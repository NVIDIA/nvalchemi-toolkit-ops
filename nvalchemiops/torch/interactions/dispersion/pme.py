# SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PyTorch bindings for dispersion (LJ-PME).

Reciprocal-space and self-energy components (PR1):
:func:`pme_dispersion_reciprocal_space`,
:func:`pme_dispersion_green_structure_factor`,
:func:`pme_dispersion_energy_corrections`.

Real-space damped LJ-PME term (PR2): :func:`lj_pme_real_space`.

The total LJ-PME energy combines these as
:math:`V_{\\text{total}} = V_{\\text{real}} + V_{\\text{recip}} - V_{\\text{self}}`.
A top-level ``lj_pme()`` orchestrator with automatic β/mesh estimation
arrives in a later PR.

References
----------
Wennberg, Hess, Lindahl (2013). JCTC 9, 3527.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import warp as wp

from nvalchemiops.interactions.dispersion.pme_dispersion_kernels import (
    _batch_pme_dispersion_green_structure_factor_kernel_overload,
    _batch_pme_dispersion_self_energy_kernel_overload,
    _lj_pme_real_space_energy_forces_kernel_overload,
    _lj_pme_real_space_energy_forces_virial_kernel_overload,
    _lj_pme_real_space_energy_kernel_overload,
    _pme_dispersion_green_structure_factor_kernel_overload,
    _pme_dispersion_self_energy_kernel_overload,
)
from nvalchemiops.torch.autograd import (
    OutputSpec,
    WarpAutogradContextManager,
    attach_for_backward,
    needs_grad,
    warp_custom_op,
    warp_from_torch,
)
from nvalchemiops.torch.interactions.dispersion.parameters import (
    DISPERSION_DEFAULT_ACCURACY,
    DISPERSION_DEFAULT_CUTOFF,
    estimate_pme_dispersion_mesh_dimensions,
    solve_dispersion_beta,
)
from nvalchemiops.torch.interactions.electrostatics.k_vectors import (
    generate_k_vectors_pme,
)
from nvalchemiops.torch.spline import spline_gather, spline_gather_vec3, spline_spread
from nvalchemiops.torch.types import get_wp_dtype, get_wp_mat_dtype, get_wp_vec_dtype

__all__ = [
    "pme_dispersion_reciprocal_space",
    "pme_dispersion_green_structure_factor",
    "pme_dispersion_energy_corrections",
    "lj_pme_real_space",
    "lj_pme",
]


###########################################################################################
########################### Helpers #######################################################
###########################################################################################


def _prepare_beta(
    beta: float | torch.Tensor,
    num_systems: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Convert beta to a per-system tensor of length num_systems."""
    if isinstance(beta, (int, float)):
        return torch.full((num_systems,), float(beta), dtype=dtype, device=device)
    if isinstance(beta, torch.Tensor):
        if beta.dim() == 0:
            return beta.expand(num_systems).to(dtype=dtype, device=device)
        if beta.shape[0] != num_systems:
            if beta.shape[0] == 1:
                return beta.expand(num_systems).to(dtype=dtype, device=device)
            raise ValueError(
                f"beta has {beta.shape[0]} values but there are {num_systems} systems"
            )
        return beta.to(dtype=dtype, device=device)
    raise TypeError(f"beta must be float or torch.Tensor, got {type(beta)}")


def _prepare_cell(cell: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Ensure cell is (B, 3, 3) and return number of systems."""
    if cell.dim() == 2:
        cell = cell.unsqueeze(0)
    return cell, cell.shape[0]


def _materialize_complex(tensor: torch.Tensor) -> torch.Tensor:
    """Force a fresh complex tensor for compiled FFT consumers."""
    if not tensor.is_complex():
        return tensor
    return torch.complex(tensor.real, tensor.imag)


###########################################################################################
########################### Green's function custom ops ###################################
###########################################################################################


def _green_output_shape(k_squared, *_):
    return k_squared.shape


def _struct_output_shape(k_squared, *_):
    return k_squared.shape


@warp_custom_op(
    name="alchemiops::_pme_dispersion_green_structure_factor",
    outputs=[
        OutputSpec("green_function", wp.array(dtype=Any, ndim=3), _green_output_shape),
        OutputSpec(
            "structure_factor_sq", wp.array(dtype=Any, ndim=3), _struct_output_shape
        ),
    ],
    grad_arrays=["green_function", "k_squared", "beta", "volume"],
)
def _pme_dispersion_green_structure_factor_op(
    k_squared: torch.Tensor,
    miller_x: torch.Tensor,
    miller_y: torch.Tensor,
    miller_z: torch.Tensor,
    beta: torch.Tensor,
    volume: torch.Tensor,
    mesh_nx: int,
    mesh_ny: int,
    mesh_nz: int,
    spline_order: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-system dispersion Green's function and B-spline correction."""
    device = wp.device_from_torch(k_squared.device)
    input_dtype = k_squared.dtype
    wp_dtype = get_wp_dtype(input_dtype)
    nx, ny, nz_rfft = k_squared.shape
    needs_grad_flag = needs_grad(k_squared, beta, volume)

    wp_k_squared = warp_from_torch(
        k_squared.contiguous(), wp_dtype, requires_grad=needs_grad_flag
    )
    wp_miller_x = warp_from_torch(
        miller_x.to(input_dtype).contiguous(), wp_dtype, requires_grad=False
    )
    wp_miller_y = warp_from_torch(
        miller_y.to(input_dtype).contiguous(), wp_dtype, requires_grad=False
    )
    wp_miller_z = warp_from_torch(
        miller_z.to(input_dtype).contiguous(), wp_dtype, requires_grad=False
    )
    wp_beta = warp_from_torch(
        beta.to(input_dtype).contiguous(), wp_dtype, requires_grad=needs_grad_flag
    )
    wp_volume = warp_from_torch(
        volume.to(input_dtype).contiguous(), wp_dtype, requires_grad=needs_grad_flag
    )

    green_function = torch.zeros(
        (nx, ny, nz_rfft), dtype=input_dtype, device=k_squared.device
    )
    structure_factor_sq = torch.zeros(
        (nx, ny, nz_rfft), dtype=input_dtype, device=k_squared.device
    )

    wp_green = warp_from_torch(green_function, wp_dtype, requires_grad=needs_grad_flag)
    wp_struct = warp_from_torch(structure_factor_sq, wp_dtype, requires_grad=False)

    kernel = _pme_dispersion_green_structure_factor_kernel_overload[wp_dtype]

    with WarpAutogradContextManager(needs_grad_flag) as tape:
        wp.launch(
            kernel,
            dim=(nx, ny, nz_rfft),
            inputs=[
                wp_k_squared,
                wp_miller_x,
                wp_miller_y,
                wp_miller_z,
                wp_beta,
                wp_volume,
                wp.int32(mesh_nx),
                wp.int32(mesh_ny),
                wp.int32(mesh_nz),
                wp.int32(spline_order),
            ],
            outputs=[wp_green, wp_struct],
            device=device,
        )

    if needs_grad_flag:
        attach_for_backward(
            green_function,
            tape=tape,
            green_function=wp_green,
            k_squared=wp_k_squared,
            beta=wp_beta,
            volume=wp_volume,
        )

    return green_function, structure_factor_sq


def _batch_green_output_shape(
    k_squared, miller_x, miller_y, miller_z, beta, volumes, *_
):
    if k_squared.dim() == 3:
        _, nx, ny, nz_rfft = (1,) + k_squared.shape
    else:
        _, nx, ny, nz_rfft = k_squared.shape
    num_systems = volumes.shape[0]
    return (num_systems, nx, ny, nz_rfft)


def _batch_struct_output_shape(k_squared, *_):
    if k_squared.dim() == 3:
        return k_squared.shape
    return k_squared.shape[1:]


@warp_custom_op(
    name="alchemiops::_batch_pme_dispersion_green_structure_factor",
    outputs=[
        OutputSpec(
            "green_function", wp.array(dtype=Any, ndim=4), _batch_green_output_shape
        ),
        OutputSpec(
            "structure_factor_sq",
            wp.array(dtype=Any, ndim=3),
            _batch_struct_output_shape,
        ),
    ],
    grad_arrays=["green_function", "k_squared", "beta", "volumes"],
)
def _batch_pme_dispersion_green_structure_factor_op(
    k_squared: torch.Tensor,
    miller_x: torch.Tensor,
    miller_y: torch.Tensor,
    miller_z: torch.Tensor,
    beta: torch.Tensor,
    volumes: torch.Tensor,
    mesh_nx: int,
    mesh_ny: int,
    mesh_nz: int,
    spline_order: int,
    num_systems: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched dispersion Green's function and B-spline correction."""
    device = wp.device_from_torch(k_squared.device)
    if k_squared.dim() == 3:
        k_squared = k_squared.unsqueeze(0)
    input_dtype = k_squared.dtype
    wp_dtype = get_wp_dtype(input_dtype)
    _, nx, ny, nz_rfft = k_squared.shape
    needs_grad_flag = needs_grad(k_squared, beta, volumes)

    wp_k_squared = warp_from_torch(
        k_squared.contiguous(), wp_dtype, requires_grad=needs_grad_flag
    )
    wp_miller_x = warp_from_torch(
        miller_x.to(input_dtype).contiguous(), wp_dtype, requires_grad=False
    )
    wp_miller_y = warp_from_torch(
        miller_y.to(input_dtype).contiguous(), wp_dtype, requires_grad=False
    )
    wp_miller_z = warp_from_torch(
        miller_z.to(input_dtype).contiguous(), wp_dtype, requires_grad=False
    )
    wp_beta = warp_from_torch(
        beta.to(input_dtype).contiguous(), wp_dtype, requires_grad=needs_grad_flag
    )
    wp_volumes = warp_from_torch(
        volumes.to(input_dtype).contiguous(), wp_dtype, requires_grad=needs_grad_flag
    )

    green_function = torch.zeros(
        (num_systems, nx, ny, nz_rfft), dtype=input_dtype, device=k_squared.device
    )
    structure_factor_sq = torch.zeros(
        (nx, ny, nz_rfft), dtype=input_dtype, device=k_squared.device
    )

    wp_green = warp_from_torch(green_function, wp_dtype, requires_grad=needs_grad_flag)
    wp_struct = warp_from_torch(structure_factor_sq, wp_dtype, requires_grad=False)

    kernel = _batch_pme_dispersion_green_structure_factor_kernel_overload[wp_dtype]

    with WarpAutogradContextManager(needs_grad_flag) as tape:
        wp.launch(
            kernel,
            dim=(num_systems, nx, ny, nz_rfft),
            inputs=[
                wp_k_squared,
                wp_miller_x,
                wp_miller_y,
                wp_miller_z,
                wp_beta,
                wp_volumes,
                wp.int32(mesh_nx),
                wp.int32(mesh_ny),
                wp.int32(mesh_nz),
                wp.int32(spline_order),
            ],
            outputs=[wp_green, wp_struct],
            device=device,
        )

    if needs_grad_flag:
        attach_for_backward(
            green_function,
            tape=tape,
            green_function=wp_green,
            k_squared=wp_k_squared,
            beta=wp_beta,
            volumes=wp_volumes,
        )

    return green_function, structure_factor_sq


def pme_dispersion_green_structure_factor(
    k_squared: torch.Tensor,
    mesh_dimensions: tuple[int, int, int],
    beta: torch.Tensor,
    cell: torch.Tensor,
    spline_order: int = 4,
    batch_idx: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute dispersion Green's function and B-spline structure factor.

    Green's function:

    .. math::

        G_{\\text{disp}}(k) = \\frac{\\pi^{3/2} \\beta^3}{2V}
            \\, f\\!\\left(\\tfrac{|k|}{2\\beta}\\right),
        \\qquad
        f(x) = \\tfrac{1}{3}\\bigl[(1 - 2x^2) e^{-x^2}
            + 2 x^3 \\sqrt{\\pi}\\, \\text{erfc}(x)\\bigr].
    """
    mesh_nx, mesh_ny, mesh_nz = mesh_dimensions
    device = k_squared.device
    input_dtype = k_squared.dtype

    cell = cell if cell.dim() == 3 else cell.unsqueeze(0)
    volume = torch.abs(torch.det(cell)).to(input_dtype)

    miller_x = torch.fft.fftfreq(
        mesh_nx, d=1.0 / mesh_nx, device=device, dtype=input_dtype
    )
    miller_y = torch.fft.fftfreq(
        mesh_ny, d=1.0 / mesh_ny, device=device, dtype=input_dtype
    )
    miller_z = torch.fft.rfftfreq(
        mesh_nz, d=1.0 / mesh_nz, device=device, dtype=input_dtype
    )

    if batch_idx is None:
        return _pme_dispersion_green_structure_factor_op(
            k_squared,
            miller_x,
            miller_y,
            miller_z,
            beta.to(input_dtype),
            volume,
            mesh_nx,
            mesh_ny,
            mesh_nz,
            spline_order,
        )
    num_systems = cell.shape[0]
    return _batch_pme_dispersion_green_structure_factor_op(
        k_squared,
        miller_x,
        miller_y,
        miller_z,
        beta.to(input_dtype),
        volume,
        mesh_nx,
        mesh_ny,
        mesh_nz,
        spline_order,
        num_systems,
    )


###########################################################################################
########################### Self-energy custom ops ########################################
###########################################################################################


@warp_custom_op(
    name="alchemiops::_pme_dispersion_self_energy",
    outputs=[
        OutputSpec(
            "energy_correction",
            wp.array(dtype=Any, ndim=1),
            lambda c6_coefficients, *_: (c6_coefficients.shape[0],),
        ),
    ],
    grad_arrays=["energy_correction", "c6_coefficients", "beta"],
)
def _pme_dispersion_self_energy_op(
    c6_coefficients: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """Per-atom dispersion self-energy: :math:`-\\beta^6 C_{6,ii}/12`."""
    device = wp.device_from_torch(c6_coefficients.device)
    input_dtype = c6_coefficients.dtype
    wp_dtype = get_wp_dtype(input_dtype)
    num_atoms = c6_coefficients.shape[0]
    needs_grad_flag = needs_grad(c6_coefficients, beta)

    wp_c6 = warp_from_torch(
        c6_coefficients.contiguous(), wp_dtype, requires_grad=needs_grad_flag
    )
    wp_beta = warp_from_torch(
        beta.to(input_dtype).contiguous(), wp_dtype, requires_grad=needs_grad_flag
    )

    energy_correction = torch.zeros(
        num_atoms, dtype=input_dtype, device=c6_coefficients.device
    )
    wp_energy = warp_from_torch(
        energy_correction, wp_dtype, requires_grad=needs_grad_flag
    )

    kernel = _pme_dispersion_self_energy_kernel_overload[wp_dtype]

    with WarpAutogradContextManager(needs_grad_flag) as tape:
        wp.launch(
            kernel,
            dim=num_atoms,
            inputs=[wp_c6, wp_beta],
            outputs=[wp_energy],
            device=device,
        )

    if needs_grad_flag:
        attach_for_backward(
            energy_correction,
            tape=tape,
            energy_correction=wp_energy,
            c6_coefficients=wp_c6,
            beta=wp_beta,
        )
    return energy_correction


@warp_custom_op(
    name="alchemiops::_batch_pme_dispersion_self_energy",
    outputs=[
        OutputSpec(
            "energy_correction",
            wp.array(dtype=Any, ndim=1),
            lambda c6_coefficients, *_: (c6_coefficients.shape[0],),
        ),
    ],
    grad_arrays=["energy_correction", "c6_coefficients", "beta"],
)
def _batch_pme_dispersion_self_energy_op(
    c6_coefficients: torch.Tensor,
    batch_idx: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """Batched per-atom dispersion self-energy."""
    device = wp.device_from_torch(c6_coefficients.device)
    input_dtype = c6_coefficients.dtype
    wp_dtype = get_wp_dtype(input_dtype)
    num_atoms = c6_coefficients.shape[0]
    needs_grad_flag = needs_grad(c6_coefficients, beta)

    wp_c6 = warp_from_torch(
        c6_coefficients.contiguous(), wp_dtype, requires_grad=needs_grad_flag
    )
    wp_batch_idx = warp_from_torch(
        batch_idx.contiguous(), wp.int32, requires_grad=False
    )
    wp_beta = warp_from_torch(
        beta.to(input_dtype).contiguous(), wp_dtype, requires_grad=needs_grad_flag
    )

    energy_correction = torch.zeros(
        num_atoms, dtype=input_dtype, device=c6_coefficients.device
    )
    wp_energy = warp_from_torch(
        energy_correction, wp_dtype, requires_grad=needs_grad_flag
    )

    kernel = _batch_pme_dispersion_self_energy_kernel_overload[wp_dtype]

    with WarpAutogradContextManager(needs_grad_flag) as tape:
        wp.launch(
            kernel,
            dim=num_atoms,
            inputs=[wp_c6, wp_batch_idx, wp_beta],
            outputs=[wp_energy],
            device=device,
        )

    if needs_grad_flag:
        attach_for_backward(
            energy_correction,
            tape=tape,
            energy_correction=wp_energy,
            c6_coefficients=wp_c6,
            beta=wp_beta,
        )
    return energy_correction


def pme_dispersion_energy_corrections(
    c6_coefficients: torch.Tensor,
    beta: torch.Tensor,
    batch_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the dispersion self-energy correction term.

    Returns the per-system self-energy

    .. math::

        V_{\\text{self}} = -\\frac{\\beta^6}{12} \\sum_i C_{6,ii}.

    The total LJ-PME energy is :math:`V_{\\text{total}} = V_{\\text{real}}
    + V_{\\text{recip}} - V_{\\text{self}}`.

    Parameters
    ----------
    c6_coefficients : torch.Tensor, shape (N,) or (N_total,)
        Per-atom homoatomic :math:`C_{6,ii}` values.
    beta : torch.Tensor
        Dispersion Ewald splitting parameter.
        - Single-system: shape (1,)
        - Batch: shape (B,)
    batch_idx : torch.Tensor | None
        System index for each atom. If provided, uses batched kernel.

    Returns
    -------
    energy_corrections : torch.Tensor
        - Single-system: shape (1,)
        - Batch: shape (B,)
    """
    input_dtype = c6_coefficients.dtype

    if batch_idx is None:
        per_atom = _pme_dispersion_self_energy_op(
            c6_coefficients,
            beta.to(input_dtype),
        )
        return per_atom.sum().reshape(1)

    num_systems = beta.shape[0]
    per_atom = _batch_pme_dispersion_self_energy_op(
        c6_coefficients,
        batch_idx.to(torch.int32),
        beta.to(input_dtype),
    )
    out = torch.zeros(num_systems, dtype=input_dtype, device=c6_coefficients.device)
    out.scatter_add_(0, batch_idx.to(torch.int64), per_atom)
    return out


###########################################################################################
########################### Reciprocal-space dispersion PME ###############################
###########################################################################################


def pme_dispersion_reciprocal_space(
    positions: torch.Tensor,
    c6_coefficients: torch.Tensor,
    cell: torch.Tensor,
    beta: float | torch.Tensor,
    mesh_dimensions: tuple[int, int, int] | None = None,
    mesh_spacing: float | None = None,
    spline_order: int = 4,
    batch_idx: torch.Tensor | None = None,
    k_vectors: torch.Tensor | None = None,
    k_squared: torch.Tensor | None = None,
    compute_forces: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Compute the dispersion PME reciprocal-space energy.

    Implements the FFT-based long-range :math:`r^{-6}` contribution using
    B-spline interpolation and convolution with the dispersion Green's
    function:

    .. math::

        V_{\\text{recip}} = \\frac{\\pi^{3/2}\\beta^3}{2V}
            \\sum_{m \\neq 0} f(\\pi|m|/\\beta)\\, |\\rho_{\\text{disp}}(m)|^2.

    The spread quantity is :math:`\\sqrt{C_{6,ii}}` (geometric combination).

    Parameters
    ----------
    positions : torch.Tensor, shape (N, 3)
        Atomic coordinates.
    c6_coefficients : torch.Tensor, shape (N,)
        Per-atom homoatomic :math:`C_{6,ii}` values (non-negative).
    cell : torch.Tensor, shape (3, 3) or (B, 3, 3)
        Unit cell matrices.
    beta : float or torch.Tensor
        Dispersion Ewald splitting parameter. Scalar (broadcast) or shape (B,).
    mesh_dimensions : tuple[int, int, int], optional
        Explicit FFT mesh dimensions.
    mesh_spacing : float, optional
        Target mesh spacing in cell units.
    spline_order : int, default=4
        B-spline interpolation order.
    batch_idx : torch.Tensor | None
        System index for each atom.
    k_vectors : torch.Tensor, optional
        Precomputed k-vectors from ``generate_k_vectors_pme``.
    k_squared : torch.Tensor, optional
        Precomputed :math:`|k|^2` values.

    Returns
    -------
    energy : torch.Tensor
        - Single-system: shape (1,).
        - Batch: shape (B,).
    """
    num_atoms = positions.shape[0]
    input_dtype = positions.dtype
    device = positions.device
    is_batch = batch_idx is not None
    fft_dims = (1, 2, 3) if is_batch else (0, 1, 2)

    cell_b, num_systems = _prepare_cell(cell)

    if num_atoms == 0:
        empty_e = torch.zeros(
            num_systems if is_batch else 1, dtype=input_dtype, device=device
        )
        if compute_forces:
            return empty_e, torch.zeros((0, 3), dtype=input_dtype, device=device)
        return empty_e

    if mesh_dimensions is None:
        if mesh_spacing is None:
            raise ValueError("Either mesh_dimensions or mesh_spacing must be provided")
        cell_lengths = torch.norm(cell_b[0], dim=1)
        mesh_dimensions = tuple(
            int(math.ceil(float(length) / mesh_spacing)) for length in cell_lengths
        )

    mesh_nx, mesh_ny, mesh_nz = mesh_dimensions
    beta_tensor = _prepare_beta(beta, num_systems, input_dtype, device)

    c6 = c6_coefficients.to(input_dtype)
    sqrt_c6 = torch.sqrt(torch.clamp(c6, min=0.0))

    mesh_grid = spline_spread(
        positions,
        sqrt_c6,
        cell_b,
        mesh_dims=(mesh_nx, mesh_ny, mesh_nz),
        spline_order=spline_order,
        batch_idx=batch_idx,
    )

    mesh_fft = torch.fft.rfftn(mesh_grid, norm="backward", dim=fft_dims)

    if k_vectors is None or k_squared is None:
        k_vectors, k_squared = generate_k_vectors_pme(cell_b, mesh_dimensions)

    green_function, structure_factor_sq = pme_dispersion_green_structure_factor(
        k_squared,
        mesh_dimensions,
        beta_tensor,
        cell_b,
        spline_order,
        batch_idx=batch_idx,
    )

    mesh_fft = mesh_fft / structure_factor_sq
    convolved_mesh = _materialize_complex(mesh_fft * green_function)

    potential_mesh = torch.fft.irfftn(
        convolved_mesh, norm="forward", s=mesh_dimensions, dim=fft_dims
    )
    potential_mesh = potential_mesh.to(input_dtype)

    raw_potential = spline_gather(
        positions,
        potential_mesh,
        cell_b,
        spline_order=spline_order,
        batch_idx=batch_idx,
    )

    per_atom = sqrt_c6 * raw_potential

    if is_batch:
        energy = torch.zeros(num_systems, dtype=input_dtype, device=device)
        energy.scatter_add_(0, batch_idx.to(torch.int64), per_atom)
    else:
        energy = per_atom.sum().reshape(1)

    if not compute_forces:
        return energy

    # Forces via Fourier gradient (same trick as Coulomb PME).
    Ex_fft = _materialize_complex(-1j * k_vectors[..., 0] * convolved_mesh)
    Ey_fft = _materialize_complex(-1j * k_vectors[..., 1] * convolved_mesh)
    Ez_fft = _materialize_complex(-1j * k_vectors[..., 2] * convolved_mesh)
    Ex = torch.fft.irfftn(Ex_fft, norm="forward", s=mesh_dimensions, dim=fft_dims)
    Ey = torch.fft.irfftn(Ey_fft, norm="forward", s=mesh_dimensions, dim=fft_dims)
    Ez = torch.fft.irfftn(Ez_fft, norm="forward", s=mesh_dimensions, dim=fft_dims)
    gradient_field_mesh = torch.stack([Ex, Ey, Ez], dim=-1).to(input_dtype)

    interpolated_gradient = spline_gather_vec3(
        positions,
        sqrt_c6,
        gradient_field_mesh,
        cell_b,
        spline_order=spline_order,
        batch_idx=batch_idx,
    )
    forces = 2.0 * interpolated_gradient
    return energy, forces


###########################################################################################
########################### Real-space LJ-PME custom ops (PR2) ############################
###########################################################################################


@warp_custom_op(
    name="alchemiops::_lj_pme_real_space_energy",
    outputs=[
        OutputSpec(
            "atomic_energies",
            lambda pos, *_: get_wp_dtype(pos.dtype),
            lambda pos, *_: (pos.shape[0],),
        ),
    ],
    grad_arrays=[
        "atomic_energies",
        "positions",
        "c6_coefficients",
        "c12_coefficients",
        "cell",
        "beta",
    ],
)
def _lj_pme_real_space_energy_op(
    positions: torch.Tensor,
    c6_coefficients: torch.Tensor,
    c12_coefficients: torch.Tensor,
    cell: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    num_neighbors: torch.Tensor,
    beta: torch.Tensor,
    cutoff: torch.Tensor,
    mask_value: int,
    half_neighbor_list: bool,
) -> torch.Tensor:
    """Internal: real-space LJ-PME energies (single system, neighbor matrix)."""
    num_atoms = positions.shape[0]
    input_dtype = positions.dtype
    device = wp.device_from_torch(positions.device)

    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)
    needs_grad_flag = needs_grad(
        positions, c6_coefficients, c12_coefficients, cell, beta
    )

    wp_positions = warp_from_torch(positions, wp_vec, requires_grad=needs_grad_flag)
    wp_c6 = warp_from_torch(c6_coefficients, wp_scalar, requires_grad=needs_grad_flag)
    wp_c12 = warp_from_torch(c12_coefficients, wp_scalar, requires_grad=needs_grad_flag)
    wp_cell = warp_from_torch(cell, wp_mat, requires_grad=needs_grad_flag)
    wp_nbr = warp_from_torch(neighbor_matrix, wp.int32)
    wp_nbr_shifts = warp_from_torch(neighbor_matrix_shifts, wp.vec3i)
    wp_num_nbrs = warp_from_torch(num_neighbors, wp.int32)
    wp_beta = warp_from_torch(beta, wp_scalar, requires_grad=needs_grad_flag)
    wp_cutoff = warp_from_torch(cutoff, wp_scalar)

    energies = torch.zeros(num_atoms, dtype=input_dtype, device=positions.device)
    wp_energies = warp_from_torch(energies, wp_scalar, requires_grad=needs_grad_flag)

    kernel = _lj_pme_real_space_energy_kernel_overload[wp_scalar]
    with WarpAutogradContextManager(needs_grad_flag) as tape:
        wp.launch(
            kernel,
            dim=num_atoms,
            inputs=[
                wp_positions,
                wp_c6,
                wp_c12,
                wp_cell,
                wp_nbr,
                wp_nbr_shifts,
                wp_num_nbrs,
                wp_beta,
                wp_cutoff,
                wp.int32(mask_value),
                bool(half_neighbor_list),
            ],
            outputs=[wp_energies],
            device=device,
        )
    if needs_grad_flag:
        attach_for_backward(
            energies,
            tape=tape,
            atomic_energies=wp_energies,
            positions=wp_positions,
            c6_coefficients=wp_c6,
            c12_coefficients=wp_c12,
            cell=wp_cell,
            beta=wp_beta,
        )
    return energies


@warp_custom_op(
    name="alchemiops::_lj_pme_real_space_energy_forces",
    outputs=[
        OutputSpec(
            "atomic_energies",
            lambda pos, *_: get_wp_dtype(pos.dtype),
            lambda pos, *_: (pos.shape[0],),
        ),
        OutputSpec(
            "atomic_forces",
            lambda pos, *_: get_wp_vec_dtype(pos.dtype),
            lambda pos, *_: (pos.shape[0], 3),
        ),
    ],
    grad_arrays=[
        "atomic_energies",
        "atomic_forces",
        "positions",
        "c6_coefficients",
        "c12_coefficients",
        "cell",
        "beta",
    ],
)
def _lj_pme_real_space_energy_forces_op(
    positions: torch.Tensor,
    c6_coefficients: torch.Tensor,
    c12_coefficients: torch.Tensor,
    cell: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    num_neighbors: torch.Tensor,
    beta: torch.Tensor,
    cutoff: torch.Tensor,
    mask_value: int,
    half_neighbor_list: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Internal: real-space LJ-PME energies + forces."""
    num_atoms = positions.shape[0]
    input_dtype = positions.dtype
    device = wp.device_from_torch(positions.device)

    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)
    needs_grad_flag = needs_grad(
        positions, c6_coefficients, c12_coefficients, cell, beta
    )

    wp_positions = warp_from_torch(positions, wp_vec, requires_grad=needs_grad_flag)
    wp_c6 = warp_from_torch(c6_coefficients, wp_scalar, requires_grad=needs_grad_flag)
    wp_c12 = warp_from_torch(c12_coefficients, wp_scalar, requires_grad=needs_grad_flag)
    wp_cell = warp_from_torch(cell, wp_mat, requires_grad=needs_grad_flag)
    wp_nbr = warp_from_torch(neighbor_matrix, wp.int32)
    wp_nbr_shifts = warp_from_torch(neighbor_matrix_shifts, wp.vec3i)
    wp_num_nbrs = warp_from_torch(num_neighbors, wp.int32)
    wp_beta = warp_from_torch(beta, wp_scalar, requires_grad=needs_grad_flag)
    wp_cutoff = warp_from_torch(cutoff, wp_scalar)

    energies = torch.zeros(num_atoms, dtype=input_dtype, device=positions.device)
    forces = torch.zeros((num_atoms, 3), dtype=input_dtype, device=positions.device)
    wp_energies = warp_from_torch(energies, wp_scalar, requires_grad=needs_grad_flag)
    wp_forces = warp_from_torch(forces, wp_vec, requires_grad=needs_grad_flag)

    kernel = _lj_pme_real_space_energy_forces_kernel_overload[wp_scalar]
    with WarpAutogradContextManager(needs_grad_flag) as tape:
        wp.launch(
            kernel,
            dim=num_atoms,
            inputs=[
                wp_positions,
                wp_c6,
                wp_c12,
                wp_cell,
                wp_nbr,
                wp_nbr_shifts,
                wp_num_nbrs,
                wp_beta,
                wp_cutoff,
                wp.int32(mask_value),
                bool(half_neighbor_list),
            ],
            outputs=[wp_energies, wp_forces],
            device=device,
        )
    if needs_grad_flag:
        attach_for_backward(
            energies,
            tape=tape,
            atomic_energies=wp_energies,
            atomic_forces=wp_forces,
            positions=wp_positions,
            c6_coefficients=wp_c6,
            c12_coefficients=wp_c12,
            cell=wp_cell,
            beta=wp_beta,
        )
    return energies, forces


@warp_custom_op(
    name="alchemiops::_lj_pme_real_space_energy_forces_virial",
    outputs=[
        OutputSpec(
            "atomic_energies",
            lambda pos, *_: get_wp_dtype(pos.dtype),
            lambda pos, *_: (pos.shape[0],),
        ),
        OutputSpec(
            "atomic_forces",
            lambda pos, *_: get_wp_vec_dtype(pos.dtype),
            lambda pos, *_: (pos.shape[0], 3),
        ),
        OutputSpec(
            "virial",
            lambda pos, *_: get_wp_dtype(pos.dtype),
            lambda *_: (9,),
        ),
    ],
    grad_arrays=[
        "atomic_energies",
        "atomic_forces",
        "virial",
        "positions",
        "c6_coefficients",
        "c12_coefficients",
        "cell",
        "beta",
    ],
)
def _lj_pme_real_space_energy_forces_virial_op(
    positions: torch.Tensor,
    c6_coefficients: torch.Tensor,
    c12_coefficients: torch.Tensor,
    cell: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    num_neighbors: torch.Tensor,
    beta: torch.Tensor,
    cutoff: torch.Tensor,
    mask_value: int,
    half_neighbor_list: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Internal: real-space LJ-PME energies + forces + virial."""
    num_atoms = positions.shape[0]
    input_dtype = positions.dtype
    device = wp.device_from_torch(positions.device)

    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)
    needs_grad_flag = needs_grad(
        positions, c6_coefficients, c12_coefficients, cell, beta
    )

    wp_positions = warp_from_torch(positions, wp_vec, requires_grad=needs_grad_flag)
    wp_c6 = warp_from_torch(c6_coefficients, wp_scalar, requires_grad=needs_grad_flag)
    wp_c12 = warp_from_torch(c12_coefficients, wp_scalar, requires_grad=needs_grad_flag)
    wp_cell = warp_from_torch(cell, wp_mat, requires_grad=needs_grad_flag)
    wp_nbr = warp_from_torch(neighbor_matrix, wp.int32)
    wp_nbr_shifts = warp_from_torch(neighbor_matrix_shifts, wp.vec3i)
    wp_num_nbrs = warp_from_torch(num_neighbors, wp.int32)
    wp_beta = warp_from_torch(beta, wp_scalar, requires_grad=needs_grad_flag)
    wp_cutoff = warp_from_torch(cutoff, wp_scalar)

    energies = torch.zeros(num_atoms, dtype=input_dtype, device=positions.device)
    forces = torch.zeros((num_atoms, 3), dtype=input_dtype, device=positions.device)
    virial_flat = torch.zeros(9, dtype=input_dtype, device=positions.device)
    wp_energies = warp_from_torch(energies, wp_scalar, requires_grad=needs_grad_flag)
    wp_forces = warp_from_torch(forces, wp_vec, requires_grad=needs_grad_flag)
    wp_virial = warp_from_torch(virial_flat, wp_scalar, requires_grad=needs_grad_flag)

    kernel = _lj_pme_real_space_energy_forces_virial_kernel_overload[wp_scalar]
    with WarpAutogradContextManager(needs_grad_flag) as tape:
        wp.launch(
            kernel,
            dim=num_atoms,
            inputs=[
                wp_positions,
                wp_c6,
                wp_c12,
                wp_cell,
                wp_nbr,
                wp_nbr_shifts,
                wp_num_nbrs,
                wp_beta,
                wp_cutoff,
                wp.int32(mask_value),
                bool(half_neighbor_list),
            ],
            outputs=[wp_energies, wp_forces, wp_virial],
            device=device,
        )
    if needs_grad_flag:
        attach_for_backward(
            energies,
            tape=tape,
            atomic_energies=wp_energies,
            atomic_forces=wp_forces,
            virial=wp_virial,
            positions=wp_positions,
            c6_coefficients=wp_c6,
            c12_coefficients=wp_c12,
            cell=wp_cell,
            beta=wp_beta,
        )
    return energies, forces, virial_flat


def lj_pme_real_space(
    positions: torch.Tensor,
    c6_coefficients: torch.Tensor,
    c12_coefficients: torch.Tensor,
    cell: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    beta: float | torch.Tensor,
    cutoff: float,
    num_neighbors: torch.Tensor | None = None,
    mask_value: int | None = None,
    compute_forces: bool = False,
    compute_virial: bool = False,
    half_neighbor_list: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Real-space LJ-PME pair energy (and optionally forces and virial).

    For each pair (i, j) with :math:`r_{ij} < r_{\\text{cut}}` the kernel
    computes the damped Lennard-Jones pair contribution

    .. math::

        V_{ij} = \\frac{C_{12,ij}}{r_{ij}^{12}}
                 - \\frac{C_{6,ij}\\, g(\\beta r_{ij})}{r_{ij}^{6}},
        \\qquad
        g(x) = e^{-x^2}\\bigl(1 + x^2 + x^4/2\\bigr),

    with geometric combination rules
    :math:`C_{6,ij} = \\sqrt{C_{6,ii} C_{6,jj}}`,
    :math:`C_{12,ij} = \\sqrt{C_{12,ii} C_{12,jj}}`. The damping function
    :math:`g` is the Wennberg complement of the long-range PME term so
    that :math:`V_{\\text{real}} + V_{\\text{recip}} - V_{\\text{self}}`
    recovers the bare :math:`r^{-6}` lattice sum.

    Parameters
    ----------
    positions : torch.Tensor, shape (N, 3)
        Atomic coordinates.
    c6_coefficients : torch.Tensor, shape (N,)
        Per-atom homoatomic :math:`C_{6,ii}` values.
    c12_coefficients : torch.Tensor, shape (N,)
        Per-atom homoatomic :math:`C_{12,ii}` values.
    cell : torch.Tensor, shape (3, 3) or (1, 3, 3)
        Unit cell matrix (single-system only in this PR).
    neighbor_matrix : torch.Tensor, shape (N, max_neighbors), dtype int32
        Half neighbor matrix: each pair (i, j) appears exactly once.
        Invalid entries should equal ``mask_value``.
    neighbor_matrix_shifts : torch.Tensor, shape (N, max_neighbors, 3), dtype int32
        Periodic image shifts.
    beta : float or torch.Tensor
        Dispersion Ewald splitting parameter.
    cutoff : float
        Real-space cutoff radius.
    num_neighbors : torch.Tensor | None, shape (N,), dtype int32
        Valid neighbor count per atom. If None, inferred from ``neighbor_matrix``.
    mask_value : int | None
        Sentinel value marking invalid entries. Defaults to ``num_atoms``.
    compute_forces : bool, default=False
        If True, return explicit forces alongside the per-atom energy.
    compute_virial : bool, default=False
        If True, return the 3×3 virial tensor
        :math:`W_{ab} = \\sum_{i<j} r_{ij,a}\\, F_{ij,b}`.

    Returns
    -------
    energies : torch.Tensor, shape (N,)
        Per-atom real-space dispersion energy (half-counted per pair).
    forces : torch.Tensor, shape (N, 3), optional
        Per-atom forces. Only returned when ``compute_forces=True``.
    virial : torch.Tensor, shape (3, 3), optional
        Virial tensor. Only returned when ``compute_virial=True``; always
        last in the return tuple.
    """
    num_atoms = positions.shape[0]
    input_dtype = positions.dtype
    device = positions.device

    if cell.dim() == 2:
        cell_b = cell.unsqueeze(0)
    else:
        cell_b = cell
    if cell_b.shape[0] != 1:
        raise ValueError(
            "lj_pme_real_space currently supports single-system cells; "
            f"got cell shape {tuple(cell_b.shape)}."
        )

    c6 = c6_coefficients.to(input_dtype)
    c12 = c12_coefficients.to(input_dtype)
    cell_cast = cell_b.to(input_dtype)

    if isinstance(beta, (int, float)):
        beta_t = torch.tensor([float(beta)], dtype=input_dtype, device=device)
    else:
        beta_t = beta.to(input_dtype)
        if beta_t.dim() == 0:
            beta_t = beta_t.reshape(1)
        if beta_t.numel() != 1:
            raise ValueError(
                "lj_pme_real_space currently supports a single beta value; "
                f"got shape {tuple(beta_t.shape)}."
            )

    cutoff_t = torch.tensor([float(cutoff)], dtype=input_dtype, device=device)

    nbr_mat_i32 = neighbor_matrix.to(torch.int32)
    nbr_shifts_i32 = neighbor_matrix_shifts.to(torch.int32)

    if mask_value is None:
        mask_value = num_atoms

    if num_neighbors is None:
        valid = nbr_mat_i32 != mask_value
        num_neighbors_t = valid.sum(dim=1).to(torch.int32)
    else:
        num_neighbors_t = num_neighbors.to(torch.int32)

    if compute_virial:
        energies, forces, virial_flat = _lj_pme_real_space_energy_forces_virial_op(
            positions,
            c6,
            c12,
            cell_cast,
            nbr_mat_i32,
            nbr_shifts_i32,
            num_neighbors_t,
            beta_t,
            cutoff_t,
            int(mask_value),
            bool(half_neighbor_list),
        )
        virial = virial_flat.reshape(3, 3)
        if compute_forces:
            return energies, forces, virial
        return energies, virial
    elif compute_forces:
        energies, forces = _lj_pme_real_space_energy_forces_op(
            positions,
            c6,
            c12,
            cell_cast,
            nbr_mat_i32,
            nbr_shifts_i32,
            num_neighbors_t,
            beta_t,
            cutoff_t,
            int(mask_value),
            bool(half_neighbor_list),
        )
        return energies, forces
    else:
        return _lj_pme_real_space_energy_op(
            positions,
            c6,
            c12,
            cell_cast,
            nbr_mat_i32,
            nbr_shifts_i32,
            num_neighbors_t,
            beta_t,
            cutoff_t,
            int(mask_value),
            bool(half_neighbor_list),
        )


###########################################################################################
########################### Top-level LJ-PME orchestrator (PR3) ###########################
###########################################################################################


def lj_pme(
    positions: torch.Tensor,
    c6_coefficients: torch.Tensor,
    c12_coefficients: torch.Tensor,
    cell: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    beta: float | torch.Tensor | None = None,
    cutoff: float | None = None,
    mesh_spacing: float | None = None,
    mesh_dimensions: tuple[int, int, int] | None = None,
    spline_order: int = 4,
    batch_idx: torch.Tensor | None = None,
    num_neighbors: torch.Tensor | None = None,
    mask_value: int | None = None,
    compute_forces: bool = False,
    accuracy: float = DISPERSION_DEFAULT_ACCURACY,
    half_neighbor_list: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Complete LJ-PME energy (and optionally forces) for one or more systems.

    Computes :math:`V_{\\text{total}} = V_{\\text{real}} + V_{\\text{recip}} - V_{\\text{self}}`
    with geometric combination rules:

    * Real-space damped sum (:func:`lj_pme_real_space`)
    * Reciprocal-space FFT sum (:func:`pme_dispersion_reciprocal_space`)
    * Self-energy correction (:func:`pme_dispersion_energy_corrections`)

    If ``beta``, ``cutoff``, or ``mesh_dimensions`` are not provided they
    are estimated jointly from ``accuracy`` (GROMACS-style matched-tail
    criterion; see :func:`estimate_pme_dispersion_parameters`).

    Parameters
    ----------
    positions : torch.Tensor, shape (N, 3)
    c6_coefficients : torch.Tensor, shape (N,)
    c12_coefficients : torch.Tensor, shape (N,)
    cell : torch.Tensor, shape (3, 3) or (B, 3, 3)
    neighbor_matrix : torch.Tensor, shape (N, max_neighbors), dtype int32
    neighbor_matrix_shifts : torch.Tensor, shape (N, max_neighbors, 3), dtype int32
    beta : float | torch.Tensor | None
    cutoff : float | None
    mesh_spacing : float | None
    mesh_dimensions : tuple[int, int, int] | None
    spline_order : int, default=4
    batch_idx : torch.Tensor | None
    num_neighbors : torch.Tensor | None
    mask_value : int | None
    compute_forces : bool, default=False
    accuracy : float, default=1e-3

    Returns
    -------
    energy : torch.Tensor
        Shape (1,) or (B,).
    forces : torch.Tensor, optional
        Shape (N, 3) if ``compute_forces=True``.
    """
    is_batch = batch_idx is not None
    if cell.dim() == 2:
        cell_b = cell.unsqueeze(0)
    else:
        cell_b = cell
    num_systems = cell_b.shape[0]
    input_dtype = positions.dtype
    device = positions.device

    if cutoff is None:
        cutoff = DISPERSION_DEFAULT_CUTOFF
    if beta is None:
        beta_scalar = solve_dispersion_beta(cutoff, accuracy)
        beta_arr = torch.full(
            (num_systems,), beta_scalar, dtype=input_dtype, device=device
        )
    elif isinstance(beta, (int, float)):
        beta_arr = torch.full(
            (num_systems,), float(beta), dtype=input_dtype, device=device
        )
    else:
        beta_arr = beta.to(dtype=input_dtype, device=device)
        if beta_arr.dim() == 0:
            beta_arr = beta_arr.expand(num_systems).contiguous()
        elif beta_arr.shape[0] == 1 and num_systems > 1:
            beta_arr = beta_arr.expand(num_systems).contiguous()

    if mesh_dimensions is None:
        if mesh_spacing is not None:
            cell_lengths = torch.linalg.norm(cell_b[0], dim=1)
            mesh_dimensions = tuple(
                int(math.ceil(float(length) / mesh_spacing)) for length in cell_lengths
            )
        else:
            mesh_dimensions = estimate_pme_dispersion_mesh_dimensions(
                cell_b, beta_arr, accuracy
            )

    # Real-space contribution.
    if compute_forces:
        e_real_per_atom, f_real = lj_pme_real_space(
            positions,
            c6_coefficients,
            c12_coefficients,
            cell_b if is_batch else cell,
            neighbor_matrix,
            neighbor_matrix_shifts,
            beta=beta_arr if not is_batch else beta_arr[:1],
            cutoff=cutoff,
            num_neighbors=num_neighbors,
            mask_value=mask_value,
            compute_forces=True,
            half_neighbor_list=half_neighbor_list,
        )
    else:
        e_real_per_atom = lj_pme_real_space(
            positions,
            c6_coefficients,
            c12_coefficients,
            cell_b if is_batch else cell,
            neighbor_matrix,
            neighbor_matrix_shifts,
            beta=beta_arr if not is_batch else beta_arr[:1],
            cutoff=cutoff,
            num_neighbors=num_neighbors,
            mask_value=mask_value,
            compute_forces=False,
            half_neighbor_list=half_neighbor_list,
        )
        f_real = None

    # Reciprocal-space contribution.
    if compute_forces:
        e_recip, f_recip = pme_dispersion_reciprocal_space(
            positions,
            c6_coefficients,
            cell_b if is_batch else cell,
            beta_arr,
            mesh_dimensions=mesh_dimensions,
            spline_order=spline_order,
            batch_idx=batch_idx,
            compute_forces=True,
        )
    else:
        e_recip = pme_dispersion_reciprocal_space(
            positions,
            c6_coefficients,
            cell_b if is_batch else cell,
            beta_arr,
            mesh_dimensions=mesh_dimensions,
            spline_order=spline_order,
            batch_idx=batch_idx,
        )
        f_recip = None

    # Self-energy.
    e_self = pme_dispersion_energy_corrections(
        c6_coefficients, beta_arr, batch_idx=batch_idx
    )

    if is_batch:
        e_real_per_system = torch.zeros(num_systems, dtype=input_dtype, device=device)
        e_real_per_system.scatter_add_(0, batch_idx.to(torch.int64), e_real_per_atom)
    else:
        e_real_per_system = e_real_per_atom.sum().reshape(1)

    energy = e_real_per_system + e_recip - e_self

    if compute_forces:
        forces = f_real + f_recip
        return energy, forces
    return energy
