# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Unified Particle Mesh Ewald (PME) API
=====================================

This module provides a unified GPU-accelerated API for Particle Mesh Ewald that
handles both single-system and batched calculations transparently. PME achieves
:math:`O(N \\log N)` scaling compared to :math:`O(N^2)` for direct summation, making it efficient
for large systems.

API STRUCTURE
=============

Primary APIs (public, with autograd support):
    particle_mesh_ewald(): Complete PME calculation (real + reciprocal)
    pme_reciprocal_space(): Reciprocal-space FFT-based component only

Helper APIs:
    pme_green_structure_factor(): Green's function and B-spline correction
    pme_energy_corrections(): Self-energy and background corrections

The batch_idx parameter determines kernel dispatch:
    batch_idx=None → Single-system kernels
    batch_idx provided → Batch kernels (multiple independent systems)

MATHEMATICAL FORMULATION
========================

PME uses B-spline interpolation to assign charges to a mesh, computes the
convolution with the Coulomb kernel efficiently via FFT, then interpolates
back to get energies and forces.

.. math::

    E_{\\text{total}} = E_{\\text{real}} + E_{\\text{reciprocal}} - E_{\\text{self}} - E_{\\text{background}}

Reciprocal-Space Steps:

1. Charge assignment:

.. math::

    Q(x) = \\sum_i q_i M_p(x - r_i)

where :math:`M_p` is the pth-order cardinal B-spline

2. FFT:

.. math::

    \\tilde{Q}(k) = \\text{FFT}[Q(x)]

3. Convolution in k-space:

.. math::

    \\tilde{\\Phi}(k) = \\frac{G(k)}{C^2(k)} \\tilde{Q}(k)

where :math:`G(k) = \\frac{2\\pi}{V} \\frac{\\exp(-k^2/(4\\alpha^2))}{k^2}` and :math:`C(k) = [\\text{sinc products}]^p` is the B-spline correction

4. Inverse FFT for potential and field:

.. math::

    \\begin{aligned}
    \\Phi(x) &= \\text{IFFT}[\\tilde{\\Phi}(k)] \\\\
    E(x) &= \\text{IFFT}[-ik \\tilde{\\Phi}(k)]
    \\end{aligned}

5. Energy and force interpolation:

.. math::

    \\begin{aligned}
    E_i &= q_i \\cdot \\text{interpolate}(\\Phi, r_i) \\\\
    F_i &= q_i \\cdot \\text{interpolate}(E, r_i)
    \\end{aligned}

Corrections:

.. math::

    \\begin{aligned}
    E_{\\text{self}} &= \\sum_i \\frac{\\alpha}{\\sqrt{\\pi}} q_i^2 \\\\
    E_{\\text{background}} &= \\sum_i \\frac{\\pi}{2\\alpha^2 V} q_i Q_{\\text{total}}
    \\end{aligned}

USAGE EXAMPLES
==============

Automatic parameter estimation::

    >>> from nvalchemiops.interactions.electrostatics import particle_mesh_ewald
    >>> energies, forces = particle_mesh_ewald(
    ...     positions, charges, cell,
    ...     neighbor_list=nl, neighbor_shifts=shifts,
    ...     accuracy=1e-6,  # alpha and mesh estimated automatically
    ... )

Explicit parameters::

    >>> energies, forces = particle_mesh_ewald(
    ...     positions, charges, cell,
    ...     alpha=0.3,
    ...     mesh_dimensions=(32, 32, 32),
    ...     spline_order=4,
    ...     neighbor_list=nl, neighbor_shifts=shifts,
    ... )

Batched systems::

    >>> energies, forces = particle_mesh_ewald(
    ...     positions, charges, cells,  # cells shape (B, 3, 3)
    ...     alpha=torch.tensor([0.3, 0.35]),
    ...     batch_idx=batch_idx,
    ...     mesh_dimensions=(32, 32, 32),
    ...     neighbor_list=nl, neighbor_shifts=shifts,
    ... )

Reciprocal-space only (no real-space)::

    >>> energies = pme_reciprocal_space(
    ...     positions, charges, cell,
    ...     alpha=0.3, mesh_dimensions=(32, 32, 32),
    ... )

REFERENCES
==========

- Essmann et al. (1995). J. Chem. Phys. 103, 8577 (SPME paper)
- Darden et al. (1993). J. Chem. Phys. 98, 10089 (Original PME)
- torchpme: https://github.com/lab-cosmo/torch-pme (Reference implementation)
"""

import math
from typing import Any

import torch
import warp as wp

from nvalchemiops.autograd import (
    OutputSpec,
    WarpAutogradContextManager,
    attach_for_backward,
    needs_grad,
    warp_custom_op,
    warp_from_torch,
)
from nvalchemiops.interactions.electrostatics.ewald import ewald_real_space
from nvalchemiops.interactions.electrostatics.k_vectors import generate_k_vectors_pme
from nvalchemiops.interactions.electrostatics.parameters import (
    estimate_pme_mesh_dimensions,
    estimate_pme_parameters,
    mesh_spacing_to_dimensions,
)
from nvalchemiops.interactions.electrostatics.pme_kernels import (
    _batch_pme_energy_corrections_kernel_overload,
    _batch_pme_energy_corrections_with_charge_grad_kernel_overload,
    _batch_pme_green_structure_factor_kernel_overload,
    _pme_energy_corrections_kernel_overload,
    _pme_energy_corrections_with_charge_grad_kernel_overload,
    _pme_green_structure_factor_kernel_overload,
)
from nvalchemiops.spline import (
    spline_gather,
    spline_gather_vec3,
    spline_spread,
)
from nvalchemiops.types import get_wp_dtype

# Mathematical constants
PI = math.pi
TWOPI = 2.0 * PI
FOURPI = 4.0 * PI


###########################################################################################
########################### Helper Functions ##############################################
###########################################################################################


def _prepare_alpha(
    alpha: float | torch.Tensor,
    num_systems: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Convert alpha to a per-system tensor.

    Parameters
    ----------
    alpha : float or torch.Tensor
        Ewald splitting parameter. Can be:
        - A scalar float (broadcast to all systems)
        - A 0-d tensor (broadcast to all systems)
        - A 1-d tensor of shape (num_systems,) for per-system values
    num_systems : int
        Number of systems in the batch.
    dtype : torch.dtype
        Target dtype (typically float64).
    device : torch.device
        Target device.

    Returns
    -------
    torch.Tensor, shape (num_systems,)
        Per-system alpha values.
    """
    if isinstance(alpha, (int, float)):
        return torch.full((num_systems,), float(alpha), dtype=dtype, device=device)
    elif isinstance(alpha, torch.Tensor):
        if alpha.dim() == 0:
            return alpha.expand(num_systems).to(dtype=dtype, device=device)
        elif alpha.shape[0] != num_systems:
            raise ValueError(
                f"alpha has {alpha.shape[0]} values but there are {num_systems} systems"
            )
        return alpha.to(dtype=dtype, device=device)
    else:
        raise TypeError(f"alpha must be float or torch.Tensor, got {type(alpha)}")


def _prepare_cell(cell: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Ensure cell is 3D (B, 3, 3) and return number of systems.

    Parameters
    ----------
    cell : torch.Tensor
        Unit cell matrix. Shape (3, 3) for single system or (B, 3, 3) for batch.

    Returns
    -------
    cell : torch.Tensor, shape (B, 3, 3)
        Cell with batch dimension.
    num_systems : int
        Number of systems (B).
    """
    if cell.dim() == 2:
        cell = cell.unsqueeze(0)
    return cell, cell.shape[0]


###########################################################################################
########################### Green Function & Structure Factor Custom Ops ##################
###########################################################################################


def _green_output_shape(k_squared, *_):
    """Helper to compute output shape for Green's function.

    Uses k_squared.shape directly since it already has shape (Nx, Ny, Nz_rfft).
    """
    return k_squared.shape


def _struct_output_shape(k_squared, *_):
    """Helper to compute output shape for structure factor.

    Uses k_squared.shape directly since it already has shape (Nx, Ny, Nz_rfft).
    """
    return k_squared.shape


@warp_custom_op(
    name="alchemiops::_pme_green_structure_factor",
    outputs=[
        OutputSpec("green_function", wp.array(dtype=Any, ndim=3), _green_output_shape),
        OutputSpec(
            "structure_factor_sq", wp.array(dtype=Any, ndim=3), _struct_output_shape
        ),
    ],
    grad_arrays=["green_function", "k_squared", "alpha", "volume"],
)
def _pme_green_structure_factor(
    k_squared: torch.Tensor,
    miller_x: torch.Tensor,
    miller_y: torch.Tensor,
    miller_z: torch.Tensor,
    alpha: torch.Tensor,
    volume: torch.Tensor,
    mesh_nx: int,
    mesh_ny: int,
    mesh_nz: int,
    spline_order: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Compute Green's function and structure factor for single-system PME.

    The Green's function includes volume normalization:

    .. math::

        G(k) = \frac{4\pi \exp(-k^2/(4\alpha^2))}{V k^2}

    Supports both float32 and float64 dtypes via kernel overloads.

    Parameters
    ----------
    k_squared : torch.Tensor, shape (Nx, Ny, Nz_rfft)
        :math:`|k|^2` for each grid point.
    miller_x : torch.Tensor, shape (Nx,)
        Miller indices in x direction.
    miller_y : torch.Tensor, shape (Ny,)
        Miller indices in y direction.
    miller_z : torch.Tensor, shape (Nz_rfft,)
        Miller indices in z direction.
    alpha : torch.Tensor, shape (1,)
        Ewald splitting parameter.
    volume : torch.Tensor, shape (1,)
        Cell volume.
    mesh_nx, mesh_ny, mesh_nz : int
        Full mesh dimensions.
    spline_order : int
        B-spline order.

    Returns
    -------
    green_function : torch.Tensor, shape (Nx, Ny, Nz_rfft)
        Green's function values (volume-normalized).
    structure_factor_sq : torch.Tensor, shape (Nx, Ny, Nz_rfft)
        Structure factor squared.
    """
    device = wp.device_from_torch(k_squared.device)
    input_dtype = k_squared.dtype
    wp_dtype = get_wp_dtype(input_dtype)
    nx, ny, nz_rfft = k_squared.shape
    needs_grad_flag = needs_grad(k_squared, alpha, volume)

    # Prepare inputs with appropriate dtype
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
    wp_alpha = warp_from_torch(
        alpha.to(input_dtype).contiguous(), wp_dtype, requires_grad=needs_grad_flag
    )
    wp_volume = warp_from_torch(
        volume.to(input_dtype).contiguous(), wp_dtype, requires_grad=needs_grad_flag
    )

    # Allocate outputs with input dtype
    green_function = torch.zeros(
        (nx, ny, nz_rfft), dtype=input_dtype, device=k_squared.device
    )
    structure_factor_sq = torch.zeros(
        (nx, ny, nz_rfft), dtype=input_dtype, device=k_squared.device
    )

    wp_green = warp_from_torch(green_function, wp_dtype, requires_grad=needs_grad_flag)
    wp_struct = warp_from_torch(structure_factor_sq, wp_dtype, requires_grad=False)

    # Select kernel based on dtype
    kernel = _pme_green_structure_factor_kernel_overload[wp_dtype]

    with WarpAutogradContextManager(needs_grad_flag) as tape:
        wp.launch(
            kernel,
            dim=(nx, ny, nz_rfft),
            inputs=[
                wp_k_squared,
                wp_miller_x,
                wp_miller_y,
                wp_miller_z,
                wp_alpha,
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
            alpha=wp_alpha,
            volume=wp_volume,
        )

    return green_function, structure_factor_sq


def _batch_green_output_shape(
    k_squared, miller_x, miller_y, miller_z, alpha, volumes, *_
):
    """Helper to compute output shapes for batch Green's function."""
    if k_squared.dim() == 3:
        _, nx, ny, nz_rfft = (1,) + k_squared.shape
    else:
        _, nx, ny, nz_rfft = k_squared.shape
    num_systems = volumes.shape[0]
    return (num_systems, nx, ny, nz_rfft)


def _batch_struct_output_shape(k_squared, *_):
    """Helper to compute output shape for structure factor in batch case."""
    if k_squared.dim() == 3:
        return k_squared.shape
    else:
        return k_squared.shape[1:]  # Remove batch dim


@warp_custom_op(
    name="alchemiops::_batch_pme_green_structure_factor",
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
    grad_arrays=["green_function", "k_squared", "alpha", "volumes"],
)
def _batch_pme_green_structure_factor(
    k_squared: torch.Tensor,
    miller_x: torch.Tensor,
    miller_y: torch.Tensor,
    miller_z: torch.Tensor,
    alpha: torch.Tensor,
    volumes: torch.Tensor,
    mesh_nx: int,
    mesh_ny: int,
    mesh_nz: int,
    spline_order: int,
    num_systems: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Green's function and structure factor for batch PME.

    The Green's function includes volume normalization:

    .. math::

        G(k) = \\frac{4\\pi \\exp(-k^2/(4\\alpha^2))}{V k^2}

    Supports both float32 and float64 dtypes via kernel overloads.

    Parameters
    ----------
    k_squared : torch.Tensor, shape (B, Nx, Ny, Nz_rfft)
        :math:`|k|^2` for each grid point per system.
    miller_x, miller_y, miller_z : torch.Tensor
        Miller indices for each dimension.
    alpha : torch.Tensor, shape (B,)
        Per-system Ewald splitting parameter.
    volumes : torch.Tensor, shape (B,)
        Per-system cell volumes.
    num_systems : int
        Number of systems.

    Returns
    -------
    green_function : torch.Tensor, shape (B, Nx, Ny, Nz_rfft)
        Green's function values per system (volume-normalized).
    structure_factor_sq : torch.Tensor, shape (Nx, Ny, Nz_rfft)
        Structure factor :math:`C^2(k)` squared (same for all systems).
    """
    device = wp.device_from_torch(k_squared.device)
    if k_squared.dim() == 3:
        k_squared = k_squared.unsqueeze(0)
    input_dtype = k_squared.dtype
    wp_dtype = get_wp_dtype(input_dtype)
    _, nx, ny, nz_rfft = k_squared.shape
    needs_grad_flag = needs_grad(k_squared, alpha, volumes)

    # Prepare inputs with appropriate dtype
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
    wp_alpha = warp_from_torch(
        alpha.to(input_dtype).contiguous(), wp_dtype, requires_grad=needs_grad_flag
    )
    wp_volumes = warp_from_torch(
        volumes.to(input_dtype).contiguous(), wp_dtype, requires_grad=needs_grad_flag
    )

    # Allocate outputs with input dtype
    green_function = torch.zeros(
        (num_systems, nx, ny, nz_rfft), dtype=input_dtype, device=k_squared.device
    )
    structure_factor_sq = torch.zeros(
        (nx, ny, nz_rfft), dtype=input_dtype, device=k_squared.device
    )

    wp_green = warp_from_torch(green_function, wp_dtype, requires_grad=needs_grad_flag)
    wp_struct = warp_from_torch(structure_factor_sq, wp_dtype, requires_grad=False)

    # Select kernel based on dtype
    kernel = _batch_pme_green_structure_factor_kernel_overload[wp_dtype]

    with WarpAutogradContextManager(needs_grad_flag) as tape:
        wp.launch(
            kernel,
            dim=(num_systems, nx, ny, nz_rfft),
            inputs=[
                wp_k_squared,
                wp_miller_x,
                wp_miller_y,
                wp_miller_z,
                wp_alpha,
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
            alpha=wp_alpha,
            volumes=wp_volumes,
        )
    return green_function, structure_factor_sq


def pme_green_structure_factor(
    k_squared: torch.Tensor,
    mesh_dimensions: tuple[int, int, int],
    alpha: torch.Tensor,
    cell: torch.Tensor,
    spline_order: int = 4,
    batch_idx: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Green's function and B-spline structure factor correction.

    Computes the Coulomb Green's function with volume normalization and the
    B-spline aliasing correction factor for PME.

    Green's function (volume-normalized):

    .. math::

        G(k) = \\frac{2\\pi}{V} \\frac{\\exp(-k^2/(4\\alpha^2))}{k^2}

    Structure factor correction (for B-spline deconvolution):

    .. math::

        C^2(k) = \\left[\\text{sinc}(m_x/N_x) \\cdot \\text{sinc}(m_y/N_y) \\cdot \\text{sinc}(m_z/N_z)\\right]^{2p}

    where p is the spline order.

    Supports both float32 and float64 dtypes.

    Parameters
    ----------
    k_squared : torch.Tensor
        :math:`|k|^2` values at each FFT grid point.
        - Single-system: shape (Nx, Ny, Nz_rfft)
        - Batch: shape (B, Nx, Ny, Nz_rfft)
    mesh_dimensions : tuple[int, int, int]
        Full mesh dimensions (Nx, Ny, Nz) before rfft.
    alpha : torch.Tensor
        Ewald splitting parameter.
        - Single-system: shape (1,)
        - Batch: shape (B,)
    cell : torch.Tensor
        Unit cell matrices.
        - Single-system: shape (3, 3) or (1, 3, 3)
        - Batch: shape (B, 3, 3)
    spline_order : int, default=4
        B-spline interpolation order (typically 4 for cubic B-splines).
    batch_idx : torch.Tensor | None, default=None
        If provided, dispatches to batch kernels.

    Returns
    -------
    green_function : torch.Tensor
        Volume-normalized Green's function :math:`G(k)`.
        - Single-system: shape (Nx, Ny, Nz_rfft)
        - Batch: shape (B, Nx, Ny, Nz_rfft)
    structure_factor_sq : torch.Tensor
        Squared structure factor :math:`C^2(k)` for B-spline deconvolution.
        Shape (Nx, Ny, Nz_rfft), shared across batch.

    Notes
    -----
    - :math:`G(k=0)` is set to zero to avoid singularity
    - The volume normalization in :math:`G(k)` eliminates later divisions
    - Structure factor is mesh-dependent only, so shared across batch
    """
    mesh_nx, mesh_ny, mesh_nz = mesh_dimensions
    device = k_squared.device
    input_dtype = k_squared.dtype

    # Ensure cell is correct shape
    cell = cell if cell.dim() == 3 else cell.unsqueeze(0)
    volume = torch.abs(torch.det(cell)).to(input_dtype)

    # Generate Miller indices in input dtype
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
        # Single system
        result = _pme_green_structure_factor(
            k_squared,
            miller_x,
            miller_y,
            miller_z,
            alpha.to(input_dtype),
            volume,
            mesh_nx,
            mesh_ny,
            mesh_nz,
            spline_order,
        )
    else:
        # Batch - num_systems from k_squared shape
        num_systems = cell.shape[0]
        result = _batch_pme_green_structure_factor(
            k_squared,
            miller_x,
            miller_y,
            miller_z,
            alpha.to(input_dtype),
            volume,
            mesh_nx,
            mesh_ny,
            mesh_nz,
            spline_order,
            num_systems,
        )
    return result


###########################################################################################
########################### PME Energy Corrections Custom Ops #############################
###########################################################################################


@warp_custom_op(
    name="alchemiops::_pme_energy_corrections",
    outputs=[
        OutputSpec(
            "corrected_energies",
            wp.array(dtype=Any, ndim=1),
            lambda raw_energies, *_: (raw_energies.shape[0],),
        ),
    ],
    grad_arrays=[
        "corrected_energies",
        "raw_energies",
        "charges",
        "volume",
        "alpha",
        "total_charge",
    ],
)
def _pme_energy_corrections(
    raw_energies: torch.Tensor,
    charges: torch.Tensor,
    volume: torch.Tensor,
    alpha: torch.Tensor,
    total_charge: torch.Tensor,
) -> torch.Tensor:
    """Apply self-energy and background corrections to PME energies.

    Uses unified prefactors. For energy-only calculations,
    the caller should multiply by 0.5 at the end.

    Supports both float32 and float64 dtypes via kernel overloads.

    Parameters
    ----------
    raw_energies : torch.Tensor, shape (N,)
        Raw interpolated energies from potential mesh.
    charges : torch.Tensor, shape (N,)
        Atomic charges.
    volume : torch.Tensor, shape (1,)
        Cell volume.
    alpha : torch.Tensor, shape (1,)
        Ewald splitting parameter.
    total_charge : torch.Tensor, shape (1,)
        Total system charge.

    Returns
    -------
    corrected_energies : torch.Tensor, shape (N,)
        Corrected energies per atom.
    """
    device = wp.device_from_torch(raw_energies.device)
    input_dtype = raw_energies.dtype
    wp_dtype = get_wp_dtype(input_dtype)
    num_atoms = raw_energies.shape[0]
    needs_grad_flag = needs_grad(raw_energies, charges, volume, alpha, total_charge)

    wp_raw = warp_from_torch(
        raw_energies.contiguous(), wp_dtype, requires_grad=needs_grad_flag
    )
    wp_charges = warp_from_torch(
        charges.to(input_dtype).contiguous(), wp_dtype, requires_grad=needs_grad_flag
    )
    wp_volume = warp_from_torch(
        volume.to(input_dtype).contiguous(), wp_dtype, requires_grad=needs_grad_flag
    )
    wp_alpha = warp_from_torch(
        alpha.to(input_dtype).contiguous(), wp_dtype, requires_grad=needs_grad_flag
    )
    wp_total_charge = warp_from_torch(
        total_charge.to(input_dtype).contiguous(),
        wp_dtype,
        requires_grad=needs_grad_flag,
    )
    corrected_energies = torch.zeros(
        num_atoms, dtype=input_dtype, device=raw_energies.device
    )
    wp_corrected = warp_from_torch(
        corrected_energies, wp_dtype, requires_grad=needs_grad_flag
    )

    # Select kernel based on dtype
    kernel = _pme_energy_corrections_kernel_overload[wp_dtype]

    with WarpAutogradContextManager(needs_grad_flag) as tape:
        wp.launch(
            kernel,
            dim=num_atoms,
            inputs=[
                wp_raw,
                wp_charges,
                wp_volume,
                wp_alpha,
                wp_total_charge,
            ],
            outputs=[wp_corrected],
            device=device,
        )

    if needs_grad_flag:
        attach_for_backward(
            corrected_energies,
            tape=tape,
            corrected_energies=wp_corrected,
            raw_energies=wp_raw,
            charges=wp_charges,
            volume=wp_volume,
            alpha=wp_alpha,
            total_charge=wp_total_charge,
        )
    return corrected_energies


@warp_custom_op(
    name="alchemiops::_batch_pme_energy_corrections",
    outputs=[
        OutputSpec(
            "corrected_energies",
            wp.array(dtype=Any, ndim=1),
            lambda raw_energies, *_: (raw_energies.shape[0],),
        ),
    ],
    grad_arrays=[
        "corrected_energies",
        "raw_energies",
        "charges",
        "volumes",
        "alpha",
        "total_charges",
    ],
)
def _batch_pme_energy_corrections(
    raw_energies: torch.Tensor,
    charges: torch.Tensor,
    batch_idx: torch.Tensor,
    volumes: torch.Tensor,
    alpha: torch.Tensor,
    total_charges: torch.Tensor,
) -> torch.Tensor:
    """Apply corrections for batch PME.

    Uses unified prefactors. For energy-only calculations,
    the caller should multiply by 0.5 at the end.

    Supports both float32 and float64 dtypes via kernel overloads.

    Parameters
    ----------
    raw_energies : torch.Tensor, shape (N_total,)
        Raw interpolated energies.
    charges : torch.Tensor, shape (N_total,)
        Atomic charges.
    batch_idx : torch.Tensor, shape (N_total,)
        System index for each atom.
    volumes : torch.Tensor, shape (B,)
        Cell volumes per system.
    alpha : torch.Tensor, shape (B,)
        Per-system Ewald splitting parameter.
    total_charges : torch.Tensor, shape (B,)
        Total charge per system.

    Returns
    -------
    corrected_energies : torch.Tensor, shape (N_total,)
        Corrected energies per atom.
    """
    device = wp.device_from_torch(raw_energies.device)
    input_dtype = raw_energies.dtype
    wp_dtype = get_wp_dtype(input_dtype)
    num_atoms = raw_energies.shape[0]
    needs_grad_flag = needs_grad(raw_energies, charges, volumes, alpha, total_charges)

    wp_raw = warp_from_torch(
        raw_energies.contiguous(), wp_dtype, requires_grad=needs_grad_flag
    )
    wp_charges = warp_from_torch(
        charges.to(input_dtype).contiguous(), wp_dtype, requires_grad=needs_grad_flag
    )
    wp_batch_idx = warp_from_torch(
        batch_idx.contiguous(), wp.int32, requires_grad=False
    )
    wp_volumes = warp_from_torch(
        volumes.to(input_dtype).contiguous(), wp_dtype, requires_grad=needs_grad_flag
    )
    wp_alpha = warp_from_torch(
        alpha.to(input_dtype).contiguous(), wp_dtype, requires_grad=needs_grad_flag
    )
    wp_total_charges = warp_from_torch(
        total_charges.to(input_dtype).contiguous(),
        wp_dtype,
        requires_grad=needs_grad_flag,
    )
    corrected_energies = torch.zeros(
        num_atoms, dtype=input_dtype, device=raw_energies.device
    )
    wp_corrected = warp_from_torch(
        corrected_energies, wp_dtype, requires_grad=needs_grad_flag
    )

    # Select kernel based on dtype
    kernel = _batch_pme_energy_corrections_kernel_overload[wp_dtype]

    with WarpAutogradContextManager(needs_grad_flag) as tape:
        wp.launch(
            kernel,
            dim=num_atoms,
            inputs=[
                wp_raw,
                wp_charges,
                wp_batch_idx,
                wp_volumes,
                wp_alpha,
                wp_total_charges,
            ],
            outputs=[wp_corrected],
            device=device,
        )

    if needs_grad_flag:
        attach_for_backward(
            corrected_energies,
            tape=tape,
            corrected_energies=wp_corrected,
            raw_energies=wp_raw,
            charges=wp_charges,
            volumes=wp_volumes,
            alpha=wp_alpha,
            total_charges=wp_total_charges,
        )
    return corrected_energies


@warp_custom_op(
    name="alchemiops::_pme_energy_corrections_with_charge_grad",
    outputs=[
        OutputSpec(
            "corrected_energies",
            wp.float64,
            lambda raw_energies, *_: (raw_energies.shape[0],),
        ),
        OutputSpec(
            "charge_gradients",
            wp.float64,
            lambda raw_energies, *_: (raw_energies.shape[0],),
        ),
    ],
    grad_arrays=[
        "corrected_energies",
        "charge_gradients",
        "raw_energies",
        "charges",
        "volume",
        "alpha",
        "total_charge",
    ],
)
def _pme_energy_corrections_with_charge_grad(
    raw_energies: torch.Tensor,
    charges: torch.Tensor,
    volume: torch.Tensor,
    alpha: torch.Tensor,
    total_charge: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply self-energy and background corrections and compute charge gradients.

    Computes both corrected energies and analytical charge gradients for PME.

    Parameters
    ----------
    raw_energies : torch.Tensor, shape (N,)
        Raw interpolated potential φ_i from mesh.
    charges : torch.Tensor, shape (N,)
        Atomic charges.
    volume : torch.Tensor, shape (1,)
        Cell volume.
    alpha : torch.Tensor, shape (1,)
        Ewald splitting parameter.
    total_charge : torch.Tensor, shape (1,)
        Total system charge.

    Returns
    -------
    corrected_energies : torch.Tensor, shape (N,)
        Corrected energies per atom.
    charge_gradients : torch.Tensor, shape (N,)
        Analytical charge gradients ∂E/∂q_i.
    """
    device = wp.device_from_torch(raw_energies.device)
    input_dtype = raw_energies.dtype
    wp_dtype = get_wp_dtype(input_dtype)
    num_atoms = raw_energies.shape[0]
    needs_grad_flag = needs_grad(raw_energies, charges, volume, alpha, total_charge)

    wp_raw = warp_from_torch(
        raw_energies.contiguous(), wp_dtype, requires_grad=needs_grad_flag
    )
    wp_charges = warp_from_torch(
        charges.to(input_dtype).contiguous(), wp_dtype, requires_grad=needs_grad_flag
    )
    wp_volume = warp_from_torch(
        volume.to(input_dtype).contiguous(), wp_dtype, requires_grad=needs_grad_flag
    )
    wp_alpha = warp_from_torch(
        alpha.to(input_dtype).contiguous(), wp_dtype, requires_grad=needs_grad_flag
    )
    wp_total_charge = warp_from_torch(
        total_charge.to(input_dtype).contiguous(),
        wp_dtype,
        requires_grad=needs_grad_flag,
    )

    corrected_energies = torch.zeros(
        num_atoms, dtype=input_dtype, device=raw_energies.device
    )
    charge_gradients = torch.zeros(
        num_atoms, dtype=input_dtype, device=raw_energies.device
    )

    wp_corrected = warp_from_torch(
        corrected_energies, wp_dtype, requires_grad=needs_grad_flag
    )
    wp_charge_grads = warp_from_torch(
        charge_gradients, wp_dtype, requires_grad=needs_grad_flag
    )

    kernel = _pme_energy_corrections_with_charge_grad_kernel_overload[wp_dtype]

    with WarpAutogradContextManager(needs_grad_flag) as tape:
        wp.launch(
            kernel,
            dim=num_atoms,
            inputs=[
                wp_raw,
                wp_charges,
                wp_volume,
                wp_alpha,
                wp_total_charge,
            ],
            outputs=[wp_corrected, wp_charge_grads],
            device=device,
        )

    if needs_grad_flag:
        attach_for_backward(
            corrected_energies,
            tape=tape,
            corrected_energies=wp_corrected,
            charge_gradients=wp_charge_grads,
            raw_energies=wp_raw,
            charges=wp_charges,
            volume=wp_volume,
            alpha=wp_alpha,
            total_charge=wp_total_charge,
        )

    return corrected_energies, charge_gradients


@warp_custom_op(
    name="alchemiops::_batch_pme_energy_corrections_with_charge_grad",
    outputs=[
        OutputSpec(
            "corrected_energies",
            wp.float64,
            lambda raw_energies, *_: (raw_energies.shape[0],),
        ),
        OutputSpec(
            "charge_gradients",
            wp.float64,
            lambda raw_energies, *_: (raw_energies.shape[0],),
        ),
    ],
    grad_arrays=[
        "corrected_energies",
        "charge_gradients",
        "raw_energies",
        "charges",
        "volumes",
        "alpha",
        "total_charges",
    ],
)
def _batch_pme_energy_corrections_with_charge_grad(
    raw_energies: torch.Tensor,
    charges: torch.Tensor,
    batch_idx: torch.Tensor,
    volumes: torch.Tensor,
    alpha: torch.Tensor,
    total_charges: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply corrections and compute charge gradients for batch PME.

    Parameters
    ----------
    raw_energies : torch.Tensor, shape (N_total,)
        Raw interpolated potential.
    charges : torch.Tensor, shape (N_total,)
        Atomic charges.
    batch_idx : torch.Tensor, shape (N_total,)
        System index for each atom.
    volumes : torch.Tensor, shape (B,)
        Cell volumes per system.
    alpha : torch.Tensor, shape (B,)
        Per-system Ewald splitting parameter.
    total_charges : torch.Tensor, shape (B,)
        Total charge per system.

    Returns
    -------
    corrected_energies : torch.Tensor, shape (N_total,)
        Corrected energies per atom.
    charge_gradients : torch.Tensor, shape (N_total,)
        Analytical charge gradients ∂E/∂q_i.
    """
    device = wp.device_from_torch(raw_energies.device)
    input_dtype = raw_energies.dtype
    wp_dtype = get_wp_dtype(input_dtype)
    num_atoms = raw_energies.shape[0]
    needs_grad_flag = needs_grad(raw_energies, charges, volumes, alpha, total_charges)

    wp_raw = warp_from_torch(
        raw_energies.contiguous(), wp_dtype, requires_grad=needs_grad_flag
    )
    wp_charges = warp_from_torch(
        charges.to(input_dtype).contiguous(), wp_dtype, requires_grad=needs_grad_flag
    )
    wp_batch_idx = warp_from_torch(
        batch_idx.contiguous(), wp.int32, requires_grad=False
    )
    wp_volumes = warp_from_torch(
        volumes.to(input_dtype).contiguous(), wp_dtype, requires_grad=needs_grad_flag
    )
    wp_alpha = warp_from_torch(
        alpha.to(input_dtype).contiguous(), wp_dtype, requires_grad=needs_grad_flag
    )
    wp_total_charges = warp_from_torch(
        total_charges.to(input_dtype).contiguous(),
        wp_dtype,
        requires_grad=needs_grad_flag,
    )

    corrected_energies = torch.zeros(
        num_atoms, dtype=input_dtype, device=raw_energies.device
    )
    charge_gradients = torch.zeros(
        num_atoms, dtype=input_dtype, device=raw_energies.device
    )

    wp_corrected = warp_from_torch(
        corrected_energies, wp_dtype, requires_grad=needs_grad_flag
    )
    wp_charge_grads = warp_from_torch(
        charge_gradients, wp_dtype, requires_grad=needs_grad_flag
    )

    kernel = _batch_pme_energy_corrections_with_charge_grad_kernel_overload[wp_dtype]

    with WarpAutogradContextManager(needs_grad_flag) as tape:
        wp.launch(
            kernel,
            dim=num_atoms,
            inputs=[
                wp_raw,
                wp_charges,
                wp_batch_idx,
                wp_volumes,
                wp_alpha,
                wp_total_charges,
            ],
            outputs=[wp_corrected, wp_charge_grads],
            device=device,
        )

    if needs_grad_flag:
        attach_for_backward(
            corrected_energies,
            tape=tape,
            corrected_energies=wp_corrected,
            charge_gradients=wp_charge_grads,
            raw_energies=wp_raw,
            charges=wp_charges,
            volumes=wp_volumes,
            alpha=wp_alpha,
            total_charges=wp_total_charges,
        )

    return corrected_energies, charge_gradients


def pme_energy_corrections(
    raw_energies: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    alpha: torch.Tensor,
    batch_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply self-energy and background corrections to PME energies.

    Converts raw interpolated potential to energy and subtracts corrections:

    .. math::

        E_i = q_i \\phi_i - E_{\\text{self},i} - E_{\\text{background},i}

    Self-energy correction (removes Gaussian self-interaction):

    .. math::

        E_{\\text{self},i} = \\frac{\\alpha}{\\sqrt{\\pi}} q_i^2

    Background correction (for non-neutral systems):

    .. math::

        E_{\\text{background},i} = \\frac{\\pi}{2\\alpha^2 V} q_i Q_{\\text{total}}

    Parameters
    ----------
    raw_energies : torch.Tensor, shape (N,) or (N_total,)
        Raw potential values :math:`\\phi_i` from mesh interpolation.
    charges : torch.Tensor, shape (N,) or (N_total,)
        Atomic charges.
    cell : torch.Tensor
        Unit cell matrices.
        - Single-system: shape (3, 3) or (1, 3, 3)
        - Batch: shape (B, 3, 3)
    alpha : torch.Tensor
        Ewald splitting parameter.
        - Single-system: shape (1,)
        - Batch: shape (B,)
    batch_idx : torch.Tensor | None, default=None
        System index for each atom. If provided, uses batch kernels.

    Returns
    -------
    corrected_energies : torch.Tensor, shape (N,) or (N_total,)
        Final per-atom reciprocal-space energy with corrections applied.

    Notes
    -----
    - For neutral systems, background correction is zero
    - Matches torchpme's self_contribution and background_correction formulas
    - Supports both float32 and float64 dtypes
    """
    input_dtype = raw_energies.dtype

    if batch_idx is None:
        # Single system - ensure tensors are 1D for kernel indexing
        total_charge = charges.sum().reshape(1)
        volume = torch.abs(torch.det(cell)).reshape(1)

        result = _pme_energy_corrections(
            raw_energies,
            charges.to(input_dtype),
            volume.to(input_dtype),
            alpha.to(input_dtype),
            total_charge.to(input_dtype),
        )
    else:
        # Batch
        num_systems = cell.shape[0]
        volumes = torch.abs(torch.linalg.det(cell)).to(input_dtype)

        # Compute total charge per system
        total_charges = torch.zeros(
            num_systems, dtype=input_dtype, device=raw_energies.device
        )
        total_charges.scatter_add_(0, batch_idx, charges.to(input_dtype))

        result = _batch_pme_energy_corrections(
            raw_energies,
            charges.to(input_dtype),
            batch_idx,
            volumes,
            alpha.to(input_dtype),
            total_charges,
        )

    return result


def pme_energy_corrections_with_charge_grad(
    raw_energies: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    alpha: torch.Tensor,
    batch_idx: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply corrections and compute charge gradients for PME energies.

    Computes both corrected energies and analytical charge gradients:
        E_i = q_i * φ_i - E_self_i - E_background_i
        ∂E/∂q_i = 2*φ_i - 2*(α/√π)*q_i - (π/(α²V))*Q_total

    The factor of 2 on φ_i arises because changing q_i affects both the
    direct energy term (q_i * φ_i) and all other potentials through the
    structure factor (∑_j q_j * ∂φ_j/∂q_i = φ_i).

    Parameters
    ----------
    raw_energies : torch.Tensor, shape (N,) or (N_total,)
        Raw potential values φ_i from mesh interpolation.
    charges : torch.Tensor, shape (N,) or (N_total,)
        Atomic charges.
    cell : torch.Tensor
        Unit cell matrices.
        - Single-system: shape (3, 3) or (1, 3, 3)
        - Batch: shape (B, 3, 3)
    alpha : torch.Tensor
        Ewald splitting parameter.
        - Single-system: shape (1,)
        - Batch: shape (B,)
    batch_idx : torch.Tensor | None, default=None
        System index for each atom. If provided, uses batch kernels.

    Returns
    -------
    corrected_energies : torch.Tensor, shape (N,) or (N_total,)
        Final per-atom reciprocal-space energy with corrections applied.
    charge_gradients : torch.Tensor, shape (N,) or (N_total,)
        Analytical charge gradients ∂E/∂q_i.
    """
    input_dtype = raw_energies.dtype

    if batch_idx is None:
        # Single system
        total_charge = charges.sum().reshape(1)
        volume = torch.abs(torch.det(cell)).reshape(1)
        return _pme_energy_corrections_with_charge_grad(
            raw_energies,
            charges.to(input_dtype),
            volume.to(input_dtype),
            alpha.to(input_dtype),
            total_charge.to(input_dtype),
        )
    else:
        # Batch
        num_systems = cell.shape[0]
        volumes = torch.abs(torch.linalg.det(cell)).to(input_dtype)

        # Compute total charge per system
        total_charges = torch.zeros(
            num_systems, dtype=input_dtype, device=raw_energies.device
        )
        total_charges.scatter_add_(0, batch_idx, charges.to(input_dtype))

        return _batch_pme_energy_corrections_with_charge_grad(
            raw_energies,
            charges.to(input_dtype),
            batch_idx,
            volumes,
            alpha.to(input_dtype),
            total_charges,
        )


###########################################################################################
########################### Unified PME Reciprocal Space ##################################
###########################################################################################


def _pme_reciprocal_space_impl(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    alpha: torch.Tensor,
    mesh_dimensions: tuple[int, int, int],
    spline_order: int,
    batch_idx: torch.Tensor | None,
    compute_forces: bool = False,
    compute_charge_gradients: bool = False,
    k_vectors: torch.Tensor | None = None,
    k_squared: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Internal implementation of PME reciprocal space calculation.

    Uses unified spline functions from nvalchemiops.spline for charge assignment
    and potential interpolation, and Warp kernels for Green's function and corrections.

    Supports both float32 and float64 dtypes - all operations are performed
    in the input dtype without conversion.
    """
    device = positions.device
    input_dtype = positions.dtype
    num_atoms = positions.shape[0]
    is_batch = batch_idx is not None
    fft_dims = (1, 2, 3) if is_batch else (0, 1, 2)

    if num_atoms == 0:
        energies = torch.zeros(num_atoms, device=device, dtype=input_dtype)
        forces = (
            torch.zeros(num_atoms, 3, device=device, dtype=input_dtype)
            if compute_forces
            else None
        )
        charge_grads = (
            torch.zeros(num_atoms, device=device, dtype=input_dtype)
            if compute_charge_gradients
            else None
        )
        return energies, forces, charge_grads

    mesh_nx, mesh_ny, mesh_nz = mesh_dimensions

    # Precompute cell inverse ONCE and derive what we need for all operations
    cell_inv = torch.linalg.inv_ex(cell)[0]
    cell_inv_t = cell_inv.transpose(-1, -2).contiguous()
    reciprocal_cell = TWOPI * cell_inv

    # Step 1: Charge assignment using unified spline_spread API
    mesh_grid = spline_spread(
        positions,
        charges,
        cell,
        mesh_dims=(mesh_nx, mesh_ny, mesh_nz),
        spline_order=spline_order,
        batch_idx=batch_idx,
        cell_inv_t=cell_inv_t,
    )

    # Step 2: FFT of charge mesh
    mesh_fft = torch.fft.rfftn(mesh_grid, norm="backward", dim=fft_dims)

    # Step 3: Generate k-space grid and compute Green's function + structure factor
    # Green's function now includes volume normalization: G(k) = 4*pi * exp(-k^2/(2*alpha^2)) / (V * k^2)
    # Use precomputed k_vectors/k_squared if provided, otherwise generate them
    if k_vectors is None or k_squared is None:
        k_vectors, k_squared = generate_k_vectors_pme(
            cell, mesh_dimensions=mesh_dimensions, reciprocal_cell=reciprocal_cell
        )

    green_function, structure_factor_sq = pme_green_structure_factor(
        k_squared,
        mesh_dimensions,
        alpha,
        cell,
        spline_order,
        batch_idx=batch_idx,
    )

    # Step 4: Apply B-spline deconvolution and convolve with Green's function
    mesh_fft = mesh_fft / structure_factor_sq
    convolved_mesh = mesh_fft * green_function

    # Step 5: Inverse FFT to get potential mesh
    potential_mesh = torch.fft.irfftn(
        convolved_mesh, norm="forward", s=mesh_dimensions, dim=fft_dims
    )
    potential_mesh = potential_mesh.to(input_dtype)

    # Step 6: Interpolate potential to atomic positions using unified spline_gather API
    # Note: raw_energies are already volume-normalized from Green's function
    raw_energies = spline_gather(
        positions,
        potential_mesh,
        cell,
        spline_order=spline_order,
        batch_idx=batch_idx,
        cell_inv_t=cell_inv_t,
    )

    # Step 7: Apply corrections using Warp kernel
    # Use charge gradient version if requested
    charge_grads = None
    if compute_charge_gradients:
        reciprocal_energies, charge_grads = pme_energy_corrections_with_charge_grad(
            raw_energies, charges, cell, alpha, batch_idx
        )
    else:
        reciprocal_energies = pme_energy_corrections(
            raw_energies, charges, cell, alpha, batch_idx
        )

    # Step 8: Compute forces if needed
    forces = None
    if compute_forces:
        # Compute electric field by taking gradient in Fourier space
        # Note: convolved_mesh is already volume-normalized from Green's function
        Ex_fft = -1j * k_vectors[..., 0] * convolved_mesh
        Ey_fft = -1j * k_vectors[..., 1] * convolved_mesh
        Ez_fft = -1j * k_vectors[..., 2] * convolved_mesh

        Ex = torch.fft.irfftn(Ex_fft, norm="forward", s=mesh_dimensions, dim=fft_dims)
        Ey = torch.fft.irfftn(Ey_fft, norm="forward", s=mesh_dimensions, dim=fft_dims)
        Ez = torch.fft.irfftn(Ez_fft, norm="forward", s=mesh_dimensions, dim=fft_dims)

        electric_field_mesh = torch.stack([Ex, Ey, Ez], dim=-1).to(input_dtype)

        # Use unified spline_gather_vec3 API to interpolate electric field
        interpolated_field = spline_gather_vec3(
            positions,
            charges,
            electric_field_mesh,
            cell,
            spline_order=spline_order,
            batch_idx=batch_idx,
            cell_inv_t=cell_inv_t,
        )

        # Compute forces: F = 2 * q * E / V
        forces = 2.0 * interpolated_field

    return reciprocal_energies, forces, charge_grads


def pme_reciprocal_space(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    alpha: float | torch.Tensor,
    mesh_dimensions: tuple[int, int, int] | None = None,
    mesh_spacing: float | None = None,
    spline_order: int = 4,
    batch_idx: torch.Tensor | None = None,
    k_vectors: torch.Tensor | None = None,
    k_squared: torch.Tensor | None = None,
    compute_forces: bool = False,
    compute_charge_gradients: bool = False,
) -> (
    torch.Tensor
    | tuple[torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
):
    """Compute PME reciprocal-space energy and optionally forces and/or charge gradients.

    Performs the FFT-based reciprocal-space calculation using the Particle Mesh
    Ewald algorithm. This achieves O(N log N) scaling through:

    1. B-spline charge interpolation to mesh (spreading)
    2. FFT of charge mesh to reciprocal space
    3. Convolution with Green's function (multiply by G(k))
    4. Inverse FFT back to real space (potential mesh)
    5. B-spline interpolation of potential to atoms (gathering)
    6. Self-energy and background corrections

    Formula
    -------
    The reciprocal-space energy is computed via the mesh potential:
    .. math::
        \\varphi_{\\text{mesh}}(k) = G(k) \\times B^2(k) \\times \\rho_{\\text{mesh}}(k)

    where:
    .. math::
        G(k) = (4\\pi/k^2) \\times exp(-k^2/(4\\alpha^2))   Green's function
        B(k) = B-spline structure factor   Interpolation correction
    .. math::
        \\rho_{\\text{mesh}}(k) = FFT of interpolated charges

    Parameters
    ----------
    positions : torch.Tensor, shape (N, 3)
        Atomic coordinates. Supports float32 or float64 dtype.
    charges : torch.Tensor, shape (N,)
        Atomic partial charges in elementary charge units.
    cell : torch.Tensor, shape (3, 3) or (B, 3, 3)
        Unit cell matrices with lattice vectors as rows. Shape (3, 3) is
        automatically promoted to (1, 3, 3).
    alpha : float or torch.Tensor
        Ewald splitting parameter controlling real/reciprocal space balance.
        - float: Same α for all systems
        - Tensor shape (B,): Per-system α values
    mesh_dimensions : tuple[int, int, int], optional
        Explicit FFT mesh dimensions (nx, ny, nz). Power-of-2 values are
        optimal for FFT performance. Either mesh_dimensions or mesh_spacing
        must be provided.
    mesh_spacing : float, optional
        Target mesh spacing in same units as cell. Mesh dimensions computed as
        ceil(cell_length / mesh_spacing). Typical value: ~1 Å.
    spline_order : int, default=4
        B-spline interpolation order. Higher orders are more accurate but slower.
        - 4: Cubic B-splines (good balance, most common)
        - 5-6: Higher accuracy for demanding applications
        - Must be ≥ 3 for smooth interpolation
    batch_idx : torch.Tensor, shape (N,), dtype=int32, optional
        System index for each atom (0 to B-1). Determines kernel dispatch:
        - None: Single-system optimized kernels
        - Provided: Batched kernels for multiple independent systems
    k_vectors : torch.Tensor, shape (nx, ny, nz//2+1, 3), optional
        Precomputed k-vectors from ``generate_k_vectors_pme``. Providing this
        along with k_squared skips k-vector generation (~15% speedup).
        Can be precomputed once and reused when cell and mesh are unchanged.
    k_squared : torch.Tensor, shape (nx, ny, nz//2+1), optional
        Precomputed |k|² values. Must be provided together with k_vectors.
    compute_forces : bool, default=False
        Whether to compute explicit reciprocal-space forces.
    compute_charge_gradients : bool, default=False
        Whether to compute analytical charge gradients ∂E/∂q_i. Useful for
        computing charge Hessians in ML potential training.

    Returns
    -------
    energies : torch.Tensor, shape (N,)
        Per-atom reciprocal-space energy (includes self and background corrections).
    forces : torch.Tensor, shape (N, 3), optional
        Reciprocal-space forces. Only returned if compute_forces=True.
    charge_gradients : torch.Tensor, shape (N,), optional
        Charge gradients ∂E_recip/∂q_i. Only returned if compute_charge_gradients=True.

    Return Patterns
    ---------------
    - ``compute_forces=False, compute_charge_gradients=False``: energies
    - ``compute_forces=True, compute_charge_gradients=False``: (energies, forces)
    - ``compute_forces=False, compute_charge_gradients=True``: (energies, charge_gradients)
    - ``compute_forces=True, compute_charge_gradients=True``: (energies, forces, charge_gradients)

    Raises
    ------
    ValueError
        If neither mesh_dimensions nor mesh_spacing is provided.

    Examples
    --------
    Energy only with explicit mesh dimensions::

        >>> energies = pme_reciprocal_space(
        ...     positions, charges, cell,
        ...     alpha=0.3, mesh_dimensions=(32, 32, 32),
        ... )
        >>> total_recip_energy = energies.sum()

    With forces using mesh spacing::

        >>> energies, forces = pme_reciprocal_space(
        ...     positions, charges, cell,
        ...     alpha=0.3, mesh_spacing=1.0,
        ...     compute_forces=True,
        ... )

    Precomputed k-vectors for MD loop (fixed cell)::

        >>> from nvalchemiops.interactions.electrostatics import generate_k_vectors_pme
        >>> mesh_dims = (32, 32, 32)
        >>> k_vectors, k_squared = generate_k_vectors_pme(cell, mesh_dims)
        >>> for step in range(num_steps):
        ...     energies = pme_reciprocal_space(
        ...         positions, charges, cell,
        ...         alpha=0.3, mesh_dimensions=mesh_dims,
        ...         k_vectors=k_vectors, k_squared=k_squared,
        ...     )

    With charge gradients for ML training::

        >>> energies, charge_grads = pme_reciprocal_space(
        ...     positions, charges, cell,
        ...     alpha=0.3, mesh_dimensions=(32, 32, 32),
        ...     compute_charge_gradients=True,
        ... )

    See Also
    --------
    particle_mesh_ewald : Complete PME calculation (real + reciprocal).
    generate_k_vectors_pme : Generate k-vectors for this function.
    pme_green_structure_factor : Compute Green's function on mesh.
    """
    cell, num_systems = _prepare_cell(cell)
    alpha_tensor = _prepare_alpha(alpha, num_systems, torch.float64, positions.device)

    # Determine mesh dimensions
    if mesh_dimensions is None:
        if mesh_spacing is None:
            raise ValueError("Either mesh_dimensions or mesh_spacing must be provided")
        cell_lengths = torch.norm(cell[0], dim=1)
        mesh_dimensions = tuple(
            int(torch.ceil(length / mesh_spacing).item()) for length in cell_lengths
        )

    energies, forces, charge_grads = _pme_reciprocal_space_impl(
        positions,
        charges,
        cell,
        alpha_tensor,
        mesh_dimensions,
        spline_order,
        batch_idx,
        compute_forces=compute_forces,
        compute_charge_gradients=compute_charge_gradients,
        k_vectors=k_vectors,
        k_squared=k_squared,
    )

    # Build return tuple based on flags
    if compute_forces and compute_charge_gradients:
        return energies, forces, charge_grads
    elif compute_forces:
        return energies, forces
    elif compute_charge_gradients:
        return energies, charge_grads
    else:
        return energies


###########################################################################################
########################### Unified PME API ###############################################
###########################################################################################


def particle_mesh_ewald(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    alpha: float | torch.Tensor | None = None,
    mesh_spacing: float | None = None,
    mesh_dimensions: tuple[int, int, int] | None = None,
    spline_order: int = 4,
    batch_idx: torch.Tensor | None = None,
    k_vectors: torch.Tensor | None = None,
    k_squared: torch.Tensor | None = None,
    neighbor_list: torch.Tensor | None = None,
    neighbor_ptr: torch.Tensor | None = None,
    neighbor_shifts: torch.Tensor | None = None,
    neighbor_matrix: torch.Tensor | None = None,
    neighbor_matrix_shifts: torch.Tensor | None = None,
    mask_value: int | None = None,
    compute_forces: bool = False,
    compute_charge_gradients: bool = False,
    accuracy: float = 1e-6,
) -> (
    torch.Tensor
    | tuple[torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
):
    """Complete Particle Mesh Ewald (PME) calculation for long-range electrostatics.

    Computes total Coulomb energy using the PME method, which achieves :math:`O(N \\log N)`
    scaling through FFT-based reciprocal space calculations. Combines:
    1. Real-space contribution (short-range, erfc-damped)
    2. Reciprocal-space contribution (long-range, FFT + B-spline interpolation)
    3. Self-energy and background corrections

    Total Energy Formula:

    .. math::

        E_{\\text{total}} = E_{\\text{real}} + E_{\\text{reciprocal}} - E_{\\text{self}} - E_{\\text{background}}

    where:

    .. math::

        E_{\\text{real}} = \\frac{1}{2} \\sum_{i \\neq j} q_i q_j \\frac{\\text{erfc}(\\alpha r_{ij}/\\sqrt{2})}{r_{ij}}
        E_{\\text{reciprocal}} = FFT-based smooth long-range contribution
        E_{\\text{self}} = \\sum_i \\frac{\\alpha}{\\sqrt{2\\pi}} q_i^2
        E_{\\text{background}} = \\frac{\\pi}{2\\alpha^2 V} Q_{\\text{total}}^2

    Parameters
    ----------
    positions : torch.Tensor, shape (N, 3)
        Atomic coordinates. Supports float32 or float64 dtype.
    charges : torch.Tensor, shape (N,)
        Atomic partial charges in elementary charge units.
    cell : torch.Tensor, shape (3, 3) or (B, 3, 3)
        Unit cell matrices with lattice vectors as rows. Shape (3, 3) is
        automatically promoted to (1, 3, 3) for single-system mode.
    alpha : float, torch.Tensor, or None, default=None
        Ewald splitting parameter controlling real/reciprocal space balance.
        - float: Same α for all systems
        - Tensor shape (B,): Per-system α values
        - None: Automatically estimated using Kolafa-Perram formula
        Larger α shifts more computation to reciprocal space.
    mesh_spacing : float, optional
        Target mesh spacing in same units as cell (typically Å). Mesh dimensions
        computed as ceil(cell_length / mesh_spacing). Typical value: 0.8-1.2 Å.
    mesh_dimensions : tuple[int, int, int], optional
        Explicit FFT mesh dimensions (nx, ny, nz). Power-of-2 values recommended
        for optimal FFT performance. If None and mesh_spacing is None, computed
        from accuracy parameter.
    spline_order : int, default=4
        B-spline interpolation order. Higher orders are more accurate but slower.
        - 4: Cubic B-splines (standard, good accuracy/speed balance)
        - 5-6: Higher accuracy for demanding applications
    batch_idx : torch.Tensor, shape (N,), dtype=int32, optional
        System index for each atom (0 to B-1). Determines execution mode:
        - None: Single-system optimized kernels
        - Provided: Batched kernels for multiple independent systems
    k_vectors : torch.Tensor, shape (nx, ny, nz//2+1, 3), optional
        Precomputed k-vectors from ``generate_k_vectors_pme``. Providing this
        along with k_squared skips k-vector generation (~15% speedup).
        Useful for fixed-cell MD simulations (NVT/NVE).
    k_squared : torch.Tensor, shape (nx, ny, nz//2+1), optional
        Precomputed |k|² values. Must be provided together with k_vectors.
    neighbor_list : torch.Tensor, shape (2, M), dtype=int32, optional
        Neighbor pairs for real-space in COO format. Row 0 = source indices,
        row 1 = target indices. Mutually exclusive with neighbor_matrix.
    neighbor_ptr : torch.Tensor, shape (N+1,), dtype=int32, optional
        CSR row pointers for neighbor_list. neighbor_ptr[i] gives the starting
        index in neighbor_list for atom i's neighbors. Required with neighbor_list.
    neighbor_shifts : torch.Tensor, shape (M, 3), dtype=int32, optional
        Periodic image shifts for neighbor_list. Required with neighbor_list.
    neighbor_matrix : torch.Tensor, shape (N, max_neighbors), dtype=int32, optional
        Dense neighbor matrix format. Entry [i, k] = j means j is k-th neighbor of i.
        Invalid entries should be set to mask_value.
        Mutually exclusive with neighbor_list.
    neighbor_matrix_shifts : torch.Tensor, shape (N, max_neighbors, 3), dtype=int32, optional
        Periodic image shifts for neighbor_matrix. Required with neighbor_matrix.
    mask_value : int, optional
        Value indicating invalid entries in neighbor_matrix. Defaults to N.
    compute_forces : bool, default=False
        Whether to compute explicit analytical forces.
    compute_charge_gradients : bool, default=False
        Whether to compute analytical charge gradients ∂E/∂q_i. Useful for
        training ML potentials that require second derivatives (charge Hessians).
    accuracy : float, default=1e-6
        Target relative accuracy for automatic parameter estimation (α, mesh dims).
        Only used when alpha or mesh_dimensions is None.
        Smaller values increase accuracy but also computational cost.

    Returns
    -------
    energies : torch.Tensor, shape (N,)
        Per-atom contribution to total PME energy. Sum gives total energy.
    forces : torch.Tensor, shape (N, 3), optional
        Forces on each atom. Only returned if compute_forces=True.
    charge_gradients : torch.Tensor, shape (N,), optional
        Charge gradients ∂E/∂q_i. Only returned if compute_charge_gradients=True.

    Return Patterns
    ---------------
    - ``compute_forces=False, compute_charge_gradients=False``: energies
    - ``compute_forces=True, compute_charge_gradients=False``: (energies, forces)
    - ``compute_forces=False, compute_charge_gradients=True``: (energies, charge_gradients)
    - ``compute_forces=True, compute_charge_gradients=True``: (energies, forces, charge_gradients)

    Raises
    ------
    ValueError
        If neither neighbor_list nor neighbor_matrix is provided for real-space.
    TypeError
        If alpha has an unsupported type.

    Examples
    --------
    Automatic parameter estimation (recommended for most cases)::

        >>> energies = particle_mesh_ewald(
        ...     positions, charges, cell,
        ...     neighbor_list=nl, neighbor_shifts=shifts,
        ...     neighbor_ptr=nptr, accuracy=1e-6,
        ... )
        >>> total_energy = energies.sum()

    Explicit parameters for reproducibility::

        >>> energies, forces = particle_mesh_ewald(
        ...     positions, charges, cell,
        ...     alpha=0.3, mesh_dimensions=(32, 32, 32),
        ...     spline_order=4, neighbor_list=nl,
        ... neighbor_shifts=shifts, neighbor_ptr=nptr,
        ...     compute_forces=True,
        ... )

    Using mesh spacing for automatic mesh sizing::

        >>> energies, forces = particle_mesh_ewald(
        ...     positions, charges, cell,
        ...     alpha=0.3, mesh_spacing=1.0,  # ~1 Å spacing
        ...     neighbor_list=nl, neighbor_shifts=shifts,
        ...     neighbor_ptr=nptr, compute_forces=True,
        ... )

    Batched systems (multiple independent structures)::

        >>> # positions: concatenated atoms from all systems
        >>> # batch_idx: [0,0,0,0, 1,1,1,1, 2,2,2,2] for 4 atoms × 3 systems
        >>> energies, forces = particle_mesh_ewald(
        ...     positions, charges, cells,  # cells shape (3, 3, 3)
        ...     alpha=torch.tensor([0.3, 0.35, 0.3]),
        ...     batch_idx=batch_idx,
        ...     mesh_dimensions=(32, 32, 32),
        ...     neighbor_list=nl,
        ...     neighbor_shifts=shifts, neighbor_ptr=nptr,
        ...     compute_forces=True,
        ... )

    Precomputed k-vectors for MD loop (fixed cell)::

        >>> from nvalchemiops.interactions.electrostatics import generate_k_vectors_pme
        >>> mesh_dims = (32, 32, 32)
        >>> k_vectors, k_squared = generate_k_vectors_pme(cell, mesh_dims)
        >>> for step in range(num_steps):
        ...     energies, forces = particle_mesh_ewald(
        ...         positions, charges, cell,
        ...         alpha=0.3, mesh_dimensions=mesh_dims,
        ...         k_vectors=k_vectors, k_squared=k_squared,
        ...         neighbor_list=nl, neighbor_shifts=shifts,
        ...         neighbor_ptr=nptr,
        ...         compute_forces=True,
        ...     )

    With charge gradients for ML training::

        >>> energies, forces, charge_grads = particle_mesh_ewald(
        ...     positions, charges, cell,
        ...     alpha=0.3, mesh_dimensions=(32, 32, 32),
        ...     neighbor_list=nl, neighbor_shifts=shifts,
        ...     neighbor_ptr=nptr,
        ...     compute_forces=True, compute_charge_gradients=True,
        ... )
        >>> # Use charge_grads for training on ∂E/∂q

    Using PyTorch autograd::

        >>> positions.requires_grad_(True)
        >>> energies = particle_mesh_ewald(
        ...     positions, charges, cell,
        ...     alpha=0.3, mesh_dimensions=(32, 32, 32),
        ...     neighbor_list=nl, neighbor_shifts=shifts,
        ...     neighbor_ptr=nptr,
        ... )
        >>> total_energy = energies.sum()
        >>> total_energy.backward()
        >>> autograd_forces = -positions.grad  # Should match explicit forces

    Notes
    -----
    Automatic Parameter Estimation (when alpha is None):
        Uses Kolafa-Perram formula:

    .. math::

        \\begin{aligned}
        \\eta &= \\frac{(V^2 / N)^{1/6}}{\\sqrt{2\\pi}} \\\\
        \\alpha &= \\frac{1}{2\\eta}
        \\end{aligned}

    Mesh dimensions (when mesh_dimensions is None):

    .. math::

        n_x = \\left\\lceil \\frac{2 \\alpha L_x}{3 \\varepsilon^{1/5}} \\right\\rceil

    Autograd Support:
        All inputs (positions, charges, cell) support gradient computation.

    See Also
    --------
    pme_reciprocal_space : Reciprocal-space component only
    ewald_real_space : Real-space component (used internally)
    estimate_pme_parameters : Automatic parameter estimation
    PMEParameters : Container for PME parameters
    """
    num_atoms = positions.shape[0]

    # Prepare cell
    cell, num_systems = _prepare_cell(cell)

    # Estimate parameters if not provided
    if alpha is None:
        params = estimate_pme_parameters(positions, cell, batch_idx, accuracy)
        alpha = params.alpha
        if mesh_dimensions is None and mesh_spacing is None:
            mesh_dimensions = tuple(params.mesh_dimensions)  # Unpack the tuple

    # Prepare alpha tensor
    alpha = _prepare_alpha(alpha, num_systems, positions.dtype, positions.device)

    if mask_value is None:
        mask_value = num_atoms

    # Determine mesh dimensions
    if mesh_dimensions is None:
        if mesh_spacing is not None:
            mesh_dimensions = mesh_spacing_to_dimensions(cell, mesh_spacing)
        else:
            # Use accuracy-based estimation
            mesh_dimensions = estimate_pme_mesh_dimensions(cell, alpha, accuracy)

    # Compute real-space contribution
    rs = ewald_real_space(
        positions=positions,
        charges=charges,
        cell=cell,
        alpha=alpha,
        neighbor_list=neighbor_list,
        neighbor_ptr=neighbor_ptr,
        neighbor_shifts=neighbor_shifts,
        neighbor_matrix=neighbor_matrix,
        neighbor_matrix_shifts=neighbor_matrix_shifts,
        mask_value=mask_value,
        batch_idx=batch_idx,
        compute_forces=compute_forces,
        compute_charge_gradients=compute_charge_gradients,
    )

    # Compute reciprocal-space contribution
    rec = pme_reciprocal_space(
        positions=positions,
        charges=charges,
        cell=cell,
        alpha=alpha,
        mesh_dimensions=mesh_dimensions,
        spline_order=spline_order,
        batch_idx=batch_idx,
        compute_forces=compute_forces,
        compute_charge_gradients=compute_charge_gradients,
        k_vectors=k_vectors,
        k_squared=k_squared,
    )

    # Combine results based on flags
    if compute_forces and compute_charge_gradients:
        # rs = (energies, forces, charge_grads), rec = (energies, forces, charge_grads)
        total_energies = rs[0] + rec[0]
        total_forces = rs[1] + rec[1]
        total_charge_grads = rs[2] + rec[2]
        return total_energies, total_forces, total_charge_grads
    elif compute_forces:
        # rs = (energies, forces), rec = (energies, forces)
        total_energies = rs[0] + rec[0]
        total_forces = rs[1] + rec[1]
        return total_energies, total_forces
    elif compute_charge_gradients:
        # rs = (energies, charge_grads), rec = (energies, charge_grads)
        total_energies = rs[0] + rec[0]
        total_charge_grads = rs[1] + rec[1]
        return total_energies, total_charge_grads
    else:
        result = rs + rec
        return result
