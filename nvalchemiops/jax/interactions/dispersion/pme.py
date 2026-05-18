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

"""JAX bindings for dispersion (LJ-PME).

This module provides:

* The reciprocal-space and self-energy components of LJ-PME (PR1):
  :func:`pme_dispersion_reciprocal_space`,
  :func:`pme_dispersion_green_structure_factor`,
  :func:`pme_dispersion_energy_corrections`.
* The damped real-space LJ-PME term (PR2): :func:`lj_pme_real_space` —
  computes :math:`V = C_{12}/r^{12} - C_6\\,g(\\beta r)/r^6` for atom pairs
  within a cutoff, optionally with explicit forces and a virial tensor.

The total LJ-PME energy combines these as

.. math::

    V_{\\text{total}} = V_{\\text{real}} + V_{\\text{recip}} - V_{\\text{self}}.

The top-level ``lj_pme()`` orchestrator (with automatic β/mesh estimation)
arrives in a later PR.

References
----------
Wennberg, Hess, Lindahl (2013). JCTC 9, 3527.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import warp as wp
from warp.jax_experimental import jax_kernel

from nvalchemiops.interactions.dispersion.pme_dispersion_kernels import (
    _batch_pme_dispersion_green_structure_factor_kernel_overload,
    _batch_pme_dispersion_self_energy_kernel_overload,
    _lj_pme_real_space_energy_forces_kernel_overload,
    _lj_pme_real_space_energy_forces_virial_kernel_overload,
    _lj_pme_real_space_energy_kernel_overload,
    _pme_dispersion_green_structure_factor_kernel_overload,
    _pme_dispersion_self_energy_kernel_overload,
)
from nvalchemiops.jax.interactions.dispersion.parameters import (
    DISPERSION_DEFAULT_ACCURACY,
    DISPERSION_DEFAULT_CUTOFF,
    estimate_pme_dispersion_mesh_dimensions,
    solve_dispersion_beta,
)
from nvalchemiops.jax.interactions.electrostatics.k_vectors import (
    generate_k_vectors_pme,
)
from nvalchemiops.jax.spline import (
    spline_gather,
    spline_gather_vec3,
    spline_spread,
)

__all__ = [
    "pme_dispersion_reciprocal_space",
    "pme_dispersion_green_structure_factor",
    "pme_dispersion_energy_corrections",
    "lj_pme_real_space",
    "lj_pme",
]


# ==============================================================================
# Helpers
# ==============================================================================


def _make_jax_kernels(
    wp_overload_dict: dict,
    num_outputs: int,
    in_out_argnames: list[str],
) -> dict:
    """Map JAX dtypes to Warp kernel overloads."""
    _JAX_TO_WP = {jnp.float32: wp.float32, jnp.float64: wp.float64}
    return {
        jax_dtype: jax_kernel(
            wp_overload_dict[wp_dtype],
            num_outputs=num_outputs,
            in_out_argnames=in_out_argnames,
            enable_backward=False,
        )
        for jax_dtype, wp_dtype in _JAX_TO_WP.items()
    }


def _normalize_dtype(dtype):
    """Normalize dtype for kernel dictionary lookup."""
    if dtype == jnp.float32 or str(dtype) == "float32":
        return jnp.float32
    elif dtype == jnp.float64 or str(dtype) == "float64":
        return jnp.float64
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


# ==============================================================================
# JAX Kernel Wrappers
# ==============================================================================

_jax_pme_disp_green_sf = _make_jax_kernels(
    _pme_dispersion_green_structure_factor_kernel_overload,
    2,
    ["green_function", "structure_factor_sq"],
)

_jax_batch_pme_disp_green_sf = _make_jax_kernels(
    _batch_pme_dispersion_green_structure_factor_kernel_overload,
    2,
    ["green_function", "structure_factor_sq"],
)

_jax_pme_disp_self_energy = _make_jax_kernels(
    _pme_dispersion_self_energy_kernel_overload,
    1,
    ["energy_correction"],
)

_jax_batch_pme_disp_self_energy = _make_jax_kernels(
    _batch_pme_dispersion_self_energy_kernel_overload,
    1,
    ["energy_correction"],
)

_jax_lj_pme_real_space_energy = _make_jax_kernels(
    _lj_pme_real_space_energy_kernel_overload,
    1,
    ["atomic_energies"],
)
_jax_lj_pme_real_space_energy_forces = _make_jax_kernels(
    _lj_pme_real_space_energy_forces_kernel_overload,
    2,
    ["atomic_energies", "atomic_forces"],
)
_jax_lj_pme_real_space_energy_forces_virial = _make_jax_kernels(
    _lj_pme_real_space_energy_forces_virial_kernel_overload,
    3,
    ["atomic_energies", "atomic_forces", "virial"],
)


# ==============================================================================
# Public API: Green's function + structure factor
# ==============================================================================


def pme_dispersion_green_structure_factor(
    k_squared: jax.Array,
    mesh_dimensions: tuple[int, int, int],
    beta: jax.Array,
    cell: jax.Array,
    spline_order: int = 4,
    batch_idx: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Compute dispersion Green's function and B-spline structure factor.

    Green's function (volume-normalized):

    .. math::

        G_{\\text{disp}}(k) = \\frac{\\pi^{3/2} \\beta^3}{2V}
            \\, f\\!\\left(\\tfrac{|k|}{2\\beta}\\right),
        \\qquad
        f(x) = \\tfrac{1}{3}\\bigl[(1 - 2x^2) e^{-x^2}
            + 2 x^3 \\sqrt{\\pi}\\, \\text{erfc}(x)\\bigr].

    Structure-factor correction :math:`C^2(k)` is identical to Coulomb PME
    (depends only on the mesh and spline order).

    Parameters
    ----------
    k_squared : jax.Array
        :math:`|k|^2` at each FFT grid point.
        - Single-system: shape (Nx, Ny, Nz_rfft)
        - Batch: shape (B, Nx, Ny, Nz_rfft)
    mesh_dimensions : tuple[int, int, int]
        Full mesh dimensions (Nx, Ny, Nz).
    beta : jax.Array
        Dispersion Ewald splitting parameter.
        - Single-system: shape (1,) or scalar
        - Batch: shape (B,)
    cell : jax.Array
        Unit cell matrices. Shape (3, 3) / (1, 3, 3) / (B, 3, 3).
    spline_order : int, default=4
        B-spline interpolation order.
    batch_idx : jax.Array | None
        If provided, dispatches to batch kernels.

    Returns
    -------
    green_function : jax.Array
        Volume-normalized dispersion Green's function.
    structure_factor_sq : jax.Array
        Squared structure factor :math:`C^2(k)` for B-spline deconvolution.
    """
    mesh_nx, mesh_ny, mesh_nz = mesh_dimensions
    input_dtype = _normalize_dtype(k_squared.dtype)

    if cell.ndim == 2:
        cell = cell[jnp.newaxis, :, :]
    volume = jnp.abs(jnp.linalg.det(cell)).astype(input_dtype)

    miller_x = jnp.fft.fftfreq(mesh_nx, d=1.0 / mesh_nx).astype(input_dtype)
    miller_y = jnp.fft.fftfreq(mesh_ny, d=1.0 / mesh_ny).astype(input_dtype)
    miller_z = jnp.fft.rfftfreq(mesh_nz, d=1.0 / mesh_nz).astype(input_dtype)

    if beta.ndim == 0:
        beta = beta.reshape(1)
    beta = beta.astype(input_dtype)

    if batch_idx is None:
        kernel = _jax_pme_disp_green_sf[input_dtype]

        green_function = jnp.zeros(
            (mesh_nx, mesh_ny, mesh_nz // 2 + 1), dtype=input_dtype
        )
        structure_factor_sq = jnp.zeros(
            (mesh_nx, mesh_ny, mesh_nz // 2 + 1), dtype=input_dtype
        )

        green_out, sf_out = kernel(
            k_squared.astype(input_dtype),
            miller_x,
            miller_y,
            miller_z,
            beta,
            volume,
            int(mesh_nx),
            int(mesh_ny),
            int(mesh_nz),
            int(spline_order),
            green_function,
            structure_factor_sq,
            launch_dims=(mesh_nx, mesh_ny, mesh_nz // 2 + 1),
        )
        return green_out, sf_out
    else:
        num_systems = cell.shape[0]
        kernel = _jax_batch_pme_disp_green_sf[input_dtype]

        k_sq = k_squared.astype(input_dtype)
        if k_sq.ndim == 3:
            k_sq = jnp.broadcast_to(
                k_sq[jnp.newaxis], (num_systems, mesh_nx, mesh_ny, mesh_nz // 2 + 1)
            )

        green_function = jnp.zeros(
            (num_systems, mesh_nx, mesh_ny, mesh_nz // 2 + 1), dtype=input_dtype
        )
        structure_factor_sq = jnp.zeros(
            (mesh_nx, mesh_ny, mesh_nz // 2 + 1), dtype=input_dtype
        )

        green_out, sf_out = kernel(
            k_sq,
            miller_x,
            miller_y,
            miller_z,
            beta,
            volume,
            int(mesh_nx),
            int(mesh_ny),
            int(mesh_nz),
            int(spline_order),
            green_function,
            structure_factor_sq,
            launch_dims=(num_systems, mesh_nx, mesh_ny, mesh_nz // 2 + 1),
        )
        return green_out, sf_out


# ==============================================================================
# Public API: Self-energy correction
# ==============================================================================


def pme_dispersion_energy_corrections(
    c6_coefficients: jax.Array,
    beta: jax.Array,
    batch_idx: jax.Array | None = None,
) -> jax.Array:
    """Compute the dispersion self-energy correction term.

    Returns the per-system self-energy

    .. math::

        V_{\\text{self}} = -\\frac{\\beta^6}{12} \\sum_i C_{6,ii}.

    Following the proposal convention, the total LJ-PME energy is then
    :math:`V_{\\text{total}} = V_{\\text{real}} + V_{\\text{recip}} - V_{\\text{self}}`.

    Parameters
    ----------
    c6_coefficients : jax.Array, shape (N,) or (N_total,)
        Per-atom homoatomic :math:`C_{6,ii}` values.
    beta : jax.Array
        Dispersion Ewald splitting parameter.
        - Single-system: shape (1,) or scalar
        - Batch: shape (B,)
    batch_idx : jax.Array | None
        System index for each atom. If provided, uses batched kernel.

    Returns
    -------
    energy_corrections : jax.Array
        - Single-system: shape (1,) — total :math:`V_{\\text{self}}` for the system.
        - Batch: shape (B,) — total per system.
    """
    input_dtype = _normalize_dtype(c6_coefficients.dtype)
    num_atoms = c6_coefficients.shape[0]
    c6 = c6_coefficients.astype(input_dtype)

    if beta.ndim == 0:
        beta = beta.reshape(1)
    beta = beta.astype(input_dtype)

    if batch_idx is None:
        kernel = _jax_pme_disp_self_energy[input_dtype]
        per_atom = jnp.zeros(num_atoms, dtype=input_dtype)
        (per_atom_out,) = kernel(
            c6,
            beta,
            per_atom,
            launch_dims=(num_atoms,),
        )
        return per_atom_out.sum().reshape(1)
    else:
        kernel = _jax_batch_pme_disp_self_energy[input_dtype]
        num_systems = int(beta.shape[0])
        per_atom = jnp.zeros(num_atoms, dtype=input_dtype)
        (per_atom_out,) = kernel(
            c6,
            batch_idx.astype(jnp.int32),
            beta,
            per_atom,
            launch_dims=(num_atoms,),
        )
        out = jnp.zeros(num_systems, dtype=input_dtype)
        return out.at[batch_idx].add(per_atom_out)


# ==============================================================================
# Public API: Reciprocal-space dispersion PME
# ==============================================================================


def pme_dispersion_reciprocal_space(
    positions: jax.Array,
    c6_coefficients: jax.Array,
    cell: jax.Array,
    beta: jax.Array,
    mesh_dimensions: tuple[int, int, int] | None = None,
    mesh_spacing: float | None = None,
    spline_order: int = 4,
    batch_idx: jax.Array | None = None,
    k_vectors: jax.Array | None = None,
    k_squared: jax.Array | None = None,
    compute_forces: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Compute the dispersion PME reciprocal-space energy.

    Implements the FFT-based long-range :math:`r^{-6}` contribution using
    B-spline interpolation and convolution with the dispersion Green's
    function:

    .. math::

        V_{\\text{recip}} = \\frac{\\pi^{3/2}\\beta^3}{2V}
            \\sum_{m \\neq 0} f(\\pi|m|/\\beta)\\, |\\rho_{\\text{disp}}(m)|^2.

    The spread quantity is :math:`\\sqrt{C_{6,ii}}` (geometric combination).

    Pipeline:

    1. Spread :math:`\\sqrt{C_{6,ii}}` to mesh (``spline_spread``).
    2. FFT to reciprocal space.
    3. Multiply by :math:`G_{\\text{disp}}(k) / C^2(k)`.
    4. Inverse FFT to potential mesh.
    5. Gather potential at atoms (``spline_gather``).
    6. Sum :math:`\\sqrt{C_{6,ii}} \\cdot \\varphi_i` over atoms.

    Parameters
    ----------
    positions : jax.Array, shape (N, 3)
        Atomic coordinates.
    c6_coefficients : jax.Array, shape (N,)
        Per-atom homoatomic :math:`C_{6,ii}` values (non-negative).
    cell : jax.Array, shape (3, 3) or (B, 3, 3)
        Unit cell matrices.
    beta : jax.Array
        Dispersion Ewald splitting parameter.
        - Single-system: shape (1,) or scalar
        - Batch: shape (B,)
    mesh_dimensions : tuple[int, int, int], optional
        Explicit FFT mesh dimensions. Required if mesh_spacing is None.
    mesh_spacing : float, optional
        Target mesh spacing in cell units. Used to compute mesh_dimensions
        if those are not provided.
    spline_order : int, default=4
        B-spline interpolation order.
    batch_idx : jax.Array | None
        System index for each atom.
    k_vectors : jax.Array, optional
        Precomputed k-vectors from ``generate_k_vectors_pme``.
    k_squared : jax.Array, optional
        Precomputed :math:`|k|^2` values.

    Returns
    -------
    energy : jax.Array
        - Single-system: shape (1,) — total reciprocal-space dispersion energy.
        - Batch: shape (B,) — total per system.

    Notes
    -----
    Unlike Coulomb PME, dispersion is short-range physically but
    long-range for slowly-decaying-tail PME purposes; the returned
    energy is the lattice-summed reciprocal contribution and is meant to
    be combined with the real-space damped term and the self-energy
    correction to form the total LJ-PME energy.
    """
    num_atoms = positions.shape[0]
    input_dtype = _normalize_dtype(positions.dtype)
    is_batch = batch_idx is not None
    fft_dims = (1, 2, 3) if is_batch else (0, 1, 2)

    if cell.ndim == 2:
        cell_b = cell[jnp.newaxis, :, :]
    else:
        cell_b = cell
    num_systems = cell_b.shape[0]

    # Handle empty systems
    if num_atoms == 0:
        empty_e = jnp.zeros(num_systems if is_batch else 1, dtype=input_dtype)
        if compute_forces:
            return empty_e, jnp.zeros((0, 3), dtype=input_dtype)
        return empty_e

    # Determine mesh dimensions
    if mesh_dimensions is None:
        if mesh_spacing is None:
            raise ValueError("Either mesh_dimensions or mesh_spacing must be provided")
        cell_lengths = jnp.linalg.norm(cell_b[0], axis=1)
        mesh_dimensions = tuple(
            int(math.ceil(float(length) / mesh_spacing)) for length in cell_lengths
        )

    mesh_nx, mesh_ny, mesh_nz = mesh_dimensions

    # Spread sqrt(C6) — geometric combination factorizes the kernel
    sqrt_c6 = jnp.sqrt(jnp.clip(c6_coefficients.astype(input_dtype), min=0.0))

    mesh_grid = spline_spread(
        positions,
        sqrt_c6,
        cell_b if is_batch else cell,
        mesh_dims=mesh_dimensions,
        spline_order=spline_order,
        batch_idx=batch_idx,
    )

    # FFT to reciprocal space
    mesh_fft = jnp.fft.rfftn(mesh_grid, axes=fft_dims, norm="backward")

    # k-vectors and Green's function
    if k_vectors is None or k_squared is None:
        k_vectors, k_squared = generate_k_vectors_pme(cell_b, mesh_dimensions)

    # Prepare beta in correct shape
    if beta.ndim == 0:
        beta = beta.reshape(1)
    beta = beta.astype(input_dtype)
    if is_batch and beta.shape[0] == 1 and num_systems > 1:
        beta = jnp.broadcast_to(beta, (num_systems,))

    green_function, structure_factor_sq = pme_dispersion_green_structure_factor(
        k_squared,
        mesh_dimensions,
        beta,
        cell_b,
        spline_order,
        batch_idx,
    )

    # Apply B-spline deconvolution and convolve with Green's function
    complex_dtype = jnp.complex64 if input_dtype == jnp.float32 else jnp.complex128
    mesh_fft = mesh_fft.astype(complex_dtype) / structure_factor_sq
    convolved_mesh = mesh_fft * green_function

    # IFFT to potential mesh
    potential_mesh = jnp.fft.irfftn(
        convolved_mesh, s=mesh_dimensions, axes=fft_dims, norm="forward"
    )

    # Gather potential at atoms
    raw_potential = spline_gather(
        positions,
        potential_mesh,
        cell_b if is_batch else cell,
        spline_order=spline_order,
        batch_idx=batch_idx,
    )

    # Energy: per-atom sqrt(C6) * phi summed (Green's function already includes
    # the 1/(2V) prefactor + sign for the reciprocal sum).
    per_atom_energy = sqrt_c6 * raw_potential

    if is_batch:
        energy = jnp.zeros(num_systems, dtype=input_dtype)
        energy = energy.at[batch_idx].add(per_atom_energy)
    else:
        energy = per_atom_energy.sum().reshape(1)

    if not compute_forces:
        return energy

    # Forces via Fourier gradient (same trick as Coulomb PME):
    #
    #   V_recip = Σ_i √C6_i · φ(r_i)       with φ on mesh from gather step
    #   F_i = -dV_recip/dr_i = 2 · √C6_i · ∇φ(r_i)   (factor 2 from the
    #         double-counting in the quadratic |ρ|² structure).
    #
    # Compute ∇φ at atom positions by inverse-FFTing -i k * convolved_mesh,
    # then gathering with weight √C6_i. Sign convention: ∇φ ↔ -i k.
    Ex_fft = -1j * k_vectors[..., 0] * convolved_mesh
    Ey_fft = -1j * k_vectors[..., 1] * convolved_mesh
    Ez_fft = -1j * k_vectors[..., 2] * convolved_mesh
    Ex = jnp.fft.irfftn(Ex_fft, s=mesh_dimensions, axes=fft_dims, norm="forward")
    Ey = jnp.fft.irfftn(Ey_fft, s=mesh_dimensions, axes=fft_dims, norm="forward")
    Ez = jnp.fft.irfftn(Ez_fft, s=mesh_dimensions, axes=fft_dims, norm="forward")
    gradient_field_mesh = jnp.stack([Ex, Ey, Ez], axis=-1).astype(input_dtype)

    interpolated_gradient = spline_gather_vec3(
        positions,
        sqrt_c6,
        gradient_field_mesh,
        cell_b if is_batch else cell,
        spline_order=spline_order,
        batch_idx=batch_idx,
    )
    forces = 2.0 * interpolated_gradient
    return energy, forces


# ==============================================================================
# Public API: Real-space LJ-PME (PR2)
# ==============================================================================


def lj_pme_real_space(
    positions: jax.Array,
    c6_coefficients: jax.Array,
    c12_coefficients: jax.Array,
    cell: jax.Array,
    neighbor_matrix: jax.Array,
    neighbor_matrix_shifts: jax.Array,
    beta: jax.Array,
    cutoff: float,
    num_neighbors: jax.Array | None = None,
    mask_value: int | None = None,
    compute_forces: bool = False,
    compute_virial: bool = False,
    half_neighbor_list: bool = True,
) -> jax.Array | tuple[jax.Array, ...]:
    """Real-space LJ-PME pair energy (and optionally forces and virial).

    For each pair (i, j) with :math:`r_{ij} < r_{\\text{cut}}` the kernel
    computes the damped Lennard-Jones pair contribution

    .. math::

        V_{ij} = \\frac{C_{12,ij}}{r_{ij}^{12}}
                 - \\frac{C_{6,ij} \\, g(\\beta r_{ij})}{r_{ij}^{6}},
        \\qquad
        g(x) = e^{-x^2}\\bigl(1 + x^2 + x^4/2\\bigr),

    with geometric combination rules
    :math:`C_{6,ij} = \\sqrt{C_{6,ii} C_{6,jj}}`,
    :math:`C_{12,ij} = \\sqrt{C_{12,ii} C_{12,jj}}`. The damping function
    :math:`g` is the Wennberg complement of the long-range PME term so that
    :math:`V_{\\text{real}} + V_{\\text{recip}} - V_{\\text{self}}` recovers
    the bare :math:`r^{-6}` lattice sum.

    Parameters
    ----------
    positions : jax.Array, shape (N, 3)
        Atomic coordinates.
    c6_coefficients : jax.Array, shape (N,)
        Per-atom homoatomic :math:`C_{6,ii}` values.
    c12_coefficients : jax.Array, shape (N,)
        Per-atom homoatomic :math:`C_{12,ii}` values.
    cell : jax.Array, shape (3, 3) or (1, 3, 3)
        Unit cell matrix.
    neighbor_matrix : jax.Array, shape (N, max_neighbors), dtype int32
        Half neighbor matrix: each pair (i, j) appears exactly once.
        Invalid entries should be marked with ``mask_value``.
    neighbor_matrix_shifts : jax.Array, shape (N, max_neighbors, 3), dtype int32
        Periodic image shifts.
    beta : jax.Array, scalar or shape (1,)
        Dispersion Ewald splitting parameter.
    cutoff : float
        Real-space cutoff radius. Pairs with :math:`r \\geq r_{\\text{cut}}`
        are skipped.
    num_neighbors : jax.Array | None, shape (N,), dtype int32
        Valid neighbor count per atom. If None, inferred as
        ``(neighbor_matrix != mask_value).sum(axis=1)``.
    mask_value : int | None
        Sentinel value marking invalid entries in ``neighbor_matrix``.
        Defaults to ``num_atoms``.
    compute_forces : bool, default=False
        If True, return explicit forces alongside the per-atom energy.
    compute_virial : bool, default=False
        If True, return the virial tensor
        :math:`W_{ab} = \\sum_{i<j} r_{ij,a} F_{ij,b}` (3×3) alongside the
        energy.

    Returns
    -------
    energies : jax.Array, shape (N,)
        Per-atom real-space dispersion energy (half-counted per pair).
    forces : jax.Array, shape (N, 3), optional
        Per-atom forces. Only returned if ``compute_forces=True``.
    virial : jax.Array, shape (3, 3), optional
        Virial tensor. Only returned if ``compute_virial=True``;
        always last in the return tuple.
    """
    input_dtype = _normalize_dtype(positions.dtype)
    num_atoms = positions.shape[0]

    if cell.ndim == 2:
        cell_b = cell[jnp.newaxis, :, :]
    elif cell.shape[0] == 1:
        cell_b = cell
    else:
        raise ValueError(
            "lj_pme_real_space currently supports single-system cells; "
            f"got cell shape {cell.shape}."
        )

    positions_cast = positions.astype(input_dtype)
    c6_cast = c6_coefficients.astype(input_dtype)
    c12_cast = c12_coefficients.astype(input_dtype)
    cell_cast = cell_b.astype(input_dtype)
    if beta.ndim == 0:
        beta = beta.reshape(1)
    beta_cast = beta.astype(input_dtype)
    cutoff_cast = jnp.array([float(cutoff)], dtype=input_dtype)

    nbr_mat = neighbor_matrix.astype(jnp.int32)
    nbr_shifts = neighbor_matrix_shifts.astype(jnp.int32)

    if mask_value is None:
        mask_value = num_atoms

    if num_neighbors is None:
        valid = nbr_mat != jnp.int32(mask_value)
        num_neighbors_i32 = valid.sum(axis=1).astype(jnp.int32)
    else:
        num_neighbors_i32 = num_neighbors.astype(jnp.int32)

    atomic_energies = jnp.zeros(num_atoms, dtype=input_dtype)

    half_flag = bool(half_neighbor_list)
    if compute_virial:
        atomic_forces = jnp.zeros((num_atoms, 3), dtype=input_dtype)
        virial_flat = jnp.zeros(9, dtype=input_dtype)
        kernel = _jax_lj_pme_real_space_energy_forces_virial[input_dtype]
        atomic_energies, atomic_forces, virial_flat = kernel(
            positions_cast,
            c6_cast,
            c12_cast,
            cell_cast,
            nbr_mat,
            nbr_shifts,
            num_neighbors_i32,
            beta_cast,
            cutoff_cast,
            int(mask_value),
            half_flag,
            atomic_energies,
            atomic_forces,
            virial_flat,
            launch_dims=(num_atoms,),
        )
        virial = virial_flat.reshape(3, 3)
        if compute_forces:
            return atomic_energies, atomic_forces, virial
        return atomic_energies, virial
    elif compute_forces:
        atomic_forces = jnp.zeros((num_atoms, 3), dtype=input_dtype)
        kernel = _jax_lj_pme_real_space_energy_forces[input_dtype]
        atomic_energies, atomic_forces = kernel(
            positions_cast,
            c6_cast,
            c12_cast,
            cell_cast,
            nbr_mat,
            nbr_shifts,
            num_neighbors_i32,
            beta_cast,
            cutoff_cast,
            int(mask_value),
            half_flag,
            atomic_energies,
            atomic_forces,
            launch_dims=(num_atoms,),
        )
        return atomic_energies, atomic_forces
    else:
        kernel = _jax_lj_pme_real_space_energy[input_dtype]
        (atomic_energies,) = kernel(
            positions_cast,
            c6_cast,
            c12_cast,
            cell_cast,
            nbr_mat,
            nbr_shifts,
            num_neighbors_i32,
            beta_cast,
            cutoff_cast,
            int(mask_value),
            half_flag,
            atomic_energies,
            launch_dims=(num_atoms,),
        )
        return atomic_energies


# ==============================================================================
# Public API: Top-level LJ-PME orchestrator (PR3)
# ==============================================================================


def lj_pme(
    positions: jax.Array,
    c6_coefficients: jax.Array,
    c12_coefficients: jax.Array,
    cell: jax.Array,
    neighbor_matrix: jax.Array,
    neighbor_matrix_shifts: jax.Array,
    beta: float | jax.Array | None = None,
    cutoff: float | None = None,
    mesh_spacing: float | None = None,
    mesh_dimensions: tuple[int, int, int] | None = None,
    spline_order: int = 4,
    batch_idx: jax.Array | None = None,
    num_neighbors: jax.Array | None = None,
    mask_value: int | None = None,
    compute_forces: bool = False,
    accuracy: float = DISPERSION_DEFAULT_ACCURACY,
    half_neighbor_list: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Complete LJ-PME energy (and optionally forces) for one or more systems.

    Computes :math:`V_{\\text{total}} = V_{\\text{real}} + V_{\\text{recip}} - V_{\\text{self}}`
    with geometric combination rules, where

    * :math:`V_{\\text{real}}` is the damped real-space pair sum
      :math:`C_{12,ij}/r^{12} - C_{6,ij} g(\\beta r)/r^6` (from
      :func:`lj_pme_real_space`),
    * :math:`V_{\\text{recip}}` is the FFT-based long-range :math:`r^{-6}`
      lattice sum (from :func:`pme_dispersion_reciprocal_space`),
    * :math:`V_{\\text{self}} = -\\beta^6 \\sum_i C_{6,ii} / 12` is the
      self-energy correction (from :func:`pme_dispersion_energy_corrections`).

    If ``beta``, ``cutoff``, or ``mesh_dimensions`` are not provided they
    are estimated jointly from the target ``accuracy`` (GROMACS-style
    matched-tail criterion, see
    :func:`estimate_pme_dispersion_parameters`).

    Parameters
    ----------
    positions : jax.Array, shape (N, 3)
        Atomic coordinates.
    c6_coefficients : jax.Array, shape (N,)
        Per-atom homoatomic :math:`C_{6,ii}` values.
    c12_coefficients : jax.Array, shape (N,)
        Per-atom homoatomic :math:`C_{12,ii}` values.
    cell : jax.Array, shape (3, 3) or (B, 3, 3)
        Unit cell matrices.
    neighbor_matrix : jax.Array, shape (N, max_neighbors), dtype int32
        Half neighbor matrix for the real-space term.
    neighbor_matrix_shifts : jax.Array, shape (N, max_neighbors, 3), dtype int32
        Periodic image shifts for ``neighbor_matrix``.
    beta : float | jax.Array | None
        Dispersion splitting parameter. If None, estimated from
        ``cutoff`` and ``accuracy``.
    cutoff : float | None
        Real-space cutoff radius. If None, defaults to
        :data:`DISPERSION_DEFAULT_CUTOFF` (9 Å).
    mesh_spacing : float | None
        Target mesh spacing. Alternative to ``mesh_dimensions``.
    mesh_dimensions : tuple[int, int, int] | None
        Explicit FFT mesh dimensions. If None, estimated.
    spline_order : int, default=4
        B-spline interpolation order.
    batch_idx : jax.Array | None
        System index for each atom (for batched evaluation).
    num_neighbors : jax.Array | None, shape (N,), dtype int32
        Valid neighbor count per atom (passed through to the real-space
        kernel; inferred from ``mask_value`` if None).
    mask_value : int | None
        Sentinel value marking invalid entries in ``neighbor_matrix``.
        Defaults to ``num_atoms``.
    compute_forces : bool, default=False
        If True, return the per-atom force tensor alongside the energy.
    accuracy : float, default=1e-3
        Target relative accuracy for joint parameter estimation
        (GROMACS's ``ewald-rtol-lj`` default).

    Returns
    -------
    energy : jax.Array
        Per-system total LJ-PME energy. Shape (1,) for single system,
        (B,) for batched.
    forces : jax.Array, optional
        Per-atom forces (shape (N, 3)) if ``compute_forces=True``.
    """
    is_batch = batch_idx is not None

    # Joint parameter estimation if any of (beta, cutoff, mesh) is missing.
    if cell.ndim == 2:
        cell_b = cell[jnp.newaxis, :, :]
    else:
        cell_b = cell
    num_systems = cell_b.shape[0]

    if cutoff is None:
        cutoff = DISPERSION_DEFAULT_CUTOFF
    if beta is None:
        beta_scalar = solve_dispersion_beta(cutoff, accuracy)
        beta_arr = jnp.full(num_systems, beta_scalar, dtype=cell_b.dtype)
    elif isinstance(beta, (int, float)):
        beta_arr = jnp.full(num_systems, float(beta), dtype=cell_b.dtype)
    else:
        beta_arr = jnp.asarray(beta, dtype=cell_b.dtype)
        if beta_arr.ndim == 0:
            beta_arr = jnp.broadcast_to(beta_arr[None], (num_systems,))
        elif beta_arr.shape[0] == 1 and num_systems > 1:
            beta_arr = jnp.broadcast_to(beta_arr, (num_systems,))

    if mesh_dimensions is None:
        if mesh_spacing is not None:
            cell_lengths_np = jnp.linalg.norm(cell_b[0], axis=1)
            mesh_dimensions = tuple(
                int(math.ceil(float(length) / mesh_spacing))
                for length in cell_lengths_np
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
            beta_arr if not is_batch else beta_arr[0:1],
            cutoff,
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
            beta_arr if not is_batch else beta_arr[0:1],
            cutoff,
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

    # Combine per-system.
    if is_batch:
        e_real_per_system = jnp.zeros(num_systems, dtype=positions.dtype)
        e_real_per_system = e_real_per_system.at[batch_idx].add(e_real_per_atom)
    else:
        e_real_per_system = e_real_per_atom.sum().reshape(1)

    energy = e_real_per_system + e_recip - e_self

    if compute_forces:
        forces = f_real + f_recip
        return energy, forces
    return energy
