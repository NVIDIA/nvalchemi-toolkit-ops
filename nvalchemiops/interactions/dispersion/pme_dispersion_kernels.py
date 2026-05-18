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

"""
PME Dispersion Kernels (LJ-PME)
================================

GPU-accelerated Warp kernels for Particle Mesh Ewald applied to the
attractive dispersion (:math:`r^{-6}`) component of the Lennard-Jones
interaction. With geometric combination rules
(:math:`C_{6,ij} = \\sqrt{C_{6,ii} \\cdot C_{6,jj}}`) the long-range sum
factorizes into a single FFT — the same cost as Coulomb PME.

MATHEMATICAL FORMULATION
========================

The total LJ-PME energy is decomposed (Wennberg et al. JCTC 2013) as:

.. math::

    V_{\\text{total}} = V_{\\text{real}} + V_{\\text{recip}} - V_{\\text{self}}

with reciprocal-space term

.. math::

    V_{\\text{recip}} = \\frac{\\pi^{3/2} \\beta^3}{2V}
        \\sum_{m \\neq 0} f(\\pi|m|/\\beta) \\, |\\rho_{\\text{disp}}(m)|^2

where :math:`\\rho_{\\text{disp}}(m) = \\sum_i \\sqrt{C_{6,ii}}\\, e^{2\\pi i\\,m \\cdot r_i}`
is the dispersion structure factor spread on the mesh and

.. math::

    f(x) = \\tfrac{1}{3}\\bigl[(1 - 2x^2) e^{-x^2}
                + 2 x^3 \\sqrt{\\pi}\\, \\text{erfc}(x)\\bigr].

In terms of the reciprocal-vector magnitude on the mesh, :math:`|k| = 2\\pi |m|`,
so :math:`x = \\pi|m|/\\beta = |k|/(2\\beta)` and :math:`x^2 = |k|^2/(4\\beta^2)`.

The self-energy correction is

.. math::

    V_{\\text{self}} = -\\frac{\\beta^6}{12} \\sum_i C_{6,ii}.

The B-spline structure-factor correction :math:`C^2(k)` is identical to
Coulomb PME (mesh- and spline-order-dependent only).

KERNEL ORGANIZATION
===================

Green's function kernels:
    _pme_dispersion_green_structure_factor_kernel        — single system
    _batch_pme_dispersion_green_structure_factor_kernel  — batched

Self-energy correction kernels:
    _pme_dispersion_self_energy_kernel        — single system, per-atom
    _batch_pme_dispersion_self_energy_kernel  — batched, per-atom

Each emits a per-atom value :math:`-\\beta^6 C_{6,ii}/12`. The binding layer
reduces over atoms (or per system in the batched case) to obtain the total
self-energy correction.

REFERENCES
==========

- Wennberg, Hess, Lindahl (2013). JCTC 9, 3527. (LJ lattice summation.)
- Essmann et al. (1995). J. Chem. Phys. 103, 8577 (SPME).
"""

import math
from typing import Any

import warp as wp

# Mathematical constants
PI = math.pi
TWOPI = 2.0 * PI
SQRT_PI = math.sqrt(PI)
PI_3_2 = PI * SQRT_PI  # pi^(3/2)


###########################################################################################
########################### Helper Functions ##############################################
###########################################################################################


@wp.func
def compute_sinc(x: Any) -> Any:
    """Compute normalized sinc function: :math:`\\sin(\\pi x)/(\\pi x)`.

    Uses Taylor expansion near zero for numerical stability.
    """
    abs_x = wp.abs(x)
    one = type(x)(1.0)
    threshold = type(x)(1e-6)

    if abs_x < threshold:
        return one

    pi_x = type(x)(PI) * x
    return wp.sin(pi_x) / pi_x


@wp.func
def dispersion_green_f(x_sq: Any) -> Any:
    """Compute the dispersion Green's function radial factor.

    .. math::

        f(x) = \\tfrac{1}{3}\\bigl[(1 - 2x^2) e^{-x^2}
                    + 2 x^3 \\sqrt{\\pi}\\, \\text{erfc}(x)\\bigr]

    Takes :math:`x^2` as input to avoid an extra sqrt when not needed.
    """
    one = type(x_sq)(1.0)
    two = type(x_sq)(2.0)
    one_third = type(x_sq)(1.0 / 3.0)
    sqrt_pi = type(x_sq)(SQRT_PI)

    x = wp.sqrt(x_sq)
    exp_term = (one - two * x_sq) * wp.exp(-x_sq)
    erfc_term = two * x_sq * x * sqrt_pi * wp.erfc(x)
    return one_third * (exp_term + erfc_term)


###########################################################################################
########################### Green Function with Structure Factor ##########################
###########################################################################################


@wp.kernel
def _pme_dispersion_green_structure_factor_kernel(
    k_squared: wp.array3d(dtype=Any),  # (Nx, Ny, Nz_rfft)
    miller_x: wp.array(dtype=Any),  # (Nx,)
    miller_y: wp.array(dtype=Any),  # (Ny,)
    miller_z: wp.array(dtype=Any),  # (Nz_rfft,)
    beta: wp.array(dtype=Any),  # (1,)
    volume: wp.array(dtype=Any),  # (1,)
    mesh_nx: wp.int32,
    mesh_ny: wp.int32,
    mesh_nz: wp.int32,
    spline_order: wp.int32,
    green_function: wp.array3d(dtype=Any),  # (Nx, Ny, Nz_rfft)
    structure_factor_sq: wp.array3d(dtype=Any),  # (Nx, Ny, Nz_rfft)
):
    """Compute dispersion PME Green's function and B-spline correction.

    Computes two arrays needed for LJ-PME reciprocal space:

    1. Green's function: :math:`G_{\\text{disp}}(k) = (\\pi^{3/2}\\beta^3/(2V))
       \\cdot f(|k|/(2\\beta))`.
    2. Structure factor squared: :math:`|B(k)|^2` for B-spline dealiasing
       (identical to Coulomb PME).

    Launch Grid
    -----------
    dim = [Nx, Ny, Nz_rfft]

    Each thread processes one grid point in the FFT mesh (using rfft symmetry).

    Parameters
    ----------
    k_squared : wp.array3d, shape (Nx, Ny, Nz_rfft)
        Squared magnitude of k-vectors at each grid point.
    miller_x, miller_y, miller_z : wp.array
        Miller indices (from fftfreq/rfftfreq).
    beta : wp.array, shape (1,)
        Dispersion Ewald splitting parameter.
    volume : wp.array, shape (1,)
        Unit cell volume.
    mesh_nx, mesh_ny, mesh_nz : wp.int32
        Full mesh dimensions (Nz is the full size, not rfft size).
    spline_order : wp.int32
        B-spline order (1-4). Order 4 (cubic) recommended.
    green_function : wp.array3d
        OUTPUT: Dispersion Green's function values per grid point.
    structure_factor_sq : wp.array3d
        OUTPUT: :math:`|B(k)|^2` structure factor squared per grid point.

    Notes
    -----
    - k=0 (grid point [0,0,0]) is explicitly set to zero (the m=0 term
      is excluded from the dispersion reciprocal sum).
    - Near-zero k² values are clamped to zero in the Green's function.
    - Structure factor is clamped to avoid division by zero in dealiasing.
    """
    i, j, k = wp.tid()

    k_sq = k_squared[i, j, k]
    beta_ = beta[0]
    volume_ = volume[0]
    mi_x = miller_x[i]
    mi_y = miller_y[j]
    mi_z = miller_z[k]

    # Get dtype-specific constants
    zero = type(k_sq)(0.0)
    two = type(k_sq)(2.0)
    four = type(k_sq)(4.0)
    threshold = type(k_sq)(1e-10)
    clamp_threshold = type(k_sq)(1e-10)
    pi32 = type(k_sq)(PI_3_2)

    # Green's function:
    #   x^2 = k^2 / (4 * beta^2)
    #   G(k) = -(pi^(3/2) * beta^3) / (2V) * f(x)
    # The minus sign reflects the attractive r^-6 dispersion: V_recip is
    # the lattice-summed long-range complement -(1/2)Σ √C6·√C6·(1-g)/r^6
    # so the FT kernel acquires the same overall sign.
    if k_sq < threshold:
        green_function[i, j, k] = zero
    else:
        x_sq = k_sq / (four * beta_ * beta_)
        f_val = dispersion_green_f(x_sq)
        beta_cubed = beta_ * beta_ * beta_
        green_function[i, j, k] = -pi32 * beta_cubed * f_val / (two * volume_)

    if i == 0 and j == 0 and k == 0:
        green_function[i, j, k] = zero

    # Structure factor: sinc(mi_x/Nx) * sinc(mi_y/Ny) * sinc(mi_z/Nz)
    sinc_x = compute_sinc(mi_x / type(mi_x)(mesh_nx))
    sinc_y = compute_sinc(mi_y / type(mi_y)(mesh_ny))
    sinc_z = compute_sinc(mi_z / type(mi_z)(mesh_nz))

    sinc_product = sinc_x * sinc_y * sinc_z

    # Raise to spline_order power
    sf = sinc_product
    for _ in range(1, 4):  # Max order 4
        if _ < spline_order:
            sf = sf * sinc_product

    if sf < clamp_threshold:
        sf = clamp_threshold

    structure_factor_sq[i, j, k] = sf * sf


@wp.kernel
def _batch_pme_dispersion_green_structure_factor_kernel(
    k_squared: wp.array4d(dtype=Any),  # (B, Nx, Ny, Nz_rfft)
    miller_x: wp.array(dtype=Any),  # (Nx,)
    miller_y: wp.array(dtype=Any),  # (Ny,)
    miller_z: wp.array(dtype=Any),  # (Nz_rfft,)
    beta: wp.array(dtype=Any),  # (B,)
    volumes: wp.array(dtype=Any),  # (B,)
    mesh_nx: wp.int32,
    mesh_ny: wp.int32,
    mesh_nz: wp.int32,
    spline_order: wp.int32,
    green_function: wp.array4d(dtype=Any),  # (B, Nx, Ny, Nz_rfft)
    structure_factor_sq: wp.array3d(dtype=Any),  # (Nx, Ny, Nz_rfft)
):
    """Compute dispersion PME Green's function and B-spline correction (batched).

    Batched version. Each system can have different beta and volume values,
    but shares the same mesh dimensions.

    Launch Grid
    -----------
    dim = [B, Nx, Ny, Nz_rfft]
    """
    batch_idx, i, j, k = wp.tid()

    k_sq = k_squared[batch_idx, i, j, k]
    system_beta = beta[batch_idx]
    system_volume = volumes[batch_idx]
    mi_x = miller_x[i]
    mi_y = miller_y[j]
    mi_z = miller_z[k]

    zero = type(k_sq)(0.0)
    two = type(k_sq)(2.0)
    four = type(k_sq)(4.0)
    threshold = type(k_sq)(1e-10)
    clamp_threshold = type(k_sq)(1e-10)
    pi32 = type(k_sq)(PI_3_2)

    if k_sq < threshold:
        green_function[batch_idx, i, j, k] = zero
    else:
        x_sq = k_sq / (four * system_beta * system_beta)
        f_val = dispersion_green_f(x_sq)
        beta_cubed = system_beta * system_beta * system_beta
        green_function[batch_idx, i, j, k] = (
            -pi32 * beta_cubed * f_val / (two * system_volume)
        )

    if i == 0 and j == 0 and k == 0:
        green_function[batch_idx, i, j, k] = zero

    # Structure factor only computed once per k-point (at batch_idx=0)
    if batch_idx == wp.int32(0):
        sinc_x = compute_sinc(mi_x / type(mi_x)(mesh_nx))
        sinc_y = compute_sinc(mi_y / type(mi_y)(mesh_ny))
        sinc_z = compute_sinc(mi_z / type(mi_z)(mesh_nz))

        sinc_product = sinc_x * sinc_y * sinc_z

        sf = sinc_product
        for _ in range(1, 4):
            if _ < spline_order:
                sf = sf * sinc_product

        if sf < clamp_threshold:
            sf = clamp_threshold

        structure_factor_sq[i, j, k] = sf * sf


###########################################################################################
########################### Dispersion Self-Energy Correction #############################
###########################################################################################


@wp.kernel
def _pme_dispersion_self_energy_kernel(
    c6_coefficients: wp.array(dtype=Any),  # (N,) homoatomic C6_ii
    beta: wp.array(dtype=Any),  # (1,)
    energy_correction: wp.array(dtype=Any),  # OUT (N,) per-atom contribution
):
    """Compute per-atom dispersion self-energy correction.

    Each atom's contribution is

    .. math::

        V_{\\text{self},i} = -\\frac{\\beta^6}{12} C_{6,ii}.

    The total :math:`V_{\\text{self}}` is obtained by summing over atoms; this
    is handled in the binding layer.

    Launch Grid
    -----------
    dim = [num_atoms]
    """
    atom_idx = wp.tid()

    c6 = c6_coefficients[atom_idx]
    beta_ = beta[0]

    twelve = type(c6)(12.0)
    beta_sq = beta_ * beta_
    beta_sixth = beta_sq * beta_sq * beta_sq

    energy_correction[atom_idx] = -beta_sixth * c6 / twelve


@wp.kernel
def _batch_pme_dispersion_self_energy_kernel(
    c6_coefficients: wp.array(dtype=Any),  # (N_total,) homoatomic C6_ii
    batch_idx: wp.array(dtype=wp.int32),  # (N_total,) system index per atom
    beta: wp.array(dtype=Any),  # (B,)
    energy_correction: wp.array(dtype=Any),  # OUT (N_total,)
):
    """Batched per-atom dispersion self-energy correction.

    Each atom uses its system's beta via ``batch_idx``.

    Launch Grid
    -----------
    dim = [num_atoms_total]
    """
    atom_idx = wp.tid()

    system_id = batch_idx[atom_idx]
    c6 = c6_coefficients[atom_idx]
    beta_ = beta[system_id]

    twelve = type(c6)(12.0)
    beta_sq = beta_ * beta_
    beta_sixth = beta_sq * beta_sq * beta_sq

    energy_correction[atom_idx] = -beta_sixth * c6 / twelve


###########################################################################################
########################### Real-Space LJ-PME Kernels #####################################
###########################################################################################
#
# Real-space (short-range) term of LJ-PME:
#
#   V_pair(r) = C12_ij / r^12 - C6_ij * g(beta*r) / r^6
#
# where g(x) = exp(-x^2) * (1 + x^2 + x^4/2) is the Wennberg damping function
# (its long-range complement (1 - g(beta*r))/r^6 is what reciprocal-space PME
# evaluates). Geometric combination rules:
#
#   C6_ij  = sqrt(c6_ii * c6_jj)
#   C12_ij = sqrt(c12_ii * c12_jj)
#
# Forces (derived from -dV/dr * r_hat with r_hat = (r_i - r_j)/r):
#
#   F_i = [ 12*C12/r^13  -  C6 * beta^6 * exp(-beta^2 r^2) / r  -  6*C6*g/r^7 ] * r_hat
#       = [ 12*C12/r^14  -  C6 * beta^6 * exp(-beta^2 r^2) / r^2 -  6*C6*g/r^8 ] * r_ij
#
# The bracket is precomputed as ``force_mag_over_r`` and applied to r_ij.
# Sign convention: r_ij = r_i - r_j, half neighbor list — pair appears once
# in the (i, j) row; Newton's third law is applied to atom j inside the kernel.


@wp.func
def _lj_pme_damping_g(x_sq: Any) -> Any:
    """Wennberg damping factor g(x) = exp(-x^2)*(1 + x^2 + x^4/2). Takes x^2."""
    one = type(x_sq)(1.0)
    half = type(x_sq)(0.5)
    return wp.exp(-x_sq) * (one + x_sq + half * x_sq * x_sq)


@wp.kernel
def _lj_pme_real_space_energy_kernel(
    positions: wp.array(dtype=Any),  # (N,) vec3
    c6_coefficients: wp.array(dtype=Any),  # (N,)
    c12_coefficients: wp.array(dtype=Any),  # (N,)
    cell: wp.array(dtype=Any),  # (1,) mat33
    neighbor_matrix: wp.array2d(dtype=wp.int32),
    neighbor_matrix_shifts: wp.array2d(dtype=wp.vec3i),
    num_neighbors: wp.array(dtype=wp.int32),
    beta: wp.array(dtype=Any),  # (1,)
    cutoff: wp.array(dtype=Any),  # (1,)
    mask_value: wp.int32,
    half_neighbor_list: wp.bool,
    atomic_energies: wp.array(dtype=Any),  # OUT (N,)
):
    """Real-space LJ-PME energy with neighbor matrix.

    Computes ``V = C12/r^12 - C6 * g(beta*r) / r^6`` for each pair within the
    cutoff. Energy accumulation depends on ``half_neighbor_list``:

    * half list: each pair (i, j) appears once; adds V/2 to both i and j.
    * full list: each pair appears twice (once in i's row, once in j's row);
      adds V/2 to atom_i only (the matching (j, i) row handles atom j).

    Launch Grid
    -----------
    dim = [num_atoms]
    """
    atom_i = wp.tid()
    num_atoms = positions.shape[0]
    max_neighbors = neighbor_matrix.shape[1]
    if atom_i >= num_atoms:
        return

    ri = positions[atom_i]
    c6_i = wp.float64(c6_coefficients[atom_i])
    c12_i = wp.float64(c12_coefficients[atom_i])
    cell_t = wp.transpose(cell[0])
    beta_ = wp.float64(beta[0])
    cut = wp.float64(cutoff[0])
    cutoff_sq = cut * cut

    n_neighbors = num_neighbors[atom_i]
    for slot in range(n_neighbors):
        if slot >= max_neighbors:
            break
        j = neighbor_matrix[atom_i, slot]
        if j == mask_value or j >= num_atoms:
            continue

        rj = positions[j]
        shift = neighbor_matrix_shifts[atom_i, slot]
        shift_vec = cell_t * type(ri)(
            type(ri[0])(shift[0]),
            type(ri[0])(shift[1]),
            type(ri[0])(shift[2]),
        )
        r_ij = ri - rj - shift_vec
        r_sq = wp.float64(wp.dot(r_ij, r_ij))
        if r_sq >= cutoff_sq or r_sq < wp.float64(1e-10):
            continue

        c6_j = wp.float64(c6_coefficients[j])
        c12_j = wp.float64(c12_coefficients[j])
        c6_ij = wp.sqrt(c6_i * c6_j)
        c12_ij = wp.sqrt(c12_i * c12_j)

        r_sq_2 = r_sq * r_sq
        r_sq_3 = r_sq_2 * r_sq
        r_sq_6 = r_sq_3 * r_sq_3
        inv_r6 = wp.float64(1.0) / r_sq_3
        inv_r12 = wp.float64(1.0) / r_sq_6

        x_sq = beta_ * beta_ * r_sq
        g = _lj_pme_damping_g(x_sq)

        pair_energy = c12_ij * inv_r12 - c6_ij * g * inv_r6
        half_energy = wp.float64(0.5) * pair_energy
        wp.atomic_add(atomic_energies, atom_i, type(atomic_energies[0])(half_energy))
        if half_neighbor_list:
            wp.atomic_add(atomic_energies, j, type(atomic_energies[0])(half_energy))


@wp.kernel
def _lj_pme_real_space_energy_forces_kernel(
    positions: wp.array(dtype=Any),
    c6_coefficients: wp.array(dtype=Any),
    c12_coefficients: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    neighbor_matrix: wp.array2d(dtype=wp.int32),
    neighbor_matrix_shifts: wp.array2d(dtype=wp.vec3i),
    num_neighbors: wp.array(dtype=wp.int32),
    beta: wp.array(dtype=Any),
    cutoff: wp.array(dtype=Any),
    mask_value: wp.int32,
    half_neighbor_list: wp.bool,
    atomic_energies: wp.array(dtype=Any),
    atomic_forces: wp.array(dtype=Any),  # OUT (N,) vec3
):
    """Real-space LJ-PME energy and forces.

    Half list: applies Newton's third law and updates both i and j.
    Full list: updates only atom_i (the matching (j, i) row handles j).

    Force on atom i:

        F_i = [ 12 C12/r^14 - C6 beta^6 exp(-beta^2 r^2)/r^2 - 6 C6 g/r^8 ] * r_ij
    """
    atom_i = wp.tid()
    num_atoms = positions.shape[0]
    max_neighbors = neighbor_matrix.shape[1]
    if atom_i >= num_atoms:
        return

    ri = positions[atom_i]
    c6_i = wp.float64(c6_coefficients[atom_i])
    c12_i = wp.float64(c12_coefficients[atom_i])
    cell_t = wp.transpose(cell[0])
    beta_ = wp.float64(beta[0])
    cut = wp.float64(cutoff[0])
    cutoff_sq = cut * cut

    force_acc = type(ri)(
        type(ri[0])(0.0),
        type(ri[0])(0.0),
        type(ri[0])(0.0),
    )

    n_neighbors = num_neighbors[atom_i]
    for slot in range(n_neighbors):
        if slot >= max_neighbors:
            break
        j = neighbor_matrix[atom_i, slot]
        if j == mask_value or j >= num_atoms:
            continue

        rj = positions[j]
        shift = neighbor_matrix_shifts[atom_i, slot]
        shift_vec = cell_t * type(ri)(
            type(ri[0])(shift[0]),
            type(ri[0])(shift[1]),
            type(ri[0])(shift[2]),
        )
        r_ij = ri - rj - shift_vec
        r_sq = wp.float64(wp.dot(r_ij, r_ij))
        if r_sq >= cutoff_sq or r_sq < wp.float64(1e-10):
            continue

        c6_j = wp.float64(c6_coefficients[j])
        c12_j = wp.float64(c12_coefficients[j])
        c6_ij = wp.sqrt(c6_i * c6_j)
        c12_ij = wp.sqrt(c12_i * c12_j)

        r_sq_2 = r_sq * r_sq
        r_sq_3 = r_sq_2 * r_sq
        r_sq_4 = r_sq_2 * r_sq_2
        r_sq_6 = r_sq_3 * r_sq_3
        r_sq_7 = r_sq_4 * r_sq_3
        inv_r2 = wp.float64(1.0) / r_sq
        inv_r6 = wp.float64(1.0) / r_sq_3
        inv_r8 = wp.float64(1.0) / r_sq_4
        inv_r12 = wp.float64(1.0) / r_sq_6
        inv_r14 = wp.float64(1.0) / r_sq_7

        beta_sq = beta_ * beta_
        x_sq = beta_sq * r_sq
        beta_sixth = beta_sq * beta_sq * beta_sq
        exp_x_sq = wp.exp(-x_sq)
        g = exp_x_sq * (wp.float64(1.0) + x_sq + wp.float64(0.5) * x_sq * x_sq)

        pair_energy = c12_ij * inv_r12 - c6_ij * g * inv_r6
        # F_i = (12 C12/r^14 - C6 beta^6 e^(-beta^2 r^2)/r^2 - 6 C6 g/r^8) * r_ij
        force_mag_over_r = (
            wp.float64(12.0) * c12_ij * inv_r14
            - c6_ij * beta_sixth * exp_x_sq * inv_r2
            - wp.float64(6.0) * c6_ij * g * inv_r8
        )

        half_energy = wp.float64(0.5) * pair_energy
        wp.atomic_add(atomic_energies, atom_i, type(atomic_energies[0])(half_energy))
        if half_neighbor_list:
            wp.atomic_add(atomic_energies, j, type(atomic_energies[0])(half_energy))

        force_ij = type(ri)(
            type(ri[0])(force_mag_over_r) * r_ij[0],
            type(ri[0])(force_mag_over_r) * r_ij[1],
            type(ri[0])(force_mag_over_r) * r_ij[2],
        )
        force_acc += force_ij
        if half_neighbor_list:
            wp.atomic_sub(atomic_forces, j, force_ij)
    wp.atomic_add(atomic_forces, atom_i, force_acc)


@wp.kernel
def _lj_pme_real_space_energy_forces_virial_kernel(
    positions: wp.array(dtype=Any),
    c6_coefficients: wp.array(dtype=Any),
    c12_coefficients: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    neighbor_matrix: wp.array2d(dtype=wp.int32),
    neighbor_matrix_shifts: wp.array2d(dtype=wp.vec3i),
    num_neighbors: wp.array(dtype=wp.int32),
    beta: wp.array(dtype=Any),
    cutoff: wp.array(dtype=Any),
    mask_value: wp.int32,
    half_neighbor_list: wp.bool,
    atomic_energies: wp.array(dtype=Any),
    atomic_forces: wp.array(dtype=Any),
    virial: wp.array(dtype=Any),  # OUT (9,) flattened 3x3
):
    """Real-space LJ-PME energy, forces, and virial.

    Virial convention: ``W_{ab} = sum_{i<j} r_ij,a * F_ij,b`` where F_ij is the
    force on atom i due to the pair. Output is the flattened 9-element matrix.

    For ``half_neighbor_list=False`` (full neighbor matrix), each pair appears
    twice; the per-pair virial contribution is scaled by 1/2 to compensate.
    """
    atom_i = wp.tid()
    num_atoms = positions.shape[0]
    max_neighbors = neighbor_matrix.shape[1]
    if atom_i >= num_atoms:
        return

    ri = positions[atom_i]
    c6_i = wp.float64(c6_coefficients[atom_i])
    c12_i = wp.float64(c12_coefficients[atom_i])
    cell_t = wp.transpose(cell[0])
    beta_ = wp.float64(beta[0])
    cut = wp.float64(cutoff[0])
    cutoff_sq = cut * cut

    force_acc = type(ri)(
        type(ri[0])(0.0),
        type(ri[0])(0.0),
        type(ri[0])(0.0),
    )

    vir_xx = wp.float64(0.0)
    vir_xy = wp.float64(0.0)
    vir_xz = wp.float64(0.0)
    vir_yx = wp.float64(0.0)
    vir_yy = wp.float64(0.0)
    vir_yz = wp.float64(0.0)
    vir_zx = wp.float64(0.0)
    vir_zy = wp.float64(0.0)
    vir_zz = wp.float64(0.0)

    n_neighbors = num_neighbors[atom_i]
    for slot in range(n_neighbors):
        if slot >= max_neighbors:
            break
        j = neighbor_matrix[atom_i, slot]
        if j == mask_value or j >= num_atoms:
            continue

        rj = positions[j]
        shift = neighbor_matrix_shifts[atom_i, slot]
        shift_vec = cell_t * type(ri)(
            type(ri[0])(shift[0]),
            type(ri[0])(shift[1]),
            type(ri[0])(shift[2]),
        )
        r_ij = ri - rj - shift_vec
        r_sq = wp.float64(wp.dot(r_ij, r_ij))
        if r_sq >= cutoff_sq or r_sq < wp.float64(1e-10):
            continue

        c6_j = wp.float64(c6_coefficients[j])
        c12_j = wp.float64(c12_coefficients[j])
        c6_ij = wp.sqrt(c6_i * c6_j)
        c12_ij = wp.sqrt(c12_i * c12_j)

        r_sq_2 = r_sq * r_sq
        r_sq_3 = r_sq_2 * r_sq
        r_sq_4 = r_sq_2 * r_sq_2
        r_sq_6 = r_sq_3 * r_sq_3
        r_sq_7 = r_sq_4 * r_sq_3
        inv_r2 = wp.float64(1.0) / r_sq
        inv_r6 = wp.float64(1.0) / r_sq_3
        inv_r8 = wp.float64(1.0) / r_sq_4
        inv_r12 = wp.float64(1.0) / r_sq_6
        inv_r14 = wp.float64(1.0) / r_sq_7

        beta_sq = beta_ * beta_
        x_sq = beta_sq * r_sq
        beta_sixth = beta_sq * beta_sq * beta_sq
        exp_x_sq = wp.exp(-x_sq)
        g = exp_x_sq * (wp.float64(1.0) + x_sq + wp.float64(0.5) * x_sq * x_sq)

        pair_energy = c12_ij * inv_r12 - c6_ij * g * inv_r6
        force_mag_over_r = (
            wp.float64(12.0) * c12_ij * inv_r14
            - c6_ij * beta_sixth * exp_x_sq * inv_r2
            - wp.float64(6.0) * c6_ij * g * inv_r8
        )

        half_energy = wp.float64(0.5) * pair_energy
        wp.atomic_add(atomic_energies, atom_i, type(atomic_energies[0])(half_energy))
        if half_neighbor_list:
            wp.atomic_add(atomic_energies, j, type(atomic_energies[0])(half_energy))

        force_ij = type(ri)(
            type(ri[0])(force_mag_over_r) * r_ij[0],
            type(ri[0])(force_mag_over_r) * r_ij[1],
            type(ri[0])(force_mag_over_r) * r_ij[2],
        )
        force_acc += force_ij
        if half_neighbor_list:
            wp.atomic_sub(atomic_forces, j, force_ij)

        r_ij_0 = wp.float64(r_ij[0])
        r_ij_1 = wp.float64(r_ij[1])
        r_ij_2 = wp.float64(r_ij[2])
        f_ij_0 = wp.float64(force_ij[0])
        f_ij_1 = wp.float64(force_ij[1])
        f_ij_2 = wp.float64(force_ij[2])
        # Virial scaling: full list double-counts each pair, so scale by 1/2.
        vir_scale = wp.float64(1.0) if half_neighbor_list else wp.float64(0.5)
        vir_xx += vir_scale * (r_ij_0 * f_ij_0)
        vir_xy += vir_scale * (r_ij_0 * f_ij_1)
        vir_xz += vir_scale * (r_ij_0 * f_ij_2)
        vir_yx += vir_scale * (r_ij_1 * f_ij_0)
        vir_yy += vir_scale * (r_ij_1 * f_ij_1)
        vir_yz += vir_scale * (r_ij_1 * f_ij_2)
        vir_zx += vir_scale * (r_ij_2 * f_ij_0)
        vir_zy += vir_scale * (r_ij_2 * f_ij_1)
        vir_zz += vir_scale * (r_ij_2 * f_ij_2)

    wp.atomic_add(atomic_forces, atom_i, force_acc)

    wp.atomic_add(virial, 0, type(virial[0])(vir_xx))
    wp.atomic_add(virial, 1, type(virial[0])(vir_xy))
    wp.atomic_add(virial, 2, type(virial[0])(vir_xz))
    wp.atomic_add(virial, 3, type(virial[0])(vir_yx))
    wp.atomic_add(virial, 4, type(virial[0])(vir_yy))
    wp.atomic_add(virial, 5, type(virial[0])(vir_yz))
    wp.atomic_add(virial, 6, type(virial[0])(vir_zx))
    wp.atomic_add(virial, 7, type(virial[0])(vir_zy))
    wp.atomic_add(virial, 8, type(virial[0])(vir_zz))


###########################################################################################
########################### Kernel Overloads for Dtype Flexibility ########################
###########################################################################################

_T = [wp.float32, wp.float64]
_V = [wp.vec3f, wp.vec3d]
_M = [wp.mat33f, wp.mat33d]

_pme_dispersion_green_structure_factor_kernel_overload = {}
_batch_pme_dispersion_green_structure_factor_kernel_overload = {}
_pme_dispersion_self_energy_kernel_overload = {}
_batch_pme_dispersion_self_energy_kernel_overload = {}
_lj_pme_real_space_energy_kernel_overload = {}
_lj_pme_real_space_energy_forces_kernel_overload = {}
_lj_pme_real_space_energy_forces_virial_kernel_overload = {}

for t in _T:
    _pme_dispersion_green_structure_factor_kernel_overload[t] = wp.overload(
        _pme_dispersion_green_structure_factor_kernel,
        [
            wp.array3d(dtype=t),  # k_squared
            wp.array(dtype=t),  # miller_x
            wp.array(dtype=t),  # miller_y
            wp.array(dtype=t),  # miller_z
            wp.array(dtype=t),  # beta
            wp.array(dtype=t),  # volume
            wp.int32,  # mesh_nx
            wp.int32,  # mesh_ny
            wp.int32,  # mesh_nz
            wp.int32,  # spline_order
            wp.array3d(dtype=t),  # green_function
            wp.array3d(dtype=t),  # structure_factor_sq
        ],
    )

    _batch_pme_dispersion_green_structure_factor_kernel_overload[t] = wp.overload(
        _batch_pme_dispersion_green_structure_factor_kernel,
        [
            wp.array4d(dtype=t),  # k_squared
            wp.array(dtype=t),  # miller_x
            wp.array(dtype=t),  # miller_y
            wp.array(dtype=t),  # miller_z
            wp.array(dtype=t),  # beta
            wp.array(dtype=t),  # volumes
            wp.int32,  # mesh_nx
            wp.int32,  # mesh_ny
            wp.int32,  # mesh_nz
            wp.int32,  # spline_order
            wp.array4d(dtype=t),  # green_function
            wp.array3d(dtype=t),  # structure_factor_sq
        ],
    )

    _pme_dispersion_self_energy_kernel_overload[t] = wp.overload(
        _pme_dispersion_self_energy_kernel,
        [
            wp.array(dtype=t),  # c6_coefficients
            wp.array(dtype=t),  # beta
            wp.array(dtype=t),  # energy_correction
        ],
    )

    _batch_pme_dispersion_self_energy_kernel_overload[t] = wp.overload(
        _batch_pme_dispersion_self_energy_kernel,
        [
            wp.array(dtype=t),  # c6_coefficients
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array(dtype=t),  # beta
            wp.array(dtype=t),  # energy_correction
        ],
    )

for t, v, m in zip(_T, _V, _M):
    _lj_pme_real_space_energy_kernel_overload[t] = wp.overload(
        _lj_pme_real_space_energy_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=t),  # c6_coefficients
            wp.array(dtype=t),  # c12_coefficients
            wp.array(dtype=m),  # cell
            wp.array2d(dtype=wp.int32),  # neighbor_matrix
            wp.array2d(dtype=wp.vec3i),  # neighbor_matrix_shifts
            wp.array(dtype=wp.int32),  # num_neighbors
            wp.array(dtype=t),  # beta
            wp.array(dtype=t),  # cutoff
            wp.int32,  # mask_value
            wp.bool,  # half_neighbor_list
            wp.array(dtype=t),  # atomic_energies
        ],
    )

    _lj_pme_real_space_energy_forces_kernel_overload[t] = wp.overload(
        _lj_pme_real_space_energy_forces_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=t),  # c6_coefficients
            wp.array(dtype=t),  # c12_coefficients
            wp.array(dtype=m),  # cell
            wp.array2d(dtype=wp.int32),  # neighbor_matrix
            wp.array2d(dtype=wp.vec3i),  # neighbor_matrix_shifts
            wp.array(dtype=wp.int32),  # num_neighbors
            wp.array(dtype=t),  # beta
            wp.array(dtype=t),  # cutoff
            wp.int32,  # mask_value
            wp.bool,  # half_neighbor_list
            wp.array(dtype=t),  # atomic_energies
            wp.array(dtype=v),  # atomic_forces
        ],
    )

    _lj_pme_real_space_energy_forces_virial_kernel_overload[t] = wp.overload(
        _lj_pme_real_space_energy_forces_virial_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=t),  # c6_coefficients
            wp.array(dtype=t),  # c12_coefficients
            wp.array(dtype=m),  # cell
            wp.array2d(dtype=wp.int32),  # neighbor_matrix
            wp.array2d(dtype=wp.vec3i),  # neighbor_matrix_shifts
            wp.array(dtype=wp.int32),  # num_neighbors
            wp.array(dtype=t),  # beta
            wp.array(dtype=t),  # cutoff
            wp.int32,  # mask_value
            wp.bool,  # half_neighbor_list
            wp.array(dtype=t),  # atomic_energies
            wp.array(dtype=v),  # atomic_forces
            wp.array(dtype=t),  # virial
        ],
    )


###########################################################################################
########################### Warp Launcher Functions #######################################
###########################################################################################


def pme_dispersion_green_structure_factor(
    k_squared: wp.array,
    miller_x: wp.array,
    miller_y: wp.array,
    miller_z: wp.array,
    beta: wp.array,
    volume: wp.array,
    mesh_nx: int,
    mesh_ny: int,
    mesh_nz: int,
    spline_order: int,
    green_function: wp.array,
    structure_factor_sq: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Framework-agnostic launcher for the single-system dispersion Green's function.

    Computes :math:`G_{\\text{disp}}(k) = (\\pi^{3/2}\\beta^3/(2V))\\cdot f(|k|/(2\\beta))`
    and the B-spline structure-factor squared :math:`|B(k)|^2`.

    Parameters
    ----------
    k_squared : wp.array, shape (Nx, Ny, Nz_rfft)
        Squared magnitude of k-vectors at each grid point.
    miller_x, miller_y, miller_z : wp.array
        Miller indices.
    beta : wp.array, shape (1,)
        Dispersion Ewald splitting parameter.
    volume : wp.array, shape (1,)
        Unit cell volume.
    mesh_nx, mesh_ny, mesh_nz : int
        Full mesh dimensions.
    spline_order : int
        B-spline order (1-4).
    green_function : wp.array, shape (Nx, Ny, Nz_rfft)
        OUTPUT: dispersion Green's function values.
    structure_factor_sq : wp.array, shape (Nx, Ny, Nz_rfft)
        OUTPUT: :math:`|B(k)|^2`.
    wp_dtype : type
        Warp scalar dtype (wp.float32 or wp.float64).
    device : str | None
        Warp device string. If None, inferred from arrays.
    """
    nx, ny, nz_rfft = k_squared.shape[0], k_squared.shape[1], k_squared.shape[2]

    kernel = _pme_dispersion_green_structure_factor_kernel_overload[wp_dtype]
    wp.launch(
        kernel,
        dim=(nx, ny, nz_rfft),
        inputs=[
            k_squared,
            miller_x,
            miller_y,
            miller_z,
            beta,
            volume,
            wp.int32(mesh_nx),
            wp.int32(mesh_ny),
            wp.int32(mesh_nz),
            wp.int32(spline_order),
        ],
        outputs=[green_function, structure_factor_sq],
        device=device,
    )


def batch_pme_dispersion_green_structure_factor(
    k_squared: wp.array,
    miller_x: wp.array,
    miller_y: wp.array,
    miller_z: wp.array,
    beta: wp.array,
    volumes: wp.array,
    mesh_nx: int,
    mesh_ny: int,
    mesh_nz: int,
    spline_order: int,
    green_function: wp.array,
    structure_factor_sq: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Framework-agnostic batched launcher for the dispersion Green's function.

    Each system can have different beta and volume values but shares the same
    mesh dimensions.
    """
    num_systems = k_squared.shape[0]
    nx, ny, nz_rfft = k_squared.shape[1], k_squared.shape[2], k_squared.shape[3]

    kernel = _batch_pme_dispersion_green_structure_factor_kernel_overload[wp_dtype]
    wp.launch(
        kernel,
        dim=(num_systems, nx, ny, nz_rfft),
        inputs=[
            k_squared,
            miller_x,
            miller_y,
            miller_z,
            beta,
            volumes,
            wp.int32(mesh_nx),
            wp.int32(mesh_ny),
            wp.int32(mesh_nz),
            wp.int32(spline_order),
        ],
        outputs=[green_function, structure_factor_sq],
        device=device,
    )


def pme_dispersion_self_energy(
    c6_coefficients: wp.array,
    beta: wp.array,
    energy_correction: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Framework-agnostic launcher for the per-atom dispersion self-energy.

    Computes :math:`V_{\\text{self},i} = -\\beta^6 C_{6,ii}/12` for each atom.
    The binding layer reduces over atoms to obtain the total :math:`V_{\\text{self}}`.

    Parameters
    ----------
    c6_coefficients : wp.array, shape (N,)
        Per-atom homoatomic :math:`C_{6,ii}` values.
    beta : wp.array, shape (1,)
        Dispersion Ewald splitting parameter.
    energy_correction : wp.array, shape (N,)
        OUTPUT: per-atom self-energy contribution.
    wp_dtype : type
        Warp scalar dtype.
    device : str | None
        Warp device string.
    """
    num_atoms = c6_coefficients.shape[0]

    kernel = _pme_dispersion_self_energy_kernel_overload[wp_dtype]
    wp.launch(
        kernel,
        dim=num_atoms,
        inputs=[c6_coefficients, beta],
        outputs=[energy_correction],
        device=device,
    )


def batch_pme_dispersion_self_energy(
    c6_coefficients: wp.array,
    batch_idx: wp.array,
    beta: wp.array,
    energy_correction: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Batched framework-agnostic launcher for the per-atom dispersion self-energy."""
    num_atoms = c6_coefficients.shape[0]

    kernel = _batch_pme_dispersion_self_energy_kernel_overload[wp_dtype]
    wp.launch(
        kernel,
        dim=num_atoms,
        inputs=[c6_coefficients, batch_idx, beta],
        outputs=[energy_correction],
        device=device,
    )


def lj_pme_real_space_energy(
    positions: wp.array,
    c6_coefficients: wp.array,
    c12_coefficients: wp.array,
    cell: wp.array,
    neighbor_matrix: wp.array,
    neighbor_matrix_shifts: wp.array,
    num_neighbors: wp.array,
    beta: wp.array,
    cutoff: wp.array,
    mask_value: int,
    atomic_energies: wp.array,
    wp_dtype: type,
    half_neighbor_list: bool = True,
    device: str | None = None,
) -> None:
    """Real-space LJ-PME energy launcher (single system, neighbor matrix)."""
    num_atoms = positions.shape[0]
    kernel = _lj_pme_real_space_energy_kernel_overload[wp_dtype]
    wp.launch(
        kernel,
        dim=num_atoms,
        inputs=[
            positions,
            c6_coefficients,
            c12_coefficients,
            cell,
            neighbor_matrix,
            neighbor_matrix_shifts,
            num_neighbors,
            beta,
            cutoff,
            wp.int32(mask_value),
            bool(half_neighbor_list),
        ],
        outputs=[atomic_energies],
        device=device,
    )


def lj_pme_real_space_energy_forces(
    positions: wp.array,
    c6_coefficients: wp.array,
    c12_coefficients: wp.array,
    cell: wp.array,
    neighbor_matrix: wp.array,
    neighbor_matrix_shifts: wp.array,
    num_neighbors: wp.array,
    beta: wp.array,
    cutoff: wp.array,
    mask_value: int,
    atomic_energies: wp.array,
    atomic_forces: wp.array,
    wp_dtype: type,
    half_neighbor_list: bool = True,
    device: str | None = None,
) -> None:
    """Real-space LJ-PME energy and forces launcher."""
    num_atoms = positions.shape[0]
    kernel = _lj_pme_real_space_energy_forces_kernel_overload[wp_dtype]
    wp.launch(
        kernel,
        dim=num_atoms,
        inputs=[
            positions,
            c6_coefficients,
            c12_coefficients,
            cell,
            neighbor_matrix,
            neighbor_matrix_shifts,
            num_neighbors,
            beta,
            cutoff,
            wp.int32(mask_value),
            bool(half_neighbor_list),
        ],
        outputs=[atomic_energies, atomic_forces],
        device=device,
    )


def lj_pme_real_space_energy_forces_virial(
    positions: wp.array,
    c6_coefficients: wp.array,
    c12_coefficients: wp.array,
    cell: wp.array,
    neighbor_matrix: wp.array,
    neighbor_matrix_shifts: wp.array,
    num_neighbors: wp.array,
    beta: wp.array,
    cutoff: wp.array,
    mask_value: int,
    atomic_energies: wp.array,
    atomic_forces: wp.array,
    virial: wp.array,
    wp_dtype: type,
    half_neighbor_list: bool = True,
    device: str | None = None,
) -> None:
    """Real-space LJ-PME energy, forces, and virial launcher."""
    num_atoms = positions.shape[0]
    kernel = _lj_pme_real_space_energy_forces_virial_kernel_overload[wp_dtype]
    wp.launch(
        kernel,
        dim=num_atoms,
        inputs=[
            positions,
            c6_coefficients,
            c12_coefficients,
            cell,
            neighbor_matrix,
            neighbor_matrix_shifts,
            num_neighbors,
            beta,
            cutoff,
            wp.int32(mask_value),
            bool(half_neighbor_list),
        ],
        outputs=[atomic_energies, atomic_forces, virial],
        device=device,
    )


###########################################################################################
########################### Module Exports ################################################
###########################################################################################

__all__ = [
    # Reciprocal-space kernel overloads (PR1)
    "_pme_dispersion_green_structure_factor_kernel_overload",
    "_batch_pme_dispersion_green_structure_factor_kernel_overload",
    "_pme_dispersion_self_energy_kernel_overload",
    "_batch_pme_dispersion_self_energy_kernel_overload",
    # Real-space kernel overloads (PR2)
    "_lj_pme_real_space_energy_kernel_overload",
    "_lj_pme_real_space_energy_forces_kernel_overload",
    "_lj_pme_real_space_energy_forces_virial_kernel_overload",
    # Reciprocal-space Warp launchers (PR1)
    "pme_dispersion_green_structure_factor",
    "batch_pme_dispersion_green_structure_factor",
    "pme_dispersion_self_energy",
    "batch_pme_dispersion_self_energy",
    # Real-space Warp launchers (PR2)
    "lj_pme_real_space_energy",
    "lj_pme_real_space_energy_forces",
    "lj_pme_real_space_energy_forces_virial",
]
