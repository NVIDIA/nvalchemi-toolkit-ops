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
Unified Ewald Summation Kernels
===============================

This module provides GPU-accelerated Warp kernels for Ewald summation,
enabling efficient calculation of long-range Coulomb interactions. All kernels
support both single-system and batched calculations via the batch_idx parameter.

DTYPE FLEXIBILITY
=================

All kernels support both float32 and float64 input types via Warp's overload system:
- Input tensors (positions, charges, cell, alpha): float32 or float64
- Accumulators (energies, structure factors): Always float64 for numerical stability
- Forces: Match input positions dtype (float32 or float64)

Use the `_*_overload` dictionaries to select the appropriate kernel based on dtype.

MATHEMATICAL FORMULATION
========================

The Ewald method splits the Coulomb energy into tractable components:

.. math::

    E_{\\text{total}}(s) = E_{\\text{real}}(s) + E_{\\text{reciprocal}}(s) - E_{\\text{self}}(s) - E_{\\text{background}}(s)

Real-Space Component (damped short-range):

.. math::

    E_{\\text{real}}(s) = \\frac{1}{2} \\sum_{i \\neq j \\in s} q_i q_j \\frac{\\text{erfc}(\\alpha r_{ij})}{r_{ij}}

The erfc damping rapidly suppresses interactions beyond a cutoff distance.
Force:

.. math::

    F_{ij} = q_i q_j \\left[\\frac{\\text{erfc}(\\alpha r_{ij})}{r^2} + \\frac{2\\alpha}{\\sqrt{\\pi}} \\frac{\\exp(-\\alpha^2 r^2)}{r}\\right] \\hat{r}_{ij}

Reciprocal-Space Component (smooth long-range):

.. math::

    E_{\\text{reciprocal}}(s) = \\frac{1}{2} \\sum_{i \\in s} q_i \\phi_i

where :math:`\\phi_i = \\frac{1}{V} \\sum_{k \\neq 0} G(k) [S_{\\text{real}}(k) \\cos(k \\cdot r_i) + S_{\\text{imag}}(k) \\sin(k \\cdot r_i)]`

Green's function:

.. math::

    G(k) = \\frac{8\\pi}{k^2} \\exp\\left(-\\frac{k^2}{4\\alpha^2}\\right)

Structure factors:

.. math::

    S(k) = \\sum_j q_j \\exp(ik \\cdot r_j)

    Note: G(k) uses 8*pi (not 4*pi) because we use half-space k-vectors, exploiting
    the symmetry S(-k) = S*(k). This halves the number of k-vectors while
    maintaining correct energies/forces.

Self-Energy Correction (removes spurious self-interaction):

.. math::

    E_{\\text{self}}(s) = \\sum_{i \\in s} \\frac{\\alpha}{\\sqrt{\\pi}} q_i^2

Background Correction (for non-neutral systems):

.. math::

    E_{\\text{background}}(s) = \\sum_{i \\in s} \\frac{\\pi}{2\\alpha^2 V} q_i Q_{\\text{total}}

KERNEL ORGANIZATION
===================

Real-Space Kernels:
    - _ewald_real_space_energy_kernel: Single-system, neighbor list format
    - _ewald_real_space_energy_forces_kernel: Single-system with forces
    - _ewald_real_space_energy_neighbor_matrix_kernel: Neighbor matrix format
    - _ewald_real_space_energy_forces_neighbor_matrix_kernel: Matrix with forces
    - _batch_ewald_real_space_*: Batched versions of above

Reciprocal-Space Kernels:
    - _ewald_reciprocal_space_energy_kernel_fill_structure_factors: Compute S(k)
    - _ewald_reciprocal_space_energy_kernel_compute_energy: Energy from S(k)
    - _ewald_reciprocal_space_energy_forces_kernel: Energy + forces from S(k)
    - _ewald_subtract_self_energy_kernel: Apply self + background corrections
    - _batch_ewald_reciprocal_space_*: Batched versions of above

PERFORMANCE TUNING
==================

Environment variables for performance tuning:

ALCH_EWALD_BATCH_BLOCK_SIZE (default: 16)
    Block size for batched structure factor computation. Each thread processes
    a block of atoms, reducing atomic contention. Benchmark results show:
    - 16 is optimal for most scenarios (2-3x faster than atom-major)
    - Atom-major (no blocking) only wins for very large atom counts (>100K atoms)
    - Tune this if you have unusual workloads (many small or few large systems)

REFERENCES
==========

- Ewald, P. P. (1921). Ann. Phys. 369, 253-287 (Original Ewald method)
- Kolafa, J. & Perram, J. W. (1992). Mol. Sim. 9, 351-368 (Parameter optimization)
- Essmann et al. (1995). J. Chem. Phys. 103, 8577 (PME method)
"""

import math
import os
from typing import Any

import warp as wp

from nvalchemiops.math import wp_erfc, wp_exp_kernel

# Mathematical constants
PI = math.pi
TWOPI = 2.0 * PI
FOURPI = 4.0 * PI
EIGHTPI = 8.0 * PI  # Used for half-space k-vector optimization (2x FOURPI)

# Block size for batch structure factor accumulation
# Benchmark results show 16 is optimal for most cases (except very large atom counts)
BATCH_BLOCK_SIZE = int(os.environ.get("ALCH_EWALD_BATCH_BLOCK_SIZE", 16))
BATCH_BLOCK_SIZE = BATCH_BLOCK_SIZE if BATCH_BLOCK_SIZE > 0 else 16


###########################################################################################
########################### Helper Functions (always float64) #############################
###########################################################################################


@wp.func
def _ewald_real_space_energy_kernel_compute_energy(
    qi: wp.float64,
    qj: wp.float64,
    distance: wp.float64,
    alpha: wp.float64,
) -> wp.float64:
    """Compute damped Coulomb energy for a single pair.

    Formula:

    .. math::

        E_{ij} = \\frac{1}{2} q_i q_j \\frac{\\text{erfc}(\\alpha r)}{r}

    The 0.5 factor accounts for pair double-counting when iterating
    over all (i,j) pairs.

    Parameters
    ----------
    qi, qj : wp.float64
        Charges of atoms i and j.
    distance : wp.float64
        Distance |r_j - r_i|.
    alpha : wp.float64
        Ewald splitting parameter.

    Returns
    -------
    wp.float64
        Damped Coulomb energy contribution.
    """
    return wp.float64(0.5) * qi * qj * wp_erfc(alpha * distance) / distance


@wp.func
def _ewald_real_space_force_magnitude(
    qi: wp.float64,
    qj: wp.float64,
    distance: wp.float64,
    alpha: wp.float64,
) -> wp.float64:
    """Compute damped Coulomb force magnitude factor for a single pair.

    Returns the scalar part of the force:

    .. math::

        F = q_i q_j \\left[\\frac{\\text{erfc}(\\alpha r)}{r^3} + \\frac{2\\alpha}{\\sqrt{\\pi}} \\frac{\\exp(-\\alpha^2 r^2)}{r^2}\\right]

    To get the force vector, multiply by the separation vector.

    Parameters
    ----------
    qi, qj : wp.float64
        Charges of atoms i and j.
    distance : wp.float64
        Distance |r_j - r_i|.
    alpha : wp.float64
        Ewald splitting parameter.

    Returns
    -------
    wp.float64
        Force magnitude factor.
    """
    two_over_sqrt_pi = wp.float64(2.0 / 1.7724538509055159)

    prefactor = wp.float64(0.5) * qi * qj
    alpha_r = alpha * distance
    alpha_r_squared = alpha_r * alpha_r

    erfc_alpha_r = wp_erfc(alpha_r)
    exp_term = wp.exp(-alpha_r_squared)

    # Force magnitude / r^2
    force_mag_over_r = erfc_alpha_r / (
        distance * distance * distance
    ) + two_over_sqrt_pi * alpha * exp_term / (distance * distance)
    return prefactor * force_mag_over_r


@wp.func
def _ewald_real_space_charge_grad_potential(
    distance: wp.float64,
    alpha: wp.float64,
) -> wp.float64:
    """Compute the damped Coulomb potential for charge gradient.

    Returns (1/2) * erfc(α·r) / r, which when multiplied by q_j gives
    the charge gradient contribution to atom i.

    For pair (i,j) with energy E_ij = (1/2) * q_i * q_j * erfc(α·r) / r:
        ∂E_ij/∂q_i = (1/2) * q_j * erfc(α·r) / r = potential * q_j
        ∂E_ij/∂q_j = (1/2) * q_i * erfc(α·r) / r = potential * q_i

    Parameters
    ----------
    distance : wp.float64
        Distance |r_j - r_i|.
    alpha : wp.float64
        Ewald splitting parameter.

    Returns
    -------
    wp.float64
        Potential factor for charge gradient computation.
    """
    return wp.float64(0.5) * wp_erfc(alpha * distance) / distance


###########################################################################################
########################### Real-Space Kernels (dtype-flexible) ###########################
###########################################################################################


@wp.kernel
def _ewald_real_space_energy_neighbor_matrix_kernel(
    positions: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    neighbor_matrix: wp.array2d(dtype=wp.int32),
    unit_shifts_matrix: wp.array2d(dtype=wp.vec3i),
    mask_value: wp.int32,
    alpha: wp.array(dtype=Any),
    pair_energies: wp.array(dtype=wp.float64),
):
    """Compute real-space Ewald energies using neighbor matrix format.

    Each thread processes one atom and loops over all its neighbors in the
    neighbor matrix. This 1D launch pattern is more efficient than 2D launch
    as it reduces thread divergence and improves memory access patterns.
    Invalid neighbors (marked with mask_value) are skipped. Pairs that are
    too close (less than 1e-8 distance) are also skipped.

    Launch Grid
    -----------
    dim = [N_atoms]

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        Atomic coordinates.
    charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Atomic charges.
    cell : wp.array, shape (1, 3, 3), dtype=wp.mat33f or wp.mat33d
        Unit cell matrix.
    neighbor_matrix : wp.array2d, shape (N, max_neighbors), dtype=wp.int32
        Neighbor indices. Entry [i, k] = j means atom j is the k-th neighbor of i.
        Invalid entries contain mask_value.
    unit_shifts_matrix : wp.array2d, shape (N, max_neighbors), dtype=wp.vec3i
        Periodic image shifts for each neighbor pair.
    mask_value : wp.int32
        Value indicating invalid/padded neighbor entries.
    alpha : wp.array, shape (1,), dtype=wp.float32 or wp.float64
        Ewald splitting parameter.
    pair_energies : wp.array, shape (N,), dtype=wp.float64
        OUTPUT: Accumulated real-space energy per atom.

    Notes
    -----
    Energy is accumulated in a local register then written once, reducing atomic
    contention. Internal computations use float64 for numerical stability.
    """
    atom_idx = wp.tid()

    # Load atom i data once
    qi = wp.float64(charges[atom_idx])
    pos_i = positions[atom_idx]
    alpha_ = wp.float64(alpha[0])
    cell_t = wp.transpose(cell[0])

    # Accumulate energy in local register
    energy_acc = wp.float64(0.0)
    max_neighbors = neighbor_matrix.shape[1]

    for neighbor_idx in range(max_neighbors):
        j = neighbor_matrix[atom_idx, neighbor_idx]
        if j == mask_value:
            continue

        qj = wp.float64(charges[j])
        pos_j = positions[j]

        # Compute periodic shift (in input precision, then cast)
        shift_vec = unit_shifts_matrix[atom_idx, neighbor_idx]
        periodic_shift = cell_t * type(pos_j)(
            type(pos_j[0])(shift_vec[0]),
            type(pos_j[0])(shift_vec[1]),
            type(pos_j[0])(shift_vec[2]),
        )

        separation_vector = pos_j - pos_i + periodic_shift
        distance = wp.float64(wp.length(separation_vector))

        if distance > wp.float64(1e-8):
            energy_acc += _ewald_real_space_energy_kernel_compute_energy(
                qi, qj, distance, alpha_
            )

    # Write accumulated energy once
    wp.atomic_add(pair_energies, atom_idx, energy_acc)


@wp.kernel
def _ewald_real_space_energy_kernel(
    positions: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    idx_j: wp.array(dtype=wp.int32),
    neighbor_ptr: wp.array(dtype=wp.int32),
    unit_shifts: wp.array(dtype=wp.vec3i),
    alpha: wp.array(dtype=Any),
    pair_energies: wp.array(dtype=wp.float64),
):
    """Compute real-space Ewald energies using neighbor list (CSR) format.

    Each thread processes one atom and loops over its neighbors using CSR
    pointers. This 1D launch pattern is more efficient than one-thread-per-pair
    as it reduces atomic contention and allows local accumulation.
    Pairs too close (less than 1e-8 distance) are skipped.

    Launch Grid
    -----------
    dim = [num_atoms]

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        Atomic coordinates.
    charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Atomic charges.
    cell : wp.array, shape (1, 3, 3), dtype=wp.mat33f or wp.mat33d
        Unit cell matrix.
    idx_j : wp.array, shape (M,), dtype=wp.int32
        Target atom indices for each pair (flattened CSR data).
    neighbor_ptr : wp.array, shape (N+1,), dtype=wp.int32
        CSR row pointers. neighbor_ptr[i] to neighbor_ptr[i+1] gives the range
        of neighbors for atom i in idx_j.
    unit_shifts : wp.array, shape (M,), dtype=wp.vec3i
        Periodic image shifts for each pair.
    alpha : wp.array, shape (1,), dtype=wp.float32 or wp.float64
        Ewald splitting parameter.
    pair_energies : wp.array, shape (N,), dtype=wp.float64
        OUTPUT: Accumulated real-space energy per atom.

    Notes
    -----
    Energy is accumulated in a local register then written once, reducing
    atomic contention. Internal computations use float64 for numerical stability.
    """
    atom_i = wp.tid()

    # Load atom i data once
    qi = wp.float64(charges[atom_i])
    pos_i = positions[atom_i]
    alpha_ = wp.float64(alpha[0])
    cell_t = wp.transpose(cell[0])

    # Accumulate energy in local register
    energy_acc = wp.float64(0.0)

    # Iterate over neighbors using CSR pointers
    j_range_start = neighbor_ptr[atom_i]
    j_range_end = neighbor_ptr[atom_i + 1]

    for edge_idx in range(j_range_start, j_range_end):
        j = idx_j[edge_idx]

        qj = wp.float64(charges[j])
        pos_j = positions[j]

        # Compute periodic shift
        shift_vec = unit_shifts[edge_idx]
        periodic_shift = cell_t * type(pos_i)(
            type(pos_i[0])(shift_vec[0]),
            type(pos_i[0])(shift_vec[1]),
            type(pos_i[0])(shift_vec[2]),
        )

        # Compute separation vector
        separation_vector = pos_j - pos_i + periodic_shift
        distance = wp.float64(wp.length(separation_vector))

        # Compute real-space energy with erfc damping
        if distance > wp.float64(1e-8):
            energy_acc += _ewald_real_space_energy_kernel_compute_energy(
                qi, qj, distance, alpha_
            )

    # Write accumulated energy once
    wp.atomic_add(pair_energies, atom_i, energy_acc)


@wp.kernel
def _ewald_real_space_energy_forces_kernel(
    positions: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    idx_j: wp.array(dtype=wp.int32),
    neighbor_ptr: wp.array(dtype=wp.int32),
    unit_shifts: wp.array(dtype=wp.vec3i),
    alpha: wp.array(dtype=Any),
    pair_energies: wp.array(dtype=wp.float64),
    atomic_forces: wp.array(dtype=Any),
):
    """Compute real-space Ewald energy and forces using neighbor list (CSR) format.

    Each thread processes one atom and loops over its neighbors using CSR
    pointers. Energy and force on atom i are accumulated locally. Force on
    atom j uses atomic_add. Pairs too close (less than 1e-8 distance) are skipped.

    Launch Grid
    -----------
    dim = [num_atoms]

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        Atomic coordinates.
    charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Atomic charges.
    cell : wp.array, shape (1, 3, 3), dtype=wp.mat33f or wp.mat33d
        Unit cell matrix.
    idx_j : wp.array, shape (M,), dtype=wp.int32
        Target atom indices for each pair (flattened CSR data).
    neighbor_ptr : wp.array, shape (N+1,), dtype=wp.int32
        CSR row pointers. neighbor_ptr[i] to neighbor_ptr[i+1] gives the range
        of neighbors for atom i in idx_j.
    unit_shifts : wp.array, shape (M,), dtype=wp.vec3i
        Periodic image shifts for each pair.
    alpha : wp.array, shape (1,), dtype=wp.float32 or wp.float64
        Ewald splitting parameter.
    pair_energies : wp.array, shape (N,), dtype=wp.float64
        OUTPUT: Accumulated real-space energy per atom.
    atomic_forces : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        OUTPUT: Accumulated forces per atom (matches positions dtype).

    Notes
    -----
    Energy accumulated locally then written once. Force on atom i accumulated
    locally; force on atom j uses atomic_add (Newton's 3rd law).
    """
    atom_i = wp.tid()

    # Load atom i data once
    qi = wp.float64(charges[atom_i])
    pos_i = positions[atom_i]
    alpha_ = wp.float64(alpha[0])
    cell_t = wp.transpose(cell[0])

    # Accumulators for energy and force on atom i
    energy_acc = wp.float64(0.0)
    force_i_acc = type(pos_i)(
        type(pos_i[0])(0.0), type(pos_i[0])(0.0), type(pos_i[0])(0.0)
    )

    # Iterate over neighbors using CSR pointers
    j_range_start = neighbor_ptr[atom_i]
    j_range_end = neighbor_ptr[atom_i + 1]
    for edge_idx in range(j_range_start, j_range_end):
        j = idx_j[edge_idx]
        qj = wp.float64(charges[j])
        pos_j = positions[j]

        # Apply periodic shift
        shift_vec = unit_shifts[edge_idx]
        periodic_shift = cell_t * type(pos_i)(
            type(pos_i[0])(shift_vec[0]),
            type(pos_i[0])(shift_vec[1]),
            type(pos_i[0])(shift_vec[2]),
        )
        separation_vector = pos_j - pos_i + periodic_shift
        distance = wp.float64(wp.length(separation_vector))

        if distance > wp.float64(1e-8):
            # Compute damped Coulomb energy
            energy_acc += _ewald_real_space_energy_kernel_compute_energy(
                qi, qj, distance, alpha_
            )

            # Compute force magnitude (in float64)
            force_mag = _ewald_real_space_force_magnitude(qi, qj, distance, alpha_)

            # Apply force in positions dtype
            force = type(pos_i)(
                type(pos_i[0])(force_mag) * separation_vector[0],
                type(pos_i[0])(force_mag) * separation_vector[1],
                type(pos_i[0])(force_mag) * separation_vector[2],
            )
            force_i_acc -= force
            wp.atomic_add(atomic_forces, j, force)

    # Write accumulated values once
    wp.atomic_add(pair_energies, atom_i, energy_acc)
    wp.atomic_add(atomic_forces, atom_i, force_i_acc)


@wp.kernel
def _ewald_real_space_energy_forces_neighbor_matrix_kernel(
    positions: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    neighbor_matrix: wp.array2d(dtype=wp.int32),
    unit_shifts_matrix: wp.array2d(dtype=wp.vec3i),
    mask_value: wp.int32,
    alpha: wp.array(dtype=Any),
    pair_energies: wp.array(dtype=wp.float64),
    atomic_forces: wp.array(dtype=Any),
):
    """Compute real-space Ewald energy and forces using neighbor matrix format.

    Each thread processes one atom and loops over all its neighbors. This 1D
    launch pattern is more efficient than 2D launch as it reduces thread
    divergence and improves memory access patterns. Energy is accumulated in
    a local register and written once. Force on atom j uses atomic_add.
    Pairs too close (less than 1e-8 distance) or invalid are skipped.

    Launch Grid
    -----------
    dim = [N_atoms]

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        Atomic coordinates.
    charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Atomic charges.
    cell : wp.array, shape (1, 3, 3), dtype=wp.mat33f or wp.mat33d
        Unit cell matrix.
    neighbor_matrix : wp.array2d, shape (N, max_neighbors), dtype=wp.int32
        Neighbor indices. Invalid entries contain mask_value.
    unit_shifts_matrix : wp.array2d, shape (N, max_neighbors), dtype=wp.vec3i
        Periodic image shifts for each neighbor pair.
    mask_value : wp.int32
        Value indicating invalid/padded neighbor entries.
    alpha : wp.array, shape (1,), dtype=wp.float32 or wp.float64
        Ewald splitting parameter.
    pair_energies : wp.array, shape (N,), dtype=wp.float64
        OUTPUT: Accumulated real-space energy per atom.
    atomic_forces : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        OUTPUT: Accumulated forces per atom (matches positions dtype).

    Notes
    -----
    Energy accumulated locally then written once. Forces on atom i accumulated
    locally; forces on atom j use atomic_add (Newton's 3rd law).
    """
    atom_idx = wp.tid()

    # Load atom i data once
    qi = wp.float64(charges[atom_idx])
    pos_i = positions[atom_idx]
    alpha_ = wp.float64(alpha[0])
    cell_t = wp.transpose(cell[0])

    # Accumulators for energy and force on atom i
    energy_acc = wp.float64(0.0)
    force_i_acc = type(pos_i)(
        type(pos_i[0])(0.0), type(pos_i[0])(0.0), type(pos_i[0])(0.0)
    )
    max_neighbors = neighbor_matrix.shape[1]

    for neighbor_idx in range(max_neighbors):
        j = neighbor_matrix[atom_idx, neighbor_idx]
        if j == mask_value:
            continue

        qj = wp.float64(charges[j])
        pos_j = positions[j]

        shift_vec = unit_shifts_matrix[atom_idx, neighbor_idx]
        periodic_shift = cell_t * type(pos_j)(
            type(pos_j[0])(shift_vec[0]),
            type(pos_j[0])(shift_vec[1]),
            type(pos_j[0])(shift_vec[2]),
        )
        separation_vector = pos_j - pos_i + periodic_shift
        distance = wp.float64(wp.length(separation_vector))

        if distance > wp.float64(1e-8):
            energy_acc += _ewald_real_space_energy_kernel_compute_energy(
                qi, qj, distance, alpha_
            )

            force_mag = _ewald_real_space_force_magnitude(qi, qj, distance, alpha_)
            force = type(pos_i)(
                type(pos_i[0])(force_mag) * separation_vector[0],
                type(pos_i[0])(force_mag) * separation_vector[1],
                type(pos_i[0])(force_mag) * separation_vector[2],
            )
            force_i_acc -= force
            wp.atomic_add(atomic_forces, j, force)

    # Write accumulated values once
    wp.atomic_add(pair_energies, atom_idx, energy_acc)
    wp.atomic_add(atomic_forces, atom_idx, force_i_acc)


###########################################################################################
#################### Real-Space Kernels with Charge Gradients #############################
###########################################################################################


@wp.kernel
def _ewald_real_space_energy_forces_charge_grad_kernel(
    positions: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    idx_j: wp.array(dtype=wp.int32),
    neighbor_ptr: wp.array(dtype=wp.int32),
    unit_shifts: wp.array(dtype=wp.vec3i),
    alpha: wp.array(dtype=Any),
    pair_energies: wp.array(dtype=wp.float64),
    atomic_forces: wp.array(dtype=Any),
    charge_gradients: wp.array(dtype=wp.float64),
):
    """Compute real-space Ewald energy, forces, AND charge gradients (neighbor list CSR).

    Each thread processes one atom and loops over its neighbors using CSR pointers.
    Energy, force, and charge gradient for atom i are accumulated locally. Forces
    and charge gradients on atom j use atomic_add. Pairs too close are skipped.

    Launch Grid
    -----------
    dim = [num_atoms]

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        Atomic coordinates.
    charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Atomic charges.
    cell : wp.array, shape (1, 3, 3), dtype=wp.mat33f or wp.mat33d
        Unit cell matrix.
    idx_j : wp.array, shape (M,), dtype=wp.int32
        Target atom indices for each pair (flattened CSR data).
    neighbor_ptr : wp.array, shape (N+1,), dtype=wp.int32
        CSR row pointers.
    unit_shifts : wp.array, shape (M,), dtype=wp.vec3i
        Periodic image shifts for each pair.
    alpha : wp.array, shape (1,), dtype=wp.float32 or wp.float64
        Ewald splitting parameter.
    pair_energies : wp.array, shape (N,), dtype=wp.float64
        OUTPUT: Accumulated real-space energy per atom.
    atomic_forces : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        OUTPUT: Accumulated forces per atom (matches positions dtype).
    charge_gradients : wp.array, shape (N,), dtype=wp.float64
        OUTPUT: Accumulated charge gradients ∂E/∂q per atom.

    Notes
    -----
    Energy, force, charge gradient on atom i accumulated locally then written once.
    Forces and charge gradients on atom j use atomic_add.
    """
    atom_i = wp.tid()

    # Load atom i data once
    qi = wp.float64(charges[atom_i])
    pos_i = positions[atom_i]
    alpha_ = wp.float64(alpha[0])
    cell_t = wp.transpose(cell[0])

    # Accumulators for energy, force, and charge gradient on atom i
    energy_acc = wp.float64(0.0)
    force_i_acc = type(pos_i)(
        type(pos_i[0])(0.0), type(pos_i[0])(0.0), type(pos_i[0])(0.0)
    )
    cg_i_acc = wp.float64(0.0)

    # Iterate over neighbors using CSR pointers
    j_range_start = neighbor_ptr[atom_i]
    j_range_end = neighbor_ptr[atom_i + 1]

    for edge_idx in range(j_range_start, j_range_end):
        j = idx_j[edge_idx]

        qj = wp.float64(charges[j])
        pos_j = positions[j]

        # Apply periodic shift
        shift_vec = unit_shifts[edge_idx]
        periodic_shift = cell_t * type(pos_i)(
            type(pos_i[0])(shift_vec[0]),
            type(pos_i[0])(shift_vec[1]),
            type(pos_i[0])(shift_vec[2]),
        )
        separation_vector = pos_j - pos_i + periodic_shift
        distance = wp.float64(wp.length(separation_vector))

        if distance > wp.float64(1e-8):
            # Compute damped Coulomb energy
            energy_acc += _ewald_real_space_energy_kernel_compute_energy(
                qi, qj, distance, alpha_
            )

            # Compute force magnitude (in float64)
            force_mag = _ewald_real_space_force_magnitude(qi, qj, distance, alpha_)

            # Apply force in positions dtype
            force = type(pos_i)(
                type(pos_i[0])(force_mag) * separation_vector[0],
                type(pos_i[0])(force_mag) * separation_vector[1],
                type(pos_i[0])(force_mag) * separation_vector[2],
            )
            force_i_acc -= force
            wp.atomic_add(atomic_forces, j, force)

            # Compute charge gradients
            potential = _ewald_real_space_charge_grad_potential(distance, alpha_)
            cg_i_acc += qj * potential
            cg_j = qi * potential
            wp.atomic_add(charge_gradients, j, cg_j)

    # Write accumulated values once

    wp.atomic_add(pair_energies, atom_i, energy_acc)
    wp.atomic_add(atomic_forces, atom_i, force_i_acc)
    wp.atomic_add(charge_gradients, atom_i, cg_i_acc)


@wp.kernel
def _ewald_real_space_energy_forces_charge_grad_neighbor_matrix_kernel(
    positions: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    neighbor_matrix: wp.array2d(dtype=wp.int32),
    unit_shifts_matrix: wp.array2d(dtype=wp.vec3i),
    mask_value: wp.int32,
    alpha: wp.array(dtype=Any),
    pair_energies: wp.array(dtype=wp.float64),
    atomic_forces: wp.array(dtype=Any),
    charge_gradients: wp.array(dtype=wp.float64),
):
    """Compute real-space Ewald energy, forces, AND charge gradients (neighbor matrix).

    Each thread processes one atom and loops over all its neighbors. This 1D
    launch pattern is more efficient than 2D launch. Energy and charge gradient
    for atom i are accumulated locally and written once. Forces and charge
    gradients on atom j use atomic_add.

    Launch Grid
    -----------
    dim = [N_atoms]

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        Atomic coordinates.
    charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Atomic charges.
    cell : wp.array, shape (1, 3, 3), dtype=wp.mat33f or wp.mat33d
        Unit cell matrix.
    neighbor_matrix : wp.array2d, shape (N, max_neighbors), dtype=wp.int32
        Neighbor indices. Invalid entries contain mask_value.
    unit_shifts_matrix : wp.array2d, shape (N, max_neighbors), dtype=wp.vec3i
        Periodic image shifts for each neighbor pair.
    mask_value : wp.int32
        Value indicating invalid/padded neighbor entries.
    alpha : wp.array, shape (1,), dtype=wp.float32 or wp.float64
        Ewald splitting parameter.
    pair_energies : wp.array, shape (N,), dtype=wp.float64
        OUTPUT: Accumulated real-space energy per atom.
    atomic_forces : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        OUTPUT: Accumulated forces per atom (matches positions dtype).
    charge_gradients : wp.array, shape (N,), dtype=wp.float64
        OUTPUT: Accumulated charge gradients ∂E/∂q per atom.
    """
    atom_idx = wp.tid()

    # Load atom i data once
    qi = wp.float64(charges[atom_idx])
    pos_i = positions[atom_idx]
    alpha_ = wp.float64(alpha[0])
    cell_t = wp.transpose(cell[0])

    # Accumulators for energy, force, and charge gradient on atom i
    energy_acc = wp.float64(0.0)
    force_i_acc = type(pos_i)(
        type(pos_i[0])(0.0), type(pos_i[0])(0.0), type(pos_i[0])(0.0)
    )
    cg_i_acc = wp.float64(0.0)
    max_neighbors = neighbor_matrix.shape[1]

    for neighbor_idx in range(max_neighbors):
        j = neighbor_matrix[atom_idx, neighbor_idx]
        if j == mask_value:
            continue

        qj = wp.float64(charges[j])
        pos_j = positions[j]

        shift_vec = unit_shifts_matrix[atom_idx, neighbor_idx]
        periodic_shift = cell_t * type(pos_j)(
            type(pos_j[0])(shift_vec[0]),
            type(pos_j[0])(shift_vec[1]),
            type(pos_j[0])(shift_vec[2]),
        )
        separation_vector = pos_j - pos_i + periodic_shift
        distance = wp.float64(wp.length(separation_vector))

        if distance > wp.float64(1e-8):
            energy_acc += _ewald_real_space_energy_kernel_compute_energy(
                qi, qj, distance, alpha_
            )

            force_mag = _ewald_real_space_force_magnitude(qi, qj, distance, alpha_)
            force = type(pos_i)(
                type(pos_i[0])(force_mag) * separation_vector[0],
                type(pos_i[0])(force_mag) * separation_vector[1],
                type(pos_i[0])(force_mag) * separation_vector[2],
            )
            force_i_acc -= force
            wp.atomic_add(atomic_forces, j, force)

            # Compute charge gradients
            potential = _ewald_real_space_charge_grad_potential(distance, alpha_)
            cg_i_acc += qj * potential
            cg_j = qi * potential
            wp.atomic_add(charge_gradients, j, cg_j)

    # Write accumulated values once
    wp.atomic_add(pair_energies, atom_idx, energy_acc)
    wp.atomic_add(atomic_forces, atom_idx, force_i_acc)
    wp.atomic_add(charge_gradients, atom_idx, cg_i_acc)


###########################################################################################
########################### Batch Real-Space Kernels ######################################
###########################################################################################


@wp.kernel
def _batch_ewald_real_space_energy_neighbor_matrix_kernel(
    positions: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    batch_id: wp.array(dtype=wp.int32),
    neighbor_matrix: wp.array2d(dtype=wp.int32),
    unit_shifts_matrix: wp.array2d(dtype=wp.vec3i),
    mask_value: wp.int32,
    alpha: wp.array(dtype=Any),
    pair_energies: wp.array(dtype=wp.float64),
):
    """Compute real-space Ewald energies for batched systems (neighbor matrix).

    Each thread processes one atom and loops over its neighbors. This 1D launch
    pattern is more efficient than 2D launch. Per-system cell and alpha are
    looked up using batch_id. Energy is accumulated locally and written once.

    Launch Grid
    -----------
    dim = [N_total]

    Parameters
    ----------
    positions : wp.array, shape (N_total,), dtype=wp.vec3f or wp.vec3d
        Atomic coordinates for all systems concatenated.
    charges : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        Atomic charges for all systems concatenated.
    cell : wp.array, shape (B, 3, 3), dtype=wp.mat33f or wp.mat33d
        Unit cell matrices for each system.
    batch_id : wp.array, shape (N_total,), dtype=wp.int32
        System index for each atom (0 to B-1).
    neighbor_matrix : wp.array2d, shape (N_total, max_neighbors), dtype=wp.int32
        Neighbor indices. Invalid entries contain mask_value.
    unit_shifts_matrix : wp.array2d, shape (N_total, max_neighbors), dtype=wp.vec3i
        Periodic image shifts for each neighbor pair.
    mask_value : wp.int32
        Value indicating invalid/padded neighbor entries.
    alpha : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        Per-system Ewald splitting parameter.
    pair_energies : wp.array, shape (N_total,), dtype=wp.float64
        OUTPUT: Accumulated real-space energy per atom.
    """
    atom_idx = wp.tid()

    # Load atom i data once
    qi = wp.float64(charges[atom_idx])
    pos_i = positions[atom_idx]
    system_id = batch_id[atom_idx]
    system_cell = cell[system_id]
    system_alpha = wp.float64(alpha[system_id])
    cell_t = wp.transpose(system_cell)

    # Accumulate energy in local register
    energy_acc = wp.float64(0.0)
    max_neighbors = neighbor_matrix.shape[1]

    for neighbor_idx in range(max_neighbors):
        j = neighbor_matrix[atom_idx, neighbor_idx]
        if j == mask_value:
            continue

        qj = wp.float64(charges[j])
        pos_j = positions[j]

        shift_vec = unit_shifts_matrix[atom_idx, neighbor_idx]
        periodic_shift = cell_t * type(pos_j)(
            type(pos_j[0])(shift_vec[0]),
            type(pos_j[0])(shift_vec[1]),
            type(pos_j[0])(shift_vec[2]),
        )
        separation_vector = pos_j - pos_i + periodic_shift
        distance = wp.float64(wp.length(separation_vector))

        if distance > wp.float64(1e-8):
            energy_acc += _ewald_real_space_energy_kernel_compute_energy(
                qi, qj, distance, system_alpha
            )

    # Write accumulated energy once
    wp.atomic_add(pair_energies, atom_idx, energy_acc)


@wp.kernel
def _batch_ewald_real_space_energy_kernel(
    positions: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    batch_id: wp.array(dtype=wp.int32),
    idx_j: wp.array(dtype=wp.int32),
    neighbor_ptr: wp.array(dtype=wp.int32),
    unit_shifts: wp.array(dtype=wp.vec3i),
    alpha: wp.array(dtype=Any),
    pair_energies: wp.array(dtype=wp.float64),
):
    """Compute real-space Ewald energies for batched systems (neighbor list CSR).

    Each thread processes one atom and loops over its neighbors using CSR
    pointers. Per-system cell and alpha are looked up using batch_id.
    Energy is accumulated locally and written once.

    Launch Grid
    -----------
    dim = [num_atoms]

    Parameters
    ----------
    positions : wp.array, shape (N_total,), dtype=wp.vec3f or wp.vec3d
        Atomic coordinates for all systems concatenated.
    charges : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        Atomic charges for all systems concatenated.
    cell : wp.array, shape (B, 3, 3), dtype=wp.mat33f or wp.mat33d
        Unit cell matrices for each system.
    batch_id : wp.array, shape (N_total,), dtype=wp.int32
        System index for each atom (0 to B-1).
    idx_j : wp.array, shape (M,), dtype=wp.int32
        Target atom indices for each pair (flattened CSR data).
    neighbor_ptr : wp.array, shape (N_total+1,), dtype=wp.int32
        CSR row pointers.
    unit_shifts : wp.array, shape (M,), dtype=wp.vec3i
        Periodic image shifts for each pair.
    alpha : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        Per-system Ewald splitting parameter.
    pair_energies : wp.array, shape (N_total,), dtype=wp.float64
        OUTPUT: Accumulated real-space energy per atom.
    """
    atom_i = wp.tid()

    # Load atom i data once
    qi = wp.float64(charges[atom_i])
    pos_i = positions[atom_i]
    system_id = batch_id[atom_i]
    system_cell = cell[system_id]
    system_alpha = wp.float64(alpha[system_id])
    cell_t = wp.transpose(system_cell)

    # Accumulate energy in local register
    energy_acc = wp.float64(0.0)

    # Iterate over neighbors using CSR pointers
    j_range_start = neighbor_ptr[atom_i]
    j_range_end = neighbor_ptr[atom_i + 1]

    for edge_idx in range(j_range_start, j_range_end):
        j = idx_j[edge_idx]

        qj = wp.float64(charges[j])
        pos_j = positions[j]

        # Convert unit shifts to Cartesian using system cell
        shift_vec = unit_shifts[edge_idx]
        periodic_shift = cell_t * type(pos_i)(
            type(pos_i[0])(shift_vec[0]),
            type(pos_i[0])(shift_vec[1]),
            type(pos_i[0])(shift_vec[2]),
        )

        # Compute separation vector
        separation_vector = pos_j - pos_i + periodic_shift
        distance = wp.float64(wp.length(separation_vector))

        # Compute real-space energy with erfc damping
        if distance > wp.float64(1e-8):
            energy_acc += _ewald_real_space_energy_kernel_compute_energy(
                qi, qj, distance, system_alpha
            )

    # Write accumulated energy once
    wp.atomic_add(pair_energies, atom_i, energy_acc)


@wp.kernel
def _batch_ewald_real_space_energy_forces_kernel(
    positions: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    batch_id: wp.array(dtype=wp.int32),
    idx_j: wp.array(dtype=wp.int32),
    neighbor_ptr: wp.array(dtype=wp.int32),
    unit_shifts: wp.array(dtype=wp.vec3i),
    alpha: wp.array(dtype=Any),
    pair_energies: wp.array(dtype=wp.float64),
    atomic_forces: wp.array(dtype=Any),
):
    """Compute real-space Ewald energy and forces for batched systems (neighbor list CSR).

    Each thread processes one atom and loops over its neighbors using CSR
    pointers. Energy and force on atom i are accumulated locally. Forces on
    atom j use atomic_add.

    Launch Grid
    -----------
    dim = [num_atoms]

    Parameters
    ----------
    positions : wp.array, shape (N_total,), dtype=wp.vec3f or wp.vec3d
        Atomic coordinates for all systems concatenated.
    charges : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        Atomic charges for all systems concatenated.
    cell : wp.array, shape (B, 3, 3), dtype=wp.mat33f or wp.mat33d
        Unit cell matrices for each system.
    batch_id : wp.array, shape (N_total,), dtype=wp.int32
        System index for each atom (0 to B-1).
    idx_j : wp.array, shape (M,), dtype=wp.int32
        Target atom indices for each pair (flattened CSR data).
    neighbor_ptr : wp.array, shape (N_total+1,), dtype=wp.int32
        CSR row pointers.
    unit_shifts : wp.array, shape (M,), dtype=wp.vec3i
        Periodic image shifts for each pair.
    alpha : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        Per-system Ewald splitting parameter.
    pair_energies : wp.array, shape (N_total,), dtype=wp.float64
        OUTPUT: Accumulated real-space energy per atom.
    atomic_forces : wp.array, shape (N_total,), dtype=wp.vec3f or wp.vec3d
        OUTPUT: Accumulated forces per atom.
    """
    atom_i = wp.tid()

    # Load atom i data once
    qi = wp.float64(charges[atom_i])
    pos_i = positions[atom_i]
    system_id = batch_id[atom_i]
    system_cell = cell[system_id]
    system_alpha = wp.float64(alpha[system_id])
    cell_t = wp.transpose(system_cell)

    # Accumulators for energy and force on atom i
    energy_acc = wp.float64(0.0)
    force_i_acc = type(pos_i)(
        type(pos_i[0])(0.0), type(pos_i[0])(0.0), type(pos_i[0])(0.0)
    )

    # Iterate over neighbors using CSR pointers
    j_range_start = neighbor_ptr[atom_i]
    j_range_end = neighbor_ptr[atom_i + 1]

    for edge_idx in range(j_range_start, j_range_end):
        j = idx_j[edge_idx]

        qj = wp.float64(charges[j])
        pos_j = positions[j]

        # Apply periodic shift using system cell
        shift_vec = unit_shifts[edge_idx]
        periodic_shift = cell_t * type(pos_i)(
            type(pos_i[0])(shift_vec[0]),
            type(pos_i[0])(shift_vec[1]),
            type(pos_i[0])(shift_vec[2]),
        )
        separation_vector = pos_j - pos_i + periodic_shift
        distance = wp.float64(wp.length(separation_vector))

        if distance > wp.float64(1e-8):
            # Compute damped Coulomb energy
            energy_acc += _ewald_real_space_energy_kernel_compute_energy(
                qi, qj, distance, system_alpha
            )

            force_mag = _ewald_real_space_force_magnitude(
                qi, qj, distance, system_alpha
            )
            force = type(pos_i)(
                type(pos_i[0])(force_mag) * separation_vector[0],
                type(pos_i[0])(force_mag) * separation_vector[1],
                type(pos_i[0])(force_mag) * separation_vector[2],
            )
            force_i_acc -= force
            wp.atomic_add(atomic_forces, j, force)

    # Write accumulated values once
    wp.atomic_add(pair_energies, atom_i, energy_acc)
    wp.atomic_add(atomic_forces, atom_i, force_i_acc)


@wp.kernel
def _batch_ewald_real_space_energy_forces_neighbor_matrix_kernel(
    positions: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    batch_id: wp.array(dtype=wp.int32),
    neighbor_matrix: wp.array2d(dtype=wp.int32),
    unit_shifts_matrix: wp.array2d(dtype=wp.vec3i),
    mask_value: wp.int32,
    alpha: wp.array(dtype=Any),
    pair_energies: wp.array(dtype=wp.float64),
    atomic_forces: wp.array(dtype=Any),
):
    """Compute real-space Ewald energy and forces for batched systems (neighbor matrix).

    Each thread processes one atom and loops over its neighbors. This 1D launch
    pattern is more efficient than 2D launch. Energy and force on atom i are
    accumulated locally. Forces on atom j use atomic_add.

    Launch Grid
    -----------
    dim = [N_total]

    Parameters
    ----------
    positions : wp.array, shape (N_total,), dtype=wp.vec3f or wp.vec3d
        Atomic coordinates for all systems concatenated.
    charges : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        Atomic charges for all systems concatenated.
    cell : wp.array, shape (B, 3, 3), dtype=wp.mat33f or wp.mat33d
        Unit cell matrices for each system.
    batch_id : wp.array, shape (N_total,), dtype=wp.int32
        System index for each atom (0 to B-1).
    neighbor_matrix : wp.array2d, shape (N_total, max_neighbors), dtype=wp.int32
        Neighbor indices. Invalid entries contain mask_value.
    unit_shifts_matrix : wp.array2d, shape (N_total, max_neighbors), dtype=wp.vec3i
        Periodic image shifts for each neighbor pair.
    mask_value : wp.int32
        Value indicating invalid/padded neighbor entries.
    alpha : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        Per-system Ewald splitting parameter.
    pair_energies : wp.array, shape (N_total,), dtype=wp.float64
        OUTPUT: Accumulated real-space energy per atom.
    atomic_forces : wp.array, shape (N_total,), dtype=wp.vec3f or wp.vec3d
        OUTPUT: Accumulated forces per atom.
    """
    atom_idx = wp.tid()

    # Load atom i data once
    qi = wp.float64(charges[atom_idx])
    pos_i = positions[atom_idx]
    system_id = batch_id[atom_idx]
    system_cell = cell[system_id]
    system_alpha = wp.float64(alpha[system_id])
    cell_t = wp.transpose(system_cell)

    # Accumulators for energy and force on atom i
    energy_acc = wp.float64(0.0)
    force_i_acc = type(pos_i)(
        type(pos_i[0])(0.0), type(pos_i[0])(0.0), type(pos_i[0])(0.0)
    )
    max_neighbors = neighbor_matrix.shape[1]

    for neighbor_idx in range(max_neighbors):
        j = neighbor_matrix[atom_idx, neighbor_idx]
        if j == mask_value:
            continue

        qj = wp.float64(charges[j])
        pos_j = positions[j]

        shift_vec = unit_shifts_matrix[atom_idx, neighbor_idx]
        periodic_shift = cell_t * type(pos_j)(
            type(pos_j[0])(shift_vec[0]),
            type(pos_j[0])(shift_vec[1]),
            type(pos_j[0])(shift_vec[2]),
        )
        separation_vector = pos_j - pos_i + periodic_shift
        distance = wp.float64(wp.length(separation_vector))

        if distance > wp.float64(1e-8):
            energy_acc += _ewald_real_space_energy_kernel_compute_energy(
                qi, qj, distance, system_alpha
            )

            force_mag = _ewald_real_space_force_magnitude(
                qi, qj, distance, system_alpha
            )
            force = type(pos_i)(
                type(pos_i[0])(force_mag) * separation_vector[0],
                type(pos_i[0])(force_mag) * separation_vector[1],
                type(pos_i[0])(force_mag) * separation_vector[2],
            )
            force_i_acc -= force
            wp.atomic_add(atomic_forces, j, force)

    # Write accumulated values once
    wp.atomic_add(pair_energies, atom_idx, energy_acc)
    wp.atomic_add(atomic_forces, atom_idx, force_i_acc)


###########################################################################################
#################### Batch Real-Space Kernels with Charge Gradients #######################
###########################################################################################


@wp.kernel
def _batch_ewald_real_space_energy_forces_charge_grad_kernel(
    positions: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    batch_id: wp.array(dtype=wp.int32),
    idx_j: wp.array(dtype=wp.int32),
    neighbor_ptr: wp.array(dtype=wp.int32),
    unit_shifts: wp.array(dtype=wp.vec3i),
    alpha: wp.array(dtype=Any),
    pair_energies: wp.array(dtype=wp.float64),
    atomic_forces: wp.array(dtype=Any),
    charge_gradients: wp.array(dtype=wp.float64),
):
    """Compute real-space Ewald energy, forces, AND charge gradients (batch, CSR).

    Each thread processes one atom and loops over its neighbors using CSR
    pointers. Energy, force, and charge gradient for atom i are accumulated
    locally. Forces and charge gradients on atom j use atomic_add.

    Launch Grid
    -----------
    dim = [num_atoms]

    Parameters
    ----------
    positions : wp.array, shape (N_total,), dtype=wp.vec3f or wp.vec3d
        Atomic coordinates for all systems concatenated.
    charges : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        Atomic charges for all systems concatenated.
    cell : wp.array, shape (B, 3, 3), dtype=wp.mat33f or wp.mat33d
        Unit cell matrices for each system.
    batch_id : wp.array, shape (N_total,), dtype=wp.int32
        System index for each atom (0 to B-1).
    idx_j : wp.array, shape (M,), dtype=wp.int32
        Target atom indices for each pair (flattened CSR data).
    neighbor_ptr : wp.array, shape (N_total+1,), dtype=wp.int32
        CSR row pointers.
    unit_shifts : wp.array, shape (M,), dtype=wp.vec3i
        Periodic image shifts for each pair.
    alpha : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        Per-system Ewald splitting parameter.
    pair_energies : wp.array, shape (N_total,), dtype=wp.float64
        OUTPUT: Accumulated real-space energy per atom.
    atomic_forces : wp.array, shape (N_total,), dtype=wp.vec3f or wp.vec3d
        OUTPUT: Accumulated forces per atom.
    charge_gradients : wp.array, shape (N_total,), dtype=wp.float64
        OUTPUT: Accumulated charge gradients ∂E/∂q per atom.
    """
    atom_i = wp.tid()

    # Load atom i data once
    qi = wp.float64(charges[atom_i])
    pos_i = positions[atom_i]
    system_id = batch_id[atom_i]
    system_cell = cell[system_id]
    system_alpha = wp.float64(alpha[system_id])
    cell_t = wp.transpose(system_cell)

    # Accumulators for energy, force, and charge gradient on atom i
    energy_acc = wp.float64(0.0)
    force_i_acc = type(pos_i)(
        type(pos_i[0])(0.0), type(pos_i[0])(0.0), type(pos_i[0])(0.0)
    )
    cg_i_acc = wp.float64(0.0)

    # Iterate over neighbors using CSR pointers
    j_range_start = neighbor_ptr[atom_i]
    j_range_end = neighbor_ptr[atom_i + 1]

    for edge_idx in range(j_range_start, j_range_end):
        j = idx_j[edge_idx]

        qj = wp.float64(charges[j])
        pos_j = positions[j]

        # Apply periodic shift using system cell
        shift_vec = unit_shifts[edge_idx]
        periodic_shift = cell_t * type(pos_i)(
            type(pos_i[0])(shift_vec[0]),
            type(pos_i[0])(shift_vec[1]),
            type(pos_i[0])(shift_vec[2]),
        )
        separation_vector = pos_j - pos_i + periodic_shift
        distance = wp.float64(wp.length(separation_vector))

        if distance > wp.float64(1e-8):
            # Compute damped Coulomb energy
            energy_acc += _ewald_real_space_energy_kernel_compute_energy(
                qi, qj, distance, system_alpha
            )

            force_mag = _ewald_real_space_force_magnitude(
                qi, qj, distance, system_alpha
            )
            force = type(pos_i)(
                type(pos_i[0])(force_mag) * separation_vector[0],
                type(pos_i[0])(force_mag) * separation_vector[1],
                type(pos_i[0])(force_mag) * separation_vector[2],
            )
            force_i_acc -= force
            wp.atomic_add(atomic_forces, j, force)

            # Compute charge gradients
            potential = _ewald_real_space_charge_grad_potential(distance, system_alpha)
            cg_i_acc += qj * potential
            cg_j = qi * potential
            wp.atomic_add(charge_gradients, j, cg_j)

    # Write accumulated values once
    wp.atomic_add(pair_energies, atom_i, energy_acc)
    wp.atomic_add(atomic_forces, atom_i, force_i_acc)
    wp.atomic_add(charge_gradients, atom_i, cg_i_acc)


@wp.kernel
def _batch_ewald_real_space_energy_forces_charge_grad_neighbor_matrix_kernel(
    positions: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    batch_id: wp.array(dtype=wp.int32),
    neighbor_matrix: wp.array2d(dtype=wp.int32),
    unit_shifts_matrix: wp.array2d(dtype=wp.vec3i),
    mask_value: wp.int32,
    alpha: wp.array(dtype=Any),
    pair_energies: wp.array(dtype=wp.float64),
    atomic_forces: wp.array(dtype=Any),
    charge_gradients: wp.array(dtype=wp.float64),
):
    """Compute real-space Ewald energy, forces, AND charge gradients for batched systems.

    Each thread processes one atom and loops over its neighbors. This 1D launch
    pattern is more efficient than 2D launch. Energy, force, and charge gradient
    for atom i are accumulated locally. Forces and charge gradients on atom j
    use atomic_add.

    Launch Grid
    -----------
    dim = [N_total]

    Parameters
    ----------
    positions : wp.array, shape (N_total,), dtype=wp.vec3f or wp.vec3d
        Atomic coordinates for all systems concatenated.
    charges : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        Atomic charges for all systems concatenated.
    cell : wp.array, shape (B, 3, 3), dtype=wp.mat33f or wp.mat33d
        Unit cell matrices for each system.
    batch_id : wp.array, shape (N_total,), dtype=wp.int32
        System index for each atom (0 to B-1).
    neighbor_matrix : wp.array2d, shape (N_total, max_neighbors), dtype=wp.int32
        Neighbor indices. Invalid entries contain mask_value.
    unit_shifts_matrix : wp.array2d, shape (N_total, max_neighbors), dtype=wp.vec3i
        Periodic image shifts for each neighbor pair.
    mask_value : wp.int32
        Value indicating invalid/padded neighbor entries.
    alpha : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        Per-system Ewald splitting parameter.
    pair_energies : wp.array, shape (N_total,), dtype=wp.float64
        OUTPUT: Accumulated real-space energy per atom.
    atomic_forces : wp.array, shape (N_total,), dtype=wp.vec3f or wp.vec3d
        OUTPUT: Accumulated forces per atom.
    charge_gradients : wp.array, shape (N_total,), dtype=wp.float64
        OUTPUT: Accumulated charge gradients ∂E/∂q per atom.
    """
    atom_idx = wp.tid()

    # Load atom i data once
    qi = wp.float64(charges[atom_idx])
    pos_i = positions[atom_idx]
    system_id = batch_id[atom_idx]
    system_cell = cell[system_id]
    system_alpha = wp.float64(alpha[system_id])
    cell_t = wp.transpose(system_cell)

    # Accumulators for energy, force, and charge gradient on atom i
    energy_acc = wp.float64(0.0)
    force_i_acc = type(pos_i)(
        type(pos_i[0])(0.0), type(pos_i[0])(0.0), type(pos_i[0])(0.0)
    )
    cg_i_acc = wp.float64(0.0)
    max_neighbors = neighbor_matrix.shape[1]

    for neighbor_idx in range(max_neighbors):
        j = neighbor_matrix[atom_idx, neighbor_idx]
        if j == mask_value:
            continue

        qj = wp.float64(charges[j])
        pos_j = positions[j]

        shift_vec = unit_shifts_matrix[atom_idx, neighbor_idx]
        periodic_shift = cell_t * type(pos_j)(
            type(pos_j[0])(shift_vec[0]),
            type(pos_j[0])(shift_vec[1]),
            type(pos_j[0])(shift_vec[2]),
        )
        separation_vector = pos_j - pos_i + periodic_shift
        distance = wp.float64(wp.length(separation_vector))

        if distance > wp.float64(1e-8):
            energy_acc += _ewald_real_space_energy_kernel_compute_energy(
                qi, qj, distance, system_alpha
            )

            force_mag = _ewald_real_space_force_magnitude(
                qi, qj, distance, system_alpha
            )
            force = type(pos_i)(
                type(pos_i[0])(force_mag) * separation_vector[0],
                type(pos_i[0])(force_mag) * separation_vector[1],
                type(pos_i[0])(force_mag) * separation_vector[2],
            )
            force_i_acc -= force
            wp.atomic_add(atomic_forces, j, force)

            # Compute charge gradients
            potential = _ewald_real_space_charge_grad_potential(distance, system_alpha)
            cg_i_acc += qj * potential
            cg_j = qi * potential
            wp.atomic_add(charge_gradients, j, cg_j)

    # Write accumulated values once
    wp.atomic_add(pair_energies, atom_idx, energy_acc)
    wp.atomic_add(atomic_forces, atom_idx, force_i_acc)
    wp.atomic_add(charge_gradients, atom_idx, cg_i_acc)


###########################################################################################
########################### Reciprocal-Space Kernels ######################################
###########################################################################################


@wp.kernel
def _ewald_reciprocal_space_energy_kernel_fill_structure_factors(
    positions: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    k_vectors: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    alpha: wp.array(dtype=Any),
    total_charge: wp.array(dtype=wp.float64),
    cos_k_dot_r: wp.array2d(dtype=wp.float64),
    sin_k_dot_r: wp.array2d(dtype=wp.float64),
    real_structure_factors: wp.array(dtype=wp.float64),
    imag_structure_factors: wp.array(dtype=wp.float64),
):
    """Compute structure factors for reciprocal-space Ewald summation.

    This kernel uses K-major iteration: each thread processes one k-vector
    over all atoms. This avoids atomics entirely since each thread fully
    owns its k-vector's output.

    The weighted structure factors are:

    .. math::

        \\begin{aligned}
        S_{\\text{real}}(k) &= \\frac{G(k)}{V} \\sum_i q_i \\cos(k \\cdot r_i) \\\\
        S_{\\text{imag}}(k) &= \\frac{G(k)}{V} \\sum_i q_i \\sin(k \\cdot r_i)
        \\end{aligned}

    where :math:`G(k) = \\frac{4\\pi}{k^2} \\exp(-k^2/(4\\alpha^2))` is the Green's function.

    Launch Grid
    -----------
    dim = [K]

    Each thread processes one k-vector over all N atoms.

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        Atomic coordinates.
    charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Atomic charges.
    k_vectors : wp.array, shape (K,), dtype=wp.vec3f or wp.vec3d
        Half-space reciprocal lattice vectors (excludes -k for each k).
    cell : wp.array, shape (1, 3, 3), dtype=wp.mat33f or wp.mat33d
        Unit cell matrix (for computing volume).
    alpha : wp.array, shape (1,), dtype=wp.float32 or wp.float64
        Ewald splitting parameter.
    total_charge : wp.array, shape (1,), dtype=wp.float64
        OUTPUT: Accumulated total charge divided by volume (Q/V) for
        background correction. Only thread 1 accumulates this.
    cos_k_dot_r : wp.array2d, shape (K, N), dtype=wp.float64
        OUTPUT: :math:`\\cos(k \\cdot r_i)` for each (k, atom) pair.
    sin_k_dot_r : wp.array2d, shape (K, N), dtype=wp.float64
        OUTPUT: :math:`\\sin(k \\cdot r_i)` for each (k, atom) pair.
    real_structure_factors : wp.array, shape (K,), dtype=wp.float64
        OUTPUT: :math:`(G(k)/V) \\sum_i q_i \\cos(k \\cdot r_i)`.
    imag_structure_factors : wp.array, shape (K,), dtype=wp.float64
        OUTPUT: :math:`(G(k)/V) \\sum_i q_i \\sin(k \\cdot r_i)`.

    Notes
    -----
    - K-major iteration avoids atomics (each thread owns its k output).
    - k=0 is skipped (early return) to avoid division by zero in G(k).
    - Thread 1 accumulates total_charge as Q/V for background correction.
    - All internal computations use float64 for numerical stability.
    - cos_k_dot_r and sin_k_dot_r store unweighted phases for charge gradient computation.
    - Half-space k-vectors with 8π Green's function give ~2x speedup.
    """
    k_idx = wp.tid()
    num_atoms = positions.shape[0]

    alpha_ = wp.float64(alpha[0])
    exp_factor = wp.float64(0.25) / (alpha_ * alpha_)
    volume = wp.float64(wp.abs(wp.determinant(cell[0])))

    k_vector = k_vectors[k_idx]
    # Cast k_vector components to float64 for precision
    kx = wp.float64(k_vector[0])
    ky = wp.float64(k_vector[1])
    kz = wp.float64(k_vector[2])
    k_squared = kx * kx + ky * ky + kz * kz

    # Skip k=0 (would cause division by zero)
    if k_squared < wp.float64(1e-10):
        return

    # Compute Green's function: (8*pi/V) * exp(-k^2/(4*alpha^2)) / k^2
    green_function = wp_exp_kernel(k_squared, exp_factor) * wp.float64(EIGHTPI) / volume

    # Accumulate structure factors in registers (no atomics!)
    real_sum = wp.float64(0.0)
    imag_sum = wp.float64(0.0)

    for atom_idx in range(num_atoms):
        position = positions[atom_idx]
        charge = wp.float64(charges[atom_idx])

        # Thread 1 accumulates total charge for background correction
        if k_idx == 1:
            tc = charge / volume
            wp.atomic_add(total_charge, 0, tc)

        # Compute k*r in float64
        k_dot_r = (
            kx * wp.float64(position[0])
            + ky * wp.float64(position[1])
            + kz * wp.float64(position[2])
        )
        cos_kr = wp.cos(k_dot_r)
        sin_kr = wp.sin(k_dot_r)

        # Store per-(k, atom) UNWEIGHTED phase factors (for charge gradients)
        cos_k_dot_r[k_idx, atom_idx] = cos_kr
        sin_k_dot_r[k_idx, atom_idx] = sin_kr

        # Accumulate structure factors (charge-weighted) in registers
        real_sum += charge * cos_kr * green_function
        imag_sum += charge * sin_kr * green_function

    # Write final structure factors (no atomics needed)
    real_structure_factors[k_idx] = real_sum
    imag_structure_factors[k_idx] = imag_sum


@wp.kernel
def _ewald_reciprocal_space_energy_kernel_compute_energy(
    charges: wp.array(dtype=Any),
    cos_k_dot_r: wp.array2d(dtype=wp.float64),
    sin_k_dot_r: wp.array2d(dtype=wp.float64),
    real_structure_factors: wp.array(dtype=wp.float64),
    imag_structure_factors: wp.array(dtype=wp.float64),
    reciprocal_energies: wp.array(dtype=wp.float64),
):
    """Compute per-atom reciprocal-space energies from structure factors.

    This kernel uses atom-major iteration: each thread processes one atom
    over all k-vectors. This avoids atomics since each thread fully owns
    its atom's output.

    For each atom i:

    .. math::

        E_i = \\frac{1}{2} \\sum_k [S_{\\text{real}}(k) \\cos(k \\cdot r_i) + S_{\\text{imag}}(k) \\sin(k \\cdot r_i)] q_i

    The 0.5 factor accounts for the pair energy sum: :math:`E = \\frac{1}{2} \\sum_i q_i \\phi_i`

    Launch Grid
    -----------
    dim = [N]

    Each thread processes one atom over all K k-vectors.

    Parameters
    ----------
    charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Atomic charges.
    cos_k_dot_r : wp.array2d, shape (K, N), dtype=wp.float64
        :math:`\\cos(k \\cdot r_i)` from structure factor computation.
    sin_k_dot_r : wp.array2d, shape (K, N), dtype=wp.float64
        :math:`\\sin(k \\cdot r_i)` from structure factor computation.
    real_structure_factors : wp.array, shape (K,), dtype=wp.float64
        Precomputed :math:`S_{\\text{real}}(k) = (G(k)/V) \\sum_j q_j \\cos(k \\cdot r_j)`.
    imag_structure_factors : wp.array, shape (K,), dtype=wp.float64
        Precomputed :math:`S_{\\text{imag}}(k) = (G(k)/V) \\sum_j q_j \\sin(k \\cdot r_j)`.
    reciprocal_energies : wp.array, shape (N,), dtype=wp.float64
        OUTPUT: Reciprocal-space energy per atom.

    Notes
    -----
    - Atom-major iteration avoids atomics (each thread owns its atom output)
    - The 0.5 factor is applied here (not in structure factor computation)
    - cos_k_dot_r and sin_k_dot_r are unweighted; charge is multiplied here
    - All computations in float64
    """
    atom_idx = wp.tid()
    num_k = real_structure_factors.shape[0]
    charge = wp.float64(charges[atom_idx])

    # Accumulate potential in register (no atomics!)
    local_potential = wp.float64(0.0)

    for k_idx in range(num_k):
        cos_kr = cos_k_dot_r[k_idx, atom_idx]
        sin_kr = sin_k_dot_r[k_idx, atom_idx]
        s_real = real_structure_factors[k_idx]
        s_imag = imag_structure_factors[k_idx]

        phase_sum = s_real * cos_kr + s_imag * sin_kr
        local_potential += charge * phase_sum

    # Write final energy: E_i = (1/2) * q_i * phi_i (no atomics needed)
    reciprocal_energies[atom_idx] = wp.float64(0.5) * local_potential


@wp.kernel
def _ewald_subtract_self_energy_kernel(
    charges: wp.array(dtype=Any),
    alpha: wp.array(dtype=Any),
    total_charge: wp.array(dtype=wp.float64),
    energy_in: wp.array(dtype=wp.float64),
    energy_out: wp.array(dtype=wp.float64),
):
    """Apply self-energy and background corrections to reciprocal-space energies.

    For each atom i:

    .. math::

        E_{\\text{out},i} = E_{\\text{in},i} - E_{\\text{self},i} - E_{\\text{background},i}

    where:

    .. math::

        \\begin{aligned}
        E_{\\text{self},i} &= \\frac{\\alpha}{\\sqrt{\\pi}} q_i^2 \\\\
        E_{\\text{background},i} &= \\frac{\\pi}{2\\alpha^2} q_i \\frac{Q_{\\text{total}}}{V}
        \\end{aligned}

    The self-energy removes the spurious interaction of each Gaussian charge
    distribution with itself. The background correction accounts for the
    uniform neutralizing background charge for non-neutral systems.

    Launch Grid
    -----------
    dim = [N]

    Parameters
    ----------
    charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Atomic charges.
    alpha : wp.array, shape (1,), dtype=wp.float32 or wp.float64
        Ewald splitting parameter.
    total_charge : wp.array, shape (1,), dtype=wp.float64
        Total charge divided by volume (Q_total/V), precomputed in
        _ewald_reciprocal_space_energy_kernel_fill_structure_factors.
    energy_in : wp.array, shape (N,), dtype=wp.float64
        Raw reciprocal-space energy per atom (from potential interpolation).
    energy_out : wp.array, shape (N,), dtype=wp.float64
        OUTPUT: Corrected reciprocal-space energy per atom.

    Notes
    -----
    - Uses separate input/output arrays to avoid in-place modification,
      which would cause incorrect gradient accumulation in Warp's autodiff
    - For neutral systems, the background correction is zero
    - All computations in float64
    """
    atom_index = wp.tid()
    charge = wp.float64(charges[atom_index])
    alpha_ = wp.float64(alpha[0])
    # Compute self-energy: alpha * q^2 / sqrt(pi)
    self_energy = alpha_ * charge * charge / wp.sqrt(wp.float64(PI))

    # Background correction: pi / (2*alpha^2) * q * (Q_total/V)
    neutralization_energy = (
        wp.float64(PI) * charge * total_charge[0] / (wp.float64(2.0) * alpha_ * alpha_)
    )

    # Subtract self-energy (separate input/output to avoid autodiff issues)
    energy_out[atom_index] = energy_in[atom_index] - self_energy - neutralization_energy


@wp.kernel
def _ewald_reciprocal_space_energy_forces_kernel(
    charges: wp.array(dtype=Any),
    k_vectors: wp.array(dtype=Any),
    cos_k_dot_r: wp.array2d(dtype=wp.float64),
    sin_k_dot_r: wp.array2d(dtype=wp.float64),
    real_structure_factors: wp.array(dtype=wp.float64),
    imag_structure_factors: wp.array(dtype=wp.float64),
    reciprocal_energies: wp.array(dtype=wp.float64),
    atomic_forces: wp.array(dtype=Any),
):
    """Compute reciprocal-space Ewald energies and forces simultaneously.

    This kernel uses atom-major iteration: each thread processes one atom
    over all k-vectors. This avoids atomics since each thread fully owns
    its atom's output.

    For each atom i:

    .. math::

        \\begin{aligned}
        E_i &= \\frac{1}{2} \\sum_k [S_{\\text{real}}(k) \\cos(k \\cdot r_i) + S_{\\text{imag}}(k) \\sin(k \\cdot r_i)] q_i \\\\
        F_i &= \\sum_k k [S_{\\text{real}}(k) \\sin(k \\cdot r_i) - S_{\\text{imag}}(k) \\cos(k \\cdot r_i)] q_i
        \\end{aligned}

    The force formula comes from :math:`-\\nabla_i E`, where the gradient acts on the
    :math:`\\cos(k \\cdot r_i)` and :math:`\\sin(k \\cdot r_i)` terms:

    .. math::

        \\begin{aligned}
        \\frac{\\partial}{\\partial r_i} \\cos(k \\cdot r_i) &= -k \\sin(k \\cdot r_i) \\\\
        \\frac{\\partial}{\\partial r_i} \\sin(k \\cdot r_i) &= k \\cos(k \\cdot r_i)
        \\end{aligned}

    Launch Grid
    -----------
    dim = [N]

    Each thread processes one atom over all K k-vectors.

    Parameters
    ----------
    charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Atomic charges.
    k_vectors : wp.array, shape (K,), dtype=wp.vec3f or wp.vec3d
        Reciprocal lattice vectors.
    cos_k_dot_r : wp.array2d, shape (K, N), dtype=wp.float64
        :math:`\\cos(k \\cdot r_i)` from structure factor computation.
    sin_k_dot_r : wp.array2d, shape (K, N), dtype=wp.float64
        :math:`\\sin(k \\cdot r_i)` from structure factor computation.
    real_structure_factors : wp.array, shape (K,), dtype=wp.float64
        Precomputed S_real(k) including Green's function.
    imag_structure_factors : wp.array, shape (K,), dtype=wp.float64
        Precomputed S_imag(k) including Green's function.
    reciprocal_energies : wp.array, shape (N,), dtype=wp.float64
        OUTPUT: Reciprocal-space energy per atom.
    atomic_forces : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        OUTPUT: Reciprocal-space forces per atom (matches k_vectors dtype).

    Notes
    -----
    - Atom-major iteration avoids atomics (each thread owns its atom output)
    - The 0.5 factor is applied to energy but not to forces
    - cos_k_dot_r and sin_k_dot_r are unweighted; charge is multiplied here
    - Energy computed in float64, forces in k_vectors dtype
    """
    atom_idx = wp.tid()
    num_k = real_structure_factors.shape[0]
    charge = wp.float64(charges[atom_idx])

    # Get the zero vector in the correct type
    k0 = k_vectors[0]

    # Accumulate in registers (no atomics!)
    local_potential = wp.float64(0.0)
    local_force_x = wp.float64(0.0)
    local_force_y = wp.float64(0.0)
    local_force_z = wp.float64(0.0)

    for k_idx in range(num_k):
        cos_kr = charge * cos_k_dot_r[k_idx, atom_idx]
        sin_kr = charge * sin_k_dot_r[k_idx, atom_idx]

        # Load precomputed structure factors (already include green function)
        s_real = real_structure_factors[k_idx]
        s_imag = imag_structure_factors[k_idx]

        # Potential contribution
        phase_sum = s_real * cos_kr + s_imag * sin_kr
        local_potential += phase_sum

        # Force contribution
        force_scalar = s_real * sin_kr - s_imag * cos_kr
        k_vec = k_vectors[k_idx]
        local_force_x += force_scalar * wp.float64(k_vec[0])
        local_force_y += force_scalar * wp.float64(k_vec[1])
        local_force_z += force_scalar * wp.float64(k_vec[2])

    # Write final results with charge multiplication (no atomics needed)
    reciprocal_energies[atom_idx] = wp.float64(0.5) * local_potential
    atomic_forces[atom_idx] = type(k0)(
        type(k0[0])(local_force_x),
        type(k0[0])(local_force_y),
        type(k0[0])(local_force_z),
    )


@wp.kernel
def _ewald_reciprocal_space_energy_forces_charge_grad_kernel(
    charges: wp.array(dtype=Any),
    k_vectors: wp.array(dtype=Any),
    cos_k_dot_r: wp.array2d(dtype=wp.float64),
    sin_k_dot_r: wp.array2d(dtype=wp.float64),
    real_structure_factors: wp.array(dtype=wp.float64),
    imag_structure_factors: wp.array(dtype=wp.float64),
    reciprocal_energies: wp.array(dtype=wp.float64),
    atomic_forces: wp.array(dtype=Any),
    charge_gradients: wp.array(dtype=wp.float64),
):
    """Compute reciprocal-space energies, forces, AND charge gradients.

    This kernel computes all three quantities in a single pass:

    .. math::

        \

        \\begin{aligned}
        E_i &= \\frac{1}{2} \\sum_k [S_{\\text{real}}(k) \\cos(k \\cdot r_i) + S_{\\text{imag}}(k) \\sin(k \\cdot r_i)] q_i \\\\
        F_i &= \\sum_k k [S_{\\text{real}}(k) \\sin(k \\cdot r_i) - S_{\\text{imag}}(k) \\cos(k \\cdot r_i)] q_i \\\\
        dE_i/dq_i &= \\sum_k [S_{\\text{real}}(k) \\cos(k \\cdot r_i) + S_{\\text{imag}}(k) \\sin(k \\cdot r_i)]
        \\end{aligned}

    where :math:`\\phi_i = \\sum_k [S_{\\text{real}}(k) \\cos(k \\cdot r_i) + S_{\\text{imag}}(k) \\sin(k \\cdot r_i)]` is the
    electrostatic potential at atom i.

    Launch Grid
    -----------
    dim = [N]

    Each thread processes one atom over all K k-vectors.

    Parameters
    ----------
    charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Atomic charges.
    k_vectors : wp.array, shape (K,), dtype=wp.vec3f or wp.vec3d
        Reciprocal lattice vectors.
    cos_k_dot_r : wp.array2d, shape (K, N), dtype=wp.float64
        :math:`\\cos(k \\cdot r_i)` from structure factor computation (unweighted).
    sin_k_dot_r : wp.array2d, shape (K, N), dtype=wp.float64
        :math:`\\sin(k \\cdot r_i)` from structure factor computation (unweighted).
    real_structure_factors : wp.array, shape (K,), dtype=wp.float64
        Precomputed :math:`S_{\\text{real}}(k)` including Green's function.
    imag_structure_factors : wp.array, shape (K,), dtype=wp.float64
        Precomputed :math:`S_{\\text{imag}}(k)` including Green's function.
    reciprocal_energies : wp.array, shape (N,), dtype=wp.float64
        OUTPUT: Reciprocal-space energy per atom.
    atomic_forces : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        OUTPUT: Reciprocal-space forces per atom.
    charge_gradients : wp.array, shape (N,), dtype=wp.float64
        OUTPUT: Electrostatic potential :math:`\\phi_i` per atom (reciprocal part of charge gradient).
    """
    atom_idx = wp.tid()
    num_k = real_structure_factors.shape[0]
    charge = wp.float64(charges[atom_idx])

    # Get the zero vector in the correct type
    k0 = k_vectors[0]

    # Accumulate in registers (no atomics!)
    local_potential = wp.float64(0.0)
    local_potential_uncharged = wp.float64(0.0)
    local_force_x = wp.float64(0.0)
    local_force_y = wp.float64(0.0)
    local_force_z = wp.float64(0.0)

    for k_idx in range(num_k):
        cos_kr = cos_k_dot_r[k_idx, atom_idx]
        sin_kr = sin_k_dot_r[k_idx, atom_idx]

        # Load precomputed structure factors (already include green function)
        s_real = real_structure_factors[k_idx]
        s_imag = imag_structure_factors[k_idx]

        # Potential contribution
        phase_sum = s_real * cos_kr + s_imag * sin_kr
        local_potential += charge * phase_sum
        local_potential_uncharged += phase_sum

        # Force contribution
        force_scalar = charge * (s_real * sin_kr - s_imag * cos_kr)
        k_vec = k_vectors[k_idx]
        local_force_x += force_scalar * wp.float64(k_vec[0])
        local_force_y += force_scalar * wp.float64(k_vec[1])
        local_force_z += force_scalar * wp.float64(k_vec[2])

    # Write final results (no atomics needed)
    # Energy: E_i = (1/2) * q_i * φ_i
    reciprocal_energies[atom_idx] = wp.float64(0.5) * local_potential

    # Forces
    atomic_forces[atom_idx] = type(k0)(
        type(k0[0])(local_force_x),
        type(k0[0])(local_force_y),
        type(k0[0])(local_force_z),
    )

    # Charge gradient
    # Self-energy and background corrections applied in higher-level code
    charge_gradients[atom_idx] = local_potential_uncharged


###########################################################################################
########################### Batch Reciprocal-Space Kernels ################################
###########################################################################################


@wp.kernel
def _batch_ewald_reciprocal_space_energy_kernel_fill_structure_factors(
    positions: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    k_vectors: wp.array2d(dtype=Any),
    cell: wp.array(dtype=Any),
    alpha: wp.array(dtype=Any),
    atom_start: wp.array(dtype=wp.int32),
    atom_end: wp.array(dtype=wp.int32),
    total_charges: wp.array(dtype=wp.float64),
    cos_k_dot_r: wp.array2d(dtype=wp.float64),
    sin_k_dot_r: wp.array2d(dtype=wp.float64),
    real_structure_factors: wp.array2d(dtype=wp.float64),
    imag_structure_factors: wp.array2d(dtype=wp.float64),
):
    """Compute structure factors for batched reciprocal-space Ewald summation.

    This kernel uses a blocked strategy: each thread handles one (k-vector, system,
    atom_block) triplet. This significantly reduces atomic contention compared to
    atom-major iteration while maintaining parallelism.

    The block size is controlled by ALCH_EWALD_BATCH_BLOCK_SIZE environment variable
    (default: 16, which benchmarks show is optimal for most scenarios).

    For each system s and atom i in that system:

    .. math::

        \\begin{aligned}
        S_{\\text{real}}(s, k) &+= \\frac{G_s(k)}{V_s} q_i \\cos(k \\cdot r_i) \\\\
        S_{\\text{imag}}(s, k) &+= \\frac{G_s(k)}{V_s} q_i \\sin(k \\cdot r_i)
        \\end{aligned}

    where :math:`G_s(k) = 8\\pi * \\exp(-k^2/(4\\alpha_s^2)) / k^2` uses half-space k-vectors.

    Launch Grid
    -----------
    dim = [K, B, max_blocks_per_system]

    where max_blocks_per_system = ceil(max_atoms_per_system / BATCH_BLOCK_SIZE)

    Parameters
    ----------
    positions : wp.array, shape (N_total,), dtype=wp.vec3f or wp.vec3d
        Atomic coordinates for all systems concatenated.
    charges : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        Atomic charges for all systems concatenated.
    k_vectors : wp.array2d, shape (B, K), dtype=wp.vec3f or wp.vec3d
        Per-system half-space reciprocal lattice vectors.
    cell : wp.array, shape (B, 3, 3), dtype=wp.mat33f or wp.mat33d
        Per-system unit cell matrices.
    alpha : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        Per-system Ewald splitting parameter.
    atom_start : wp.array, shape (B,), dtype=wp.int32
        First atom index for each system.
    atom_end : wp.array, shape (B,), dtype=wp.int32
        Last atom index (exclusive) for each system.
    total_charges : wp.array, shape (B,), dtype=wp.float64
        OUTPUT: Accumulated (Q_total/V) per system for background correction.
    cos_k_dot_r : wp.array2d, shape (K, N_total), dtype=wp.float64
        OUTPUT: :math:`\\cos(k \\cdot r_i)` for each (k, atom) pair.
    sin_k_dot_r : wp.array2d, shape (K, N_total), dtype=wp.float64
        OUTPUT: :math:`\\sin(k \\cdot r_i)` for each (k, atom) pair.
    real_structure_factors : wp.array2d, shape (B, K), dtype=wp.float64
        OUTPUT: Per-system :math:`(G(k)/V) \\sum_i q_i \\cos(k \\cdot r_i)`.
    imag_structure_factors : wp.array2d, shape (B, K), dtype=wp.float64
        OUTPUT: Per-system :math:`(G(k)/V) \\sum_i q_i \\sin(k \\cdot r_i)`.

    Notes
    -----
    - Blocked iteration reduces atomic contention vs atom-major.
    - Each block computes partial sums in registers before one atomic add.
    - BATCH_BLOCK_SIZE=16 is optimal for most cases (set via environment variable ALCH_EWALD_BATCH_BLOCK_SIZE).
    - k=0 causes early return (would cause division by zero in G(k)).
    - Blocks beyond the system's atoms cause early return.
    - Thread 1 accumulates total_charges as Q/V for background correction.
    - All internal computations use float64 for numerical stability.
    - Half-space k-vectors with 8π Green's function give ~2x speedup.
    """
    k_idx, system_id, block_idx = wp.tid()

    system_cell = cell[system_id]
    system_alpha = wp.float64(alpha[system_id])

    a_start = atom_start[system_id]
    a_end = atom_end[system_id]

    # Compute atom range for this block
    block_start = a_start + block_idx * BATCH_BLOCK_SIZE
    block_end = wp.min(block_start + BATCH_BLOCK_SIZE, a_end)

    # Skip if this block is beyond the system's atoms
    if block_start >= a_end:
        return

    exp_factor = wp.float64(0.25) / (system_alpha * system_alpha)
    volume = wp.float64(wp.abs(wp.determinant(system_cell)))

    k_vector = k_vectors[system_id, k_idx]
    kx = wp.float64(k_vector[0])
    ky = wp.float64(k_vector[1])
    kz = wp.float64(k_vector[2])
    k_squared = kx * kx + ky * ky + kz * kz

    # Skip k=0 (would cause division by zero)
    if k_squared < wp.float64(1e-10):
        return

    # Compute Green's function: (4*pi/V) * exp(-k^2/(4*alpha^2)) / k^2
    green_function = wp_exp_kernel(k_squared, exp_factor) * wp.float64(EIGHTPI) / volume

    # Accumulate partial sums for this block in registers
    local_real = wp.float64(0.0)
    local_imag = wp.float64(0.0)
    local_charge = wp.float64(0.0)

    for atom_idx in range(block_start, block_end):
        position = positions[atom_idx]
        charge = wp.float64(charges[atom_idx])

        # Only first k-thread per block accumulates total charge
        if k_idx == 1:
            local_charge += charge / volume

        # Compute cos(k*r) and sin(k*r) weighted by charge
        k_dot_r = (
            kx * wp.float64(position[0])
            + ky * wp.float64(position[1])
            + kz * wp.float64(position[2])
        )
        cos_kr = wp.cos(k_dot_r)
        sin_kr = wp.sin(k_dot_r)

        # Store per-(k, atom) UNWEIGHTED phase factors (for charge gradients)
        cos_k_dot_r[k_idx, atom_idx] = cos_kr
        sin_k_dot_r[k_idx, atom_idx] = sin_kr

        # Accumulate structure factors (charge-weighted) in registers
        local_real += charge * cos_kr * green_function
        local_imag += charge * sin_kr * green_function

    # One atomic add per block (much fewer atomics than atom-major!)
    wp.atomic_add(real_structure_factors, system_id, k_idx, local_real)
    wp.atomic_add(imag_structure_factors, system_id, k_idx, local_imag)

    if k_idx == 1:
        wp.atomic_add(total_charges, system_id, local_charge)


@wp.kernel
def _batch_ewald_reciprocal_space_energy_kernel_compute_energy(
    charges: wp.array(dtype=Any),
    batch_id: wp.array(dtype=wp.int32),
    cos_k_dot_r: wp.array2d(dtype=wp.float64),
    sin_k_dot_r: wp.array2d(dtype=wp.float64),
    real_structure_factors: wp.array2d(dtype=wp.float64),
    imag_structure_factors: wp.array2d(dtype=wp.float64),
    reciprocal_energies: wp.array(dtype=wp.float64),
):
    """Compute per-atom reciprocal-space energies for batched systems.

    This kernel uses atom-major iteration: each thread processes one atom
    over all k-vectors. This avoids atomics since each thread fully owns
    its atom's output.

    For each atom i in system s:

    .. math::

        E_i = \\frac{1}{2} \\sum_k [S_{\\text{real}}(s,k) \\cos(k \\cdot r_i) + S_{\\text{imag}}(s,k) \\sin(k \\cdot r_i)] q_i

    Uses batch_id to look up the correct system's structure factors.

    Launch Grid
    -----------
    dim = [N_total]

    Each thread processes one atom over all K k-vectors.

    Parameters
    ----------
    charges : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        Atomic charges.
    batch_id : wp.array, shape (N_total,), dtype=wp.int32
        System index for each atom (0 to B-1).
    cos_k_dot_r : wp.array2d, shape (K, N_total), dtype=wp.float64
        :math:`\\cos(k \\cdot r_i)` from structure factor computation.
    sin_k_dot_r : wp.array2d, shape (K, N_total), dtype=wp.float64
        :math:`\\sin(k \\cdot r_i)` from structure factor computation.
    real_structure_factors : wp.array2d, shape (B, K), dtype=wp.float64
        Per-system :math:`S_{\\text{real}}(s, k)` including Green's function.
    imag_structure_factors : wp.array2d, shape (B, K), dtype=wp.float64
        Per-system :math:`S_{\\text{imag}}(s, k)` including Green's function.
    reciprocal_energies : wp.array, shape (N_total,), dtype=wp.float64
        OUTPUT: Reciprocal-space energy per atom.

    Notes
    -----
    - Atom-major iteration avoids atomics (each thread owns its atom output)
    - cos_k_dot_r and sin_k_dot_r are unweighted; charge is multiplied here
    - All computations in float64
    """
    atom_idx = wp.tid()
    num_k = real_structure_factors.shape[1]
    charge = wp.float64(charges[atom_idx])

    system_id = batch_id[atom_idx]

    # Accumulate potential in register (no atomics!)
    local_potential = wp.float64(0.0)

    for k_idx in range(num_k):
        cos_kr = cos_k_dot_r[k_idx, atom_idx]
        sin_kr = sin_k_dot_r[k_idx, atom_idx]
        s_real = real_structure_factors[system_id, k_idx]
        s_imag = imag_structure_factors[system_id, k_idx]

        phase_sum = s_real * cos_kr + s_imag * sin_kr
        local_potential += charge * phase_sum

    # Write final energy: E_i = (1/2) * q_i * phi_i (no atomics needed)
    reciprocal_energies[atom_idx] = wp.float64(0.5) * local_potential


@wp.kernel
def _batch_ewald_subtract_self_energy_kernel(
    charges: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    alpha: wp.array(dtype=Any),
    total_charges: wp.array(dtype=wp.float64),
    energy_in: wp.array(dtype=wp.float64),
    energy_out: wp.array(dtype=wp.float64),
):
    """Apply self-energy and background corrections for batched systems.

    For each atom i in system s:

    .. math::

        E_{\\text{out},i} = E_{\\text{in},i} - E_{\\text{self},i} - E_{\\text{background},i}

    where:

    .. math::

        \\begin{aligned}
        E_{\\text{self},i} &= \\frac{\\alpha_s}{\\sqrt{\\pi}} q_i^2 \\\\
        E_{\\text{background},i} &= \\frac{\\pi}{2\\alpha_s^2} q_i \\frac{Q_{s,\\text{total}}}{V_s}
        \\end{aligned}

    Uses per-system alpha and total_charge values looked up via batch_idx.

    Launch Grid
    -----------
    dim = [N_total]

    Parameters
    ----------
    charges : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        Atomic charges for all systems concatenated.
    batch_idx : wp.array, shape (N_total,), dtype=wp.int32
        System index for each atom (0 to B-1).
    alpha : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        Per-system Ewald splitting parameter.
    total_charges : wp.array, shape (B,), dtype=wp.float64
        Per-system (Q_total/V), precomputed in structure factor kernel.
    energy_in : wp.array, shape (N_total,), dtype=wp.float64
        Raw reciprocal-space energy per atom.
    energy_out : wp.array, shape (N_total,), dtype=wp.float64
        OUTPUT: Corrected reciprocal-space energy per atom.

    Notes
    -----
    - Uses separate input/output arrays for autodiff compatibility
    - Each system may have different alpha and total charge values
    - All computations in float64
    """
    atom_index = wp.tid()
    charge = wp.float64(charges[atom_index])
    system_id = batch_idx[atom_index]
    system_alpha = wp.float64(alpha[system_id])
    system_total_charge = total_charges[system_id]

    # Compute self-energy: alpha * q^2 / sqrt(pi)
    self_energy = system_alpha * charge * charge / wp.sqrt(wp.float64(PI))

    # Background correction: pi / (2*alpha^2) * q * (Q_total/V)
    neutralization_energy = (
        wp.float64(PI)
        * charge
        * system_total_charge
        / (wp.float64(2.0) * system_alpha * system_alpha)
    )

    # Subtract self-energy and background (separate input/output to avoid autodiff issues)
    energy_out[atom_index] = energy_in[atom_index] - self_energy - neutralization_energy


@wp.kernel
def _batch_ewald_reciprocal_space_energy_forces_kernel(
    charges: wp.array(dtype=Any),
    batch_id: wp.array(dtype=wp.int32),
    k_vectors: wp.array2d(dtype=Any),
    cos_k_dot_r: wp.array2d(dtype=wp.float64),
    sin_k_dot_r: wp.array2d(dtype=wp.float64),
    real_structure_factors: wp.array2d(dtype=wp.float64),
    imag_structure_factors: wp.array2d(dtype=wp.float64),
    reciprocal_energies: wp.array(dtype=wp.float64),
    atomic_forces: wp.array(dtype=Any),
):
    """Compute reciprocal-space energies and forces for batched systems.

    This kernel uses atom-major iteration: each thread processes one atom
    over all k-vectors. This avoids atomics since each thread fully owns
    its atom's output.

    For each atom i in system s:

    .. math::

        \\begin{aligned}
        E_i &= \\frac{1}{2} \\sum_k [S_{\\text{real}}(s,k) \\cos(k \\cdot r_i) + S_{\\text{imag}}(s,k) \\sin(k \\cdot r_i)] q_i \\\\
        F_i &= \\sum_k k [S_{\\text{real}}(s,k) \\sin(k \\cdot r_i) - S_{\\text{imag}}(s,k) \\cos(k \\cdot r_i)] q_i
        \\end{aligned}

    Uses batch_id to look up the correct system's k-vectors and structure factors.

    Launch Grid
    -----------
    dim = [N_total]

    Each thread processes one atom over all K k-vectors.

    Parameters
    ----------
    charges : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        Atomic charges.
    batch_id : wp.array, shape (N_total,), dtype=wp.int32
        System index for each atom (0 to B-1).
    k_vectors : wp.array2d, shape (B, K), dtype=wp.vec3f or wp.vec3d
        Per-system reciprocal lattice vectors.
    cos_k_dot_r : wp.array2d, shape (K, N_total), dtype=wp.float64
        :math:`\\cos(k \\cdot r_i)` from structure factor computation.
    sin_k_dot_r : wp.array2d, shape (K, N_total), dtype=wp.float64
        :math:`\\sin(k \\cdot r_i)` from structure factor computation.
    real_structure_factors : wp.array2d, shape (B, K), dtype=wp.float64
        Per-system :math:`S_{\\text{real}}(s, k)` including Green's function.
    imag_structure_factors : wp.array2d, shape (B, K), dtype=wp.float64
        Per-system :math:`S_{\\text{imag}}(s, k)` including Green's function.
    reciprocal_energies : wp.array, shape (N_total,), dtype=wp.float64
        OUTPUT: Reciprocal-space energy per atom.
    atomic_forces : wp.array, shape (N_total,), dtype=wp.vec3f or wp.vec3d
        OUTPUT: Reciprocal-space forces per atom.

    Notes
    -----
    - Atom-major iteration avoids atomics (each thread owns its atom output)
    - cos_k_dot_r and sin_k_dot_r are unweighted; charge is multiplied here
    - Energy computed in float64, forces in k_vectors dtype
    """
    atom_idx = wp.tid()
    num_k = real_structure_factors.shape[1]
    charge = wp.float64(charges[atom_idx])

    system_id = batch_id[atom_idx]

    # Get the zero vector in the correct type
    k0 = k_vectors[system_id, 0]

    # Accumulate in registers (no atomics!)
    local_potential = wp.float64(0.0)
    local_force_x = wp.float64(0.0)
    local_force_y = wp.float64(0.0)
    local_force_z = wp.float64(0.0)

    for k_idx in range(num_k):
        cos_kr = charge * cos_k_dot_r[k_idx, atom_idx]
        sin_kr = charge * sin_k_dot_r[k_idx, atom_idx]

        # Load precomputed structure factors (already include green function)
        s_real = real_structure_factors[system_id, k_idx]
        s_imag = imag_structure_factors[system_id, k_idx]

        # Potential contribution
        phase_sum = s_real * cos_kr + s_imag * sin_kr
        local_potential += phase_sum

        # Force contribution
        force_scalar = s_real * sin_kr - s_imag * cos_kr
        k_vec = k_vectors[system_id, k_idx]
        local_force_x += force_scalar * wp.float64(k_vec[0])
        local_force_y += force_scalar * wp.float64(k_vec[1])
        local_force_z += force_scalar * wp.float64(k_vec[2])

    # Write final results with charge multiplication (no atomics needed)
    reciprocal_energies[atom_idx] = wp.float64(0.5) * local_potential
    atomic_forces[atom_idx] = type(k0)(
        type(k0[0])(local_force_x),
        type(k0[0])(local_force_y),
        type(k0[0])(local_force_z),
    )


@wp.kernel
def _batch_ewald_reciprocal_space_energy_forces_charge_grad_kernel(
    charges: wp.array(dtype=Any),
    batch_id: wp.array(dtype=wp.int32),
    k_vectors: wp.array2d(dtype=Any),
    cos_k_dot_r: wp.array2d(dtype=wp.float64),
    sin_k_dot_r: wp.array2d(dtype=wp.float64),
    real_structure_factors: wp.array2d(dtype=wp.float64),
    imag_structure_factors: wp.array2d(dtype=wp.float64),
    reciprocal_energies: wp.array(dtype=wp.float64),
    atomic_forces: wp.array(dtype=Any),
    charge_gradients: wp.array(dtype=wp.float64),
):
    """Compute reciprocal-space energies, forces, AND charge gradients for batched systems.

    This kernel computes all three quantities in a single pass:

    .. math::

        \\begin{aligned}
        E_i &= \\frac{1}{2} \\sum_k [S_{\\text{real}}(s,k) \\cos(k \\cdot r_i) + S_{\\text{imag}}(s,k) \\sin(k \\cdot r_i)] q_i \\\\
        F_i &= \\sum_k k [S_{\\text{real}}(s,k) \\sin(k \\cdot r_i) - S_{\\text{imag}}(s,k) \\cos(k \\cdot r_i)] q_i \\\\
        dE_i/dq_i &= \\sum_k [S_{\\text{real}}(s,k) \\cos(k \\cdot r_i) + S_{\\text{imag}}(s,k) \\sin(k \\cdot r_i)]
        \\end{aligned}

    where :math:`\\phi_i = \\sum_k [S_{\\text{real}}(s,k) \\cos(k \\cdot r_i) + S_{\\text{imag}}(s,k) \\sin(k \\cdot r_i)]` is the
    electrostatic potential at atom i from system s.

    Launch Grid
    -----------
    dim = [N_total]

    Each thread processes one atom over all K k-vectors.

    Parameters
    ----------
    charges : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        Atomic charges.
    batch_id : wp.array, shape (N_total,), dtype=wp.int32
        System index for each atom (0 to B-1).
    k_vectors : wp.array2d, shape (B, K), dtype=wp.vec3f or wp.vec3d
        Per-system reciprocal lattice vectors.
    cos_k_dot_r : wp.array2d, shape (K, N_total), dtype=wp.float64
        :math:`\\cos(k \\cdot r_i)` from structure factor computation (unweighted).
    sin_k_dot_r : wp.array2d, shape (K, N_total), dtype=wp.float64
        :math:`\\sin(k \\cdot r_i)` from structure factor computation (unweighted).
    real_structure_factors : wp.array2d, shape (B, K), dtype=wp.float64
        Per-system :math:`S_{\\text{real}}(s, k)` including Green's function.
    imag_structure_factors : wp.array2d, shape (B, K), dtype=wp.float64
        Per-system :math:`S_{\\text{imag}}(s, k)` including Green's function.
    reciprocal_energies : wp.array, shape (N_total,), dtype=wp.float64
        OUTPUT: Reciprocal-space energy per atom.
    atomic_forces : wp.array, shape (N_total,), dtype=wp.vec3f or wp.vec3d
        OUTPUT: Reciprocal-space forces per atom.
    charge_gradients : wp.array, shape (N_total,), dtype=wp.float64
        OUTPUT: Electrostatic potential :math:`\\phi_i` per atom (reciprocal part of charge gradient).
    """
    atom_idx = wp.tid()
    num_k = real_structure_factors.shape[1]
    charge = wp.float64(charges[atom_idx])

    system_id = batch_id[atom_idx]

    # Get the zero vector in the correct type
    k0 = k_vectors[system_id, 0]

    # Accumulate in registers (no atomics!)
    local_potential = wp.float64(0.0)
    local_potential_uncharged = wp.float64(0.0)
    local_force_x = wp.float64(0.0)
    local_force_y = wp.float64(0.0)
    local_force_z = wp.float64(0.0)

    for k_idx in range(num_k):
        cos_kr = cos_k_dot_r[k_idx, atom_idx]
        sin_kr = sin_k_dot_r[k_idx, atom_idx]

        # Load precomputed structure factors (already include green function)
        s_real = real_structure_factors[system_id, k_idx]
        s_imag = imag_structure_factors[system_id, k_idx]

        # Potential contribution
        phase_sum = s_real * cos_kr + s_imag * sin_kr
        local_potential += charge * phase_sum
        local_potential_uncharged += phase_sum

        # Force contribution
        force_scalar = charge * (s_real * sin_kr - s_imag * cos_kr)
        k_vec = k_vectors[system_id, k_idx]
        local_force_x += force_scalar * wp.float64(k_vec[0])
        local_force_y += force_scalar * wp.float64(k_vec[1])
        local_force_z += force_scalar * wp.float64(k_vec[2])

    # Write final results (no atomics needed)
    # Energy
    reciprocal_energies[atom_idx] = wp.float64(0.5) * local_potential

    # Forces
    atomic_forces[atom_idx] = type(k0)(
        type(k0[0])(local_force_x),
        type(k0[0])(local_force_y),
        type(k0[0])(local_force_z),
    )

    # Charge gradient
    # Self-energy and background corrections applied in higher-level code
    charge_gradients[atom_idx] = local_potential_uncharged


###########################################################################################
########################### Kernel Overloads (float32/float64) ############################
###########################################################################################

# Type aliases for clarity
_T = [wp.float32, wp.float64]
_V = [wp.vec3f, wp.vec3d]
_M = [wp.mat33f, wp.mat33d]

# Dictionaries to store overloads, keyed by scalar type (wp.float32 or wp.float64)
# Real-space single-system kernels
_ewald_real_space_energy_kernel_overload = {}
_ewald_real_space_energy_forces_kernel_overload = {}
_ewald_real_space_energy_neighbor_matrix_kernel_overload = {}
_ewald_real_space_energy_forces_neighbor_matrix_kernel_overload = {}

# Real-space single-system kernels with charge gradients
_ewald_real_space_energy_forces_charge_grad_kernel_overload = {}
_ewald_real_space_energy_forces_charge_grad_neighbor_matrix_kernel_overload = {}

# Real-space batch kernels
_batch_ewald_real_space_energy_kernel_overload = {}
_batch_ewald_real_space_energy_forces_kernel_overload = {}
_batch_ewald_real_space_energy_neighbor_matrix_kernel_overload = {}
_batch_ewald_real_space_energy_forces_neighbor_matrix_kernel_overload = {}

# Real-space batch kernels with charge gradients
_batch_ewald_real_space_energy_forces_charge_grad_kernel_overload = {}
_batch_ewald_real_space_energy_forces_charge_grad_neighbor_matrix_kernel_overload = {}

# Reciprocal-space single-system kernels
_ewald_reciprocal_space_energy_kernel_fill_structure_factors_overload = {}
_ewald_reciprocal_space_energy_kernel_compute_energy_overload = {}
_ewald_reciprocal_space_energy_forces_kernel_overload = {}
_ewald_reciprocal_space_energy_forces_charge_grad_kernel_overload = {}
_ewald_subtract_self_energy_kernel_overload = {}

# Reciprocal-space batch kernels
_batch_ewald_reciprocal_space_energy_kernel_fill_structure_factors_overload = {}
_batch_ewald_reciprocal_space_energy_kernel_compute_energy_overload = {}
_batch_ewald_reciprocal_space_energy_forces_kernel_overload = {}
_batch_ewald_reciprocal_space_energy_forces_charge_grad_kernel_overload = {}
_batch_ewald_subtract_self_energy_kernel_overload = {}

for t, v, m in zip(_T, _V, _M):
    # ==================== Real-space single-system kernels ====================

    _ewald_real_space_energy_kernel_overload[t] = wp.overload(
        _ewald_real_space_energy_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=t),  # charges
            wp.array(dtype=m),  # cell
            wp.array(dtype=wp.int32),  # idx_i
            wp.array(dtype=wp.int32),  # idx_j
            wp.array(dtype=wp.vec3i),  # unit_shifts
            wp.array(dtype=t),  # alpha
            wp.array(dtype=wp.float64),  # pair_energies (always float64)
        ],
    )

    _ewald_real_space_energy_forces_kernel_overload[t] = wp.overload(
        _ewald_real_space_energy_forces_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=t),  # charges
            wp.array(dtype=m),  # cell
            wp.array(dtype=wp.int32),  # idx_i
            wp.array(dtype=wp.int32),  # idx_j
            wp.array(dtype=wp.vec3i),  # unit_shifts
            wp.array(dtype=t),  # alpha
            wp.array(dtype=wp.float64),  # pair_energies
            wp.array(dtype=v),  # atomic_forces (matches positions dtype)
        ],
    )

    _ewald_real_space_energy_neighbor_matrix_kernel_overload[t] = wp.overload(
        _ewald_real_space_energy_neighbor_matrix_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=t),  # charges
            wp.array(dtype=m),  # cell
            wp.array2d(dtype=wp.int32),  # neighbor_matrix
            wp.array2d(dtype=wp.vec3i),  # unit_shifts_matrix
            wp.int32,  # mask_value
            wp.array(dtype=t),  # alpha
            wp.array(dtype=wp.float64),  # pair_energies
        ],
    )

    _ewald_real_space_energy_forces_neighbor_matrix_kernel_overload[t] = wp.overload(
        _ewald_real_space_energy_forces_neighbor_matrix_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=t),  # charges
            wp.array(dtype=m),  # cell
            wp.array2d(dtype=wp.int32),  # neighbor_matrix
            wp.array2d(dtype=wp.vec3i),  # unit_shifts_matrix
            wp.int32,  # mask_value
            wp.array(dtype=t),  # alpha
            wp.array(dtype=wp.float64),  # pair_energies
            wp.array(dtype=v),  # atomic_forces
        ],
    )

    # ==================== Real-space single-system kernels with charge gradients ====================

    _ewald_real_space_energy_forces_charge_grad_kernel_overload[t] = wp.overload(
        _ewald_real_space_energy_forces_charge_grad_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=t),  # charges
            wp.array(dtype=m),  # cell
            wp.array(dtype=wp.int32),  # idx_i
            wp.array(dtype=wp.int32),  # idx_j
            wp.array(dtype=wp.vec3i),  # unit_shifts
            wp.array(dtype=t),  # alpha
            wp.array(dtype=wp.float64),  # pair_energies
            wp.array(dtype=v),  # atomic_forces
            wp.array(dtype=wp.float64),  # charge_gradients
        ],
    )

    _ewald_real_space_energy_forces_charge_grad_neighbor_matrix_kernel_overload[t] = (
        wp.overload(
            _ewald_real_space_energy_forces_charge_grad_neighbor_matrix_kernel,
            [
                wp.array(dtype=v),  # positions
                wp.array(dtype=t),  # charges
                wp.array(dtype=m),  # cell
                wp.array2d(dtype=wp.int32),  # neighbor_matrix
                wp.array2d(dtype=wp.vec3i),  # unit_shifts_matrix
                wp.int32,  # mask_value
                wp.array(dtype=t),  # alpha
                wp.array(dtype=wp.float64),  # pair_energies
                wp.array(dtype=v),  # atomic_forces
                wp.array(dtype=wp.float64),  # charge_gradients
            ],
        )
    )

    # ==================== Real-space batch kernels ====================

    _batch_ewald_real_space_energy_kernel_overload[t] = wp.overload(
        _batch_ewald_real_space_energy_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=t),  # charges
            wp.array(dtype=m),  # cell
            wp.array(dtype=wp.int32),  # batch_id
            wp.array(dtype=wp.int32),  # idx_i
            wp.array(dtype=wp.int32),  # idx_j
            wp.array(dtype=wp.vec3i),  # unit_shifts
            wp.array(dtype=t),  # alpha
            wp.array(dtype=wp.float64),  # pair_energies
        ],
    )

    _batch_ewald_real_space_energy_forces_kernel_overload[t] = wp.overload(
        _batch_ewald_real_space_energy_forces_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=t),  # charges
            wp.array(dtype=m),  # cell
            wp.array(dtype=wp.int32),  # batch_id
            wp.array(dtype=wp.int32),  # idx_i
            wp.array(dtype=wp.int32),  # idx_j
            wp.array(dtype=wp.vec3i),  # unit_shifts
            wp.array(dtype=t),  # alpha
            wp.array(dtype=wp.float64),  # pair_energies
            wp.array(dtype=v),  # atomic_forces
        ],
    )

    _batch_ewald_real_space_energy_neighbor_matrix_kernel_overload[t] = wp.overload(
        _batch_ewald_real_space_energy_neighbor_matrix_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=t),  # charges
            wp.array(dtype=m),  # cell
            wp.array(dtype=wp.int32),  # batch_id
            wp.array2d(dtype=wp.int32),  # neighbor_matrix
            wp.array2d(dtype=wp.vec3i),  # unit_shifts_matrix
            wp.int32,  # mask_value
            wp.array(dtype=t),  # alpha
            wp.array(dtype=wp.float64),  # pair_energies
        ],
    )

    _batch_ewald_real_space_energy_forces_neighbor_matrix_kernel_overload[t] = (
        wp.overload(
            _batch_ewald_real_space_energy_forces_neighbor_matrix_kernel,
            [
                wp.array(dtype=v),  # positions
                wp.array(dtype=t),  # charges
                wp.array(dtype=m),  # cell
                wp.array(dtype=wp.int32),  # batch_id
                wp.array2d(dtype=wp.int32),  # neighbor_matrix
                wp.array2d(dtype=wp.vec3i),  # unit_shifts_matrix
                wp.int32,  # mask_value
                wp.array(dtype=t),  # alpha
                wp.array(dtype=wp.float64),  # pair_energies
                wp.array(dtype=v),  # atomic_forces
            ],
        )
    )

    # ==================== Real-space batch kernels with charge gradients ====================

    _batch_ewald_real_space_energy_forces_charge_grad_kernel_overload[t] = wp.overload(
        _batch_ewald_real_space_energy_forces_charge_grad_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=t),  # charges
            wp.array(dtype=m),  # cell
            wp.array(dtype=wp.int32),  # batch_id
            wp.array(dtype=wp.int32),  # idx_i
            wp.array(dtype=wp.int32),  # idx_j
            wp.array(dtype=wp.vec3i),  # unit_shifts
            wp.array(dtype=t),  # alpha
            wp.array(dtype=wp.float64),  # pair_energies
            wp.array(dtype=v),  # atomic_forces
            wp.array(dtype=wp.float64),  # charge_gradients
        ],
    )

    _batch_ewald_real_space_energy_forces_charge_grad_neighbor_matrix_kernel_overload[
        t
    ] = wp.overload(
        _batch_ewald_real_space_energy_forces_charge_grad_neighbor_matrix_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=t),  # charges
            wp.array(dtype=m),  # cell
            wp.array(dtype=wp.int32),  # batch_id
            wp.array2d(dtype=wp.int32),  # neighbor_matrix
            wp.array2d(dtype=wp.vec3i),  # unit_shifts_matrix
            wp.int32,  # mask_value
            wp.array(dtype=t),  # alpha
            wp.array(dtype=wp.float64),  # pair_energies
            wp.array(dtype=v),  # atomic_forces
            wp.array(dtype=wp.float64),  # charge_gradients
        ],
    )

    # ==================== Reciprocal-space single-system kernels ====================

    _ewald_reciprocal_space_energy_kernel_fill_structure_factors_overload[t] = (
        wp.overload(
            _ewald_reciprocal_space_energy_kernel_fill_structure_factors,
            [
                wp.array(dtype=v),  # positions
                wp.array(dtype=t),  # charges
                wp.array(dtype=v),  # k_vectors
                wp.array(dtype=m),  # cell
                wp.array(dtype=t),  # alpha
                wp.array(dtype=wp.float64),  # total_charge
                wp.array2d(dtype=wp.float64),  # cos_k_dot_r
                wp.array2d(dtype=wp.float64),  # sin_k_dot_r
                wp.array(dtype=wp.float64),  # real_structure_factors
                wp.array(dtype=wp.float64),  # imag_structure_factors
            ],
        )
    )

    _ewald_reciprocal_space_energy_kernel_compute_energy_overload[t] = wp.overload(
        _ewald_reciprocal_space_energy_kernel_compute_energy,
        [
            wp.array(dtype=t),  # charges
            wp.array2d(dtype=wp.float64),  # cos_k_dot_r
            wp.array2d(dtype=wp.float64),  # sin_k_dot_r
            wp.array(dtype=wp.float64),  # real_structure_factors
            wp.array(dtype=wp.float64),  # imag_structure_factors
            wp.array(dtype=wp.float64),  # reciprocal_energies
        ],
    )

    _ewald_reciprocal_space_energy_forces_kernel_overload[t] = wp.overload(
        _ewald_reciprocal_space_energy_forces_kernel,
        [
            wp.array(dtype=t),  # charges
            wp.array(dtype=v),  # k_vectors
            wp.array2d(dtype=wp.float64),  # cos_k_dot_r
            wp.array2d(dtype=wp.float64),  # sin_k_dot_r
            wp.array(dtype=wp.float64),  # real_structure_factors
            wp.array(dtype=wp.float64),  # imag_structure_factors
            wp.array(dtype=wp.float64),  # reciprocal_energies
            wp.array(dtype=v),  # atomic_forces
        ],
    )

    _ewald_reciprocal_space_energy_forces_charge_grad_kernel_overload[t] = wp.overload(
        _ewald_reciprocal_space_energy_forces_charge_grad_kernel,
        [
            wp.array(dtype=t),  # charges
            wp.array(dtype=v),  # k_vectors
            wp.array2d(dtype=wp.float64),  # cos_k_dot_r
            wp.array2d(dtype=wp.float64),  # sin_k_dot_r
            wp.array(dtype=wp.float64),  # real_structure_factors
            wp.array(dtype=wp.float64),  # imag_structure_factors
            wp.array(dtype=wp.float64),  # reciprocal_energies
            wp.array(dtype=v),  # atomic_forces
            wp.array(dtype=wp.float64),  # charge_gradients
        ],
    )

    _ewald_subtract_self_energy_kernel_overload[t] = wp.overload(
        _ewald_subtract_self_energy_kernel,
        [
            wp.array(dtype=t),  # charges
            wp.array(dtype=t),  # alpha
            wp.array(dtype=wp.float64),  # total_charge
            wp.array(dtype=wp.float64),  # energy_in
            wp.array(dtype=wp.float64),  # energy_out
        ],
    )

    # ==================== Reciprocal-space batch kernels ====================

    _batch_ewald_reciprocal_space_energy_kernel_fill_structure_factors_overload[t] = (
        wp.overload(
            _batch_ewald_reciprocal_space_energy_kernel_fill_structure_factors,
            [
                wp.array(dtype=v),  # positions
                wp.array(dtype=t),  # charges
                wp.array2d(dtype=v),  # k_vectors (B, K)
                wp.array(dtype=m),  # cell
                wp.array(dtype=t),  # alpha
                wp.array(dtype=wp.int32),  # atom_start
                wp.array(dtype=wp.int32),  # atom_end
                wp.array(dtype=wp.float64),  # total_charges
                wp.array2d(dtype=wp.float64),  # cos_k_dot_r
                wp.array2d(dtype=wp.float64),  # sin_k_dot_r
                wp.array2d(dtype=wp.float64),  # real_structure_factors
                wp.array2d(dtype=wp.float64),  # imag_structure_factors
            ],
        )
    )

    _batch_ewald_reciprocal_space_energy_kernel_compute_energy_overload[t] = (
        wp.overload(
            _batch_ewald_reciprocal_space_energy_kernel_compute_energy,
            [
                wp.array(dtype=t),  # charges
                wp.array(dtype=wp.int32),  # batch_id
                wp.array2d(dtype=wp.float64),  # cos_k_dot_r
                wp.array2d(dtype=wp.float64),  # sin_k_dot_r
                wp.array2d(dtype=wp.float64),  # real_structure_factors
                wp.array2d(dtype=wp.float64),  # imag_structure_factors
                wp.array(dtype=wp.float64),  # reciprocal_energies
            ],
        )
    )

    _batch_ewald_reciprocal_space_energy_forces_kernel_overload[t] = wp.overload(
        _batch_ewald_reciprocal_space_energy_forces_kernel,
        [
            wp.array(dtype=t),  # charges
            wp.array(dtype=wp.int32),  # batch_id
            wp.array2d(dtype=v),  # k_vectors (B, K)
            wp.array2d(dtype=wp.float64),  # cos_k_dot_r
            wp.array2d(dtype=wp.float64),  # sin_k_dot_r
            wp.array2d(dtype=wp.float64),  # real_structure_factors
            wp.array2d(dtype=wp.float64),  # imag_structure_factors
            wp.array(dtype=wp.float64),  # reciprocal_energies
            wp.array(dtype=v),  # atomic_forces
        ],
    )

    _batch_ewald_reciprocal_space_energy_forces_charge_grad_kernel_overload[t] = (
        wp.overload(
            _batch_ewald_reciprocal_space_energy_forces_charge_grad_kernel,
            [
                wp.array(dtype=t),  # charges
                wp.array(dtype=wp.int32),  # batch_id
                wp.array2d(dtype=v),  # k_vectors (B, K)
                wp.array2d(dtype=wp.float64),  # cos_k_dot_r
                wp.array2d(dtype=wp.float64),  # sin_k_dot_r
                wp.array2d(dtype=wp.float64),  # real_structure_factors
                wp.array2d(dtype=wp.float64),  # imag_structure_factors
                wp.array(dtype=wp.float64),  # reciprocal_energies
                wp.array(dtype=v),  # atomic_forces
                wp.array(dtype=wp.float64),  # charge_gradients
            ],
        )
    )

    _batch_ewald_subtract_self_energy_kernel_overload[t] = wp.overload(
        _batch_ewald_subtract_self_energy_kernel,
        [
            wp.array(dtype=t),  # charges
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array(dtype=t),  # alpha
            wp.array(dtype=wp.float64),  # total_charges
            wp.array(dtype=wp.float64),  # energy_in
            wp.array(dtype=wp.float64),  # energy_out
        ],
    )
