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
Unified Ewald Summation Kernels
===============================

This module provides GPU-accelerated Warp kernels for Ewald summation,
enabling efficient calculation of long-range Coulomb interactions. All kernels
support both single-system and batched calculations via the batch_idx parameter.

DTYPE FLEXIBILITY
=================

All public launchers support both float32 and float64 input types:
- Input tensors (positions, charges, cell, alpha): float32 or float64
- Accumulators (energies, structure factors): Always float64 for numerical stability
- Forces: Match input positions dtype (float32 or float64)

Real-space and reciprocal-space launchers route through cached factory-backed
kernels.

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
    - ewald_real_space_* launchers: public low-level Warp launchers
    - get_ewald_real_kernel: factory-selected single/batch CSR/matrix kernels

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
    a block of atoms, reducing atomic contention.
    - 16 is the default block size
    - Larger values can be useful for workloads with fewer, larger systems
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

__all__ = [
    "BATCH_BLOCK_SIZE",
    "REAL_SPACE_TILED_BLOCK_DIM",
    "batch_ewald_energy_corrections",
    "batch_ewald_energy_corrections_backward",
    "batch_ewald_energy_corrections_double_backward",
    "batch_ewald_real_space_energy",
    "batch_ewald_real_space_energy_forces",
    "batch_ewald_real_space_energy_forces_charge_grad",
    "batch_ewald_real_space_energy_forces_charge_grad_matrix",
    "batch_ewald_real_space_energy_forces_matrix",
    "batch_ewald_real_space_energy_matrix",
    "batch_ewald_reciprocal_space_compute_energy",
    "batch_ewald_reciprocal_space_energy_forces",
    "batch_ewald_reciprocal_space_energy_forces_charge_grad",
    "batch_ewald_reciprocal_space_fill_structure_factors",
    "batch_ewald_subtract_self_energy",
    "ewald_energy_corrections",
    "ewald_energy_corrections_backward",
    "ewald_energy_corrections_double_backward",
    "ewald_real_space_energy",
    "ewald_real_space_energy_forces",
    "ewald_real_space_energy_forces_charge_grad",
    "ewald_real_space_energy_forces_charge_grad_matrix",
    "ewald_real_space_energy_forces_matrix",
    "ewald_real_space_energy_matrix",
    "ewald_reciprocal_space_compute_energy",
    "ewald_reciprocal_space_energy_forces",
    "ewald_reciprocal_space_energy_forces_charge_grad",
    "ewald_reciprocal_space_fill_structure_factors",
    "ewald_subtract_self_energy",
]

# Mathematical constants
PI = math.pi
SQRT_PI = math.sqrt(PI)
TWO_OVER_SQRT_PI = 2.0 / SQRT_PI
TWOPI = 2.0 * PI
FOURPI = 4.0 * PI
EIGHTPI = 8.0 * PI  # Half-space k-vector Green's function factor.

# Block size for batch structure factor accumulation
# Tunable via ALCH_EWALD_BATCH_BLOCK_SIZE.
BATCH_BLOCK_SIZE = int(os.environ.get("ALCH_EWALD_BATCH_BLOCK_SIZE", 16))
BATCH_BLOCK_SIZE = BATCH_BLOCK_SIZE if BATCH_BLOCK_SIZE > 0 else 16

# Block size for block-per-atom tiled real-space neighbor-matrix kernels.
# Each atom is handled by REAL_SPACE_TILED_BLOCK_DIM cooperating threads; the
# per-thread accumulators are reduced via wp.tile_sum to a single block-local
# result. On CPU warp clamps block_dim to 1 and the tile primitives degrade to
# scalar passthrough.
REAL_SPACE_TILED_BLOCK_DIM = 64


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
    two_over_sqrt_pi = wp.float64(TWO_OVER_SQRT_PI)

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
        background correction. Only the first k-vector thread accumulates this.
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
    - Thread 0 accumulates total_charge as Q/V for background correction.
    - All internal computations use float64 for numerical stability.
    - cos_k_dot_r and sin_k_dot_r store unweighted phases for charge gradient computation.
    - Half-space k-vectors use the corresponding 8π Green's function factor.
    """
    k_idx = wp.tid()
    num_atoms = positions.shape[0]

    alpha_ = wp.float64(alpha[0])
    exp_factor = wp.float64(0.25) / (alpha_ * alpha_)
    volume = wp.float64(wp.abs(wp.determinant(cell[0])))

    k_vector = k_vectors[k_idx]
    kx = wp.float64(k_vector[0])
    ky = wp.float64(k_vector[1])
    kz = wp.float64(k_vector[2])
    k_squared = kx * kx + ky * ky + kz * kz

    if k_squared < wp.float64(1e-10):
        if k_idx == 0:
            total_charge_accum = wp.float64(0.0)
            for atom_idx in range(num_atoms):
                total_charge_accum += wp.float64(charges[atom_idx]) / volume
            total_charge[0] = total_charge_accum
        for atom_idx in range(num_atoms):
            cos_k_dot_r[k_idx, atom_idx] = wp.float64(0.0)
            sin_k_dot_r[k_idx, atom_idx] = wp.float64(0.0)
        real_structure_factors[k_idx] = wp.float64(0.0)
        imag_structure_factors[k_idx] = wp.float64(0.0)
        return

    # Compute Green's function: (8*pi/V) * exp(-k^2/(4*alpha^2)) / k^2
    green_function = wp_exp_kernel(k_squared, exp_factor) * wp.float64(EIGHTPI) / volume

    # Accumulate structure factors in registers (no atomics!)
    real_sum = wp.float64(0.0)
    imag_sum = wp.float64(0.0)
    total_charge_accum = wp.float64(0.0)

    for atom_idx in range(num_atoms):
        position = positions[atom_idx]
        charge = wp.float64(charges[atom_idx])

        # Thread 0 accumulates total charge for background correction. This
        # keeps the correction valid for one-k-vector launches.
        if k_idx == 0:
            total_charge_accum += charge / volume

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

    if k_idx == 0:
        total_charge[0] = total_charge_accum
    real_structure_factors[k_idx] = real_sum
    imag_structure_factors[k_idx] = imag_sum


@wp.kernel
def _ewald_reciprocal_space_energy_kernel_fill_structure_factors_cellgrad(
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
    cellgrad_cache: wp.array2d(dtype=wp.float64),
):
    """Forward fill + the un-weighted cell-grad reduction (single system).

    Byte-identical to ``_ewald_reciprocal_space_energy_kernel_fill_structure_factors``
    for all existing outputs; additionally accumulates, in the SAME atom loop, the
    un-weighted per-``k`` sums consumed by the O(K) ``kspace`` backward
    (``ewald_recip_factory._make_backward_kspace_from_cache_kernel``):

      cellgrad_cache[k] = [A, B, Ra_x, Ra_y, Ra_z, Rb_x, Rb_y, Rb_z]

    with ``A = sum_i q_i cos(k.r_i)``, ``B = sum_i q_i sin(k.r_i)``,
    ``Ra = sum_i q_i cos(k.r_i) r_i``, ``Rb = sum_i q_i sin(k.r_i) r_i`` (all
    un-weighted -- the Green's function ``g_k`` is applied in the consume kernel).
    """
    k_idx = wp.tid()
    num_atoms = positions.shape[0]

    alpha_ = wp.float64(alpha[0])
    exp_factor = wp.float64(0.25) / (alpha_ * alpha_)
    volume = wp.float64(wp.abs(wp.determinant(cell[0])))

    k_vector = k_vectors[k_idx]
    kx = wp.float64(k_vector[0])
    ky = wp.float64(k_vector[1])
    kz = wp.float64(k_vector[2])
    k_squared = kx * kx + ky * ky + kz * kz

    if k_squared < wp.float64(1e-10):
        if k_idx == 0:
            total_charge_accum = wp.float64(0.0)
            for atom_idx in range(num_atoms):
                total_charge_accum += wp.float64(charges[atom_idx]) / volume
            total_charge[0] = total_charge_accum
        for atom_idx in range(num_atoms):
            cos_k_dot_r[k_idx, atom_idx] = wp.float64(0.0)
            sin_k_dot_r[k_idx, atom_idx] = wp.float64(0.0)
        real_structure_factors[k_idx] = wp.float64(0.0)
        imag_structure_factors[k_idx] = wp.float64(0.0)
        for col in range(8):
            cellgrad_cache[k_idx, col] = wp.float64(0.0)
        return

    green_function = wp_exp_kernel(k_squared, exp_factor) * wp.float64(EIGHTPI) / volume

    real_sum = wp.float64(0.0)
    imag_sum = wp.float64(0.0)
    a_sum = wp.float64(0.0)
    b_sum = wp.float64(0.0)
    ra_x = wp.float64(0.0)
    ra_y = wp.float64(0.0)
    ra_z = wp.float64(0.0)
    rb_x = wp.float64(0.0)
    rb_y = wp.float64(0.0)
    rb_z = wp.float64(0.0)
    total_charge_accum = wp.float64(0.0)

    for atom_idx in range(num_atoms):
        position = positions[atom_idx]
        charge = wp.float64(charges[atom_idx])

        if k_idx == 0:
            total_charge_accum += charge / volume

        rx = wp.float64(position[0])
        ry = wp.float64(position[1])
        rz = wp.float64(position[2])
        k_dot_r = kx * rx + ky * ry + kz * rz
        cos_kr = wp.cos(k_dot_r)
        sin_kr = wp.sin(k_dot_r)

        cos_k_dot_r[k_idx, atom_idx] = cos_kr
        sin_k_dot_r[k_idx, atom_idx] = sin_kr

        real_sum += charge * cos_kr * green_function
        imag_sum += charge * sin_kr * green_function

        # Un-weighted cell-grad reduction (marginal extra FMAs in this loop).
        qc = charge * cos_kr
        qs = charge * sin_kr
        a_sum += qc
        b_sum += qs
        ra_x += qc * rx
        ra_y += qc * ry
        ra_z += qc * rz
        rb_x += qs * rx
        rb_y += qs * ry
        rb_z += qs * rz

    real_structure_factors[k_idx] = real_sum
    imag_structure_factors[k_idx] = imag_sum
    if k_idx == 0:
        total_charge[0] = total_charge_accum
    cellgrad_cache[k_idx, 0] = a_sum
    cellgrad_cache[k_idx, 1] = b_sum
    cellgrad_cache[k_idx, 2] = ra_x
    cellgrad_cache[k_idx, 3] = ra_y
    cellgrad_cache[k_idx, 4] = ra_z
    cellgrad_cache[k_idx, 5] = rb_x
    cellgrad_cache[k_idx, 6] = rb_y
    cellgrad_cache[k_idx, 7] = rb_z


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
    if num_k == 0:
        reciprocal_energies[atom_idx] = wp.float64(0.0)
        atomic_forces[atom_idx] = type(atomic_forces[atom_idx])(
            type(atomic_forces[atom_idx][0])(0.0),
            type(atomic_forces[atom_idx][0])(0.0),
            type(atomic_forces[atom_idx][0])(0.0),
        )
        return

    charge = wp.float64(charges[atom_idx])

    # Accumulate in registers (no atomics!)
    local_potential = wp.float64(0.0)
    local_force_x = wp.float64(0.0)
    local_force_y = wp.float64(0.0)
    local_force_z = wp.float64(0.0)

    for k_idx in range(num_k):
        cos_kr = charge * cos_k_dot_r[k_idx, atom_idx]
        sin_kr = charge * sin_k_dot_r[k_idx, atom_idx]

        s_real = real_structure_factors[k_idx]
        s_imag = imag_structure_factors[k_idx]

        phase_sum = s_real * cos_kr + s_imag * sin_kr
        local_potential += phase_sum

        force_scalar = s_real * sin_kr - s_imag * cos_kr
        k_vec = k_vectors[k_idx]
        local_force_x += force_scalar * wp.float64(k_vec[0])
        local_force_y += force_scalar * wp.float64(k_vec[1])
        local_force_z += force_scalar * wp.float64(k_vec[2])

    reciprocal_energies[atom_idx] = wp.float64(0.5) * local_potential
    atomic_forces[atom_idx] = type(atomic_forces[atom_idx])(
        type(atomic_forces[atom_idx][0])(local_force_x),
        type(atomic_forces[atom_idx][0])(local_force_y),
        type(atomic_forces[atom_idx][0])(local_force_z),
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
    if num_k == 0:
        reciprocal_energies[atom_idx] = wp.float64(0.0)
        atomic_forces[atom_idx] = type(atomic_forces[atom_idx])(
            type(atomic_forces[atom_idx][0])(0.0),
            type(atomic_forces[atom_idx][0])(0.0),
            type(atomic_forces[atom_idx][0])(0.0),
        )
        charge_gradients[atom_idx] = wp.float64(0.0)
        return

    charge = wp.float64(charges[atom_idx])

    # Accumulate in registers (no atomics!)
    local_potential = wp.float64(0.0)
    local_potential_uncharged = wp.float64(0.0)
    local_force_x = wp.float64(0.0)
    local_force_y = wp.float64(0.0)
    local_force_z = wp.float64(0.0)

    for k_idx in range(num_k):
        cos_kr = cos_k_dot_r[k_idx, atom_idx]
        sin_kr = sin_k_dot_r[k_idx, atom_idx]

        s_real = real_structure_factors[k_idx]
        s_imag = imag_structure_factors[k_idx]

        phase_sum = s_real * cos_kr + s_imag * sin_kr
        local_potential += charge * phase_sum
        local_potential_uncharged += phase_sum

        force_scalar = charge * (s_real * sin_kr - s_imag * cos_kr)
        k_vec = k_vectors[k_idx]
        local_force_x += force_scalar * wp.float64(k_vec[0])
        local_force_y += force_scalar * wp.float64(k_vec[1])
        local_force_z += force_scalar * wp.float64(k_vec[2])

    reciprocal_energies[atom_idx] = wp.float64(0.5) * local_potential

    atomic_forces[atom_idx] = type(atomic_forces[atom_idx])(
        type(atomic_forces[atom_idx][0])(local_force_x),
        type(atomic_forces[atom_idx][0])(local_force_y),
        type(atomic_forces[atom_idx][0])(local_force_z),
    )

    # Self-energy and background corrections applied in higher-level code
    charge_gradients[atom_idx] = local_potential_uncharged


###########################################################################################
#################### Reciprocal-Space Virial Kernels ######################################
###########################################################################################


@wp.kernel
def _ewald_reciprocal_space_virial_kernel(
    k_vectors: wp.array(dtype=Any),
    alpha: wp.array(dtype=Any),
    volume: wp.array(dtype=wp.float64),
    real_structure_factors: wp.array(dtype=wp.float64),
    imag_structure_factors: wp.array(dtype=wp.float64),
    virial: wp.array(dtype=Any),
):
    """Compute the reciprocal-space virial tensor from precomputed structure factors.

    For each k-vector, the virial contribution is:

    .. math::

        W_{ab}(k) = E(k) \\left[ \\frac{2 k_a k_b}{k^2} \\left(1 + \\frac{k^2}{4\\alpha^2}\\right) - \\delta_{ab} \\right]

    where the per-k energy is :math:`E(k) = \\frac{|S(k)|^2}{2 G(k)}` and
    :math:`G(k) = \\frac{8\\pi}{V} \\frac{\\exp(-k^2/(4\\alpha^2))}{k^2}`.

    Launch Grid
    -----------
    dim = [K]

    Parameters
    ----------
    k_vectors : wp.array, shape (K,), dtype=wp.vec3f or wp.vec3d
        Reciprocal lattice vectors.
    alpha : wp.array, shape (1,), dtype=wp.float32 or wp.float64
        Ewald splitting parameter.
    volume : wp.array, shape (1,), dtype=wp.float64
        Unit cell volume |det(cell)|.
    real_structure_factors : wp.array, shape (K,), dtype=wp.float64
        Precomputed S_real(k) = G(k) * sum_i q_i cos(k.r_i).
    imag_structure_factors : wp.array, shape (K,), dtype=wp.float64
        Precomputed S_imag(k) = G(k) * sum_i q_i sin(k.r_i).
    virial : wp.array, shape (1,), dtype=wp.mat33f or wp.mat33d
        OUTPUT: Accumulated virial tensor.
    """
    k_idx = wp.tid()

    k_vec = k_vectors[k_idx]
    kx = wp.float64(k_vec[0])
    ky = wp.float64(k_vec[1])
    kz = wp.float64(k_vec[2])
    k_sq = kx * kx + ky * ky + kz * kz

    if k_sq < wp.float64(1e-10):
        return

    alpha_ = wp.float64(alpha[0])
    vol = volume[0]
    s_real = real_structure_factors[k_idx]
    s_imag = imag_structure_factors[k_idx]

    # |S(k)|^2
    s_sq = s_real * s_real + s_imag * s_imag

    # Green's function G(k) = (8*pi/V) * exp(-k^2/(4*alpha^2)) / k^2
    exp_factor = wp.float64(0.25) / (alpha_ * alpha_)
    green = wp.float64(EIGHTPI) / vol * wp.exp(-k_sq * exp_factor) / k_sq

    # Per-k energy: E(k) = |S|^2 / (2*G)
    energy_k = wp.float64(0.5) * s_sq / green

    # Virial W = -dE/dε.  d ln G / dε_ab = -δ_ab + 2 k_a k_b / k² (1 + k²/(4α²)),
    # so W_ab(k) = E(k) * [δ_ab - 2 k_a k_b / k² (1 + k²/(4α²))].
    k_factor = wp.float64(2.0) * (wp.float64(1.0) + k_sq * exp_factor) / k_sq

    w00 = energy_k * (wp.float64(1.0) - k_factor * kx * kx)
    w01 = energy_k * (-k_factor * kx * ky)
    w02 = energy_k * (-k_factor * kx * kz)
    w10 = energy_k * (-k_factor * ky * kx)
    w11 = energy_k * (wp.float64(1.0) - k_factor * ky * ky)
    w12 = energy_k * (-k_factor * ky * kz)
    w20 = energy_k * (-k_factor * kz * kx)
    w21 = energy_k * (-k_factor * kz * ky)
    w22 = energy_k * (wp.float64(1.0) - k_factor * kz * kz)

    _virial_ref = virial[0]
    virial_k = type(_virial_ref)(
        type(k_vec[0])(w00),
        type(k_vec[0])(w01),
        type(k_vec[0])(w02),
        type(k_vec[0])(w10),
        type(k_vec[0])(w11),
        type(k_vec[0])(w12),
        type(k_vec[0])(w20),
        type(k_vec[0])(w21),
        type(k_vec[0])(w22),
    )
    wp.atomic_add(virial, 0, virial_k)


@wp.kernel
def _batch_ewald_reciprocal_space_virial_kernel(
    k_vectors: wp.array2d(dtype=Any),
    alpha: wp.array(dtype=Any),
    volume: wp.array(dtype=wp.float64),
    real_structure_factors: wp.array2d(dtype=wp.float64),
    imag_structure_factors: wp.array2d(dtype=wp.float64),
    virial: wp.array(dtype=Any),
):
    """Compute the reciprocal-space virial tensor for batched systems.

    Same formula as single-system version, but with per-system k-vectors,
    structure factors, alpha, and volume.

    Launch Grid
    -----------
    dim = [K, B]

    Parameters
    ----------
    k_vectors : wp.array2d, shape (B, K), dtype=wp.vec3f or wp.vec3d
        Per-system reciprocal lattice vectors.
    alpha : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        Per-system Ewald splitting parameter.
    volume : wp.array, shape (B,), dtype=wp.float64
        Per-system unit cell volume |det(cell)|.
    real_structure_factors : wp.array2d, shape (B, K), dtype=wp.float64
        Per-system S_real(k).
    imag_structure_factors : wp.array2d, shape (B, K), dtype=wp.float64
        Per-system S_imag(k).
    virial : wp.array, shape (B,), dtype=wp.mat33f or wp.mat33d
        OUTPUT: Per-system accumulated virial tensor.
    """
    k_idx, system_id = wp.tid()

    k_vec = k_vectors[system_id, k_idx]
    kx = wp.float64(k_vec[0])
    ky = wp.float64(k_vec[1])
    kz = wp.float64(k_vec[2])
    k_sq = kx * kx + ky * ky + kz * kz

    if k_sq < wp.float64(1e-10):
        return

    alpha_ = wp.float64(alpha[system_id])
    vol = volume[system_id]
    s_real = real_structure_factors[system_id, k_idx]
    s_imag = imag_structure_factors[system_id, k_idx]

    # |S(k)|^2
    s_sq = s_real * s_real + s_imag * s_imag

    # Green's function G(k) = (8*pi/V) * exp(-k^2/(4*alpha^2)) / k^2
    exp_factor = wp.float64(0.25) / (alpha_ * alpha_)
    green = wp.float64(EIGHTPI) / vol * wp.exp(-k_sq * exp_factor) / k_sq

    # Per-k energy: E(k) = |S|^2 / (2*G)
    energy_k = wp.float64(0.5) * s_sq / green

    # Virial W = -dE/dε.  d ln G / dε_ab = -δ_ab + 2 k_a k_b / k² (1 + k²/(4α²)),
    # so W_ab(k) = E(k) * [δ_ab - 2 k_a k_b / k² (1 + k²/(4α²))].
    k_factor = wp.float64(2.0) * (wp.float64(1.0) + k_sq * exp_factor) / k_sq

    w00 = energy_k * (wp.float64(1.0) - k_factor * kx * kx)
    w01 = energy_k * (-k_factor * kx * ky)
    w02 = energy_k * (-k_factor * kx * kz)
    w10 = energy_k * (-k_factor * ky * kx)
    w11 = energy_k * (wp.float64(1.0) - k_factor * ky * ky)
    w12 = energy_k * (-k_factor * ky * kz)
    w20 = energy_k * (-k_factor * kz * kx)
    w21 = energy_k * (-k_factor * kz * ky)
    w22 = energy_k * (wp.float64(1.0) - k_factor * kz * kz)

    _virial_ref = virial[system_id]
    virial_k = type(_virial_ref)(
        type(k_vec[0])(w00),
        type(k_vec[0])(w01),
        type(k_vec[0])(w02),
        type(k_vec[0])(w10),
        type(k_vec[0])(w11),
        type(k_vec[0])(w12),
        type(k_vec[0])(w20),
        type(k_vec[0])(w21),
        type(k_vec[0])(w22),
    )
    wp.atomic_add(virial, system_id, virial_k)


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
    (default: 16).

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
    - BATCH_BLOCK_SIZE is set via ALCH_EWALD_BATCH_BLOCK_SIZE.
    - k=0 causes early return (would cause division by zero in G(k)).
    - Blocks beyond the system's atoms cause early return.
    - Thread 0 accumulates total_charges as Q/V for background correction.
    - All internal computations use float64 for numerical stability.
    - Half-space k-vectors use the corresponding 8π Green's function factor.
    """
    k_idx, system_id, block_idx = wp.tid()

    system_cell = cell[system_id]
    system_alpha = wp.float64(alpha[system_id])

    a_start = atom_start[system_id]
    a_end = atom_end[system_id]

    block_start = a_start + block_idx * BATCH_BLOCK_SIZE
    block_end = wp.min(block_start + BATCH_BLOCK_SIZE, a_end)

    if block_start >= a_end:
        return

    exp_factor = wp.float64(0.25) / (system_alpha * system_alpha)
    volume = wp.float64(wp.abs(wp.determinant(system_cell)))

    k_vector = k_vectors[system_id, k_idx]
    kx = wp.float64(k_vector[0])
    ky = wp.float64(k_vector[1])
    kz = wp.float64(k_vector[2])
    k_squared = kx * kx + ky * ky + kz * kz

    if k_squared < wp.float64(1e-10):
        if k_idx == 0:
            local_charge = wp.float64(0.0)
            for atom_idx in range(block_start, block_end):
                local_charge += wp.float64(charges[atom_idx]) / volume
            wp.atomic_add(total_charges, system_id, local_charge)
        for atom_idx in range(block_start, block_end):
            cos_k_dot_r[k_idx, atom_idx] = wp.float64(0.0)
            sin_k_dot_r[k_idx, atom_idx] = wp.float64(0.0)
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

        # Only first k-thread per block accumulates total charge.
        if k_idx == 0:
            local_charge += charge / volume

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

    if k_idx == 0:
        wp.atomic_add(total_charges, system_id, local_charge)


@wp.kernel
def _batch_ewald_reciprocal_space_energy_kernel_fill_structure_factors_cellgrad(
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
    cellgrad_cache: wp.array2d(dtype=wp.float64),
):
    """Compute batched structure factors plus first-order cell-gradient sums.

    Thread launch
    -------------
    ``dim = (K, B, max_blocks_per_system)``. Each thread owns one
    ``(k-vector, system, atom block)`` partial reduction.

    Modifies
    --------
    ``total_charges``, ``cos_k_dot_r``, ``sin_k_dot_r``,
    ``real_structure_factors``, ``imag_structure_factors``, and
    ``cellgrad_cache``. The cache layout is ``row = system_id * K + k_idx`` with
    columns ``[A, B, Ra_x, Ra_y, Ra_z, Rb_x, Rb_y, Rb_z]``.
    """
    k_idx, system_id, block_idx = wp.tid()

    system_cell = cell[system_id]
    system_alpha = wp.float64(alpha[system_id])

    a_start = atom_start[system_id]
    a_end = atom_end[system_id]

    block_start = a_start + block_idx * BATCH_BLOCK_SIZE
    block_end = wp.min(block_start + BATCH_BLOCK_SIZE, a_end)
    if block_start >= a_end:
        return

    exp_factor = wp.float64(0.25) / (system_alpha * system_alpha)
    volume = wp.float64(wp.abs(wp.determinant(system_cell)))

    k_vector = k_vectors[system_id, k_idx]
    kx = wp.float64(k_vector[0])
    ky = wp.float64(k_vector[1])
    kz = wp.float64(k_vector[2])
    k_squared = kx * kx + ky * ky + kz * kz

    if k_squared < wp.float64(1e-10):
        if k_idx == 0:
            local_charge = wp.float64(0.0)
            for atom_idx in range(block_start, block_end):
                local_charge += wp.float64(charges[atom_idx]) / volume
            wp.atomic_add(total_charges, system_id, local_charge)
        for atom_idx in range(block_start, block_end):
            cos_k_dot_r[k_idx, atom_idx] = wp.float64(0.0)
            sin_k_dot_r[k_idx, atom_idx] = wp.float64(0.0)
        return

    green_function = wp_exp_kernel(k_squared, exp_factor) * wp.float64(EIGHTPI) / volume

    local_real = wp.float64(0.0)
    local_imag = wp.float64(0.0)
    local_charge = wp.float64(0.0)
    local_a = wp.float64(0.0)
    local_b = wp.float64(0.0)
    local_ra_x = wp.float64(0.0)
    local_ra_y = wp.float64(0.0)
    local_ra_z = wp.float64(0.0)
    local_rb_x = wp.float64(0.0)
    local_rb_y = wp.float64(0.0)
    local_rb_z = wp.float64(0.0)

    for atom_idx in range(block_start, block_end):
        position = positions[atom_idx]
        charge = wp.float64(charges[atom_idx])

        if k_idx == 0:
            local_charge += charge / volume

        rx = wp.float64(position[0])
        ry = wp.float64(position[1])
        rz = wp.float64(position[2])
        k_dot_r = kx * rx + ky * ry + kz * rz
        cos_kr = wp.cos(k_dot_r)
        sin_kr = wp.sin(k_dot_r)

        cos_k_dot_r[k_idx, atom_idx] = cos_kr
        sin_k_dot_r[k_idx, atom_idx] = sin_kr

        local_real += charge * cos_kr * green_function
        local_imag += charge * sin_kr * green_function

        qc = charge * cos_kr
        qs = charge * sin_kr
        local_a += qc
        local_b += qs
        local_ra_x += qc * rx
        local_ra_y += qc * ry
        local_ra_z += qc * rz
        local_rb_x += qs * rx
        local_rb_y += qs * ry
        local_rb_z += qs * rz

    wp.atomic_add(real_structure_factors, system_id, k_idx, local_real)
    wp.atomic_add(imag_structure_factors, system_id, k_idx, local_imag)

    row = system_id * k_vectors.shape[1] + k_idx
    wp.atomic_add(cellgrad_cache, row, 0, local_a)
    wp.atomic_add(cellgrad_cache, row, 1, local_b)
    wp.atomic_add(cellgrad_cache, row, 2, local_ra_x)
    wp.atomic_add(cellgrad_cache, row, 3, local_ra_y)
    wp.atomic_add(cellgrad_cache, row, 4, local_ra_z)
    wp.atomic_add(cellgrad_cache, row, 5, local_rb_x)
    wp.atomic_add(cellgrad_cache, row, 6, local_rb_y)
    wp.atomic_add(cellgrad_cache, row, 7, local_rb_z)

    if k_idx == 0:
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


###########################################################################################
########################### Ewald Correction Autograd Kernels ##############################
###########################################################################################


@wp.kernel
def _ewald_energy_corrections_kernel(
    raw_energies: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    volume: wp.array(dtype=Any),
    alpha: wp.array(dtype=Any),
    total_charge: wp.array(dtype=Any),
    corrected_energies: wp.array(dtype=Any),
):
    """Apply Ewald reciprocal self/background corrections.

    Thread launch
    -------------
    One thread per atom.

    Modifies
    --------
    corrected_energies
        Per-atom corrected reciprocal energies.
    """
    i = wp.tid()
    q = charges[i]
    r = raw_energies[i]
    v = volume[0]
    a = alpha[0]
    qtot = total_charge[0]

    pi = type(q)(PI)
    two = type(q)(2.0)
    self_contrib = a * q * q / wp.sqrt(pi)
    background_contrib = pi * q * qtot / (two * a * a * v)
    corrected_energies[i] = r - self_contrib - background_contrib


@wp.kernel
def _batch_ewald_energy_corrections_kernel(
    raw_energies: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    volumes: wp.array(dtype=Any),
    alpha: wp.array(dtype=Any),
    total_charges: wp.array(dtype=Any),
    corrected_energies: wp.array(dtype=Any),
):
    """Batched Ewald reciprocal self/background corrections.

    Thread launch
    -------------
    One thread per atom.

    Modifies
    --------
    corrected_energies
        Per-atom corrected reciprocal energies.
    """
    i = wp.tid()
    s = batch_idx[i]
    q = charges[i]
    r = raw_energies[i]
    v = volumes[s]
    a = alpha[s]
    qtot = total_charges[s]

    pi = type(q)(PI)
    two = type(q)(2.0)
    self_contrib = a * q * q / wp.sqrt(pi)
    background_contrib = pi * q * qtot / (two * a * a * v)
    corrected_energies[i] = r - self_contrib - background_contrib


@wp.kernel
def _ewald_energy_corrections_backward_kernel(
    grad_E: wp.array(dtype=Any),
    raw_energies: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    volume: wp.array(dtype=Any),
    alpha: wp.array(dtype=Any),
    total_charge: wp.array(dtype=Any),
    grad_raw: wp.array(dtype=Any),
    grad_charges: wp.array(dtype=Any),
    grad_volume: wp.array(dtype=Any),
    grad_alpha: wp.array(dtype=Any),
    grad_total_charge: wp.array(dtype=Any),
):
    """Single-system backward for Ewald reciprocal corrections."""
    i = wp.tid()
    g = grad_E[i]
    q = charges[i]
    a = alpha[0]
    v = volume[0]
    qtot = total_charge[0]

    pi = type(g)(PI)
    two = type(g)(2.0)
    sqrt_pi = wp.sqrt(pi)
    c2 = pi / (two * a * a * v)

    grad_raw[i] = g
    grad_charges[i] = g * (-two * a * q / sqrt_pi - c2 * qtot)

    d_alpha = g * (-(q * q) / sqrt_pi + pi * q * qtot / (a * a * a * v))
    wp.atomic_add(grad_alpha, 0, d_alpha)

    d_volume = g * pi * q * qtot / (two * a * a * v * v)
    wp.atomic_add(grad_volume, 0, d_volume)

    d_qtot = -g * c2 * q
    wp.atomic_add(grad_total_charge, 0, d_qtot)


@wp.kernel
def _batch_ewald_energy_corrections_backward_kernel(
    grad_E: wp.array(dtype=Any),
    raw_energies: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    volumes: wp.array(dtype=Any),
    alpha: wp.array(dtype=Any),
    total_charges: wp.array(dtype=Any),
    grad_raw: wp.array(dtype=Any),
    grad_charges: wp.array(dtype=Any),
    grad_volumes: wp.array(dtype=Any),
    grad_alpha: wp.array(dtype=Any),
    grad_total_charges: wp.array(dtype=Any),
):
    """Batched backward for Ewald reciprocal corrections."""
    i = wp.tid()
    s = batch_idx[i]
    g = grad_E[i]
    q = charges[i]
    a = alpha[s]
    v = volumes[s]
    qtot = total_charges[s]

    pi = type(g)(PI)
    two = type(g)(2.0)
    sqrt_pi = wp.sqrt(pi)
    c2 = pi / (two * a * a * v)

    grad_raw[i] = g
    grad_charges[i] = g * (-two * a * q / sqrt_pi - c2 * qtot)

    d_alpha = g * (-(q * q) / sqrt_pi + pi * q * qtot / (a * a * a * v))
    wp.atomic_add(grad_alpha, s, d_alpha)

    d_volume = g * pi * q * qtot / (two * a * a * v * v)
    wp.atomic_add(grad_volumes, s, d_volume)

    d_qtot = -g * c2 * q
    wp.atomic_add(grad_total_charges, s, d_qtot)


@wp.kernel
def _ewald_energy_corrections_double_backward_kernel(
    h_raw: wp.array(dtype=Any),
    h_chg: wp.array(dtype=Any),
    h_vol: wp.array(dtype=Any),
    h_alpha: wp.array(dtype=Any),
    h_qtot: wp.array(dtype=Any),
    grad_E: wp.array(dtype=Any),
    raw_energies: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    volume: wp.array(dtype=Any),
    alpha: wp.array(dtype=Any),
    total_charge: wp.array(dtype=Any),
    grad_grad_E: wp.array(dtype=Any),
    grad_raw: wp.array(dtype=Any),
    grad_charges: wp.array(dtype=Any),
    grad_volume: wp.array(dtype=Any),
    grad_alpha: wp.array(dtype=Any),
    grad_total_charge: wp.array(dtype=Any),
):
    """Single-system double-backward for Ewald reciprocal corrections."""
    i = wp.tid()
    g_i = grad_E[i]
    q = charges[i]
    a = alpha[0]
    v = volume[0]
    qtot = total_charge[0]
    hr = h_raw[i]
    hc = h_chg[i]
    hv = h_vol[0]
    ha = h_alpha[0]
    hq = h_qtot[0]

    pi = type(g_i)(PI)
    two = type(g_i)(2.0)
    three = type(g_i)(3.0)
    sqrt_pi = wp.sqrt(pi)
    c1 = a / sqrt_pi
    c2 = pi / (two * a * a * v)
    c_i = -two * c1 * q - c2 * qtot
    a_i = -(q * q) / sqrt_pi + pi * q * qtot / (a * a * a * v)
    b_i = pi * q * qtot / (two * a * a * v * v)
    d_i = -pi * q / (two * a * a * v)

    grad_grad_E[i] = hr + hc * c_i + ha * a_i + hv * b_i + hq * d_i
    grad_raw[i] = type(g_i)(0.0)

    grad_charges[i] = g_i * (
        hc * (-two * c1)
        + ha * (-two * q / sqrt_pi + pi * qtot / (a * a * a * v))
        + hv * (pi * qtot / (two * a * a * v * v))
        + hq * (-pi / (two * a * a * v))
    )

    g_q = g_i * q
    dV_atom = (
        hc * g_i * qtot * pi / (two * a * a * v * v)
        + ha * (-pi * qtot / (a * a * a * v * v)) * g_q
        + hv * (-pi * qtot / (a * a * v * v * v)) * g_q
        + hq * (pi / (two * a * a * v * v)) * g_q
    )
    wp.atomic_add(grad_volume, 0, dV_atom)

    dA_atom = (
        hc * g_i * (-two * q / sqrt_pi + pi * qtot / (a * a * a * v))
        + ha * (-three * pi * qtot / (a * a * a * a * v)) * g_q
        + hv * (-pi * qtot / (a * a * a * v * v)) * g_q
        + hq * (pi / (a * a * a * v)) * g_q
    )
    wp.atomic_add(grad_alpha, 0, dA_atom)

    dQ_atom = (
        hc * g_i * (-pi / (two * a * a * v))
        + ha * (pi / (a * a * a * v)) * g_q
        + hv * (pi / (two * a * a * v * v)) * g_q
    )
    wp.atomic_add(grad_total_charge, 0, dQ_atom)


@wp.kernel
def _batch_ewald_energy_corrections_double_backward_kernel(
    h_raw: wp.array(dtype=Any),
    h_chg: wp.array(dtype=Any),
    h_vol: wp.array(dtype=Any),
    h_alpha: wp.array(dtype=Any),
    h_qtot: wp.array(dtype=Any),
    grad_E: wp.array(dtype=Any),
    raw_energies: wp.array(dtype=Any),
    charges: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    volumes: wp.array(dtype=Any),
    alpha: wp.array(dtype=Any),
    total_charges: wp.array(dtype=Any),
    grad_grad_E: wp.array(dtype=Any),
    grad_raw: wp.array(dtype=Any),
    grad_charges: wp.array(dtype=Any),
    grad_volumes: wp.array(dtype=Any),
    grad_alpha: wp.array(dtype=Any),
    grad_total_charges: wp.array(dtype=Any),
):
    """Batched double-backward for Ewald reciprocal corrections."""
    i = wp.tid()
    s = batch_idx[i]
    g_i = grad_E[i]
    q = charges[i]
    a = alpha[s]
    v = volumes[s]
    qtot = total_charges[s]
    hr = h_raw[i]
    hc = h_chg[i]
    hv = h_vol[s]
    ha = h_alpha[s]
    hq = h_qtot[s]

    pi = type(g_i)(PI)
    two = type(g_i)(2.0)
    three = type(g_i)(3.0)
    sqrt_pi = wp.sqrt(pi)
    c1 = a / sqrt_pi
    c2 = pi / (two * a * a * v)
    c_i = -two * c1 * q - c2 * qtot
    a_i = -(q * q) / sqrt_pi + pi * q * qtot / (a * a * a * v)
    b_i = pi * q * qtot / (two * a * a * v * v)
    d_i = -pi * q / (two * a * a * v)

    grad_grad_E[i] = hr + hc * c_i + ha * a_i + hv * b_i + hq * d_i
    grad_raw[i] = type(g_i)(0.0)

    grad_charges[i] = g_i * (
        hc * (-two * c1)
        + ha * (-two * q / sqrt_pi + pi * qtot / (a * a * a * v))
        + hv * (pi * qtot / (two * a * a * v * v))
        + hq * (-pi / (two * a * a * v))
    )

    g_q = g_i * q
    dV_atom = (
        hc * g_i * qtot * pi / (two * a * a * v * v)
        + ha * (-pi * qtot / (a * a * a * v * v)) * g_q
        + hv * (-pi * qtot / (a * a * v * v * v)) * g_q
        + hq * (pi / (two * a * a * v * v)) * g_q
    )
    wp.atomic_add(grad_volumes, s, dV_atom)

    dA_atom = (
        hc * g_i * (-two * q / sqrt_pi + pi * qtot / (a * a * a * v))
        + ha * (-three * pi * qtot / (a * a * a * a * v)) * g_q
        + hv * (-pi * qtot / (a * a * a * v * v)) * g_q
        + hq * (pi / (a * a * a * v)) * g_q
    )
    wp.atomic_add(grad_alpha, s, dA_atom)

    dQ_atom = (
        hc * g_i * (-pi / (two * a * a * v))
        + ha * (pi / (a * a * a * v)) * g_q
        + hv * (pi / (two * a * a * v * v)) * g_q
    )
    wp.atomic_add(grad_total_charges, s, dQ_atom)


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
    system_id = batch_id[atom_idx]
    if num_k == 0:
        reciprocal_energies[atom_idx] = wp.float64(0.0)
        atomic_forces[atom_idx] = type(atomic_forces[atom_idx])(
            type(atomic_forces[atom_idx][0])(0.0),
            type(atomic_forces[atom_idx][0])(0.0),
            type(atomic_forces[atom_idx][0])(0.0),
        )
        return

    charge = wp.float64(charges[atom_idx])

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

        phase_sum = s_real * cos_kr + s_imag * sin_kr
        local_potential += phase_sum

        force_scalar = s_real * sin_kr - s_imag * cos_kr
        k_vec = k_vectors[system_id, k_idx]
        local_force_x += force_scalar * wp.float64(k_vec[0])
        local_force_y += force_scalar * wp.float64(k_vec[1])
        local_force_z += force_scalar * wp.float64(k_vec[2])

    reciprocal_energies[atom_idx] = wp.float64(0.5) * local_potential
    atomic_forces[atom_idx] = type(atomic_forces[atom_idx])(
        type(atomic_forces[atom_idx][0])(local_force_x),
        type(atomic_forces[atom_idx][0])(local_force_y),
        type(atomic_forces[atom_idx][0])(local_force_z),
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
    system_id = batch_id[atom_idx]
    if num_k == 0:
        reciprocal_energies[atom_idx] = wp.float64(0.0)
        atomic_forces[atom_idx] = type(atomic_forces[atom_idx])(
            type(atomic_forces[atom_idx][0])(0.0),
            type(atomic_forces[atom_idx][0])(0.0),
            type(atomic_forces[atom_idx][0])(0.0),
        )
        charge_gradients[atom_idx] = wp.float64(0.0)
        return

    charge = wp.float64(charges[atom_idx])

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

        phase_sum = s_real * cos_kr + s_imag * sin_kr
        local_potential += charge * phase_sum
        local_potential_uncharged += phase_sum

        force_scalar = charge * (s_real * sin_kr - s_imag * cos_kr)
        k_vec = k_vectors[system_id, k_idx]
        local_force_x += force_scalar * wp.float64(k_vec[0])
        local_force_y += force_scalar * wp.float64(k_vec[1])
        local_force_z += force_scalar * wp.float64(k_vec[2])

    reciprocal_energies[atom_idx] = wp.float64(0.5) * local_potential

    atomic_forces[atom_idx] = type(atomic_forces[atom_idx])(
        type(atomic_forces[atom_idx][0])(local_force_x),
        type(atomic_forces[atom_idx][0])(local_force_y),
        type(atomic_forces[atom_idx][0])(local_force_z),
    )

    # Self-energy and background corrections applied in higher-level code
    charge_gradients[atom_idx] = local_potential_uncharged


###########################################################################################
########################### Warp Launchers (Framework-Agnostic) ############################
###########################################################################################


def _launch_ewald_real_forward_factory(
    positions: wp.array,
    charges: wp.array,
    cell: wp.array,
    alpha: wp.array,
    pair_energies: wp.array,
    wp_dtype: type,
    device: str | None,
    *,
    batched: bool,
    neighbor_input: str,
    batch_id: wp.array | None = None,
    idx_j: wp.array | None = None,
    neighbor_ptr: wp.array | None = None,
    unit_shifts: wp.array | None = None,
    neighbor_matrix: wp.array | None = None,
    unit_shifts_matrix: wp.array | None = None,
    mask_value: int = 0,
    atomic_forces: wp.array | None = None,
    charge_gradients: wp.array | None = None,
    virial: wp.array | None = None,
    compute_virial: bool = False,
) -> None:
    """Launch the factory-backed Ewald real forward kernel."""
    if device is None:
        device = str(positions.device)

    if compute_virial and atomic_forces is None:
        raise ValueError("atomic_forces is required when compute_virial=True")

    from nvalchemiops.interactions.electrostatics._factory_common import _DerivState
    from nvalchemiops.interactions.electrostatics.ewald_real_factory import (
        alloc_ewald_real_sentinels,
        get_ewald_real_kernel,
    )

    if charge_gradients is not None:
        deriv_state = _DerivState.E_F_dQ
    elif atomic_forces is not None:
        deriv_state = _DerivState.E_F
    else:
        deriv_state = _DerivState.E

    sentinels = alloc_ewald_real_sentinels(wp_dtype, device)
    kernel = get_ewald_real_kernel(
        wp_dtype,
        batched=batched,
        neighbor_input=neighbor_input,
        deriv_state=deriv_state,
        cell_grad=compute_virial,
        tiled=neighbor_input == "matrix",
    )

    batch_arg = batch_id if batched else sentinels["batch_id"]
    if neighbor_input == "matrix":
        if neighbor_matrix is None or unit_shifts_matrix is None:
            raise ValueError(
                "neighbor_matrix and unit_shifts_matrix are required for matrix input"
            )
        launch_dim = int(neighbor_matrix.shape[0])
        idx_arg = sentinels["idx_j"]
        ptr_arg = sentinels["neighbor_ptr"]
        shifts_arg = sentinels["unit_shifts"]
        matrix_arg = neighbor_matrix
        matrix_shifts_arg = unit_shifts_matrix
    else:
        if idx_j is None or neighbor_ptr is None or unit_shifts is None:
            raise ValueError(
                "idx_j, neighbor_ptr, and unit_shifts are required for CSR input"
            )
        launch_dim = int(positions.shape[0])
        idx_arg = idx_j
        ptr_arg = neighbor_ptr
        shifts_arg = unit_shifts
        matrix_arg = sentinels["neighbor_matrix"]
        matrix_shifts_arg = sentinels["unit_shifts_matrix"]

    launch_inputs = [
        positions,
        charges,
        cell,
        batch_arg,
        idx_arg,
        ptr_arg,
        shifts_arg,
        matrix_arg,
        matrix_shifts_arg,
        wp.int32(mask_value),
        alpha,
        pair_energies,
        atomic_forces if atomic_forces is not None else sentinels["atomic_forces"],
        charge_gradients
        if charge_gradients is not None
        else sentinels["charge_gradients"],
        virial if compute_virial else sentinels["virial"],
    ]
    if neighbor_input == "matrix":
        wp.launch_tiled(
            kernel=kernel,
            dim=launch_dim,
            inputs=launch_inputs,
            block_dim=REAL_SPACE_TILED_BLOCK_DIM,
            device=device,
        )
    else:
        wp.launch(
            kernel=kernel,
            dim=launch_dim,
            inputs=launch_inputs,
            device=device,
        )


def ewald_real_space_energy(
    positions: wp.array,
    charges: wp.array,
    cell: wp.array,
    idx_j: wp.array,
    neighbor_ptr: wp.array,
    unit_shifts: wp.array,
    alpha: wp.array,
    pair_energies: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Launch Ewald real-space energy kernel using CSR neighbor list format.

    This is a framework-agnostic launcher that accepts warp arrays directly.

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        Atomic positions.
    charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Atomic charges.
    cell : wp.array, shape (1,), dtype=wp.mat33f or wp.mat33d
        Unit cell matrix.
    idx_j : wp.array, shape (M,), dtype=wp.int32
        Target atom indices (CSR data).
    neighbor_ptr : wp.array, shape (N+1,), dtype=wp.int32
        CSR row pointers.
    unit_shifts : wp.array, shape (M,), dtype=wp.vec3i
        Periodic image shifts.
    alpha : wp.array, shape (1,), dtype=wp.float32 or wp.float64
        Ewald splitting parameter.
    pair_energies : wp.array, shape (N,), dtype=wp.float64
        OUTPUT: Per-atom energies. Must be pre-allocated and zeroed.
    wp_dtype : type
        Warp scalar type (wp.float32 or wp.float64).
    device : str, optional
        Warp device. If None, inferred from positions.
    """
    _launch_ewald_real_forward_factory(
        positions,
        charges,
        cell,
        alpha,
        pair_energies,
        wp_dtype,
        device,
        batched=False,
        neighbor_input="list",
        idx_j=idx_j,
        neighbor_ptr=neighbor_ptr,
        unit_shifts=unit_shifts,
    )


def ewald_real_space_energy_forces(
    positions: wp.array,
    charges: wp.array,
    cell: wp.array,
    idx_j: wp.array,
    neighbor_ptr: wp.array,
    unit_shifts: wp.array,
    alpha: wp.array,
    pair_energies: wp.array,
    atomic_forces: wp.array,
    virial: wp.array,
    wp_dtype: type,
    device: str | None = None,
    compute_virial: bool = False,
) -> None:
    """Launch Ewald real-space energy and forces kernel using CSR neighbor list.

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        Atomic positions.
    charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Atomic charges.
    cell : wp.array, shape (1,), dtype=wp.mat33f or wp.mat33d
        Unit cell matrix.
    idx_j : wp.array, shape (M,), dtype=wp.int32
        Target atom indices (CSR data).
    neighbor_ptr : wp.array, shape (N+1,), dtype=wp.int32
        CSR row pointers.
    unit_shifts : wp.array, shape (M,), dtype=wp.vec3i
        Periodic image shifts.
    alpha : wp.array, shape (1,), dtype=wp.float32 or wp.float64
        Ewald splitting parameter.
    pair_energies : wp.array, shape (N,), dtype=wp.float64
        OUTPUT: Per-atom energies.
    atomic_forces : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        OUTPUT: Per-atom forces.
    virial : wp.array, shape (1,), dtype=wp.mat33f or wp.mat33d
        OUTPUT: Virial tensor. Only written when compute_virial=True.
        Must be pre-allocated and zeroed by caller.
    wp_dtype : type
        Warp scalar type (wp.float32 or wp.float64).
    device : str, optional
        Warp device.
    compute_virial : bool, optional
        Whether to compute the virial tensor. Default False.
    """
    _launch_ewald_real_forward_factory(
        positions,
        charges,
        cell,
        alpha,
        pair_energies,
        wp_dtype,
        device,
        batched=False,
        neighbor_input="list",
        idx_j=idx_j,
        neighbor_ptr=neighbor_ptr,
        unit_shifts=unit_shifts,
        atomic_forces=atomic_forces,
        virial=virial,
        compute_virial=compute_virial,
    )


def ewald_real_space_energy_matrix(
    positions: wp.array,
    charges: wp.array,
    cell: wp.array,
    neighbor_matrix: wp.array,
    unit_shifts_matrix: wp.array,
    mask_value: int,
    alpha: wp.array,
    pair_energies: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Launch Ewald real-space energy kernel using neighbor matrix format.

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        Atomic positions.
    charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Atomic charges.
    cell : wp.array, shape (1,), dtype=wp.mat33f or wp.mat33d
        Unit cell matrix.
    neighbor_matrix : wp.array2d, shape (N, max_neighbors), dtype=wp.int32
        Neighbor indices.
    unit_shifts_matrix : wp.array2d, shape (N, max_neighbors), dtype=wp.vec3i
        Periodic image shifts.
    mask_value : int
        Value indicating invalid/padded entries.
    alpha : wp.array, shape (1,), dtype=wp.float32 or wp.float64
        Ewald splitting parameter.
    pair_energies : wp.array, shape (N,), dtype=wp.float64
        OUTPUT: Per-atom energies.
    wp_dtype : type
        Warp scalar type (wp.float32 or wp.float64).
    device : str, optional
        Warp device.
    """
    _launch_ewald_real_forward_factory(
        positions,
        charges,
        cell,
        alpha,
        pair_energies,
        wp_dtype,
        device,
        batched=False,
        neighbor_input="matrix",
        neighbor_matrix=neighbor_matrix,
        unit_shifts_matrix=unit_shifts_matrix,
        mask_value=mask_value,
    )


def ewald_real_space_energy_forces_matrix(
    positions: wp.array,
    charges: wp.array,
    cell: wp.array,
    neighbor_matrix: wp.array,
    unit_shifts_matrix: wp.array,
    mask_value: int,
    alpha: wp.array,
    pair_energies: wp.array,
    atomic_forces: wp.array,
    virial: wp.array,
    wp_dtype: type,
    device: str | None = None,
    compute_virial: bool = False,
) -> None:
    """Launch Ewald real-space energy and forces kernel using neighbor matrix.

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        Atomic positions.
    charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Atomic charges.
    cell : wp.array, shape (1,), dtype=wp.mat33f or wp.mat33d
        Unit cell matrix.
    neighbor_matrix : wp.array2d, shape (N, max_neighbors), dtype=wp.int32
        Neighbor indices.
    unit_shifts_matrix : wp.array2d, shape (N, max_neighbors), dtype=wp.vec3i
        Periodic image shifts.
    mask_value : int
        Value indicating invalid/padded entries.
    alpha : wp.array, shape (1,), dtype=wp.float32 or wp.float64
        Ewald splitting parameter.
    pair_energies : wp.array, shape (N,), dtype=wp.float64
        OUTPUT: Per-atom energies.
    atomic_forces : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        OUTPUT: Per-atom forces.
    wp_dtype : type
        Warp scalar type (wp.float32 or wp.float64).
    device : str, optional
        Warp device.
    compute_virial : bool, optional
        Whether to compute the virial tensor. Default False.
    virial : wp.array, optional
        OUTPUT: Virial tensor. Must be pre-allocated by caller.
    """
    _launch_ewald_real_forward_factory(
        positions,
        charges,
        cell,
        alpha,
        pair_energies,
        wp_dtype,
        device,
        batched=False,
        neighbor_input="matrix",
        neighbor_matrix=neighbor_matrix,
        unit_shifts_matrix=unit_shifts_matrix,
        mask_value=mask_value,
        atomic_forces=atomic_forces,
        virial=virial,
        compute_virial=compute_virial,
    )


def ewald_real_space_energy_forces_charge_grad(
    positions: wp.array,
    charges: wp.array,
    cell: wp.array,
    idx_j: wp.array,
    neighbor_ptr: wp.array,
    unit_shifts: wp.array,
    alpha: wp.array,
    pair_energies: wp.array,
    atomic_forces: wp.array,
    charge_gradients: wp.array,
    virial: wp.array,
    wp_dtype: type,
    device: str | None = None,
    compute_virial: bool = False,
) -> None:
    """Launch Ewald real-space energy, forces, and charge gradients kernel (CSR).

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        Atomic positions.
    charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Atomic charges.
    cell : wp.array, shape (1,), dtype=wp.mat33f or wp.mat33d
        Unit cell matrix.
    idx_j : wp.array, shape (M,), dtype=wp.int32
        Target atom indices (CSR data).
    neighbor_ptr : wp.array, shape (N+1,), dtype=wp.int32
        CSR row pointers.
    unit_shifts : wp.array, shape (M,), dtype=wp.vec3i
        Periodic image shifts.
    alpha : wp.array, shape (1,), dtype=wp.float32 or wp.float64
        Ewald splitting parameter.
    pair_energies : wp.array, shape (N,), dtype=wp.float64
        OUTPUT: Per-atom energies.
    atomic_forces : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        OUTPUT: Per-atom forces.
    charge_gradients : wp.array, shape (N,), dtype=wp.float64
        OUTPUT: Per-atom charge gradients.
    wp_dtype : type
        Warp scalar type (wp.float32 or wp.float64).
    device : str, optional
        Warp device.
    compute_virial : bool, optional
        Whether to compute the virial tensor. Default False.
    virial : wp.array, optional
        OUTPUT: Virial tensor. Must be pre-allocated by caller.
    """
    _launch_ewald_real_forward_factory(
        positions,
        charges,
        cell,
        alpha,
        pair_energies,
        wp_dtype,
        device,
        batched=False,
        neighbor_input="list",
        idx_j=idx_j,
        neighbor_ptr=neighbor_ptr,
        unit_shifts=unit_shifts,
        atomic_forces=atomic_forces,
        charge_gradients=charge_gradients,
        virial=virial,
        compute_virial=compute_virial,
    )


def ewald_real_space_energy_forces_charge_grad_matrix(
    positions: wp.array,
    charges: wp.array,
    cell: wp.array,
    neighbor_matrix: wp.array,
    unit_shifts_matrix: wp.array,
    mask_value: int,
    alpha: wp.array,
    pair_energies: wp.array,
    atomic_forces: wp.array,
    charge_gradients: wp.array,
    virial: wp.array,
    wp_dtype: type,
    device: str | None = None,
    compute_virial: bool = False,
) -> None:
    """Launch Ewald real-space energy, forces, and charge gradients kernel (matrix).

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        Atomic positions.
    charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Atomic charges.
    cell : wp.array, shape (1,), dtype=wp.mat33f or wp.mat33d
        Unit cell matrix.
    neighbor_matrix : wp.array2d, shape (N, max_neighbors), dtype=wp.int32
        Neighbor indices.
    unit_shifts_matrix : wp.array2d, shape (N, max_neighbors), dtype=wp.vec3i
        Periodic image shifts.
    mask_value : int
        Value indicating invalid/padded entries.
    alpha : wp.array, shape (1,), dtype=wp.float32 or wp.float64
        Ewald splitting parameter.
    pair_energies : wp.array, shape (N,), dtype=wp.float64
        OUTPUT: Per-atom energies.
    atomic_forces : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        OUTPUT: Per-atom forces.
    charge_gradients : wp.array, shape (N,), dtype=wp.float64
        OUTPUT: Per-atom charge gradients.
    virial : wp.array, shape (1,), dtype=wp.mat33f or wp.mat33d
        OUTPUT: Virial tensor. Only written when compute_virial=True.
        Must be pre-allocated by caller.
    wp_dtype : type
        Warp scalar type (wp.float32 or wp.float64).
    device : str, optional
        Warp device.
    compute_virial : bool, optional
        Whether to compute the virial tensor. Default False.
    """
    _launch_ewald_real_forward_factory(
        positions,
        charges,
        cell,
        alpha,
        pair_energies,
        wp_dtype,
        device,
        batched=False,
        neighbor_input="matrix",
        neighbor_matrix=neighbor_matrix,
        unit_shifts_matrix=unit_shifts_matrix,
        mask_value=mask_value,
        atomic_forces=atomic_forces,
        charge_gradients=charge_gradients,
        virial=virial,
        compute_virial=compute_virial,
    )


# ==================== Batch Real-Space Launchers ====================


def batch_ewald_real_space_energy(
    positions: wp.array,
    charges: wp.array,
    cell: wp.array,
    batch_id: wp.array,
    idx_j: wp.array,
    neighbor_ptr: wp.array,
    unit_shifts: wp.array,
    alpha: wp.array,
    pair_energies: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Launch batched Ewald real-space energy kernel using CSR neighbor list.

    Parameters
    ----------
    positions : wp.array, shape (N_total,), dtype=wp.vec3f or wp.vec3d
        Atomic positions (all systems concatenated).
    charges : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        Atomic charges.
    cell : wp.array, shape (B,), dtype=wp.mat33f or wp.mat33d
        Unit cell matrices for each system.
    batch_id : wp.array, shape (N_total,), dtype=wp.int32
        System index for each atom.
    idx_j : wp.array, shape (M,), dtype=wp.int32
        Target atom indices (CSR data).
    neighbor_ptr : wp.array, shape (N_total+1,), dtype=wp.int32
        CSR row pointers.
    unit_shifts : wp.array, shape (M,), dtype=wp.vec3i
        Periodic image shifts.
    alpha : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        Per-system Ewald splitting parameter.
    pair_energies : wp.array, shape (N_total,), dtype=wp.float64
        OUTPUT: Per-atom energies.
    wp_dtype : type
        Warp scalar type (wp.float32 or wp.float64).
    device : str, optional
        Warp device.
    """
    _launch_ewald_real_forward_factory(
        positions,
        charges,
        cell,
        alpha,
        pair_energies,
        wp_dtype,
        device,
        batched=True,
        neighbor_input="list",
        batch_id=batch_id,
        idx_j=idx_j,
        neighbor_ptr=neighbor_ptr,
        unit_shifts=unit_shifts,
    )


def batch_ewald_real_space_energy_forces(
    positions: wp.array,
    charges: wp.array,
    cell: wp.array,
    batch_id: wp.array,
    idx_j: wp.array,
    neighbor_ptr: wp.array,
    unit_shifts: wp.array,
    alpha: wp.array,
    pair_energies: wp.array,
    atomic_forces: wp.array,
    virial: wp.array,
    wp_dtype: type,
    device: str | None = None,
    compute_virial: bool = False,
) -> None:
    """Launch batched Ewald real-space energy and forces kernel (CSR).

    Parameters
    ----------
    positions : wp.array, shape (N_total,), dtype=wp.vec3f or wp.vec3d
        Atomic positions.
    charges : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        Atomic charges.
    cell : wp.array, shape (B,), dtype=wp.mat33f or wp.mat33d
        Unit cell matrices.
    batch_id : wp.array, shape (N_total,), dtype=wp.int32
        System index for each atom.
    idx_j : wp.array, shape (M,), dtype=wp.int32
        Target atom indices.
    neighbor_ptr : wp.array, shape (N_total+1,), dtype=wp.int32
        CSR row pointers.
    unit_shifts : wp.array, shape (M,), dtype=wp.vec3i
        Periodic image shifts.
    alpha : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        Per-system Ewald splitting parameter.
    pair_energies : wp.array, shape (N_total,), dtype=wp.float64
        OUTPUT: Per-atom energies.
    atomic_forces : wp.array, shape (N_total,), dtype=wp.vec3f or wp.vec3d
        OUTPUT: Per-atom forces.
    wp_dtype : type
        Warp scalar type.
    device : str, optional
        Warp device.
    compute_virial : bool, optional
        Whether to compute the virial tensor. Default False.
    virial : wp.array, optional
        OUTPUT: Virial tensor, shape (B,). If None, a dummy array is created.
    """
    _launch_ewald_real_forward_factory(
        positions,
        charges,
        cell,
        alpha,
        pair_energies,
        wp_dtype,
        device,
        batched=True,
        neighbor_input="list",
        batch_id=batch_id,
        idx_j=idx_j,
        neighbor_ptr=neighbor_ptr,
        unit_shifts=unit_shifts,
        atomic_forces=atomic_forces,
        virial=virial,
        compute_virial=compute_virial,
    )


def batch_ewald_real_space_energy_matrix(
    positions: wp.array,
    charges: wp.array,
    cell: wp.array,
    batch_id: wp.array,
    neighbor_matrix: wp.array,
    unit_shifts_matrix: wp.array,
    mask_value: int,
    alpha: wp.array,
    pair_energies: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Launch batched Ewald real-space energy kernel using neighbor matrix.

    Parameters
    ----------
    positions : wp.array, shape (N_total,), dtype=wp.vec3f or wp.vec3d
        Atomic positions.
    charges : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        Atomic charges.
    cell : wp.array, shape (B,), dtype=wp.mat33f or wp.mat33d
        Unit cell matrices.
    batch_id : wp.array, shape (N_total,), dtype=wp.int32
        System index for each atom.
    neighbor_matrix : wp.array2d, shape (N_total, max_neighbors), dtype=wp.int32
        Neighbor indices.
    unit_shifts_matrix : wp.array2d, shape (N_total, max_neighbors), dtype=wp.vec3i
        Periodic image shifts.
    mask_value : int
        Value indicating invalid entries.
    alpha : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        Per-system Ewald splitting parameter.
    pair_energies : wp.array, shape (N_total,), dtype=wp.float64
        OUTPUT: Per-atom energies.
    wp_dtype : type
        Warp scalar type.
    device : str, optional
        Warp device.
    """
    _launch_ewald_real_forward_factory(
        positions,
        charges,
        cell,
        alpha,
        pair_energies,
        wp_dtype,
        device,
        batched=True,
        neighbor_input="matrix",
        batch_id=batch_id,
        neighbor_matrix=neighbor_matrix,
        unit_shifts_matrix=unit_shifts_matrix,
        mask_value=mask_value,
    )


def batch_ewald_real_space_energy_forces_matrix(
    positions: wp.array,
    charges: wp.array,
    cell: wp.array,
    batch_id: wp.array,
    neighbor_matrix: wp.array,
    unit_shifts_matrix: wp.array,
    mask_value: int,
    alpha: wp.array,
    pair_energies: wp.array,
    atomic_forces: wp.array,
    virial: wp.array,
    wp_dtype: type,
    device: str | None = None,
    compute_virial: bool = False,
) -> None:
    """Launch batched Ewald real-space energy and forces kernel (matrix).

    Parameters
    ----------
    positions : wp.array, shape (N_total,), dtype=wp.vec3f or wp.vec3d
        Atomic positions.
    charges : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        Atomic charges.
    cell : wp.array, shape (B,), dtype=wp.mat33f or wp.mat33d
        Unit cell matrices.
    batch_id : wp.array, shape (N_total,), dtype=wp.int32
        System index for each atom.
    neighbor_matrix : wp.array2d, shape (N_total, max_neighbors), dtype=wp.int32
        Neighbor indices.
    unit_shifts_matrix : wp.array2d, shape (N_total, max_neighbors), dtype=wp.vec3i
        Periodic image shifts.
    mask_value : int
        Value indicating invalid entries.
    alpha : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        Per-system Ewald splitting parameter.
    pair_energies : wp.array, shape (N_total,), dtype=wp.float64
        OUTPUT: Per-atom energies.
    atomic_forces : wp.array, shape (N_total,), dtype=wp.vec3f or wp.vec3d
        OUTPUT: Per-atom forces.
    wp_dtype : type
        Warp scalar type.
    device : str, optional
        Warp device.
    compute_virial : bool, optional
        Whether to compute the virial tensor. Default False.
    virial : wp.array, optional
        OUTPUT: Virial tensor, shape (B,). If None, a dummy array is created.
    """
    _launch_ewald_real_forward_factory(
        positions,
        charges,
        cell,
        alpha,
        pair_energies,
        wp_dtype,
        device,
        batched=True,
        neighbor_input="matrix",
        batch_id=batch_id,
        neighbor_matrix=neighbor_matrix,
        unit_shifts_matrix=unit_shifts_matrix,
        mask_value=mask_value,
        atomic_forces=atomic_forces,
        virial=virial,
        compute_virial=compute_virial,
    )


def batch_ewald_real_space_energy_forces_charge_grad(
    positions: wp.array,
    charges: wp.array,
    cell: wp.array,
    batch_id: wp.array,
    idx_j: wp.array,
    neighbor_ptr: wp.array,
    unit_shifts: wp.array,
    alpha: wp.array,
    pair_energies: wp.array,
    atomic_forces: wp.array,
    charge_gradients: wp.array,
    virial: wp.array,
    wp_dtype: type,
    device: str | None = None,
    compute_virial: bool = False,
) -> None:
    """Launch batched Ewald real-space energy, forces, charge gradients kernel (CSR).

    Parameters
    ----------
    positions : wp.array, shape (N_total,), dtype=wp.vec3f or wp.vec3d
        Atomic positions.
    charges : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        Atomic charges.
    cell : wp.array, shape (B,), dtype=wp.mat33f or wp.mat33d
        Unit cell matrices.
    batch_id : wp.array, shape (N_total,), dtype=wp.int32
        System index for each atom.
    idx_j : wp.array, shape (M,), dtype=wp.int32
        Target atom indices.
    neighbor_ptr : wp.array, shape (N_total+1,), dtype=wp.int32
        CSR row pointers.
    unit_shifts : wp.array, shape (M,), dtype=wp.vec3i
        Periodic image shifts.
    alpha : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        Per-system Ewald splitting parameter.
    pair_energies : wp.array, shape (N_total,), dtype=wp.float64
        OUTPUT: Per-atom energies.
    atomic_forces : wp.array, shape (N_total,), dtype=wp.vec3f or wp.vec3d
        OUTPUT: Per-atom forces.
    charge_gradients : wp.array, shape (N_total,), dtype=wp.float64
        OUTPUT: Per-atom charge gradients.
    wp_dtype : type
        Warp scalar type.
    device : str, optional
        Warp device.
    compute_virial : bool, optional
        Whether to compute the virial tensor. Default False.
    virial : wp.array, optional
        OUTPUT: Virial tensor, shape (B,). If None, a dummy array is created.
    """
    _launch_ewald_real_forward_factory(
        positions,
        charges,
        cell,
        alpha,
        pair_energies,
        wp_dtype,
        device,
        batched=True,
        neighbor_input="list",
        batch_id=batch_id,
        idx_j=idx_j,
        neighbor_ptr=neighbor_ptr,
        unit_shifts=unit_shifts,
        atomic_forces=atomic_forces,
        charge_gradients=charge_gradients,
        virial=virial,
        compute_virial=compute_virial,
    )


def batch_ewald_real_space_energy_forces_charge_grad_matrix(
    positions: wp.array,
    charges: wp.array,
    cell: wp.array,
    batch_id: wp.array,
    neighbor_matrix: wp.array,
    unit_shifts_matrix: wp.array,
    mask_value: int,
    alpha: wp.array,
    pair_energies: wp.array,
    atomic_forces: wp.array,
    charge_gradients: wp.array,
    virial: wp.array,
    wp_dtype: type,
    device: str | None = None,
    compute_virial: bool = False,
) -> None:
    """Launch batched Ewald real-space energy, forces, charge gradients kernel (matrix).

    Parameters
    ----------
    positions : wp.array, shape (N_total,), dtype=wp.vec3f or wp.vec3d
        Atomic positions.
    charges : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        Atomic charges.
    cell : wp.array, shape (B,), dtype=wp.mat33f or wp.mat33d
        Unit cell matrices.
    batch_id : wp.array, shape (N_total,), dtype=wp.int32
        System index for each atom.
    neighbor_matrix : wp.array2d, shape (N_total, max_neighbors), dtype=wp.int32
        Neighbor indices.
    unit_shifts_matrix : wp.array2d, shape (N_total, max_neighbors), dtype=wp.vec3i
        Periodic image shifts.
    mask_value : int
        Value indicating invalid entries.
    alpha : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        Per-system Ewald splitting parameter.
    pair_energies : wp.array, shape (N_total,), dtype=wp.float64
        OUTPUT: Per-atom energies.
    atomic_forces : wp.array, shape (N_total,), dtype=wp.vec3f or wp.vec3d
        OUTPUT: Per-atom forces.
    charge_gradients : wp.array, shape (N_total,), dtype=wp.float64
        OUTPUT: Per-atom charge gradients.
    wp_dtype : type
        Warp scalar type.
    device : str, optional
        Warp device.
    compute_virial : bool, optional
        Whether to compute the virial tensor. Default False.
    virial : wp.array, optional
        OUTPUT: Virial tensor, shape (B,). If None, a dummy array is created.
    """
    _launch_ewald_real_forward_factory(
        positions,
        charges,
        cell,
        alpha,
        pair_energies,
        wp_dtype,
        device,
        batched=True,
        neighbor_input="matrix",
        batch_id=batch_id,
        neighbor_matrix=neighbor_matrix,
        unit_shifts_matrix=unit_shifts_matrix,
        mask_value=mask_value,
        atomic_forces=atomic_forces,
        charge_gradients=charge_gradients,
        virial=virial,
        compute_virial=compute_virial,
    )


# ==================== Reciprocal-Space Launchers ====================


def _get_ewald_recip_component_factory_kernel(
    wp_dtype: type,
    *,
    component: str,
    batched: bool = False,
) -> wp.Kernel:
    """Return an Ewald reciprocal factory kernel without a module import cycle."""
    from nvalchemiops.interactions.electrostatics.ewald_recip_factory import (
        get_ewald_recip_component_kernel,
    )

    return get_ewald_recip_component_kernel(
        wp_dtype, component=component, batched=batched
    )


def ewald_reciprocal_space_fill_structure_factors(
    positions: wp.array,
    charges: wp.array,
    k_vectors: wp.array,
    cell: wp.array,
    alpha: wp.array,
    total_charge: wp.array,
    cos_k_dot_r: wp.array,
    sin_k_dot_r: wp.array,
    real_structure_factors: wp.array,
    imag_structure_factors: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Launch kernel to compute structure factors for reciprocal-space Ewald.

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        Atomic positions.
    charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Atomic charges.
    k_vectors : wp.array, shape (K,), dtype=wp.vec3f or wp.vec3d
        Half-space reciprocal lattice vectors.
    cell : wp.array, shape (1,), dtype=wp.mat33f or wp.mat33d
        Unit cell matrix.
    alpha : wp.array, shape (1,), dtype=wp.float32 or wp.float64
        Ewald splitting parameter.
    total_charge : wp.array, shape (1,), dtype=wp.float64
        OUTPUT: Q_total/V for background correction.
    cos_k_dot_r : wp.array2d, shape (K, N), dtype=wp.float64
        OUTPUT: cos(k.r) for each (k, atom) pair.
    sin_k_dot_r : wp.array2d, shape (K, N), dtype=wp.float64
        OUTPUT: sin(k.r) for each (k, atom) pair.
    real_structure_factors : wp.array, shape (K,), dtype=wp.float64
        OUTPUT: Real part of weighted structure factors.
    imag_structure_factors : wp.array, shape (K,), dtype=wp.float64
        OUTPUT: Imaginary part of weighted structure factors.
    wp_dtype : type
        Warp scalar type.
    device : str, optional
        Warp device.
    """
    num_k = k_vectors.shape[0]
    if device is None:
        device = str(positions.device)

    wp.launch(
        _get_ewald_recip_component_factory_kernel(wp_dtype, component="fill"),
        dim=num_k,
        inputs=[
            positions,
            charges,
            k_vectors,
            cell,
            alpha,
            total_charge,
            cos_k_dot_r,
            sin_k_dot_r,
            real_structure_factors,
            imag_structure_factors,
        ],
        device=device,
    )


def ewald_reciprocal_space_compute_energy(
    charges: wp.array,
    cos_k_dot_r: wp.array,
    sin_k_dot_r: wp.array,
    real_structure_factors: wp.array,
    imag_structure_factors: wp.array,
    reciprocal_energies: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Launch kernel to compute per-atom reciprocal-space energies.

    Parameters
    ----------
    charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Atomic charges.
    cos_k_dot_r : wp.array2d, shape (K, N), dtype=wp.float64
        cos(k.r) from structure factor computation.
    sin_k_dot_r : wp.array2d, shape (K, N), dtype=wp.float64
        sin(k.r) from structure factor computation.
    real_structure_factors : wp.array, shape (K,), dtype=wp.float64
        Real structure factors.
    imag_structure_factors : wp.array, shape (K,), dtype=wp.float64
        Imaginary structure factors.
    reciprocal_energies : wp.array, shape (N,), dtype=wp.float64
        OUTPUT: Per-atom energies.
    wp_dtype : type
        Warp scalar type.
    device : str, optional
        Warp device.
    """
    num_atoms = charges.shape[0]
    if device is None:
        device = str(charges.device)

    wp.launch(
        _get_ewald_recip_component_factory_kernel(wp_dtype, component="compute_energy"),
        dim=num_atoms,
        inputs=[
            charges,
            cos_k_dot_r,
            sin_k_dot_r,
            real_structure_factors,
            imag_structure_factors,
            reciprocal_energies,
        ],
        device=device,
    )


def ewald_subtract_self_energy(
    charges: wp.array,
    alpha: wp.array,
    total_charge: wp.array,
    energy_in: wp.array,
    energy_out: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Launch kernel to apply self-energy and background corrections.

    Parameters
    ----------
    charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Atomic charges.
    alpha : wp.array, shape (1,), dtype=wp.float32 or wp.float64
        Ewald splitting parameter.
    total_charge : wp.array, shape (1,), dtype=wp.float64
        Q_total/V from structure factor computation.
    energy_in : wp.array, shape (N,), dtype=wp.float64
        Raw reciprocal-space energies.
    energy_out : wp.array, shape (N,), dtype=wp.float64
        OUTPUT: Corrected energies.
    wp_dtype : type
        Warp scalar type.
    device : str, optional
        Warp device.
    """
    num_atoms = charges.shape[0]
    if device is None:
        device = str(charges.device)

    wp.launch(
        _get_ewald_recip_component_factory_kernel(wp_dtype, component="subtract_self"),
        dim=num_atoms,
        inputs=[charges, alpha, total_charge, energy_in, energy_out],
        device=device,
    )


def ewald_energy_corrections(
    raw_energies: wp.array,
    charges: wp.array,
    volume: wp.array,
    alpha: wp.array,
    total_charge: wp.array,
    corrected_energies: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Launch single-system differentiable Ewald reciprocal corrections."""
    num_atoms = raw_energies.shape[0]
    if device is None:
        device = str(raw_energies.device)

    wp.launch(
        _get_ewald_recip_component_factory_kernel(wp_dtype, component="corrections"),
        dim=num_atoms,
        inputs=[raw_energies, charges, volume, alpha, total_charge],
        outputs=[corrected_energies],
        device=device,
    )


def ewald_energy_corrections_backward(
    grad_E: wp.array,
    raw_energies: wp.array,
    charges: wp.array,
    volume: wp.array,
    alpha: wp.array,
    total_charge: wp.array,
    grad_raw: wp.array,
    grad_charges: wp.array,
    grad_volume: wp.array,
    grad_alpha: wp.array,
    grad_total_charge: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Launch single-system Ewald reciprocal correction backward."""
    num_atoms = raw_energies.shape[0]
    if device is None:
        device = str(raw_energies.device)

    wp.launch(
        _get_ewald_recip_component_factory_kernel(
            wp_dtype, component="corrections_backward"
        ),
        dim=num_atoms,
        inputs=[grad_E, raw_energies, charges, volume, alpha, total_charge],
        outputs=[grad_raw, grad_charges, grad_volume, grad_alpha, grad_total_charge],
        device=device,
    )


def ewald_energy_corrections_double_backward(
    h_raw: wp.array,
    h_chg: wp.array,
    h_vol: wp.array,
    h_alpha: wp.array,
    h_qtot: wp.array,
    grad_E: wp.array,
    raw_energies: wp.array,
    charges: wp.array,
    volume: wp.array,
    alpha: wp.array,
    total_charge: wp.array,
    grad_grad_E: wp.array,
    grad_raw: wp.array,
    grad_charges: wp.array,
    grad_volume: wp.array,
    grad_alpha: wp.array,
    grad_total_charge: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Launch single-system Ewald reciprocal correction double-backward."""
    num_atoms = raw_energies.shape[0]
    if device is None:
        device = str(raw_energies.device)

    wp.launch(
        _get_ewald_recip_component_factory_kernel(
            wp_dtype, component="corrections_double_backward"
        ),
        dim=num_atoms,
        inputs=[
            h_raw,
            h_chg,
            h_vol,
            h_alpha,
            h_qtot,
            grad_E,
            raw_energies,
            charges,
            volume,
            alpha,
            total_charge,
        ],
        outputs=[
            grad_grad_E,
            grad_raw,
            grad_charges,
            grad_volume,
            grad_alpha,
            grad_total_charge,
        ],
        device=device,
    )


def ewald_reciprocal_space_energy_forces(
    charges: wp.array,
    k_vectors: wp.array,
    cos_k_dot_r: wp.array,
    sin_k_dot_r: wp.array,
    real_structure_factors: wp.array,
    imag_structure_factors: wp.array,
    reciprocal_energies: wp.array,
    atomic_forces: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Launch kernel to compute reciprocal-space energies and forces.

    Parameters
    ----------
    charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Atomic charges.
    k_vectors : wp.array, shape (K,), dtype=wp.vec3f or wp.vec3d
        Reciprocal lattice vectors.
    cos_k_dot_r : wp.array2d, shape (K, N), dtype=wp.float64
        cos(k.r) from structure factor computation.
    sin_k_dot_r : wp.array2d, shape (K, N), dtype=wp.float64
        sin(k.r) from structure factor computation.
    real_structure_factors : wp.array, shape (K,), dtype=wp.float64
        Real structure factors.
    imag_structure_factors : wp.array, shape (K,), dtype=wp.float64
        Imaginary structure factors.
    reciprocal_energies : wp.array, shape (N,), dtype=wp.float64
        OUTPUT: Per-atom energies.
    atomic_forces : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        OUTPUT: Per-atom forces.
    wp_dtype : type
        Warp scalar type.
    device : str, optional
        Warp device.
    """
    num_atoms = charges.shape[0]
    if device is None:
        device = str(charges.device)

    wp.launch(
        _get_ewald_recip_component_factory_kernel(
            wp_dtype, component="compute_energy_forces"
        ),
        dim=num_atoms,
        inputs=[
            charges,
            k_vectors,
            cos_k_dot_r,
            sin_k_dot_r,
            real_structure_factors,
            imag_structure_factors,
            reciprocal_energies,
            atomic_forces,
        ],
        device=device,
    )


def ewald_reciprocal_space_energy_forces_charge_grad(
    charges: wp.array,
    k_vectors: wp.array,
    cos_k_dot_r: wp.array,
    sin_k_dot_r: wp.array,
    real_structure_factors: wp.array,
    imag_structure_factors: wp.array,
    reciprocal_energies: wp.array,
    atomic_forces: wp.array,
    charge_gradients: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Launch kernel to compute reciprocal-space energies, forces, and charge gradients.

    Parameters
    ----------
    charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Atomic charges.
    k_vectors : wp.array, shape (K,), dtype=wp.vec3f or wp.vec3d
        Reciprocal lattice vectors.
    cos_k_dot_r : wp.array2d, shape (K, N), dtype=wp.float64
        cos(k.r) from structure factor computation.
    sin_k_dot_r : wp.array2d, shape (K, N), dtype=wp.float64
        sin(k.r) from structure factor computation.
    real_structure_factors : wp.array, shape (K,), dtype=wp.float64
        Real structure factors.
    imag_structure_factors : wp.array, shape (K,), dtype=wp.float64
        Imaginary structure factors.
    reciprocal_energies : wp.array, shape (N,), dtype=wp.float64
        OUTPUT: Per-atom energies.
    atomic_forces : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        OUTPUT: Per-atom forces.
    charge_gradients : wp.array, shape (N,), dtype=wp.float64
        OUTPUT: Per-atom charge gradients.
    wp_dtype : type
        Warp scalar type.
    device : str, optional
        Warp device.
    """
    num_atoms = charges.shape[0]
    if device is None:
        device = str(charges.device)

    wp.launch(
        _get_ewald_recip_component_factory_kernel(
            wp_dtype, component="compute_energy_forces_charge_grad"
        ),
        dim=num_atoms,
        inputs=[
            charges,
            k_vectors,
            cos_k_dot_r,
            sin_k_dot_r,
            real_structure_factors,
            imag_structure_factors,
            reciprocal_energies,
            atomic_forces,
            charge_gradients,
        ],
        device=device,
    )


# ==================== Batch Reciprocal-Space Launchers ====================


def batch_ewald_reciprocal_space_fill_structure_factors(
    positions: wp.array,
    charges: wp.array,
    k_vectors: wp.array,
    cell: wp.array,
    alpha: wp.array,
    atom_start: wp.array,
    atom_end: wp.array,
    total_charges: wp.array,
    cos_k_dot_r: wp.array,
    sin_k_dot_r: wp.array,
    real_structure_factors: wp.array,
    imag_structure_factors: wp.array,
    num_k: int,
    num_systems: int,
    max_blocks_per_system: int,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Launch batched kernel to compute structure factors for reciprocal-space Ewald.

    Parameters
    ----------
    positions : wp.array, shape (N_total,), dtype=wp.vec3f or wp.vec3d
        Atomic positions.
    charges : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        Atomic charges.
    k_vectors : wp.array2d, shape (B, K), dtype=wp.vec3f or wp.vec3d
        Per-system reciprocal lattice vectors.
    cell : wp.array, shape (B,), dtype=wp.mat33f or wp.mat33d
        Per-system unit cell matrices.
    alpha : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        Per-system Ewald splitting parameter.
    atom_start : wp.array, shape (B,), dtype=wp.int32
        First atom index for each system.
    atom_end : wp.array, shape (B,), dtype=wp.int32
        Last atom index (exclusive) for each system.
    total_charges : wp.array, shape (B,), dtype=wp.float64
        OUTPUT: Per-system Q_total/V.
    cos_k_dot_r : wp.array2d, shape (K, N_total), dtype=wp.float64
        OUTPUT: cos(k.r) for each (k, atom) pair.
    sin_k_dot_r : wp.array2d, shape (K, N_total), dtype=wp.float64
        OUTPUT: sin(k.r) for each (k, atom) pair.
    real_structure_factors : wp.array2d, shape (B, K), dtype=wp.float64
        OUTPUT: Per-system real structure factors.
    imag_structure_factors : wp.array2d, shape (B, K), dtype=wp.float64
        OUTPUT: Per-system imaginary structure factors.
    num_k : int
        Number of k-vectors per system.
    num_systems : int
        Number of systems in the batch.
    max_blocks_per_system : int
        Maximum atom blocks per system.
    wp_dtype : type
        Warp scalar type.
    device : str, optional
        Warp device.
    """
    if device is None:
        device = str(positions.device)

    wp.launch(
        _get_ewald_recip_component_factory_kernel(
            wp_dtype, component="fill", batched=True
        ),
        dim=(num_k, num_systems, max_blocks_per_system),
        inputs=[
            positions,
            charges,
            k_vectors,
            cell,
            alpha,
            atom_start,
            atom_end,
            total_charges,
            cos_k_dot_r,
            sin_k_dot_r,
            real_structure_factors,
            imag_structure_factors,
        ],
        device=device,
    )


def batch_ewald_reciprocal_space_compute_energy(
    charges: wp.array,
    batch_id: wp.array,
    cos_k_dot_r: wp.array,
    sin_k_dot_r: wp.array,
    real_structure_factors: wp.array,
    imag_structure_factors: wp.array,
    reciprocal_energies: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Launch batched kernel to compute per-atom reciprocal-space energies.

    Parameters
    ----------
    charges : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        Atomic charges.
    batch_id : wp.array, shape (N_total,), dtype=wp.int32
        System index for each atom.
    cos_k_dot_r : wp.array2d, shape (K, N_total), dtype=wp.float64
        cos(k.r) from structure factor computation.
    sin_k_dot_r : wp.array2d, shape (K, N_total), dtype=wp.float64
        sin(k.r) from structure factor computation.
    real_structure_factors : wp.array2d, shape (B, K), dtype=wp.float64
        Per-system real structure factors.
    imag_structure_factors : wp.array2d, shape (B, K), dtype=wp.float64
        Per-system imaginary structure factors.
    reciprocal_energies : wp.array, shape (N_total,), dtype=wp.float64
        OUTPUT: Per-atom energies.
    wp_dtype : type
        Warp scalar type.
    device : str, optional
        Warp device.
    """
    num_atoms = charges.shape[0]
    if device is None:
        device = str(charges.device)

    wp.launch(
        _get_ewald_recip_component_factory_kernel(
            wp_dtype, component="compute_energy", batched=True
        ),
        dim=num_atoms,
        inputs=[
            charges,
            batch_id,
            cos_k_dot_r,
            sin_k_dot_r,
            real_structure_factors,
            imag_structure_factors,
            reciprocal_energies,
        ],
        device=device,
    )


def batch_ewald_subtract_self_energy(
    charges: wp.array,
    batch_idx: wp.array,
    alpha: wp.array,
    total_charges: wp.array,
    energy_in: wp.array,
    energy_out: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Launch batched kernel to apply self-energy and background corrections.

    Parameters
    ----------
    charges : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        Atomic charges.
    batch_idx : wp.array, shape (N_total,), dtype=wp.int32
        System index for each atom.
    alpha : wp.array, shape (B,), dtype=wp.float32 or wp.float64
        Per-system Ewald splitting parameter.
    total_charges : wp.array, shape (B,), dtype=wp.float64
        Per-system Q_total/V.
    energy_in : wp.array, shape (N_total,), dtype=wp.float64
        Raw reciprocal-space energies.
    energy_out : wp.array, shape (N_total,), dtype=wp.float64
        OUTPUT: Corrected energies.
    wp_dtype : type
        Warp scalar type.
    device : str, optional
        Warp device.
    """
    num_atoms = charges.shape[0]
    if device is None:
        device = str(charges.device)

    wp.launch(
        _get_ewald_recip_component_factory_kernel(
            wp_dtype, component="subtract_self", batched=True
        ),
        dim=num_atoms,
        inputs=[charges, batch_idx, alpha, total_charges, energy_in, energy_out],
        device=device,
    )


def batch_ewald_energy_corrections(
    raw_energies: wp.array,
    charges: wp.array,
    batch_idx: wp.array,
    volumes: wp.array,
    alpha: wp.array,
    total_charges: wp.array,
    corrected_energies: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Launch batched differentiable Ewald reciprocal corrections."""
    num_atoms = raw_energies.shape[0]
    if device is None:
        device = str(raw_energies.device)

    wp.launch(
        _get_ewald_recip_component_factory_kernel(
            wp_dtype, component="corrections", batched=True
        ),
        dim=num_atoms,
        inputs=[raw_energies, charges, batch_idx, volumes, alpha, total_charges],
        outputs=[corrected_energies],
        device=device,
    )


def batch_ewald_energy_corrections_backward(
    grad_E: wp.array,
    raw_energies: wp.array,
    charges: wp.array,
    batch_idx: wp.array,
    volumes: wp.array,
    alpha: wp.array,
    total_charges: wp.array,
    grad_raw: wp.array,
    grad_charges: wp.array,
    grad_volumes: wp.array,
    grad_alpha: wp.array,
    grad_total_charges: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Launch batched Ewald reciprocal correction backward."""
    num_atoms = raw_energies.shape[0]
    if device is None:
        device = str(raw_energies.device)

    wp.launch(
        _get_ewald_recip_component_factory_kernel(
            wp_dtype, component="corrections_backward", batched=True
        ),
        dim=num_atoms,
        inputs=[
            grad_E,
            raw_energies,
            charges,
            batch_idx,
            volumes,
            alpha,
            total_charges,
        ],
        outputs=[grad_raw, grad_charges, grad_volumes, grad_alpha, grad_total_charges],
        device=device,
    )


def batch_ewald_energy_corrections_double_backward(
    h_raw: wp.array,
    h_chg: wp.array,
    h_vol: wp.array,
    h_alpha: wp.array,
    h_qtot: wp.array,
    grad_E: wp.array,
    raw_energies: wp.array,
    charges: wp.array,
    batch_idx: wp.array,
    volumes: wp.array,
    alpha: wp.array,
    total_charges: wp.array,
    grad_grad_E: wp.array,
    grad_raw: wp.array,
    grad_charges: wp.array,
    grad_volumes: wp.array,
    grad_alpha: wp.array,
    grad_total_charges: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Launch batched Ewald reciprocal correction double-backward."""
    num_atoms = raw_energies.shape[0]
    if device is None:
        device = str(raw_energies.device)

    wp.launch(
        _get_ewald_recip_component_factory_kernel(
            wp_dtype, component="corrections_double_backward", batched=True
        ),
        dim=num_atoms,
        inputs=[
            h_raw,
            h_chg,
            h_vol,
            h_alpha,
            h_qtot,
            grad_E,
            raw_energies,
            charges,
            batch_idx,
            volumes,
            alpha,
            total_charges,
        ],
        outputs=[
            grad_grad_E,
            grad_raw,
            grad_charges,
            grad_volumes,
            grad_alpha,
            grad_total_charges,
        ],
        device=device,
    )


def batch_ewald_reciprocal_space_energy_forces(
    charges: wp.array,
    batch_id: wp.array,
    k_vectors: wp.array,
    cos_k_dot_r: wp.array,
    sin_k_dot_r: wp.array,
    real_structure_factors: wp.array,
    imag_structure_factors: wp.array,
    reciprocal_energies: wp.array,
    atomic_forces: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Launch batched kernel to compute reciprocal-space energies and forces.

    Parameters
    ----------
    charges : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        Atomic charges.
    batch_id : wp.array, shape (N_total,), dtype=wp.int32
        System index for each atom.
    k_vectors : wp.array2d, shape (B, K), dtype=wp.vec3f or wp.vec3d
        Per-system reciprocal lattice vectors.
    cos_k_dot_r : wp.array2d, shape (K, N_total), dtype=wp.float64
        cos(k.r) from structure factor computation.
    sin_k_dot_r : wp.array2d, shape (K, N_total), dtype=wp.float64
        sin(k.r) from structure factor computation.
    real_structure_factors : wp.array2d, shape (B, K), dtype=wp.float64
        Per-system real structure factors.
    imag_structure_factors : wp.array2d, shape (B, K), dtype=wp.float64
        Per-system imaginary structure factors.
    reciprocal_energies : wp.array, shape (N_total,), dtype=wp.float64
        OUTPUT: Per-atom energies.
    atomic_forces : wp.array, shape (N_total,), dtype=wp.vec3f or wp.vec3d
        OUTPUT: Per-atom forces.
    wp_dtype : type
        Warp scalar type.
    device : str, optional
        Warp device.
    """
    num_atoms = charges.shape[0]
    if device is None:
        device = str(charges.device)

    wp.launch(
        _get_ewald_recip_component_factory_kernel(
            wp_dtype, component="compute_energy_forces", batched=True
        ),
        dim=num_atoms,
        inputs=[
            charges,
            batch_id,
            k_vectors,
            cos_k_dot_r,
            sin_k_dot_r,
            real_structure_factors,
            imag_structure_factors,
            reciprocal_energies,
            atomic_forces,
        ],
        device=device,
    )


def batch_ewald_reciprocal_space_energy_forces_charge_grad(
    charges: wp.array,
    batch_id: wp.array,
    k_vectors: wp.array,
    cos_k_dot_r: wp.array,
    sin_k_dot_r: wp.array,
    real_structure_factors: wp.array,
    imag_structure_factors: wp.array,
    reciprocal_energies: wp.array,
    atomic_forces: wp.array,
    charge_gradients: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Launch batched kernel for reciprocal-space energies, forces, charge gradients.

    Parameters
    ----------
    charges : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        Atomic charges.
    batch_id : wp.array, shape (N_total,), dtype=wp.int32
        System index for each atom.
    k_vectors : wp.array2d, shape (B, K), dtype=wp.vec3f or wp.vec3d
        Per-system reciprocal lattice vectors.
    cos_k_dot_r : wp.array2d, shape (K, N_total), dtype=wp.float64
        cos(k.r) from structure factor computation.
    sin_k_dot_r : wp.array2d, shape (K, N_total), dtype=wp.float64
        sin(k.r) from structure factor computation.
    real_structure_factors : wp.array2d, shape (B, K), dtype=wp.float64
        Per-system real structure factors.
    imag_structure_factors : wp.array2d, shape (B, K), dtype=wp.float64
        Per-system imaginary structure factors.
    reciprocal_energies : wp.array, shape (N_total,), dtype=wp.float64
        OUTPUT: Per-atom energies.
    atomic_forces : wp.array, shape (N_total,), dtype=wp.vec3f or wp.vec3d
        OUTPUT: Per-atom forces.
    charge_gradients : wp.array, shape (N_total,), dtype=wp.float64
        OUTPUT: Per-atom charge gradients.
    wp_dtype : type
        Warp scalar type.
    device : str, optional
        Warp device.
    """
    num_atoms = charges.shape[0]
    if device is None:
        device = str(charges.device)

    wp.launch(
        _get_ewald_recip_component_factory_kernel(
            wp_dtype, component="compute_energy_forces_charge_grad", batched=True
        ),
        dim=num_atoms,
        inputs=[
            charges,
            batch_id,
            k_vectors,
            cos_k_dot_r,
            sin_k_dot_r,
            real_structure_factors,
            imag_structure_factors,
            reciprocal_energies,
            atomic_forces,
            charge_gradients,
        ],
        device=device,
    )


###########################################################################################
########################### Kernel Overloads (float32/float64) ############################
###########################################################################################
