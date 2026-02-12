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
Coulomb Electrostatic Interactions - Warp Kernel Implementation
===============================================================

This module implements direct Coulomb energy and force calculations for electrostatic
interactions using Warp GPU/CPU kernels. Includes both undamped (direct) and
damped (Ewald/PME real-space) variants.

Architecture
------------
This module provides two layers:

1. **Warp Kernels** (pure Warp, framework-agnostic):
   - ``_coulomb_energy_kernel``, ``_coulomb_energy_forces_kernel``
   - ``_coulomb_energy_matrix_kernel``, ``_coulomb_energy_forces_matrix_kernel``
   - Batch variants of all above

2. **Warp Launchers** (framework-agnostic API):
   - ``coulomb_energy()``, ``coulomb_energy_forces()``
   - ``coulomb_energy_matrix()``, ``coulomb_energy_forces_matrix()``
   - Batch variants of all above

For PyTorch integration, see ``nvalchemiops.torch.interactions.electrostatics.coulomb``.

Mathematical Formulation
------------------------

1. Coulomb Energy (Undamped):
   The energy between two charges :math:`q_i` and :math:`q_j` separated by distance r is:

   .. math::

       E_{ij} = \\frac{q_i q_j}{r}

2. Coulomb Force (Undamped):

   .. math::

       F_{ij} = \\frac{q_i q_j}{r^2} \\hat{r}

   where :math:`\\hat{r} = r_{ij} / |r_{ij}|` is the unit vector from j to i.

3. Damped Coulomb (Ewald/PME Real-Space):
   For Ewald splitting with parameter :math:`\\alpha`:

   Energy:

   .. math::

       E_{ij} = q_i q_j \\frac{\\text{erfc}(\\alpha r)}{r}

   Force:

   .. math::

       F_{ij} = q_i q_j \\left[\\frac{\\text{erfc}(\\alpha r)}{r^2} + \\frac{2\\alpha}{\\sqrt{\\pi}} \\frac{\\exp(-\\alpha^2 r^2)}{r}\\right] \\hat{r}

   where erfc(x) is the complementary error function.

.. note::
   This implementation assumes a **half neighbor list** where each pair (i, j)
   appears only once (i.e., only for i < j or only for i > j). If using a
   symmetric neighbor list where both (i, j) and (j, i) appear, the total
   energy will be doubled.

Neighbor Formats
----------------

This module supports two neighbor formats:

1. **Neighbor List (CSR format)**: ``idx_j`` is shape (num_pairs,) containing
   destination indices, ``neighbor_ptr`` is shape (N+1,) with CSR row pointers.

2. **Neighbor Matrix**: ``neighbor_matrix`` is shape (N, max_neighbors) where
   each row contains neighbor indices for that atom.

References
----------
- Allen & Tildesley, "Computer Simulation of Liquids" (1987)
- Essmann et al., J. Chem. Phys. 103, 8577 (1995) - PME paper
"""

from __future__ import annotations

import math

import warp as wp

from nvalchemiops.math import wp_erfc

__all__ = [
    # Warp launchers (framework-agnostic public API)
    "coulomb_energy",
    "coulomb_energy_forces",
    "coulomb_energy_matrix",
    "coulomb_energy_forces_matrix",
    "batch_coulomb_energy",
    "batch_coulomb_energy_forces",
    "batch_coulomb_energy_matrix",
    "batch_coulomb_energy_forces_matrix",
]

# Mathematical constants
PI = math.pi
SQRT_PI = math.sqrt(PI)
TWO_OVER_SQRT_PI = 2.0 / SQRT_PI


# ==============================================================================
# Warp Kernels - Energy Only (Neighbor List Format)
# ==============================================================================


@wp.kernel
def _coulomb_energy_kernel(
    positions: wp.array(dtype=wp.vec3d),
    charges: wp.array(dtype=wp.float64),
    cell: wp.array(dtype=wp.mat33d),
    idx_j: wp.array(dtype=wp.int32),
    neighbor_ptr: wp.array(dtype=wp.int32),
    unit_shifts: wp.array(dtype=wp.vec3i),
    cutoff: wp.float64,
    alpha: wp.float64,
    energies: wp.array(dtype=wp.float64),
):
    """Compute Coulomb energies (damped or undamped based on alpha).

    Formula (undamped, alpha=0):

    .. math::

        E_{ij} = \\frac{1}{2} \\frac{q_i q_j}{r}

    Formula (damped, alpha>0):

    .. math::

        E_{ij} = \\frac{1}{2} q_i q_j \\frac{\\text{erfc}(\\alpha r)}{r}

    Launch Grid: dim = [num_atoms]
    Each thread processes one atom and loops over its neighbors using CSR format.

    Note: Uses atomic_add to accumulate to per-atom energies.
    """
    atom_i = wp.tid()
    num_atoms = positions.shape[0]

    if atom_i >= num_atoms:
        return

    ri = positions[atom_i]
    qi = charges[atom_i]
    cell_t = wp.transpose(cell[0])

    energy_acc = wp.float64(0.0)

    j_start = neighbor_ptr[atom_i]
    j_end = neighbor_ptr[atom_i + 1]

    for edge_idx in range(j_start, j_end):
        j = idx_j[edge_idx]

        rj = positions[j]
        qj = charges[j]

        shift_vec = cell_t * type(ri)(unit_shifts[edge_idx])
        r_ij = ri - rj - shift_vec
        r = wp.length(r_ij)

        if r >= cutoff or r < wp.float64(1e-10):
            continue

        prefactor = wp.float64(0.5) * qi * qj

        if alpha > wp.float64(0.0):
            # Damped: E = q_i * q_j * erfc(alpha*r) / r
            alpha_r = alpha * r
            erfc_term = wp_erfc(alpha_r)
            energy_acc += prefactor * erfc_term / r
        else:
            # Undamped: E = q_i * q_j / r
            energy_acc += prefactor / r

    wp.atomic_add(energies, atom_i, energy_acc)


@wp.kernel
def _coulomb_energy_forces_kernel(
    positions: wp.array(dtype=wp.vec3d),
    charges: wp.array(dtype=wp.float64),
    cell: wp.array(dtype=wp.mat33d),
    idx_j: wp.array(dtype=wp.int32),
    neighbor_ptr: wp.array(dtype=wp.int32),
    unit_shifts: wp.array(dtype=wp.vec3i),
    cutoff: wp.float64,
    alpha: wp.float64,
    energies: wp.array(dtype=wp.float64),
    forces: wp.array(dtype=wp.vec3d),
):
    """Compute Coulomb energies and forces (damped or undamped based on alpha).

    Launch Grid: dim = [num_atoms]
    Each thread processes one atom and loops over its neighbors using CSR format.

    Note: Uses atomic_add to accumulate to per-atom arrays.
    """
    atom_i = wp.tid()
    num_atoms = positions.shape[0]

    if atom_i >= num_atoms:
        return

    ri = positions[atom_i]
    qi = charges[atom_i]
    cell_t = wp.transpose(cell[0])

    energy_acc = wp.float64(0.0)
    force_acc = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))

    j_start = neighbor_ptr[atom_i]
    j_end = neighbor_ptr[atom_i + 1]

    for edge_idx in range(j_start, j_end):
        j = idx_j[edge_idx]

        rj = positions[j]
        qj = charges[j]

        shift_vec = cell_t * type(ri)(unit_shifts[edge_idx])
        r_ij = ri - rj - shift_vec
        r = wp.length(r_ij)

        if r >= cutoff or r < wp.float64(1e-10):
            continue

        prefactor = wp.float64(0.5) * qi * qj

        if alpha > wp.float64(0.0):
            # Damped
            alpha_r = alpha * r
            alpha_r_sq = alpha_r * alpha_r
            erfc_term = wp_erfc(alpha_r)
            exp_term = wp.exp(-alpha_r_sq)

            # Energy: E = q_i * q_j * erfc(alphar) / r
            energy_acc += prefactor * erfc_term / r

            # Force: F = q_i * q_j *
            # [erfc(alpha*r)/r^3 + 2*alpha/sqrt(pi) *
            # exp(-alpha^2*r^2)/r^2] * r_ij
            two_over_sqrt_pi = wp.float64(1.1283791670955126)
            force_mag_over_r = erfc_term / (
                r * r * r
            ) + two_over_sqrt_pi * alpha * exp_term / (r * r)
            force_ij = prefactor * force_mag_over_r * r_ij
        else:
            # Undamped: E = q_i * q_j / r, F = q_i * q_j / r^3 * r_ij
            energy_acc += prefactor / r
            force_mag_over_r = prefactor / (r * r * r)
            force_ij = force_mag_over_r * r_ij

        # Accumulate force on i, apply Newton's 3rd law to j
        force_acc += force_ij
        wp.atomic_add(forces, j, -force_ij)

    wp.atomic_add(energies, atom_i, energy_acc)
    wp.atomic_add(forces, atom_i, force_acc)


# ==============================================================================
# Warp Kernels - Neighbor Matrix Format
# ==============================================================================


@wp.kernel
def _coulomb_energy_matrix_kernel(
    positions: wp.array(dtype=wp.vec3d),
    charges: wp.array(dtype=wp.float64),
    cell: wp.array(dtype=wp.mat33d),
    neighbor_matrix: wp.array2d(dtype=wp.int32),
    neighbor_matrix_shifts: wp.array2d(dtype=wp.vec3i),
    cutoff: wp.float64,
    alpha: wp.float64,
    fill_value: wp.int32,
    atomic_energies: wp.array(dtype=wp.float64),
):
    """Compute Coulomb energies using neighbor matrix format.

    Launch Grid: dim = [num_atoms]
    Each thread processes one atom and loops over its neighbors.
    """
    atom_idx = wp.tid()
    num_atoms = positions.shape[0]
    max_neighbors = neighbor_matrix.shape[1]

    if atom_idx >= num_atoms:
        return

    ri = positions[atom_idx]
    qi = charges[atom_idx]
    cell_t = wp.transpose(cell[0])

    energy_acc = wp.float64(0.0)

    for neighbor_slot in range(max_neighbors):
        j = neighbor_matrix[atom_idx, neighbor_slot]
        if j >= fill_value or j >= num_atoms:
            continue

        rj = positions[j]
        qj = charges[j]

        shift = neighbor_matrix_shifts[atom_idx, neighbor_slot]
        shift_vec = cell_t * type(ri)(shift)
        r_ij = ri - rj - shift_vec
        r = wp.length(r_ij)

        if r >= cutoff or r < wp.float64(1e-10):
            continue

        prefactor = qi * qj

        if alpha > wp.float64(0.0):
            alpha_r = alpha * r
            erfc_term = wp_erfc(alpha_r)
            energy_acc += prefactor * erfc_term / r
        else:
            energy_acc += prefactor / r

    wp.atomic_add(atomic_energies, atom_idx, energy_acc)


@wp.kernel
def _coulomb_energy_forces_matrix_kernel(
    positions: wp.array(dtype=wp.vec3d),
    charges: wp.array(dtype=wp.float64),
    cell: wp.array(dtype=wp.mat33d),
    neighbor_matrix: wp.array2d(dtype=wp.int32),
    neighbor_matrix_shifts: wp.array2d(dtype=wp.vec3i),
    cutoff: wp.float64,
    alpha: wp.float64,
    fill_value: wp.int32,
    atomic_energies: wp.array(dtype=wp.float64),
    atomic_forces: wp.array(dtype=wp.vec3d),
):
    """Compute Coulomb energies and forces using neighbor matrix format.

    Launch Grid: dim = [num_atoms]
    Each thread processes one atom and loops over its neighbors.
    """
    atom_idx = wp.tid()
    num_atoms = positions.shape[0]
    max_neighbors = neighbor_matrix.shape[1]

    if atom_idx >= num_atoms:
        return

    ri = positions[atom_idx]
    qi = charges[atom_idx]
    cell_t = wp.transpose(cell[0])

    energy_acc = wp.float64(0.0)
    force_acc = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))

    for neighbor_slot in range(max_neighbors):
        j = neighbor_matrix[atom_idx, neighbor_slot]
        if j >= fill_value or j >= num_atoms:
            continue

        rj = positions[j]
        qj = charges[j]

        shift = neighbor_matrix_shifts[atom_idx, neighbor_slot]
        shift_vec = cell_t * type(ri)(shift)
        r_ij = ri - rj - shift_vec
        r = wp.length(r_ij)

        if r >= cutoff or r < wp.float64(1e-10):
            continue

        prefactor = wp.float64(0.5) * qi * qj

        if alpha > wp.float64(0.0):
            alpha_r = alpha * r
            alpha_r_sq = alpha_r * alpha_r
            erfc_term = wp_erfc(alpha_r)
            exp_term = wp.exp(-alpha_r_sq)

            energy_acc += prefactor * erfc_term / r
            two_over_sqrt_pi = wp.float64(1.1283791670955126)
            force_mag_over_r = erfc_term / (
                r * r * r
            ) + two_over_sqrt_pi * alpha * exp_term / (r * r)
            force_ij = prefactor * force_mag_over_r * r_ij
        else:
            energy_acc += prefactor / r
            force_mag_over_r = prefactor / (r * r * r)
            force_ij = force_mag_over_r * r_ij

        force_acc += force_ij
        wp.atomic_add(atomic_forces, j, -force_ij)

    wp.atomic_add(atomic_energies, atom_idx, energy_acc)
    wp.atomic_add(atomic_forces, atom_idx, force_acc)


# ==============================================================================
# Warp Kernels - Batch Versions (Neighbor List Format)
# ==============================================================================


@wp.kernel
def _batch_coulomb_energy_kernel(
    positions: wp.array(dtype=wp.vec3d),
    charges: wp.array(dtype=wp.float64),
    cell: wp.array(dtype=wp.mat33d),
    idx_j: wp.array(dtype=wp.int32),
    neighbor_ptr: wp.array(dtype=wp.int32),
    unit_shifts: wp.array(dtype=wp.vec3i),
    batch_idx: wp.array(dtype=wp.int32),
    cutoff: wp.float64,
    alpha: wp.float64,
    energies: wp.array(dtype=wp.float64),
):
    """Compute Coulomb energies for batched systems.

    Launch Grid: dim = [num_atoms]
    Each thread processes one atom and loops over its neighbors using CSR format.

    Note: Uses atomic_add to accumulate to per-atom energies.
    """
    atom_i = wp.tid()
    num_atoms = positions.shape[0]

    if atom_i >= num_atoms:
        return

    system_id = batch_idx[atom_i]
    ri = positions[atom_i]
    qi = charges[atom_i]
    cell_t = wp.transpose(cell[system_id])

    energy_acc = wp.float64(0.0)

    j_start = neighbor_ptr[atom_i]
    j_end = neighbor_ptr[atom_i + 1]

    for edge_idx in range(j_start, j_end):
        j = idx_j[edge_idx]

        rj = positions[j]
        qj = charges[j]

        shift_vec = cell_t * type(ri)(unit_shifts[edge_idx])
        r_ij = ri - rj - shift_vec
        r = wp.length(r_ij)

        if r >= cutoff or r < wp.float64(1e-10):
            continue

        prefactor = wp.float64(0.5) * qi * qj

        if alpha > wp.float64(0.0):
            alpha_r = alpha * r
            erfc_term = wp_erfc(alpha_r)
            energy_acc += prefactor * erfc_term / r
        else:
            energy_acc += prefactor / r

    wp.atomic_add(energies, atom_i, energy_acc)


@wp.kernel
def _batch_coulomb_energy_forces_kernel(
    positions: wp.array(dtype=wp.vec3d),
    charges: wp.array(dtype=wp.float64),
    cell: wp.array(dtype=wp.mat33d),
    idx_j: wp.array(dtype=wp.int32),
    neighbor_ptr: wp.array(dtype=wp.int32),
    unit_shifts: wp.array(dtype=wp.vec3i),
    batch_idx: wp.array(dtype=wp.int32),
    cutoff: wp.float64,
    alpha: wp.float64,
    energies: wp.array(dtype=wp.float64),
    forces: wp.array(dtype=wp.vec3d),
):
    """Compute Coulomb energies and forces for batched systems.

    Launch Grid: dim = [num_atoms]
    Each thread processes one atom and loops over its neighbors using CSR format.

    Note: Uses atomic_add to accumulate to per-atom arrays.
    """
    atom_i = wp.tid()
    num_atoms = positions.shape[0]

    if atom_i >= num_atoms:
        return

    system_id = batch_idx[atom_i]
    ri = positions[atom_i]
    qi = charges[atom_i]
    cell_t = wp.transpose(cell[system_id])

    energy_acc = wp.float64(0.0)
    force_acc = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))

    j_start = neighbor_ptr[atom_i]
    j_end = neighbor_ptr[atom_i + 1]

    for edge_idx in range(j_start, j_end):
        j = idx_j[edge_idx]

        rj = positions[j]
        qj = charges[j]

        shift_vec = cell_t * type(ri)(unit_shifts[edge_idx])
        r_ij = ri - rj - shift_vec
        r = wp.length(r_ij)

        if r >= cutoff or r < wp.float64(1e-10):
            continue

        prefactor = wp.float64(0.5) * qi * qj

        if alpha > wp.float64(0.0):
            alpha_r = alpha * r
            alpha_r_sq = alpha_r * alpha_r
            erfc_term = wp_erfc(alpha_r)
            exp_term = wp.exp(-alpha_r_sq)

            energy_acc += prefactor * erfc_term / r

            two_over_sqrt_pi = wp.float64(1.1283791670955126)
            force_mag_over_r = erfc_term / (
                r * r * r
            ) + two_over_sqrt_pi * alpha * exp_term / (r * r)
            force_ij = prefactor * force_mag_over_r * r_ij
        else:
            energy_acc += prefactor / r
            force_mag_over_r = prefactor / (r * r * r)
            force_ij = force_mag_over_r * r_ij

        force_acc += force_ij
        wp.atomic_add(forces, j, -force_ij)

    wp.atomic_add(energies, atom_i, energy_acc)
    wp.atomic_add(forces, atom_i, force_acc)


# ==============================================================================
# Warp Kernels - Batch Versions (Neighbor Matrix Format)
# ==============================================================================


@wp.kernel
def _batch_coulomb_energy_matrix_kernel(
    positions: wp.array(dtype=wp.vec3d),
    charges: wp.array(dtype=wp.float64),
    cell: wp.array(dtype=wp.mat33d),
    neighbor_matrix: wp.array2d(dtype=wp.int32),
    neighbor_matrix_shifts: wp.array2d(dtype=wp.vec3i),
    batch_idx: wp.array(dtype=wp.int32),
    cutoff: wp.float64,
    alpha: wp.float64,
    fill_value: wp.int32,
    atomic_energies: wp.array(dtype=wp.float64),
):
    """Compute Coulomb energies for batched systems using neighbor matrix.

    Launch Grid: dim = [num_atoms]
    Each thread processes one atom and loops over its neighbors.
    """
    atom_idx = wp.tid()
    num_atoms = positions.shape[0]
    max_neighbors = neighbor_matrix.shape[1]

    if atom_idx >= num_atoms:
        return

    system_id = batch_idx[atom_idx]
    ri = positions[atom_idx]
    qi = charges[atom_idx]
    cell_t = wp.transpose(cell[system_id])

    energy_acc = wp.float64(0.0)

    for neighbor_slot in range(max_neighbors):
        j = neighbor_matrix[atom_idx, neighbor_slot]
        if j >= fill_value or j >= num_atoms:
            continue

        rj = positions[j]
        qj = charges[j]

        shift = neighbor_matrix_shifts[atom_idx, neighbor_slot]
        shift_vec = cell_t * type(ri)(shift)
        r_ij = ri - rj - shift_vec
        r = wp.length(r_ij)

        if r >= cutoff or r < wp.float64(1e-10):
            continue

        prefactor = qi * qj

        if alpha > wp.float64(0.0):
            alpha_r = alpha * r
            erfc_term = wp_erfc(alpha_r)
            energy_acc += prefactor * erfc_term / r
        else:
            energy_acc += prefactor / r

    wp.atomic_add(atomic_energies, atom_idx, energy_acc)


@wp.kernel
def _batch_coulomb_energy_forces_matrix_kernel(
    positions: wp.array(dtype=wp.vec3d),
    charges: wp.array(dtype=wp.float64),
    cell: wp.array(dtype=wp.mat33d),
    neighbor_matrix: wp.array2d(dtype=wp.int32),
    neighbor_matrix_shifts: wp.array2d(dtype=wp.vec3i),
    batch_idx: wp.array(dtype=wp.int32),
    cutoff: wp.float64,
    alpha: wp.float64,
    fill_value: wp.int32,
    atomic_energies: wp.array(dtype=wp.float64),
    atomic_forces: wp.array(dtype=wp.vec3d),
):
    """Compute Coulomb energies and forces for batched systems using neighbor matrix.

    Launch Grid: dim = [num_atoms]
    Each thread processes one atom and loops over its neighbors.
    """
    atom_idx = wp.tid()
    num_atoms = positions.shape[0]
    max_neighbors = neighbor_matrix.shape[1]

    if atom_idx >= num_atoms:
        return

    system_id = batch_idx[atom_idx]
    ri = positions[atom_idx]
    qi = charges[atom_idx]
    cell_t = wp.transpose(cell[system_id])

    energy_acc = wp.float64(0.0)
    force_acc = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))

    for neighbor_slot in range(max_neighbors):
        j = neighbor_matrix[atom_idx, neighbor_slot]
        if j >= fill_value or j >= num_atoms:
            continue

        rj = positions[j]
        qj = charges[j]

        shift = neighbor_matrix_shifts[atom_idx, neighbor_slot]
        shift_vec = cell_t * type(ri)(shift)
        r_ij = ri - rj - shift_vec
        r = wp.length(r_ij)

        if r >= cutoff or r < wp.float64(1e-10):
            continue

        prefactor = wp.float64(0.5) * qi * qj

        if alpha > wp.float64(0.0):
            alpha_r = alpha * r
            alpha_r_sq = alpha_r * alpha_r
            erfc_term = wp_erfc(alpha_r)
            exp_term = wp.exp(-alpha_r_sq)

            energy_acc += prefactor * erfc_term / r
            two_over_sqrt_pi = wp.float64(1.1283791670955126)
            force_mag_over_r = erfc_term / (
                r * r * r
            ) + two_over_sqrt_pi * alpha * exp_term / (r * r)
            force_ij = prefactor * force_mag_over_r * r_ij
        else:
            energy_acc += prefactor / r
            force_mag_over_r = prefactor / (r * r * r)
            force_ij = force_mag_over_r * r_ij

        force_acc += force_ij
        wp.atomic_add(atomic_forces, j, -force_ij)

    wp.atomic_add(atomic_energies, atom_idx, energy_acc)
    wp.atomic_add(atomic_forces, atom_idx, force_acc)


# ==============================================================================
# Warp Launchers (Framework-Agnostic)
# ==============================================================================


def coulomb_energy(
    positions: wp.array,
    charges: wp.array,
    cell: wp.array,
    idx_j: wp.array,
    neighbor_ptr: wp.array,
    unit_shifts: wp.array,
    cutoff: float,
    alpha: float,
    energies: wp.array,
    device: str | None = None,
) -> None:
    """Launch Coulomb energy kernel using CSR neighbor list format.

    This is a framework-agnostic launcher that accepts warp arrays directly.
    Framework-specific wrappers (PyTorch, JAX) handle tensor-to-warp conversion.

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3d
        Atomic positions.
    charges : wp.array, shape (N,), dtype=wp.float64
        Atomic charges.
    cell : wp.array, shape (1,), dtype=wp.mat33d
        Unit cell matrix.
    idx_j : wp.array, shape (num_pairs,), dtype=wp.int32
        Destination atom indices in CSR format.
    neighbor_ptr : wp.array, shape (N+1,), dtype=wp.int32
        CSR row pointers.
    unit_shifts : wp.array, shape (num_pairs,), dtype=wp.vec3i
        Integer unit cell shifts.
    cutoff : float
        Cutoff distance.
    alpha : float
        Ewald splitting parameter (0.0 for undamped).
    energies : wp.array, shape (N,), dtype=wp.float64
        OUTPUT: Per-atom energies. Must be pre-allocated and zeroed.
    device : str, optional
        Warp device. If None, inferred from positions.

    Returns
    -------
    None
        Results are written to energies array in-place.
    """
    num_atoms = positions.shape[0]
    if device is None:
        device = str(positions.device)

    wp.launch(
        _coulomb_energy_kernel,
        dim=num_atoms,
        inputs=[
            positions,
            charges,
            cell,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            wp.float64(cutoff),
            wp.float64(alpha),
            energies,
        ],
        device=device,
    )


def coulomb_energy_forces(
    positions: wp.array,
    charges: wp.array,
    cell: wp.array,
    idx_j: wp.array,
    neighbor_ptr: wp.array,
    unit_shifts: wp.array,
    cutoff: float,
    alpha: float,
    energies: wp.array,
    forces: wp.array,
    device: str | None = None,
) -> None:
    """Launch Coulomb energy and forces kernel using CSR neighbor list format.

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3d
        Atomic positions.
    charges : wp.array, shape (N,), dtype=wp.float64
        Atomic charges.
    cell : wp.array, shape (1,), dtype=wp.mat33d
        Unit cell matrix.
    idx_j : wp.array, shape (num_pairs,), dtype=wp.int32
        Destination atom indices in CSR format.
    neighbor_ptr : wp.array, shape (N+1,), dtype=wp.int32
        CSR row pointers.
    unit_shifts : wp.array, shape (num_pairs,), dtype=wp.vec3i
        Integer unit cell shifts.
    cutoff : float
        Cutoff distance.
    alpha : float
        Ewald splitting parameter (0.0 for undamped).
    energies : wp.array, shape (N,), dtype=wp.float64
        OUTPUT: Per-atom energies. Must be pre-allocated and zeroed.
    forces : wp.array, shape (N,), dtype=wp.vec3d
        OUTPUT: Per-atom forces. Must be pre-allocated and zeroed.
    device : str, optional
        Warp device. If None, inferred from positions.

    Returns
    -------
    None
        Results are written to energies and forces arrays in-place.
    """
    num_atoms = positions.shape[0]
    if device is None:
        device = str(positions.device)

    wp.launch(
        _coulomb_energy_forces_kernel,
        dim=num_atoms,
        inputs=[
            positions,
            charges,
            cell,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            wp.float64(cutoff),
            wp.float64(alpha),
            energies,
            forces,
        ],
        device=device,
    )


def coulomb_energy_matrix(
    positions: wp.array,
    charges: wp.array,
    cell: wp.array,
    neighbor_matrix: wp.array,
    neighbor_matrix_shifts: wp.array,
    cutoff: float,
    alpha: float,
    fill_value: int,
    energies: wp.array,
    device: str | None = None,
) -> None:
    """Launch Coulomb energy kernel using neighbor matrix format.

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3d
        Atomic positions.
    charges : wp.array, shape (N,), dtype=wp.float64
        Atomic charges.
    cell : wp.array, shape (1,), dtype=wp.mat33d
        Unit cell matrix.
    neighbor_matrix : wp.array2d, shape (N, max_neighbors), dtype=wp.int32
        Neighbor indices.
    neighbor_matrix_shifts : wp.array2d, shape (N, max_neighbors), dtype=wp.vec3i
        Integer unit cell shifts.
    cutoff : float
        Cutoff distance.
    alpha : float
        Ewald splitting parameter (0.0 for undamped).
    fill_value : int
        Value indicating padding in neighbor_matrix.
    energies : wp.array, shape (N,), dtype=wp.float64
        OUTPUT: Per-atom energies. Must be pre-allocated and zeroed.
    device : str, optional
        Warp device. If None, inferred from positions.

    Returns
    -------
    None
        Results are written to energies array in-place.
    """
    num_atoms = positions.shape[0]
    if device is None:
        device = str(positions.device)

    wp.launch(
        _coulomb_energy_matrix_kernel,
        dim=num_atoms,
        inputs=[
            positions,
            charges,
            cell,
            neighbor_matrix,
            neighbor_matrix_shifts,
            wp.float64(cutoff),
            wp.float64(alpha),
            wp.int32(fill_value),
            energies,
        ],
        device=device,
    )


def coulomb_energy_forces_matrix(
    positions: wp.array,
    charges: wp.array,
    cell: wp.array,
    neighbor_matrix: wp.array,
    neighbor_matrix_shifts: wp.array,
    cutoff: float,
    alpha: float,
    fill_value: int,
    energies: wp.array,
    forces: wp.array,
    device: str | None = None,
) -> None:
    """Launch Coulomb energy and forces kernel using neighbor matrix format.

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3d
        Atomic positions.
    charges : wp.array, shape (N,), dtype=wp.float64
        Atomic charges.
    cell : wp.array, shape (1,), dtype=wp.mat33d
        Unit cell matrix.
    neighbor_matrix : wp.array2d, shape (N, max_neighbors), dtype=wp.int32
        Neighbor indices.
    neighbor_matrix_shifts : wp.array2d, shape (N, max_neighbors), dtype=wp.vec3i
        Integer unit cell shifts.
    cutoff : float
        Cutoff distance.
    alpha : float
        Ewald splitting parameter (0.0 for undamped).
    fill_value : int
        Value indicating padding in neighbor_matrix.
    energies : wp.array, shape (N,), dtype=wp.float64
        OUTPUT: Per-atom energies. Must be pre-allocated and zeroed.
    forces : wp.array, shape (N,), dtype=wp.vec3d
        OUTPUT: Per-atom forces. Must be pre-allocated and zeroed.
    device : str, optional
        Warp device. If None, inferred from positions.

    Returns
    -------
    None
        Results are written to energies and forces arrays in-place.
    """
    num_atoms = positions.shape[0]
    if device is None:
        device = str(positions.device)

    wp.launch(
        _coulomb_energy_forces_matrix_kernel,
        dim=num_atoms,
        inputs=[
            positions,
            charges,
            cell,
            neighbor_matrix,
            neighbor_matrix_shifts,
            wp.float64(cutoff),
            wp.float64(alpha),
            wp.int32(fill_value),
            energies,
            forces,
        ],
        device=device,
    )


def batch_coulomb_energy(
    positions: wp.array,
    charges: wp.array,
    cell: wp.array,
    idx_j: wp.array,
    neighbor_ptr: wp.array,
    unit_shifts: wp.array,
    batch_idx: wp.array,
    cutoff: float,
    alpha: float,
    energies: wp.array,
    device: str | None = None,
) -> None:
    """Launch batched Coulomb energy kernel using CSR neighbor list format.

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3d
        Atomic positions (all systems concatenated).
    charges : wp.array, shape (N,), dtype=wp.float64
        Atomic charges.
    cell : wp.array, shape (B,), dtype=wp.mat33d
        Unit cell matrices for each system.
    idx_j : wp.array, shape (num_pairs,), dtype=wp.int32
        Destination atom indices in CSR format.
    neighbor_ptr : wp.array, shape (N+1,), dtype=wp.int32
        CSR row pointers.
    unit_shifts : wp.array, shape (num_pairs,), dtype=wp.vec3i
        Integer unit cell shifts.
    batch_idx : wp.array, shape (N,), dtype=wp.int32
        System index for each atom.
    cutoff : float
        Cutoff distance.
    alpha : float
        Ewald splitting parameter (0.0 for undamped).
    energies : wp.array, shape (N,), dtype=wp.float64
        OUTPUT: Per-atom energies. Must be pre-allocated and zeroed.
    device : str, optional
        Warp device. If None, inferred from positions.

    Returns
    -------
    None
        Results are written to energies array in-place.
    """
    num_atoms = positions.shape[0]
    if device is None:
        device = str(positions.device)

    wp.launch(
        _batch_coulomb_energy_kernel,
        dim=num_atoms,
        inputs=[
            positions,
            charges,
            cell,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            batch_idx,
            wp.float64(cutoff),
            wp.float64(alpha),
            energies,
        ],
        device=device,
    )


def batch_coulomb_energy_forces(
    positions: wp.array,
    charges: wp.array,
    cell: wp.array,
    idx_j: wp.array,
    neighbor_ptr: wp.array,
    unit_shifts: wp.array,
    batch_idx: wp.array,
    cutoff: float,
    alpha: float,
    energies: wp.array,
    forces: wp.array,
    device: str | None = None,
) -> None:
    """Launch batched Coulomb energy and forces kernel using CSR neighbor list format.

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3d
        Atomic positions (all systems concatenated).
    charges : wp.array, shape (N,), dtype=wp.float64
        Atomic charges.
    cell : wp.array, shape (B,), dtype=wp.mat33d
        Unit cell matrices for each system.
    idx_j : wp.array, shape (num_pairs,), dtype=wp.int32
        Destination atom indices in CSR format.
    neighbor_ptr : wp.array, shape (N+1,), dtype=wp.int32
        CSR row pointers.
    unit_shifts : wp.array, shape (num_pairs,), dtype=wp.vec3i
        Integer unit cell shifts.
    batch_idx : wp.array, shape (N,), dtype=wp.int32
        System index for each atom.
    cutoff : float
        Cutoff distance.
    alpha : float
        Ewald splitting parameter (0.0 for undamped).
    energies : wp.array, shape (N,), dtype=wp.float64
        OUTPUT: Per-atom energies. Must be pre-allocated and zeroed.
    forces : wp.array, shape (N,), dtype=wp.vec3d
        OUTPUT: Per-atom forces. Must be pre-allocated and zeroed.
    device : str, optional
        Warp device. If None, inferred from positions.

    Returns
    -------
    None
        Results are written to energies and forces arrays in-place.
    """
    num_atoms = positions.shape[0]
    if device is None:
        device = str(positions.device)

    wp.launch(
        _batch_coulomb_energy_forces_kernel,
        dim=num_atoms,
        inputs=[
            positions,
            charges,
            cell,
            idx_j,
            neighbor_ptr,
            unit_shifts,
            batch_idx,
            wp.float64(cutoff),
            wp.float64(alpha),
            energies,
            forces,
        ],
        device=device,
    )


def batch_coulomb_energy_matrix(
    positions: wp.array,
    charges: wp.array,
    cell: wp.array,
    neighbor_matrix: wp.array,
    neighbor_matrix_shifts: wp.array,
    batch_idx: wp.array,
    cutoff: float,
    alpha: float,
    fill_value: int,
    energies: wp.array,
    device: str | None = None,
) -> None:
    """Launch batched Coulomb energy kernel using neighbor matrix format.

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3d
        Atomic positions (all systems concatenated).
    charges : wp.array, shape (N,), dtype=wp.float64
        Atomic charges.
    cell : wp.array, shape (B,), dtype=wp.mat33d
        Unit cell matrices for each system.
    neighbor_matrix : wp.array2d, shape (N, max_neighbors), dtype=wp.int32
        Neighbor indices.
    neighbor_matrix_shifts : wp.array2d, shape (N, max_neighbors), dtype=wp.vec3i
        Integer unit cell shifts.
    batch_idx : wp.array, shape (N,), dtype=wp.int32
        System index for each atom.
    cutoff : float
        Cutoff distance.
    alpha : float
        Ewald splitting parameter (0.0 for undamped).
    fill_value : int
        Value indicating padding in neighbor_matrix.
    energies : wp.array, shape (N,), dtype=wp.float64
        OUTPUT: Per-atom energies. Must be pre-allocated and zeroed.
    device : str, optional
        Warp device. If None, inferred from positions.

    Returns
    -------
    None
        Results are written to energies array in-place.
    """
    num_atoms = positions.shape[0]
    if device is None:
        device = str(positions.device)

    wp.launch(
        _batch_coulomb_energy_matrix_kernel,
        dim=num_atoms,
        inputs=[
            positions,
            charges,
            cell,
            neighbor_matrix,
            neighbor_matrix_shifts,
            batch_idx,
            wp.float64(cutoff),
            wp.float64(alpha),
            wp.int32(fill_value),
            energies,
        ],
        device=device,
    )


def batch_coulomb_energy_forces_matrix(
    positions: wp.array,
    charges: wp.array,
    cell: wp.array,
    neighbor_matrix: wp.array,
    neighbor_matrix_shifts: wp.array,
    batch_idx: wp.array,
    cutoff: float,
    alpha: float,
    fill_value: int,
    energies: wp.array,
    forces: wp.array,
    device: str | None = None,
) -> None:
    """Launch batched Coulomb energy and forces kernel using neighbor matrix format.

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3d
        Atomic positions (all systems concatenated).
    charges : wp.array, shape (N,), dtype=wp.float64
        Atomic charges.
    cell : wp.array, shape (B,), dtype=wp.mat33d
        Unit cell matrices for each system.
    neighbor_matrix : wp.array2d, shape (N, max_neighbors), dtype=wp.int32
        Neighbor indices.
    neighbor_matrix_shifts : wp.array2d, shape (N, max_neighbors), dtype=wp.vec3i
        Integer unit cell shifts.
    batch_idx : wp.array, shape (N,), dtype=wp.int32
        System index for each atom.
    cutoff : float
        Cutoff distance.
    alpha : float
        Ewald splitting parameter (0.0 for undamped).
    fill_value : int
        Value indicating padding in neighbor_matrix.
    energies : wp.array, shape (N,), dtype=wp.float64
        OUTPUT: Per-atom energies. Must be pre-allocated and zeroed.
    forces : wp.array, shape (N,), dtype=wp.vec3d
        OUTPUT: Per-atom forces. Must be pre-allocated and zeroed.
    device : str, optional
        Warp device. If None, inferred from positions.

    Returns
    -------
    None
        Results are written to energies and forces arrays in-place.
    """
    num_atoms = positions.shape[0]
    if device is None:
        device = str(positions.device)

    wp.launch(
        _batch_coulomb_energy_forces_matrix_kernel,
        dim=num_atoms,
        inputs=[
            positions,
            charges,
            cell,
            neighbor_matrix,
            neighbor_matrix_shifts,
            batch_idx,
            wp.float64(cutoff),
            wp.float64(alpha),
            wp.int32(fill_value),
            energies,
            forces,
        ],
        device=device,
    )
