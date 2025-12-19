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
Coulomb Electrostatic Interactions
==================================

This module implements direct Coulomb energy and force calculations for electrostatic
interactions, including both undamped (direct) and damped (Ewald/PME real-space) variants.

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

1. **Neighbor List (COO format)**: `neighbor_list` is shape (2, num_pairs) where
   neighbor_list[0] are source indices and neighbor_list[1] are target indices.

2. **Neighbor Matrix**: `neighbor_matrix` is shape (N, max_neighbors) where
   each row contains neighbor indices for that atom.

API Structure
-------------

Internal Custom Ops (with autograd):
    - `_coulomb_energy_list`: Energy-only, neighbor list format
    - `_coulomb_energy_forces_list`: Energy+forces, neighbor list format
    - `_coulomb_energy_matrix`: Energy-only, neighbor matrix format
    - `_coulomb_energy_forces_matrix`: Energy+forces, neighbor matrix format
    - Batch versions of all above

Public Wrappers:
    - `coulomb_energy()`: Compute energies only
    - `coulomb_forces()`: Compute forces only (convenience)
    - `coulomb_energy_forces()`: Compute both energies and forces

References
----------
- Allen & Tildesley, "Computer Simulation of Liquids" (1987)
- Essmann et al., J. Chem. Phys. 103, 8577 (1995) - PME paper

Examples
--------
>>> # Direct Coulomb energy and forces
>>> energy, forces = coulomb_energy_forces(
...     positions, charges, cell, cutoff=10.0,
...     neighbor_list=neighbor_list, neighbor_shifts=neighbor_shifts
... )

>>> # Ewald/PME real-space contribution (damped)
>>> energy, forces = coulomb_energy_forces(
...     positions, charges, cell, cutoff=10.0, alpha=0.3,
...     neighbor_list=neighbor_list, neighbor_shifts=neighbor_shifts
... )
"""

from __future__ import annotations

import math

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
from nvalchemiops.math import wp_erfc

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
# Internal Custom Ops - Neighbor List Format
# ==============================================================================


@warp_custom_op(
    name="nvalchemiops::_coulomb_energy_list",
    outputs=[OutputSpec("energies", wp.float64, lambda pos, *_: (pos.shape[0],))],
    grad_arrays=["energies", "positions", "charges", "cell"],
)
def _coulomb_energy_list(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    neighbor_list: torch.Tensor,
    neighbor_ptr: torch.Tensor,
    neighbor_shifts: torch.Tensor,
    cutoff: float,
    alpha: float,
) -> torch.Tensor:
    """Internal: Compute Coulomb energies using neighbor list CSR format."""
    num_atoms = positions.shape[0]
    num_pairs = neighbor_list.shape[1]

    if num_pairs == 0:
        return torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)

    idx_j = neighbor_list[1].contiguous()

    device = wp.device_from_torch(positions.device)
    needs_grad_flag = needs_grad(positions, charges, cell)

    wp_positions = warp_from_torch(positions, wp.vec3d, requires_grad=needs_grad_flag)
    wp_charges = warp_from_torch(charges, wp.float64, requires_grad=needs_grad_flag)
    wp_cell = warp_from_torch(cell, wp.mat33d, requires_grad=needs_grad_flag)
    wp_idx_j = warp_from_torch(idx_j, wp.int32)
    wp_neighbor_ptr = warp_from_torch(neighbor_ptr, wp.int32)
    wp_unit_shifts = warp_from_torch(neighbor_shifts, wp.vec3i)

    energies = torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)
    wp_energies = warp_from_torch(energies, wp.float64, requires_grad=needs_grad_flag)

    with WarpAutogradContextManager(needs_grad_flag) as tape:
        wp.launch(
            _coulomb_energy_kernel,
            dim=num_atoms,
            inputs=[
                wp_positions,
                wp_charges,
                wp_cell,
                wp_idx_j,
                wp_neighbor_ptr,
                wp_unit_shifts,
                wp.float64(cutoff),
                wp.float64(alpha),
                wp_energies,
            ],
            device=device,
        )

    if needs_grad_flag:
        attach_for_backward(
            energies,
            tape=tape,
            energies=wp_energies,
            positions=wp_positions,
            charges=wp_charges,
            cell=wp_cell,
        )

    return energies


@warp_custom_op(
    name="nvalchemiops::_coulomb_energy_forces_list",
    outputs=[
        OutputSpec("energies", wp.float64, lambda pos, *_: (pos.shape[0],)),
        OutputSpec("forces", wp.vec3d, lambda pos, *_: (pos.shape[0], 3)),
    ],
    grad_arrays=["energies", "forces", "positions", "charges", "cell"],
)
def _coulomb_energy_forces_list(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    neighbor_list: torch.Tensor,
    neighbor_ptr: torch.Tensor,
    neighbor_shifts: torch.Tensor,
    cutoff: float,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Internal: Compute Coulomb energies and forces using neighbor list CSR format."""
    num_atoms = positions.shape[0]
    num_pairs = neighbor_list.shape[1]

    if num_pairs == 0:
        return (
            torch.zeros(num_atoms, device=positions.device, dtype=torch.float64),
            torch.zeros((num_atoms, 3), device=positions.device, dtype=torch.float64),
        )

    idx_j = neighbor_list[1].contiguous()

    device = wp.device_from_torch(positions.device)
    needs_grad_flag = needs_grad(positions, charges, cell)

    wp_positions = warp_from_torch(positions, wp.vec3d, requires_grad=needs_grad_flag)
    wp_charges = warp_from_torch(charges, wp.float64, requires_grad=needs_grad_flag)
    wp_cell = warp_from_torch(cell, wp.mat33d, requires_grad=needs_grad_flag)
    wp_idx_j = warp_from_torch(idx_j, wp.int32)
    wp_neighbor_ptr = warp_from_torch(neighbor_ptr, wp.int32)
    wp_unit_shifts = warp_from_torch(neighbor_shifts, wp.vec3i)

    energies = torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)
    forces = torch.zeros((num_atoms, 3), device=positions.device, dtype=torch.float64)
    wp_energies = warp_from_torch(energies, wp.float64, requires_grad=needs_grad_flag)
    wp_forces = warp_from_torch(forces, wp.vec3d, requires_grad=needs_grad_flag)

    with WarpAutogradContextManager(needs_grad_flag) as tape:
        wp.launch(
            _coulomb_energy_forces_kernel,
            dim=num_atoms,
            inputs=[
                wp_positions,
                wp_charges,
                wp_cell,
                wp_idx_j,
                wp_neighbor_ptr,
                wp_unit_shifts,
                wp.float64(cutoff),
                wp.float64(alpha),
                wp_energies,
                wp_forces,
            ],
            device=device,
        )

    if needs_grad_flag:
        attach_for_backward(
            energies,
            tape=tape,
            energies=wp_energies,
            forces=wp_forces,
            positions=wp_positions,
            charges=wp_charges,
            cell=wp_cell,
        )

    return energies, forces


# ==============================================================================
# Internal Custom Ops - Neighbor Matrix Format
# ==============================================================================


@warp_custom_op(
    name="nvalchemiops::_coulomb_energy_matrix",
    outputs=[OutputSpec("energies", wp.float64, lambda pos, *_: (pos.shape[0],))],
    grad_arrays=["energies", "positions", "charges", "cell"],
)
def _coulomb_energy_matrix(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    cutoff: float,
    alpha: float,
    fill_value: int,
) -> torch.Tensor:
    """Internal: Compute Coulomb energies using neighbor matrix format."""
    num_atoms = positions.shape[0]
    max_neighbors = neighbor_matrix.shape[1]

    if num_atoms == 0 or max_neighbors == 0:
        return torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)

    device = wp.device_from_torch(positions.device)
    needs_grad_flag = needs_grad(positions, charges, cell)

    wp_positions = warp_from_torch(positions, wp.vec3d, requires_grad=needs_grad_flag)
    wp_charges = warp_from_torch(charges, wp.float64, requires_grad=needs_grad_flag)
    wp_cell = warp_from_torch(cell, wp.mat33d, requires_grad=needs_grad_flag)
    wp_neighbor_matrix = warp_from_torch(neighbor_matrix, wp.int32)
    wp_neighbor_matrix_shifts = warp_from_torch(neighbor_matrix_shifts, wp.vec3i)

    atomic_energies = torch.zeros(
        num_atoms, device=positions.device, dtype=torch.float64
    )
    wp_energies = warp_from_torch(
        atomic_energies, wp.float64, requires_grad=needs_grad_flag
    )

    with WarpAutogradContextManager(needs_grad_flag) as tape:
        wp.launch(
            _coulomb_energy_matrix_kernel,
            dim=num_atoms,
            inputs=[
                wp_positions,
                wp_charges,
                wp_cell,
                wp_neighbor_matrix,
                wp_neighbor_matrix_shifts,
                wp.float64(cutoff),
                wp.float64(alpha),
                wp.int32(fill_value),
                wp_energies,
            ],
            device=device,
        )

    if needs_grad_flag:
        attach_for_backward(
            atomic_energies,
            tape=tape,
            energies=wp_energies,
            positions=wp_positions,
            charges=wp_charges,
            cell=wp_cell,
        )

    return atomic_energies


@warp_custom_op(
    name="nvalchemiops::_coulomb_energy_forces_matrix",
    outputs=[
        OutputSpec("energies", wp.float64, lambda pos, *_: (pos.shape[0],)),
        OutputSpec("forces", wp.vec3d, lambda pos, *_: (pos.shape[0], 3)),
    ],
    grad_arrays=["energies", "forces", "positions", "charges", "cell"],
)
def _coulomb_energy_forces_matrix(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    cutoff: float,
    alpha: float,
    fill_value: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Internal: Compute Coulomb energies and forces using neighbor matrix format."""
    num_atoms = positions.shape[0]
    max_neighbors = neighbor_matrix.shape[1]

    if num_atoms == 0 or max_neighbors == 0:
        return (
            torch.zeros(num_atoms, device=positions.device, dtype=torch.float64),
            torch.zeros((num_atoms, 3), device=positions.device, dtype=torch.float64),
        )

    device = wp.device_from_torch(positions.device)
    needs_grad_flag = needs_grad(positions, charges, cell)

    wp_positions = warp_from_torch(positions, wp.vec3d, requires_grad=needs_grad_flag)
    wp_charges = warp_from_torch(charges, wp.float64, requires_grad=needs_grad_flag)
    wp_cell = warp_from_torch(cell, wp.mat33d, requires_grad=needs_grad_flag)
    wp_neighbor_matrix = warp_from_torch(neighbor_matrix, wp.int32)
    wp_neighbor_matrix_shifts = warp_from_torch(neighbor_matrix_shifts, wp.vec3i)

    atomic_energies = torch.zeros(
        num_atoms, device=positions.device, dtype=torch.float64
    )
    forces = torch.zeros((num_atoms, 3), device=positions.device, dtype=torch.float64)
    wp_energies = warp_from_torch(
        atomic_energies, wp.float64, requires_grad=needs_grad_flag
    )
    wp_forces = warp_from_torch(forces, wp.vec3d, requires_grad=needs_grad_flag)

    with WarpAutogradContextManager(needs_grad_flag) as tape:
        wp.launch(
            _coulomb_energy_forces_matrix_kernel,
            dim=num_atoms,
            inputs=[
                wp_positions,
                wp_charges,
                wp_cell,
                wp_neighbor_matrix,
                wp_neighbor_matrix_shifts,
                wp.float64(cutoff),
                wp.float64(alpha),
                wp.int32(fill_value),
                wp_energies,
                wp_forces,
            ],
            device=device,
        )

    if needs_grad_flag:
        attach_for_backward(
            atomic_energies,
            tape=tape,
            energies=wp_energies,
            forces=wp_forces,
            positions=wp_positions,
            charges=wp_charges,
            cell=wp_cell,
        )

    return atomic_energies, forces


# ==============================================================================
# Internal Custom Ops - Batch Versions (Neighbor List Format)
# ==============================================================================


@warp_custom_op(
    name="nvalchemiops::_batch_coulomb_energy_list",
    outputs=[OutputSpec("energies", wp.float64, lambda pos, *_: (pos.shape[0],))],
    grad_arrays=["energies", "positions", "charges", "cell"],
)
def _batch_coulomb_energy_list(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    neighbor_list: torch.Tensor,
    neighbor_ptr: torch.Tensor,
    neighbor_shifts: torch.Tensor,
    batch_idx: torch.Tensor,
    cutoff: float,
    alpha: float,
) -> torch.Tensor:
    """Internal: Compute Coulomb energies for batched systems using neighbor list CSR format."""
    num_atoms = positions.shape[0]
    num_pairs = neighbor_list.shape[1]

    if num_pairs == 0:
        return torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)

    idx_j = neighbor_list[1].contiguous()

    device = wp.device_from_torch(positions.device)
    needs_grad_flag = needs_grad(positions, charges, cell)

    wp_positions = warp_from_torch(positions, wp.vec3d, requires_grad=needs_grad_flag)
    wp_charges = warp_from_torch(charges, wp.float64, requires_grad=needs_grad_flag)
    wp_cell = warp_from_torch(cell, wp.mat33d, requires_grad=needs_grad_flag)
    wp_idx_j = warp_from_torch(idx_j, wp.int32)
    wp_neighbor_ptr = warp_from_torch(neighbor_ptr, wp.int32)
    wp_unit_shifts = warp_from_torch(neighbor_shifts, wp.vec3i)
    wp_batch_idx = warp_from_torch(batch_idx, wp.int32)

    energies = torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)
    wp_energies = warp_from_torch(energies, wp.float64, requires_grad=needs_grad_flag)

    with WarpAutogradContextManager(needs_grad_flag) as tape:
        wp.launch(
            _batch_coulomb_energy_kernel,
            dim=num_atoms,
            inputs=[
                wp_positions,
                wp_charges,
                wp_cell,
                wp_idx_j,
                wp_neighbor_ptr,
                wp_unit_shifts,
                wp_batch_idx,
                wp.float64(cutoff),
                wp.float64(alpha),
                wp_energies,
            ],
            device=device,
        )

    if needs_grad_flag:
        attach_for_backward(
            energies,
            tape=tape,
            energies=wp_energies,
            positions=wp_positions,
            charges=wp_charges,
            cell=wp_cell,
        )

    return energies


@warp_custom_op(
    name="nvalchemiops::_batch_coulomb_energy_forces_list",
    outputs=[
        OutputSpec("energies", wp.float64, lambda pos, *_: (pos.shape[0],)),
        OutputSpec("forces", wp.vec3d, lambda pos, *_: (pos.shape[0], 3)),
    ],
    grad_arrays=["energies", "forces", "positions", "charges", "cell"],
)
def _batch_coulomb_energy_forces_list(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    neighbor_list: torch.Tensor,
    neighbor_ptr: torch.Tensor,
    neighbor_shifts: torch.Tensor,
    batch_idx: torch.Tensor,
    cutoff: float,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Internal: Compute Coulomb energies and forces for batched systems using neighbor list CSR format."""
    num_atoms = positions.shape[0]
    num_pairs = neighbor_list.shape[1]

    if num_pairs == 0:
        return (
            torch.zeros(num_atoms, device=positions.device, dtype=torch.float64),
            torch.zeros((num_atoms, 3), device=positions.device, dtype=torch.float64),
        )

    idx_j = neighbor_list[1].contiguous()

    device = wp.device_from_torch(positions.device)
    needs_grad_flag = needs_grad(positions, charges, cell)

    wp_positions = warp_from_torch(positions, wp.vec3d, requires_grad=needs_grad_flag)
    wp_charges = warp_from_torch(charges, wp.float64, requires_grad=needs_grad_flag)
    wp_cell = warp_from_torch(cell, wp.mat33d, requires_grad=needs_grad_flag)
    wp_idx_j = warp_from_torch(idx_j, wp.int32)
    wp_neighbor_ptr = warp_from_torch(neighbor_ptr, wp.int32)
    wp_unit_shifts = warp_from_torch(neighbor_shifts, wp.vec3i)
    wp_batch_idx = warp_from_torch(batch_idx, wp.int32)

    energies = torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)
    forces = torch.zeros((num_atoms, 3), device=positions.device, dtype=torch.float64)
    wp_energies = warp_from_torch(energies, wp.float64, requires_grad=needs_grad_flag)
    wp_forces = warp_from_torch(forces, wp.vec3d, requires_grad=needs_grad_flag)

    with WarpAutogradContextManager(needs_grad_flag) as tape:
        wp.launch(
            _batch_coulomb_energy_forces_kernel,
            dim=num_atoms,
            inputs=[
                wp_positions,
                wp_charges,
                wp_cell,
                wp_idx_j,
                wp_neighbor_ptr,
                wp_unit_shifts,
                wp_batch_idx,
                wp.float64(cutoff),
                wp.float64(alpha),
                wp_energies,
                wp_forces,
            ],
            device=device,
        )

    if needs_grad_flag:
        attach_for_backward(
            energies,
            tape=tape,
            energies=wp_energies,
            forces=wp_forces,
            positions=wp_positions,
            charges=wp_charges,
            cell=wp_cell,
        )

    return energies, forces


# ==============================================================================
# Internal Custom Ops - Batch Versions (Neighbor Matrix Format)
# ==============================================================================


@warp_custom_op(
    name="nvalchemiops::_batch_coulomb_energy_matrix",
    outputs=[OutputSpec("energies", wp.float64, lambda pos, *_: (pos.shape[0],))],
    grad_arrays=["energies", "positions", "charges", "cell"],
)
def _batch_coulomb_energy_matrix(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    batch_idx: torch.Tensor,
    cutoff: float,
    alpha: float,
    fill_value: int,
) -> torch.Tensor:
    """Internal: Compute Coulomb energies for batched systems using neighbor matrix format."""
    num_atoms = positions.shape[0]
    max_neighbors = neighbor_matrix.shape[1]

    if num_atoms == 0 or max_neighbors == 0:
        return torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)

    device = wp.device_from_torch(positions.device)
    needs_grad_flag = needs_grad(positions, charges, cell)

    wp_positions = warp_from_torch(positions, wp.vec3d, requires_grad=needs_grad_flag)
    wp_charges = warp_from_torch(charges, wp.float64, requires_grad=needs_grad_flag)
    wp_cell = warp_from_torch(cell, wp.mat33d, requires_grad=needs_grad_flag)
    wp_neighbor_matrix = warp_from_torch(neighbor_matrix, wp.int32)
    wp_neighbor_matrix_shifts = warp_from_torch(neighbor_matrix_shifts, wp.vec3i)
    wp_batch_idx = warp_from_torch(batch_idx, wp.int32)

    atomic_energies = torch.zeros(
        num_atoms, device=positions.device, dtype=torch.float64
    )
    wp_energies = warp_from_torch(
        atomic_energies, wp.float64, requires_grad=needs_grad_flag
    )

    with WarpAutogradContextManager(needs_grad_flag) as tape:
        wp.launch(
            _batch_coulomb_energy_matrix_kernel,
            dim=num_atoms,
            inputs=[
                wp_positions,
                wp_charges,
                wp_cell,
                wp_neighbor_matrix,
                wp_neighbor_matrix_shifts,
                wp_batch_idx,
                wp.float64(cutoff),
                wp.float64(alpha),
                wp.int32(fill_value),
                wp_energies,
            ],
            device=device,
        )

    if needs_grad_flag:
        attach_for_backward(
            atomic_energies,
            tape=tape,
            energies=wp_energies,
            positions=wp_positions,
            charges=wp_charges,
            cell=wp_cell,
        )

    return atomic_energies


@warp_custom_op(
    name="nvalchemiops::_batch_coulomb_energy_forces_matrix",
    outputs=[
        OutputSpec("energies", wp.float64, lambda pos, *_: (pos.shape[0],)),
        OutputSpec("forces", wp.vec3d, lambda pos, *_: (pos.shape[0], 3)),
    ],
    grad_arrays=["energies", "forces", "positions", "charges", "cell"],
)
def _batch_coulomb_energy_forces_matrix(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    batch_idx: torch.Tensor,
    cutoff: float,
    alpha: float,
    fill_value: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Internal: Compute Coulomb energies and forces for batched systems using neighbor matrix format."""
    num_atoms = positions.shape[0]
    max_neighbors = neighbor_matrix.shape[1]

    if num_atoms == 0 or max_neighbors == 0:
        return (
            torch.zeros(num_atoms, device=positions.device, dtype=torch.float64),
            torch.zeros((num_atoms, 3), device=positions.device, dtype=torch.float64),
        )

    device = wp.device_from_torch(positions.device)
    needs_grad_flag = needs_grad(positions, charges, cell)

    wp_positions = warp_from_torch(positions, wp.vec3d, requires_grad=needs_grad_flag)
    wp_charges = warp_from_torch(charges, wp.float64, requires_grad=needs_grad_flag)
    wp_cell = warp_from_torch(cell, wp.mat33d, requires_grad=needs_grad_flag)
    wp_neighbor_matrix = warp_from_torch(neighbor_matrix, wp.int32)
    wp_neighbor_matrix_shifts = warp_from_torch(neighbor_matrix_shifts, wp.vec3i)
    wp_batch_idx = warp_from_torch(batch_idx, wp.int32)

    atomic_energies = torch.zeros(
        num_atoms, device=positions.device, dtype=torch.float64
    )
    forces = torch.zeros((num_atoms, 3), device=positions.device, dtype=torch.float64)
    wp_energies = warp_from_torch(
        atomic_energies, wp.float64, requires_grad=needs_grad_flag
    )
    wp_forces = warp_from_torch(forces, wp.vec3d, requires_grad=needs_grad_flag)

    with WarpAutogradContextManager(needs_grad_flag) as tape:
        wp.launch(
            _batch_coulomb_energy_forces_matrix_kernel,
            dim=num_atoms,
            inputs=[
                wp_positions,
                wp_charges,
                wp_cell,
                wp_neighbor_matrix,
                wp_neighbor_matrix_shifts,
                wp_batch_idx,
                wp.float64(cutoff),
                wp.float64(alpha),
                wp.int32(fill_value),
                wp_energies,
                wp_forces,
            ],
            device=device,
        )

    if needs_grad_flag:
        attach_for_backward(
            atomic_energies,
            tape=tape,
            energies=wp_energies,
            forces=wp_forces,
            positions=wp_positions,
            charges=wp_charges,
            cell=wp_cell,
        )

    return atomic_energies, forces


# ==============================================================================
# Public API
# ==============================================================================


def coulomb_energy(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    cutoff: float,
    alpha: float = 0.0,
    neighbor_list: torch.Tensor | None = None,
    neighbor_ptr: torch.Tensor | None = None,
    neighbor_shifts: torch.Tensor | None = None,
    neighbor_matrix: torch.Tensor | None = None,
    neighbor_matrix_shifts: torch.Tensor | None = None,
    fill_value: int | None = None,
    batch_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute Coulomb electrostatic energies.

    Computes pairwise electrostatic energies using the Coulomb law,
    with optional erfc damping for Ewald/PME real-space calculations.
    Supports automatic differentiation with respect to positions, charges, and cell.

    Parameters
    ----------
    positions : torch.Tensor, shape (N, 3)
        Atomic coordinates.
    charges : torch.Tensor, shape (N,)
        Atomic charges.
    cell : torch.Tensor, shape (1, 3, 3) or (B, 3, 3)
        Unit cell matrix. Shape (B, 3, 3) for batched calculations.
    cutoff : float
        Cutoff distance for interactions.
    alpha : float, default=0.0
        Ewald splitting parameter. Use 0.0 for undamped Coulomb.
    neighbor_list : torch.Tensor | None, shape (2, num_pairs)
        Neighbor pairs in COO format. Row 0 = source, Row 1 = target.
    neighbor_ptr : torch.Tensor | None, shape (N+1,)
        CSR row pointers for neighbor list. Required with neighbor_list.
        Provided by neighborlist module.
    neighbor_shifts : torch.Tensor | None, shape (num_pairs, 3)
        Integer unit cell shifts for neighbor list format.
    neighbor_matrix : torch.Tensor | None, shape (N, max_neighbors)
        Neighbor indices in matrix format.
    neighbor_matrix_shifts : torch.Tensor | None, shape (N, max_neighbors, 3)
        Integer unit cell shifts for matrix format.
    fill_value : int | None
        Fill value for neighbor matrix padding.
    batch_idx : torch.Tensor | None, shape (N,)
        Batch indices for each atom.

    Returns
    -------
    energies : torch.Tensor, shape (N,)
        Per-atom energies. Sum to get total energy.

    Examples
    --------
    >>> # Direct Coulomb (undamped)
    >>> energies = coulomb_energy(
    ...     positions, charges, cell, cutoff=10.0, alpha=0.0,
    ...     neighbor_list=neighbor_list, neighbor_ptr=neighbor_ptr,
    ...     neighbor_shifts=neighbor_shifts
    ... )
    >>> total_energy = energies.sum()

    >>> # Ewald/PME real-space (damped) with autograd
    >>> positions.requires_grad_(True)
    >>> energies = coulomb_energy(
    ...     positions, charges, cell, cutoff=10.0, alpha=0.3,
    ...     neighbor_list=neighbor_list, neighbor_ptr=neighbor_ptr,
    ...     neighbor_shifts=neighbor_shifts
    ... )
    >>> energies.sum().backward()
    >>> forces = -positions.grad
    """
    # Validate inputs
    use_list = neighbor_list is not None and neighbor_shifts is not None
    use_matrix = neighbor_matrix is not None and neighbor_matrix_shifts is not None

    if not use_list and not use_matrix:
        raise ValueError(
            "Must provide either neighbor_list/neighbor_shifts or neighbor_matrix/neighbor_matrix_shifts"
        )

    if use_list and use_matrix:
        raise ValueError(
            "Cannot provide both neighbor list and neighbor matrix formats"
        )

    # Convert to float64 for numerical stability
    positions_f64 = positions.to(torch.float64)
    charges_f64 = charges.to(torch.float64)
    cell_f64 = cell.to(torch.float64)

    is_batched = batch_idx is not None

    if use_list:
        if neighbor_ptr is None:
            raise ValueError("neighbor_ptr is required when using neighbor_list format")
        neighbor_list_cont = neighbor_list.contiguous()
        neighbor_shifts_cont = neighbor_shifts.contiguous()

        if is_batched:
            energies = _batch_coulomb_energy_list(
                positions_f64,
                charges_f64,
                cell_f64,
                neighbor_list_cont,
                neighbor_ptr,
                neighbor_shifts_cont,
                batch_idx,
                cutoff,
                alpha,
            )
        else:
            energies = _coulomb_energy_list(
                positions_f64,
                charges_f64,
                cell_f64,
                neighbor_list_cont,
                neighbor_ptr,
                neighbor_shifts_cont,
                cutoff,
                alpha,
            )
    else:
        neighbor_matrix_cont = neighbor_matrix.contiguous()
        neighbor_matrix_shifts_cont = neighbor_matrix_shifts.contiguous()
        if fill_value is None:
            fill_value = positions.shape[0]

        if is_batched:
            energies = _batch_coulomb_energy_matrix(
                positions_f64,
                charges_f64,
                cell_f64,
                neighbor_matrix_cont,
                neighbor_matrix_shifts_cont,
                batch_idx,
                cutoff,
                alpha,
                fill_value,
            )
        else:
            energies = _coulomb_energy_matrix(
                positions_f64,
                charges_f64,
                cell_f64,
                neighbor_matrix_cont,
                neighbor_matrix_shifts_cont,
                cutoff,
                alpha,
                fill_value,
            )

    return energies.to(positions.dtype)


def coulomb_forces(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    cutoff: float,
    alpha: float = 0.0,
    neighbor_list: torch.Tensor | None = None,
    neighbor_ptr: torch.Tensor | None = None,
    neighbor_shifts: torch.Tensor | None = None,
    neighbor_matrix: torch.Tensor | None = None,
    neighbor_matrix_shifts: torch.Tensor | None = None,
    fill_value: int | None = None,
    batch_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute Coulomb electrostatic forces.

    Convenience wrapper that returns only forces (no energies).

    Parameters
    ----------
    See coulomb_energy for parameter descriptions.

    Returns
    -------
    forces : torch.Tensor, shape (N, 3)
        Forces on each atom.

    See Also
    --------
    coulomb_energy_forces : Compute both energies and forces
    """
    _, forces = coulomb_energy_forces(
        positions=positions,
        charges=charges,
        cell=cell,
        cutoff=cutoff,
        alpha=alpha,
        neighbor_list=neighbor_list,
        neighbor_ptr=neighbor_ptr,
        neighbor_shifts=neighbor_shifts,
        neighbor_matrix=neighbor_matrix,
        neighbor_matrix_shifts=neighbor_matrix_shifts,
        fill_value=fill_value,
        batch_idx=batch_idx,
    )
    return forces


def coulomb_energy_forces(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    cutoff: float,
    alpha: float = 0.0,
    neighbor_list: torch.Tensor | None = None,
    neighbor_ptr: torch.Tensor | None = None,
    neighbor_shifts: torch.Tensor | None = None,
    neighbor_matrix: torch.Tensor | None = None,
    neighbor_matrix_shifts: torch.Tensor | None = None,
    fill_value: int | None = None,
    batch_idx: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Coulomb electrostatic energies and forces.

    Computes pairwise electrostatic energies and forces using the Coulomb law,
    with optional erfc damping for Ewald/PME real-space calculations.
    Supports automatic differentiation with respect to positions, charges, and cell.

    Parameters
    ----------
    positions : torch.Tensor, shape (N, 3)
        Atomic coordinates.
    charges : torch.Tensor, shape (N,)
        Atomic charges.
    cell : torch.Tensor, shape (1, 3, 3) or (B, 3, 3)
        Unit cell matrix. Shape (B, 3, 3) for batched calculations.
    cutoff : float
        Cutoff distance for interactions.
    alpha : float, default=0.0
        Ewald splitting parameter. Use 0.0 for undamped Coulomb.
    neighbor_list : torch.Tensor | None, shape (2, num_pairs)
        Neighbor pairs in COO format.
    neighbor_ptr : torch.Tensor | None, shape (N+1,)
        CSR row pointers for neighbor list. Required with neighbor_list.
        Provided by neighborlist module.
    neighbor_shifts : torch.Tensor | None, shape (num_pairs, 3)
        Integer unit cell shifts for neighbor list format.
    neighbor_matrix : torch.Tensor | None, shape (N, max_neighbors)
        Neighbor indices in matrix format.
    neighbor_matrix_shifts : torch.Tensor | None, shape (N, max_neighbors, 3)
        Integer unit cell shifts for matrix format.
    fill_value : int | None
        Fill value for neighbor matrix padding.
    batch_idx : torch.Tensor | None, shape (N,)
        Batch indices for each atom.

    Returns
    -------
    energies : torch.Tensor, shape (N,)
        Per-atom energies.
    forces : torch.Tensor, shape (N, 3)
        Forces on each atom.

    Examples
    --------
    >>> # Direct Coulomb
    >>> energies, forces = coulomb_energy_forces(
    ...     positions, charges, cell, cutoff=10.0, alpha=0.0,
    ...     neighbor_list=neighbor_list, neighbor_ptr=neighbor_ptr,
    ...     neighbor_shifts=neighbor_shifts
    ... )

    >>> # Ewald/PME real-space
    >>> energies, forces = coulomb_energy_forces(
    ...     positions, charges, cell, cutoff=10.0, alpha=0.3,
    ...     neighbor_matrix=neighbor_matrix, neighbor_matrix_shifts=neighbor_matrix_shifts,
    ...     fill_value=num_atoms
    ... )
    """
    # Validate inputs
    use_list = neighbor_list is not None and neighbor_shifts is not None
    use_matrix = neighbor_matrix is not None and neighbor_matrix_shifts is not None

    if not use_list and not use_matrix:
        raise ValueError(
            "Must provide either neighbor_list/neighbor_shifts or neighbor_matrix/neighbor_matrix_shifts"
        )

    if use_list and use_matrix:
        raise ValueError(
            "Cannot provide both neighbor list and neighbor matrix formats"
        )

    # Convert to float64 for numerical stability
    positions_f64 = positions.to(torch.float64)
    charges_f64 = charges.to(torch.float64)
    cell_f64 = cell.to(torch.float64)

    is_batched = batch_idx is not None

    if use_list:
        if neighbor_ptr is None:
            raise ValueError("neighbor_ptr is required when using neighbor_list format")
        neighbor_list_cont = neighbor_list.contiguous()
        neighbor_shifts_cont = neighbor_shifts.contiguous()

        if is_batched:
            energies, forces = _batch_coulomb_energy_forces_list(
                positions_f64,
                charges_f64,
                cell_f64,
                neighbor_list_cont,
                neighbor_ptr,
                neighbor_shifts_cont,
                batch_idx,
                cutoff,
                alpha,
            )
        else:
            energies, forces = _coulomb_energy_forces_list(
                positions_f64,
                charges_f64,
                cell_f64,
                neighbor_list_cont,
                neighbor_ptr,
                neighbor_shifts_cont,
                cutoff,
                alpha,
            )
    else:
        neighbor_matrix_cont = neighbor_matrix.contiguous()
        neighbor_matrix_shifts_cont = neighbor_matrix_shifts.contiguous()
        if fill_value is None:
            fill_value = positions.shape[0]

        if is_batched:
            energies, forces = _batch_coulomb_energy_forces_matrix(
                positions_f64,
                charges_f64,
                cell_f64,
                neighbor_matrix_cont,
                neighbor_matrix_shifts_cont,
                batch_idx,
                cutoff,
                alpha,
                fill_value,
            )
        else:
            energies, forces = _coulomb_energy_forces_matrix(
                positions_f64,
                charges_f64,
                cell_f64,
                neighbor_matrix_cont,
                neighbor_matrix_shifts_cont,
                cutoff,
                alpha,
                fill_value,
            )

    return energies.to(positions.dtype), forces.to(positions.dtype)
