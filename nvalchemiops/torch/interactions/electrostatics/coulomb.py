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
Coulomb Electrostatic Interactions - PyTorch Bindings
=====================================================

This module provides PyTorch bindings for Coulomb electrostatic calculations.
It wraps the framework-agnostic Warp launchers from
``nvalchemiops.interactions.electrostatics.coulomb``.

Public API
----------
- ``coulomb_energy()``: Compute energies only
- ``coulomb_forces()``: Compute forces only (convenience)
- ``coulomb_energy_forces()``: Compute both energies and forces

All functions support:
- Undamped (direct) and damped (Ewald/PME real-space) Coulomb
- Both neighbor list (CSR) and neighbor matrix formats
- Batched calculations
- Full autograd support

Examples
--------
>>> # Direct Coulomb energy and forces
>>> energy, forces = coulomb_energy_forces(
...     positions, charges, cell, cutoff=10.0,
...     neighbor_list=neighbor_list, neighbor_ptr=neighbor_ptr,
...     neighbor_shifts=neighbor_shifts
... )

>>> # Ewald/PME real-space contribution (damped)
>>> energy, forces = coulomb_energy_forces(
...     positions, charges, cell, cutoff=10.0, alpha=0.3,
...     neighbor_list=neighbor_list, neighbor_ptr=neighbor_ptr,
...     neighbor_shifts=neighbor_shifts
... )
"""

from __future__ import annotations

import torch
import warp as wp

from nvalchemiops.interactions.electrostatics.coulomb import (
    _batch_coulomb_energy_forces_kernel,
    _batch_coulomb_energy_forces_matrix_kernel,
    _batch_coulomb_energy_kernel,
    _batch_coulomb_energy_matrix_kernel,
    _coulomb_energy_forces_kernel,
    _coulomb_energy_forces_matrix_kernel,
    _coulomb_energy_kernel,
    _coulomb_energy_matrix_kernel,
)
from nvalchemiops.torch.autograd import (
    OutputSpec,
    WarpAutogradContextManager,
    attach_for_backward,
    needs_grad,
    warp_custom_op,
    warp_from_torch,
)
from nvalchemiops.types import get_wp_vec_dtype

__all__ = [
    "coulomb_energy",
    "coulomb_forces",
    "coulomb_energy_forces",
]


# ==============================================================================
# Internal Custom Ops - Neighbor List Format
# ==============================================================================

# Output dtype convention:
#   - Energies: always wp.float64 for numerical stability during accumulation.
#   - Forces: match input precision via get_wp_vec_dtype(pos.dtype) -- vec3f for
#     float32 inputs, vec3d for float64.


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
        OutputSpec(
            "forces",
            lambda pos, *_: get_wp_vec_dtype(pos.dtype),
            lambda pos, *_: (pos.shape[0], 3),
        ),
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
        OutputSpec(
            "forces",
            lambda pos, *_: get_wp_vec_dtype(pos.dtype),
            lambda pos, *_: (pos.shape[0], 3),
        ),
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
    """Internal: Compute Coulomb energies for batched systems using neighbor list."""
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
        OutputSpec(
            "forces",
            lambda pos, *_: get_wp_vec_dtype(pos.dtype),
            lambda pos, *_: (pos.shape[0], 3),
        ),
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
    """Internal: Compute Coulomb energies and forces for batched systems."""
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
    """Internal: Compute Coulomb energies for batched systems using neighbor matrix."""
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
        OutputSpec(
            "forces",
            lambda pos, *_: get_wp_vec_dtype(pos.dtype),
            lambda pos, *_: (pos.shape[0], 3),
        ),
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
    """Internal: Compute Coulomb energies and forces for batched systems."""
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
            "Must provide either neighbor_list/neighbor_shifts or "
            "neighbor_matrix/neighbor_matrix_shifts"
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

    Note
    ----
    Energies are always float64 for numerical stability during accumulation.
    Forces match the input dtype (float32 or float64).

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
            "Must provide either neighbor_list/neighbor_shifts or "
            "neighbor_matrix/neighbor_matrix_shifts"
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
