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

r"""
Pairwise Dispersion (r^-6) Interactions - PyTorch Bindings
==========================================================

PyTorch bindings for direct pairwise dispersion (van der Waals attractive
:math:`r^{-6}`) calculations. Wraps the framework-agnostic Warp kernels from
``nvalchemiops.interactions.dispersion.lj_real_kernels``.

This is the dispersion analog of
``nvalchemiops.torch.interactions.electrostatics.coulomb``. The public API takes
per-atom Lennard-Jones :math:`\sigma` / :math:`\epsilon` and converts internally
to the geometric-rule dispersion charge

.. math::

    C_{6,i} = 4 \epsilon_i \sigma_i^6, \qquad
    b_i = \sqrt{C_{6,i}} = 2\sqrt{\epsilon_i}\,\sigma_i^3 .

The pair energy uses :math:`C_{6,ij} = \sqrt{C_{6,i} C_{6,j}} = b_i b_j`.

Public API
----------
- ``lj_dispersion_energy()``: per-atom energies
- ``lj_dispersion_forces()``: forces (convenience)
- ``lj_dispersion_energy_forces()``: both

All functions support plain (``alpha=0``) and :math:`\beta`-damped real-space
dispersion, both neighbor formats, batching, and autograd w.r.t. positions,
``sigma``, ``epsilon`` and ``cell``.
"""

from __future__ import annotations

import torch
import warp as wp

from nvalchemiops.interactions.dispersion.lj_real_kernels import (
    _batch_lj_dispersion_energy_forces_kernel,
    _batch_lj_dispersion_energy_forces_matrix_kernel,
    _batch_lj_dispersion_energy_kernel,
    _batch_lj_dispersion_energy_matrix_kernel,
    _lj_dispersion_energy_forces_kernel,
    _lj_dispersion_energy_forces_matrix_kernel,
    _lj_dispersion_energy_kernel,
    _lj_dispersion_energy_matrix_kernel,
)
from nvalchemiops.torch.autograd import (
    OutputSpec,
    WarpAutogradContextManager,
    attach_for_backward,
    needs_grad,
    warp_custom_op,
    warp_from_torch,
)
from nvalchemiops.torch.types import get_wp_vec_dtype

__all__ = [
    "lj_dispersion_energy",
    "lj_dispersion_forces",
    "lj_dispersion_energy_forces",
    "sigma_epsilon_to_dispersion_charge",
]


# ==============================================================================
# Helpers
# ==============================================================================


def sigma_epsilon_to_dispersion_charge(
    sigma: torch.Tensor, epsilon: torch.Tensor
) -> torch.Tensor:
    r"""Convert per-atom LJ :math:`\sigma, \epsilon` to dispersion charge.

    Uses ``C6_i = 4 * epsilon_i * sigma_i**6`` and returns
    ``b_i = sqrt(C6_i) = 2 * sqrt(epsilon_i) * sigma_i**3`` (geometric rule).

    The computation is differentiable so autograd flows from ``b`` back to
    ``sigma`` and ``epsilon``.
    """
    sigma_f64 = sigma.to(torch.float64)
    epsilon_f64 = epsilon.to(torch.float64)
    return 2.0 * torch.sqrt(epsilon_f64) * sigma_f64**3


# ==============================================================================
# Internal Custom Ops - Neighbor List Format
# ==============================================================================

# Output dtype convention (matches electrostatics):
#   - Energies: always wp.float64 for stable accumulation.
#   - Forces: match input precision via get_wp_vec_dtype(pos.dtype).


@warp_custom_op(
    name="nvalchemiops::_lj_dispersion_energy_list",
    outputs=[OutputSpec("energies", wp.float64, lambda pos, *_: (pos.shape[0],))],
    grad_arrays=["energies", "positions", "b", "cell"],
)
def _lj_dispersion_energy_list(
    positions: torch.Tensor,
    b: torch.Tensor,
    cell: torch.Tensor,
    neighbor_list: torch.Tensor,
    neighbor_ptr: torch.Tensor,
    neighbor_shifts: torch.Tensor,
    cutoff: float,
    alpha: float,
) -> torch.Tensor:
    """Internal: dispersion energies using neighbor list CSR format."""
    num_atoms = positions.shape[0]
    num_pairs = neighbor_list.shape[1]

    if num_pairs == 0:
        return torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)

    idx_j = neighbor_list[1].contiguous()

    device = wp.device_from_torch(positions.device)
    needs_grad_flag = needs_grad(positions, b, cell)

    wp_positions = warp_from_torch(positions, wp.vec3d, requires_grad=needs_grad_flag)
    wp_b = warp_from_torch(b, wp.float64, requires_grad=needs_grad_flag)
    wp_cell = warp_from_torch(cell, wp.mat33d, requires_grad=needs_grad_flag)
    wp_idx_j = warp_from_torch(idx_j, wp.int32)
    wp_neighbor_ptr = warp_from_torch(neighbor_ptr, wp.int32)
    wp_unit_shifts = warp_from_torch(neighbor_shifts, wp.vec3i)

    energies = torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)
    wp_energies = warp_from_torch(energies, wp.float64, requires_grad=needs_grad_flag)

    with WarpAutogradContextManager(needs_grad_flag) as tape:
        wp.launch(
            _lj_dispersion_energy_kernel,
            dim=num_atoms,
            inputs=[
                wp_positions,
                wp_b,
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
            b=wp_b,
            cell=wp_cell,
        )

    return energies


@warp_custom_op(
    name="nvalchemiops::_lj_dispersion_energy_forces_list",
    outputs=[
        OutputSpec("energies", wp.float64, lambda pos, *_: (pos.shape[0],)),
        OutputSpec(
            "forces",
            lambda pos, *_: get_wp_vec_dtype(pos.dtype),
            lambda pos, *_: (pos.shape[0], 3),
        ),
    ],
    grad_arrays=["energies", "forces", "positions", "b", "cell"],
)
def _lj_dispersion_energy_forces_list(
    positions: torch.Tensor,
    b: torch.Tensor,
    cell: torch.Tensor,
    neighbor_list: torch.Tensor,
    neighbor_ptr: torch.Tensor,
    neighbor_shifts: torch.Tensor,
    cutoff: float,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Internal: dispersion energies and forces using neighbor list CSR format."""
    num_atoms = positions.shape[0]
    num_pairs = neighbor_list.shape[1]

    if num_pairs == 0:
        return (
            torch.zeros(num_atoms, device=positions.device, dtype=torch.float64),
            torch.zeros((num_atoms, 3), device=positions.device, dtype=torch.float64),
        )

    idx_j = neighbor_list[1].contiguous()

    device = wp.device_from_torch(positions.device)
    needs_grad_flag = needs_grad(positions, b, cell)

    wp_positions = warp_from_torch(positions, wp.vec3d, requires_grad=needs_grad_flag)
    wp_b = warp_from_torch(b, wp.float64, requires_grad=needs_grad_flag)
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
            _lj_dispersion_energy_forces_kernel,
            dim=num_atoms,
            inputs=[
                wp_positions,
                wp_b,
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
            b=wp_b,
            cell=wp_cell,
        )

    return energies, forces


# ==============================================================================
# Internal Custom Ops - Neighbor Matrix Format
# ==============================================================================


@warp_custom_op(
    name="nvalchemiops::_lj_dispersion_energy_matrix",
    outputs=[OutputSpec("energies", wp.float64, lambda pos, *_: (pos.shape[0],))],
    grad_arrays=["energies", "positions", "b", "cell"],
)
def _lj_dispersion_energy_matrix(
    positions: torch.Tensor,
    b: torch.Tensor,
    cell: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    cutoff: float,
    alpha: float,
    fill_value: int,
) -> torch.Tensor:
    """Internal: dispersion energies using neighbor matrix format."""
    num_atoms = positions.shape[0]
    max_neighbors = neighbor_matrix.shape[1]

    if num_atoms == 0 or max_neighbors == 0:
        return torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)

    device = wp.device_from_torch(positions.device)
    needs_grad_flag = needs_grad(positions, b, cell)

    wp_positions = warp_from_torch(positions, wp.vec3d, requires_grad=needs_grad_flag)
    wp_b = warp_from_torch(b, wp.float64, requires_grad=needs_grad_flag)
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
            _lj_dispersion_energy_matrix_kernel,
            dim=num_atoms,
            inputs=[
                wp_positions,
                wp_b,
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
            b=wp_b,
            cell=wp_cell,
        )

    return atomic_energies


@warp_custom_op(
    name="nvalchemiops::_lj_dispersion_energy_forces_matrix",
    outputs=[
        OutputSpec("energies", wp.float64, lambda pos, *_: (pos.shape[0],)),
        OutputSpec(
            "forces",
            lambda pos, *_: get_wp_vec_dtype(pos.dtype),
            lambda pos, *_: (pos.shape[0], 3),
        ),
    ],
    grad_arrays=["energies", "forces", "positions", "b", "cell"],
)
def _lj_dispersion_energy_forces_matrix(
    positions: torch.Tensor,
    b: torch.Tensor,
    cell: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    cutoff: float,
    alpha: float,
    fill_value: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Internal: dispersion energies and forces using neighbor matrix format."""
    num_atoms = positions.shape[0]
    max_neighbors = neighbor_matrix.shape[1]

    if num_atoms == 0 or max_neighbors == 0:
        return (
            torch.zeros(num_atoms, device=positions.device, dtype=torch.float64),
            torch.zeros((num_atoms, 3), device=positions.device, dtype=torch.float64),
        )

    device = wp.device_from_torch(positions.device)
    needs_grad_flag = needs_grad(positions, b, cell)

    wp_positions = warp_from_torch(positions, wp.vec3d, requires_grad=needs_grad_flag)
    wp_b = warp_from_torch(b, wp.float64, requires_grad=needs_grad_flag)
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
            _lj_dispersion_energy_forces_matrix_kernel,
            dim=num_atoms,
            inputs=[
                wp_positions,
                wp_b,
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
            b=wp_b,
            cell=wp_cell,
        )

    return atomic_energies, forces


# ==============================================================================
# Internal Custom Ops - Batch Versions (Neighbor List Format)
# ==============================================================================


@warp_custom_op(
    name="nvalchemiops::_batch_lj_dispersion_energy_list",
    outputs=[OutputSpec("energies", wp.float64, lambda pos, *_: (pos.shape[0],))],
    grad_arrays=["energies", "positions", "b", "cell"],
)
def _batch_lj_dispersion_energy_list(
    positions: torch.Tensor,
    b: torch.Tensor,
    cell: torch.Tensor,
    neighbor_list: torch.Tensor,
    neighbor_ptr: torch.Tensor,
    neighbor_shifts: torch.Tensor,
    batch_idx: torch.Tensor,
    cutoff: float,
    alpha: float,
) -> torch.Tensor:
    """Internal: batched dispersion energies using neighbor list."""
    num_atoms = positions.shape[0]
    num_pairs = neighbor_list.shape[1]

    if num_pairs == 0:
        return torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)

    idx_j = neighbor_list[1].contiguous()

    device = wp.device_from_torch(positions.device)
    needs_grad_flag = needs_grad(positions, b, cell)

    wp_positions = warp_from_torch(positions, wp.vec3d, requires_grad=needs_grad_flag)
    wp_b = warp_from_torch(b, wp.float64, requires_grad=needs_grad_flag)
    wp_cell = warp_from_torch(cell, wp.mat33d, requires_grad=needs_grad_flag)
    wp_idx_j = warp_from_torch(idx_j, wp.int32)
    wp_neighbor_ptr = warp_from_torch(neighbor_ptr, wp.int32)
    wp_unit_shifts = warp_from_torch(neighbor_shifts, wp.vec3i)
    wp_batch_idx = warp_from_torch(batch_idx, wp.int32)

    energies = torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)
    wp_energies = warp_from_torch(energies, wp.float64, requires_grad=needs_grad_flag)

    with WarpAutogradContextManager(needs_grad_flag) as tape:
        wp.launch(
            _batch_lj_dispersion_energy_kernel,
            dim=num_atoms,
            inputs=[
                wp_positions,
                wp_b,
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
            b=wp_b,
            cell=wp_cell,
        )

    return energies


@warp_custom_op(
    name="nvalchemiops::_batch_lj_dispersion_energy_forces_list",
    outputs=[
        OutputSpec("energies", wp.float64, lambda pos, *_: (pos.shape[0],)),
        OutputSpec(
            "forces",
            lambda pos, *_: get_wp_vec_dtype(pos.dtype),
            lambda pos, *_: (pos.shape[0], 3),
        ),
    ],
    grad_arrays=["energies", "forces", "positions", "b", "cell"],
)
def _batch_lj_dispersion_energy_forces_list(
    positions: torch.Tensor,
    b: torch.Tensor,
    cell: torch.Tensor,
    neighbor_list: torch.Tensor,
    neighbor_ptr: torch.Tensor,
    neighbor_shifts: torch.Tensor,
    batch_idx: torch.Tensor,
    cutoff: float,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Internal: batched dispersion energies and forces using neighbor list."""
    num_atoms = positions.shape[0]
    num_pairs = neighbor_list.shape[1]

    if num_pairs == 0:
        return (
            torch.zeros(num_atoms, device=positions.device, dtype=torch.float64),
            torch.zeros((num_atoms, 3), device=positions.device, dtype=torch.float64),
        )

    idx_j = neighbor_list[1].contiguous()

    device = wp.device_from_torch(positions.device)
    needs_grad_flag = needs_grad(positions, b, cell)

    wp_positions = warp_from_torch(positions, wp.vec3d, requires_grad=needs_grad_flag)
    wp_b = warp_from_torch(b, wp.float64, requires_grad=needs_grad_flag)
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
            _batch_lj_dispersion_energy_forces_kernel,
            dim=num_atoms,
            inputs=[
                wp_positions,
                wp_b,
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
            b=wp_b,
            cell=wp_cell,
        )

    return energies, forces


# ==============================================================================
# Internal Custom Ops - Batch Versions (Neighbor Matrix Format)
# ==============================================================================


@warp_custom_op(
    name="nvalchemiops::_batch_lj_dispersion_energy_matrix",
    outputs=[OutputSpec("energies", wp.float64, lambda pos, *_: (pos.shape[0],))],
    grad_arrays=["energies", "positions", "b", "cell"],
)
def _batch_lj_dispersion_energy_matrix(
    positions: torch.Tensor,
    b: torch.Tensor,
    cell: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    batch_idx: torch.Tensor,
    cutoff: float,
    alpha: float,
    fill_value: int,
) -> torch.Tensor:
    """Internal: batched dispersion energies using neighbor matrix."""
    num_atoms = positions.shape[0]
    max_neighbors = neighbor_matrix.shape[1]

    if num_atoms == 0 or max_neighbors == 0:
        return torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)

    device = wp.device_from_torch(positions.device)
    needs_grad_flag = needs_grad(positions, b, cell)

    wp_positions = warp_from_torch(positions, wp.vec3d, requires_grad=needs_grad_flag)
    wp_b = warp_from_torch(b, wp.float64, requires_grad=needs_grad_flag)
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
            _batch_lj_dispersion_energy_matrix_kernel,
            dim=num_atoms,
            inputs=[
                wp_positions,
                wp_b,
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
            b=wp_b,
            cell=wp_cell,
        )

    return atomic_energies


@warp_custom_op(
    name="nvalchemiops::_batch_lj_dispersion_energy_forces_matrix",
    outputs=[
        OutputSpec("energies", wp.float64, lambda pos, *_: (pos.shape[0],)),
        OutputSpec(
            "forces",
            lambda pos, *_: get_wp_vec_dtype(pos.dtype),
            lambda pos, *_: (pos.shape[0], 3),
        ),
    ],
    grad_arrays=["energies", "forces", "positions", "b", "cell"],
)
def _batch_lj_dispersion_energy_forces_matrix(
    positions: torch.Tensor,
    b: torch.Tensor,
    cell: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    batch_idx: torch.Tensor,
    cutoff: float,
    alpha: float,
    fill_value: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Internal: batched dispersion energies and forces using neighbor matrix."""
    num_atoms = positions.shape[0]
    max_neighbors = neighbor_matrix.shape[1]

    if num_atoms == 0 or max_neighbors == 0:
        return (
            torch.zeros(num_atoms, device=positions.device, dtype=torch.float64),
            torch.zeros((num_atoms, 3), device=positions.device, dtype=torch.float64),
        )

    device = wp.device_from_torch(positions.device)
    needs_grad_flag = needs_grad(positions, b, cell)

    wp_positions = warp_from_torch(positions, wp.vec3d, requires_grad=needs_grad_flag)
    wp_b = warp_from_torch(b, wp.float64, requires_grad=needs_grad_flag)
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
            _batch_lj_dispersion_energy_forces_matrix_kernel,
            dim=num_atoms,
            inputs=[
                wp_positions,
                wp_b,
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
            b=wp_b,
            cell=wp_cell,
        )

    return atomic_energies, forces


# ==============================================================================
# Public API
# ==============================================================================


def _validate_neighbors(
    neighbor_list,
    neighbor_shifts,
    neighbor_matrix,
    neighbor_matrix_shifts,
):
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
    return use_list, use_matrix


def lj_dispersion_energy(
    positions: torch.Tensor,
    sigma: torch.Tensor,
    epsilon: torch.Tensor,
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
    r"""Compute pairwise dispersion (:math:`r^{-6}`) energies.

    Parameters
    ----------
    positions : torch.Tensor, shape (N, 3)
        Atomic coordinates.
    sigma : torch.Tensor, shape (N,)
        Per-atom Lennard-Jones :math:`\sigma`.
    epsilon : torch.Tensor, shape (N,)
        Per-atom Lennard-Jones :math:`\epsilon`.
    cell : torch.Tensor, shape (1, 3, 3) or (B, 3, 3)
        Unit cell matrix. Shape (B, 3, 3) for batched calculations.
    cutoff : float
        Cutoff distance for interactions.
    alpha : float, default=0.0
        Dispersion Ewald splitting parameter :math:`\beta`. Use 0.0 for the
        plain :math:`-C_6/r^6` interaction; > 0 selects the damped real-space
        term used by dispersion PME.
    neighbor_list, neighbor_ptr, neighbor_shifts : torch.Tensor | None
        CSR neighbor list (COO pairs, row pointers, integer shifts).
    neighbor_matrix, neighbor_matrix_shifts : torch.Tensor | None
        Neighbor matrix format.
    fill_value : int | None
        Padding value for neighbor matrix.
    batch_idx : torch.Tensor | None, shape (N,)
        Batch index for each atom.

    Returns
    -------
    energies : torch.Tensor, shape (N,)
        Per-atom energies (input dtype). Sum to get total energy.
    """
    use_list, _ = _validate_neighbors(
        neighbor_list, neighbor_shifts, neighbor_matrix, neighbor_matrix_shifts
    )

    positions_f64 = positions.to(torch.float64)
    cell_f64 = cell.to(torch.float64)
    b = sigma_epsilon_to_dispersion_charge(sigma, epsilon)

    is_batched = batch_idx is not None

    if use_list:
        if neighbor_ptr is None:
            raise ValueError("neighbor_ptr is required when using neighbor_list format")
        neighbor_list_cont = neighbor_list.contiguous()
        neighbor_shifts_cont = neighbor_shifts.contiguous()

        if is_batched:
            energies = _batch_lj_dispersion_energy_list(
                positions_f64,
                b,
                cell_f64,
                neighbor_list_cont,
                neighbor_ptr,
                neighbor_shifts_cont,
                batch_idx,
                cutoff,
                alpha,
            )
        else:
            energies = _lj_dispersion_energy_list(
                positions_f64,
                b,
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
            energies = _batch_lj_dispersion_energy_matrix(
                positions_f64,
                b,
                cell_f64,
                neighbor_matrix_cont,
                neighbor_matrix_shifts_cont,
                batch_idx,
                cutoff,
                alpha,
                fill_value,
            )
        else:
            energies = _lj_dispersion_energy_matrix(
                positions_f64,
                b,
                cell_f64,
                neighbor_matrix_cont,
                neighbor_matrix_shifts_cont,
                cutoff,
                alpha,
                fill_value,
            )

    return energies.to(positions.dtype)


def lj_dispersion_forces(
    positions: torch.Tensor,
    sigma: torch.Tensor,
    epsilon: torch.Tensor,
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
    """Compute pairwise dispersion forces (convenience wrapper).

    See :func:`lj_dispersion_energy` for parameter descriptions.
    """
    _, forces = lj_dispersion_energy_forces(
        positions=positions,
        sigma=sigma,
        epsilon=epsilon,
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


def lj_dispersion_energy_forces(
    positions: torch.Tensor,
    sigma: torch.Tensor,
    epsilon: torch.Tensor,
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
    r"""Compute pairwise dispersion (:math:`r^{-6}`) energies and forces.

    See :func:`lj_dispersion_energy` for parameter descriptions.

    Returns
    -------
    energies : torch.Tensor, shape (N,)
        Per-atom energies (input dtype).
    forces : torch.Tensor, shape (N, 3)
        Forces on each atom (input dtype).

    Note
    ----
    Energies are accumulated in float64 internally for stability. Forces match
    the input dtype.
    """
    use_list, _ = _validate_neighbors(
        neighbor_list, neighbor_shifts, neighbor_matrix, neighbor_matrix_shifts
    )

    positions_f64 = positions.to(torch.float64)
    cell_f64 = cell.to(torch.float64)
    b = sigma_epsilon_to_dispersion_charge(sigma, epsilon)

    is_batched = batch_idx is not None

    if use_list:
        if neighbor_ptr is None:
            raise ValueError("neighbor_ptr is required when using neighbor_list format")
        neighbor_list_cont = neighbor_list.contiguous()
        neighbor_shifts_cont = neighbor_shifts.contiguous()

        if is_batched:
            energies, forces = _batch_lj_dispersion_energy_forces_list(
                positions_f64,
                b,
                cell_f64,
                neighbor_list_cont,
                neighbor_ptr,
                neighbor_shifts_cont,
                batch_idx,
                cutoff,
                alpha,
            )
        else:
            energies, forces = _lj_dispersion_energy_forces_list(
                positions_f64,
                b,
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
            energies, forces = _batch_lj_dispersion_energy_forces_matrix(
                positions_f64,
                b,
                cell_f64,
                neighbor_matrix_cont,
                neighbor_matrix_shifts_cont,
                batch_idx,
                cutoff,
                alpha,
                fill_value,
            )
        else:
            energies, forces = _lj_dispersion_energy_forces_matrix(
                positions_f64,
                b,
                cell_f64,
                neighbor_matrix_cont,
                neighbor_matrix_shifts_cont,
                cutoff,
                alpha,
                fill_value,
            )

    return energies.to(positions.dtype), forces.to(positions.dtype)
