# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Ewald Summation - PyTorch Bindings
==================================

This module provides PyTorch bindings for Ewald summation calculations.
It wraps the framework-agnostic Warp launchers from
``nvalchemiops.interactions.electrostatics.ewald_kernels``.

Public API
----------
- ``ewald_real_space()``: Real-space component of Ewald summation
- ``ewald_reciprocal_space()``: Reciprocal-space component
- ``ewald_summation()``: Complete Ewald summation (real + reciprocal)

Mathematical Formulation
------------------------
The Ewald method splits long-range Coulomb interactions into components:

.. math::

    E_{\text{total}} = E_{\text{real}} + E_{\text{reciprocal}} - E_{\text{self}} - E_{\text{background}}

All functions support:
- Both neighbor list (CSR) and neighbor matrix formats
- Batched calculations
- Full autograd support
- Optional explicit forces and charge gradients

Examples
--------
>>> # Complete Ewald summation
>>> energies, forces = ewald_summation(
...     positions, charges, cell,
...     neighbor_list=nl, neighbor_ptr=neighbor_ptr, neighbor_shifts=shifts,
...     accuracy=1e-6,
...     compute_forces=True,
... )

>>> # Separate real and reciprocal components
>>> e_real, f_real = ewald_real_space(
...     positions, charges, cell, alpha,
...     neighbor_list=nl, neighbor_ptr=neighbor_ptr, neighbor_shifts=shifts,
...     compute_forces=True,
... )
>>> e_recip, f_recip = ewald_reciprocal_space(
...     positions, charges, cell, k_vectors, alpha,
...     compute_forces=True,
... )
"""

from __future__ import annotations

import math

import torch
import warp as wp

from nvalchemiops.interactions.electrostatics.ewald_kernels import (
    BATCH_BLOCK_SIZE,
    _batch_ewald_real_space_energy_forces_charge_grad_kernel_overload,
    _batch_ewald_real_space_energy_forces_charge_grad_neighbor_matrix_kernel_overload,
    _batch_ewald_real_space_energy_forces_kernel_overload,
    _batch_ewald_real_space_energy_forces_neighbor_matrix_kernel_overload,
    _batch_ewald_real_space_energy_kernel_overload,
    _batch_ewald_real_space_energy_neighbor_matrix_kernel_overload,
    _batch_ewald_reciprocal_space_energy_forces_charge_grad_kernel_overload,
    _batch_ewald_reciprocal_space_energy_forces_kernel_overload,
    _batch_ewald_reciprocal_space_energy_kernel_compute_energy_overload,
    _batch_ewald_reciprocal_space_energy_kernel_fill_structure_factors_overload,
    _batch_ewald_reciprocal_space_virial_kernel_overload,
    _batch_ewald_subtract_self_energy_kernel_overload,
    _ewald_real_space_energy_forces_charge_grad_kernel_overload,
    _ewald_real_space_energy_forces_charge_grad_neighbor_matrix_kernel_overload,
    _ewald_real_space_energy_forces_kernel_overload,
    _ewald_real_space_energy_forces_neighbor_matrix_kernel_overload,
    _ewald_real_space_energy_kernel_overload,
    _ewald_real_space_energy_neighbor_matrix_kernel_overload,
    _ewald_reciprocal_space_energy_forces_charge_grad_kernel_overload,
    _ewald_reciprocal_space_energy_forces_kernel_overload,
    _ewald_reciprocal_space_energy_kernel_compute_energy_overload,
    _ewald_reciprocal_space_energy_kernel_fill_structure_factors_overload,
    _ewald_reciprocal_space_virial_kernel_overload,
    _ewald_subtract_self_energy_kernel_overload,
)
from nvalchemiops.torch.autograd import (
    OutputSpec,
    WarpAutogradContextManager,
    attach_for_backward,
    needs_grad,
    warp_custom_op,
    warp_from_torch,
)
from nvalchemiops.torch.interactions.electrostatics.k_vectors import (
    generate_k_vectors_ewald_summation,
)
from nvalchemiops.torch.interactions.electrostatics.parameters import (
    estimate_ewald_parameters,
)
from nvalchemiops.types import get_wp_dtype, get_wp_mat_dtype, get_wp_vec_dtype

__all__ = [
    "ewald_real_space",
    "ewald_reciprocal_space",
    "ewald_summation",
]


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
        Target dtype for the output tensor.
    device : torch.device
        Target device for the output tensor.

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
########################### Real-Space Internal Custom Ops ################################
###########################################################################################

# Output dtype convention:
#   - Energies: always wp.float64 for numerical stability during accumulation.
#   - Forces: match input precision via get_wp_vec_dtype(pos.dtype) -- vec3f for
#     float32 inputs, vec3d for float64.  This was changed from the previous
#     hardcoded wp.vec3d to fix a dtype mismatch when positions are float32.
#   - Virial: match input precision via get_wp_mat_dtype(pos.dtype) -- mat33f for
#     float32 inputs, mat33d for float64.


@warp_custom_op(
    name="alchemiops::_ewald_real_space_energy",
    outputs=[
        OutputSpec("energies", wp.float64, lambda pos, *_: (pos.shape[0],)),
    ],
    grad_arrays=["energies", "positions", "charges", "cell", "alpha"],
)
def _ewald_real_space_energy(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    alpha: torch.Tensor,
    neighbor_list: torch.Tensor,
    neighbor_ptr: torch.Tensor,
    neighbor_shifts: torch.Tensor,
) -> torch.Tensor:
    """Internal: Compute real-space Ewald energies (single system, neighbor list CSR)."""
    num_atoms = positions.shape[0]
    input_dtype = positions.dtype
    empty_nl = neighbor_list.shape[1] == 0

    idx_j = neighbor_list[1]
    device = wp.device_from_torch(positions.device)
    needs_grad_flag = needs_grad(positions, charges, cell)

    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)

    wp_positions = warp_from_torch(positions, wp_vec, requires_grad=needs_grad_flag)
    wp_charges = warp_from_torch(charges, wp_scalar, requires_grad=needs_grad_flag)
    wp_cell = warp_from_torch(cell, wp_mat, requires_grad=needs_grad_flag)
    wp_alpha = warp_from_torch(alpha, wp_scalar, requires_grad=needs_grad_flag)
    wp_idx_j = warp_from_torch(idx_j, wp.int32)
    wp_neighbor_ptr = warp_from_torch(neighbor_ptr, wp.int32)
    wp_unit_shifts = warp_from_torch(neighbor_shifts, wp.vec3i)

    energies = torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)
    wp_energies = warp_from_torch(energies, wp.float64, requires_grad=needs_grad_flag)

    with WarpAutogradContextManager(needs_grad_flag) as tape:
        if not empty_nl:
            wp.launch(
                _ewald_real_space_energy_kernel_overload[wp_scalar],
                dim=[num_atoms],
                inputs=[
                    wp_positions,
                    wp_charges,
                    wp_cell,
                    wp_idx_j,
                    wp_neighbor_ptr,
                    wp_unit_shifts,
                    wp_alpha,
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
            alpha=wp_alpha,
        )
    return energies


@warp_custom_op(
    name="alchemiops::_ewald_real_space_energy_forces",
    outputs=[
        OutputSpec("energies", wp.float64, lambda pos, *_: (pos.shape[0],)),
        OutputSpec(
            "forces",
            lambda pos, *_: get_wp_vec_dtype(pos.dtype),
            lambda pos, *_: (pos.shape[0], 3),
        ),
        OutputSpec(
            "virial",
            lambda pos, *_: get_wp_mat_dtype(pos.dtype),
            lambda pos, charges, cell, *_: (cell.shape[0], 3, 3),
        ),
    ],
    grad_arrays=[
        "energies",
        "forces",
        "virial",
        "positions",
        "charges",
        "cell",
        "alpha",
    ],
)
def _ewald_real_space_energy_forces(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    alpha: torch.Tensor,
    neighbor_list: torch.Tensor,
    neighbor_ptr: torch.Tensor,
    neighbor_shifts: torch.Tensor,
    compute_virial: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Internal: Compute real-space Ewald energies, forces, and optionally virial (single, CSR)."""
    num_atoms = positions.shape[0]
    input_dtype = positions.dtype
    empty_nl = neighbor_list.shape[1] == 0

    idx_j = neighbor_list[1]
    device = wp.device_from_torch(positions.device)
    needs_grad_flag = needs_grad(positions, charges, cell)
    virial_grad = needs_grad_flag and compute_virial

    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)

    wp_positions = warp_from_torch(positions, wp_vec, requires_grad=needs_grad_flag)
    wp_charges = warp_from_torch(charges, wp_scalar, requires_grad=needs_grad_flag)
    wp_cell = warp_from_torch(cell, wp_mat, requires_grad=needs_grad_flag)
    wp_alpha = warp_from_torch(alpha, wp_scalar, requires_grad=needs_grad_flag)
    wp_idx_j = warp_from_torch(idx_j, wp.int32)
    wp_neighbor_ptr = warp_from_torch(neighbor_ptr, wp.int32)
    wp_unit_shifts = warp_from_torch(neighbor_shifts, wp.vec3i)

    energies = torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)
    forces = torch.zeros(num_atoms, 3, device=positions.device, dtype=input_dtype)
    virial = torch.zeros(1, 3, 3, device=positions.device, dtype=input_dtype)
    wp_energies = warp_from_torch(energies, wp.float64, requires_grad=needs_grad_flag)
    wp_forces = warp_from_torch(forces, wp_vec, requires_grad=needs_grad_flag)
    wp_virial = warp_from_torch(virial, wp_mat, requires_grad=virial_grad)

    with WarpAutogradContextManager(needs_grad_flag) as tape:
        if not empty_nl:
            wp.launch(
                _ewald_real_space_energy_forces_kernel_overload[wp_scalar],
                dim=[num_atoms],
                inputs=[
                    wp_positions,
                    wp_charges,
                    wp_cell,
                    wp_idx_j,
                    wp_neighbor_ptr,
                    wp_unit_shifts,
                    wp_alpha,
                    compute_virial,
                    wp_energies,
                    wp_forces,
                    wp_virial,
                ],
                device=device,
            )

    if needs_grad_flag:
        backward_kw = dict(
            energies=wp_energies,
            forces=wp_forces,
            positions=wp_positions,
            charges=wp_charges,
            cell=wp_cell,
            alpha=wp_alpha,
        )
        if virial_grad:
            backward_kw["virial"] = wp_virial
        attach_for_backward(energies, tape=tape, **backward_kw)
    return energies, forces, virial


@warp_custom_op(
    name="alchemiops::_ewald_real_space_energy_matrix",
    outputs=[OutputSpec("energies", wp.float64, lambda pos, *_: (pos.shape[0],))],
    grad_arrays=["energies", "positions", "charges", "cell", "alpha"],
)
def _ewald_real_space_energy_matrix(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    alpha: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    mask_value: int,
) -> torch.Tensor:
    """Internal: Compute real-space Ewald energies (single system, neighbor matrix)."""
    num_atoms = positions.shape[0]
    input_dtype = positions.dtype
    empty_nm = neighbor_matrix.shape[0] == 0

    device = wp.device_from_torch(positions.device)
    needs_grad_flag = needs_grad(positions, charges, cell)

    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)

    wp_positions = warp_from_torch(positions, wp_vec, requires_grad=needs_grad_flag)
    wp_charges = warp_from_torch(charges, wp_scalar, requires_grad=needs_grad_flag)
    wp_cell = warp_from_torch(cell, wp_mat, requires_grad=needs_grad_flag)
    wp_alpha = warp_from_torch(alpha, wp_scalar, requires_grad=needs_grad_flag)
    wp_neighbor_matrix = warp_from_torch(neighbor_matrix, wp.int32)
    wp_unit_shifts_matrix = warp_from_torch(neighbor_matrix_shifts, wp.vec3i)

    energies = torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)
    wp_energies = warp_from_torch(energies, wp.float64, requires_grad=needs_grad_flag)

    with WarpAutogradContextManager(needs_grad_flag) as tape:
        if not empty_nm:
            wp.launch(
                _ewald_real_space_energy_neighbor_matrix_kernel_overload[wp_scalar],
                dim=[neighbor_matrix.shape[0]],
                inputs=[
                    wp_positions,
                    wp_charges,
                    wp_cell,
                    wp_neighbor_matrix,
                    wp_unit_shifts_matrix,
                    wp.int32(mask_value),
                    wp_alpha,
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
            alpha=wp_alpha,
        )
    return energies


@warp_custom_op(
    name="alchemiops::_ewald_real_space_energy_forces_matrix",
    outputs=[
        OutputSpec("energies", wp.float64, lambda pos, *_: (pos.shape[0],)),
        OutputSpec(
            "forces",
            lambda pos, *_: get_wp_vec_dtype(pos.dtype),
            lambda pos, *_: (pos.shape[0], 3),
        ),
        OutputSpec(
            "virial",
            lambda pos, *_: get_wp_mat_dtype(pos.dtype),
            lambda pos, charges, cell, *_: (cell.shape[0], 3, 3),
        ),
    ],
    grad_arrays=[
        "energies",
        "forces",
        "virial",
        "positions",
        "charges",
        "cell",
        "alpha",
    ],
)
def _ewald_real_space_energy_forces_matrix(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    alpha: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    mask_value: int,
    compute_virial: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Internal: Compute real-space Ewald energies, forces, and optionally virial (single, matrix)."""
    num_atoms = positions.shape[0]
    input_dtype = positions.dtype
    empty_nm = neighbor_matrix.shape[0] == 0

    device = wp.device_from_torch(positions.device)
    needs_grad_flag = needs_grad(positions, charges, cell)
    virial_grad = needs_grad_flag and compute_virial

    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)

    wp_positions = warp_from_torch(positions, wp_vec, requires_grad=needs_grad_flag)
    wp_charges = warp_from_torch(charges, wp_scalar, requires_grad=needs_grad_flag)
    wp_cell = warp_from_torch(cell, wp_mat, requires_grad=needs_grad_flag)
    wp_alpha = warp_from_torch(alpha, wp_scalar, requires_grad=needs_grad_flag)
    wp_neighbor_matrix = warp_from_torch(neighbor_matrix, wp.int32)
    wp_unit_shifts_matrix = warp_from_torch(neighbor_matrix_shifts, wp.vec3i)

    energies = torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)
    forces = torch.zeros(num_atoms, 3, device=positions.device, dtype=input_dtype)
    virial = torch.zeros(1, 3, 3, device=positions.device, dtype=input_dtype)
    wp_energies = warp_from_torch(energies, wp.float64, requires_grad=needs_grad_flag)
    wp_forces = warp_from_torch(forces, wp_vec, requires_grad=needs_grad_flag)
    wp_virial = warp_from_torch(virial, wp_mat, requires_grad=virial_grad)

    with WarpAutogradContextManager(needs_grad_flag) as tape:
        if not empty_nm:
            wp.launch(
                _ewald_real_space_energy_forces_neighbor_matrix_kernel_overload[
                    wp_scalar
                ],
                dim=[neighbor_matrix.shape[0]],
                inputs=[
                    wp_positions,
                    wp_charges,
                    wp_cell,
                    wp_neighbor_matrix,
                    wp_unit_shifts_matrix,
                    wp.int32(mask_value),
                    wp_alpha,
                    compute_virial,
                    wp_energies,
                    wp_forces,
                    wp_virial,
                ],
                device=device,
            )

    if needs_grad_flag:
        backward_kw = dict(
            energies=wp_energies,
            forces=wp_forces,
            positions=wp_positions,
            charges=wp_charges,
            cell=wp_cell,
            alpha=wp_alpha,
        )
        if virial_grad:
            backward_kw["virial"] = wp_virial
        attach_for_backward(energies, tape=tape, **backward_kw)
    return energies, forces, virial


###########################################################################################
################## Real-Space with Charge Gradients Internal Custom Ops ###################
###########################################################################################


@warp_custom_op(
    name="alchemiops::_ewald_real_space_energy_forces_charge_grad",
    outputs=[
        OutputSpec("energies", wp.float64, lambda pos, *_: (pos.shape[0],)),
        OutputSpec(
            "forces",
            lambda pos, *_: get_wp_vec_dtype(pos.dtype),
            lambda pos, *_: (pos.shape[0], 3),
        ),
        OutputSpec("charge_gradients", wp.float64, lambda pos, *_: (pos.shape[0],)),
        OutputSpec(
            "virial",
            lambda pos, *_: get_wp_mat_dtype(pos.dtype),
            lambda pos, charges, cell, *_: (cell.shape[0], 3, 3),
        ),
    ],
    grad_arrays=[
        "energies",
        "forces",
        "charge_gradients",
        "virial",
        "positions",
        "charges",
        "cell",
        "alpha",
    ],
)
def _ewald_real_space_energy_forces_charge_grad(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    alpha: torch.Tensor,
    neighbor_list: torch.Tensor,
    neighbor_ptr: torch.Tensor,
    neighbor_shifts: torch.Tensor,
    compute_virial: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Internal: Compute real-space Ewald E+F+charge_grad+virial (single, CSR)."""
    num_atoms = positions.shape[0]
    input_dtype = positions.dtype
    empty_nl = neighbor_list.shape[1] == 0

    idx_j = neighbor_list[1]
    device = wp.device_from_torch(positions.device)
    needs_grad_flag = needs_grad(positions, charges, cell)
    virial_grad = needs_grad_flag and compute_virial

    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)

    wp_positions = warp_from_torch(positions, wp_vec, requires_grad=needs_grad_flag)
    wp_charges = warp_from_torch(charges, wp_scalar, requires_grad=needs_grad_flag)
    wp_cell = warp_from_torch(cell, wp_mat, requires_grad=needs_grad_flag)
    wp_alpha = warp_from_torch(alpha, wp_scalar, requires_grad=needs_grad_flag)
    wp_idx_j = warp_from_torch(idx_j, wp.int32)
    wp_neighbor_ptr = warp_from_torch(neighbor_ptr, wp.int32)
    wp_unit_shifts = warp_from_torch(neighbor_shifts, wp.vec3i)

    energies = torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)
    forces = torch.zeros(num_atoms, 3, device=positions.device, dtype=input_dtype)
    charge_grads = torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)
    virial = torch.zeros(1, 3, 3, device=positions.device, dtype=input_dtype)
    wp_energies = warp_from_torch(energies, wp.float64, requires_grad=needs_grad_flag)
    wp_forces = warp_from_torch(forces, wp_vec, requires_grad=needs_grad_flag)
    wp_charge_grads = warp_from_torch(
        charge_grads, wp.float64, requires_grad=needs_grad_flag
    )
    wp_virial = warp_from_torch(virial, wp_mat, requires_grad=virial_grad)

    with WarpAutogradContextManager(needs_grad_flag) as tape:
        if not empty_nl:
            wp.launch(
                _ewald_real_space_energy_forces_charge_grad_kernel_overload[wp_scalar],
                dim=[num_atoms],
                inputs=[
                    wp_positions,
                    wp_charges,
                    wp_cell,
                    wp_idx_j,
                    wp_neighbor_ptr,
                    wp_unit_shifts,
                    wp_alpha,
                    compute_virial,
                    wp_energies,
                    wp_forces,
                    wp_charge_grads,
                    wp_virial,
                ],
                device=device,
            )

    if needs_grad_flag:
        backward_kw = dict(
            energies=wp_energies,
            forces=wp_forces,
            charge_gradients=wp_charge_grads,
            positions=wp_positions,
            charges=wp_charges,
            cell=wp_cell,
            alpha=wp_alpha,
        )
        if virial_grad:
            backward_kw["virial"] = wp_virial
        attach_for_backward(energies, tape=tape, **backward_kw)
    return energies, forces, charge_grads, virial


@warp_custom_op(
    name="alchemiops::_ewald_real_space_energy_forces_charge_grad_matrix",
    outputs=[
        OutputSpec("energies", wp.float64, lambda pos, *_: (pos.shape[0],)),
        OutputSpec(
            "forces",
            lambda pos, *_: get_wp_vec_dtype(pos.dtype),
            lambda pos, *_: (pos.shape[0], 3),
        ),
        OutputSpec("charge_gradients", wp.float64, lambda pos, *_: (pos.shape[0],)),
        OutputSpec(
            "virial",
            lambda pos, *_: get_wp_mat_dtype(pos.dtype),
            lambda pos, charges, cell, *_: (cell.shape[0], 3, 3),
        ),
    ],
    grad_arrays=[
        "energies",
        "forces",
        "charge_gradients",
        "virial",
        "positions",
        "charges",
        "cell",
        "alpha",
    ],
)
def _ewald_real_space_energy_forces_charge_grad_matrix(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    alpha: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    mask_value: int,
    compute_virial: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Internal: Compute real-space Ewald E+F+charge_grad+virial (single, matrix)."""
    num_atoms = positions.shape[0]
    input_dtype = positions.dtype
    empty_nm = neighbor_matrix.shape[0] == 0

    device = wp.device_from_torch(positions.device)
    needs_grad_flag = needs_grad(positions, charges, cell)
    virial_grad = needs_grad_flag and compute_virial

    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)

    wp_positions = warp_from_torch(positions, wp_vec, requires_grad=needs_grad_flag)
    wp_charges = warp_from_torch(charges, wp_scalar, requires_grad=needs_grad_flag)
    wp_cell = warp_from_torch(cell, wp_mat, requires_grad=needs_grad_flag)
    wp_alpha = warp_from_torch(alpha, wp_scalar, requires_grad=needs_grad_flag)
    wp_neighbor_matrix = warp_from_torch(neighbor_matrix, wp.int32)
    wp_unit_shifts_matrix = warp_from_torch(neighbor_matrix_shifts, wp.vec3i)

    energies = torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)
    forces = torch.zeros(num_atoms, 3, device=positions.device, dtype=input_dtype)
    charge_grads = torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)
    virial = torch.zeros(1, 3, 3, device=positions.device, dtype=input_dtype)
    wp_energies = warp_from_torch(energies, wp.float64, requires_grad=needs_grad_flag)
    wp_forces = warp_from_torch(forces, wp_vec, requires_grad=needs_grad_flag)
    wp_charge_grads = warp_from_torch(
        charge_grads, wp.float64, requires_grad=needs_grad_flag
    )
    wp_virial = warp_from_torch(virial, wp_mat, requires_grad=virial_grad)

    with WarpAutogradContextManager(needs_grad_flag) as tape:
        if not empty_nm:
            wp.launch(
                _ewald_real_space_energy_forces_charge_grad_neighbor_matrix_kernel_overload[
                    wp_scalar
                ],
                dim=[neighbor_matrix.shape[0]],
                inputs=[
                    wp_positions,
                    wp_charges,
                    wp_cell,
                    wp_neighbor_matrix,
                    wp_unit_shifts_matrix,
                    wp.int32(mask_value),
                    wp_alpha,
                    compute_virial,
                    wp_energies,
                    wp_forces,
                    wp_charge_grads,
                    wp_virial,
                ],
                device=device,
            )

    if needs_grad_flag:
        backward_kw = dict(
            energies=wp_energies,
            forces=wp_forces,
            charge_gradients=wp_charge_grads,
            positions=wp_positions,
            charges=wp_charges,
            cell=wp_cell,
            alpha=wp_alpha,
        )
        if virial_grad:
            backward_kw["virial"] = wp_virial
        attach_for_backward(energies, tape=tape, **backward_kw)
    return energies, forces, charge_grads, virial


###########################################################################################
########################### Batch Real-Space Internal Custom Ops ##########################
###########################################################################################


@warp_custom_op(
    name="alchemiops::_batch_ewald_real_space_energy",
    outputs=[OutputSpec("energies", wp.float64, lambda pos, *_: (pos.shape[0],))],
    grad_arrays=["energies", "positions", "charges", "cell", "alpha"],
)
def _batch_ewald_real_space_energy(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    alpha: torch.Tensor,
    batch_idx: torch.Tensor,
    neighbor_list: torch.Tensor,
    neighbor_ptr: torch.Tensor,
    neighbor_shifts: torch.Tensor,
) -> torch.Tensor:
    """Internal: Compute real-space Ewald energies (batch, neighbor list CSR)."""
    num_atoms = positions.shape[0]
    input_dtype = positions.dtype
    device = wp.device_from_torch(positions.device)
    needs_grad_flag = needs_grad(positions, charges, cell)
    empty_nl = neighbor_list.shape[1] == 0

    idx_j = neighbor_list[1]

    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)

    wp_positions = warp_from_torch(positions, wp_vec, requires_grad=needs_grad_flag)
    wp_charges = warp_from_torch(charges, wp_scalar, requires_grad=needs_grad_flag)
    wp_cell = warp_from_torch(cell, wp_mat, requires_grad=needs_grad_flag)
    wp_alpha = warp_from_torch(alpha, wp_scalar, requires_grad=needs_grad_flag)
    wp_batch_idx = warp_from_torch(batch_idx, wp.int32)
    wp_idx_j = warp_from_torch(idx_j, wp.int32)
    wp_neighbor_ptr = warp_from_torch(neighbor_ptr, wp.int32)
    wp_unit_shifts = warp_from_torch(neighbor_shifts, wp.vec3i)

    energies = torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)
    wp_energies = warp_from_torch(energies, wp.float64, requires_grad=needs_grad_flag)

    with WarpAutogradContextManager(needs_grad_flag) as tape:
        if not empty_nl:
            wp.launch(
                _batch_ewald_real_space_energy_kernel_overload[wp_scalar],
                dim=[num_atoms],
                inputs=[
                    wp_positions,
                    wp_charges,
                    wp_cell,
                    wp_batch_idx,
                    wp_idx_j,
                    wp_neighbor_ptr,
                    wp_unit_shifts,
                    wp_alpha,
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
            alpha=wp_alpha,
        )
    return energies


@warp_custom_op(
    name="alchemiops::_batch_ewald_real_space_energy_forces",
    outputs=[
        OutputSpec("energies", wp.float64, lambda pos, *_: (pos.shape[0],)),
        OutputSpec(
            "forces",
            lambda pos, *_: get_wp_vec_dtype(pos.dtype),
            lambda pos, *_: (pos.shape[0], 3),
        ),
        OutputSpec(
            "virial",
            lambda pos, *_: get_wp_mat_dtype(pos.dtype),
            lambda pos, charges, cell, *_: (cell.shape[0], 3, 3),
        ),
    ],
    grad_arrays=[
        "energies",
        "forces",
        "virial",
        "positions",
        "charges",
        "cell",
        "alpha",
    ],
)
def _batch_ewald_real_space_energy_forces(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    alpha: torch.Tensor,
    batch_idx: torch.Tensor,
    neighbor_list: torch.Tensor,
    neighbor_ptr: torch.Tensor,
    neighbor_shifts: torch.Tensor,
    compute_virial: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Internal: Compute real-space Ewald energies, forces, and optionally virial (batch, CSR)."""
    num_atoms = positions.shape[0]
    input_dtype = positions.dtype
    empty_nl = neighbor_list.shape[1] == 0

    idx_j = neighbor_list[1]
    device = wp.device_from_torch(positions.device)
    needs_grad_flag = needs_grad(positions, charges, cell)
    virial_grad = needs_grad_flag and compute_virial

    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)

    wp_positions = warp_from_torch(positions, wp_vec, requires_grad=needs_grad_flag)
    wp_charges = warp_from_torch(charges, wp_scalar, requires_grad=needs_grad_flag)
    wp_cell = warp_from_torch(cell, wp_mat, requires_grad=needs_grad_flag)
    wp_alpha = warp_from_torch(alpha, wp_scalar, requires_grad=needs_grad_flag)
    wp_batch_idx = warp_from_torch(batch_idx, wp.int32)
    wp_idx_j = warp_from_torch(idx_j, wp.int32)
    wp_neighbor_ptr = warp_from_torch(neighbor_ptr, wp.int32)
    wp_unit_shifts = warp_from_torch(neighbor_shifts, wp.vec3i)

    num_systems = cell.shape[0]
    energies = torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)
    forces = torch.zeros(num_atoms, 3, device=positions.device, dtype=input_dtype)
    virial = torch.zeros(num_systems, 3, 3, device=positions.device, dtype=input_dtype)
    wp_energies = warp_from_torch(energies, wp.float64, requires_grad=needs_grad_flag)
    wp_forces = warp_from_torch(forces, wp_vec, requires_grad=needs_grad_flag)
    wp_virial = warp_from_torch(virial, wp_mat, requires_grad=virial_grad)

    with WarpAutogradContextManager(needs_grad_flag) as tape:
        if not empty_nl:
            wp.launch(
                _batch_ewald_real_space_energy_forces_kernel_overload[wp_scalar],
                dim=[num_atoms],
                inputs=[
                    wp_positions,
                    wp_charges,
                    wp_cell,
                    wp_batch_idx,
                    wp_idx_j,
                    wp_neighbor_ptr,
                    wp_unit_shifts,
                    wp_alpha,
                    compute_virial,
                    wp_energies,
                    wp_forces,
                    wp_virial,
                ],
                device=device,
            )

    if needs_grad_flag:
        backward_kw = dict(
            energies=wp_energies,
            forces=wp_forces,
            positions=wp_positions,
            charges=wp_charges,
            cell=wp_cell,
            alpha=wp_alpha,
        )
        if virial_grad:
            backward_kw["virial"] = wp_virial
        attach_for_backward(energies, tape=tape, **backward_kw)
    return energies, forces, virial


@warp_custom_op(
    name="alchemiops::_batch_ewald_real_space_energy_matrix",
    outputs=[OutputSpec("energies", wp.float64, lambda pos, *_: (pos.shape[0],))],
    grad_arrays=["energies", "positions", "charges", "cell", "alpha"],
)
def _batch_ewald_real_space_energy_matrix(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    alpha: torch.Tensor,
    batch_idx: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    mask_value: int,
) -> torch.Tensor:
    """Internal: Compute real-space Ewald energies (batch, neighbor matrix)."""
    num_atoms = positions.shape[0]
    input_dtype = positions.dtype
    empty_nm = neighbor_matrix.shape[0] == 0

    device = wp.device_from_torch(positions.device)
    needs_grad_flag = needs_grad(positions, charges, cell)

    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)

    wp_positions = warp_from_torch(positions, wp_vec, requires_grad=needs_grad_flag)
    wp_charges = warp_from_torch(charges, wp_scalar, requires_grad=needs_grad_flag)
    wp_cell = warp_from_torch(cell, wp_mat, requires_grad=needs_grad_flag)
    wp_alpha = warp_from_torch(alpha, wp_scalar, requires_grad=needs_grad_flag)
    wp_batch_idx = warp_from_torch(batch_idx, wp.int32)
    wp_neighbor_matrix = warp_from_torch(neighbor_matrix, wp.int32)
    wp_unit_shifts_matrix = warp_from_torch(neighbor_matrix_shifts, wp.vec3i)

    energies = torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)
    wp_energies = warp_from_torch(energies, wp.float64, requires_grad=needs_grad_flag)

    with WarpAutogradContextManager(needs_grad_flag) as tape:
        if not empty_nm:
            wp.launch(
                _batch_ewald_real_space_energy_neighbor_matrix_kernel_overload[
                    wp_scalar
                ],
                dim=[neighbor_matrix.shape[0]],
                inputs=[
                    wp_positions,
                    wp_charges,
                    wp_cell,
                    wp_batch_idx,
                    wp_neighbor_matrix,
                    wp_unit_shifts_matrix,
                    wp.int32(mask_value),
                    wp_alpha,
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
            alpha=wp_alpha,
        )
    return energies


@warp_custom_op(
    name="alchemiops::_batch_ewald_real_space_energy_forces_matrix",
    outputs=[
        OutputSpec("energies", wp.float64, lambda pos, *_: (pos.shape[0],)),
        OutputSpec(
            "forces",
            lambda pos, *_: get_wp_vec_dtype(pos.dtype),
            lambda pos, *_: (pos.shape[0], 3),
        ),
        OutputSpec(
            "virial",
            lambda pos, *_: get_wp_mat_dtype(pos.dtype),
            lambda pos, charges, cell, *_: (cell.shape[0], 3, 3),
        ),
    ],
    grad_arrays=[
        "energies",
        "forces",
        "virial",
        "positions",
        "charges",
        "cell",
        "alpha",
    ],
)
def _batch_ewald_real_space_energy_forces_matrix(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    alpha: torch.Tensor,
    batch_idx: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    mask_value: int,
    compute_virial: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Internal: Compute real-space Ewald energies, forces, and optionally virial (batch, matrix)."""
    num_atoms = positions.shape[0]
    input_dtype = positions.dtype
    empty_nm = neighbor_matrix.shape[0] == 0

    device = wp.device_from_torch(positions.device)
    needs_grad_flag = needs_grad(positions, charges, cell)
    virial_grad = needs_grad_flag and compute_virial

    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)

    wp_positions = warp_from_torch(positions, wp_vec, requires_grad=needs_grad_flag)
    wp_charges = warp_from_torch(charges, wp_scalar, requires_grad=needs_grad_flag)
    wp_cell = warp_from_torch(cell, wp_mat, requires_grad=needs_grad_flag)
    wp_alpha = warp_from_torch(alpha, wp_scalar, requires_grad=needs_grad_flag)
    wp_batch_idx = warp_from_torch(batch_idx, wp.int32)
    wp_neighbor_matrix = warp_from_torch(neighbor_matrix, wp.int32)
    wp_unit_shifts_matrix = warp_from_torch(neighbor_matrix_shifts, wp.vec3i)

    num_systems = cell.shape[0]
    energies = torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)
    forces = torch.zeros(num_atoms, 3, device=positions.device, dtype=input_dtype)
    virial = torch.zeros(num_systems, 3, 3, device=positions.device, dtype=input_dtype)
    wp_energies = warp_from_torch(energies, wp.float64, requires_grad=needs_grad_flag)
    wp_forces = warp_from_torch(forces, wp_vec, requires_grad=needs_grad_flag)
    wp_virial = warp_from_torch(virial, wp_mat, requires_grad=virial_grad)

    with WarpAutogradContextManager(needs_grad_flag) as tape:
        if not empty_nm:
            wp.launch(
                _batch_ewald_real_space_energy_forces_neighbor_matrix_kernel_overload[
                    wp_scalar
                ],
                dim=[neighbor_matrix.shape[0]],
                inputs=[
                    wp_positions,
                    wp_charges,
                    wp_cell,
                    wp_batch_idx,
                    wp_neighbor_matrix,
                    wp_unit_shifts_matrix,
                    wp.int32(mask_value),
                    wp_alpha,
                    compute_virial,
                    wp_energies,
                    wp_forces,
                    wp_virial,
                ],
                device=device,
            )

    if needs_grad_flag:
        backward_kw = dict(
            energies=wp_energies,
            forces=wp_forces,
            positions=wp_positions,
            charges=wp_charges,
            cell=wp_cell,
            alpha=wp_alpha,
        )
        if virial_grad:
            backward_kw["virial"] = wp_virial
        attach_for_backward(energies, tape=tape, **backward_kw)
    return energies, forces, virial


###########################################################################################
################ Batch Real-Space with Charge Gradients Internal Custom Ops ###############
###########################################################################################


@warp_custom_op(
    name="alchemiops::_batch_ewald_real_space_energy_forces_charge_grad",
    outputs=[
        OutputSpec("energies", wp.float64, lambda pos, *_: (pos.shape[0],)),
        OutputSpec(
            "forces",
            lambda pos, *_: get_wp_vec_dtype(pos.dtype),
            lambda pos, *_: (pos.shape[0], 3),
        ),
        OutputSpec("charge_gradients", wp.float64, lambda pos, *_: (pos.shape[0],)),
        OutputSpec(
            "virial",
            lambda pos, *_: get_wp_mat_dtype(pos.dtype),
            lambda pos, charges, cell, *_: (cell.shape[0], 3, 3),
        ),
    ],
    grad_arrays=[
        "energies",
        "forces",
        "charge_gradients",
        "virial",
        "positions",
        "charges",
        "cell",
        "alpha",
    ],
)
def _batch_ewald_real_space_energy_forces_charge_grad(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    alpha: torch.Tensor,
    batch_idx: torch.Tensor,
    neighbor_list: torch.Tensor,
    neighbor_ptr: torch.Tensor,
    neighbor_shifts: torch.Tensor,
    compute_virial: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Internal: Compute real-space Ewald E+F+charge_grad+virial (batch, CSR)."""
    num_atoms = positions.shape[0]
    input_dtype = positions.dtype
    empty_nl = neighbor_list.shape[1] == 0

    idx_j = neighbor_list[1]
    device = wp.device_from_torch(positions.device)
    needs_grad_flag = needs_grad(positions, charges, cell)
    virial_grad = needs_grad_flag and compute_virial

    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)

    wp_positions = warp_from_torch(positions, wp_vec, requires_grad=needs_grad_flag)
    wp_charges = warp_from_torch(charges, wp_scalar, requires_grad=needs_grad_flag)
    wp_cell = warp_from_torch(cell, wp_mat, requires_grad=needs_grad_flag)
    wp_alpha = warp_from_torch(alpha, wp_scalar, requires_grad=needs_grad_flag)
    wp_batch_idx = warp_from_torch(batch_idx, wp.int32)
    wp_idx_j = warp_from_torch(idx_j, wp.int32)
    wp_neighbor_ptr = warp_from_torch(neighbor_ptr, wp.int32)
    wp_unit_shifts = warp_from_torch(neighbor_shifts, wp.vec3i)

    num_systems = cell.shape[0]
    energies = torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)
    forces = torch.zeros(num_atoms, 3, device=positions.device, dtype=input_dtype)
    charge_grads = torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)
    virial = torch.zeros(num_systems, 3, 3, device=positions.device, dtype=input_dtype)
    wp_energies = warp_from_torch(energies, wp.float64, requires_grad=needs_grad_flag)
    wp_forces = warp_from_torch(forces, wp_vec, requires_grad=needs_grad_flag)
    wp_charge_grads = warp_from_torch(
        charge_grads, wp.float64, requires_grad=needs_grad_flag
    )
    wp_virial = warp_from_torch(virial, wp_mat, requires_grad=virial_grad)

    with WarpAutogradContextManager(needs_grad_flag) as tape:
        if not empty_nl:
            wp.launch(
                _batch_ewald_real_space_energy_forces_charge_grad_kernel_overload[
                    wp_scalar
                ],
                dim=[num_atoms],
                inputs=[
                    wp_positions,
                    wp_charges,
                    wp_cell,
                    wp_batch_idx,
                    wp_idx_j,
                    wp_neighbor_ptr,
                    wp_unit_shifts,
                    wp_alpha,
                    compute_virial,
                    wp_energies,
                    wp_forces,
                    wp_charge_grads,
                    wp_virial,
                ],
                device=device,
            )

    if needs_grad_flag:
        backward_kw = dict(
            energies=wp_energies,
            forces=wp_forces,
            charge_gradients=wp_charge_grads,
            positions=wp_positions,
            charges=wp_charges,
            cell=wp_cell,
            alpha=wp_alpha,
        )
        if virial_grad:
            backward_kw["virial"] = wp_virial
        attach_for_backward(energies, tape=tape, **backward_kw)
    return energies, forces, charge_grads, virial


@warp_custom_op(
    name="alchemiops::_batch_ewald_real_space_energy_forces_charge_grad_matrix",
    outputs=[
        OutputSpec("energies", wp.float64, lambda pos, *_: (pos.shape[0],)),
        OutputSpec(
            "forces",
            lambda pos, *_: get_wp_vec_dtype(pos.dtype),
            lambda pos, *_: (pos.shape[0], 3),
        ),
        OutputSpec("charge_gradients", wp.float64, lambda pos, *_: (pos.shape[0],)),
        OutputSpec(
            "virial",
            lambda pos, *_: get_wp_mat_dtype(pos.dtype),
            lambda pos, charges, cell, *_: (cell.shape[0], 3, 3),
        ),
    ],
    grad_arrays=[
        "energies",
        "forces",
        "charge_gradients",
        "virial",
        "positions",
        "charges",
        "cell",
        "alpha",
    ],
)
def _batch_ewald_real_space_energy_forces_charge_grad_matrix(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    alpha: torch.Tensor,
    batch_idx: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    mask_value: int,
    compute_virial: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Internal: Compute real-space Ewald E+F+charge_grad+virial (batch, matrix)."""
    num_atoms = positions.shape[0]
    input_dtype = positions.dtype
    empty_nm = neighbor_matrix.shape[0] == 0

    device = wp.device_from_torch(positions.device)
    needs_grad_flag = needs_grad(positions, charges, cell)
    virial_grad = needs_grad_flag and compute_virial

    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)

    wp_positions = warp_from_torch(positions, wp_vec, requires_grad=needs_grad_flag)
    wp_charges = warp_from_torch(charges, wp_scalar, requires_grad=needs_grad_flag)
    wp_cell = warp_from_torch(cell, wp_mat, requires_grad=needs_grad_flag)
    wp_alpha = warp_from_torch(alpha, wp_scalar, requires_grad=needs_grad_flag)
    wp_batch_idx = warp_from_torch(batch_idx, wp.int32)
    wp_neighbor_matrix = warp_from_torch(neighbor_matrix, wp.int32)
    wp_unit_shifts_matrix = warp_from_torch(neighbor_matrix_shifts, wp.vec3i)

    num_systems = cell.shape[0]
    energies = torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)
    forces = torch.zeros(num_atoms, 3, device=positions.device, dtype=input_dtype)
    charge_grads = torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)
    virial = torch.zeros(num_systems, 3, 3, device=positions.device, dtype=input_dtype)
    wp_energies = warp_from_torch(energies, wp.float64, requires_grad=needs_grad_flag)
    wp_forces = warp_from_torch(forces, wp_vec, requires_grad=needs_grad_flag)
    wp_charge_grads = warp_from_torch(
        charge_grads, wp.float64, requires_grad=needs_grad_flag
    )
    wp_virial = warp_from_torch(virial, wp_mat, requires_grad=virial_grad)

    with WarpAutogradContextManager(needs_grad_flag) as tape:
        if not empty_nm:
            wp.launch(
                _batch_ewald_real_space_energy_forces_charge_grad_neighbor_matrix_kernel_overload[
                    wp_scalar
                ],
                dim=[neighbor_matrix.shape[0]],
                inputs=[
                    wp_positions,
                    wp_charges,
                    wp_cell,
                    wp_batch_idx,
                    wp_neighbor_matrix,
                    wp_unit_shifts_matrix,
                    wp.int32(mask_value),
                    wp_alpha,
                    compute_virial,
                    wp_energies,
                    wp_forces,
                    wp_charge_grads,
                    wp_virial,
                ],
                device=device,
            )

    if needs_grad_flag:
        backward_kw = dict(
            energies=wp_energies,
            forces=wp_forces,
            charge_gradients=wp_charge_grads,
            positions=wp_positions,
            charges=wp_charges,
            cell=wp_cell,
            alpha=wp_alpha,
        )
        if virial_grad:
            backward_kw["virial"] = wp_virial
        attach_for_backward(energies, tape=tape, **backward_kw)
    return energies, forces, charge_grads, virial


###########################################################################################
########################### Reciprocal-Space Internal Custom Ops ##########################
###########################################################################################


@warp_custom_op(
    name="alchemiops::_ewald_reciprocal_space_energy",
    outputs=[
        OutputSpec("energies", wp.float64, lambda pos, *_: (pos.shape[0],)),
        OutputSpec(
            "virial",
            lambda pos, *_: get_wp_mat_dtype(pos.dtype),
            lambda pos, charges, cell, *_: (cell.shape[0], 3, 3),
        ),
    ],
    grad_arrays=[
        "energies",
        "virial",
        "positions",
        "charges",
        "cell",
        "k_vectors",
        "alpha",
    ],
)
def _ewald_reciprocal_space_energy(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    k_vectors: torch.Tensor,
    alpha: torch.Tensor,
    compute_virial: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Internal: Compute reciprocal-space Ewald energies and optionally virial (single)."""
    num_k = k_vectors.shape[0]
    num_atoms = positions.shape[0]
    device = wp.device_from_torch(positions.device)
    needs_grad_flag = needs_grad(positions, charges, cell)
    virial_grad = needs_grad_flag and compute_virial

    input_dtype = positions.dtype
    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)

    wp_positions = warp_from_torch(positions, wp_vec, requires_grad=needs_grad_flag)
    wp_charges = warp_from_torch(charges, wp_scalar, requires_grad=needs_grad_flag)
    wp_cell = warp_from_torch(cell, wp_mat, requires_grad=needs_grad_flag)
    k_vectors_typed = k_vectors.to(input_dtype)
    wp_k_vectors = warp_from_torch(
        k_vectors_typed, wp_vec, requires_grad=needs_grad_flag
    )
    wp_alpha = warp_from_torch(alpha, wp_scalar, requires_grad=needs_grad_flag)

    # Intermediate arrays
    wp_cos_k_dot_r = warp_from_torch(
        torch.zeros((num_k, num_atoms), device=positions.device, dtype=torch.float64),
        wp.float64,
        requires_grad=needs_grad_flag,
    )
    wp_sin_k_dot_r = warp_from_torch(
        torch.zeros((num_k, num_atoms), device=positions.device, dtype=torch.float64),
        wp.float64,
        requires_grad=needs_grad_flag,
    )
    real_sf = torch.zeros(num_k, device=positions.device, dtype=torch.float64)
    imag_sf = torch.zeros(num_k, device=positions.device, dtype=torch.float64)
    wp_real_sf = warp_from_torch(real_sf, wp.float64, requires_grad=needs_grad_flag)
    wp_imag_sf = warp_from_torch(imag_sf, wp.float64, requires_grad=needs_grad_flag)
    total_charge = torch.zeros(1, device=positions.device, dtype=torch.float64)
    wp_total_charge = warp_from_torch(
        total_charge, wp.float64, requires_grad=needs_grad_flag
    )
    raw_energies = torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)
    wp_raw_energies = warp_from_torch(
        raw_energies, wp.float64, requires_grad=needs_grad_flag
    )
    energies = torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)
    wp_energies = warp_from_torch(energies, wp.float64, requires_grad=needs_grad_flag)

    wp_virial = None
    with WarpAutogradContextManager(needs_grad_flag) as tape:
        wp.launch(
            _ewald_reciprocal_space_energy_kernel_fill_structure_factors_overload[
                wp_scalar
            ],
            dim=num_k,
            inputs=[wp_positions, wp_charges, wp_k_vectors, wp_cell, wp_alpha],
            outputs=[
                wp_total_charge,
                wp_cos_k_dot_r,
                wp_sin_k_dot_r,
                wp_real_sf,
                wp_imag_sf,
            ],
            device=device,
        )
        wp.launch(
            _ewald_reciprocal_space_energy_kernel_compute_energy_overload[wp_scalar],
            dim=num_atoms,
            inputs=[wp_charges, wp_cos_k_dot_r, wp_sin_k_dot_r, wp_real_sf, wp_imag_sf],
            outputs=[wp_raw_energies],
            device=device,
        )
        wp.launch(
            _ewald_subtract_self_energy_kernel_overload[wp_scalar],
            dim=num_atoms,
            inputs=[wp_charges, wp_alpha, wp_total_charge, wp_raw_energies],
            outputs=[wp_energies],
            device=device,
        )
        if compute_virial:
            virial = torch.zeros(1, 3, 3, device=positions.device, dtype=input_dtype)
            wp_virial = warp_from_torch(virial, wp_mat, requires_grad=virial_grad)
            volume = torch.abs(torch.det(cell[0].to(torch.float64))).view(1)
            wp_volume = warp_from_torch(volume, wp.float64)
            wp.launch(
                _ewald_reciprocal_space_virial_kernel_overload[wp_scalar],
                dim=num_k,
                inputs=[
                    wp_k_vectors,
                    wp_alpha,
                    wp_volume,
                    wp_real_sf,
                    wp_imag_sf,
                    wp_virial,
                ],
                device=device,
            )
        else:
            virial = torch.zeros(1, 3, 3, device=positions.device, dtype=input_dtype)

    if needs_grad_flag:
        backward_kw = dict(
            energies=wp_energies,
            positions=wp_positions,
            charges=wp_charges,
            cell=wp_cell,
            k_vectors=wp_k_vectors,
            alpha=wp_alpha,
        )
        if virial_grad and wp_virial is not None:
            backward_kw["virial"] = wp_virial
        attach_for_backward(energies, tape=tape, **backward_kw)
    return energies, virial


@warp_custom_op(
    name="alchemiops::_ewald_reciprocal_space_energy_forces",
    outputs=[
        OutputSpec("energies", wp.float64, lambda pos, *_: (pos.shape[0],)),
        OutputSpec(
            "forces",
            lambda pos, *_: get_wp_vec_dtype(pos.dtype),
            lambda pos, *_: (pos.shape[0], 3),
        ),
        OutputSpec(
            "virial",
            lambda pos, *_: get_wp_mat_dtype(pos.dtype),
            lambda pos, charges, cell, *_: (cell.shape[0], 3, 3),
        ),
    ],
    grad_arrays=[
        "energies",
        "forces",
        "virial",
        "positions",
        "charges",
        "cell",
        "k_vectors",
        "alpha",
    ],
)
def _ewald_reciprocal_space_energy_forces(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    k_vectors: torch.Tensor,
    alpha: torch.Tensor,
    compute_virial: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Internal: Compute reciprocal-space Ewald energies, forces, and optionally virial (single)."""
    num_k = k_vectors.shape[0]
    num_atoms = positions.shape[0]
    device = wp.device_from_torch(positions.device)
    input_dtype = positions.dtype

    if num_k == 0 or num_atoms == 0:
        return (
            torch.zeros(num_atoms, device=positions.device, dtype=input_dtype),
            torch.zeros(num_atoms, 3, device=positions.device, dtype=input_dtype),
            torch.zeros(1, 3, 3, device=positions.device, dtype=input_dtype),
        )

    needs_grad_flag = needs_grad(positions, charges, cell)
    virial_grad = needs_grad_flag and compute_virial

    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)

    wp_positions = warp_from_torch(positions, wp_vec, requires_grad=needs_grad_flag)
    wp_charges = warp_from_torch(charges, wp_scalar, requires_grad=needs_grad_flag)
    wp_cell = warp_from_torch(cell, wp_mat, requires_grad=needs_grad_flag)
    k_vectors_typed = k_vectors.to(input_dtype)
    wp_k_vectors = warp_from_torch(
        k_vectors_typed, wp_vec, requires_grad=needs_grad_flag
    )
    wp_alpha = warp_from_torch(alpha, wp_scalar, requires_grad=needs_grad_flag)

    # Intermediate arrays
    wp_cos_k_dot_r = warp_from_torch(
        torch.zeros((num_k, num_atoms), device=positions.device, dtype=torch.float64),
        wp.float64,
        requires_grad=needs_grad_flag,
    )
    wp_sin_k_dot_r = warp_from_torch(
        torch.zeros((num_k, num_atoms), device=positions.device, dtype=torch.float64),
        wp.float64,
        requires_grad=needs_grad_flag,
    )
    real_sf = torch.zeros(num_k, device=positions.device, dtype=torch.float64)
    imag_sf = torch.zeros(num_k, device=positions.device, dtype=torch.float64)
    wp_real_sf = warp_from_torch(real_sf, wp.float64, requires_grad=needs_grad_flag)
    wp_imag_sf = warp_from_torch(imag_sf, wp.float64, requires_grad=needs_grad_flag)
    total_charge = torch.zeros(1, device=positions.device, dtype=torch.float64)
    wp_total_charge = warp_from_torch(
        total_charge, wp.float64, requires_grad=needs_grad_flag
    )
    wp_raw_energies = warp_from_torch(
        torch.zeros(num_atoms, device=positions.device, dtype=torch.float64),
        wp.float64,
        requires_grad=needs_grad_flag,
    )
    energies = torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)
    forces = torch.zeros(num_atoms, 3, device=positions.device, dtype=input_dtype)
    wp_energies = warp_from_torch(energies, wp.float64, requires_grad=needs_grad_flag)
    wp_forces = warp_from_torch(forces, wp_vec, requires_grad=needs_grad_flag)

    wp_virial = None
    with WarpAutogradContextManager(needs_grad_flag) as tape:
        wp.launch(
            _ewald_reciprocal_space_energy_kernel_fill_structure_factors_overload[
                wp_scalar
            ],
            dim=num_k,
            inputs=[wp_positions, wp_charges, wp_k_vectors, wp_cell, wp_alpha],
            outputs=[
                wp_total_charge,
                wp_cos_k_dot_r,
                wp_sin_k_dot_r,
                wp_real_sf,
                wp_imag_sf,
            ],
            device=device,
        )
        wp.launch(
            _ewald_reciprocal_space_energy_forces_kernel_overload[wp_scalar],
            dim=num_atoms,
            inputs=[
                wp_charges,
                wp_k_vectors,
                wp_cos_k_dot_r,
                wp_sin_k_dot_r,
                wp_real_sf,
                wp_imag_sf,
            ],
            outputs=[wp_raw_energies, wp_forces],
            device=device,
        )
        wp.launch(
            _ewald_subtract_self_energy_kernel_overload[wp_scalar],
            dim=num_atoms,
            inputs=[wp_charges, wp_alpha, wp_total_charge, wp_raw_energies],
            outputs=[wp_energies],
            device=device,
        )
        if compute_virial:
            virial = torch.zeros(1, 3, 3, device=positions.device, dtype=input_dtype)
            wp_virial = warp_from_torch(virial, wp_mat, requires_grad=virial_grad)
            volume = torch.abs(torch.det(cell[0].to(torch.float64))).view(1)
            wp_volume = warp_from_torch(volume, wp.float64)
            wp.launch(
                _ewald_reciprocal_space_virial_kernel_overload[wp_scalar],
                dim=num_k,
                inputs=[
                    wp_k_vectors,
                    wp_alpha,
                    wp_volume,
                    wp_real_sf,
                    wp_imag_sf,
                    wp_virial,
                ],
                device=device,
            )
        else:
            virial = torch.zeros(1, 3, 3, device=positions.device, dtype=input_dtype)

    if needs_grad_flag:
        backward_kw = dict(
            energies=wp_energies,
            forces=wp_forces,
            positions=wp_positions,
            charges=wp_charges,
            cell=wp_cell,
            k_vectors=wp_k_vectors,
            alpha=wp_alpha,
        )
        if virial_grad and wp_virial is not None:
            backward_kw["virial"] = wp_virial
        attach_for_backward(energies, tape=tape, **backward_kw)
    return energies, forces, virial


@warp_custom_op(
    name="alchemiops::_ewald_reciprocal_space_energy_forces_charge_grad",
    outputs=[
        OutputSpec("energies", wp.float64, lambda pos, *_: (pos.shape[0],)),
        OutputSpec(
            "forces",
            lambda pos, *_: get_wp_vec_dtype(pos.dtype),
            lambda pos, *_: (pos.shape[0], 3),
        ),
        OutputSpec("charge_gradients", wp.float64, lambda pos, *_: (pos.shape[0],)),
        OutputSpec(
            "virial",
            lambda pos, *_: get_wp_mat_dtype(pos.dtype),
            lambda pos, charges, cell, *_: (cell.shape[0], 3, 3),
        ),
    ],
    grad_arrays=[
        "energies",
        "forces",
        "charge_gradients",
        "virial",
        "positions",
        "charges",
        "cell",
        "k_vectors",
        "alpha",
    ],
)
def _ewald_reciprocal_space_energy_forces_charge_grad(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    k_vectors: torch.Tensor,
    alpha: torch.Tensor,
    compute_virial: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Internal: Compute reciprocal-space Ewald E+F+charge_grad+virial (single)."""
    num_k = k_vectors.shape[0]
    num_atoms = positions.shape[0]
    device = wp.device_from_torch(positions.device)
    input_dtype = positions.dtype

    if num_k == 0 or num_atoms == 0:
        return (
            torch.zeros(num_atoms, device=positions.device, dtype=input_dtype),
            torch.zeros(num_atoms, 3, device=positions.device, dtype=input_dtype),
            torch.zeros(num_atoms, device=positions.device, dtype=input_dtype),
            torch.zeros(1, 3, 3, device=positions.device, dtype=input_dtype),
        )

    needs_grad_flag = needs_grad(positions, charges, cell)
    virial_grad = needs_grad_flag and compute_virial

    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)

    wp_positions = warp_from_torch(positions, wp_vec, requires_grad=needs_grad_flag)
    wp_charges = warp_from_torch(charges, wp_scalar, requires_grad=needs_grad_flag)
    wp_cell = warp_from_torch(cell, wp_mat, requires_grad=needs_grad_flag)
    k_vectors_typed = k_vectors.to(input_dtype)
    wp_k_vectors = warp_from_torch(
        k_vectors_typed, wp_vec, requires_grad=needs_grad_flag
    )
    wp_alpha = warp_from_torch(alpha, wp_scalar, requires_grad=needs_grad_flag)

    # Intermediate arrays
    cos_k_dot_r = torch.zeros(
        num_k, num_atoms, device=positions.device, dtype=torch.float64
    )
    sin_k_dot_r = torch.zeros(
        num_k, num_atoms, device=positions.device, dtype=torch.float64
    )
    real_sf = torch.zeros(num_k, device=positions.device, dtype=torch.float64)
    imag_sf = torch.zeros(num_k, device=positions.device, dtype=torch.float64)
    wp_cos_k_dot_r = warp_from_torch(
        cos_k_dot_r, wp.float64, requires_grad=needs_grad_flag
    )
    wp_sin_k_dot_r = warp_from_torch(
        sin_k_dot_r, wp.float64, requires_grad=needs_grad_flag
    )
    wp_real_sf = warp_from_torch(real_sf, wp.float64, requires_grad=needs_grad_flag)
    wp_imag_sf = warp_from_torch(imag_sf, wp.float64, requires_grad=needs_grad_flag)
    total_charge = torch.zeros(1, device=positions.device, dtype=torch.float64)
    wp_total_charge = warp_from_torch(
        total_charge, wp.float64, requires_grad=needs_grad_flag
    )
    wp_raw_energies = warp_from_torch(
        torch.zeros(num_atoms, device=positions.device, dtype=torch.float64),
        wp.float64,
        requires_grad=needs_grad_flag,
    )
    energies = torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)
    forces = torch.zeros(num_atoms, 3, device=positions.device, dtype=input_dtype)
    charge_grads = torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)
    wp_energies = warp_from_torch(energies, wp.float64, requires_grad=needs_grad_flag)
    wp_forces = warp_from_torch(forces, wp_vec, requires_grad=needs_grad_flag)
    wp_charge_grads = warp_from_torch(
        charge_grads, wp.float64, requires_grad=needs_grad_flag
    )

    wp_virial = None
    with WarpAutogradContextManager(needs_grad_flag) as tape:
        wp.launch(
            _ewald_reciprocal_space_energy_kernel_fill_structure_factors_overload[
                wp_scalar
            ],
            dim=num_k,
            inputs=[wp_positions, wp_charges, wp_k_vectors, wp_cell, wp_alpha],
            outputs=[
                wp_total_charge,
                wp_cos_k_dot_r,
                wp_sin_k_dot_r,
                wp_real_sf,
                wp_imag_sf,
            ],
            device=device,
        )
        wp.launch(
            _ewald_reciprocal_space_energy_forces_charge_grad_kernel_overload[
                wp_scalar
            ],
            dim=num_atoms,
            inputs=[
                wp_charges,
                wp_k_vectors,
                wp_cos_k_dot_r,
                wp_sin_k_dot_r,
                wp_real_sf,
                wp_imag_sf,
            ],
            outputs=[wp_raw_energies, wp_forces, wp_charge_grads],
            device=device,
        )
        wp.launch(
            _ewald_subtract_self_energy_kernel_overload[wp_scalar],
            dim=num_atoms,
            inputs=[wp_charges, wp_alpha, wp_total_charge, wp_raw_energies],
            outputs=[wp_energies],
            device=device,
        )
        if compute_virial:
            virial = torch.zeros(1, 3, 3, device=positions.device, dtype=input_dtype)
            wp_virial = warp_from_torch(virial, wp_mat, requires_grad=virial_grad)
            volume = torch.abs(torch.det(cell[0].to(torch.float64))).view(1)
            wp_volume = warp_from_torch(volume, wp.float64)
            wp.launch(
                _ewald_reciprocal_space_virial_kernel_overload[wp_scalar],
                dim=num_k,
                inputs=[
                    wp_k_vectors,
                    wp_alpha,
                    wp_volume,
                    wp_real_sf,
                    wp_imag_sf,
                    wp_virial,
                ],
                device=device,
            )
        else:
            virial = torch.zeros(1, 3, 3, device=positions.device, dtype=input_dtype)

    # Apply self-energy and background corrections to charge gradients
    alpha_val = alpha[0].item()
    self_energy_grad = 2.0 * alpha_val / math.sqrt(math.pi) * charges
    background_grad = math.pi / (alpha_val * alpha_val) * total_charge[0]
    charge_grads = charge_grads - self_energy_grad - background_grad

    if needs_grad_flag:
        backward_kw = dict(
            energies=wp_energies,
            forces=wp_forces,
            charge_gradients=wp_charge_grads,
            positions=wp_positions,
            charges=wp_charges,
            cell=wp_cell,
            k_vectors=wp_k_vectors,
            alpha=wp_alpha,
        )
        if virial_grad and wp_virial is not None:
            backward_kw["virial"] = wp_virial
        attach_for_backward(energies, tape=tape, **backward_kw)
    return energies, forces, charge_grads.to(input_dtype), virial


###########################################################################################
########################### Batch Reciprocal-Space Internal Custom Ops ####################
###########################################################################################


@warp_custom_op(
    name="alchemiops::_batch_ewald_reciprocal_space_energy",
    outputs=[
        OutputSpec("energies", wp.float64, lambda pos, *_: (pos.shape[0],)),
        OutputSpec(
            "virial",
            lambda pos, *_: get_wp_mat_dtype(pos.dtype),
            lambda pos, charges, cell, *_: (cell.shape[0], 3, 3),
        ),
    ],
    grad_arrays=[
        "energies",
        "virial",
        "positions",
        "charges",
        "cell",
        "k_vectors",
        "alpha",
    ],
)
def _batch_ewald_reciprocal_space_energy(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    k_vectors: torch.Tensor,
    alpha: torch.Tensor,
    batch_idx: torch.Tensor,
    compute_virial: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Internal: Compute reciprocal-space Ewald energies and optionally virial (batch)."""
    num_k = k_vectors.shape[1]
    num_atoms = positions.shape[0]
    num_systems = cell.shape[0]
    device = wp.device_from_torch(positions.device)
    input_dtype = positions.dtype

    if num_k == 0 or num_atoms == 0:
        return (
            torch.zeros(num_atoms, device=positions.device, dtype=input_dtype),
            torch.zeros(num_systems, 3, 3, device=positions.device, dtype=input_dtype),
        )

    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)

    atom_counts = torch.bincount(batch_idx, minlength=num_systems)
    atom_end = torch.cumsum(atom_counts, dim=0).to(torch.int32)
    atom_start = torch.cat(
        [torch.zeros(1, device=positions.device, dtype=torch.int32), atom_end[:-1]]
    )
    max_atoms_per_system = atom_counts.max().item()
    max_blocks_per_system = (
        max_atoms_per_system + BATCH_BLOCK_SIZE - 1
    ) // BATCH_BLOCK_SIZE

    needs_grad_flag = needs_grad(positions, charges, cell)
    virial_grad = needs_grad_flag and compute_virial

    wp_positions = warp_from_torch(positions, wp_vec, requires_grad=needs_grad_flag)
    wp_charges = warp_from_torch(charges, wp_scalar, requires_grad=needs_grad_flag)
    wp_cell = warp_from_torch(cell, wp_mat, requires_grad=needs_grad_flag)
    k_vectors_typed = k_vectors.to(input_dtype)
    wp_k_vectors = warp_from_torch(
        k_vectors_typed, wp_vec, requires_grad=needs_grad_flag
    )
    wp_alpha = warp_from_torch(alpha, wp_scalar, requires_grad=needs_grad_flag)
    wp_batch_idx = warp_from_torch(batch_idx, wp.int32)
    wp_atom_start = warp_from_torch(atom_start, wp.int32)
    wp_atom_end = warp_from_torch(atom_end, wp.int32)

    # Intermediate arrays
    wp_cos_k_dot_r = warp_from_torch(
        torch.zeros((num_k, num_atoms), device=positions.device, dtype=torch.float64),
        wp.float64,
        requires_grad=needs_grad_flag,
    )
    wp_sin_k_dot_r = warp_from_torch(
        torch.zeros((num_k, num_atoms), device=positions.device, dtype=torch.float64),
        wp.float64,
        requires_grad=needs_grad_flag,
    )
    real_sf = torch.zeros(
        (num_systems, num_k), device=positions.device, dtype=torch.float64
    )
    imag_sf = torch.zeros(
        (num_systems, num_k), device=positions.device, dtype=torch.float64
    )
    wp_real_sf = warp_from_torch(real_sf, wp.float64, requires_grad=needs_grad_flag)
    wp_imag_sf = warp_from_torch(imag_sf, wp.float64, requires_grad=needs_grad_flag)
    wp_total_charge = warp_from_torch(
        torch.zeros(num_systems, device=positions.device, dtype=torch.float64),
        wp.float64,
        requires_grad=needs_grad_flag,
    )
    raw_energies = torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)
    wp_raw_energies = warp_from_torch(
        raw_energies, wp.float64, requires_grad=needs_grad_flag
    )
    energies = torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)
    wp_energies = warp_from_torch(energies, wp.float64, requires_grad=needs_grad_flag)

    wp_virial = None
    with WarpAutogradContextManager(needs_grad_flag) as tape:
        wp.launch(
            _batch_ewald_reciprocal_space_energy_kernel_fill_structure_factors_overload[
                wp_scalar
            ],
            dim=(num_k, num_systems, max_blocks_per_system),
            inputs=[
                wp_positions,
                wp_charges,
                wp_k_vectors,
                wp_cell,
                wp_alpha,
                wp_atom_start,
                wp_atom_end,
            ],
            outputs=[
                wp_total_charge,
                wp_cos_k_dot_r,
                wp_sin_k_dot_r,
                wp_real_sf,
                wp_imag_sf,
            ],
            device=device,
        )
        wp.launch(
            _batch_ewald_reciprocal_space_energy_kernel_compute_energy_overload[
                wp_scalar
            ],
            dim=num_atoms,
            inputs=[
                wp_charges,
                wp_batch_idx,
                wp_cos_k_dot_r,
                wp_sin_k_dot_r,
                wp_real_sf,
                wp_imag_sf,
            ],
            outputs=[wp_raw_energies],
            device=device,
        )
        wp.launch(
            _batch_ewald_subtract_self_energy_kernel_overload[wp_scalar],
            dim=num_atoms,
            inputs=[
                wp_charges,
                wp_batch_idx,
                wp_alpha,
                wp_total_charge,
                wp_raw_energies,
            ],
            outputs=[wp_energies],
            device=device,
        )
        if compute_virial:
            virial = torch.zeros(
                num_systems, 3, 3, device=positions.device, dtype=input_dtype
            )
            wp_virial = warp_from_torch(virial, wp_mat, requires_grad=virial_grad)
            volume = torch.abs(torch.det(cell.to(torch.float64)))
            wp_volume = warp_from_torch(volume, wp.float64)
            wp.launch(
                _batch_ewald_reciprocal_space_virial_kernel_overload[wp_scalar],
                dim=(num_k, num_systems),
                inputs=[
                    wp_k_vectors,
                    wp_alpha,
                    wp_volume,
                    wp_real_sf,
                    wp_imag_sf,
                    wp_virial,
                ],
                device=device,
            )
        else:
            virial = torch.zeros(
                num_systems, 3, 3, device=positions.device, dtype=input_dtype
            )

    if needs_grad_flag:
        backward_kw = dict(
            energies=wp_energies,
            positions=wp_positions,
            charges=wp_charges,
            cell=wp_cell,
            k_vectors=wp_k_vectors,
            alpha=wp_alpha,
        )
        if virial_grad and wp_virial is not None:
            backward_kw["virial"] = wp_virial
        attach_for_backward(energies, tape=tape, **backward_kw)
    return energies, virial


@warp_custom_op(
    name="alchemiops::_batch_ewald_reciprocal_space_energy_forces",
    outputs=[
        OutputSpec("energies", wp.float64, lambda pos, *_: (pos.shape[0],)),
        OutputSpec(
            "forces",
            lambda pos, *_: get_wp_vec_dtype(pos.dtype),
            lambda pos, *_: (pos.shape[0], 3),
        ),
        OutputSpec(
            "virial",
            lambda pos, *_: get_wp_mat_dtype(pos.dtype),
            lambda pos, charges, cell, *_: (cell.shape[0], 3, 3),
        ),
    ],
    grad_arrays=[
        "energies",
        "forces",
        "virial",
        "positions",
        "charges",
        "cell",
        "k_vectors",
        "alpha",
    ],
)
def _batch_ewald_reciprocal_space_energy_forces(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    k_vectors: torch.Tensor,
    alpha: torch.Tensor,
    batch_idx: torch.Tensor,
    compute_virial: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Internal: Compute reciprocal-space Ewald energies, forces, and optionally virial (batch)."""
    num_k = k_vectors.shape[1]
    num_atoms = positions.shape[0]
    num_systems = cell.shape[0]
    device = wp.device_from_torch(positions.device)
    input_dtype = positions.dtype

    if num_k == 0 or num_atoms == 0:
        return (
            torch.zeros(num_atoms, device=positions.device, dtype=input_dtype),
            torch.zeros(num_atoms, 3, device=positions.device, dtype=input_dtype),
            torch.zeros(num_systems, 3, 3, device=positions.device, dtype=input_dtype),
        )

    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)

    atom_counts = torch.bincount(batch_idx, minlength=num_systems)
    atom_end = torch.cumsum(atom_counts, dim=0).to(torch.int32)
    atom_start = torch.cat(
        [torch.zeros(1, device=positions.device, dtype=torch.int32), atom_end[:-1]]
    )
    max_atoms_per_system = atom_counts.max().item()
    max_blocks_per_system = (
        max_atoms_per_system + BATCH_BLOCK_SIZE - 1
    ) // BATCH_BLOCK_SIZE

    needs_grad_flag = needs_grad(positions, charges, cell)
    virial_grad = needs_grad_flag and compute_virial

    wp_positions = warp_from_torch(positions, wp_vec, requires_grad=needs_grad_flag)
    wp_charges = warp_from_torch(charges, wp_scalar, requires_grad=needs_grad_flag)
    wp_cell = warp_from_torch(cell, wp_mat, requires_grad=needs_grad_flag)
    k_vectors_typed = k_vectors.to(input_dtype)
    wp_k_vectors = warp_from_torch(
        k_vectors_typed, wp_vec, requires_grad=needs_grad_flag
    )
    wp_alpha = warp_from_torch(alpha, wp_scalar, requires_grad=needs_grad_flag)
    wp_batch_idx = warp_from_torch(batch_idx, wp.int32)
    wp_atom_start = warp_from_torch(atom_start, wp.int32)
    wp_atom_end = warp_from_torch(atom_end, wp.int32)

    # Intermediate arrays
    wp_cos_k_dot_r = warp_from_torch(
        torch.zeros((num_k, num_atoms), device=positions.device, dtype=torch.float64),
        wp.float64,
        requires_grad=needs_grad_flag,
    )
    wp_sin_k_dot_r = warp_from_torch(
        torch.zeros((num_k, num_atoms), device=positions.device, dtype=torch.float64),
        wp.float64,
        requires_grad=needs_grad_flag,
    )
    real_sf = torch.zeros(
        (num_systems, num_k), device=positions.device, dtype=torch.float64
    )
    imag_sf = torch.zeros(
        (num_systems, num_k), device=positions.device, dtype=torch.float64
    )
    wp_real_sf = warp_from_torch(real_sf, wp.float64, requires_grad=needs_grad_flag)
    wp_imag_sf = warp_from_torch(imag_sf, wp.float64, requires_grad=needs_grad_flag)
    wp_total_charge = warp_from_torch(
        torch.zeros(num_systems, device=positions.device, dtype=torch.float64),
        wp.float64,
        requires_grad=needs_grad_flag,
    )
    raw_energies = torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)
    wp_raw_energies = warp_from_torch(
        raw_energies, wp.float64, requires_grad=needs_grad_flag
    )
    energies = torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)
    forces = torch.zeros(num_atoms, 3, device=positions.device, dtype=input_dtype)
    wp_energies = warp_from_torch(energies, wp.float64, requires_grad=needs_grad_flag)
    wp_forces = warp_from_torch(forces, wp_vec, requires_grad=needs_grad_flag)

    wp_virial = None
    with WarpAutogradContextManager(needs_grad_flag) as tape:
        wp.launch(
            _batch_ewald_reciprocal_space_energy_kernel_fill_structure_factors_overload[
                wp_scalar
            ],
            dim=(num_k, num_systems, max_blocks_per_system),
            inputs=[
                wp_positions,
                wp_charges,
                wp_k_vectors,
                wp_cell,
                wp_alpha,
                wp_atom_start,
                wp_atom_end,
            ],
            outputs=[
                wp_total_charge,
                wp_cos_k_dot_r,
                wp_sin_k_dot_r,
                wp_real_sf,
                wp_imag_sf,
            ],
            device=device,
        )
        wp.launch(
            _batch_ewald_reciprocal_space_energy_forces_kernel_overload[wp_scalar],
            dim=num_atoms,
            inputs=[
                wp_charges,
                wp_batch_idx,
                wp_k_vectors,
                wp_cos_k_dot_r,
                wp_sin_k_dot_r,
                wp_real_sf,
                wp_imag_sf,
            ],
            outputs=[wp_raw_energies, wp_forces],
            device=device,
        )
        wp.launch(
            _batch_ewald_subtract_self_energy_kernel_overload[wp_scalar],
            dim=num_atoms,
            inputs=[
                wp_charges,
                wp_batch_idx,
                wp_alpha,
                wp_total_charge,
                wp_raw_energies,
            ],
            outputs=[wp_energies],
            device=device,
        )
        if compute_virial:
            virial = torch.zeros(
                num_systems, 3, 3, device=positions.device, dtype=input_dtype
            )
            wp_virial = warp_from_torch(virial, wp_mat, requires_grad=virial_grad)
            volume = torch.abs(torch.det(cell.to(torch.float64)))
            wp_volume = warp_from_torch(volume, wp.float64)
            wp.launch(
                _batch_ewald_reciprocal_space_virial_kernel_overload[wp_scalar],
                dim=(num_k, num_systems),
                inputs=[
                    wp_k_vectors,
                    wp_alpha,
                    wp_volume,
                    wp_real_sf,
                    wp_imag_sf,
                    wp_virial,
                ],
                device=device,
            )
        else:
            virial = torch.zeros(
                num_systems, 3, 3, device=positions.device, dtype=input_dtype
            )

    if needs_grad_flag:
        backward_kw = dict(
            energies=wp_energies,
            forces=wp_forces,
            positions=wp_positions,
            charges=wp_charges,
            cell=wp_cell,
            k_vectors=wp_k_vectors,
            alpha=wp_alpha,
        )
        if virial_grad and wp_virial is not None:
            backward_kw["virial"] = wp_virial
        attach_for_backward(energies, tape=tape, **backward_kw)
    return energies, forces, virial


@warp_custom_op(
    name="alchemiops::_batch_ewald_reciprocal_space_energy_forces_charge_grad",
    outputs=[
        OutputSpec("energies", wp.float64, lambda pos, *_: (pos.shape[0],)),
        OutputSpec(
            "forces",
            lambda pos, *_: get_wp_vec_dtype(pos.dtype),
            lambda pos, *_: (pos.shape[0], 3),
        ),
        OutputSpec("charge_gradients", wp.float64, lambda pos, *_: (pos.shape[0],)),
        OutputSpec(
            "virial",
            lambda pos, *_: get_wp_mat_dtype(pos.dtype),
            lambda pos, charges, cell, *_: (cell.shape[0], 3, 3),
        ),
    ],
    grad_arrays=[
        "energies",
        "forces",
        "charge_gradients",
        "virial",
        "positions",
        "charges",
        "cell",
        "k_vectors",
        "alpha",
    ],
)
def _batch_ewald_reciprocal_space_energy_forces_charge_grad(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    k_vectors: torch.Tensor,
    alpha: torch.Tensor,
    batch_idx: torch.Tensor,
    compute_virial: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Internal: Compute reciprocal-space Ewald E+F+charge_grad+virial (batch)."""
    num_k = k_vectors.shape[1]
    num_atoms = positions.shape[0]
    num_systems = cell.shape[0]
    device = wp.device_from_torch(positions.device)
    input_dtype = positions.dtype

    if num_k == 0 or num_atoms == 0:
        return (
            torch.zeros(num_atoms, device=positions.device, dtype=input_dtype),
            torch.zeros(num_atoms, 3, device=positions.device, dtype=input_dtype),
            torch.zeros(num_atoms, device=positions.device, dtype=input_dtype),
            torch.zeros(num_systems, 3, 3, device=positions.device, dtype=input_dtype),
        )

    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)

    atom_counts = torch.bincount(batch_idx, minlength=num_systems)
    atom_end = torch.cumsum(atom_counts, dim=0).to(torch.int32)
    atom_start = torch.cat(
        [torch.zeros(1, device=positions.device, dtype=torch.int32), atom_end[:-1]]
    )
    max_atoms_per_system = atom_counts.max().item()
    max_blocks_per_system = (
        max_atoms_per_system + BATCH_BLOCK_SIZE - 1
    ) // BATCH_BLOCK_SIZE

    needs_grad_flag = needs_grad(positions, charges, cell)
    virial_grad = needs_grad_flag and compute_virial

    wp_positions = warp_from_torch(positions, wp_vec, requires_grad=needs_grad_flag)
    wp_charges = warp_from_torch(charges, wp_scalar, requires_grad=needs_grad_flag)
    wp_cell = warp_from_torch(cell, wp_mat, requires_grad=needs_grad_flag)
    k_vectors_typed = k_vectors.to(input_dtype)
    wp_k_vectors = warp_from_torch(
        k_vectors_typed, wp_vec, requires_grad=needs_grad_flag
    )
    wp_alpha = warp_from_torch(alpha, wp_scalar, requires_grad=needs_grad_flag)
    wp_batch_idx = warp_from_torch(batch_idx, wp.int32)
    wp_atom_start = warp_from_torch(atom_start, wp.int32)
    wp_atom_end = warp_from_torch(atom_end, wp.int32)

    # Intermediate arrays
    wp_cos_k_dot_r = warp_from_torch(
        torch.zeros((num_k, num_atoms), device=positions.device, dtype=torch.float64),
        wp.float64,
        requires_grad=needs_grad_flag,
    )
    wp_sin_k_dot_r = warp_from_torch(
        torch.zeros((num_k, num_atoms), device=positions.device, dtype=torch.float64),
        wp.float64,
        requires_grad=needs_grad_flag,
    )
    real_sf = torch.zeros(
        (num_systems, num_k), device=positions.device, dtype=torch.float64
    )
    imag_sf = torch.zeros(
        (num_systems, num_k), device=positions.device, dtype=torch.float64
    )
    wp_real_sf = warp_from_torch(real_sf, wp.float64, requires_grad=needs_grad_flag)
    wp_imag_sf = warp_from_torch(imag_sf, wp.float64, requires_grad=needs_grad_flag)
    total_charge_batch = torch.zeros(
        num_systems, device=positions.device, dtype=torch.float64
    )
    wp_total_charge = warp_from_torch(
        total_charge_batch,
        wp.float64,
        requires_grad=needs_grad_flag,
    )
    raw_energies = torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)
    wp_raw_energies = warp_from_torch(
        raw_energies, wp.float64, requires_grad=needs_grad_flag
    )
    energies = torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)
    forces = torch.zeros(num_atoms, 3, device=positions.device, dtype=input_dtype)
    charge_grads = torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)
    wp_energies = warp_from_torch(energies, wp.float64, requires_grad=needs_grad_flag)
    wp_forces = warp_from_torch(forces, wp_vec, requires_grad=needs_grad_flag)
    wp_charge_grads = warp_from_torch(
        charge_grads, wp.float64, requires_grad=needs_grad_flag
    )

    wp_virial = None
    with WarpAutogradContextManager(needs_grad_flag) as tape:
        wp.launch(
            _batch_ewald_reciprocal_space_energy_kernel_fill_structure_factors_overload[
                wp_scalar
            ],
            dim=(num_k, num_systems, max_blocks_per_system),
            inputs=[
                wp_positions,
                wp_charges,
                wp_k_vectors,
                wp_cell,
                wp_alpha,
                wp_atom_start,
                wp_atom_end,
            ],
            outputs=[
                wp_total_charge,
                wp_cos_k_dot_r,
                wp_sin_k_dot_r,
                wp_real_sf,
                wp_imag_sf,
            ],
            device=device,
        )
        wp.launch(
            _batch_ewald_reciprocal_space_energy_forces_charge_grad_kernel_overload[
                wp_scalar
            ],
            dim=num_atoms,
            inputs=[
                wp_charges,
                wp_batch_idx,
                wp_k_vectors,
                wp_cos_k_dot_r,
                wp_sin_k_dot_r,
                wp_real_sf,
                wp_imag_sf,
            ],
            outputs=[wp_raw_energies, wp_forces, wp_charge_grads],
            device=device,
        )
        wp.launch(
            _batch_ewald_subtract_self_energy_kernel_overload[wp_scalar],
            dim=num_atoms,
            inputs=[
                wp_charges,
                wp_batch_idx,
                wp_alpha,
                wp_total_charge,
                wp_raw_energies,
            ],
            outputs=[wp_energies],
            device=device,
        )
        if compute_virial:
            virial = torch.zeros(
                num_systems, 3, 3, device=positions.device, dtype=input_dtype
            )
            wp_virial = warp_from_torch(virial, wp_mat, requires_grad=virial_grad)
            volume = torch.abs(torch.det(cell.to(torch.float64)))
            wp_volume = warp_from_torch(volume, wp.float64)
            wp.launch(
                _batch_ewald_reciprocal_space_virial_kernel_overload[wp_scalar],
                dim=(num_k, num_systems),
                inputs=[
                    wp_k_vectors,
                    wp_alpha,
                    wp_volume,
                    wp_real_sf,
                    wp_imag_sf,
                    wp_virial,
                ],
                device=device,
            )
        else:
            virial = torch.zeros(
                num_systems, 3, 3, device=positions.device, dtype=input_dtype
            )

    # Apply self-energy and background corrections to charge gradients
    alpha_per_atom = alpha[batch_idx]
    total_charge_per_atom = total_charge_batch[batch_idx]

    self_energy_grad = 2.0 / math.sqrt(math.pi) * alpha_per_atom * charges
    background_grad = (
        math.pi / (alpha_per_atom * alpha_per_atom) * total_charge_per_atom
    )
    charge_grads = charge_grads - self_energy_grad - background_grad

    if needs_grad_flag:
        backward_kw = dict(
            energies=wp_energies,
            forces=wp_forces,
            charge_gradients=wp_charge_grads,
            positions=wp_positions,
            charges=wp_charges,
            cell=wp_cell,
            k_vectors=wp_k_vectors,
            alpha=wp_alpha,
        )
        if virial_grad and wp_virial is not None:
            backward_kw["virial"] = wp_virial
        attach_for_backward(energies, tape=tape, **backward_kw)
    return energies, forces, charge_grads.to(input_dtype), virial


###########################################################################################
########################### Public Wrapper APIs ###########################################
###########################################################################################


def ewald_real_space(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    alpha: torch.Tensor,
    neighbor_list: torch.Tensor | None = None,
    neighbor_ptr: torch.Tensor | None = None,
    neighbor_shifts: torch.Tensor | None = None,
    neighbor_matrix: torch.Tensor | None = None,
    neighbor_matrix_shifts: torch.Tensor | None = None,
    mask_value: int = -1,
    batch_idx: torch.Tensor | None = None,
    compute_forces: bool = False,
    compute_charge_gradients: bool = False,
    compute_virial: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Compute real-space Ewald energy and optionally forces, charge gradients, and virial.

    Computes the damped Coulomb interactions for atom pairs within the real-space
    cutoff. The complementary error function (erfc) damping ensures rapid
    convergence in real space.

    Parameters
    ----------
    positions : torch.Tensor, shape (N, 3)
        Atomic coordinates.
    charges : torch.Tensor, shape (N,)
        Atomic partial charges.
    cell : torch.Tensor, shape (3, 3) or (B, 3, 3)
        Unit cell matrices.
    alpha : torch.Tensor, shape (1,) or (B,)
        Ewald splitting parameter(s).
    neighbor_list : torch.Tensor, shape (2, M), optional
        Neighbor list in COO format.
    neighbor_ptr : torch.Tensor, shape (N+1,), optional
        CSR row pointers for neighbor list.
    neighbor_shifts : torch.Tensor, shape (M, 3), optional
        Periodic image shifts for neighbor list.
    neighbor_matrix : torch.Tensor, shape (N, max_neighbors), optional
        Dense neighbor matrix format.
    neighbor_matrix_shifts : torch.Tensor, shape (N, max_neighbors, 3), optional
        Periodic image shifts for neighbor_matrix.
    mask_value : int, default=-1
        Value indicating invalid entries in neighbor_matrix.
    batch_idx : torch.Tensor, shape (N,), optional
        System index for each atom.
    compute_forces : bool, default=False
        Whether to compute explicit forces.
    compute_charge_gradients : bool, default=False
        Whether to compute charge gradients.
    compute_virial : bool, default=False
        Whether to compute the virial tensor W = -dE/d(epsilon).
        Stress = virial / volume.

    Returns
    -------
    energies : torch.Tensor, shape (N,)
        Per-atom real-space energy.
    forces : torch.Tensor, shape (N, 3), optional
        Forces (if compute_forces=True).
    charge_gradients : torch.Tensor, shape (N,), optional
        Charge gradients (if compute_charge_gradients=True).
    virial : torch.Tensor, shape (1, 3, 3) or (B, 3, 3), optional
        Virial tensor (if compute_virial=True). Always last in the tuple.

    Note
    ----
    Energies are always float64 for numerical stability during accumulation.
    Forces and virial match the input dtype (float32 or float64).

    """
    is_batch = batch_idx is not None

    # The virial tensor is computed as the outer product of separation vectors and
    # pair forces (W += r_ij ⊗ F_ij), which is accumulated inside the force kernel.
    # Therefore, even when only virial is requested (compute_forces=False,
    # compute_virial=True), we must dispatch a force-capable kernel.
    need_force_kernel = compute_forces or compute_virial

    # Helper to build the return tuple from raw outputs using match dispatch.
    def _build_result(energies, forces=None, charge_grads=None, virial=None):
        match (
            compute_forces and forces is not None,
            compute_charge_gradients and charge_grads is not None,
            compute_virial and virial is not None,
        ):
            case (True, True, True):
                return energies, forces, charge_grads, virial
            case (True, True, False):
                return energies, forces, charge_grads
            case (True, False, True):
                return energies, forces, virial
            case (True, False, False):
                return energies, forces
            case (False, True, True):
                return energies, charge_grads, virial
            case (False, True, False):
                return energies, charge_grads
            case (False, False, True):
                return energies, virial
            case _:
                return energies

    if compute_charge_gradients:
        if neighbor_list is not None:
            if neighbor_ptr is None:
                raise ValueError(
                    "neighbor_ptr is required when using neighbor_list format"
                )
            if is_batch:
                energies, forces, charge_grads, virial = (
                    _batch_ewald_real_space_energy_forces_charge_grad(
                        positions,
                        charges,
                        cell,
                        alpha,
                        batch_idx,
                        neighbor_list,
                        neighbor_ptr,
                        neighbor_shifts,
                        compute_virial=compute_virial,
                    )
                )
            else:
                energies, forces, charge_grads, virial = (
                    _ewald_real_space_energy_forces_charge_grad(
                        positions,
                        charges,
                        cell,
                        alpha,
                        neighbor_list,
                        neighbor_ptr,
                        neighbor_shifts,
                        compute_virial=compute_virial,
                    )
                )
        elif neighbor_matrix is not None:
            if is_batch:
                energies, forces, charge_grads, virial = (
                    _batch_ewald_real_space_energy_forces_charge_grad_matrix(
                        positions,
                        charges,
                        cell,
                        alpha,
                        batch_idx,
                        neighbor_matrix,
                        neighbor_matrix_shifts,
                        mask_value,
                        compute_virial=compute_virial,
                    )
                )
            else:
                energies, forces, charge_grads, virial = (
                    _ewald_real_space_energy_forces_charge_grad_matrix(
                        positions,
                        charges,
                        cell,
                        alpha,
                        neighbor_matrix,
                        neighbor_matrix_shifts,
                        mask_value,
                        compute_virial=compute_virial,
                    )
                )
        else:
            raise ValueError("Either neighbor_list or neighbor_matrix must be provided")

        return _build_result(energies, forces, charge_grads, virial)

    # No charge gradients requested
    if neighbor_list is not None:
        if neighbor_ptr is None:
            raise ValueError("neighbor_ptr is required when using neighbor_list format")
        if is_batch:
            if need_force_kernel:
                energies, forces, virial = _batch_ewald_real_space_energy_forces(
                    positions,
                    charges,
                    cell,
                    alpha,
                    batch_idx,
                    neighbor_list,
                    neighbor_ptr,
                    neighbor_shifts,
                    compute_virial=compute_virial,
                )
                return _build_result(energies, forces, virial=virial)
            else:
                energies = _batch_ewald_real_space_energy(
                    positions,
                    charges,
                    cell,
                    alpha,
                    batch_idx,
                    neighbor_list,
                    neighbor_ptr,
                    neighbor_shifts,
                )
                return _build_result(energies)
        else:
            if need_force_kernel:
                energies, forces, virial = _ewald_real_space_energy_forces(
                    positions,
                    charges,
                    cell,
                    alpha,
                    neighbor_list,
                    neighbor_ptr,
                    neighbor_shifts,
                    compute_virial=compute_virial,
                )
                return _build_result(energies, forces, virial=virial)
            else:
                energies = _ewald_real_space_energy(
                    positions,
                    charges,
                    cell,
                    alpha,
                    neighbor_list,
                    neighbor_ptr,
                    neighbor_shifts,
                )
                return _build_result(energies)
    elif neighbor_matrix is not None:
        if is_batch:
            if need_force_kernel:
                energies, forces, virial = _batch_ewald_real_space_energy_forces_matrix(
                    positions,
                    charges,
                    cell,
                    alpha,
                    batch_idx,
                    neighbor_matrix,
                    neighbor_matrix_shifts,
                    mask_value,
                    compute_virial=compute_virial,
                )
                return _build_result(energies, forces, virial=virial)
            else:
                energies = _batch_ewald_real_space_energy_matrix(
                    positions,
                    charges,
                    cell,
                    alpha,
                    batch_idx,
                    neighbor_matrix,
                    neighbor_matrix_shifts,
                    mask_value,
                )
                return _build_result(energies)
        else:
            if need_force_kernel:
                energies, forces, virial = _ewald_real_space_energy_forces_matrix(
                    positions,
                    charges,
                    cell,
                    alpha,
                    neighbor_matrix,
                    neighbor_matrix_shifts,
                    mask_value,
                    compute_virial=compute_virial,
                )
                return _build_result(energies, forces, virial=virial)
            else:
                energies = _ewald_real_space_energy_matrix(
                    positions,
                    charges,
                    cell,
                    alpha,
                    neighbor_matrix,
                    neighbor_matrix_shifts,
                    mask_value,
                )
                return _build_result(energies)
    else:
        raise ValueError("Either neighbor_list or neighbor_matrix must be provided")


def ewald_reciprocal_space(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    k_vectors: torch.Tensor,
    alpha: torch.Tensor,
    batch_idx: torch.Tensor | None = None,
    compute_forces: bool = False,
    compute_charge_gradients: bool = False,
    compute_virial: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    r"""Compute reciprocal-space Ewald energy and optionally forces, charge gradients, virial.

    Computes the smooth long-range electrostatic contribution using structure
    factors in reciprocal space.

    Parameters
    ----------
    positions : torch.Tensor, shape (N, 3)
        Atomic coordinates.
    charges : torch.Tensor, shape (N,)
        Atomic partial charges.
    cell : torch.Tensor, shape (3, 3) or (B, 3, 3)
        Unit cell matrices.
    k_vectors : torch.Tensor
        Reciprocal lattice vectors. Shape (K, 3) for single system, (B, K, 3) for batch.
    alpha : torch.Tensor, shape (1,) or (B,)
        Ewald splitting parameter(s).
    batch_idx : torch.Tensor, shape (N,), optional
        System index for each atom.
    compute_forces : bool, default=False
        Whether to compute explicit forces.
    compute_charge_gradients : bool, default=False
        Whether to compute charge gradients.
    compute_virial : bool, default=False
        Whether to compute the virial tensor W = -dE/d(epsilon).
        Stress = virial / volume.

    Returns
    -------
    energies : torch.Tensor, shape (N,)
        Per-atom reciprocal-space energy.
    forces : torch.Tensor, shape (N, 3), optional
        Forces (if compute_forces=True).
    charge_gradients : torch.Tensor, shape (N,), optional
        Charge gradients (if compute_charge_gradients=True).
    virial : torch.Tensor, shape (1, 3, 3) or (B, 3, 3), optional
        Virial tensor (if compute_virial=True). Always last in the tuple.

    Note
    ----
    Energies are always float64 for numerical stability during accumulation.
    Forces and virial match the input dtype (float32 or float64).
    """
    is_batch = batch_idx is not None

    # Normalize k-vector rank based on dispatch mode.
    # Batch kernels expect (B, K, 3), single kernels expect (K, 3).
    if is_batch and k_vectors.dim() == 2:
        k_vectors = k_vectors.unsqueeze(0)
    elif not is_batch and k_vectors.dim() == 3 and k_vectors.shape[0] == 1:
        k_vectors = k_vectors.squeeze(0)

    # Helper to build the return tuple from raw outputs using match dispatch.
    def _build_result(energies, forces=None, charge_grads=None, virial=None):
        match (
            compute_forces and forces is not None,
            compute_charge_gradients and charge_grads is not None,
            compute_virial and virial is not None,
        ):
            case (True, True, True):
                return energies, forces, charge_grads, virial
            case (True, True, False):
                return energies, forces, charge_grads
            case (True, False, True):
                return energies, forces, virial
            case (True, False, False):
                return energies, forces
            case (False, True, True):
                return energies, charge_grads, virial
            case (False, True, False):
                return energies, charge_grads
            case (False, False, True):
                return energies, virial
            case _:
                return energies

    if compute_charge_gradients:
        if is_batch:
            energies, forces, charge_grads, virial = (
                _batch_ewald_reciprocal_space_energy_forces_charge_grad(
                    positions,
                    charges,
                    cell,
                    k_vectors,
                    alpha,
                    batch_idx,
                    compute_virial=compute_virial,
                )
            )
        else:
            energies, forces, charge_grads, virial = (
                _ewald_reciprocal_space_energy_forces_charge_grad(
                    positions,
                    charges,
                    cell,
                    k_vectors,
                    alpha,
                    compute_virial=compute_virial,
                )
            )

        return _build_result(energies, forces, charge_grads, virial)

    # No charge gradients
    if is_batch:
        if compute_forces:
            energies, forces, virial = _batch_ewald_reciprocal_space_energy_forces(
                positions,
                charges,
                cell,
                k_vectors,
                alpha,
                batch_idx,
                compute_virial=compute_virial,
            )
            return _build_result(energies, forces, virial=virial)
        else:
            energies, virial = _batch_ewald_reciprocal_space_energy(
                positions,
                charges,
                cell,
                k_vectors,
                alpha,
                batch_idx,
                compute_virial=compute_virial,
            )
            return _build_result(energies, virial=virial)
    else:
        if compute_forces:
            energies, forces, virial = _ewald_reciprocal_space_energy_forces(
                positions,
                charges,
                cell,
                k_vectors,
                alpha,
                compute_virial=compute_virial,
            )
            return _build_result(energies, forces, virial=virial)
        else:
            energies, virial = _ewald_reciprocal_space_energy(
                positions,
                charges,
                cell,
                k_vectors,
                alpha,
                compute_virial=compute_virial,
            )
            return _build_result(energies, virial=virial)


def ewald_summation(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    alpha: float | torch.Tensor | None = None,
    k_vectors: torch.Tensor | None = None,
    k_cutoff: float | None = None,
    batch_idx: torch.Tensor | None = None,
    neighbor_list: torch.Tensor | None = None,
    neighbor_ptr: torch.Tensor | None = None,
    neighbor_shifts: torch.Tensor | None = None,
    neighbor_matrix: torch.Tensor | None = None,
    neighbor_matrix_shifts: torch.Tensor | None = None,
    mask_value: int | None = None,
    compute_forces: bool = False,
    compute_virial: bool = False,
    accuracy: float = 1e-6,
) -> tuple[torch.Tensor, ...] | torch.Tensor:
    """Complete Ewald summation for long-range electrostatics.

    Computes total Coulomb energy by combining real-space and reciprocal-space
    contributions with self-energy and background corrections.

    Parameters
    ----------
    positions : torch.Tensor, shape (N, 3)
        Atomic coordinates.
    charges : torch.Tensor, shape (N,)
        Atomic partial charges.
    cell : torch.Tensor, shape (3, 3) or (B, 3, 3)
        Unit cell matrices.
    alpha : float, torch.Tensor, or None, default=None
        Ewald splitting parameter. Auto-estimated if None.
    k_vectors : torch.Tensor, optional
        Pre-computed reciprocal lattice vectors.
    k_cutoff : float, optional
        K-space cutoff for generating k_vectors.
    batch_idx : torch.Tensor, shape (N,), optional
        System index for each atom.
    neighbor_list : torch.Tensor, shape (2, M), optional
        Neighbor pairs in COO format.
    neighbor_ptr : torch.Tensor, shape (N+1,), optional
        CSR row pointers.
    neighbor_shifts : torch.Tensor, shape (M, 3), optional
        Periodic image shifts for neighbor list.
    neighbor_matrix : torch.Tensor, shape (N, max_neighbors), optional
        Dense neighbor matrix.
    neighbor_matrix_shifts : torch.Tensor, shape (N, max_neighbors, 3), optional
        Periodic image shifts for neighbor_matrix.
    mask_value : int, optional
        Value indicating invalid entries. Defaults to N.
    compute_forces : bool, default=False
        Whether to compute explicit forces.
    compute_virial : bool, default=False
        Whether to compute the virial tensor W = -dE/d(epsilon).
        Stress = virial / volume.
    accuracy : float, default=1e-6
        Target accuracy for parameter estimation.

    Returns
    -------
    energies : torch.Tensor, shape (N,)
        Per-atom total Ewald energy.
    forces : torch.Tensor, shape (N, 3), optional
        Forces (if compute_forces=True).
    virial : torch.Tensor, shape (1, 3, 3) or (B, 3, 3), optional
        Virial tensor (if compute_virial=True). Always last in the tuple.

    Note
    ----
    Energies are always float64 for numerical stability during accumulation.
    Forces and virial match the input dtype (float32 or float64).
    """
    device = positions.device
    dtype = positions.dtype
    num_atoms = positions.shape[0]

    cell, num_systems = _prepare_cell(cell)

    if alpha is None or (k_cutoff is None and k_vectors is None):
        params = estimate_ewald_parameters(positions, cell, batch_idx, accuracy)
        if alpha is None:
            alpha = params.alpha
        if k_cutoff is None:
            k_cutoff = params.reciprocal_space_cutoff

    alpha_tensor = _prepare_alpha(alpha, num_systems, dtype, device)

    if k_vectors is None:
        k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff)

    if mask_value is None:
        mask_value = num_atoms

    # Compute real-space
    rs = ewald_real_space(
        positions=positions,
        charges=charges,
        cell=cell,
        alpha=alpha_tensor,
        neighbor_list=neighbor_list,
        neighbor_ptr=neighbor_ptr,
        neighbor_shifts=neighbor_shifts,
        neighbor_matrix=neighbor_matrix,
        neighbor_matrix_shifts=neighbor_matrix_shifts,
        mask_value=mask_value,
        batch_idx=batch_idx,
        compute_forces=compute_forces,
        compute_virial=compute_virial,
    )

    # Compute reciprocal-space
    rec = ewald_reciprocal_space(
        positions=positions,
        charges=charges,
        cell=cell,
        k_vectors=k_vectors,
        alpha=alpha_tensor,
        batch_idx=batch_idx,
        compute_forces=compute_forces,
        compute_virial=compute_virial,
    )

    # Normalize return tuples for element-wise combination
    rs_tuple = rs if isinstance(rs, tuple) else (rs,)
    rec_tuple = rec if isinstance(rec, tuple) else (rec,)

    results = tuple(r + s for r, s in zip(rs_tuple, rec_tuple))

    if len(results) == 1:
        return results[0]
    return results
