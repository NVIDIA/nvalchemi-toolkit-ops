# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

r"""
Unified Ewald Summation API
===========================

This module provides a unified GPU-accelerated API for Ewald summation that
handles both single-system and batched calculations transparently. All functions
support automatic differentiation through PyTorch's autograd system.

API STRUCTURE
=============

Primary APIs (public, with autograd support):
    ewald_summation(): Complete Ewald calculation (real + reciprocal)
    ewald_real_space(): Real-space damped Coulomb component
    ewald_reciprocal_space(): Reciprocal-space smooth long-range component

The batch_idx parameter determines kernel dispatch:
    batch_idx=None → Single-system kernels (optimized, cell shape (1,3,3))
    batch_idx provided → Batch kernels (multiple systems, cell shape (B,3,3))

MATHEMATICAL FORMULATION
========================

The Ewald method splits long-range Coulomb interactions into components:

.. math::

    E_{\text{total}} = E_{\text{real}} + E_{\text{reciprocal}} - E_{\text{self}} - E_{\text{background}}

Real-Space Component (short-range, damped):

.. math::

    E_{\text{real}} = \frac{1}{2} \sum_{i \neq j} q_i q_j \frac{\text{erfc}(\alpha r_{ij})}{r_{ij}}

The erfc function rapidly damps interactions beyond :math:`r \sim 3/\alpha`.

Reciprocal-Space Component (long-range, smooth):

.. math::

    E_{\text{reciprocal}} = \frac{1}{2V} \sum_{k \in halfspace} \frac{8\pi}{k^2} \exp\left(-\frac{k^2}{4\alpha^2}\right) |S(k)|^2

where :math:`S(k) = \sum_j q_j \exp(ik \cdot r_j)` is the structure factor.

Self-Energy Correction:

.. math::

    E_{\text{self}} = \sum_i \frac{\alpha}{\sqrt{\pi}} q_i^2

Background Correction (for non-neutral systems):

.. math::

    E_{\text{background}} = \sum_i \frac{\pi}{2\alpha^2 V} q_i Q_{\text{total}}

USAGE EXAMPLES
==============

Single system with automatic parameter estimation::

    >>> from nvalchemiops.interactions.electrostatics import ewald_summation
    >>> energies, forces = ewald_summation(
    ...     positions, charges, cell,
    ...     neighbor_list=neighbor_list,
    ...     neighbor_shifts=neighbor_shifts,
    ...     accuracy=1e-6,  # alpha and k_cutoff estimated automatically
    ... )

Single system with explicit parameters::

    >>> energies, forces = ewald_summation(
    ...     positions, charges, cell,
    ...     alpha=0.3, k_cutoff=8.0,
    ...     neighbor_matrix=neighbor_matrix,
    ...     neighbor_matrix_shifts=shifts,
    ...     mask_value=-1,
    ... )

Batched systems (multiple independent structures)::

    >>> # positions: concatenated atoms from all systems
    >>> # batch_idx: system index for each atom
    >>> energies, forces = ewald_summation(
    ...     positions, charges, cells,  # cells shape (B, 3, 3)
    ...     alpha=torch.tensor([0.3, 0.3, 0.3]),  # per-system alpha
    ...     batch_idx=batch_idx,
    ...     k_cutoff=8.0,
    ...     neighbor_list=neighbor_list,
    ...     neighbor_shifts=neighbor_shifts,
    ... )

Energy-only (no force computation)::

    >>> energies = ewald_summation(
    ...     positions, charges, cell, alpha=0.3, k_cutoff=8.0,
    ...     neighbor_list=nl, neighbor_shifts=shifts,
    ...     compute_forces=False,
    ... )

Autograd for gradients::

    >>> positions.requires_grad_(True)
    >>> energies, forces = ewald_summation(positions, charges, cell, ...)
    >>> total_energy = energies.sum()
    >>> total_energy.backward()
    >>> autograd_forces = -positions.grad  # Should match explicit forces

REFERENCES
==========

- Ewald, P. P. (1921). Ann. Phys. 369, 253-287 (Original Ewald method)
- Kolafa, J. & Perram, J. W. (1992). Mol. Sim. 9, 351-368 (Parameter optimization)
- Essmann et al. (1995). J. Chem. Phys. 103, 8577 (PME method)
"""

import math
import os

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
from nvalchemiops.interactions.electrostatics.ewald_kernels import (
    _batch_ewald_real_space_energy_forces_charge_grad_kernel_overload,
    _batch_ewald_real_space_energy_forces_charge_grad_neighbor_matrix_kernel_overload,
    _batch_ewald_real_space_energy_forces_kernel_overload,
    _batch_ewald_real_space_energy_forces_neighbor_matrix_kernel_overload,
    # Batch real-space kernel overloads
    _batch_ewald_real_space_energy_kernel_overload,
    _batch_ewald_real_space_energy_neighbor_matrix_kernel_overload,
    _batch_ewald_reciprocal_space_energy_forces_charge_grad_kernel_overload,
    _batch_ewald_reciprocal_space_energy_forces_kernel_overload,
    _batch_ewald_reciprocal_space_energy_kernel_compute_energy_overload,
    # Batch reciprocal-space kernel overloads
    _batch_ewald_reciprocal_space_energy_kernel_fill_structure_factors_overload,
    _batch_ewald_subtract_self_energy_kernel_overload,
    _ewald_real_space_energy_forces_charge_grad_kernel_overload,
    _ewald_real_space_energy_forces_charge_grad_neighbor_matrix_kernel_overload,
    _ewald_real_space_energy_forces_kernel_overload,
    _ewald_real_space_energy_forces_neighbor_matrix_kernel_overload,
    # Single-system real-space kernel overloads
    _ewald_real_space_energy_kernel_overload,
    _ewald_real_space_energy_neighbor_matrix_kernel_overload,
    _ewald_reciprocal_space_energy_forces_charge_grad_kernel_overload,
    _ewald_reciprocal_space_energy_forces_kernel_overload,
    # Kernel compute (dtype-independent, all float64)
    _ewald_reciprocal_space_energy_kernel_compute_energy_overload,
    # Single-system reciprocal-space kernel overloads
    _ewald_reciprocal_space_energy_kernel_fill_structure_factors_overload,
    _ewald_subtract_self_energy_kernel_overload,
)
from nvalchemiops.interactions.electrostatics.k_vectors import (
    generate_k_vectors_ewald_summation,
)
from nvalchemiops.interactions.electrostatics.parameters import (
    estimate_ewald_parameters,
)
from nvalchemiops.types import get_wp_dtype, get_wp_mat_dtype, get_wp_vec_dtype

# Mathematical constants
PI = math.pi
TWOPI = 2.0 * PI
BATCH_BLOCK_SIZE = int(os.environ.get("ALCH_EWALD_BATCH_BLOCK_SIZE", 16))


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

    Raises
    ------
    ValueError
        If alpha tensor has wrong number of elements.
    TypeError
        If alpha is neither float nor tensor.
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
    """Internal: Compute real-space Ewald energies (single system, neighbor list CSR).

    Supports both float32 and float64 input dtypes. Returns energies in input dtype.
    Uses CSR format (neighbor_ptr + idx_j) for optimized kernel launch.
    """
    num_atoms = positions.shape[0]
    input_dtype = positions.dtype
    empty_nl = neighbor_list.shape[1] == 0

    idx_j = neighbor_list[1]  # Only need idx_j for CSR format
    device = wp.device_from_torch(positions.device)
    needs_grad_flag = needs_grad(positions, charges, cell)

    # Get warp types based on input dtype
    input_dtype = positions.dtype
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

    # Output energies are always float64 for precision
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
    # Return energies in input dtype
    return energies.to(input_dtype)


@warp_custom_op(
    name="alchemiops::_ewald_real_space_energy_forces",
    outputs=[
        OutputSpec("energies", wp.float64, lambda pos, *_: (pos.shape[0],)),
        OutputSpec("forces", wp.vec3d, lambda pos, *_: (pos.shape[0], 3)),
    ],
    grad_arrays=["energies", "forces", "positions", "charges", "cell", "alpha"],
)
def _ewald_real_space_energy_forces(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    alpha: torch.Tensor,
    neighbor_list: torch.Tensor,
    neighbor_ptr: torch.Tensor,
    neighbor_shifts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Internal: Compute real-space Ewald energies and forces (single system, CSR).

    Supports both float32 and float64 input dtypes. Returns both in input dtype.
    Uses CSR format (neighbor_ptr + idx_j) for optimized kernel launch.
    """
    num_atoms = positions.shape[0]
    input_dtype = positions.dtype

    empty_nl = neighbor_list.shape[1] == 0
    idx_j = neighbor_list[1]  # Only need idx_j for CSR format
    device = wp.device_from_torch(positions.device)
    needs_grad_flag = needs_grad(positions, charges, cell)

    # Get warp types based on input dtype
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

    # Energies always float64, forces match positions dtype
    energies = torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)
    forces = torch.zeros(num_atoms, 3, device=positions.device, dtype=input_dtype)
    wp_energies = warp_from_torch(energies, wp.float64, requires_grad=needs_grad_flag)
    wp_forces = warp_from_torch(forces, wp_vec, requires_grad=needs_grad_flag)

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
            alpha=wp_alpha,
        )
    # Return energies in input dtype (forces already in input dtype)
    return energies.to(input_dtype), forces


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
    """Internal: Compute real-space Ewald energies (single system, neighbor matrix).

    Supports both float32 and float64 input dtypes. Returns energies in input dtype.
    """
    num_atoms = positions.shape[0]
    input_dtype = positions.dtype
    empty_nm = neighbor_matrix.shape[0] == 0

    device = wp.device_from_torch(positions.device)
    needs_grad_flag = needs_grad(positions, charges, cell)

    # Get warp types based on input dtype
    input_dtype = positions.dtype
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
    # Return energies in input dtype
    return energies.to(input_dtype)


@warp_custom_op(
    name="alchemiops::_ewald_real_space_energy_forces_matrix",
    outputs=[
        OutputSpec("energies", wp.float64, lambda pos, *_: (pos.shape[0],)),
        OutputSpec("forces", wp.vec3d, lambda pos, *_: (pos.shape[0], 3)),
    ],
    grad_arrays=["energies", "forces", "positions", "charges", "cell", "alpha"],
)
def _ewald_real_space_energy_forces_matrix(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    alpha: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    mask_value: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Internal: Compute real-space Ewald energies and forces (single system, neighbor matrix).

    Supports both float32 and float64 input dtypes. Returns both in input dtype.
    """
    num_atoms = positions.shape[0]
    input_dtype = positions.dtype

    empty_nm = neighbor_matrix.shape[0] == 0

    device = wp.device_from_torch(positions.device)
    needs_grad_flag = needs_grad(positions, charges, cell)

    # Get warp types based on input dtype
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
    wp_energies = warp_from_torch(energies, wp.float64, requires_grad=needs_grad_flag)
    wp_forces = warp_from_torch(forces, wp_vec, requires_grad=needs_grad_flag)

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
            alpha=wp_alpha,
        )
    # Return energies in input dtype (forces already in input dtype)
    return energies.to(input_dtype), forces


###########################################################################################
################## Real-Space with Charge Gradients Internal Custom Ops ###################
###########################################################################################


@warp_custom_op(
    name="alchemiops::_ewald_real_space_energy_forces_charge_grad",
    outputs=[
        OutputSpec("energies", wp.float64, lambda pos, *_: (pos.shape[0],)),
        OutputSpec("forces", wp.vec3d, lambda pos, *_: (pos.shape[0], 3)),
        OutputSpec("charge_gradients", wp.float64, lambda pos, *_: (pos.shape[0],)),
    ],
    grad_arrays=[
        "energies",
        "forces",
        "charge_gradients",
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Internal: Compute real-space Ewald energies, forces, AND charge gradients (CSR).

    Single system, neighbor list CSR format.
    Supports both float32 and float64 input dtypes.

    Returns
    -------
    energies : torch.Tensor, shape (N,)
        Per-atom energy contributions.
    forces : torch.Tensor, shape (N, 3)
        Forces on each atom.
    charge_gradients : torch.Tensor, shape (N,)
        Gradient of total energy with respect to each atom's charge: ∂E/∂q_i.
    """
    num_atoms = positions.shape[0]
    input_dtype = positions.dtype

    empty_nl = neighbor_list.shape[1] == 0

    idx_j = neighbor_list[1]  # Only need idx_j for CSR format
    device = wp.device_from_torch(positions.device)
    needs_grad_flag = needs_grad(positions, charges, cell)

    # Get warp types based on input dtype
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

    # Output tensors
    energies = torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)
    forces = torch.zeros(num_atoms, 3, device=positions.device, dtype=input_dtype)
    charge_grads = torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)
    wp_energies = warp_from_torch(energies, wp.float64, requires_grad=needs_grad_flag)
    wp_forces = warp_from_torch(forces, wp_vec, requires_grad=needs_grad_flag)
    wp_charge_grads = warp_from_torch(
        charge_grads, wp.float64, requires_grad=needs_grad_flag
    )

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
                    wp_energies,
                    wp_forces,
                    wp_charge_grads,
                ],
                device=device,
            )

    if needs_grad_flag:
        attach_for_backward(
            energies,
            tape=tape,
            energies=wp_energies,
            forces=wp_forces,
            charge_gradients=wp_charge_grads,
            positions=wp_positions,
            charges=wp_charges,
            cell=wp_cell,
            alpha=wp_alpha,
        )
    return energies.to(input_dtype), forces, charge_grads


@warp_custom_op(
    name="alchemiops::_ewald_real_space_energy_forces_charge_grad_matrix",
    outputs=[
        OutputSpec("energies", wp.float64, lambda pos, *_: (pos.shape[0],)),
        OutputSpec("forces", wp.vec3d, lambda pos, *_: (pos.shape[0], 3)),
        OutputSpec("charge_gradients", wp.float64, lambda pos, *_: (pos.shape[0],)),
    ],
    grad_arrays=[
        "energies",
        "forces",
        "charge_gradients",
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Internal: Compute real-space Ewald energies, forces, AND charge gradients.

    Single system, neighbor matrix format.
    Supports both float32 and float64 input dtypes.

    Returns
    -------
    energies : torch.Tensor, shape (N,)
        Per-atom energy contributions.
    forces : torch.Tensor, shape (N, 3)
        Forces on each atom.
    charge_gradients : torch.Tensor, shape (N,)
        Gradient of total energy with respect to each atom's charge: ∂E/∂q_i.
    """
    num_atoms = positions.shape[0]
    input_dtype = positions.dtype

    empty_nm = neighbor_matrix.shape[0] == 0

    device = wp.device_from_torch(positions.device)
    needs_grad_flag = needs_grad(positions, charges, cell)

    # Get warp types based on input dtype
    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)

    wp_positions = warp_from_torch(positions, wp_vec, requires_grad=needs_grad_flag)
    wp_charges = warp_from_torch(charges, wp_scalar, requires_grad=needs_grad_flag)
    wp_cell = warp_from_torch(cell, wp_mat, requires_grad=needs_grad_flag)
    wp_alpha = warp_from_torch(alpha, wp_scalar, requires_grad=needs_grad_flag)
    wp_neighbor_matrix = warp_from_torch(neighbor_matrix, wp.int32)
    wp_unit_shifts_matrix = warp_from_torch(neighbor_matrix_shifts, wp.vec3i)

    # Output tensors
    energies = torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)
    forces = torch.zeros(num_atoms, 3, device=positions.device, dtype=input_dtype)
    charge_grads = torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)
    wp_energies = warp_from_torch(energies, wp.float64, requires_grad=needs_grad_flag)
    wp_forces = warp_from_torch(forces, wp_vec, requires_grad=needs_grad_flag)
    wp_charge_grads = warp_from_torch(
        charge_grads, wp.float64, requires_grad=needs_grad_flag
    )

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
                    wp_energies,
                    wp_forces,
                    wp_charge_grads,
                ],
                device=device,
            )

    if needs_grad_flag:
        attach_for_backward(
            energies,
            tape=tape,
            energies=wp_energies,
            forces=wp_forces,
            charge_gradients=wp_charge_grads,
            positions=wp_positions,
            charges=wp_charges,
            cell=wp_cell,
            alpha=wp_alpha,
        )
    return energies.to(input_dtype), forces, charge_grads


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
    """Internal: Compute real-space Ewald energies (batch, neighbor list CSR).

    Supports both float32 and float64 input dtypes. Returns energies in input dtype.
    Uses CSR format (neighbor_ptr + idx_j) for optimized kernel launch.
    """
    num_atoms = positions.shape[0]
    input_dtype = positions.dtype
    device = wp.device_from_torch(positions.device)
    needs_grad_flag = needs_grad(positions, charges, cell)

    empty_nl = neighbor_list.shape[1] == 0

    idx_j = neighbor_list[1]  # Only need idx_j for CSR format

    # Get warp types based on input dtype
    input_dtype = positions.dtype
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
    # Return energies in input dtype
    return energies.to(input_dtype)


@warp_custom_op(
    name="alchemiops::_batch_ewald_real_space_energy_forces",
    outputs=[
        OutputSpec("energies", wp.float64, lambda pos, *_: (pos.shape[0],)),
        OutputSpec("forces", wp.vec3d, lambda pos, *_: (pos.shape[0], 3)),
    ],
    grad_arrays=["energies", "forces", "positions", "charges", "cell", "alpha"],
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Internal: Compute real-space Ewald energies and forces (batch, CSR).

    Supports both float32 and float64 input dtypes. Returns both in input dtype.
    Uses CSR format (neighbor_ptr + idx_j) for optimized kernel launch.
    """
    num_atoms = positions.shape[0]
    input_dtype = positions.dtype

    empty_nl = neighbor_list.shape[1] == 0

    idx_j = neighbor_list[1]  # Only need idx_j for CSR format
    device = wp.device_from_torch(positions.device)
    needs_grad_flag = needs_grad(positions, charges, cell)

    # Get warp types based on input dtype
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
    forces = torch.zeros(num_atoms, 3, device=positions.device, dtype=input_dtype)
    wp_energies = warp_from_torch(energies, wp.float64, requires_grad=needs_grad_flag)
    wp_forces = warp_from_torch(forces, wp_vec, requires_grad=needs_grad_flag)

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
            alpha=wp_alpha,
        )
    # Return energies in input dtype (forces already in input dtype)
    return energies.to(input_dtype), forces


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
    """Internal: Compute real-space Ewald energies (batch, neighbor matrix).

    Supports both float32 and float64 input dtypes. Returns energies in input dtype.
    """
    num_atoms = positions.shape[0]
    input_dtype = positions.dtype
    empty_nm = neighbor_matrix.shape[0] == 0

    device = wp.device_from_torch(positions.device)
    needs_grad_flag = needs_grad(positions, charges, cell)

    # Get warp types based on input dtype
    input_dtype = positions.dtype
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
    # Return energies in input dtype
    return energies.to(input_dtype)


@warp_custom_op(
    name="alchemiops::_batch_ewald_real_space_energy_forces_matrix",
    outputs=[
        OutputSpec("energies", wp.float64, lambda pos, *_: (pos.shape[0],)),
        OutputSpec("forces", wp.vec3d, lambda pos, *_: (pos.shape[0], 3)),
    ],
    grad_arrays=["energies", "forces", "positions", "charges", "cell", "alpha"],
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Internal: Compute real-space Ewald energies and forces (batch, neighbor matrix).

    Supports both float32 and float64 input dtypes. Returns both in input dtype.
    """
    num_atoms = positions.shape[0]
    input_dtype = positions.dtype

    empty_nm = neighbor_matrix.shape[0] == 0

    device = wp.device_from_torch(positions.device)
    needs_grad_flag = needs_grad(positions, charges, cell)

    # Get warp types based on input dtype
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
    forces = torch.zeros(num_atoms, 3, device=positions.device, dtype=input_dtype)
    wp_energies = warp_from_torch(energies, wp.float64, requires_grad=needs_grad_flag)
    wp_forces = warp_from_torch(forces, wp_vec, requires_grad=needs_grad_flag)

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
            alpha=wp_alpha,
        )
    # Return energies in input dtype (forces already in input dtype)
    return energies.to(input_dtype), forces


###########################################################################################
################ Batch Real-Space with Charge Gradients Internal Custom Ops ###############
###########################################################################################


@warp_custom_op(
    name="alchemiops::_batch_ewald_real_space_energy_forces_charge_grad",
    outputs=[
        OutputSpec("energies", wp.float64, lambda pos, *_: (pos.shape[0],)),
        OutputSpec("forces", wp.vec3d, lambda pos, *_: (pos.shape[0], 3)),
        OutputSpec("charge_gradients", wp.float64, lambda pos, *_: (pos.shape[0],)),
    ],
    grad_arrays=[
        "energies",
        "forces",
        "charge_gradients",
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Internal: Compute real-space Ewald energies, forces, AND charge gradients (CSR).

    Batch mode, neighbor list CSR format.
    Supports both float32 and float64 input dtypes.

    Returns
    -------
    energies : torch.Tensor, shape (N,)
        Per-atom energy contributions.
    forces : torch.Tensor, shape (N, 3)
        Forces on each atom.
    charge_gradients : torch.Tensor, shape (N,)
        Gradient of total energy with respect to each atom's charge: ∂E/∂q_i.
    """
    num_atoms = positions.shape[0]
    input_dtype = positions.dtype

    empty_nl = neighbor_list.shape[1] == 0

    idx_j = neighbor_list[1]  # Only need idx_j for CSR format
    device = wp.device_from_torch(positions.device)
    needs_grad_flag = needs_grad(positions, charges, cell)

    # Get warp types based on input dtype
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

    # Output tensors
    energies = torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)
    forces = torch.zeros(num_atoms, 3, device=positions.device, dtype=input_dtype)
    charge_grads = torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)
    wp_energies = warp_from_torch(energies, wp.float64, requires_grad=needs_grad_flag)
    wp_forces = warp_from_torch(forces, wp_vec, requires_grad=needs_grad_flag)
    wp_charge_grads = warp_from_torch(
        charge_grads, wp.float64, requires_grad=needs_grad_flag
    )

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
                    wp_energies,
                    wp_forces,
                    wp_charge_grads,
                ],
                device=device,
            )

    if needs_grad_flag:
        attach_for_backward(
            energies,
            tape=tape,
            energies=wp_energies,
            forces=wp_forces,
            charge_gradients=wp_charge_grads,
            positions=wp_positions,
            charges=wp_charges,
            cell=wp_cell,
            alpha=wp_alpha,
        )
    return energies.to(input_dtype), forces, charge_grads


@warp_custom_op(
    name="alchemiops::_batch_ewald_real_space_energy_forces_charge_grad_matrix",
    outputs=[
        OutputSpec("energies", wp.float64, lambda pos, *_: (pos.shape[0],)),
        OutputSpec("forces", wp.vec3d, lambda pos, *_: (pos.shape[0], 3)),
        OutputSpec("charge_gradients", wp.float64, lambda pos, *_: (pos.shape[0],)),
    ],
    grad_arrays=[
        "energies",
        "forces",
        "charge_gradients",
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Internal: Compute real-space Ewald energies, forces, AND charge gradients.

    Batch mode, neighbor matrix format.
    Supports both float32 and float64 input dtypes.

    Returns
    -------
    energies : torch.Tensor, shape (N,)
        Per-atom energy contributions.
    forces : torch.Tensor, shape (N, 3)
        Forces on each atom.
    charge_gradients : torch.Tensor, shape (N,)
        Gradient of total energy with respect to each atom's charge: ∂E/∂q_i.
    """
    num_atoms = positions.shape[0]
    input_dtype = positions.dtype

    empty_nm = neighbor_matrix.shape[0] == 0

    device = wp.device_from_torch(positions.device)
    needs_grad_flag = needs_grad(positions, charges, cell)

    # Get warp types based on input dtype
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

    # Output tensors
    energies = torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)
    forces = torch.zeros(num_atoms, 3, device=positions.device, dtype=input_dtype)
    charge_grads = torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)
    wp_energies = warp_from_torch(energies, wp.float64, requires_grad=needs_grad_flag)
    wp_forces = warp_from_torch(forces, wp_vec, requires_grad=needs_grad_flag)
    wp_charge_grads = warp_from_torch(
        charge_grads, wp.float64, requires_grad=needs_grad_flag
    )

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
                    wp_energies,
                    wp_forces,
                    wp_charge_grads,
                ],
                device=device,
            )

    if needs_grad_flag:
        attach_for_backward(
            energies,
            tape=tape,
            energies=wp_energies,
            forces=wp_forces,
            charge_gradients=wp_charge_grads,
            positions=wp_positions,
            charges=wp_charges,
            cell=wp_cell,
            alpha=wp_alpha,
        )
    return energies.to(input_dtype), forces, charge_grads


###########################################################################################
########################### Reciprocal-Space Internal Custom Ops ##########################
###########################################################################################


@warp_custom_op(
    name="alchemiops::_ewald_reciprocal_space_energy",
    outputs=[OutputSpec("energies", wp.float64, lambda pos, *_: (pos.shape[0],))],
    grad_arrays=["energies", "positions", "charges", "cell", "k_vectors", "alpha"],
)
def _ewald_reciprocal_space_energy(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    k_vectors: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """Internal: Compute reciprocal-space Ewald energies (single system).

    Supports both float32 and float64 input dtypes. Energies and structure factors
    are always computed in float64 for numerical stability.
    """
    num_k = k_vectors.shape[0]
    num_atoms = positions.shape[0]
    device = wp.device_from_torch(positions.device)

    needs_grad_flag = needs_grad(positions, charges, cell)

    # Get warp types based on input dtype
    input_dtype = positions.dtype
    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)
    wp_positions = warp_from_torch(positions, wp_vec, requires_grad=needs_grad_flag)
    wp_charges = warp_from_torch(charges, wp_scalar, requires_grad=needs_grad_flag)
    wp_cell = warp_from_torch(cell, wp_mat, requires_grad=needs_grad_flag)
    # Ensure k_vectors have same dtype as positions
    k_vectors_typed = k_vectors.to(input_dtype)
    wp_k_vectors = warp_from_torch(
        k_vectors_typed, wp_vec, requires_grad=needs_grad_flag
    )
    wp_alpha = warp_from_torch(alpha, wp_scalar, requires_grad=needs_grad_flag)
    # Intermediate arrays (always float64 for precision)
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
    with WarpAutogradContextManager(needs_grad_flag) as tape:
        # K-major: one thread per k-vector
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
        # Atom-major: one thread per atom
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

    if needs_grad_flag:
        attach_for_backward(
            energies,
            tape=tape,
            energies=wp_energies,
            positions=wp_positions,
            charges=wp_charges,
            cell=wp_cell,
            k_vectors=wp_k_vectors,
            alpha=wp_alpha,
        )
    # Return energies in input dtype
    return energies.to(input_dtype)


@warp_custom_op(
    name="alchemiops::_ewald_reciprocal_space_energy_forces",
    outputs=[
        OutputSpec("energies", wp.float64, lambda pos, *_: (pos.shape[0],)),
        OutputSpec("forces", wp.vec3d, lambda pos, *_: (pos.shape[0], 3)),
    ],
    grad_arrays=[
        "energies",
        "forces",
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Internal: Compute reciprocal-space Ewald energies and forces (single system).

    Supports both float32 and float64 input dtypes. Returns both in input dtype.
    """
    num_k = k_vectors.shape[0]
    num_atoms = positions.shape[0]
    device = wp.device_from_torch(positions.device)
    input_dtype = positions.dtype

    if num_k == 0 or num_atoms == 0:
        return (
            torch.zeros(num_atoms, device=positions.device, dtype=input_dtype),
            torch.zeros(num_atoms, 3, device=positions.device, dtype=input_dtype),
        )

    needs_grad_flag = needs_grad(positions, charges, cell)

    # Get warp types based on input dtype
    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)

    wp_positions = warp_from_torch(positions, wp_vec, requires_grad=needs_grad_flag)
    wp_charges = warp_from_torch(charges, wp_scalar, requires_grad=needs_grad_flag)
    wp_cell = warp_from_torch(cell, wp_mat, requires_grad=needs_grad_flag)
    # Ensure k_vectors have same dtype as positions
    k_vectors_typed = k_vectors.to(input_dtype)
    wp_k_vectors = warp_from_torch(
        k_vectors_typed, wp_vec, requires_grad=needs_grad_flag
    )
    wp_alpha = warp_from_torch(alpha, wp_scalar, requires_grad=needs_grad_flag)

    # Intermediate arrays (always float64 for precision)
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
    wp_real_sf = warp_from_torch(
        torch.zeros(num_k, device=positions.device, dtype=torch.float64),
        wp.float64,
        requires_grad=needs_grad_flag,
    )
    wp_imag_sf = warp_from_torch(
        torch.zeros(num_k, device=positions.device, dtype=torch.float64),
        wp.float64,
        requires_grad=needs_grad_flag,
    )
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

    with WarpAutogradContextManager(needs_grad_flag) as tape:
        # K-major: one thread per k-vector
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
        # Atom-major: one thread per atom
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

    if needs_grad_flag:
        attach_for_backward(
            energies,
            tape=tape,
            energies=wp_energies,
            forces=wp_forces,
            positions=wp_positions,
            charges=wp_charges,
            cell=wp_cell,
            k_vectors=wp_k_vectors,
            alpha=wp_alpha,
        )
    # Return energies in input dtype (forces already in input dtype)
    return energies.to(input_dtype), forces


@warp_custom_op(
    name="alchemiops::_ewald_reciprocal_space_energy_forces_charge_grad",
    outputs=[
        OutputSpec("energies", wp.float64, lambda pos, *_: (pos.shape[0],)),
        OutputSpec("forces", wp.vec3d, lambda pos, *_: (pos.shape[0], 3)),
        OutputSpec("charge_gradients", wp.float64, lambda pos, *_: (pos.shape[0],)),
    ],
    grad_arrays=[
        "energies",
        "forces",
        "charge_gradients",
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Internal: Compute reciprocal-space Ewald energies, forces, and charge gradients.

    Supports both float32 and float64 input dtypes. Returns all in input dtype.
    """
    num_k = k_vectors.shape[0]
    num_atoms = positions.shape[0]
    device = wp.device_from_torch(positions.device)
    input_dtype = positions.dtype

    if num_k == 0 or num_atoms == 0:
        return (
            torch.zeros(num_atoms, device=positions.device, dtype=input_dtype),
            torch.zeros(num_atoms, 3, device=positions.device, dtype=input_dtype),
            torch.zeros(num_atoms, device=positions.device, dtype=input_dtype),
        )

    needs_grad_flag = needs_grad(positions, charges, cell)

    # Get warp types based on input dtype
    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)

    wp_positions = warp_from_torch(positions, wp_vec, requires_grad=needs_grad_flag)
    wp_charges = warp_from_torch(charges, wp_scalar, requires_grad=needs_grad_flag)
    wp_cell = warp_from_torch(cell, wp_mat, requires_grad=needs_grad_flag)
    # Ensure k_vectors have same dtype as positions
    k_vectors_typed = k_vectors.to(input_dtype)
    wp_k_vectors = warp_from_torch(
        k_vectors_typed, wp_vec, requires_grad=needs_grad_flag
    )
    wp_alpha = warp_from_torch(alpha, wp_scalar, requires_grad=needs_grad_flag)

    # Intermediate arrays (always float64 for precision)
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

    with WarpAutogradContextManager(needs_grad_flag) as tape:
        # K-major: one thread per k-vector
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
        # Atom-major: one thread per atom (energy, forces, and charge gradients)
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

    # Apply self-energy and background corrections to charge gradients
    # charge_grads contains φ_i (electrostatic potential)
    # Full charge gradient: ∂E/∂q_i = φ_i - 2(α/√π)q_i - (π/α²)(Q_total/V)
    alpha_val = alpha[0].item()
    self_energy_grad = 2.0 * alpha_val / math.sqrt(PI) * charges
    background_grad = PI / (alpha_val * alpha_val) * total_charge[0]
    charge_grads = charge_grads - self_energy_grad - background_grad

    if needs_grad_flag:
        attach_for_backward(
            energies,
            tape=tape,
            energies=wp_energies,
            forces=wp_forces,
            charge_gradients=wp_charge_grads,
            positions=wp_positions,
            charges=wp_charges,
            cell=wp_cell,
            k_vectors=wp_k_vectors,
            alpha=wp_alpha,
        )
    # Return in appropriate dtypes
    return energies.to(input_dtype), forces, charge_grads.to(input_dtype)


###########################################################################################
########################### Batch Reciprocal-Space Internal Custom Ops ####################
###########################################################################################


@warp_custom_op(
    name="alchemiops::_batch_ewald_reciprocal_space_energy",
    outputs=[OutputSpec("energies", wp.float64, lambda pos, *_: (pos.shape[0],))],
    grad_arrays=["energies", "positions", "charges", "cell", "k_vectors", "alpha"],
)
def _batch_ewald_reciprocal_space_energy(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    k_vectors: torch.Tensor,
    alpha: torch.Tensor,
    batch_idx: torch.Tensor,
) -> torch.Tensor:
    """Internal: Compute reciprocal-space Ewald energies (batch).

    Supports both float32 and float64 input dtypes. Returns energies in input dtype.
    """
    num_k = k_vectors.shape[1]
    num_atoms = positions.shape[0]
    num_systems = cell.shape[0]
    device = wp.device_from_torch(positions.device)
    input_dtype = positions.dtype

    if num_k == 0 or num_atoms == 0:
        return torch.zeros(num_atoms, device=positions.device, dtype=input_dtype)

    # Get warp types based on input dtype
    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)
    # Compute atom_start and atom_end for each system from batch_idx
    # This enables the blocked kernel to efficiently process atoms per system
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
    wp_positions = warp_from_torch(positions, wp_vec, requires_grad=needs_grad_flag)
    wp_charges = warp_from_torch(charges, wp_scalar, requires_grad=needs_grad_flag)
    wp_cell = warp_from_torch(cell, wp_mat, requires_grad=needs_grad_flag)
    # Ensure k_vectors have same dtype as positions
    k_vectors_typed = k_vectors.to(input_dtype)
    wp_k_vectors = warp_from_torch(
        k_vectors_typed, wp_vec, requires_grad=needs_grad_flag
    )
    wp_alpha = warp_from_torch(alpha, wp_scalar, requires_grad=needs_grad_flag)
    wp_batch_idx = warp_from_torch(batch_idx, wp.int32)
    wp_atom_start = warp_from_torch(atom_start, wp.int32)
    wp_atom_end = warp_from_torch(atom_end, wp.int32)
    # Intermediate arrays (always float64)
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
    with WarpAutogradContextManager(needs_grad_flag) as tape:
        # Blocked: one thread per (k-vector, system, atom_block)
        # Much fewer atomics than atom-major iteration
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
        # Atom-major: one thread per atom
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

    if needs_grad_flag:
        attach_for_backward(
            energies,
            tape=tape,
            energies=wp_energies,
            positions=wp_positions,
            charges=wp_charges,
            cell=wp_cell,
            k_vectors=wp_k_vectors,
            alpha=wp_alpha,
        )
    # Return energies in input dtype
    return energies.to(input_dtype)


@warp_custom_op(
    name="alchemiops::_batch_ewald_reciprocal_space_energy_forces",
    outputs=[
        OutputSpec("energies", wp.float64, lambda pos, *_: (pos.shape[0],)),
        OutputSpec("forces", wp.vec3d, lambda pos, *_: (pos.shape[0], 3)),
    ],
    grad_arrays=[
        "energies",
        "forces",
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Internal: Compute reciprocal-space Ewald energies and forces (batch).

    Supports both float32 and float64 input dtypes. Returns both in input dtype.
    """
    num_k = k_vectors.shape[1]
    num_atoms = positions.shape[0]
    num_systems = cell.shape[0]
    device = wp.device_from_torch(positions.device)
    input_dtype = positions.dtype

    if num_k == 0 or num_atoms == 0:
        return (
            torch.zeros(num_atoms, device=positions.device, dtype=input_dtype),
            torch.zeros(num_atoms, 3, device=positions.device, dtype=input_dtype),
        )

    # Get warp types based on input dtype
    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)

    # Compute atom_start and atom_end for each system from batch_idx
    # This enables the blocked kernel to efficiently process atoms per system
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
    wp_positions = warp_from_torch(positions, wp_vec, requires_grad=needs_grad_flag)
    wp_charges = warp_from_torch(charges, wp_scalar, requires_grad=needs_grad_flag)
    wp_cell = warp_from_torch(cell, wp_mat, requires_grad=needs_grad_flag)
    # Ensure k_vectors have same dtype as positions
    k_vectors_typed = k_vectors.to(input_dtype)
    wp_k_vectors = warp_from_torch(
        k_vectors_typed, wp_vec, requires_grad=needs_grad_flag
    )
    wp_alpha = warp_from_torch(alpha, wp_scalar, requires_grad=needs_grad_flag)
    wp_batch_idx = warp_from_torch(batch_idx, wp.int32)
    wp_atom_start = warp_from_torch(atom_start, wp.int32)
    wp_atom_end = warp_from_torch(atom_end, wp.int32)

    # Intermediate arrays (always float64)
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

    with WarpAutogradContextManager(needs_grad_flag) as tape:
        # Blocked: one thread per (k-vector, system, atom_block)
        # Much fewer atomics than atom-major iteration
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
        # Atom-major: one thread per atom
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

    if needs_grad_flag:
        attach_for_backward(
            energies,
            tape=tape,
            energies=wp_energies,
            forces=wp_forces,
            positions=wp_positions,
            charges=wp_charges,
            cell=wp_cell,
            k_vectors=wp_k_vectors,
            alpha=wp_alpha,
        )
    # Return energies in input dtype (forces already in input dtype)
    return energies.to(input_dtype), forces


@warp_custom_op(
    name="alchemiops::_batch_ewald_reciprocal_space_energy_forces_charge_grad",
    outputs=[
        OutputSpec("energies", wp.float64, lambda pos, *_: (pos.shape[0],)),
        OutputSpec("forces", wp.vec3d, lambda pos, *_: (pos.shape[0], 3)),
        OutputSpec("charge_gradients", wp.float64, lambda pos, *_: (pos.shape[0],)),
    ],
    grad_arrays=[
        "energies",
        "forces",
        "charge_gradients",
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Internal: Compute reciprocal-space Ewald energies, forces, and charge grads (batch).

    Supports both float32 and float64 input dtypes. Returns all in input dtype.
    """
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
        )

    # Get warp types based on input dtype
    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)

    # Compute atom_start and atom_end for each system from batch_idx
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
    wp_positions = warp_from_torch(positions, wp_vec, requires_grad=needs_grad_flag)
    wp_charges = warp_from_torch(charges, wp_scalar, requires_grad=needs_grad_flag)
    wp_cell = warp_from_torch(cell, wp_mat, requires_grad=needs_grad_flag)
    # Ensure k_vectors have same dtype as positions
    k_vectors_typed = k_vectors.to(input_dtype)
    wp_k_vectors = warp_from_torch(
        k_vectors_typed, wp_vec, requires_grad=needs_grad_flag
    )
    wp_alpha = warp_from_torch(alpha, wp_scalar, requires_grad=needs_grad_flag)
    wp_batch_idx = warp_from_torch(batch_idx, wp.int32)
    wp_atom_start = warp_from_torch(atom_start, wp.int32)
    wp_atom_end = warp_from_torch(atom_end, wp.int32)

    # Intermediate arrays (always float64)
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

    with WarpAutogradContextManager(needs_grad_flag) as tape:
        # Blocked: one thread per (k-vector, system, atom_block)
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
        # Atom-major: one thread per atom (energy, forces, charge gradients)
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

    # Apply self-energy and background corrections to charge gradients
    # charge_grads contains φ_i (electrostatic potential)
    # Full charge gradient: ∂E/∂q_i = φ_i - 2(α/√π)q_i - (π/α²)(Q_total/V)
    # For batch mode, gather per-atom values from per-system arrays
    alpha_per_atom = alpha[batch_idx]  # shape (N,)
    total_charge_per_atom = total_charge_batch[batch_idx]  # shape (N,)

    self_energy_grad = 2.0 / math.sqrt(PI) * alpha_per_atom * charges
    background_grad = PI / (alpha_per_atom * alpha_per_atom) * total_charge_per_atom
    charge_grads = charge_grads - self_energy_grad - background_grad

    if needs_grad_flag:
        attach_for_backward(
            energies,
            tape=tape,
            energies=wp_energies,
            forces=wp_forces,
            charge_gradients=wp_charge_grads,
            positions=wp_positions,
            charges=wp_charges,
            cell=wp_cell,
            k_vectors=wp_k_vectors,
            alpha=wp_alpha,
        )
    # Return in appropriate dtypes
    return energies.to(input_dtype), forces, charge_grads.to(input_dtype)


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
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Compute real-space Ewald energy and optionally forces and charge gradients.

    Computes the damped Coulomb interactions for atom pairs within the real-space
    cutoff. The complementary error function (erfc) damping ensures rapid
    convergence in real space.

    Formula:

    .. math::

        E_{\\text{real}} = \\frac{1}{2} \\sum_{i \\neq j} q_i q_j \\frac{\\text{erfc}(\\alpha r_{ij})}{r_{ij}}

    Parameters
    ----------
    positions : torch.Tensor, shape (N, 3)
        Atomic coordinates. Supports float32 or float64 dtype.
    charges : torch.Tensor, shape (N,)
        Atomic partial charges in elementary charge units.
    cell : torch.Tensor, shape (3, 3) or (B, 3, 3)
        Unit cell matrices with lattice vectors as rows. Shape (3, 3) is
        automatically promoted to (1, 3, 3).
    alpha : torch.Tensor, shape (1,) or (B,)
        Ewald splitting parameter(s). Controls the real/reciprocal space split.
        Larger α shifts more computation to reciprocal space.
    neighbor_list : torch.Tensor, shape (2, M), dtype=int32, optional
        Neighbor list in COO format. Row 0 contains source atom indices (i),
        row 1 contains target atom indices (j). Each pair should appear once
        (not symmetrized). Mutually exclusive with neighbor_matrix.
    neighbor_ptr : torch.Tensor, shape (N+1,), dtype=int32, optional
        CSR row pointers for neighbor list. neighbor_ptr[i] gives the starting
        index in idx_j (neighbor_list[1]) for atom i's neighbors. Required
        when using neighbor_list format. Provided by neighborlist module.
    neighbor_shifts : torch.Tensor, shape (M, 3), dtype=int32, optional
        Periodic image shifts for each neighbor pair. Entry [k, :] gives the
        integer cell translation for pair k. Required with neighbor_list.
    neighbor_matrix : torch.Tensor, shape (N, max_neighbors), dtype=int32, optional
        Dense neighbor matrix format. Entry [i, k] = j means atom j is the k-th
        neighbor of atom i. Invalid entries should be set to mask_value.
        More cache-friendly for small, fixed neighbor counts.
        Mutually exclusive with neighbor_list.
    neighbor_matrix_shifts : torch.Tensor, shape (N, max_neighbors, 3), dtype=int32, optional
        Periodic image shifts for neighbor_matrix. Required with neighbor_matrix.
    mask_value : int, default=-1
        Value indicating invalid/padded entries in neighbor_matrix.
    batch_idx : torch.Tensor, shape (N,), dtype=int32, optional
        System index for each atom (0 to B-1). Determines kernel dispatch:
        - None: Single-system optimized kernels
        - Provided: Batched kernels for multiple independent systems
    compute_forces : bool, default=False
        Whether to compute explicit forces. Forces are computed analytically
        within the kernel (not via autograd).
    compute_charge_gradients : bool, default=False
        Whether to compute analytical charge gradients (∂E/∂q_i). Useful for
        second-derivative training in ML potentials, as Warp requires analytical
        first derivatives to compute second derivatives via autograd.

    Returns
    -------
    energies : torch.Tensor, shape (N,)
        Per-atom real-space energy contribution (sum gives total E_real).
    forces : torch.Tensor, shape (N, 3), optional
        Real-space forces on each atom. Only returned if compute_forces=True.
    charge_gradients : torch.Tensor, shape (N,), optional
        Gradient ∂E_real/∂q_i. Only returned if compute_charge_gradients=True.

    Return Patterns
    ---------------
    - ``compute_forces=False, compute_charge_gradients=False``: energies
    - ``compute_forces=True, compute_charge_gradients=False``: (energies, forces)
    - ``compute_forces=False, compute_charge_gradients=True``: (energies, charge_gradients)
    - ``compute_forces=True, compute_charge_gradients=True``: (energies, forces, charge_gradients)

    Raises
    ------
    ValueError
        If neither neighbor_list nor neighbor_matrix is provided.

    Examples
    --------
    Energy only with neighbor list::

        >>> energies = ewald_real_space(
        ...     positions, charges, cell, alpha,
        ...     neighbor_list=nl, neighbor_shifts=shifts,
        ... )
        >>> total_energy = energies.sum()

    With explicit forces::

        >>> energies, forces = ewald_real_space(
        ...     positions, charges, cell, alpha,
        ...     neighbor_list=nl, neighbor_shifts=shifts,
        ...     compute_forces=True,
        ... )

    With charge gradients for ML training::

        >>> energies, forces, charge_grads = ewald_real_space(
        ...     positions, charges, cell, alpha,
        ...     neighbor_list=nl, neighbor_shifts=shifts,
        ...     compute_forces=True, compute_charge_gradients=True,
        ... )
        >>> # charge_grads can be used to compute charge Hessian via autograd:

    Using neighbor matrix format::

        >>> energies = ewald_real_space(
        ...     positions, charges, cell, alpha,
        ...     neighbor_matrix=nm, neighbor_matrix_shifts=nm_shifts,
        ...     mask_value=-1,
        ... )

    Batched systems::

        >>> # positions: concatenated atoms, batch_idx: system assignment
        >>> energies = ewald_real_space(
        ...     positions, charges, cells,  # cells shape (B, 3, 3)
        ...     alpha,  # shape (B,)
        ...     batch_idx=batch_idx,
        ...     neighbor_list=nl, neighbor_shifts=shifts,
        ... )

    See Also
    --------
    ewald_reciprocal_space : Reciprocal-space component of Ewald summation.
    ewald_summation : Complete Ewald summation (real + reciprocal).
    estimate_ewald_parameters : Automatic parameter estimation.
    """
    is_batch = batch_idx is not None

    # Dispatch based on compute_charge_gradients, then compute_forces
    if compute_charge_gradients:
        # Use the combined energy+forces+charge_grad kernels
        if neighbor_list is not None:
            if neighbor_ptr is None:
                raise ValueError(
                    "neighbor_ptr is required when using neighbor_list format"
                )
            if is_batch:
                energies, forces, charge_grads = (
                    _batch_ewald_real_space_energy_forces_charge_grad(
                        positions,
                        charges,
                        cell,
                        alpha,
                        batch_idx,
                        neighbor_list,
                        neighbor_ptr,
                        neighbor_shifts,
                    )
                )
            else:
                energies, forces, charge_grads = (
                    _ewald_real_space_energy_forces_charge_grad(
                        positions,
                        charges,
                        cell,
                        alpha,
                        neighbor_list,
                        neighbor_ptr,
                        neighbor_shifts,
                    )
                )
        elif neighbor_matrix is not None:
            if is_batch:
                energies, forces, charge_grads = (
                    _batch_ewald_real_space_energy_forces_charge_grad_matrix(
                        positions,
                        charges,
                        cell,
                        alpha,
                        batch_idx,
                        neighbor_matrix,
                        neighbor_matrix_shifts,
                        mask_value,
                    )
                )
            else:
                energies, forces, charge_grads = (
                    _ewald_real_space_energy_forces_charge_grad_matrix(
                        positions,
                        charges,
                        cell,
                        alpha,
                        neighbor_matrix,
                        neighbor_matrix_shifts,
                        mask_value,
                    )
                )
        else:
            raise ValueError("Either neighbor_list or neighbor_matrix must be provided")

        # Return based on compute_forces flag
        if compute_forces:
            return energies, forces, charge_grads
        else:
            return energies, charge_grads

    # No charge gradients requested - use existing kernels
    if neighbor_list is not None:
        if neighbor_ptr is None:
            raise ValueError("neighbor_ptr is required when using neighbor_list format")
        if is_batch:
            if compute_forces:
                return _batch_ewald_real_space_energy_forces(
                    positions,
                    charges,
                    cell,
                    alpha,
                    batch_idx,
                    neighbor_list,
                    neighbor_ptr,
                    neighbor_shifts,
                )
            else:
                return _batch_ewald_real_space_energy(
                    positions,
                    charges,
                    cell,
                    alpha,
                    batch_idx,
                    neighbor_list,
                    neighbor_ptr,
                    neighbor_shifts,
                )
        else:
            if compute_forces:
                return _ewald_real_space_energy_forces(
                    positions,
                    charges,
                    cell,
                    alpha,
                    neighbor_list,
                    neighbor_ptr,
                    neighbor_shifts,
                )
            else:
                return _ewald_real_space_energy(
                    positions,
                    charges,
                    cell,
                    alpha,
                    neighbor_list,
                    neighbor_ptr,
                    neighbor_shifts,
                )
    elif neighbor_matrix is not None:
        if is_batch:
            if compute_forces:
                return _batch_ewald_real_space_energy_forces_matrix(
                    positions,
                    charges,
                    cell,
                    alpha,
                    batch_idx,
                    neighbor_matrix,
                    neighbor_matrix_shifts,
                    mask_value,
                )
            else:
                return _batch_ewald_real_space_energy_matrix(
                    positions,
                    charges,
                    cell,
                    alpha,
                    batch_idx,
                    neighbor_matrix,
                    neighbor_matrix_shifts,
                    mask_value,
                )
        else:
            if compute_forces:
                return _ewald_real_space_energy_forces_matrix(
                    positions,
                    charges,
                    cell,
                    alpha,
                    neighbor_matrix,
                    neighbor_matrix_shifts,
                    mask_value,
                )
            else:
                return _ewald_real_space_energy_matrix(
                    positions,
                    charges,
                    cell,
                    alpha,
                    neighbor_matrix,
                    neighbor_matrix_shifts,
                    mask_value,
                )
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
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    r"""Compute reciprocal-space Ewald energy and optionally forces and charge gradients.

    Computes the smooth long-range electrostatic contribution using structure
    factors in reciprocal space. Automatically applies self-energy and background
    corrections.

    The total energy is given by

    .. math::

        E_{\text{reciprocal}} = \frac{1}{2V} \sum_{k \in halfspace} G(k) \vert S(k) \vert^2 - E_{\text{self}} - E_{\text{background}}

    where the components are:

    - Green's function: :math:`G(k) = \frac{8\pi}{k^2} \exp\left(-\frac{k^2}{4\alpha^2}\right)`
    - Structure factor: :math:`S(k) = \sum_j q_j \exp(ik \cdot r_j)`
    - Self-energy correction: :math:`E_{\text{self}} = \sum_i \frac{\alpha}{\sqrt{\pi}} q_i^2`
    - Background correction: :math:`E_{\text{background}} = \frac{\pi}{2\alpha^2 V} Q_{\text{total}}^2` (for non-neutral systems)

    Parameters
    ----------
    positions : torch.Tensor, shape (N, 3)
        Atomic coordinates. Supports float32 or float64 dtype.
    charges : torch.Tensor, shape (N,)
        Atomic partial charges in elementary charge units.
    cell : torch.Tensor, shape (3, 3) or (B, 3, 3)
        Unit cell matrices with lattice vectors as rows. Shape (3, 3) is
        automatically promoted to (1, 3, 3).
    k_vectors : torch.Tensor
        Reciprocal lattice vectors from ``generate_k_vectors_ewald_summation``.
        Shape (K, 3) for single system, (B, K, 3) for batch.
        Must be half-space vectors (excludes k=0 and -k for each +k).
    alpha : torch.Tensor, shape (1,) or (B,)
        Ewald splitting parameter(s). Must match values used for real-space.
    batch_idx : torch.Tensor, shape (N,), dtype=int32, optional
        System index for each atom (0 to B-1). Determines kernel dispatch:
        - None: Single-system optimized kernels
        - Provided: Batched kernels for multiple independent systems
    compute_forces : bool, default=False
        Whether to compute explicit reciprocal-space forces.
    compute_charge_gradients : bool, default=False
        Whether to compute analytical charge gradients (∂E/∂q_i). Useful for
        computing charge Hessians in ML potential training.

    Returns
    -------
    energies : torch.Tensor, shape (N,)
        Per-atom reciprocal-space energy (includes self and background corrections).
    forces : torch.Tensor, shape (N, 3), optional
        Reciprocal-space forces. Only returned if compute_forces=True.
    charge_gradients : torch.Tensor, shape (N,), optional
        Charge gradients ∂E_recip/∂q_i including corrections.
        Only returned if compute_charge_gradients=True.

    Return Patterns
    ---------------
    - ``compute_forces=False, compute_charge_gradients=False``: energies
    - ``compute_forces=True, compute_charge_gradients=False``: (energies, forces)
    - ``compute_forces=False, compute_charge_gradients=True``: (energies, charge_gradients)
    - ``compute_forces=True, compute_charge_gradients=True``: (energies, forces, charge_gradients)

    Examples
    --------
    Generate k-vectors and compute energy::

        >>> from nvalchemiops.interactions.electrostatics import (
        ...     generate_k_vectors_ewald_summation
        ... )
        >>> k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff=8.0)
        >>> energies = ewald_reciprocal_space(
        ...     positions, charges, cell, k_vectors, alpha,
        ... )
        >>> total_recip_energy = energies.sum()

    With forces::

        >>> energies, forces = ewald_reciprocal_space(
        ...     positions, charges, cell, k_vectors, alpha,
        ...     compute_forces=True,
        ... )

    With charge gradients for ML training::

        >>> energies, charge_grads = ewald_reciprocal_space(
        ...     positions, charges, cell, k_vectors, alpha,
        ...     compute_charge_gradients=True,
        ... )

    Batched systems::

        >>> # k_vectors shape: (B, K, 3) with same K for all systems
        >>> energies = ewald_reciprocal_space(
        ...     positions, charges, cells, k_vectors, alpha,
        ...     batch_idx=batch_idx,
        ... )

    Notes
    -----
    - k_vectors MUST be generated using ``generate_k_vectors_ewald_summation``,
      which provides half-space k-vectors. Using full k-space vectors will
      double-count and give incorrect energies.
    - For batch mode with varying cell sizes, use the same k_cutoff for all
      systems to ensure consistent K dimension.
    - The charge gradient formula includes corrections for self-energy and
      background, making it suitable for training on charge derivatives.

    See Also
    --------
    ewald_real_space : Real-space component of Ewald summation.
    ewald_summation : Complete Ewald summation (real + reciprocal).
    generate_k_vectors_ewald_summation : Generate k-vectors for this function.
    """
    is_batch = batch_idx is not None

    # Handle charge gradients case (uses combined kernel that computes all three)
    if compute_charge_gradients:
        if is_batch:
            energies, forces, charge_grads = (
                _batch_ewald_reciprocal_space_energy_forces_charge_grad(
                    positions, charges, cell, k_vectors, alpha, batch_idx
                )
            )
        else:
            energies, forces, charge_grads = (
                _ewald_reciprocal_space_energy_forces_charge_grad(
                    positions, charges, cell, k_vectors, alpha
                )
            )

        # Return based on compute_forces flag
        if compute_forces:
            return energies, forces, charge_grads
        else:
            return energies, charge_grads

    # No charge gradients requested - use existing kernels
    if is_batch:
        if compute_forces:
            return _batch_ewald_reciprocal_space_energy_forces(
                positions, charges, cell, k_vectors, alpha, batch_idx
            )
        else:
            return _batch_ewald_reciprocal_space_energy(
                positions, charges, cell, k_vectors, alpha, batch_idx
            )
    else:
        if compute_forces:
            e, f = _ewald_reciprocal_space_energy_forces(
                positions, charges, cell, k_vectors, alpha
            )
            return e, f
        else:
            return _ewald_reciprocal_space_energy(
                positions, charges, cell, k_vectors, alpha
            )


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
    accuracy: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    """Complete Ewald summation for long-range electrostatics.

    Computes total Coulomb energy by combining real-space and reciprocal-space
    contributions with self-energy and background corrections. Supports automatic
    parameter estimation, batched calculations, and automatic differentiation.

    Formula
    -------
    The total Ewald energy is:

    Total Energy Formula:

    .. math::

        E_{\\text{total}} = E_{\\text{real}} + E_{\\text{reciprocal}} - E_{\\text{self}} - E_{\\text{background}}

    where:

    .. math::

        E_{\\text{real}} = \\frac{1}{2} \\sum_{i \\neq j} q_i q_j \\frac{\\text{erfc}(\\alpha r_{ij}/\\sqrt{2})}{r_{ij}}
        E_{\\text{reciprocal}} = \\frac{1}{2V} \\sum_{k \\in halfspace} G(k) \\vert S(k) \\vert^2
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
        - None: Automatically estimated from accuracy using Kolafa-Perram formula
        Larger α shifts more computation to reciprocal space.
    k_vectors : torch.Tensor, optional
        Pre-computed reciprocal lattice vectors. Shape (K, 3) for single system,
        (B, K, 3) for batch. If None, generated from k_cutoff using
        ``generate_k_vectors_ewald_summation``.
    k_cutoff : float, optional
        K-space cutoff (maximum |k| magnitude) for generating k_vectors.
        If None with alpha=None, estimated from accuracy.
        Typical values: 8-12 Å⁻¹.
    batch_idx : torch.Tensor, shape (N,), dtype=int32, optional
        System index for each atom (0 to B-1). Determines execution mode:
        - None: Single-system optimized kernels
        - Provided: Batched kernels for multiple independent systems
    neighbor_list : torch.Tensor, shape (2, M), dtype=int32, optional
        Neighbor pairs in COO format. Row 0 = source indices, row 1 = target.
        Mutually exclusive with neighbor_matrix.
    neighbor_ptr : torch.Tensor, shape (N+1,), dtype=int32, optional
        CSR row pointers for neighbor list. neighbor_ptr[i] gives the starting
        index in idx_j for atom i's neighbors. Required with neighbor_list.
        Provided by neighborlist module.
    neighbor_shifts : torch.Tensor, shape (M, 3), dtype=int32, optional
        Periodic image shifts for each neighbor pair. Required with neighbor_list.
    neighbor_matrix : torch.Tensor, shape (N, max_neighbors), dtype=int32, optional
        Dense neighbor matrix. Entry [i, k] = j means j is k-th neighbor of i.
        Invalid entries should be set to mask_value.
        Mutually exclusive with neighbor_list.
    neighbor_matrix_shifts : torch.Tensor, shape (N, max_neighbors, 3), dtype=int32, optional
        Periodic image shifts for neighbor_matrix. Required with neighbor_matrix.
    mask_value : int, optional
        Value indicating invalid entries in neighbor_matrix. Defaults to N.
    compute_forces : bool, default=False
        Whether to compute explicit analytical forces.
    accuracy : float, default=1e-6
        Target relative accuracy for automatic parameter estimation.
        Only used when alpha or k_cutoff is None.
        Smaller values increase accuracy but also computational cost.

    Returns
    -------
    energies : torch.Tensor, shape (N,)
        Per-atom contribution to total Ewald energy. Sum gives total energy.
    forces : torch.Tensor, shape (N, 3), optional
        Forces on each atom. Only returned if compute_forces=True.

    Raises
    ------
    ValueError
        If neither neighbor_list nor neighbor_matrix is provided.
    TypeError
        If alpha has an unsupported type.

    Examples
    --------
    Automatic parameter estimation (recommended for most cases)::

        >>> energies, forces = ewald_summation(
        ...     positions, charges, cell,
        ...     neighbor_list=nl, neighbor_shifts=shifts,
        ...     accuracy=1e-6,
        ... )
        >>> total_energy = energies.sum()

    Explicit parameters for reproducibility::

        >>> energies, forces = ewald_summation(
        ...     positions, charges, cell,
        ...     alpha=0.3, k_cutoff=8.0,
        ...     neighbor_list=nl, neighbor_shifts=shifts,
        ... )

    Using neighbor matrix format::

        >>> energies, forces = ewald_summation(
        ...     positions, charges, cell,
        ...     alpha=0.3, k_cutoff=8.0,
        ...     neighbor_matrix=nm, neighbor_matrix_shifts=nm_shifts,
        ...     mask_value=-1,
        ... )

    Batched systems (multiple independent structures)::

        >>> # positions: concatenated atoms from all systems
        >>> # batch_idx: [0,0,0,0, 1,1,1,1, 2,2,2,2] for 4 atoms × 3 systems
        >>> energies, forces = ewald_summation(
        ...     positions, charges, cells,  # cells shape (3, 3, 3)
        ...     alpha=torch.tensor([0.3, 0.35, 0.3]),
        ...     batch_idx=batch_idx,
        ...     k_cutoff=8.0,
        ...     neighbor_list=nl, neighbor_shifts=shifts,
        ... )

    Energy-only (skips force computation for speed)::

        >>> energies = ewald_summation(
        ...     positions, charges, cell,
        ...     alpha=0.3, k_cutoff=8.0,
        ...     neighbor_list=nl, neighbor_shifts=shifts,
        ...     compute_forces=False,
        ... )

    Using autograd for gradients::

        >>> positions.requires_grad_(True)
        >>> energies, forces = ewald_summation(
        ...     positions, charges, cell,
        ...     neighbor_list=nl, neighbor_shifts=shifts,
        ... )
        >>> total_energy = energies.sum()
        >>> total_energy.backward()
        >>> autograd_forces = -positions.grad

    Notes
    -----
    Automatic Parameter Estimation (when alpha or k_cutoff is None):
        Uses the Kolafa-Perram formula:

    .. math::

        \\begin{aligned}
        \\eta &= \\frac{(V^2 / N)^{1/6}}{\\sqrt{2\\pi}} \\\\
        \\alpha &= \\frac{1}{\\sqrt{2} \\eta} \\\\
        k_{\\text{cutoff}} &= \\frac{\\sqrt{-2 \\ln(\\varepsilon)}}{\\eta} \\\\
        r_{\\text{cutoff}} &= \\sqrt{-2 \\ln(\\varepsilon)} \\cdot \\eta
        \\end{aligned}

    This balances computational cost between real and reciprocal space.

    Autograd Support:
        All inputs (positions, charges, cell) support gradient computation.
        For positions, :math:`-\\nabla E` gives forces, which should match the explicit forces.

    See Also
    --------
    ewald_real_space : Real-space component only
    ewald_reciprocal_space : Reciprocal-space component only
    estimate_ewald_parameters : Automatic parameter estimation
    EwaldParameters : Container for Ewald parameters
    """
    device = positions.device
    dtype = positions.dtype
    num_atoms = positions.shape[0]

    # Prepare cell
    cell, num_systems = _prepare_cell(cell)

    # Estimate parameters if not provided
    if alpha is None or (k_cutoff is None and k_vectors is None):
        params = estimate_ewald_parameters(positions, cell, batch_idx, accuracy)
        if alpha is None:
            alpha = params.alpha
        if k_cutoff is None:
            # For batch mode, use max k_cutoff to ensure convergence for all systems
            k_cutoff = params.reciprocal_space_cutoff

    # Prepare alpha
    alpha_tensor = _prepare_alpha(alpha, num_systems, dtype, device)

    # Generate k_vectors if not provided
    if k_vectors is None:
        k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff)

    # Set default mask_value
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
    )
    # Combine results
    if compute_forces:
        total_energies = rs[0] + rec[0]
        total_forces = rs[1] + rec[1]
        return total_energies, total_forces
    else:
        return rs + rec
