# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Damped Shifted Force (DSF) Electrostatics - PyTorch Bindings
=============================================================

This module provides PyTorch bindings for DSF electrostatic calculations.
It wraps the framework-agnostic Warp launchers from
``nvalchemiops.interactions.electrostatics.dsf``.

Public API
----------
- ``dsf_coulomb()``: Compute DSF electrostatic energy, forces, and virial

Features:
- Both undamped (alpha=0, shifted-force Coulomb) and damped DSF
- Both neighbor list (CSR) and neighbor matrix formats
- Batched calculations
- Charge gradient support for MLIP training (via straight-through trick)
- Optional forces and virial computation
- float32 and float64 precision support

Integration Pattern
-------------------
This module uses ``torch.library.custom_op`` with ``mutates_args`` and
``register_fake`` instead of the ``@warp_custom_op`` / ``WarpAutogradContextManager``
pattern used by the Coulomb bindings. This is intentional:

- DSF does not require double backward (Hessian / gradients of forces w.r.t.
  positions). Forces and charge gradients are computed analytically in the
  forward Warp kernel, so there is no need for a Warp backward tape.
- Charge gradients are propagated through PyTorch autograd via a
  "straight-through trick": a zero-valued correction term whose gradient
  equals the kernel-computed dE/dq is added to the energy tensor.
- ``register_fake`` enables ``torch.compile`` compatibility.

Examples
--------
>>> # Basic DSF energy and forces
>>> energy, forces = dsf_coulomb(
...     positions, charges, cutoff=10.0, alpha=0.2,
...     neighbor_list=neighbor_list, neighbor_ptr=neighbor_ptr,
... )

>>> # With charge gradients for MLIP training
>>> charges.requires_grad_(True)
>>> energy, forces = dsf_coulomb(
...     positions, charges, cutoff=10.0, alpha=0.2,
...     neighbor_list=neighbor_list, neighbor_ptr=neighbor_ptr,
... )
>>> energy.sum().backward()
>>> charge_grads = charges.grad  # dE/dq_i
"""

from __future__ import annotations

import torch
import warp as wp

from nvalchemiops.interactions.electrostatics.dsf import (
    dsf_csr as wp_dsf_csr,
)
from nvalchemiops.interactions.electrostatics.dsf import (
    dsf_matrix as wp_dsf_matrix,
)
from nvalchemiops.torch.types import get_wp_dtype, get_wp_mat_dtype, get_wp_vec_dtype

__all__ = [
    "dsf_coulomb",
]


# ==============================================================================
# Internal Custom Ops
# ==============================================================================


@torch.library.custom_op(
    "nvalchemiops::dsf_csr_op",
    mutates_args=("energy", "forces", "virial", "charge_grad"),
)
def _dsf_csr_op(
    positions: torch.Tensor,
    charges: torch.Tensor,
    idx_j: torch.Tensor,
    neighbor_ptr: torch.Tensor,
    cutoff: float,
    alpha: float,
    energy: torch.Tensor,
    forces: torch.Tensor,
    virial: torch.Tensor,
    charge_grad: torch.Tensor,
    compute_forces: bool = True,
    compute_virial: bool = False,
    compute_charge_grad: bool = False,
    cell: torch.Tensor | None = None,
    unit_shifts: torch.Tensor | None = None,
    batch_idx: torch.Tensor | None = None,
    device: str | None = None,
) -> None:
    """Internal custom op: DSF with CSR neighbor list (optional PBC)."""
    num_atoms = positions.size(0)
    if num_atoms == 0:
        return

    if device is None:
        device = str(positions.device)

    energy.zero_()
    if compute_forces:
        forces.zero_()
    if compute_virial:
        virial.zero_()
    if compute_charge_grad:
        charge_grad.zero_()

    input_dtype = positions.dtype
    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)

    positions_wp = wp.from_torch(positions.detach(), dtype=wp_vec, return_ctype=True)
    charges_wp = wp.from_torch(charges.detach(), dtype=wp_scalar, return_ctype=True)
    idx_j_wp = wp.from_torch(idx_j, dtype=wp.int32, return_ctype=True)
    neighbor_ptr_wp = wp.from_torch(neighbor_ptr, dtype=wp.int32, return_ctype=True)

    energy_wp = wp.from_torch(energy, dtype=wp.float64, return_ctype=True)
    forces_wp = wp.from_torch(forces, dtype=wp_vec, return_ctype=True)
    virial_wp = wp.from_torch(virial, dtype=wp_mat, return_ctype=True)
    charge_grad_wp = wp.from_torch(charge_grad, dtype=wp_scalar, return_ctype=True)

    cell_wp = None
    unit_shifts_wp = None
    if cell is not None:
        cell_wp = wp.from_torch(cell.detach(), dtype=wp_mat, return_ctype=True)
    if unit_shifts is not None:
        unit_shifts_wp = wp.from_torch(unit_shifts, dtype=wp.vec3i, return_ctype=True)

    batch_idx_wp = None
    if batch_idx is not None:
        batch_idx_wp = wp.from_torch(batch_idx, dtype=wp.int32, return_ctype=True)

    wp_dsf_csr(
        positions=positions_wp,
        charges=charges_wp,
        idx_j=idx_j_wp,
        neighbor_ptr=neighbor_ptr_wp,
        cutoff=cutoff,
        alpha=alpha,
        energy=energy_wp,
        forces=forces_wp,
        virial=virial_wp,
        charge_grad=charge_grad_wp,
        cell=cell_wp,
        unit_shifts=unit_shifts_wp,
        device=device,
        batch_idx=batch_idx_wp,
        compute_forces=compute_forces,
        compute_virial=compute_virial,
        compute_charge_grad=compute_charge_grad,
        wp_scalar_type=wp_scalar,
    )


@_dsf_csr_op.register_fake
def _dsf_csr_op_fake(
    positions: torch.Tensor,
    charges: torch.Tensor,
    idx_j: torch.Tensor,
    neighbor_ptr: torch.Tensor,
    cutoff: float,
    alpha: float,
    energy: torch.Tensor,
    forces: torch.Tensor,
    virial: torch.Tensor,
    charge_grad: torch.Tensor,
    compute_forces: bool = True,
    compute_virial: bool = False,
    compute_charge_grad: bool = False,
    cell: torch.Tensor | None = None,
    unit_shifts: torch.Tensor | None = None,
    batch_idx: torch.Tensor | None = None,
    device: str | None = None,
) -> None:
    pass


@torch.library.custom_op(
    "nvalchemiops::dsf_matrix_op",
    mutates_args=("energy", "forces", "virial", "charge_grad"),
)
def _dsf_matrix_op(
    positions: torch.Tensor,
    charges: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    cutoff: float,
    alpha: float,
    fill_value: int,
    energy: torch.Tensor,
    forces: torch.Tensor,
    virial: torch.Tensor,
    charge_grad: torch.Tensor,
    compute_forces: bool = True,
    compute_virial: bool = False,
    compute_charge_grad: bool = False,
    cell: torch.Tensor | None = None,
    neighbor_matrix_shifts: torch.Tensor | None = None,
    batch_idx: torch.Tensor | None = None,
    device: str | None = None,
) -> None:
    """Internal custom op: DSF with neighbor matrix (optional PBC)."""
    num_atoms = positions.size(0)
    if num_atoms == 0:
        return

    if device is None:
        device = str(positions.device)

    energy.zero_()
    if compute_forces:
        forces.zero_()
    if compute_virial:
        virial.zero_()
    if compute_charge_grad:
        charge_grad.zero_()

    input_dtype = positions.dtype
    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)

    positions_wp = wp.from_torch(positions.detach(), dtype=wp_vec, return_ctype=True)
    charges_wp = wp.from_torch(charges.detach(), dtype=wp_scalar, return_ctype=True)
    neighbor_matrix_wp = wp.from_torch(
        neighbor_matrix, dtype=wp.int32, return_ctype=True
    )

    energy_wp = wp.from_torch(energy, dtype=wp.float64, return_ctype=True)
    forces_wp = wp.from_torch(forces, dtype=wp_vec, return_ctype=True)
    virial_wp = wp.from_torch(virial, dtype=wp_mat, return_ctype=True)
    charge_grad_wp = wp.from_torch(charge_grad, dtype=wp_scalar, return_ctype=True)

    cell_wp = None
    neighbor_matrix_shifts_wp = None
    if cell is not None:
        cell_wp = wp.from_torch(cell.detach(), dtype=wp_mat, return_ctype=True)
    if neighbor_matrix_shifts is not None:
        neighbor_matrix_shifts_wp = wp.from_torch(
            neighbor_matrix_shifts, dtype=wp.vec3i, return_ctype=True
        )

    batch_idx_wp = None
    if batch_idx is not None:
        batch_idx_wp = wp.from_torch(batch_idx, dtype=wp.int32, return_ctype=True)

    wp_dsf_matrix(
        positions=positions_wp,
        charges=charges_wp,
        neighbor_matrix=neighbor_matrix_wp,
        cutoff=cutoff,
        alpha=alpha,
        fill_value=fill_value,
        energy=energy_wp,
        forces=forces_wp,
        virial=virial_wp,
        charge_grad=charge_grad_wp,
        cell=cell_wp,
        neighbor_matrix_shifts=neighbor_matrix_shifts_wp,
        device=device,
        batch_idx=batch_idx_wp,
        compute_forces=compute_forces,
        compute_virial=compute_virial,
        compute_charge_grad=compute_charge_grad,
        wp_scalar_type=wp_scalar,
    )


@_dsf_matrix_op.register_fake
def _dsf_matrix_op_fake(
    positions: torch.Tensor,
    charges: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    cutoff: float,
    alpha: float,
    fill_value: int,
    energy: torch.Tensor,
    forces: torch.Tensor,
    virial: torch.Tensor,
    charge_grad: torch.Tensor,
    compute_forces: bool = True,
    compute_virial: bool = False,
    compute_charge_grad: bool = False,
    cell: torch.Tensor | None = None,
    neighbor_matrix_shifts: torch.Tensor | None = None,
    batch_idx: torch.Tensor | None = None,
    device: str | None = None,
) -> None:
    pass


# ==============================================================================
# Public API
# ==============================================================================


def dsf_coulomb(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cutoff: float,
    alpha: float = 0.2,
    cell: torch.Tensor | None = None,
    batch_idx: torch.Tensor | None = None,
    # Neighbor list (CSR) format
    neighbor_list: torch.Tensor | None = None,
    neighbor_ptr: torch.Tensor | None = None,
    unit_shifts: torch.Tensor | None = None,
    # Neighbor matrix format
    neighbor_matrix: torch.Tensor | None = None,
    neighbor_matrix_shifts: torch.Tensor | None = None,
    fill_value: int | None = None,
    # Control flags
    compute_forces: bool = True,
    compute_virial: bool = False,
    num_systems: int | None = None,
    device: str | None = None,
) -> (
    tuple[torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
):
    """Compute DSF electrostatic energy, forces, and virial.

    The Damped Shifted Force (DSF) method is a pairwise O(N) electrostatic
    summation technique that ensures both potential energy and forces smoothly
    vanish at a defined cutoff radius.

    Supports float32 and float64 input precision. Energy is always returned
    in float64. Forces, virial, and charge gradients match the input precision.

    Parameters
    ----------
    positions : torch.Tensor, shape (num_atoms, 3)
        Atomic coordinates (float32 or float64).
    charges : torch.Tensor, shape (num_atoms,)
        Atomic charges (must match positions dtype). If requires_grad=True,
        charge gradients (dE/dq) will be propagated through autograd.
    cutoff : float
        Cutoff radius beyond which interactions are zero.
    alpha : float, default 0.2
        Damping parameter. Set to 0.0 for shifted-force bare Coulomb.
    cell : torch.Tensor, shape (num_systems, 3, 3), optional
        Unit cell matrices for periodic boundary conditions.
    batch_idx : torch.Tensor, shape (num_atoms,), dtype=int32, optional
        System index for each atom. If None, all atoms in one system.
    neighbor_list : torch.Tensor, shape (2, num_pairs), dtype=int32, optional
        Neighbor list in COO format. Row 1 contains destination atoms.
    neighbor_ptr : torch.Tensor, shape (num_atoms+1,), dtype=int32, optional
        CSR row pointers (required with neighbor_list).
    unit_shifts : torch.Tensor, shape (num_pairs, 3), dtype=int32, optional
        Integer unit cell shifts for PBC (required with neighbor_list + cell).
    neighbor_matrix : torch.Tensor, shape (num_atoms, max_neighbors), dtype=int32, optional
        Dense neighbor matrix format.
    neighbor_matrix_shifts : torch.Tensor, shape (num_atoms, max_neighbors, 3), dtype=int32, optional
        Integer unit cell shifts for matrix format PBC.
    fill_value : int, optional
        Padding indicator for neighbor_matrix. Defaults to num_atoms.
    compute_forces : bool, default True
        Whether to compute forces.
    compute_virial : bool, default False
        Whether to compute virial tensor (requires PBC and compute_forces).
    num_systems : int, optional
        Number of systems. Inferred from batch_idx or cell if not given.
    device : str, optional
        Warp device string. Inferred from positions if not given.

    Returns
    -------
    energy : torch.Tensor, shape (num_systems,), dtype=float64
        Per-system electrostatic energy (always float64). If charges.requires_grad,
        this tensor is connected to the autograd graph for charge gradients.
    forces : torch.Tensor, shape (num_atoms, 3), dtype matches input
        Per-atom forces. Only returned if compute_forces=True.
    virial : torch.Tensor, shape (num_systems, 3, 3), dtype matches input
        Per-system virial tensor. Only returned if compute_virial=True.

    Notes
    -----
    - Assumes a full neighbor list (each pair appears in both directions).
    - For MLIP training with geometry-dependent charges, set
      ``charges.requires_grad_(True)`` before calling. After
      ``energy.sum().backward()``, ``charges.grad`` will contain dE/dq.
    - Charge gradients (dE/dq) are computed when
      ``charges.requires_grad=True``, regardless of ``compute_forces``.
    - The returned ``energy`` tensor is **not** differentiable w.r.t.
      ``positions`` or ``cell`` through PyTorch autograd. Forces are
      computed analytically by the Warp kernel, not via autograd.

    Examples
    --------
    >>> # Basic energy + forces
    >>> energy, forces = dsf_coulomb(positions, charges, cutoff=10.0, alpha=0.2,
    ...     neighbor_list=nl, neighbor_ptr=ptr)

    >>> # MLIP workflow with charge gradients
    >>> charges = model(positions)  # Predict charges from geometry
    >>> charges.requires_grad_(True)
    >>> energy, forces = dsf_coulomb(positions, charges, cutoff=10.0, alpha=0.2,
    ...     neighbor_list=nl, neighbor_ptr=ptr)
    >>> loss = (energy - ref_energy).pow(2).sum()
    >>> loss.backward()  # charges.grad now contains dE/dq * dloss/dE
    """
    # Validate inputs
    if compute_virial and not compute_forces:
        raise ValueError("compute_virial=True requires compute_forces=True")

    if compute_virial and cell is None:
        raise ValueError(
            "compute_virial=True requires periodic boundary conditions (cell)"
        )

    # Validate neighbor format: exactly one of list or matrix must be provided
    use_list = neighbor_list is not None
    use_matrix = neighbor_matrix is not None

    if not use_list and not use_matrix:
        raise ValueError(
            "Must provide either neighbor_list (with neighbor_ptr) or neighbor_matrix"
        )

    if use_list and use_matrix:
        raise ValueError(
            "Cannot provide both neighbor list and neighbor matrix formats"
        )

    if use_list and neighbor_ptr is None:
        raise ValueError("neighbor_ptr is required when using neighbor_list format")

    # Validate PBC shift tensors when cell is provided
    if cell is not None:
        if use_list and unit_shifts is None:
            raise ValueError(
                "unit_shifts is required when using neighbor_list format with "
                "periodic boundary conditions (cell)"
            )
        if use_matrix and neighbor_matrix_shifts is None:
            raise ValueError(
                "neighbor_matrix_shifts is required when using neighbor_matrix format "
                "with periodic boundary conditions (cell)"
            )

    if charges.dtype != positions.dtype:
        raise ValueError(
            f"charges dtype ({charges.dtype}) must match positions dtype ({positions.dtype})"
        )

    input_dtype = positions.dtype

    # Charge gradients are computed whenever charges require grad,
    # regardless of whether forces are requested.
    compute_charge_grad = charges.requires_grad

    # Get shapes
    num_atoms = positions.size(0)

    if num_atoms == 0:
        if num_systems is None:
            num_systems = 1
        dev = positions.device
        empty_energy = torch.zeros(num_systems, dtype=torch.float64, device=dev)
        if not compute_forces:
            return (empty_energy,)
        empty_forces = torch.zeros((0, 3), dtype=input_dtype, device=dev)
        if not compute_virial:
            return empty_energy, empty_forces
        empty_virial = torch.zeros((num_systems, 3, 3), dtype=input_dtype, device=dev)
        return empty_energy, empty_forces, empty_virial

    # Determine number of systems
    if num_systems is None:
        if batch_idx is None:
            num_systems = 1
        elif cell is not None:
            num_systems = cell.size(0)
        else:
            num_systems = int(batch_idx.max().item()) + 1

    # Ensure cell matches input dtype
    if cell is not None:
        cell = cell.to(dtype=input_dtype)

    # Allocate output tensors
    dev = positions.device
    energy = torch.zeros(num_systems, dtype=torch.float64, device=dev)

    if compute_forces:
        forces_out = torch.zeros((num_atoms, 3), dtype=input_dtype, device=dev)
    else:
        forces_out = torch.empty((0, 3), dtype=input_dtype, device=dev)

    if compute_charge_grad:
        charge_grad_out = torch.zeros(num_atoms, dtype=input_dtype, device=dev)
    else:
        charge_grad_out = torch.empty(0, dtype=input_dtype, device=dev)

    if compute_virial:
        virial_out = torch.zeros((num_systems, 3, 3), dtype=input_dtype, device=dev)
    else:
        virial_out = torch.empty((0, 3, 3), dtype=input_dtype, device=dev)

    # Dispatch to appropriate custom op (2-way: by neighbor format)
    if neighbor_matrix is not None:
        if fill_value is None:
            fill_value = num_atoms

        _dsf_matrix_op(
            positions=positions,
            charges=charges,
            neighbor_matrix=neighbor_matrix,
            cutoff=cutoff,
            alpha=alpha,
            fill_value=fill_value,
            energy=energy,
            forces=forces_out,
            virial=virial_out,
            charge_grad=charge_grad_out,
            compute_forces=compute_forces,
            compute_virial=compute_virial,
            compute_charge_grad=compute_charge_grad,
            cell=cell,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            batch_idx=batch_idx,
            device=device,
        )
    else:
        idx_j = neighbor_list[1].contiguous()

        _dsf_csr_op(
            positions=positions,
            charges=charges,
            idx_j=idx_j,
            neighbor_ptr=neighbor_ptr,
            cutoff=cutoff,
            alpha=alpha,
            energy=energy,
            forces=forces_out,
            virial=virial_out,
            charge_grad=charge_grad_out,
            compute_forces=compute_forces,
            compute_virial=compute_virial,
            compute_charge_grad=compute_charge_grad,
            cell=cell,
            unit_shifts=unit_shifts,
            batch_idx=batch_idx,
            device=device,
        )

    # Charge gradient support via straight-through trick
    # This makes energy differentiable w.r.t. charges without Warp tape.
    # The correction term has value 0 but gradient dE/dq w.r.t. charges.
    if compute_charge_grad:
        # Cast to float64 for numerical stability in the correction computation.
        # charge_grad_out is in input precision (possibly float32).
        cg_f64 = charge_grad_out.to(dtype=torch.float64)
        charges_f64 = charges.to(dtype=torch.float64)
        charges_detached_f64 = charges_f64.detach()
        # correction[i] = dE_dq[i] * (q[i] - q_detached[i]) = dE_dq[i] * 0 = 0
        # but d(correction)/d(q[i]) = dE_dq[i]
        correction = cg_f64.detach() * (charges_f64 - charges_detached_f64)

        if batch_idx is not None:
            system_correction = torch.zeros_like(energy)
            system_correction.scatter_add_(0, batch_idx.long(), correction)
        else:
            # Single system: sum all corrections
            system_correction = correction.sum().unsqueeze(0)

        energy = energy + system_correction

    # Build return tuple
    if not compute_forces:
        return (energy,)
    if not compute_virial:
        return energy, forces_out
    return energy, forces_out, virial_out
