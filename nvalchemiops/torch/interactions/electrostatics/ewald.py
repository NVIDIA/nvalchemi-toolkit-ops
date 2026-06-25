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
- Energy autograd for differentiable training

The full ``ewald_summation`` API treats energy autograd as the differentiable
contract; its direct-output flags warn and are deprecated. The component APIs
(``ewald_real_space`` and ``ewald_reciprocal_space``) intentionally retain direct
forces as no-autograd MD/inference escape hatches. Component charge-gradient,
virial, and hybrid direct outputs are deprecated training-style outputs and warn.

Examples
--------
>>> # Complete Ewald summation
>>> energies = ewald_summation(
...     positions, charges, cell,
...     neighbor_list=nl, neighbor_ptr=neighbor_ptr, neighbor_shifts=shifts,
...     accuracy=1e-6,
... )
>>> forces = -torch.autograd.grad(energies.sum(), positions, create_graph=True)[0]

>>> # Fixed-cell loop: precompute the actual reciprocal vectors once
>>> k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff=8.0)
>>> energies = ewald_summation(
...     positions, charges, cell,
...     alpha=alpha, k_vectors=k_vectors,
...     neighbor_list=nl, neighbor_ptr=neighbor_ptr, neighbor_shifts=shifts,
... )

>>> # Changing-cell loop: precompute conservative Miller half-bounds once
>>> energies = ewald_summation(
...     positions, charges, cell_t,
...     alpha=alpha, k_cutoff=8.0, miller_bounds=(16, 16, 16),
...     neighbor_list=nl, neighbor_ptr=neighbor_ptr, neighbor_shifts=shifts,
... )

>>> # Separate real and reciprocal components with direct no-autograd forces
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

import warnings

import torch

from nvalchemiops.torch.interactions.electrostatics._ewald_corrections_chain import (
    ewald_energy_corrections,
    ewald_energy_corrections_batch,
)
from nvalchemiops.torch.interactions.electrostatics._ewald_direct import (
    reciprocal_space_direct,
)
from nvalchemiops.torch.interactions.electrostatics._ewald_real_chain import (
    real_space_cell_connect,
)
from nvalchemiops.torch.interactions.electrostatics._registration import (
    ensure_electrostatics_ops_registered,
)
from nvalchemiops.torch.interactions.electrostatics._util import (
    _build_electrostatic_result,
    _combine_electrostatic_outputs,
    _compiled_direct_output_deprecation_signal,
    _component_direct_output_deprecation_msg,
    _detach_setup_tensor,
    _direct_output_deprecation_msg,
    _has_potentially_geometry_dependent_charges,
    _InjectCachedEvalGrad,
    _InjectCachedEvalGradWithFallback,
    _InjectChargeGrad,
    _unpack_electrostatic_outputs,
)
from nvalchemiops.torch.interactions.electrostatics.k_vectors import (
    generate_k_vectors_ewald_summation,
)
from nvalchemiops.torch.interactions.electrostatics.parameters import (
    estimate_ewald_parameters,
)
from nvalchemiops.torch.interactions.electrostatics.slab import (
    compute_slab_correction as _compute_slab_correction,
)

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
########################### Internal Energy Assembly (explicit chains) ####################
###########################################################################################
#
# The differentiable public energy is assembled from the explicit factory chains
# registered in ``_ewald_real_chain`` / ``_ewald_recip_chain`` (forward energy ->
# backward -> double_backward), plus Torch-native pieces that the
# kernels do not own:
#
#   * real-space ``cell`` gradient -- :func:`real_space_cell_connect` (the literal
#     ``dE/dcell`` via the differentiable periodic shift ``unit_shifts @ cell``);
#   * reciprocal self-energy + background corrections (closed-form in charges /
#     alpha / volume) and the ``k_vectors(cell)`` / ``volume(cell)`` maps that carry
#     the reciprocal ``cell`` gradient through Torch.
#
# When nothing requires grad, the forward-only ``_DerivState.E`` kernel runs with no
# derivative state (inference performance preserved). The deprecated direct flags
# (``compute_forces`` / ``compute_charge_gradients`` / ``compute_virial`` /
# ``hybrid_forces``) are served by :mod:`_ewald_direct` (tape-free forward kernels).

# Output dtype convention (unchanged):
#   - Energies: always float64 for numerical-stability accumulation.
#   - Forces / virial / charge gradients: accumulated in float64, returned in the
#     input precision (float32 or float64).


def _atom_ranges(
    batch_idx: torch.Tensor, num_systems: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-system [start, end) atom index ranges from a sorted ``batch_idx``.

    The batched factory kernels assume atoms are grouped contiguously by system
    (the existing batched-Ewald contract); ``atom_start``/``atom_end`` are int32.
    """
    device = batch_idx.device
    counts = torch.zeros(num_systems, dtype=torch.long, device=device)
    counts = counts.index_add(
        0,
        batch_idx,
        torch.ones(batch_idx.shape[0], dtype=counts.dtype, device=device),
    )
    ends = torch.cumsum(counts, dim=0)
    starts = ends - counts
    return starts.to(torch.int32), ends.to(torch.int32)


def _attach_virial_charge_grad(
    virial_value: torch.Tensor,
    charges: torch.Tensor,
    energy_fn,
    positions: torch.Tensor,
    cell: torch.Tensor,
    batch_idx: torch.Tensor | None,
    k_vectors_2d: torch.Tensor | None = None,
) -> torch.Tensor:
    """Give the direct (kernel) ``virial`` a charge gradient via strain autograd.

    The deprecated ``compute_virial`` output remains differentiable w.r.t.
    ``charges``. The direct factory kernel output is forward-only, so this
    re-attaches the charge gradient with a straight-through:
    the value stays the kernel ``virial_value`` while the gradient comes from the
    row-vector displacement virial ``W = -dE/dstrain`` of the autograd-connected
    energy, recomputed with ``positions`` / ``cell`` detached so only the
    ``charges`` pathway is live (forces / cell gradients of the direct virial
    stay forward-only, as before).

    ``k_vectors_2d``: when given (reciprocal path), the k-vectors are deformed with
    the strain as ``k_s = k @ inv(deform).T`` (the reciprocal lattice transforms
    contravariantly with ``cell_s = cell @ deform``) and the 4-argument
    ``energy_fn(p, q, c, k)`` is called, matching the kernel virial's k-vector
    strain response.
    """
    if not charges.requires_grad:
        return virial_value
    num_systems = cell.shape[0]
    pos_d = positions.detach()
    cell_d = cell.detach()
    eye = torch.eye(3, device=positions.device, dtype=positions.dtype).unsqueeze(0)
    strain = torch.zeros(
        num_systems,
        3,
        3,
        device=positions.device,
        dtype=positions.dtype,
        requires_grad=True,
    )
    deform = eye + strain  # (S, 3, 3)
    atom_sys = (
        torch.zeros(positions.shape[0], dtype=torch.int32, device=positions.device)
        if batch_idx is None
        else batch_idx
    )
    pos_s = torch.einsum("ni,nij->nj", pos_d, deform[atom_sys])
    cell_s = torch.einsum("bij,bjk->bik", cell_d, deform)
    if k_vectors_2d is None:
        energy = energy_fn(pos_s, charges, cell_s).sum()
    else:
        k_s = torch.matmul(
            k_vectors_2d.detach(), torch.linalg.inv(deform).transpose(1, 2)
        )
        energy = energy_fn(pos_s, charges, cell_s, k_s).sum()
    (dE_dstrain,) = torch.autograd.grad(energy, strain, create_graph=True)
    w_torch = (-dE_dstrain).to(virial_value.dtype)
    # Straight-through: value from the kernel, charge gradient from ``w_torch``.
    return virial_value.detach() + (w_torch - w_torch.detach())


def _real_space_energy_outputs(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    alpha: torch.Tensor,
    *,
    batch_idx: torch.Tensor | None,
    idx_j: torch.Tensor | None,
    neighbor_ptr: torch.Tensor | None,
    neighbor_shifts: torch.Tensor | None,
    neighbor_matrix: torch.Tensor | None,
    neighbor_matrix_shifts: torch.Tensor | None,
    mask_value: int,
    want_forces: bool = False,
    want_charge_grad: bool = False,
    want_virial: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Per-atom real-space Ewald energy plus optional deprecated direct outputs.

    Differentiable in ``positions`` / ``charges`` through the explicit chain and in ``cell``
    through :func:`real_space_cell_connect` (literal ``dE/dcell``). When no input
    requires grad, the plain forward-only chain op runs (no derivative state).
    Deprecated direct outputs reuse the same forward launch and remain forward-only.
    """
    num_atoms = positions.shape[0]
    device = positions.device
    use_matrix = neighbor_matrix is not None

    if num_atoms == 0:
        energy = torch.zeros(num_atoms, device=device, dtype=torch.float64)
        forces = (
            torch.zeros(num_atoms, 3, device=device, dtype=positions.dtype)
            if want_forces
            else None
        )
        charge_grads = (
            torch.zeros(num_atoms, device=device, dtype=torch.float64)
            if want_charge_grad
            else None
        )
        virial = (
            torch.zeros(cell.shape[0], 3, 3, device=device, dtype=positions.dtype)
            if want_virial
            else None
        )
        return energy, forces, charge_grads, virial

    if use_matrix:
        if neighbor_matrix_shifts is None:
            raise ValueError(
                "neighbor_matrix_shifts is required when using neighbor_matrix format"
            )
        idx_j_t = torch.zeros(0, dtype=torch.int32, device=device)
        neighbor_ptr_t = torch.zeros(0, dtype=torch.int32, device=device)
        neighbor_shifts_t = torch.zeros(0, 3, dtype=torch.int32, device=device)
        neighbor_matrix_t = neighbor_matrix.to(torch.int32)
        neighbor_matrix_shifts_t = neighbor_matrix_shifts.to(torch.int32)
    else:
        if idx_j is None:
            raise ValueError("neighbor_ptr is required when using neighbor_list format")
        if neighbor_shifts is None:
            raise ValueError(
                "neighbor_shifts is required when using neighbor_list format"
            )
        idx_j_t = idx_j.to(torch.int32)
        neighbor_ptr_t = neighbor_ptr.to(torch.int32)
        neighbor_shifts_t = neighbor_shifts.to(torch.int32)
        neighbor_matrix_t = torch.zeros(num_atoms, 0, dtype=torch.int32, device=device)
        neighbor_matrix_shifts_t = torch.zeros(
            num_atoms, 0, 3, dtype=torch.int32, device=device
        )

    # Forward-precompute gating: pick the fused forward specialization by the
    # requires-grad set so energy + the dE/dR / dE/dq caches come from ONE launch and
    # the first backward is a cheap scale. The cell first-order grad is owned by the
    # Torch ``_RealCellGrad`` connector (literal dE/dcell), so the chain caches only
    # the position / charge first-order state.
    need_pos = bool(positions.requires_grad or want_forces)
    need_charge = bool(charges.requires_grad or want_charge_grad)
    need_cell = bool(cell.requires_grad)
    ensure_electrostatics_ops_registered()
    if batch_idx is None:
        energy, dEdR, dEdq, dedcell, direct_virial = (
            torch.ops.nvalchemiops.ewald_real_energy_single(
                positions,
                charges,
                cell,
                alpha,
                idx_j_t,
                neighbor_ptr_t,
                neighbor_shifts_t,
                neighbor_matrix_t,
                neighbor_matrix_shifts_t,
                int(mask_value),
                use_matrix,
                need_pos,
                need_charge,
                need_cell,
                want_virial,
            )
        )
    else:
        energy, dEdR, dEdq, dedcell, direct_virial = (
            torch.ops.nvalchemiops.ewald_real_energy_batch(
                positions,
                charges,
                cell,
                alpha,
                batch_idx.to(torch.int32),
                idx_j_t,
                neighbor_ptr_t,
                neighbor_shifts_t,
                neighbor_matrix_t,
                neighbor_matrix_shifts_t,
                int(mask_value),
                use_matrix,
                need_pos,
                need_charge,
                need_cell,
                want_virial,
            )
        )

    forces = -dEdR.to(positions.dtype) if want_forces else None
    charge_grads = dEdq if want_charge_grad else None
    virial = direct_virial if want_virial else None

    # Connect ``cell`` (literal dE/dcell) when it carries grad. The connector is a
    # value-zero straight-through term, so the energy value is unchanged. For the
    # matrix layout ``dedcell`` is the forward-fused per-atom cache (first-order
    # backward is a pure scatter, no kernel, no edge list): the (potentially huge)
    # edge-list build is DEFERRED -- the raw neighbor matrix is threaded through and
    # flattened to edges only inside the rare double-backward branch. For CSR the
    # cheap edges are built here (the edge-kernel path needs them) and the matrix is
    # passed empty.
    if need_cell:
        empty_edges = torch.zeros(0, dtype=torch.long, device=device)
        if use_matrix:
            edge_i = edge_j = empty_edges
            unit_shifts = torch.zeros(0, 3, dtype=torch.int32, device=device)
            nm_for_grad = neighbor_matrix_t
            nms_for_grad = neighbor_matrix_shifts_t
        else:
            # CSR -> edges: edge_i is the row owner per edge.
            edge_i = torch.repeat_interleave(
                torch.arange(num_atoms, device=device),
                neighbor_ptr_t.to(torch.long).diff(),
            )
            edge_j = idx_j_t.to(torch.long)
            unit_shifts = neighbor_shifts_t
            # Empty matrix -> the connector backward uses the CSR edges directly.
            nm_for_grad = torch.zeros(num_atoms, 0, dtype=torch.int32, device=device)
            nms_for_grad = torch.zeros(
                num_atoms, 0, 3, dtype=torch.int32, device=device
            )
        energy = real_space_cell_connect(
            energy,
            positions,
            charges,
            cell,
            alpha,
            edge_i,
            edge_j,
            unit_shifts,
            batch_idx,
            dedcell.detach(),
            nm_for_grad,
            nms_for_grad,
            int(mask_value),
        )
    return energy, forces, charge_grads, virial


def _real_space_energy(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    alpha: torch.Tensor,
    *,
    batch_idx: torch.Tensor | None,
    idx_j: torch.Tensor | None,
    neighbor_ptr: torch.Tensor | None,
    neighbor_shifts: torch.Tensor | None,
    neighbor_matrix: torch.Tensor | None,
    neighbor_matrix_shifts: torch.Tensor | None,
    mask_value: int,
) -> torch.Tensor:
    """Per-atom real-space Ewald energy, connected to autograd via the explicit chain."""
    energy, _, _, _ = _real_space_energy_outputs(
        positions,
        charges,
        cell,
        alpha,
        batch_idx=batch_idx,
        idx_j=idx_j,
        neighbor_ptr=neighbor_ptr,
        neighbor_shifts=neighbor_shifts,
        neighbor_matrix=neighbor_matrix,
        neighbor_matrix_shifts=neighbor_matrix_shifts,
        mask_value=mask_value,
    )
    return energy


def _apply_reciprocal_corrections(
    e_ksum: torch.Tensor,
    charges: torch.Tensor,
    volume: torch.Tensor,
    alpha: torch.Tensor,
    batch_idx: torch.Tensor | None,
) -> torch.Tensor:
    """Apply Ewald reciprocal self-energy + background corrections.

    ``volume`` and ``total_charge`` carry gradients back to ``cell`` and
    ``charges``; ``alpha`` is setup-only and is detached from public autograd.
    """
    alpha = alpha.detach()
    if batch_idx is None:
        total_charge = charges.sum().reshape(1)
        return ewald_energy_corrections(e_ksum, charges, volume, alpha, total_charge)
    total_charges = torch.zeros(
        volume.shape[0],
        dtype=charges.dtype,
        device=charges.device,
    )
    total_charges = total_charges.index_add(0, batch_idx.to(torch.long), charges)
    return ewald_energy_corrections_batch(
        e_ksum,
        charges,
        batch_idx.to(torch.int32),
        volume,
        alpha,
        total_charges,
    )


def _reciprocal_space_energy(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    k_vectors: torch.Tensor,
    alpha: torch.Tensor,
    *,
    batch_idx: torch.Tensor | None,
    max_atoms_per_system: int | None = None,
) -> torch.Tensor:
    """Per-atom reciprocal-space Ewald energy, connected to autograd via the chain.

    Energy = k-sum (explicit chain, differentiable in positions / charges and,
    for internally generated reciprocal geometry, cell) minus the Torch-native
    self + background corrections. Public k-vector leaf gradients are outside
    the electrostatics contract.
    """
    num_atoms = positions.shape[0]
    device = positions.device

    if num_atoms == 0:
        return torch.zeros(num_atoms, device=device, dtype=torch.float64)

    # Forward-precompute gating: the fused recip forward emits energy + the
    # detached dE/dR / dE/dq caches in one atom-major pass when positions / charges
    # require grad, so the first backward scales them instead of re-running the
    # atom-major ``compute`` kernel (the k/V cell grads stay on the cheap k-major
    # recompute). ``cell`` flows through ``k_vectors(cell)`` / ``volume`` as before.
    need_pos = bool(positions.requires_grad)
    need_charge = bool(charges.requires_grad)
    # The recip chain owns the cell first order via grad_kvectors / grad_volume (the
    # k-major ``kspace`` recompute). Public k-vector leaf gradients are not part of
    # the contract; the cell path is only valid when k-vectors were generated from
    # this cell inside the full Ewald call.
    need_cell = bool(cell.requires_grad)
    ensure_electrostatics_ops_registered()
    if batch_idx is None:
        num_systems = 1
        num_k = k_vectors.shape[-2]
        k_vectors_2d = k_vectors.reshape(1, num_k, 3)
        volume = torch.abs(
            torch.det(cell.reshape(3, 3) if cell.dim() == 2 else cell[0])
        ).reshape(1)
        e_ksum, _, _, _ = torch.ops.nvalchemiops.ewald_recip_energy_single(
            positions,
            charges,
            cell,
            k_vectors_2d,
            volume.to(torch.float64),
            alpha,
            need_pos,
            need_charge,
            need_cell,
        )
    else:
        num_systems = cell.shape[0]
        num_k = k_vectors.shape[-2]
        k_vectors_2d = (
            k_vectors
            if k_vectors.dim() == 3
            else k_vectors.reshape(1, num_k, 3).expand(num_systems, num_k, 3)
        )
        volume = torch.abs(torch.linalg.det(cell)).to(torch.float64)
        atom_start, atom_end = _atom_ranges(batch_idx, num_systems)
        max_atoms_bound = 0
        if max_atoms_per_system is not None:
            max_atoms_bound = int(max_atoms_per_system)
            if positions.shape[0] == 0:
                if max_atoms_bound < 0:
                    raise ValueError("max_atoms_per_system must be non-negative")
            elif max_atoms_bound <= 0:
                raise ValueError(
                    "max_atoms_per_system must be positive for non-empty batches"
                )
            elif max_atoms_bound > positions.shape[0]:
                raise ValueError(
                    "max_atoms_per_system cannot exceed the total number of atoms"
                )
        e_ksum, _, _, _ = torch.ops.nvalchemiops.ewald_recip_energy_batch(
            positions,
            charges,
            cell,
            k_vectors_2d,
            volume,
            alpha,
            batch_idx.to(torch.int32),
            atom_start,
            atom_end,
            need_pos,
            need_charge,
            need_cell,
            max_atoms_bound,
        )

    return _apply_reciprocal_corrections(e_ksum, charges, volume, alpha, batch_idx)


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
    mask_value: int | None = None,
    batch_idx: torch.Tensor | None = None,
    compute_forces: bool = False,
    compute_charge_gradients: bool = False,
    compute_virial: bool = False,
    hybrid_forces: bool = False,
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
    mask_value : int, optional
        Value indicating invalid entries in neighbor_matrix. Defaults to N.
    batch_idx : torch.Tensor, shape (N,), optional
        System index for each atom. When provided, atoms must be grouped by
        system: ``batch_idx`` must be contiguous, nondecreasing, and use system
        IDs ``0..B-1``.
    compute_forces : bool, default=False
        Whether to compute explicit component forces. This direct output is kept
        for no-autograd MD/inference use; use energy autograd for differentiable
        training.
    compute_charge_gradients : bool, default=False
        Whether to compute explicit component charge gradients. This direct
        output follows the same no-autograd contract as ``compute_forces``.
    compute_virial : bool, default=False
        Whether to compute the component virial tensor W = -dE/d(epsilon).
        Stress = -virial / volume.
    hybrid_forces : bool, default=False
        When True, positions and cell are detached from the autograd graph and
        charge gradients are attached to the energy via a straight-through
        trick.  Forces and virial are forward-only (not differentiable).
        This is intended for efficient inference with geometry-dependent
        charges ``q(R)``, where explicit forces provide ``dE/dR|_q`` and
        autograd through the energy provides the charge chain-rule term
        ``(dE/dq)(dq/dR)``.

    Returns
    -------
    energies : torch.Tensor, shape (N,)
        Per-atom real-space energy.
    forces : torch.Tensor, shape (N, 3), optional
        Direct component forces (if compute_forces=True).
    charge_gradients : torch.Tensor, shape (N,), optional
        Direct component charge gradients (if compute_charge_gradients=True).
    virial : torch.Tensor, shape (1, 3, 3) or (B, 3, 3), optional
        Virial tensor (if compute_virial=True). Always last in the tuple.

    Note
    ----
    Energies are always float64 for numerical stability during accumulation.
    Forces, virial, and charge gradients match the input dtype (float32 or float64).

    When ``charges`` is a non-leaf tensor that may depend on ``positions``
    (q = q(R)), ordinary first-order losses may use cached partial derivatives
    and let PyTorch apply dE/dq * dq/dR once. Weighted losses and higher-order
    derivatives recompute safe partials or connected gradients as needed to
    avoid double-counting that chain term (issue #115). Hybrid direct-output
    mode uses the same cached fallback connector so weighted q(R) losses can
    recover a valid energy gradient when the forward energy was detached.

    """
    component_deprecated_flags = tuple(
        name
        for name, enabled in (
            ("compute_charge_gradients", compute_charge_gradients),
            ("compute_virial", compute_virial),
            ("hybrid_forces", hybrid_forces),
        )
        if enabled
    )
    if component_deprecated_flags and not torch.compiler.is_compiling():
        warnings.warn(
            _component_direct_output_deprecation_msg(
                "ewald_real_space", component_deprecated_flags
            ),
            DeprecationWarning,
            stacklevel=2,
        )

    if mask_value is None:
        mask_value = positions.shape[0]

    # The factory kernels index ``alpha[isys]``; accept a 0-d scalar alpha.
    if alpha.dim() == 0:
        alpha = alpha.reshape(1)
    alpha = _detach_setup_tensor(alpha)

    if neighbor_list is None and neighbor_matrix is None:
        raise ValueError("Either neighbor_list or neighbor_matrix must be provided")
    if neighbor_list is not None and neighbor_ptr is None:
        raise ValueError("neighbor_ptr is required when using neighbor_list format")

    idx_j = neighbor_list[1] if neighbor_list is not None else None

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

    want_direct = compute_forces or compute_charge_gradients or compute_virial

    if hybrid_forces:
        # Positions/cell detached from the graph; charge gradients attached via the
        # lazy cached-eval trick. Forces/virial forward-only.
        energies, forces, charge_grads, virial = _real_space_energy_outputs(
            positions.detach(),
            charges.detach(),
            cell.detach(),
            alpha.detach(),
            batch_idx=batch_idx,
            idx_j=idx_j,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            mask_value=mask_value,
            want_forces=compute_forces,
            want_charge_grad=True,
            want_virial=compute_virial,
        )
        if charges.requires_grad:

            def _fallback(p, q, c):
                return _real_space_energy(
                    p,
                    q,
                    c,
                    alpha,
                    batch_idx=batch_idx,
                    idx_j=idx_j,
                    neighbor_ptr=neighbor_ptr,
                    neighbor_shifts=neighbor_shifts,
                    neighbor_matrix=neighbor_matrix,
                    neighbor_matrix_shifts=neighbor_matrix_shifts,
                    mask_value=mask_value,
                )

            energies = _InjectCachedEvalGradWithFallback.apply(
                energies,
                positions,
                charges,
                cell,
                None,
                charge_grads.detach(),
                None,
                batch_idx,
                _fallback,
            )
        return _build_result(energies, forces, charge_grads.to(positions.dtype), virial)

    if (
        not want_direct
        and not cell.requires_grad
        and not alpha.requires_grad
        and (positions.requires_grad or charges.requires_grad)
        and not _has_potentially_geometry_dependent_charges(positions, charges)
    ):
        # Ordinary scalar first-derivative evaluations can use detached direct
        # caches. Weighted losses and create_graph=True rebuild the true energy
        # graph lazily in the custom backward below.
        energies, forces, charge_grads, _virial = _real_space_energy_outputs(
            positions.detach(),
            charges.detach(),
            cell.detach(),
            alpha.detach(),
            batch_idx=batch_idx,
            idx_j=idx_j,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            mask_value=mask_value,
            want_forces=positions.requires_grad,
            want_charge_grad=charges.requires_grad,
            want_virial=False,
        )

        def _fallback(p, q, c):
            return _real_space_energy(
                p,
                q,
                c,
                alpha,
                batch_idx=batch_idx,
                idx_j=idx_j,
                neighbor_ptr=neighbor_ptr,
                neighbor_shifts=neighbor_shifts,
                neighbor_matrix=neighbor_matrix,
                neighbor_matrix_shifts=neighbor_matrix_shifts,
                mask_value=mask_value,
            )

        energies = _InjectCachedEvalGradWithFallback.apply(
            energies,
            positions,
            charges,
            cell,
            -forces.detach() if positions.requires_grad else None,
            charge_grads.detach() if charges.requires_grad else None,
            None,
            batch_idx,
            _fallback,
        )
        return energies

    # Differentiable energy plus optional deprecated direct outputs from one chain
    # forward launch. Autograd still propagates only from the energy output.
    energies, forces, charge_grads, virial = _real_space_energy_outputs(
        positions,
        charges,
        cell,
        alpha,
        batch_idx=batch_idx,
        idx_j=idx_j,
        neighbor_ptr=neighbor_ptr,
        neighbor_shifts=neighbor_shifts,
        neighbor_matrix=neighbor_matrix,
        neighbor_matrix_shifts=neighbor_matrix_shifts,
        mask_value=mask_value,
        want_forces=compute_forces,
        want_charge_grad=compute_charge_gradients or charges.requires_grad,
        want_virial=compute_virial,
    )

    if not want_direct:
        return energies

    # The deprecated direct virial is differentiable w.r.t. charges; re-attach that
    # gradient (value stays the kernel output) via the strain virial of the
    # autograd-connected real-space energy.
    if compute_virial and charges.requires_grad:

        def _rs_energy_fn(p, q, c):
            return _real_space_energy(
                p,
                q,
                c,
                alpha,
                batch_idx=batch_idx,
                idx_j=idx_j,
                neighbor_ptr=neighbor_ptr,
                neighbor_shifts=neighbor_shifts,
                neighbor_matrix=neighbor_matrix,
                neighbor_matrix_shifts=neighbor_matrix_shifts,
                mask_value=mask_value,
            )

        virial = _attach_virial_charge_grad(
            virial, charges, _rs_energy_fn, positions, cell, batch_idx
        )

    charge_grads_out = (
        charge_grads.to(positions.dtype) if charge_grads is not None else None
    )
    if (
        not cell.requires_grad
        and (positions.requires_grad or charges.requires_grad)
        and (forces is not None or charge_grads is not None)
    ):
        energies = _InjectCachedEvalGrad.apply(
            energies,
            positions,
            charges,
            cell,
            -forces.detach()
            if positions.requires_grad and forces is not None
            else None,
            charge_grads.detach()
            if charges.requires_grad and charge_grads is not None
            else None,
            None,
            batch_idx,
        )
    return _build_result(energies, forces, charge_grads_out, virial)


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
    hybrid_forces: bool = False,
    *,
    max_atoms_per_system: int | None = None,
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
        System index for each atom. When provided, atoms must be grouped by
        system: ``batch_idx`` must be contiguous, nondecreasing, and use system
        IDs ``0..B-1``.
    compute_forces : bool, default=False
        Whether to compute explicit component forces. This direct output is kept
        for no-autograd MD/inference use; use energy autograd for differentiable
        training.
    compute_charge_gradients : bool, default=False
        Whether to compute explicit component charge gradients. This direct
        output follows the same no-autograd contract as ``compute_forces``.
    compute_virial : bool, default=False
        Whether to compute the component virial tensor W = -dE/d(epsilon).
        Stress = -virial / volume.
    hybrid_forces : bool, default=False
        When True, positions and cell are detached from the autograd graph and
        charge gradients are attached to the energy via a straight-through
        trick.  Forces and virial are forward-only (not differentiable).
        See :func:`ewald_real_space` for details.
    max_atoms_per_system : int, optional, keyword-only
        Maximum number of atoms in any single system when ``batch_idx`` is
        provided. Passing this host-known upper bound avoids CUDA host
        synchronization from launch-size inference in the reciprocal kernel.
        Overestimates are safe but may launch extra blocks. When omitted, the
        bound is inferred from ``atom_start`` / ``atom_end`` and may
        synchronize on CUDA.

    Returns
    -------
    energies : torch.Tensor, shape (N,)
        Per-atom reciprocal-space energy.
    forces : torch.Tensor, shape (N, 3), optional
        Direct component forces (if compute_forces=True).
    charge_gradients : torch.Tensor, shape (N,), optional
        Direct component charge gradients (if compute_charge_gradients=True).
    virial : torch.Tensor, shape (1, 3, 3) or (B, 3, 3), optional
        Virial tensor (if compute_virial=True). Always last in the tuple.

    Note
    ----
    Energies are always float64 for numerical stability during accumulation.
    Forces, virial, and charge gradients match the input dtype (float32 or float64).
    ``k_vectors`` are setup metadata. Caller-supplied vectors are treated as
    static values that correspond to the current ``cell``.

    When ``charges`` is a non-leaf tensor that may depend on ``positions``
    (q = q(R)), ordinary first-order losses may use cached partial derivatives
    and let PyTorch apply dE/dq * dq/dR once. Weighted losses and higher-order
    derivatives recompute safe partials or connected gradients as needed to
    avoid double-counting that chain term (issue #115).
    """
    return _ewald_reciprocal_space(
        positions=positions,
        charges=charges,
        cell=cell,
        k_vectors=k_vectors,
        alpha=alpha,
        batch_idx=batch_idx,
        compute_forces=compute_forces,
        compute_charge_gradients=compute_charge_gradients,
        compute_virial=compute_virial,
        hybrid_forces=hybrid_forces,
        allow_cell_grad_with_k_vectors=False,
        max_atoms_per_system=max_atoms_per_system,
    )


def _ewald_reciprocal_space(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    k_vectors: torch.Tensor,
    alpha: torch.Tensor,
    batch_idx: torch.Tensor | None = None,
    compute_forces: bool = False,
    compute_charge_gradients: bool = False,
    compute_virial: bool = False,
    hybrid_forces: bool = False,
    allow_cell_grad_with_k_vectors: bool = False,
    max_atoms_per_system: int | None = None,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Private reciprocal-space implementation with an internal cell-gradient path."""
    component_deprecated_flags = tuple(
        name
        for name, enabled in (
            ("compute_charge_gradients", compute_charge_gradients),
            ("compute_virial", compute_virial),
            ("hybrid_forces", hybrid_forces),
        )
        if enabled
    )
    if component_deprecated_flags and not torch.compiler.is_compiling():
        warnings.warn(
            _component_direct_output_deprecation_msg(
                "ewald_reciprocal_space", component_deprecated_flags
            ),
            DeprecationWarning,
            stacklevel=2,
        )

    is_batch = batch_idx is not None

    # The factory kernels index ``alpha[isys]``; accept a 0-d scalar alpha.
    if alpha.dim() == 0:
        alpha = alpha.reshape(1)
    alpha = _detach_setup_tensor(alpha)

    # Normalize k-vectors to a (S, K, 3) tensor for the factory kernels.
    if is_batch:
        num_systems = cell.shape[0] if cell.dim() == 3 else 1
        if k_vectors.dim() == 2:
            k_vectors_2d = k_vectors.unsqueeze(0).expand(num_systems, *k_vectors.shape)
        else:
            k_vectors_2d = k_vectors
    else:
        if k_vectors.dim() == 3:
            k_vectors_2d = k_vectors[:1]
        else:
            k_vectors_2d = k_vectors.unsqueeze(0)
    if not allow_cell_grad_with_k_vectors:
        k_vectors_2d = k_vectors_2d.detach()

    atom_start = atom_end = None
    if is_batch:
        atom_start, atom_end = _atom_ranges(batch_idx, k_vectors_2d.shape[0])

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

    want_direct = compute_forces or compute_charge_gradients or compute_virial

    # No atoms have no reciprocal contribution. Empty k-vector sets still need
    # self/background corrections below.
    num_atoms = positions.shape[0]
    if num_atoms == 0:
        num_systems = k_vectors_2d.shape[0]
        zeros_e = torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)
        zeros_f = torch.zeros(
            num_atoms, 3, device=positions.device, dtype=positions.dtype
        )
        zeros_cg = torch.zeros(
            num_atoms, device=positions.device, dtype=positions.dtype
        )
        zeros_v = torch.zeros(
            num_systems, 3, 3, device=positions.device, dtype=positions.dtype
        )
        return _build_result(zeros_e, zeros_f, zeros_cg, zeros_v)

    if hybrid_forces:
        k_vectors_hybrid = k_vectors_2d.detach()
        e_ksum, forces, charge_grads, virial = reciprocal_space_direct(
            positions.detach(),
            charges.detach(),
            cell.detach(),
            k_vectors_hybrid,
            alpha.detach(),
            batch_idx=batch_idx,
            atom_start=atom_start,
            atom_end=atom_end,
            want_charge_grad=True,
            want_virial=compute_virial,
        )
        volume = torch.abs(torch.linalg.det(cell.detach().to(torch.float64))).reshape(
            k_vectors_hybrid.shape[0]
        )
        energies = _apply_reciprocal_corrections(
            e_ksum,
            charges.detach(),
            volume,
            alpha,
            batch_idx,
        )
        if charges.requires_grad:

            def _fallback(p, q, c):
                return _reciprocal_space_energy(
                    p,
                    q,
                    c,
                    k_vectors_hybrid,
                    alpha,
                    batch_idx=batch_idx,
                    max_atoms_per_system=max_atoms_per_system,
                )

            energies = _InjectCachedEvalGradWithFallback.apply(
                energies,
                positions,
                charges,
                cell,
                None,
                charge_grads.detach(),
                None,
                batch_idx,
                _fallback,
            )
        return _build_result(energies, forces, charge_grads.to(positions.dtype), virial)

    differentiable_inputs = (
        positions.requires_grad or charges.requires_grad or cell.requires_grad
    )
    if want_direct and not differentiable_inputs:
        e_ksum, forces, charge_grads, virial = reciprocal_space_direct(
            positions,
            charges,
            cell,
            k_vectors_2d,
            alpha,
            batch_idx=batch_idx,
            atom_start=atom_start,
            atom_end=atom_end,
            want_charge_grad=compute_charge_gradients,
            want_virial=compute_virial,
        )
        volume = torch.abs(torch.linalg.det(cell.to(torch.float64))).reshape(
            k_vectors_2d.shape[0]
        )
        energies = _apply_reciprocal_corrections(
            e_ksum,
            charges,
            volume,
            alpha,
            batch_idx,
        )
        charge_grads_out = (
            charge_grads.to(positions.dtype) if charge_grads is not None else None
        )
        return _build_result(energies, forces, charge_grads_out, virial)

    energies = _reciprocal_space_energy(
        positions,
        charges,
        cell,
        k_vectors_2d,
        alpha,
        batch_idx=batch_idx,
        max_atoms_per_system=max_atoms_per_system,
    )

    if not want_direct:
        return energies

    _e_ksum, forces, charge_grads, virial = reciprocal_space_direct(
        positions,
        charges,
        cell,
        k_vectors_2d,
        alpha,
        batch_idx=batch_idx,
        atom_start=atom_start,
        atom_end=atom_end,
        want_charge_grad=compute_charge_gradients or charges.requires_grad,
        want_virial=compute_virial,
    )

    # Re-attach the direct virial's charge gradient (value stays the kernel
    # output) via the strain virial of the reciprocal energy. The strain deforms
    # the k-vectors too (``_attach_virial_charge_grad`` with ``k_vectors_2d``), so
    # the charge gradient matches the kernel virial's ``k_factor`` term.
    if compute_virial and charges.requires_grad:

        def _rec_energy_fn(p, q, c, k):
            return _reciprocal_space_energy(
                p,
                q,
                c,
                k,
                alpha,
                batch_idx=batch_idx,
                max_atoms_per_system=max_atoms_per_system,
            )

        virial = _attach_virial_charge_grad(
            virial,
            charges,
            _rec_energy_fn,
            positions,
            cell,
            batch_idx,
            k_vectors_2d=k_vectors_2d,
        )

    if (
        not cell.requires_grad
        and (positions.requires_grad or charges.requires_grad)
        and (forces is not None or charge_grads is not None)
    ):
        energies = _InjectCachedEvalGrad.apply(
            energies,
            positions,
            charges,
            cell,
            -forces.detach()
            if positions.requires_grad and forces is not None
            else None,
            charge_grads.detach()
            if charges.requires_grad and charge_grads is not None
            else None,
            None,
            batch_idx,
        )

    charge_grads_out = (
        charge_grads.to(positions.dtype) if charge_grads is not None else None
    )
    return _build_result(energies, forces, charge_grads_out, virial)


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
    compute_charge_gradients: bool = False,
    compute_virial: bool = False,
    accuracy: float = 1e-6,
    hybrid_forces: bool = False,
    pbc: torch.Tensor | None = None,
    slab_correction: bool = False,
    *,
    miller_bounds: tuple[int, int, int] | torch.Tensor | None = None,
    max_atoms_per_system: int | None = None,
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
    miller_bounds : tuple[int, int, int] or torch.Tensor, optional, keyword-only
        Precomputed Miller-index half-bounds used when ``k_vectors`` is not
        supplied. Passing Python integer bounds avoids deriving range sizes from
        device tensors inside regenerated-k-vector loops.
    max_atoms_per_system : int, optional, keyword-only
        Maximum number of atoms in any single system when ``batch_idx`` is
        provided. See :func:`ewald_reciprocal_space` for the sync-free launch
        contract.
    batch_idx : torch.Tensor, shape (N,), optional
        System index for each atom. When provided, atoms must be grouped by
        system: ``batch_idx`` must be contiguous, nondecreasing, and use system
        IDs ``0..B-1``.
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
        Deprecated direct-output flag. Compute energy and use
        ``torch.autograd.grad`` for differentiable forces.
    compute_charge_gradients : bool, default=False
        Deprecated direct-output flag. Compute energy and use
        ``torch.autograd.grad`` for ``dE/dq_i``.
    compute_virial : bool, default=False
        Deprecated direct-output flag for the virial tensor W = -dE/d(epsilon).
        Stress = -virial / volume.
    accuracy : float, default=1e-6
        Target accuracy for parameter estimation.
    hybrid_forces : bool, default=False
        When True, positions and cell are detached from the autograd graph and
        charge gradients are attached to the energy via a straight-through
        trick.  Forces and virial are forward-only (not differentiable).
        See :func:`ewald_real_space` for details.
    pbc : torch.Tensor, shape (3,) or (B, 3), dtype=bool, optional
        Per-system periodic boundary conditions. Required when
        ``slab_correction=True``. Each row has True for periodic directions
        and False for the non-periodic (slab) direction. A (3,) tensor is
        accepted only for single-system calls; batched calls require explicit
        (B, 3) per-system pbc. This argument controls the slab correction
        geometry; real-space periodic images are determined by the neighbor
        list supplied to the Ewald real-space term.
    slab_correction : bool, default=False
        When True, apply the Yeh-Berkowitz slab correction (with the
        Ballenegger et al. 2009 Eq. 29 non-neutral extension) to the total
        energy and to forces/charge_grads/virial when those are requested.
        Orthorhombic and triclinic slab cells are supported.

    Returns
    -------
    energies : torch.Tensor, shape (N,)
        Per-atom total Ewald energy.
    forces : torch.Tensor, shape (N, 3), optional
        Deprecated direct forces (if compute_forces=True).
    charge_gradients : torch.Tensor, shape (N,), optional
        Deprecated direct charge gradients (if compute_charge_gradients=True).
    virial : torch.Tensor, shape (1, 3, 3) or (B, 3, 3), optional
        Virial tensor (if compute_virial=True). Always last in the tuple.

    Note
    ----
    Energies are accumulated in float64 for numerical stability. Deprecated
    direct forces, charge gradients, and virials match the input dtype where the
    underlying component path returns typed outputs.

    When ``charges`` is a non-leaf tensor that may depend on ``positions``
    (q = q(R)), ordinary first-order losses may use cached partial derivatives
    and let PyTorch apply dE/dq * dq/dR once. Weighted losses and higher-order
    derivatives recompute safe partials or connected gradients as needed to
    avoid double-counting that chain term (issue #115).

    Enabled output flags are appended in order: energies, [forces],
    [charge_gradients], [virial]. A single output is returned unwrapped;
    multiple outputs are returned as a tuple.

    Examples
    --------
    Automatic parameter estimation (recommended for most cases)::

        >>> energies = ewald_summation(
        ...     positions, charges, cell,
        ...     neighbor_list=nl, neighbor_ptr=nptr, neighbor_shifts=shifts,
        ...     accuracy=1e-6,
        ... )
        >>> total_energy = energies.sum()

    Explicit parameters with forces::

        >>> energies, forces = ewald_summation(
        ...     positions, charges, cell,
        ...     alpha=0.3, k_cutoff=8.0,
        ...     neighbor_list=nl, neighbor_ptr=nptr, neighbor_shifts=shifts,
        ...     compute_forces=True,
        ... )

    Slab correction for two-dimensional periodic systems::

        >>> pbc_slab = torch.tensor([[True, True, False]], device=positions.device)
        >>> energies, forces = ewald_summation(
        ...     positions, charges, cell,
        ...     alpha=0.3, k_cutoff=8.0,
        ...     neighbor_list=nl, neighbor_ptr=nptr, neighbor_shifts=shifts,
        ...     pbc=pbc_slab, slab_correction=True,
        ...     compute_forces=True,
        ... )
    """
    if compute_forces or compute_virial or compute_charge_gradients or hybrid_forces:
        if torch.compiler.is_compiling():
            _compiled_direct_output_deprecation_signal("ewald_summation")
        else:
            warnings.warn(
                _direct_output_deprecation_msg("ewald_summation"),
                DeprecationWarning,
                stacklevel=2,
            )

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

    alpha_tensor = _detach_setup_tensor(
        _prepare_alpha(alpha, num_systems, dtype, device)
    )

    generated_k_vectors = k_vectors is None
    if k_vectors is None:
        k_vectors = generate_k_vectors_ewald_summation(
            cell, k_cutoff, miller_bounds=miller_bounds
        )

    if mask_value is None:
        mask_value = num_atoms

    def _compute_components():
        # Compute real-space
        real = ewald_real_space(
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
            compute_charge_gradients=compute_charge_gradients,
            compute_virial=compute_virial,
            hybrid_forces=hybrid_forces,
        )

        # Compute reciprocal-space
        reciprocal = _ewald_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            k_vectors=k_vectors,
            alpha=alpha_tensor,
            batch_idx=batch_idx,
            compute_forces=compute_forces,
            compute_charge_gradients=compute_charge_gradients,
            compute_virial=compute_virial,
            hybrid_forces=hybrid_forces,
            allow_cell_grad_with_k_vectors=generated_k_vectors,
            max_atoms_per_system=max_atoms_per_system,
        )
        return real, reciprocal

    if torch.compiler.is_compiling():
        rs, rec = _compute_components()
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"The component direct-output flag\(s\).*",
                category=DeprecationWarning,
            )
            rs, rec = _compute_components()

    # Optional slab correction: returns a same-shape tuple as rs/rec,
    # so it composes uniformly with the named-field combination below.
    slab_result: torch.Tensor | tuple[torch.Tensor, ...] | None = None
    if slab_correction:
        if hybrid_forces:
            slab_out = _compute_slab_correction(
                positions.detach(),
                charges.detach(),
                cell.detach(),
                pbc,
                batch_idx=batch_idx,
                compute_forces=compute_forces,
                compute_charge_gradients=True,
                compute_virial=compute_virial,
            )
            slab_energies, slab_forces, slab_charge_grads, slab_virial = (
                _unpack_electrostatic_outputs(
                    slab_out,
                    compute_forces,
                    compute_charge_gradients=True,
                    compute_virial=compute_virial,
                )
            )

            if charges.requires_grad:
                slab_energies = _compute_slab_correction(
                    positions,
                    charges,
                    cell,
                    pbc,
                    batch_idx=batch_idx,
                    compute_forces=False,
                    compute_charge_gradients=False,
                    compute_virial=False,
                )
                slab_energies = _InjectChargeGrad.apply(
                    slab_energies, charges, slab_charge_grads, batch_idx
                )

            slab_result = _build_electrostatic_result(
                slab_energies,
                slab_forces,
                slab_charge_grads,
                slab_virial,
                compute_forces,
                compute_charge_gradients,
                compute_virial,
            )
        else:
            slab_result = _compute_slab_correction(
                positions,
                charges,
                cell,
                pbc,
                batch_idx=batch_idx,
                compute_forces=compute_forces,
                compute_charge_gradients=compute_charge_gradients,
                compute_virial=compute_virial,
            )

    return _combine_electrostatic_outputs(
        rs,
        rec,
        slab_result,
        compute_forces,
        compute_charge_gradients,
        compute_virial,
    )
