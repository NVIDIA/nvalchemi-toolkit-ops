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

"""Autograd wiring for per-pair distances and vectors from any neighbor-list
family.

Forward calls a family-specific closure that runs the warp neighbor-list
launcher for the requested per-pair outputs.  The closure returns a
:class:`_NeighborForwardOutput` containing the requested user-visible
``distances`` / ``vectors`` plus the integer indices and shifts needed to
reconstruct ``r = x_j - x_i + S @ cell`` in the backward.

Backward is a single pure-torch pass: gather, subtract, matmul with cell,
norm, clamp, divide, scatter, einsum.  All operations are differentiable, so
``create_graph=True`` works without any second-tier wrapping — torch derives
second-order gradients automatically.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import torch

__all__ = [
    "_DISTANCE_DERIVATIVE_EPSILON",
    "_NeighborForwardOutput",
    "_NeighborDistanceVectorFn",
    "_route_pair_outputs",
    "_flatten_active_pairs",
    "_reconstruct_matrix_geometry",
    "_reconstruct_coo_geometry",
]

#: Stabilization for ``d_safe = d.clamp(min=eps)`` in the reconstruction.
#: Keys are the torch dtypes the differentiable ``positions`` tensor may have.
_DISTANCE_DERIVATIVE_EPSILON: dict[torch.dtype, float] = {
    torch.float16: 1e-3,
    torch.float32: 1e-6,
    torch.float64: 1e-12,
}


class _NeighborForwardOutput(NamedTuple):
    """Uniform forward result returned by each family's closure."""

    distances: torch.Tensor | None
    """Per-pair scalar distances, or ``None`` when not requested."""

    vectors: torch.Tensor | None
    """Per-pair displacement vectors, or ``None`` when not requested."""

    extra_outputs: tuple[torch.Tensor, ...]
    """Non-differentiable user-visible tensors the wrapper returns alongside
    ``distances`` / ``vectors``: typically the neighbor matrix / list, the
    per-atom counts / CSR pointers, and the integer shift tensor.  Order is
    family-specific; the wrapper re-packages them into its public return.
    ``distances`` or ``vectors`` can be ``None`` when the corresponding public
    output was not requested.
    """

    i_idx_flat: torch.Tensor
    """``(P,)`` int32: atom-i index per active pair."""

    j_idx_flat: torch.Tensor
    """``(P,)`` int32: atom-j index per active pair."""

    shifts_flat: torch.Tensor
    """``(P, 3)`` int32: PBC shift per active pair.  Cast to positions dtype
    in the backward for the ``shifts @ cell`` matmul.
    """

    batch_idx_flat: torch.Tensor | None
    """``(P,)`` int32 for batched paths: per-pair system index.  ``None``
    for single-system paths.
    """

    active_mask: torch.Tensor | None
    """``(K, M)`` bool for matrix layout: True at active slots.  ``None``
    for COO layout (grad tensors are already flat ``(P, ...)``).
    Used in backward to gather upstream grad_distances / grad_vectors into
    the same row-major order as ``i_idx_flat`` / ``j_idx_flat``.
    """

    matrix_shape: tuple[int, int] | None
    """``(K, M)`` for matrix layout, ``None`` for COO.  Sentinel that
    selects the grad-flatten path in backward.
    """


def _flatten_active_pairs(
    neighbor_matrix: torch.Tensor,
    num_neighbors: torch.Tensor,
    shifts: torch.Tensor,
    target_indices: torch.Tensor | None = None,
    batch_idx: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor,
]:
    """Flatten the active slots of a matrix-format neighbor list.

    Returns
    -------
    i_idx_flat : (P,) long
    j_idx_flat : (P,) long
    shifts_flat : (P, 3) int
    batch_idx_flat : (P,) long or None
    active_mask : (K, M) bool
    """
    K, M = neighbor_matrix.shape
    col_idx = torch.arange(M, device=neighbor_matrix.device, dtype=torch.int32)
    active_mask = col_idx[None, :] < num_neighbors[:, None]  # (K, M) bool

    if target_indices is not None:
        row_to_atom_i = target_indices.to(torch.int32)
    else:
        row_to_atom_i = torch.arange(
            K, device=neighbor_matrix.device, dtype=torch.int32
        )
    i_idx_2d = row_to_atom_i.unsqueeze(-1).expand(-1, M)

    i_idx_flat = i_idx_2d[active_mask].contiguous()
    j_idx_flat = neighbor_matrix[active_mask].contiguous()
    shifts_flat = shifts[active_mask].contiguous()

    batch_idx_flat: torch.Tensor | None = None
    if batch_idx is not None:
        batch_idx_flat = batch_idx.to(torch.int32)[i_idx_flat]

    return i_idx_flat, j_idx_flat, shifts_flat, batch_idx_flat, active_mask


def _reconstruct_matrix_geometry(
    positions: torch.Tensor,
    cell: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    num_neighbors: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    batch_idx_atom: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct zero-padded matrix geometry from fixed topology tensors."""
    n_rows, max_neighbors = neighbor_matrix.shape
    slots = torch.arange(
        max_neighbors, dtype=num_neighbors.dtype, device=neighbor_matrix.device
    )
    active = slots.unsqueeze(0) < num_neighbors.unsqueeze(1)
    i_idx = torch.arange(n_rows, dtype=torch.int32, device=neighbor_matrix.device)
    i_idx = i_idx.unsqueeze(1).expand_as(neighbor_matrix)
    safe_i = torch.where(active, i_idx, 0).reshape(-1)
    safe_j = torch.where(active, neighbor_matrix, 0).reshape(-1)
    safe_shifts = torch.where(active.unsqueeze(-1), neighbor_matrix_shifts, 0).reshape(
        -1, 3
    )

    shifts = safe_shifts.to(positions.dtype)
    if batch_idx_atom is None:
        cell_3x3 = cell.squeeze(0) if cell.ndim == 3 else cell
        shift_displacement = shifts @ cell_3x3
    else:
        pair_batch_idx = batch_idx_atom[safe_i.to(torch.long)]
        shift_displacement = torch.einsum(
            "pa,pab->pb", shifts, cell[pair_batch_idx.to(torch.long)]
        )
    vectors = positions[safe_j.to(torch.long)] - positions[safe_i.to(torch.long)]
    vectors = vectors + shift_displacement
    raw = vectors.norm(dim=-1)
    eps = _DISTANCE_DERIVATIVE_EPSILON.get(positions.dtype, 1e-6)
    near_zero = raw < eps
    far_vectors = torch.where(
        near_zero.unsqueeze(-1), torch.ones_like(vectors), vectors
    )
    potential = torch.where(
        near_zero,
        vectors.square().sum(dim=-1) / (2.0 * eps) + eps / 2.0,
        far_vectors.norm(dim=-1),
    )
    distances = potential + (raw - potential).detach()
    distances = distances.reshape(n_rows, max_neighbors)
    vectors = vectors.reshape(n_rows, max_neighbors, 3)
    return (
        torch.where(active, distances, torch.zeros_like(distances)),
        torch.where(active.unsqueeze(-1), vectors, torch.zeros_like(vectors)),
    )


def _reconstruct_coo_geometry(
    positions: torch.Tensor,
    cell: torch.Tensor,
    neighbor_list: torch.Tensor,
    neighbor_list_shifts: torch.Tensor,
    batch_idx_atom: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct differentiable compact-COO distances and vectors."""
    i_idx, j_idx = neighbor_list
    vectors = positions[j_idx.long()] - positions[i_idx.long()]
    shifts = neighbor_list_shifts.to(positions.dtype)
    if batch_idx_atom is None:
        cell_3x3 = cell.squeeze(0) if cell.ndim == 3 else cell
        vectors = vectors + shifts @ cell_3x3
    else:
        pair_batch = batch_idx_atom[i_idx.long()]
        vectors = vectors + torch.einsum("pa,pab->pb", shifts, cell[pair_batch.long()])
    raw = vectors.norm(dim=-1)
    eps = _DISTANCE_DERIVATIVE_EPSILON.get(positions.dtype, 1e-6)
    near_zero = raw < eps
    potential = torch.where(
        near_zero,
        vectors.square().sum(dim=-1) / (2.0 * eps) + eps / 2.0,
        torch.where(near_zero.unsqueeze(-1), torch.ones_like(vectors), vectors).norm(
            dim=-1
        ),
    )
    return potential + (raw - potential).detach(), vectors


# Sentinel empty tensors used to stand in for ``None`` in saved_tensors.
def _empty_i32(device: torch.device) -> torch.Tensor:
    return torch.empty(0, dtype=torch.int32, device=device)


def _empty_bool(device: torch.device) -> torch.Tensor:
    return torch.empty(0, dtype=torch.bool, device=device)


class _NeighborDistanceVectorFn(torch.autograd.Function):
    """Autograd over per-pair (d, r) — family- and format-agnostic.

    Forward returns ``(distances, vectors, *extra_outputs)``; caller unpacks.

    Saved tensors: ``positions``, ``cell`` (or 0-d sentinel), ``i_idx``,
    ``j_idx``, ``shifts_int``, ``batch_idx`` (or 0-d sentinel),
    ``active_mask`` (or 0-d sentinel for COO).

    Saved attributes: presence flags + ``matrix_shape`` + ``cell_shape`` +
    extra-output count.

    Backward reconstructs ``r`` from ``positions`` and ``cell``, so the
    backward itself is differentiable: ``create_graph=True`` for second-order
    just works.
    """

    @staticmethod
    def forward(ctx, positions, cell, forward_fn, forward_kwargs):
        out = forward_fn(positions, cell, **forward_kwargs)
        device = positions.device

        cell_or_empty = (
            cell
            if cell is not None
            else torch.empty(0, dtype=positions.dtype, device=device)
        )
        batch_or_empty = (
            out.batch_idx_flat if out.batch_idx_flat is not None else _empty_i32(device)
        )
        mask_or_empty = (
            out.active_mask if out.active_mask is not None else _empty_bool(device)
        )
        ctx.save_for_backward(
            positions,
            cell_or_empty,
            out.i_idx_flat,
            out.j_idx_flat,
            out.shifts_flat,
            batch_or_empty,
            mask_or_empty,
        )
        ctx.has_cell = cell is not None
        ctx.has_batch_idx = out.batch_idx_flat is not None
        ctx.matrix_shape = out.matrix_shape
        ctx.cell_shape = tuple(cell.shape) if cell is not None else None
        ctx.n_extra = len(out.extra_outputs)
        return (out.distances, out.vectors, *out.extra_outputs)

    @staticmethod
    def backward(ctx, grad_distances, grad_vectors, *grad_extra):
        (
            positions,
            cell_or_empty,
            i_idx,
            j_idx,
            shifts_int,
            batch_or_empty,
            mask_or_empty,
        ) = ctx.saved_tensors
        cell = cell_or_empty if ctx.has_cell else None
        batch_idx_flat = batch_or_empty if ctx.has_batch_idx else None
        active_mask = mask_or_empty if ctx.matrix_shape is not None else None

        eps = _DISTANCE_DERIVATIVE_EPSILON.get(positions.dtype, 1e-6)
        shifts_pt = shifts_int.to(positions.dtype)  # (P, 3)

        # Reconstruct r in a torch-differentiable way.
        if cell is not None:
            if batch_idx_flat is not None:
                cell_per_pair = cell[batch_idx_flat]  # (P, 3, 3)
                shift_displacement = torch.einsum(
                    "pa,pab->pb", shifts_pt, cell_per_pair
                )
            else:
                c = cell.squeeze(0) if cell.ndim == 3 else cell
                shift_displacement = shifts_pt @ c  # (P, 3)
        else:
            shift_displacement = torch.zeros_like(shifts_pt)

        r_active = positions[j_idx] - positions[i_idx] + shift_displacement  # (P, 3)
        d_active = r_active.norm(dim=-1)  # (P,)
        d_safe = d_active.clamp(min=eps)  # (P,)
        u_active = r_active / d_safe.unsqueeze(-1)  # (P, 3)

        # Flatten upstream grads to (P,) and (P, 3).
        if ctx.matrix_shape is not None:
            grad_d_flat = (
                grad_distances[active_mask] if grad_distances is not None else None
            )
            grad_r_flat = (
                grad_vectors[active_mask] if grad_vectors is not None else None
            )
        else:
            grad_d_flat = grad_distances if grad_distances is not None else None
            grad_r_flat = grad_vectors if grad_vectors is not None else None

        # Build per-pair contribution.
        contrib = torch.zeros_like(r_active)  # (P, 3)
        if grad_d_flat is not None:
            contrib = contrib + grad_d_flat.unsqueeze(-1) * u_active
        if grad_r_flat is not None:
            contrib = contrib + grad_r_flat

        # Scatter to grad_positions.
        N = positions.shape[0]
        grad_positions = torch.zeros(
            (N, 3), dtype=positions.dtype, device=positions.device
        )
        grad_positions = grad_positions.index_add(0, i_idx, -contrib)
        grad_positions = grad_positions.index_add(0, j_idx, +contrib)

        # Accumulate grad_cell.
        grad_cell: torch.Tensor | None = None
        if cell is not None:
            if batch_idx_flat is not None:
                # Per-system outer product; index_add into (S, 3, 3).
                S = cell.shape[0]
                grad_cell_flat = torch.zeros(
                    (S, 3, 3), dtype=positions.dtype, device=positions.device
                )
                # outer[p, a, b] = contrib[p, b] * shifts[p, a] gives
                #   ∂L/∂cell[a, b] = Σ_p ∂L/∂r[p, b] · shifts[p, a]
                # which matches dr/dcell[a, b] = shifts[a] · e_b.
                per_pair_outer = torch.einsum("pa,pb->pab", shifts_pt, contrib)
                grad_cell_flat = grad_cell_flat.index_add(
                    0, batch_idx_flat, per_pair_outer
                )
                grad_cell = grad_cell_flat.reshape(ctx.cell_shape)
            else:
                # Single-system: outer product summed.
                # dd/dcell[a, b] = u[b] * shifts[a]  per pair
                # dL/dcell[a, b] = Σ_p contrib[p, b] * shifts[p, a]
                grad_cell_3x3 = torch.einsum("pa,pb->ab", shifts_pt, contrib)
                grad_cell = grad_cell_3x3.reshape(ctx.cell_shape)

        # Return gradients in the order matching forward inputs:
        # (positions, cell, forward_fn, forward_kwargs).
        # forward_fn and forward_kwargs are non-tensor → None.
        return grad_positions, grad_cell, None, None


def _route_pair_outputs(
    positions: torch.Tensor,
    cell: torch.Tensor | None,
    forward_fn: Callable[..., _NeighborForwardOutput],
    forward_kwargs: dict,
) -> tuple[torch.Tensor | None, ...]:
    """Route to the autograd Function iff any input requires grad.

    Returns ``(distances, vectors, *extra_outputs)`` in the order the
    family's forward closure emits.  The caller repackages this into its
    public return tuple; unrequested geometry entries are ``None``.
    """
    needs_grad = positions.requires_grad or (cell is not None and cell.requires_grad)
    if needs_grad:
        return _NeighborDistanceVectorFn.apply(
            positions,
            cell,
            forward_fn,
            forward_kwargs,
        )
    out = forward_fn(positions, cell, **forward_kwargs)
    return (out.distances, out.vectors, *out.extra_outputs)
