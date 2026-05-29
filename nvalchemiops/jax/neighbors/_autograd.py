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

"""JAX autograd wiring for per-pair distances and vectors.

Mirrors :mod:`nvalchemiops.torch.neighbors._autograd`.  The Warp launchers
do not propagate gradients across the JAX boundary (``enable_backward=False``
on every ``jax_kernel`` / ``jax_callable``), so the differentiable behaviour
is implemented entirely in pure JAX via :func:`jax.custom_vjp`.

The autograd primitive is a *straight-through* function: in the forward it
returns the warp-produced ``distances`` and ``vectors`` unchanged; on the
backward it reconstructs the per-pair displacement ``r`` from ``positions``
and ``cell`` using the integer indices and shifts the warp kernel emitted,
then scatters the analytical gradient.

JIT compatibility constraint
----------------------------
JAX requires concrete bool indices, so this module keeps per-pair tensors at
the full ``(K, M)`` matrix shape (not the compact ``(P,)`` shape used on the
torch side).  Inactive slots are zeroed out via an ``active_mask`` instead of
being filtered with boolean indexing.  Scatter sites for inactive slots are
still written (to safe sentinel atom indices), but with a zero contribution
so the gradient is unaffected.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp

__all__ = [
    "_DISTANCE_DERIVATIVE_EPSILON",
    "_NeighborForwardOutput",
    "_attach_neighbor_pair_grads",
    "_route_pair_outputs",
    "_build_index_residuals",
]

#: Stabilization for ``d_safe = maximum(d, eps)`` in the reconstruction.
_DISTANCE_DERIVATIVE_EPSILON: dict[jnp.dtype, float] = {
    jnp.float16: 1e-3,
    jnp.float32: 1e-6,
    jnp.float64: 1e-12,
}


class _NeighborForwardOutput(NamedTuple):
    """Uniform forward result returned by each family's closure.

    All per-slot tensors are kept at full matrix shape ``(K, M, ...)``.  The
    ``active_mask`` zeroes the inactive slots in the backward.
    """

    distances: jax.Array
    """Per-pair scalar distances ``(K, M)`` matrix or ``(P,)`` COO."""

    vectors: jax.Array
    """Per-pair displacement vectors ``(K, M, 3)`` or ``(P, 3)``."""

    extra_outputs: tuple[jax.Array, ...]
    """Non-differentiable user-visible tensors the wrapper returns alongside
    ``distances`` / ``vectors`` (neighbor matrix / list, counts / CSR ptr,
    integer shifts).
    """

    i_idx: jax.Array
    """``(K, M)`` int32: atom-i per slot.  Inactive-slot rows can hold any
    safe value since ``active_mask`` zeros their contribution.
    """

    j_idx: jax.Array
    """``(K, M)`` int32: atom-j per slot.  For inactive slots, equals the
    sentinel ``N`` produced by the kernel — clipped to a safe in-range value
    in the backward.
    """

    shifts: jax.Array
    """``(K, M, 3)`` int32: PBC shift per slot."""

    batch_idx: jax.Array | None
    """``(N,)`` int32 mapping atom → system for batched cells; ``None`` for
    single-system.  Indexed by ``i_idx`` in the backward to route per-system
    cell gradients.
    """

    active_mask: jax.Array
    """``(K, M)`` bool: ``True`` for active emitted slots."""

    matrix_shape: tuple[int, int]
    """``(K, M)`` — included for API parity with the torch version."""


def _build_index_residuals(
    neighbor_matrix: jax.Array,
    num_neighbors: jax.Array,
    shifts: jax.Array,
    target_indices: jax.Array | None = None,
    batch_idx: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array | None, jax.Array]:
    """Build the integer-index residuals the backward consumes.

    Returns ``(i_idx, j_idx, shifts, batch_idx, active_mask)`` all at full
    matrix shape ``(K, M, ...)``.  ``batch_idx`` is returned as-is (shape
    ``(N,)`` or ``None``); the backward gathers it on-demand via ``i_idx``.
    """
    K, M = neighbor_matrix.shape
    col_idx = jnp.arange(M, dtype=jnp.int32)
    active_mask = col_idx[None, :] < num_neighbors[:, None]
    if target_indices is not None:
        row_to_atom_i = target_indices.astype(jnp.int32)
    else:
        row_to_atom_i = jnp.arange(K, dtype=jnp.int32)
    i_idx = jnp.broadcast_to(row_to_atom_i[:, None], (K, M))
    j_idx = neighbor_matrix.astype(jnp.int32)
    return i_idx, j_idx, shifts, batch_idx, active_mask


# ---------------------------------------------------------------------------
# custom_vjp primitive.
#
# JAX rule: array-typed args must be in the DIFFERENTIABLE positions (not
# nondiff_argnums).  nondiff_argnums is restricted to non-tracer args
# (function-valued, tuples of Python ints, bools, etc.).
#
# Convention used here:
#   - Differentiable: positions, cell.
#   - Constants-with-zero-gradient (array-typed): distances_primal,
#     vectors_primal, i_idx, j_idx, shifts_int, batch_idx_or_sentinel,
#     active_mask.  Backward returns zero cotangents for these.
#   - nondiff_argnums: matrix_shape, cell_shape, has_batch_idx (all
#     tuples-of-ints or bools).
# ---------------------------------------------------------------------------


@functools.partial(jax.custom_vjp, nondiff_argnums=(9, 10, 11))
def _attach_neighbor_pair_grads(
    positions: jax.Array,
    cell: jax.Array,
    distances_primal: jax.Array,
    vectors_primal: jax.Array,
    i_idx: jax.Array,
    j_idx: jax.Array,
    shifts_int: jax.Array,
    batch_idx_or_sentinel: jax.Array,
    active_mask: jax.Array,
    # nondiff (static) args below:
    matrix_shape: tuple[int, int],
    cell_shape: tuple,
    has_batch_idx: bool,
) -> tuple[jax.Array, jax.Array]:
    """Straight-through forward: returns the warp-produced primals.

    The autograd graph is attached by ``defvjp`` below.
    """
    return distances_primal, vectors_primal


def _attach_neighbor_pair_grads_fwd(
    positions,
    cell,
    distances_primal,
    vectors_primal,
    i_idx,
    j_idx,
    shifts_int,
    batch_idx_or_sentinel,
    active_mask,
    matrix_shape,
    cell_shape,
    has_batch_idx,
):
    residuals = (
        positions,
        cell,
        i_idx,
        j_idx,
        shifts_int,
        batch_idx_or_sentinel,
        active_mask,
    )
    return (distances_primal, vectors_primal), residuals


def _attach_neighbor_pair_grads_bwd(
    matrix_shape,
    cell_shape,
    has_batch_idx,
    residuals,
    cotangents,
):
    (
        positions,
        cell,
        i_idx,
        j_idx,
        shifts_int,
        batch_idx_or_sentinel,
        active_mask,
    ) = residuals
    grad_distances, grad_vectors = cotangents
    eps = _DISTANCE_DERIVATIVE_EPSILON.get(positions.dtype, 1e-6)
    N = positions.shape[0]
    batch_idx = batch_idx_or_sentinel if has_batch_idx else None

    shifts_pt = shifts_int.astype(positions.dtype)  # (K, M, 3)

    # Clip j_idx to a safe in-range value so the gather doesn't fault on
    # sentinel (N) values.  Inactive slots get zeroed via active_mask below,
    # so the actual gather result for those is irrelevant.
    j_safe = jnp.clip(j_idx, 0, N - 1)

    # Reconstruct r per slot.
    if batch_idx is not None:
        # Per-pair cell = cell[batch_idx[i_idx]] — index gather chain.
        # i_idx shape (K, M); batch_idx shape (N,); cell shape (S, 3, 3).
        batch_idx_safe = batch_idx.astype(jnp.int32)
        per_atom_system = batch_idx_safe[i_idx]  # (K, M)
        cell_per_slot = cell[per_atom_system]  # (K, M, 3, 3)
        shift_displacement = jnp.einsum("kma,kmab->kmb", shifts_pt, cell_per_slot)
    else:
        c = jnp.squeeze(cell, 0) if cell.ndim == 3 else cell
        shift_displacement = shifts_pt @ c  # (K, M, 3)

    r_per_slot = positions[j_safe] - positions[i_idx] + shift_displacement
    # Replace inactive slots' ``r`` with a safe non-zero vector before
    # taking ``1 / d_safe``.  Without this, ``r = 0`` on inactive slots
    # makes ``d_safe = eps`` and the *second* derivative of ``1 / d_safe``
    # blows up, producing NaN under ``jax.grad(jax.grad(...))``.  The
    # ``active_mask`` zeros the resulting contribution below, so the
    # forward output is unchanged.
    r_per_slot_safe = jnp.where(
        active_mask[..., None], r_per_slot, jnp.ones_like(r_per_slot)
    )
    d_safe = jnp.maximum(jnp.linalg.norm(r_per_slot_safe, axis=-1), eps)
    u_per_slot = r_per_slot_safe / d_safe[..., None]

    # Per-slot contribution, masked to zero on inactive slots.
    contrib = grad_distances[..., None] * u_per_slot + grad_vectors
    contrib = jnp.where(active_mask[..., None], contrib, 0.0)

    # Scatter to grad_positions.
    i_flat = i_idx.reshape(-1)
    j_flat = j_safe.reshape(-1)
    contrib_flat = contrib.reshape(-1, 3)
    grad_positions = jnp.zeros((N, 3), dtype=positions.dtype)
    grad_positions = grad_positions.at[i_flat].add(-contrib_flat)
    grad_positions = grad_positions.at[j_flat].add(+contrib_flat)

    # Accumulate grad_cell.
    shifts_flat = shifts_pt.reshape(-1, 3)
    if batch_idx is not None:
        # per_pair_outer[p] = shifts_flat[p] outer contrib_flat[p].
        per_pair_outer = jnp.einsum("pa,pb->pab", shifts_flat, contrib_flat)
        batch_idx_safe = batch_idx.astype(jnp.int32)
        per_slot_system = batch_idx_safe[i_idx].reshape(-1)
        grad_cell = jnp.zeros(cell_shape, dtype=positions.dtype)
        grad_cell = grad_cell.at[per_slot_system].add(per_pair_outer)
    else:
        grad_cell_3x3 = jnp.einsum("pa,pb->ab", shifts_flat, contrib_flat)
        grad_cell = grad_cell_3x3.reshape(cell_shape)

    # Return cotangents matching the differentiable inputs of the primal:
    # (positions, cell, distances_primal, vectors_primal, i_idx, j_idx,
    # shifts_int, batch_idx_or_sentinel, active_mask).  Only positions and
    # cell receive non-zero gradient; the rest are constants-from-forward.
    return (
        grad_positions,
        grad_cell,
        jnp.zeros_like(grad_distances),
        jnp.zeros_like(grad_vectors),
        jnp.zeros_like(i_idx),
        jnp.zeros_like(j_idx),
        jnp.zeros_like(shifts_int),
        jnp.zeros_like(batch_idx_or_sentinel),
        jnp.zeros_like(active_mask),
    )


_attach_neighbor_pair_grads.defvjp(
    _attach_neighbor_pair_grads_fwd, _attach_neighbor_pair_grads_bwd
)


def _route_pair_outputs(
    positions: jax.Array,
    cell: jax.Array | None,
    forward_fn: Callable[..., _NeighborForwardOutput],
    forward_kwargs: dict,
) -> tuple[jax.Array, ...]:
    """Route every pair-output call through the autograd primitive.

    Unlike the torch side, we don't sniff ``requires_grad``.  JAX traces
    lazily; the backward is never built unless someone calls ``jax.grad``.
    """
    out: _NeighborForwardOutput = forward_fn(positions, cell, **forward_kwargs)
    if cell is None:
        cell_for_residual = jnp.zeros((1, 3, 3), dtype=positions.dtype)
        cell_shape = (1, 3, 3)
    else:
        cell_for_residual = cell
        cell_shape = tuple(cell.shape)

    has_batch_idx = out.batch_idx is not None
    if has_batch_idx:
        batch_idx_arr = out.batch_idx
    else:
        # Sentinel: zero-length int32 array.  Never actually indexed in bwd.
        batch_idx_arr = jnp.zeros(0, dtype=jnp.int32)

    distances_diff, vectors_diff = _attach_neighbor_pair_grads(
        positions,
        cell_for_residual,
        out.distances,
        out.vectors,
        out.i_idx,
        out.j_idx,
        out.shifts,
        batch_idx_arr,
        out.active_mask,
        out.matrix_shape,
        cell_shape,
        has_batch_idx,
    )
    return (distances_diff, vectors_diff, *out.extra_outputs)
