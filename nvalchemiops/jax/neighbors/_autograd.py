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
is implemented entirely in pure JAX.

The Warp kernel determines only the *topology* — which pairs are neighbours,
with their integer atom indices and PBC shifts.  The differentiable geometry is
then **reconstructed in pure JAX** from that topology
(:func:`_reconstruct_pair_geometry`): ``r = positions[j] - positions[i] +
shifts @ cell``.  Because this is a plain JAX expression, gradients of *all
orders* (forces, Hessian / HVP) are handled natively and correctly for any
downstream loss — including losses nonlinear in distance.  (An earlier
``jax.custom_vjp`` straight-through that returned the detached kernel distances
and re-attached only a first-order gradient produced a wrong HVP whenever the
loss was nonlinear in distance; reconstructing live removes that failure mode.)

JIT compatibility constraint
----------------------------
JAX requires concrete bool indices, so this module keeps per-pair tensors at
the full ``(K, M)`` matrix shape (not the compact ``(P,)`` shape used on the
torch side).  Inactive slots are zeroed out via an ``active_mask`` instead of
being filtered with boolean indexing.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp

__all__ = [
    "_DISTANCE_DERIVATIVE_EPSILON",
    "_NeighborForwardOutput",
    "_reconstruct_pair_geometry",
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


def _reconstruct_pair_geometry(
    positions: jax.Array,
    cell: jax.Array,
    i_idx: jax.Array,
    j_idx: jax.Array,
    shifts_int: jax.Array,
    active_mask: jax.Array,
    batch_idx: jax.Array | None,
    has_batch_idx: bool,
) -> tuple[jax.Array, jax.Array]:
    """Reconstruct per-pair ``(distances, vectors)`` as a *live*, differentiable
    function of ``positions`` / ``cell`` from the kernel-emitted topology.

    The Warp kernels are ``enable_backward=False`` (opaque to JAX autodiff), so the
    geometry is recomputed here in pure JAX:
    ``r = positions[j] - positions[i] + shifts @ cell``.  This reproduces the
    kernel's exact displacement (identical formula), so the returned values match
    its ``neighbor_distances`` / ``neighbor_vectors`` to floating-point round-off
    while being differentiable to *all orders* — JAX handles gradients, HVP, and the
    Hessian natively, correctly even for losses nonlinear in distance.

    Safe-norm handling (correct under ``jax.grad(jax.grad(...))``): ``jnp.linalg.norm``
    has a ``0 / 0`` (NaN) derivative at ``r == 0``, which occurs on inactive slots
    *and* on the degenerate-but-kernel-emitted active pair of two distinct atoms at
    identical coordinates.  Any such zero-vector slot is replaced by a unit
    placeholder before ``norm`` (so the gradient is finite to all orders) and then
    masked to ``0`` in ``distances`` — a coincident active pair contributes a finite
    ``0`` gradient, matching the torch reference.  Real separations keep the exact
    ``linalg.norm`` (the same op torch uses), so machine-precision HVP parity holds.
    """
    eps = _DISTANCE_DERIVATIVE_EPSILON.get(positions.dtype, 1e-6)
    N = positions.shape[0]
    shifts_pt = shifts_int.astype(positions.dtype)  # (K, M, 3)

    # Clip sentinel (N) j-indices to a safe row; inactive slots are masked below.
    j_safe = jnp.clip(j_idx, 0, N - 1)

    if has_batch_idx and batch_idx is not None:
        # Per-pair cell = cell[batch_idx[i_idx]] — index gather chain.
        batch_idx_safe = batch_idx.astype(jnp.int32)
        per_atom_system = batch_idx_safe[i_idx]  # (K, M)
        cell_per_slot = cell[per_atom_system]  # (K, M, 3, 3)
        shift_displacement = jnp.einsum("kma,kmab->kmb", shifts_pt, cell_per_slot)
    else:
        c = jnp.squeeze(cell, 0) if cell.ndim == 3 else cell
        shift_displacement = shifts_pt @ c  # (K, M, 3)

    r = positions[j_safe] - positions[i_idx] + shift_displacement
    # Replace zero-vector slots with a unit placeholder so ``norm`` has a finite
    # gradient; the placeholder distance is discarded by ``keep`` below.
    is_zero = jnp.sum(r * r, axis=-1) <= eps * eps  # (K, M)
    r_for_norm = jnp.where(is_zero[..., None], jnp.ones_like(r), r)
    d = jnp.linalg.norm(r_for_norm, axis=-1)

    keep = active_mask & jnp.logical_not(is_zero)
    distances = jnp.where(keep, d, 0.0)
    vectors = jnp.where(active_mask[..., None], r, 0.0)
    return distances, vectors


def _route_pair_outputs(
    positions: jax.Array,
    cell: jax.Array | None,
    forward_fn: Callable[..., _NeighborForwardOutput],
    forward_kwargs: dict,
) -> tuple[jax.Array, ...]:
    """Run a pair-output forward closure and return differentiable per-pair
    distances / vectors.

    The Warp launcher determines only the *topology* (which pairs are neighbours,
    with integer atom indices and PBC shifts).  The differentiable geometry is
    reconstructed in pure JAX from that topology via
    :func:`_reconstruct_pair_geometry`, so JAX's native autodiff covers gradients
    to all orders.  Unlike the torch side we don't sniff ``requires_grad``: JAX
    traces lazily, so the backward is built only when someone calls ``jax.grad``.
    """
    out: _NeighborForwardOutput = forward_fn(positions, cell, **forward_kwargs)
    cell_for_residual = (
        jnp.zeros((1, 3, 3), dtype=positions.dtype) if cell is None else cell
    )
    has_batch_idx = out.batch_idx is not None

    distances_diff, vectors_diff = _reconstruct_pair_geometry(
        positions,
        cell_for_residual,
        out.i_idx,
        out.j_idx,
        out.shifts,
        out.active_mask,
        out.batch_idx if has_batch_idx else None,
        has_batch_idx,
    )
    return (distances_diff, vectors_diff, *out.extra_outputs)
