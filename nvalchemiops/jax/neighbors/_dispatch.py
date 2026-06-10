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


"""Private JAX neighbor-list dispatch helpers."""

from __future__ import annotations

from collections.abc import Iterable

import jax
import jax.numpy as jnp
import numpy as np
import warp as wp
from warp.jax_experimental import jax_kernel

from nvalchemiops.neighbors.base_dispatch import (
    _FLAG_NAMES,
    _OPTION_TARGET_INDICES,
    DEFAULT_BATCH_MAX_NBINS,
    DEFAULT_SINGLE_MAX_NBINS,
    FEATURE_BATCHED,
    FEATURE_CUDA,
    FEATURE_POSITIONS_FLOAT32,
    auto_base_constants,
    finalize_neighbor_list_method,
    get_select_neighbor_list_method_cost_kernel,
    neighbor_list_strategy_run_args,
    optional_outputs_mask,
)
from nvalchemiops.neighbors.base_dispatch import (
    estimate_neighbor_list_costs as _estimate_neighbor_list_costs_wp,
)

__all__ = [
    "_auto_base_method_from_geometry",
    "_auto_method_from_geometry",
    "_reject_unsupported_cluster_tile_combo",
    "estimate_neighbor_list_costs",
    "suggest_neighbor_list_method",
    "synthesize_cell_for_cell_list",
]

_jax_select_method_f32 = jax_kernel(
    get_select_neighbor_list_method_cost_kernel(wp.float32),
    num_outputs=2,
    in_out_argnames=["costs", "flags"],
    enable_backward=False,
)

_jax_select_method_f64 = jax_kernel(
    get_select_neighbor_list_method_cost_kernel(wp.float64),
    num_outputs=2,
    in_out_argnames=["costs", "flags"],
    enable_backward=False,
)


def _is_jax_cpu_array(array: jax.Array) -> bool:
    """Return whether ``array`` is backed by a CPU device."""
    try:
        return all(device.platform == "cpu" for device in array.devices())
    except AttributeError:
        return True


def _jax_selector_cpu_fallback(
    batch_ptr: jax.Array,
    batch_idx: jax.Array | None,
    cell: jax.Array,
    pbc: jax.Array,
    cutoff: float,
    *,
    max_nbins: int | None,
    option_mask: int,
    feature_mask: int,
    target_count: int | None,
) -> list[tuple[str, float]]:
    """Run the shared Warp selector directly on CPU for host-backed JAX arrays."""
    wp_batch_ptr = wp.array(
        np.asarray(jax.device_get(batch_ptr), dtype=np.int32),
        dtype=wp.int32,
        device="cpu",
    )
    wp_batch_idx = (
        wp.array(
            np.asarray(jax.device_get(batch_idx), dtype=np.int32),
            dtype=wp.int32,
            device="cpu",
        )
        if batch_idx is not None
        else None
    )
    cell_np = np.asarray(jax.device_get(cell))
    wp_cell_dtype = wp.mat33d if cell_np.dtype == np.float64 else wp.mat33f
    wp_cell = wp.array(cell_np, dtype=wp_cell_dtype, device="cpu")
    wp_pbc = wp.array(
        np.asarray(jax.device_get(pbc), dtype=np.bool_),
        dtype=wp.bool,
        device="cpu",
    )
    return _estimate_neighbor_list_costs_wp(
        wp_batch_ptr,
        wp_cell,
        wp_pbc,
        cutoff,
        batch_idx=wp_batch_idx,
        max_nbins=max_nbins,
        option_mask=option_mask,
        feature_mask=feature_mask,
        target_count=target_count,
    )


def _selector_batch_ptr_from_geometry(
    positions: jax.Array,
    batch_idx: jax.Array | None,
    batch_ptr: jax.Array | None,
    num_systems: int,
) -> jax.Array:
    """Return an int32 batch pointer for selector metadata."""
    if batch_ptr is not None:
        return batch_ptr.astype(jnp.int32)
    if batch_idx is not None:
        counts = jnp.bincount(batch_idx, length=int(num_systems)).astype(jnp.int32)
        return jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), jnp.cumsum(counts)])

    base = positions.shape[0] // max(int(num_systems), 1)
    counts = jnp.full((int(num_systems),), base, dtype=jnp.int32)
    if num_systems > 0:
        counts = counts.at[0].add(positions.shape[0] - base * int(num_systems))
    return jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), jnp.cumsum(counts)])


def _synthesize_cell_for_geometry(
    positions: jax.Array,
    batch_idx: jax.Array | None,
    batch_ptr: jax.Array,
    cutoff: float,
) -> tuple[jax.Array, jax.Array]:
    """Build non-PBC bounding-box cells for selector metadata."""
    num_systems = int(batch_ptr.shape[0]) - 1
    padding = jnp.asarray(float(cutoff) * 0.1, dtype=positions.dtype)
    if batch_idx is None or num_systems == 1:
        pos_min = jnp.min(positions, axis=0)
        shifted = positions - pos_min
        lengths = jnp.max(shifted, axis=0) + padding
        cell = jnp.diag(lengths).reshape(1, 3, 3)
        if num_systems > 1:
            cell = jnp.broadcast_to(cell, (num_systems, 3, 3))
        pbc = jnp.zeros((num_systems, 3), dtype=jnp.bool_)
        return cell, pbc

    pos_min = jax.ops.segment_min(positions, batch_idx, num_segments=num_systems)
    pos_max = jax.ops.segment_max(positions, batch_idx, num_segments=num_systems)
    counts = batch_ptr[1:] - batch_ptr[:-1]
    lengths = pos_max - pos_min + padding
    lengths = jnp.where(counts[:, None] > 0, lengths, padding)
    cell = lengths[:, :, None] * jnp.eye(3, dtype=positions.dtype)
    pbc = jnp.zeros((num_systems, 3), dtype=jnp.bool_)
    return cell, pbc


def synthesize_cell_for_cell_list(
    positions: jax.Array,
    cutoff: float,
    batch_idx: jax.Array | None = None,
    batch_ptr: jax.Array | None = None,
    num_systems: int = 1,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Build shifted non-PBC bounding-box cells for JAX cell-list dispatch."""
    num_systems = int(num_systems)
    if positions.shape[0] == 0:
        cell = jnp.tile(jnp.eye(3, dtype=positions.dtype), (num_systems, 1, 1))
        pbc = jnp.zeros((num_systems, 3), dtype=jnp.bool_)
        return positions, cell, pbc

    padding = jnp.asarray(float(cutoff) * 0.1, dtype=positions.dtype)
    if batch_idx is None and batch_ptr is not None:
        counts = batch_ptr[1:] - batch_ptr[:-1]
        batch_idx = jnp.repeat(jnp.arange(num_systems, dtype=jnp.int32), counts)

    if batch_idx is None or num_systems == 1:
        pos_min = jnp.min(positions, axis=0)
        shifted = positions - pos_min
        lengths = jnp.max(shifted, axis=0) + padding
        cell = jnp.diag(lengths).reshape(1, 3, 3)
        if num_systems > 1:
            cell = jnp.broadcast_to(cell, (num_systems, 3, 3))
            pbc = jnp.zeros((num_systems, 3), dtype=jnp.bool_)
        else:
            pbc = jnp.zeros((3,), dtype=jnp.bool_)
        return shifted, cell, pbc

    pos_min = jax.ops.segment_min(positions, batch_idx, num_segments=num_systems)
    pos_max = jax.ops.segment_max(positions, batch_idx, num_segments=num_systems)
    counts = (
        batch_ptr[1:] - batch_ptr[:-1]
        if batch_ptr is not None
        else jnp.bincount(batch_idx, length=num_systems)
    )
    lengths = pos_max - pos_min + padding
    lengths = jnp.where(counts[:, None] > 0, lengths, padding)
    shifted = positions - pos_min[batch_idx]
    cell = lengths[:, :, None] * jnp.eye(3, dtype=positions.dtype)
    pbc = jnp.zeros((num_systems, 3), dtype=jnp.bool_)
    return shifted, cell, pbc


def _normalize_selector_cell_pbc(
    cell: jax.Array,
    pbc: jax.Array,
    num_systems: int,
) -> tuple[jax.Array, jax.Array]:
    """Return selector metadata with one cell per system."""
    if cell.ndim == 2:
        cell = cell[jnp.newaxis, :, :]
    if cell.ndim != 3 or cell.shape[-2:] != (3, 3):
        raise ValueError("cell must have shape (3, 3) or (num_systems, 3, 3)")
    if cell.shape[0] == 1 and num_systems > 1:
        cell = jnp.broadcast_to(cell, (num_systems, 3, 3))
    if cell.shape[0] != num_systems:
        raise ValueError("cell must contain one matrix per system")

    if pbc.ndim not in (1, 2):
        raise ValueError("pbc must have shape (3,) or (num_systems, 3)")
    if pbc.ndim == 1 and pbc.shape[0] != 3:
        raise ValueError("pbc must have shape (3,) or (num_systems, 3)")
    if pbc.ndim == 2 and pbc.shape[0] == 1 and num_systems > 1:
        pbc = jnp.broadcast_to(pbc, (num_systems, 3))
    if pbc.ndim == 2 and (pbc.shape[0] != num_systems or pbc.shape[1] != 3):
        raise ValueError("pbc must have shape (3,) or (num_systems, 3)")
    return cell, pbc.astype(jnp.bool_)


def estimate_neighbor_list_costs(
    batch_ptr: jax.Array,
    cell: jax.Array,
    pbc: jax.Array,
    cutoff: float,
    *,
    batch_idx: jax.Array | None = None,
    max_nbins: int | None = None,
    optional_outputs: Iterable[str] | None = None,
    cutoff2: float | None = None,
    half_fill: bool = False,
    return_neighbor_list: bool = False,
    target_indices: jax.Array | None = None,
    return_vectors: bool = False,
    return_distances: bool = False,
    use_pair_fn: bool = False,
    rebuild_flags: jax.Array | None = None,
    wrap_positions: bool = True,
    positions_dtype=None,
) -> list[tuple[str, float]]:
    """Report feasible JAX neighbor-list strategies and their estimated cost.

    Parameters
    ----------
    batch_ptr : jax.Array, shape (num_systems + 1,), dtype=jnp.int32
        Cumulative atom counts. ``batch_ptr[-1]`` is the total atom count.
    cell : jax.Array, shape (3, 3) or (num_systems, 3, 3)
        Per-system cells, or one shared cell to broadcast.
    pbc : jax.Array, shape (3,) or (num_systems, 3), dtype=bool
        Shared or per-system PBC flags.
    cutoff : float
        Neighbor cutoff.
    max_nbins : int, optional
        Per-system cell-list cell cap. Defaults to the same cap used by the
        active single-system or batched frontend.

    Returns
    -------
    list of (str, float)
        Feasible strategies and their relative estimated cost (lower is
        faster), sorted cheapest-first.  Host-only: this syncs a tiny selector
        result, so call it outside ``jax.jit`` and pass the chosen name as an
        explicit ``method=``.
    """
    if batch_ptr.ndim != 1:
        raise ValueError("batch_ptr must be a 1-D array")
    num_systems = max(int(batch_ptr.shape[0]) - 1, 0)
    cell, pbc = _normalize_selector_cell_pbc(cell, pbc, num_systems)
    batch_ptr = batch_ptr.astype(jnp.int32)
    batch_idx_is_provided = batch_idx is not None
    if batch_idx is not None:
        batch_idx = batch_idx.astype(jnp.int32)
    if num_systems == 0:
        return [("cell_list_atom_centric", 0.0)]

    if max_nbins is None:
        max_nbins = (
            DEFAULT_SINGLE_MAX_NBINS if num_systems == 1 else DEFAULT_BATCH_MAX_NBINS
        )
    if int(max_nbins) <= 0:
        raise ValueError("max_nbins must be positive")

    costs = jnp.zeros(5, dtype=jnp.float32)
    flags = jnp.zeros(len(_FLAG_NAMES), dtype=jnp.int32)
    shell, setup = auto_base_constants()
    pbc_is_batched = pbc.ndim == 2
    pbc_single = pbc if not pbc_is_batched else jnp.zeros((0,), dtype=jnp.bool_)
    pbc_batch = pbc if pbc_is_batched else jnp.zeros((0, 3), dtype=jnp.bool_)
    batch_idx_arg = (
        batch_idx if batch_idx is not None else jnp.zeros((0,), dtype=jnp.int32)
    )
    feature_mask = 0
    try:
        if (
            "gpu" in str(cell.devices()).lower()
            or "cuda" in str(cell.devices()).lower()
        ):
            feature_mask |= FEATURE_CUDA
    except (AttributeError, TypeError):
        pass
    position_dtype = cell.dtype if positions_dtype is None else positions_dtype
    if position_dtype == jnp.float32:
        feature_mask |= FEATURE_POSITIONS_FLOAT32
    if num_systems > 1:
        feature_mask |= FEATURE_BATCHED
    options = optional_outputs_mask(
        optional_outputs,
        cutoff2=cutoff2,
        half_fill=half_fill,
        return_neighbor_list=return_neighbor_list,
        target_indices=target_indices,
        return_vectors=return_vectors,
        return_distances=return_distances,
        use_pair_fn=use_pair_fn,
        rebuild_flags=rebuild_flags,
        wrap_positions=wrap_positions,
    )
    target_count = int(target_indices.shape[0]) if target_indices is not None else 0
    target_count_arg = target_count if target_indices is not None else None
    if options & _OPTION_TARGET_INDICES and target_indices is None:
        raise ValueError(
            "target_count is required when target_indices is included in "
            "optional_outputs"
        )

    if _is_jax_cpu_array(cell):
        return _jax_selector_cpu_fallback(
            batch_ptr,
            batch_idx,
            cell,
            pbc,
            cutoff,
            max_nbins=max_nbins,
            option_mask=options,
            feature_mask=feature_mask,
            target_count=target_count_arg,
        )

    if cell.dtype == jnp.float64:
        kernel = _jax_select_method_f64
        cell = cell.astype(jnp.float64)
    else:
        kernel = _jax_select_method_f32
        cell = cell.astype(jnp.float32)

    costs, flags = kernel(
        batch_ptr,
        batch_idx_arg,
        batch_idx_is_provided,
        cell,
        pbc_single,
        pbc_batch,
        pbc_is_batched,
        float(cutoff),
        float(shell),
        float(setup),
        int(max_nbins),
        int(2**31 - 1),
        int(options),
        int(feature_mask),
        int(target_count),
        costs,
        flags,
        launch_dims=(
            max(num_systems, int(batch_idx.shape[0]) if batch_idx is not None else 0),
        ),
    )
    strategies = finalize_neighbor_list_method(
        jax.device_get(costs), jax.device_get(flags)
    )
    if num_systems > 1:
        strategies = [("batch_" + name, cost) for name, cost in strategies]
    return strategies


def suggest_neighbor_list_method(*args, **kwargs) -> str:
    """Return the cheapest feasible JAX neighbor-list strategy name.

    Thin wrapper over :func:`estimate_neighbor_list_costs` returning only the
    top-ranked strategy.  Same arguments and same host-only sync caveat.
    """
    return estimate_neighbor_list_costs(*args, **kwargs)[0][0]


def _reject_unsupported_cluster_tile_combo(
    pbc: jax.Array | None,
    half_fill: bool,
) -> None:
    """Raise ``NotImplementedError`` for unsupported explicit cluster-tile combos."""
    if pbc is None:
        raise NotImplementedError(
            "method='cluster_tile' / 'batch_cluster_tile' is "
            "fundamentally PBC-implicit; non-periodic systems (pbc=None) "
            "are not supported.  Use method='naive' or 'cell_list', "
            "or pass a cell with fully periodic pbc."
        )
    try:
        all_periodic = bool(jax.device_get(jnp.all(pbc)))
    except RuntimeError:
        all_periodic = True
    if not all_periodic:
        raise NotImplementedError(
            "method='cluster_tile' / 'batch_cluster_tile' is "
            "fundamentally PBC-implicit; pbc with any False entry "
            "is not supported.  Use method='naive' or 'cell_list'."
        )
    if half_fill:
        raise NotImplementedError(
            "method='cluster_tile' / 'batch_cluster_tile' uses "
            "tile-level upper-triangular iteration; atom-level "
            "half_fill=True is not supported.  Use method='naive' or "
            "'cell_list' for half-fill output."
        )


def _auto_method_from_geometry(
    positions: jax.Array,
    cutoff: float,
    cell: jax.Array | None,
    pbc: jax.Array | None,
    batch_idx: jax.Array | None,
    batch_ptr: jax.Array | None,
    num_systems: int,
    *,
    optional_outputs: Iterable[str] | None = None,
    cutoff2: float | None = None,
    half_fill: bool = False,
    return_neighbor_list: bool = False,
    target_indices: jax.Array | None = None,
    return_vectors: bool = False,
    return_distances: bool = False,
    use_pair_fn: bool = False,
    rebuild_flags: jax.Array | None = None,
    wrap_positions: bool = True,
) -> str:
    """Return the cheapest base strategy name for the geometry (no ``batch_``).

    The ``batch_`` prefix is added by the caller from ``num_systems``; this
    returns the unbatched fine-grained name, e.g. ``"naive_tile"``.
    """
    if positions.shape[0] == 0:
        return "cell_list_atom_centric"

    selector_batch_ptr = _selector_batch_ptr_from_geometry(
        positions, batch_idx, batch_ptr, num_systems
    )
    if cell is None:
        selector_cell, selector_pbc = _synthesize_cell_for_geometry(
            positions, batch_idx, selector_batch_ptr, cutoff
        )
    else:
        if pbc is None:
            raise ValueError("pbc is required when cell is provided")
        selector_cell = cell
        selector_pbc = pbc

    name = suggest_neighbor_list_method(
        selector_batch_ptr,
        selector_cell,
        selector_pbc,
        cutoff,
        batch_idx=batch_idx,
        optional_outputs=optional_outputs,
        cutoff2=cutoff2,
        half_fill=half_fill,
        return_neighbor_list=return_neighbor_list,
        target_indices=target_indices,
        return_vectors=return_vectors,
        return_distances=return_distances,
        use_pair_fn=use_pair_fn,
        rebuild_flags=rebuild_flags,
        wrap_positions=wrap_positions,
        positions_dtype=positions.dtype,
    )
    return name[len("batch_") :] if name.startswith("batch_") else name


def _auto_base_method_from_geometry(
    positions: jax.Array,
    cutoff: float,
    cell: jax.Array | None,
    pbc: jax.Array | None,
    batch_idx: jax.Array | None,
    batch_ptr: jax.Array | None,
    num_systems: int,
) -> str:
    """Pick the base method (``naive`` / ``cell_list`` / ``cluster_tile``)."""
    name = _auto_method_from_geometry(
        positions, cutoff, cell, pbc, batch_idx, batch_ptr, num_systems
    )
    return neighbor_list_strategy_run_args(name)[0]
