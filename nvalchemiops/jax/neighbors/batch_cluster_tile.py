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

"""JAX bindings for the batched cluster-pair tile neighbor list.

Mirrors :mod:`nvalchemiops.torch.neighbors.batch_cluster_tile`: the per-system
Morton sort + padded SoA gather happens in JAX, then the bbox reduction +
tile-pair enumeration and the final conversion (matrix / COO) run on the
Warp side via ``jax_callable`` callbacks.

Scope: triclinic-safe, float32, ``N >= 0``, variable per-system N.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import warp as wp
from warp.jax_experimental import GraphMode, jax_callable

from nvalchemiops.neighbors.cluster_tile import (
    TILE_GROUP_SIZE,
)
from nvalchemiops.neighbors.cluster_tile import (
    batch_build_cluster_tile_list as _warp_batch_build_cluster_tile_list,
)
from nvalchemiops.neighbors.cluster_tile import (
    batch_query_cluster_tile as _warp_batch_query_cluster_tile,
)
from nvalchemiops.neighbors.cluster_tile import (
    batch_query_cluster_tile_coo as _warp_batch_query_cluster_tile_coo,
)
from nvalchemiops.neighbors.cluster_tile import (
    estimate_batch_cluster_tile_segments as _warp_estimate_batch_cluster_tile_segments,
)
from nvalchemiops.neighbors.cluster_tile import (
    estimate_max_tiles_per_group as _estimate_max_tiles_per_group,
)
from nvalchemiops.neighbors.neighbor_utils import (
    NeighborOverflowError,
    estimate_max_neighbors,
)

__all__ = [
    "TILE_GROUP_SIZE",
    "allocate_batch_cluster_tile_list",
    "batch_cluster_tile_neighbor_list",
    "batch_query_cluster_tile_coo",
    "batch_query_cluster_tile",
    "batch_build_cluster_tile_list",
    "estimate_batch_cluster_tile_list_sizes",
    "estimate_batch_cluster_tile_segments",
]


# =============================================================================
# Sizing helper (pure JAX, no Warp launches)
# =============================================================================
def _batch_tile_buffer_max_tiles_per_group(
    positions, batch_ptr, cutoff, cell_batch
) -> int:
    """``max_tiles_per_group`` for the batched compact tile buffer.

    The compact buffer is ``ngroup_total * min(ngroup_total, mtpg)``.
    ``batch_ptr`` is structural (always concrete), so per-system ``ngroup`` is
    available.  When ``positions`` and ``cell_batch`` are concrete we
    density-size from each system's geometry and take the max.  When traced
    (e.g. ``grad`` w.r.t. cell, or ``jit``) we fall back to ``max_i ngroup_i``:
    the compact capacity ``ngroup_total * max_i ngroup_i >= sum_i ngroup_i**2``
    then bounds the upper-triangular maximum, so it can never overflow.
    """
    counts = np.asarray(batch_ptr).reshape(-1)
    per_sys = (counts[1:] - counts[:-1]).astype(np.int64)
    ngroups = [(int(n) + TILE_GROUP_SIZE - 1) // TILE_GROUP_SIZE for n in per_sys]
    max_ng = max(ngroups) if ngroups else 1
    cutoff_concrete = not isinstance(cutoff, jax.core.Tracer)
    vols = _concrete_cell_batch_volumes(cell_batch)
    if isinstance(positions, jax.core.Tracer) or vols is None or not cutoff_concrete:
        return max(max_ng, 1)
    best = 256
    for n, vol in zip(per_sys, vols):
        best = max(
            best, _estimate_max_tiles_per_group(int(n), float(cutoff), float(vol))
        )
    return best


def _concrete_cell_batch_volumes(cell_batch) -> list[float] | None:
    """Per-system ``abs(det(cell))`` if ``cell_batch`` is concrete, else None."""
    try:
        arr = np.asarray(cell_batch).reshape(-1, 3, 3)
    except Exception:
        return None
    return [float(abs(np.linalg.det(c))) for c in arr]


def estimate_batch_cluster_tile_list_sizes(
    batch_ptr: jax.Array,
    max_tiles_per_group: int = 256,
) -> tuple[int, int, int, int, int]:
    """Estimate allocation sizes for the batched tile neighbor list state.

    Mirrors
    :func:`nvalchemiops.torch.neighbors.batch_cluster_tile.estimate_batch_cluster_tile_list_sizes`.

    Returns ``(n_padded, ngroup, ngroup_padded, max_tiles, num_systems)``.
    The ``.item()`` syncs are necessary to size buffers; cache the result
    if calling from a hot loop.
    """
    num_systems = int(batch_ptr.shape[0]) - 1
    natom_per_system = (batch_ptr[1:] - batch_ptr[:-1]).astype(jnp.int32)
    natom_padded_per_system = (
        (natom_per_system + TILE_GROUP_SIZE - 1) // TILE_GROUP_SIZE
    ) * TILE_GROUP_SIZE
    n_padded = int(natom_padded_per_system.sum())
    ngroup = n_padded // TILE_GROUP_SIZE
    ngroup_padded = (
        (ngroup + TILE_GROUP_SIZE - 1) // TILE_GROUP_SIZE
    ) * TILE_GROUP_SIZE + TILE_GROUP_SIZE
    max_tiles = ngroup * min(ngroup, max_tiles_per_group)
    return n_padded, ngroup, ngroup_padded, max_tiles, num_systems


def estimate_batch_cluster_tile_segments(
    batch_ptr: jax.Array,
    max_neighbors: int,
    max_tiles_per_group: int = 256,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Estimate fixed per-system tile and COO segment buffers.

    Returns ``(tile_capacities, tile_offsets, pair_capacities, pair_offsets)``
    as int32 JAX arrays on the same backend as ``batch_ptr``.
    """
    tile_caps, tile_offsets, pair_caps, pair_offsets = (
        _warp_estimate_batch_cluster_tile_segments(
            batch_ptr,
            max_neighbors=int(max_neighbors),
            max_tiles_per_group=int(max_tiles_per_group),
        )
    )
    try:
        device = next(iter(batch_ptr.devices()))
    except AttributeError:
        device = batch_ptr.device() if callable(batch_ptr.device) else batch_ptr.device
    return (
        jax.device_put(jnp.asarray(tile_caps, dtype=jnp.int32), device),
        jax.device_put(jnp.asarray(tile_offsets, dtype=jnp.int32), device),
        jax.device_put(jnp.asarray(pair_caps, dtype=jnp.int32), device),
        jax.device_put(jnp.asarray(pair_offsets, dtype=jnp.int32), device),
    )


def allocate_batch_cluster_tile_list(
    batch_ptr: jax.Array,
    max_neighbors: int,
    *,
    max_tiles_per_group: int = 256,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Allocate zeroed persistent tile buffers for the selective rebuild path.

    JAX counterpart of
    :func:`nvalchemiops.torch.neighbors.batch_cluster_tile.allocate_batch_cluster_tile_list`.
    The dense ``format="tile"`` build allocates its own state, but the
    *selective* rebuild path of :func:`batch_cluster_tile_neighbor_list`
    threads persistent tile buffers across steps and the caller must supply
    them.  ``tile_system`` in particular MUST be zero-initialized: the
    segmented + batched query kernel reads ``tile_system[tile]`` for every
    allocated slot (including unwritten gap slots) *before* bounds-guarding, so
    an uninitialized buffer (e.g. ``jnp.empty``) can drive out-of-bounds
    indexing.  This helper guarantees the zeroing and sizes the buffers
    consistently with :func:`estimate_batch_cluster_tile_segments`.

    Parameters
    ----------
    batch_ptr : jax.Array, shape (S + 1,), dtype=int32
        Cumulative atom counts.
    max_neighbors : int
        Per-atom neighbor capacity; sizes the per-system tile segments.
    max_tiles_per_group : int, default 256
        Per-group tile cap used by the segment estimator.

    Returns
    -------
    tuple of jax.Array
        ``(num_tiles, tile_row_group, tile_col_group, tile_system,
        tile_counts, tile_offsets)``, all int32 on ``batch_ptr``'s backend.
        Pass them to :func:`batch_cluster_tile_neighbor_list` as
        ``previous_num_tiles``, ``previous_tile_row_group``,
        ``previous_tile_col_group``, ``previous_tile_system``,
        ``previous_tile_counts``, and ``tile_offsets`` respectively.
    """
    num_systems = int(batch_ptr.shape[0]) - 1
    _tile_caps, tile_offsets, _pair_caps, _pair_offsets = (
        estimate_batch_cluster_tile_segments(
            batch_ptr,
            max_neighbors=int(max_neighbors),
            max_tiles_per_group=int(max_tiles_per_group),
        )
    )
    max_tiles = int(tile_offsets[-1])
    try:
        device = next(iter(batch_ptr.devices()))
    except AttributeError:
        device = batch_ptr.device() if callable(batch_ptr.device) else batch_ptr.device

    def _zeros(n: int) -> jax.Array:
        return jax.device_put(jnp.zeros(n, dtype=jnp.int32), device)

    return (
        _zeros(1),  # num_tiles
        _zeros(max_tiles),  # tile_row_group
        _zeros(max_tiles),  # tile_col_group
        _zeros(max_tiles),  # tile_system
        _zeros(num_systems),  # tile_counts
        tile_offsets,  # tile_offsets (already on device)
    )


# =============================================================================
# Morton helpers (JAX, mirror the torch path)
# =============================================================================
def _spread_10bit(x: jax.Array) -> jax.Array:
    """Spread the low 10 bits of ``x`` across a 30-bit Morton code."""
    x = x & 0x3FF
    x = (x | (x << 16)) & 0x030000FF
    x = (x | (x << 8)) & 0x0300F00F
    x = (x | (x << 4)) & 0x030C30C3
    x = (x | (x << 2)) & 0x09249249
    return x


def _per_atom_morton_codes(
    positions: jax.Array,
    batch_idx: jax.Array,
    inv_cell_batch: jax.Array,
) -> jax.Array:
    """Compute per-atom 30-bit Morton code using the per-system inverse cell."""
    inv_per_atom = inv_cell_batch[batch_idx]
    # frac = positions @ inv^T per atom
    frac = jnp.einsum("ij,ikj->ik", positions, inv_per_atom)
    frac = frac - jnp.floor(frac)
    bucket = jnp.clip((frac * 1024.0).astype(jnp.int32), 0, 1023)
    ix, iy, iz = bucket[:, 0], bucket[:, 1], bucket[:, 2]
    return (_spread_10bit(iz) << 2) | (_spread_10bit(iy) << 1) | _spread_10bit(ix)


def _batched_morton_sort_padded(
    positions: jax.Array,
    batch_idx: jax.Array,
    batch_ptr: jax.Array,
    inv_cell_batch: jax.Array,
    n_padded: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Per-system Morton sort into the padded SoA layout.

    Returns ``(sorted_atom_index, sort_inv, sorted_pos_x, sorted_pos_y,
    sorted_pos_z, batch_idx_sorted)`` plus the running padded prefix
    ``batch_ptr_padded`` (returned last in a 7-tuple actually — see code).
    """
    N = positions.shape[0]
    num_systems = inv_cell_batch.shape[0]

    natom_per_system = batch_ptr[1:] - batch_ptr[:-1]
    natom_padded_per_system = (
        (natom_per_system + TILE_GROUP_SIZE - 1) // TILE_GROUP_SIZE
    ) * TILE_GROUP_SIZE
    batch_ptr_padded = jnp.concatenate(
        [
            jnp.zeros(1, dtype=jnp.int32),
            jnp.cumsum(natom_padded_per_system, dtype=jnp.int32),
        ]
    )

    codes = _per_atom_morton_codes(positions, batch_idx, inv_cell_batch)
    # Lexicographic (batch_idx, codes) sort via two stable passes — JAX
    # disables int64 by default, so packing (batch_idx << 32 | codes) into
    # an int64 key would silently truncate and break the per-system
    # ordering for batches with multiple systems.
    perm_morton = jnp.argsort(codes, stable=True)
    bi_after_morton = batch_idx[perm_morton]
    perm_outer = jnp.argsort(bi_after_morton, stable=True)
    sorted_atom_index_real = perm_morton[perm_outer].astype(jnp.int32)

    # Inverse permutation over real atoms.
    sort_inv = jnp.zeros(N, dtype=jnp.int32)
    sort_inv = sort_inv.at[sorted_atom_index_real].set(
        jnp.arange(N, dtype=jnp.int32),
    )

    # Placement indices for each real atom in the padded layout.
    batch_idx_sorted_real = batch_idx[sorted_atom_index_real]
    within_system = jnp.arange(N, dtype=jnp.int32) - batch_ptr[batch_idx_sorted_real]
    padded_slot = batch_ptr_padded[batch_idx_sorted_real] + within_system

    # sorted_atom_index[k] = N is the padding sentinel.
    sp = jnp.full((n_padded,), N, dtype=jnp.int32)
    sp = sp.at[padded_slot].set(sorted_atom_index_real)

    # System index for every padded slot.
    batch_idx_sorted = jnp.repeat(
        jnp.arange(num_systems, dtype=jnp.int32),
        natom_padded_per_system.astype(jnp.int32),
        total_repeat_length=n_padded,
    )

    # Padding slots duplicate each system's first real atom.
    first_atom_per_system = batch_ptr[:-1]
    position_index = jnp.where(
        sp == N,
        first_atom_per_system[batch_idx_sorted],
        sp,
    )
    sorted_pos = positions[position_index]
    return (
        sp,
        sort_inv,
        jnp.asarray(sorted_pos[:, 0]),
        jnp.asarray(sorted_pos[:, 1]),
        jnp.asarray(sorted_pos[:, 2]),
        batch_idx_sorted,
        batch_ptr_padded,
    )


# =============================================================================
# jax_callable wrappers
# =============================================================================
def _batch_build_cluster_tile_list_callback(
    sorted_pos_x: wp.array(dtype=wp.float32),
    sorted_pos_y: wp.array(dtype=wp.float32),
    sorted_pos_z: wp.array(dtype=wp.float32),
    group_system: wp.array(dtype=wp.int32),
    group_ptr: wp.array(dtype=wp.int32),
    cell_batch: wp.array(dtype=wp.mat33f),
    inv_cell_batch: wp.array(dtype=wp.mat33f),
    group_ctr_x: wp.array(dtype=wp.float32),
    group_ctr_y: wp.array(dtype=wp.float32),
    group_ctr_z: wp.array(dtype=wp.float32),
    group_ext_x: wp.array(dtype=wp.float32),
    group_ext_y: wp.array(dtype=wp.float32),
    group_ext_z: wp.array(dtype=wp.float32),
    num_tiles: wp.array(dtype=wp.int32),
    tile_row_group: wp.array(dtype=wp.int32),
    tile_col_group: wp.array(dtype=wp.int32),
    tile_system: wp.array(dtype=wp.int32),
    cutoff: wp.float32,
) -> None:
    _warp_batch_build_cluster_tile_list(
        sorted_pos_x=sorted_pos_x,
        sorted_pos_y=sorted_pos_y,
        sorted_pos_z=sorted_pos_z,
        group_system=group_system,
        group_ptr=group_ptr,
        cell_batch=cell_batch,
        inv_cell_batch=inv_cell_batch,
        cutoff=float(cutoff),
        num_tiles=num_tiles,
        tile_row_group=tile_row_group,
        tile_col_group=tile_col_group,
        tile_system=tile_system,
        wp_dtype=wp.float32,
        device=str(sorted_pos_x.device),
        group_ctr_x_buffer=group_ctr_x,
        group_ctr_y_buffer=group_ctr_y,
        group_ctr_z_buffer=group_ctr_z,
        group_ext_x_buffer=group_ext_x,
        group_ext_y_buffer=group_ext_y,
        group_ext_z_buffer=group_ext_z,
    )


def _batch_build_cluster_tile_list_selective_callback(
    sorted_pos_x: wp.array(dtype=wp.float32),
    sorted_pos_y: wp.array(dtype=wp.float32),
    sorted_pos_z: wp.array(dtype=wp.float32),
    group_system: wp.array(dtype=wp.int32),
    group_ptr: wp.array(dtype=wp.int32),
    cell_batch: wp.array(dtype=wp.mat33f),
    inv_cell_batch: wp.array(dtype=wp.mat33f),
    group_ctr_x: wp.array(dtype=wp.float32),
    group_ctr_y: wp.array(dtype=wp.float32),
    group_ctr_z: wp.array(dtype=wp.float32),
    group_ext_x: wp.array(dtype=wp.float32),
    group_ext_y: wp.array(dtype=wp.float32),
    group_ext_z: wp.array(dtype=wp.float32),
    num_tiles: wp.array(dtype=wp.int32),
    tile_offsets: wp.array(dtype=wp.int32),
    tile_counts: wp.array(dtype=wp.int32),
    rebuild_flags: wp.array(dtype=wp.bool),
    tile_row_group: wp.array(dtype=wp.int32),
    tile_col_group: wp.array(dtype=wp.int32),
    tile_system: wp.array(dtype=wp.int32),
    cutoff: wp.float32,
) -> None:
    """jax_callable callback for selective batched tile enumeration."""
    _warp_batch_build_cluster_tile_list(
        sorted_pos_x=sorted_pos_x,
        sorted_pos_y=sorted_pos_y,
        sorted_pos_z=sorted_pos_z,
        group_system=group_system,
        group_ptr=group_ptr,
        cell_batch=cell_batch,
        inv_cell_batch=inv_cell_batch,
        cutoff=float(cutoff),
        num_tiles=num_tiles,
        tile_row_group=tile_row_group,
        tile_col_group=tile_col_group,
        tile_system=tile_system,
        wp_dtype=wp.float32,
        device=str(sorted_pos_x.device),
        group_ctr_x_buffer=group_ctr_x,
        group_ctr_y_buffer=group_ctr_y,
        group_ctr_z_buffer=group_ctr_z,
        group_ext_x_buffer=group_ext_x,
        group_ext_y_buffer=group_ext_y,
        group_ext_z_buffer=group_ext_z,
        rebuild_flags=rebuild_flags,
        tile_offsets=tile_offsets,
        tile_counts=tile_counts,
    )


def _batch_query_cluster_tile_callback(
    sorted_atom_index: wp.array(dtype=wp.int32),
    sorted_pos_x: wp.array(dtype=wp.float32),
    sorted_pos_y: wp.array(dtype=wp.float32),
    sorted_pos_z: wp.array(dtype=wp.float32),
    cell_batch: wp.array(dtype=wp.mat33f),
    inv_cell_batch: wp.array(dtype=wp.mat33f),
    num_tiles: wp.array(dtype=wp.int32),
    tile_row_group: wp.array(dtype=wp.int32),
    tile_col_group: wp.array(dtype=wp.int32),
    tile_system: wp.array(dtype=wp.int32),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    neighbor_matrix_shifts: wp.array(dtype=wp.int32, ndim=3),
    cutoff: wp.float32,
    natom: wp.int32,
) -> None:
    # No ``n_tiles`` scalar — kernel guards per-tile.
    _warp_batch_query_cluster_tile(
        sorted_atom_index=sorted_atom_index,
        sorted_pos_x=sorted_pos_x,
        sorted_pos_y=sorted_pos_y,
        sorted_pos_z=sorted_pos_z,
        cell_batch=cell_batch,
        inv_cell_batch=inv_cell_batch,
        num_tiles=num_tiles,
        tile_row_group=tile_row_group,
        tile_col_group=tile_col_group,
        tile_system=tile_system,
        cutoff=float(cutoff),
        natom=int(natom),
        neighbor_matrix=neighbor_matrix,
        num_neighbors=num_neighbors,
        neighbor_matrix_shifts=neighbor_matrix_shifts,
        wp_dtype=wp.float32,
        device=str(sorted_pos_x.device),
    )


def _batch_query_cluster_tile_selective_callback(
    sorted_atom_index: wp.array(dtype=wp.int32),
    sorted_pos_x: wp.array(dtype=wp.float32),
    sorted_pos_y: wp.array(dtype=wp.float32),
    sorted_pos_z: wp.array(dtype=wp.float32),
    cell_batch: wp.array(dtype=wp.mat33f),
    inv_cell_batch: wp.array(dtype=wp.mat33f),
    num_tiles: wp.array(dtype=wp.int32),
    tile_offsets: wp.array(dtype=wp.int32),
    tile_counts: wp.array(dtype=wp.int32),
    rebuild_flags: wp.array(dtype=wp.bool),
    tile_row_group: wp.array(dtype=wp.int32),
    tile_col_group: wp.array(dtype=wp.int32),
    tile_system: wp.array(dtype=wp.int32),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    neighbor_matrix_shifts: wp.array(dtype=wp.int32, ndim=3),
    cutoff: wp.float32,
    natom: wp.int32,
) -> None:
    """jax_callable callback for selective batched tile-pair → matrix conversion."""
    _warp_batch_query_cluster_tile(
        sorted_atom_index=sorted_atom_index,
        sorted_pos_x=sorted_pos_x,
        sorted_pos_y=sorted_pos_y,
        sorted_pos_z=sorted_pos_z,
        cell_batch=cell_batch,
        inv_cell_batch=inv_cell_batch,
        num_tiles=num_tiles,
        tile_row_group=tile_row_group,
        tile_col_group=tile_col_group,
        tile_system=tile_system,
        cutoff=float(cutoff),
        natom=int(natom),
        neighbor_matrix=neighbor_matrix,
        num_neighbors=num_neighbors,
        neighbor_matrix_shifts=neighbor_matrix_shifts,
        wp_dtype=wp.float32,
        device=str(sorted_pos_x.device),
        rebuild_flags=rebuild_flags,
        tile_offsets=tile_offsets,
        tile_counts=tile_counts,
    )


def _batch_query_cluster_tile_dual_callback(
    sorted_atom_index: wp.array(dtype=wp.int32),
    sorted_pos_x: wp.array(dtype=wp.float32),
    sorted_pos_y: wp.array(dtype=wp.float32),
    sorted_pos_z: wp.array(dtype=wp.float32),
    cell_batch: wp.array(dtype=wp.mat33f),
    inv_cell_batch: wp.array(dtype=wp.mat33f),
    num_tiles: wp.array(dtype=wp.int32),
    tile_row_group: wp.array(dtype=wp.int32),
    tile_col_group: wp.array(dtype=wp.int32),
    tile_system: wp.array(dtype=wp.int32),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    neighbor_matrix_shifts: wp.array(dtype=wp.int32, ndim=3),
    neighbor_matrix2: wp.array(dtype=wp.int32, ndim=2),
    num_neighbors2: wp.array(dtype=wp.int32),
    neighbor_matrix_shifts2: wp.array(dtype=wp.int32, ndim=3),
    cutoff: wp.float32,
    cutoff2: wp.float32,
    natom: wp.int32,
) -> None:
    """jax_callable callback for batched dual-cutoff matrix conversion."""
    _warp_batch_query_cluster_tile(
        sorted_atom_index=sorted_atom_index,
        sorted_pos_x=sorted_pos_x,
        sorted_pos_y=sorted_pos_y,
        sorted_pos_z=sorted_pos_z,
        cell_batch=cell_batch,
        inv_cell_batch=inv_cell_batch,
        num_tiles=num_tiles,
        tile_row_group=tile_row_group,
        tile_col_group=tile_col_group,
        tile_system=tile_system,
        cutoff=float(cutoff),
        cutoff2=float(cutoff2),
        natom=int(natom),
        neighbor_matrix=neighbor_matrix,
        num_neighbors=num_neighbors,
        neighbor_matrix_shifts=neighbor_matrix_shifts,
        neighbor_matrix2=neighbor_matrix2,
        num_neighbors2=num_neighbors2,
        neighbor_matrix_shifts2=neighbor_matrix_shifts2,
        wp_dtype=wp.float32,
        device=str(sorted_pos_x.device),
    )


def _batch_query_cluster_tile_dual_selective_callback(
    sorted_atom_index: wp.array(dtype=wp.int32),
    sorted_pos_x: wp.array(dtype=wp.float32),
    sorted_pos_y: wp.array(dtype=wp.float32),
    sorted_pos_z: wp.array(dtype=wp.float32),
    cell_batch: wp.array(dtype=wp.mat33f),
    inv_cell_batch: wp.array(dtype=wp.mat33f),
    num_tiles: wp.array(dtype=wp.int32),
    tile_offsets: wp.array(dtype=wp.int32),
    tile_counts: wp.array(dtype=wp.int32),
    rebuild_flags: wp.array(dtype=wp.bool),
    tile_row_group: wp.array(dtype=wp.int32),
    tile_col_group: wp.array(dtype=wp.int32),
    tile_system: wp.array(dtype=wp.int32),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    neighbor_matrix_shifts: wp.array(dtype=wp.int32, ndim=3),
    neighbor_matrix2: wp.array(dtype=wp.int32, ndim=2),
    num_neighbors2: wp.array(dtype=wp.int32),
    neighbor_matrix_shifts2: wp.array(dtype=wp.int32, ndim=3),
    cutoff: wp.float32,
    cutoff2: wp.float32,
    natom: wp.int32,
) -> None:
    """jax_callable callback for selective batched dual-cutoff conversion."""
    _warp_batch_query_cluster_tile(
        sorted_atom_index=sorted_atom_index,
        sorted_pos_x=sorted_pos_x,
        sorted_pos_y=sorted_pos_y,
        sorted_pos_z=sorted_pos_z,
        cell_batch=cell_batch,
        inv_cell_batch=inv_cell_batch,
        num_tiles=num_tiles,
        tile_row_group=tile_row_group,
        tile_col_group=tile_col_group,
        tile_system=tile_system,
        cutoff=float(cutoff),
        cutoff2=float(cutoff2),
        natom=int(natom),
        neighbor_matrix=neighbor_matrix,
        num_neighbors=num_neighbors,
        neighbor_matrix_shifts=neighbor_matrix_shifts,
        neighbor_matrix2=neighbor_matrix2,
        num_neighbors2=num_neighbors2,
        neighbor_matrix_shifts2=neighbor_matrix_shifts2,
        wp_dtype=wp.float32,
        device=str(sorted_pos_x.device),
        rebuild_flags=rebuild_flags,
        tile_offsets=tile_offsets,
        tile_counts=tile_counts,
    )


def _batch_query_cluster_tile_pair_callback(
    sorted_atom_index: wp.array(dtype=wp.int32),
    sorted_pos_x: wp.array(dtype=wp.float32),
    sorted_pos_y: wp.array(dtype=wp.float32),
    sorted_pos_z: wp.array(dtype=wp.float32),
    cell_batch: wp.array(dtype=wp.mat33f),
    inv_cell_batch: wp.array(dtype=wp.mat33f),
    num_tiles: wp.array(dtype=wp.int32),
    tile_row_group: wp.array(dtype=wp.int32),
    tile_col_group: wp.array(dtype=wp.int32),
    tile_system: wp.array(dtype=wp.int32),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    neighbor_matrix_shifts: wp.array(dtype=wp.int32, ndim=3),
    # ``neighbor_vectors`` is typed ``wp.array(dtype=wp.vec3f, ndim=2)`` so
    # the JAX-side ``(N, M, 3)`` float32 buffer maps to a ``(N, M)`` array
    # of ``vec3f`` — same convention as
    # :func:`nvalchemiops.jax.neighbors.cluster_tile._query_cluster_tile_pair_callback`.
    neighbor_vectors: wp.array(dtype=wp.vec3f, ndim=2),
    neighbor_distances: wp.array(dtype=wp.float32, ndim=2),
    cutoff: wp.float32,
    natom: wp.int32,
) -> None:
    # No ``n_tiles`` scalar — kernel guards per-tile.
    _warp_batch_query_cluster_tile(
        sorted_atom_index=sorted_atom_index,
        sorted_pos_x=sorted_pos_x,
        sorted_pos_y=sorted_pos_y,
        sorted_pos_z=sorted_pos_z,
        cell_batch=cell_batch,
        inv_cell_batch=inv_cell_batch,
        num_tiles=num_tiles,
        tile_row_group=tile_row_group,
        tile_col_group=tile_col_group,
        tile_system=tile_system,
        cutoff=float(cutoff),
        natom=int(natom),
        neighbor_matrix=neighbor_matrix,
        num_neighbors=num_neighbors,
        neighbor_matrix_shifts=neighbor_matrix_shifts,
        wp_dtype=wp.float32,
        device=str(sorted_pos_x.device),
        return_vectors=True,
        return_distances=True,
        neighbor_vectors=neighbor_vectors,
        neighbor_distances=neighbor_distances,
    )


def _batch_query_cluster_tile_coo_callback(
    sorted_atom_index: wp.array(dtype=wp.int32),
    sorted_pos_x: wp.array(dtype=wp.float32),
    sorted_pos_y: wp.array(dtype=wp.float32),
    sorted_pos_z: wp.array(dtype=wp.float32),
    cell_batch: wp.array(dtype=wp.mat33f),
    inv_cell_batch: wp.array(dtype=wp.mat33f),
    num_tiles: wp.array(dtype=wp.int32),
    tile_row_group: wp.array(dtype=wp.int32),
    tile_col_group: wp.array(dtype=wp.int32),
    tile_system: wp.array(dtype=wp.int32),
    pair_counter: wp.array(dtype=wp.int32),
    coo_list: wp.array(dtype=wp.int32, ndim=2),
    coo_shifts: wp.array(dtype=wp.int32, ndim=2),
    cutoff: wp.float32,
    natom: wp.int32,
    max_pairs: wp.int32,
) -> None:
    # No ``n_tiles`` scalar — kernel guards per-tile.
    _warp_batch_query_cluster_tile_coo(
        sorted_atom_index=sorted_atom_index,
        sorted_pos_x=sorted_pos_x,
        sorted_pos_y=sorted_pos_y,
        sorted_pos_z=sorted_pos_z,
        cell_batch=cell_batch,
        inv_cell_batch=inv_cell_batch,
        num_tiles=num_tiles,
        tile_row_group=tile_row_group,
        tile_col_group=tile_col_group,
        tile_system=tile_system,
        cutoff=float(cutoff),
        natom=int(natom),
        max_pairs=int(max_pairs),
        pair_counter=pair_counter,
        coo_list=coo_list,
        coo_shifts=coo_shifts,
        wp_dtype=wp.float32,
        device=str(sorted_pos_x.device),
    )


def _batch_query_cluster_tile_coo_segmented_callback(
    sorted_atom_index: wp.array(dtype=wp.int32),
    sorted_pos_x: wp.array(dtype=wp.float32),
    sorted_pos_y: wp.array(dtype=wp.float32),
    sorted_pos_z: wp.array(dtype=wp.float32),
    cell_batch: wp.array(dtype=wp.mat33f),
    inv_cell_batch: wp.array(dtype=wp.mat33f),
    num_tiles: wp.array(dtype=wp.int32),
    tile_offsets: wp.array(dtype=wp.int32),
    tile_counts: wp.array(dtype=wp.int32),
    rebuild_flags: wp.array(dtype=wp.bool),
    tile_row_group: wp.array(dtype=wp.int32),
    tile_col_group: wp.array(dtype=wp.int32),
    tile_system: wp.array(dtype=wp.int32),
    pair_counter: wp.array(dtype=wp.int32),
    pair_offsets: wp.array(dtype=wp.int32),
    pair_counts: wp.array(dtype=wp.int32),
    coo_list: wp.array(dtype=wp.int32, ndim=2),
    coo_shifts: wp.array(dtype=wp.int32, ndim=2),
    cutoff: wp.float32,
    natom: wp.int32,
    max_pairs: wp.int32,
) -> None:
    """jax_callable callback for batched fixed-segment selective COO."""
    _warp_batch_query_cluster_tile_coo(
        sorted_atom_index=sorted_atom_index,
        sorted_pos_x=sorted_pos_x,
        sorted_pos_y=sorted_pos_y,
        sorted_pos_z=sorted_pos_z,
        cell_batch=cell_batch,
        inv_cell_batch=inv_cell_batch,
        num_tiles=num_tiles,
        tile_row_group=tile_row_group,
        tile_col_group=tile_col_group,
        tile_system=tile_system,
        cutoff=float(cutoff),
        natom=int(natom),
        max_pairs=int(max_pairs),
        pair_counter=pair_counter,
        coo_list=coo_list,
        coo_shifts=coo_shifts,
        wp_dtype=wp.float32,
        device=str(sorted_pos_x.device),
        rebuild_flags=rebuild_flags,
        tile_offsets=tile_offsets,
        tile_counts=tile_counts,
        pair_offsets=pair_offsets,
        pair_counts=pair_counts,
    )


_jax_batch_build_cluster_tile_list = jax_callable(
    _batch_build_cluster_tile_list_callback,
    num_outputs=10,
    in_out_argnames=[
        "group_ctr_x",
        "group_ctr_y",
        "group_ctr_z",
        "group_ext_x",
        "group_ext_y",
        "group_ext_z",
        "num_tiles",
        "tile_row_group",
        "tile_col_group",
        "tile_system",
    ],
    graph_mode=GraphMode.WARP,
)

_jax_batch_build_cluster_tile_list_selective = jax_callable(
    _batch_build_cluster_tile_list_selective_callback,
    num_outputs=11,
    in_out_argnames=[
        "group_ctr_x",
        "group_ctr_y",
        "group_ctr_z",
        "group_ext_x",
        "group_ext_y",
        "group_ext_z",
        "num_tiles",
        "tile_counts",
        "tile_row_group",
        "tile_col_group",
        "tile_system",
    ],
    graph_mode=GraphMode.WARP,
)


_jax_batch_query_cluster_tile = jax_callable(
    _batch_query_cluster_tile_callback,
    num_outputs=3,
    in_out_argnames=["neighbor_matrix", "num_neighbors", "neighbor_matrix_shifts"],
    graph_mode=GraphMode.WARP,
)

_jax_batch_query_cluster_tile_selective = jax_callable(
    _batch_query_cluster_tile_selective_callback,
    num_outputs=3,
    in_out_argnames=["neighbor_matrix", "num_neighbors", "neighbor_matrix_shifts"],
    graph_mode=GraphMode.WARP,
)

_jax_batch_query_cluster_tile_dual = jax_callable(
    _batch_query_cluster_tile_dual_callback,
    num_outputs=6,
    in_out_argnames=[
        "neighbor_matrix",
        "num_neighbors",
        "neighbor_matrix_shifts",
        "neighbor_matrix2",
        "num_neighbors2",
        "neighbor_matrix_shifts2",
    ],
    graph_mode=GraphMode.WARP,
)

_jax_batch_query_cluster_tile_dual_selective = jax_callable(
    _batch_query_cluster_tile_dual_selective_callback,
    num_outputs=6,
    in_out_argnames=[
        "neighbor_matrix",
        "num_neighbors",
        "neighbor_matrix_shifts",
        "neighbor_matrix2",
        "num_neighbors2",
        "neighbor_matrix_shifts2",
    ],
    graph_mode=GraphMode.WARP,
)


_jax_batch_query_cluster_tile_pair = jax_callable(
    _batch_query_cluster_tile_pair_callback,
    num_outputs=5,
    in_out_argnames=[
        "neighbor_matrix",
        "num_neighbors",
        "neighbor_matrix_shifts",
        "neighbor_vectors",
        "neighbor_distances",
    ],
    graph_mode=GraphMode.WARP,
)


@functools.cache
def _get_jax_batch_cluster_tile_pair_fn_callable(pair_fn):
    """Build (and cache) a batched ``jax_callable`` closing over ``pair_fn`` for the
    cluster-tile pair-output → matrix kernel.

    Batched analogue of
    ``cluster_tile._get_jax_cluster_tile_pair_fn_callable``: adds the ``pair_params``
    input + ``pair_energies`` / ``pair_forces`` outputs.  Cached by ``pair_fn``
    identity; fp32-only.
    """

    def _callback(
        sorted_atom_index: wp.array(dtype=wp.int32),
        sorted_pos_x: wp.array(dtype=wp.float32),
        sorted_pos_y: wp.array(dtype=wp.float32),
        sorted_pos_z: wp.array(dtype=wp.float32),
        cell_batch: wp.array(dtype=wp.mat33f),
        inv_cell_batch: wp.array(dtype=wp.mat33f),
        num_tiles: wp.array(dtype=wp.int32),
        tile_row_group: wp.array(dtype=wp.int32),
        tile_col_group: wp.array(dtype=wp.int32),
        tile_system: wp.array(dtype=wp.int32),
        neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
        num_neighbors: wp.array(dtype=wp.int32),
        neighbor_matrix_shifts: wp.array(dtype=wp.int32, ndim=3),
        neighbor_vectors: wp.array(dtype=wp.vec3f, ndim=2),
        neighbor_distances: wp.array(dtype=wp.float32, ndim=2),
        pair_params: wp.array(dtype=wp.float32, ndim=2),
        pair_energies: wp.array(dtype=wp.float32, ndim=2),
        pair_forces: wp.array(dtype=wp.vec3f, ndim=2),
        cutoff: wp.float32,
        natom: wp.int32,
    ) -> None:
        _warp_batch_query_cluster_tile(
            sorted_atom_index=sorted_atom_index,
            sorted_pos_x=sorted_pos_x,
            sorted_pos_y=sorted_pos_y,
            sorted_pos_z=sorted_pos_z,
            cell_batch=cell_batch,
            inv_cell_batch=inv_cell_batch,
            num_tiles=num_tiles,
            tile_row_group=tile_row_group,
            tile_col_group=tile_col_group,
            tile_system=tile_system,
            cutoff=float(cutoff),
            natom=int(natom),
            neighbor_matrix=neighbor_matrix,
            num_neighbors=num_neighbors,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            wp_dtype=wp.float32,
            device=str(sorted_pos_x.device),
            return_vectors=True,
            return_distances=True,
            neighbor_vectors=neighbor_vectors,
            neighbor_distances=neighbor_distances,
            pair_fn=pair_fn,
            pair_params=pair_params,
            pair_energies=pair_energies,
            pair_forces=pair_forces,
        )

    return jax_callable(
        _callback,
        num_outputs=7,
        in_out_argnames=[
            "neighbor_matrix",
            "num_neighbors",
            "neighbor_matrix_shifts",
            "neighbor_vectors",
            "neighbor_distances",
            "pair_energies",
            "pair_forces",
        ],
        graph_mode=GraphMode.WARP,
    )


_jax_batch_query_cluster_tile_coo = jax_callable(
    _batch_query_cluster_tile_coo_callback,
    num_outputs=3,
    in_out_argnames=["pair_counter", "coo_list", "coo_shifts"],
    graph_mode=GraphMode.WARP,
)

_jax_batch_query_cluster_tile_coo_segmented = jax_callable(
    _batch_query_cluster_tile_coo_segmented_callback,
    num_outputs=4,
    in_out_argnames=["pair_counter", "pair_counts", "coo_list", "coo_shifts"],
    graph_mode=GraphMode.WARP,
)


# =============================================================================
# User-facing entry points
# =============================================================================
def _make_batch_idx(batch_ptr: jax.Array) -> jax.Array:
    """Build per-atom system index from a batch_ptr (cumulative atom counts)."""
    num_systems = int(batch_ptr.shape[0]) - 1
    natom_per_system = (batch_ptr[1:] - batch_ptr[:-1]).astype(jnp.int32)
    return jnp.repeat(
        jnp.arange(num_systems, dtype=jnp.int32),
        natom_per_system,
        total_repeat_length=int(batch_ptr[-1]),
    )


def batch_build_cluster_tile_list(
    positions: jax.Array,
    cutoff: float,
    cell_batch: jax.Array,
    batch_ptr: jax.Array,
    *,
    rebuild_flags: jax.Array | None = None,
    tile_offsets: jax.Array | None = None,
    tile_counts: jax.Array | None = None,
    num_tiles: jax.Array | None = None,
    tile_row_group: jax.Array | None = None,
    tile_col_group: jax.Array | None = None,
    tile_system: jax.Array | None = None,
) -> tuple[jax.Array, ...]:
    """Build the batched tile neighbor list state.

    Runs per-system Morton sort + padded SoA gather in JAX, then the
    bbox reduction + tile-pair enumeration on the Warp side.

    Parameters
    ----------
    positions : jax.Array, shape (N, 3), dtype=float32
        Concatenated atomic coordinates for all systems.
    cutoff : float
        Bbox cutoff distance.
    cell_batch : jax.Array, shape (S, 3, 3), dtype=float32
        Per-system unit cell matrices.
    batch_ptr : jax.Array, shape (S + 1,), dtype=int32
        Cumulative atom counts.

    Returns
    -------
    tuple of jax.Array
        ``(sorted_atom_index, sort_inv, sorted_pos_x, sorted_pos_y,
        sorted_pos_z, batch_idx_sorted, batch_ptr_padded, group_system,
        group_ptr, group_ctr_x, group_ctr_y, group_ctr_z, group_ext_x,
        group_ext_y, group_ext_z, num_tiles, tile_row_group,
        tile_col_group, tile_system)``.
    """
    if positions.dtype != jnp.float32:
        raise TypeError(
            "positions must be float32 (batch_cluster_tile kernels are float32 only)"
        )
    if cell_batch.dtype != jnp.float32:
        raise TypeError("cell_batch must be float32")
    if cell_batch.ndim != 3 or cell_batch.shape[1:] != (3, 3):
        raise ValueError(
            f"cell_batch must have shape (S, 3, 3); got {cell_batch.shape}"
        )
    if batch_ptr.dtype != jnp.int32:
        batch_ptr = batch_ptr.astype(jnp.int32)

    # Geometry-size the compact tile buffer so dense/high-cutoff systems don't
    # silently overflow; trace-safe ``max_i ngroup_i`` fallback when traced.
    max_tiles_per_group = _batch_tile_buffer_max_tiles_per_group(
        positions, batch_ptr, cutoff, cell_batch
    )
    n_padded, ngroup, ngroup_padded, max_tiles, _num_systems = (
        estimate_batch_cluster_tile_list_sizes(
            batch_ptr, max_tiles_per_group=max_tiles_per_group
        )
    )

    inv_cell_batch = jnp.linalg.inv(cell_batch)
    batch_idx = _make_batch_idx(batch_ptr)

    (
        sorted_atom_index,
        sort_inv,
        sorted_pos_x,
        sorted_pos_y,
        sorted_pos_z,
        batch_idx_sorted,
        batch_ptr_padded,
    ) = _batched_morton_sort_padded(
        positions,
        batch_idx,
        batch_ptr,
        inv_cell_batch,
        n_padded,
    )

    # Per-tile derived structure.
    group_ptr = (batch_ptr_padded // TILE_GROUP_SIZE).astype(jnp.int32)
    group_system = batch_idx_sorted[::TILE_GROUP_SIZE]

    group_ctr_x = jnp.zeros(ngroup_padded, dtype=jnp.float32)
    group_ctr_y = jnp.zeros(ngroup_padded, dtype=jnp.float32)
    group_ctr_z = jnp.zeros(ngroup_padded, dtype=jnp.float32)
    group_ext_x = jnp.zeros(ngroup_padded, dtype=jnp.float32)
    group_ext_y = jnp.zeros(ngroup_padded, dtype=jnp.float32)
    group_ext_z = jnp.zeros(ngroup_padded, dtype=jnp.float32)
    if num_tiles is None:
        num_tiles = jnp.zeros(1, dtype=jnp.int32)
    if tile_row_group is None:
        tile_row_group = jnp.zeros(max_tiles, dtype=jnp.int32)
    if tile_col_group is None:
        tile_col_group = jnp.zeros(max_tiles, dtype=jnp.int32)
    if tile_system is None:
        tile_system = jnp.zeros(max_tiles, dtype=jnp.int32)

    if rebuild_flags is None:
        (
            group_ctr_x,
            group_ctr_y,
            group_ctr_z,
            group_ext_x,
            group_ext_y,
            group_ext_z,
            num_tiles,
            tile_row_group,
            tile_col_group,
            tile_system,
        ) = _jax_batch_build_cluster_tile_list(
            sorted_pos_x,
            sorted_pos_y,
            sorted_pos_z,
            group_system,
            group_ptr,
            cell_batch,
            inv_cell_batch,
            group_ctr_x,
            group_ctr_y,
            group_ctr_z,
            group_ext_x,
            group_ext_y,
            group_ext_z,
            num_tiles,
            tile_row_group,
            tile_col_group,
            tile_system,
            float(cutoff),
        )
    else:
        if tile_offsets is None or tile_counts is None:
            raise ValueError(
                "rebuild_flags requires tile_offsets and tile_counts for "
                "batched cluster_tile builds"
            )
        rf = rebuild_flags.astype(jnp.bool_)
        (
            group_ctr_x,
            group_ctr_y,
            group_ctr_z,
            group_ext_x,
            group_ext_y,
            group_ext_z,
            num_tiles,
            tile_counts,
            tile_row_group,
            tile_col_group,
            tile_system,
        ) = _jax_batch_build_cluster_tile_list_selective(
            sorted_pos_x,
            sorted_pos_y,
            sorted_pos_z,
            group_system,
            group_ptr,
            cell_batch,
            inv_cell_batch,
            group_ctr_x,
            group_ctr_y,
            group_ctr_z,
            group_ext_x,
            group_ext_y,
            group_ext_z,
            num_tiles,
            tile_offsets.astype(jnp.int32),
            tile_counts.astype(jnp.int32),
            rf,
            tile_row_group,
            tile_col_group,
            tile_system,
            float(cutoff),
        )

    del ngroup  # implicit in group_system.shape[0]
    result = (
        sorted_atom_index,
        sort_inv,
        sorted_pos_x,
        sorted_pos_y,
        sorted_pos_z,
        batch_idx_sorted,
        batch_ptr_padded,
        group_system,
        group_ptr,
        group_ctr_x,
        group_ctr_y,
        group_ctr_z,
        group_ext_x,
        group_ext_y,
        group_ext_z,
        num_tiles,
        tile_row_group,
        tile_col_group,
        tile_system,
    )
    if rebuild_flags is not None:
        return (*result, tile_counts)
    return result


def batch_query_cluster_tile(
    sorted_atom_index: jax.Array,
    sorted_pos_x: jax.Array,
    sorted_pos_y: jax.Array,
    sorted_pos_z: jax.Array,
    cell_batch: jax.Array,
    num_tiles: jax.Array,
    tile_row_group: jax.Array,
    tile_col_group: jax.Array,
    tile_system: jax.Array,
    cutoff: float,
    natom: int,
    max_neighbors: int,
    *,
    fill_value: int | None = None,
    cutoff2: float | None = None,
    rebuild_flags: jax.Array | None = None,
    tile_offsets: jax.Array | None = None,
    tile_counts: jax.Array | None = None,
    batch_idx: jax.Array | None = None,
    neighbor_matrix: jax.Array | None = None,
    num_neighbors: jax.Array | None = None,
    neighbor_matrix_shifts: jax.Array | None = None,
    neighbor_matrix2: jax.Array | None = None,
    num_neighbors2: jax.Array | None = None,
    neighbor_matrix_shifts2: jax.Array | None = None,
    return_vectors: bool = False,
    return_distances: bool = False,
    pair_fn: wp.Function | None = None,
    pair_params: jax.Array | None = None,
    neighbor_vectors: jax.Array | None = None,
    neighbor_distances: jax.Array | None = None,
    pair_energies: jax.Array | None = None,
    pair_forces: jax.Array | None = None,
) -> tuple[jax.Array, ...]:
    """Convert the batched tile pair list to dense neighbor-matrix form.

    Cluster-tile does not support partial neighbor lists; no
    ``target_indices`` kwarg.  Pair-output kwargs raise
    ``NotImplementedError`` — see
    :func:`nvalchemiops.jax.neighbors.cluster_tile.query_cluster_tile` for
    the rationale.  Use the torch binding when these axes are needed.
    """

    if pair_fn is not None and pair_params is None:
        raise ValueError(
            "pair_fn requires pair_params (a per-atom (n_atoms, K) parameter array).",
        )
    has_pair_outputs = (
        bool(return_vectors) or bool(return_distances) or (pair_fn is not None)
    )
    dual_cutoff = cutoff2 is not None
    selective = rebuild_flags is not None
    if (dual_cutoff or selective) and has_pair_outputs:
        raise NotImplementedError(
            "JAX batch_cluster_tile cutoff2 / rebuild_flags are "
            "matrix-topology features in this pass and cannot be combined "
            "with return_distances or return_vectors.",
        )
    if selective:
        if tile_offsets is None or tile_counts is None:
            raise ValueError(
                "rebuild_flags requires tile_offsets and tile_counts for "
                "JAX batch_cluster_tile matrix queries"
            )
        if batch_idx is None:
            raise ValueError("batch_idx is required when rebuild_flags is provided")
    if fill_value is None:
        fill_value = natom
    inv_cell_batch = jnp.linalg.inv(cell_batch)

    if neighbor_matrix is None:
        neighbor_matrix = jnp.zeros((natom, max_neighbors), dtype=jnp.int32)
    elif not selective:
        neighbor_matrix = neighbor_matrix.at[:].set(jnp.int32(0))
    if num_neighbors is None:
        num_neighbors = jnp.zeros(natom, dtype=jnp.int32)
    elif not selective:
        num_neighbors = num_neighbors.at[:].set(jnp.int32(0))
    if neighbor_matrix_shifts is None:
        neighbor_matrix_shifts = jnp.zeros((natom, max_neighbors, 3), dtype=jnp.int32)
    elif not selective:
        neighbor_matrix_shifts = neighbor_matrix_shifts.at[:].set(jnp.int32(0))

    if selective:
        rf = rebuild_flags.astype(jnp.bool_)
        atom_rebuild = rf[batch_idx.astype(jnp.int32)]
        num_neighbors = jnp.where(
            atom_rebuild, jnp.zeros_like(num_neighbors), num_neighbors
        )
    else:
        rf = None

    if dual_cutoff:
        if neighbor_matrix2 is None:
            neighbor_matrix2 = jnp.zeros((natom, max_neighbors), dtype=jnp.int32)
        elif not selective:
            neighbor_matrix2 = neighbor_matrix2.at[:].set(jnp.int32(0))
        if num_neighbors2 is None:
            num_neighbors2 = jnp.zeros(natom, dtype=jnp.int32)
        elif not selective:
            num_neighbors2 = num_neighbors2.at[:].set(jnp.int32(0))
        if neighbor_matrix_shifts2 is None:
            neighbor_matrix_shifts2 = jnp.zeros(
                (natom, max_neighbors, 3), dtype=jnp.int32
            )
        elif not selective:
            neighbor_matrix_shifts2 = neighbor_matrix_shifts2.at[:].set(jnp.int32(0))
        if selective:
            num_neighbors2 = jnp.where(
                atom_rebuild, jnp.zeros_like(num_neighbors2), num_neighbors2
            )

    # No host-side ``int(num_tiles[0])`` read: the warp kernel launches
    # at the allocated tile-buffer capacity and early-returns per-tile
    # using the device-side ``num_tiles`` array.
    if has_pair_outputs:
        if neighbor_vectors is None:
            neighbor_vectors = jnp.zeros((natom, max_neighbors, 3), dtype=jnp.float32)
        if neighbor_distances is None:
            neighbor_distances = jnp.zeros((natom, max_neighbors), dtype=jnp.float32)
        if pair_fn is not None:
            if pair_energies is None:
                pair_energies = jnp.zeros((natom, max_neighbors), dtype=jnp.float32)
            if pair_forces is None:
                pair_forces = jnp.zeros((natom, max_neighbors, 3), dtype=jnp.float32)
            pair_params_arg = jnp.asarray(pair_params, dtype=jnp.float32)
            pair_callable = _get_jax_batch_cluster_tile_pair_fn_callable(pair_fn)
            (
                neighbor_matrix,
                num_neighbors,
                neighbor_matrix_shifts,
                neighbor_vectors,
                neighbor_distances,
                pair_energies,
                pair_forces,
            ) = pair_callable(
                sorted_atom_index,
                sorted_pos_x,
                sorted_pos_y,
                sorted_pos_z,
                cell_batch,
                inv_cell_batch,
                num_tiles,
                tile_row_group,
                tile_col_group,
                tile_system,
                neighbor_matrix,
                num_neighbors,
                neighbor_matrix_shifts,
                neighbor_vectors,
                neighbor_distances,
                pair_params_arg,
                pair_energies,
                pair_forces,
                float(cutoff),
                int(natom),
            )
            col_idx = jnp.arange(max_neighbors, dtype=jnp.int32)[jnp.newaxis, :]
            active = col_idx < num_neighbors[:, jnp.newaxis]
            neighbor_matrix = jnp.where(active, neighbor_matrix, jnp.int32(fill_value))
            return (
                neighbor_matrix,
                num_neighbors,
                neighbor_matrix_shifts,
                neighbor_vectors,
                neighbor_distances,
                pair_energies,
                pair_forces,
            )
        (
            neighbor_matrix,
            num_neighbors,
            neighbor_matrix_shifts,
            neighbor_vectors,
            neighbor_distances,
        ) = _jax_batch_query_cluster_tile_pair(
            sorted_atom_index,
            sorted_pos_x,
            sorted_pos_y,
            sorted_pos_z,
            cell_batch,
            inv_cell_batch,
            num_tiles,
            tile_row_group,
            tile_col_group,
            tile_system,
            neighbor_matrix,
            num_neighbors,
            neighbor_matrix_shifts,
            neighbor_vectors,
            neighbor_distances,
            float(cutoff),
            int(natom),
        )
        col_idx = jnp.arange(max_neighbors, dtype=jnp.int32)[jnp.newaxis, :]
        active = col_idx < num_neighbors[:, jnp.newaxis]
        neighbor_matrix = jnp.where(active, neighbor_matrix, jnp.int32(fill_value))
        return (
            neighbor_matrix,
            num_neighbors,
            neighbor_matrix_shifts,
            neighbor_vectors,
            neighbor_distances,
        )

    if dual_cutoff and selective:
        (
            neighbor_matrix,
            num_neighbors,
            neighbor_matrix_shifts,
            neighbor_matrix2,
            num_neighbors2,
            neighbor_matrix_shifts2,
        ) = _jax_batch_query_cluster_tile_dual_selective(
            sorted_atom_index,
            sorted_pos_x,
            sorted_pos_y,
            sorted_pos_z,
            cell_batch,
            inv_cell_batch,
            num_tiles,
            tile_offsets.astype(jnp.int32),
            tile_counts.astype(jnp.int32),
            rf,
            tile_row_group,
            tile_col_group,
            tile_system,
            neighbor_matrix,
            num_neighbors,
            neighbor_matrix_shifts,
            neighbor_matrix2,
            num_neighbors2,
            neighbor_matrix_shifts2,
            float(cutoff),
            float(cutoff2),
            int(natom),
        )
    elif dual_cutoff:
        (
            neighbor_matrix,
            num_neighbors,
            neighbor_matrix_shifts,
            neighbor_matrix2,
            num_neighbors2,
            neighbor_matrix_shifts2,
        ) = _jax_batch_query_cluster_tile_dual(
            sorted_atom_index,
            sorted_pos_x,
            sorted_pos_y,
            sorted_pos_z,
            cell_batch,
            inv_cell_batch,
            num_tiles,
            tile_row_group,
            tile_col_group,
            tile_system,
            neighbor_matrix,
            num_neighbors,
            neighbor_matrix_shifts,
            neighbor_matrix2,
            num_neighbors2,
            neighbor_matrix_shifts2,
            float(cutoff),
            float(cutoff2),
            int(natom),
        )
    elif selective:
        neighbor_matrix, num_neighbors, neighbor_matrix_shifts = (
            _jax_batch_query_cluster_tile_selective(
                sorted_atom_index,
                sorted_pos_x,
                sorted_pos_y,
                sorted_pos_z,
                cell_batch,
                inv_cell_batch,
                num_tiles,
                tile_offsets.astype(jnp.int32),
                tile_counts.astype(jnp.int32),
                rf,
                tile_row_group,
                tile_col_group,
                tile_system,
                neighbor_matrix,
                num_neighbors,
                neighbor_matrix_shifts,
                float(cutoff),
                int(natom),
            )
        )
    else:
        neighbor_matrix, num_neighbors, neighbor_matrix_shifts = (
            _jax_batch_query_cluster_tile(
                sorted_atom_index,
                sorted_pos_x,
                sorted_pos_y,
                sorted_pos_z,
                cell_batch,
                inv_cell_batch,
                num_tiles,
                tile_row_group,
                tile_col_group,
                tile_system,
                neighbor_matrix,
                num_neighbors,
                neighbor_matrix_shifts,
                float(cutoff),
                int(natom),
            )
        )

    col_idx = jnp.arange(max_neighbors, dtype=jnp.int32)[jnp.newaxis, :]
    active = col_idx < num_neighbors[:, jnp.newaxis]
    neighbor_matrix = jnp.where(active, neighbor_matrix, jnp.int32(fill_value))
    if dual_cutoff:
        active2 = col_idx < num_neighbors2[:, jnp.newaxis]
        neighbor_matrix2 = jnp.where(active2, neighbor_matrix2, jnp.int32(fill_value))
        return (
            neighbor_matrix,
            num_neighbors,
            neighbor_matrix_shifts,
            neighbor_matrix2,
            num_neighbors2,
            neighbor_matrix_shifts2,
        )
    return neighbor_matrix, num_neighbors, neighbor_matrix_shifts


def batch_query_cluster_tile_coo(
    sorted_atom_index: jax.Array,
    sorted_pos_x: jax.Array,
    sorted_pos_y: jax.Array,
    sorted_pos_z: jax.Array,
    cell_batch: jax.Array,
    num_tiles: jax.Array,
    tile_row_group: jax.Array,
    tile_col_group: jax.Array,
    tile_system: jax.Array,
    cutoff: float,
    natom: int,
    max_pairs: int,
    *,
    rebuild_flags: jax.Array | None = None,
    tile_offsets: jax.Array | None = None,
    tile_counts: jax.Array | None = None,
    pair_offsets: jax.Array | None = None,
    pair_counts: jax.Array | None = None,
    neighbor_list: jax.Array | None = None,
    neighbor_list_shifts: jax.Array | None = None,
) -> tuple[jax.Array, ...]:
    """Convert the batched tile pair list to flat COO form."""
    segmented = pair_offsets is not None or pair_counts is not None
    if (pair_offsets is None) != (pair_counts is None):
        raise ValueError("Pass both 'pair_offsets' and 'pair_counts', or neither.")
    if (tile_offsets is None) != (tile_counts is None):
        raise ValueError("Pass both 'tile_offsets' and 'tile_counts', or neither.")
    if rebuild_flags is not None and not segmented:
        raise ValueError("rebuild_flags requires pair_offsets and pair_counts")
    if rebuild_flags is not None and (tile_offsets is None or tile_counts is None):
        raise ValueError("rebuild_flags requires tile_offsets and tile_counts")

    inv_cell_batch = jnp.linalg.inv(cell_batch)
    pair_counter = jnp.zeros(1, dtype=jnp.int32)

    if segmented:
        max_pairs = int(pair_offsets[-1])
        if neighbor_list is None:
            neighbor_list = jnp.zeros((2, max_pairs), dtype=jnp.int32)
        if neighbor_list_shifts is None:
            neighbor_list_shifts = jnp.zeros((max_pairs, 3), dtype=jnp.int32)
        coo_list = neighbor_list.T.copy()
        coo_shifts = neighbor_list_shifts
        rf = (
            rebuild_flags.astype(jnp.bool_)
            if rebuild_flags is not None
            else jnp.ones_like(pair_counts, dtype=jnp.bool_)
        )
        pair_counter, pair_counts, coo_list, coo_shifts = (
            _jax_batch_query_cluster_tile_coo_segmented(
                sorted_atom_index,
                sorted_pos_x,
                sorted_pos_y,
                sorted_pos_z,
                cell_batch,
                inv_cell_batch,
                num_tiles,
                tile_offsets.astype(jnp.int32),
                tile_counts.astype(jnp.int32),
                rf,
                tile_row_group,
                tile_col_group,
                tile_system,
                pair_counter,
                pair_offsets.astype(jnp.int32),
                pair_counts.astype(jnp.int32),
                coo_list,
                coo_shifts,
                float(cutoff),
                int(natom),
                int(max_pairs),
            )
        )
        del pair_counter
        return coo_list.T, pair_offsets, pair_counts, coo_shifts

    coo_list = jnp.zeros((max_pairs, 2), dtype=jnp.int32)
    coo_shifts = jnp.zeros((max_pairs, 3), dtype=jnp.int32)
    pair_counter, coo_list, coo_shifts = _jax_batch_query_cluster_tile_coo(
        sorted_atom_index,
        sorted_pos_x,
        sorted_pos_y,
        sorted_pos_z,
        cell_batch,
        inv_cell_batch,
        num_tiles,
        tile_row_group,
        tile_col_group,
        tile_system,
        pair_counter,
        coo_list,
        coo_shifts,
        float(cutoff),
        int(natom),
        int(max_pairs),
    )

    npairs = int(pair_counter[0])
    coo_list_trim = coo_list[:npairs]
    coo_shifts_trim = coo_shifts[:npairs]
    neighbor_list = coo_list_trim.T
    per_atom = jnp.bincount(neighbor_list[0], length=natom).astype(jnp.int32)
    neighbor_ptr = jnp.concatenate(
        [
            jnp.zeros(1, dtype=jnp.int32),
            jnp.cumsum(per_atom, dtype=jnp.int32),
        ]
    )
    return neighbor_list, neighbor_ptr, coo_shifts_trim


def batch_cluster_tile_neighbor_list(
    positions: jax.Array,
    cutoff: float,
    cell_batch: jax.Array,
    batch_ptr: jax.Array,
    max_neighbors: int | None = None,
    fill_value: int | None = None,
    format: str = "matrix",
    max_pairs: int | None = None,
    *,
    cutoff2: float | None = None,
    rebuild_flags: jax.Array | None = None,
    tile_offsets: jax.Array | None = None,
    pair_offsets: jax.Array | None = None,
    previous_tile_counts: jax.Array | None = None,
    previous_pair_counts: jax.Array | None = None,
    previous_neighbor_list: jax.Array | None = None,
    previous_neighbor_list_shifts: jax.Array | None = None,
    previous_num_tiles: jax.Array | None = None,
    previous_tile_row_group: jax.Array | None = None,
    previous_tile_col_group: jax.Array | None = None,
    previous_tile_system: jax.Array | None = None,
    previous_neighbor_matrix: jax.Array | None = None,
    previous_num_neighbors: jax.Array | None = None,
    previous_neighbor_matrix_shifts: jax.Array | None = None,
    previous_neighbor_matrix2: jax.Array | None = None,
    previous_num_neighbors2: jax.Array | None = None,
    previous_neighbor_matrix_shifts2: jax.Array | None = None,
    return_vectors: bool = False,
    return_distances: bool = False,
    pair_fn: wp.Function | None = None,
    pair_params: jax.Array | None = None,
    neighbor_vectors: jax.Array | None = None,
    neighbor_distances: jax.Array | None = None,
    pair_energies: jax.Array | None = None,
    pair_forces: jax.Array | None = None,
) -> tuple[jax.Array, ...]:
    """Build a batched cluster-pair tile neighbor list (one-shot convenience).

    Batched JAX binding for the cluster-pair tile algorithm.  Per-system
    Morton sort and padded SoA gather happen in JAX; bbox reduction,
    tile-pair enumeration, and conversion (matrix / COO) run on the Warp
    side via ``jax_callable`` callbacks.  Mirrors
    :func:`nvalchemiops.torch.neighbors.batch_cluster_tile.batch_cluster_tile_neighbor_list`.

    Parameters
    ----------
    positions : jax.Array, shape (total_atoms, 3), dtype=float32
        Concatenated atomic coordinates across systems.
    cutoff : float
        Cutoff distance in Cartesian units. Must be positive.
    cutoff2 : float, optional
        Matrix-format second cutoff. Cannot be combined with pair outputs
        or COO/tile formats.
    rebuild_flags : jax.Array, shape (num_systems,), dtype=bool, optional
        Selective rebuild flags for matrix or segmented COO output. Requires
        fixed tile segments and previous output buffers.
    cell_batch : jax.Array, shape (num_systems, 3, 3), dtype=float32
        Per-system unit cell matrices. Cluster-tile assumes fully
        periodic boundaries.
    batch_ptr : jax.Array, shape (num_systems + 1,), dtype=int32
        CSR pointer separating systems.  Assumes positions are laid out
        in system-contiguous order; interleaved layouts are **not
        supported** and will silently emit cross-system pairs.
    max_neighbors : int, optional
        Max neighbors per atom (``"matrix"`` format only). Falls back to
        :func:`estimate_max_neighbors`.
    fill_value : int, optional
        Matrix sentinel; defaults to ``total_atoms``.
    format : {"matrix", "coo", "tile"}, default "matrix"
        Output representation. See Returns.
    max_pairs : int, optional
        Upper bound for compact COO output; defaults to
        ``total_atoms * max_neighbors``.
    tile_offsets, previous_tile_counts, pair_offsets, previous_pair_counts : jax.Array, optional
        Fixed per-system segmented tile/COO buffers used with
        ``rebuild_flags``. Size them with
        :func:`estimate_batch_cluster_tile_segments`.
    previous_neighbor_list, previous_neighbor_list_shifts : jax.Array, optional
        Fixed segmented-COO output buffers used with ``rebuild_flags`` and
        ``format="coo"``.
    return_vectors, return_distances : bool, default False
        If True, append per-pair displacement vectors / scalar distances
        to the matrix-format return tuple. Matrix format only.
    pair_fn, pair_params, neighbor_vectors, neighbor_distances, pair_energies, pair_forces : optional
        Reserved for parity with the torch binding. Raise
        ``NotImplementedError`` on the JAX path.

    Returns
    -------
    For ``format == "matrix"``:
        ``(neighbor_matrix, num_neighbors, neighbor_matrix_shifts)``, with
        optional ``(*, distances)`` and/or ``(*, vectors)`` appended when
        ``return_distances`` / ``return_vectors`` is True.
    For ``format == "coo"``:
        ``(neighbor_list, neighbor_ptr, neighbor_list_shifts)`` in compact
        mode, or ``(neighbor_list, pair_offsets, pair_counts,
        neighbor_list_shifts, tile_offsets, tile_counts, num_tiles,
        tile_row_group, tile_col_group, tile_system)`` with
        ``rebuild_flags`` segmented mode.
    For ``format == "tile"``:
        ``(num_tiles, tile_row_group, tile_col_group, tile_system,
        sorted_atom_index, sorted_pos_x, sorted_pos_y, sorted_pos_z,
        batch_idx_sorted, batch_ptr_padded, group_ptr)`` — same 11-tuple
        as the torch sibling so downstream tile consumers can be
        backend-agnostic.

    Notes
    -----
    - Cluster-tile is CUDA float32 only.
    - Cluster-tile does not support partial neighbor lists (no
      ``target_indices`` kwarg).
    - The unified :func:`nvalchemiops.jax.neighbors.neighbor_list` entry
      point may select this binding automatically when the selector guards
      and cost model prefer it; pass ``method="batch_cluster_tile"`` to
      force it.

    See Also
    --------
    nvalchemiops.jax.neighbors.cluster_tile_neighbor_list :
        Single-system companion entry point.
    nvalchemiops.jax.neighbors.batch_cluster_tile.batch_build_cluster_tile_list :
        Lower-level build step exposed for caching across queries.
    nvalchemiops.jax.neighbors.batch_cluster_tile.batch_query_cluster_tile :
        Lower-level query step.
    """

    from nvalchemiops.jax.neighbors._autograd import (
        _build_index_residuals,
        _NeighborForwardOutput,
        _route_pair_outputs,
    )
    from nvalchemiops.jax.neighbors.neighbor_utils import (
        coo_pack_pair_geometry,
        get_neighbor_list_from_neighbor_matrix,
    )

    if format not in ("matrix", "coo", "tile"):
        raise ValueError(
            f"format must be 'matrix' | 'coo' | 'tile'; got {format!r}",
        )
    if pair_fn is not None and pair_params is None:
        raise ValueError(
            "pair_fn requires pair_params (a per-atom (n_atoms, K) parameter array).",
        )
    has_pair_outputs = (
        bool(return_vectors) or bool(return_distances) or (pair_fn is not None)
    )
    dual_cutoff = cutoff2 is not None
    selective = rebuild_flags is not None
    if has_pair_outputs and format == "tile":
        raise NotImplementedError(
            "return_distances / return_vectors / pair_fn are not supported with "
            "format='tile' on the JAX batch_cluster_tile binding; use 'matrix' or "
            "'coo'.",
        )
    if dual_cutoff and format != "matrix":
        raise NotImplementedError(
            "JAX batch_cluster_tile cutoff2 is supported only with format='matrix'.",
        )
    if selective and format not in {"matrix", "coo"}:
        raise NotImplementedError(
            "JAX batch_cluster_tile rebuild_flags are supported only with "
            "format='matrix' or segmented format='coo'.",
        )
    if (dual_cutoff or selective) and has_pair_outputs:
        raise NotImplementedError(
            "JAX batch_cluster_tile cutoff2 / rebuild_flags cannot be "
            "combined with return_distances or return_vectors in this pass.",
        )
    if selective:
        required = {
            "tile_offsets": tile_offsets,
            "previous_tile_counts": previous_tile_counts,
            "previous_num_tiles": previous_num_tiles,
            "previous_tile_row_group": previous_tile_row_group,
            "previous_tile_col_group": previous_tile_col_group,
            "previous_tile_system": previous_tile_system,
        }
        if format == "matrix":
            required.update(
                {
                    "previous_neighbor_matrix": previous_neighbor_matrix,
                    "previous_num_neighbors": previous_num_neighbors,
                    "previous_neighbor_matrix_shifts": previous_neighbor_matrix_shifts,
                }
            )
            if dual_cutoff:
                required.update(
                    {
                        "previous_neighbor_matrix2": previous_neighbor_matrix2,
                        "previous_num_neighbors2": previous_num_neighbors2,
                        "previous_neighbor_matrix_shifts2": previous_neighbor_matrix_shifts2,
                    }
                )
        elif format == "coo":
            required.update(
                {
                    "pair_offsets": pair_offsets,
                    "previous_pair_counts": previous_pair_counts,
                    "previous_neighbor_list": previous_neighbor_list,
                    "previous_neighbor_list_shifts": previous_neighbor_list_shifts,
                }
            )
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                "rebuild_flags requires previous batch_cluster_tile state: "
                + ", ".join(missing)
            )
    N = positions.shape[0]
    if max_neighbors is None:
        max_neighbors = estimate_max_neighbors(
            cutoff2 if cutoff2 is not None else cutoff
        )

    if N == 0:
        # Public docstring promises any ``N >= 0``.  Empty batches would
        # otherwise trip ``positions[-1:]`` in the JAX morton sort and
        # zero-length ``jax.lax.dynamic_slice`` ops downstream.
        nm0 = jnp.empty((0, int(max_neighbors)), dtype=jnp.int32)
        nn0 = jnp.empty(0, dtype=jnp.int32)
        ns0 = jnp.empty((0, int(max_neighbors), 3), dtype=jnp.int32)
        if format == "coo":
            coo_base = (
                jnp.empty((2, 0), dtype=jnp.int32),
                jnp.zeros(1, dtype=jnp.int32),
                jnp.empty((0, 3), dtype=jnp.int32),
            )
            coo_tail: list = []
            if return_distances:
                coo_tail.append(jnp.empty(0, dtype=positions.dtype))
            if return_vectors:
                coo_tail.append(jnp.empty((0, 3), dtype=positions.dtype))
            if pair_fn is not None:
                coo_tail.extend(
                    (
                        jnp.empty(0, dtype=positions.dtype),
                        jnp.empty((0, 3), dtype=positions.dtype),
                    )
                )
            return (*coo_base, *coo_tail)
        if format == "tile":
            empty_i32 = jnp.empty(0, dtype=jnp.int32)
            empty_f32 = jnp.empty(0, dtype=positions.dtype)
            num_systems = int(batch_ptr.shape[0]) - 1
            empty_bptr = jnp.zeros(num_systems + 1, dtype=jnp.int32)
            empty_gptr = jnp.zeros(num_systems + 1, dtype=jnp.int32)
            # 11-tuple matching the non-empty ``"tile"`` branch.
            return (
                jnp.zeros(1, dtype=jnp.int32),  # num_tiles
                empty_i32,  # tile_row_group
                empty_i32,  # tile_col_group
                empty_i32,  # tile_system
                empty_i32,  # sorted_atom_index
                empty_f32,  # sorted_pos_x
                empty_f32,  # sorted_pos_y
                empty_f32,  # sorted_pos_z
                empty_i32,  # batch_idx_sorted
                empty_bptr,  # batch_ptr_padded
                empty_gptr,  # group_ptr
            )
        # format == "matrix"
        if return_distances and return_vectors:
            return (
                nm0,
                nn0,
                ns0,
                jnp.empty((0, int(max_neighbors)), dtype=positions.dtype),
                jnp.empty((0, int(max_neighbors), 3), dtype=positions.dtype),
            )
        if return_distances:
            return (
                nm0,
                nn0,
                ns0,
                jnp.empty((0, int(max_neighbors)), dtype=positions.dtype),
            )
        if return_vectors:
            return (
                nm0,
                nn0,
                ns0,
                jnp.empty((0, int(max_neighbors), 3), dtype=positions.dtype),
            )
        if dual_cutoff:
            matrix_out = (nm0, nn0, ns0, nm0, nn0, ns0)
        else:
            matrix_out = (nm0, nn0, ns0)
        if selective:
            return (
                *matrix_out,
                tile_offsets,
                previous_tile_counts,
                previous_num_tiles,
                previous_tile_row_group,
                previous_tile_col_group,
                previous_tile_system,
            )
        return matrix_out

    if has_pair_outputs:
        if fill_value is None:
            fill_value = N
        batch_idx_arr = _make_batch_idx(batch_ptr).astype(jnp.int32)

        def _forward(p: jax.Array, c: jax.Array) -> _NeighborForwardOutput:
            p_det = jax.lax.stop_gradient(p)
            c_det = jax.lax.stop_gradient(c)
            (
                sai,
                _sort_inv,
                spx,
                spy,
                spz,
                _bis,
                _bpp,
                _gs,
                _gp,
                _gcx,
                _gcy,
                _gcz,
                _gex,
                _gey,
                _gez,
                nt,
                trg,
                tcg,
                ts,
            ) = batch_build_cluster_tile_list(p_det, cutoff, c_det, batch_ptr)
            out = batch_query_cluster_tile(
                sai,
                spx,
                spy,
                spz,
                c_det,
                nt,
                trg,
                tcg,
                ts,
                cutoff,
                p_det.shape[0],
                int(max_neighbors),
                fill_value=int(fill_value),
                return_vectors=True,
                return_distances=True,
                pair_fn=pair_fn,
                pair_params=pair_params,
            )
            if pair_fn is not None:
                nm, nn, shifts, vec, dist, pe, pf = out
            else:
                nm, nn, shifts, vec, dist = out
            i_idx, j_idx, shifts_ret, _, mask_ = _build_index_residuals(
                nm,
                nn,
                shifts,
            )
            K, M = nm.shape
            extra_outputs = (
                (nm, nn, shifts, pe, pf) if pair_fn is not None else (nm, nn, shifts)
            )
            return _NeighborForwardOutput(
                distances=dist,
                vectors=vec,
                extra_outputs=extra_outputs,
                i_idx=i_idx,
                j_idx=j_idx,
                shifts=shifts_ret,
                batch_idx=batch_idx_arr,
                active_mask=mask_,
                matrix_shape=(K, M),
            )

        route_out = _route_pair_outputs(positions, cell_batch, _forward, {})
        if pair_fn is not None:
            (
                distances_out,
                vectors_out,
                nm_out,
                nn_out,
                shifts_out,
                pe_out,
                pf_out,
            ) = route_out
        else:
            distances_out, vectors_out, nm_out, nn_out, shifts_out = route_out
            pe_out = pf_out = None
        if format == "coo":
            # Mirror the batch_cell_list COO contract: convert the matrix topology
            # to a flat neighbor list and repack the per-pair geometry (and pair_fn
            # outputs) into the same COO order.  Eager-only, like the matrix->COO
            # index conversion (the pair count is data-dependent).
            nl, nptr, nl_shifts = get_neighbor_list_from_neighbor_matrix(
                nm_out,
                num_neighbors=nn_out,
                neighbor_shift_matrix=shifts_out,
                fill_value=int(fill_value),
            )
            base = (nl, nptr, nl_shifts)
            active = nm_out != int(fill_value)
            distances_out, vectors_out = coo_pack_pair_geometry(
                active, distances_out, vectors_out
            )
            if pair_fn is not None:
                pe_out, pf_out = coo_pack_pair_geometry(active, pe_out, pf_out)
        else:
            base = (nm_out, nn_out, shifts_out)
        # Return tail mirrors the torch contract: optional distances / vectors,
        # then (pe, pf) when ``pair_fn`` is set.
        tail: list = []
        if return_distances:
            tail.append(distances_out)
        if return_vectors:
            tail.append(vectors_out)
        if pair_fn is not None:
            tail.extend((pe_out, pf_out))
        return (*base, *tail)

    positions_topology = jax.lax.stop_gradient(positions)
    cell_batch_topology = jax.lax.stop_gradient(cell_batch)

    # Tile candidates must cover the larger radius so the cutoff2 matrix cannot
    # miss pairs in the (cutoff, cutoff2] shell; the query filters each matrix
    # by its own cutoff.
    build_cutoff = cutoff if cutoff2 is None else max(float(cutoff), float(cutoff2))
    build_out = batch_build_cluster_tile_list(
        positions_topology,
        build_cutoff,
        cell_batch_topology,
        batch_ptr,
        rebuild_flags=rebuild_flags,
        tile_offsets=tile_offsets,
        tile_counts=previous_tile_counts,
        num_tiles=previous_num_tiles,
        tile_row_group=previous_tile_row_group,
        tile_col_group=previous_tile_col_group,
        tile_system=previous_tile_system,
    )
    if selective:
        (
            sorted_atom_index,
            _sort_inv,
            sorted_pos_x,
            sorted_pos_y,
            sorted_pos_z,
            batch_idx_sorted,
            batch_ptr_padded,
            _group_system,
            group_ptr,
            _group_ctr_x,
            _group_ctr_y,
            _group_ctr_z,
            _group_ext_x,
            _group_ext_y,
            _group_ext_z,
            num_tiles,
            tile_row_group,
            tile_col_group,
            tile_system,
            tile_counts,
        ) = build_out
    else:
        (
            sorted_atom_index,
            _sort_inv,
            sorted_pos_x,
            sorted_pos_y,
            sorted_pos_z,
            batch_idx_sorted,
            batch_ptr_padded,
            _group_system,
            group_ptr,
            _group_ctr_x,
            _group_ctr_y,
            _group_ctr_z,
            _group_ext_x,
            _group_ext_y,
            _group_ext_z,
            num_tiles,
            tile_row_group,
            tile_col_group,
            tile_system,
        ) = build_out
        tile_counts = None

    # Eager guard: raise on tile-buffer overflow instead of silently dropping
    # tiles.  Skipped under trace (the geometry fallback sizes the buffer so it
    # can never overflow).  Compact path checks the global ``num_tiles``;
    # segmented (selective) checks per-system ``tile_counts``.
    if not isinstance(num_tiles, jax.core.Tracer):
        tile_capacity = int(tile_row_group.shape[0])
        if tile_offsets is None:
            n_tiles_host = int(num_tiles[0])
            if n_tiles_host > tile_capacity:
                raise NeighborOverflowError(tile_capacity, n_tiles_host)
        elif tile_counts is not None and not isinstance(tile_counts, jax.core.Tracer):
            counts_host = np.asarray(tile_counts).reshape(-1)
            offs = np.asarray(tile_offsets).reshape(-1)
            seg_caps = offs[1:] - offs[:-1]
            over = np.nonzero(counts_host > seg_caps)[0]
            if over.size > 0:
                isys = int(over[0])
                raise NeighborOverflowError(
                    int(seg_caps[isys]), int(counts_host[isys]), system_index=isys
                )

    if format == "tile":
        # 11-tuple matching the torch sibling at
        # ``nvalchemiops/torch/neighbors/batch_cluster_tile.py:batch_cluster_tile_neighbor_list``
        # so downstream consumers can rely on a single shape across
        # backends.
        return (
            num_tiles,
            tile_row_group,
            tile_col_group,
            tile_system,
            sorted_atom_index,
            sorted_pos_x,
            sorted_pos_y,
            sorted_pos_z,
            batch_idx_sorted,
            batch_ptr_padded,
            group_ptr,
        )

    if format == "coo":
        if selective:
            coo_out = batch_query_cluster_tile_coo(
                sorted_atom_index,
                sorted_pos_x,
                sorted_pos_y,
                sorted_pos_z,
                cell_batch_topology,
                num_tiles,
                tile_row_group,
                tile_col_group,
                tile_system,
                cutoff,
                N,
                int(pair_offsets[-1]),
                rebuild_flags=rebuild_flags,
                tile_offsets=tile_offsets,
                tile_counts=tile_counts,
                pair_offsets=pair_offsets,
                pair_counts=previous_pair_counts,
                neighbor_list=previous_neighbor_list,
                neighbor_list_shifts=previous_neighbor_list_shifts,
            )
            return (
                *coo_out,
                tile_offsets,
                tile_counts,
                num_tiles,
                tile_row_group,
                tile_col_group,
                tile_system,
            )
        if max_pairs is None:
            max_pairs = N * max_neighbors
        return batch_query_cluster_tile_coo(
            sorted_atom_index,
            sorted_pos_x,
            sorted_pos_y,
            sorted_pos_z,
            cell_batch_topology,
            num_tiles,
            tile_row_group,
            tile_col_group,
            tile_system,
            cutoff,
            N,
            int(max_pairs),
        )

    matrix_out = batch_query_cluster_tile(
        sorted_atom_index,
        sorted_pos_x,
        sorted_pos_y,
        sorted_pos_z,
        cell_batch_topology,
        num_tiles,
        tile_row_group,
        tile_col_group,
        tile_system,
        cutoff,
        N,
        int(max_neighbors),
        fill_value=fill_value,
        cutoff2=cutoff2,
        rebuild_flags=rebuild_flags,
        tile_offsets=tile_offsets,
        tile_counts=tile_counts,
        batch_idx=_make_batch_idx(batch_ptr).astype(jnp.int32) if selective else None,
        neighbor_matrix=previous_neighbor_matrix,
        num_neighbors=previous_num_neighbors,
        neighbor_matrix_shifts=previous_neighbor_matrix_shifts,
        neighbor_matrix2=previous_neighbor_matrix2,
        num_neighbors2=previous_num_neighbors2,
        neighbor_matrix_shifts2=previous_neighbor_matrix_shifts2,
    )
    if selective:
        return (
            *matrix_out,
            tile_offsets,
            tile_counts,
            num_tiles,
            tile_row_group,
            tile_col_group,
            tile_system,
        )
    return matrix_out
