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

"""JAX bindings for the single-system cluster-pair tile neighbor list.

Mirrors :mod:`nvalchemiops.torch.neighbors.cluster_tile`: the Morton sort +
SoA gather happens in JAX, then the bbox reduction + tile-pair enumeration
and the final conversion (matrix / COO) run on the Warp side via
``jax_callable`` callbacks.  Tile-format output returns the raw cluster
state for consumers that want to plug into shared-memory tile loads
directly.

Scope: single system, triclinic-safe, float32 only, ``N >= 0`` (any
non-32-aligned ``N`` is padded internally).
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import warp as wp
from warp.jax_experimental import GraphMode, jax_callable

from nvalchemiops.jax.neighbors._autograd import (
    _build_index_residuals,
    _NeighborForwardOutput,
    _route_pair_outputs,
)
from nvalchemiops.jax.neighbors.neighbor_utils import (
    coo_pack_pair_geometry,
    get_neighbor_list_from_neighbor_matrix,
)
from nvalchemiops.neighbors.cluster_tile import (
    TILE_GROUP_SIZE,
)
from nvalchemiops.neighbors.cluster_tile import (
    build_cluster_tile_list as _warp_build_cluster_tile_list,
)
from nvalchemiops.neighbors.cluster_tile import (
    estimate_max_tiles_per_group as _estimate_max_tiles_per_group,
)
from nvalchemiops.neighbors.cluster_tile import (
    query_cluster_tile as _warp_query_cluster_tile,
)
from nvalchemiops.neighbors.cluster_tile import (
    query_cluster_tile_coo as _warp_query_cluster_tile_coo,
)
from nvalchemiops.neighbors.neighbor_utils import (
    NeighborOverflowError,
    estimate_max_neighbors,
)

__all__ = [
    "TILE_GROUP_SIZE",
    "allocate_cluster_tile_list",
    "build_cluster_tile_list",
    "estimate_cluster_tile_list_sizes",
    "cluster_tile_neighbor_list",
    "query_cluster_tile_coo",
    "query_cluster_tile",
]


# =============================================================================
# Sizing + allocation helpers (pure JAX, no Warp launches)
# =============================================================================
def _concrete_cell_volume(cell) -> float | None:
    """Return ``abs(det(cell))`` if ``cell`` is concrete, else ``None``.

    Under ``jit``/``grad`` the cell may be a tracer; ``np.asarray`` raises and
    we return ``None`` so the caller falls back to a trace-safe static size.
    """
    try:
        arr = np.asarray(cell)
    except Exception:
        return None
    try:
        arr = arr.reshape(-1, 3, 3)[0] if arr.ndim == 3 else arr.reshape(3, 3)
        return float(abs(np.linalg.det(arr)))
    except Exception:
        return None


def _tile_buffer_max_tiles_per_group(positions, total_atoms: int, cutoff, cell) -> int:
    """``max_tiles_per_group`` for the JAX tile buffer.

    The tile-buffer size must be a *static* Python int (it's an array shape).
    When ``positions`` and ``cell`` are concrete (eager, or ``grad`` w.r.t.
    positions where the cell is a closure constant) we density-size from the
    geometry so dense/high-cutoff systems don't silently overflow.  When either
    is traced (e.g. ``grad`` w.r.t. cell, or ``jit`` of the whole call) the
    volume is unavailable, so fall back to ``ngroup`` -> capacity ``ngroup**2``,
    the upper-triangular maximum, which can *never* overflow (at the cost of
    more memory for very large traced systems).
    """
    ngroup = (int(total_atoms) + TILE_GROUP_SIZE - 1) // TILE_GROUP_SIZE
    if ngroup <= 1:
        return 1
    cutoff_concrete = not isinstance(cutoff, jax.core.Tracer)
    vol = _concrete_cell_volume(cell)
    if isinstance(positions, jax.core.Tracer) or vol is None or not cutoff_concrete:
        return ngroup  # trace-safe upper-triangular cap (capacity ngroup**2)
    return _estimate_max_tiles_per_group(int(total_atoms), float(cutoff), vol)


def estimate_cluster_tile_list_sizes(
    total_atoms: int,
    max_tiles_per_group: int = 256,
) -> tuple[int, int, int, int]:
    """Estimate allocation sizes for the tile neighbor list state.

    Mirrors :func:`nvalchemiops.torch.neighbors.cluster_tile.estimate_cluster_tile_list_sizes`.

    Parameters
    ----------
    total_atoms : int
        Real atom count.  Any ``total_atoms >= 0`` is accepted.
    max_tiles_per_group : int, default 256
        Upper bound on neighbor groups per row_group.

    Returns
    -------
    n_padded : int
        Padded atom count = ``ceil(total_atoms / TILE_GROUP_SIZE) * TILE_GROUP_SIZE``.
    ngroup : int
        ``n_padded // TILE_GROUP_SIZE``.
    ngroup_padded : int
        Group-array pad length, multiple of TILE_GROUP_SIZE.
    max_tiles : int
        Upper bound on the tile-pair list size.
    """
    if total_atoms < 0:
        raise ValueError(f"total_atoms must be >= 0; got {total_atoms}")
    n_padded = (
        (total_atoms + TILE_GROUP_SIZE - 1) // TILE_GROUP_SIZE
    ) * TILE_GROUP_SIZE
    if n_padded == 0:
        n_padded = TILE_GROUP_SIZE
    ngroup = n_padded // TILE_GROUP_SIZE
    ngroup_padded = (
        (ngroup + TILE_GROUP_SIZE - 1) // TILE_GROUP_SIZE
    ) * TILE_GROUP_SIZE
    if ngroup_padded == ngroup:
        ngroup_padded = ngroup + TILE_GROUP_SIZE
    max_tiles = ngroup * min(ngroup, max_tiles_per_group)
    return n_padded, ngroup, ngroup_padded, max_tiles


def allocate_cluster_tile_list(
    total_atoms: int,
    dtype: jnp.dtype = jnp.float32,
    max_tiles_per_group: int = 256,
) -> tuple[jax.Array, ...]:
    """Allocate the state tensors consumed by :func:`build_cluster_tile_list`.

    Returns ``(sorted_atom_index, morton_codes, sorted_pos_x, sorted_pos_y,
    sorted_pos_z, group_ctr_x, group_ctr_y, group_ctr_z, group_ext_x,
    group_ext_y, group_ext_z, num_tiles, tile_row_group, tile_col_group)``.
    """
    n_padded, ngroup, ngroup_padded, max_tiles = estimate_cluster_tile_list_sizes(
        total_atoms,
        max_tiles_per_group=max_tiles_per_group,
    )
    return (
        jnp.zeros(n_padded, dtype=jnp.int32),  # sorted_atom_index
        jnp.zeros(n_padded, dtype=jnp.int32),  # morton_codes
        jnp.zeros(n_padded, dtype=dtype),  # sorted_pos_x
        jnp.zeros(n_padded, dtype=dtype),  # sorted_pos_y
        jnp.zeros(n_padded, dtype=dtype),  # sorted_pos_z
        jnp.zeros(ngroup_padded, dtype=dtype),  # group_ctr_x
        jnp.zeros(ngroup_padded, dtype=dtype),  # group_ctr_y
        jnp.zeros(ngroup_padded, dtype=dtype),  # group_ctr_z
        jnp.zeros(ngroup_padded, dtype=dtype),  # group_ext_x
        jnp.zeros(ngroup_padded, dtype=dtype),  # group_ext_y
        jnp.zeros(ngroup_padded, dtype=dtype),  # group_ext_z
        jnp.zeros(1, dtype=jnp.int32),  # num_tiles
        jnp.zeros(max_tiles, dtype=jnp.int32),  # tile_row_group
        jnp.zeros(max_tiles, dtype=jnp.int32),  # tile_col_group
    )


# =============================================================================
# Morton code helpers (JAX, mirror the torch path)
# =============================================================================
def _spread_10bit(x: jax.Array) -> jax.Array:
    """Spread the low 10 bits of ``x`` across a 30-bit Morton code."""
    x = x & 0x3FF
    x = (x | (x << 16)) & 0x030000FF
    x = (x | (x << 8)) & 0x0300F00F
    x = (x | (x << 4)) & 0x030C30C3
    x = (x | (x << 2)) & 0x09249249
    return x


def _compute_morton_codes(
    positions_padded: jax.Array,
    inv_cell: jax.Array,
    natom: int,
    n_padded: int,
) -> jax.Array:
    """Compute 30-bit Morton codes for atoms, sentinel-pad the tail.

    Pads positions are assumed to copy a real atom (in-cell coordinate);
    after Morton coding their 30-bit code is OR-ed with ``0x40000000`` so
    they sort after every real-atom code.
    """
    inv_cell_2d = inv_cell[0] if inv_cell.ndim == 3 else inv_cell
    frac = positions_padded @ inv_cell_2d.T
    frac = frac - jnp.floor(frac)
    bucket = jnp.clip((frac * 1024.0).astype(jnp.int32), 0, 1023)
    ix, iy, iz = bucket[:, 0], bucket[:, 1], bucket[:, 2]
    codes = (_spread_10bit(iz) << 2) | (_spread_10bit(iy) << 1) | _spread_10bit(ix)
    if natom < n_padded:
        sentinel = jnp.full((n_padded - natom,), 0x40000000, dtype=jnp.int32)
        codes = jnp.concatenate([codes[:natom], sentinel])
    return codes


# =============================================================================
# jax_callable wrappers (Warp launchers behind GraphMode.WARP callbacks)
# =============================================================================
def _build_cluster_tile_list_callback(
    sorted_pos_x: wp.array(dtype=wp.float32),
    sorted_pos_y: wp.array(dtype=wp.float32),
    sorted_pos_z: wp.array(dtype=wp.float32),
    cell: wp.array(dtype=wp.mat33f),
    inv_cell: wp.array(dtype=wp.mat33f),
    group_ctr_x: wp.array(dtype=wp.float32),
    group_ctr_y: wp.array(dtype=wp.float32),
    group_ctr_z: wp.array(dtype=wp.float32),
    group_ext_x: wp.array(dtype=wp.float32),
    group_ext_y: wp.array(dtype=wp.float32),
    group_ext_z: wp.array(dtype=wp.float32),
    num_tiles: wp.array(dtype=wp.int32),
    tile_row_group: wp.array(dtype=wp.int32),
    tile_col_group: wp.array(dtype=wp.int32),
    cutoff: wp.float32,
) -> None:
    """jax_callable callback for the SS tile bbox + tile-pair enumeration."""
    _warp_build_cluster_tile_list(
        sorted_pos_x=sorted_pos_x,
        sorted_pos_y=sorted_pos_y,
        sorted_pos_z=sorted_pos_z,
        cell=cell,
        inv_cell=inv_cell,
        cutoff=float(cutoff),
        num_tiles=num_tiles,
        tile_row_group=tile_row_group,
        tile_col_group=tile_col_group,
        wp_dtype=wp.float32,
        device=str(sorted_pos_x.device),
        group_ctr_x_buffer=group_ctr_x,
        group_ctr_y_buffer=group_ctr_y,
        group_ctr_z_buffer=group_ctr_z,
        group_ext_x_buffer=group_ext_x,
        group_ext_y_buffer=group_ext_y,
        group_ext_z_buffer=group_ext_z,
    )


def _build_cluster_tile_list_selective_callback(
    sorted_pos_x: wp.array(dtype=wp.float32),
    sorted_pos_y: wp.array(dtype=wp.float32),
    sorted_pos_z: wp.array(dtype=wp.float32),
    cell: wp.array(dtype=wp.mat33f),
    inv_cell: wp.array(dtype=wp.mat33f),
    group_ctr_x: wp.array(dtype=wp.float32),
    group_ctr_y: wp.array(dtype=wp.float32),
    group_ctr_z: wp.array(dtype=wp.float32),
    group_ext_x: wp.array(dtype=wp.float32),
    group_ext_y: wp.array(dtype=wp.float32),
    group_ext_z: wp.array(dtype=wp.float32),
    num_tiles: wp.array(dtype=wp.int32),
    tile_row_group: wp.array(dtype=wp.int32),
    tile_col_group: wp.array(dtype=wp.int32),
    rebuild_flags: wp.array(dtype=wp.bool),
    cutoff: wp.float32,
) -> None:
    """jax_callable callback for selective SS tile enumeration."""
    _warp_build_cluster_tile_list(
        sorted_pos_x=sorted_pos_x,
        sorted_pos_y=sorted_pos_y,
        sorted_pos_z=sorted_pos_z,
        cell=cell,
        inv_cell=inv_cell,
        cutoff=float(cutoff),
        num_tiles=num_tiles,
        tile_row_group=tile_row_group,
        tile_col_group=tile_col_group,
        wp_dtype=wp.float32,
        device=str(sorted_pos_x.device),
        group_ctr_x_buffer=group_ctr_x,
        group_ctr_y_buffer=group_ctr_y,
        group_ctr_z_buffer=group_ctr_z,
        group_ext_x_buffer=group_ext_x,
        group_ext_y_buffer=group_ext_y,
        group_ext_z_buffer=group_ext_z,
        rebuild_flags=rebuild_flags,
    )


def _query_cluster_tile_callback(
    sorted_atom_index: wp.array(dtype=wp.int32),
    sorted_pos_x: wp.array(dtype=wp.float32),
    sorted_pos_y: wp.array(dtype=wp.float32),
    sorted_pos_z: wp.array(dtype=wp.float32),
    num_tiles: wp.array(dtype=wp.int32),
    tile_row_group: wp.array(dtype=wp.int32),
    tile_col_group: wp.array(dtype=wp.int32),
    cell: wp.array(dtype=wp.mat33f),
    inv_cell: wp.array(dtype=wp.mat33f),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    neighbor_matrix_shifts: wp.array(dtype=wp.int32, ndim=3),
    cutoff: wp.float32,
    natom: wp.int32,
) -> None:
    """jax_callable callback for the tile-pair → matrix conversion kernel.

    No ``n_tiles`` scalar — the warp launcher launches at the allocated
    ``tile_row_group`` capacity and guards per-tile via the device-side
    ``num_tiles`` array.
    """
    _warp_query_cluster_tile(
        sorted_atom_index=sorted_atom_index,
        sorted_pos_x=sorted_pos_x,
        sorted_pos_y=sorted_pos_y,
        sorted_pos_z=sorted_pos_z,
        num_tiles=num_tiles,
        tile_row_group=tile_row_group,
        tile_col_group=tile_col_group,
        cell=cell,
        inv_cell=inv_cell,
        cutoff=float(cutoff),
        natom=int(natom),
        neighbor_matrix=neighbor_matrix,
        num_neighbors=num_neighbors,
        neighbor_matrix_shifts=neighbor_matrix_shifts,
        wp_dtype=wp.float32,
        device=str(sorted_pos_x.device),
    )


def _query_cluster_tile_selective_callback(
    sorted_atom_index: wp.array(dtype=wp.int32),
    sorted_pos_x: wp.array(dtype=wp.float32),
    sorted_pos_y: wp.array(dtype=wp.float32),
    sorted_pos_z: wp.array(dtype=wp.float32),
    num_tiles: wp.array(dtype=wp.int32),
    tile_row_group: wp.array(dtype=wp.int32),
    tile_col_group: wp.array(dtype=wp.int32),
    cell: wp.array(dtype=wp.mat33f),
    inv_cell: wp.array(dtype=wp.mat33f),
    rebuild_flags: wp.array(dtype=wp.bool),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    neighbor_matrix_shifts: wp.array(dtype=wp.int32, ndim=3),
    cutoff: wp.float32,
    natom: wp.int32,
) -> None:
    """jax_callable callback for selective tile-pair → matrix conversion."""
    _warp_query_cluster_tile(
        sorted_atom_index=sorted_atom_index,
        sorted_pos_x=sorted_pos_x,
        sorted_pos_y=sorted_pos_y,
        sorted_pos_z=sorted_pos_z,
        num_tiles=num_tiles,
        tile_row_group=tile_row_group,
        tile_col_group=tile_col_group,
        cell=cell,
        inv_cell=inv_cell,
        cutoff=float(cutoff),
        natom=int(natom),
        neighbor_matrix=neighbor_matrix,
        num_neighbors=num_neighbors,
        neighbor_matrix_shifts=neighbor_matrix_shifts,
        wp_dtype=wp.float32,
        device=str(sorted_pos_x.device),
        rebuild_flags=rebuild_flags,
    )


def _query_cluster_tile_dual_callback(
    sorted_atom_index: wp.array(dtype=wp.int32),
    sorted_pos_x: wp.array(dtype=wp.float32),
    sorted_pos_y: wp.array(dtype=wp.float32),
    sorted_pos_z: wp.array(dtype=wp.float32),
    num_tiles: wp.array(dtype=wp.int32),
    tile_row_group: wp.array(dtype=wp.int32),
    tile_col_group: wp.array(dtype=wp.int32),
    cell: wp.array(dtype=wp.mat33f),
    inv_cell: wp.array(dtype=wp.mat33f),
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
    """jax_callable callback for dual-cutoff tile-pair → matrix conversion."""
    _warp_query_cluster_tile(
        sorted_atom_index=sorted_atom_index,
        sorted_pos_x=sorted_pos_x,
        sorted_pos_y=sorted_pos_y,
        sorted_pos_z=sorted_pos_z,
        num_tiles=num_tiles,
        tile_row_group=tile_row_group,
        tile_col_group=tile_col_group,
        cell=cell,
        inv_cell=inv_cell,
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


def _query_cluster_tile_dual_selective_callback(
    sorted_atom_index: wp.array(dtype=wp.int32),
    sorted_pos_x: wp.array(dtype=wp.float32),
    sorted_pos_y: wp.array(dtype=wp.float32),
    sorted_pos_z: wp.array(dtype=wp.float32),
    num_tiles: wp.array(dtype=wp.int32),
    tile_row_group: wp.array(dtype=wp.int32),
    tile_col_group: wp.array(dtype=wp.int32),
    cell: wp.array(dtype=wp.mat33f),
    inv_cell: wp.array(dtype=wp.mat33f),
    rebuild_flags: wp.array(dtype=wp.bool),
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
    """jax_callable callback for selective dual-cutoff matrix conversion."""
    _warp_query_cluster_tile(
        sorted_atom_index=sorted_atom_index,
        sorted_pos_x=sorted_pos_x,
        sorted_pos_y=sorted_pos_y,
        sorted_pos_z=sorted_pos_z,
        num_tiles=num_tiles,
        tile_row_group=tile_row_group,
        tile_col_group=tile_col_group,
        cell=cell,
        inv_cell=inv_cell,
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
    )


def _query_cluster_tile_pair_callback(
    sorted_atom_index: wp.array(dtype=wp.int32),
    sorted_pos_x: wp.array(dtype=wp.float32),
    sorted_pos_y: wp.array(dtype=wp.float32),
    sorted_pos_z: wp.array(dtype=wp.float32),
    num_tiles: wp.array(dtype=wp.int32),
    tile_row_group: wp.array(dtype=wp.int32),
    tile_col_group: wp.array(dtype=wp.int32),
    cell: wp.array(dtype=wp.mat33f),
    inv_cell: wp.array(dtype=wp.mat33f),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    neighbor_matrix_shifts: wp.array(dtype=wp.int32, ndim=3),
    neighbor_vectors: wp.array(dtype=wp.vec3f, ndim=2),
    neighbor_distances: wp.array(dtype=wp.float32, ndim=2),
    cutoff: wp.float32,
    natom: wp.int32,
) -> None:
    """jax_callable callback for the pair-output tile-pair → matrix kernel.

    Same as :func:`_query_cluster_tile_callback` but with additional
    ``neighbor_vectors`` / ``neighbor_distances`` in/out arrays that the
    underlying Warp launcher fills with per-pair displacements and scalar
    distances when ``return_vectors`` / ``return_distances`` are enabled.

    Dtype contract: ``neighbor_vectors`` is typed
    ``wp.array(dtype=wp.vec3f, ndim=2)`` so that the JAX-side ``(N, M, 3)``
    float32 buffer maps to a ``(N, M)`` array of ``vec3f``.  This matches
    the convention used by every other neighbor-list kernel in this
    package (see ``neighbor_vectors: wp.array(dtype=vec_dtype, ndim=2)``
    in :mod:`nvalchemiops.neighbors.naive.kernels` and the cell_list
    kernels).
    """
    _warp_query_cluster_tile(
        sorted_atom_index=sorted_atom_index,
        sorted_pos_x=sorted_pos_x,
        sorted_pos_y=sorted_pos_y,
        sorted_pos_z=sorted_pos_z,
        num_tiles=num_tiles,
        tile_row_group=tile_row_group,
        tile_col_group=tile_col_group,
        cell=cell,
        inv_cell=inv_cell,
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


def _query_cluster_tile_coo_callback(
    sorted_atom_index: wp.array(dtype=wp.int32),
    sorted_pos_x: wp.array(dtype=wp.float32),
    sorted_pos_y: wp.array(dtype=wp.float32),
    sorted_pos_z: wp.array(dtype=wp.float32),
    num_tiles: wp.array(dtype=wp.int32),
    tile_row_group: wp.array(dtype=wp.int32),
    tile_col_group: wp.array(dtype=wp.int32),
    cell: wp.array(dtype=wp.mat33f),
    inv_cell: wp.array(dtype=wp.mat33f),
    pair_counter: wp.array(dtype=wp.int32),
    coo_list: wp.array(dtype=wp.int32, ndim=2),
    coo_shifts: wp.array(dtype=wp.int32, ndim=2),
    cutoff: wp.float32,
    natom: wp.int32,
    max_pairs: wp.int32,
) -> None:
    """jax_callable callback for the tile-pair → flat COO conversion kernel.

    No ``n_tiles`` scalar — the warp launcher launches at the allocated
    ``tile_row_group`` capacity and guards per-tile via the device-side
    ``num_tiles`` array.
    """
    _warp_query_cluster_tile_coo(
        sorted_atom_index=sorted_atom_index,
        sorted_pos_x=sorted_pos_x,
        sorted_pos_y=sorted_pos_y,
        sorted_pos_z=sorted_pos_z,
        num_tiles=num_tiles,
        tile_row_group=tile_row_group,
        tile_col_group=tile_col_group,
        cell=cell,
        inv_cell=inv_cell,
        cutoff=float(cutoff),
        natom=int(natom),
        max_pairs=int(max_pairs),
        pair_counter=pair_counter,
        coo_list=coo_list,
        coo_shifts=coo_shifts,
        wp_dtype=wp.float32,
        device=str(sorted_pos_x.device),
    )


def _query_cluster_tile_coo_segmented_callback(
    sorted_atom_index: wp.array(dtype=wp.int32),
    sorted_pos_x: wp.array(dtype=wp.float32),
    sorted_pos_y: wp.array(dtype=wp.float32),
    sorted_pos_z: wp.array(dtype=wp.float32),
    num_tiles: wp.array(dtype=wp.int32),
    tile_row_group: wp.array(dtype=wp.int32),
    tile_col_group: wp.array(dtype=wp.int32),
    cell: wp.array(dtype=wp.mat33f),
    inv_cell: wp.array(dtype=wp.mat33f),
    rebuild_flags: wp.array(dtype=wp.bool),
    pair_counter: wp.array(dtype=wp.int32),
    pair_offsets: wp.array(dtype=wp.int32),
    pair_counts: wp.array(dtype=wp.int32),
    coo_list: wp.array(dtype=wp.int32, ndim=2),
    coo_shifts: wp.array(dtype=wp.int32, ndim=2),
    cutoff: wp.float32,
    natom: wp.int32,
    max_pairs: wp.int32,
) -> None:
    """jax_callable callback for fixed-segment selective COO conversion."""
    _warp_query_cluster_tile_coo(
        sorted_atom_index=sorted_atom_index,
        sorted_pos_x=sorted_pos_x,
        sorted_pos_y=sorted_pos_y,
        sorted_pos_z=sorted_pos_z,
        num_tiles=num_tiles,
        tile_row_group=tile_row_group,
        tile_col_group=tile_col_group,
        cell=cell,
        inv_cell=inv_cell,
        cutoff=float(cutoff),
        natom=int(natom),
        max_pairs=int(max_pairs),
        pair_counter=pair_counter,
        coo_list=coo_list,
        coo_shifts=coo_shifts,
        wp_dtype=wp.float32,
        device=str(sorted_pos_x.device),
        rebuild_flags=rebuild_flags,
        pair_offsets=pair_offsets,
        pair_counts=pair_counts,
    )


_jax_build_cluster_tile_list = jax_callable(
    _build_cluster_tile_list_callback,
    num_outputs=9,
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
    ],
    graph_mode=GraphMode.WARP,
)

_jax_build_cluster_tile_list_selective = jax_callable(
    _build_cluster_tile_list_selective_callback,
    num_outputs=9,
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
    ],
    graph_mode=GraphMode.WARP,
)


_jax_query_cluster_tile = jax_callable(
    _query_cluster_tile_callback,
    num_outputs=3,
    in_out_argnames=["neighbor_matrix", "num_neighbors", "neighbor_matrix_shifts"],
    graph_mode=GraphMode.WARP,
)

_jax_query_cluster_tile_selective = jax_callable(
    _query_cluster_tile_selective_callback,
    num_outputs=3,
    in_out_argnames=["neighbor_matrix", "num_neighbors", "neighbor_matrix_shifts"],
    graph_mode=GraphMode.WARP,
)

_jax_query_cluster_tile_dual = jax_callable(
    _query_cluster_tile_dual_callback,
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

_jax_query_cluster_tile_dual_selective = jax_callable(
    _query_cluster_tile_dual_selective_callback,
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


_jax_query_cluster_tile_pair = jax_callable(
    _query_cluster_tile_pair_callback,
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
def _get_jax_cluster_tile_pair_fn_callable(pair_fn):
    """Build (and cache) a ``jax_callable`` that closes over ``pair_fn`` for the
    cluster-tile pair-output → matrix kernel.

    Same as ``_jax_query_cluster_tile_pair`` but the callback closes over
    ``pair_fn`` (which cannot cross the JAX trace boundary as data) and adds the
    ``pair_params`` input + ``pair_energies`` / ``pair_forces`` outputs.  Cached by
    ``pair_fn`` identity; one recompile per distinct ``pair_fn``.  fp32-only, like
    the rest of the cluster-tile JAX binding.
    """

    def _callback(
        sorted_atom_index: wp.array(dtype=wp.int32),
        sorted_pos_x: wp.array(dtype=wp.float32),
        sorted_pos_y: wp.array(dtype=wp.float32),
        sorted_pos_z: wp.array(dtype=wp.float32),
        num_tiles: wp.array(dtype=wp.int32),
        tile_row_group: wp.array(dtype=wp.int32),
        tile_col_group: wp.array(dtype=wp.int32),
        cell: wp.array(dtype=wp.mat33f),
        inv_cell: wp.array(dtype=wp.mat33f),
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
        _warp_query_cluster_tile(
            sorted_atom_index=sorted_atom_index,
            sorted_pos_x=sorted_pos_x,
            sorted_pos_y=sorted_pos_y,
            sorted_pos_z=sorted_pos_z,
            num_tiles=num_tiles,
            tile_row_group=tile_row_group,
            tile_col_group=tile_col_group,
            cell=cell,
            inv_cell=inv_cell,
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


_jax_query_cluster_tile_coo = jax_callable(
    _query_cluster_tile_coo_callback,
    num_outputs=3,
    in_out_argnames=["pair_counter", "coo_list", "coo_shifts"],
    graph_mode=GraphMode.WARP,
)

_jax_query_cluster_tile_coo_segmented = jax_callable(
    _query_cluster_tile_coo_segmented_callback,
    num_outputs=4,
    in_out_argnames=["pair_counter", "pair_counts", "coo_list", "coo_shifts"],
    graph_mode=GraphMode.WARP,
)


# =============================================================================
# User-facing JAX entry points
# =============================================================================
def _normalize_cell(cell: jax.Array, dtype: jnp.dtype) -> jax.Array:
    """Coerce ``(3, 3)`` or ``(1, 3, 3)`` cell to ``(1, 3, 3)`` with given dtype."""
    if cell.ndim == 2:
        cell = cell[jnp.newaxis, :, :]
    if cell.shape != (1, 3, 3):
        raise ValueError(f"cell must have shape (3, 3) or (1, 3, 3); got {cell.shape}")
    return cell.astype(dtype)


def _morton_sort_and_gather(
    positions: jax.Array,
    cell: jax.Array,
    n_padded: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Pad positions to ``n_padded``, compute Morton codes, argsort, gather SoA.

    Returns ``(sorted_atom_index, morton_codes, sorted_pos_x, sorted_pos_y,
    sorted_pos_z)``.
    """
    N = positions.shape[0]
    inv_cell = jnp.linalg.inv(cell[0])
    if N < n_padded:
        # Pad with the last real atom (any in-cell point works).
        pad = jnp.broadcast_to(
            positions[-1:],
            (n_padded - N, 3),
        )
        padded_positions = jnp.concatenate([positions, pad], axis=0)
    else:
        padded_positions = positions
    morton_codes = _compute_morton_codes(padded_positions, inv_cell, N, n_padded)
    perm = jnp.argsort(morton_codes).astype(jnp.int32)
    sorted_pos = padded_positions[perm]
    return (
        perm,
        morton_codes,
        jnp.asarray(sorted_pos[:, 0]),
        jnp.asarray(sorted_pos[:, 1]),
        jnp.asarray(sorted_pos[:, 2]),
    )


def build_cluster_tile_list(
    positions: jax.Array,
    cutoff: float,
    cell: jax.Array,
    *,
    rebuild_flags: jax.Array | None = None,
    num_tiles: jax.Array | None = None,
    tile_row_group: jax.Array | None = None,
    tile_col_group: jax.Array | None = None,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Build the tile neighbor list state in pre-allocated form.

    Runs Morton sort + SoA gather in JAX, then the bbox reduction + tile
    enumeration on the Warp side via a ``jax_callable`` callback.

    Parameters
    ----------
    positions : jax.Array, shape (N, 3), dtype=float32
        Atomic coordinates.  Any ``N >= 0``; non-32-aligned ``N`` is padded
        internally.
    cutoff : float
        Cutoff distance for the bbox filter.
    cell : jax.Array, shape (3, 3) or (1, 3, 3), dtype=float32
        Unit cell matrix (orthorhombic or triclinic).

    Returns
    -------
    tuple of jax.Array
        ``(sorted_atom_index, morton_codes, sorted_pos_x, sorted_pos_y,
        sorted_pos_z, group_ctr_x, group_ctr_y, group_ctr_z, group_ext_x,
        group_ext_y, group_ext_z, num_tiles, tile_row_group,
        tile_col_group)``.

    Notes
    -----
    Float32 only.  The cluster-pair tile kernels currently only support
    ``wp.float32``.
    """
    if positions.dtype != jnp.float32:
        raise TypeError(
            "positions must be float32 (cluster_tile kernels are float32 only)"
        )
    N = positions.shape[0]
    # Geometry-size the tile buffer so dense/high-cutoff systems don't silently
    # overflow; trace-safe ``ngroup**2`` fallback when positions/cell are tracers.
    max_tiles_per_group = _tile_buffer_max_tiles_per_group(positions, N, cutoff, cell)
    n_padded, ngroup, ngroup_padded, max_tiles = estimate_cluster_tile_list_sizes(
        N, max_tiles_per_group=max_tiles_per_group
    )

    cell_n = _normalize_cell(cell, jnp.float32)
    inv_cell_n = jnp.linalg.inv(cell_n[0])[jnp.newaxis, :, :]

    sorted_atom_index, morton_codes, sorted_pos_x, sorted_pos_y, sorted_pos_z = (
        _morton_sort_and_gather(positions, cell_n, n_padded)
    )

    # Allocate the bbox + tile output buffers.
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
        ) = _jax_build_cluster_tile_list(
            sorted_pos_x,
            sorted_pos_y,
            sorted_pos_z,
            cell_n,
            inv_cell_n,
            group_ctr_x,
            group_ctr_y,
            group_ctr_z,
            group_ext_x,
            group_ext_y,
            group_ext_z,
            num_tiles,
            tile_row_group,
            tile_col_group,
            float(cutoff),
        )
    else:
        rf = rebuild_flags.flatten()[:1].astype(jnp.bool_)
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
        ) = _jax_build_cluster_tile_list_selective(
            sorted_pos_x,
            sorted_pos_y,
            sorted_pos_z,
            cell_n,
            inv_cell_n,
            group_ctr_x,
            group_ctr_y,
            group_ctr_z,
            group_ext_x,
            group_ext_y,
            group_ext_z,
            num_tiles,
            tile_row_group,
            tile_col_group,
            rf,
            float(cutoff),
        )

    del ngroup  # implicit in group_*_x.shape[0]
    return (
        sorted_atom_index,
        morton_codes,
        sorted_pos_x,
        sorted_pos_y,
        sorted_pos_z,
        group_ctr_x,
        group_ctr_y,
        group_ctr_z,
        group_ext_x,
        group_ext_y,
        group_ext_z,
        num_tiles,
        tile_row_group,
        tile_col_group,
    )


def query_cluster_tile(
    sorted_atom_index: jax.Array,
    sorted_pos_x: jax.Array,
    sorted_pos_y: jax.Array,
    sorted_pos_z: jax.Array,
    num_tiles: jax.Array,
    tile_row_group: jax.Array,
    tile_col_group: jax.Array,
    cell: jax.Array,
    cutoff: float,
    natom: int,
    max_neighbors: int,
    *,
    fill_value: int | None = None,
    cutoff2: float | None = None,
    neighbor_matrix: jax.Array | None = None,
    num_neighbors: jax.Array | None = None,
    neighbor_matrix_shifts: jax.Array | None = None,
    neighbor_matrix2: jax.Array | None = None,
    num_neighbors2: jax.Array | None = None,
    neighbor_matrix_shifts2: jax.Array | None = None,
    rebuild_flags: jax.Array | None = None,
    return_vectors: bool = False,
    return_distances: bool = False,
    pair_fn: wp.Function | None = None,
    pair_params: jax.Array | None = None,
    neighbor_vectors: jax.Array | None = None,
    neighbor_distances: jax.Array | None = None,
    pair_energies: jax.Array | None = None,
    pair_forces: jax.Array | None = None,
) -> tuple[jax.Array, ...]:
    """Convert the tile pair list to dense neighbor-matrix form.

    Returns ``(neighbor_matrix, num_neighbors, neighbor_matrix_shifts)``.
    Skip-prefill: the warp kernel only writes the active entries, then a
    JAX-side ``jnp.where`` fills the unused tail columns with
    ``fill_value`` (defaults to ``natom``).

    Cluster-tile does not support partial neighbor lists; there is no
    ``target_indices`` kwarg.  Pair-output kwargs (``return_vectors`` /
    ``return_distances`` / ``pair_fn`` and associated buffers) are
    rejected with ``NotImplementedError`` when set because the JAX
    ``jax_callable`` pathway here cannot carry a
    callable ``pair_fn`` across the trace boundary.  Callers needing
    these axes should use the torch binding
    (:func:`nvalchemiops.torch.neighbors.cluster_tile.query_cluster_tile`)
    or call
    :func:`nvalchemiops.neighbors.cluster_tile.query_cluster_tile` directly
    on Warp arrays.
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
            "JAX cluster_tile cutoff2 / rebuild_flags are matrix-topology "
            "features in this pass and cannot be combined with "
            "return_distances or return_vectors.",
        )
    if fill_value is None:
        fill_value = natom
    cell_n = _normalize_cell(cell, jnp.float32)
    inv_cell_n = jnp.linalg.inv(cell_n[0])[jnp.newaxis, :, :]

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

    rf = None
    if selective:
        rf = rebuild_flags.flatten()[:1].astype(jnp.bool_)
        num_neighbors = jnp.where(rf[0], jnp.zeros_like(num_neighbors), num_neighbors)
        if dual_cutoff and num_neighbors2 is not None:
            num_neighbors2 = jnp.where(
                rf[0], jnp.zeros_like(num_neighbors2), num_neighbors2
            )

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

    # No host-side ``int(num_tiles[0])`` read: the warp kernel launches
    # at the allocated tile-buffer capacity and early-returns per-tile
    # using the device-side ``num_tiles`` array.
    if has_pair_outputs:
        if neighbor_vectors is None:
            neighbor_vectors = jnp.zeros((natom, max_neighbors, 3), dtype=jnp.float32)
        if neighbor_distances is None:
            neighbor_distances = jnp.zeros((natom, max_neighbors), dtype=jnp.float32)
        if pair_fn is not None:
            # pair_fn path: auto-allocate energy/force buffers and run the
            # call-time callable that closes over ``pair_fn`` (returns 7 outputs).
            if pair_energies is None:
                pair_energies = jnp.zeros((natom, max_neighbors), dtype=jnp.float32)
            if pair_forces is None:
                pair_forces = jnp.zeros((natom, max_neighbors, 3), dtype=jnp.float32)
            pair_params_arg = jnp.asarray(pair_params, dtype=jnp.float32)
            pair_callable = _get_jax_cluster_tile_pair_fn_callable(pair_fn)
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
                num_tiles,
                tile_row_group,
                tile_col_group,
                cell_n,
                inv_cell_n,
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
        ) = _jax_query_cluster_tile_pair(
            sorted_atom_index,
            sorted_pos_x,
            sorted_pos_y,
            sorted_pos_z,
            num_tiles,
            tile_row_group,
            tile_col_group,
            cell_n,
            inv_cell_n,
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
        ) = _jax_query_cluster_tile_dual_selective(
            sorted_atom_index,
            sorted_pos_x,
            sorted_pos_y,
            sorted_pos_z,
            num_tiles,
            tile_row_group,
            tile_col_group,
            cell_n,
            inv_cell_n,
            rf,
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
        ) = _jax_query_cluster_tile_dual(
            sorted_atom_index,
            sorted_pos_x,
            sorted_pos_y,
            sorted_pos_z,
            num_tiles,
            tile_row_group,
            tile_col_group,
            cell_n,
            inv_cell_n,
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
            _jax_query_cluster_tile_selective(
                sorted_atom_index,
                sorted_pos_x,
                sorted_pos_y,
                sorted_pos_z,
                num_tiles,
                tile_row_group,
                tile_col_group,
                cell_n,
                inv_cell_n,
                rf,
                neighbor_matrix,
                num_neighbors,
                neighbor_matrix_shifts,
                float(cutoff),
                int(natom),
            )
        )
    else:
        neighbor_matrix, num_neighbors, neighbor_matrix_shifts = (
            _jax_query_cluster_tile(
                sorted_atom_index,
                sorted_pos_x,
                sorted_pos_y,
                sorted_pos_z,
                num_tiles,
                tile_row_group,
                tile_col_group,
                cell_n,
                inv_cell_n,
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


def query_cluster_tile_coo(
    sorted_atom_index: jax.Array,
    sorted_pos_x: jax.Array,
    sorted_pos_y: jax.Array,
    sorted_pos_z: jax.Array,
    num_tiles: jax.Array,
    tile_row_group: jax.Array,
    tile_col_group: jax.Array,
    cell: jax.Array,
    cutoff: float,
    natom: int,
    max_pairs: int,
    *,
    rebuild_flags: jax.Array | None = None,
    pair_offsets: jax.Array | None = None,
    pair_counts: jax.Array | None = None,
    neighbor_list: jax.Array | None = None,
    neighbor_list_shifts: jax.Array | None = None,
) -> tuple[jax.Array, ...]:
    """Convert the tile pair list to flat COO form.

    Compact mode returns (neighbor_list, neighbor_ptr, neighbor_list_shifts).
    Segmented mode returns fixed buffers as
    (neighbor_list, pair_offsets, pair_counts, neighbor_list_shifts).
    """
    segmented = pair_offsets is not None or pair_counts is not None
    if (pair_offsets is None) != (pair_counts is None):
        raise ValueError("Pass both 'pair_offsets' and 'pair_counts', or neither.")
    if rebuild_flags is not None and not segmented:
        raise ValueError("rebuild_flags requires pair_offsets and pair_counts")

    cell_n = _normalize_cell(cell, jnp.float32)
    inv_cell_n = jnp.linalg.inv(cell_n[0])[jnp.newaxis, :, :]

    pair_counter = jnp.zeros(1, dtype=jnp.int32)
    if segmented:
        max_pairs = int(pair_offsets[-1])
        if neighbor_list is None:
            neighbor_list = jnp.zeros((2, max_pairs), dtype=jnp.int32)
        if neighbor_list_shifts is None:
            neighbor_list_shifts = jnp.zeros((max_pairs, 3), dtype=jnp.int32)
        coo_list = neighbor_list.T.copy()
        coo_shifts = neighbor_list_shifts
        if rebuild_flags is None:
            rf = jnp.ones(1, dtype=jnp.bool_)
        else:
            rf = rebuild_flags.flatten()[:1].astype(jnp.bool_)
        pair_counter, pair_counts, coo_list, coo_shifts = (
            _jax_query_cluster_tile_coo_segmented(
                sorted_atom_index,
                sorted_pos_x,
                sorted_pos_y,
                sorted_pos_z,
                num_tiles,
                tile_row_group,
                tile_col_group,
                cell_n,
                inv_cell_n,
                rf,
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
    pair_counter, coo_list, coo_shifts = _jax_query_cluster_tile_coo(
        sorted_atom_index,
        sorted_pos_x,
        sorted_pos_y,
        sorted_pos_z,
        num_tiles,
        tile_row_group,
        tile_col_group,
        cell_n,
        inv_cell_n,
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


def _cluster_tile_pair_outputs_forward(
    positions: jax.Array,
    cell: jax.Array,
    *,
    cutoff: float,
    max_neighbors: int,
    fill_value: int,
    pair_fn=None,
    pair_params: jax.Array | None = None,
) -> _NeighborForwardOutput:
    """Forward closure for the cluster_tile autograd path.

    Builds the cluster tiles + runs the pair-output query kernel inside a
    single closure.  positions/cell are stop_gradient'd before any Warp
    launch; the autograd primitive's reconstruction backward gets the
    live tensors separately.  When ``pair_fn`` is set, ``query_cluster_tile``
    returns per-pair ``pair_energies`` / ``pair_forces`` which ride along in
    ``extra_outputs`` (forward-only).
    """
    positions = jax.lax.stop_gradient(positions)
    cell = jax.lax.stop_gradient(cell)
    natom = positions.shape[0]
    (
        sorted_atom_index,
        _morton_codes,
        sorted_pos_x,
        sorted_pos_y,
        sorted_pos_z,
        _group_ctr_x,
        _group_ctr_y,
        _group_ctr_z,
        _group_ext_x,
        _group_ext_y,
        _group_ext_z,
        num_tiles,
        tile_row_group,
        tile_col_group,
    ) = build_cluster_tile_list(positions, cutoff, cell)

    out = query_cluster_tile(
        sorted_atom_index,
        sorted_pos_x,
        sorted_pos_y,
        sorted_pos_z,
        num_tiles,
        tile_row_group,
        tile_col_group,
        cell,
        cutoff,
        natom,
        int(max_neighbors),
        fill_value=fill_value,
        return_vectors=True,
        return_distances=True,
        pair_fn=pair_fn,
        pair_params=pair_params,
    )
    has_pair_fn = pair_fn is not None
    if has_pair_fn:
        nm, nn, shifts, vec, dist, pe, pf = out
    else:
        nm, nn, shifts, vec, dist = out

    i_idx, j_idx, shifts_ret, _, mask_ = _build_index_residuals(
        nm,
        nn,
        shifts,
    )
    K, M = nm.shape
    extra_outputs = (nm, nn, shifts, pe, pf) if has_pair_fn else (nm, nn, shifts)
    return _NeighborForwardOutput(
        distances=dist,
        vectors=vec,
        extra_outputs=extra_outputs,
        i_idx=i_idx,
        j_idx=j_idx,
        shifts=shifts_ret,
        batch_idx=None,
        active_mask=mask_,
        matrix_shape=(K, M),
    )


def cluster_tile_neighbor_list(
    positions: jax.Array,
    cutoff: float,
    cell: jax.Array,
    max_neighbors: int | None = None,
    fill_value: int | None = None,
    format: str = "matrix",
    max_pairs: int | None = None,
    *,
    cutoff2: float | None = None,
    rebuild_flags: jax.Array | None = None,
    pair_offsets: jax.Array | None = None,
    previous_pair_counts: jax.Array | None = None,
    previous_neighbor_list: jax.Array | None = None,
    previous_neighbor_list_shifts: jax.Array | None = None,
    previous_num_tiles: jax.Array | None = None,
    previous_tile_row_group: jax.Array | None = None,
    previous_tile_col_group: jax.Array | None = None,
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
    """Build a cluster-pair tile neighbor list (one-shot convenience).

    Single-system JAX binding for the cluster-pair tile algorithm.  Runs
    Morton sort, bounding-box reduction, and tile enumeration, then emits
    the result in one of three formats selected by ``format=``.  Mirrors
    :func:`nvalchemiops.torch.neighbors.cluster_tile.cluster_tile_neighbor_list`.

    Parameters
    ----------
    positions : jax.Array, shape (N, 3), dtype=float32
        Atomic coordinates. Any ``N >= 0``; non-32-aligned ``N`` is padded
        internally to a multiple of ``TILE_GROUP_SIZE``.
    cutoff : float
        Cutoff distance in Cartesian units. Must be positive.
    cutoff2 : float, optional
        Matrix-format second cutoff. Cannot be combined with pair outputs
        or COO/tile formats.
    rebuild_flags : jax.Array, shape (1,), dtype=bool, optional
        Selective rebuild flag for matrix or segmented COO output. Requires
        previous tile state plus previous output buffers.
    cell : jax.Array, shape (3, 3) or (1, 3, 3), dtype=float32
        Unit cell matrix. Cluster-tile assumes fully periodic boundaries.
    max_neighbors : int, optional
        Max neighbors per atom (``"matrix"`` format only). Falls back to
        :func:`estimate_max_neighbors`.
    fill_value : int, optional
        Matrix sentinel; defaults to ``N``.
    format : {"matrix", "coo", "tile"}, default "matrix"
        Output representation. See Returns.
    max_pairs : int, optional
        Upper bound for compact COO output; defaults to ``N * max_neighbors``.
    pair_offsets, previous_pair_counts, previous_neighbor_list, previous_neighbor_list_shifts : jax.Array, optional
        Fixed segmented-COO buffers used with ``rebuild_flags`` and
        ``format="coo"``. The return tuple preserves the fixed buffer
        shapes and reports the updated ``pair_counts``.
    return_vectors, return_distances : bool, default False
        If True, append per-pair displacement vectors / scalar distances
        to the matrix-format return tuple. Matrix format only.
    pair_fn, pair_params, neighbor_vectors, neighbor_distances, pair_energies, pair_forces : optional
        Reserved for parity with the torch binding. Raise
        ``NotImplementedError`` on the JAX path because the JAX
        ``jax_callable`` pathway cannot carry a callable ``pair_fn`` across
        the trace boundary.

    Returns
    -------
    For ``format == "matrix"``:
        ``(neighbor_matrix, num_neighbors, neighbor_matrix_shifts)``, with
        optional ``(*, distances)`` and/or ``(*, vectors)`` appended when
        ``return_distances`` / ``return_vectors`` is True.
    For ``format == "coo"``:
        ``(neighbor_list, neighbor_ptr, neighbor_list_shifts)`` in compact
        mode, or ``(neighbor_list, pair_offsets, pair_counts,
        neighbor_list_shifts, num_tiles, tile_row_group, tile_col_group)``
        with ``rebuild_flags`` segmented mode.
    For ``format == "tile"``:
        ``(num_tiles, tile_row_group, tile_col_group, sorted_atom_index,
        sorted_pos_x, sorted_pos_y, sorted_pos_z)`` --- the raw cluster
        state for downstream tile consumers.

    Notes
    -----
    - Cluster-tile is CUDA float32 only.
    - Cluster-tile does not support partial neighbor lists; there is no
      ``target_indices`` kwarg.
    - The unified
      :func:`nvalchemiops.jax.neighbors.neighbor_list` entry point selects
      this binding automatically for fully-periodic float32 CUDA inputs
      with at least 2000 atoms and no pair-output kwargs; call this
      function directly to force the strategy.

    See Also
    --------
    nvalchemiops.jax.neighbors.batch_cluster_tile_neighbor_list :
        Batched companion entry point.
    nvalchemiops.jax.neighbors.cluster_tile.build_cluster_tile_list :
        Lower-level build step exposed for caching across queries.
    nvalchemiops.jax.neighbors.cluster_tile.query_cluster_tile :
        Lower-level query step.
    """

    if positions.dtype != jnp.float32:
        raise TypeError("positions must be float32")
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
            "format='tile' on the JAX cluster_tile binding; use 'matrix' or 'coo'.",
        )
    if dual_cutoff and format != "matrix":
        raise NotImplementedError(
            "JAX cluster_tile cutoff2 is supported only with format='matrix'.",
        )
    if selective and format not in {"matrix", "coo"}:
        raise NotImplementedError(
            "JAX cluster_tile rebuild_flags are supported only with "
            "format='matrix' or segmented format='coo'.",
        )
    if (dual_cutoff or selective) and has_pair_outputs:
        raise NotImplementedError(
            "JAX cluster_tile cutoff2 / rebuild_flags cannot be combined "
            "with return_distances or return_vectors in this pass.",
        )
    if selective:
        required = {
            "previous_num_tiles": previous_num_tiles,
            "previous_tile_row_group": previous_tile_row_group,
            "previous_tile_col_group": previous_tile_col_group,
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
                "rebuild_flags requires previous cluster_tile state: "
                + ", ".join(missing)
            )
    N = positions.shape[0]
    if max_neighbors is None:
        max_neighbors = estimate_max_neighbors(
            cutoff2 if cutoff2 is not None else cutoff
        )

    if N == 0:
        # Public docstring promises ``N >= 0``.  Returning zero-shaped
        # arrays of the right structure avoids the ``positions[-1:]``
        # padding step in :func:`_morton_sort_and_gather` and gives
        # callers a no-op result that broadcasts naturally downstream.
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
            # 7-tuple matching the non-empty ``"tile"`` branch shape.
            empty_i32 = jnp.empty(0, dtype=jnp.int32)
            empty_f32 = jnp.empty(0, dtype=positions.dtype)
            return (
                jnp.zeros(1, dtype=jnp.int32),  # num_tiles
                empty_i32,  # tile_row_group
                empty_i32,  # tile_col_group
                empty_i32,  # sorted_atom_index
                empty_f32,  # sorted_pos_x
                empty_f32,  # sorted_pos_y
                empty_f32,  # sorted_pos_z
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
                previous_num_tiles,
                previous_tile_row_group,
                previous_tile_col_group,
            )
        return matrix_out

    if has_pair_outputs:
        if fill_value is None:
            fill_value = N
        forward_kwargs = {
            "cutoff": float(cutoff),
            "max_neighbors": int(max_neighbors),
            "fill_value": int(fill_value),
            "pair_fn": pair_fn,
            "pair_params": pair_params,
        }
        route_out = _route_pair_outputs(
            positions,
            cell,
            _cluster_tile_pair_outputs_forward,
            forward_kwargs,
        )
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
            # Mirror the naive / cell_list COO contract: convert the matrix
            # topology to a flat neighbor list and repack the per-pair geometry
            # (and pair_fn outputs) into the same COO order.  Eager-only, like the
            # matrix->COO index conversion (the pair count is data-dependent).
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

    # Tile candidates must cover the larger radius so the cutoff2 matrix cannot
    # miss pairs in the (cutoff, cutoff2] shell; the query filters each matrix
    # by its own cutoff.
    build_cutoff = cutoff if cutoff2 is None else max(float(cutoff), float(cutoff2))
    (
        sorted_atom_index,
        _morton_codes,
        sorted_pos_x,
        sorted_pos_y,
        sorted_pos_z,
        _group_ctr_x,
        _group_ctr_y,
        _group_ctr_z,
        _group_ext_x,
        _group_ext_y,
        _group_ext_z,
        num_tiles,
        tile_row_group,
        tile_col_group,
    ) = build_cluster_tile_list(
        positions,
        build_cutoff,
        cell,
        rebuild_flags=rebuild_flags,
        num_tiles=previous_num_tiles,
        tile_row_group=previous_tile_row_group,
        tile_col_group=previous_tile_col_group,
    )

    # Eager guard: raise on tile-buffer overflow instead of silently dropping
    # tiles.  Skipped under trace (positions/num_tiles are tracers) -- there the
    # ``ngroup**2`` geometry fallback guarantees the buffer never overflows.
    if not isinstance(num_tiles, jax.core.Tracer):
        n_tiles_host = int(num_tiles[0])
        tile_capacity = int(tile_row_group.shape[0])
        if n_tiles_host > tile_capacity:
            raise NeighborOverflowError(tile_capacity, n_tiles_host)

    if format == "tile":
        return (
            num_tiles,
            tile_row_group,
            tile_col_group,
            sorted_atom_index,
            sorted_pos_x,
            sorted_pos_y,
            sorted_pos_z,
        )

    if format == "coo":
        if selective:
            coo_out = query_cluster_tile_coo(
                sorted_atom_index,
                sorted_pos_x,
                sorted_pos_y,
                sorted_pos_z,
                num_tiles,
                tile_row_group,
                tile_col_group,
                cell,
                cutoff,
                N,
                int(pair_offsets[-1]),
                rebuild_flags=rebuild_flags,
                pair_offsets=pair_offsets,
                pair_counts=previous_pair_counts,
                neighbor_list=previous_neighbor_list,
                neighbor_list_shifts=previous_neighbor_list_shifts,
            )
            return (*coo_out, num_tiles, tile_row_group, tile_col_group)
        if max_pairs is None:
            max_pairs = N * max_neighbors
        return query_cluster_tile_coo(
            sorted_atom_index,
            sorted_pos_x,
            sorted_pos_y,
            sorted_pos_z,
            num_tiles,
            tile_row_group,
            tile_col_group,
            cell,
            cutoff,
            N,
            int(max_pairs),
        )

    # format == "matrix"
    matrix_out = query_cluster_tile(
        sorted_atom_index,
        sorted_pos_x,
        sorted_pos_y,
        sorted_pos_z,
        num_tiles,
        tile_row_group,
        tile_col_group,
        cell,
        cutoff,
        N,
        int(max_neighbors),
        fill_value=fill_value,
        cutoff2=cutoff2,
        rebuild_flags=rebuild_flags,
        neighbor_matrix=previous_neighbor_matrix,
        num_neighbors=previous_num_neighbors,
        neighbor_matrix_shifts=previous_neighbor_matrix_shifts,
        neighbor_matrix2=previous_neighbor_matrix2,
        num_neighbors2=previous_num_neighbors2,
        neighbor_matrix_shifts2=previous_neighbor_matrix_shifts2,
    )
    if selective:
        return (*matrix_out, num_tiles, tile_row_group, tile_col_group)
    return matrix_out
