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


"""Generated Warp kernels and factories for cluster-tile neighbor lists."""

from functools import lru_cache

import warp as wp

from nvalchemiops.neighbors.neighbor_utils import (
    _append_specialization_doc,
    kernel_specialization_name,
    set_fn_doc,
    set_fn_name,
)

__all__ = [
    "TILE_GROUP_SIZE",
    "_compute_morton_kernel",
    "_get_reset_cluster_tile_counts_kernel",
    "_get_build_cluster_tiles_kernel",
    "_get_tile_sort_specialization",
    "_group_buffer",
    "_group_scratch_size",
    "_permute_gather_soa_kernel",
    "_rank_to_group_kernel",
    "_require_f32",
    "get_batch_query_cluster_tile_coo_kernel",
    "get_batch_query_cluster_tile_kernel",
    "get_query_cluster_tile_coo_kernel",
    "get_query_cluster_tile_kernel",
]


TILE_GROUP_SIZE = 32
TILE = wp.constant(TILE_GROUP_SIZE)

_MORTON_PADDING_SENTINEL = 0x40000000


def _pair_output_features(
    *,
    selective: bool = False,
    dual_cutoff: bool = False,
    coo_segmented: bool = False,
    return_vectors: bool = False,
    return_distances: bool = False,
    pair_fn: wp.Function | None = None,
) -> tuple[str, ...]:
    """Return specialization-name tokens for cluster-tile pair outputs."""
    return tuple(
        feature
        for feature in (
            "selective" if selective else "",
            "dual_cutoff" if dual_cutoff else "",
            "coo_segmented" if coo_segmented else "",
            "vectors" if return_vectors else "",
            "distances" if return_distances else "",
            "pair_fn" if pair_fn is not None else "",
        )
        if feature
    )


def _require_f32(wp_dtype: type) -> None:
    """Reject non-float32 dtypes for cluster-tile public launchers."""
    if wp_dtype is not wp.float32:
        raise ValueError(
            f"cluster_tile kernels are float32-only; got {wp_dtype!r}",
        )


def _group_scratch_size(ngroup: int) -> int:
    """Return padded group scratch length for tile-load safety."""
    ngroup_padded = (
        (ngroup + TILE_GROUP_SIZE - 1) // TILE_GROUP_SIZE
    ) * TILE_GROUP_SIZE
    if ngroup_padded == ngroup:
        return ngroup + TILE_GROUP_SIZE
    return ngroup_padded


def _group_buffer(buffer: wp.array | None, size: int, device: str) -> wp.array:
    """Return caller-owned or transient float32 group scratch."""
    if buffer is not None:
        return buffer
    return wp.empty(size, dtype=wp.float32, device=device)


@wp.func
def _wrap_triclinic(
    d: wp.vec3f,
    cell: wp.mat33f,
    inv_cell: wp.mat33f,
):
    """Return triclinic minimum-image displacement and integer shift

    Parameters
    ----------
    d : wp.vec3f
        Cartesian displacement before wrapping.
    cell : wp.mat33f
        Cell matrix.
    inv_cell : wp.mat33f
        Inverse cell matrix.

    Returns
    -------
    tuple[wp.vec3f, wp.vec3i]
        Minimum-image displacement and integer shift vector.
    """
    inv_cell_T = wp.transpose(inv_cell)
    f = inv_cell_T * d
    s_a = -wp.int32(wp.floor(f[0] + wp.float32(0.5)))
    s_b = -wp.int32(wp.floor(f[1] + wp.float32(0.5)))
    s_c = -wp.int32(wp.floor(f[2] + wp.float32(0.5)))
    f = f + wp.vec3f(wp.float32(s_a), wp.float32(s_b), wp.float32(s_c))
    cell_T = wp.transpose(cell)
    d_new = cell_T * f
    return d_new, wp.vec3i(s_a, s_b, s_c)


@wp.kernel(enable_backward=False, module="morton_build")
def _compute_morton_kernel(
    positions: wp.array(dtype=wp.vec3f),
    inv_cell: wp.array(dtype=wp.mat33f),
    natom: wp.int32,
    morton_codes: wp.array(dtype=wp.int32),
    sorted_atom_index: wp.array(dtype=wp.int32),
    num_neighbors: wp.array(dtype=wp.int32),
    num_tiles: wp.array(dtype=wp.int32),
) -> None:
    """Compute per-atom Morton codes and initialize tile scratch

    Parameters
    ----------
    positions : wp.array, shape (natom_padded,), dtype=wp.vec3f
        Atomic positions.
    inv_cell : wp.array, shape (1,), dtype=wp.mat33f
        Inverse cell matrix.
    natom : wp.int32
        Number of real atoms before padding.
    morton_codes : wp.array, shape (natom_padded,), dtype=wp.int32
        OUTPUT: Morton code per atom or padding sentinel.
    sorted_atom_index : wp.array, shape (natom_padded,), dtype=wp.int32
        OUTPUT: Initial atom-index permutation.
    num_neighbors : wp.array, shape (natom,), dtype=wp.int32
        OUTPUT: Neighbor counts initialized to zero.
    num_tiles : wp.array, shape (1,), dtype=wp.int32
        OUTPUT: Tile counter initialized to zero.

    Returns
    -------
    None
        This function modifies the input arrays in-place.

    Notes
    -----
    - Thread launch: 1D over the padded atom count.
    - Modifies: morton_codes, sorted_atom_index, num_neighbors, num_tiles.

    See Also
    --------
    _compute_morton : Launch Morton-code generation for cluster-tile construction.
    """
    i = wp.tid()
    if i == 0:
        num_tiles[0] = 0
    if i >= natom:
        morton_codes[i] = wp.int32(_MORTON_PADDING_SENTINEL)
        sorted_atom_index[i] = i
        return
    pos = positions[i]
    inv_cell_mat = inv_cell[0]
    inv_cell_T = wp.transpose(inv_cell_mat)
    f = inv_cell_T * pos
    fx = f[0] - wp.floor(f[0])
    fy = f[1] - wp.floor(f[1])
    fz = f[2] - wp.floor(f[2])
    ix = wp.int32(fx * 1024.0)
    iy = wp.int32(fy * 1024.0)
    iz = wp.int32(fz * 1024.0)
    if ix < 0:
        ix = 0
    if ix > 1023:
        ix = 1023
    if iy < 0:
        iy = 0
    if iy > 1023:
        iy = 1023
    if iz < 0:
        iz = 0
    if iz > 1023:
        iz = 1023
    ix = ix & 0x000003FF
    ix = (ix | (ix << 16)) & 0x030000FF
    ix = (ix | (ix << 8)) & 0x0300F00F
    ix = (ix | (ix << 4)) & 0x030C30C3
    ix = (ix | (ix << 2)) & 0x09249249
    iy = iy & 0x000003FF
    iy = (iy | (iy << 16)) & 0x030000FF
    iy = (iy | (iy << 8)) & 0x0300F00F
    iy = (iy | (iy << 4)) & 0x030C30C3
    iy = (iy | (iy << 2)) & 0x09249249
    iz = iz & 0x000003FF
    iz = (iz | (iz << 16)) & 0x030000FF
    iz = (iz | (iz << 8)) & 0x0300F00F
    iz = (iz | (iz << 4)) & 0x030C30C3
    iz = (iz | (iz << 2)) & 0x09249249
    morton_codes[i] = (iz << 2) | (iy << 1) | ix
    sorted_atom_index[i] = i
    num_neighbors[i] = 0


@wp.kernel(enable_backward=False, module="morton_build")
def _permute_gather_soa_kernel(
    positions: wp.array(dtype=wp.vec3f),
    sorted_atom_index: wp.array(dtype=wp.int32),
    natom: wp.int32,
    sorted_pos_x: wp.array(dtype=wp.float32),
    sorted_pos_y: wp.array(dtype=wp.float32),
    sorted_pos_z: wp.array(dtype=wp.float32),
) -> None:
    """Gather AoS positions into Morton-sorted SoA arrays

    Parameters
    ----------
    positions : wp.array, shape (natom,), dtype=wp.vec3f
        Atomic positions in original ordering.
    sorted_atom_index : wp.array, shape (natom,), dtype=wp.int32
        Source atom index for each sorted slot.
    natom : wp.int32
        Number of real atoms.
    sorted_pos_x : wp.array, shape (natom,), dtype=wp.float32
        OUTPUT: Sorted x coordinates.
    sorted_pos_y : wp.array, shape (natom,), dtype=wp.float32
        OUTPUT: Sorted y coordinates.
    sorted_pos_z : wp.array, shape (natom,), dtype=wp.float32
        OUTPUT: Sorted z coordinates.

    Returns
    -------
    None
        This function modifies the input arrays in-place.

    Notes
    -----
    - Thread launch: 1D over ``natom``.
    - Modifies: sorted_pos_x, sorted_pos_y, sorted_pos_z.

    See Also
    --------
    _permute_gather_soa : Launch sorted-position and tile-coordinate gathering.
    """
    i = wp.tid()
    if i >= natom:
        return
    src = sorted_atom_index[i]
    pos = positions[src]
    sorted_pos_x[i] = pos[0]
    sorted_pos_y[i] = pos[1]
    sorted_pos_z[i] = pos[2]


TILE_SORT_N_1024 = wp.constant(1024)
TILE_SORT_N_2048 = wp.constant(2048)


@wp.kernel(enable_backward=False, module="warp_tile_sort_1024")
def _tile_sort_kernel_1024(
    keys: wp.array(dtype=wp.int32),
    values: wp.array(dtype=wp.int32),
) -> None:
    """Sort 1024 int32 key/value pairs in-place with one tiled block

    Parameters
    ----------
    keys : wp.array, shape (1024,), dtype=wp.int32
        MODIFIED: Sort keys.
    values : wp.array, shape (1024,), dtype=wp.int32
        MODIFIED: Values permuted with ``keys``.

    Returns
    -------
    None
        This function modifies the input arrays in-place.

    Notes
    -----
    - Thread launch: one tiled block with fixed size 1024.
    - Modifies: keys and values.

    See Also
    --------
    _tile_sort_pairs : Launch tile-pair sorting for cluster-tile construction.
    """
    keys_tile = wp.tile_load(keys, shape=TILE_SORT_N_1024, storage="shared")
    values_tile = wp.tile_load(values, shape=TILE_SORT_N_1024, storage="shared")
    wp.tile_sort(keys_tile, values_tile)
    wp.tile_store(keys, keys_tile)
    wp.tile_store(values, values_tile)


@wp.kernel(enable_backward=False, module="warp_tile_sort_2048")
def _tile_sort_kernel_2048(
    keys: wp.array(dtype=wp.int32),
    values: wp.array(dtype=wp.int32),
) -> None:
    """Sort 2048 int32 key/value pairs in-place with one tiled block

    Parameters
    ----------
    keys : wp.array, shape (2048,), dtype=wp.int32
        MODIFIED: Sort keys.
    values : wp.array, shape (2048,), dtype=wp.int32
        MODIFIED: Values permuted with ``keys``.

    Returns
    -------
    None
        This function modifies the input arrays in-place.

    Notes
    -----
    - Thread launch: one tiled block with fixed size 2048.
    - Modifies: keys and values.

    See Also
    --------
    _tile_sort_pairs : Launch tile-pair sorting for cluster-tile construction.
    """
    keys_tile = wp.tile_load(keys, shape=TILE_SORT_N_2048, storage="shared")
    values_tile = wp.tile_load(values, shape=TILE_SORT_N_2048, storage="shared")
    wp.tile_sort(keys_tile, values_tile)
    wp.tile_store(keys, keys_tile)
    wp.tile_store(values, values_tile)


_TILE_SORT_SPECIALIZATIONS = {
    1024: (_tile_sort_kernel_1024, 512),
    2048: (_tile_sort_kernel_2048, 512),
}


def _get_tile_sort_specialization(natom: int) -> tuple[wp.Kernel, int] | None:
    """Return the fixed-size tile-sort specialization for ``natom``."""
    return _TILE_SORT_SPECIALIZATIONS.get(int(natom))


@wp.kernel(enable_backward=False)
def _rank_to_group_kernel(
    sorted_pos_x: wp.array(dtype=wp.float32),
    sorted_pos_y: wp.array(dtype=wp.float32),
    sorted_pos_z: wp.array(dtype=wp.float32),
    group_ctr_x: wp.array(dtype=wp.float32),
    group_ctr_y: wp.array(dtype=wp.float32),
    group_ctr_z: wp.array(dtype=wp.float32),
    group_ext_x: wp.array(dtype=wp.float32),
    group_ext_y: wp.array(dtype=wp.float32),
    group_ext_z: wp.array(dtype=wp.float32),
) -> None:
    """Compute one axis-aligned bounding box for each 32-atom cluster

    Parameters
    ----------
    sorted_pos_x : wp.array, shape (natom_padded,), dtype=wp.float32
        Sorted atom x coordinates.
    sorted_pos_y : wp.array, shape (natom_padded,), dtype=wp.float32
        Sorted atom y coordinates.
    sorted_pos_z : wp.array, shape (natom_padded,), dtype=wp.float32
        Sorted atom z coordinates.
    group_ctr_x : wp.array, shape (ngroup,), dtype=wp.float32
        OUTPUT: Group center x coordinates.
    group_ctr_y : wp.array, shape (ngroup,), dtype=wp.float32
        OUTPUT: Group center y coordinates.
    group_ctr_z : wp.array, shape (ngroup,), dtype=wp.float32
        OUTPUT: Group center z coordinates.
    group_ext_x : wp.array, shape (ngroup,), dtype=wp.float32
        OUTPUT: Group half-extent x coordinates.
    group_ext_y : wp.array, shape (ngroup,), dtype=wp.float32
        OUTPUT: Group half-extent y coordinates.
    group_ext_z : wp.array, shape (ngroup,), dtype=wp.float32
        OUTPUT: Group half-extent z coordinates.

    Returns
    -------
    None
        This function modifies the input arrays in-place.

    Notes
    -----
    - Thread launch: tiled with one block per group and block dimension ``TILE_GROUP_SIZE``.
    - Modifies: group center and extent arrays.

    See Also
    --------
    _tile_sort_pairs : Launch tile-group rank construction after sorting.
    """
    g = wp.tid()
    x_tile = wp.tile_load(sorted_pos_x, shape=TILE, offset=g * TILE)
    y_tile = wp.tile_load(sorted_pos_y, shape=TILE, offset=g * TILE)
    z_tile = wp.tile_load(sorted_pos_z, shape=TILE, offset=g * TILE)

    x_min = wp.tile_extract(wp.tile_min(x_tile), 0)
    x_max = wp.tile_extract(wp.tile_max(x_tile), 0)
    y_min = wp.tile_extract(wp.tile_min(y_tile), 0)
    y_max = wp.tile_extract(wp.tile_max(y_tile), 0)
    z_min = wp.tile_extract(wp.tile_min(z_tile), 0)
    z_max = wp.tile_extract(wp.tile_max(z_tile), 0)

    lane_tile = wp.tile_arange(TILE, dtype=wp.int32)
    lane = wp.untile(lane_tile)
    if lane == 0:
        group_ctr_x[g] = wp.float32(0.5) * (x_max + x_min)
        group_ctr_y[g] = wp.float32(0.5) * (y_max + y_min)
        group_ctr_z[g] = wp.float32(0.5) * (z_max + z_min)
        group_ext_x[g] = wp.float32(0.5) * (x_max - x_min)
        group_ext_y[g] = wp.float32(0.5) * (y_max - y_min)
        group_ext_z[g] = wp.float32(0.5) * (z_max - z_min)


@wp.func
def _bbox_distance_sq(
    d: wp.vec3f,
    rg_ext: wp.vec3f,
    cg_ext: wp.vec3f,
) -> wp.float32:
    """Return squared distance between two Cartesian axis-aligned boxes.

    Parameters
    ----------
    d : wp.vec3f
        Cartesian displacement between the two box centers.
    rg_ext : wp.vec3f
        Half-extents of the row-group bounding box.
    cg_ext : wp.vec3f
        Half-extents of the column-group bounding box.

    Returns
    -------
    wp.float32
        Squared gap distance between the two boxes, or zero when they overlap.
    """
    dx = wp.max(wp.abs(d[0]) - rg_ext[0] - cg_ext[0], wp.float32(0.0))
    dy = wp.max(wp.abs(d[1]) - rg_ext[1] - cg_ext[1], wp.float32(0.0))
    dz = wp.max(wp.abs(d[2]) - rg_ext[2] - cg_ext[2], wp.float32(0.0))
    return dx * dx + dy * dy + dz * dz


@wp.func
def _bbox_valid(
    col_idx: wp.int32,
    row_group: wp.int32,
    col_limit: wp.int32,
    rg_ctr: wp.vec3f,
    rg_ext: wp.vec3f,
    cg_ctr: wp.vec3f,
    cg_ext: wp.vec3f,
    cell: wp.mat33f,
    inv_cell: wp.mat33f,
    cutoff_sq: wp.float32,
) -> wp.int32:
    """Return whether a group pair can contain an in-cutoff atom pair

    Parameters
    ----------
    col_idx : wp.int32
        Candidate column group index.
    row_group : wp.int32
        Row group index.
    col_limit : wp.int32
        Exclusive upper bound for valid column groups.
    rg_ctr : wp.vec3f
        Row-group bounding-box center.
    rg_ext : wp.vec3f
        Row-group bounding-box half extents.
    cg_ctr : wp.vec3f
        Column-group bounding-box center.
    cg_ext : wp.vec3f
        Column-group bounding-box half extents.
    cell : wp.mat33f
        Cell matrix.
    inv_cell : wp.mat33f
        Inverse cell matrix.
    cutoff_sq : wp.float32
        Squared cutoff distance.

    Returns
    -------
    wp.int32
        One when the group pair can contain an in-cutoff atom pair, otherwise zero.
    """
    if col_idx < row_group:
        return 0
    if col_idx >= col_limit:
        return 0
    d_ctr = wp.vec3f(
        rg_ctr[0] - cg_ctr[0],
        rg_ctr[1] - cg_ctr[1],
        rg_ctr[2] - cg_ctr[2],
    )
    d_wrapped, _s = _wrap_triclinic(d_ctr, cell, inv_cell)
    if _bbox_distance_sq(d_wrapped, rg_ext, cg_ext) < cutoff_sq:
        return 1

    cell_T = wp.transpose(cell)
    for ia in range(3):
        shift_a = wp.float32(ia - 1)
        for ib in range(3):
            shift_b = wp.float32(ib - 1)
            for ic in range(3):
                if ia != 1 or ib != 1 or ic != 1:
                    shift_c = wp.float32(ic - 1)
                    d_image = d_wrapped + cell_T * wp.vec3f(
                        shift_a,
                        shift_b,
                        shift_c,
                    )
                    if _bbox_distance_sq(d_image, rg_ext, cg_ext) < cutoff_sq:
                        return 1
    return 0


@lru_cache(maxsize=None)
def _get_reset_cluster_tile_counts_kernel(*, selective: bool) -> wp.Kernel:
    """Build a counter-reset kernel for cluster-tile segmented state."""
    SELECTIVE = wp.constant(bool(selective))

    @wp.kernel(enable_backward=False)
    def _kernel(
        counts: wp.array(dtype=wp.int32),
        rebuild_flags: wp.array(dtype=wp.bool),
    ) -> None:
        """Reset cluster-tile per-system counters

        Parameters
        ----------
        counts : wp.array, shape (num_systems,), dtype=wp.int32
            OUTPUT: Counters to reset before a selective or segmented launch.
        rebuild_flags : wp.array, shape (num_systems,), dtype=wp.bool
            Per-system rebuild flags. Read only in selective specializations.

        Returns
        -------
        None
            This function modifies the input arrays in-place.

        Notes
        -----
        - Thread launch: One thread per counter entry.
        - Modifies: ``counts`` entries selected by ``rebuild_flags`` or all entries.
        ``SELECTIVE`` is a static specialization. The reset is a separate
        pre-pass so main kernels never race by zeroing and atomically
        incrementing the same per-system counter.

        See Also
        --------
        _get_reset_cluster_tile_counts_kernel : Return the specialized counter-reset kernel.
        """
        isys = wp.tid()
        if SELECTIVE:
            if rebuild_flags[isys]:
                counts[isys] = 0
        else:
            counts[isys] = 0

    name = kernel_specialization_name(
        "_reset_cluster_tile_counts",
        features=("selective" if selective else "",),
    )
    return set_fn_doc(
        set_fn_name(_kernel, name),
        _append_specialization_doc(
            _kernel.__doc__,
            entries=(
                ("operation", "reset_cluster_tile_counts"),
                ("selective", bool(selective)),
            ),
        ),
    )


@lru_cache(maxsize=None)
def _get_build_cluster_tiles_kernel(
    *, batched: bool, segmented: bool = False, selective: bool = False
) -> wp.Kernel:
    """Build the single/batched group-to-tile enumeration kernel."""
    BATCHED = wp.constant(bool(batched))
    SEGMENTED = wp.constant(bool(segmented))
    SELECTIVE = wp.constant(bool(selective))

    @wp.kernel(enable_backward=False)
    def _kernel(
        group_ctr_x: wp.array(dtype=wp.float32),
        group_ctr_y: wp.array(dtype=wp.float32),
        group_ctr_z: wp.array(dtype=wp.float32),
        group_ext_x: wp.array(dtype=wp.float32),
        group_ext_y: wp.array(dtype=wp.float32),
        group_ext_z: wp.array(dtype=wp.float32),
        group_system: wp.array(dtype=wp.int32),
        group_ptr: wp.array(dtype=wp.int32),
        cell: wp.array(dtype=wp.mat33f),
        inv_cell: wp.array(dtype=wp.mat33f),
        cutoff_sq: wp.float32,
        ngroup: wp.int32,
        num_tiles: wp.array(dtype=wp.int32),
        tile_offsets: wp.array(dtype=wp.int32),
        tile_counts: wp.array(dtype=wp.int32),
        rebuild_flags: wp.array(dtype=wp.bool),
        tile_row_group: wp.array(dtype=wp.int32),
        tile_col_group: wp.array(dtype=wp.int32),
        tile_system: wp.array(dtype=wp.int32),
        max_tiles: wp.int32,
    ) -> None:
        """Build cluster-tile metadata

        Parameters
        ----------
        group_ctr_x, group_ctr_y, group_ctr_z : wp.array, shape (ngroup,), dtype=wp.float32
            Group center coordinates.
        group_ext_x, group_ext_y, group_ext_z : wp.array, shape (ngroup,), dtype=wp.float32
            Group half-extent coordinates.
        group_system : wp.array, shape (ngroup,), dtype=wp.int32
            System index per group for batched mode. Sentinel in single-system mode.
        group_ptr : wp.array, shape (num_systems + 1,), dtype=wp.int32
            Group prefix offsets per system for batched mode.
        cell, inv_cell : wp.array, shape (num_systems,), dtype=wp.mat33f
            Cell and inverse-cell matrices.
        cutoff_sq : wp.float32
            Squared cutoff distance.
        ngroup : wp.int32
            Total number of groups.
        num_tiles : wp.array, shape (1,), dtype=wp.int32
            OUTPUT: Compact tile count for non-segmented launches.
        tile_offsets : wp.array, shape (num_systems + 1,), dtype=wp.int32
            Caller-owned fixed output offsets for segmented batched launches.
        tile_counts : wp.array, shape (num_systems,), dtype=wp.int32
            OUTPUT: Per-system tile counts for segmented batched launches.
        rebuild_flags : wp.array, shape (num_systems,), dtype=wp.bool
            Per-system selective rebuild flags. Sentinel when inactive.
        tile_row_group, tile_col_group : wp.array, shape (max_tiles,), dtype=wp.int32
            OUTPUT: Row and column group index for each emitted tile.
        tile_system : wp.array, shape (max_tiles,), dtype=wp.int32
            OUTPUT: System index for each emitted tile in batched mode.
        max_tiles : wp.int32
            Compact tile capacity for non-segmented launches.

        Returns
        -------
        None
            This function modifies the input arrays in-place.

        Notes
        -----
        - Thread launch: Tiled launch with one block per row group and block dimension ``TILE_GROUP_SIZE``.
        - Modifies: ``num_tiles`` or ``tile_counts`` plus ``tile_row_group``, ``tile_col_group``, and ``tile_system``.
        ``BATCHED``, ``SEGMENTED``, and ``SELECTIVE`` are static specializations. Segmented
        batched mode writes each system into ``[tile_offsets[isys],
        tile_offsets[isys + 1])`` using ``tile_counts[isys]`` as the local
        atomic counter.

        See Also
        --------
        build_cluster_tile_list : Launch cluster-tile list construction.
        """
        row_group = wp.tid()
        lane_tile = wp.tile_arange(TILE, dtype=wp.int32)
        lane = wp.untile(lane_tile)

        system_idx = wp.int32(0)
        col_limit = ngroup
        if BATCHED:
            system_idx = group_system[row_group]
            col_limit = group_ptr[system_idx + 1]

        if SELECTIVE:
            if not rebuild_flags[system_idx]:
                return

        cell_mat = cell[system_idx]
        inv_cell_mat = inv_cell[system_idx]

        rg_ctr = wp.vec3f(
            group_ctr_x[row_group],
            group_ctr_y[row_group],
            group_ctr_z[row_group],
        )
        rg_ext = wp.vec3f(
            group_ext_x[row_group],
            group_ext_y[row_group],
            group_ext_z[row_group],
        )

        col_start_aligned = (row_group // TILE) * TILE
        for col_start in range(col_start_aligned, col_limit, TILE):
            cg_ctr_x_tile = wp.tile_load(group_ctr_x, shape=TILE, offset=col_start)
            cg_ctr_y_tile = wp.tile_load(group_ctr_y, shape=TILE, offset=col_start)
            cg_ctr_z_tile = wp.tile_load(group_ctr_z, shape=TILE, offset=col_start)
            cg_ext_x_tile = wp.tile_load(group_ext_x, shape=TILE, offset=col_start)
            cg_ext_y_tile = wp.tile_load(group_ext_y, shape=TILE, offset=col_start)
            cg_ext_z_tile = wp.tile_load(group_ext_z, shape=TILE, offset=col_start)

            cg = col_start + lane
            cg_ctr = wp.vec3f(
                wp.untile(cg_ctr_x_tile),
                wp.untile(cg_ctr_y_tile),
                wp.untile(cg_ctr_z_tile),
            )
            cg_ext = wp.vec3f(
                wp.untile(cg_ext_x_tile),
                wp.untile(cg_ext_y_tile),
                wp.untile(cg_ext_z_tile),
            )

            valid = _bbox_valid(
                cg,
                row_group,
                col_limit,
                rg_ctr,
                rg_ext,
                cg_ctr,
                cg_ext,
                cell_mat,
                inv_cell_mat,
                cutoff_sq,
            )

            valid_tile = wp.tile(valid)
            scan_tile = wp.tile_scan_inclusive(valid_tile)
            tile_total = wp.tile_extract(scan_tile, TILE - 1)

            if tile_total > 0:
                base = wp.int32(0)
                capacity = max_tiles
                offset = wp.int32(0)
                if SEGMENTED:
                    capacity = tile_offsets[system_idx + 1] - tile_offsets[system_idx]
                    offset = tile_offsets[system_idx]
                    if lane == 0:
                        base = wp.atomic_add(tile_counts, system_idx, tile_total)
                else:
                    if lane == 0:
                        base = wp.atomic_add(num_tiles, 0, tile_total)
                base_tile = wp.tile_from_thread(
                    shape=TILE,
                    value=base,
                    thread_idx=0,
                    storage="shared",
                )
                base_b = wp.untile(base_tile)
                scan_v = wp.untile(scan_tile)
                local_slot = base_b + scan_v - 1
                slot = offset + local_slot

                if valid == 1 and local_slot < capacity:
                    tile_row_group[slot] = row_group
                    tile_col_group[slot] = cg
                    if BATCHED:
                        tile_system[slot] = system_idx

    name = kernel_specialization_name(
        "_build_cluster_tiles",
        wp_dtype=wp.float32,
        features=(
            "batched" if batched else "",
            "segmented" if segmented else "",
            "selective" if selective else "",
        ),
    )
    return set_fn_doc(
        set_fn_name(_kernel, name),
        _append_specialization_doc(
            _kernel.__doc__,
            dtype=wp.float32,
            entries=(
                ("operation", "build_cluster_tiles"),
                ("batched", bool(batched)),
                ("segmented", bool(segmented)),
                ("selective", bool(selective)),
            ),
        ),
    )


@lru_cache(maxsize=None)
def _make_query_cluster_tile_kernel(
    *,
    batched: bool,
    tile_segmented: bool,
    selective: bool,
    dual_cutoff: bool,
    return_vectors: bool,
    return_distances: bool,
    pair_fn: wp.Function | None,
) -> wp.Kernel:
    """Build the tile-to-matrix kernel for one batching/feature combination."""
    BATCHED = wp.constant(bool(batched))
    TILE_SEGMENTED = wp.constant(bool(tile_segmented))
    SELECTIVE = wp.constant(bool(selective))
    DUAL_CUTOFF = wp.constant(bool(dual_cutoff))
    RETURN_VECTORS = wp.constant(bool(return_vectors))
    RETURN_DISTANCES = wp.constant(bool(return_distances))
    HAS_PAIR_FN = wp.constant(pair_fn is not None)

    @wp.kernel(enable_backward=False, module="unique")
    def _kernel(
        sorted_pos_x: wp.array(dtype=wp.float32),
        sorted_pos_y: wp.array(dtype=wp.float32),
        sorted_pos_z: wp.array(dtype=wp.float32),
        sorted_atom_index: wp.array(dtype=wp.int32),
        cell: wp.array(dtype=wp.mat33f),
        inv_cell: wp.array(dtype=wp.mat33f),
        cutoff_sq: wp.float32,
        cutoff_sq2: wp.float32,
        natom: wp.int32,
        max_neighbors: wp.int32,
        max_neighbors2: wp.int32,
        tile_row_group: wp.array(dtype=wp.int32),
        tile_col_group: wp.array(dtype=wp.int32),
        tile_system: wp.array(dtype=wp.int32),
        num_tiles: wp.array(dtype=wp.int32),
        tile_offsets: wp.array(dtype=wp.int32),
        tile_counts: wp.array(dtype=wp.int32),
        rebuild_flags: wp.array(dtype=wp.bool),
        neighbor_matrix: wp.array2d(dtype=wp.int32),
        num_neighbors: wp.array(dtype=wp.int32),
        neighbor_matrix_shifts: wp.array3d(dtype=wp.int32),
        neighbor_matrix2: wp.array2d(dtype=wp.int32),
        num_neighbors2: wp.array(dtype=wp.int32),
        neighbor_matrix_shifts2: wp.array3d(dtype=wp.int32),
        neighbor_vectors: wp.array2d(dtype=wp.vec3f),
        neighbor_distances: wp.array2d(dtype=wp.float32),
        pair_params: wp.array2d(dtype=wp.float32),
        pair_energies: wp.array2d(dtype=wp.float32),
        pair_forces: wp.array2d(dtype=wp.vec3f),
    ) -> None:
        """Fill cluster-tile neighbor-matrix rows

        Parameters
        ----------
        sorted_pos_x, sorted_pos_y, sorted_pos_z : wp.array, shape (natom_padded,), dtype=wp.float32
            Sorted atom coordinates.
        sorted_atom_index : wp.array, shape (natom_padded,), dtype=wp.int32
            Original atom index for each sorted slot.
        cell, inv_cell : wp.array, shape (num_systems,), dtype=wp.mat33f
            Cell and inverse-cell matrices.
        cutoff_sq, cutoff_sq2 : wp.float32
            Squared primary and secondary cutoffs. ``cutoff_sq2`` is read only
            in dual-cutoff specializations.
        natom : wp.int32
            Number of real atoms before padding.
        max_neighbors, max_neighbors2 : wp.int32
            Maximum row capacities for the primary and secondary matrices.
        tile_row_group, tile_col_group, tile_system : wp.array, shape (num_tiles,), dtype=wp.int32
            Cluster-tile records from the build launcher.
        num_tiles : wp.array, shape (1,), dtype=wp.int32
            Compact tile count for non-segmented tile lists.
        tile_offsets : wp.array, shape (num_systems + 1,), dtype=wp.int32
            Fixed per-system tile offsets for segmented tile lists.
        tile_counts : wp.array, shape (num_systems,), dtype=wp.int32
            Per-system tile counts for segmented tile lists.
        rebuild_flags : wp.array, shape (num_systems,), dtype=wp.bool
            Per-system selective rebuild flags. Sentinel when inactive.
        neighbor_matrix, neighbor_matrix2 : wp.array, shape (natom, max_neighbors*), dtype=wp.int32
            OUTPUT: Primary and optional secondary neighbor atom indices.
        num_neighbors, num_neighbors2 : wp.array, shape (natom,), dtype=wp.int32
            OUTPUT: Primary and optional secondary row counts, updated atomically.
        neighbor_matrix_shifts, neighbor_matrix_shifts2 : wp.array, shape (natom, max_neighbors*, 3), dtype=wp.int32
            OUTPUT: Periodic shift vectors for each emitted neighbor.
        neighbor_vectors : wp.array, shape (natom, max_neighbors), dtype=wp.vec3f
            OUTPUT: Optional primary displacement vectors. Sentinel when disabled.
        neighbor_distances : wp.array, shape (natom, max_neighbors), dtype=wp.float32
            OUTPUT: Optional primary pair distances. Sentinel when disabled.
        pair_params : wp.array, shape (natom, K), dtype=wp.float32
            Pair-function parameters. Sentinel when no pair function is active.
        pair_energies : wp.array, shape (natom, max_neighbors), dtype=wp.float32
            OUTPUT: Optional primary pair-function energies. Sentinel when disabled.
        pair_forces : wp.array, shape (natom, max_neighbors), dtype=wp.vec3f
            OUTPUT: Optional primary pair-function forces. Sentinel when disabled.

        Returns
        -------
        None
            This function modifies the input arrays in-place.

        Notes
        -----
        - Thread launch: Tiled launch with one block per allocated tile slot and block dimension ``TILE_GROUP_SIZE``.
        - Modifies: neighbor matrices, neighbor counts, shift matrices, and enabled pair-output buffers.
        Batched, segmented-tile, selective, dual-cutoff, and pair-output modes
        are static specializations. Selective kernels skip unflagged systems;
        their row counters must be reset by a separate pre-pass.

        See Also
        --------
        query_cluster_tile : Launch cluster-tile neighbor search with matrix-style outputs.
        """
        tile = wp.tid()
        system_idx = wp.int32(0)

        if TILE_SEGMENTED:
            if BATCHED:
                system_idx = tile_system[tile]
            local_tile = tile - tile_offsets[system_idx]
            # Bound to the per-system region capacity (mirrors the build's
            # ``local_slot < capacity`` write clamp).  ``tile_counts`` is
            # incremented unclamped, so on a tile-capacity overflow it can
            # exceed ``capacity``; without this clamp an over-counted system
            # plus zeroed ``tile_system`` gap slots can misattribute tiles
            # from a later region.
            capacity = tile_offsets[system_idx + 1] - tile_offsets[system_idx]
            if (
                local_tile < 0
                or local_tile >= tile_counts[system_idx]
                or local_tile >= capacity
            ):
                return
        else:
            if tile >= num_tiles[0]:
                return
            if BATCHED:
                system_idx = tile_system[tile]

        if SELECTIVE:
            if not rebuild_flags[system_idx]:
                return

        lane_tile = wp.tile_arange(TILE, dtype=wp.int32)
        lane = wp.untile(lane_tile)

        row_group = tile_row_group[tile]
        col_group = tile_col_group[tile]

        cell_mat = cell[system_idx]
        inv_cell_mat = inv_cell[system_idx]

        j_sorted = col_group * TILE + lane
        j_orig_t = sorted_atom_index[j_sorted]
        pj_x = sorted_pos_x[j_sorted]
        pj_y = sorted_pos_y[j_sorted]
        pj_z = sorted_pos_z[j_sorted]

        pi_x_tile = wp.tile_load(sorted_pos_x, shape=TILE, offset=row_group * TILE)
        pi_y_tile = wp.tile_load(sorted_pos_y, shape=TILE, offset=row_group * TILE)
        pi_z_tile = wp.tile_load(sorted_pos_z, shape=TILE, offset=row_group * TILE)
        i_orig_tile = wp.tile_load(
            sorted_atom_index, shape=TILE, offset=row_group * TILE
        )

        for i_local in range(TILE_GROUP_SIZE):
            i_sorted = row_group * TILE + i_local
            i_orig = wp.tile_extract(i_orig_tile, i_local)
            pi_x = wp.tile_extract(pi_x_tile, i_local)
            pi_y = wp.tile_extract(pi_y_tile, i_local)
            pi_z = wp.tile_extract(pi_z_tile, i_local)

            d = wp.vec3f(pj_x - pi_x, pj_y - pi_y, pj_z - pi_z)
            d_wrapped, s_shift = _wrap_triclinic(d, cell_mat, inv_cell_mat)
            dist_sq = (
                d_wrapped[0] * d_wrapped[0]
                + d_wrapped[1] * d_wrapped[1]
                + d_wrapped[2] * d_wrapped[2]
            )

            in_real_atoms = wp.bool(False)
            if BATCHED:
                if i_sorted < j_sorted and i_orig < natom and j_orig_t < natom:
                    in_real_atoms = True
            else:
                if i_sorted < j_sorted and i_sorted < natom and j_sorted < natom:
                    in_real_atoms = True

            valid = wp.int32(0)
            if in_real_atoms and dist_sq < cutoff_sq:
                valid = wp.int32(1)

            valid_tile = wp.tile(valid)
            scan_tile = wp.tile_scan_inclusive(valid_tile)
            row_count = wp.tile_extract(scan_tile, TILE - 1)

            if row_count > 0:
                base = wp.int32(0)
                if lane == 0:
                    base = wp.atomic_add(num_neighbors, i_orig, row_count)
                base_tile = wp.tile_from_thread(
                    shape=TILE,
                    value=base,
                    thread_idx=0,
                    storage="shared",
                )
                base_b = wp.untile(base_tile)
                scan_v = wp.untile(scan_tile)
                write_col = base_b + scan_v - 1

                if valid == 1 and write_col < max_neighbors:
                    neighbor_matrix[i_orig, write_col] = j_orig_t
                    neighbor_matrix_shifts[i_orig, write_col, 0] = s_shift[0]
                    neighbor_matrix_shifts[i_orig, write_col, 1] = s_shift[1]
                    neighbor_matrix_shifts[i_orig, write_col, 2] = s_shift[2]
                    if RETURN_VECTORS:
                        neighbor_vectors[i_orig, write_col] = d_wrapped
                    if RETURN_DISTANCES or HAS_PAIR_FN:
                        distance = wp.sqrt(dist_sq)
                        if RETURN_DISTANCES:
                            neighbor_distances[i_orig, write_col] = distance
                        if HAS_PAIR_FN:
                            pair_energy, pair_force = pair_fn(
                                d_wrapped,
                                distance,
                                pair_params,
                                i_orig,
                                j_orig_t,
                            )
                            pair_energies[i_orig, write_col] = pair_energy
                            pair_forces[i_orig, write_col] = pair_force

            # Full-fill: emit the reverse (j, i) pair into atom j's row with
            # negated shift/vector and the (j, i) pair-function result.  Each
            # unordered pair is visited once (i_sorted < j_sorted), so forward
            # plus reverse yields each ordered pair exactly once -- a symmetric
            # full matrix matching cell_list / naive (half_fill=False).
            if valid == 1:
                pos_j = wp.atomic_add(num_neighbors, j_orig_t, 1)
                if pos_j < max_neighbors:
                    neighbor_matrix[j_orig_t, pos_j] = i_orig
                    neighbor_matrix_shifts[j_orig_t, pos_j, 0] = -s_shift[0]
                    neighbor_matrix_shifts[j_orig_t, pos_j, 1] = -s_shift[1]
                    neighbor_matrix_shifts[j_orig_t, pos_j, 2] = -s_shift[2]
                    if RETURN_VECTORS:
                        neighbor_vectors[j_orig_t, pos_j] = -d_wrapped
                    if RETURN_DISTANCES or HAS_PAIR_FN:
                        distance_r = wp.sqrt(dist_sq)
                        if RETURN_DISTANCES:
                            neighbor_distances[j_orig_t, pos_j] = distance_r
                        if HAS_PAIR_FN:
                            pair_energy_r, pair_force_r = pair_fn(
                                -d_wrapped,
                                distance_r,
                                pair_params,
                                j_orig_t,
                                i_orig,
                            )
                            pair_energies[j_orig_t, pos_j] = pair_energy_r
                            pair_forces[j_orig_t, pos_j] = pair_force_r

            if DUAL_CUTOFF:
                valid2 = wp.int32(0)
                if in_real_atoms and dist_sq < cutoff_sq2:
                    valid2 = wp.int32(1)
                valid2_tile = wp.tile(valid2)
                scan2_tile = wp.tile_scan_inclusive(valid2_tile)
                row_count2 = wp.tile_extract(scan2_tile, TILE - 1)

                if row_count2 > 0:
                    base2 = wp.int32(0)
                    if lane == 0:
                        base2 = wp.atomic_add(num_neighbors2, i_orig, row_count2)
                    base2_tile = wp.tile_from_thread(
                        shape=TILE,
                        value=base2,
                        thread_idx=0,
                        storage="shared",
                    )
                    base2_b = wp.untile(base2_tile)
                    scan2_v = wp.untile(scan2_tile)
                    write_col2 = base2_b + scan2_v - 1

                    if valid2 == 1 and write_col2 < max_neighbors2:
                        neighbor_matrix2[i_orig, write_col2] = j_orig_t
                        neighbor_matrix_shifts2[i_orig, write_col2, 0] = s_shift[0]
                        neighbor_matrix_shifts2[i_orig, write_col2, 1] = s_shift[1]
                        neighbor_matrix_shifts2[i_orig, write_col2, 2] = s_shift[2]

                # Full-fill reverse for the secondary (cutoff2) matrix.
                if valid2 == 1:
                    pos_j2 = wp.atomic_add(num_neighbors2, j_orig_t, 1)
                    if pos_j2 < max_neighbors2:
                        neighbor_matrix2[j_orig_t, pos_j2] = i_orig
                        neighbor_matrix_shifts2[j_orig_t, pos_j2, 0] = -s_shift[0]
                        neighbor_matrix_shifts2[j_orig_t, pos_j2, 1] = -s_shift[1]
                        neighbor_matrix_shifts2[j_orig_t, pos_j2, 2] = -s_shift[2]

    name = kernel_specialization_name(
        "_query_cluster_tile",
        wp_dtype=wp.float32,
        features=(
            "batched" if batched else "",
            "tile_segmented" if tile_segmented else "",
            *_pair_output_features(
                selective=selective,
                dual_cutoff=dual_cutoff,
                return_vectors=return_vectors,
                return_distances=return_distances,
                pair_fn=pair_fn,
            ),
        ),
    )
    return set_fn_doc(
        set_fn_name(_kernel, name),
        _append_specialization_doc(
            _kernel.__doc__,
            dtype=wp.float32,
            entries=(
                ("operation", "query_cluster_tile"),
                ("output", "matrix"),
                ("batched", bool(batched)),
                ("tile_segmented", bool(tile_segmented)),
                ("selective", bool(selective)),
                ("dual_cutoff", bool(dual_cutoff)),
                ("return_vectors", bool(return_vectors)),
                ("return_distances", bool(return_distances)),
                ("pair_fn", pair_fn is not None),
            ),
        ),
    )


def get_query_cluster_tile_kernel(
    *,
    tile_segmented: bool = False,
    selective: bool = False,
    dual_cutoff: bool = False,
    return_vectors: bool = False,
    return_distances: bool = False,
    pair_fn: wp.Function | None = None,
) -> wp.Kernel:
    """Return a cached single-system tile-to-matrix kernel.

    The cache key stores the Warp function object directly, so module-scope
    ``@wp.func`` singletons reuse the same compiled kernel across calls.
    """
    return _make_query_cluster_tile_kernel(
        batched=False,
        tile_segmented=bool(tile_segmented),
        selective=bool(selective),
        dual_cutoff=bool(dual_cutoff),
        return_vectors=bool(return_vectors),
        return_distances=bool(return_distances),
        pair_fn=pair_fn,
    )


def get_batch_query_cluster_tile_kernel(
    *,
    tile_segmented: bool = False,
    selective: bool = False,
    dual_cutoff: bool = False,
    return_vectors: bool = False,
    return_distances: bool = False,
    pair_fn: wp.Function | None = None,
) -> wp.Kernel:
    """Return a cached batched tile-to-matrix kernel."""
    return _make_query_cluster_tile_kernel(
        batched=True,
        tile_segmented=bool(tile_segmented),
        selective=bool(selective),
        dual_cutoff=bool(dual_cutoff),
        return_vectors=bool(return_vectors),
        return_distances=bool(return_distances),
        pair_fn=pair_fn,
    )


@lru_cache(maxsize=None)
def _make_query_cluster_tile_coo_kernel(
    *,
    batched: bool,
    tile_segmented: bool = False,
    coo_segmented: bool = False,
    selective: bool = False,
    return_vectors: bool = False,
    return_distances: bool = False,
    pair_fn: wp.Function | None = None,
) -> wp.Kernel:
    """Build the tile-to-COO conversion kernel for one feature combination."""
    BATCHED = wp.constant(bool(batched))
    TILE_SEGMENTED = wp.constant(bool(tile_segmented))
    COO_SEGMENTED = wp.constant(bool(coo_segmented))
    SELECTIVE = wp.constant(bool(selective))
    RETURN_VECTORS = wp.constant(bool(return_vectors))
    RETURN_DISTANCES = wp.constant(bool(return_distances))
    HAS_PAIR_FN = wp.constant(pair_fn is not None)

    @wp.kernel(enable_backward=False, module="unique")
    def _kernel(
        sorted_pos_x: wp.array(dtype=wp.float32),
        sorted_pos_y: wp.array(dtype=wp.float32),
        sorted_pos_z: wp.array(dtype=wp.float32),
        sorted_atom_index: wp.array(dtype=wp.int32),
        cell: wp.array(dtype=wp.mat33f),
        inv_cell: wp.array(dtype=wp.mat33f),
        cutoff_sq: wp.float32,
        natom: wp.int32,
        max_pairs: wp.int32,
        tile_row_group: wp.array(dtype=wp.int32),
        tile_col_group: wp.array(dtype=wp.int32),
        tile_system: wp.array(dtype=wp.int32),
        num_tiles: wp.array(dtype=wp.int32),
        tile_offsets: wp.array(dtype=wp.int32),
        tile_counts: wp.array(dtype=wp.int32),
        rebuild_flags: wp.array(dtype=wp.bool),
        pair_counter: wp.array(dtype=wp.int32),
        pair_offsets: wp.array(dtype=wp.int32),
        pair_counts: wp.array(dtype=wp.int32),
        coo_list: wp.array2d(dtype=wp.int32),
        coo_shifts: wp.array2d(dtype=wp.int32),
        neighbor_vectors: wp.array(dtype=wp.vec3f),
        neighbor_distances: wp.array(dtype=wp.float32),
        pair_params: wp.array2d(dtype=wp.float32),
        pair_energies: wp.array(dtype=wp.float32),
        pair_forces: wp.array(dtype=wp.vec3f),
    ) -> None:
        """Convert cluster-tile pairs into COO rows

        Parameters
        ----------
        sorted_pos_x, sorted_pos_y, sorted_pos_z : wp.array, shape (natom_padded,), dtype=wp.float32
            Sorted atom coordinates.
        sorted_atom_index : wp.array, shape (natom_padded,), dtype=wp.int32
            Original atom index for each sorted slot.
        cell, inv_cell : wp.array, shape (num_systems,), dtype=wp.mat33f
            Cell and inverse-cell matrices.
        cutoff_sq : wp.float32
            Squared cutoff distance.
        natom : wp.int32
            Number of real atoms before padding.
        max_pairs : wp.int32
            Compact COO capacity. Sentinel in segmented COO specializations.
        tile_row_group, tile_col_group, tile_system : wp.array, shape (num_tiles,), dtype=wp.int32
            Cluster-tile records from the build launcher.
        num_tiles : wp.array, shape (1,), dtype=wp.int32
            Compact tile count for non-segmented tile lists.
        tile_offsets : wp.array, shape (num_systems + 1,), dtype=wp.int32
            Fixed per-system tile offsets for segmented tile lists.
        tile_counts : wp.array, shape (num_systems,), dtype=wp.int32
            Per-system tile counts for segmented tile lists.
        rebuild_flags : wp.array, shape (num_systems,), dtype=wp.bool
            Per-system selective rebuild flags. Sentinel when inactive.
        pair_counter : wp.array, shape (1,), dtype=wp.int32
            OUTPUT: Compact COO pair count. Sentinel in segmented COO mode.
        pair_offsets : wp.array, shape (num_systems + 1,), dtype=wp.int32
            Caller-owned fixed output offsets for segmented COO mode.
        pair_counts : wp.array, shape (num_systems,), dtype=wp.int32
            OUTPUT: Per-system pair counts for segmented COO mode.
        coo_list : wp.array, shape (max_pairs, 2), dtype=wp.int32
            OUTPUT: COO atom-index pairs.
        coo_shifts : wp.array, shape (max_pairs, 3), dtype=wp.int32
            OUTPUT: Periodic shift vectors for COO pairs.
        neighbor_vectors : wp.array, shape (max_pairs,), dtype=wp.vec3f
            OUTPUT: Optional displacement vectors. Sentinel when disabled.
        neighbor_distances : wp.array, shape (max_pairs,), dtype=wp.float32
            OUTPUT: Optional pair distances. Sentinel when disabled.
        pair_params : wp.array, shape (natom, K), dtype=wp.float32
            Pair-function parameters. Sentinel when no pair function is active.
        pair_energies : wp.array, shape (max_pairs,), dtype=wp.float32
            OUTPUT: Optional pair-function energies. Sentinel when disabled.
        pair_forces : wp.array, shape (max_pairs,), dtype=wp.vec3f
            OUTPUT: Optional pair-function forces. Sentinel when disabled.

        Returns
        -------
        None
            This function modifies the input arrays in-place.

        Notes
        -----
        - Thread launch: Tiled launch with one block per allocated tile slot and block dimension ``TILE_GROUP_SIZE``.
        - Modifies: ``pair_counter`` or ``pair_counts``, ``coo_list``, ``coo_shifts``, and enabled pair-output buffers.
        Segmented COO writes use ``local = atomic_add(pair_counts, isys,
        count)`` and write only if ``local < pair_offsets[isys + 1] -
        pair_offsets[isys]``; overflow is reported by the wrapper from the
        final count.

        See Also
        --------
        query_cluster_tile_coo : Launch cluster-tile neighbor search with COO-style outputs.
        """
        tile = wp.tid()
        system_idx = wp.int32(0)

        if TILE_SEGMENTED:
            if BATCHED:
                system_idx = tile_system[tile]
            local_tile = tile - tile_offsets[system_idx]
            # Bound to the per-system region capacity (mirrors the build's
            # ``local_slot < capacity`` write clamp).  ``tile_counts`` is
            # incremented unclamped, so on a tile-capacity overflow it can
            # exceed ``capacity``; without this clamp an over-counted system
            # plus zeroed ``tile_system`` gap slots can misattribute tiles
            # from a later region.
            capacity = tile_offsets[system_idx + 1] - tile_offsets[system_idx]
            if (
                local_tile < 0
                or local_tile >= tile_counts[system_idx]
                or local_tile >= capacity
            ):
                return
        else:
            if tile >= num_tiles[0]:
                return
            if BATCHED:
                system_idx = tile_system[tile]

        if SELECTIVE:
            if not rebuild_flags[system_idx]:
                return

        lane_tile = wp.tile_arange(TILE, dtype=wp.int32)
        lane = wp.untile(lane_tile)

        row_group = tile_row_group[tile]
        col_group = tile_col_group[tile]

        cell_mat = cell[system_idx]
        inv_cell_mat = inv_cell[system_idx]

        j_sorted = col_group * TILE + lane
        j_orig_t = sorted_atom_index[j_sorted]
        pj_x = sorted_pos_x[j_sorted]
        pj_y = sorted_pos_y[j_sorted]
        pj_z = sorted_pos_z[j_sorted]

        pi_x_tile = wp.tile_load(sorted_pos_x, shape=TILE, offset=row_group * TILE)
        pi_y_tile = wp.tile_load(sorted_pos_y, shape=TILE, offset=row_group * TILE)
        pi_z_tile = wp.tile_load(sorted_pos_z, shape=TILE, offset=row_group * TILE)
        i_orig_tile = wp.tile_load(
            sorted_atom_index, shape=TILE, offset=row_group * TILE
        )

        for i_local in range(TILE_GROUP_SIZE):
            i_sorted = row_group * TILE + i_local
            i_orig = wp.tile_extract(i_orig_tile, i_local)
            pi_x = wp.tile_extract(pi_x_tile, i_local)
            pi_y = wp.tile_extract(pi_y_tile, i_local)
            pi_z = wp.tile_extract(pi_z_tile, i_local)

            d = wp.vec3f(pj_x - pi_x, pj_y - pi_y, pj_z - pi_z)
            d_wrapped, s_shift = _wrap_triclinic(d, cell_mat, inv_cell_mat)
            dist_sq = (
                d_wrapped[0] * d_wrapped[0]
                + d_wrapped[1] * d_wrapped[1]
                + d_wrapped[2] * d_wrapped[2]
            )

            valid = wp.int32(0)
            if BATCHED:
                if (
                    i_sorted < j_sorted
                    and i_orig < natom
                    and j_orig_t < natom
                    and dist_sq < cutoff_sq
                ):
                    valid = wp.int32(1)
            else:
                if (
                    i_sorted < j_sorted
                    and i_sorted < natom
                    and j_sorted < natom
                    and dist_sq < cutoff_sq
                ):
                    valid = wp.int32(1)

            valid_tile = wp.tile(valid)
            scan_tile = wp.tile_scan_inclusive(valid_tile)
            i_count = wp.tile_extract(scan_tile, TILE - 1)

            if i_count > 0:
                base = wp.int32(0)
                capacity = max_pairs
                offset = wp.int32(0)
                # Full-fill: reserve two slots per valid pair (forward (i, j)
                # and reverse (j, i)), so the COO list carries each ordered
                # pair once -- matching cell_list / naive (half_fill=False).
                if COO_SEGMENTED:
                    capacity = pair_offsets[system_idx + 1] - pair_offsets[system_idx]
                    offset = pair_offsets[system_idx]
                    if lane == 0:
                        base = wp.atomic_add(pair_counts, system_idx, 2 * i_count)
                else:
                    if lane == 0:
                        base = wp.atomic_add(pair_counter, 0, 2 * i_count)
                base_tile = wp.tile_from_thread(
                    shape=TILE,
                    value=base,
                    thread_idx=0,
                    storage="shared",
                )
                base_b = wp.untile(base_tile)
                scan_v = wp.untile(scan_tile)
                local_pos = base_b + scan_v - 1
                local_rev = base_b + i_count + scan_v - 1
                write_pos = offset + local_pos
                write_rev = offset + local_rev

                if valid == 1 and local_pos < capacity:
                    coo_list[write_pos, 0] = i_orig
                    coo_list[write_pos, 1] = j_orig_t
                    coo_shifts[write_pos, 0] = s_shift[0]
                    coo_shifts[write_pos, 1] = s_shift[1]
                    coo_shifts[write_pos, 2] = s_shift[2]
                    if RETURN_VECTORS:
                        neighbor_vectors[write_pos] = d_wrapped
                    if RETURN_DISTANCES or HAS_PAIR_FN:
                        distance = wp.sqrt(dist_sq)
                        if RETURN_DISTANCES:
                            neighbor_distances[write_pos] = distance
                        if HAS_PAIR_FN:
                            pair_energy, pair_force = pair_fn(
                                d_wrapped,
                                distance,
                                pair_params,
                                i_orig,
                                j_orig_t,
                            )
                            pair_energies[write_pos] = pair_energy
                            pair_forces[write_pos] = pair_force

                if valid == 1 and local_rev < capacity:
                    coo_list[write_rev, 0] = j_orig_t
                    coo_list[write_rev, 1] = i_orig
                    coo_shifts[write_rev, 0] = -s_shift[0]
                    coo_shifts[write_rev, 1] = -s_shift[1]
                    coo_shifts[write_rev, 2] = -s_shift[2]
                    if RETURN_VECTORS:
                        neighbor_vectors[write_rev] = -d_wrapped
                    if RETURN_DISTANCES or HAS_PAIR_FN:
                        distance_r = wp.sqrt(dist_sq)
                        if RETURN_DISTANCES:
                            neighbor_distances[write_rev] = distance_r
                        if HAS_PAIR_FN:
                            pair_energy_r, pair_force_r = pair_fn(
                                -d_wrapped,
                                distance_r,
                                pair_params,
                                j_orig_t,
                                i_orig,
                            )
                            pair_energies[write_rev] = pair_energy_r
                            pair_forces[write_rev] = pair_force_r

    name = kernel_specialization_name(
        "_query_cluster_tile_coo",
        wp_dtype=wp.float32,
        features=(
            "batched" if batched else "",
            "tile_segmented" if tile_segmented else "",
            *_pair_output_features(
                selective=selective,
                coo_segmented=coo_segmented,
                return_vectors=return_vectors,
                return_distances=return_distances,
                pair_fn=pair_fn,
            ),
        ),
    )
    return set_fn_doc(
        set_fn_name(_kernel, name),
        _append_specialization_doc(
            _kernel.__doc__,
            dtype=wp.float32,
            entries=(
                ("operation", "query_cluster_tile_coo"),
                ("output", "coo"),
                ("batched", bool(batched)),
                ("tile_segmented", bool(tile_segmented)),
                ("coo_segmented", bool(coo_segmented)),
                ("selective", bool(selective)),
                ("return_vectors", bool(return_vectors)),
                ("return_distances", bool(return_distances)),
                ("pair_fn", pair_fn is not None),
            ),
        ),
    )


def get_query_cluster_tile_coo_kernel(
    *,
    tile_segmented: bool = False,
    coo_segmented: bool = False,
    selective: bool = False,
    return_vectors: bool = False,
    return_distances: bool = False,
    pair_fn: wp.Function | None = None,
) -> wp.Kernel:
    """Return a cached single-system tile-to-COO kernel."""
    return _make_query_cluster_tile_coo_kernel(
        batched=False,
        tile_segmented=bool(tile_segmented),
        coo_segmented=bool(coo_segmented),
        selective=bool(selective),
        return_vectors=bool(return_vectors),
        return_distances=bool(return_distances),
        pair_fn=pair_fn,
    )


def get_batch_query_cluster_tile_coo_kernel(
    *,
    tile_segmented: bool = False,
    coo_segmented: bool = False,
    selective: bool = False,
    return_vectors: bool = False,
    return_distances: bool = False,
    pair_fn: wp.Function | None = None,
) -> wp.Kernel:
    """Return a cached batched tile-to-COO kernel."""
    return _make_query_cluster_tile_coo_kernel(
        batched=True,
        tile_segmented=bool(tile_segmented),
        coo_segmented=bool(coo_segmented),
        selective=bool(selective),
        return_vectors=bool(return_vectors),
        return_distances=bool(return_distances),
        pair_fn=pair_fn,
    )
