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


"""Single-system cluster-pair tile neighbor-list launchers.

GROMACS-style NxM cluster pair list: Morton-sorted atoms grouped into
32-atom clusters. This module contains the public Warp-facing launchers;
compiled kernels and factories live in :mod:`nvalchemiops.neighbors.cluster_tile.kernels`.
"""

import math

import warp as wp

from nvalchemiops.neighbors.cluster_tile.kernels import (
    TILE_GROUP_SIZE,
    _compute_morton_kernel,
    _get_build_cluster_tiles_kernel,
    _get_reset_cluster_tile_counts_kernel,
    _get_tile_sort_specialization,
    _group_buffer,
    _group_scratch_size,
    _permute_gather_soa_kernel,
    _rank_to_group_kernel,
    _require_f32,
    get_batch_query_cluster_tile_coo_kernel,
    get_batch_query_cluster_tile_kernel,
    get_query_cluster_tile_coo_kernel,
    get_query_cluster_tile_kernel,
)
from nvalchemiops.neighbors.neighbor_utils import empty_sentinel as _empty_sentinel
from nvalchemiops.neighbors.output_args import (
    _prepare_coo_pair_output_args,
    _prepare_pair_output_args,
)

__all__ = [
    "TILE_GROUP_SIZE",
    "estimate_max_tiles_per_group",
    "estimate_batch_cluster_tile_segments",
    "batch_query_cluster_tile_coo",
    "batch_query_cluster_tile",
    "batch_build_cluster_tile_list",
    "build_cluster_tile_list",
    "get_batch_query_cluster_tile_kernel",
    "get_query_cluster_tile_kernel",
    "query_cluster_tile_coo",
    "query_cluster_tile",
]


def estimate_max_tiles_per_group(
    total_atoms: int,
    cutoff: float,
    cell_volume: float | None,
    *,
    safety: float = 2.0,
    floor: int = 256,
) -> int:
    """Estimate neighbor cluster-groups per row group from density and cutoff.

    The tile-list capacity is ``ngroup * min(ngroup, max_tiles_per_group)``.
    The fixed default (256) silently truncates dense / high-cutoff periodic
    systems (e.g. ammonia 4:5 at >=20 A, ``ngroup=1024``), so size it from the
    expected number of 32-atom clusters whose bounding box can fall within
    ``cutoff`` of a row group.  ``min(ngroup, ...)`` in the capacity formula
    clamps the per-row count, so over-estimates cost nothing; the ``floor``
    keeps parity with the old default for sparse / low-cutoff systems.

    Parameters
    ----------
    total_atoms : int
        Atom count for the system (single system, not batched total).
    cutoff : float
        Cartesian cutoff (use ``max(cutoff, cutoff2)`` for dual-cutoff).
    cell_volume : float or None
        ``abs(det(cell))``.  ``None`` / non-positive falls back to ``floor``.
    safety : float, default 2.0
        Multiplier on the volumetric estimate.
    floor : int, default 256
        Minimum returned value (the historical default).
    """
    ngroup = (int(total_atoms) + TILE_GROUP_SIZE - 1) // TILE_GROUP_SIZE
    if ngroup <= 1:
        return 1
    if cell_volume is None or cell_volume <= 0.0 or cutoff <= 0.0:
        return min(ngroup, floor)
    cluster_vol = float(cell_volume) / ngroup
    cluster_extent = cluster_vol ** (1.0 / 3.0)
    radius = float(cutoff) + 2.0 * cluster_extent
    n_neigh = (4.0 / 3.0) * math.pi * radius * radius * radius / cluster_vol
    return max(int(floor), int(math.ceil(safety * n_neigh)))


def estimate_batch_cluster_tile_segments(
    batch_ptr,
    max_neighbors: int,
    max_tiles_per_group: int = 256,
) -> tuple[list[int], list[int], list[int], list[int]]:
    """Estimate per-system tile and COO segment capacities.

    Parameters
    ----------
    batch_ptr : sequence of int
        CSR atom pointer with length ``num_systems + 1``.
    max_neighbors : int
        Per-atom COO capacity multiplier.
    max_tiles_per_group : int, default 256
        Upper bound on neighbor groups per row group.

    Returns
    -------
    tile_capacities, tile_offsets, pair_capacities, pair_offsets : list[int]
        Per-system capacities and exclusive prefix offsets. ``tile_offsets``
        and ``pair_offsets`` are caller-owned fixed inputs for segmented
        cluster-tile build and COO query launchers; ``tile_counts`` and
        ``pair_counts`` are separate output counters with length
        ``num_systems``.
    """
    if hasattr(batch_ptr, "detach"):
        values = [int(v) for v in batch_ptr.detach().cpu().tolist()]
    elif hasattr(batch_ptr, "tolist"):
        values = [int(v) for v in batch_ptr.tolist()]
    else:
        values = [int(v) for v in batch_ptr]
    if len(values) < 2:
        raise ValueError("batch_ptr must have length at least 2")
    if max_neighbors < 0:
        raise ValueError("max_neighbors must be >= 0")
    if max_tiles_per_group < 0:
        raise ValueError("max_tiles_per_group must be >= 0")

    tile_capacities: list[int] = []
    pair_capacities: list[int] = []
    for start, stop in zip(values[:-1], values[1:]):
        natom = int(stop) - int(start)
        if natom < 0:
            raise ValueError("batch_ptr must be non-decreasing")
        n_padded = ((natom + TILE_GROUP_SIZE - 1) // TILE_GROUP_SIZE) * TILE_GROUP_SIZE
        ngroup = n_padded // TILE_GROUP_SIZE
        tile_capacities.append(ngroup * min(ngroup, int(max_tiles_per_group)))
        pair_capacities.append(natom * int(max_neighbors))

    tile_offsets = [0]
    pair_offsets = [0]
    for capacity in tile_capacities:
        tile_offsets.append(tile_offsets[-1] + int(capacity))
    for capacity in pair_capacities:
        pair_offsets.append(pair_offsets[-1] + int(capacity))
    return tile_capacities, tile_offsets, pair_capacities, pair_offsets


def _reset_cluster_tile_counts(
    counts: wp.array,
    rebuild_flags: wp.array | None,
    device: str,
    *,
    selective: bool,
) -> None:
    """Reset compact or per-system cluster-tile counters before a launch."""
    count = int(counts.shape[0])
    if count <= 0:
        return
    flags_arg = (
        rebuild_flags
        if rebuild_flags is not None
        else _empty_sentinel(1, wp.bool, device)
    )
    wp.launch(
        kernel=_get_reset_cluster_tile_counts_kernel(selective=bool(selective)),
        dim=count,
        inputs=[counts, flags_arg],
        block_dim=TILE_GROUP_SIZE,
        device=device,
    )


def _require_all_or_none(name_a: str, value_a, name_b: str, value_b) -> bool:
    """Return True when both values are present, or raise for partial pairs."""
    present_a = value_a is not None
    present_b = value_b is not None
    if present_a != present_b:
        raise ValueError(f"Pass both {name_a!r} and {name_b!r}, or neither.")
    return present_a


def _compute_morton(
    positions: wp.array,
    inv_cell: wp.array,
    natom: int,
    morton_codes: wp.array,
    sorted_atom_index: wp.array,
    num_neighbors: wp.array,
    num_tiles: wp.array,
    device: str,
) -> None:
    """Compute Morton codes and initialize tile scratch.

    Parameters
    ----------
    positions : wp.array, shape ``(n_padded,)``, dtype=wp.vec3f
        Padded position array.
    inv_cell : wp.array, shape ``(1,)``, dtype=wp.mat33f
        Inverse cell matrix used to compute fractional coordinates.
    natom : int
        Real atom count before padding.
    morton_codes : wp.array, shape ``(n_padded,)``, dtype=wp.int32
        OUTPUT: per-atom Morton codes.
    sorted_atom_index : wp.array, shape ``(n_padded,)``, dtype=wp.int32
        OUTPUT: identity permutation consumed by the sort.
    num_neighbors : wp.array, shape ``(natom,)``, dtype=wp.int32
        OUTPUT: zeroed for downstream accumulation.
    num_tiles : wp.array, shape ``(1,)``, dtype=wp.int32
        OUTPUT: element 0 zeroed.
    device : str
        Warp device string.
    """
    wp.launch(
        kernel=_compute_morton_kernel,
        dim=int(morton_codes.shape[0]),
        inputs=[
            positions,
            inv_cell,
            int(natom),
            morton_codes,
            sorted_atom_index,
            num_neighbors,
            num_tiles,
        ],
        device=device,
    )


def _permute_gather_soa(
    positions: wp.array,
    sorted_atom_index: wp.array,
    natom: int,
    sorted_pos_x: wp.array,
    sorted_pos_y: wp.array,
    sorted_pos_z: wp.array,
    device: str,
) -> None:
    """Gather AoS positions into Morton-sorted SoA arrays.

    Parameters
    ----------
    positions : wp.array, shape ``(>=natom,)``, dtype=wp.vec3f
        Atomic positions in original order.
    sorted_atom_index : wp.array, shape ``(>=natom,)``, dtype=wp.int32
        Morton-sort permutation.
    natom : int
        Real atom count.
    sorted_pos_x, sorted_pos_y, sorted_pos_z : wp.array, shape ``(natom,)``, dtype=wp.float32
        OUTPUT arrays for sorted position components.
    device : str
        Warp device string.
    """
    wp.launch(
        kernel=_permute_gather_soa_kernel,
        dim=int(natom),
        inputs=[
            positions,
            sorted_atom_index,
            int(natom),
            sorted_pos_x,
            sorted_pos_y,
            sorted_pos_z,
        ],
        device=device,
    )


def _tile_sort_pairs(
    keys: wp.array,
    values: wp.array,
    natom: int,
    device: str,
) -> bool:
    """Sort key/value int32 pairs in-place for supported fixed sizes.

    Parameters
    ----------
    keys, values : wp.array, shape ``(>=natom,)``, dtype=wp.int32
        Key/value arrays sorted in-place.
    natom : int
        Number of leading entries to sort. Supported values are 1024 and 2048.
    device : str
        Warp device string.

    Returns
    -------
    bool
        ``True`` if a tile-sort specialization was launched, otherwise
        ``False`` so callers can fall back to radix sort.
    """
    spec = _get_tile_sort_specialization(int(natom))
    if spec is None:
        return False
    kernel, block_dim = spec
    wp.launch_tiled(
        kernel=kernel,
        dim=[1],
        inputs=[keys, values],
        block_dim=block_dim,
        device=device,
    )
    return True


def build_cluster_tile_list(
    sorted_pos_x: wp.array,
    sorted_pos_y: wp.array,
    sorted_pos_z: wp.array,
    cell: wp.array,
    inv_cell: wp.array,
    cutoff: float,
    num_tiles: wp.array,
    tile_row_group: wp.array,
    tile_col_group: wp.array,
    wp_dtype: type,
    device: str,
    *,
    group_ctr_x_buffer: wp.array | None = None,
    group_ctr_y_buffer: wp.array | None = None,
    group_ctr_z_buffer: wp.array | None = None,
    group_ext_x_buffer: wp.array | None = None,
    group_ext_y_buffer: wp.array | None = None,
    group_ext_z_buffer: wp.array | None = None,
    rebuild_flags: wp.array | None = None,
) -> None:
    """Enumerate cluster-tile pairs on pre-sorted positions.

    Walks Morton-sorted 32-atom clusters, computes per-group bounding boxes,
    and emits the tile (row-group, col-group) pairs whose boxes are within
    ``cutoff``. Cluster-tile kernels are CUDA float32 only.

    Parameters
    ----------
    sorted_pos_x, sorted_pos_y, sorted_pos_z : wp.array, shape (n_padded,), dtype=wp.float32
        Morton-sorted SoA positions. The padded length must be a multiple of
        ``TILE_GROUP_SIZE``.
    cell, inv_cell : wp.array, shape (1,), dtype=wp.mat33f
        Cell and inverse-cell matrices.
    cutoff : float
        Bounding-box filter cutoff in Cartesian units.
    num_tiles : wp.array, shape (1,), dtype=wp.int32
        OUTPUT: tile counter incremented atomically. Caller must zero
        before launch.
    tile_row_group, tile_col_group : wp.array, shape (max_tiles,), dtype=wp.int32
        OUTPUT: paired tile indices.
    wp_dtype : type
        Must be ``wp.float32``; cluster-tile kernels are float32-only.
    device : str
        Warp device string (e.g. ``"cuda:0"``).
    group_ctr_x_buffer, group_ctr_y_buffer, group_ctr_z_buffer : wp.array, optional
        Caller-owned per-group center-of-bbox scratch. Transient buffers are
        allocated when omitted.
    group_ext_x_buffer, group_ext_y_buffer, group_ext_z_buffer : wp.array, optional
        Caller-owned per-group bbox-extent scratch. Transient buffers are
        allocated when omitted.

    Returns
    -------
    None
        This function modifies the input arrays in-place:

        - ``num_tiles`` : counts emitted tile pairs.
        - ``tile_row_group``, ``tile_col_group`` : populated with paired
          tile indices for ``[0:num_tiles[0]]``.

    Notes
    -----
    - Thread launch: tiled over ``(ngroup,)`` with ``block_dim=TILE_GROUP_SIZE``.
    - Modifies: ``num_tiles``, ``tile_row_group``, ``tile_col_group``.
    - The caller is responsible for Morton-sorting positions before
      invoking this launcher. See the framework bindings under
      ``nvalchemiops.{jax,torch}.neighbors.cluster_tile`` for the full
      sort+build+query path.

    See Also
    --------
    query_cluster_tile : Consume tile pairs into a per-atom neighbor matrix.
    query_cluster_tile_coo : Consume tile pairs into a flat COO neighbor list.
    batch_build_cluster_tile_list : Batched companion launcher.
    """
    _require_f32(wp_dtype)
    ngroup = int(sorted_pos_x.shape[0]) // TILE_GROUP_SIZE
    max_tiles = int(tile_row_group.shape[0])
    scratch_size = _group_scratch_size(ngroup)

    group_ctr_x = _group_buffer(group_ctr_x_buffer, scratch_size, device)
    group_ctr_y = _group_buffer(group_ctr_y_buffer, scratch_size, device)
    group_ctr_z = _group_buffer(group_ctr_z_buffer, scratch_size, device)
    group_ext_x = _group_buffer(group_ext_x_buffer, scratch_size, device)
    group_ext_y = _group_buffer(group_ext_y_buffer, scratch_size, device)
    group_ext_z = _group_buffer(group_ext_z_buffer, scratch_size, device)

    wp.launch_tiled(
        kernel=_rank_to_group_kernel,
        dim=[ngroup],
        inputs=[
            sorted_pos_x,
            sorted_pos_y,
            sorted_pos_z,
            group_ctr_x,
            group_ctr_y,
            group_ctr_z,
            group_ext_x,
            group_ext_y,
            group_ext_z,
        ],
        block_dim=TILE_GROUP_SIZE,
        device=device,
    )

    int32_sentinel = _empty_sentinel(1, wp.int32, device)
    bool_sentinel = _empty_sentinel(1, wp.bool, device)
    selective = rebuild_flags is not None
    rebuild_flags_arg = rebuild_flags if rebuild_flags is not None else bool_sentinel
    if selective:
        _reset_cluster_tile_counts(num_tiles, rebuild_flags_arg, device, selective=True)
    wp.launch_tiled(
        kernel=_get_build_cluster_tiles_kernel(
            batched=False, segmented=False, selective=selective
        ),
        dim=[ngroup],
        inputs=[
            group_ctr_x,
            group_ctr_y,
            group_ctr_z,
            group_ext_x,
            group_ext_y,
            group_ext_z,
            int32_sentinel,
            int32_sentinel,
            cell,
            inv_cell,
            wp.float32(cutoff * cutoff),
            int(ngroup),
            num_tiles,
            int32_sentinel,
            int32_sentinel,
            rebuild_flags_arg,
            tile_row_group,
            tile_col_group,
            int32_sentinel,
            int(max_tiles),
        ],
        block_dim=TILE_GROUP_SIZE,
        device=device,
    )


def query_cluster_tile(
    sorted_atom_index: wp.array,
    sorted_pos_x: wp.array,
    sorted_pos_y: wp.array,
    sorted_pos_z: wp.array,
    num_tiles: wp.array,
    tile_row_group: wp.array,
    tile_col_group: wp.array,
    cell: wp.array,
    inv_cell: wp.array,
    cutoff: float,
    natom: int,
    neighbor_matrix: wp.array,
    num_neighbors: wp.array,
    neighbor_matrix_shifts: wp.array,
    wp_dtype: type,
    device: str,
    *,
    n_tiles: int | None = None,
    cutoff2: float | None = None,
    neighbor_matrix2: wp.array | None = None,
    num_neighbors2: wp.array | None = None,
    neighbor_matrix_shifts2: wp.array | None = None,
    rebuild_flags: wp.array | None = None,
    tile_offsets: wp.array | None = None,
    tile_counts: wp.array | None = None,
    return_vectors: bool = False,
    return_distances: bool = False,
    pair_fn: wp.Function | None = None,
    pair_params: wp.array | None = None,
    neighbor_vectors: wp.array | None = None,
    neighbor_distances: wp.array | None = None,
    pair_energies: wp.array | None = None,
    pair_forces: wp.array | None = None,
) -> None:
    """Convert cluster-tile pairs into a per-atom neighbor matrix.

    Iterates the tile pairs emitted by :func:`build_cluster_tile_list` and
    fills the per-atom neighbor matrix (plus shifts and optional pair-output
    buffers). Cluster-tile kernels are CUDA float32 only.

    Parameters
    ----------
    sorted_atom_index : wp.array, shape (n_padded,), dtype=wp.int32
        Morton-sort permutation mapping cluster slot to original atom index.
    sorted_pos_x, sorted_pos_y, sorted_pos_z : wp.array, shape (n_padded,), dtype=wp.float32
        Morton-sorted SoA positions.
    num_tiles : wp.array, shape (1,), dtype=wp.int32
        Tile counter populated by :func:`build_cluster_tile_list`.
    tile_row_group, tile_col_group : wp.array, shape (max_tiles,), dtype=wp.int32
        Paired tile indices populated by :func:`build_cluster_tile_list`.
    cell, inv_cell : wp.array, shape (1,), dtype=wp.mat33f
        Cell and inverse-cell matrices.
    cutoff : float
        Pair cutoff in Cartesian units.
    natom : int
        Real (unpadded) atom count.
    neighbor_matrix : wp.array, shape (natom, max_neighbors), dtype=wp.int32
        OUTPUT: per-atom neighbor indices.
    num_neighbors : wp.array, shape (natom,), dtype=wp.int32
        OUTPUT: per-atom neighbor counts. Caller must zero before launch.
    neighbor_matrix_shifts : wp.array, shape (natom, max_neighbors, 3), dtype=wp.int32
        OUTPUT: per-pair periodic shift vectors.
    wp_dtype : type
        Must be ``wp.float32``.
    device : str
        Warp device string (e.g. ``"cuda:0"``).
    return_vectors : bool, default False
        If True, write per-pair displacement vectors into ``neighbor_vectors``.
    return_distances : bool, default False
        If True, write per-pair distances into ``neighbor_distances``.
    pair_fn : wp.Function or None, optional
        Optional pair function; when set, ``pair_energies`` and ``pair_forces``
        are populated.
    pair_params : wp.array or None, optional
        Parameter table consumed by ``pair_fn``.
    neighbor_vectors : wp.array or None, optional
        OUTPUT (when ``return_vectors``): pair displacement vectors.
    neighbor_distances : wp.array or None, optional
        OUTPUT (when ``return_distances``): pair distances.
    pair_energies : wp.array or None, optional
        OUTPUT (when ``pair_fn`` is set): pair-function energies.
    pair_forces : wp.array or None, optional
        OUTPUT (when ``pair_fn`` is set): pair-function forces.

    Returns
    -------
    None
        This function modifies the input arrays in-place:

        - ``neighbor_matrix``, ``num_neighbors``, ``neighbor_matrix_shifts``
          are always written.
        - ``neighbor_vectors``, ``neighbor_distances``, ``pair_energies``,
          ``pair_forces`` are written when their enabling flag is set.

    Notes
    -----
    - Thread launch: tiled over the **allocated** ``tile_row_group``
      buffer with ``block_dim=TILE_GROUP_SIZE``.  Threads whose tile
      index exceeds ``num_tiles[0]`` (the actual emitted-tile count
      from :func:`build_cluster_tile_list`) early-return inside the
      kernel — this removes the host-side ``num_tiles.item()`` sync
      that wrappers used to do to set the launch dimension.
    - Modifies: ``neighbor_matrix``, ``num_neighbors``,
      ``neighbor_matrix_shifts``, and any enabled pair-output buffers.
    - Cluster-tile iterates emitted tile pairs rather than source atoms,
      so partial neighbor lists (``target_indices``) are not supported
      here. Use :func:`nvalchemiops.neighbors.cell_list.query_cell_list`
      or :func:`nvalchemiops.neighbors.naive.naive_neighbor_matrix` for
      partial neighbor lists.

    See Also
    --------
    build_cluster_tile_list : Emit the tile pairs consumed by this launcher.
    query_cluster_tile_coo : COO-format variant.
    batch_query_cluster_tile : Batched companion launcher.
    """

    _require_f32(wp_dtype)
    tile_capacity = int(tile_row_group.shape[0])
    # ``n_tiles`` (host-synced emitted-tile count) tightens the launch to the
    # real tiles; ``None`` falls back to the full buffer (the kernel's
    # ``tile >= num_tiles[0]`` guard still no-ops the surplus blocks).
    launch_tiles = (
        tile_capacity if n_tiles is None else max(0, min(int(n_tiles), tile_capacity))
    )
    if launch_tiles <= 0:
        return

    dual_cutoff = cutoff2 is not None
    has_pair_outputs = (
        bool(return_vectors) or bool(return_distances) or pair_fn is not None
    )
    if dual_cutoff and has_pair_outputs:
        raise ValueError(
            "cluster_tile cutoff2 is matrix-only and cannot be combined with pair outputs"
        )
    if dual_cutoff:
        if (
            neighbor_matrix2 is None
            or num_neighbors2 is None
            or neighbor_matrix_shifts2 is None
        ):
            raise ValueError(
                "neighbor_matrix2, num_neighbors2, and neighbor_matrix_shifts2 are required when cutoff2 is provided"
            )
    int32_1d_sentinel = _empty_sentinel(1, wp.int32, device)
    int32_2d_sentinel = _empty_sentinel(2, wp.int32, device)
    int32_3d_sentinel = _empty_sentinel(3, wp.int32, device)
    bool_sentinel = _empty_sentinel(1, wp.bool, device)
    tile_segmented = _require_all_or_none(
        "tile_offsets", tile_offsets, "tile_counts", tile_counts
    )
    selective = rebuild_flags is not None

    (
        vectors_arg,
        distances_arg,
        pair_params_arg,
        energies_arg,
        forces_arg,
    ) = _prepare_pair_output_args(
        wp.float32,
        device,
        return_vectors=return_vectors,
        return_distances=return_distances,
        pair_fn=pair_fn,
        pair_params=pair_params,
        neighbor_vectors=neighbor_vectors,
        neighbor_distances=neighbor_distances,
        pair_energies=pair_energies,
        pair_forces=pair_forces,
    )

    wp.launch_tiled(
        kernel=get_query_cluster_tile_kernel(
            tile_segmented=tile_segmented,
            selective=selective,
            dual_cutoff=dual_cutoff,
            return_vectors=bool(return_vectors),
            return_distances=bool(return_distances),
            pair_fn=pair_fn,
        ),
        dim=[launch_tiles],
        inputs=[
            sorted_pos_x,
            sorted_pos_y,
            sorted_pos_z,
            sorted_atom_index,
            cell,
            inv_cell,
            wp.float32(cutoff * cutoff),
            wp.float32(
                (cutoff2 if cutoff2 is not None else cutoff)
                * (cutoff2 if cutoff2 is not None else cutoff)
            ),
            int(natom),
            int(neighbor_matrix.shape[1]),
            int(neighbor_matrix2.shape[1]) if neighbor_matrix2 is not None else 0,
            tile_row_group,
            tile_col_group,
            _empty_sentinel(1, wp.int32, device),
            num_tiles,
            tile_offsets if tile_offsets is not None else int32_1d_sentinel,
            tile_counts if tile_counts is not None else int32_1d_sentinel,
            rebuild_flags if rebuild_flags is not None else bool_sentinel,
            neighbor_matrix,
            num_neighbors,
            neighbor_matrix_shifts,
            neighbor_matrix2 if neighbor_matrix2 is not None else int32_2d_sentinel,
            num_neighbors2 if num_neighbors2 is not None else int32_1d_sentinel,
            neighbor_matrix_shifts2
            if neighbor_matrix_shifts2 is not None
            else int32_3d_sentinel,
            vectors_arg,
            distances_arg,
            pair_params_arg,
            energies_arg,
            forces_arg,
        ],
        block_dim=TILE_GROUP_SIZE,
        device=device,
    )


def query_cluster_tile_coo(
    sorted_atom_index: wp.array,
    sorted_pos_x: wp.array,
    sorted_pos_y: wp.array,
    sorted_pos_z: wp.array,
    num_tiles: wp.array,
    tile_row_group: wp.array,
    tile_col_group: wp.array,
    cell: wp.array,
    inv_cell: wp.array,
    cutoff: float,
    natom: int,
    max_pairs: int,
    pair_counter: wp.array,
    coo_list: wp.array,
    coo_shifts: wp.array,
    wp_dtype: type,
    device: str,
    *,
    n_tiles: int | None = None,
    rebuild_flags: wp.array | None = None,
    tile_offsets: wp.array | None = None,
    tile_counts: wp.array | None = None,
    pair_offsets: wp.array | None = None,
    pair_counts: wp.array | None = None,
    return_vectors: bool = False,
    return_distances: bool = False,
    pair_fn: wp.Function | None = None,
    pair_params: wp.array | None = None,
    neighbor_vectors: wp.array | None = None,
    neighbor_distances: wp.array | None = None,
    pair_energies: wp.array | None = None,
    pair_forces: wp.array | None = None,
) -> None:
    """Convert cluster-tile pairs into a flat COO pair list.

    Iterates the tile pairs emitted by :func:`build_cluster_tile_list` and
    writes a flat COO pair list (``coo_list``), per-pair shifts, and optional
    pair-output buffers. Cluster-tile kernels are CUDA float32 only.

    Parameters
    ----------
    sorted_atom_index : wp.array, shape (n_padded,), dtype=wp.int32
        Morton-sort permutation mapping cluster slot to original atom index.
    sorted_pos_x, sorted_pos_y, sorted_pos_z : wp.array, shape (n_padded,), dtype=wp.float32
        Morton-sorted SoA positions.
    num_tiles : wp.array, shape (1,), dtype=wp.int32
        Tile counter populated by :func:`build_cluster_tile_list`.
    tile_row_group, tile_col_group : wp.array, shape (max_tiles,), dtype=wp.int32
        Paired tile indices populated by :func:`build_cluster_tile_list`.
    cell, inv_cell : wp.array, shape (1,), dtype=wp.mat33f
        Cell and inverse-cell matrices.
    cutoff : float
        Pair cutoff in Cartesian units.
    natom : int
        Real (unpadded) atom count.
    max_pairs : int
        Capacity of ``coo_list`` and ``coo_shifts``.
    pair_counter : wp.array, shape (1,), dtype=wp.int32
        OUTPUT: pair counter incremented atomically. Caller must zero
        before launch.
    coo_list : wp.array, shape (max_pairs, 2), dtype=wp.int32
        OUTPUT: flat COO pair list ``[source_atom, target_atom]``.
    coo_shifts : wp.array, shape (max_pairs, 3), dtype=wp.int32
        OUTPUT: per-pair periodic shift vectors.
    wp_dtype : type
        Must be ``wp.float32``.
    device : str
        Warp device string (e.g. ``"cuda:0"``).
    return_vectors : bool, default False
        If True, write per-pair displacement vectors into ``neighbor_vectors``.
    return_distances : bool, default False
        If True, write per-pair distances into ``neighbor_distances``.
    pair_fn : wp.Function or None, optional
        Optional pair function; when set, ``pair_energies`` and ``pair_forces``
        are populated.
    pair_params : wp.array or None, optional
        Parameter table consumed by ``pair_fn``.
    neighbor_vectors, neighbor_distances, pair_energies, pair_forces : wp.array or None, optional
        OUTPUT buffers, written only when the corresponding enable flag /
        ``pair_fn`` is active.

    Returns
    -------
    None
        This function modifies the input arrays in-place:

        - ``pair_counter``, ``coo_list``, ``coo_shifts`` are always written.
        - ``neighbor_vectors``, ``neighbor_distances``, ``pair_energies``,
          ``pair_forces`` are written when their enabling flag is set.

    Notes
    -----
    - Thread launch: tiled over the **allocated** ``tile_row_group``
      buffer with ``block_dim=TILE_GROUP_SIZE``.  Threads whose tile
      index exceeds ``num_tiles[0]`` early-return inside the kernel.
    - Modifies: ``pair_counter``, ``coo_list``, ``coo_shifts``, and any
      enabled pair-output buffers.
    - Cluster-tile does not support partial neighbor lists; use
      :func:`nvalchemiops.neighbors.cell_list.query_cell_list` or
      :func:`nvalchemiops.neighbors.naive.naive_neighbor_matrix` instead.

    See Also
    --------
    build_cluster_tile_list : Emit the tile pairs consumed by this launcher.
    query_cluster_tile : Per-atom neighbor-matrix variant.
    batch_query_cluster_tile_coo : Batched companion launcher.
    """

    _require_f32(wp_dtype)
    coo_segmented = _require_all_or_none(
        "pair_offsets", pair_offsets, "pair_counts", pair_counts
    )
    tile_segmented = _require_all_or_none(
        "tile_offsets", tile_offsets, "tile_counts", tile_counts
    )
    selective = rebuild_flags is not None
    if selective and not coo_segmented:
        raise ValueError(
            "selective cluster_tile COO requires pair_offsets and pair_counts"
        )
    int32_1d_sentinel = _empty_sentinel(1, wp.int32, device)
    bool_sentinel = _empty_sentinel(1, wp.bool, device)
    if coo_segmented:
        _reset_cluster_tile_counts(
            pair_counts,
            rebuild_flags if rebuild_flags is not None else bool_sentinel,
            device,
            selective=selective,
        )
    (
        vectors_arg,
        distances_arg,
        pair_params_arg,
        energies_arg,
        forces_arg,
    ) = _prepare_coo_pair_output_args(
        wp.float32,
        device,
        int(max_pairs),
        return_vectors=return_vectors,
        return_distances=return_distances,
        pair_fn=pair_fn,
        pair_params=pair_params,
        neighbor_vectors=neighbor_vectors,
        neighbor_distances=neighbor_distances,
        pair_energies=pair_energies,
        pair_forces=pair_forces,
    )
    tile_capacity = int(tile_row_group.shape[0])
    # ``n_tiles`` (host-synced emitted-tile count) tightens the launch to the
    # real tiles; ``None`` falls back to the full buffer (the kernel's
    # ``tile >= num_tiles[0]`` guard still no-ops the surplus blocks).
    launch_tiles = (
        tile_capacity if n_tiles is None else max(0, min(int(n_tiles), tile_capacity))
    )
    if launch_tiles <= 0:
        return

    wp.launch_tiled(
        kernel=get_query_cluster_tile_coo_kernel(
            tile_segmented=tile_segmented,
            coo_segmented=coo_segmented,
            selective=selective,
            return_vectors=bool(return_vectors),
            return_distances=bool(return_distances),
            pair_fn=pair_fn,
        ),
        dim=[launch_tiles],
        inputs=[
            sorted_pos_x,
            sorted_pos_y,
            sorted_pos_z,
            sorted_atom_index,
            cell,
            inv_cell,
            wp.float32(cutoff * cutoff),
            int(natom),
            int(max_pairs),
            tile_row_group,
            tile_col_group,
            _empty_sentinel(1, wp.int32, device),
            num_tiles,
            tile_offsets if tile_offsets is not None else int32_1d_sentinel,
            tile_counts if tile_counts is not None else int32_1d_sentinel,
            rebuild_flags if rebuild_flags is not None else bool_sentinel,
            pair_counter,
            pair_offsets if pair_offsets is not None else int32_1d_sentinel,
            pair_counts if pair_counts is not None else int32_1d_sentinel,
            coo_list,
            coo_shifts,
            vectors_arg,
            distances_arg,
            pair_params_arg,
            energies_arg,
            forces_arg,
        ],
        block_dim=TILE_GROUP_SIZE,
        device=device,
    )


def batch_build_cluster_tile_list(
    sorted_pos_x: wp.array,
    sorted_pos_y: wp.array,
    sorted_pos_z: wp.array,
    group_system: wp.array,
    group_ptr: wp.array,
    cell_batch: wp.array,
    inv_cell_batch: wp.array,
    cutoff: float,
    num_tiles: wp.array,
    tile_row_group: wp.array,
    tile_col_group: wp.array,
    tile_system: wp.array,
    wp_dtype: type,
    device: str,
    *,
    group_ctr_x_buffer: wp.array | None = None,
    group_ctr_y_buffer: wp.array | None = None,
    group_ctr_z_buffer: wp.array | None = None,
    group_ext_x_buffer: wp.array | None = None,
    group_ext_y_buffer: wp.array | None = None,
    group_ext_z_buffer: wp.array | None = None,
    rebuild_flags: wp.array | None = None,
    tile_offsets: wp.array | None = None,
    tile_counts: wp.array | None = None,
) -> None:
    """Enumerate per-system cluster-tile pairs on pre-sorted positions.

    Batched companion of :func:`build_cluster_tile_list`: walks the
    concatenated per-system Morton-sorted clusters and emits tile pairs
    restricted to each system. Cluster-tile kernels are CUDA float32 only.

    Parameters
    ----------
    sorted_pos_x, sorted_pos_y, sorted_pos_z : wp.array, shape (n_padded,), dtype=wp.float32
        Per-system Morton-sorted SoA positions concatenated across systems.
    group_system : wp.array, shape (ngroup,), dtype=wp.int32
        System index for each 32-atom group.
    group_ptr : wp.array, shape (num_systems + 1,), dtype=wp.int32
        CSR pointer for groups per system.
    cell_batch, inv_cell_batch : wp.array, shape (num_systems,), dtype=wp.mat33f
        Per-system cell and inverse-cell matrices.
    cutoff : float
        Bounding-box filter cutoff in Cartesian units.
    num_tiles : wp.array, shape (1,), dtype=wp.int32
        OUTPUT: tile counter incremented atomically. Caller must zero
        before launch.
    tile_row_group, tile_col_group, tile_system : wp.array, shape (max_tiles,), dtype=wp.int32
        OUTPUT: paired tile indices and per-tile system index.
    wp_dtype : type
        Must be ``wp.float32``; cluster-tile kernels are float32-only.
    device : str
        Warp device string (e.g. ``"cuda:0"``).
    group_ctr_x_buffer, group_ctr_y_buffer, group_ctr_z_buffer : wp.array, optional
        Caller-owned per-group center-of-bbox scratch. Transient buffers are
        allocated when omitted.
    group_ext_x_buffer, group_ext_y_buffer, group_ext_z_buffer : wp.array, optional
        Caller-owned per-group bbox-extent scratch. Transient buffers are
        allocated when omitted.

    Returns
    -------
    None
        This function modifies the input arrays in-place:

        - ``num_tiles`` : counts emitted tile pairs.
        - ``tile_row_group``, ``tile_col_group``, ``tile_system`` : populated
          for ``[0:num_tiles[0]]``.

    Notes
    -----
    - Thread launch: tiled over ``(ngroup,)`` with ``block_dim=TILE_GROUP_SIZE``.
    - Modifies: ``num_tiles``, ``tile_row_group``, ``tile_col_group``,
      ``tile_system``.
    - Pairs are emitted only within the same system; cross-system pairs
      are filtered out.

    See Also
    --------
    build_cluster_tile_list : Single-system companion launcher.
    batch_query_cluster_tile : Consume batched tile pairs into a neighbor matrix.
    batch_query_cluster_tile_coo : Consume batched tile pairs into a COO list.
    """
    _require_f32(wp_dtype)
    ngroup = int(sorted_pos_x.shape[0]) // TILE_GROUP_SIZE
    max_tiles = int(tile_row_group.shape[0])
    scratch_size = _group_scratch_size(ngroup)

    group_ctr_x = _group_buffer(group_ctr_x_buffer, scratch_size, device)
    group_ctr_y = _group_buffer(group_ctr_y_buffer, scratch_size, device)
    group_ctr_z = _group_buffer(group_ctr_z_buffer, scratch_size, device)
    group_ext_x = _group_buffer(group_ext_x_buffer, scratch_size, device)
    group_ext_y = _group_buffer(group_ext_y_buffer, scratch_size, device)
    group_ext_z = _group_buffer(group_ext_z_buffer, scratch_size, device)

    wp.launch_tiled(
        kernel=_rank_to_group_kernel,
        dim=[ngroup],
        inputs=[
            sorted_pos_x,
            sorted_pos_y,
            sorted_pos_z,
            group_ctr_x,
            group_ctr_y,
            group_ctr_z,
            group_ext_x,
            group_ext_y,
            group_ext_z,
        ],
        block_dim=TILE_GROUP_SIZE,
        device=device,
    )

    int32_sentinel = _empty_sentinel(1, wp.int32, device)
    bool_sentinel = _empty_sentinel(1, wp.bool, device)
    segmented = _require_all_or_none(
        "tile_offsets", tile_offsets, "tile_counts", tile_counts
    )
    selective = rebuild_flags is not None
    if selective and not segmented:
        raise ValueError(
            "batched selective cluster_tile build requires tile_offsets and tile_counts"
        )
    tile_offsets_arg = tile_offsets if tile_offsets is not None else int32_sentinel
    tile_counts_arg = tile_counts if tile_counts is not None else int32_sentinel
    rebuild_flags_arg = rebuild_flags if rebuild_flags is not None else bool_sentinel
    if segmented:
        _reset_cluster_tile_counts(
            tile_counts_arg, rebuild_flags_arg, device, selective=selective
        )

    wp.launch_tiled(
        kernel=_get_build_cluster_tiles_kernel(
            batched=True, segmented=segmented, selective=selective
        ),
        dim=[ngroup],
        inputs=[
            group_ctr_x,
            group_ctr_y,
            group_ctr_z,
            group_ext_x,
            group_ext_y,
            group_ext_z,
            group_system,
            group_ptr,
            cell_batch,
            inv_cell_batch,
            wp.float32(cutoff * cutoff),
            int(ngroup),
            num_tiles,
            tile_offsets_arg,
            tile_counts_arg,
            rebuild_flags_arg,
            tile_row_group,
            tile_col_group,
            tile_system,
            int(max_tiles),
        ],
        block_dim=TILE_GROUP_SIZE,
        device=device,
    )


def batch_query_cluster_tile(
    sorted_atom_index: wp.array,
    sorted_pos_x: wp.array,
    sorted_pos_y: wp.array,
    sorted_pos_z: wp.array,
    cell_batch: wp.array,
    inv_cell_batch: wp.array,
    num_tiles: wp.array,
    tile_row_group: wp.array,
    tile_col_group: wp.array,
    tile_system: wp.array,
    cutoff: float,
    natom: int,
    neighbor_matrix: wp.array,
    num_neighbors: wp.array,
    neighbor_matrix_shifts: wp.array,
    wp_dtype: type,
    device: str,
    *,
    n_tiles: int | None = None,
    cutoff2: float | None = None,
    neighbor_matrix2: wp.array | None = None,
    num_neighbors2: wp.array | None = None,
    neighbor_matrix_shifts2: wp.array | None = None,
    rebuild_flags: wp.array | None = None,
    tile_offsets: wp.array | None = None,
    tile_counts: wp.array | None = None,
    return_vectors: bool = False,
    return_distances: bool = False,
    pair_fn: wp.Function | None = None,
    pair_params: wp.array | None = None,
    neighbor_vectors: wp.array | None = None,
    neighbor_distances: wp.array | None = None,
    pair_energies: wp.array | None = None,
    pair_forces: wp.array | None = None,
) -> None:
    """Convert batched cluster-tile pairs into a global per-atom neighbor matrix.

    Batched companion of :func:`query_cluster_tile`: iterates the tile pairs
    emitted by :func:`batch_build_cluster_tile_list` and fills a global
    (cross-system) per-atom neighbor matrix. Cluster-tile kernels are CUDA
    float32 only.

    Parameters
    ----------
    sorted_atom_index : wp.array, shape (n_padded,), dtype=wp.int32
        Morton-sort permutation; padding slots carry ``sorted_atom_index == natom``.
    sorted_pos_x, sorted_pos_y, sorted_pos_z : wp.array, shape (n_padded,), dtype=wp.float32
        Per-system Morton-sorted SoA positions.
    cell_batch, inv_cell_batch : wp.array, shape (num_systems,), dtype=wp.mat33f
        Per-system cell and inverse-cell matrices.
    num_tiles : wp.array, shape (1,), dtype=wp.int32
        Tile counter populated by :func:`batch_build_cluster_tile_list`.
    tile_row_group, tile_col_group, tile_system : wp.array, shape (max_tiles,), dtype=wp.int32
        Paired tile indices and per-tile system index populated by
        :func:`batch_build_cluster_tile_list`.
    cutoff : float
        Pair cutoff in Cartesian units.
    natom : int
        Total real (unpadded) atom count across systems.
    neighbor_matrix : wp.array, shape (natom, max_neighbors), dtype=wp.int32
        OUTPUT: global per-atom neighbor indices.
    num_neighbors : wp.array, shape (natom,), dtype=wp.int32
        OUTPUT: per-atom neighbor counts. Caller must zero before launch.
    neighbor_matrix_shifts : wp.array, shape (natom, max_neighbors, 3), dtype=wp.int32
        OUTPUT: per-pair periodic shift vectors.
    wp_dtype : type
        Must be ``wp.float32``.
    device : str
        Warp device string (e.g. ``"cuda:0"``).
    return_vectors : bool, default False
        If True, write per-pair displacement vectors into ``neighbor_vectors``.
    return_distances : bool, default False
        If True, write per-pair distances into ``neighbor_distances``.
    pair_fn : wp.Function or None, optional
        Optional pair function; when set, ``pair_energies`` and ``pair_forces``
        are populated.
    pair_params : wp.array or None, optional
        Parameter table consumed by ``pair_fn``.
    neighbor_vectors, neighbor_distances, pair_energies, pair_forces : wp.array or None, optional
        OUTPUT buffers, written only when the corresponding enable flag /
        ``pair_fn`` is active.

    Returns
    -------
    None
        This function modifies the input arrays in-place:

        - ``neighbor_matrix``, ``num_neighbors``, ``neighbor_matrix_shifts``
          are always written.
        - Optional pair-output buffers are written when their enable flag
          is set.

    Notes
    -----
    - Thread launch: tiled over the **allocated** ``tile_row_group``
      buffer with ``block_dim=TILE_GROUP_SIZE``.  Threads whose tile
      index exceeds ``num_tiles[0]`` early-return inside the kernel.
    - Modifies: ``neighbor_matrix``, ``num_neighbors``,
      ``neighbor_matrix_shifts``, and any enabled pair-output buffers.
    - Cluster-tile does not support partial neighbor lists; use
      :func:`nvalchemiops.neighbors.cell_list.batch_query_cell_list` or
      :func:`nvalchemiops.neighbors.naive.batch_naive_neighbor_matrix`
      instead.

    See Also
    --------
    batch_build_cluster_tile_list : Emit the tile pairs consumed here.
    batch_query_cluster_tile_coo : COO-format variant.
    query_cluster_tile : Single-system companion launcher.
    """

    _require_f32(wp_dtype)
    tile_capacity = int(tile_row_group.shape[0])
    # ``n_tiles`` (host-synced emitted-tile count) tightens the launch to the
    # real tiles; ``None`` falls back to the full buffer (the kernel's
    # ``tile >= num_tiles[0]`` guard still no-ops the surplus blocks).
    launch_tiles = (
        tile_capacity if n_tiles is None else max(0, min(int(n_tiles), tile_capacity))
    )
    if launch_tiles <= 0:
        return

    dual_cutoff = cutoff2 is not None
    has_pair_outputs = (
        bool(return_vectors) or bool(return_distances) or pair_fn is not None
    )
    if dual_cutoff and has_pair_outputs:
        raise ValueError(
            "cluster_tile cutoff2 is matrix-only and cannot be combined with pair outputs"
        )
    if dual_cutoff:
        if (
            neighbor_matrix2 is None
            or num_neighbors2 is None
            or neighbor_matrix_shifts2 is None
        ):
            raise ValueError(
                "neighbor_matrix2, num_neighbors2, and neighbor_matrix_shifts2 are required when cutoff2 is provided"
            )
    int32_1d_sentinel = _empty_sentinel(1, wp.int32, device)
    int32_2d_sentinel = _empty_sentinel(2, wp.int32, device)
    int32_3d_sentinel = _empty_sentinel(3, wp.int32, device)
    bool_sentinel = _empty_sentinel(1, wp.bool, device)
    tile_segmented = _require_all_or_none(
        "tile_offsets", tile_offsets, "tile_counts", tile_counts
    )
    selective = rebuild_flags is not None
    if selective and not tile_segmented:
        raise ValueError(
            "batched selective cluster_tile matrix query requires tile_offsets and tile_counts"
        )

    (
        vectors_arg,
        distances_arg,
        pair_params_arg,
        energies_arg,
        forces_arg,
    ) = _prepare_pair_output_args(
        wp.float32,
        device,
        return_vectors=return_vectors,
        return_distances=return_distances,
        pair_fn=pair_fn,
        pair_params=pair_params,
        neighbor_vectors=neighbor_vectors,
        neighbor_distances=neighbor_distances,
        pair_energies=pair_energies,
        pair_forces=pair_forces,
    )

    wp.launch_tiled(
        kernel=get_batch_query_cluster_tile_kernel(
            tile_segmented=tile_segmented,
            selective=selective,
            dual_cutoff=dual_cutoff,
            return_vectors=bool(return_vectors),
            return_distances=bool(return_distances),
            pair_fn=pair_fn,
        ),
        dim=[launch_tiles],
        inputs=[
            sorted_pos_x,
            sorted_pos_y,
            sorted_pos_z,
            sorted_atom_index,
            cell_batch,
            inv_cell_batch,
            wp.float32(cutoff * cutoff),
            wp.float32(
                (cutoff2 if cutoff2 is not None else cutoff)
                * (cutoff2 if cutoff2 is not None else cutoff)
            ),
            int(natom),
            int(neighbor_matrix.shape[1]),
            int(neighbor_matrix2.shape[1]) if neighbor_matrix2 is not None else 0,
            tile_row_group,
            tile_col_group,
            tile_system,
            num_tiles,
            tile_offsets if tile_offsets is not None else int32_1d_sentinel,
            tile_counts if tile_counts is not None else int32_1d_sentinel,
            rebuild_flags if rebuild_flags is not None else bool_sentinel,
            neighbor_matrix,
            num_neighbors,
            neighbor_matrix_shifts,
            neighbor_matrix2 if neighbor_matrix2 is not None else int32_2d_sentinel,
            num_neighbors2 if num_neighbors2 is not None else int32_1d_sentinel,
            neighbor_matrix_shifts2
            if neighbor_matrix_shifts2 is not None
            else int32_3d_sentinel,
            vectors_arg,
            distances_arg,
            pair_params_arg,
            energies_arg,
            forces_arg,
        ],
        block_dim=TILE_GROUP_SIZE,
        device=device,
    )


def batch_query_cluster_tile_coo(
    sorted_atom_index: wp.array,
    sorted_pos_x: wp.array,
    sorted_pos_y: wp.array,
    sorted_pos_z: wp.array,
    cell_batch: wp.array,
    inv_cell_batch: wp.array,
    num_tiles: wp.array,
    tile_row_group: wp.array,
    tile_col_group: wp.array,
    tile_system: wp.array,
    cutoff: float,
    natom: int,
    max_pairs: int,
    pair_counter: wp.array,
    coo_list: wp.array,
    coo_shifts: wp.array,
    wp_dtype: type,
    device: str,
    *,
    n_tiles: int | None = None,
    rebuild_flags: wp.array | None = None,
    tile_offsets: wp.array | None = None,
    tile_counts: wp.array | None = None,
    pair_offsets: wp.array | None = None,
    pair_counts: wp.array | None = None,
    return_vectors: bool = False,
    return_distances: bool = False,
    pair_fn: wp.Function | None = None,
    pair_params: wp.array | None = None,
    neighbor_vectors: wp.array | None = None,
    neighbor_distances: wp.array | None = None,
    pair_energies: wp.array | None = None,
    pair_forces: wp.array | None = None,
) -> None:
    """Convert batched cluster-tile pairs into a flat COO pair list.

    Batched companion of :func:`query_cluster_tile_coo`: iterates the tile
    pairs emitted by :func:`batch_build_cluster_tile_list` and writes a flat
    COO pair list, per-pair shifts, and optional pair-output buffers.
    Cluster-tile kernels are CUDA float32 only.

    Parameters
    ----------
    sorted_atom_index : wp.array, shape (n_padded,), dtype=wp.int32
        Morton-sort permutation; padding slots carry ``sorted_atom_index == natom``.
    sorted_pos_x, sorted_pos_y, sorted_pos_z : wp.array, shape (n_padded,), dtype=wp.float32
        Per-system Morton-sorted SoA positions.
    cell_batch, inv_cell_batch : wp.array, shape (num_systems,), dtype=wp.mat33f
        Per-system cell and inverse-cell matrices.
    num_tiles : wp.array, shape (1,), dtype=wp.int32
        Tile counter populated by :func:`batch_build_cluster_tile_list`.
    tile_row_group, tile_col_group, tile_system : wp.array, shape (max_tiles,), dtype=wp.int32
        Paired tile indices and per-tile system index populated by
        :func:`batch_build_cluster_tile_list`.
    cutoff : float
        Pair cutoff in Cartesian units.
    natom : int
        Total real (unpadded) atom count across systems.
    max_pairs : int
        Capacity of ``coo_list`` and ``coo_shifts``.
    pair_counter : wp.array, shape (1,), dtype=wp.int32
        OUTPUT: pair counter incremented atomically. Caller must zero
        before launch.
    coo_list : wp.array, shape (max_pairs, 2), dtype=wp.int32
        OUTPUT: flat COO pair list ``[source_atom, target_atom]``.
    coo_shifts : wp.array, shape (max_pairs, 3), dtype=wp.int32
        OUTPUT: per-pair periodic shift vectors.
    wp_dtype : type
        Must be ``wp.float32``.
    device : str
        Warp device string (e.g. ``"cuda:0"``).
    return_vectors : bool, default False
        If True, write per-pair displacement vectors into ``neighbor_vectors``.
    return_distances : bool, default False
        If True, write per-pair distances into ``neighbor_distances``.
    pair_fn : wp.Function or None, optional
        Optional pair function; when set, ``pair_energies`` and ``pair_forces``
        are populated.
    pair_params : wp.array or None, optional
        Parameter table consumed by ``pair_fn``.
    neighbor_vectors, neighbor_distances, pair_energies, pair_forces : wp.array or None, optional
        OUTPUT buffers, written only when the corresponding enable flag /
        ``pair_fn`` is active.

    Returns
    -------
    None
        This function modifies the input arrays in-place:

        - ``pair_counter``, ``coo_list``, ``coo_shifts`` are always written.
        - Optional pair-output buffers are written when their enable flag
          is set.

    Notes
    -----
    - Thread launch: tiled over the **allocated** ``tile_row_group``
      buffer with ``block_dim=TILE_GROUP_SIZE``.  Threads whose tile
      index exceeds ``num_tiles[0]`` early-return inside the kernel.
    - Modifies: ``pair_counter``, ``coo_list``, ``coo_shifts``, and any
      enabled pair-output buffers.
    - Cluster-tile does not support partial neighbor lists; use
      :func:`nvalchemiops.neighbors.cell_list.batch_query_cell_list` or
      :func:`nvalchemiops.neighbors.naive.batch_naive_neighbor_matrix`
      instead.

    See Also
    --------
    batch_build_cluster_tile_list : Emit the tile pairs consumed here.
    batch_query_cluster_tile : Per-atom neighbor-matrix variant.
    query_cluster_tile_coo : Single-system companion launcher.
    """

    _require_f32(wp_dtype)
    coo_segmented = _require_all_or_none(
        "pair_offsets", pair_offsets, "pair_counts", pair_counts
    )
    tile_segmented = _require_all_or_none(
        "tile_offsets", tile_offsets, "tile_counts", tile_counts
    )
    selective = rebuild_flags is not None
    if selective and not coo_segmented:
        raise ValueError(
            "selective cluster_tile COO requires pair_offsets and pair_counts"
        )
    if selective and not tile_segmented:
        raise ValueError(
            "selective cluster_tile COO requires tile_offsets and tile_counts"
        )
    int32_1d_sentinel = _empty_sentinel(1, wp.int32, device)
    bool_sentinel = _empty_sentinel(1, wp.bool, device)
    if coo_segmented:
        _reset_cluster_tile_counts(
            pair_counts,
            rebuild_flags if rebuild_flags is not None else bool_sentinel,
            device,
            selective=selective,
        )
    (
        vectors_arg,
        distances_arg,
        pair_params_arg,
        energies_arg,
        forces_arg,
    ) = _prepare_coo_pair_output_args(
        wp.float32,
        device,
        int(max_pairs),
        return_vectors=return_vectors,
        return_distances=return_distances,
        pair_fn=pair_fn,
        pair_params=pair_params,
        neighbor_vectors=neighbor_vectors,
        neighbor_distances=neighbor_distances,
        pair_energies=pair_energies,
        pair_forces=pair_forces,
    )
    tile_capacity = int(tile_row_group.shape[0])
    # ``n_tiles`` (host-synced emitted-tile count) tightens the launch to the
    # real tiles; ``None`` falls back to the full buffer (the kernel's
    # ``tile >= num_tiles[0]`` guard still no-ops the surplus blocks).
    launch_tiles = (
        tile_capacity if n_tiles is None else max(0, min(int(n_tiles), tile_capacity))
    )
    if launch_tiles <= 0:
        return

    wp.launch_tiled(
        kernel=get_batch_query_cluster_tile_coo_kernel(
            tile_segmented=tile_segmented,
            coo_segmented=coo_segmented,
            selective=selective,
            return_vectors=bool(return_vectors),
            return_distances=bool(return_distances),
            pair_fn=pair_fn,
        ),
        dim=[launch_tiles],
        inputs=[
            sorted_pos_x,
            sorted_pos_y,
            sorted_pos_z,
            sorted_atom_index,
            cell_batch,
            inv_cell_batch,
            wp.float32(cutoff * cutoff),
            int(natom),
            int(max_pairs),
            tile_row_group,
            tile_col_group,
            tile_system,
            num_tiles,
            tile_offsets if tile_offsets is not None else int32_1d_sentinel,
            tile_counts if tile_counts is not None else int32_1d_sentinel,
            rebuild_flags if rebuild_flags is not None else bool_sentinel,
            pair_counter,
            pair_offsets if pair_offsets is not None else int32_1d_sentinel,
            pair_counts if pair_counts is not None else int32_1d_sentinel,
            coo_list,
            coo_shifts,
            vectors_arg,
            distances_arg,
            pair_params_arg,
            energies_arg,
            forces_arg,
        ],
        block_dim=TILE_GROUP_SIZE,
        device=device,
    )
