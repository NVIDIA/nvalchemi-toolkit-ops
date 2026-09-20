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

"""PyTorch bindings for batched cluster-pair tile neighbor list.

All torch-side work (per-system Morton sort with system-major key,
per-system padding to a multiple of ``TILE_GROUP_SIZE``, SoA gather of
sorted positions, ``inv_cell_batch`` inversion, ``wp.from_torch``
conversion) lives in this module; the Warp layer at
``nvalchemiops.neighbors.cluster_tile`` only sees ``wp.array``
inputs.

Scope: float32, orthorhombic or triclinic PBC, one cell per system,
arbitrary natom per system (padded internally).
"""

from typing import TYPE_CHECKING, NamedTuple

import torch
import warp as wp

if TYPE_CHECKING:
    from nvalchemiops.torch.neighbors.prepared_cluster_tile import ClusterTileState

from nvalchemiops.neighbors.cluster_tile import (
    TILE_GROUP_SIZE,
)
from nvalchemiops.neighbors.cluster_tile import (
    batch_build_cluster_tile_list as wp_batch_build_cluster_tile_list,
)
from nvalchemiops.neighbors.cluster_tile import (
    batch_query_cluster_tile as wp_batch_query_cluster_tile,
)
from nvalchemiops.neighbors.cluster_tile import (
    batch_query_cluster_tile_coo as wp_batch_query_cluster_tile_coo,
)
from nvalchemiops.neighbors.cluster_tile import (
    estimate_batch_cluster_tile_segments as wp_estimate_batch_cluster_tile_segments,
)
from nvalchemiops.neighbors.cluster_tile import (
    estimate_batch_max_tiles_per_group as wp_estimate_batch_max_tiles_per_group,
)
from nvalchemiops.neighbors.cluster_tile.kernels import (
    _get_query_cluster_tile_direct_csr_count_kernel,
    _get_query_cluster_tile_direct_csr_fill_kernel,
)
from nvalchemiops.neighbors.neighbor_utils import (
    _selective_fill_neighbor_matrix_tail as wp_selective_fill_neighbor_matrix_tail,
)
from nvalchemiops.neighbors.neighbor_utils import estimate_max_neighbors
from nvalchemiops.neighbors.neighbor_utils import (
    fill_neighbor_matrix_tail as wp_fill_neighbor_matrix_tail,
)
from nvalchemiops.neighbors.neighbor_utils import (
    selective_zero_num_neighbors as wp_selective_zero_num_neighbors,
)
from nvalchemiops.neighbors.output_args import (
    _has_partial_or_pair_outputs,
    _prepare_coo_pair_output_args,
)
from nvalchemiops.torch._warp_op_helpers import scoped_torch_warp_stream
from nvalchemiops.torch.neighbors._autograd import (
    _reconstruct_coo_geometry,
    _reconstruct_matrix_geometry,
)
from nvalchemiops.torch.neighbors.neighbor_utils import (
    _check_neighbor_capacity,
    _check_tile_buffer_capacity,
    _compact_coo_prefix,
    _prepare_compact_coo_geometry_buffers,
    _validate_cluster_tile_matrix_outputs,
    _validate_compact_coo_outputs,
    _validate_segmented_coo_state,
)
from nvalchemiops.torch.types import get_wp_dtype

__all__ = [
    "TILE_GROUP_SIZE",
    "estimate_batch_max_tiles_per_group",
    "estimate_batch_cluster_tile_list_sizes",
    "estimate_batch_cluster_tile_segments",
    "allocate_batch_cluster_tile_list",
    "batch_build_cluster_tile_list",
    "batch_query_cluster_tile",
    "batch_query_cluster_tile_coo",
    "batch_cluster_tile_neighbor_list",
]


@torch.library.custom_op(
    "nvalchemiops::_batch_cluster_tile_direct_csr_count", mutates_args=("row_counts",)
)
@scoped_torch_warp_stream
def _batch_cluster_tile_direct_csr_count_op(
    cutoff: float,
    natom: int,
    cell: torch.Tensor,
    inv_cell: torch.Tensor,
    sorted_atom_index: torch.Tensor,
    sorted_pos_x: torch.Tensor,
    sorted_pos_y: torch.Tensor,
    sorted_pos_z: torch.Tensor,
    num_tiles: torch.Tensor,
    tile_row_group: torch.Tensor,
    tile_col_group: torch.Tensor,
    tile_system: torch.Tensor,
    row_counts: torch.Tensor,
    launch_tiles: int,
) -> None:
    wp.launch_tiled(
        _get_query_cluster_tile_direct_csr_count_kernel(batched=True),
        dim=[launch_tiles],
        inputs=[
            wp.from_torch(x, dtype=d, return_ctype=True)
            for x, d in (
                (sorted_pos_x, wp.float32),
                (sorted_pos_y, wp.float32),
                (sorted_pos_z, wp.float32),
                (sorted_atom_index, wp.int32),
                (cell.contiguous(), wp.mat33f),
                (inv_cell.contiguous(), wp.mat33f),
            )
        ]
        + [
            float(cutoff * cutoff),
            int(natom),
            wp.from_torch(num_tiles, dtype=wp.int32, return_ctype=True),
            wp.from_torch(tile_row_group, dtype=wp.int32, return_ctype=True),
            wp.from_torch(tile_col_group, dtype=wp.int32, return_ctype=True),
            wp.from_torch(tile_system, dtype=wp.int32, return_ctype=True),
            wp.from_torch(row_counts, dtype=wp.int32, return_ctype=True),
        ],
        device=str(sorted_pos_x.device),
        block_dim=TILE_GROUP_SIZE,
    )


@_batch_cluster_tile_direct_csr_count_op.register_fake
def _(*args, **kwargs) -> None:
    return None


@torch.library.custom_op(
    "nvalchemiops::_batch_cluster_tile_direct_csr_fill",
    mutates_args=(
        "cursors",
        "coo_list",
        "coo_shifts",
        "neighbor_vectors",
        "neighbor_distances",
    ),
)
@scoped_torch_warp_stream
def _batch_cluster_tile_direct_csr_fill_op(
    cutoff: float,
    natom: int,
    cell: torch.Tensor,
    inv_cell: torch.Tensor,
    sorted_atom_index: torch.Tensor,
    sorted_pos_x: torch.Tensor,
    sorted_pos_y: torch.Tensor,
    sorted_pos_z: torch.Tensor,
    num_tiles: torch.Tensor,
    tile_row_group: torch.Tensor,
    tile_col_group: torch.Tensor,
    tile_system: torch.Tensor,
    cursors: torch.Tensor,
    coo_list: torch.Tensor,
    coo_shifts: torch.Tensor,
    neighbor_vectors: torch.Tensor,
    neighbor_distances: torch.Tensor,
    return_vectors: bool,
    return_distances: bool,
    max_pairs: int,
    launch_tiles: int,
) -> None:
    wp.launch_tiled(
        _get_query_cluster_tile_direct_csr_fill_kernel(
            batched=True,
            return_vectors=return_vectors,
            return_distances=return_distances,
        ),
        dim=[launch_tiles],
        inputs=[
            wp.from_torch(x, dtype=d, return_ctype=True)
            for x, d in (
                (sorted_pos_x, wp.float32),
                (sorted_pos_y, wp.float32),
                (sorted_pos_z, wp.float32),
                (sorted_atom_index, wp.int32),
                (cell.contiguous(), wp.mat33f),
                (inv_cell.contiguous(), wp.mat33f),
            )
        ]
        + [
            float(cutoff * cutoff),
            int(natom),
            int(max_pairs),
            wp.from_torch(num_tiles, dtype=wp.int32, return_ctype=True),
            wp.from_torch(tile_row_group, dtype=wp.int32, return_ctype=True),
            wp.from_torch(tile_col_group, dtype=wp.int32, return_ctype=True),
            wp.from_torch(tile_system, dtype=wp.int32, return_ctype=True),
            wp.from_torch(cursors, dtype=wp.int32, return_ctype=True),
            wp.from_torch(coo_list, dtype=wp.int32, return_ctype=True),
            wp.from_torch(coo_shifts, dtype=wp.int32, return_ctype=True),
            wp.from_torch(neighbor_vectors, dtype=wp.vec3f, return_ctype=True),
            wp.from_torch(neighbor_distances, dtype=wp.float32, return_ctype=True),
            wp.empty((1, 1), dtype=wp.float32, device=str(sorted_pos_x.device)),
            wp.empty(1, dtype=wp.float32, device=str(sorted_pos_x.device)),
            wp.empty(1, dtype=wp.vec3f, device=str(sorted_pos_x.device)),
        ],
        device=str(sorted_pos_x.device),
        block_dim=TILE_GROUP_SIZE,
    )


@_batch_cluster_tile_direct_csr_fill_op.register_fake
def _(*args, **kwargs) -> None:
    return None


@scoped_torch_warp_stream
def _batch_cluster_tile_direct_csr_fill_pair_outputs(
    cutoff: float,
    natom: int,
    cell: torch.Tensor,
    inv_cell: torch.Tensor,
    sorted_atom_index: torch.Tensor,
    sorted_pos_x: torch.Tensor,
    sorted_pos_y: torch.Tensor,
    sorted_pos_z: torch.Tensor,
    num_tiles: torch.Tensor,
    tile_row_group: torch.Tensor,
    tile_col_group: torch.Tensor,
    tile_system: torch.Tensor,
    cursors: torch.Tensor,
    coo_list: torch.Tensor,
    coo_shifts: torch.Tensor,
    pair_fn: wp.Function,
    pair_params: torch.Tensor,
    pair_energies: torch.Tensor,
    pair_forces: torch.Tensor,
    neighbor_vectors: torch.Tensor | None,
    neighbor_distances: torch.Tensor | None,
    return_vectors: bool,
    return_distances: bool,
    max_pairs: int,
    launch_tiles: int,
) -> None:
    """Fill batched compact CSR rows and raw pair outputs together."""
    device = str(sorted_pos_x.device)
    wp_pair_params = wp.from_torch(pair_params, dtype=wp.float32)
    wp_neighbor_vectors = (
        wp.from_torch(neighbor_vectors, dtype=wp.vec3f)
        if neighbor_vectors is not None
        else None
    )
    wp_neighbor_distances = (
        wp.from_torch(neighbor_distances, dtype=wp.float32)
        if neighbor_distances is not None
        else None
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
        max_pairs,
        return_vectors=return_vectors,
        return_distances=return_distances,
        pair_fn=pair_fn,
        pair_params=wp_pair_params,
        neighbor_vectors=wp_neighbor_vectors,
        neighbor_distances=wp_neighbor_distances,
        pair_energies=wp.from_torch(pair_energies, dtype=wp.float32),
        pair_forces=wp.from_torch(pair_forces, dtype=wp.vec3f),
    )
    wp.launch_tiled(
        _get_query_cluster_tile_direct_csr_fill_kernel(
            batched=True,
            return_vectors=return_vectors,
            return_distances=return_distances,
            pair_fn=pair_fn,
        ),
        dim=[launch_tiles],
        inputs=[
            wp.from_torch(sorted_pos_x, dtype=wp.float32),
            wp.from_torch(sorted_pos_y, dtype=wp.float32),
            wp.from_torch(sorted_pos_z, dtype=wp.float32),
            wp.from_torch(sorted_atom_index, dtype=wp.int32),
            wp.from_torch(cell.contiguous(), dtype=wp.mat33f),
            wp.from_torch(inv_cell.contiguous(), dtype=wp.mat33f),
            float(cutoff * cutoff),
            int(natom),
            int(max_pairs),
            wp.from_torch(num_tiles, dtype=wp.int32),
            wp.from_torch(tile_row_group, dtype=wp.int32),
            wp.from_torch(tile_col_group, dtype=wp.int32),
            wp.from_torch(tile_system, dtype=wp.int32),
            wp.from_torch(cursors, dtype=wp.int32),
            wp.from_torch(coo_list, dtype=wp.int32),
            wp.from_torch(coo_shifts, dtype=wp.int32),
            vectors_arg,
            distances_arg,
            pair_params_arg,
            energies_arg,
            forces_arg,
        ],
        device=device,
        block_dim=TILE_GROUP_SIZE,
    )


@torch.library.custom_op(
    "nvalchemiops::_batch_cluster_tile_fill_neighbor_matrix_tail",
    mutates_args=("neighbor_matrix",),
)
@scoped_torch_warp_stream
def _batch_cluster_tile_fill_neighbor_matrix_tail_op(
    num_neighbors: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    n_rows: int,
    max_neighbors: int,
    fill_value: int,
) -> None:
    if max_neighbors <= 0:
        return
    wp_fill_neighbor_matrix_tail(
        wp.from_torch(
            num_neighbors, dtype=wp.int32, requires_grad=False, return_ctype=True
        ),
        int(n_rows),
        int(max_neighbors),
        int(fill_value),
        wp.from_torch(
            neighbor_matrix, dtype=wp.int32, requires_grad=False, return_ctype=True
        ),
        str(neighbor_matrix.device),
    )


@_batch_cluster_tile_fill_neighbor_matrix_tail_op.register_fake
def _(
    num_neighbors: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    n_rows: int,
    max_neighbors: int,
    fill_value: int,
) -> None:
    return None


# =============================================================================
# Sizing + allocation helpers
# =============================================================================
def estimate_batch_max_tiles_per_group(
    batch_ptr: torch.Tensor,
    cutoff: float,
    cell_batch: torch.Tensor,
    *,
    safety: float = 2.0,
    floor: int = 256,
) -> int:
    """Estimate batched ``max_tiles_per_group`` from per-system cells.

    Parameters
    ----------
    batch_ptr : torch.Tensor, shape (num_systems + 1,)
        Cumulative atom counts.
    cutoff : float
        Cartesian cutoff used for cluster-tile construction.
    cell_batch : torch.Tensor, shape (num_systems, 3, 3)
        Per-system cell matrices.
    safety : float, default 2.0
        Multiplier on the volumetric estimate.
    floor : int, default 256
        Minimum returned value for batched compact buffers.

    Returns
    -------
    int
        Shared ``max_tiles_per_group`` for the batched compact tile buffer.
    """
    if batch_ptr.shape[0] < 2:
        raise ValueError("batch_ptr must have length at least 2")
    volumes = torch.linalg.det(cell_batch.to(torch.float64)).abs().view(-1)
    return wp_estimate_batch_max_tiles_per_group(
        batch_ptr,
        cutoff,
        volumes,
        safety=safety,
        floor=floor,
    )


def estimate_batch_cluster_tile_list_sizes(
    batch_ptr: torch.Tensor,
    max_tiles_per_group: int = 256,
) -> tuple[int, int, int, int, int]:
    """Estimate allocation sizes for the batched tile neighbor list state.

    Parameters
    ----------
    batch_ptr : torch.Tensor, shape (num_systems + 1,), dtype=int32
        Cumulative atom counts defining per-system ranges.
    max_tiles_per_group : int, default 256
        Sets the capacity of the tile-pair buffer shared by all row groups. If
        system ``i`` has ``g_i`` groups, its compact capacity contribution is
        ``g_i * min(g_i, max_tiles_per_group)`` entries. Contributions are
        summed into one pooled buffer shared by the whole batch.

    Returns
    -------
    n_padded : int
        Total padded atom count (sum of per-system ``ceil(natom/32)*32``).
    ngroup : int
        Number of 32-atom groups: ``n_padded // 32``.
    ngroup_padded : int
        Group-array pad length for in-bounds ``wp.tile_load`` at any
        TILE-aligned offset.
    max_tiles : int
        Allocated tile-pair list capacity.
    num_systems : int
    """
    if batch_ptr.shape[0] < 2:
        raise ValueError("batch_ptr must have length at least 2")
    num_systems = int(batch_ptr.shape[0]) - 1
    natom_per_system = (batch_ptr[1:] - batch_ptr[:-1]).to(torch.int64)
    natom_padded_per_system = (
        (natom_per_system + TILE_GROUP_SIZE - 1) // TILE_GROUP_SIZE
    ) * TILE_GROUP_SIZE
    n_padded = int(natom_padded_per_system.sum().item())
    ngroup_per_system = natom_padded_per_system // TILE_GROUP_SIZE
    ngroup = int(ngroup_per_system.sum().item())
    ngroup_padded = (
        (ngroup + TILE_GROUP_SIZE - 1) // TILE_GROUP_SIZE
    ) * TILE_GROUP_SIZE + TILE_GROUP_SIZE
    max_tiles = int(
        (
            ngroup_per_system
            * torch.clamp(ngroup_per_system, max=int(max_tiles_per_group))
        )
        .sum()
        .item()
    )
    return n_padded, ngroup, ngroup_padded, max_tiles, num_systems


def estimate_batch_cluster_tile_segments(
    batch_ptr: torch.Tensor,
    max_neighbors: int,
    max_tiles_per_group: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate fixed per-system tile and COO segment buffers.

    Returns ``(tile_capacities, tile_offsets, pair_capacities, pair_offsets)``
    as int32 tensors on ``batch_ptr.device``. ``tile_offsets`` and
    ``pair_offsets`` are caller-owned fixed inputs for segmented cluster-tile
    build / COO query paths; ``tile_counts`` and ``pair_counts`` are separate
    output counters with length ``num_systems``.

    Parameters
    ----------
    batch_ptr : torch.Tensor, shape (num_systems + 1,), dtype=int32
        Cumulative atom counts defining per-system ranges.
    max_neighbors : int
        Upper bound on neighbors per atom used to size per-system COO segments.
    max_tiles_per_group : int, default 256
        Sets each system's tile-pair segment capacity. A system with ``g_i``
        groups receives ``g_i * min(g_i, max_tiles_per_group)`` entries. Its
        group count is ``ceil(num_atoms_i / 32)``.

    Returns
    -------
    tile_capacities : torch.Tensor, shape (num_systems,), dtype=int32
        Per-system tile buffer capacities.
    tile_offsets : torch.Tensor, shape (num_systems + 1,), dtype=int32
        CSR-style offsets into the segmented tile buffer (fixed input to build).
    pair_capacities : torch.Tensor, shape (num_systems,), dtype=int32
        Per-system pair buffer capacities.
    pair_offsets : torch.Tensor, shape (num_systems + 1,), dtype=int32
        CSR-style offsets into the segmented COO pair buffer (fixed input to query).
    """
    if batch_ptr.shape[0] < 2:
        raise ValueError("batch_ptr must have length at least 2")
    tile_caps, tile_offsets, pair_caps, pair_offsets = (
        wp_estimate_batch_cluster_tile_segments(
            batch_ptr,
            max_neighbors=int(max_neighbors),
            max_tiles_per_group=int(max_tiles_per_group),
        )
    )
    device = batch_ptr.device
    return (
        torch.tensor(tile_caps, dtype=torch.int32, device=device),
        torch.tensor(tile_offsets, dtype=torch.int32, device=device),
        torch.tensor(pair_caps, dtype=torch.int32, device=device),
        torch.tensor(pair_offsets, dtype=torch.int32, device=device),
    )


def allocate_batch_cluster_tile_list(
    batch_ptr: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    max_tiles_per_group: int = 256,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Allocate all state tensors consumed by ``batch_build_cluster_tile_list``.

    Parameters
    ----------
    batch_ptr : torch.Tensor, shape (num_systems + 1,), dtype=int32
        Cumulative atom counts defining per-system ranges.
    device : torch.device
        Target device for all allocated tensors.
    dtype : torch.dtype, default torch.float32
        Floating-point dtype for position and group-centroid tensors.
    max_tiles_per_group : int, default 256
        Capacity factor for the intermediate tile-pair buffer. A system with
        ``g_i`` row groups contributes
        ``g_i * min(g_i, max_tiles_per_group)`` entries to one pooled buffer.
        Increasing the value uses more memory and accommodates more candidate
        tile pairs. See :ref:`cluster-tile-buffer-capacity` for sizing details.

    Returns
    -------
    sorted_atom_index : torch.Tensor, shape (n_padded,), dtype=int32
        Permutation mapping padded slot to original atom index; sentinel value
        ``N`` (total real atoms) marks padding slots.
    sort_inv : torch.Tensor, shape (N,), dtype=int32
        Inverse permutation over the real (un-padded) atoms.
    sorted_pos_x : torch.Tensor, shape (n_padded,), dtype=dtype
        Morton-sorted x-coordinates in the padded SoA layout.
    sorted_pos_y : torch.Tensor, shape (n_padded,), dtype=dtype
        Morton-sorted y-coordinates in the padded SoA layout.
    sorted_pos_z : torch.Tensor, shape (n_padded,), dtype=dtype
        Morton-sorted z-coordinates in the padded SoA layout.
    batch_idx_sorted : torch.Tensor, shape (n_padded,), dtype=int32
        System index for every padded slot after Morton sort.
    batch_ptr_padded : torch.Tensor, shape (num_systems + 1,), dtype=int32
        Cumulative padded atom counts (multiples of ``TILE_GROUP_SIZE``).
    group_system : torch.Tensor, shape (ngroup,), dtype=int32
        System index for each 32-atom group.
    group_ptr : torch.Tensor, shape (num_systems + 1,), dtype=int32
        CSR group-level pointer derived from ``batch_ptr_padded``.
    group_ctr_x : torch.Tensor, shape (ngroup_padded,), dtype=dtype
        Group bounding-box centre x-coordinate (zeroed; written by build).
    group_ctr_y : torch.Tensor, shape (ngroup_padded,), dtype=dtype
        Group bounding-box centre y-coordinate (zeroed; written by build).
    group_ctr_z : torch.Tensor, shape (ngroup_padded,), dtype=dtype
        Group bounding-box centre z-coordinate (zeroed; written by build).
    group_ext_x : torch.Tensor, shape (ngroup_padded,), dtype=dtype
        Group bounding-box half-extent x (zeroed; written by build).
    group_ext_y : torch.Tensor, shape (ngroup_padded,), dtype=dtype
        Group bounding-box half-extent y (zeroed; written by build).
    group_ext_z : torch.Tensor, shape (ngroup_padded,), dtype=dtype
        Group bounding-box half-extent z (zeroed; written by build).
    num_tiles : torch.Tensor, shape (1,), dtype=int32
        Scalar tile-pair count; written by build, read by query.
    tile_row_group : torch.Tensor, shape (max_tiles,), dtype=int32
        Row group index for each tile pair (zeroed; written by build).
    tile_col_group : torch.Tensor, shape (max_tiles,), dtype=int32
        Column group index for each tile pair (zeroed; written by build).
    tile_system : torch.Tensor, shape (max_tiles,), dtype=int32
        System index for each tile pair (zeroed; written by build).
    """
    n_padded, ngroup, ngroup_padded, max_tiles, num_systems = (
        estimate_batch_cluster_tile_list_sizes(
            batch_ptr,
            max_tiles_per_group=max_tiles_per_group,
        )
    )
    N = int(batch_ptr[-1].item())
    sorted_atom_index = torch.empty(n_padded, dtype=torch.int32, device=device)
    sort_inv = torch.empty(N, dtype=torch.int32, device=device)
    sorted_pos_x = torch.empty(n_padded, dtype=dtype, device=device)
    sorted_pos_y = torch.empty(n_padded, dtype=dtype, device=device)
    sorted_pos_z = torch.empty(n_padded, dtype=dtype, device=device)
    batch_idx_sorted = torch.empty(n_padded, dtype=torch.int32, device=device)
    batch_ptr_padded = torch.empty(num_systems + 1, dtype=torch.int32, device=device)
    group_system = torch.empty(ngroup, dtype=torch.int32, device=device)
    group_ptr = torch.empty(num_systems + 1, dtype=torch.int32, device=device)
    group_ctr_x = torch.zeros(ngroup_padded, dtype=dtype, device=device)
    group_ctr_y = torch.zeros(ngroup_padded, dtype=dtype, device=device)
    group_ctr_z = torch.zeros(ngroup_padded, dtype=dtype, device=device)
    group_ext_x = torch.zeros(ngroup_padded, dtype=dtype, device=device)
    group_ext_y = torch.zeros(ngroup_padded, dtype=dtype, device=device)
    group_ext_z = torch.zeros(ngroup_padded, dtype=dtype, device=device)
    num_tiles = torch.zeros(1, dtype=torch.int32, device=device)
    tile_row_group = torch.zeros(max_tiles, dtype=torch.int32, device=device)
    tile_col_group = torch.zeros(max_tiles, dtype=torch.int32, device=device)
    tile_system = torch.zeros(max_tiles, dtype=torch.int32, device=device)
    return (
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


# =============================================================================
# Morton sort + padded scatter (torch side)
# =============================================================================
class _BatchPartitionMetadata(NamedTuple):
    """Tensors derived only from a fixed batch partition."""

    atom_system: torch.Tensor
    batch_ptr_padded: torch.Tensor
    padded_slot_system: torch.Tensor
    real_sorted_rank_to_padded_slot: torch.Tensor
    group_ptr: torch.Tensor
    group_system: torch.Tensor


def _prepare_batch_partition_metadata(
    batch_ptr: torch.Tensor,
    *,
    num_atoms: int,
    padded_slot_system: torch.Tensor,
    batch_ptr_padded: torch.Tensor,
    group_system: torch.Tensor,
    group_ptr: torch.Tensor,
) -> _BatchPartitionMetadata:
    """Build partition metadata into the fixed scratch views."""
    num_systems = batch_ptr.shape[0] - 1
    device = batch_ptr.device
    atoms_per_system = batch_ptr[1:] - batch_ptr[:-1]
    padded_atoms_per_system = (
        (atoms_per_system + TILE_GROUP_SIZE - 1) // TILE_GROUP_SIZE
    ) * TILE_GROUP_SIZE

    atom_system = torch.repeat_interleave(
        torch.arange(num_systems, dtype=torch.int32, device=device),
        atoms_per_system.to(torch.int64),
        output_size=num_atoms,
    ).contiguous()
    padded_ptr = torch.zeros(num_systems + 1, dtype=torch.int32, device=device)
    torch.cumsum(padded_atoms_per_system, dim=0, out=padded_ptr[1:])
    slot_system = torch.repeat_interleave(
        torch.arange(num_systems, dtype=torch.int32, device=device),
        padded_atoms_per_system.to(torch.int64),
        output_size=padded_slot_system.shape[0],
    ).contiguous()
    real_rank = torch.arange(num_atoms, dtype=torch.int32, device=device)
    padded_slot = (
        padded_ptr[atom_system] + real_rank - batch_ptr[atom_system]
    ).contiguous()
    group_level_ptr = (padded_ptr // TILE_GROUP_SIZE).to(torch.int32).contiguous()
    group_owner = slot_system[::TILE_GROUP_SIZE].to(torch.int32).contiguous()

    batch_ptr_padded.copy_(padded_ptr)
    padded_slot_system.copy_(slot_system)
    group_ptr.copy_(group_level_ptr)
    group_system.copy_(group_owner)
    return _BatchPartitionMetadata(
        atom_system=atom_system,
        batch_ptr_padded=batch_ptr_padded,
        padded_slot_system=padded_slot_system,
        real_sorted_rank_to_padded_slot=padded_slot,
        group_ptr=group_ptr,
        group_system=group_system,
    )


def _per_atom_morton_codes(
    positions: torch.Tensor,
    batch_idx: torch.Tensor,
    inv_cell_batch: torch.Tensor,
) -> torch.Tensor:
    """30-bit Morton codes using per-atom fractional coords (triclinic-safe)."""
    inv_per_atom = inv_cell_batch[batch_idx]
    frac = torch.einsum("ni,nij->nj", positions, inv_per_atom)
    frac = frac - torch.floor(frac)
    bucket = (frac * 1024.0).clamp(0, 1023).to(torch.int32)
    ix, iy, iz = bucket.unbind(dim=-1)

    def _spread(x: torch.Tensor) -> torch.Tensor:
        x = x & 0x3FF
        x = (x | (x << 16)) & 0x030000FF
        x = (x | (x << 8)) & 0x0300F00F
        x = (x | (x << 4)) & 0x030C30C3
        x = (x | (x << 2)) & 0x09249249
        return x

    return (_spread(iz) << 2) | (_spread(iy) << 1) | _spread(ix)


def _batched_morton_sort_padded(
    positions: torch.Tensor,
    batch_ptr: torch.Tensor,
    inv_cell_batch: torch.Tensor,
    atom_system: torch.Tensor,
    padded_slot_system: torch.Tensor,
    real_sorted_rank_to_padded_slot: torch.Tensor,
    sorted_atom_index: torch.Tensor,
    sort_inv: torch.Tensor,
    sorted_pos_x: torch.Tensor,
    sorted_pos_y: torch.Tensor,
    sorted_pos_z: torch.Tensor,
) -> None:
    """Per-system Morton sort into the padded layout (torch ops only)."""
    N = positions.shape[0]
    device = positions.device

    codes = _per_atom_morton_codes(positions, atom_system, inv_cell_batch)
    # System-major key: system index in high 32 bits, Morton code in low.
    system_major = codes.to(torch.int64) | (atom_system.to(torch.int64) << 32)
    sorted_atom_index_real = torch.argsort(system_major).to(torch.int32)

    # Inverse permutation over real atoms.
    inv = torch.empty(N, dtype=torch.int32, device=device)
    inv[sorted_atom_index_real] = torch.arange(
        N,
        dtype=torch.int32,
        device=device,
    )
    sort_inv.copy_(inv)

    # sorted_atom_index[k] = N is the padding sentinel.
    sp = torch.full(
        (sorted_atom_index.shape[0],),
        N,
        dtype=torch.int32,
        device=device,
    )
    sp[real_sorted_rank_to_padded_slot] = sorted_atom_index_real
    sorted_atom_index.copy_(sp)

    # Padding slots duplicate each system's first real atom, so
    # tile_min / tile_max over the group remain well-formed.  Duplicates
    # cannot cause a miss because they add a point already inside the
    # system's spatial region.
    first_atom_per_system = batch_ptr[:-1]
    position_index = torch.where(
        sp == N,
        first_atom_per_system[padded_slot_system],
        sp,
    )
    sorted_pos = positions[position_index].contiguous()
    sorted_pos_x.copy_(sorted_pos[:, 0].contiguous())
    sorted_pos_y.copy_(sorted_pos[:, 1].contiguous())
    sorted_pos_z.copy_(sorted_pos[:, 2].contiguous())


# =============================================================================
# Component ops (torch wrappers around the warp launchers)
# =============================================================================
@torch.library.custom_op(
    "nvalchemiops::_batch_build_cluster_tile_list",
    mutates_args=(
        "sorted_atom_index",
        "sort_inv",
        "sorted_pos_x",
        "sorted_pos_y",
        "sorted_pos_z",
        "group_ctr_x",
        "group_ctr_y",
        "group_ctr_z",
        "group_ext_x",
        "group_ext_y",
        "group_ext_z",
        "inv_cell_batch",
        "num_tiles",
        "tile_row_group",
        "tile_col_group",
        "tile_system",
        "tile_counts",
    ),
)
@scoped_torch_warp_stream
def _batch_build_cluster_tile_list_op(
    positions: torch.Tensor,
    cutoff: float,
    cell_batch: torch.Tensor,
    inv_cell_batch: torch.Tensor,
    batch_ptr: torch.Tensor,
    atom_system: torch.Tensor,
    real_sorted_rank_to_padded_slot: torch.Tensor,
    sorted_atom_index: torch.Tensor,
    sort_inv: torch.Tensor,
    sorted_pos_x: torch.Tensor,
    sorted_pos_y: torch.Tensor,
    sorted_pos_z: torch.Tensor,
    batch_idx_sorted: torch.Tensor,
    batch_ptr_padded: torch.Tensor,
    group_system: torch.Tensor,
    group_ptr: torch.Tensor,
    group_ctr_x: torch.Tensor,
    group_ctr_y: torch.Tensor,
    group_ctr_z: torch.Tensor,
    group_ext_x: torch.Tensor,
    group_ext_y: torch.Tensor,
    group_ext_z: torch.Tensor,
    num_tiles: torch.Tensor,
    tile_row_group: torch.Tensor,
    tile_col_group: torch.Tensor,
    tile_system: torch.Tensor,
    rebuild_flags: torch.Tensor,
    tile_offsets: torch.Tensor,
    tile_counts: torch.Tensor,
    use_rebuild_flags: bool,
    use_segmented: bool,
    rebuild_observed: bool,
    compute_inv_cell_batch: bool,
) -> None:
    device = positions.device
    with torch.cuda.device(device):
        capturing = torch.cuda.is_current_stream_capturing()
    if (
        use_rebuild_flags
        and not capturing
        and not rebuild_observed
        and not bool(rebuild_flags.any().item())
    ):
        return
    if compute_inv_cell_batch:
        computed_inv_cell_batch, info = torch.linalg.inv_ex(
            cell_batch,
            check_errors=False,
        )
        torch._assert_async(
            (info == 0).all(),
            "cell matrices must be non-singular",
        )
        inv_cell_batch.copy_(computed_inv_cell_batch)

    wp_device = str(device)
    wp_dtype = get_wp_dtype(positions.dtype)

    _batched_morton_sort_padded(
        positions,
        batch_ptr,
        inv_cell_batch,
        atom_system,
        batch_idx_sorted,
        real_sorted_rank_to_padded_slot,
        sorted_atom_index,
        sort_inv,
        sorted_pos_x,
        sorted_pos_y,
        sorted_pos_z,
    )

    if not use_segmented:
        num_tiles.zero_()
    group_ctr_x.zero_()
    group_ctr_y.zero_()
    group_ctr_z.zero_()
    group_ext_x.zero_()
    group_ext_y.zero_()
    group_ext_z.zero_()

    wp_batch_build_cluster_tile_list(
        sorted_pos_x=wp.from_torch(sorted_pos_x, dtype=wp_dtype, return_ctype=True),
        sorted_pos_y=wp.from_torch(sorted_pos_y, dtype=wp_dtype, return_ctype=True),
        sorted_pos_z=wp.from_torch(sorted_pos_z, dtype=wp_dtype, return_ctype=True),
        group_system=wp.from_torch(group_system, dtype=wp.int32, return_ctype=True),
        group_ptr=wp.from_torch(group_ptr, dtype=wp.int32, return_ctype=True),
        cell_batch=wp.from_torch(
            cell_batch.contiguous(),
            dtype=wp.mat33f,
            return_ctype=True,
        ),
        inv_cell_batch=wp.from_torch(
            inv_cell_batch.contiguous(),
            dtype=wp.mat33f,
            return_ctype=True,
        ),
        cutoff=float(cutoff),
        num_tiles=wp.from_torch(num_tiles, dtype=wp.int32, return_ctype=True),
        tile_row_group=wp.from_torch(tile_row_group, dtype=wp.int32, return_ctype=True),
        tile_col_group=wp.from_torch(
            tile_col_group,
            dtype=wp.int32,
            return_ctype=True,
        ),
        tile_system=wp.from_torch(tile_system, dtype=wp.int32, return_ctype=True),
        wp_dtype=wp_dtype,
        device=wp_device,
        rebuild_flags=(
            wp.from_torch(rebuild_flags, dtype=wp.bool, return_ctype=True)
            if use_rebuild_flags
            else None
        ),
        tile_offsets=(
            wp.from_torch(tile_offsets, dtype=wp.int32, return_ctype=True)
            if use_segmented
            else None
        ),
        tile_counts=(
            wp.from_torch(tile_counts, dtype=wp.int32, return_ctype=True)
            if use_segmented
            else None
        ),
        group_ctr_x_buffer=wp.from_torch(
            group_ctr_x, dtype=wp_dtype, return_ctype=True
        ),
        group_ctr_y_buffer=wp.from_torch(
            group_ctr_y, dtype=wp_dtype, return_ctype=True
        ),
        group_ctr_z_buffer=wp.from_torch(
            group_ctr_z, dtype=wp_dtype, return_ctype=True
        ),
        group_ext_x_buffer=wp.from_torch(
            group_ext_x, dtype=wp_dtype, return_ctype=True
        ),
        group_ext_y_buffer=wp.from_torch(
            group_ext_y, dtype=wp_dtype, return_ctype=True
        ),
        group_ext_z_buffer=wp.from_torch(
            group_ext_z, dtype=wp_dtype, return_ctype=True
        ),
    )


@_batch_build_cluster_tile_list_op.register_fake
def _(
    positions: torch.Tensor,
    cutoff: float,
    cell_batch: torch.Tensor,
    inv_cell_batch: torch.Tensor,
    batch_ptr: torch.Tensor,
    atom_system: torch.Tensor,
    real_sorted_rank_to_padded_slot: torch.Tensor,
    sorted_atom_index: torch.Tensor,
    sort_inv: torch.Tensor,
    sorted_pos_x: torch.Tensor,
    sorted_pos_y: torch.Tensor,
    sorted_pos_z: torch.Tensor,
    batch_idx_sorted: torch.Tensor,
    batch_ptr_padded: torch.Tensor,
    group_system: torch.Tensor,
    group_ptr: torch.Tensor,
    group_ctr_x: torch.Tensor,
    group_ctr_y: torch.Tensor,
    group_ctr_z: torch.Tensor,
    group_ext_x: torch.Tensor,
    group_ext_y: torch.Tensor,
    group_ext_z: torch.Tensor,
    num_tiles: torch.Tensor,
    tile_row_group: torch.Tensor,
    tile_col_group: torch.Tensor,
    tile_system: torch.Tensor,
    rebuild_flags: torch.Tensor,
    tile_offsets: torch.Tensor,
    tile_counts: torch.Tensor,
    use_rebuild_flags: bool,
    use_segmented: bool,
    rebuild_observed: bool,
    compute_inv_cell_batch: bool,
) -> None:
    return None


def _validate_batch_cluster_tile_scratch(
    *,
    positions: torch.Tensor,
    batch_ptr: torch.Tensor,
    sorted_atom_index: torch.Tensor,
    sort_inv: torch.Tensor,
    sorted_pos_x: torch.Tensor,
    sorted_pos_y: torch.Tensor,
    sorted_pos_z: torch.Tensor,
    batch_idx_sorted: torch.Tensor,
    batch_ptr_padded: torch.Tensor,
    group_system: torch.Tensor,
    group_ptr: torch.Tensor,
    group_ctr_x: torch.Tensor,
    group_ctr_y: torch.Tensor,
    group_ctr_z: torch.Tensor,
    group_ext_x: torch.Tensor,
    group_ext_y: torch.Tensor,
    group_ext_z: torch.Tensor,
    num_tiles: torch.Tensor,
    tile_row_group: torch.Tensor,
    tile_col_group: torch.Tensor,
    tile_system: torch.Tensor,
) -> None:
    """Validate caller-owned batch build scratch before it is modified."""
    int_tensors = {
        "sorted_atom_index": sorted_atom_index,
        "sort_inv": sort_inv,
        "batch_idx_sorted": batch_idx_sorted,
        "batch_ptr_padded": batch_ptr_padded,
        "group_system": group_system,
        "group_ptr": group_ptr,
        "num_tiles": num_tiles,
        "tile_row_group": tile_row_group,
        "tile_col_group": tile_col_group,
        "tile_system": tile_system,
    }
    float_tensors = {
        "sorted_pos_x": sorted_pos_x,
        "sorted_pos_y": sorted_pos_y,
        "sorted_pos_z": sorted_pos_z,
        "group_ctr_x": group_ctr_x,
        "group_ctr_y": group_ctr_y,
        "group_ctr_z": group_ctr_z,
        "group_ext_x": group_ext_x,
        "group_ext_y": group_ext_y,
        "group_ext_z": group_ext_z,
    }
    for name, tensor in int_tensors.items():
        if tensor.device != positions.device or tensor.dtype != torch.int32:
            raise ValueError(f"{name} must be an int32 tensor on positions.device")
        if tensor.ndim != 1:
            raise ValueError(f"{name} must be 1D")
    for name, tensor in float_tensors.items():
        if tensor.device != positions.device or tensor.dtype != positions.dtype:
            raise ValueError(f"{name} must match positions device and dtype")
        if tensor.ndim != 1:
            raise ValueError(f"{name} must be 1D")

    n_atoms = positions.shape[0]
    num_systems = batch_ptr.shape[0] - 1
    if sort_inv.shape[0] != n_atoms:
        raise ValueError("sort_inv must have length N")
    if batch_ptr_padded.shape[0] != num_systems + 1:
        raise ValueError("batch_ptr_padded must have length S + 1")
    if group_ptr.shape[0] != num_systems + 1:
        raise ValueError("group_ptr must have length S + 1")
    if num_tiles.shape[0] != 1:
        raise ValueError("num_tiles must have length 1")
    tile_capacity = tile_row_group.shape[0]
    if (
        tile_col_group.shape[0] != tile_capacity
        or tile_system.shape[0] != tile_capacity
    ):
        raise ValueError(
            "tile_row_group, tile_col_group, and tile_system must have equal capacity"
        )

    padded_length = sorted_atom_index.shape[0]
    padded_tensors = (sorted_pos_x, sorted_pos_y, sorted_pos_z, batch_idx_sorted)
    if any(tensor.shape[0] != padded_length for tensor in padded_tensors):
        raise ValueError("padded atom scratch tensors must have a common length")
    if padded_length % TILE_GROUP_SIZE != 0:
        raise ValueError(
            "padded atom scratch length must be divisible by TILE_GROUP_SIZE"
        )
    ngroup = padded_length // TILE_GROUP_SIZE
    if group_system.shape[0] != ngroup:
        raise ValueError("group_system must have length ngroup")
    group_padded = (
        (ngroup + TILE_GROUP_SIZE - 1) // TILE_GROUP_SIZE
    ) * TILE_GROUP_SIZE + TILE_GROUP_SIZE
    if any(
        tensor.shape[0] != group_padded
        for tensor in (
            group_ctr_x,
            group_ctr_y,
            group_ctr_z,
            group_ext_x,
            group_ext_y,
            group_ext_z,
        )
    ):
        raise ValueError("group bounding-box scratch must have padded ngroup length")

    current_padded_atoms = (
        ((batch_ptr[1:] - batch_ptr[:-1] + TILE_GROUP_SIZE - 1) // TILE_GROUP_SIZE)
        * TILE_GROUP_SIZE
    ).sum()
    if torch.compiler.is_compiling():
        torch._assert_async(
            current_padded_atoms == padded_length,
            "scratch padded atom length must match the current batch_ptr",
        )
    elif int(current_padded_atoms.item()) != padded_length:
        raise ValueError("scratch padded atom length must match the current batch_ptr")


def _batch_build_cluster_tile_list_normalized(
    partition_metadata: _BatchPartitionMetadata | None,
    positions: torch.Tensor,
    cutoff: float,
    cell_batch: torch.Tensor,
    batch_ptr: torch.Tensor,
    sorted_atom_index: torch.Tensor,
    sort_inv: torch.Tensor,
    sorted_pos_x: torch.Tensor,
    sorted_pos_y: torch.Tensor,
    sorted_pos_z: torch.Tensor,
    batch_idx_sorted: torch.Tensor,
    batch_ptr_padded: torch.Tensor,
    group_system: torch.Tensor,
    group_ptr: torch.Tensor,
    group_ctr_x: torch.Tensor,
    group_ctr_y: torch.Tensor,
    group_ctr_z: torch.Tensor,
    group_ext_x: torch.Tensor,
    group_ext_y: torch.Tensor,
    group_ext_z: torch.Tensor,
    num_tiles: torch.Tensor,
    tile_row_group: torch.Tensor,
    tile_col_group: torch.Tensor,
    tile_system: torch.Tensor,
    inv_cell_batch: torch.Tensor | None = None,
    rebuild_flags: torch.Tensor | None = None,
    tile_offsets: torch.Tensor | None = None,
    tile_counts: torch.Tensor | None = None,
    rebuild_observed: bool = False,
) -> torch.Tensor:
    """Build batched tile neighbor list state into pre-allocated outputs.

    Runs the per-system Morton sort + padded SoA gather in torch, then
    the bbox reduction + tile-pair enumeration in warp.  All output tensors
    are modified in-place.

    Parameters
    ----------
    positions : torch.Tensor, shape (N, 3), dtype=float32
        Concatenated atomic coordinates across all systems.
    cutoff : float
        Cartesian cutoff radius used for tile-pair enumeration.
    cell_batch : torch.Tensor, shape (num_systems, 3, 3), dtype=float32
        Per-system unit cell matrices.
    batch_ptr : torch.Tensor, shape (num_systems + 1,), dtype=int32
        CSR pointer giving atom ranges per system.
    sorted_atom_index : torch.Tensor, shape (n_padded,), dtype=int32
        Output permutation array; modified in-place.
    sort_inv : torch.Tensor, shape (N,), dtype=int32
        Output inverse permutation; modified in-place.
    sorted_pos_x : torch.Tensor, shape (n_padded,), dtype=float32
        Output SoA x-positions; modified in-place.
    sorted_pos_y : torch.Tensor, shape (n_padded,), dtype=float32
        Output SoA y-positions; modified in-place.
    sorted_pos_z : torch.Tensor, shape (n_padded,), dtype=float32
        Output SoA z-positions; modified in-place.
    batch_idx_sorted : torch.Tensor, shape (n_padded,), dtype=int32
        Output system index per padded slot; modified in-place.
    batch_ptr_padded : torch.Tensor, shape (num_systems + 1,), dtype=int32
        Output padded CSR pointer; modified in-place.
    group_system : torch.Tensor, shape (ngroup,), dtype=int32
        Output system index per group; modified in-place.
    group_ptr : torch.Tensor, shape (num_systems + 1,), dtype=int32
        Output group-level CSR pointer; modified in-place.
    group_ctr_x : torch.Tensor, shape (ngroup_padded,), dtype=float32
        Output group bounding-box centre x; modified in-place.
    group_ctr_y : torch.Tensor, shape (ngroup_padded,), dtype=float32
        Output group bounding-box centre y; modified in-place.
    group_ctr_z : torch.Tensor, shape (ngroup_padded,), dtype=float32
        Output group bounding-box centre z; modified in-place.
    group_ext_x : torch.Tensor, shape (ngroup_padded,), dtype=float32
        Output group bounding-box half-extent x; modified in-place.
    group_ext_y : torch.Tensor, shape (ngroup_padded,), dtype=float32
        Output group bounding-box half-extent y; modified in-place.
    group_ext_z : torch.Tensor, shape (ngroup_padded,), dtype=float32
        Output group bounding-box half-extent z; modified in-place.
    num_tiles : torch.Tensor, shape (1,), dtype=int32
        Output tile-pair count (scalar); modified in-place.
    tile_row_group : torch.Tensor, shape (max_tiles,), dtype=int32
        Output row group per tile pair; modified in-place.
    tile_col_group : torch.Tensor, shape (max_tiles,), dtype=int32
        Output column group per tile pair; modified in-place.
    tile_system : torch.Tensor, shape (max_tiles,), dtype=int32
        Output system index per tile pair; modified in-place.
    inv_cell_batch : torch.Tensor, shape (num_systems, 3, 3), dtype=float32, optional
        Pre-computed inverse of ``cell_batch``.  Computed internally when None.
    rebuild_flags : torch.Tensor, shape (num_systems,), dtype=bool, optional
        Per-system selective rebuild flags.  Requires ``tile_offsets`` and
        ``tile_counts`` (segmented path).
    tile_offsets : torch.Tensor, shape (num_systems + 1,), dtype=int32, optional
        Fixed per-system tile segment offsets for the segmented build path.
        Must be provided together with ``tile_counts``.
    tile_counts : torch.Tensor, shape (num_systems,), dtype=int32, optional
        Output per-system tile counts for the segmented build path; modified
        in-place.  Must be provided together with ``tile_offsets``.
    """
    if positions.dtype != torch.float32:
        raise TypeError("positions must be float32")
    if (
        cell_batch.dtype != torch.float32
        or cell_batch.ndim != 3
        or cell_batch.shape[1:] != (3, 3)
    ):
        raise ValueError(
            f"cell_batch must be (S, 3, 3) float32; got {tuple(cell_batch.shape)}",
        )
    if partition_metadata is None:
        if batch_ptr.dtype != torch.int32 or batch_ptr.ndim != 1:
            raise ValueError("batch_ptr must be 1D int32")
        if batch_ptr.shape[0] < 2:
            raise ValueError("batch_ptr must have length at least 2")
    num_systems = cell_batch.shape[0]
    if batch_ptr.shape[0] != num_systems + 1:
        raise ValueError(
            f"batch_ptr length {batch_ptr.shape[0]} != num_systems+1 = {num_systems + 1}",
        )
    if partition_metadata is None:
        N = positions.shape[0]
        if torch.compiler.is_compiling():
            torch._assert_async(
                batch_ptr[-1] == N,
                "batch_ptr[-1] must equal positions.shape[0]",
            )
        elif int(batch_ptr[-1].item()) != N:
            raise ValueError(f"batch_ptr[-1] ({int(batch_ptr[-1])}) != N ({N})")
        _validate_batch_cluster_tile_scratch(
            positions=positions,
            batch_ptr=batch_ptr,
            sorted_atom_index=sorted_atom_index,
            sort_inv=sort_inv,
            sorted_pos_x=sorted_pos_x,
            sorted_pos_y=sorted_pos_y,
            sorted_pos_z=sorted_pos_z,
            batch_idx_sorted=batch_idx_sorted,
            batch_ptr_padded=batch_ptr_padded,
            group_system=group_system,
            group_ptr=group_ptr,
            group_ctr_x=group_ctr_x,
            group_ctr_y=group_ctr_y,
            group_ctr_z=group_ctr_z,
            group_ext_x=group_ext_x,
            group_ext_y=group_ext_y,
            group_ext_z=group_ext_z,
            num_tiles=num_tiles,
            tile_row_group=tile_row_group,
            tile_col_group=tile_col_group,
            tile_system=tile_system,
        )
        partition_metadata = _prepare_batch_partition_metadata(
            batch_ptr,
            num_atoms=positions.shape[0],
            padded_slot_system=batch_idx_sorted,
            batch_ptr_padded=batch_ptr_padded,
            group_system=group_system,
            group_ptr=group_ptr,
        )
    elif (
        partition_metadata.batch_ptr_padded is not batch_ptr_padded
        or partition_metadata.padded_slot_system is not batch_idx_sorted
        or partition_metadata.group_system is not group_system
        or partition_metadata.group_ptr is not group_ptr
    ):
        raise ValueError("prepared partition metadata must belong to batch scratch")

    compute_inv_cell_batch = inv_cell_batch is None and torch.compiler.is_compiling()
    if inv_cell_batch is None:
        inv_cell_batch = (
            torch.empty_like(cell_batch)
            if compute_inv_cell_batch
            else torch.linalg.inv(cell_batch).contiguous()
        )

    use_segmented = (tile_offsets is not None) or (tile_counts is not None)
    if (tile_offsets is None) != (tile_counts is None):
        raise ValueError("Pass both 'tile_offsets' and 'tile_counts', or neither.")
    use_rebuild_flags = rebuild_flags is not None
    if use_rebuild_flags and not use_segmented:
        raise ValueError(
            "rebuild_flags requires tile_offsets and tile_counts for batch_cluster_tile"
        )
    dummy_i32 = torch.empty(1, dtype=torch.int32, device=positions.device)
    dummy_bool = torch.empty(1, dtype=torch.bool, device=positions.device)

    _batch_build_cluster_tile_list_op(
        positions,
        float(cutoff),
        cell_batch,
        inv_cell_batch,
        batch_ptr,
        partition_metadata.atom_system,
        partition_metadata.real_sorted_rank_to_padded_slot,
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
        rebuild_flags if rebuild_flags is not None else dummy_bool,
        tile_offsets if tile_offsets is not None else dummy_i32,
        tile_counts if tile_counts is not None else dummy_i32,
        bool(use_rebuild_flags),
        bool(use_segmented),
        bool(rebuild_observed),
        bool(compute_inv_cell_batch),
    )
    return inv_cell_batch


def batch_build_cluster_tile_list(
    positions: torch.Tensor,
    cutoff: float,
    cell_batch: torch.Tensor,
    batch_ptr: torch.Tensor,
    sorted_atom_index: torch.Tensor,
    sort_inv: torch.Tensor,
    sorted_pos_x: torch.Tensor,
    sorted_pos_y: torch.Tensor,
    sorted_pos_z: torch.Tensor,
    batch_idx_sorted: torch.Tensor,
    batch_ptr_padded: torch.Tensor,
    group_system: torch.Tensor,
    group_ptr: torch.Tensor,
    group_ctr_x: torch.Tensor,
    group_ctr_y: torch.Tensor,
    group_ctr_z: torch.Tensor,
    group_ext_x: torch.Tensor,
    group_ext_y: torch.Tensor,
    group_ext_z: torch.Tensor,
    num_tiles: torch.Tensor,
    tile_row_group: torch.Tensor,
    tile_col_group: torch.Tensor,
    tile_system: torch.Tensor,
    inv_cell_batch: torch.Tensor | None = None,
    rebuild_flags: torch.Tensor | None = None,
    tile_offsets: torch.Tensor | None = None,
    tile_counts: torch.Tensor | None = None,
) -> None:
    """Build batched cluster-tile state into caller-owned scratch."""
    _batch_build_cluster_tile_list_normalized(
        None,
        positions,
        cutoff,
        cell_batch,
        batch_ptr,
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
        inv_cell_batch=inv_cell_batch,
        rebuild_flags=rebuild_flags,
        tile_offsets=tile_offsets,
        tile_counts=tile_counts,
    )


batch_build_cluster_tile_list.__doc__ = (
    _batch_build_cluster_tile_list_normalized.__doc__
)


@torch.library.custom_op(
    "nvalchemiops::_batch_query_cluster_tile",
    mutates_args=(
        "neighbor_matrix",
        "num_neighbors",
        "neighbor_matrix_shifts",
        "neighbor_vectors",
        "neighbor_distances",
    ),
)
@scoped_torch_warp_stream
def _batch_query_cluster_tile_op(
    cutoff: float,
    natom: int,
    sorted_atom_index: torch.Tensor,
    sorted_pos_x: torch.Tensor,
    sorted_pos_y: torch.Tensor,
    sorted_pos_z: torch.Tensor,
    cell_batch: torch.Tensor,
    inv_cell_batch: torch.Tensor,
    num_tiles: torch.Tensor,
    tile_row_group: torch.Tensor,
    tile_col_group: torch.Tensor,
    tile_system: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    num_neighbors: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    neighbor_vectors: torch.Tensor,
    neighbor_distances: torch.Tensor,
    n_tiles: int,
    return_vectors: bool,
    return_distances: bool,
) -> None:
    # ``n_tiles`` (host-synced compact tile count) tightens the launch; the
    # kernel still guards per-tile defensively.  ``n_tiles <= 0`` (segmented
    # path, count unknown here) falls back to the full-buffer launch.
    # Warp writes active pairs only. Clear padding inside the same opaque
    # mutation boundary before launching the query.
    if return_vectors:
        neighbor_vectors.zero_()
    if return_distances:
        neighbor_distances.zero_()

    wp_device = str(sorted_pos_x.device)
    wp_dtype = get_wp_dtype(sorted_pos_x.dtype)
    wp_batch_query_cluster_tile(
        n_tiles=int(n_tiles) if n_tiles > 0 else None,
        sorted_atom_index=wp.from_torch(
            sorted_atom_index, dtype=wp.int32, return_ctype=True
        ),
        sorted_pos_x=wp.from_torch(sorted_pos_x, dtype=wp_dtype, return_ctype=True),
        sorted_pos_y=wp.from_torch(sorted_pos_y, dtype=wp_dtype, return_ctype=True),
        sorted_pos_z=wp.from_torch(sorted_pos_z, dtype=wp_dtype, return_ctype=True),
        cell_batch=wp.from_torch(
            cell_batch.contiguous(),
            dtype=wp.mat33f,
            return_ctype=True,
        ),
        inv_cell_batch=wp.from_torch(
            inv_cell_batch.contiguous(),
            dtype=wp.mat33f,
            return_ctype=True,
        ),
        num_tiles=wp.from_torch(num_tiles, dtype=wp.int32, return_ctype=True),
        tile_row_group=wp.from_torch(tile_row_group, dtype=wp.int32, return_ctype=True),
        tile_col_group=wp.from_torch(
            tile_col_group,
            dtype=wp.int32,
            return_ctype=True,
        ),
        tile_system=wp.from_torch(tile_system, dtype=wp.int32, return_ctype=True),
        cutoff=float(cutoff),
        natom=int(natom),
        neighbor_matrix=wp.from_torch(
            neighbor_matrix,
            dtype=wp.int32,
            return_ctype=True,
        ),
        num_neighbors=wp.from_torch(
            num_neighbors,
            dtype=wp.int32,
            return_ctype=True,
        ),
        neighbor_matrix_shifts=wp.from_torch(
            neighbor_matrix_shifts,
            dtype=wp.int32,
            return_ctype=True,
        ),
        wp_dtype=wp_dtype,
        device=wp_device,
        return_vectors=bool(return_vectors),
        return_distances=bool(return_distances),
        neighbor_vectors=(
            wp.from_torch(neighbor_vectors, dtype=wp.vec3f, return_ctype=True)
            if return_vectors
            else None
        ),
        neighbor_distances=(
            wp.from_torch(neighbor_distances, dtype=wp_dtype, return_ctype=True)
            if return_distances
            else None
        ),
    )


@_batch_query_cluster_tile_op.register_fake
def _(
    cutoff: float,
    natom: int,
    sorted_atom_index: torch.Tensor,
    sorted_pos_x: torch.Tensor,
    sorted_pos_y: torch.Tensor,
    sorted_pos_z: torch.Tensor,
    cell_batch: torch.Tensor,
    inv_cell_batch: torch.Tensor,
    num_tiles: torch.Tensor,
    tile_row_group: torch.Tensor,
    tile_col_group: torch.Tensor,
    tile_system: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    num_neighbors: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    neighbor_vectors: torch.Tensor,
    neighbor_distances: torch.Tensor,
    n_tiles: int,
    return_vectors: bool,
    return_distances: bool,
) -> None:
    return None


@torch.library.custom_op(
    "nvalchemiops::_batch_query_cluster_tile_topology",
    mutates_args=(
        "neighbor_matrix",
        "num_neighbors",
        "neighbor_matrix_shifts",
        "neighbor_matrix2",
        "num_neighbors2",
        "neighbor_matrix_shifts2",
        "neighbor_vectors",
        "neighbor_distances",
    ),
)
@scoped_torch_warp_stream
def _batch_query_cluster_tile_topology_op(
    cutoff: float,
    cutoff2: float,
    natom: int,
    sorted_atom_index: torch.Tensor,
    sorted_pos_x: torch.Tensor,
    sorted_pos_y: torch.Tensor,
    sorted_pos_z: torch.Tensor,
    cell_batch: torch.Tensor,
    inv_cell_batch: torch.Tensor,
    num_tiles: torch.Tensor,
    tile_row_group: torch.Tensor,
    tile_col_group: torch.Tensor,
    tile_system: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    num_neighbors: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    neighbor_matrix2: torch.Tensor,
    num_neighbors2: torch.Tensor,
    neighbor_matrix_shifts2: torch.Tensor,
    neighbor_vectors: torch.Tensor,
    neighbor_distances: torch.Tensor,
    rebuild_flags: torch.Tensor,
    tile_offsets: torch.Tensor,
    tile_counts: torch.Tensor,
    batch_idx: torch.Tensor,
    n_tiles: int,
    use_cutoff2: bool,
    use_rebuild_flags: bool,
    use_segmented: bool,
    return_vectors: bool,
    return_distances: bool,
) -> None:
    if use_rebuild_flags:
        wp_selective_zero_num_neighbors(
            wp.from_torch(
                num_neighbors,
                dtype=wp.int32,
                requires_grad=False,
                return_ctype=True,
            ),
            wp.from_torch(batch_idx, dtype=wp.int32, return_ctype=True),
            wp.from_torch(rebuild_flags, dtype=wp.bool, return_ctype=True),
            str(sorted_pos_x.device),
        )
        if use_cutoff2:
            wp_selective_zero_num_neighbors(
                wp.from_torch(
                    num_neighbors2,
                    dtype=wp.int32,
                    requires_grad=False,
                    return_ctype=True,
                ),
                wp.from_torch(batch_idx, dtype=wp.int32, return_ctype=True),
                wp.from_torch(rebuild_flags, dtype=wp.bool, return_ctype=True),
                str(sorted_pos_x.device),
            )
        # Clear complete rows only for rebuilt systems. False-flag rows retain
        # active entries and padding exactly as supplied by the caller.
        row_update = rebuild_flags[batch_idx.to(torch.long)]
        if return_vectors:
            neighbor_vectors.masked_fill_(row_update[:, None, None], 0)
        if return_distances:
            neighbor_distances.masked_fill_(row_update[:, None], 0)
    else:
        if return_vectors:
            neighbor_vectors.zero_()
        if return_distances:
            neighbor_distances.zero_()
    _batch_query_cluster_tile_optional(
        cell_batch,
        inv_cell_batch,
        natom,
        cutoff,
        sorted_atom_index,
        sorted_pos_x,
        sorted_pos_y,
        sorted_pos_z,
        num_tiles,
        tile_row_group,
        tile_col_group,
        tile_system,
        neighbor_matrix,
        num_neighbors,
        neighbor_matrix_shifts,
        n_tiles=n_tiles,
        cutoff2=cutoff2 if use_cutoff2 else None,
        neighbor_matrix2=neighbor_matrix2 if use_cutoff2 else None,
        num_neighbors2=num_neighbors2 if use_cutoff2 else None,
        neighbor_matrix_shifts2=neighbor_matrix_shifts2 if use_cutoff2 else None,
        rebuild_flags=rebuild_flags if use_rebuild_flags else None,
        tile_offsets=tile_offsets if use_segmented else None,
        tile_counts=tile_counts if use_segmented else None,
        return_vectors=return_vectors,
        return_distances=return_distances,
        pair_fn=None,
        pair_params=None,
        neighbor_vectors=neighbor_vectors if return_vectors else None,
        neighbor_distances=neighbor_distances if return_distances else None,
        pair_energies=None,
        pair_forces=None,
    )


@_batch_query_cluster_tile_topology_op.register_fake
def _(
    cutoff: float,
    cutoff2: float,
    natom: int,
    sorted_atom_index: torch.Tensor,
    sorted_pos_x: torch.Tensor,
    sorted_pos_y: torch.Tensor,
    sorted_pos_z: torch.Tensor,
    cell_batch: torch.Tensor,
    inv_cell_batch: torch.Tensor,
    num_tiles: torch.Tensor,
    tile_row_group: torch.Tensor,
    tile_col_group: torch.Tensor,
    tile_system: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    num_neighbors: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    neighbor_matrix2: torch.Tensor,
    num_neighbors2: torch.Tensor,
    neighbor_matrix_shifts2: torch.Tensor,
    neighbor_vectors: torch.Tensor,
    neighbor_distances: torch.Tensor,
    rebuild_flags: torch.Tensor,
    tile_offsets: torch.Tensor,
    tile_counts: torch.Tensor,
    batch_idx: torch.Tensor,
    n_tiles: int,
    use_cutoff2: bool,
    use_rebuild_flags: bool,
    use_segmented: bool,
    return_vectors: bool,
    return_distances: bool,
) -> None:
    return None


@torch.library.custom_op(
    "nvalchemiops::_batch_cluster_tile_selective_fill_neighbor_matrix_tail",
    mutates_args=("neighbor_matrix", "neighbor_matrix2"),
)
@scoped_torch_warp_stream
def _batch_cluster_tile_selective_fill_neighbor_matrix_tail_op(
    num_neighbors: torch.Tensor,
    num_neighbors2: torch.Tensor,
    batch_idx: torch.Tensor,
    rebuild_flags: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    neighbor_matrix2: torch.Tensor,
    max_neighbors: int,
    fill_value: int,
    use_cutoff2: bool,
) -> None:
    if max_neighbors <= 0:
        return
    wp_device = str(neighbor_matrix.device)
    wp_selective_fill_neighbor_matrix_tail(
        wp.from_torch(
            num_neighbors, dtype=wp.int32, requires_grad=False, return_ctype=True
        ),
        wp.from_torch(
            batch_idx, dtype=wp.int32, requires_grad=False, return_ctype=True
        ),
        wp.from_torch(
            rebuild_flags, dtype=wp.bool, requires_grad=False, return_ctype=True
        ),
        int(num_neighbors.shape[0]),
        int(max_neighbors),
        int(fill_value),
        wp.from_torch(
            neighbor_matrix, dtype=wp.int32, requires_grad=False, return_ctype=True
        ),
        wp_device,
        batched=True,
    )
    if use_cutoff2:
        wp_selective_fill_neighbor_matrix_tail(
            wp.from_torch(
                num_neighbors2, dtype=wp.int32, requires_grad=False, return_ctype=True
            ),
            wp.from_torch(
                batch_idx, dtype=wp.int32, requires_grad=False, return_ctype=True
            ),
            wp.from_torch(
                rebuild_flags, dtype=wp.bool, requires_grad=False, return_ctype=True
            ),
            int(num_neighbors2.shape[0]),
            int(max_neighbors),
            int(fill_value),
            wp.from_torch(
                neighbor_matrix2,
                dtype=wp.int32,
                requires_grad=False,
                return_ctype=True,
            ),
            wp_device,
            batched=True,
        )


@_batch_cluster_tile_selective_fill_neighbor_matrix_tail_op.register_fake
def _(
    num_neighbors: torch.Tensor,
    num_neighbors2: torch.Tensor,
    batch_idx: torch.Tensor,
    rebuild_flags: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    neighbor_matrix2: torch.Tensor,
    max_neighbors: int,
    fill_value: int,
    use_cutoff2: bool,
) -> None:
    return None


@scoped_torch_warp_stream
def batch_query_cluster_tile(
    sorted_atom_index: torch.Tensor,
    sorted_pos_x: torch.Tensor,
    sorted_pos_y: torch.Tensor,
    sorted_pos_z: torch.Tensor,
    cell_batch: torch.Tensor,
    num_tiles: torch.Tensor,
    tile_row_group: torch.Tensor,
    tile_col_group: torch.Tensor,
    tile_system: torch.Tensor,
    cutoff: float,
    natom: int,
    neighbor_matrix: torch.Tensor,
    num_neighbors: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    inv_cell_batch: torch.Tensor | None = None,
    *,
    cutoff2: float | None = None,
    neighbor_matrix2: torch.Tensor | None = None,
    num_neighbors2: torch.Tensor | None = None,
    neighbor_matrix_shifts2: torch.Tensor | None = None,
    rebuild_flags: torch.Tensor | None = None,
    tile_offsets: torch.Tensor | None = None,
    tile_counts: torch.Tensor | None = None,
    batch_idx: torch.Tensor | None = None,
    return_vectors: bool = False,
    return_distances: bool = False,
    pair_fn: wp.Function | None = None,
    pair_params: torch.Tensor | None = None,
    neighbor_vectors: torch.Tensor | None = None,
    neighbor_distances: torch.Tensor | None = None,
    pair_energies: torch.Tensor | None = None,
    pair_forces: torch.Tensor | None = None,
) -> None:
    """Convert the batched tile pair list to neighbor_matrix in place.

    Cluster-tile does not support partial neighbor lists; there is no
    ``target_indices`` kwarg.  See
    :func:`nvalchemiops.neighbors.cluster_tile.batch_query_cluster_tile`
    for the full pair-output kwarg semantics.  Pair outputs follow
    the same pattern as the single-system :func:`query_cluster_tile`
    binding: when any pair-output kwarg is set the call bypasses the torch
    custom op and forwards directly to the warp launcher (custom ops
    cannot carry callable ``pair_fn`` across their schema boundary).

    Parameters
    ----------
    sorted_atom_index : torch.Tensor, shape (n_padded,), dtype=int32
        Permutation from padded slot to original atom index (sentinel ``N``
        marks padding slots).
    sorted_pos_x : torch.Tensor, shape (n_padded,), dtype=float32
        Morton-sorted x-coordinates in the padded SoA layout.
    sorted_pos_y : torch.Tensor, shape (n_padded,), dtype=float32
        Morton-sorted y-coordinates in the padded SoA layout.
    sorted_pos_z : torch.Tensor, shape (n_padded,), dtype=float32
        Morton-sorted z-coordinates in the padded SoA layout.
    cell_batch : torch.Tensor, shape (num_systems, 3, 3), dtype=float32
        Per-system unit cell matrices.
    num_tiles : torch.Tensor, shape (1,), dtype=int32
        Scalar tile-pair count produced by ``batch_build_cluster_tile_list``.
    tile_row_group : torch.Tensor, shape (max_tiles,), dtype=int32
        Row group index per tile pair.
    tile_col_group : torch.Tensor, shape (max_tiles,), dtype=int32
        Column group index per tile pair.
    tile_system : torch.Tensor, shape (max_tiles,), dtype=int32
        System index per tile pair.
    cutoff : float
        Cartesian cutoff radius for pair filtering.
    natom : int
        Total number of real atoms across all systems.
    neighbor_matrix : torch.Tensor, shape (natom, max_neighbors), dtype=int32
        Output neighbor indices; modified in-place.
    num_neighbors : torch.Tensor, shape (natom,), dtype=int32
        Output per-atom neighbor counts; modified in-place.
    neighbor_matrix_shifts : torch.Tensor, shape (natom, max_neighbors, 3), dtype=int32
        Output periodic image shift vectors; modified in-place.
    inv_cell_batch : torch.Tensor, shape (num_systems, 3, 3), dtype=float32, optional
        Pre-computed inverse of ``cell_batch``.  Computed internally when None.
    cutoff2 : float, optional
        Secondary cutoff for a second neighbor matrix.  Matrix-format only.
    neighbor_matrix2 : torch.Tensor, shape (natom, max_neighbors), dtype=int32, optional
        Output neighbor indices for the secondary cutoff; modified in-place.
    num_neighbors2 : torch.Tensor, shape (natom,), dtype=int32, optional
        Output per-atom counts for the secondary cutoff; modified in-place.
    neighbor_matrix_shifts2 : torch.Tensor, shape (natom, max_neighbors, 3), dtype=int32, optional
        Output shift vectors for the secondary cutoff; modified in-place.
    rebuild_flags : torch.Tensor, shape (num_systems,), dtype=bool, optional
        Per-system selective rebuild flags.  Requires ``batch_idx``,
        ``tile_offsets``, and ``tile_counts``.
    tile_offsets : torch.Tensor, shape (num_systems + 1,), dtype=int32, optional
        Fixed per-system tile segment offsets for the segmented path.
    tile_counts : torch.Tensor, shape (num_systems,), dtype=int32, optional
        Per-system tile counts for the segmented path.
    batch_idx : torch.Tensor, shape (natom,), dtype=int32, optional
        System index per atom; required when ``rebuild_flags`` is provided.
    return_vectors : bool, default False
        Write per-pair displacement vectors to ``neighbor_vectors``.
    return_distances : bool, default False
        Write per-pair scalar distances to ``neighbor_distances``.
    pair_fn : wp.Function, optional
        Module-scope Warp function evaluated for each pair.
    pair_params : torch.Tensor, optional
        Per-atom parameters forwarded to ``pair_fn``.
    neighbor_vectors : torch.Tensor, shape (natom, max_neighbors, 3), optional
        Output displacement vectors; modified in-place when ``return_vectors``.
    neighbor_distances : torch.Tensor, shape (natom, max_neighbors), optional
        Output scalar distances; modified in-place when ``return_distances``.
    pair_energies : torch.Tensor, shape (natom, max_neighbors), optional
        Output pair energies; modified in-place when ``pair_fn`` is set.
    pair_forces : torch.Tensor, shape (natom, max_neighbors, 3), optional
        Output pair forces; modified in-place when ``pair_fn`` is set.
    """

    _validate_cluster_tile_matrix_outputs(
        device=sorted_pos_x.device,
        dtype=sorted_pos_x.dtype,
        natom=int(natom),
        max_neighbors=int(neighbor_matrix.shape[1]),
        cutoff2=cutoff2,
        neighbor_matrix2=neighbor_matrix2,
        num_neighbors2=num_neighbors2,
        neighbor_matrix_shifts2=neighbor_matrix_shifts2,
        return_vectors=return_vectors,
        return_distances=return_distances,
        neighbor_vectors=neighbor_vectors,
        neighbor_distances=neighbor_distances,
        allocate_missing=False,
    )

    if inv_cell_batch is None:
        inv_cell_batch = torch.linalg.inv(cell_batch).contiguous()

    # Eager execution tightens the launch to the emitted tile count. Compiled
    # execution validates on device and launches the full static capacity. The
    # segmented path keeps the full-buffer launch and is checked per system.
    tile_capacity = int(tile_row_group.shape[0])
    if tile_offsets is None:
        n_tiles = _check_tile_buffer_capacity(num_tiles, tile_capacity)
    else:
        n_tiles = 0
        if tile_counts is not None:
            seg_caps = tile_offsets[1:] - tile_offsets[:-1]
            _check_tile_buffer_capacity(
                tile_counts,
                seg_caps,
                segmented=True,
            )

    geometry_only = (
        (return_vectors or return_distances)
        and cutoff2 is None
        and rebuild_flags is None
        and tile_offsets is None
        and tile_counts is None
        and pair_fn is None
        and pair_params is None
        and pair_energies is None
        and pair_forces is None
    )
    needs_direct = (
        cutoff2 is not None
        or rebuild_flags is not None
        or tile_offsets is not None
        or tile_counts is not None
        or _has_partial_or_pair_outputs(
            return_vectors=return_vectors,
            return_distances=return_distances,
            pair_fn=pair_fn,
            pair_params=pair_params,
            neighbor_vectors=neighbor_vectors,
            neighbor_distances=neighbor_distances,
            pair_energies=pair_energies,
            pair_forces=pair_forces,
        )
    )
    topology_only = not _has_partial_or_pair_outputs(
        return_vectors=return_vectors,
        return_distances=return_distances,
        pair_fn=pair_fn,
        pair_params=pair_params,
        neighbor_vectors=neighbor_vectors,
        neighbor_distances=neighbor_distances,
        pair_energies=pair_energies,
        pair_forces=pair_forces,
    )
    geometry_without_callback = (
        (return_vectors or return_distances)
        and pair_fn is None
        and pair_params is None
        and pair_energies is None
        and pair_forces is None
    )
    if (topology_only or geometry_without_callback) and (
        cutoff2 is not None
        or rebuild_flags is not None
        or tile_offsets is not None
        or tile_counts is not None
    ):
        if rebuild_flags is not None and batch_idx is None:
            raise ValueError("batch_idx is required when rebuild_flags is provided")
        device = sorted_pos_x.device
        dummy_matrix = torch.empty((1, 1), dtype=torch.int32, device=device)
        dummy_counts = torch.empty(1, dtype=torch.int32, device=device)
        dummy_shifts = torch.empty((1, 1, 3), dtype=torch.int32, device=device)
        dummy_vectors = torch.empty((1, 3), dtype=sorted_pos_x.dtype, device=device)
        dummy_distances = torch.empty(1, dtype=sorted_pos_x.dtype, device=device)
        dummy_i32 = torch.empty(1, dtype=torch.int32, device=device)
        dummy_bool = torch.empty(1, dtype=torch.bool, device=device)
        _batch_query_cluster_tile_topology_op(
            float(cutoff),
            float(cutoff2) if cutoff2 is not None else 0.0,
            int(natom),
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
            neighbor_matrix2 if neighbor_matrix2 is not None else dummy_matrix,
            num_neighbors2 if num_neighbors2 is not None else dummy_counts,
            neighbor_matrix_shifts2
            if neighbor_matrix_shifts2 is not None
            else dummy_shifts,
            neighbor_vectors if neighbor_vectors is not None else dummy_vectors,
            neighbor_distances if neighbor_distances is not None else dummy_distances,
            rebuild_flags if rebuild_flags is not None else dummy_bool,
            tile_offsets if tile_offsets is not None else dummy_i32,
            tile_counts if tile_counts is not None else dummy_i32,
            batch_idx if batch_idx is not None else dummy_i32,
            int(n_tiles),
            bool(cutoff2 is not None),
            bool(rebuild_flags is not None),
            bool(tile_offsets is not None),
            bool(return_vectors),
            bool(return_distances),
        )
        return
    if geometry_only:
        device = sorted_pos_x.device
        dummy_vectors = torch.empty((1, 3), dtype=sorted_pos_x.dtype, device=device)
        dummy_distances = torch.empty(1, dtype=sorted_pos_x.dtype, device=device)
        _batch_query_cluster_tile_op(
            cutoff,
            natom,
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
            neighbor_vectors if neighbor_vectors is not None else dummy_vectors,
            neighbor_distances if neighbor_distances is not None else dummy_distances,
            n_tiles,
            bool(return_vectors),
            bool(return_distances),
        )
        return
    if needs_direct:
        if rebuild_flags is not None:
            if batch_idx is None:
                raise ValueError("batch_idx is required when rebuild_flags is provided")
            wp_selective_zero_num_neighbors(
                wp.from_torch(num_neighbors, dtype=wp.int32, return_ctype=True),
                wp.from_torch(batch_idx, dtype=wp.int32, return_ctype=True),
                wp.from_torch(rebuild_flags, dtype=wp.bool, return_ctype=True),
                str(sorted_pos_x.device),
            )
            if cutoff2 is not None and num_neighbors2 is not None:
                wp_selective_zero_num_neighbors(
                    wp.from_torch(num_neighbors2, dtype=wp.int32, return_ctype=True),
                    wp.from_torch(batch_idx, dtype=wp.int32, return_ctype=True),
                    wp.from_torch(rebuild_flags, dtype=wp.bool, return_ctype=True),
                    str(sorted_pos_x.device),
                )
        _batch_query_cluster_tile_optional(
            cell_batch,
            inv_cell_batch,
            natom,
            cutoff,
            sorted_atom_index,
            sorted_pos_x,
            sorted_pos_y,
            sorted_pos_z,
            num_tiles,
            tile_row_group,
            tile_col_group,
            tile_system,
            neighbor_matrix,
            num_neighbors,
            neighbor_matrix_shifts,
            n_tiles=n_tiles,
            cutoff2=cutoff2,
            neighbor_matrix2=neighbor_matrix2,
            num_neighbors2=num_neighbors2,
            neighbor_matrix_shifts2=neighbor_matrix_shifts2,
            rebuild_flags=rebuild_flags,
            tile_offsets=tile_offsets,
            tile_counts=tile_counts,
            return_vectors=return_vectors,
            return_distances=return_distances,
            pair_fn=pair_fn,
            pair_params=pair_params,
            neighbor_vectors=neighbor_vectors,
            neighbor_distances=neighbor_distances,
            pair_energies=pair_energies,
            pair_forces=pair_forces,
        )
        return

    _batch_query_cluster_tile_op(
        cutoff,
        natom,
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
        torch.empty((1, 3), dtype=sorted_pos_x.dtype, device=sorted_pos_x.device),
        torch.empty(1, dtype=sorted_pos_x.dtype, device=sorted_pos_x.device),
        n_tiles,
        False,
        False,
    )


@scoped_torch_warp_stream
def _batch_query_cluster_tile_optional(
    cell_batch: torch.Tensor,
    inv_cell_batch: torch.Tensor,
    natom: int,
    cutoff: float,
    sorted_atom_index: torch.Tensor,
    sorted_pos_x: torch.Tensor,
    sorted_pos_y: torch.Tensor,
    sorted_pos_z: torch.Tensor,
    num_tiles: torch.Tensor,
    tile_row_group: torch.Tensor,
    tile_col_group: torch.Tensor,
    tile_system: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    num_neighbors: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    *,
    n_tiles: int,
    cutoff2: float | None,
    neighbor_matrix2: torch.Tensor | None,
    num_neighbors2: torch.Tensor | None,
    neighbor_matrix_shifts2: torch.Tensor | None,
    rebuild_flags: torch.Tensor | None,
    tile_offsets: torch.Tensor | None,
    tile_counts: torch.Tensor | None,
    return_vectors: bool,
    return_distances: bool,
    pair_fn: wp.Function | None,
    pair_params: torch.Tensor | None,
    neighbor_vectors: torch.Tensor | None,
    neighbor_distances: torch.Tensor | None,
    pair_energies: torch.Tensor | None,
    pair_forces: torch.Tensor | None,
) -> None:
    """Pair-output path for the batched binding.  Bypasses the torch
    custom op and calls the warp launcher directly so a callable
    ``pair_fn`` can cross the binding boundary.  ``n_tiles`` (compact tile
    count, or ``<= 0`` for the segmented full-buffer launch) sets the launch
    dimension; the kernel still guards per-tile defensively.
    """
    device = sorted_pos_x.device
    wp_device = str(device)
    wp_dtype = get_wp_dtype(sorted_pos_x.dtype)
    wp_vec_dtype = wp.vec3f
    wp_pair_params = (
        wp.from_torch(pair_params, dtype=wp_dtype) if pair_params is not None else None
    )
    wp_neighbor_vectors = (
        wp.from_torch(neighbor_vectors, dtype=wp_vec_dtype)
        if neighbor_vectors is not None
        else None
    )
    wp_neighbor_distances = (
        wp.from_torch(neighbor_distances, dtype=wp_dtype)
        if neighbor_distances is not None
        else None
    )
    wp_pair_energies = (
        wp.from_torch(pair_energies, dtype=wp_dtype)
        if pair_energies is not None
        else None
    )
    wp_pair_forces = (
        wp.from_torch(pair_forces, dtype=wp_vec_dtype)
        if pair_forces is not None
        else None
    )
    wp_batch_query_cluster_tile(
        sorted_atom_index=wp.from_torch(sorted_atom_index, dtype=wp.int32),
        sorted_pos_x=wp.from_torch(sorted_pos_x, dtype=wp_dtype),
        sorted_pos_y=wp.from_torch(sorted_pos_y, dtype=wp_dtype),
        sorted_pos_z=wp.from_torch(sorted_pos_z, dtype=wp_dtype),
        cell_batch=wp.from_torch(cell_batch.contiguous(), dtype=wp.mat33f),
        inv_cell_batch=wp.from_torch(inv_cell_batch.contiguous(), dtype=wp.mat33f),
        num_tiles=wp.from_torch(num_tiles, dtype=wp.int32),
        tile_row_group=wp.from_torch(tile_row_group, dtype=wp.int32),
        tile_col_group=wp.from_torch(tile_col_group, dtype=wp.int32),
        tile_system=wp.from_torch(tile_system, dtype=wp.int32),
        cutoff=float(cutoff),
        natom=int(natom),
        neighbor_matrix=wp.from_torch(neighbor_matrix, dtype=wp.int32),
        num_neighbors=wp.from_torch(num_neighbors, dtype=wp.int32),
        neighbor_matrix_shifts=wp.from_torch(neighbor_matrix_shifts, dtype=wp.int32),
        wp_dtype=wp_dtype,
        device=wp_device,
        n_tiles=int(n_tiles) if n_tiles > 0 else None,
        cutoff2=cutoff2,
        neighbor_matrix2=(
            wp.from_torch(neighbor_matrix2, dtype=wp.int32)
            if neighbor_matrix2 is not None
            else None
        ),
        num_neighbors2=(
            wp.from_torch(num_neighbors2, dtype=wp.int32)
            if num_neighbors2 is not None
            else None
        ),
        neighbor_matrix_shifts2=(
            wp.from_torch(neighbor_matrix_shifts2, dtype=wp.int32)
            if neighbor_matrix_shifts2 is not None
            else None
        ),
        rebuild_flags=(
            wp.from_torch(rebuild_flags, dtype=wp.bool)
            if rebuild_flags is not None
            else None
        ),
        tile_offsets=(
            wp.from_torch(tile_offsets, dtype=wp.int32)
            if tile_offsets is not None
            else None
        ),
        tile_counts=(
            wp.from_torch(tile_counts, dtype=wp.int32)
            if tile_counts is not None
            else None
        ),
        return_vectors=bool(return_vectors),
        return_distances=bool(return_distances),
        pair_fn=pair_fn,
        pair_params=wp_pair_params,
        neighbor_vectors=wp_neighbor_vectors,
        neighbor_distances=wp_neighbor_distances,
        pair_energies=wp_pair_energies,
        pair_forces=wp_pair_forces,
    )


@torch.library.custom_op(
    "nvalchemiops::_batch_query_cluster_tile_coo",
    mutates_args=("pair_counter", "coo_list", "coo_shifts"),
)
@scoped_torch_warp_stream
def _batch_query_cluster_tile_coo_op(
    cutoff: float,
    natom: int,
    max_pairs: int,
    sorted_atom_index: torch.Tensor,
    sorted_pos_x: torch.Tensor,
    sorted_pos_y: torch.Tensor,
    sorted_pos_z: torch.Tensor,
    cell_batch: torch.Tensor,
    inv_cell_batch: torch.Tensor,
    num_tiles: torch.Tensor,
    tile_row_group: torch.Tensor,
    tile_col_group: torch.Tensor,
    tile_system: torch.Tensor,
    pair_counter: torch.Tensor,
    coo_list: torch.Tensor,
    coo_shifts: torch.Tensor,
    n_tiles: int,
) -> None:
    # ``n_tiles`` (host-synced compact tile count) tightens the launch; the
    # kernel still guards per-tile defensively.
    wp_device = str(sorted_pos_x.device)
    wp_dtype = get_wp_dtype(sorted_pos_x.dtype)
    wp_batch_query_cluster_tile_coo(
        n_tiles=int(n_tiles) if n_tiles > 0 else None,
        sorted_atom_index=wp.from_torch(
            sorted_atom_index, dtype=wp.int32, return_ctype=True
        ),
        sorted_pos_x=wp.from_torch(sorted_pos_x, dtype=wp_dtype, return_ctype=True),
        sorted_pos_y=wp.from_torch(sorted_pos_y, dtype=wp_dtype, return_ctype=True),
        sorted_pos_z=wp.from_torch(sorted_pos_z, dtype=wp_dtype, return_ctype=True),
        cell_batch=wp.from_torch(
            cell_batch.contiguous(),
            dtype=wp.mat33f,
            return_ctype=True,
        ),
        inv_cell_batch=wp.from_torch(
            inv_cell_batch.contiguous(),
            dtype=wp.mat33f,
            return_ctype=True,
        ),
        num_tiles=wp.from_torch(num_tiles, dtype=wp.int32, return_ctype=True),
        tile_row_group=wp.from_torch(tile_row_group, dtype=wp.int32, return_ctype=True),
        tile_col_group=wp.from_torch(
            tile_col_group,
            dtype=wp.int32,
            return_ctype=True,
        ),
        tile_system=wp.from_torch(tile_system, dtype=wp.int32, return_ctype=True),
        cutoff=float(cutoff),
        natom=int(natom),
        max_pairs=int(max_pairs),
        pair_counter=wp.from_torch(pair_counter, dtype=wp.int32, return_ctype=True),
        coo_list=wp.from_torch(coo_list, dtype=wp.int32, return_ctype=True),
        coo_shifts=wp.from_torch(coo_shifts, dtype=wp.int32, return_ctype=True),
        wp_dtype=wp_dtype,
        device=wp_device,
    )


@_batch_query_cluster_tile_coo_op.register_fake
def _(
    cutoff: float,
    natom: int,
    max_pairs: int,
    sorted_atom_index: torch.Tensor,
    sorted_pos_x: torch.Tensor,
    sorted_pos_y: torch.Tensor,
    sorted_pos_z: torch.Tensor,
    cell_batch: torch.Tensor,
    inv_cell_batch: torch.Tensor,
    num_tiles: torch.Tensor,
    tile_row_group: torch.Tensor,
    tile_col_group: torch.Tensor,
    tile_system: torch.Tensor,
    pair_counter: torch.Tensor,
    coo_list: torch.Tensor,
    coo_shifts: torch.Tensor,
    n_tiles: int,
) -> None:
    return None


@scoped_torch_warp_stream
def _batch_query_cluster_tile_coo_optional(
    cell_batch: torch.Tensor,
    inv_cell_batch: torch.Tensor,
    natom: int,
    max_pairs: int,
    cutoff: float,
    sorted_atom_index: torch.Tensor,
    sorted_pos_x: torch.Tensor,
    sorted_pos_y: torch.Tensor,
    sorted_pos_z: torch.Tensor,
    num_tiles: torch.Tensor,
    tile_row_group: torch.Tensor,
    tile_col_group: torch.Tensor,
    tile_system: torch.Tensor,
    pair_counter: torch.Tensor,
    coo_list: torch.Tensor,
    coo_shifts: torch.Tensor,
    *,
    n_tiles: int,
    rebuild_flags: torch.Tensor | None,
    tile_offsets: torch.Tensor | None,
    tile_counts: torch.Tensor | None,
    pair_offsets: torch.Tensor | None,
    pair_counts: torch.Tensor | None,
    return_vectors: bool,
    return_distances: bool,
    pair_fn: wp.Function | None,
    pair_params: torch.Tensor | None,
    neighbor_vectors: torch.Tensor | None,
    neighbor_distances: torch.Tensor | None,
    pair_energies: torch.Tensor | None,
    pair_forces: torch.Tensor | None,
) -> None:
    """Pair-output COO path for the batched binding.

    ``n_tiles`` (compact tile count, or ``<= 0`` for the segmented full-buffer
    launch) sets the launch dimension; the kernel still guards per-tile.
    """
    wp_device = str(sorted_pos_x.device)
    wp_dtype = get_wp_dtype(sorted_pos_x.dtype)
    wp_vec_dtype = wp.vec3f
    wp_pair_params = (
        wp.from_torch(pair_params, dtype=wp_dtype) if pair_params is not None else None
    )
    wp_neighbor_vectors = (
        wp.from_torch(neighbor_vectors, dtype=wp_vec_dtype)
        if neighbor_vectors is not None
        else None
    )
    wp_neighbor_distances = (
        wp.from_torch(neighbor_distances, dtype=wp_dtype)
        if neighbor_distances is not None
        else None
    )
    wp_pair_energies = (
        wp.from_torch(pair_energies, dtype=wp_dtype)
        if pair_energies is not None
        else None
    )
    wp_pair_forces = (
        wp.from_torch(pair_forces, dtype=wp_vec_dtype)
        if pair_forces is not None
        else None
    )
    wp_batch_query_cluster_tile_coo(
        sorted_atom_index=wp.from_torch(sorted_atom_index, dtype=wp.int32),
        sorted_pos_x=wp.from_torch(sorted_pos_x, dtype=wp_dtype),
        sorted_pos_y=wp.from_torch(sorted_pos_y, dtype=wp_dtype),
        sorted_pos_z=wp.from_torch(sorted_pos_z, dtype=wp_dtype),
        cell_batch=wp.from_torch(cell_batch.contiguous(), dtype=wp.mat33f),
        inv_cell_batch=wp.from_torch(inv_cell_batch.contiguous(), dtype=wp.mat33f),
        num_tiles=wp.from_torch(num_tiles, dtype=wp.int32),
        tile_row_group=wp.from_torch(tile_row_group, dtype=wp.int32),
        tile_col_group=wp.from_torch(tile_col_group, dtype=wp.int32),
        tile_system=wp.from_torch(tile_system, dtype=wp.int32),
        cutoff=float(cutoff),
        natom=int(natom),
        max_pairs=int(max_pairs),
        pair_counter=wp.from_torch(pair_counter, dtype=wp.int32),
        coo_list=wp.from_torch(coo_list, dtype=wp.int32),
        coo_shifts=wp.from_torch(coo_shifts, dtype=wp.int32),
        wp_dtype=wp_dtype,
        device=wp_device,
        n_tiles=int(n_tiles) if n_tiles > 0 else None,
        rebuild_flags=(
            wp.from_torch(rebuild_flags, dtype=wp.bool)
            if rebuild_flags is not None
            else None
        ),
        tile_offsets=(
            wp.from_torch(tile_offsets, dtype=wp.int32)
            if tile_offsets is not None
            else None
        ),
        tile_counts=(
            wp.from_torch(tile_counts, dtype=wp.int32)
            if tile_counts is not None
            else None
        ),
        pair_offsets=(
            wp.from_torch(pair_offsets, dtype=wp.int32)
            if pair_offsets is not None
            else None
        ),
        pair_counts=(
            wp.from_torch(pair_counts, dtype=wp.int32)
            if pair_counts is not None
            else None
        ),
        return_vectors=bool(return_vectors),
        return_distances=bool(return_distances),
        pair_fn=pair_fn,
        pair_params=wp_pair_params,
        neighbor_vectors=wp_neighbor_vectors,
        neighbor_distances=wp_neighbor_distances,
        pair_energies=wp_pair_energies,
        pair_forces=wp_pair_forces,
    )


def batch_query_cluster_tile_coo(
    sorted_atom_index: torch.Tensor,
    sorted_pos_x: torch.Tensor,
    sorted_pos_y: torch.Tensor,
    sorted_pos_z: torch.Tensor,
    cell_batch: torch.Tensor,
    num_tiles: torch.Tensor,
    tile_row_group: torch.Tensor,
    tile_col_group: torch.Tensor,
    tile_system: torch.Tensor,
    cutoff: float,
    natom: int,
    max_pairs: int,
    pair_counter: torch.Tensor,
    coo_list: torch.Tensor,
    coo_shifts: torch.Tensor,
    inv_cell_batch: torch.Tensor | None = None,
    *,
    rebuild_flags: torch.Tensor | None = None,
    tile_offsets: torch.Tensor | None = None,
    tile_counts: torch.Tensor | None = None,
    pair_offsets: torch.Tensor | None = None,
    pair_counts: torch.Tensor | None = None,
    return_vectors: bool = False,
    return_distances: bool = False,
    pair_fn: wp.Function | None = None,
    pair_params: torch.Tensor | None = None,
    neighbor_vectors: torch.Tensor | None = None,
    neighbor_distances: torch.Tensor | None = None,
    pair_energies: torch.Tensor | None = None,
    pair_forces: torch.Tensor | None = None,
) -> None:
    """Convert the batched tile pair list to flat COO pair list in place.

    Cluster-tile does not support partial neighbor lists; there is no
    ``target_indices`` kwarg.  Optional pair outputs use flat COO
    buffers with length ``max_pairs``; they are written in the same
    order as ``coo_list``.

    Parameters
    ----------
    sorted_atom_index : torch.Tensor, shape (n_padded,), dtype=int32
        Permutation from padded slot to original atom index.
    sorted_pos_x : torch.Tensor, shape (n_padded,), dtype=float32
        Morton-sorted x-coordinates in the padded SoA layout.
    sorted_pos_y : torch.Tensor, shape (n_padded,), dtype=float32
        Morton-sorted y-coordinates in the padded SoA layout.
    sorted_pos_z : torch.Tensor, shape (n_padded,), dtype=float32
        Morton-sorted z-coordinates in the padded SoA layout.
    cell_batch : torch.Tensor, shape (num_systems, 3, 3), dtype=float32
        Per-system unit cell matrices.
    num_tiles : torch.Tensor, shape (1,), dtype=int32
        Scalar tile-pair count from ``batch_build_cluster_tile_list``.
    tile_row_group : torch.Tensor, shape (max_tiles,), dtype=int32
        Row group index per tile pair.
    tile_col_group : torch.Tensor, shape (max_tiles,), dtype=int32
        Column group index per tile pair.
    tile_system : torch.Tensor, shape (max_tiles,), dtype=int32
        System index per tile pair.
    cutoff : float
        Cartesian cutoff radius for pair filtering.
    natom : int
        Total number of real atoms across all systems.
    max_pairs : int
        Upper bound on the total number of pairs written to ``coo_list``.
    pair_counter : torch.Tensor, shape (1,), dtype=int32
        Output scalar pair count; zeroed then modified in-place.
    coo_list : torch.Tensor, shape (max_pairs, 2), dtype=int32
        Output COO pair list ``[i, j]``; modified in-place.
    coo_shifts : torch.Tensor, shape (max_pairs, 3), dtype=int32
        Output periodic image shift vectors per pair; modified in-place.
    inv_cell_batch : torch.Tensor, shape (num_systems, 3, 3), dtype=float32, optional
        Pre-computed inverse of ``cell_batch``.  Computed internally when None.
    rebuild_flags : torch.Tensor, shape (num_systems,), dtype=bool, optional
        Per-system selective rebuild flags.  Requires ``tile_offsets``,
        ``tile_counts``, ``pair_offsets``, and ``pair_counts``.
    tile_offsets : torch.Tensor, shape (num_systems + 1,), dtype=int32, optional
        Fixed per-system tile segment offsets for the segmented path.
    tile_counts : torch.Tensor, shape (num_systems,), dtype=int32, optional
        Per-system tile counts for the segmented path.
    pair_offsets : torch.Tensor, shape (num_systems + 1,), dtype=int32, optional
        Fixed per-system COO segment offsets for the segmented path.
    pair_counts : torch.Tensor, shape (num_systems,), dtype=int32, optional
        Output per-system pair counts for the segmented path; modified in-place.
    return_vectors : bool, default False
        Write per-pair displacement vectors to ``neighbor_vectors``.
    return_distances : bool, default False
        Write per-pair scalar distances to ``neighbor_distances``.
    pair_fn : wp.Function, optional
        Module-scope Warp function evaluated for each pair.
    pair_params : torch.Tensor, optional
        Per-atom parameters forwarded to ``pair_fn``.
    neighbor_vectors : torch.Tensor, shape (max_pairs, 3), optional
        Output displacement vectors; modified in-place when ``return_vectors``.
    neighbor_distances : torch.Tensor, shape (max_pairs,), optional
        Output scalar distances; modified in-place when ``return_distances``.
    pair_energies : torch.Tensor, shape (max_pairs,), optional
        Output pair energies; modified in-place when ``pair_fn`` is set.
    pair_forces : torch.Tensor, shape (max_pairs, 3), optional
        Output pair forces; modified in-place when ``pair_fn`` is set.
    """

    if pair_offsets is not None:
        if pair_counts is None:
            raise ValueError("pair_counts is required with pair_offsets")
        _validate_segmented_coo_state(
            device=coo_list.device,
            num_systems=int(pair_offsets.shape[0]) - 1,
            neighbor_list=coo_list.transpose(0, 1),
            neighbor_list_shifts=coo_shifts,
            pair_offsets=pair_offsets,
            pair_counts=pair_counts,
            rebuild_flags=rebuild_flags,
            tile_offsets=tile_offsets,
            tile_counts=tile_counts,
            num_tiles=num_tiles,
            tile_row_group=tile_row_group,
            tile_col_group=tile_col_group,
            tile_system=tile_system,
        )

    if inv_cell_batch is None:
        inv_cell_batch = torch.linalg.inv(cell_batch).contiguous()
    pair_counter.zero_()

    # Eager execution tightens the launch to the emitted tile count. Compiled
    # execution validates on device and launches the full static capacity. The
    # segmented path keeps the full-buffer launch and is checked per system.
    tile_capacity = int(tile_row_group.shape[0])
    if tile_offsets is None:
        n_tiles = _check_tile_buffer_capacity(num_tiles, tile_capacity)
    else:
        n_tiles = 0
        if tile_counts is not None:
            seg_caps = tile_offsets[1:] - tile_offsets[:-1]
            _check_tile_buffer_capacity(
                tile_counts,
                seg_caps,
                segmented=True,
            )

    needs_direct = (
        rebuild_flags is not None
        or tile_offsets is not None
        or tile_counts is not None
        or pair_offsets is not None
        or pair_counts is not None
        or _has_partial_or_pair_outputs(
            return_vectors=return_vectors,
            return_distances=return_distances,
            pair_fn=pair_fn,
            pair_params=pair_params,
            neighbor_vectors=neighbor_vectors,
            neighbor_distances=neighbor_distances,
            pair_energies=pair_energies,
            pair_forces=pair_forces,
        )
    )
    if needs_direct:
        _batch_query_cluster_tile_coo_optional(
            cell_batch,
            inv_cell_batch,
            natom,
            max_pairs,
            cutoff,
            sorted_atom_index,
            sorted_pos_x,
            sorted_pos_y,
            sorted_pos_z,
            num_tiles,
            tile_row_group,
            tile_col_group,
            tile_system,
            pair_counter,
            coo_list,
            coo_shifts,
            n_tiles=n_tiles,
            rebuild_flags=rebuild_flags,
            tile_offsets=tile_offsets,
            tile_counts=tile_counts,
            pair_offsets=pair_offsets,
            pair_counts=pair_counts,
            return_vectors=return_vectors,
            return_distances=return_distances,
            pair_fn=pair_fn,
            pair_params=pair_params,
            neighbor_vectors=neighbor_vectors,
            neighbor_distances=neighbor_distances,
            pair_energies=pair_energies,
            pair_forces=pair_forces,
        )
        return

    _batch_query_cluster_tile_coo_op(
        cutoff,
        natom,
        max_pairs,
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
        n_tiles,
    )


def _validate_batch_matrix_state(
    *,
    device: torch.device,
    natom: int,
    num_systems: int,
    max_neighbors: int,
    neighbor_matrix: torch.Tensor,
    num_neighbors: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    num_tiles: torch.Tensor,
    tile_offsets: torch.Tensor,
    tile_counts: torch.Tensor,
    tile_row_group: torch.Tensor,
    tile_col_group: torch.Tensor,
    tile_system: torch.Tensor,
    neighbor_matrix2: torch.Tensor | None = None,
    num_neighbors2: torch.Tensor | None = None,
    neighbor_matrix_shifts2: torch.Tensor | None = None,
) -> None:
    """Validate caller-owned selective matrix and tile state."""
    state = {
        "neighbor_matrix": neighbor_matrix,
        "num_neighbors": num_neighbors,
        "neighbor_matrix_shifts": neighbor_matrix_shifts,
        "num_tiles": num_tiles,
        "tile_offsets": tile_offsets,
        "tile_counts": tile_counts,
        "tile_row_group": tile_row_group,
        "tile_col_group": tile_col_group,
        "tile_system": tile_system,
    }
    for name, value in state.items():
        if value.device != device or value.dtype != torch.int32:
            raise ValueError(f"{name} must be an int32 tensor on positions.device")
    if neighbor_matrix.shape != (natom, max_neighbors):
        raise ValueError("neighbor_matrix must have shape (N, max_neighbors)")
    if num_neighbors.shape != (natom,):
        raise ValueError("num_neighbors must have shape (N,)")
    if neighbor_matrix_shifts.shape != (natom, max_neighbors, 3):
        raise ValueError("neighbor_matrix_shifts must have shape (N, max_neighbors, 3)")
    if (
        tile_offsets.shape != (num_systems + 1,)
        or tile_counts.shape != (num_systems,)
        or num_tiles.shape != (1,)
        or tile_row_group.ndim != 1
        or tile_col_group.shape != tile_row_group.shape
        or tile_system.shape != tile_row_group.shape
    ):
        raise ValueError("batched tile state has an invalid shape")
    for name, matrix, counts, shifts in (
        ("secondary", neighbor_matrix2, num_neighbors2, neighbor_matrix_shifts2),
    ):
        if matrix is not None:
            if (
                counts is None
                or shifts is None
                or matrix.device != device
                or matrix.dtype != torch.int32
                or counts.device != device
                or counts.dtype != torch.int32
                or shifts.device != device
                or shifts.dtype != torch.int32
                or matrix.shape != (natom, max_neighbors)
                or counts.shape != (natom,)
                or shifts.shape != (natom, max_neighbors, 3)
            ):
                raise ValueError(f"{name} matrix state has an invalid shape or dtype")


def _batch_cluster_tile_neighbor_list_impl(
    partition_metadata: _BatchPartitionMetadata | None,
    positions: torch.Tensor,
    cutoff: float | None = None,
    cell_batch: torch.Tensor | None = None,
    batch_ptr: torch.Tensor | None = None,
    max_neighbors: int | None = None,
    fill_value: int | None = None,
    format: str = "matrix",
    max_pairs: int | None = None,
    cutoff2: float | None = None,
    rebuild_flags: torch.Tensor | None = None,
    return_state: bool = False,
    # matrix-format outputs
    neighbor_matrix: torch.Tensor | None = None,
    neighbor_matrix_shifts: torch.Tensor | None = None,
    num_neighbors: torch.Tensor | None = None,
    neighbor_matrix2: torch.Tensor | None = None,
    neighbor_matrix_shifts2: torch.Tensor | None = None,
    num_neighbors2: torch.Tensor | None = None,
    # coo-format outputs
    neighbor_list: torch.Tensor | None = None,
    neighbor_list_shifts: torch.Tensor | None = None,
    pair_counter: torch.Tensor | None = None,
    pair_offsets: torch.Tensor | None = None,
    pair_counts: torch.Tensor | None = None,
    inv_cell_batch: torch.Tensor | None = None,
    # scratch buffers
    sorted_atom_index: torch.Tensor | None = None,
    sort_inv: torch.Tensor | None = None,
    sorted_pos_x: torch.Tensor | None = None,
    sorted_pos_y: torch.Tensor | None = None,
    sorted_pos_z: torch.Tensor | None = None,
    batch_idx_sorted: torch.Tensor | None = None,
    batch_ptr_padded: torch.Tensor | None = None,
    group_system: torch.Tensor | None = None,
    group_ptr: torch.Tensor | None = None,
    group_ctr_x: torch.Tensor | None = None,
    group_ctr_y: torch.Tensor | None = None,
    group_ctr_z: torch.Tensor | None = None,
    group_ext_x: torch.Tensor | None = None,
    group_ext_y: torch.Tensor | None = None,
    group_ext_z: torch.Tensor | None = None,
    num_tiles: torch.Tensor | None = None,
    tile_row_group: torch.Tensor | None = None,
    tile_col_group: torch.Tensor | None = None,
    tile_system: torch.Tensor | None = None,
    tile_offsets: torch.Tensor | None = None,
    tile_counts: torch.Tensor | None = None,
    # Optional matrix outputs / pair-fn surface
    return_vectors: bool = False,
    return_distances: bool = False,
    pair_fn: wp.Function | None = None,
    pair_params: torch.Tensor | None = None,
    neighbor_vectors: torch.Tensor | None = None,
    neighbor_distances: torch.Tensor | None = None,
    pair_energies: torch.Tensor | None = None,
    pair_forces: torch.Tensor | None = None,
    max_tiles_per_group: int | None = None,
    *,
    state: "ClusterTileState | None" = None,
) -> tuple[torch.Tensor, ...]:
    """Build and query a batched cluster-pair tile neighbor list in one call.

    Batched PyTorch binding for the cluster-pair tile algorithm.  Supports
    triclinic ``cell_batch`` of shape ``(num_systems, 3, 3)`` and arbitrary
    per-system atom counts (padded internally to a multiple of
    ``TILE_GROUP_SIZE``). Cluster-tile is CUDA float32 only.

    Parameters
    ----------
    positions : torch.Tensor, shape (total_atoms, 3), dtype=float32
        Concatenated atomic coordinates across systems.
    cutoff : float, optional
        Cutoff distance in Cartesian units. Must be positive. Required without
        ``state`` and ignored when ``state`` is provided.
    cell_batch : torch.Tensor, shape (num_systems, 3, 3), dtype=float32
        Per-system unit cell matrices. Cluster-tile assumes fully
        periodic boundaries. Required on every call, including prepared
        execution.
    batch_ptr : torch.Tensor, shape (num_systems + 1,), dtype=int32 or int64, optional
        CSR pointer separating systems.  Assumes positions are laid out
        in system-contiguous order (system 0 atoms first, then system 1,
        and so on); interleaved layouts are **not supported** and will
        silently emit cross-system pairs.  When invoked via
        :func:`nvalchemiops.torch.neighbors.neighbor_list` with a
        ``batch_idx`` argument, the dispatcher derives ``batch_ptr`` by
        assuming ``batch_idx`` is sorted by system — the same contract.
        Required without ``state``. A prepared state owns its partition, so a
        supplied ``batch_ptr`` is ignored when ``state`` is provided.
    max_neighbors : int, optional
        Falls back to ``estimate_max_neighbors`` using the larger active cutoff.
        Matrix format only.
    fill_value : int, optional
        Matrix sentinel; defaults to ``total_atoms``.
    format : {"matrix", "coo", "tile"}, default "matrix"
        Output representation. See Returns.
    max_pairs : int, optional
        Upper bound for COO output; defaults to
        ``total_atoms * max_neighbors``.
    cutoff2 : float, optional
        Cutoff for the second matrix. It is normally the outer cutoff and may
        equal ``cutoff``. Either order is accepted because the tile buffer is
        sized for the larger value. Dual cutoff is matrix-only and cannot be
        combined with pair-output buffers.
    rebuild_flags : torch.Tensor, shape (num_systems,), dtype=torch.bool, optional
        Per-system selective rebuild flags. Supported for matrix output and
        segmented COO output. Across selective calls, callers must retain and
        reuse ``tile_offsets``, ``tile_counts``, and the complete scratch/tile
        block triggered by ``sorted_atom_index``. Matrix output additionally
        requires persistent neighbor matrix/count/shift buffers for every
        active cutoff when an unflagged system must be preserved.
    return_state : bool, default=False
        When used with ``rebuild_flags``, append the reusable tile state
        ``(tile_offsets, tile_counts, num_tiles, tile_row_group,
        tile_col_group, tile_system)`` to matrix or segmented-COO output.
        These are the same tensor objects as caller-supplied state buffers.
        Requires ``rebuild_flags``.
    tile_offsets, tile_counts : torch.Tensor, optional
        Fixed per-system tile offsets and OUTPUT tile counters for segmented
        tile-list state. Use ``estimate_batch_cluster_tile_segments`` to size
        these arrays. Caller-owned persistent buffers are required across
        selective rebuild calls.
    pair_offsets, pair_counts : torch.Tensor, optional
        Fixed per-system COO offsets and OUTPUT pair counters for segmented
        COO output. Compact COO cannot be combined with ``rebuild_flags``.
        Before launch, offsets, counts, topology buffers, and tile state are
        validated against their physical capacities. Eager calls containing a
        false flag require caller-owned topology and tile state; batched
        ``torch.compile(fullgraph=True)`` selective COO is not supported.
    neighbor_matrix, num_neighbors, neighbor_matrix_shifts : torch.Tensor, optional
        Pre-allocated matrix-format outputs for the primary cutoff, shapes
        ``(total_atoms, max_neighbors)``, ``(total_atoms,)``, and
        ``(total_atoms, max_neighbors, 3)`` respectively, dtype int32.
        All-or-nothing: provide all three or none; when omitted, the trio is
        allocated. The complete trio must be caller-owned persistent state when
        selective rebuild must preserve prior neighbor topology for an
        unflagged system, and for compiled selective matrix calls.
    neighbor_matrix2 : torch.Tensor, shape (total_atoms, max_neighbors), dtype=torch.int32, optional
        Pre-allocated neighbor indices for the secondary cutoff
        (``cutoff2``). All-or-nothing with ``num_neighbors2`` and
        ``neighbor_matrix_shifts2``: provide the complete trio or none. With
        selective rebuild, retain and reuse the complete trio across calls
        that preserve an unflagged system.
    neighbor_matrix_shifts2 : torch.Tensor, shape (total_atoms, max_neighbors, 3), dtype=torch.int32, optional
        Pre-allocated periodic shift vectors for the secondary cutoff.
    num_neighbors2 : torch.Tensor, shape (total_atoms,), dtype=torch.int32, optional
        Pre-allocated per-atom neighbor counts for the secondary cutoff.
    neighbor_list, neighbor_list_shifts, pair_counter : torch.Tensor, optional
        Pre-allocated COO-format outputs. Shapes ``(2, max_pairs)``,
        ``(max_pairs, 3)``, and ``(1,)``, dtype int32 and on the same device
        as ``positions``. Compact COO buffers may be supplied independently;
        omitted buffers are allocated. Selective calls that preserve an
        unflagged system require complete caller-owned segmented topology
        state.
    inv_cell_batch : torch.Tensor, optional
        Pre-computed inverse cell matrices.
    sorted_atom_index, sort_inv, sorted_pos_x, sorted_pos_y, sorted_pos_z, batch_idx_sorted, batch_ptr_padded, group_system, group_ptr : torch.Tensor, optional
        Pre-allocated scratch buffers (shapes as returned by
        ``allocate_batch_cluster_tile_list``). All-or-nothing: provide
        every scratch tensor or none. The trigger is ``sorted_atom_index``.
        Retain the complete scratch block across selective rebuild calls.
    group_ctr_x : torch.Tensor, shape (ngroup_padded,), dtype=float32, optional
        Pre-allocated group bounding-box centre x-coordinate scratch; written
        by build.  Part of the all-or-nothing scratch block triggered by
        ``sorted_atom_index``.
    group_ctr_y : torch.Tensor, shape (ngroup_padded,), dtype=float32, optional
        Pre-allocated group bounding-box centre y-coordinate scratch; written
        by build.
    group_ctr_z : torch.Tensor, shape (ngroup_padded,), dtype=float32, optional
        Pre-allocated group bounding-box centre z-coordinate scratch; written
        by build.
    group_ext_x : torch.Tensor, shape (ngroup_padded,), dtype=float32, optional
        Pre-allocated group bounding-box half-extent x scratch; written by build.
    group_ext_y : torch.Tensor, shape (ngroup_padded,), dtype=float32, optional
        Pre-allocated group bounding-box half-extent y scratch; written by build.
    group_ext_z : torch.Tensor, shape (ngroup_padded,), dtype=float32, optional
        Pre-allocated group bounding-box half-extent z scratch; written by build.
    num_tiles, tile_row_group, tile_col_group, tile_system : torch.Tensor, optional
        Pre-allocated tile-list state (shapes as returned by
        ``allocate_batch_cluster_tile_list``).  Part of the all-or-nothing
        scratch block triggered by ``sorted_atom_index`` and persistent across
        selective rebuild calls.
    return_vectors, return_distances : bool, default False
        Return per-pair displacements / distances on nonselective calls and
        write their active prefixes to supplied ``neighbor_vectors`` /
        ``neighbor_distances`` buffers. Matrix format uses
        ``(total_atoms, max_neighbors, ...)`` buffers; compact COO uses flat
        capacity buffers and returns newly allocated exact geometry.
    pair_fn : wp.Function, optional
        Module-scope Warp ``@wp.func`` of signature
        ``(r_ij, distance, pair_params, i, j) -> (energy, force)``.
    pair_params : torch.Tensor, optional
        Per-atom pair-function parameters; required with ``pair_fn``.
        ``pair_params`` is rejected with prepared state because prepared pair
        callbacks are unsupported.
    neighbor_vectors, neighbor_distances, pair_energies, pair_forces : torch.Tensor, optional
        OUTPUT buffers, written only when the corresponding enable flag
        / ``pair_fn`` is active. Geometry buffers must not require gradients.
        When geometry is reconstructed for autograd, supplied buffers receive
        detached value snapshots while the returned geometry uses separate
        differentiable tensors. Omitted geometry buffers are not allocated.
    max_tiles_per_group : int, optional
        Capacity factor for an internally allocated intermediate tile-pair
        buffer. A system with ``g_i`` row groups contributes
        ``g_i * min(g_i, max_tiles_per_group)`` entries to one pooled buffer.
        Eager calls estimate the value when it is ``None``. Caller-owned tile
        arrays determine the actual physical capacity. See
        :ref:`cluster-tile-buffer-capacity` for sizing details.
    state : ClusterTileState, optional
        Prepared configuration and reusable storage returned by
        :func:`prepare_cluster_tile`. The cell remains a required per-call
        input. The state controls cutoffs, format, capacities, optional outputs,
        and scratch storage; redundant static arguments are ignored. Explicit
        scratch or output buffers and ``return_state=True`` are rejected. The
        state must be batched; its prepared partition is used even when
        ``batch_ptr`` is supplied.

    Returns
    -------
    tuple of torch.Tensor
        Shape depends on ``format``:

        - ``"matrix"`` (default): ``(neighbor_matrix, num_neighbors,
          neighbor_matrix_shifts)``. When ``cutoff2`` is set, the secondary
          ``(neighbor_matrix2, num_neighbors2, neighbor_matrix_shifts2)``
          triple follows the primary triple. On the non-selective path without
          ``pair_fn``, optional distances and/or vectors are appended when
          ``return_distances`` / ``return_vectors`` is True. Selective queries
          and ``pair_fn`` queries instead write requested pair-output buffers
          in place and return only the topology triple (or dual-cutoff
          sextuple). Selective calls with
          ``return_state=True`` append ``(tile_offsets, tile_counts,
          num_tiles, tile_row_group, tile_col_group, tile_system)``, yielding
          nine tensors for one cutoff or twelve for two cutoffs.
          Differentiable reconstructed geometry does not alias supplied output
          buffers; otherwise returned geometry is the supplied or internally
          allocated buffer.
        - ``"coo"``: ``(neighbor_list, neighbor_ptr, neighbor_list_shifts)``
          built directly in source-owned CSR rows without a matrix
          intermediate. Requested distances and/or vectors are appended in
          that order, have the exact pair length, and do not alias reusable
          capacity buffers. Pair callbacks keep the topology-only return and
          write their buffers in place. Pair order within a row is unspecified. With
          segmented COO, returns
          ``(neighbor_list, pair_offsets, pair_counts, neighbor_list_shifts)``
          without trimming the caller-owned fixed segments; selective calls
          with ``return_state=True`` append the six state tensors above,
          yielding ten tensors.
        - ``"tile"``: 11-tuple ``(num_tiles, tile_row_group,
          tile_col_group, tile_system, sorted_atom_index, sorted_pos_x,
          sorted_pos_y, sorted_pos_z, batch_idx_sorted, batch_ptr_padded,
          group_ptr)`` --- per-tile and per-system mapping arrays a
          batch tile consumer needs.

    Notes
    -----
    - Cluster-tile is CUDA float32 only; float64 ``positions`` is rejected.
    - Cluster-tile does not support partial neighbor lists (no
      ``target_indices`` kwarg).
    - ``torch.compile(fullgraph=True)`` supports tile and matrix output,
      including dual-cutoff matrices and differentiable geometry. On PyTorch
      2.10 or newer, nonselective exact COO output also supports fullgraph and
      may return a different pair count on each call. Compiled calls require
      the complete scratch tuple returned by
      ``allocate_batch_cluster_tile_list``. Selective matrix calls also
      require fixed tile offsets, counters, and output buffers. Compiled
      capacity failures use asynchronous device assertions. Pair callbacks
      remain eager-only.
    - Build differentiable losses from returned geometry, not from supplied
      output buffers, which are non-differentiable value snapshots when
      reconstruction is required.
    - The unified
      :func:`nvalchemiops.torch.neighbors.neighbor_list` entry point may
      select this binding automatically when the selector guards and cost
      model prefer it; pass ``method="batch_cluster_tile"`` to force it.

    See Also
    --------
    nvalchemiops.torch.neighbors.cluster_tile_neighbor_list :
        Single-system companion entry point.
    nvalchemiops.torch.neighbors.batch_cluster_tile.batch_build_cluster_tile_list :
        Lower-level build step exposed for caching across queries.
    nvalchemiops.torch.neighbors.batch_cluster_tile.batch_query_cluster_tile :
        Lower-level query step.
    """

    if state is not None:
        from nvalchemiops.torch.neighbors.prepared_cluster_tile import (
            ClusterTileState,
            _execute_prepared_cluster_tile,
        )

        if not isinstance(state, ClusterTileState):
            raise TypeError("state must be a ClusterTileState")
        if not state.is_batched:
            raise ValueError(
                "batch_cluster_tile_neighbor_list requires a batched state"
            )
        if cell_batch is None:
            raise ValueError("cell_batch is required when state is provided")
        conflicts = [
            name
            for name, value in {
                "neighbor_matrix": neighbor_matrix,
                "neighbor_matrix_shifts": neighbor_matrix_shifts,
                "num_neighbors": num_neighbors,
                "neighbor_matrix2": neighbor_matrix2,
                "neighbor_matrix_shifts2": neighbor_matrix_shifts2,
                "num_neighbors2": num_neighbors2,
                "neighbor_list": neighbor_list,
                "neighbor_list_shifts": neighbor_list_shifts,
                "pair_counter": pair_counter,
                "pair_offsets": pair_offsets,
                "pair_counts": pair_counts,
                "inv_cell_batch": inv_cell_batch,
                "sorted_atom_index": sorted_atom_index,
                "sort_inv": sort_inv,
                "sorted_pos_x": sorted_pos_x,
                "sorted_pos_y": sorted_pos_y,
                "sorted_pos_z": sorted_pos_z,
                "batch_idx_sorted": batch_idx_sorted,
                "batch_ptr_padded": batch_ptr_padded,
                "group_system": group_system,
                "group_ptr": group_ptr,
                "group_ctr_x": group_ctr_x,
                "group_ctr_y": group_ctr_y,
                "group_ctr_z": group_ctr_z,
                "group_ext_x": group_ext_x,
                "group_ext_y": group_ext_y,
                "group_ext_z": group_ext_z,
                "num_tiles": num_tiles,
                "tile_row_group": tile_row_group,
                "tile_col_group": tile_col_group,
                "tile_system": tile_system,
                "tile_offsets": tile_offsets,
                "tile_counts": tile_counts,
                "neighbor_vectors": neighbor_vectors,
                "neighbor_distances": neighbor_distances,
                "pair_energies": pair_energies,
                "pair_forces": pair_forces,
            }.items()
            if value is not None
        ]
        if return_state:
            conflicts.insert(0, "return_state")
        if conflicts:
            raise ValueError(
                "state controls configuration and storage; conflicting arguments: "
                + ", ".join(conflicts)
            )
        if pair_params is not None:
            raise ValueError(
                "pair_params is not supported with state because prepared pair "
                "callbacks are unsupported"
            )
        return _execute_prepared_cluster_tile(
            positions,
            cell_batch,
            state,
            rebuild_flags=rebuild_flags,
        )
    if cutoff is None or cell_batch is None or batch_ptr is None:
        missing = [
            name
            for name, value in (
                ("cutoff", cutoff),
                ("cell_batch", cell_batch),
                ("batch_ptr", batch_ptr),
            )
            if value is None
        ]
        raise ValueError(
            "missing required arguments without state: " + ", ".join(missing)
        )
    if positions.dtype != torch.float32:
        raise TypeError("positions must be float32")
    if format not in ("matrix", "coo", "tile"):
        raise ValueError(
            f"format must be 'matrix' | 'coo' | 'tile'; got {format!r}",
        )
    if max_tiles_per_group is not None and (
        not isinstance(max_tiles_per_group, int)
        or isinstance(max_tiles_per_group, bool)
        or max_tiles_per_group <= 0
    ):
        raise ValueError("max_tiles_per_group must be a positive integer")
    has_pair_outputs = (
        bool(return_vectors)
        or bool(return_distances)
        or pair_fn is not None
        or pair_params is not None
        or neighbor_vectors is not None
        or neighbor_distances is not None
        or pair_energies is not None
        or pair_forces is not None
    )
    is_compiling = torch.compiler.is_compiling()
    eager_rebuild_values = (
        tuple(bool(value) for value in rebuild_flags.flatten().tolist())
        if rebuild_flags is not None and not is_compiling
        else None
    )
    eager_any_rebuild = (
        any(eager_rebuild_values) if eager_rebuild_values is not None else False
    )
    eager_all_true = (
        all(eager_rebuild_values) if eager_rebuild_values is not None else False
    )
    if return_state and rebuild_flags is None:
        raise ValueError("return_state=True requires rebuild_flags")
    if has_pair_outputs and format == "tile":
        raise NotImplementedError(
            "Pair outputs (return_vectors / return_distances / pair_fn) "
            "are not supported with format='tile'. Use format='matrix' "
            "or format='coo'.",
        )
    if cutoff2 is not None:
        if format != "matrix":
            raise ValueError(
                "cluster_tile cutoff2 is supported only with format='matrix'"
            )
        if has_pair_outputs:
            raise ValueError(
                "cluster_tile cutoff2 cannot be combined with pair outputs"
            )
    if rebuild_flags is not None and format == "tile":
        raise ValueError(
            "cluster_tile selective rebuild is not supported with format='tile'"
        )
    if (
        rebuild_flags is not None
        and format == "coo"
        and (pair_offsets is None or pair_counts is None)
        and (is_compiling or not eager_all_true or not return_state)
    ):
        raise ValueError(
            "cluster_tile selective COO requires pair_offsets and pair_counts"
        )
    if (pair_offsets is None) != (pair_counts is None):
        raise ValueError("Pass both 'pair_offsets' and 'pair_counts', or neither.")
    if (tile_offsets is None) != (tile_counts is None):
        raise ValueError("Pass both 'tile_offsets' and 'tile_counts', or neither.")
    has_cached_partition_metadata = partition_metadata is not None
    if not has_cached_partition_metadata:
        if batch_ptr.dtype != torch.int32 or batch_ptr.ndim != 1:
            raise ValueError("batch_ptr must be 1D int32")
        if batch_ptr.shape[0] < 2:
            raise ValueError("batch_ptr must have length at least 2")
    device = positions.device
    N = positions.shape[0]
    num_systems = int(batch_ptr.shape[0]) - 1
    if (
        is_compiling
        and format == "coo"
        and rebuild_flags is None
        and pair_fn is not None
    ):
        raise NotImplementedError(
            "torch.compile(fullgraph=True) cluster-tile pair_fn is not supported"
        )
    if not has_cached_partition_metadata:
        if is_compiling:
            torch._assert_async(
                batch_ptr[-1] == N,
                "batch_ptr[-1] must equal positions.shape[0]",
            )
        elif int(batch_ptr[-1].item()) != N:
            raise ValueError(f"batch_ptr[-1] ({int(batch_ptr[-1])}) != N ({N})")

    scratch_values = (
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
    has_complete_scratch = all(value is not None for value in scratch_values)
    if is_compiling and not has_complete_scratch:
        raise ValueError(
            "compiled batch_cluster_tile_neighbor_list requires complete scratch "
            "from allocate_batch_cluster_tile_list"
        )
    if not is_compiling and sorted_atom_index is not None and not has_complete_scratch:
        raise ValueError(
            "batch_cluster_tile scratch must include every tensor returned by "
            "allocate_batch_cluster_tile_list"
        )
    if rebuild_flags is not None and format == "coo":
        if neighbor_list is not None and pair_offsets is None and pair_counts is None:
            raise ValueError(
                "selective segmented COO bootstrap requires complete caller-owned "
                "state or no persistent state"
            )
    eager_has_false = (
        rebuild_flags is not None and not is_compiling and not eager_all_true
    )

    if eager_has_false:
        required_state: dict[str, torch.Tensor | None] = {
            "num_tiles": num_tiles,
            "tile_row_group": tile_row_group,
            "tile_col_group": tile_col_group,
            "tile_system": tile_system,
            "tile_offsets": tile_offsets,
            "tile_counts": tile_counts,
        }
        if format == "coo":
            required_state.update(
                {
                    "neighbor_list": neighbor_list,
                    "neighbor_list_shifts": neighbor_list_shifts,
                    "pair_offsets": pair_offsets,
                    "pair_counts": pair_counts,
                }
            )
        else:
            required_state.update(
                {
                    "neighbor_matrix": neighbor_matrix,
                    "num_neighbors": num_neighbors,
                    "neighbor_matrix_shifts": neighbor_matrix_shifts,
                }
            )
            if cutoff2 is not None:
                required_state.update(
                    {
                        "neighbor_matrix2": neighbor_matrix2,
                        "num_neighbors2": num_neighbors2,
                        "neighbor_matrix_shifts2": neighbor_matrix_shifts2,
                    }
                )
        missing_state = [
            name for name, value in required_state.items() if value is None
        ]
        if missing_state:
            raise ValueError(
                "selective segmented COO with rebuild_flags=False requires "
                "caller-owned topology and tile state: " + ", ".join(missing_state)
            )
    if is_compiling and rebuild_flags is not None and format == "matrix":
        compiled_state = (
            neighbor_matrix,
            num_neighbors,
            neighbor_matrix_shifts,
            num_tiles,
            tile_offsets,
            tile_counts,
            tile_row_group,
            tile_col_group,
            tile_system,
        )
        if any(value is None for value in compiled_state):
            raise ValueError(
                "selective compiled matrix requires complete caller-owned state"
            )
        if cutoff2 is not None and any(
            value is None
            for value in (neighbor_matrix2, num_neighbors2, neighbor_matrix_shifts2)
        ):
            raise ValueError(
                "selective compiled dual-cutoff matrix requires both output triples"
            )

    if max_neighbors is None:
        max_neighbors = max(
            estimate_max_neighbors(
                cutoff if cutoff2 is None else max(float(cutoff), float(cutoff2))
            ),
            TILE_GROUP_SIZE,
        )
    if fill_value is None:
        fill_value = N
    if format == "matrix":
        _validate_cluster_tile_matrix_outputs(
            device=device,
            dtype=positions.dtype,
            natom=N,
            max_neighbors=int(max_neighbors),
            cutoff2=cutoff2,
            neighbor_matrix2=neighbor_matrix2,
            num_neighbors2=num_neighbors2,
            neighbor_matrix_shifts2=neighbor_matrix_shifts2,
            return_vectors=return_vectors,
            return_distances=return_distances,
            neighbor_vectors=neighbor_vectors,
            neighbor_distances=neighbor_distances,
            allocate_missing=True,
        )
    build_cutoff = cutoff if cutoff2 is None else max(float(cutoff), float(cutoff2))
    needs_segmented_tiles = rebuild_flags is not None or pair_offsets is not None
    if max_tiles_per_group is None:
        if has_complete_scratch and not (
            needs_segmented_tiles and tile_offsets is None and not is_compiling
        ):
            max_tiles_per_group = 1
        else:
            max_tiles_per_group = estimate_batch_max_tiles_per_group(
                batch_ptr, build_cutoff, cell_batch
            )
    else:
        max_tiles_per_group = int(max_tiles_per_group)
    if (
        rebuild_flags is not None
        and format == "matrix"
        and neighbor_matrix is not None
        and num_neighbors is not None
        and neighbor_matrix_shifts is not None
        and num_tiles is not None
        and tile_offsets is not None
        and tile_counts is not None
        and tile_row_group is not None
        and tile_col_group is not None
        and tile_system is not None
    ):
        _validate_batch_matrix_state(
            device=device,
            natom=N,
            num_systems=num_systems,
            max_neighbors=int(max_neighbors),
            neighbor_matrix=neighbor_matrix,
            num_neighbors=num_neighbors,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            num_tiles=num_tiles,
            tile_offsets=tile_offsets,
            tile_counts=tile_counts,
            tile_row_group=tile_row_group,
            tile_col_group=tile_col_group,
            tile_system=tile_system,
            neighbor_matrix2=neighbor_matrix2,
            num_neighbors2=num_neighbors2,
            neighbor_matrix_shifts2=neighbor_matrix_shifts2,
        )

    if needs_segmented_tiles and tile_offsets is None:
        if is_compiling:
            raise ValueError(
                "compiled segmented cluster-tile calls require tile_offsets and "
                "tile_counts"
            )
        _tile_caps, tile_offsets, _pair_caps, default_pair_offsets = (
            estimate_batch_cluster_tile_segments(
                batch_ptr,
                int(max_neighbors),
                max_tiles_per_group=int(max_tiles_per_group),
            )
        )
        tile_counts = torch.zeros(
            int(tile_offsets.shape[0]) - 1, dtype=torch.int32, device=device
        )
        if format == "coo" and pair_offsets is None:
            pair_offsets = default_pair_offsets
            pair_counts = torch.zeros(
                int(pair_offsets.shape[0]) - 1, dtype=torch.int32, device=device
            )
            if max_pairs is not None and int(max_pairs) != int(pair_offsets[-1].item()):
                raise ValueError(
                    "max_pairs must equal the estimated segmented COO capacity"
                )
    elif needs_segmented_tiles and tile_counts is None:
        raise ValueError("tile_counts is required when tile_offsets is provided")

    geometry_requested = (
        (return_vectors or return_distances)
        and pair_fn is None
        and pair_params is None
        and pair_energies is None
        and pair_forces is None
    )
    requires_reconstruction = (
        geometry_requested
        and torch.is_grad_enabled()
        and (positions.requires_grad or cell_batch.requires_grad)
    )
    snapshot_vectors = neighbor_vectors is not None
    snapshot_distances = neighbor_distances is not None
    if format == "coo" and pair_offsets is None:
        if max_pairs is None:
            max_pairs = N * max_neighbors
        _validate_compact_coo_outputs(
            device=device,
            capacity=int(max_pairs),
            neighbor_list=neighbor_list,
            neighbor_list_shifts=neighbor_list_shifts,
            pair_counter=pair_counter,
        )
        if geometry_requested:
            neighbor_vectors, neighbor_distances = (
                _prepare_compact_coo_geometry_buffers(
                    device=device,
                    dtype=positions.dtype,
                    capacity=int(max_pairs),
                    return_vectors=bool(return_vectors),
                    return_distances=bool(return_distances),
                    neighbor_vectors=neighbor_vectors,
                    neighbor_distances=neighbor_distances,
                    allocate_missing=not requires_reconstruction,
                )
            )

    if sorted_atom_index is None:
        previous_tile_state = (
            num_tiles,
            tile_row_group,
            tile_col_group,
            tile_system,
        )
        (
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
        ) = allocate_batch_cluster_tile_list(
            batch_ptr,
            device,
            dtype=positions.dtype,
            max_tiles_per_group=max_tiles_per_group,
        )
        (
            previous_num_tiles,
            previous_tile_row_group,
            previous_tile_col_group,
            previous_tile_system,
        ) = previous_tile_state
        if previous_num_tiles is not None:
            num_tiles = previous_num_tiles
        if previous_tile_row_group is not None:
            tile_row_group = previous_tile_row_group
        if previous_tile_col_group is not None:
            tile_col_group = previous_tile_col_group
        if previous_tile_system is not None:
            tile_system = previous_tile_system

    segmented_pair_capacity: int | None = None
    if format == "coo" and pair_offsets is not None:
        if neighbor_list is None:
            capacity = int(pair_offsets[-1].item())
            neighbor_list = torch.empty(
                (2, capacity),
                dtype=torch.int32,
                device=device,
            )
        if neighbor_list_shifts is None:
            neighbor_list_shifts = torch.empty(
                (neighbor_list.shape[1], 3),
                dtype=torch.int32,
                device=device,
            )
        segmented_pair_capacity = _validate_segmented_coo_state(
            device=device,
            num_systems=num_systems,
            neighbor_list=neighbor_list,
            neighbor_list_shifts=neighbor_list_shifts,
            pair_offsets=pair_offsets,
            pair_counts=pair_counts,
            rebuild_flags=rebuild_flags,
            tile_offsets=tile_offsets,
            tile_counts=tile_counts,
            num_tiles=num_tiles,
            tile_row_group=tile_row_group,
            tile_col_group=tile_col_group,
            tile_system=tile_system,
        )

    if partition_metadata is None:
        _validate_batch_cluster_tile_scratch(
            positions=positions,
            batch_ptr=batch_ptr,
            sorted_atom_index=sorted_atom_index,
            sort_inv=sort_inv,
            sorted_pos_x=sorted_pos_x,
            sorted_pos_y=sorted_pos_y,
            sorted_pos_z=sorted_pos_z,
            batch_idx_sorted=batch_idx_sorted,
            batch_ptr_padded=batch_ptr_padded,
            group_system=group_system,
            group_ptr=group_ptr,
            group_ctr_x=group_ctr_x,
            group_ctr_y=group_ctr_y,
            group_ctr_z=group_ctr_z,
            group_ext_x=group_ext_x,
            group_ext_y=group_ext_y,
            group_ext_z=group_ext_z,
            num_tiles=num_tiles,
            tile_row_group=tile_row_group,
            tile_col_group=tile_col_group,
            tile_system=tile_system,
        )

    if rebuild_flags is not None and not is_compiling and not eager_any_rebuild:
        if format == "coo":
            outputs = (
                neighbor_list,
                pair_offsets,
                pair_counts,
                neighbor_list_shifts,
            )
        elif cutoff2 is not None:
            outputs = (
                neighbor_matrix,
                num_neighbors,
                neighbor_matrix_shifts,
                neighbor_matrix2,
                num_neighbors2,
                neighbor_matrix_shifts2,
            )
        else:
            outputs = (neighbor_matrix, num_neighbors, neighbor_matrix_shifts)
        if return_state:
            return (
                *outputs,
                tile_offsets,
                tile_counts,
                num_tiles,
                tile_row_group,
                tile_col_group,
                tile_system,
            )
        return outputs

    if partition_metadata is None:
        partition_metadata = _prepare_batch_partition_metadata(
            batch_ptr,
            num_atoms=N,
            padded_slot_system=batch_idx_sorted,
            batch_ptr_padded=batch_ptr_padded,
            group_system=group_system,
            group_ptr=group_ptr,
        )

    inv_cell_batch = _batch_build_cluster_tile_list_normalized(
        partition_metadata,
        positions.detach() if requires_reconstruction else positions,
        build_cutoff,
        cell_batch.detach() if requires_reconstruction else cell_batch,
        batch_ptr,
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
        inv_cell_batch=(
            inv_cell_batch.detach()
            if requires_reconstruction and inv_cell_batch is not None
            else inv_cell_batch
        ),
        rebuild_flags=rebuild_flags,
        tile_offsets=tile_offsets,
        tile_counts=tile_counts,
        rebuild_observed=eager_any_rebuild,
    )

    if format == "tile":
        # Raw-tile callers must not receive a silently truncated tile list
        # (format='tile' is always the compact, non-selective path).
        tile_capacity = int(tile_row_group.shape[0])
        _check_tile_buffer_capacity(num_tiles, tile_capacity)
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
        segmented_coo = pair_offsets is not None
        if segmented_coo:
            if segmented_pair_capacity is None:
                raise RuntimeError("segmented COO state was not validated")
            max_pairs = segmented_pair_capacity
        elif max_pairs is None:
            max_pairs = N * max_neighbors
        if neighbor_list is None:
            coo_buf = torch.empty(
                (max_pairs, 2),
                dtype=torch.int32,
                device=device,
            )
        else:
            coo_buf = neighbor_list.transpose(0, 1)
        if neighbor_list_shifts is None:
            neighbor_list_shifts = torch.empty(
                (max_pairs, 3),
                dtype=torch.int32,
                device=device,
            )
        if pair_counter is None:
            pair_counter = torch.zeros(1, dtype=torch.int32, device=device)
        if segmented_coo and pair_counts is None:
            pair_counts = torch.empty(
                int(pair_offsets.shape[0]) - 1, dtype=torch.int32, device=device
            )
        if not segmented_coo:
            if is_compiling:
                version = torch.__version__.split("+", maxsplit=1)[0]
                major, minor = (int(part) for part in version.split(".")[:2])
                if (major, minor) < (2, 10):
                    raise RuntimeError(
                        "torch.compile(fullgraph=True) exact cluster-tile COO requires PyTorch 2.10 or newer"
                    )
            copy_vectors = bool(
                geometry_requested and return_vectors and not requires_reconstruction
            )
            copy_distances = bool(
                geometry_requested and return_distances and not requires_reconstruction
            )
            if not geometry_requested:
                prefix_vectors = torch.empty(
                    (1, 3), dtype=positions.dtype, device=device
                )
                prefix_distances = torch.empty(1, dtype=positions.dtype, device=device)
            if geometry_requested:
                prefix_vectors = neighbor_vectors
                prefix_distances = neighbor_distances
            tile_capacity = int(tile_row_group.shape[0])
            n_tiles = _check_tile_buffer_capacity(num_tiles, tile_capacity)
            launch_tiles = tile_capacity if is_compiling else n_tiles
            row_counts = torch.zeros(N, dtype=torch.int32, device=device)
            direct_cell = cell_batch.detach() if requires_reconstruction else cell_batch
            direct_inv_cell = (
                torch.linalg.inv(direct_cell).contiguous()
                if inv_cell_batch is None
                else (
                    inv_cell_batch.detach()
                    if requires_reconstruction
                    else inv_cell_batch
                )
            )
            _batch_cluster_tile_direct_csr_count_op(
                cutoff,
                N,
                direct_cell,
                direct_inv_cell,
                sorted_atom_index,
                sorted_pos_x,
                sorted_pos_y,
                sorted_pos_z,
                num_tiles,
                tile_row_group,
                tile_col_group,
                tile_system,
                row_counts,
                launch_tiles,
            )
            neighbor_ptr = torch.cat(
                (
                    torch.zeros(1, dtype=torch.int32, device=device),
                    torch.cumsum(row_counts, dim=0, dtype=torch.int32),
                )
            )
            total = neighbor_ptr[-1:]
            resolved_count = _check_neighbor_capacity(
                total,
                int(max_pairs),
                kind="coo",
            )
            pair_counter.copy_(total)
            cursors = neighbor_ptr[:-1].clone()
            if pair_fn is None:
                _batch_cluster_tile_direct_csr_fill_op(
                    cutoff,
                    N,
                    direct_cell,
                    direct_inv_cell,
                    sorted_atom_index,
                    sorted_pos_x,
                    sorted_pos_y,
                    sorted_pos_z,
                    num_tiles,
                    tile_row_group,
                    tile_col_group,
                    tile_system,
                    cursors,
                    coo_buf,
                    neighbor_list_shifts,
                    prefix_vectors,
                    prefix_distances,
                    copy_vectors,
                    copy_distances,
                    int(max_pairs),
                    launch_tiles,
                )
            else:
                if pair_params is None or pair_energies is None or pair_forces is None:
                    raise ValueError(
                        "pair_fn requires pair_params, pair_energies, and pair_forces"
                    )
                _batch_cluster_tile_direct_csr_fill_pair_outputs(
                    cutoff,
                    N,
                    direct_cell,
                    direct_inv_cell,
                    sorted_atom_index,
                    sorted_pos_x,
                    sorted_pos_y,
                    sorted_pos_z,
                    num_tiles,
                    tile_row_group,
                    tile_col_group,
                    tile_system,
                    cursors,
                    coo_buf,
                    neighbor_list_shifts,
                    pair_fn,
                    pair_params,
                    pair_energies,
                    pair_forces,
                    neighbor_vectors,
                    neighbor_distances,
                    return_vectors,
                    return_distances,
                    int(max_pairs),
                    launch_tiles,
                )
            nl, nls, exact_vectors, exact_distances = _compact_coo_prefix(
                total,
                coo_buf,
                neighbor_list_shifts,
                prefix_vectors,
                prefix_distances,
                int(max_pairs),
                -1 if is_compiling else resolved_count,
                copy_vectors,
                copy_distances,
            )
            if geometry_requested:
                if requires_reconstruction:
                    batch_idx_atom = (
                        partition_metadata.atom_system
                        if partition_metadata is not None
                        else torch.repeat_interleave(
                            torch.arange(num_systems, dtype=torch.int32, device=device),
                            (batch_ptr[1:] - batch_ptr[:-1]).to(torch.int64),
                            output_size=N,
                        )
                    )
                    exact_distances, exact_vectors = _reconstruct_coo_geometry(
                        positions, cell_batch, nl, nls, batch_idx_atom
                    )
                    if return_vectors and snapshot_vectors:
                        neighbor_vectors[: exact_vectors.shape[0]].copy_(
                            exact_vectors.detach()
                        )
                    if return_distances and snapshot_distances:
                        neighbor_distances[: exact_distances.shape[0]].copy_(
                            exact_distances.detach()
                        )
                outputs = (nl, neighbor_ptr, nls)
                if return_distances:
                    outputs = (*outputs, exact_distances)
                if return_vectors:
                    outputs = (*outputs, exact_vectors)
                return outputs
            return nl, neighbor_ptr, nls
        batch_query_cluster_tile_coo(
            sorted_atom_index,
            sorted_pos_x,
            sorted_pos_y,
            sorted_pos_z,
            cell_batch,
            num_tiles,
            tile_row_group,
            tile_col_group,
            tile_system,
            cutoff,
            N,
            int(max_pairs),
            pair_counter,
            coo_buf,
            neighbor_list_shifts,
            inv_cell_batch=inv_cell_batch,
            rebuild_flags=rebuild_flags,
            tile_offsets=tile_offsets,
            tile_counts=tile_counts,
            pair_offsets=pair_offsets,
            pair_counts=pair_counts,
            return_vectors=return_vectors,
            return_distances=return_distances,
            pair_fn=pair_fn,
            pair_params=pair_params,
            neighbor_vectors=neighbor_vectors,
            neighbor_distances=neighbor_distances,
            pair_energies=pair_energies,
            pair_forces=pair_forces,
        )
        if segmented_coo:
            segment_caps = pair_offsets[1:] - pair_offsets[:-1]
            _check_neighbor_capacity(
                pair_counts,
                segment_caps,
                segmented=True,
                kind="coo",
            )
            outputs = (
                neighbor_list,
                pair_offsets,
                pair_counts,
                neighbor_list_shifts,
            )
            if return_state:
                return (
                    *outputs,
                    tile_offsets,
                    tile_counts,
                    num_tiles,
                    tile_row_group,
                    tile_col_group,
                    tile_system,
                )
            return outputs

    # ``format == "matrix"``: skip-prefill matrix path with tail fill.
    if neighbor_matrix is None:
        neighbor_matrix = torch.empty(
            (N, max_neighbors),
            dtype=torch.int32,
            device=device,
        )
    if num_neighbors is None:
        num_neighbors = torch.zeros(N, dtype=torch.int32, device=device)
    elif rebuild_flags is None:
        num_neighbors.zero_()
    if neighbor_matrix_shifts is None:
        neighbor_matrix_shifts = torch.empty(
            (N, max_neighbors, 3),
            dtype=torch.int32,
            device=device,
        )
    if cutoff2 is not None:
        if neighbor_matrix2 is None:
            neighbor_matrix2 = torch.empty(
                (N, max_neighbors), dtype=torch.int32, device=device
            )
        if num_neighbors2 is None:
            num_neighbors2 = torch.zeros(N, dtype=torch.int32, device=device)
        elif rebuild_flags is None:
            num_neighbors2.zero_()
        if neighbor_matrix_shifts2 is None:
            neighbor_matrix_shifts2 = torch.empty(
                (N, max_neighbors, 3), dtype=torch.int32, device=device
            )

    # Reconstructed geometry is returned directly and needs an output buffer
    # only when the caller requested a detached snapshot.
    if return_vectors and neighbor_vectors is None and not requires_reconstruction:
        neighbor_vectors = torch.empty(
            (N, max_neighbors, 3),
            dtype=positions.dtype,
            device=device,
        )
    if return_distances and neighbor_distances is None and not requires_reconstruction:
        neighbor_distances = torch.empty(
            (N, max_neighbors),
            dtype=positions.dtype,
            device=device,
        )
    if pair_fn is not None:
        if pair_energies is None:
            pair_energies = torch.empty(
                (N, max_neighbors),
                dtype=positions.dtype,
                device=device,
            )
        if pair_forces is None:
            pair_forces = torch.empty(
                (N, max_neighbors, 3),
                dtype=positions.dtype,
                device=device,
            )

    batch_idx_atom = None
    if rebuild_flags is not None or requires_reconstruction:
        batch_idx_atom = (
            partition_metadata.atom_system
            if partition_metadata is not None
            else torch.repeat_interleave(
                torch.arange(num_systems, dtype=torch.int32, device=device),
                (batch_ptr[1:] - batch_ptr[:-1]).to(torch.int64),
                output_size=N,
            )
        )
    batch_query_cluster_tile(
        sorted_atom_index,
        sorted_pos_x,
        sorted_pos_y,
        sorted_pos_z,
        cell_batch.detach() if requires_reconstruction else cell_batch,
        num_tiles,
        tile_row_group,
        tile_col_group,
        tile_system,
        cutoff,
        N,
        neighbor_matrix,
        num_neighbors,
        neighbor_matrix_shifts,
        inv_cell_batch=(
            inv_cell_batch.detach()
            if requires_reconstruction and inv_cell_batch is not None
            else inv_cell_batch
        ),
        cutoff2=cutoff2,
        neighbor_matrix2=neighbor_matrix2,
        num_neighbors2=num_neighbors2,
        neighbor_matrix_shifts2=neighbor_matrix_shifts2,
        rebuild_flags=rebuild_flags,
        tile_offsets=tile_offsets,
        tile_counts=tile_counts,
        batch_idx=batch_idx_atom,
        return_vectors=return_vectors and not requires_reconstruction,
        return_distances=return_distances and not requires_reconstruction,
        pair_fn=pair_fn,
        pair_params=pair_params,
        neighbor_vectors=None if requires_reconstruction else neighbor_vectors,
        neighbor_distances=None if requires_reconstruction else neighbor_distances,
        pair_energies=pair_energies,
        pair_forces=pair_forces,
    )

    if max_neighbors > 0:
        if rebuild_flags is not None:
            secondary_matrix = (
                neighbor_matrix2
                if neighbor_matrix2 is not None
                else torch.empty((1, 1), dtype=torch.int32, device=device)
            )
            secondary_counts = (
                num_neighbors2
                if num_neighbors2 is not None
                else torch.empty(1, dtype=torch.int32, device=device)
            )
            _batch_cluster_tile_selective_fill_neighbor_matrix_tail_op(
                num_neighbors,
                secondary_counts,
                batch_idx_atom,
                rebuild_flags,
                neighbor_matrix,
                secondary_matrix,
                int(max_neighbors),
                int(fill_value),
                bool(cutoff2 is not None),
            )
        else:
            _batch_cluster_tile_fill_neighbor_matrix_tail_op(
                num_neighbors,
                neighbor_matrix,
                int(N),
                int(max_neighbors),
                int(fill_value),
            )
            if cutoff2 is not None:
                _batch_cluster_tile_fill_neighbor_matrix_tail_op(
                    num_neighbors2,
                    neighbor_matrix2,
                    int(N),
                    int(max_neighbors),
                    int(fill_value),
                )

    _check_neighbor_capacity(num_neighbors, int(max_neighbors))
    if cutoff2 is not None and num_neighbors2 is not None:
        _check_neighbor_capacity(num_neighbors2, int(max_neighbors))

    if requires_reconstruction:
        distances, vectors = _reconstruct_matrix_geometry(
            positions,
            cell_batch,
            neighbor_matrix,
            num_neighbors,
            neighbor_matrix_shifts,
            batch_idx_atom,
        )
        row_update = (
            rebuild_flags[batch_idx_atom.to(torch.long)]
            if rebuild_flags is not None
            else torch.ones(N, dtype=torch.bool, device=device)
        )
        if return_vectors and snapshot_vectors:
            neighbor_vectors.copy_(
                torch.where(
                    row_update[:, None, None],
                    vectors.detach(),
                    neighbor_vectors.detach(),
                )
            )
        if return_distances and snapshot_distances:
            neighbor_distances.copy_(
                torch.where(
                    row_update[:, None],
                    distances.detach(),
                    neighbor_distances.detach(),
                )
            )

    if cutoff2 is not None:
        outputs = (
            neighbor_matrix,
            num_neighbors,
            neighbor_matrix_shifts,
            neighbor_matrix2,
            num_neighbors2,
            neighbor_matrix_shifts2,
        )
    else:
        outputs = (neighbor_matrix, num_neighbors, neighbor_matrix_shifts)
    if geometry_requested and rebuild_flags is None:
        if return_distances:
            outputs = (
                *outputs,
                distances if requires_reconstruction else neighbor_distances,
            )
        if return_vectors:
            outputs = (
                *outputs,
                vectors if requires_reconstruction else neighbor_vectors,
            )
    if return_state:
        return (
            *outputs,
            tile_offsets,
            tile_counts,
            num_tiles,
            tile_row_group,
            tile_col_group,
            tile_system,
        )
    return outputs


def batch_cluster_tile_neighbor_list(
    positions: torch.Tensor,
    cutoff: float | None = None,
    cell_batch: torch.Tensor | None = None,
    batch_ptr: torch.Tensor | None = None,
    max_neighbors: int | None = None,
    fill_value: int | None = None,
    format: str = "matrix",
    max_pairs: int | None = None,
    cutoff2: float | None = None,
    rebuild_flags: torch.Tensor | None = None,
    return_state: bool = False,
    neighbor_matrix: torch.Tensor | None = None,
    neighbor_matrix_shifts: torch.Tensor | None = None,
    num_neighbors: torch.Tensor | None = None,
    neighbor_matrix2: torch.Tensor | None = None,
    neighbor_matrix_shifts2: torch.Tensor | None = None,
    num_neighbors2: torch.Tensor | None = None,
    neighbor_list: torch.Tensor | None = None,
    neighbor_list_shifts: torch.Tensor | None = None,
    pair_counter: torch.Tensor | None = None,
    pair_offsets: torch.Tensor | None = None,
    pair_counts: torch.Tensor | None = None,
    inv_cell_batch: torch.Tensor | None = None,
    sorted_atom_index: torch.Tensor | None = None,
    sort_inv: torch.Tensor | None = None,
    sorted_pos_x: torch.Tensor | None = None,
    sorted_pos_y: torch.Tensor | None = None,
    sorted_pos_z: torch.Tensor | None = None,
    batch_idx_sorted: torch.Tensor | None = None,
    batch_ptr_padded: torch.Tensor | None = None,
    group_system: torch.Tensor | None = None,
    group_ptr: torch.Tensor | None = None,
    group_ctr_x: torch.Tensor | None = None,
    group_ctr_y: torch.Tensor | None = None,
    group_ctr_z: torch.Tensor | None = None,
    group_ext_x: torch.Tensor | None = None,
    group_ext_y: torch.Tensor | None = None,
    group_ext_z: torch.Tensor | None = None,
    num_tiles: torch.Tensor | None = None,
    tile_row_group: torch.Tensor | None = None,
    tile_col_group: torch.Tensor | None = None,
    tile_system: torch.Tensor | None = None,
    tile_offsets: torch.Tensor | None = None,
    tile_counts: torch.Tensor | None = None,
    return_vectors: bool = False,
    return_distances: bool = False,
    pair_fn: wp.Function | None = None,
    pair_params: torch.Tensor | None = None,
    neighbor_vectors: torch.Tensor | None = None,
    neighbor_distances: torch.Tensor | None = None,
    pair_energies: torch.Tensor | None = None,
    pair_forces: torch.Tensor | None = None,
    max_tiles_per_group: int | None = None,
    *,
    state: "ClusterTileState | None" = None,
) -> tuple[torch.Tensor, ...]:
    """Build and query a batched cluster-pair tile neighbor list in one call."""
    return _batch_cluster_tile_neighbor_list_impl(
        None,
        positions,
        cutoff,
        cell_batch,
        batch_ptr,
        max_neighbors=max_neighbors,
        fill_value=fill_value,
        format=format,
        max_pairs=max_pairs,
        cutoff2=cutoff2,
        rebuild_flags=rebuild_flags,
        return_state=return_state,
        neighbor_matrix=neighbor_matrix,
        neighbor_matrix_shifts=neighbor_matrix_shifts,
        num_neighbors=num_neighbors,
        neighbor_matrix2=neighbor_matrix2,
        neighbor_matrix_shifts2=neighbor_matrix_shifts2,
        num_neighbors2=num_neighbors2,
        neighbor_list=neighbor_list,
        neighbor_list_shifts=neighbor_list_shifts,
        pair_counter=pair_counter,
        pair_offsets=pair_offsets,
        pair_counts=pair_counts,
        inv_cell_batch=inv_cell_batch,
        sorted_atom_index=sorted_atom_index,
        sort_inv=sort_inv,
        sorted_pos_x=sorted_pos_x,
        sorted_pos_y=sorted_pos_y,
        sorted_pos_z=sorted_pos_z,
        batch_idx_sorted=batch_idx_sorted,
        batch_ptr_padded=batch_ptr_padded,
        group_system=group_system,
        group_ptr=group_ptr,
        group_ctr_x=group_ctr_x,
        group_ctr_y=group_ctr_y,
        group_ctr_z=group_ctr_z,
        group_ext_x=group_ext_x,
        group_ext_y=group_ext_y,
        group_ext_z=group_ext_z,
        num_tiles=num_tiles,
        tile_row_group=tile_row_group,
        tile_col_group=tile_col_group,
        tile_system=tile_system,
        tile_offsets=tile_offsets,
        tile_counts=tile_counts,
        return_vectors=return_vectors,
        return_distances=return_distances,
        pair_fn=pair_fn,
        pair_params=pair_params,
        neighbor_vectors=neighbor_vectors,
        neighbor_distances=neighbor_distances,
        pair_energies=pair_energies,
        pair_forces=pair_forces,
        max_tiles_per_group=max_tiles_per_group,
        state=state,
    )


batch_cluster_tile_neighbor_list.__doc__ = (
    _batch_cluster_tile_neighbor_list_impl.__doc__
)
