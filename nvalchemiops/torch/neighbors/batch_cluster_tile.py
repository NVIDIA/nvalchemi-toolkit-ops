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

from typing import TYPE_CHECKING

import torch
import warp as wp

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
from nvalchemiops.neighbors.neighbor_utils import (
    NeighborOverflowError,
    estimate_max_neighbors,
)
from nvalchemiops.neighbors.neighbor_utils import (
    fill_neighbor_matrix_tail as wp_fill_neighbor_matrix_tail,
)
from nvalchemiops.neighbors.neighbor_utils import (
    selective_zero_num_neighbors as wp_selective_zero_num_neighbors,
)
from nvalchemiops.neighbors.output_args import _has_partial_or_pair_outputs
from nvalchemiops.torch.types import get_wp_dtype

if TYPE_CHECKING:
    from nvalchemiops.torch.neighbors._autograd import _NeighborForwardOutput

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
    "nvalchemiops::_batch_cluster_tile_fill_neighbor_matrix_tail",
    mutates_args=("neighbor_matrix",),
)
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
    max_tiles_per_group : int, default 512
        Upper bound on neighbor groups per row_group (dense-cutoff cap).

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
        Upper bound on the tile pair list size.
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
    ngroup = n_padded // TILE_GROUP_SIZE
    ngroup_padded = (
        (ngroup + TILE_GROUP_SIZE - 1) // TILE_GROUP_SIZE
    ) * TILE_GROUP_SIZE + TILE_GROUP_SIZE
    max_tiles = ngroup * min(ngroup, max_tiles_per_group)
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

    Returns ``(sorted_atom_index, sort_inv_real, sorted_pos_x, sorted_pos_y,
    sorted_pos_z, batch_idx_sorted, batch_ptr_padded, group_system,
    group_ptr, group_ctr_x, group_ctr_y, group_ctr_z, group_ext_x,
    group_ext_y, group_ext_z, num_tiles, tile_row_group, tile_col_group,
    tile_system)``.
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
    batch_idx: torch.Tensor,
    batch_ptr: torch.Tensor,
    inv_cell_batch: torch.Tensor,
    sorted_atom_index: torch.Tensor,
    sort_inv: torch.Tensor,
    sorted_pos_x: torch.Tensor,
    sorted_pos_y: torch.Tensor,
    sorted_pos_z: torch.Tensor,
    batch_idx_sorted: torch.Tensor,
    batch_ptr_padded: torch.Tensor,
) -> None:
    """Per-system Morton sort into the padded layout (torch ops only)."""
    N = positions.shape[0]
    num_systems = inv_cell_batch.shape[0]
    device = positions.device

    natom_per_system = batch_ptr[1:] - batch_ptr[:-1]
    natom_padded_per_system = (
        (natom_per_system + TILE_GROUP_SIZE - 1) // TILE_GROUP_SIZE
    ) * TILE_GROUP_SIZE

    bpp = torch.zeros(num_systems + 1, dtype=torch.int32, device=device)
    torch.cumsum(natom_padded_per_system, dim=0, out=bpp[1:])
    batch_ptr_padded.copy_(bpp)

    codes = _per_atom_morton_codes(positions, batch_idx, inv_cell_batch)
    # System-major key: system index in high 32 bits, Morton code in low.
    system_major = codes.to(torch.int64) | (batch_idx.to(torch.int64) << 32)
    sorted_atom_index_real = torch.argsort(system_major).to(torch.int32)

    # Inverse permutation over real atoms.
    inv = torch.empty(N, dtype=torch.int32, device=device)
    inv[sorted_atom_index_real] = torch.arange(
        N,
        dtype=torch.int32,
        device=device,
    )
    sort_inv.copy_(inv)

    # Placement indices for each real atom in the padded layout.
    batch_idx_sorted_real = batch_idx[sorted_atom_index_real]
    within_system = (
        torch.arange(
            N,
            dtype=torch.int32,
            device=device,
        )
        - batch_ptr[batch_idx_sorted_real]
    )
    padded_slot = bpp[batch_idx_sorted_real] + within_system

    # sorted_atom_index[k] = N is the padding sentinel.
    sp = torch.full(
        (sorted_atom_index.shape[0],),
        N,
        dtype=torch.int32,
        device=device,
    )
    sp[padded_slot] = sorted_atom_index_real
    sorted_atom_index.copy_(sp)

    # System index for every padded slot.
    bis = (
        torch.repeat_interleave(
            torch.arange(num_systems, dtype=torch.int32, device=device),
            natom_padded_per_system,
        )
        .to(torch.int32)
        .contiguous()
    )
    batch_idx_sorted.copy_(bis)

    # Padding slots duplicate each system's first real atom, so
    # tile_min / tile_max over the group remain well-formed.  Duplicates
    # cannot cause a miss because they add a point already inside the
    # system's spatial region.
    first_atom_per_system = batch_ptr[:-1]
    position_index = torch.where(
        sp == N,
        first_atom_per_system[bis],
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
        "batch_idx_sorted",
        "batch_ptr_padded",
        "group_system",
        "group_ptr",
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
        "tile_counts",
    ),
)
def _batch_build_cluster_tile_list_op(
    positions: torch.Tensor,
    cutoff: float,
    cell_batch: torch.Tensor,
    inv_cell_batch: torch.Tensor,
    batch_ptr: torch.Tensor,
    batch_idx: torch.Tensor,
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
) -> None:
    wp_device = str(positions.device)
    wp_dtype = get_wp_dtype(positions.dtype)

    _batched_morton_sort_padded(
        positions,
        batch_idx,
        batch_ptr,
        inv_cell_batch,
        sorted_atom_index,
        sort_inv,
        sorted_pos_x,
        sorted_pos_y,
        sorted_pos_z,
        batch_idx_sorted,
        batch_ptr_padded,
    )

    group_ptr.copy_((batch_ptr_padded // TILE_GROUP_SIZE).to(torch.int32).contiguous())
    group_system.copy_(
        batch_idx_sorted[::TILE_GROUP_SIZE].to(torch.int32).contiguous(),
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
    batch_idx: torch.Tensor,
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
) -> None:
    return None


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
    """Build batched tile neighbor list state into pre-allocated outputs.

    Runs the per-system Morton sort + padded SoA gather in torch, then
    the bbox reduction + tile-pair enumeration in warp.
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
    if batch_ptr.dtype != torch.int32 or batch_ptr.ndim != 1:
        raise ValueError("batch_ptr must be 1D int32")
    if batch_ptr.shape[0] < 2:
        raise ValueError("batch_ptr must have length at least 2")

    num_systems = cell_batch.shape[0]
    if batch_ptr.shape[0] != num_systems + 1:
        raise ValueError(
            f"batch_ptr length {batch_ptr.shape[0]} != num_systems+1 = {num_systems + 1}",
        )
    N = positions.shape[0]
    if int(batch_ptr[-1].item()) != N:
        raise ValueError(f"batch_ptr[-1] ({int(batch_ptr[-1])}) != N ({N})")

    if inv_cell_batch is None:
        inv_cell_batch = torch.linalg.inv(cell_batch).contiguous()

    batch_idx = torch.repeat_interleave(
        torch.arange(num_systems, dtype=torch.int32, device=positions.device),
        (batch_ptr[1:] - batch_ptr[:-1]).to(torch.int64),
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
        batch_idx,
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
    )


@torch.library.custom_op(
    "nvalchemiops::_batch_query_cluster_tile",
    mutates_args=("neighbor_matrix", "num_neighbors", "neighbor_matrix_shifts"),
)
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
    n_tiles: int,
) -> None:
    # ``n_tiles`` (host-synced compact tile count) tightens the launch; the
    # kernel still guards per-tile defensively.  ``n_tiles <= 0`` (segmented
    # path, count unknown here) falls back to the full-buffer launch.
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
    n_tiles: int,
) -> None:
    return None


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
    """

    if inv_cell_batch is None:
        inv_cell_batch = torch.linalg.inv(cell_batch).contiguous()

    # Host-sync the tile count to tighten the launch and raise on tile-buffer
    # overflow.  The default (non-selective, non-COO-segmented) batch path uses
    # a single compact ``num_tiles`` counter with contiguous tile writes, so the
    # tight launch and a single global overflow check apply.  The segmented path
    # (selective rebuild / segmented COO) keeps the full-buffer launch and is
    # overflow-checked per system below.
    tile_capacity = int(tile_row_group.shape[0])
    if tile_offsets is None:
        if torch.compiler.is_compiling():
            n_tiles = tile_capacity
        else:
            n_tiles = int(num_tiles.item())
            if n_tiles > tile_capacity:
                raise NeighborOverflowError(tile_capacity, n_tiles)
    else:
        n_tiles = 0  # segmented: count is per-system; full-buffer launch
        if tile_counts is not None:
            seg_caps = tile_offsets[1:] - tile_offsets[:-1]
            overflow = tile_counts > seg_caps
            if bool(overflow.any().item()):
                isys = int(torch.nonzero(overflow, as_tuple=False)[0, 0].item())
                raise NeighborOverflowError(
                    int(seg_caps[isys].item()),
                    int(tile_counts[isys].item()),
                    system_index=isys,
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
        n_tiles,
    )


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
    """

    if inv_cell_batch is None:
        inv_cell_batch = torch.linalg.inv(cell_batch).contiguous()
    pair_counter.zero_()

    # Host-sync the tile count: tighten the launch on the compact path and
    # raise on tile-buffer overflow (missing tiles -> missing pairs).  The
    # segmented path is overflow-checked per system.
    tile_capacity = int(tile_row_group.shape[0])
    if tile_offsets is None:
        if torch.compiler.is_compiling():
            n_tiles = tile_capacity
        else:
            n_tiles = int(num_tiles.item())
            if n_tiles > tile_capacity:
                raise NeighborOverflowError(tile_capacity, n_tiles)
    else:
        n_tiles = 0  # segmented: full-buffer launch
        if tile_counts is not None:
            seg_caps = tile_offsets[1:] - tile_offsets[:-1]
            overflow = tile_counts > seg_caps
            if bool(overflow.any().item()):
                isys = int(torch.nonzero(overflow, as_tuple=False)[0, 0].item())
                raise NeighborOverflowError(
                    int(seg_caps[isys].item()),
                    int(tile_counts[isys].item()),
                    system_index=isys,
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


# =============================================================================
# High-level convenience
# =============================================================================
def _batch_cluster_tile_pair_outputs_forward(
    positions: torch.Tensor,
    cell_batch: torch.Tensor,
    *,
    batch_ptr: torch.Tensor,
    cutoff: float,
    max_neighbors: int,
    fill_value: int,
    batch_idx_atom: torch.Tensor,
) -> "_NeighborForwardOutput":
    """Forward closure for the torch batch_cluster_tile autograd path."""
    from nvalchemiops.torch.neighbors._autograd import (
        _flatten_active_pairs,
        _NeighborForwardOutput,
    )

    positions_det = positions.detach()
    cell_det = cell_batch.detach()
    device = positions_det.device
    N = int(batch_ptr[-1].item())
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
        dtype=positions_det.dtype,
    )
    batch_build_cluster_tile_list(
        positions_det,
        cutoff,
        cell_det,
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
    )
    nm = torch.empty((N, max_neighbors), dtype=torch.int32, device=device)
    nn = torch.zeros(N, dtype=torch.int32, device=device)
    nms = torch.empty((N, max_neighbors, 3), dtype=torch.int32, device=device)
    nv = torch.zeros((N, max_neighbors, 3), dtype=positions_det.dtype, device=device)
    nd = torch.zeros((N, max_neighbors), dtype=positions_det.dtype, device=device)
    batch_query_cluster_tile(
        sorted_atom_index,
        sorted_pos_x,
        sorted_pos_y,
        sorted_pos_z,
        cell_det,
        num_tiles,
        tile_row_group,
        tile_col_group,
        tile_system,
        cutoff,
        N,
        nm,
        nn,
        nms,
        return_vectors=True,
        return_distances=True,
        neighbor_vectors=nv,
        neighbor_distances=nd,
    )
    if max_neighbors > 0:
        _batch_cluster_tile_fill_neighbor_matrix_tail_op(
            nn,
            nm,
            int(N),
            int(max_neighbors),
            int(fill_value),
        )
    i_idx, j_idx, shifts_flat, batch_idx_flat, mask = _flatten_active_pairs(
        nm,
        nn,
        nms,
        batch_idx=batch_idx_atom,
    )
    K, M = nm.shape
    return _NeighborForwardOutput(
        distances=nd,
        vectors=nv,
        extra_outputs=(nm, nn, nms),
        i_idx_flat=i_idx,
        j_idx_flat=j_idx,
        shifts_flat=shifts_flat,
        batch_idx_flat=batch_idx_flat,
        active_mask=mask,
        matrix_shape=(K, M),
    )


def batch_cluster_tile_neighbor_list(
    positions: torch.Tensor,
    cutoff: float,
    cell_batch: torch.Tensor,
    batch_ptr: torch.Tensor,
    max_neighbors: int | None = None,
    fill_value: int | None = None,
    format: str = "matrix",
    max_pairs: int | None = None,
    cutoff2: float | None = None,
    rebuild_flags: torch.Tensor | None = None,
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
) -> tuple[torch.Tensor, ...]:
    """Build a batched cluster-pair tile neighbor list (one-shot convenience).

    Batched PyTorch binding for the cluster-pair tile algorithm.  Supports
    triclinic ``cell_batch`` of shape ``(num_systems, 3, 3)`` and arbitrary
    per-system atom counts (padded internally to a multiple of
    ``TILE_GROUP_SIZE``). Cluster-tile is CUDA float32 only.

    Parameters
    ----------
    positions : torch.Tensor, shape (total_atoms, 3), dtype=float32
        Concatenated atomic coordinates across systems.
    cutoff : float
        Cutoff distance in Cartesian units. Must be positive.
    cell_batch : torch.Tensor, shape (num_systems, 3, 3), dtype=float32
        Per-system unit cell matrices. Cluster-tile assumes fully
        periodic boundaries.
    batch_ptr : torch.Tensor, shape (num_systems + 1,), dtype=int32 or int64
        CSR pointer separating systems.  Assumes positions are laid out
        in system-contiguous order (system 0 atoms first, then system 1,
        and so on); interleaved layouts are **not supported** and will
        silently emit cross-system pairs.  When invoked via
        :func:`nvalchemiops.torch.neighbors.neighbor_list` with a
        ``batch_idx`` argument, the dispatcher derives ``batch_ptr`` by
        assuming ``batch_idx`` is sorted by system — the same contract.
    max_neighbors : int, optional
        Max neighbors per atom (``"matrix"`` format only). Falls back to
        :func:`estimate_max_neighbors`.
    fill_value : int, optional
        Matrix sentinel; defaults to ``total_atoms``.
    format : {"matrix", "coo", "tile"}, default "matrix"
        Output representation. See Returns.
    max_pairs : int, optional
        Upper bound for COO output; defaults to
        ``total_atoms * max_neighbors``.
    cutoff2 : float, optional
        Secondary cutoff for matrix output. Dual cutoff is matrix-only and
        cannot be combined with pair-output buffers.
    rebuild_flags : torch.Tensor, shape (num_systems,), dtype=torch.bool, optional
        Per-system selective rebuild flags. Supported for matrix output and
        segmented COO output.
    tile_offsets, tile_counts : torch.Tensor, optional
        Fixed per-system tile offsets and OUTPUT tile counters for segmented
        tile-list state. Use ``estimate_batch_cluster_tile_segments`` to size
        these arrays.
    pair_offsets, pair_counts : torch.Tensor, optional
        Fixed per-system COO offsets and OUTPUT pair counters for segmented
        COO output. Compact COO cannot be combined with ``rebuild_flags``.
    neighbor_matrix, num_neighbors, neighbor_matrix_shifts : torch.Tensor, optional
        Pre-allocated matrix-format outputs. All-or-nothing across the trio.
    neighbor_list, neighbor_list_shifts, pair_counter : torch.Tensor, optional
        Pre-allocated COO-format outputs. Shapes ``(2, max_pairs)``,
        ``(max_pairs, 3)``, ``(1,)`` int32. All-or-nothing across the trio.
    inv_cell_batch : torch.Tensor, optional
        Pre-computed inverse cell matrices.
    sorted_atom_index, sort_inv, sorted_pos_x, sorted_pos_y, sorted_pos_z, batch_idx_sorted, batch_ptr_padded, group_system, group_ptr, group_ctr_*, group_ext_*, num_tiles, tile_row_group, tile_col_group, tile_system : torch.Tensor, optional
        Pre-allocated scratch buffers (shapes as returned by
        ``allocate_batch_cluster_tile_list``). All-or-nothing: provide
        every scratch tensor or none. The trigger is ``sorted_atom_index``.
    return_vectors, return_distances : bool, default False
        Write per-pair displacements / scalar distances to
        ``neighbor_vectors`` / ``neighbor_distances``. Matrix format uses
        ``(total_atoms, max_neighbors, ...)`` buffers; COO format uses
        flat ``(max_pairs, ...)`` buffers.
    pair_fn : wp.Function, optional
        Module-scope Warp ``@wp.func`` of signature
        ``(r_ij, distance, pair_params, i, j) -> (energy, force)``.
    pair_params : torch.Tensor, optional
        Per-atom pair-function parameters; required with ``pair_fn``.
    neighbor_vectors, neighbor_distances, pair_energies, pair_forces : torch.Tensor, optional
        OUTPUT buffers, written only when the corresponding enable flag
        / ``pair_fn`` is active.
    max_tiles_per_group : int, optional
        Upper bound on neighbor groups per row group for scratch allocation.
        Passing this skips the geometry-aware sizing preflight, which otherwise
        synchronizes per-system counts and cell volumes to the host.

    Returns
    -------
    tuple of torch.Tensor
        Shape depends on ``format``:

        - ``"matrix"`` (default): ``(neighbor_matrix, num_neighbors,
          neighbor_matrix_shifts)``, with optional ``(*, distances)``
          and/or ``(*, vectors)`` appended when ``return_distances`` /
          ``return_vectors`` is True, and optional ``(*, pair_energies,
          pair_forces)`` when ``pair_fn`` is set.
        - ``"coo"``: ``(neighbor_list, neighbor_ptr, neighbor_list_shifts)``
          via the direct ``batch_query_cluster_tile_coo`` path (no
          matrix intermediate). ``neighbor_ptr`` is reconstructed from
          ``bincount(neighbor_list[0])``. With segmented COO, returns
          ``(neighbor_list, pair_offsets, pair_counts, neighbor_list_shifts)``
          without trimming the caller-owned fixed segments.
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

    if positions.dtype != torch.float32:
        raise TypeError("positions must be float32")
    if format not in ("matrix", "coo", "tile"):
        raise ValueError(
            f"format must be 'matrix' | 'coo' | 'tile'; got {format!r}",
        )
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
    ):
        raise ValueError(
            "cluster_tile selective COO requires pair_offsets and pair_counts"
        )
    if (pair_offsets is None) != (pair_counts is None):
        raise ValueError("Pass both 'pair_offsets' and 'pair_counts', or neither.")
    if (tile_offsets is None) != (tile_counts is None):
        raise ValueError("Pass both 'tile_offsets' and 'tile_counts', or neither.")
    if batch_ptr.shape[0] < 2:
        raise ValueError("batch_ptr must have length at least 2")
    device = positions.device
    N = int(batch_ptr[-1].item())

    if max_neighbors is None:
        max_neighbors = max(
            estimate_max_neighbors(cutoff2 if cutoff2 is not None else cutoff),
            TILE_GROUP_SIZE,
        )
    if fill_value is None:
        fill_value = N

    needs_segmented_tiles = rebuild_flags is not None or pair_offsets is not None
    if needs_segmented_tiles and tile_offsets is None:
        _tile_caps, tile_offsets, _pair_caps, default_pair_offsets = (
            estimate_batch_cluster_tile_segments(batch_ptr, int(max_neighbors))
        )
        tile_counts = torch.empty(
            int(tile_offsets.shape[0]) - 1, dtype=torch.int32, device=device
        )
        if format == "coo" and pair_offsets is None:
            pair_offsets = default_pair_offsets
            pair_counts = torch.empty(
                int(pair_offsets.shape[0]) - 1, dtype=torch.int32, device=device
            )
    elif needs_segmented_tiles and tile_counts is None:
        raise ValueError("tile_counts is required when tile_offsets is provided")

    # Autograd routing: distances/vectors only (no pair_fn).
    if (
        (bool(return_distances) or bool(return_vectors))
        and pair_fn is None
        and pair_params is None
        and pair_energies is None
        and pair_forces is None
        and format == "matrix"
        and rebuild_flags is None
    ):
        from nvalchemiops.torch.neighbors._autograd import _route_pair_outputs

        # Build atom -> system mapping for the backward (per-system cell grad).
        per_sys_counts = batch_ptr[1:] - batch_ptr[:-1]
        batch_idx_atom = torch.repeat_interleave(
            torch.arange(per_sys_counts.shape[0], dtype=torch.int32, device=device),
            per_sys_counts.to(torch.int64),
        )
        forward_kwargs = {
            "batch_ptr": batch_ptr,
            "cutoff": cutoff,
            "max_neighbors": int(max_neighbors),
            "fill_value": int(fill_value),
            "batch_idx_atom": batch_idx_atom,
        }
        distances_out, vectors_out, nm_out, nn_out, shifts_out = _route_pair_outputs(
            positions,
            cell_batch,
            _batch_cluster_tile_pair_outputs_forward,
            forward_kwargs,
        )
        base = (nm_out, nn_out, shifts_out)
        if return_distances and return_vectors:
            return (*base, distances_out, vectors_out)
        if return_distances:
            return (*base, distances_out)
        return (*base, vectors_out)

    # Tile candidates must cover the larger radius so the cutoff2 matrix cannot
    # miss pairs in the (cutoff, cutoff2] shell; the query filters each matrix
    # by its own cutoff.
    build_cutoff = cutoff if cutoff2 is None else max(float(cutoff), float(cutoff2))

    if sorted_atom_index is None:
        if max_tiles_per_group is None:
            max_tiles_per_group = estimate_batch_max_tiles_per_group(
                batch_ptr, build_cutoff, cell_batch
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
    batch_build_cluster_tile_list(
        positions,
        build_cutoff,
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

    if format == "tile":
        # Raw-tile callers must not receive a silently truncated tile list
        # (format='tile' is always the compact, non-selective path).
        n_tiles = int(num_tiles.item())
        tile_capacity = int(tile_row_group.shape[0])
        if n_tiles > tile_capacity:
            raise NeighborOverflowError(tile_capacity, n_tiles)
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
            max_pairs = int(pair_offsets[-1].item())
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
            overflow = pair_counts > segment_caps
            if bool(overflow.any().item()):
                isys = int(torch.nonzero(overflow, as_tuple=False)[0, 0].item())
                raise NeighborOverflowError(
                    int(segment_caps[isys].item()),
                    int(pair_counts[isys].item()),
                    system_index=isys,
                )
            return (
                coo_buf.transpose(0, 1).contiguous(),
                pair_offsets,
                pair_counts,
                neighbor_list_shifts,
            )

        npairs = int(pair_counter.item())
        if npairs > int(max_pairs):
            raise NeighborOverflowError(int(max_pairs), npairs)
        nl = coo_buf[:npairs].transpose(0, 1).contiguous()
        nls = neighbor_list_shifts[:npairs].contiguous()
        per_atom_counts = torch.bincount(nl[0].long(), minlength=N).to(
            torch.int32,
        )
        neighbor_ptr = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=device),
                torch.cumsum(per_atom_counts, dim=0).to(torch.int32),
            ],
        )
        return nl, neighbor_ptr, nls

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

    # Allocate pair output buffers when caller omits them.
    if return_vectors and neighbor_vectors is None:
        neighbor_vectors = torch.empty(
            (N, max_neighbors, 3),
            dtype=positions.dtype,
            device=device,
        )
    if return_distances and neighbor_distances is None:
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

    batch_query_cluster_tile(
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
        neighbor_matrix,
        num_neighbors,
        neighbor_matrix_shifts,
        inv_cell_batch=inv_cell_batch,
        cutoff2=cutoff2,
        neighbor_matrix2=neighbor_matrix2,
        num_neighbors2=num_neighbors2,
        neighbor_matrix_shifts2=neighbor_matrix_shifts2,
        rebuild_flags=rebuild_flags,
        tile_offsets=tile_offsets,
        tile_counts=tile_counts,
        batch_idx=torch.repeat_interleave(
            torch.arange(int(batch_ptr.shape[0]) - 1, dtype=torch.int32, device=device),
            (batch_ptr[1:] - batch_ptr[:-1]).to(torch.int64),
        )
        if rebuild_flags is not None
        else None,
        return_vectors=return_vectors,
        return_distances=return_distances,
        pair_fn=pair_fn,
        pair_params=pair_params,
        neighbor_vectors=neighbor_vectors,
        neighbor_distances=neighbor_distances,
        pair_energies=pair_energies,
        pair_forces=pair_forces,
    )

    max_seen = int(num_neighbors.max().item()) if N > 0 else 0
    if max_seen > int(max_neighbors):
        raise NeighborOverflowError(int(max_neighbors), max_seen)
    if cutoff2 is not None and num_neighbors2 is not None:
        max_seen2 = int(num_neighbors2.max().item()) if N > 0 else 0
        if max_seen2 > int(max_neighbors):
            raise NeighborOverflowError(int(max_neighbors), max_seen2)

    if max_neighbors > 0:
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

    if cutoff2 is not None:
        return (
            neighbor_matrix,
            num_neighbors,
            neighbor_matrix_shifts,
            neighbor_matrix2,
            num_neighbors2,
            neighbor_matrix_shifts2,
        )
    return neighbor_matrix, num_neighbors, neighbor_matrix_shifts
