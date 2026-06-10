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

"""PyTorch bindings for single-system cluster-pair tile neighbor list.

Mirrors the ``cell_list`` / ``naive`` wrapper pattern: exposes low-level
component ops (``build_cluster_tile_list``, ``query_cluster_tile``,
``query_cluster_tile_coo``) that fill pre-allocated tensors, plus a high-level
convenience entry point ``cluster_tile_neighbor_list``.  All torch-side work
(Morton sort, allocation, ``wp.from_torch`` conversion) lives in this
module; the Warp layer at ``nvalchemiops.neighbors.cluster_tile`` only
sees ``wp.array`` inputs.

Scope: single system, orthorhombic or triclinic PBC, float32, any
``N >= 0`` (the wrapper pads internally to a multiple of TILE_GROUP_SIZE).
"""

from typing import TYPE_CHECKING

import torch
import warp as wp

from nvalchemiops.neighbors.cluster_tile import (
    TILE_GROUP_SIZE,
    estimate_max_tiles_per_group,
)
from nvalchemiops.neighbors.cluster_tile import (
    build_cluster_tile_list as wp_build_cluster_tile_list,
)
from nvalchemiops.neighbors.cluster_tile import (
    query_cluster_tile as wp_query_cluster_tile,
)
from nvalchemiops.neighbors.cluster_tile import (
    query_cluster_tile_coo as wp_query_cluster_tile_coo,
)
from nvalchemiops.neighbors.neighbor_utils import (
    NeighborOverflowError,
    estimate_max_neighbors,
)
from nvalchemiops.neighbors.neighbor_utils import (
    fill_neighbor_matrix_tail as wp_fill_neighbor_matrix_tail,
)
from nvalchemiops.neighbors.neighbor_utils import (
    selective_zero_num_neighbors_single as wp_selective_zero_num_neighbors_single,
)
from nvalchemiops.neighbors.output_args import _has_partial_or_pair_outputs
from nvalchemiops.torch.types import get_wp_dtype

if TYPE_CHECKING:
    from nvalchemiops.torch.neighbors._autograd import _NeighborForwardOutput

__all__ = [
    "TILE_GROUP_SIZE",
    "estimate_cluster_tile_list_sizes",
    "allocate_cluster_tile_list",
    "build_cluster_tile_list",
    "query_cluster_tile",
    "query_cluster_tile_coo",
    "cluster_tile_neighbor_list",
]


@torch.library.custom_op(
    "nvalchemiops::_cluster_tile_fill_neighbor_matrix_tail",
    mutates_args=("neighbor_matrix",),
)
def _cluster_tile_fill_neighbor_matrix_tail_op(
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


@_cluster_tile_fill_neighbor_matrix_tail_op.register_fake
def _(
    num_neighbors: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    n_rows: int,
    max_neighbors: int,
    fill_value: int,
) -> None:
    return None


# =============================================================================
# Sizing + allocation helpers (torch-side, not ``custom_op``-wrapped)
# =============================================================================
def estimate_cluster_tile_list_sizes(
    total_atoms: int,
    max_tiles_per_group: int = 256,
) -> tuple[int, int, int, int]:
    """Estimate allocation sizes for the tile neighbor list state.

    Any ``total_atoms >= 0`` is accepted; internally the state is sized
    at ``n_padded = ceil(total_atoms / TILE_GROUP_SIZE) * TILE_GROUP_SIZE`` so
    the kernels see a 32-aligned layout.  Padding slots receive a
    sentinel max Morton code (see ``_compute_morton_kernel``) so they
    sort to the end and are dropped by the convert/coo kernels'
    ``i_sorted < natom`` filter.

    Parameters
    ----------
    total_atoms : int
        Real atom count.
    max_tiles_per_group : int, default 256
        Upper bound on neighbor groups per row_group (dense-cutoff cap).

    Returns
    -------
    n_padded : int
        Padded atom count = ``ceil(total_atoms / TILE_GROUP_SIZE) * TILE_GROUP_SIZE``.
        Used to size positions / sorted SoA / sorted_atom_index scratch arrays.
    ngroup : int
        Number of 32-atom groups: ``n_padded // TILE_GROUP_SIZE``.
    ngroup_padded : int
        Group-array pad length for in-bounds ``wp.tile_load`` at any
        TILE-aligned offset.  Multiple of ``TILE_GROUP_SIZE``; at least one
        TILE slack over ``ngroup``.
    max_tiles : int
        Upper bound on the tile-pair list size.
    """
    if total_atoms < 0:
        raise ValueError(f"total_atoms must be >= 0; got {total_atoms}")
    n_padded = (
        (total_atoms + TILE_GROUP_SIZE - 1) // TILE_GROUP_SIZE
    ) * TILE_GROUP_SIZE
    if n_padded == 0:
        n_padded = TILE_GROUP_SIZE  # always reserve at least one tile
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
]:
    """Allocate all state tensors consumed by ``build_cluster_tile_list``.

    Returns ``(sorted_atom_index, morton_codes, sorted_pos_x, sorted_pos_y,
    sorted_pos_z, group_ctr_x, group_ctr_y, group_ctr_z, group_ext_x,
    group_ext_y, group_ext_z, num_tiles, tile_row_group, tile_col_group)``.
    """
    n_padded, ngroup, ngroup_padded, max_tiles = estimate_cluster_tile_list_sizes(
        total_atoms,
        max_tiles_per_group=max_tiles_per_group,
    )
    # Scratch arrays sized at the padded layout so non-32-aligned
    # ``total_atoms`` is handled inside ``_build_cluster_tile_list_op``.
    # Padding slots are populated with sentinel Morton codes (see
    # ``_compute_morton_kernel``) and dropped by the convert/coo
    # kernels' ``i_sorted < natom`` filter.
    sorted_atom_index = torch.empty(n_padded, dtype=torch.int32, device=device)
    morton_codes = torch.empty(n_padded, dtype=torch.int32, device=device)
    sorted_pos_x = torch.empty(n_padded, dtype=dtype, device=device)
    sorted_pos_y = torch.empty(n_padded, dtype=dtype, device=device)
    sorted_pos_z = torch.empty(n_padded, dtype=dtype, device=device)
    group_ctr_x = torch.zeros(ngroup_padded, dtype=dtype, device=device)
    group_ctr_y = torch.zeros(ngroup_padded, dtype=dtype, device=device)
    group_ctr_z = torch.zeros(ngroup_padded, dtype=dtype, device=device)
    group_ext_x = torch.zeros(ngroup_padded, dtype=dtype, device=device)
    group_ext_y = torch.zeros(ngroup_padded, dtype=dtype, device=device)
    group_ext_z = torch.zeros(ngroup_padded, dtype=dtype, device=device)
    num_tiles = torch.zeros(1, dtype=torch.int32, device=device)
    tile_row_group = torch.zeros(max_tiles, dtype=torch.int32, device=device)
    tile_col_group = torch.zeros(max_tiles, dtype=torch.int32, device=device)
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


# =============================================================================
# Internal helpers
# =============================================================================
def _cell_invcell_from_cell(
    cell: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize a ``(3, 3)`` or ``(1, 3, 3)`` cell to ``(cell_3x3,
    inv_cell_3x3)`` torch tensors for the Warp kernels.

    Accepts any non-degenerate cell (orthorhombic or triclinic).
    The downstream Warp kernels use ``_wrap_triclinic`` which is a
    strict superset of the old orthorhombic-only path.
    """
    if cell.ndim == 3:
        if cell.shape[0] != 1:
            raise ValueError(
                f"single-system cluster_tile expects (1, 3, 3) cell; got {tuple(cell.shape)}"
            )
        cell_mat = cell[0]
    elif cell.ndim == 2:
        if cell.shape != (3, 3):
            raise ValueError(
                f"cell must be (3, 3) or (1, 3, 3); got {tuple(cell.shape)}"
            )
        cell_mat = cell
    else:
        raise ValueError(f"cell must be (3, 3) or (1, 3, 3); got {tuple(cell.shape)}")
    cell_mat = cell_mat.contiguous()
    inv_cell_mat = torch.linalg.inv(cell_mat).contiguous()
    return cell_mat, inv_cell_mat


def _cell_volume(cell: torch.Tensor) -> float:
    """Return ``abs(det(cell))`` for a ``(3, 3)`` or ``(1, 3, 3)`` cell."""
    cell_mat = cell[0] if cell.ndim == 3 else cell
    return float(torch.linalg.det(cell_mat.to(torch.float64)).abs().item())


def _mat33f_from_torch(mat: torch.Tensor):
    """Zero-copy view a ``(1, 3, 3)`` or ``(3, 3)`` torch tensor as a
    ``wp.array(dtype=wp.mat33f, shape=(1,))``.

    These Warp kernels read the cell / inv_cell as length-1
    ``wp.array(dtype=wp.mat33f)`` and dereference ``cell[0]`` inside the
    kernel body, matching the cell_list pattern.  This avoids a per-call
    host sync.
    """
    if mat.ndim == 2:
        mat = mat.unsqueeze(0)
    return wp.from_torch(
        mat.detach().contiguous().to(torch.float32),
        dtype=wp.mat33f,
        requires_grad=False,
        return_ctype=True,
    )


# =============================================================================
# Component ops (torch.library.custom_op wrappers)
# =============================================================================
@torch.library.custom_op(
    "nvalchemiops::_build_cluster_tile_list",
    mutates_args=(
        "sorted_atom_index",
        "morton_codes",
        "sorted_pos_x",
        "sorted_pos_y",
        "sorted_pos_z",
        "group_ctr_x",
        "group_ctr_y",
        "group_ctr_z",
        "group_ext_x",
        "group_ext_y",
        "group_ext_z",
        "num_tiles",
        "tile_row_group",
        "tile_col_group",
    ),
)
def _build_cluster_tile_list_op(
    positions: torch.Tensor,
    cutoff: float,
    cell: torch.Tensor,
    inv_cell: torch.Tensor,
    sorted_atom_index: torch.Tensor,
    morton_codes: torch.Tensor,
    sorted_pos_x: torch.Tensor,
    sorted_pos_y: torch.Tensor,
    sorted_pos_z: torch.Tensor,
    group_ctr_x: torch.Tensor,
    group_ctr_y: torch.Tensor,
    group_ctr_z: torch.Tensor,
    group_ext_x: torch.Tensor,
    group_ext_y: torch.Tensor,
    group_ext_z: torch.Tensor,
    num_tiles: torch.Tensor,
    tile_row_group: torch.Tensor,
    tile_col_group: torch.Tensor,
) -> None:
    """Compute Morton codes + argsort + SoA gather in torch, then run
    bbox reduction + tile enumeration on the warp side.

    Triclinic-safe via the (cell, inv_cell) pair; orthorhombic cells
    just have a diagonal inv_cell and produce the same result as the
    pre-triclinic implementation.

    See Also
    --------
    nvalchemiops.neighbors.cluster_tile.build_cluster_tile_list : warp launcher
    """
    N = positions.shape[0]
    if N == 0:
        return
    device = positions.device
    wp_device = str(device)
    wp_dtype = get_wp_dtype(positions.dtype)

    num_tiles.zero_()

    wp_cell = _mat33f_from_torch(cell)
    wp_inv_cell = _mat33f_from_torch(inv_cell)

    # ---- Steps 1-3: Morton codes + argsort + gather (torch-side) ----
    # Fractional coords via inv_cell; wrap into [0, 1) so the bucket
    # produces a deterministic 30-bit Morton code.  Orthorhombic cells
    # collapse this back to the cheaper diagonal multiply.
    #
    # Pad to ``n_padded = ceil(N / TILE_GROUP_SIZE) * TILE_GROUP_SIZE`` so the
    # kernels see a 32-aligned layout.  Padding slots get a sentinel
    # max 30-bit Morton code (0x3FFFFFFF) so radix sort places them at
    # the end of the sorted layout, where the convert/coo kernels'
    # ``i_sorted < natom`` filter naturally drops any pair involving
    # them.  Padding positions copy the last real atom (any in-cell
    # position is safe) so the SoA gather has well-defined reads.
    n_padded = int(sorted_atom_index.shape[0])
    if N < n_padded:
        padded_positions = torch.empty(
            (n_padded, 3),
            dtype=positions.dtype,
            device=positions.device,
        )
        padded_positions[:N].copy_(positions)
        padded_positions[N:].copy_(positions[-1:].expand(n_padded - N, 3))
    else:
        padded_positions = positions

    frac = padded_positions @ inv_cell.T
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

    codes32 = (_spread(iz) << 2) | (_spread(iy) << 1) | _spread(ix)
    if N < n_padded:
        # Sentinel: one bit above any real 30-bit Morton code.  Real
        # codes saturate at 0x3FFFFFFF; 0x40000000 sorts after all of
        # them.  See ``_compute_morton_kernel`` for the matching
        # warp-side value.
        codes32[N:] = 0x40000000
    morton_codes.copy_(codes32)
    perm = torch.argsort(codes32)
    sorted_atom_index.copy_(perm.to(torch.int32))
    sorted_pos = padded_positions[perm].contiguous()
    sorted_pos_x.copy_(sorted_pos[:, 0].contiguous())
    sorted_pos_y.copy_(sorted_pos[:, 1].contiguous())
    sorted_pos_z.copy_(sorted_pos[:, 2].contiguous())

    # ---- Step 4: rank2group + group2tile (warp launcher) ----
    wp_sorted_pos_x = wp.from_torch(
        sorted_pos_x,
        dtype=wp_dtype,
        return_ctype=True,
    )
    wp_sorted_pos_y = wp.from_torch(
        sorted_pos_y,
        dtype=wp_dtype,
        return_ctype=True,
    )
    wp_sorted_pos_z = wp.from_torch(
        sorted_pos_z,
        dtype=wp_dtype,
        return_ctype=True,
    )
    wp_group_ctr_x = wp.from_torch(group_ctr_x, dtype=wp_dtype, return_ctype=True)
    wp_group_ctr_y = wp.from_torch(group_ctr_y, dtype=wp_dtype, return_ctype=True)
    wp_group_ctr_z = wp.from_torch(group_ctr_z, dtype=wp_dtype, return_ctype=True)
    wp_group_ext_x = wp.from_torch(group_ext_x, dtype=wp_dtype, return_ctype=True)
    wp_group_ext_y = wp.from_torch(group_ext_y, dtype=wp_dtype, return_ctype=True)
    wp_group_ext_z = wp.from_torch(group_ext_z, dtype=wp_dtype, return_ctype=True)
    wp_num_tiles = wp.from_torch(num_tiles, dtype=wp.int32, return_ctype=True)
    wp_tile_row_group = wp.from_torch(tile_row_group, dtype=wp.int32, return_ctype=True)
    wp_tile_col_group = wp.from_torch(
        tile_col_group,
        dtype=wp.int32,
        return_ctype=True,
    )
    wp_build_cluster_tile_list(
        sorted_pos_x=wp_sorted_pos_x,
        sorted_pos_y=wp_sorted_pos_y,
        sorted_pos_z=wp_sorted_pos_z,
        cell=wp_cell,
        inv_cell=wp_inv_cell,
        cutoff=float(cutoff),
        num_tiles=wp_num_tiles,
        tile_row_group=wp_tile_row_group,
        tile_col_group=wp_tile_col_group,
        wp_dtype=wp_dtype,
        device=wp_device,
        group_ctr_x_buffer=wp_group_ctr_x,
        group_ctr_y_buffer=wp_group_ctr_y,
        group_ctr_z_buffer=wp_group_ctr_z,
        group_ext_x_buffer=wp_group_ext_x,
        group_ext_y_buffer=wp_group_ext_y,
        group_ext_z_buffer=wp_group_ext_z,
    )


@_build_cluster_tile_list_op.register_fake
def _(
    positions: torch.Tensor,
    cutoff: float,
    cell: torch.Tensor,
    inv_cell: torch.Tensor,
    sorted_atom_index: torch.Tensor,
    morton_codes: torch.Tensor,
    sorted_pos_x: torch.Tensor,
    sorted_pos_y: torch.Tensor,
    sorted_pos_z: torch.Tensor,
    group_ctr_x: torch.Tensor,
    group_ctr_y: torch.Tensor,
    group_ctr_z: torch.Tensor,
    group_ext_x: torch.Tensor,
    group_ext_y: torch.Tensor,
    group_ext_z: torch.Tensor,
    num_tiles: torch.Tensor,
    tile_row_group: torch.Tensor,
    tile_col_group: torch.Tensor,
) -> None:
    return None


def build_cluster_tile_list(
    positions: torch.Tensor,
    cutoff: float,
    cell: torch.Tensor,
    sorted_atom_index: torch.Tensor,
    morton_codes: torch.Tensor,
    sorted_pos_x: torch.Tensor,
    sorted_pos_y: torch.Tensor,
    sorted_pos_z: torch.Tensor,
    group_ctr_x: torch.Tensor,
    group_ctr_y: torch.Tensor,
    group_ctr_z: torch.Tensor,
    group_ext_x: torch.Tensor,
    group_ext_y: torch.Tensor,
    group_ext_z: torch.Tensor,
    num_tiles: torch.Tensor,
    tile_row_group: torch.Tensor,
    tile_col_group: torch.Tensor,
) -> None:
    """Build cluster-tile neighbor list state into pre-allocated tensors.

    Normalizes ``cell`` to a ``(3, 3)`` matrix + computes ``inv_cell``,
    then runs Morton sort (torch) + warp bbox reduction + warp
    tile-pair enumeration.  Triclinic cells supported.  All output
    tensors are filled in place.

    Parameters
    ----------
    positions : (N, 3) float
        Atomic coordinates, ``N % 32 == 0``, wrapped to the primary cell.
    cutoff : float
    cell : (1, 3, 3) or (3, 3) float
        Any non-degenerate cell (orthorhombic or triclinic).
    sorted_atom_index : (N,) int32 OUT
    morton_codes : (N,) int32 OUT (scratch used by the op)
    sorted_pos_x/y/z : (N,) float OUT
    group_ctr_x/y/z, group_ext_x/y/z : (ngroup_padded,) float OUT
    num_tiles : (1,) int32 OUT (atomic counter; reset internally)
    tile_row_group, tile_col_group : (max_tiles,) int32 OUT

    See Also
    --------
    nvalchemiops.neighbors.cluster_tile.build_cluster_tile_list : warp launcher
    """
    if positions.dtype != torch.float32:
        raise TypeError("positions must be float32")
    cell_mat, inv_cell_mat = _cell_invcell_from_cell(cell)
    cell_mat = cell_mat.to(positions.dtype)
    inv_cell_mat = inv_cell_mat.to(positions.dtype)
    _build_cluster_tile_list_op(
        positions,
        cutoff,
        cell_mat,
        inv_cell_mat,
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


@torch.library.custom_op(
    "nvalchemiops::_query_cluster_tile",
    mutates_args=("neighbor_matrix", "neighbor_matrix_shifts", "num_neighbors"),
)
def _query_cluster_tile_op(
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
    neighbor_matrix: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    num_neighbors: torch.Tensor,
    n_tiles: int,
) -> None:
    # ``n_tiles`` (host-synced emitted-tile count from the caller) sets the
    # launch dimension so we don't launch over the full allocated tile
    # buffer.  The kernel still guards ``tile >= num_tiles[0]`` defensively.
    device = sorted_pos_x.device
    wp_device = str(device)
    wp_dtype = get_wp_dtype(sorted_pos_x.dtype)
    wp_cell = _mat33f_from_torch(cell)
    wp_inv_cell = _mat33f_from_torch(inv_cell)
    wp_query_cluster_tile(
        sorted_atom_index=wp.from_torch(
            sorted_atom_index, dtype=wp.int32, return_ctype=True
        ),
        sorted_pos_x=wp.from_torch(sorted_pos_x, dtype=wp_dtype, return_ctype=True),
        sorted_pos_y=wp.from_torch(sorted_pos_y, dtype=wp_dtype, return_ctype=True),
        sorted_pos_z=wp.from_torch(sorted_pos_z, dtype=wp_dtype, return_ctype=True),
        num_tiles=wp.from_torch(num_tiles, dtype=wp.int32, return_ctype=True),
        tile_row_group=wp.from_torch(tile_row_group, dtype=wp.int32, return_ctype=True),
        tile_col_group=wp.from_torch(
            tile_col_group,
            dtype=wp.int32,
            return_ctype=True,
        ),
        cell=wp_cell,
        inv_cell=wp_inv_cell,
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
        n_tiles=int(n_tiles),
    )


@_query_cluster_tile_op.register_fake
def _(
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
    neighbor_matrix: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    num_neighbors: torch.Tensor,
    n_tiles: int,
) -> None:
    return None


def query_cluster_tile(
    sorted_atom_index: torch.Tensor,
    sorted_pos_x: torch.Tensor,
    sorted_pos_y: torch.Tensor,
    sorted_pos_z: torch.Tensor,
    num_tiles: torch.Tensor,
    tile_row_group: torch.Tensor,
    tile_col_group: torch.Tensor,
    cell: torch.Tensor,
    cutoff: float,
    natom: int,
    neighbor_matrix: torch.Tensor,
    num_neighbors: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    *,
    cutoff2: float | None = None,
    neighbor_matrix2: torch.Tensor | None = None,
    num_neighbors2: torch.Tensor | None = None,
    neighbor_matrix_shifts2: torch.Tensor | None = None,
    rebuild_flags: torch.Tensor | None = None,
    return_vectors: bool = False,
    return_distances: bool = False,
    pair_fn: wp.Function | None = None,
    pair_params: torch.Tensor | None = None,
    neighbor_vectors: torch.Tensor | None = None,
    neighbor_distances: torch.Tensor | None = None,
    pair_energies: torch.Tensor | None = None,
    pair_forces: torch.Tensor | None = None,
) -> None:
    """Convert the tile pair list to neighbor_matrix form in place.

    Cluster-tile does not support partial neighbor lists; there is no
    ``target_indices`` kwarg.  Use :func:`naive_neighbor_list` or
    :func:`cell_list` for partial neighbor lists.

    Parameters
    ----------
    return_vectors, return_distances : bool, default ``False``
        Write per-pair displacements / distances to
        ``neighbor_vectors`` / ``neighbor_distances``.
    pair_fn : callable, optional
        Module-scope Warp ``@wp.func`` of signature
        ``(r_ij, distance, pair_params, i, j) -> (energy, force)``.
    pair_params : torch.Tensor, shape ``(num_atoms, num_parameters)``, optional
        Per-atom pair-function parameters; required with ``pair_fn``.
    neighbor_vectors, neighbor_distances : torch.Tensor, optional
        OUTPUT buffers for per-pair displacements / distances.
    pair_energies, pair_forces : torch.Tensor, optional
        OUTPUT buffers for per-pair energies / forces; required with
        ``pair_fn``.
    """

    cell_mat, inv_cell_mat = _cell_invcell_from_cell(cell)
    cell_mat = cell_mat.to(sorted_pos_x.dtype)
    inv_cell_mat = inv_cell_mat.to(sorted_pos_x.dtype)
    # Host-sync the emitted-tile count: this tightens the launch dimension
    # to the real tiles (vs the full buffer) and lets us raise on tile-buffer
    # overflow instead of silently dropping tiles.  This ``.item()`` makes the
    # matrix path non-CUDA-graph/torch.compile-capturable by design.
    tile_capacity = int(tile_row_group.shape[0])
    if torch.compiler.is_compiling():
        n_tiles = tile_capacity
    else:
        n_tiles = int(num_tiles.item())
        if n_tiles > tile_capacity:
            raise NeighborOverflowError(tile_capacity, n_tiles)

    feature_path = (
        cutoff2 is not None
        or rebuild_flags is not None
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
    if feature_path:
        if (
            pair_fn is None
            and pair_params is None
            and pair_energies is None
            and pair_forces is None
        ):
            return _query_cluster_tile_optional_no_pair_fn_op(
                cell_mat,
                inv_cell_mat,
                natom,
                cutoff,
                sorted_atom_index,
                sorted_pos_x,
                sorted_pos_y,
                sorted_pos_z,
                num_tiles,
                tile_row_group,
                tile_col_group,
                neighbor_matrix,
                num_neighbors,
                neighbor_matrix_shifts,
                neighbor_matrix2,
                num_neighbors2,
                neighbor_matrix_shifts2,
                rebuild_flags,
                neighbor_vectors,
                neighbor_distances,
                n_tiles,
                cutoff2,
                return_vectors,
                return_distances,
            )
        if torch.compiler.is_compiling():
            raise NotImplementedError(
                "cluster_tile pair_fn outputs are eager-only because callable Warp "
                "functions cannot cross a torch.library.custom_op schema boundary.",
            )
        # Pair outputs are exercised - bypass the torch custom op because it
        # cannot carry a callable ``pair_fn``.
        _query_cluster_tile_optional(
            cell_mat,
            inv_cell_mat,
            natom,
            cutoff,
            sorted_atom_index,
            sorted_pos_x,
            sorted_pos_y,
            sorted_pos_z,
            num_tiles,
            tile_row_group,
            tile_col_group,
            neighbor_matrix,
            num_neighbors,
            neighbor_matrix_shifts,
            cutoff2=cutoff2,
            neighbor_matrix2=neighbor_matrix2,
            num_neighbors2=num_neighbors2,
            neighbor_matrix_shifts2=neighbor_matrix_shifts2,
            rebuild_flags=rebuild_flags,
            return_vectors=return_vectors,
            return_distances=return_distances,
            pair_fn=pair_fn,
            pair_params=pair_params,
            neighbor_vectors=neighbor_vectors,
            neighbor_distances=neighbor_distances,
            pair_energies=pair_energies,
            pair_forces=pair_forces,
            n_tiles=n_tiles,
        )
        return

    _query_cluster_tile_op(
        cutoff,
        natom,
        cell_mat,
        inv_cell_mat,
        sorted_atom_index,
        sorted_pos_x,
        sorted_pos_y,
        sorted_pos_z,
        num_tiles,
        tile_row_group,
        tile_col_group,
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
        n_tiles,
    )


@torch.library.custom_op(
    "nvalchemiops::_query_cluster_tile_optional_no_pair_fn",
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
def _query_cluster_tile_optional_no_pair_fn_op(
    cell_mat: torch.Tensor,
    inv_cell_mat: torch.Tensor,
    natom: int,
    cutoff: float,
    sorted_atom_index: torch.Tensor,
    sorted_pos_x: torch.Tensor,
    sorted_pos_y: torch.Tensor,
    sorted_pos_z: torch.Tensor,
    num_tiles: torch.Tensor,
    tile_row_group: torch.Tensor,
    tile_col_group: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    num_neighbors: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    neighbor_matrix2: torch.Tensor | None,
    num_neighbors2: torch.Tensor | None,
    neighbor_matrix_shifts2: torch.Tensor | None,
    rebuild_flags: torch.Tensor | None,
    neighbor_vectors: torch.Tensor | None,
    neighbor_distances: torch.Tensor | None,
    n_tiles: int,
    cutoff2: float | None,
    return_vectors: bool,
    return_distances: bool,
) -> None:
    _query_cluster_tile_optional(
        cell_mat,
        inv_cell_mat,
        natom,
        cutoff,
        sorted_atom_index,
        sorted_pos_x,
        sorted_pos_y,
        sorted_pos_z,
        num_tiles,
        tile_row_group,
        tile_col_group,
        neighbor_matrix,
        num_neighbors,
        neighbor_matrix_shifts,
        n_tiles=n_tiles,
        cutoff2=cutoff2,
        neighbor_matrix2=neighbor_matrix2,
        num_neighbors2=num_neighbors2,
        neighbor_matrix_shifts2=neighbor_matrix_shifts2,
        rebuild_flags=rebuild_flags,
        return_vectors=return_vectors,
        return_distances=return_distances,
        pair_fn=None,
        pair_params=None,
        neighbor_vectors=neighbor_vectors,
        neighbor_distances=neighbor_distances,
        pair_energies=None,
        pair_forces=None,
    )


@_query_cluster_tile_optional_no_pair_fn_op.register_fake
def _(
    cell_mat: torch.Tensor,
    inv_cell_mat: torch.Tensor,
    natom: int,
    cutoff: float,
    sorted_atom_index: torch.Tensor,
    sorted_pos_x: torch.Tensor,
    sorted_pos_y: torch.Tensor,
    sorted_pos_z: torch.Tensor,
    num_tiles: torch.Tensor,
    tile_row_group: torch.Tensor,
    tile_col_group: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    num_neighbors: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    neighbor_matrix2: torch.Tensor | None,
    num_neighbors2: torch.Tensor | None,
    neighbor_matrix_shifts2: torch.Tensor | None,
    rebuild_flags: torch.Tensor | None,
    neighbor_vectors: torch.Tensor | None,
    neighbor_distances: torch.Tensor | None,
    n_tiles: int,
    cutoff2: float | None,
    return_vectors: bool,
    return_distances: bool,
) -> None:
    return None


def _query_cluster_tile_optional(
    cell_mat: torch.Tensor,
    inv_cell_mat: torch.Tensor,
    natom: int,
    cutoff: float,
    sorted_atom_index: torch.Tensor,
    sorted_pos_x: torch.Tensor,
    sorted_pos_y: torch.Tensor,
    sorted_pos_z: torch.Tensor,
    num_tiles: torch.Tensor,
    tile_row_group: torch.Tensor,
    tile_col_group: torch.Tensor,
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
    return_vectors: bool,
    return_distances: bool,
    pair_fn: wp.Function | None,
    pair_params: torch.Tensor | None,
    neighbor_vectors: torch.Tensor | None,
    neighbor_distances: torch.Tensor | None,
    pair_energies: torch.Tensor | None,
    pair_forces: torch.Tensor | None,
) -> None:
    """Pair-output path: bypass the torch custom op + call warp directly.

    Mirrors :func:`nvalchemiops.torch.neighbors.cell_list._query_cell_list_optional`.
    Torch custom ops cannot carry a callable ``pair_fn`` across their
    schema boundary, so the pair-output path drops down to the warp
    launcher with ``wp.from_torch``-wrapped tensors directly.  No
    host-side ``num_tiles.item()`` sync: the warp kernel guards
    per-tile via the device-side ``num_tiles`` array.
    """
    device = sorted_pos_x.device
    wp_device = str(device)
    wp_dtype = get_wp_dtype(sorted_pos_x.dtype)
    wp_vec_dtype = wp.vec3f
    wp_cell = _mat33f_from_torch(cell_mat)
    wp_inv_cell = _mat33f_from_torch(inv_cell_mat)
    wp_pair_params = (
        wp.from_torch(pair_params, dtype=wp_dtype, requires_grad=False)
        if pair_params is not None
        else None
    )
    wp_neighbor_vectors = (
        wp.from_torch(neighbor_vectors, dtype=wp_vec_dtype, requires_grad=False)
        if neighbor_vectors is not None
        else None
    )
    wp_neighbor_distances = (
        wp.from_torch(neighbor_distances, dtype=wp_dtype, requires_grad=False)
        if neighbor_distances is not None
        else None
    )
    wp_pair_energies = (
        wp.from_torch(pair_energies, dtype=wp_dtype, requires_grad=False)
        if pair_energies is not None
        else None
    )
    wp_pair_forces = (
        wp.from_torch(pair_forces, dtype=wp_vec_dtype, requires_grad=False)
        if pair_forces is not None
        else None
    )
    wp_query_cluster_tile(
        sorted_atom_index=wp.from_torch(
            sorted_atom_index, dtype=wp.int32, requires_grad=False
        ),
        sorted_pos_x=wp.from_torch(sorted_pos_x, dtype=wp_dtype, requires_grad=False),
        sorted_pos_y=wp.from_torch(sorted_pos_y, dtype=wp_dtype, requires_grad=False),
        sorted_pos_z=wp.from_torch(sorted_pos_z, dtype=wp_dtype, requires_grad=False),
        num_tiles=wp.from_torch(num_tiles, dtype=wp.int32, requires_grad=False),
        tile_row_group=wp.from_torch(
            tile_row_group, dtype=wp.int32, requires_grad=False
        ),
        tile_col_group=wp.from_torch(
            tile_col_group, dtype=wp.int32, requires_grad=False
        ),
        cell=wp_cell,
        inv_cell=wp_inv_cell,
        cutoff=float(cutoff),
        natom=int(natom),
        neighbor_matrix=wp.from_torch(
            neighbor_matrix, dtype=wp.int32, requires_grad=False
        ),
        num_neighbors=wp.from_torch(num_neighbors, dtype=wp.int32, requires_grad=False),
        neighbor_matrix_shifts=wp.from_torch(
            neighbor_matrix_shifts, dtype=wp.int32, requires_grad=False
        ),
        wp_dtype=wp_dtype,
        device=wp_device,
        n_tiles=int(n_tiles),
        cutoff2=cutoff2,
        neighbor_matrix2=(
            wp.from_torch(neighbor_matrix2, dtype=wp.int32, requires_grad=False)
            if neighbor_matrix2 is not None
            else None
        ),
        num_neighbors2=(
            wp.from_torch(num_neighbors2, dtype=wp.int32, requires_grad=False)
            if num_neighbors2 is not None
            else None
        ),
        neighbor_matrix_shifts2=(
            wp.from_torch(neighbor_matrix_shifts2, dtype=wp.int32, requires_grad=False)
            if neighbor_matrix_shifts2 is not None
            else None
        ),
        rebuild_flags=(
            wp.from_torch(rebuild_flags, dtype=wp.bool, requires_grad=False)
            if rebuild_flags is not None
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
    "nvalchemiops::_query_cluster_tile_coo",
    mutates_args=("pair_counter", "coo_list", "coo_shifts"),
)
def _query_cluster_tile_coo_op(
    cutoff: float,
    natom: int,
    max_pairs: int,
    cell: torch.Tensor,
    inv_cell: torch.Tensor,
    sorted_atom_index: torch.Tensor,
    sorted_pos_x: torch.Tensor,
    sorted_pos_y: torch.Tensor,
    sorted_pos_z: torch.Tensor,
    num_tiles: torch.Tensor,
    tile_row_group: torch.Tensor,
    tile_col_group: torch.Tensor,
    pair_counter: torch.Tensor,
    coo_list: torch.Tensor,
    coo_shifts: torch.Tensor,
    n_tiles: int,
) -> None:
    # ``n_tiles`` (host-synced emitted-tile count from the caller) tightens
    # the launch dimension; the kernel still guards per-tile defensively.
    device = sorted_pos_x.device
    wp_device = str(device)
    wp_dtype = get_wp_dtype(sorted_pos_x.dtype)
    wp_cell = _mat33f_from_torch(cell)
    wp_inv_cell = _mat33f_from_torch(inv_cell)
    wp_query_cluster_tile_coo(
        sorted_atom_index=wp.from_torch(
            sorted_atom_index, dtype=wp.int32, return_ctype=True
        ),
        sorted_pos_x=wp.from_torch(sorted_pos_x, dtype=wp_dtype, return_ctype=True),
        sorted_pos_y=wp.from_torch(sorted_pos_y, dtype=wp_dtype, return_ctype=True),
        sorted_pos_z=wp.from_torch(sorted_pos_z, dtype=wp_dtype, return_ctype=True),
        num_tiles=wp.from_torch(num_tiles, dtype=wp.int32, return_ctype=True),
        tile_row_group=wp.from_torch(tile_row_group, dtype=wp.int32, return_ctype=True),
        tile_col_group=wp.from_torch(
            tile_col_group,
            dtype=wp.int32,
            return_ctype=True,
        ),
        cell=wp_cell,
        inv_cell=wp_inv_cell,
        cutoff=float(cutoff),
        natom=int(natom),
        max_pairs=int(max_pairs),
        pair_counter=wp.from_torch(
            pair_counter,
            dtype=wp.int32,
            return_ctype=True,
        ),
        coo_list=wp.from_torch(coo_list, dtype=wp.int32, return_ctype=True),
        coo_shifts=wp.from_torch(coo_shifts, dtype=wp.int32, return_ctype=True),
        wp_dtype=wp_dtype,
        device=wp_device,
        n_tiles=int(n_tiles),
    )


@_query_cluster_tile_coo_op.register_fake
def _(
    cutoff: float,
    natom: int,
    max_pairs: int,
    cell: torch.Tensor,
    inv_cell: torch.Tensor,
    sorted_atom_index: torch.Tensor,
    sorted_pos_x: torch.Tensor,
    sorted_pos_y: torch.Tensor,
    sorted_pos_z: torch.Tensor,
    num_tiles: torch.Tensor,
    tile_row_group: torch.Tensor,
    tile_col_group: torch.Tensor,
    pair_counter: torch.Tensor,
    coo_list: torch.Tensor,
    coo_shifts: torch.Tensor,
    n_tiles: int,
) -> None:
    return None


@torch.library.custom_op(
    "nvalchemiops::_query_cluster_tile_coo_optional_no_pair_fn",
    mutates_args=(
        "pair_counter",
        "coo_list",
        "coo_shifts",
        "neighbor_vectors",
        "neighbor_distances",
    ),
)
def _query_cluster_tile_coo_optional_no_pair_fn_op(
    cell_mat: torch.Tensor,
    inv_cell_mat: torch.Tensor,
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
    pair_counter: torch.Tensor,
    coo_list: torch.Tensor,
    coo_shifts: torch.Tensor,
    neighbor_vectors: torch.Tensor | None,
    neighbor_distances: torch.Tensor | None,
    n_tiles: int,
    return_vectors: bool,
    return_distances: bool,
) -> None:
    _query_cluster_tile_coo_optional(
        cell_mat,
        inv_cell_mat,
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
        pair_counter,
        coo_list,
        coo_shifts,
        n_tiles=n_tiles,
        return_vectors=return_vectors,
        return_distances=return_distances,
        pair_fn=None,
        pair_params=None,
        neighbor_vectors=neighbor_vectors,
        neighbor_distances=neighbor_distances,
        pair_energies=None,
        pair_forces=None,
    )


@_query_cluster_tile_coo_optional_no_pair_fn_op.register_fake
def _(
    cell_mat: torch.Tensor,
    inv_cell_mat: torch.Tensor,
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
    pair_counter: torch.Tensor,
    coo_list: torch.Tensor,
    coo_shifts: torch.Tensor,
    neighbor_vectors: torch.Tensor | None,
    neighbor_distances: torch.Tensor | None,
    n_tiles: int,
    return_vectors: bool,
    return_distances: bool,
) -> None:
    return None


def _query_cluster_tile_coo_optional(
    cell_mat: torch.Tensor,
    inv_cell_mat: torch.Tensor,
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
    pair_counter: torch.Tensor,
    coo_list: torch.Tensor,
    coo_shifts: torch.Tensor,
    *,
    n_tiles: int,
    return_vectors: bool,
    return_distances: bool,
    pair_fn: wp.Function | None,
    pair_params: torch.Tensor | None,
    neighbor_vectors: torch.Tensor | None,
    neighbor_distances: torch.Tensor | None,
    pair_energies: torch.Tensor | None,
    pair_forces: torch.Tensor | None,
) -> None:
    """Pair-output COO path: bypass the torch custom op.

    ``n_tiles`` (host-synced emitted-tile count) sets the launch dimension;
    the kernel still guards per-tile defensively.
    """
    device = sorted_pos_x.device
    wp_device = str(device)
    wp_dtype = get_wp_dtype(sorted_pos_x.dtype)
    wp_vec_dtype = wp.vec3f
    wp_pair_params = (
        wp.from_torch(pair_params, dtype=wp_dtype, requires_grad=False)
        if pair_params is not None
        else None
    )
    wp_neighbor_vectors = (
        wp.from_torch(neighbor_vectors, dtype=wp_vec_dtype, requires_grad=False)
        if neighbor_vectors is not None
        else None
    )
    wp_neighbor_distances = (
        wp.from_torch(neighbor_distances, dtype=wp_dtype, requires_grad=False)
        if neighbor_distances is not None
        else None
    )
    wp_pair_energies = (
        wp.from_torch(pair_energies, dtype=wp_dtype, requires_grad=False)
        if pair_energies is not None
        else None
    )
    wp_pair_forces = (
        wp.from_torch(pair_forces, dtype=wp_vec_dtype, requires_grad=False)
        if pair_forces is not None
        else None
    )
    wp_query_cluster_tile_coo(
        sorted_atom_index=wp.from_torch(
            sorted_atom_index, dtype=wp.int32, requires_grad=False
        ),
        sorted_pos_x=wp.from_torch(sorted_pos_x, dtype=wp_dtype, requires_grad=False),
        sorted_pos_y=wp.from_torch(sorted_pos_y, dtype=wp_dtype, requires_grad=False),
        sorted_pos_z=wp.from_torch(sorted_pos_z, dtype=wp_dtype, requires_grad=False),
        num_tiles=wp.from_torch(num_tiles, dtype=wp.int32, requires_grad=False),
        tile_row_group=wp.from_torch(
            tile_row_group, dtype=wp.int32, requires_grad=False
        ),
        tile_col_group=wp.from_torch(
            tile_col_group, dtype=wp.int32, requires_grad=False
        ),
        cell=_mat33f_from_torch(cell_mat),
        inv_cell=_mat33f_from_torch(inv_cell_mat),
        cutoff=float(cutoff),
        natom=int(natom),
        max_pairs=int(max_pairs),
        pair_counter=wp.from_torch(pair_counter, dtype=wp.int32, requires_grad=False),
        coo_list=wp.from_torch(coo_list, dtype=wp.int32, requires_grad=False),
        coo_shifts=wp.from_torch(coo_shifts, dtype=wp.int32, requires_grad=False),
        wp_dtype=wp_dtype,
        device=wp_device,
        n_tiles=int(n_tiles),
        return_vectors=bool(return_vectors),
        return_distances=bool(return_distances),
        pair_fn=pair_fn,
        pair_params=wp_pair_params,
        neighbor_vectors=wp_neighbor_vectors,
        neighbor_distances=wp_neighbor_distances,
        pair_energies=wp_pair_energies,
        pair_forces=wp_pair_forces,
    )


def query_cluster_tile_coo(
    sorted_atom_index: torch.Tensor,
    sorted_pos_x: torch.Tensor,
    sorted_pos_y: torch.Tensor,
    sorted_pos_z: torch.Tensor,
    num_tiles: torch.Tensor,
    tile_row_group: torch.Tensor,
    tile_col_group: torch.Tensor,
    cell: torch.Tensor,
    cutoff: float,
    natom: int,
    max_pairs: int,
    pair_counter: torch.Tensor,
    coo_list: torch.Tensor,
    coo_shifts: torch.Tensor,
    *,
    return_vectors: bool = False,
    return_distances: bool = False,
    pair_fn: wp.Function | None = None,
    pair_params: torch.Tensor | None = None,
    neighbor_vectors: torch.Tensor | None = None,
    neighbor_distances: torch.Tensor | None = None,
    pair_energies: torch.Tensor | None = None,
    pair_forces: torch.Tensor | None = None,
) -> None:
    """Convert the tile pair list to flat COO format in place.

    Cluster-tile does not support partial neighbor lists; there is no
    ``target_indices`` kwarg.  Optional pair outputs use flat COO
    buffers with length ``max_pairs``; they are written in the same
    order as ``coo_list``.
    """

    cell_mat, inv_cell_mat = _cell_invcell_from_cell(cell)
    cell_mat = cell_mat.to(sorted_pos_x.dtype)
    inv_cell_mat = inv_cell_mat.to(sorted_pos_x.dtype)
    # Host-sync the emitted-tile count to tighten the launch and raise on
    # tile-buffer overflow (missing tiles -> missing pairs) instead of
    # silently dropping them.
    tile_capacity = int(tile_row_group.shape[0])
    if torch.compiler.is_compiling():
        n_tiles = tile_capacity
    else:
        n_tiles = int(num_tiles.item())
        if n_tiles > tile_capacity:
            raise NeighborOverflowError(tile_capacity, n_tiles)
    pair_counter.zero_()

    if _has_partial_or_pair_outputs(
        return_vectors=return_vectors,
        return_distances=return_distances,
        pair_fn=pair_fn,
        pair_params=pair_params,
        neighbor_vectors=neighbor_vectors,
        neighbor_distances=neighbor_distances,
        pair_energies=pair_energies,
        pair_forces=pair_forces,
    ):
        if (
            pair_fn is None
            and pair_params is None
            and pair_energies is None
            and pair_forces is None
        ):
            return _query_cluster_tile_coo_optional_no_pair_fn_op(
                cell_mat,
                inv_cell_mat,
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
                pair_counter,
                coo_list,
                coo_shifts,
                neighbor_vectors,
                neighbor_distances,
                n_tiles,
                return_vectors,
                return_distances,
            )
        if torch.compiler.is_compiling():
            raise NotImplementedError(
                "cluster_tile COO pair_fn outputs are eager-only because callable "
                "Warp functions cannot cross a torch.library.custom_op schema boundary.",
            )
        _query_cluster_tile_coo_optional(
            cell_mat,
            inv_cell_mat,
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
            pair_counter,
            coo_list,
            coo_shifts,
            n_tiles=n_tiles,
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

    _query_cluster_tile_coo_op(
        cutoff,
        natom,
        max_pairs,
        cell_mat,
        inv_cell_mat,
        sorted_atom_index,
        sorted_pos_x,
        sorted_pos_y,
        sorted_pos_z,
        num_tiles,
        tile_row_group,
        tile_col_group,
        pair_counter,
        coo_list,
        coo_shifts,
        n_tiles,
    )


# =============================================================================
# High-level convenience
# =============================================================================
def _cluster_tile_pair_outputs_forward(
    positions: torch.Tensor,
    cell: torch.Tensor,
    *,
    cutoff: float,
    max_neighbors: int,
    fill_value: int,
) -> "_NeighborForwardOutput":
    """Forward closure for the torch cluster_tile autograd path."""
    from nvalchemiops.torch.neighbors._autograd import (
        _flatten_active_pairs,
        _NeighborForwardOutput,
    )

    positions_det = positions.detach()
    cell_det = cell.detach()
    N = positions_det.shape[0]
    device = positions_det.device
    (
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
    ) = allocate_cluster_tile_list(N, device, dtype=positions_det.dtype)
    build_cluster_tile_list(
        positions_det,
        cutoff,
        cell_det,
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
    nm = torch.empty((N, max_neighbors), dtype=torch.int32, device=device)
    nn = torch.zeros(N, dtype=torch.int32, device=device)
    nms = torch.empty((N, max_neighbors, 3), dtype=torch.int32, device=device)
    nv = torch.zeros((N, max_neighbors, 3), dtype=positions_det.dtype, device=device)
    nd = torch.zeros((N, max_neighbors), dtype=positions_det.dtype, device=device)
    query_cluster_tile(
        sorted_atom_index,
        sorted_pos_x,
        sorted_pos_y,
        sorted_pos_z,
        num_tiles,
        tile_row_group,
        tile_col_group,
        cell_det,
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
        _cluster_tile_fill_neighbor_matrix_tail_op(
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


def cluster_tile_neighbor_list(
    positions: torch.Tensor,
    cutoff: float,
    cell: torch.Tensor,
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
    # scratch buffers
    sorted_atom_index: torch.Tensor | None = None,
    morton_codes: torch.Tensor | None = None,
    sorted_pos_x: torch.Tensor | None = None,
    sorted_pos_y: torch.Tensor | None = None,
    sorted_pos_z: torch.Tensor | None = None,
    group_ctr_x: torch.Tensor | None = None,
    group_ctr_y: torch.Tensor | None = None,
    group_ctr_z: torch.Tensor | None = None,
    group_ext_x: torch.Tensor | None = None,
    group_ext_y: torch.Tensor | None = None,
    group_ext_z: torch.Tensor | None = None,
    num_tiles: torch.Tensor | None = None,
    tile_row_group: torch.Tensor | None = None,
    tile_col_group: torch.Tensor | None = None,
    # Optional matrix outputs / pair-fn surface
    return_vectors: bool = False,
    return_distances: bool = False,
    pair_fn: wp.Function | None = None,
    pair_params: torch.Tensor | None = None,
    neighbor_vectors: torch.Tensor | None = None,
    neighbor_distances: torch.Tensor | None = None,
    pair_energies: torch.Tensor | None = None,
    pair_forces: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """Build a cluster-pair tile neighbor list (one-shot convenience).

    Single-system PyTorch binding for the cluster-pair tile algorithm.
    Runs Morton sort, Warp bounding-box reduction, and tile enumeration,
    then emits the result in one of three formats selected by ``format=``.
    Supports orthorhombic and triclinic cells alike via
    ``_wrap_triclinic``. Cluster-tile is CUDA float32 only.

    Parameters
    ----------
    positions : torch.Tensor, shape (N, 3), dtype=float32
        Atomic coordinates. Any ``N >= 0``; non-32-aligned ``N`` is
        supported via internal padding to
        ``ceil(N / TILE_GROUP_SIZE) * TILE_GROUP_SIZE``. Padding slots
        use sentinel Morton codes and are filtered out by the
        convert/coo kernels.
    cutoff : float
        Cutoff distance in Cartesian units. Must be positive.
    cutoff2 : float, optional
        Matrix-format second cutoff. When provided, the function returns a
        second ``(neighbor_matrix2, num_neighbors2, neighbor_matrix_shifts2)``
        group for neighbors within ``cutoff2``. Cannot be combined with pair
        outputs or COO/tile formats.
    cell : torch.Tensor, shape (1, 3, 3) or (3, 3), dtype=float32
        Any non-degenerate cell (orthorhombic or triclinic).
    max_neighbors : int, optional
        Falls back to ``estimate_max_neighbors(cutoff)``.  Matrix
        format only.
    fill_value : int, optional
        Matrix sentinel; defaults to ``N``.
    format : {"matrix", "coo", "tile"}, default "matrix"
        Output representation:

        - ``"matrix"``: returns
          ``(neighbor_matrix, num_neighbors, neighbor_matrix_shifts)`` —
          the dense ``(N, max_neighbors)`` row-padded form used by
          ``cell_list`` and ``naive``.
        - ``"coo"``: returns
          ``(neighbor_list, neighbor_ptr, neighbor_list_shifts)`` — flat
          pair list emitted directly by ``query_cluster_tile_coo`` (no matrix
          intermediate).  ``neighbor_ptr`` is reconstructed from
          ``bincount(neighbor_list[0])`` (cheap; requires a single CPU
          sync on ``pair_counter[0]`` that's needed for the trim
          anyway).
        - ``"tile"``: returns the native cluster-pair tile state as a
          7-tuple
          ``(num_tiles, tile_row_group, tile_col_group,
          sorted_atom_index, sorted_pos_x, sorted_pos_y, sorted_pos_z)``.
          No convert kernel is run.  Intended for downstream kernels
          that consume the tile-pair list directly with shared-memory
          tile loads.  Tile pairs are group-level half-fill: every
          emitted pair has ``tile_col_group[t] >= tile_row_group[t]``.
          The consumer chooses atom-level fill.
    max_pairs : int, optional
        Upper bound for COO output; defaults to ``N * max_neighbors``.
    rebuild_flags : torch.Tensor, shape (1,), dtype=bool, optional
        Matrix-format selective rebuild flag. Requires previous tile state
        and previous matrix outputs. When ``rebuild_flags[0]`` is False,
        the previous outputs are returned unchanged.
    neighbor_matrix, num_neighbors, neighbor_matrix_shifts : optional
        Pre-allocated matrix-format outputs.  All-or-nothing only across
        the trio; supply all three or none.
    neighbor_list, neighbor_list_shifts, pair_counter : optional
        Pre-allocated COO-format outputs.  Shapes
        ``(2, max_pairs)``, ``(max_pairs, 3)``, ``(1,)`` int32.
        Same all-or-nothing semantics.
    sorted_atom_index, morton_codes, sorted_pos_x, sorted_pos_y, sorted_pos_z, group_ctr_x, group_ctr_y, group_ctr_z, group_ext_x, group_ext_y, group_ext_z, num_tiles, tile_row_group, tile_col_group : torch.Tensor, optional
        Pre-allocated scratch buffers (shapes as returned by
        ``allocate_cluster_tile_list``).  All-or-nothing: either
        provide every scratch buffer or none.  The trigger is
        ``sorted_atom_index``.  Reuse is safe — ``num_tiles`` is reset
        each call and every other scratch tensor is either fully
        overwritten or only read in regions the kernels just wrote.
    return_vectors, return_distances : bool, default ``False``
        Write per-pair Cartesian displacements / scalar distances to
        ``neighbor_vectors`` / ``neighbor_distances``.
        Matrix format uses ``(N, max_neighbors, ...)`` buffers; COO format
        uses flat ``(max_pairs, ...)`` buffers.
    pair_fn : callable, optional
        Module-scope Warp ``@wp.func`` of signature
        ``(r_ij, distance, pair_params, i, j) -> (energy, force)``.  Writes
        per-pair energies / forces to ``pair_energies`` /
        ``pair_forces``. Matrix format uses row-padded buffers; COO
        format uses flat buffers written in pair-list order.
    pair_params : torch.Tensor, shape ``(num_atoms, num_parameters)``, optional
        Per-atom pair-function parameters; required with ``pair_fn``.
    neighbor_vectors, neighbor_distances : torch.Tensor, optional
        OUTPUT buffers for per-pair displacements / distances. Matrix
        format allocates them when omitted; COO format requires caller-owned
        flat buffers.
    pair_energies, pair_forces : torch.Tensor, optional
        OUTPUT buffers for per-pair energies / forces. Matrix format
        allocates them when omitted; COO format requires caller-owned flat
        buffers.

    Returns
    -------
    tuple of torch.Tensor
        Shape depends on ``format``:

        - ``"matrix"`` (default): ``(neighbor_matrix, num_neighbors,
          neighbor_matrix_shifts)``, with optional ``(*, distances)`` and/or
          ``(*, vectors)`` appended when ``return_distances`` /
          ``return_vectors`` is True, and optional ``(*, pair_energies,
          pair_forces)`` when ``pair_fn`` is set. With ``cutoff2``, returns
          the primary group followed by the secondary cutoff group.
        - ``"coo"``: ``(neighbor_list, neighbor_ptr, neighbor_list_shifts)``.
        - ``"tile"``: ``(num_tiles, tile_row_group, tile_col_group,
          sorted_atom_index, sorted_pos_x, sorted_pos_y, sorted_pos_z)``.

    Notes
    -----
    - Cluster-tile is CUDA float32 only; float64 ``positions`` is rejected.
    - Cluster-tile does not support partial neighbor lists (no
      ``target_indices`` kwarg).
    - The unified
      :func:`nvalchemiops.torch.neighbors.neighbor_list` entry point may
      select this binding automatically when the selector guards and cost
      model prefer it; pass ``method="cluster_tile"`` to force it.

    See Also
    --------
    nvalchemiops.torch.neighbors.batch_cluster_tile_neighbor_list :
        Batched companion entry point.
    nvalchemiops.torch.neighbors.cluster_tile.build_cluster_tile_list :
        Lower-level build step exposed for caching across queries.
    nvalchemiops.torch.neighbors.cluster_tile.query_cluster_tile :
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
    dual_cutoff = cutoff2 is not None
    selective = rebuild_flags is not None
    if has_pair_outputs and format == "tile":
        raise NotImplementedError(
            "Pair outputs (return_vectors / return_distances / pair_fn) "
            "are not supported with format='tile'. Use format='matrix' "
            "or format='coo'.",
        )
    if dual_cutoff:
        if format != "matrix":
            raise ValueError(
                "cluster_tile cutoff2 is supported only with format='matrix'"
            )
        if has_pair_outputs:
            raise ValueError(
                "cluster_tile cutoff2 cannot be combined with pair outputs"
            )
    if selective:
        if format != "matrix":
            raise ValueError(
                "cluster_tile selective rebuild is supported only with format='matrix'"
            )
        if has_pair_outputs:
            raise ValueError(
                "cluster_tile selective rebuild cannot be combined with pair outputs"
            )
        required = {
            "num_tiles": num_tiles,
            "tile_row_group": tile_row_group,
            "tile_col_group": tile_col_group,
            "neighbor_matrix": neighbor_matrix,
            "num_neighbors": num_neighbors,
            "neighbor_matrix_shifts": neighbor_matrix_shifts,
        }
        if dual_cutoff:
            required.update(
                {
                    "neighbor_matrix2": neighbor_matrix2,
                    "num_neighbors2": num_neighbors2,
                    "neighbor_matrix_shifts2": neighbor_matrix_shifts2,
                }
            )
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                "rebuild_flags requires previous cluster_tile state: "
                + ", ".join(missing)
            )
    N = positions.shape[0]
    device = positions.device
    if max_neighbors is None:
        max_neighbors = max(
            estimate_max_neighbors(cutoff2 if cutoff2 is not None else cutoff),
            TILE_GROUP_SIZE,
        )
    if fill_value is None:
        fill_value = N

    if selective and not bool(rebuild_flags.flatten()[0].item()):
        if dual_cutoff:
            return (
                neighbor_matrix,
                num_neighbors,
                neighbor_matrix_shifts,
                neighbor_matrix2,
                num_neighbors2,
                neighbor_matrix_shifts2,
            )
        return neighbor_matrix, num_neighbors, neighbor_matrix_shifts

    # Autograd routing: when only distances/vectors are requested (no
    # pair_fn / energies / forces), route through the autograd primitive
    # so the per-pair outputs are differentiable w.r.t. positions / cell.
    if (
        (bool(return_distances) or bool(return_vectors))
        and pair_fn is None
        and pair_params is None
        and pair_energies is None
        and pair_forces is None
        and format == "matrix"
        and not dual_cutoff
        and not selective
    ):
        from nvalchemiops.torch.neighbors._autograd import _route_pair_outputs

        forward_kwargs = {
            "cutoff": cutoff,
            "max_neighbors": int(max_neighbors),
            "fill_value": int(fill_value),
        }
        distances_out, vectors_out, nm_out, nn_out, shifts_out = _route_pair_outputs(
            positions,
            cell,
            _cluster_tile_pair_outputs_forward,
            forward_kwargs,
        )
        base = (nm_out, nn_out, shifts_out)
        if return_distances and return_vectors:
            return (*base, distances_out, vectors_out)
        if return_distances:
            return (*base, distances_out)
        return (*base, vectors_out)

    # Tile candidates must cover the *larger* radius so the cutoff2 matrix
    # cannot miss pairs in the (cutoff, cutoff2] shell; the query then filters
    # each matrix by its own cutoff.
    build_cutoff = cutoff if cutoff2 is None else max(float(cutoff), float(cutoff2))

    # Allocate scratch if caller didn't supply.  ``sorted_atom_index`` is the
    # all-or-nothing sentinel.
    if sorted_atom_index is None:
        cell_volume = float(_cell_volume(cell))
        max_tiles_per_group = estimate_max_tiles_per_group(N, build_cutoff, cell_volume)
        (
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
        ) = allocate_cluster_tile_list(
            N,
            device,
            dtype=positions.dtype,
            max_tiles_per_group=max_tiles_per_group,
        )
    build_cluster_tile_list(
        positions,
        build_cutoff,
        cell,
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

    if format == "tile":
        # Raw-tile callers must not receive a silently truncated tile list.
        n_tiles = int(num_tiles.item())
        tile_capacity = int(tile_row_group.shape[0])
        if n_tiles > tile_capacity:
            raise NeighborOverflowError(tile_capacity, n_tiles)
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
        if max_pairs is None:
            max_pairs = N * max_neighbors
        # ``query_cluster_tile_coo`` writes row-major (max_pairs, 2); we transpose to
        # package-canonical (2, num_pairs) on the way out.  Pre-allocation
        # kwargs accept the package layout (2, max_pairs); we view-as-flat
        # then reshape for the kernel.
        if neighbor_list is None:
            coo_buf = torch.empty(
                (max_pairs, 2),
                dtype=torch.int32,
                device=device,
            )
        else:
            # Caller passed (2, max_pairs).  Use a transposed view; the
            # kernel writes row-major into this buffer.
            coo_buf = neighbor_list.transpose(0, 1)
        if neighbor_list_shifts is None:
            neighbor_list_shifts = torch.empty(
                (max_pairs, 3),
                dtype=torch.int32,
                device=device,
            )
        if pair_counter is None:
            pair_counter = torch.zeros(1, dtype=torch.int32, device=device)
        query_cluster_tile_coo(
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
            pair_counter,
            coo_buf,
            neighbor_list_shifts,
            return_vectors=return_vectors,
            return_distances=return_distances,
            pair_fn=pair_fn,
            pair_params=pair_params,
            neighbor_vectors=neighbor_vectors,
            neighbor_distances=neighbor_distances,
            pair_energies=pair_energies,
            pair_forces=pair_forces,
        )
        # Trim to actual pair count and rebuild CSR neighbor_ptr.
        # The ``.item()`` is the only sync; it's needed for the slice
        # anyway, so the bincount is on a CPU-known-length tensor.
        npairs = int(pair_counter.item())
        if npairs > int(max_pairs):
            raise NeighborOverflowError(int(max_pairs), npairs)
        nl = coo_buf[:npairs].transpose(0, 1).contiguous()  # (2, npairs)
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
    elif selective:
        wp_selective_zero_num_neighbors_single(
            wp.from_torch(num_neighbors, dtype=wp.int32, return_ctype=True),
            wp.from_torch(rebuild_flags, dtype=wp.bool, return_ctype=True),
            str(device),
        )
    else:
        num_neighbors.zero_()
    if neighbor_matrix_shifts is None:
        neighbor_matrix_shifts = torch.empty(
            (N, max_neighbors, 3),
            dtype=torch.int32,
            device=device,
        )
    if dual_cutoff:
        if neighbor_matrix2 is None:
            neighbor_matrix2 = torch.empty(
                (N, max_neighbors), dtype=torch.int32, device=device
            )
        if num_neighbors2 is None:
            num_neighbors2 = torch.zeros(N, dtype=torch.int32, device=device)
        elif selective:
            wp_selective_zero_num_neighbors_single(
                wp.from_torch(num_neighbors2, dtype=wp.int32, return_ctype=True),
                wp.from_torch(rebuild_flags, dtype=wp.bool, return_ctype=True),
                str(device),
            )
        else:
            num_neighbors2.zero_()
        if neighbor_matrix_shifts2 is None:
            neighbor_matrix_shifts2 = torch.empty(
                (N, max_neighbors, 3), dtype=torch.int32, device=device
            )

    # Pair-output buffer allocation: caller may omit any of the four
    # OUTPUT buffer kwargs and have them allocated via framework-native
    # ``torch.empty`` here.  Required-presence rules
    # (``return_vectors`` ⇒ ``neighbor_vectors``,
    # ``pair_fn`` ⇒ ``pair_{energies,forces}_buffer``) are enforced by
    # the warp launcher.
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

    query_cluster_tile(
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
        neighbor_matrix,
        num_neighbors,
        neighbor_matrix_shifts,
        cutoff2=cutoff2,
        neighbor_matrix2=neighbor_matrix2,
        num_neighbors2=num_neighbors2,
        neighbor_matrix_shifts2=neighbor_matrix_shifts2,
        rebuild_flags=rebuild_flags,
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
    if dual_cutoff and num_neighbors2 is not None:
        max_seen2 = int(num_neighbors2.max().item()) if N > 0 else 0
        if max_seen2 > int(max_neighbors):
            raise NeighborOverflowError(int(max_neighbors), max_seen2)

    # Skip-prefill tail fill: write ``fill_value`` into the unused columns
    # of ``neighbor_matrix``.  Pairs with the always-write-shifts kernel
    # above to eliminate the per-step ``neighbor_matrix.fill_`` and
    # ``neighbor_matrix_shifts.zero_`` ops.
    if max_neighbors > 0:
        _cluster_tile_fill_neighbor_matrix_tail_op(
            num_neighbors,
            neighbor_matrix,
            int(N),
            int(max_neighbors),
            int(fill_value),
        )
        if dual_cutoff:
            _cluster_tile_fill_neighbor_matrix_tail_op(
                num_neighbors2,
                neighbor_matrix2,
                int(N),
                int(max_neighbors),
                int(fill_value),
            )

    if dual_cutoff:
        return (
            neighbor_matrix,
            num_neighbors,
            neighbor_matrix_shifts,
            neighbor_matrix2,
            num_neighbors2,
            neighbor_matrix_shifts2,
        )
    return neighbor_matrix, num_neighbors, neighbor_matrix_shifts
