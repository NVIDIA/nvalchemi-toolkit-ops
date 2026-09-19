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

"""PyTorch utilities for neighbor list construction.

This module contains PyTorch-specific helper functions for neighbor list operations.
"""

from __future__ import annotations

from typing import Literal

import torch
import warp as wp

from nvalchemiops.neighbors.neighbor_utils import (
    NeighborOverflowError,
    TileBufferOverflow,
    estimate_max_neighbors,
)
from nvalchemiops.neighbors.neighbor_utils import (
    compute_naive_num_shifts as wp_compute_naive_num_shifts,
)
from nvalchemiops.torch._warp_op_helpers import scoped_torch_warp_stream
from nvalchemiops.torch.types import get_wp_dtype, get_wp_mat_dtype

__all__ = [
    "compute_naive_num_shifts",
    "get_neighbor_list_from_neighbor_matrix",
    "prepare_batch_idx_ptr",
    "allocate_cell_list",
    "estimate_max_neighbors",
    "synthesize_cell_for_batch",
    "synthesize_cell_for_ss",
    "NeighborOverflowError",
    "TileBufferOverflow",
]


def _raise_if_compiling_host_only(name: str, replacement: str) -> None:
    """Raise a clear error when a host-only helper is traced by Dynamo."""
    if torch.compiler.is_compiling() or torch._dynamo.is_compiling():
        raise RuntimeError(
            f"{name} is a host-only neighbor-list helper and cannot run inside "
            f"torch.compile. {replacement}"
        )


def _check_tile_buffer_capacity(
    counts: torch.Tensor,
    capacities: int | torch.Tensor,
    *,
    segmented: bool = False,
) -> int:
    """Validate cluster-tile buffer capacity without breaking compilation.

    Returns the observed compact count in eager execution, the static compact
    capacity during compilation, and zero for segmented state.
    """
    if torch.compiler.is_compiling():
        torch._assert_async(
            torch.all(counts <= capacities),
            "cluster-tile buffer capacity exceeded",
        )
        return 0 if segmented or not isinstance(capacities, int) else capacities

    if segmented:
        overflow = counts > capacities
        if bool(overflow.any().item()):
            system_index = int(overflow.nonzero(as_tuple=False)[0, 0].item())
            max_tiles = int(capacities[system_index].item())
            num_tiles = int(counts[system_index].item())
            raise TileBufferOverflow(
                max_tiles,
                num_tiles,
                system_index=system_index,
            )
        return 0

    num_tiles = int(counts.max().item()) if counts.numel() > 0 else 0
    max_tiles = (
        int(capacities.item()) if isinstance(capacities, torch.Tensor) else capacities
    )
    if num_tiles > max_tiles:
        raise TileBufferOverflow(max_tiles, num_tiles)
    return num_tiles


def _check_neighbor_capacity(
    counts: torch.Tensor,
    capacities: int | torch.Tensor,
    *,
    segmented: bool = False,
    kind: Literal["matrix", "coo"] = "matrix",
) -> int:
    """Validate matrix or COO capacity without breaking compilation.

    Returns the maximum observed count for a nonsegmented eager call. The
    return value is zero for segmented state and is not meaningful while
    compiling.
    """
    if kind == "matrix":
        compiled_message = "cluster-tile neighbor matrix capacity exceeded"
    elif kind == "coo":
        compiled_message = "cluster-tile COO pair capacity exceeded"
    else:
        raise ValueError("kind must be 'matrix' or 'coo'")

    if torch.compiler.is_compiling():
        torch._assert_async(
            torch.all(counts <= capacities),
            compiled_message,
        )
        return 0

    if segmented:
        overflow = counts > capacities
        if bool(overflow.any().item()):
            system_index = int(overflow.nonzero(as_tuple=False)[0, 0].item())
            max_neighbors = int(capacities[system_index].item())
            num_neighbors = int(counts[system_index].item())
            raise NeighborOverflowError(
                max_neighbors,
                num_neighbors,
                system_index=system_index,
            )
        return 0

    num_neighbors = int(counts.max().item()) if counts.numel() > 0 else 0
    max_neighbors = (
        int(capacities.item()) if isinstance(capacities, torch.Tensor) else capacities
    )
    if num_neighbors > max_neighbors:
        raise NeighborOverflowError(max_neighbors, num_neighbors)
    return num_neighbors


def _prepare_compact_coo_geometry_buffers(
    *,
    device: torch.device,
    dtype: torch.dtype,
    capacity: int,
    return_vectors: bool,
    return_distances: bool,
    neighbor_vectors: torch.Tensor | None,
    neighbor_distances: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate or validate reusable compact-COO geometry buffers."""
    if return_vectors:
        if neighbor_vectors is None:
            neighbor_vectors = torch.empty((capacity, 3), dtype=dtype, device=device)
        elif (
            neighbor_vectors.device != device
            or neighbor_vectors.dtype != dtype
            or neighbor_vectors.ndim != 2
            or neighbor_vectors.shape[1] != 3
            or neighbor_vectors.shape[0] < capacity
        ):
            raise ValueError(
                "neighbor_vectors must have shape (capacity, 3), matching "
                "positions dtype and device, with capacity at least max_pairs"
            )
        if neighbor_vectors.requires_grad:
            raise ValueError(
                "neighbor_vectors must not require gradients; differentiate the "
                "returned geometry instead"
            )
    else:
        neighbor_vectors = torch.empty((1, 3), dtype=dtype, device=device)

    if return_distances:
        if neighbor_distances is None:
            neighbor_distances = torch.empty(capacity, dtype=dtype, device=device)
        elif (
            neighbor_distances.device != device
            or neighbor_distances.dtype != dtype
            or neighbor_distances.ndim != 1
            or neighbor_distances.shape[0] < capacity
        ):
            raise ValueError(
                "neighbor_distances must have shape (capacity,), matching "
                "positions dtype and device, with capacity at least max_pairs"
            )
        if neighbor_distances.requires_grad:
            raise ValueError(
                "neighbor_distances must not require gradients; differentiate the "
                "returned geometry instead"
            )
    else:
        neighbor_distances = torch.empty(1, dtype=dtype, device=device)

    return neighbor_vectors, neighbor_distances


@torch.library.custom_op("nvalchemiops::_compact_coo_prefix", mutates_args=())
def _compact_coo_prefix(
    pair_count: torch.Tensor,
    coo_list: torch.Tensor,
    coo_shifts: torch.Tensor,
    neighbor_vectors: torch.Tensor,
    neighbor_distances: torch.Tensor,
    capacity: int,
    copy_vectors: bool,
    copy_distances: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Copy the active compact-COO prefix into exact, non-aliasing outputs."""
    num_pairs = int(pair_count.reshape(-1)[0].item())
    copy_length = max(0, min(num_pairs, int(capacity)))
    storage_length = max(copy_length, 1)

    pairs = coo_list.new_empty((2, storage_length))[:, :copy_length]
    shifts = coo_shifts.new_empty((storage_length, 3))[:copy_length]
    if copy_length:
        pairs.copy_(coo_list[:copy_length].transpose(0, 1))
        shifts.copy_(coo_shifts[:copy_length])

    vector_length = storage_length if copy_vectors else 1
    vectors = neighbor_vectors.new_empty((vector_length, 3))[
        : copy_length if copy_vectors else 0
    ]
    if copy_vectors and copy_length:
        vectors.copy_(neighbor_vectors[:copy_length])

    distance_length = storage_length if copy_distances else 1
    distances = neighbor_distances.new_empty(distance_length)[
        : copy_length if copy_distances else 0
    ]
    if copy_distances and copy_length:
        distances.copy_(neighbor_distances[:copy_length])
    return pairs, shifts, vectors, distances


@_compact_coo_prefix.register_fake
def _(
    pair_count: torch.Tensor,
    coo_list: torch.Tensor,
    coo_shifts: torch.Tensor,
    neighbor_vectors: torch.Tensor,
    neighbor_distances: torch.Tensor,
    capacity: int,
    copy_vectors: bool,
    copy_distances: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ctx = torch.library.get_ctx()
    num_pairs = ctx.new_dynamic_size(min=0, max=int(capacity))
    pairs = coo_list.new_empty((2, num_pairs))
    shifts = coo_shifts.new_empty((num_pairs, 3))
    vectors = (
        neighbor_vectors.new_empty((num_pairs, 3))
        if copy_vectors
        else neighbor_vectors.new_empty((0, 3))
    )
    distances = (
        neighbor_distances.new_empty((num_pairs,))
        if copy_distances
        else neighbor_distances.new_empty((0,))
    )
    return pairs, shifts, vectors, distances


def _validate_pair_params_present(
    pair_fn: object,
    pair_params: torch.Tensor | None,
) -> None:
    """Validate the torch pair-function parameter contract."""
    if pair_fn is not None and pair_params is None:
        raise ValueError("pair_params is required when pair_fn is provided")


def _validate_cluster_tile_matrix_outputs(
    *,
    device: torch.device,
    dtype: torch.dtype,
    natom: int,
    max_neighbors: int,
    cutoff2: float | None,
    neighbor_matrix2: torch.Tensor | None,
    num_neighbors2: torch.Tensor | None,
    neighbor_matrix_shifts2: torch.Tensor | None,
    return_vectors: bool,
    return_distances: bool,
    neighbor_vectors: torch.Tensor | None,
    neighbor_distances: torch.Tensor | None,
    allocate_missing: bool,
) -> None:
    """Validate optional cluster-tile matrix outputs before mutation."""

    def validate(
        name: str,
        tensor: torch.Tensor,
        expected_shape: tuple[int, ...],
        expected_dtype: torch.dtype,
    ) -> None:
        if tensor.device != device:
            raise ValueError(f"{name} must be on the query device")
        if tensor.dtype != expected_dtype or tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape} and dtype "
                f"{expected_dtype}; got shape {tuple(tensor.shape)} and "
                f"dtype {tensor.dtype}."
            )

    secondary = (neighbor_matrix2, num_neighbors2, neighbor_matrix_shifts2)
    num_secondary = sum(value is not None for value in secondary)
    if 0 < num_secondary < len(secondary):
        raise ValueError(
            "neighbor_matrix2, num_neighbors2, and neighbor_matrix_shifts2 "
            "must be supplied together"
        )
    if cutoff2 is None:
        if num_secondary:
            raise ValueError("secondary matrix outputs require cutoff2")
    elif num_secondary == 0:
        if not allocate_missing:
            raise ValueError(
                "cutoff2 requires neighbor_matrix2, num_neighbors2, and "
                "neighbor_matrix_shifts2"
            )
    else:
        validate(
            "neighbor_matrix2",
            neighbor_matrix2,
            (natom, max_neighbors),
            torch.int32,
        )
        validate("num_neighbors2", num_neighbors2, (natom,), torch.int32)
        validate(
            "neighbor_matrix_shifts2",
            neighbor_matrix_shifts2,
            (natom, max_neighbors, 3),
            torch.int32,
        )

    geometry = (
        (
            "neighbor_vectors",
            return_vectors,
            neighbor_vectors,
            (natom, max_neighbors, 3),
        ),
        (
            "neighbor_distances",
            return_distances,
            neighbor_distances,
            (natom, max_neighbors),
        ),
    )
    for name, enabled, tensor, expected_shape in geometry:
        if not enabled:
            if tensor is not None:
                raise ValueError(f"{name} is only valid when its output is enabled")
            continue
        if tensor is None:
            if allocate_missing:
                continue
            raise ValueError(f"{name} is required when its output is enabled")
        validate(name, tensor, expected_shape, dtype)
        if tensor.requires_grad:
            raise ValueError(
                f"{name} must not require gradients; differentiate the returned "
                "geometry instead"
            )


def _validate_segmented_coo_structure(
    *,
    device: torch.device,
    num_systems: int,
    neighbor_list: torch.Tensor,
    neighbor_list_shifts: torch.Tensor,
    pair_offsets: torch.Tensor,
    pair_counts: torch.Tensor,
    rebuild_flags: torch.Tensor | None,
    tile_offsets: torch.Tensor | None = None,
    tile_counts: torch.Tensor | None = None,
    num_tiles: torch.Tensor | None = None,
    tile_row_group: torch.Tensor | None = None,
    tile_col_group: torch.Tensor | None = None,
    tile_system: torch.Tensor | None = None,
) -> int:
    """Validate segmented COO shapes, dtypes, devices, and capacities."""
    tensors = {
        "neighbor_list": neighbor_list,
        "neighbor_list_shifts": neighbor_list_shifts,
        "pair_offsets": pair_offsets,
        "pair_counts": pair_counts,
    }
    if rebuild_flags is not None:
        tensors["rebuild_flags"] = rebuild_flags
    optional_tensors = {
        "tile_offsets": tile_offsets,
        "tile_counts": tile_counts,
        "num_tiles": num_tiles,
        "tile_row_group": tile_row_group,
        "tile_col_group": tile_col_group,
        "tile_system": tile_system,
    }
    tensors.update(
        {name: value for name, value in optional_tensors.items() if value is not None}
    )
    for name, value in tensors.items():
        if value.device != device:
            raise ValueError(f"{name} must match positions.device")

    if (
        neighbor_list.dtype != torch.int32
        or neighbor_list.ndim != 2
        or neighbor_list.shape[0] != 2
    ):
        raise ValueError("neighbor_list must have shape (2, capacity) and dtype int32")
    capacity = int(neighbor_list.shape[1])
    if neighbor_list_shifts.dtype != torch.int32 or neighbor_list_shifts.shape != (
        capacity,
        3,
    ):
        raise ValueError(
            "neighbor_list_shifts must have shape (neighbor_list capacity, 3) "
            "and dtype int32"
        )
    if pair_offsets.dtype != torch.int32 or pair_offsets.shape != (num_systems + 1,):
        raise ValueError(
            "pair_offsets must have shape (num_systems + 1,) and dtype int32"
        )
    if pair_counts.dtype != torch.int32 or pair_counts.shape != (num_systems,):
        raise ValueError("pair_counts must have shape (num_systems,) and dtype int32")
    if rebuild_flags is not None and (
        rebuild_flags.dtype != torch.bool or rebuild_flags.shape != (num_systems,)
    ):
        raise ValueError("rebuild_flags must have shape (num_systems,) and dtype bool")

    tile_values = (tile_offsets, tile_counts, num_tiles, tile_row_group, tile_col_group)
    if any(value is not None for value in tile_values):
        if any(value is None for value in tile_values):
            raise ValueError("segmented COO tile state must be supplied completely")
        if (
            tile_offsets.dtype != torch.int32
            or tile_offsets.shape != (num_systems + 1,)
            or tile_counts.dtype != torch.int32
            or tile_counts.shape != (num_systems,)
            or num_tiles.dtype != torch.int32
            or num_tiles.shape != (1,)
        ):
            raise ValueError(
                "segmented COO tile metadata has an invalid shape or dtype"
            )
        if (
            tile_row_group.dtype != torch.int32
            or tile_col_group.dtype != torch.int32
            or tile_row_group.ndim != 1
            or tile_col_group.ndim != 1
            or tile_row_group.shape != tile_col_group.shape
        ):
            raise ValueError(
                "tile row and column buffers must be matching 1D int32 arrays"
            )
        if tile_system is not None and (
            tile_system.dtype != torch.int32
            or tile_system.ndim != 1
            or tile_system.shape != tile_row_group.shape
        ):
            raise ValueError("tile_system must match tile row and column buffer shapes")

    return capacity


def _validate_segmented_coo_values(
    *,
    neighbor_list: torch.Tensor,
    pair_offsets: torch.Tensor,
    pair_counts: torch.Tensor,
    tile_offsets: torch.Tensor | None = None,
    tile_counts: torch.Tensor | None = None,
    tile_row_group: torch.Tensor | None = None,
) -> None:
    """Eagerly validate segmented COO metadata values before mutation."""
    capacity = int(neighbor_list.shape[1])
    pair_offsets_host = pair_offsets.cpu()
    pair_counts_host = pair_counts.cpu()
    if int(pair_offsets_host[0]) != 0:
        raise ValueError("pair_offsets must start at zero")
    if bool((pair_offsets_host[1:] < pair_offsets_host[:-1]).any()):
        raise ValueError("pair_offsets must be nondecreasing")
    if int(pair_offsets_host[-1]) != capacity:
        raise ValueError("pair_offsets final value must equal neighbor_list capacity")
    pair_capacities = pair_offsets_host[1:] - pair_offsets_host[:-1]
    if bool((pair_counts_host < 0).any()) or bool(
        (pair_counts_host > pair_capacities).any()
    ):
        raise ValueError("pair_counts must lie within their pair_offsets segments")

    if tile_offsets is not None:
        tile_offsets_host = tile_offsets.cpu()
        tile_counts_host = tile_counts.cpu()
        if int(tile_offsets_host[0]) != 0:
            raise ValueError("tile_offsets must start at zero")
        if bool((tile_offsets_host[1:] < tile_offsets_host[:-1]).any()):
            raise ValueError("tile_offsets must be nondecreasing")
        if int(tile_offsets_host[-1]) > int(tile_row_group.shape[0]):
            raise ValueError("tile_offsets exceed the physical tile-buffer capacity")
        tile_capacities = tile_offsets_host[1:] - tile_offsets_host[:-1]
        if bool((tile_counts_host < 0).any()) or bool(
            (tile_counts_host > tile_capacities).any()
        ):
            raise ValueError("tile_counts must lie within their tile_offsets segments")


def _validate_segmented_coo_state(
    *,
    device: torch.device,
    num_systems: int,
    neighbor_list: torch.Tensor,
    neighbor_list_shifts: torch.Tensor,
    pair_offsets: torch.Tensor,
    pair_counts: torch.Tensor,
    rebuild_flags: torch.Tensor | None,
    tile_offsets: torch.Tensor | None = None,
    tile_counts: torch.Tensor | None = None,
    num_tiles: torch.Tensor | None = None,
    tile_row_group: torch.Tensor | None = None,
    tile_col_group: torch.Tensor | None = None,
    tile_system: torch.Tensor | None = None,
) -> int:
    """Validate fixed-capacity segmented COO state before a kernel launch."""
    capacity = _validate_segmented_coo_structure(
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
    if not torch.compiler.is_compiling():
        _validate_segmented_coo_values(
            neighbor_list=neighbor_list,
            pair_offsets=pair_offsets,
            pair_counts=pair_counts,
            tile_offsets=tile_offsets,
            tile_counts=tile_counts,
            tile_row_group=tile_row_group,
        )
    return capacity


def _normalize_compiled_single_segment_coo_count(
    *,
    pair_offsets: torch.Tensor,
    pair_counts: torch.Tensor,
    rebuild_flags: torch.Tensor,
    physical_capacity: int,
) -> None:
    """Fail closed for malformed compiled single-segment COO metadata.

    This compiled-path helper uses only device-side int32 operations. A true
    rebuild asserts when the resulting count exceeds the fixed segment. A
    skipped rebuild preserves a valid saved count, while malformed offsets or
    an invalid saved count are normalized to zero. This is deliberately not a
    generic batched normalizer.
    """
    offsets_valid = (pair_offsets[0] == 0) & (pair_offsets[1] == physical_capacity)
    rebuilt_count = torch.where(
        offsets_valid & rebuild_flags.flatten()[0],
        pair_counts,
        torch.zeros_like(pair_counts),
    )
    _check_neighbor_capacity(
        rebuilt_count,
        physical_capacity,
        kind="coo",
    )
    clamped_count = torch.clamp(pair_counts, min=0, max=physical_capacity)
    saved_count_valid = (pair_counts >= 0) & (pair_counts <= physical_capacity)
    normalized_counts = torch.where(
        offsets_valid & (rebuild_flags.flatten()[0] | saved_count_valid),
        clamped_count,
        torch.zeros_like(pair_counts),
    )
    pair_counts.copy_(normalized_counts)


@scoped_torch_warp_stream
def compute_naive_num_shifts(
    cell: torch.Tensor,
    cutoff: float,
    pbc: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Compute periodic image shifts needed for neighbor searching.

    Parameters
    ----------
    cell : torch.Tensor, shape (num_systems, 3, 3)
        Cell matrices defining lattice vectors in Cartesian coordinates.
        Each 3x3 matrix represents one system's periodic cell.
    cutoff : float
        Cutoff distance for neighbor searching in Cartesian units.
        Must be positive and typically less than half the minimum cell dimension.
    pbc : torch.Tensor, shape (num_systems, 3), dtype=bool
        Periodic boundary condition flags for each dimension.
        True enables periodicity in that direction.

    Returns
    -------
    shift_range : torch.Tensor, shape (num_systems, 3), dtype=int32
        Maximum shift indices in each dimension for each system.
    num_shifts : torch.Tensor, shape (num_systems,), dtype=int32
        Number of periodic shifts for each system.
    max_shifts : int
        Maximum per-system shift count across all systems.

    Raises
    ------
    ValueError
        If any per-system shift count exceeds int32 range.

    See Also
    --------
    nvalchemiops.neighbors.neighbor_utils.compute_naive_num_shifts : Core warp launcher
    """
    _raise_if_compiling_host_only(
        "compute_naive_num_shifts",
        "Call it before compiling and pass shift_range_per_dimension, "
        "num_shifts_per_system, and max_shifts_per_system to the compiled "
        "neighbor-list call.",
    )
    num_systems = cell.shape[0]
    device = cell.device

    num_shifts_i32 = torch.empty(num_systems, dtype=torch.int32, device=device)
    shift_range = torch.empty((num_systems, 3), dtype=torch.int32, device=device)

    wp_dtype = get_wp_dtype(cell.dtype)
    wp_mat_dtype = get_wp_mat_dtype(cell.dtype)
    wp_device = wp.device_from_torch(device)

    wp_cell = wp.from_torch(cell, dtype=wp_mat_dtype, requires_grad=False)
    wp_pbc = wp.from_torch(pbc, dtype=wp.bool, requires_grad=False)
    wp_num_shifts = wp.from_torch(num_shifts_i32, dtype=wp.int32, requires_grad=False)
    wp_shift_range = wp.from_torch(shift_range, dtype=wp.vec3i, requires_grad=False)

    wp_compute_naive_num_shifts(
        cell=wp_cell,
        cutoff=cutoff,
        pbc=wp_pbc,
        num_shifts=wp_num_shifts,
        shift_range=wp_shift_range,
        wp_dtype=wp_dtype,
        device=str(wp_device),
    )

    s = shift_range.to(torch.int64)
    k1 = 2 * s[:, 1] + 1
    k2 = 2 * s[:, 2] + 1
    num_shifts_i64 = s[:, 0] * k1 * k2 + s[:, 1] * k2 + s[:, 2] + 1

    max_shifts_i64 = num_shifts_i64.max().item() if num_systems > 0 else 0
    if max_shifts_i64 > 2**31 - 1:
        raise ValueError(
            f"Per-system shift count ({max_shifts_i64}) exceeds int32 max "
            f"(2^31 - 1). Reduce the cutoff, increase cell size, or use a "
            f"cell-list method for very small cells."
        )

    num_shifts = num_shifts_i64.to(torch.int32)
    return shift_range, num_shifts, int(max_shifts_i64)


def get_neighbor_list_from_neighbor_matrix(
    neighbor_matrix: torch.Tensor,
    num_neighbors: torch.Tensor,
    neighbor_shift_matrix: torch.Tensor | None = None,
    fill_value: int = -1,
) -> (
    tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
):
    """Convert neighbor matrix format to neighbor list format.

    Parameters
    ----------
    neighbor_matrix : torch.Tensor, shape (total_atoms, max_neighbors), dtype=int32
        The neighbor matrix with neighbor atom indices.
    num_neighbors : torch.Tensor, shape (total_atoms,), dtype=int32
        The number of neighbors for each atom.
    neighbor_shift_matrix : torch.Tensor | None, shape (total_atoms, max_neighbors, 3), dtype=int32
        Optional neighbor shift matrix with periodic shift vectors.
    fill_value : int, default=-1
        The fill value used in the neighbor matrix to indicate empty slots.
        This is used to create a mask from the neighbor matrix.

    Returns
    -------
    neighbor_list : torch.Tensor, shape (2, num_pairs), dtype=int32
        The neighbor list in COO format [source_atoms, target_atoms].
    neighbor_ptr : torch.Tensor, shape (total_atoms + 1,), dtype=int32
        CSR-style pointer array where neighbor_ptr[i]:neighbor_ptr[i+1] gives the range of
        neighbors for atom i in the flattened neighbor list.
    neighbor_list_shifts : torch.Tensor, shape (num_pairs, 3), dtype=int32
        The neighbor shift vectors (only returned if neighbor_shift_matrix is not None).

    Raises
    ------
    NeighborOverflowError
        If eager execution finds more neighbors than the neighbor matrix can
        hold.  The exception retains the allocated capacity and the observed
        maximum neighbor count in ``max_neighbors`` and ``num_neighbors``.
    RuntimeError
        If compiled execution finds an insufficient neighbor-matrix capacity.
        The compiled assertion is device-side and may be reported
        asynchronously by the active device runtime.

    Notes
    -----
    This is a pure PyTorch utility function with no warp dependencies. It converts
    from the fixed-width matrix format to the variable-width list format by masking
    out fill values and flattening the result. Compiled callers use a device-side
    capacity assertion to avoid converting a tensor to a Python scalar. Exact output
    allocation through ``nonzero`` is data-dependent and requires host synchronization.

    See Also
    --------
    nvalchemiops.torch.neighbors.naive_neighbor_list : Uses this for format conversion
    nvalchemiops.torch.neighbors.cell_list : Uses this for format conversion
    """
    # Handle empty case
    if num_neighbors.shape[0] == 0:
        neighbor_list = torch.zeros(
            2, 0, dtype=neighbor_matrix.dtype, device=neighbor_matrix.device
        )
        neighbor_ptr = torch.zeros(1, dtype=torch.int32, device=neighbor_matrix.device)
        if neighbor_shift_matrix is not None:
            neighbor_shift_list = torch.empty(
                0,
                3,
                dtype=neighbor_shift_matrix.dtype,
                device=neighbor_shift_matrix.device,
            )
            return neighbor_list, neighbor_ptr, neighbor_shift_list
        else:
            return neighbor_list, neighbor_ptr

    # Validate that the neighbor matrix is large enough.  Eager callers retain
    # the structured overflow exception; compiled callers need a device-side
    # assertion because converting ``max_found`` to a Python scalar would break
    # graph capture and would synchronize the device.
    max_found = num_neighbors.max()
    if torch.compiler.is_compiling():
        torch._assert_async(
            max_found <= neighbor_matrix.shape[1],
            "neighbor matrix capacity is insufficient for the requested COO output",
        )
    else:
        max_found_value = int(max_found.item())
        if max_found_value > neighbor_matrix.shape[1]:
            raise NeighborOverflowError(
                neighbor_matrix.shape[1],
                max_found_value,
            )

    # Create mask and extract neighbor pairs.  ``nonzero`` returns the row and
    # slot coordinates together, avoiding separate dynamic mask compactions and
    # retaining row-major ordering.
    mask = neighbor_matrix != fill_value
    dtype = neighbor_matrix.dtype
    i_idx, slot_idx = mask.nonzero(as_tuple=True)
    j_idx = neighbor_matrix[i_idx, slot_idx].to(dtype)
    neighbor_list = torch.stack([i_idx.to(dtype), j_idx], dim=0)

    # Create CSR-style pointer array
    neighbor_ptr = torch.zeros(
        num_neighbors.shape[0] + 1, dtype=torch.int32, device=neighbor_matrix.device
    )
    torch.cumsum(num_neighbors, dim=0, out=neighbor_ptr[1:])

    if neighbor_shift_matrix is not None:
        neighbor_list_shifts = neighbor_shift_matrix[i_idx, slot_idx]
        return neighbor_list, neighbor_ptr, neighbor_list_shifts
    else:
        return neighbor_list, neighbor_ptr


def coo_pack_pair_geometry(
    active_mask: torch.Tensor,
    distances: torch.Tensor | None = None,
    vectors: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Repack matrix-layout per-pair geometry into COO order.

    ``active_mask`` is ``neighbor_matrix != fill_value``.  Flattening it in
    row-major order yields the active-slot indices in the same order
    :func:`get_neighbor_list_from_neighbor_matrix` uses, so the gathered
    distances ``(num_pairs,)`` and vectors ``(num_pairs, 3)`` index-align with
    the returned neighbor list.  ``index_select`` keeps the autograd link.

    Parameters
    ----------
    active_mask : torch.Tensor, shape (total_atoms, max_neighbors), dtype=bool
        Mask of active neighbor-matrix slots.
    distances : torch.Tensor | None, shape (total_atoms, max_neighbors)
        Per-pair distances in matrix layout, or ``None``.
    vectors : torch.Tensor | None, shape (total_atoms, max_neighbors, 3)
        Per-pair displacement vectors in matrix layout, or ``None``.

    Returns
    -------
    tuple of (torch.Tensor | None, torch.Tensor | None)
        ``(distances, vectors)`` in COO layout, each unchanged if it was
        ``None``.
    """
    flat_active = active_mask.reshape(-1).nonzero(as_tuple=True)[0]
    if distances is not None:
        distances = distances.reshape(-1).index_select(0, flat_active)
    if vectors is not None:
        vectors = vectors.reshape(-1, vectors.shape[-1]).index_select(0, flat_active)
    return distances, vectors


def prepare_batch_idx_ptr(
    batch_idx: torch.Tensor | None,
    batch_ptr: torch.Tensor | None,
    num_atoms: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare batch index and pointer tensors from either representation.

    Utility function to ensure both batch_idx and batch_ptr are available,
    computing one from the other if needed.

    Parameters
    ----------
    batch_idx : torch.Tensor | None, shape (total_atoms,), dtype=int32
        Tensor indicating the batch index for each atom.
    batch_ptr : torch.Tensor | None, shape (num_systems + 1,), dtype=int32
        Tensor indicating the start index of each batch in the atom list.
    num_atoms : int
        Total number of atoms across all systems.
    device : torch.device
        Device on which to create tensors if needed.

    Returns
    -------
    batch_idx : torch.Tensor, shape (total_atoms,), dtype=int32
        Prepared batch index tensor.
    batch_ptr : torch.Tensor, shape (num_systems + 1,), dtype=int32
        Prepared batch pointer tensor.

    Raises
    ------
    ValueError
        If both batch_idx and batch_ptr are None.
    RuntimeError
        If batch_idx length does not match num_atoms (only checked in eager mode).

    Notes
    -----
    This is a pure PyTorch utility function with no warp dependencies. It provides
    convenience for batch operations by converting between dense (batch_idx) and
    sparse (batch_ptr) batch representations.

    The batch_idx size validation is only performed in eager mode to avoid graph
    breaks during torch.compile tracing. During compiled execution, mismatched
    sizes will result in undefined behavior.

    See Also
    --------
    nvalchemiops.torch.neighbors.batch_naive_neighbor_list : Uses this for batch setup
    nvalchemiops.torch.neighbors.batch_cell_list : Uses this for batch setup
    """
    if batch_idx is None and batch_ptr is None:
        raise ValueError("Either batch_idx or batch_ptr must be provided.")

    if batch_ptr is not None and batch_ptr.shape[0] < 2:
        raise ValueError("batch_ptr must have length at least 2")

    # Validate batch_idx size in eager mode only to avoid graph breaks
    if not torch.compiler.is_compiling():
        if batch_idx is not None and batch_idx.shape[0] != num_atoms:
            raise RuntimeError(
                f"batch_idx length ({batch_idx.shape[0]}) does not match "
                f"num_atoms ({num_atoms}). batch_idx must have one entry per atom."
            )

    if batch_idx is None:
        num_systems = batch_ptr.shape[0] - 1
        num_atoms_per_system = batch_ptr[1:] - batch_ptr[:-1]
        batch_idx = torch.repeat_interleave(
            torch.arange(num_systems, dtype=torch.int32, device=device),
            num_atoms_per_system,
        )

    elif batch_ptr is None:
        num_systems = batch_idx.max() + 1
        num_atoms_per_system = torch.bincount(batch_idx, minlength=num_systems)
        batch_ptr = torch.zeros(num_systems + 1, dtype=torch.int32, device=device)
        torch.cumsum(num_atoms_per_system, dim=0, out=batch_ptr[1:])

    return batch_idx, batch_ptr


def synthesize_cell_for_ss(
    positions: torch.Tensor,
    cutoff: float,
    padding_fraction: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build an orthorhombic non-PBC cell around ``positions``.

    Used by ``neighbor_list(method=..., cell=None)`` so callers without
    a real simulation cell still get a tight bounding box.  The
    returned positions are shifted so the minimum corner is at the
    origin; the cell diagonal is the extent plus ``padding_fraction *
    cutoff`` so atoms never sit on the boundary.

    Parameters
    ----------
    positions : (N, 3) float
        Atomic coordinates (any frame).
    cutoff : float
        Neighbor cutoff; only used to size the boundary padding.
    padding_fraction : float, default 0.1
        Padding around the bounding box, expressed as a fraction of
        ``cutoff``.

    Returns
    -------
    positions : (N, 3) float
        Same dtype/device as input; shifted so ``min == 0``.
    cell : (1, 3, 3) float
        Orthorhombic cell whose diagonal is the (padded) extent.
    pbc : (3,) bool
        ``[False, False, False]`` — synthesized cells are non-periodic.
    """
    pbc = torch.zeros(3, dtype=torch.bool, device=positions.device)
    if positions.shape[0] == 0:
        cell = torch.eye(3, dtype=positions.dtype, device=positions.device).reshape(
            1, 3, 3
        )
        return positions, cell, pbc
    pos_min = positions.min(dim=0).values
    positions = positions - pos_min
    pos_max = positions.max(dim=0).values
    cell_lengths = pos_max + padding_fraction * cutoff
    cell = torch.diag(cell_lengths).reshape(1, 3, 3)
    return positions, cell, pbc


def synthesize_cell_for_batch(
    positions: torch.Tensor,
    batch_idx: torch.Tensor,
    batch_ptr: torch.Tensor,
    cutoff: float,
    padding_fraction: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-system bounding-box cells around ``positions`` for batched inputs.

    Companion to :func:`synthesize_cell_for_ss` for the batched
    entry points (``batch_cell_list``, ``batch_cluster_tile_neighbor_list``).
    For each system, computes the tight ``(min, max)`` bbox via
    ``scatter_reduce`` and synthesizes an orthorhombic non-PBC cell
    with ``padding_fraction * cutoff`` of slack on each side.
    Positions are shifted so each system's minimum corner is at the
    origin.

    Parameters
    ----------
    positions : (total_atoms, 3) float
        Concatenated atomic coordinates.
    batch_idx : (total_atoms,) int32
        System index per atom.
    batch_ptr : (num_systems + 1,) int32
        CSR offsets — used only to derive ``num_systems``. Must have length at least 2.
    cutoff : float
        Neighbor cutoff.
    padding_fraction : float, default 0.1
        Padding around each system's bbox.

    Returns
    -------
    positions : (total_atoms, 3) float
        Same dtype/device as input; shifted per-system so each system's
        min corner is at the origin.
    cell : (num_systems, 3, 3) float
        Per-system orthorhombic cells.
    pbc : (num_systems, 3) bool
        All False — synthesized cells are non-periodic.
    """
    if batch_ptr.shape[0] < 2:
        raise ValueError("batch_ptr must have length at least 2")
    num_systems = int(batch_ptr.shape[0]) - 1
    if positions.shape[0] == 0:
        cell = (
            torch.eye(3, dtype=positions.dtype, device=positions.device)
            .reshape(1, 3, 3)
            .expand(num_systems, -1, -1)
            .contiguous()
        )
        pbc = torch.zeros((num_systems, 3), dtype=torch.bool, device=positions.device)
        return positions, cell, pbc
    expanded_idx = batch_idx.unsqueeze(1).expand_as(positions)
    pos_min = torch.full(
        (num_systems, 3),
        float("inf"),
        dtype=positions.dtype,
        device=positions.device,
    )
    pos_min.scatter_reduce_(0, expanded_idx, positions, reduce="amin")
    pos_max = torch.full(
        (num_systems, 3),
        float("-inf"),
        dtype=positions.dtype,
        device=positions.device,
    )
    pos_max.scatter_reduce_(0, expanded_idx, positions, reduce="amax")
    # TODO: switch to segment_ops once #17 is merged
    positions = positions - torch.index_select(pos_min, 0, batch_idx)
    cell_lengths = pos_max - pos_min + padding_fraction * cutoff
    cell = torch.diag_embed(cell_lengths)
    pbc = torch.zeros(
        (num_systems, 3),
        dtype=torch.bool,
        device=positions.device,
    )
    return positions, cell, pbc


def allocate_cell_list(
    total_atoms: int,
    max_total_cells: int,
    neighbor_search_radius: torch.Tensor,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Allocate memory tensors for cell list data structures.

    Parameters
    ----------
    total_atoms : int
        Total number of atoms across all systems.
    max_total_cells : int
        Maximum number of cells to allocate.
    neighbor_search_radius : torch.Tensor, shape (3,) or (num_systems, 3), dtype=int32
        Radius of neighboring cells to search in each dimension.
    device : torch.device
        Device on which to create tensors.

    Returns
    -------
    cells_per_dimension : torch.Tensor, shape (3,) or (num_systems, 3), dtype=int32
        Number of cells in x, y, z directions (to be filled by build_cell_list).
    neighbor_search_radius : torch.Tensor, shape (3,) or (num_systems, 3), dtype=int32
        Radius of neighboring cells to search (passed through for convenience).
    atom_periodic_shifts : torch.Tensor, shape (total_atoms, 3), dtype=int32
        Periodic boundary crossings for each atom (to be filled by build_cell_list).
    atom_to_cell_mapping : torch.Tensor, shape (total_atoms, 3), dtype=int32
        3D cell coordinates for each atom (to be filled by build_cell_list).
    atoms_per_cell_count : torch.Tensor, shape (max_total_cells,), dtype=int32
        Number of atoms in each cell (to be filled by build_cell_list).
    cell_atom_start_indices : torch.Tensor, shape (max_total_cells,), dtype=int32
        Starting index in cell_atom_list for each cell (to be filled by build_cell_list).
    cell_atom_list : torch.Tensor, shape (total_atoms,), dtype=int32
        Flattened list of atom indices organized by cell (to be filled by build_cell_list).

    Notes
    -----
    This is a pure PyTorch utility function with no warp dependencies. It pre-allocates
    all tensors needed for cell list construction, supporting both single-system and
    batched operations based on the shape of neighbor_search_radius.

    See Also
    --------
    nvalchemiops.neighbors.cell_list.build_cell_list : Warp launcher that uses these tensors
    nvalchemiops.torch.neighbors.cell_list.build_cell_list : High-level PyTorch wrapper
    nvalchemiops.torch.neighbors.batch_cell_list.batch_build_cell_list : Batched version
    """
    if max_total_cells < 0:
        raise ValueError(
            f"allocate_cell_list: max_total_cells={max_total_cells} < 0 "
            "(cell-count overflow or bad estimate)."
        )
    # Detect number of systems from neighbor_search_radius shape
    is_batched = neighbor_search_radius.ndim == 2
    num_systems = neighbor_search_radius.shape[0] if is_batched else 1
    cells_per_dimension = torch.zeros(
        (3,) if not is_batched else (num_systems, 3),
        dtype=torch.int32,
        device=device,
    )

    atom_periodic_shifts = torch.zeros(
        (total_atoms, 3), dtype=torch.int32, device=device
    )
    atom_to_cell_mapping = torch.zeros(
        (total_atoms, 3), dtype=torch.int32, device=device
    )
    atoms_per_cell_count = torch.zeros(
        (max_total_cells,), dtype=torch.int32, device=device
    )
    cell_atom_start_indices = torch.zeros(
        (max_total_cells,), dtype=torch.int32, device=device
    )
    cell_atom_list = torch.zeros((total_atoms,), dtype=torch.int32, device=device)
    return (
        cells_per_dimension,
        neighbor_search_radius,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
    )
