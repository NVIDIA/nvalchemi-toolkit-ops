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

"""Reusable storage for Torch cluster-tile neighbor-list execution."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from nvalchemiops.neighbors.cluster_tile import estimate_max_tiles_per_group
from nvalchemiops.neighbors.neighbor_utils import estimate_max_neighbors
from nvalchemiops.torch.neighbors.batch_cluster_tile import (
    TILE_GROUP_SIZE,
    _batch_cluster_tile_neighbor_list_normalized,
    _BatchPartitionMetadata,
    _prepare_batch_partition_metadata,
    allocate_batch_cluster_tile_list,
    estimate_batch_max_tiles_per_group,
)
from nvalchemiops.torch.neighbors.cluster_tile import (
    _cell_volume,
    allocate_cluster_tile_list,
    cluster_tile_neighbor_list,
)

__all__ = [
    "ClusterTileState",
    "prepare_cluster_tile",
    "cluster_tile_neighbor_list_prepared",
]


@dataclass(frozen=True, slots=True)
class ClusterTileState:
    """Fixed configuration and reusable storage for cluster-tile execution.

    Create a state with :func:`prepare_cluster_tile`. Configuration attributes
    cannot be reassigned. ``neighbor_vectors`` and ``neighbor_distances`` are
    borrowed, non-differentiable snapshot buffers that a later execution may
    overwrite. When matrix geometry requires autograd, execution returns fresh
    differentiable tensors and writes matching detached values to these
    buffers. Build losses from the returned geometry. Batched state caches only
    metadata derived from the fixed partition; geometry-dependent sorting and
    bounds are recomputed for every execution.
    """

    format: str
    is_batched: bool
    num_atoms: int
    num_systems: int
    device: torch.device
    dtype: torch.dtype
    cutoff: float
    cutoff2: float | None
    max_neighbors: int
    max_pairs: int | None
    fill_value: int
    return_vectors: bool
    return_distances: bool
    max_tiles_per_group: int
    _batch_ptr: torch.Tensor | None = field(repr=False)
    _partition_metadata: _BatchPartitionMetadata | None = field(repr=False)
    _cell_shape: tuple[int, ...] = field(repr=False)
    _scratch: tuple[torch.Tensor, ...] = field(repr=False)
    _topology: tuple[torch.Tensor, ...] = field(repr=False)
    _neighbor_vectors: torch.Tensor | None = field(repr=False)
    _neighbor_distances: torch.Tensor | None = field(repr=False)

    @property
    def neighbor_vectors(self) -> torch.Tensor | None:
        """Borrowed non-differentiable vector snapshot, if configured."""
        return self._neighbor_vectors

    @property
    def neighbor_distances(self) -> torch.Tensor | None:
        """Borrowed non-differentiable distance snapshot, if configured."""
        return self._neighbor_distances


def _positive_int(name: str, value: int | None) -> int | None:
    """Validate an optional positive integer."""
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_positions(positions: torch.Tensor) -> None:
    """Validate positions accepted by cluster-tile preparation."""
    if positions.dtype != torch.float32:
        raise TypeError("positions must be float32")
    if not positions.is_cuda or positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must be a CUDA tensor with shape (N, 3)")


def _prepare_batch_ptr(
    batch_ptr: torch.Tensor | None,
    positions: torch.Tensor,
) -> torch.Tensor | None:
    """Validate and copy a fixed batch partition."""
    if batch_ptr is None:
        return None
    if (
        batch_ptr.dtype != torch.int32
        or batch_ptr.device != positions.device
        or batch_ptr.ndim != 1
        or batch_ptr.numel() < 2
    ):
        raise ValueError("batch_ptr must be a CUDA int32 tensor with shape (B + 1,)")
    values = batch_ptr.detach().cpu().tolist()
    if (
        values[0] != 0
        or values[-1] != positions.shape[0]
        or any(stop < start for start, stop in zip(values, values[1:]))
    ):
        raise ValueError("batch_ptr must start at 0, be monotone, and end at N")
    return batch_ptr.detach().clone().contiguous()


def _validate_cell(
    cell: torch.Tensor,
    positions: torch.Tensor,
    batch_ptr: torch.Tensor | None,
) -> None:
    """Validate the fixed cell representation."""
    if cell.dtype != positions.dtype or cell.device != positions.device:
        raise ValueError("cell must match positions dtype and device")
    if batch_ptr is None:
        if cell.shape not in ((3, 3), (1, 3, 3)):
            raise ValueError("single-system cell must have shape (3, 3) or (1, 3, 3)")
    elif cell.shape != (batch_ptr.numel() - 1, 3, 3):
        raise ValueError("batched cell must have shape (B, 3, 3)")


def _allocate_matrix_topology(
    num_atoms: int,
    max_neighbors: int,
    device: torch.device,
    dual_cutoff: bool,
) -> tuple[torch.Tensor, ...]:
    """Allocate one or two matrix topology triples."""

    def allocate_one() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.empty((num_atoms, max_neighbors), dtype=torch.int32, device=device),
            torch.zeros(num_atoms, dtype=torch.int32, device=device),
            torch.empty(
                (num_atoms, max_neighbors, 3), dtype=torch.int32, device=device
            ),
        )

    primary = allocate_one()
    return (*primary, *allocate_one()) if dual_cutoff else primary


def prepare_cluster_tile(
    positions: torch.Tensor,
    cutoff: float,
    cell: torch.Tensor,
    *,
    format: str,
    batch_ptr: torch.Tensor | None = None,
    max_neighbors: int | None = None,
    fill_value: int | None = None,
    max_pairs: int | None = None,
    cutoff2: float | None = None,
    return_vectors: bool = False,
    return_distances: bool = False,
    max_tiles_per_group: int | None = None,
) -> ClusterTileState:
    """Allocate reusable storage for a fixed cluster-tile configuration.

    Parameters
    ----------
    positions : torch.Tensor, shape (N, 3), dtype=float32, CUDA
        Coordinates used to fix the atom count, dtype, and device.
    cutoff : float
        Positive primary cutoff.
    cell : torch.Tensor
        Single ``(3, 3)`` or ``(1, 3, 3)`` cell, or batched ``(B, 3, 3)``
        cells when ``batch_ptr`` is provided.
    format : {"tile", "matrix", "coo"}
        Fixed output representation.
    batch_ptr : torch.Tensor, optional
        Int32 cumulative atom counts. Preparation copies the partition.
    max_neighbors : int, optional
        Matrix row capacity. Also determines the default COO capacity.
    fill_value : int, optional
        Matrix padding value. Defaults to the atom count.
    max_pairs : int, optional
        Exact COO capacity. Defaults to ``N * max_neighbors``.
    cutoff2 : float, optional
        Secondary cutoff for matrix topology output. Dual-cutoff geometry is
        not supported.
    return_vectors, return_distances : bool, default=False
        Allocate reusable geometry buffers.
    max_tiles_per_group : int, optional
        Tile-pair capacity per row group. When omitted, preparation estimates
        it from the supplied geometry.

    Returns
    -------
    ClusterTileState
        Frozen configuration with private reusable topology and scratch.

    Notes
    -----
    Preparation allocates storage but does not build a neighbor list. Capture
    the returned state as a closure constant for ``torch.compile``.
    """
    _validate_positions(positions)
    if format not in ("tile", "matrix", "coo"):
        raise ValueError(f"format must be 'matrix' | 'coo' | 'tile'; got {format!r}")
    if not isinstance(cutoff, (int, float)) or isinstance(cutoff, bool) or cutoff <= 0:
        raise ValueError("cutoff must be positive")
    if cutoff2 is not None and (
        not isinstance(cutoff2, (int, float))
        or isinstance(cutoff2, bool)
        or cutoff2 <= 0
    ):
        raise ValueError("cutoff2 must be positive")
    if cutoff2 is not None and format != "matrix":
        raise ValueError("cutoff2 is supported only with format='matrix'")
    if cutoff2 is not None and (return_vectors or return_distances):
        raise ValueError(
            "cutoff2 cannot be combined with return_vectors or return_distances"
        )
    if format == "tile" and (return_vectors or return_distances):
        raise ValueError("tile output does not support vectors or distances")

    max_neighbors = _positive_int("max_neighbors", max_neighbors)
    max_pairs = _positive_int("max_pairs", max_pairs)
    max_tiles_per_group = _positive_int("max_tiles_per_group", max_tiles_per_group)
    protected_batch_ptr = _prepare_batch_ptr(batch_ptr, positions)
    _validate_cell(cell, positions, protected_batch_ptr)

    num_atoms = positions.shape[0]
    if max_neighbors is None:
        max_neighbors = max(
            estimate_max_neighbors(
                cutoff if cutoff2 is None else max(float(cutoff), float(cutoff2))
            ),
            TILE_GROUP_SIZE,
        )
    if fill_value is None:
        fill_value = num_atoms
    elif not isinstance(fill_value, int) or isinstance(fill_value, bool):
        raise ValueError("fill_value must be an integer")

    build_cutoff = max(float(cutoff), float(cutoff2 or cutoff))
    if max_tiles_per_group is None:
        if protected_batch_ptr is None:
            max_tiles_per_group = estimate_max_tiles_per_group(
                num_atoms, build_cutoff, _cell_volume(cell)
            )
        else:
            max_tiles_per_group = estimate_batch_max_tiles_per_group(
                protected_batch_ptr, build_cutoff, cell
            )

    if protected_batch_ptr is None:
        scratch = allocate_cluster_tile_list(
            num_atoms,
            positions.device,
            dtype=positions.dtype,
            max_tiles_per_group=max_tiles_per_group,
        )
    else:
        scratch = allocate_batch_cluster_tile_list(
            protected_batch_ptr,
            positions.device,
            dtype=positions.dtype,
            max_tiles_per_group=max_tiles_per_group,
        )
    partition_metadata = None
    if protected_batch_ptr is not None:
        partition_metadata = _prepare_batch_partition_metadata(
            protected_batch_ptr,
            num_atoms=num_atoms,
            padded_slot_system=scratch[5],
            batch_ptr_padded=scratch[6],
            group_system=scratch[7],
            group_ptr=scratch[8],
        )

    topology: tuple[torch.Tensor, ...] = ()
    if format == "matrix":
        topology = _allocate_matrix_topology(
            num_atoms,
            max_neighbors,
            positions.device,
            cutoff2 is not None,
        )
    elif format == "coo":
        if max_pairs is None:
            max_pairs = num_atoms * max_neighbors
        topology = (
            torch.empty((2, max_pairs), dtype=torch.int32, device=positions.device),
            torch.empty((max_pairs, 3), dtype=torch.int32, device=positions.device),
            torch.zeros(1, dtype=torch.int32, device=positions.device),
        )

    geometry_shape = (
        (num_atoms, max_neighbors) if format == "matrix" else (max_pairs or 0,)
    )
    neighbor_vectors = (
        torch.empty(
            (*geometry_shape, 3), dtype=positions.dtype, device=positions.device
        )
        if return_vectors
        else None
    )
    neighbor_distances = (
        torch.empty(geometry_shape, dtype=positions.dtype, device=positions.device)
        if return_distances
        else None
    )
    return ClusterTileState(
        format=format,
        is_batched=protected_batch_ptr is not None,
        num_atoms=num_atoms,
        num_systems=(
            int(protected_batch_ptr.numel() - 1)
            if protected_batch_ptr is not None
            else 1
        ),
        device=positions.device,
        dtype=positions.dtype,
        cutoff=float(cutoff),
        cutoff2=float(cutoff2) if cutoff2 is not None else None,
        max_neighbors=max_neighbors,
        max_pairs=max_pairs,
        fill_value=fill_value,
        return_vectors=bool(return_vectors),
        return_distances=bool(return_distances),
        max_tiles_per_group=max_tiles_per_group,
        _batch_ptr=protected_batch_ptr,
        _partition_metadata=partition_metadata,
        _cell_shape=tuple(cell.shape),
        _scratch=scratch,
        _topology=topology,
        _neighbor_vectors=neighbor_vectors,
        _neighbor_distances=neighbor_distances,
    )


def cluster_tile_neighbor_list_prepared(
    positions: torch.Tensor,
    cell: torch.Tensor,
    state: ClusterTileState,
) -> tuple[torch.Tensor, ...]:
    """Execute a previously prepared cluster-tile configuration.

    Parameters
    ----------
    positions : torch.Tensor
        Coordinates with the prepared shape, dtype, and device.
    cell : torch.Tensor
        Cell with the prepared shape, dtype, and device.
    state : ClusterTileState
        State returned by :func:`prepare_cluster_tile`. Capture it as a closure
        constant rather than passing it as a compiled graph input.

    Returns
    -------
    tuple of torch.Tensor
        The tuple returned by the matching direct cluster-tile function.

    Notes
    -----
    Returned matrix topology and tile tensors borrow state-owned storage. When
    autograd reconstruction is needed, matrix geometry is returned as fresh
    differentiable tensors and state-owned geometry buffers receive matching
    detached snapshots. Without reconstruction, returned matrix geometry
    aliases those buffers. Build losses from the returned geometry, not from
    the snapshot buffers. Exact COO tensors are exact-sized per call. Finish
    backward or copy any values that must survive before reusing ``state``.
    """
    if not isinstance(state, ClusterTileState):
        raise TypeError("state must be a ClusterTileState")
    if positions.shape != (state.num_atoms, 3):
        raise ValueError("positions shape does not match prepared state")
    if positions.dtype != state.dtype:
        raise TypeError("positions dtype does not match prepared state")
    if positions.device != state.device:
        raise ValueError("positions device does not match prepared state")
    if tuple(cell.shape) != state._cell_shape:
        raise ValueError("cell shape does not match prepared state")
    if cell.dtype != state.dtype:
        raise TypeError("cell dtype does not match prepared state")
    if cell.device != state.device:
        raise ValueError("cell device does not match prepared state")

    topology = state._topology
    matrix_kwargs: dict[str, torch.Tensor] = {}
    coo_kwargs: dict[str, torch.Tensor] = {}
    if state.format == "matrix":
        matrix_kwargs = {
            "neighbor_matrix": topology[0],
            "num_neighbors": topology[1],
            "neighbor_matrix_shifts": topology[2],
        }
        if state.cutoff2 is not None:
            matrix_kwargs.update(
                neighbor_matrix2=topology[3],
                num_neighbors2=topology[4],
                neighbor_matrix_shifts2=topology[5],
            )
    elif state.format == "coo":
        coo_kwargs = {
            "neighbor_list": topology[0],
            "neighbor_list_shifts": topology[1],
            "pair_counter": topology[2],
        }

    if state.is_batched:
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
        ) = state._scratch
        return _batch_cluster_tile_neighbor_list_normalized(
            state._partition_metadata,
            positions,
            state.cutoff,
            cell,
            state._batch_ptr,
            format=state.format,
            max_neighbors=state.max_neighbors,
            fill_value=state.fill_value,
            max_pairs=state.max_pairs,
            cutoff2=state.cutoff2,
            return_vectors=state.return_vectors,
            return_distances=state.return_distances,
            neighbor_vectors=state.neighbor_vectors,
            neighbor_distances=state.neighbor_distances,
            max_tiles_per_group=state.max_tiles_per_group,
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
            **matrix_kwargs,
            **coo_kwargs,
        )

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
    ) = state._scratch
    return cluster_tile_neighbor_list(
        positions,
        state.cutoff,
        cell,
        format=state.format,
        max_neighbors=state.max_neighbors,
        fill_value=state.fill_value,
        max_pairs=state.max_pairs,
        cutoff2=state.cutoff2,
        return_vectors=state.return_vectors,
        return_distances=state.return_distances,
        neighbor_vectors=state.neighbor_vectors,
        neighbor_distances=state.neighbor_distances,
        max_tiles_per_group=state.max_tiles_per_group,
        sorted_atom_index=sorted_atom_index,
        morton_codes=morton_codes,
        sorted_pos_x=sorted_pos_x,
        sorted_pos_y=sorted_pos_y,
        sorted_pos_z=sorted_pos_z,
        group_ctr_x=group_ctr_x,
        group_ctr_y=group_ctr_y,
        group_ctr_z=group_ctr_z,
        group_ext_x=group_ext_x,
        group_ext_y=group_ext_y,
        group_ext_z=group_ext_z,
        num_tiles=num_tiles,
        tile_row_group=tile_row_group,
        tile_col_group=tile_col_group,
        **matrix_kwargs,
        **coo_kwargs,
    )
