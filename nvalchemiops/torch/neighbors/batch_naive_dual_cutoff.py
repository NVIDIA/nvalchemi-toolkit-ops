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

"""PyTorch bindings for batched naive dual cutoff neighbor list construction."""

from __future__ import annotations

import torch
import warp as wp

from nvalchemiops.neighbors.naive import (
    batch_naive_neighbor_matrix_dual_cutoff,
    batch_naive_neighbor_matrix_pbc_dual_cutoff,
)
from nvalchemiops.neighbors.neighbor_utils import (
    estimate_max_neighbors,
)
from nvalchemiops.torch._warp_op_helpers import register_noop_fake
from nvalchemiops.torch.neighbors.neighbor_utils import (
    compute_naive_num_shifts,
    get_neighbor_list_from_neighbor_matrix,
    prepare_batch_idx_ptr,
)
from nvalchemiops.torch.types import get_wp_dtype, get_wp_mat_dtype, get_wp_vec_dtype

__all__ = ["batch_naive_neighbor_list_dual_cutoff"]


@torch.library.custom_op(
    "nvalchemiops::_batch_naive_neighbor_matrix_no_pbc_dual_cutoff",
    mutates_args=(
        "neighbor_matrix1",
        "num_neighbors1",
        "neighbor_matrix2",
        "num_neighbors2",
    ),
)
def _batch_naive_neighbor_matrix_no_pbc_dual_cutoff(
    positions: torch.Tensor,
    cutoff1: float,
    cutoff2: float,
    batch_idx: torch.Tensor,
    batch_ptr: torch.Tensor,
    neighbor_matrix1: torch.Tensor,
    num_neighbors1: torch.Tensor,
    neighbor_matrix2: torch.Tensor,
    num_neighbors2: torch.Tensor,
    half_fill: bool,
) -> None:
    """Fill two neighbor matrices for batch using dual cutoffs with naive O(N^2) algorithm.

    This function is torch compilable.

    See Also
    --------
    nvalchemiops.neighbors.batch_naive_dual_cutoff.batch_naive_neighbor_matrix_dual_cutoff : Core warp launcher
    batch_naive_neighbor_list_dual_cutoff : High-level wrapper function
    """
    device = positions.device
    wp_vec_dtype = get_wp_vec_dtype(positions.dtype)
    wp_dtype = get_wp_dtype(positions.dtype)

    wp_positions = wp.from_torch(
        positions, dtype=wp_vec_dtype, requires_grad=False, return_ctype=True
    )
    wp_batch_idx = wp.from_torch(
        batch_idx, dtype=wp.int32, requires_grad=False, return_ctype=True
    )
    wp_batch_ptr = wp.from_torch(
        batch_ptr, dtype=wp.int32, requires_grad=False, return_ctype=True
    )
    wp_neighbor_matrix1 = wp.from_torch(
        neighbor_matrix1, dtype=wp.int32, requires_grad=False, return_ctype=True
    )
    wp_num_neighbors1 = wp.from_torch(
        num_neighbors1, dtype=wp.int32, requires_grad=False, return_ctype=True
    )
    wp_neighbor_matrix2 = wp.from_torch(
        neighbor_matrix2, dtype=wp.int32, requires_grad=False, return_ctype=True
    )
    wp_num_neighbors2 = wp.from_torch(
        num_neighbors2, dtype=wp.int32, requires_grad=False, return_ctype=True
    )

    batch_naive_neighbor_matrix_dual_cutoff(
        positions=wp_positions,
        cutoff1=cutoff1,
        cutoff2=cutoff2,
        batch_idx=wp_batch_idx,
        batch_ptr=wp_batch_ptr,
        neighbor_matrix1=wp_neighbor_matrix1,
        num_neighbors1=wp_num_neighbors1,
        neighbor_matrix2=wp_neighbor_matrix2,
        num_neighbors2=wp_num_neighbors2,
        wp_dtype=wp_dtype,
        device=str(device),
        half_fill=half_fill,
    )


@torch.library.custom_op(
    "nvalchemiops::_batch_naive_neighbor_matrix_pbc_dual_cutoff",
    mutates_args=(
        "neighbor_matrix1",
        "neighbor_matrix2",
        "neighbor_matrix_shifts1",
        "neighbor_matrix_shifts2",
        "num_neighbors1",
        "num_neighbors2",
    ),
)
def _batch_naive_neighbor_matrix_pbc_dual_cutoff(
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cutoff1: float,
    cutoff2: float,
    batch_idx: torch.Tensor,
    batch_ptr: torch.Tensor,
    neighbor_matrix1: torch.Tensor,
    neighbor_matrix2: torch.Tensor,
    neighbor_matrix_shifts1: torch.Tensor,
    neighbor_matrix_shifts2: torch.Tensor,
    num_neighbors1: torch.Tensor,
    num_neighbors2: torch.Tensor,
    shift_range_per_dimension: torch.Tensor,
    num_shifts_per_system: torch.Tensor,
    max_shifts_per_system: int,
    half_fill: bool = False,
    max_atoms_per_system: int | None = None,
    wrap_positions: bool = True,
    positions_wrapped_buffer: torch.Tensor | None = None,
    per_atom_cell_offsets_buffer: torch.Tensor | None = None,
    inv_cell_buffer: torch.Tensor | None = None,
) -> None:
    """Compute batch neighbor matrices with PBC using dual cutoffs.

    This function is torch compilable.

    See Also
    --------
    nvalchemiops.neighbors.batch_naive_dual_cutoff.batch_naive_neighbor_matrix_pbc_dual_cutoff : Core warp launcher
    batch_naive_neighbor_list_dual_cutoff : High-level wrapper function
    """
    device = positions.device
    wp_vec_dtype = get_wp_vec_dtype(positions.dtype)
    wp_mat_dtype = get_wp_mat_dtype(positions.dtype)
    wp_dtype = get_wp_dtype(positions.dtype)

    wp_positions = wp.from_torch(
        positions, dtype=wp_vec_dtype, requires_grad=False, return_ctype=True
    )
    wp_cell = wp.from_torch(
        cell, dtype=wp_mat_dtype, requires_grad=False, return_ctype=True
    )
    wp_pbc = wp.from_torch(pbc, dtype=wp.bool, requires_grad=False, return_ctype=True)
    wp_shift_range = wp.from_torch(
        shift_range_per_dimension,
        dtype=wp.vec3i,
        requires_grad=False,
        return_ctype=True,
    )
    wp_num_shifts_arr = wp.from_torch(
        num_shifts_per_system, dtype=wp.int32, requires_grad=False, return_ctype=True
    )
    wp_batch_idx = wp.from_torch(
        batch_idx, dtype=wp.int32, requires_grad=False, return_ctype=True
    )
    wp_batch_ptr = wp.from_torch(
        batch_ptr, dtype=wp.int32, requires_grad=False, return_ctype=True
    )
    wp_neighbor_matrix1 = wp.from_torch(
        neighbor_matrix1, dtype=wp.int32, requires_grad=False, return_ctype=True
    )
    wp_neighbor_matrix2 = wp.from_torch(
        neighbor_matrix2, dtype=wp.int32, requires_grad=False, return_ctype=True
    )
    wp_neighbor_matrix_shifts1 = wp.from_torch(
        neighbor_matrix_shifts1, dtype=wp.vec3i, requires_grad=False, return_ctype=True
    )
    wp_neighbor_matrix_shifts2 = wp.from_torch(
        neighbor_matrix_shifts2, dtype=wp.vec3i, requires_grad=False, return_ctype=True
    )
    wp_num_neighbors1 = wp.from_torch(
        num_neighbors1, dtype=wp.int32, requires_grad=False, return_ctype=True
    )
    wp_num_neighbors2 = wp.from_torch(
        num_neighbors2, dtype=wp.int32, requires_grad=False, return_ctype=True
    )

    if max_atoms_per_system is None:
        max_atoms_per_system = (batch_ptr[1:] - batch_ptr[:-1]).max().item()

    wp_positions_wrapped = (
        wp.from_torch(
            positions_wrapped_buffer,
            dtype=wp_vec_dtype,
            requires_grad=False,
            return_ctype=True,
        )
        if positions_wrapped_buffer is not None
        else None
    )
    wp_per_atom_cell_offsets = (
        wp.from_torch(
            per_atom_cell_offsets_buffer,
            dtype=wp.vec3i,
            requires_grad=False,
            return_ctype=True,
        )
        if per_atom_cell_offsets_buffer is not None
        else None
    )
    wp_inv_cell = (
        wp.from_torch(
            inv_cell_buffer, dtype=wp_mat_dtype, requires_grad=False, return_ctype=True
        )
        if inv_cell_buffer is not None
        else None
    )

    batch_naive_neighbor_matrix_pbc_dual_cutoff(
        positions=wp_positions,
        cell=wp_cell,
        pbc=wp_pbc,
        cutoff1=cutoff1,
        cutoff2=cutoff2,
        batch_ptr=wp_batch_ptr,
        batch_idx=wp_batch_idx,
        shift_range=wp_shift_range,
        num_shifts_arr=wp_num_shifts_arr,
        max_shifts_per_system=max_shifts_per_system,
        neighbor_matrix1=wp_neighbor_matrix1,
        neighbor_matrix2=wp_neighbor_matrix2,
        neighbor_matrix_shifts1=wp_neighbor_matrix_shifts1,
        neighbor_matrix_shifts2=wp_neighbor_matrix_shifts2,
        num_neighbors1=wp_num_neighbors1,
        num_neighbors2=wp_num_neighbors2,
        wp_dtype=wp_dtype,
        device=str(device),
        max_atoms_per_system=max_atoms_per_system,
        half_fill=half_fill,
        wrap_positions=wrap_positions,
        positions_wrapped_buffer=wp_positions_wrapped,
        per_atom_cell_offsets_buffer=wp_per_atom_cell_offsets,
        inv_cell_buffer=wp_inv_cell,
    )


@torch.library.custom_op(
    "nvalchemiops::_batch_naive_neighbor_matrix_no_pbc_dual_cutoff_selective",
    mutates_args=(
        "neighbor_matrix1",
        "num_neighbors1",
        "neighbor_matrix2",
        "num_neighbors2",
    ),
)
def _batch_naive_neighbor_matrix_no_pbc_dual_cutoff_selective(
    positions: torch.Tensor,
    cutoff1: float,
    cutoff2: float,
    batch_idx: torch.Tensor,
    batch_ptr: torch.Tensor,
    neighbor_matrix1: torch.Tensor,
    num_neighbors1: torch.Tensor,
    neighbor_matrix2: torch.Tensor,
    num_neighbors2: torch.Tensor,
    rebuild_flags: torch.Tensor,
    half_fill: bool = False,
) -> None:
    """Selective batched naive dual cutoff neighbor matrix custom op (no PBC).

    Wraps the GPU-side selective kernel: per-system rebuild_flags checked on the
    device — no CPU-GPU synchronisation occurs.

    See Also
    --------
    nvalchemiops.neighbors.batch_naive_dual_cutoff.batch_naive_neighbor_matrix_dual_cutoff : Core warp launcher
    batch_naive_neighbor_list_dual_cutoff : High-level wrapper that dispatches here when rebuild_flags is set
    """
    device = positions.device
    wp_device = wp.device_from_torch(device)
    wp_vec_dtype = get_wp_vec_dtype(positions.dtype)
    wp_dtype = get_wp_dtype(positions.dtype)

    wp_positions = wp.from_torch(
        positions, dtype=wp_vec_dtype, requires_grad=False, return_ctype=True
    )
    wp_batch_idx = wp.from_torch(
        batch_idx, dtype=wp.int32, requires_grad=False, return_ctype=True
    )
    wp_batch_ptr = wp.from_torch(
        batch_ptr, dtype=wp.int32, requires_grad=False, return_ctype=True
    )
    wp_neighbor_matrix1 = wp.from_torch(
        neighbor_matrix1, dtype=wp.int32, requires_grad=False, return_ctype=True
    )
    wp_num_neighbors1 = wp.from_torch(
        num_neighbors1, dtype=wp.int32, requires_grad=False, return_ctype=True
    )
    wp_neighbor_matrix2 = wp.from_torch(
        neighbor_matrix2, dtype=wp.int32, requires_grad=False, return_ctype=True
    )
    wp_num_neighbors2 = wp.from_torch(
        num_neighbors2, dtype=wp.int32, requires_grad=False, return_ctype=True
    )
    wp_rebuild_flags = wp.from_torch(
        rebuild_flags, dtype=wp.bool, requires_grad=False, return_ctype=True
    )

    batch_naive_neighbor_matrix_dual_cutoff(
        positions=wp_positions,
        cutoff1=cutoff1,
        cutoff2=cutoff2,
        batch_idx=wp_batch_idx,
        batch_ptr=wp_batch_ptr,
        neighbor_matrix1=wp_neighbor_matrix1,
        num_neighbors1=wp_num_neighbors1,
        neighbor_matrix2=wp_neighbor_matrix2,
        num_neighbors2=wp_num_neighbors2,
        wp_dtype=wp_dtype,
        device=str(wp_device),
        half_fill=half_fill,
        rebuild_flags=wp_rebuild_flags,
    )


@torch.library.custom_op(
    "nvalchemiops::_batch_naive_neighbor_matrix_pbc_dual_cutoff_selective",
    mutates_args=(
        "neighbor_matrix1",
        "neighbor_matrix2",
        "neighbor_matrix_shifts1",
        "neighbor_matrix_shifts2",
        "num_neighbors1",
        "num_neighbors2",
    ),
)
def _batch_naive_neighbor_matrix_pbc_dual_cutoff_selective(
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cutoff1: float,
    cutoff2: float,
    batch_idx: torch.Tensor,
    batch_ptr: torch.Tensor,
    neighbor_matrix1: torch.Tensor,
    neighbor_matrix2: torch.Tensor,
    neighbor_matrix_shifts1: torch.Tensor,
    neighbor_matrix_shifts2: torch.Tensor,
    num_neighbors1: torch.Tensor,
    num_neighbors2: torch.Tensor,
    shift_range_per_dimension: torch.Tensor,
    num_shifts_per_system: torch.Tensor,
    max_shifts_per_system: int,
    rebuild_flags: torch.Tensor,
    half_fill: bool = False,
    max_atoms_per_system: int | None = None,
    wrap_positions: bool = True,
    positions_wrapped_buffer: torch.Tensor | None = None,
    per_atom_cell_offsets_buffer: torch.Tensor | None = None,
    inv_cell_buffer: torch.Tensor | None = None,
) -> None:
    """Selective batched naive dual cutoff PBC neighbor matrix custom op.

    Per-system rebuild_flags are checked on the device — no CPU-GPU
    synchronisation occurs.

    See Also
    --------
    nvalchemiops.neighbors.batch_naive_dual_cutoff.batch_naive_neighbor_matrix_pbc_dual_cutoff : Core warp launcher
    batch_naive_neighbor_list_dual_cutoff : High-level wrapper that dispatches here when rebuild_flags is set
    """
    device = positions.device
    wp_device = wp.device_from_torch(device)
    wp_vec_dtype = get_wp_vec_dtype(positions.dtype)
    wp_mat_dtype = get_wp_mat_dtype(positions.dtype)
    wp_dtype = get_wp_dtype(positions.dtype)

    wp_positions = wp.from_torch(
        positions, dtype=wp_vec_dtype, requires_grad=False, return_ctype=True
    )
    wp_cell = wp.from_torch(
        cell, dtype=wp_mat_dtype, requires_grad=False, return_ctype=True
    )
    wp_pbc = wp.from_torch(pbc, dtype=wp.bool, requires_grad=False, return_ctype=True)
    wp_shift_range = wp.from_torch(
        shift_range_per_dimension,
        dtype=wp.vec3i,
        requires_grad=False,
        return_ctype=True,
    )
    wp_num_shifts_arr = wp.from_torch(
        num_shifts_per_system, dtype=wp.int32, requires_grad=False, return_ctype=True
    )
    wp_batch_idx = wp.from_torch(
        batch_idx, dtype=wp.int32, requires_grad=False, return_ctype=True
    )
    wp_batch_ptr = wp.from_torch(
        batch_ptr, dtype=wp.int32, requires_grad=False, return_ctype=True
    )
    wp_neighbor_matrix1 = wp.from_torch(
        neighbor_matrix1, dtype=wp.int32, requires_grad=False, return_ctype=True
    )
    wp_neighbor_matrix2 = wp.from_torch(
        neighbor_matrix2, dtype=wp.int32, requires_grad=False, return_ctype=True
    )
    wp_neighbor_matrix_shifts1 = wp.from_torch(
        neighbor_matrix_shifts1, dtype=wp.vec3i, requires_grad=False, return_ctype=True
    )
    wp_neighbor_matrix_shifts2 = wp.from_torch(
        neighbor_matrix_shifts2, dtype=wp.vec3i, requires_grad=False, return_ctype=True
    )
    wp_num_neighbors1 = wp.from_torch(
        num_neighbors1, dtype=wp.int32, requires_grad=False, return_ctype=True
    )
    wp_num_neighbors2 = wp.from_torch(
        num_neighbors2, dtype=wp.int32, requires_grad=False, return_ctype=True
    )
    wp_rebuild_flags = wp.from_torch(
        rebuild_flags, dtype=wp.bool, requires_grad=False, return_ctype=True
    )
    wp_positions_wrapped = (
        wp.from_torch(
            positions_wrapped_buffer,
            dtype=wp_vec_dtype,
            requires_grad=False,
            return_ctype=True,
        )
        if positions_wrapped_buffer is not None
        else None
    )
    wp_per_atom_cell_offsets = (
        wp.from_torch(
            per_atom_cell_offsets_buffer,
            dtype=wp.vec3i,
            requires_grad=False,
            return_ctype=True,
        )
        if per_atom_cell_offsets_buffer is not None
        else None
    )
    wp_inv_cell = (
        wp.from_torch(
            inv_cell_buffer, dtype=wp_mat_dtype, requires_grad=False, return_ctype=True
        )
        if inv_cell_buffer is not None
        else None
    )

    if max_atoms_per_system is None:
        max_atoms_per_system = (batch_ptr[1:] - batch_ptr[:-1]).max().item()

    batch_naive_neighbor_matrix_pbc_dual_cutoff(
        positions=wp_positions,
        cell=wp_cell,
        pbc=wp_pbc,
        cutoff1=cutoff1,
        cutoff2=cutoff2,
        batch_ptr=wp_batch_ptr,
        batch_idx=wp_batch_idx,
        shift_range=wp_shift_range,
        num_shifts_arr=wp_num_shifts_arr,
        max_shifts_per_system=max_shifts_per_system,
        neighbor_matrix1=wp_neighbor_matrix1,
        neighbor_matrix2=wp_neighbor_matrix2,
        neighbor_matrix_shifts1=wp_neighbor_matrix_shifts1,
        neighbor_matrix_shifts2=wp_neighbor_matrix_shifts2,
        num_neighbors1=wp_num_neighbors1,
        num_neighbors2=wp_num_neighbors2,
        wp_dtype=wp_dtype,
        device=str(wp_device),
        max_atoms_per_system=max_atoms_per_system,
        half_fill=half_fill,
        rebuild_flags=wp_rebuild_flags,
        wrap_positions=wrap_positions,
        positions_wrapped_buffer=wp_positions_wrapped,
        per_atom_cell_offsets_buffer=wp_per_atom_cell_offsets,
        inv_cell_buffer=wp_inv_cell,
    )


register_noop_fake(_batch_naive_neighbor_matrix_no_pbc_dual_cutoff)
register_noop_fake(_batch_naive_neighbor_matrix_pbc_dual_cutoff)
register_noop_fake(_batch_naive_neighbor_matrix_no_pbc_dual_cutoff_selective)
register_noop_fake(_batch_naive_neighbor_matrix_pbc_dual_cutoff_selective)


def batch_naive_neighbor_list_dual_cutoff(
    positions: torch.Tensor,
    cutoff1: float,
    cutoff2: float,
    batch_idx: torch.Tensor | None = None,
    batch_ptr: torch.Tensor | None = None,
    pbc: torch.Tensor | None = None,
    cell: torch.Tensor | None = None,
    max_neighbors1: int | None = None,
    max_neighbors2: int | None = None,
    half_fill: bool = False,
    fill_value: int | None = None,
    return_neighbor_list: bool = False,
    neighbor_matrix1: torch.Tensor | None = None,
    neighbor_matrix2: torch.Tensor | None = None,
    neighbor_matrix_shifts1: torch.Tensor | None = None,
    neighbor_matrix_shifts2: torch.Tensor | None = None,
    num_neighbors1: torch.Tensor | None = None,
    num_neighbors2: torch.Tensor | None = None,
    shift_range_per_dimension: torch.Tensor | None = None,
    num_shifts_per_system: torch.Tensor | None = None,
    max_shifts_per_system: int | None = None,
    max_atoms_per_system: int | None = None,
    rebuild_flags: torch.Tensor | None = None,
    wrap_positions: bool = True,
    positions_wrapped_buffer: torch.Tensor | None = None,
    per_atom_cell_offsets_buffer: torch.Tensor | None = None,
    inv_cell_buffer: torch.Tensor | None = None,
) -> (
    tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]
    | tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
):
    """Compute batch neighbor matrices using naive O(N^2) algorithm with dual cutoffs.

    Allocates or accepts pre-allocated neighbor matrices for two independent cutoff
    radii and fills them in a single GPU pass. Supports free-space and periodic
    boundary conditions, selective per-system rebuilds via ``rebuild_flags``, and
    optional conversion to COO neighbor-list format.

    Parameters
    ----------
    positions : torch.Tensor, shape (N, 3)
        Atomic positions in Cartesian space, where N is the total number of atoms
        across all systems in the batch.
    cutoff1 : float
        Neighbour search cutoff radius for the first neighbour list.
    cutoff2 : float
        Neighbour search cutoff radius for the second neighbour list. Must satisfy
        ``cutoff2 >= cutoff1`` for correct shift-range pre-computation.
    batch_idx : torch.Tensor, shape (N,), dtype=int32, optional
        System index per atom. Pass ``None`` for a single-system batch (all atoms
        belong to system 0).
    batch_ptr : torch.Tensor, shape (num_systems + 1,), dtype=int32, optional
        CSR-style row pointer for the batch; ``batch_ptr[i]:batch_ptr[i+1]`` gives
        the atom range for system ``i``. Derived from ``batch_idx`` when ``None``.
    pbc : torch.Tensor, shape (num_systems, 3) or (3,), dtype=bool, optional
        Periodic boundary flags per system and dimension. Pass ``None`` for
        free-space (no PBC). Must be provided together with ``cell``.
    cell : torch.Tensor, shape (num_systems, 3, 3) or (1, 3, 3), optional
        Unit-cell matrices; each row is a lattice vector in Cartesian coordinates.
        Must be provided together with ``pbc``.
    max_neighbors1 : int, optional
        Column width of ``neighbor_matrix1``. Estimated automatically when ``None``
        and no pre-allocated matrix is supplied.
    max_neighbors2 : int, optional
        Column width of ``neighbor_matrix2``. Defaults to ``max_neighbors1`` when
        ``None``.
    half_fill : bool, optional
        If ``True``, only the lower-triangular half of each neighbor matrix is
        filled (each pair recorded once). Default is ``False``.
    fill_value : int, optional
        Padding sentinel for unused slots in the neighbor matrices. Defaults to N
        (total atom count).
    return_neighbor_list : bool, optional
        If ``True``, convert the neighbor matrices to COO edge-list format
        ``(neighbor_list, neighbor_ptr)`` before returning. Default is ``False``.
    neighbor_matrix1 : torch.Tensor, shape (N, max_neighbors1), dtype=int32, optional
        Pre-allocated output buffer for cutoff1 neighbour indices. Modified in-place.
        Allocated internally when ``None``.
    neighbor_matrix2 : torch.Tensor, shape (N, max_neighbors2), dtype=int32, optional
        Pre-allocated output buffer for cutoff2 neighbour indices. Modified in-place.
        Allocated internally when ``None``.
    neighbor_matrix_shifts1 : torch.Tensor, shape (N, max_neighbors1, 3), dtype=int32, optional
        Pre-allocated PBC image shift vectors for cutoff1 neighbours. Modified in-place.
        Only used when ``pbc`` is not ``None``. Allocated internally when ``None``.
    neighbor_matrix_shifts2 : torch.Tensor, shape (N, max_neighbors2, 3), dtype=int32, optional
        Pre-allocated PBC image shift vectors for cutoff2 neighbours. Modified in-place.
        Only used when ``pbc`` is not ``None``. Allocated internally when ``None``.
    num_neighbors1 : torch.Tensor, shape (N,), dtype=int32, optional
        Pre-allocated atom-wise neighbour count for cutoff1. Modified in-place.
        Allocated internally when ``None``.
    num_neighbors2 : torch.Tensor, shape (N,), dtype=int32, optional
        Pre-allocated atom-wise neighbour count for cutoff2. Modified in-place.
        Allocated internally when ``None``.
    shift_range_per_dimension : torch.Tensor, shape (num_systems, 3), dtype=int32, optional
        Half-range of image shifts per system and dimension. Computed from ``cell``
        and ``cutoff2`` when ``None``.
    num_shifts_per_system : torch.Tensor, shape (num_systems,), dtype=int32, optional
        Total number of image shifts per system. Computed when ``None``.
    max_shifts_per_system : int, optional
        Maximum value across ``num_shifts_per_system``; used as a kernel launch
        bound. Computed when ``None``.
    max_atoms_per_system : int, optional
        Maximum number of atoms in any single system; used as a kernel launch
        bound. Computed from ``batch_ptr`` when ``None``.
    rebuild_flags : torch.Tensor, shape (num_systems,), dtype=bool, optional
        Per-system boolean flags. When provided, only systems with ``True`` are
        rebuilt; other systems retain their existing neighbour data. No CPU-GPU
        synchronisation occurs.
    wrap_positions : bool, optional
        If ``True`` (default), positions are wrapped into the primary unit cell
        before distance evaluation. Only relevant when ``pbc`` is not ``None``.
    positions_wrapped_buffer : torch.Tensor, shape (N, 3), optional
        Pre-allocated buffer for wrapped positions. Allocated internally when
        ``None`` and ``wrap_positions`` is ``True``.
    per_atom_cell_offsets_buffer : torch.Tensor, shape (N, 3), dtype=int32, optional
        Pre-allocated buffer for per-atom cell offsets used during wrapping.
        Allocated internally when ``None``.
    inv_cell_buffer : torch.Tensor, shape (num_systems, 3, 3), optional
        Pre-allocated buffer for inverse cell matrices. Allocated internally when
        ``None``.

    Returns
    -------
    No PBC, ``return_neighbor_list=False`` :
        tuple of (neighbor_matrix1, num_neighbors1, neighbor_matrix2, num_neighbors2)

        neighbor_matrix1 : torch.Tensor, shape (N, max_neighbors1), dtype=int32
            Neighbour indices for cutoff1; unused slots filled with ``fill_value``.
        num_neighbors1 : torch.Tensor, shape (N,), dtype=int32
            Number of valid neighbours per atom for cutoff1.
        neighbor_matrix2 : torch.Tensor, shape (N, max_neighbors2), dtype=int32
            Neighbour indices for cutoff2; unused slots filled with ``fill_value``.
        num_neighbors2 : torch.Tensor, shape (N,), dtype=int32
            Number of valid neighbours per atom for cutoff2.

    No PBC, ``return_neighbor_list=True`` :
        tuple of (neighbor_list1, neighbor_ptr1, neighbor_list2, neighbor_ptr2)

        neighbor_list1 : torch.Tensor, shape (E1,), dtype=int32
            COO target-atom indices for cutoff1 edges.
        neighbor_ptr1 : torch.Tensor, shape (N + 1,), dtype=int32
            CSR row pointer for cutoff1 edges.
        neighbor_list2 : torch.Tensor, shape (E2,), dtype=int32
            COO target-atom indices for cutoff2 edges.
        neighbor_ptr2 : torch.Tensor, shape (N + 1,), dtype=int32
            CSR row pointer for cutoff2 edges.

    With PBC, ``return_neighbor_list=False`` :
        tuple of (neighbor_matrix1, num_neighbors1, neighbor_matrix_shifts1,
        neighbor_matrix2, num_neighbors2, neighbor_matrix_shifts2)

        neighbor_matrix1 : torch.Tensor, shape (N, max_neighbors1), dtype=int32
            Neighbour indices for cutoff1.
        num_neighbors1 : torch.Tensor, shape (N,), dtype=int32
            Neighbour counts for cutoff1.
        neighbor_matrix_shifts1 : torch.Tensor, shape (N, max_neighbors1, 3), dtype=int32
            PBC image shift vectors for cutoff1 neighbours.
        neighbor_matrix2 : torch.Tensor, shape (N, max_neighbors2), dtype=int32
            Neighbour indices for cutoff2.
        num_neighbors2 : torch.Tensor, shape (N,), dtype=int32
            Neighbour counts for cutoff2.
        neighbor_matrix_shifts2 : torch.Tensor, shape (N, max_neighbors2, 3), dtype=int32
            PBC image shift vectors for cutoff2 neighbours.

    With PBC, ``return_neighbor_list=True`` :
        tuple of (neighbor_list1, neighbor_ptr1, unit_shifts1,
        neighbor_list2, neighbor_ptr2, unit_shifts2)

        neighbor_list1 : torch.Tensor, shape (E1,), dtype=int32
            COO target-atom indices for cutoff1 edges.
        neighbor_ptr1 : torch.Tensor, shape (N + 1,), dtype=int32
            CSR row pointer for cutoff1 edges.
        unit_shifts1 : torch.Tensor, shape (E1, 3), dtype=int32
            PBC image shift vectors for cutoff1 edges.
        neighbor_list2 : torch.Tensor, shape (E2,), dtype=int32
            COO target-atom indices for cutoff2 edges.
        neighbor_ptr2 : torch.Tensor, shape (N + 1,), dtype=int32
            CSR row pointer for cutoff2 edges.
        unit_shifts2 : torch.Tensor, shape (E2, 3), dtype=int32
            PBC image shift vectors for cutoff2 edges.

    See Also
    --------
    :func:`nvalchemiops.neighbors.batch_naive_dual_cutoff.batch_naive_neighbor_matrix_dual_cutoff` : Core warp launcher (no PBC).
    :func:`nvalchemiops.neighbors.batch_naive_dual_cutoff.batch_naive_neighbor_matrix_pbc_dual_cutoff` : Core warp launcher (with PBC).
    :func:`nvalchemiops.torch.neighbors.batch_naive.batch_naive_neighbor_list` : Single-cutoff variant.
    """
    if pbc is None and cell is not None:
        raise ValueError("If cell is provided, pbc must also be provided")
    if pbc is not None and cell is None:
        raise ValueError("If pbc is provided, cell must also be provided")

    if cell is not None:
        cell = cell if cell.ndim == 3 else cell.unsqueeze(0)
    if pbc is not None:
        pbc = pbc if pbc.ndim == 2 else pbc.unsqueeze(0)

    if fill_value is None:
        fill_value = positions.shape[0]

    if max_neighbors1 is None and (
        neighbor_matrix1 is None
        or neighbor_matrix2 is None
        or (neighbor_matrix_shifts1 is None and pbc is not None)
        or (neighbor_matrix_shifts2 is None and pbc is not None)
        or num_neighbors1 is None
        or num_neighbors2 is None
    ):
        max_neighbors2 = estimate_max_neighbors(cutoff2)
        max_neighbors1 = max_neighbors2

    if max_neighbors2 is None:
        max_neighbors2 = max_neighbors1

    total_atoms = positions.shape[0]

    if neighbor_matrix1 is None:
        neighbor_matrix1 = torch.full(
            (total_atoms, max_neighbors1),
            fill_value,
            dtype=torch.int32,
            device=positions.device,
        )
    elif rebuild_flags is None:
        neighbor_matrix1.fill_(fill_value)

    if num_neighbors1 is None:
        num_neighbors1 = torch.zeros(
            total_atoms, dtype=torch.int32, device=positions.device
        )
    elif rebuild_flags is None:
        num_neighbors1.zero_()

    if neighbor_matrix2 is None:
        neighbor_matrix2 = torch.full(
            (total_atoms, max_neighbors2),
            fill_value,
            dtype=torch.int32,
            device=positions.device,
        )
    elif rebuild_flags is None:
        neighbor_matrix2.fill_(fill_value)

    if num_neighbors2 is None:
        num_neighbors2 = torch.zeros(
            total_atoms, dtype=torch.int32, device=positions.device
        )
    elif rebuild_flags is None:
        num_neighbors2.zero_()

    if pbc is not None:
        if neighbor_matrix_shifts1 is None:
            neighbor_matrix_shifts1 = torch.zeros(
                (total_atoms, max_neighbors1, 3),
                dtype=torch.int32,
                device=positions.device,
            )
        elif rebuild_flags is None:
            neighbor_matrix_shifts1.zero_()
        if neighbor_matrix_shifts2 is None:
            neighbor_matrix_shifts2 = torch.zeros(
                (total_atoms, max_neighbors2, 3),
                dtype=torch.int32,
                device=positions.device,
            )
        elif rebuild_flags is None:
            neighbor_matrix_shifts2.zero_()
        if (
            max_shifts_per_system is None
            or num_shifts_per_system is None
            or shift_range_per_dimension is None
        ):
            shift_range_per_dimension, num_shifts_per_system, max_shifts_per_system = (
                compute_naive_num_shifts(cell, cutoff2, pbc)
            )

    batch_idx, batch_ptr = prepare_batch_idx_ptr(
        batch_idx=batch_idx,
        batch_ptr=batch_ptr,
        num_atoms=total_atoms,
        device=positions.device,
    )

    # Validate batch_idx size matches total_atoms at the public batched entry point.
    if batch_idx.shape[0] != total_atoms:
        raise RuntimeError(
            f"batch_idx length ({batch_idx.shape[0]}) does not match "
            f"num_atoms ({total_atoms}). batch_idx must have one entry per atom."
        )

    if pbc is None:
        if rebuild_flags is not None:
            _batch_naive_neighbor_matrix_no_pbc_dual_cutoff_selective(
                positions=positions,
                cutoff1=cutoff1,
                cutoff2=cutoff2,
                batch_idx=batch_idx,
                batch_ptr=batch_ptr,
                neighbor_matrix1=neighbor_matrix1,
                num_neighbors1=num_neighbors1,
                neighbor_matrix2=neighbor_matrix2,
                num_neighbors2=num_neighbors2,
                rebuild_flags=rebuild_flags,
                half_fill=half_fill,
            )
        else:
            _batch_naive_neighbor_matrix_no_pbc_dual_cutoff(
                positions=positions,
                cutoff1=cutoff1,
                cutoff2=cutoff2,
                batch_idx=batch_idx,
                batch_ptr=batch_ptr,
                neighbor_matrix1=neighbor_matrix1,
                num_neighbors1=num_neighbors1,
                neighbor_matrix2=neighbor_matrix2,
                num_neighbors2=num_neighbors2,
                half_fill=half_fill,
            )
        if return_neighbor_list:
            neighbor_list1, neighbor_ptr1 = get_neighbor_list_from_neighbor_matrix(
                neighbor_matrix1, num_neighbors=num_neighbors1, fill_value=fill_value
            )
            neighbor_list2, neighbor_ptr2 = get_neighbor_list_from_neighbor_matrix(
                neighbor_matrix2, num_neighbors=num_neighbors2, fill_value=fill_value
            )
            return (
                neighbor_list1,
                neighbor_ptr1,
                neighbor_list2,
                neighbor_ptr2,
            )
        else:
            return (
                neighbor_matrix1,
                num_neighbors1,
                neighbor_matrix2,
                num_neighbors2,
            )
    else:
        if rebuild_flags is not None:
            _batch_naive_neighbor_matrix_pbc_dual_cutoff_selective(
                positions=positions,
                cell=cell,
                pbc=pbc,
                cutoff1=cutoff1,
                cutoff2=cutoff2,
                batch_idx=batch_idx,
                batch_ptr=batch_ptr,
                neighbor_matrix1=neighbor_matrix1,
                neighbor_matrix2=neighbor_matrix2,
                neighbor_matrix_shifts1=neighbor_matrix_shifts1,
                neighbor_matrix_shifts2=neighbor_matrix_shifts2,
                num_neighbors1=num_neighbors1,
                num_neighbors2=num_neighbors2,
                shift_range_per_dimension=shift_range_per_dimension,
                num_shifts_per_system=num_shifts_per_system,
                max_shifts_per_system=max_shifts_per_system,
                rebuild_flags=rebuild_flags,
                half_fill=half_fill,
                max_atoms_per_system=max_atoms_per_system,
                wrap_positions=wrap_positions,
                positions_wrapped_buffer=positions_wrapped_buffer,
                per_atom_cell_offsets_buffer=per_atom_cell_offsets_buffer,
                inv_cell_buffer=inv_cell_buffer,
            )
        else:
            _batch_naive_neighbor_matrix_pbc_dual_cutoff(
                positions=positions,
                cell=cell,
                pbc=pbc,
                cutoff1=cutoff1,
                cutoff2=cutoff2,
                batch_idx=batch_idx,
                batch_ptr=batch_ptr,
                neighbor_matrix1=neighbor_matrix1,
                neighbor_matrix2=neighbor_matrix2,
                neighbor_matrix_shifts1=neighbor_matrix_shifts1,
                neighbor_matrix_shifts2=neighbor_matrix_shifts2,
                num_neighbors1=num_neighbors1,
                num_neighbors2=num_neighbors2,
                shift_range_per_dimension=shift_range_per_dimension,
                num_shifts_per_system=num_shifts_per_system,
                max_shifts_per_system=max_shifts_per_system,
                half_fill=half_fill,
                max_atoms_per_system=max_atoms_per_system,
                wrap_positions=wrap_positions,
                positions_wrapped_buffer=positions_wrapped_buffer,
                per_atom_cell_offsets_buffer=per_atom_cell_offsets_buffer,
                inv_cell_buffer=inv_cell_buffer,
            )
        if return_neighbor_list:
            neighbor_list1, neighbor_ptr1, unit_shifts1 = (
                get_neighbor_list_from_neighbor_matrix(
                    neighbor_matrix1,
                    num_neighbors=num_neighbors1,
                    neighbor_shift_matrix=neighbor_matrix_shifts1,
                    fill_value=fill_value,
                )
            )
            neighbor_list2, neighbor_ptr2, unit_shifts2 = (
                get_neighbor_list_from_neighbor_matrix(
                    neighbor_matrix2,
                    num_neighbors=num_neighbors2,
                    neighbor_shift_matrix=neighbor_matrix_shifts2,
                    fill_value=fill_value,
                )
            )
            return (
                neighbor_list1,
                neighbor_ptr1,
                unit_shifts1,
                neighbor_list2,
                neighbor_ptr2,
                unit_shifts2,
            )
        else:
            return (
                neighbor_matrix1,
                num_neighbors1,
                neighbor_matrix_shifts1,
                neighbor_matrix2,
                num_neighbors2,
                neighbor_matrix_shifts2,
            )
