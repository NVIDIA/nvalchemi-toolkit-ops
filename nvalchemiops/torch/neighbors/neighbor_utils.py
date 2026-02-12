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

import torch
import warp as wp

from nvalchemiops.neighbors.neighbor_utils import (
    NeighborOverflowError,
    estimate_max_neighbors,
)
from nvalchemiops.neighbors.neighbor_utils import (
    compute_naive_num_shifts as wp_compute_naive_num_shifts,
)
from nvalchemiops.types import get_wp_dtype, get_wp_mat_dtype

__all__ = [
    "compute_naive_num_shifts",
    "get_neighbor_list_from_neighbor_matrix",
    "prepare_batch_idx_ptr",
    "allocate_cell_list",
    "estimate_max_neighbors",
    "NeighborOverflowError",
]


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
    shift_offset : torch.Tensor, shape (num_systems + 1,), dtype=int32
        Cumulative sum of number of shifts for each system.
    total_shifts : int
        Total number of periodic shifts needed across all systems.

    See Also
    --------
    nvalchemiops.neighbors.neighbor_utils.compute_naive_num_shifts : Core warp launcher
    """
    num_systems = cell.shape[0]
    device = cell.device

    num_shifts = torch.empty(num_systems, dtype=torch.int32, device=device)
    shift_range = torch.empty((num_systems, 3), dtype=torch.int32, device=device)

    wp_dtype = get_wp_dtype(cell.dtype)
    wp_mat_dtype = get_wp_mat_dtype(cell.dtype)
    wp_device = wp.device_from_torch(device)

    wp_cell = wp.from_torch(cell, dtype=wp_mat_dtype)
    wp_pbc = wp.from_torch(pbc, dtype=wp.bool)
    wp_num_shifts = wp.from_torch(num_shifts, dtype=wp.int32)
    wp_shift_range = wp.from_torch(shift_range, dtype=wp.vec3i)

    wp_compute_naive_num_shifts(
        cell=wp_cell,
        cutoff=cutoff,
        pbc=wp_pbc,
        num_shifts=wp_num_shifts,
        shift_range=wp_shift_range,
        wp_dtype=wp_dtype,
        device=str(wp_device),
    )

    shift_offset = torch.empty((num_systems + 1,), dtype=torch.int32, device=device)
    shift_offset[0] = 0
    torch.cumsum(num_shifts, dim=0, out=shift_offset[1:])
    return shift_range, shift_offset, shift_offset[-1].item()


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
    ValueError
        If the max number of neighbors is larger than the neighbor matrix width.

    Notes
    -----
    This is a pure PyTorch utility function with no warp dependencies. It converts
    from the fixed-width matrix format to the variable-width list format by masking
    out fill values and flattening the result.

    See Also
    --------
    nvalchemiops.torch.neighbors.unbatched.naive_neighbor_list : Uses this for format conversion
    nvalchemiops.torch.neighbors.unbatched.cell_list : Uses this for format conversion
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

    # Validate that the neighbor matrix is large enough
    max_found = num_neighbors.max()
    if max_found > neighbor_matrix.shape[1]:
        raise NeighborOverflowError(
            neighbor_matrix.shape[1],
            max_found.item() if hasattr(max_found, "item") else int(max_found),
        )

    # Create mask and extract neighbor pairs
    mask = neighbor_matrix != fill_value
    dtype = neighbor_matrix.dtype
    i_idx = torch.where(mask)[0].to(dtype)
    j_idx = neighbor_matrix[mask].to(dtype)
    neighbor_list = torch.stack([i_idx, j_idx], dim=0)

    # Create CSR-style pointer array
    neighbor_ptr = torch.zeros(
        num_neighbors.shape[0] + 1, dtype=torch.int32, device=neighbor_matrix.device
    )
    torch.cumsum(num_neighbors, dim=0, out=neighbor_ptr[1:])

    if neighbor_shift_matrix is not None:
        neighbor_list_shifts = neighbor_shift_matrix[mask]
        return neighbor_list, neighbor_ptr, neighbor_list_shifts
    else:
        return neighbor_list, neighbor_ptr


@torch.compile
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
    nvalchemiops.torch.neighbors.batched.batch_naive_neighbor_list : Uses this for batch setup
    nvalchemiops.torch.neighbors.batched.batch_cell_list : Uses this for batch setup
    """
    if batch_idx is None and batch_ptr is None:
        raise ValueError("Either batch_idx or batch_ptr must be provided.")

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
    nvalchemiops.torch.neighbors.unbatched.build_cell_list : High-level PyTorch wrapper
    nvalchemiops.torch.neighbors.batched.batch_build_cell_list : Batched version
    """
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
