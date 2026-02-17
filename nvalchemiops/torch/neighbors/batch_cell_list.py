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

"""PyTorch bindings for batched cell list neighbor construction."""

from __future__ import annotations

import warnings

import torch
import warp as wp

from nvalchemiops.neighbors.batch_cell_list import (
    _batch_estimate_cell_list_sizes_overload,
)
from nvalchemiops.neighbors.batch_cell_list import (
    batch_build_cell_list as wp_batch_build_cell_list,
)
from nvalchemiops.neighbors.batch_cell_list import (
    batch_query_cell_list as wp_batch_query_cell_list,
)
from nvalchemiops.neighbors.neighbor_utils import estimate_max_neighbors
from nvalchemiops.torch.neighbors.neighbor_utils import (
    allocate_cell_list,
    get_neighbor_list_from_neighbor_matrix,
)
from nvalchemiops.types import get_wp_dtype, get_wp_mat_dtype, get_wp_vec_dtype

__all__ = [
    "estimate_batch_cell_list_sizes",
    "batch_build_cell_list",
    "batch_query_cell_list",
    "batch_cell_list",
]


def estimate_batch_cell_list_sizes(
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cutoff: float,
    max_nbins: int = 1000,
) -> tuple[int, torch.Tensor]:
    """Estimate memory allocation sizes for batch cell list construction.

    Analyzes a batch of systems to determine conservative memory
    allocation requirements for torch.compile-friendly batch cell list building.
    Uses system sizes, cutoff distance, and safety factors to prevent overflow.

    Parameters
    ----------
    cell : torch.Tensor, shape (num_systems, 3, 3)
        Unit cell matrices for each system in the batch.
    pbc : torch.Tensor, shape (num_systems, 3), dtype=bool
        Periodic boundary condition flags for each system and dimension.
    cutoff : float
        Neighbor search cutoff distance.
    max_nbins : int, default=1000
        Maximum number of cells to allocate per system.

    Returns
    -------
    max_total_cells_across_batch : int
        Estimated maximum total cells needed across all systems combined.
    neighbor_search_radius : torch.Tensor, shape (num_systems, 3), dtype=int32
        Radius of neighboring cells to search for each system.

    Notes
    -----
    - Currently, only unit cells with a positive determinant (i.e. with
      positive volume) are supported. For non-periodic systems, pass an identity
      cell.
    - Estimates assume roughly uniform atomic distribution within each system
    - Cell sizes are determined by the smallest cutoff to ensure neighbor completeness
    - For degenerate cells or empty systems, returns conservative fallback values

    See Also
    --------
    nvalchemiops.neighbors.batch_cell_list.batch_build_cell_list : Core warp launcher
    allocate_cell_list : Allocates tensors based on these estimates
    batch_build_cell_list : High-level wrapper that uses these estimates
    """
    if cell.numel() > 0 and torch.any(cell.det() <= 0.0):
        raise RuntimeError(
            "Cells with volume <= 0 detected and are not supported."
            " Please pass unit cells with `det(cell) > 0.0`."
        )
    num_systems = cell.shape[0]

    if num_systems == 0 or cutoff <= 0:
        return 1, torch.zeros((num_systems, 3), device=cell.device, dtype=torch.int32)

    dtype = cell.dtype
    device = cell.device
    wp_device = str(device)
    wp_dtype = get_wp_dtype(dtype)
    wp_mat_dtype = get_wp_mat_dtype(dtype)

    wp_cell = wp.from_torch(cell, dtype=wp_mat_dtype, return_ctype=True)
    wp_pbc = wp.from_torch(pbc, dtype=wp.bool, return_ctype=True)

    max_total_cells = torch.zeros(num_systems, device=device, dtype=torch.int32)
    wp_max_total_cells = wp.from_torch(
        max_total_cells, dtype=wp.int32, return_ctype=True
    )
    neighbor_search_radius = torch.zeros(
        (num_systems, 3), dtype=torch.int32, device=device
    )
    wp_neighbor_search_radius = wp.from_torch(
        neighbor_search_radius, dtype=wp.vec3i, return_ctype=True
    )

    wp.launch(
        _batch_estimate_cell_list_sizes_overload[wp_dtype],
        dim=num_systems,
        inputs=[
            wp_cell,
            wp_pbc,
            wp_dtype(cutoff),
            max_nbins,
            wp_max_total_cells,
            wp_neighbor_search_radius,
        ],
        device=wp_device,
    )

    return (
        max_total_cells.sum().item(),
        neighbor_search_radius,
    )


@torch.library.custom_op(
    "nvalchemiops::batch_build_cell_list",
    mutates_args=(
        "cells_per_dimension",
        "atom_periodic_shifts",
        "atom_to_cell_mapping",
        "atoms_per_cell_count",
        "cell_atom_start_indices",
        "cell_atom_list",
    ),
)
def _batch_build_cell_list_op(
    positions: torch.Tensor,
    cutoff: float,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    batch_idx: torch.Tensor,
    cells_per_dimension: torch.Tensor,
    atom_periodic_shifts: torch.Tensor,
    atom_to_cell_mapping: torch.Tensor,
    atoms_per_cell_count: torch.Tensor,
    cell_atom_start_indices: torch.Tensor,
    cell_atom_list: torch.Tensor,
) -> None:
    """Internal custom op for building batch spatial cell lists.

    This function is torch compilable.

    Notes
    -----
    The neighbor_search_radius is not an input parameter because it's not used
    during cell list building - it's only needed for querying the cell list.

    See Also
    --------
    nvalchemiops.neighbors.batch_cell_list.batch_build_cell_list : Core warp launcher
    batch_build_cell_list : High-level wrapper function
    """
    device = positions.device
    num_systems = cell.shape[0]

    # Handle empty case
    if positions.shape[0] == 0 or cutoff <= 0:
        return

    # Get warp dtype of input tensors
    wp_dtype = get_wp_dtype(positions.dtype)
    wp_vec_dtype = get_wp_vec_dtype(positions.dtype)
    wp_mat_dtype = get_wp_mat_dtype(positions.dtype)
    wp_device = str(device)

    # Convert to warp arrays
    wp_positions = wp.from_torch(positions, dtype=wp_vec_dtype, return_ctype=True)
    wp_cell = wp.from_torch(cell, dtype=wp_mat_dtype, return_ctype=True)
    wp_pbc = wp.from_torch(pbc, dtype=wp.bool, return_ctype=True)
    wp_batch_idx = wp.from_torch(
        batch_idx.to(dtype=torch.int32), dtype=wp.int32, return_ctype=True
    )

    wp_cells_per_dimension = wp.from_torch(
        cells_per_dimension, dtype=wp.vec3i, return_ctype=True
    )

    # Allocate cell_offsets internally (shape num_systems, not num_systems+1)
    cell_offsets = torch.zeros(num_systems, dtype=torch.int32, device=device)
    wp_cell_offsets = wp.from_torch(cell_offsets, dtype=wp.int32)

    # Allocate cells_per_system scratch buffer
    cells_per_system = torch.zeros(num_systems, dtype=torch.int32, device=device)
    wp_cells_per_system = wp.from_torch(cells_per_system, dtype=wp.int32)

    wp_atom_periodic_shifts = wp.from_torch(
        atom_periodic_shifts, dtype=wp.vec3i, return_ctype=True
    )
    wp_atom_to_cell_mapping = wp.from_torch(
        atom_to_cell_mapping, dtype=wp.vec3i, return_ctype=True
    )
    # underlying warp launcher relies on Python API for array_scan
    # so `return_ctype` is omitted
    wp_atoms_per_cell_count = wp.from_torch(atoms_per_cell_count, dtype=wp.int32)
    wp_cell_atom_start_indices = wp.from_torch(cell_atom_start_indices, dtype=wp.int32)
    wp_cell_atom_list = wp.from_torch(cell_atom_list, dtype=wp.int32, return_ctype=True)

    # Zero atoms_per_cell_count before building
    atoms_per_cell_count.zero_()

    # Call core warp launcher
    wp_batch_build_cell_list(
        positions=wp_positions,
        cell=wp_cell,
        pbc=wp_pbc,
        cutoff=cutoff,
        batch_idx=wp_batch_idx,
        cells_per_dimension=wp_cells_per_dimension,
        cell_offsets=wp_cell_offsets,
        cells_per_system=wp_cells_per_system,
        atom_periodic_shifts=wp_atom_periodic_shifts,
        atom_to_cell_mapping=wp_atom_to_cell_mapping,
        atoms_per_cell_count=wp_atoms_per_cell_count,
        cell_atom_start_indices=wp_cell_atom_start_indices,
        cell_atom_list=wp_cell_atom_list,
        wp_dtype=wp_dtype,
        device=wp_device,
    )


def batch_build_cell_list(
    positions: torch.Tensor,
    cutoff: float,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    batch_idx: torch.Tensor,
    cells_per_dimension: torch.Tensor,
    neighbor_search_radius: torch.Tensor,
    atom_periodic_shifts: torch.Tensor,
    atom_to_cell_mapping: torch.Tensor,
    atoms_per_cell_count: torch.Tensor,
    cell_atom_start_indices: torch.Tensor,
    cell_atom_list: torch.Tensor,
) -> None:
    """Build batch spatial cell lists with fixed allocation sizes for torch.compile compatibility.

    This function is torch compilable.

    Parameters
    ----------
    positions : torch.Tensor, shape (total_atoms, 3)
        Concatenated atomic coordinates for all systems in the batch.
    cutoff : float
        Neighbor search cutoff distance.
    cell : torch.Tensor, shape (num_systems, 3, 3)
        Unit cell matrices for each system in the batch.
    pbc : torch.Tensor, shape (num_systems, 3), dtype=bool
        Periodic boundary condition flags for each system and dimension.
    batch_idx : torch.Tensor, shape (total_atoms,), dtype=int32
        System index for each atom.
    cells_per_dimension : torch.Tensor, shape (num_systems, 3), dtype=int32
        OUTPUT: Number of cells in x, y, z directions for each system.
    neighbor_search_radius : torch.Tensor, shape (num_systems, 3), dtype=int32
        Radius of neighboring cells to search in each dimension. Passed through
        from allocate_cell_list for API continuity but not used in this function.
    atom_periodic_shifts : torch.Tensor, shape (total_atoms, 3), dtype=int32
        OUTPUT: Periodic boundary crossings for each atom across all systems.
    atom_to_cell_mapping : torch.Tensor, shape (total_atoms, 3), dtype=int32
        OUTPUT: 3D cell coordinates assigned to each atom across all systems.
    atoms_per_cell_count : torch.Tensor, shape (max_total_cells,), dtype=int32
        OUTPUT: Number of atoms in each cell across all systems.
    cell_atom_start_indices : torch.Tensor, shape (max_total_cells,), dtype=int32
        OUTPUT: Starting index in global cell arrays for each system (CSR format).
    cell_atom_list : torch.Tensor, shape (total_atoms,), dtype=int32
        OUTPUT: Flattened list of atom indices organized by cell across all systems.

    See Also
    --------
    nvalchemiops.neighbors.batch_cell_list.batch_build_cell_list : Core warp launcher
    estimate_batch_cell_list_sizes : Estimate memory requirements
    batch_query_cell_list : Query the built cell list for neighbors
    batch_cell_list : High-level function that builds and queries in one call
    """
    return _batch_build_cell_list_op(
        positions,
        cutoff,
        cell,
        pbc,
        batch_idx,
        cells_per_dimension,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
    )


@torch.library.custom_op(
    "nvalchemiops::batch_query_cell_list",
    mutates_args=("neighbor_matrix", "neighbor_matrix_shifts", "num_neighbors"),
)
def _batch_query_cell_list_op(
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cutoff: float,
    batch_idx: torch.Tensor,
    cells_per_dimension: torch.Tensor,
    neighbor_search_radius: torch.Tensor,
    atom_periodic_shifts: torch.Tensor,
    atom_to_cell_mapping: torch.Tensor,
    atoms_per_cell_count: torch.Tensor,
    cell_atom_start_indices: torch.Tensor,
    cell_atom_list: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    num_neighbors: torch.Tensor,
    half_fill: bool = False,
) -> None:
    """Internal custom op for querying batch spatial cell lists to build neighbor matrices.

    This function is torch compilable.

    See Also
    --------
    nvalchemiops.neighbors.batch_cell_list.batch_query_cell_list : Core warp launcher
    batch_query_cell_list : High-level wrapper function
    """
    device = positions.device
    num_systems = cell.shape[0]

    # Handle empty case
    if positions.shape[0] == 0 or cutoff <= 0:
        return

    # Get warp dtypes and arrays
    wp_dtype = get_wp_dtype(positions.dtype)
    wp_vec_dtype = get_wp_vec_dtype(positions.dtype)
    wp_mat_dtype = get_wp_mat_dtype(positions.dtype)
    wp_device = str(device)

    wp_positions = wp.from_torch(positions, dtype=wp_vec_dtype, return_ctype=True)
    wp_cell = wp.from_torch(cell, dtype=wp_mat_dtype, return_ctype=True)
    wp_pbc = wp.from_torch(pbc, dtype=wp.bool, return_ctype=True)
    wp_batch_idx = wp.from_torch(
        batch_idx.to(dtype=torch.int32), dtype=wp.int32, return_ctype=True
    )

    wp_cells_per_dimension = wp.from_torch(
        cells_per_dimension, dtype=wp.vec3i, return_ctype=True
    )
    wp_neighbor_search_radius = wp.from_torch(
        neighbor_search_radius, dtype=wp.vec3i, return_ctype=True
    )

    #  cell_offsets[i] = sum of cells for systems 0..i-1
    cells_per_system = cells_per_dimension.prod(dim=1)
    cell_offsets = torch.zeros(num_systems, dtype=torch.int32, device=device)
    if num_systems > 1:
        torch.cumsum(cells_per_system[:-1], dim=0, out=cell_offsets[1:])
    # cell_offsets[0] is already 0 from zeros initialization
    wp_cell_offsets = wp.from_torch(cell_offsets, dtype=wp.int32, return_ctype=True)

    wp_atom_periodic_shifts = wp.from_torch(
        atom_periodic_shifts, dtype=wp.vec3i, return_ctype=True
    )
    wp_atom_to_cell_mapping = wp.from_torch(
        atom_to_cell_mapping, dtype=wp.vec3i, return_ctype=True
    )
    wp_atoms_per_cell_count = wp.from_torch(
        atoms_per_cell_count, dtype=wp.int32, return_ctype=True
    )
    wp_cell_atom_start_indices = wp.from_torch(
        cell_atom_start_indices, dtype=wp.int32, return_ctype=True
    )
    wp_cell_atom_list = wp.from_torch(cell_atom_list, dtype=wp.int32, return_ctype=True)

    wp_neighbor_matrix = wp.from_torch(
        neighbor_matrix, dtype=wp.int32, return_ctype=True
    )
    wp_neighbor_matrix_shifts = wp.from_torch(
        neighbor_matrix_shifts, dtype=wp.vec3i, return_ctype=True
    )
    wp_num_neighbors = wp.from_torch(num_neighbors, dtype=wp.int32, return_ctype=True)

    # Call core warp launcher
    wp_batch_query_cell_list(
        positions=wp_positions,
        cell=wp_cell,
        pbc=wp_pbc,
        cutoff=cutoff,
        batch_idx=wp_batch_idx,
        cells_per_dimension=wp_cells_per_dimension,
        neighbor_search_radius=wp_neighbor_search_radius,
        cell_offsets=wp_cell_offsets,
        atom_periodic_shifts=wp_atom_periodic_shifts,
        atom_to_cell_mapping=wp_atom_to_cell_mapping,
        atoms_per_cell_count=wp_atoms_per_cell_count,
        cell_atom_start_indices=wp_cell_atom_start_indices,
        cell_atom_list=wp_cell_atom_list,
        neighbor_matrix=wp_neighbor_matrix,
        neighbor_matrix_shifts=wp_neighbor_matrix_shifts,
        num_neighbors=wp_num_neighbors,
        wp_dtype=wp_dtype,
        device=wp_device,
        half_fill=half_fill,
    )


def batch_query_cell_list(
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cutoff: float,
    batch_idx: torch.Tensor,
    cells_per_dimension: torch.Tensor,
    neighbor_search_radius: torch.Tensor,
    atom_periodic_shifts: torch.Tensor,
    atom_to_cell_mapping: torch.Tensor,
    atoms_per_cell_count: torch.Tensor,
    cell_atom_start_indices: torch.Tensor,
    cell_atom_list: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    num_neighbors: torch.Tensor,
    half_fill: bool = False,
) -> None:
    """Query batch spatial cell lists to build neighbor matrices for multiple systems.

    This function is torch compilable.

    See Also
    --------
    nvalchemiops.neighbors.batch_cell_list.batch_query_cell_list : Core warp launcher
    batch_build_cell_list : Builds the cell list data structures
    batch_cell_list : High-level function that builds and queries in one call
    """
    return _batch_query_cell_list_op(
        positions,
        cell,
        pbc,
        cutoff,
        batch_idx,
        cells_per_dimension,
        neighbor_search_radius,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
        half_fill,
    )


def batch_cell_list(
    positions: torch.Tensor,
    cutoff: float,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    batch_idx: torch.Tensor,
    max_neighbors: int | None = None,
    half_fill: bool = False,
    fill_value: int | None = None,
    return_neighbor_list: bool = False,
    neighbor_matrix: torch.Tensor | None = None,
    neighbor_matrix_shifts: torch.Tensor | None = None,
    num_neighbors: torch.Tensor | None = None,
    cells_per_dimension: torch.Tensor | None = None,
    neighbor_search_radius: torch.Tensor | None = None,
    cell_offsets: torch.Tensor | None = None,
    atom_periodic_shifts: torch.Tensor | None = None,
    atom_to_cell_mapping: torch.Tensor | None = None,
    atoms_per_cell_count: torch.Tensor | None = None,
    cell_atom_start_indices: torch.Tensor | None = None,
    cell_atom_list: torch.Tensor | None = None,
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor]
):
    """Build complete batch neighbor matrices using spatial cell list acceleration.

    High-level convenience function that processes multiple systems
    simultaneously. Automatically estimates memory requirements, builds batch
    spatial cell list data structures, and queries them to produce complete
    neighbor matrices for all systems.

    Parameters
    ----------
    positions : torch.Tensor, shape (total_atoms, 3)
        Concatenated atomic coordinates for all systems in the batch.
    cutoff : float
        Neighbor search cutoff distance.
    cell : torch.Tensor, shape (num_systems, 3, 3)
        Unit cell matrices for each system in the batch.
    pbc : torch.Tensor, shape (num_systems, 3), dtype=bool
        Periodic boundary condition flags for each system and dimension.
    batch_idx : torch.Tensor, shape (total_atoms,), dtype=int32
        System index for each atom.
    max_neighbors : int or None, optional
        Maximum number of neighbors per atom. If None, automatically estimated.
    half_fill : bool, default=False
        If True, only fill half of the neighbor matrix.
    fill_value : int | None, optional
        Value to use for padding empty neighbor slots in the matrix. Default is total_atoms.
    return_neighbor_list : bool, optional - default=False
        If True, convert the neighbor matrix to a neighbor list (idx_i, idx_j) format.
    cells_per_dimension : torch.Tensor, shape (num_systems, 3), dtype=int32, optional
        Pre-allocated tensor for cell dimensions.
    neighbor_search_radius : torch.Tensor, shape (num_systems, 3), dtype=int32, optional
        Pre-allocated tensor for search radius.
    atom_periodic_shifts : torch.Tensor, shape (total_atoms, 3), dtype=int32, optional
        Pre-allocated tensor for periodic shifts.
    atom_to_cell_mapping : torch.Tensor, shape (total_atoms, 3), dtype=int32, optional
        Pre-allocated tensor for cell mapping.
    atoms_per_cell_count : torch.Tensor, shape (max_total_cells,), dtype=int32, optional
        Pre-allocated tensor for atom counts.
    cell_atom_start_indices : torch.Tensor, shape (max_total_cells,), dtype=int32, optional
        Pre-allocated tensor for start indices.
    cell_atom_list : torch.Tensor, shape (total_atoms,), dtype=int32, optional
        Pre-allocated tensor for atom list.

    Returns
    -------
    results : tuple of torch.Tensor
        Variable-length tuple with neighbor data in matrix or list format.

    See Also
    --------
    nvalchemiops.neighbors.batch_cell_list.batch_build_cell_list : Core warp launcher for building
    nvalchemiops.neighbors.batch_cell_list.batch_query_cell_list : Core warp launcher for querying
    batch_naive_neighbor_list : O(N²) method for small systems
    """
    total_atoms = positions.shape[0]
    device = positions.device
    if device == "cpu":
        warnings.warn(
            "The CPU version of `batch_cell_list` is known to experience"
            " issues with memory allocation and under investigation. Please"
            " ensure tensor provided as `positions` is on GPU."
        )

    # Handle empty case
    if total_atoms <= 0 or cutoff <= 0:
        if return_neighbor_list:
            return (
                torch.zeros((2, 0), dtype=torch.int32, device=device),
                torch.zeros((total_atoms + 1,), dtype=torch.int32, device=device),
                torch.zeros((0, 3), dtype=torch.int32, device=device),
            )
        else:
            return (
                torch.full((total_atoms, 0), -1, dtype=torch.int32, device=device),
                torch.zeros((total_atoms,), dtype=torch.int32, device=device),
                torch.zeros((total_atoms, 0, 3), dtype=torch.int32, device=device),
            )

    if max_neighbors is None and neighbor_matrix is None:
        max_neighbors = estimate_max_neighbors(cutoff)

    if fill_value is None:
        fill_value = total_atoms

    if neighbor_matrix is None:
        neighbor_matrix = torch.full(
            (total_atoms, max_neighbors), fill_value, dtype=torch.int32, device=device
        )
    else:
        neighbor_matrix.fill_(fill_value)
    if neighbor_matrix_shifts is None:
        neighbor_matrix_shifts = torch.zeros(
            (total_atoms, max_neighbors, 3), dtype=torch.int32, device=device
        )
    else:
        neighbor_matrix_shifts.zero_()
    if num_neighbors is None:
        num_neighbors = torch.zeros((total_atoms,), dtype=torch.int32, device=device)
    else:
        num_neighbors.zero_()

    # Allocate cell list if needed
    if (
        cells_per_dimension is None
        or neighbor_search_radius is None
        or atom_periodic_shifts is None
        or atom_to_cell_mapping is None
        or atoms_per_cell_count is None
        or cell_atom_start_indices is None
        or cell_atom_list is None
    ):
        max_total_cells, neighbor_search_radius = estimate_batch_cell_list_sizes(
            cell, pbc, cutoff
        )
        (
            cells_per_dimension,
            neighbor_search_radius,
            atom_periodic_shifts,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
        ) = allocate_cell_list(
            total_atoms,
            max_total_cells,
            neighbor_search_radius,
            device,
        )
        cell_list_cache = (
            cells_per_dimension,
            neighbor_search_radius,
            atom_periodic_shifts,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
        )
    else:
        cells_per_dimension.zero_()
        atom_periodic_shifts.zero_()
        atom_to_cell_mapping.zero_()
        atoms_per_cell_count.zero_()
        cell_atom_start_indices.zero_()
        cell_atom_list.zero_()
        cell_list_cache = (
            cells_per_dimension,
            neighbor_search_radius,
            atom_periodic_shifts,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
        )

    # Build batch cell list with fixed allocations
    batch_build_cell_list(
        positions,
        cutoff,
        cell,
        pbc,
        batch_idx,
        *cell_list_cache,
    )

    # Query neighbor lists
    batch_query_cell_list(
        positions,
        cell,
        pbc,
        cutoff,
        batch_idx,
        *cell_list_cache,
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
        half_fill,
    )

    if return_neighbor_list:
        neighbor_list, neighbor_ptr, neighbor_list_shifts = (
            get_neighbor_list_from_neighbor_matrix(
                neighbor_matrix,
                num_neighbors=num_neighbors,
                neighbor_shift_matrix=neighbor_matrix_shifts,
                fill_value=fill_value,
            )
        )
        return neighbor_list, neighbor_ptr, neighbor_list_shifts
    else:
        return neighbor_matrix, num_neighbors, neighbor_matrix_shifts
