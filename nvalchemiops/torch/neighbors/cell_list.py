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

"""PyTorch bindings for unbatched cell list neighbor construction."""

from __future__ import annotations

import torch
import warp as wp

from nvalchemiops.neighbors.cell_list import (
    build_cell_list as wp_build_cell_list,
)
from nvalchemiops.neighbors.cell_list import (
    query_cell_list as wp_query_cell_list,
)
from nvalchemiops.neighbors.neighbor_utils import estimate_max_neighbors
from nvalchemiops.torch.neighbors.neighbor_utils import (
    allocate_cell_list,
    get_neighbor_list_from_neighbor_matrix,
)
from nvalchemiops.types import get_wp_dtype, get_wp_mat_dtype, get_wp_vec_dtype

__all__ = [
    "estimate_cell_list_sizes",
    "build_cell_list",
    "query_cell_list",
    "cell_list",
]


def estimate_cell_list_sizes(
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cutoff: float,
    max_nbins: int = 1000,
) -> tuple[int, torch.Tensor]:
    """Estimate allocation sizes for torch.compile-friendly cell list construction.

    Provides conservative estimates for maximum memory allocations needed when
    building cell lists with fixed-size tensors to avoid dynamic allocation
    and graph breaks in torch.compile.

    This function is not torch.compile compatible because it returns an integer
    received from using torch.Tensor.item()

    Parameters
    ----------
    cell : torch.Tensor, shape (1, 3, 3)
        Unit cell matrix defining the simulation box.
    pbc : torch.Tensor, shape (1, 3), dtype=bool
        Flags indicating periodic boundary conditions in x, y, z directions.
    cutoff : float
        Maximum distance for neighbor search, determines minimum cell size.
    max_nbins : int, default=1000
        Maximum number of cells to allocate.

    Returns
    -------
    max_total_cells : int
        Estimated maximum number of cells needed for spatial decomposition.
        For degenerate cells, returns the total number of atoms.
    neighbor_search_radius : torch.Tensor, shape (3,), dtype=int32
        Radius of neighboring cells to search in each dimension.

    Notes
    -----
    - Cell size is determined by the cutoff distance to ensure neighboring
      cells contain all potential neighbors. The estimation assumes roughly
      cubic cells and uniform atomic distribution.
    - Currently, only unit cells with a positive determinant (i.e. with
      positive volume) are supported. For non-periodic systems, pass an identity
      cell.

    See Also
    --------
    nvalchemiops.neighbors.cell_list.build_cell_list : Core warp launcher
    allocate_cell_list : Allocates tensors based on these estimates
    build_cell_list : High-level wrapper that uses these estimates
    """
    if cell.numel() > 0 and cell.det() <= 0.0:
        raise RuntimeError(
            "Cell with volume <= 0.0 detected and is not supported."
            " Please pass unit cells with `det(cell) > 0.0`."
        )
    dtype = cell.dtype
    device = cell.device
    wp_device = str(device)
    wp_dtype = get_wp_dtype(dtype)
    wp_mat_dtype = get_wp_mat_dtype(dtype)
    wp_cell = wp.from_torch(cell, dtype=wp_mat_dtype, return_ctype=True)
    wp_pbc = wp.from_torch(pbc, dtype=wp.bool, return_ctype=True)

    if (cell.ndim == 3 and cell.shape[0] == 0) or cutoff <= 0:
        return 1, torch.zeros((3,), dtype=torch.int32, device=device)

    if cell.ndim == 2:
        cell = cell.unsqueeze(0)

    max_total_cells = torch.zeros(1, device=device, dtype=torch.int32)
    wp_max_total_cells = wp.from_torch(
        max_total_cells, dtype=wp.int32, return_ctype=True
    )

    neighbor_search_radius = torch.zeros((3,), dtype=torch.int32, device=device)
    wp_neighbor_search_radius = wp.from_torch(
        neighbor_search_radius, dtype=wp.int32, return_ctype=True
    )

    # Note: Using the _estimate_cell_list_sizes kernel from cell_list module
    from nvalchemiops.neighbors.cell_list import _estimate_cell_list_sizes_overload

    wp.launch(
        _estimate_cell_list_sizes_overload[wp_dtype],
        dim=1,
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
        max_total_cells.item(),
        neighbor_search_radius,
    )


@torch.library.custom_op(
    "nvalchemiops::build_cell_list",
    mutates_args=(
        "cells_per_dimension",
        "atom_periodic_shifts",
        "atom_to_cell_mapping",
        "atoms_per_cell_count",
        "cell_atom_start_indices",
        "cell_atom_list",
    ),
)
def _build_cell_list_op(
    positions: torch.Tensor,
    cutoff: float,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cells_per_dimension: torch.Tensor,
    atom_periodic_shifts: torch.Tensor,
    atom_to_cell_mapping: torch.Tensor,
    atoms_per_cell_count: torch.Tensor,
    cell_atom_start_indices: torch.Tensor,
    cell_atom_list: torch.Tensor,
) -> None:
    """Internal custom op for building spatial cell list.

    This function is torch compilable.

    Notes
    -----
    The neighbor_search_radius is not an input parameter because it's computed
    internally by the warp launcher and doesn't need to be passed in.

    See Also
    --------
    nvalchemiops.neighbors.cell_list.build_cell_list : Core warp launcher
    build_cell_list : High-level wrapper function
    """
    total_atoms = positions.shape[0]
    device = positions.device

    # Handle empty case
    if total_atoms == 0:
        return

    cell = cell if cell.ndim == 3 else cell.unsqueeze(0)
    pbc = pbc.squeeze(0)

    # Get warp dtypes and arrays
    wp_dtype = get_wp_dtype(positions.dtype)
    wp_vec_dtype = get_wp_vec_dtype(positions.dtype)
    wp_mat_dtype = get_wp_mat_dtype(positions.dtype)
    wp_device = str(device)

    wp_positions = wp.from_torch(positions, dtype=wp_vec_dtype, return_ctype=True)
    wp_cell = wp.from_torch(cell, dtype=wp_mat_dtype, return_ctype=True)
    wp_pbc = wp.from_torch(pbc, dtype=wp.bool, return_ctype=True)

    wp_cells_per_dimension = wp.from_torch(
        cells_per_dimension, dtype=wp.int32, return_ctype=True
    )
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
    wp_build_cell_list(
        positions=wp_positions,
        cell=wp_cell,
        pbc=wp_pbc,
        cutoff=cutoff,
        cells_per_dimension=wp_cells_per_dimension,
        atom_periodic_shifts=wp_atom_periodic_shifts,
        atom_to_cell_mapping=wp_atom_to_cell_mapping,
        atoms_per_cell_count=wp_atoms_per_cell_count,
        cell_atom_start_indices=wp_cell_atom_start_indices,
        cell_atom_list=wp_cell_atom_list,
        wp_dtype=wp_dtype,
        device=wp_device,
    )

    # Compute cell atom start indices using cumsum
    max_total_cells = atoms_per_cell_count.shape[0]
    cell_atom_start_indices[0] = 0
    if max_total_cells > 1:
        torch.cumsum(atoms_per_cell_count[:-1], dim=0, out=cell_atom_start_indices[1:])


def build_cell_list(
    positions: torch.Tensor,
    cutoff: float,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cells_per_dimension: torch.Tensor,
    neighbor_search_radius: torch.Tensor,
    atom_periodic_shifts: torch.Tensor,
    atom_to_cell_mapping: torch.Tensor,
    atoms_per_cell_count: torch.Tensor,
    cell_atom_start_indices: torch.Tensor,
    cell_atom_list: torch.Tensor,
) -> None:
    """Build spatial cell list with fixed allocation sizes for torch.compile compatibility.

    Constructs a spatial decomposition data structure for efficient neighbor searching.
    Uses fixed-size memory allocations to prevent dynamic tensor creation that would
    cause graph breaks in torch.compile.

    Parameters
    ----------
    positions : torch.Tensor, shape (total_atoms, 3)
        Atomic coordinates in Cartesian space where total_atoms is the number of atoms.
        Must be float32, float64, or float16 dtype.
    cutoff : float
        Maximum distance for neighbor search. Determines minimum cell size.
    cell : torch.Tensor, shape (1, 3, 3)
        Unit cell matrix defining the simulation box. Each row represents a
        lattice vector in Cartesian coordinates. Must match positions dtype.
    pbc : torch.Tensor, shape (3,), dtype=bool
        Flags indicating periodic boundary conditions in x, y, z directions.
        True enables PBC, False disables it for that dimension.
    cells_per_dimension : torch.Tensor, shape (3,), dtype=int32
        OUTPUT: Number of cells created in x, y, z directions.
    neighbor_search_radius : torch.Tensor, shape (3,), dtype=int32
        Radius of neighboring cells to search in each dimension. Passed through
        from allocate_cell_list for API continuity but not used in this function.
    atom_periodic_shifts : torch.Tensor, shape (total_atoms, 3), dtype=int32
        OUTPUT: Periodic boundary crossings for each atom.
    atom_to_cell_mapping : torch.Tensor, shape (total_atoms, 3), dtype=int32
        OUTPUT: 3D cell coordinates assigned to each atom.
    atoms_per_cell_count : torch.Tensor, shape (max_total_cells,), dtype=int32
        OUTPUT: Number of atoms in each cell. Only first 'total_cells' entries are valid.
    cell_atom_start_indices : torch.Tensor, shape (max_total_cells,), dtype=int32
        OUTPUT: Starting index in cell_atom_list for each cell's atoms.
    cell_atom_list : torch.Tensor, shape (total_atoms,), dtype=int32
        OUTPUT: Flattened list of atom indices organized by cell. Use with start_indices
        to extract atoms for each cell.

    Notes
    -----
    - This function is torch.compile compatible and uses only static tensor shapes
    - Memory usage is determined by max_total_cells
    - For optimal performance, use estimates from estimate_cell_list_sizes()
    - Cell list must be rebuilt when atoms move between cells or PBC/cell changes

    See Also
    --------
    nvalchemiops.neighbors.cell_list.build_cell_list : Core warp launcher
    estimate_cell_list_sizes : Estimate memory requirements
    query_cell_list : Query the built cell list for neighbors
    cell_list : High-level function that builds and queries in one call
    """
    return _build_cell_list_op(
        positions,
        cutoff,
        cell,
        pbc,
        cells_per_dimension,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
    )


@torch.library.custom_op(
    "nvalchemiops::query_cell_list",
    mutates_args=("neighbor_matrix", "neighbor_matrix_shifts", "num_neighbors"),
)
def _query_cell_list_op(
    positions: torch.Tensor,
    cutoff: float,
    cell: torch.Tensor,
    pbc: torch.Tensor,
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
    """Internal custom op for querying spatial cell list to build neighbor matrix.

    This function is torch compilable.

    See Also
    --------
    nvalchemiops.neighbors.cell_list.query_cell_list : Core warp launcher
    query_cell_list : High-level wrapper function
    """
    total_atoms = positions.shape[0]
    device = positions.device

    # Handle empty case
    if total_atoms == 0:
        return

    cell = cell if cell.ndim == 3 else cell.unsqueeze(0)
    pbc = pbc.squeeze(0)

    # Get warp dtypes and arrays
    wp_dtype = get_wp_dtype(positions.dtype)
    wp_vec_dtype = get_wp_vec_dtype(positions.dtype)
    wp_mat_dtype = get_wp_mat_dtype(positions.dtype)
    wp_device = str(device)

    wp_positions = wp.from_torch(positions, dtype=wp_vec_dtype, return_ctype=True)
    wp_cell = wp.from_torch(cell, dtype=wp_mat_dtype, return_ctype=True)
    wp_pbc = wp.from_torch(pbc, dtype=wp.bool, return_ctype=True)

    wp_cells_per_dimension = wp.from_torch(
        cells_per_dimension, dtype=wp.int32, return_ctype=True
    )
    wp_neighbor_search_radius = wp.from_torch(
        neighbor_search_radius, dtype=wp.int32, return_ctype=True
    )
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
    wp_query_cell_list(
        positions=wp_positions,
        cell=wp_cell,
        pbc=wp_pbc,
        cutoff=cutoff,
        cells_per_dimension=wp_cells_per_dimension,
        neighbor_search_radius=wp_neighbor_search_radius,
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


def query_cell_list(
    positions: torch.Tensor,
    cutoff: float,
    cell: torch.Tensor,
    pbc: torch.Tensor,
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
    """Query spatial cell list to build neighbor matrix with distance constraints.

    Uses pre-built cell list data structures to efficiently find all atom pairs
    within the specified cutoff distance. Handles periodic boundary conditions
    and returns neighbor matrix format.

    This function is torch compilable.

    Parameters
    ----------
    positions : torch.Tensor, shape (total_atoms, 3)
        Atomic coordinates in Cartesian space.
    cutoff : float
        Maximum distance for considering atoms as neighbors.
    cell : torch.Tensor, shape (1, 3, 3)
        Unit cell matrix for periodic boundary coordinate shifts.
    pbc : torch.Tensor, shape (3,), dtype=bool
        Periodic boundary condition flags.
    cells_per_dimension : torch.Tensor, shape (3,), dtype=int32
        Number of cells in x, y, z directions from build_cell_list.
    neighbor_search_radius : torch.Tensor, shape (3,), dtype=int32
        Shifts to search from build_cell_list.
    atom_periodic_shifts : torch.Tensor, shape (total_atoms, 3), dtype=int32
        Periodic boundary crossings for each atom from build_cell_list.
    atom_to_cell_mapping : torch.Tensor, shape (total_atoms, 3), dtype=int32
        3D cell coordinates for each atom from build_cell_list.
    atoms_per_cell_count : torch.Tensor, shape (max_total_cells,), dtype=int32
        Number of atoms in each cell from build_cell_list.
    cell_atom_start_indices : torch.Tensor, shape (max_total_cells,), dtype=int32
        Starting index in cell_atom_list for each cell from build_cell_list.
    cell_atom_list : torch.Tensor, shape (total_atoms,), dtype=int32
        Flattened list of atom indices organized by cell from build_cell_list.
    neighbor_matrix : torch.Tensor, shape (total_atoms, max_neighbors), dtype=int32
        OUTPUT: Neighbor matrix to be filled with neighbor atom indices.
        Must be pre-allocated.
    neighbor_matrix_shifts : torch.Tensor, shape (total_atoms, max_neighbors, 3), dtype=int32
        OUTPUT: Matrix storing shift vectors for each neighbor relationship.
        Must be pre-allocated.
    num_neighbors : torch.Tensor, shape (total_atoms,), dtype=int32
        OUTPUT: Number of neighbors found for each atom.
        Must be pre-allocated.
    half_fill : bool, default=False
        If True, only store half of the neighbor relationships.

    See Also
    --------
    nvalchemiops.neighbors.cell_list.query_cell_list : Core warp launcher
    build_cell_list : Builds the cell list data structures
    cell_list : High-level function that builds and queries in one call
    """
    return _query_cell_list_op(
        positions,
        cutoff,
        cell,
        pbc,
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


def cell_list(
    positions: torch.Tensor,
    cutoff: float,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    max_neighbors: int | None = None,
    half_fill: bool = False,
    fill_value: int | None = None,
    return_neighbor_list: bool = False,
    neighbor_matrix: torch.Tensor | None = None,
    neighbor_matrix_shifts: torch.Tensor | None = None,
    num_neighbors: torch.Tensor | None = None,
    cells_per_dimension: torch.Tensor | None = None,
    neighbor_search_radius: torch.Tensor | None = None,
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
    """Build complete neighbor matrix using spatial cell list acceleration.

    High-level convenience function that automatically estimates memory requirements,
    builds spatial cell list data structures, and queries them to produce a complete
    neighbor matrix. Combines build_cell_list and query_cell_list operations.

    Parameters
    ----------
    positions : torch.Tensor, shape (total_atoms, 3)
        Atomic coordinates in Cartesian space where total_atoms is the number of atoms.
    cutoff : float
        Maximum distance for neighbor search.
    cell : torch.Tensor, shape (1, 3, 3)
        Unit cell matrix defining the simulation box. Each row represents a
        lattice vector in Cartesian coordinates.
    pbc : torch.Tensor, shape (1, 3), dtype=bool
        Flags indicating periodic boundary conditions in x, y, z directions.
    max_neighbors : int, optional
        Maximum number of neighbors per atom. If not provided, will be estimated automatically.
    half_fill : bool, optional
        If True, only fill half of the neighbor matrix. Default is False.
    fill_value : int | None, optional
        Value to fill the neighbor matrix with. Default is total_atoms.
    return_neighbor_list : bool, optional - default = False
        If True, convert the neighbor matrix to a neighbor list (idx_i, idx_j) format by
        creating a mask over the fill_value, which can incur a performance penalty.
        We recommend using the neighbor matrix format,
        and only convert to a neighbor list format if absolutely necessary.
    neighbor_matrix : torch.Tensor, optional
        Pre-allocated tensor of shape (total_atoms, max_neighbors) for neighbor indices.
        If None, allocated internally.
    neighbor_matrix_shifts : torch.Tensor, optional
        Pre-allocated tensor of shape (total_atoms, max_neighbors, 3) for shift vectors.
        If None, allocated internally.
    num_neighbors : torch.Tensor, optional
        Pre-allocated tensor of shape (total_atoms,) for neighbor counts.
        If None, allocated internally.
    cells_per_dimension : torch.Tensor, shape (3,), dtype=int32, optional
        Number of cells in x, y, z directions.
        Pass a pre-allocated tensor to avoid reallocation for cell list construction.
        If None, allocated internally to build the cell list.
    neighbor_search_radius : torch.Tensor, shape (3,), dtype=int32, optional
        Radius of neighboring cells to search in each dimension.
        Pass a pre-allocated tensor to avoid reallocation for cell list construction.
        If None, allocated internally to build the cell list.
    atom_periodic_shifts : torch.Tensor, shape (total_atoms, 3), dtype=int32, optional
        Periodic boundary crossings for each atom.
        Pass a pre-allocated tensor to avoid reallocation for cell list construction.
        If None, allocated internally to build the cell list.
    atom_to_cell_mapping : torch.Tensor, shape (total_atoms, 3), dtype=int32, optional
        Cell coordinates for each atom.
        Pass a pre-allocated tensor to avoid reallocation for cell list construction.
        If None, allocated internally to build the cell list.
    atoms_per_cell_count : torch.Tensor, shape (max_total_cells,), dtype=int32, optional
        Number of atoms in each cell.
        Pass a pre-allocated tensor to avoid reallocation for cell list construction.
        If None, allocated internally to build the cell list.
    cell_atom_start_indices : torch.Tensor, shape (max_total_cells,), dtype=int32, optional
        Starting index in cell_atom_list for each cell.
        Pass a pre-allocated tensor to avoid reallocation for cell list construction.
        If None, allocated internally to build the cell list.
    cell_atom_list : torch.Tensor, shape (total_atoms,), dtype=int32, optional
        Flattened list of atom indices organized by cell.
        Pass a pre-allocated tensor to avoid reallocation for cell list construction.
        If None, allocated internally to build the cell list.

    Returns
    -------
    results : tuple of torch.Tensor
        Variable-length tuple depending on input parameters. The return pattern follows:

        - Matrix format (default): ``(neighbor_matrix, num_neighbors, neighbor_matrix_shifts)``
        - List format (return_neighbor_list=True): ``(neighbor_list, neighbor_ptr, neighbor_list_shifts)``

    Notes
    -----
    - This is the main user-facing API for cell list neighbor construction
    - Uses automatic memory allocation estimation for torch.compile compatibility
    - For advanced users who want to cache cell lists, use build_cell_list and query_cell_list separately
    - Returns appropriate empty tensors for systems with <= 1 atom or cutoff <= 0

    See Also
    --------
    nvalchemiops.neighbors.cell_list.build_cell_list : Core warp launcher for building
    nvalchemiops.neighbors.cell_list.query_cell_list : Core warp launcher for querying
    naive_neighbor_list : O(N²) method for small systems
    """
    total_atoms = positions.shape[0]
    device = positions.device
    if pbc is None:
        raise ValueError(
            "cell_list requires `pbc` to be specified. "
            "Pass a boolean tensor of shape (3,) or (1, 3), "
            "e.g. pbc=torch.tensor([True, True, True])."
        )
    cell = cell if cell.ndim == 3 else cell.unsqueeze(0)
    pbc = pbc.squeeze(0)

    if fill_value is None:
        fill_value = total_atoms

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
                torch.full(
                    (total_atoms, 0), fill_value, dtype=torch.int32, device=device
                ),
                torch.zeros((total_atoms,), dtype=torch.int32, device=device),
                torch.zeros((total_atoms, 0, 3), dtype=torch.int32, device=device),
            )

    if max_neighbors is None and (
        neighbor_matrix is None
        or neighbor_matrix_shifts is None
        or num_neighbors is None
    ):
        max_neighbors = estimate_max_neighbors(cutoff)

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
        max_total_cells, neighbor_search_radius = estimate_cell_list_sizes(
            cell, pbc, cutoff
        )
        cell_list_cache = allocate_cell_list(
            total_atoms,
            max_total_cells,
            neighbor_search_radius,
            device,
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

    build_cell_list(
        positions,
        cutoff,
        cell,
        pbc,
        *cell_list_cache,
    )

    query_cell_list(
        positions,
        cutoff,
        cell,
        pbc,
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
