# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Any

import torch
import warp as wp

from nvalchemiops.math import wpdivmod
from nvalchemiops.neighborlist.neighbor_utils import (
    _update_neighbor_matrix_pbc,
    allocate_cell_list,
    estimate_max_neighbors,
    get_neighbor_list_from_neighbor_matrix,
)
from nvalchemiops.types import get_wp_dtype, get_wp_mat_dtype, get_wp_vec_dtype

###########################################################################################
########################### Cell List Construction ########################################
###########################################################################################


@wp.kernel(enable_backward=False)
def _estimate_cell_list_sizes(
    cell: wp.array(dtype=Any),
    pbc: wp.array(dtype=Any),
    cell_size: Any,
    max_nbins: Any,
    number_of_cells: wp.array(dtype=Any),
    neighbor_search_radius: wp.array(dtype=Any),
) -> None:
    """Estimate allocation sizes for torch.compile-friendly cell list construction.

    Parameters
    ----------
    cell : wp.array(dtype=Any), shape (1, 3, 3)
        Unit cell matrix defining the simulation box.
    pbc : wp.array(dtype=Any), shape (3,), dtype=bool
        Flags indicating periodic boundary conditions in x, y, z directions.
        True enables PBC, False disables it for that dimension.
    cell_size : Any
        Size of the cells in the simulation box.
    max_nbins : Any
        Maximum number of cells to allocate.
    number_of_cells : wp.array(dtype=Any), shape (1,)
        Output: Number of cells in the simulation box.
    neighbor_search_radius : wp.array(dtype=Any), shape (3,)
        Output: Radius of neighboring cells to search in each dimension.
    Notes
    -----
    - Thread launch: One thread per atom (dim=total_atoms)
    - Modifies: number_of_cells
    - Handles: periodic boundaries by wrapping and clamping
    """
    # Convert cell matrix to inverse transpose for coordinate transformations
    inverse_cell_transpose = wp.transpose(wp.inverse(cell[0]))

    cells_per_dimension = wp.vec3i(0, 0, 0)
    # Calculate optimal number of cells in each dimension
    for i in range(3):
        # Distance between parallel faces in reciprocal space
        face_distance = type(cell_size)(1.0) / wp.length(inverse_cell_transpose[i])
        cells_per_dimension[i] = max(wp.int32(face_distance / cell_size), 1)

        if cells_per_dimension[i] == 1 and not pbc[i]:
            neighbor_search_radius[i] = 0
        else:
            neighbor_search_radius[i] = wp.int32(
                wp.ceil(
                    cell_size * type(cell_size)(cells_per_dimension[i]) / face_distance
                )
            )

    # Check if total cell count exceeds maximum allowed
    total_cells = int(
        cells_per_dimension[0] * cells_per_dimension[1] * cells_per_dimension[2]
    )

    # Reduce cell count if necessary by halving dimensions iteratively
    while total_cells > max_nbins:
        for i in range(3):
            cells_per_dimension[i] = max(cells_per_dimension[i] // 2, 1)
        total_cells = int(
            cells_per_dimension[0] * cells_per_dimension[1] * cells_per_dimension[2]
        )

    number_of_cells[0] = total_cells


@wp.kernel(enable_backward=False)
def _cell_list_construct_bin_size(
    cell: wp.array(dtype=Any),
    pbc: wp.array(dtype=Any),
    cells_per_dimension: wp.array(dtype=Any),
    target_cell_size: Any,
    max_cells_allowed: Any,
) -> None:
    """Determine optimal spatial decomposition parameters for cell list construction.

    This kernel calculates the number of cells needed in each spatial dimension
    and the neighbor search radius based on the simulation cell geometry and
    target cell size. Assumes a single system (not batched).

    The algorithm:
    1. Computes optimal cell count per dimension based on cell geometry
    2. Reduces cell count if total exceeds maximum allowed
    3. Calculates neighbor search radius to ensure completeness

    Parameters
    ----------
    cell : wp.array, shape (1, 3, 3), dtype=wp.mat33*
        Unit cell matrix defining simulation box geometry.
    pbc : wp.array, shape (3,), dtype=bool
        Periodic boundary condition flags for x, y, z directions.
    cells_per_dimension : wp.array, shape (3,), dtype=wp.int32
        OUTPUT: Number of cells to create in x, y, z directions.
    target_cell_size : float
        Desired cell size, typically the neighbor cutoff distance.
    max_cells_allowed : int
        Maximum total number of cells allowed (nx * ny * nz ≤ max_cells_allowed).

    Notes
    -----
    - Modifies: cells_per_dimension, neighbor_search_radius
    - Thread launch: Single thread (dim=1)
    - For non-periodic directions with only 1 cell, search radius is set to 0
    """

    # Convert cell matrix to inverse transpose for coordinate transformations
    inverse_cell_transpose = wp.transpose(wp.inverse(cell[0]))

    # Calculate optimal number of cells in each dimension
    for i in range(3):
        # Distance between parallel faces in reciprocal space
        face_distance = type(target_cell_size)(1.0) / wp.length(
            inverse_cell_transpose[i]
        )
        cells_per_dimension[i] = max(wp.int32(face_distance / target_cell_size), 1)

    # Check if total cell count exceeds maximum allowed
    total_cells = int(
        cells_per_dimension[0] * cells_per_dimension[1] * cells_per_dimension[2]
    )

    # Reduce cell count if necessary by halving dimensions iteratively
    while total_cells > max_cells_allowed:
        for i in range(3):
            cells_per_dimension[i] = max(cells_per_dimension[i] // 2, 1)
        total_cells = int(
            cells_per_dimension[0] * cells_per_dimension[1] * cells_per_dimension[2]
        )


@wp.kernel(enable_backward=False)
def _cell_list_count_atoms_per_bin(
    positions: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    pbc: wp.array(dtype=Any),
    cells_per_dimension: wp.array(dtype=Any),
    atoms_per_cell_count: wp.array(dtype=Any),
    atom_periodic_shifts: wp.array(dtype=Any),
) -> None:
    """Count atoms in each spatial cell and compute periodic boundary shifts.

    This is the first pass of the two-pass cell list construction algorithm.
    Each thread processes one atom, determines which cell it belongs to,
    handles periodic boundary conditions, and atomically increments the
    atom count for that cell.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Atomic coordinates in Cartesian space.
    cell : wp.array, shape (1, 3, 3), dtype=wp.mat33*
        Unit cell matrix for coordinate transformations.
    pbc : wp.array, shape (3,), dtype=bool
        Periodic boundary condition flags for x, y, z directions.
    cells_per_dimension : wp.array, shape (3,), dtype=wp.int32
        Number of spatial cells in x, y, z directions.
    atoms_per_cell_count : wp.array, shape (total_cells,), dtype=wp.int32
        OUTPUT: Number of atoms assigned to each cell (modified atomically).
    atom_periodic_shifts : wp.array, shape (total_atoms, 3), dtype=wp.vec3i
        OUTPUT: Periodic boundary crossings for each atom.

    Notes
    -----
    - Thread launch: One thread per atom (dim=total_atoms)
    - Modifies: atoms_per_cell_count, atom_periodic_shifts
    - Uses atomic operations for thread-safe counting
    - Handles periodic boundaries by wrapping coordinates and tracking shifts
    """
    atom_idx = wp.tid()

    # Transform to fractional coordinates
    inverse_cell = wp.inverse(cell[0])
    fractional_position = positions[atom_idx] * inverse_cell

    # Determine which cell this atom belongs to
    cell_coords = wp.vec3i(0, 0, 0)
    for dim in range(3):
        cell_coords[dim] = wp.int32(
            wp.floor(
                fractional_position[dim]
                * type(fractional_position[dim])(cells_per_dimension[dim])
            )
        )

        # Handle periodic boundary conditions
        if pbc[dim]:
            cell_before_wrap = cell_coords[dim]
            num_cells = cells_per_dimension[dim]
            quotient, remainder = wpdivmod(cell_before_wrap, num_cells)
            atom_periodic_shifts[atom_idx][dim] = quotient
            cell_coords[dim] = remainder
        else:
            # Clamp to valid cell range for non-periodic dimensions
            atom_periodic_shifts[atom_idx][dim] = 0
            cell_coords[dim] = wp.clamp(
                cell_coords[dim], 0, cells_per_dimension[dim] - 1
            )

    # Convert 3D cell coordinates to linear index
    linear_cell_index = cell_coords[0] + cells_per_dimension[0] * (
        cell_coords[1] + cells_per_dimension[1] * cell_coords[2]
    )

    # Atomically increment the count for this cell
    wp.atomic_add(atoms_per_cell_count, linear_cell_index, 1)


@wp.kernel(enable_backward=False)
def _cell_list_compute_cell_offsets(
    atoms_per_cell_count: wp.array(dtype=wp.int32),
    cell_atom_start_indices: wp.array(dtype=wp.int32),
    total_cells: int,
) -> None:
    """Compute exclusive prefix sum to determine starting indices for each cell.

    This kernel calculates where each cell's atom list begins in the flattened
    cell_atom_indices array. Uses an exclusive prefix sum so that cell i starts
    at index cell_atom_start_indices[i] and contains atoms_per_cell_count[i] atoms.

    Parameters
    ----------
    atoms_per_cell_count : wp.array, shape (total_cells,), dtype=wp.int32
        Number of atoms assigned to each cell.
    cell_atom_start_indices : wp.array, shape (total_cells,), dtype=wp.int32
        OUTPUT: Starting index in cell_atom_indices array for each cell.
    total_cells : int
        Total number of cells in the spatial decomposition.

    Notes
    -----
    - Thread launch: One thread per cell (dim=total_cells)
    - Modifies: cell_atom_start_indices
    - This is a simple O(n²) prefix sum implementation suitable for small arrays
    - For large arrays, a more efficient parallel prefix sum would be preferred
    """
    cell_idx = wp.tid()
    if cell_idx < total_cells:
        running_sum = wp.int32(0)
        for i in range(cell_idx):
            running_sum += atoms_per_cell_count[i]
        cell_atom_start_indices[cell_idx] = running_sum


@wp.kernel(enable_backward=False)
def _cell_list_bin_atoms(
    positions: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    pbc: wp.array(dtype=Any),
    cells_per_dimension: wp.array(dtype=Any),
    atom_to_cell_mapping: wp.array(dtype=Any),
    atoms_per_cell_count: wp.array(dtype=Any),
    cell_atom_start_indices: wp.array(dtype=Any),
    cell_atom_list: wp.array(dtype=Any),
) -> None:
    """Assign atoms to their spatial cells and build cell-to-atom mapping.

    This is the second pass of the two-pass cell list construction algorithm.
    Each thread processes one atom, determines its cell assignment, and adds
    it to that cell's atom list using atomic operations for thread safety.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Atomic coordinates in Cartesian space.
    cell : wp.array, shape (1, 3, 3), dtype=wp.mat33*
        Unit cell matrix for coordinate transformations.
    pbc : wp.array, shape (3,), dtype=bool
        Periodic boundary condition flags for x, y, z directions.
    cells_per_dimension : wp.array, shape (3,), dtype=wp.int32
        Number of spatial cells in x, y, z directions.
    atom_to_cell_mapping : wp.array, shape (total_atoms, 3), dtype=wp.vec3i
        OUTPUT: 3D cell coordinates for each atom.
    atoms_per_cell_count : wp.array, shape (total_cells,), dtype=wp.int32
        MODIFIED: Running count of atoms added to each cell (reset before use).
    cell_atom_start_indices : wp.array, shape (total_cells,), dtype=wp.int32
        Starting index in cell_atom_list for each cell's atoms.
    cell_atom_list : wp.array, shape (total_cells,), dtype=wp.int32
        OUTPUT: Flattened list of atom indices organized by cell.

    Notes
    -----
    - Thread launch: One thread per atom (dim=total_atoms)
    - Modifies: atom_to_cell_mapping, atoms_per_cell_count, cell_atom_list
    - atoms_per_cell_count must be zeroed before calling this kernel
    - Uses atomic operations for thread-safe list building
    """
    atom_idx = wp.tid()

    # Safety check for thread bounds
    if atom_idx >= positions.shape[0]:
        return

    # Transform to fractional coordinates
    inverse_cell = wp.inverse(cell[0])
    fractional_position = positions[atom_idx] * inverse_cell

    # Determine which cell this atom belongs to
    cell_coords = wp.vec3i(0, 0, 0)
    for dim in range(3):
        cell_coords[dim] = wp.int32(
            wp.floor(
                fractional_position[dim]
                * type(fractional_position[dim])(cells_per_dimension[dim])
            )
        )

        # Handle periodic boundary conditions
        if pbc[dim]:
            cell_before_wrap = cell_coords[dim]
            num_cells = cells_per_dimension[dim]
            _, remainder = wpdivmod(cell_before_wrap, num_cells)
            cell_coords[dim] = remainder
        else:
            # Clamp to valid cell range for non-periodic dimensions
            cell_coords[dim] = wp.clamp(
                cell_coords[dim], 0, cells_per_dimension[dim] - 1
            )

    # Store the cell assignment for this atom
    atom_to_cell_mapping[atom_idx] = cell_coords

    # Convert 3D cell coordinates to linear index
    linear_cell_index = cell_coords[0] + cells_per_dimension[0] * (
        cell_coords[1] + cells_per_dimension[1] * cell_coords[2]
    )

    # Atomically get position in this cell's atom list
    position_in_cell = wp.atomic_add(atoms_per_cell_count, linear_cell_index, 1)

    # Calculate final position in flattened atom list
    final_list_index = cell_atom_start_indices[linear_cell_index] + position_in_cell

    # Store this atom's index in the cell's atom list
    cell_atom_list[final_list_index] = atom_idx


@wp.kernel(enable_backward=False)
def _cell_list_build_neighbor_matrix(
    positions: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    pbc: wp.array(dtype=bool),
    cutoff: Any,
    cells_per_dimension: wp.array(dtype=wp.int32),
    neighbor_search_radius: wp.array(dtype=wp.int32),
    atom_periodic_shifts: wp.array(dtype=wp.vec3i),
    atom_to_cell_mapping: wp.array(dtype=wp.vec3i),
    atoms_per_cell_count: wp.array(dtype=wp.int32),
    cell_atom_start_indices: wp.array(dtype=wp.int32),
    cell_atom_list: wp.array(dtype=wp.int32),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts: wp.array(dtype=Any, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    half_fill: bool,
) -> None:
    """Build neighbor matrix with atom pairs and periodic shifts.

    For each atom, searches through neighboring cells and records all neighbor
    atoms within the cutoff distance into a fixed-size matrix format. Stores
    neighbor indices and their periodic shift vectors.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Atomic coordinates in Cartesian space.
    cell : wp.array, shape (1, 3, 3), dtype=wp.mat33*
        Unit cell matrix for periodic boundary coordinate shifts.
    pbc : wp.array, shape (3,), dtype=bool
        Periodic boundary condition flags.
    cutoff : float
        Maximum distance for considering atoms as neighbors.
    cells_per_dimension : wp.array, shape (3,), dtype=wp.int32
        Number of spatial cells in x, y, z directions.
    neighbor_search_radius : wp.array, shape (3,), dtype=wp.int32
        Radius of neighboring cells to search in each dimension.
    atom_periodic_shifts : wp.array, shape (total_atoms, 3), dtype=wp.vec3i
        Periodic boundary crossings for each atom.
    atom_to_cell_mapping : wp.array, shape (total_atoms, 3), dtype=wp.vec3i
        3D cell coordinates for each atom.
    atoms_per_cell_count : wp.array, shape (total_cells,), dtype=wp.int32
        Number of atoms in each cell.
    cell_atom_start_indices : wp.array, shape (total_cells,), dtype=wp.int32
        Starting index in cell_atom_list for each cell.
    cell_atom_list : wp.array, shape (total_atoms,), dtype=wp.int32
        Flattened list of atom indices organized by cell.
    neighbor_matrix : wp.array, shape (total_atoms, max_neighbors), dtype=wp.int32
        OUTPUT: Neighbor matrix to be filled with neighbor atom indices.
    neighbor_matrix_shifts : wp.array, shape (total_atoms, max_neighbors, 3), dtype=wp.vec3i
        OUTPUT: Shift vectors for each neighbor relationship.
    num_neighbors : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Number of neighbors found for each atom.

    Notes
    -----
    - Thread launch: One thread per atom (dim=total_atoms)
    - Each thread loops over all neighbor cell shifts internally
    - Modifies: neighbor_matrix, neighbor_matrix_shifts, num_neighbors
    - If max_neighbors is exceeded for an atom, extra neighbors are ignored

    Performance Optimizations:
    - Uses cutoff squared to avoid expensive sqrt operations
    - Caches cells_per_dimension and pbc in registers to reduce memory access
    - Uses scalar variables instead of vec3 where possible to reduce register pressure
    - Unrolls PBC boundary checks for better branch prediction
    - Explicitly computes distance components to enable vectorization
    """
    atom_idx = wp.tid()

    # Precompute cutoff squared to avoid sqrt in distance checks
    cutoff_distance_sq = cutoff * cutoff
    central_atom_position = positions[atom_idx]
    central_atom_cell = atom_to_cell_mapping[atom_idx]
    central_atom_shift = atom_periodic_shifts[atom_idx]
    max_neighbors = neighbor_matrix.shape[1]

    # Load cell matrix once
    cell_mat = cell[0]

    # Cache cells_per_dimension in registers (small, accessed frequently)
    cpd_x = cells_per_dimension[0]
    cpd_y = cells_per_dimension[1]
    cpd_z = cells_per_dimension[2]

    # Cache pbc flags in registers
    pbc_x = pbc[0]
    pbc_y = pbc[1]
    pbc_z = pbc[2]

    # Loop through all neighbor cell shifts
    for dx in range(0, neighbor_search_radius[0] + 1):
        for dy in range(-neighbor_search_radius[1], neighbor_search_radius[1] + 1):
            for dz in range(-neighbor_search_radius[2], neighbor_search_radius[2] + 1):
                if not (
                    dx > 0 or (dx == 0 and dy > 0) or (dx == 0 and dy == 0 and dz >= 0)
                ):
                    continue
                # Compute target cell coordinates
                target_x = central_atom_cell[0] + dx
                target_y = central_atom_cell[1] + dy
                target_z = central_atom_cell[2] + dz

                # For non-PBC dimensions, skip cells outside the valid range
                # Unrolled for better branch prediction
                if not pbc_x and (target_x < 0 or target_x >= cpd_x):
                    continue
                if not pbc_y and (target_y < 0 or target_y >= cpd_y):
                    continue
                if not pbc_z and (target_z < 0 or target_z >= cpd_z):
                    continue

                # Compute cell shift and wrapped cell coordinates (inline wpdivmod)
                cs_x, wc_x = wpdivmod(target_x, cpd_x)
                cs_y, wc_y = wpdivmod(target_y, cpd_y)
                cs_z, wc_z = wpdivmod(target_z, cpd_z)

                # Convert to linear cell index
                linear_cell_index = wc_x + cpd_x * (wc_y + cpd_y * wc_z)

                # Get atom range for this cell
                cell_start_index = cell_atom_start_indices[linear_cell_index]
                num_atoms_in_cell = atoms_per_cell_count[linear_cell_index]

                # Check each atom in this neighboring cell
                for cell_atom_idx in range(num_atoms_in_cell):
                    neighbor_atom_idx = cell_atom_list[cell_start_index + cell_atom_idx]

                    # Get neighbor's periodic shift
                    neighbor_atom_shift = atom_periodic_shifts[neighbor_atom_idx]

                    # Calculate unit cell shift (reuse variables to reduce register pressure)
                    # Apply PBC: add relative shift only for periodic dimensions
                    shift_x = cs_x
                    shift_y = cs_y
                    shift_z = cs_z

                    if pbc_x:
                        shift_x += central_atom_shift[0] - neighbor_atom_shift[0]
                    else:
                        shift_x = 0

                    if pbc_y:
                        shift_y += central_atom_shift[1] - neighbor_atom_shift[1]
                    else:
                        shift_y = 0

                    if pbc_z:
                        shift_z += central_atom_shift[2] - neighbor_atom_shift[2]
                    else:
                        shift_z = 0

                    # For home cell (dx=dy=dz=0), only process j > i
                    # to avoid double counting
                    if dx == 0 and dy == 0 and dz == 0:
                        if neighbor_atom_idx <= atom_idx:
                            continue

                    # Calculate Cartesian shift
                    fractional_shift = type(central_atom_position)(
                        type(central_atom_position[0])(shift_x),
                        type(central_atom_position[0])(shift_y),
                        type(central_atom_position[0])(shift_z),
                    )
                    cartesian_shift = fractional_shift * cell_mat

                    # Calculate distance squared
                    neighbor_pos = positions[neighbor_atom_idx]
                    dr = neighbor_pos - central_atom_position + cartesian_shift
                    distance_sq = wp.dot(dr, dr)

                    if distance_sq < cutoff_distance_sq:
                        # Store neighbor in matrix if space available

                        _update_neighbor_matrix_pbc(
                            atom_idx,
                            neighbor_atom_idx,
                            neighbor_matrix,
                            neighbor_matrix_shifts,
                            num_neighbors,
                            wp.vec3i(shift_x, shift_y, shift_z),
                            max_neighbors,
                            half_fill,
                        )


T = [wp.float32, wp.float64]
V = [wp.vec3f, wp.vec3d]
M = [wp.mat33f, wp.mat33d]
_estimate_cell_list_sizes_overload = {}
_cell_list_construct_bin_size_overload = {}
_cell_list_count_atoms_per_bin_overload = {}
_cell_list_bin_atoms_overload = {}
_cell_list_build_neighbor_matrix_overload = {}
for t, v, m in zip(T, V, M):
    _estimate_cell_list_sizes_overload[t] = wp.overload(
        _estimate_cell_list_sizes,
        [
            wp.array(dtype=m),
            wp.array(dtype=wp.bool),
            t,
            wp.int32,
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32),
        ],
    )
    _cell_list_construct_bin_size_overload[t] = wp.overload(
        _cell_list_construct_bin_size,
        [
            wp.array(dtype=m),
            wp.array(dtype=wp.bool),
            wp.array(dtype=wp.int32),
            t,
            wp.int32,
        ],
    )
    _cell_list_count_atoms_per_bin_overload[t] = wp.overload(
        _cell_list_count_atoms_per_bin,
        [
            wp.array(dtype=v),
            wp.array(dtype=m),
            wp.array(dtype=wp.bool),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.vec3i),
        ],
    )
    _cell_list_bin_atoms_overload[t] = wp.overload(
        _cell_list_bin_atoms,
        [
            wp.array(dtype=v),
            wp.array(dtype=m),
            wp.array(dtype=wp.bool),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.vec3i),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32),
        ],
    )
    _cell_list_build_neighbor_matrix_overload[t] = wp.overload(
        _cell_list_build_neighbor_matrix,
        [
            wp.array(dtype=v),
            wp.array(dtype=m),
            wp.array(dtype=wp.bool),
            t,
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.vec3i),
            wp.array(dtype=wp.vec3i),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32),
            wp.array2d(dtype=wp.int32),
            wp.array2d(dtype=wp.vec3i),
            wp.array(dtype=wp.int32),
            wp.bool,
        ],
    )

###########################################################################################
################################ Cell List Pytorch Wrapper ################################
###########################################################################################


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
    recieved from using torch.Tensor.item()

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
    max_atoms_per_cell : int
        Estimated maximum atoms that could be assigned to any single cell.
        Assumes roughly uniform distribution with safety margins.
    neighbor_search_radius : torch.Tensor, shape (3,), dtype=int32
        Radius of neighboring cells to search in each dimension.

    Notes
    -----
    Cell size is determined by the cutoff distance to ensure neighboring
    cells contain all potential neighbors. The estimation assumes roughly
    cubic cells and uniform atomic distribution.
    """
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
        neighbor_search_radius, dtype=wp.vec3i, return_ctype=True
    )
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
        "neighbor_search_radius",
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
    neighbor_search_radius: torch.Tensor,
    atom_periodic_shifts: torch.Tensor,
    atom_to_cell_mapping: torch.Tensor,
    atoms_per_cell_count: torch.Tensor,
    cell_atom_start_indices: torch.Tensor,
    cell_atom_list: torch.Tensor,
) -> None:
    """Internal custom op for building spatial cell list.

    Constructs a spatial decomposition data structure for efficient neighbor searching.
    Uses fixed-size memory allocations to prevent dynamic tensor creation that would
    cause graph breaks in torch.compile. Returns individual tensor components rather
    than a structured object for custom operator compatibility.

    Parameters
    ----------
    positions : torch.Tensor, shape (total_atoms, 3)
        Atomic coordinates in Cartesian space where total_atoms is the number of atoms.
        Must be float32, float64, or float16 dtype.
    cell : torch.Tensor, shape (1, 3, 3)
        Unit cell matrix defining the simulation box. Each row represents a
        lattice vector in Cartesian coordinates. Must match positions dtype.
    pbc : torch.Tensor, shape (3,), dtype=bool
        Flags indicating periodic boundary conditions in x, y, z directions.
        True enables PBC, False disables it for that dimension.
    cutoff : float
        Maximum distance for neighbor search. Determines minimum cell size.
    cells_per_dimension : torch.Tensor, shape (3,), dtype=int32
        OUTPUT: Number of cells created in x, y, z directions.
    neighbor_search_radius : torch.Tensor, shape (3,), dtype=int32
        OUTPUT: Shifts to search in each dimension.
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
    - Memory usage is determined by max_total_cells * max_atoms_per_cell
    - For optimal performance, use estimates from estimate_cell_list_sizes()
    - Cell list must be rebuilt when atoms move between cells or PBC/cell changes
    """
    total_atoms = positions.shape[0]
    device = positions.device
    dtype = positions.dtype

    # Handle empty case
    if total_atoms == 0 or cutoff <= 0:
        return

    cell = cell if cell.ndim == 3 else cell.unsqueeze(0)
    pbc = pbc.squeeze(0)

    # Get warp dtypes
    wp_dtype = get_wp_dtype(dtype)
    wp_vec_dtype = get_wp_vec_dtype(dtype)
    wp_mat_dtype = get_wp_mat_dtype(dtype)

    # Convert to warp arrays
    wp_positions = wp.from_torch(positions, dtype=wp_vec_dtype, return_ctype=True)
    wp_cell = wp.from_torch(cell, dtype=wp_mat_dtype, return_ctype=True)
    wp_pbc = wp.from_torch(pbc, dtype=wp.bool, return_ctype=True)
    wp_cutoff = wp_dtype(cutoff)
    wp_device = str(device)

    # Construct cell dimensions
    max_total_cells = atoms_per_cell_count.shape[0]
    wp_cells_per_dimension = wp.from_torch(
        cells_per_dimension, dtype=wp.vec3i, return_ctype=True
    )
    wp.launch(
        _cell_list_construct_bin_size_overload[wp_dtype],
        dim=1,
        device=wp_device,
        inputs=(
            wp_cell,
            wp_pbc,
            wp_cells_per_dimension,
            wp_cutoff,
            max_total_cells,
        ),
    )

    # Use user-specified fixed sizes instead of dynamic computation
    # These should be >= the actual needed sizes
    wp_atoms_per_cell_count = wp.from_torch(
        atoms_per_cell_count, dtype=wp.int32, return_ctype=True
    )
    wp_atom_periodic_shifts = wp.from_torch(
        atom_periodic_shifts, dtype=wp.vec3i, return_ctype=True
    )
    wp_atom_to_cell_mapping = wp.from_torch(
        atom_to_cell_mapping, dtype=wp.vec3i, return_ctype=True
    )

    # Fixed allocation for cell_atom_list using user-specified size
    wp_cell_atom_list = wp.from_torch(cell_atom_list, dtype=wp.int32, return_ctype=True)
    wp_cell_atom_start_indices = wp.from_torch(
        cell_atom_start_indices, dtype=wp.int32, return_ctype=True
    )

    # Count atoms per bin
    atoms_per_cell_count.zero_()
    wp.launch(
        _cell_list_count_atoms_per_bin_overload[wp_dtype],
        dim=total_atoms,
        inputs=[
            wp_positions,
            wp_cell,
            wp_pbc,
            wp_cells_per_dimension,
            wp_atoms_per_cell_count,
            wp_atom_periodic_shifts,
        ],
        device=wp_device,
    )

    # Compute offsets properly (like original implementation)
    cell_atom_start_indices[0] = 0
    if max_total_cells > 1:
        torch.cumsum(atoms_per_cell_count[:-1], dim=0, out=cell_atom_start_indices[1:])

    # Reset counts and bin atoms
    atoms_per_cell_count.zero_()
    wp.launch(
        _cell_list_bin_atoms_overload[wp_dtype],
        dim=total_atoms,
        inputs=[
            wp_positions,
            wp_cell,
            wp_pbc,
            wp_cells_per_dimension,
            wp_atom_to_cell_mapping,
            wp_atoms_per_cell_count,
            wp_cell_atom_start_indices,
            wp_cell_atom_list,
        ],
        device=wp_device,
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

    Uses pre-built cell list data structures to efficiently find all atom pairs
    within the specified cutoff distance. Handles periodic boundary conditions
    and returns neighbor matrix format.

    This function is torch compilable.

    Parameters
    ----------
    positions : torch.Tensor, shape (total_atoms, 3)
        Atomic coordinates in Cartesian space.
    cell : torch.Tensor, shape (1, 3, 3)
        Unit cell matrix for periodic boundary coordinate shifts.
    pbc : torch.Tensor, shape (3,), dtype=bool
        Periodic boundary condition flags.
    cutoff : float
        Maximum distance for considering atoms as neighbors.
    cells_per_dimension : torch.Tensor, shape (3,), dtype=int32
        Number of cells in x, y, z directions from build_cell_list.
    atom_periodic_shifts : torch.Tensor, shape (total_atoms, 3), dtype=int32
        Periodic boundary crossings for each atom from build_cell_list.
    neighbor_search_radius : torch.Tensor, shape (3,), dtype=int32
        Shifts to search from build_cell_list.
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

    Returns
    -------
    None
        This function modifies the input tensors in-place:

        - neighbor_matrix : Filled with neighbor atom indices
        - neighbor_matrix_shifts : Filled with corresponding shift vectors
        - num_neighbors : Updated with neighbor counts per atom
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
    wp_neighbor_search_radius = wp.from_torch(
        neighbor_search_radius, dtype=wp.int32, return_ctype=True
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

    # Build neighbor matrix
    wp.launch(
        _cell_list_build_neighbor_matrix_overload[wp_dtype],
        dim=total_atoms,
        inputs=[
            wp_positions,
            wp_cell,
            wp_pbc,
            wp_dtype(cutoff),
            wp_cells_per_dimension,
            wp_neighbor_search_radius,
            wp_atom_periodic_shifts,
            wp_atom_to_cell_mapping,
            wp_atoms_per_cell_count,
            wp_cell_atom_start_indices,
            wp_cell_atom_list,
            wp_neighbor_matrix,
            wp_neighbor_matrix_shifts,
            wp_num_neighbors,
            half_fill,
        ],
        device=wp_device,
    )


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
    cause graph breaks in torch.compile. Returns individual tensor components rather
    than a structured object for custom operator compatibility.

    Parameters
    ----------
    positions : torch.Tensor, shape (total_atoms, 3)
        Atomic coordinates in Cartesian space where total_atoms is the number of atoms.
        Must be float32, float64, or float16 dtype.
    cell : torch.Tensor, shape (1, 3, 3)
        Unit cell matrix defining the simulation box. Each row represents a
        lattice vector in Cartesian coordinates. Must match positions dtype.
    pbc : torch.Tensor, shape (3,), dtype=bool
        Flags indicating periodic boundary conditions in x, y, z directions.
        True enables PBC, False disables it for that dimension.
    cutoff : float
        Maximum distance for neighbor search. Determines minimum cell size.
    cells_per_dimension : torch.Tensor, shape (3,), dtype=int32
        OUTPUT: Number of cells created in x, y, z directions.
    neighbor_search_radius : torch.Tensor, shape (3,), dtype=int32
        OUTPUT: Shifts to search in each dimension.
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
    - Memory usage is determined by max_total_cells * max_atoms_per_cell
    - For optimal performance, use estimates from estimate_cell_list_sizes()
    - Cell list must be rebuilt when atoms move between cells or PBC/cell changes
    """
    return _build_cell_list_op(
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
    cell : torch.Tensor, shape (1, 3, 3)
        Unit cell matrix for periodic boundary coordinate shifts.
    pbc : torch.Tensor, shape (3,), dtype=bool
        Periodic boundary condition flags.
    cutoff : float
        Maximum distance for considering atoms as neighbors.
    cells_per_dimension : torch.Tensor, shape (3,), dtype=int32
        Number of cells in x, y, z directions from build_cell_list.
    atom_periodic_shifts : torch.Tensor, shape (total_atoms, 3), dtype=int32
        Periodic boundary crossings for each atom from build_cell_list.
    neighbor_search_radius : torch.Tensor, shape (3,), dtype=int32
        Shifts to search from build_cell_list.
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

    Returns
    -------
    None
        This function modifies the input tensors in-place:

        - neighbor_matrix : Filled with neighbor atom indices
        - neighbor_matrix_shifts : Filled with corresponding shift vectors
        - num_neighbors : Updated with neighbor counts per atom
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
    cell : torch.Tensor, shape (1, 3, 3)
        Unit cell matrix defining the simulation box. Each row represents a
        lattice vector in Cartesian coordinates.
    pbc : torch.Tensor, shape (1, 3), dtype=bool
        Flags indicating periodic boundary conditions in x, y, z directions.
    cutoff : float
        Maximum distance for neighbor search.
    max_neighbors : int, optional
        Maximum number of neighbors per atom. If not provided, will be estimated automatically.
    half_fill : bool, optional
        If True, only fill half of the neighbor matrix. Default is True.
    fill_value : int | None, optional
        Value to fill the neighbor matrix with. Default is -1.
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
    cells_per_dimension : torch.Tensor, shape (3,), dtype=int32
        Number of cells in x, y, z directions.
        Pass a pre-allocated tensor to avoid reallocation for cell list construction.
        If None, allocated internally to build the cell list.
    neighbor_search_radius : torch.Tensor, shape (3,), dtype=int32
        Radius of neighboring cells to search in each dimension.
        Pass a pre-allocated tensor to avoid reallocation for cell list construction.
        If None, allocated internally to build the cell list.
    atom_periodic_shifts : torch.Tensor, shape (total_atoms, 3), dtype=int32
        Periodic boundary crossings for each atom.
        Pass a pre-allocated tensor to avoid reallocation for cell list construction.
        If None, allocated internally to build the cell list.
    atom_to_cell_mapping : torch.Tensor, shape (total_atoms, 3), dtype=int32
        Cell coordinates for each atom.
        Pass a pre-allocated tensor to avoid reallocation for cell list construction.
        If None, allocated internally to build the cell list.
    atoms_per_cell_count : torch.Tensor, shape (max_total_cells,), dtype=int32
        Number of atoms in each cell.
        Pass a pre-allocated tensor to avoid reallocation for cell list construction.
        If None, allocated internally to build the cell list.
    cell_atom_start_indices : torch.Tensor, shape (max_total_cells,), dtype=int32
        Starting index in cell_atom_list for each cell.
        Pass a pre-allocated tensor to avoid reallocation for cell list construction.
        If None, allocated internally to build the cell list.
    cell_atom_list : torch.Tensor, shape (total_atoms,), dtype=int32
        Flattened list of atom indices organized by cell.
        Pass a pre-allocated tensor to avoid reallocation for cell list construction.
        If None, allocated internally to build the cell list.
    Returns
    -------
    results : tuple of torch.Tensor
        Variable-length tuple depending on input parameters. The return pattern follows:

        - Matrix format (default): ``(neighbor_matrix, num_neighbors, neighbor_matrix_shifts)``
        - List format (return_neighbor_list=True): ``(neighbor_list, neighbor_ptr, neighbor_list_shifts)``

        **Components returned:**

        - **neighbor_data** (tensor): Neighbor indices, format depends on ``return_neighbor_list``:

            * If ``return_neighbor_list=False`` (default): Returns ``neighbor_matrix``
              with shape (total_atoms, max_neighbors), dtype int32. Each row i contains
              indices of atom i's neighbors, padded with fill_value.
            * If ``return_neighbor_list=True``: Returns ``neighbor_list`` with shape
              (2, num_pairs), dtype int32, in COO format [source_atoms, target_atoms].

        - **num_neighbor_data** (tensor): Information about the number of neighbors for each atom,
          format depends on ``return_neighbor_list``:

            * If ``return_neighbor_list=False`` (default): Returns ``num_neighbors`` with shape (total_atoms,), dtype int32.
              Count of neighbors found for each atom.
            * If ``return_neighbor_list=True``: Returns ``neighbor_ptr`` with shape (total_atoms + 1,), dtype int32.
              CSR-style pointer arrays where ``neighbor_ptr_data[i]`` to ``neighbor_ptr_data[i+1]`` gives the range of
              neighbors for atom i in the flattened neighbor list.

        - **neighbor_shift_data** (tensor, optional): Periodic shift vectors for each neighbor,
          format depends on ``return_neighbor_list`` and only returned when ``pbc`` is provided:

            * If ``return_neighbor_list=False`` (default): Returns ``neighbor_matrix_shifts`` with
              shape (total_atoms, max_neighbors, 3), dtype int32.
            * If ``return_neighbor_list=True``: Returns ``unit_shifts`` with shape
              (num_pairs, 3), dtype int32.

    Notes
    -----
    - This is the main user-facing API for neighbor list construction
    - Uses automatic memory allocation estimation for torch.compile compatibility
    - For advanced users who want to cache cell lists, use build_cell_list and query_cell_list separately
    - Returns appropriate empty tensors for systems with <= 1 atom or cutoff <= 0
    """
    total_atoms = positions.shape[0]
    device = positions.device
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

    # Call query_cell_list
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
