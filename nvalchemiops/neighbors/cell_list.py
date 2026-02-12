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

"""Core warp kernels and launchers for cell list neighbor construction.

This module contains warp kernels for O(N) cell-based neighbor list computation.
See `nvalchemiops.torch.neighbors` for PyTorch bindings.
"""

from typing import Any

import warp as wp

from nvalchemiops.math import wpdivmod
from nvalchemiops.neighbors.neighbor_utils import (
    _update_neighbor_matrix_pbc,
    zero_array,
)

__all__ = [
    "build_cell_list",
    "query_cell_list",
]

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
    inverse_cell_transpose = wp.transpose(wp.inverse(cell[0]))
    fractional_position = inverse_cell_transpose * positions[atom_idx]

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
    inverse_cell_transpose = wp.transpose(wp.inverse(cell[0]))
    fractional_position = inverse_cell_transpose * positions[atom_idx]

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
    - Computes cell_transpose only when needed (late in the pipeline)
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

    # Load cell matrix once (compute transpose only when needed)
    cell_mat = cell[0]
    # OPTIMIZATION: Compute transpose ONCE outside the loop, not inside!
    cell_transpose = wp.transpose(cell_mat)

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

                    # Calculate Cartesian shift (using pre-computed transpose from line 449)
                    fractional_shift = type(central_atom_position)(
                        type(central_atom_position[0])(shift_x),
                        type(central_atom_position[0])(shift_y),
                        type(central_atom_position[0])(shift_z),
                    )
                    cartesian_shift = cell_transpose * fractional_shift

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
################################ Core Warp Launchers #######################################
###########################################################################################


def build_cell_list(
    positions: wp.array,
    cell: wp.array,
    pbc: wp.array,
    cutoff: float,
    cells_per_dimension: wp.array,
    atom_periodic_shifts: wp.array,
    atom_to_cell_mapping: wp.array,
    atoms_per_cell_count: wp.array,
    cell_atom_start_indices: wp.array,
    cell_atom_list: wp.array,
    wp_dtype: type,
    device: str,
) -> None:
    """Core warp launcher for building spatial cell list.

    Constructs a spatial decomposition data structure for efficient neighbor searching
    using pure warp operations. This function launches warp kernels to organize atoms
    into spatial cells.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Atomic coordinates in Cartesian space.
    cell : wp.array, shape (1, 3, 3), dtype=wp.mat33*
        Unit cell matrix defining the simulation box.
    pbc : wp.array, shape (3,), dtype=wp.bool
        Periodic boundary condition flags for x, y, z directions.
    cutoff : float
        Maximum distance for neighbor search.
    cells_per_dimension : wp.array, shape (3,), dtype=wp.int32
        OUTPUT: Number of cells created in x, y, z directions.
    atom_periodic_shifts : wp.array, shape (total_atoms, 3), dtype=wp.vec3i
        OUTPUT: Periodic boundary crossings for each atom.
    atom_to_cell_mapping : wp.array, shape (total_atoms, 3), dtype=wp.vec3i
        OUTPUT: 3D cell coordinates assigned to each atom.
    atoms_per_cell_count : wp.array, shape (max_total_cells,), dtype=wp.int32
        OUTPUT: Number of atoms in each cell. Must be zeroed by caller before first use.
    cell_atom_start_indices : wp.array, shape (max_total_cells,), dtype=wp.int32
        OUTPUT: Starting index in cell_atom_list for each cell's atoms.
        Must be filled with cumulative sums by caller between kernel launches.
    cell_atom_list : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Flattened list of atom indices organized by cell.
    wp_dtype : type
        Warp dtype (wp.float32, wp.float64, or wp.float16).
    device : str
        Warp device string (e.g., 'cuda:0', 'cpu').

    Notes
    -----
    - This is a low-level warp interface. Caller must ensure atoms_per_cell_count is zeroed.
    - atoms_per_cell_count must be zeroed before calling this function.
    - This function handles the cumsum internally using wp.utils.array_scan.
    - For framework bindings, use the torch/jax wrappers instead.

    See Also
    --------
    query_cell_list : Query cell list to build neighbor matrix (call after this)
    wp.utils.array_scan : Warp utility for computing prefix sums
    _cell_list_construct_bin_size : Kernel for computing cell dimensions
    _cell_list_count_atoms_per_bin : Kernel for counting atoms per cell
    _cell_list_bin_atoms : Kernel for binning atoms into cells
    """
    total_atoms = positions.shape[0]
    max_total_cells = atoms_per_cell_count.shape[0]
    wp_cutoff = wp_dtype(cutoff)

    # Construct cell dimensions
    wp.launch(
        _cell_list_construct_bin_size_overload[wp_dtype],
        dim=1,
        device=device,
        inputs=(
            cell,
            pbc,
            cells_per_dimension,
            wp_cutoff,
            max_total_cells,
        ),
    )

    # Count atoms per bin (expects atoms_per_cell_count to be zeroed by caller)
    wp.launch(
        _cell_list_count_atoms_per_bin_overload[wp_dtype],
        dim=total_atoms,
        inputs=[
            positions,
            cell,
            pbc,
            cells_per_dimension,
            atoms_per_cell_count,
            atom_periodic_shifts,
        ],
        device=device,
    )

    # Compute exclusive scan to get starting indices for each cell
    # This converts [3, 5, 2, 0, 4, ...] -> [0, 3, 8, 10, 10, ...]
    wp.utils.array_scan(atoms_per_cell_count, cell_atom_start_indices, inclusive=False)

    # Zero counts before binning atoms (second pass needs fresh counts)
    zero_array(atoms_per_cell_count, device)

    # Bin atoms (expects atoms_per_cell_count to be zeroed)
    wp.launch(
        _cell_list_bin_atoms_overload[wp_dtype],
        dim=total_atoms,
        inputs=[
            positions,
            cell,
            pbc,
            cells_per_dimension,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
        ],
        device=device,
    )


def query_cell_list(
    positions: wp.array,
    cell: wp.array,
    pbc: wp.array,
    cutoff: float,
    cells_per_dimension: wp.array,
    neighbor_search_radius: wp.array,
    atom_periodic_shifts: wp.array,
    atom_to_cell_mapping: wp.array,
    atoms_per_cell_count: wp.array,
    cell_atom_start_indices: wp.array,
    cell_atom_list: wp.array,
    neighbor_matrix: wp.array,
    neighbor_matrix_shifts: wp.array,
    num_neighbors: wp.array,
    wp_dtype: type,
    device: str,
    half_fill: bool = False,
) -> None:
    """Core warp launcher for querying spatial cell list to build neighbor matrix.

    Uses pre-built cell list data structures to efficiently find all atom pairs
    within the specified cutoff distance using pure warp operations.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Atomic coordinates in Cartesian space.
    cell : wp.array, shape (1, 3, 3), dtype=wp.mat33*
        Unit cell matrix for periodic boundary coordinate shifts.
    pbc : wp.array, shape (3,), dtype=wp.bool
        Periodic boundary condition flags.
    cutoff : float
        Maximum distance for considering atoms as neighbors.
    cells_per_dimension : wp.array, shape (3,), dtype=wp.int32
        Number of cells in x, y, z directions from build_cell_list_warp.
    neighbor_search_radius : wp.array, shape (3,), dtype=wp.int32
        Radius of neighboring cells to search in each dimension.
    atom_periodic_shifts : wp.array, shape (total_atoms, 3), dtype=wp.vec3i
        Periodic boundary crossings for each atom from build_cell_list_warp.
    atom_to_cell_mapping : wp.array, shape (total_atoms, 3), dtype=wp.vec3i
        3D cell coordinates for each atom from build_cell_list_warp.
    atoms_per_cell_count : wp.array, shape (max_total_cells,), dtype=wp.int32
        Number of atoms in each cell from build_cell_list_warp.
    cell_atom_start_indices : wp.array, shape (max_total_cells,), dtype=wp.int32
        Starting index in cell_atom_list for each cell from build_cell_list_warp.
    cell_atom_list : wp.array, shape (total_atoms,), dtype=wp.int32
        Flattened list of atom indices organized by cell from build_cell_list_warp.
    neighbor_matrix : wp.array, shape (total_atoms, max_neighbors), dtype=wp.int32
        OUTPUT: Neighbor matrix to be filled with neighbor atom indices.
    neighbor_matrix_shifts : wp.array, shape (total_atoms, max_neighbors, 3), dtype=wp.vec3i
        OUTPUT: Matrix storing shift vectors for each neighbor relationship.
    num_neighbors : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Number of neighbors found for each atom.
    wp_dtype : type
        Warp dtype (wp.float32, wp.float64, or wp.float16).
    device : str
        Warp device string (e.g., 'cuda:0', 'cpu').
    half_fill : bool, default=False
        If True, only store half of the neighbor relationships (i < j).

    Notes
    -----
    - This is a low-level warp interface. For framework bindings, use torch/jax wrappers.
    - Output arrays (neighbor_matrix, neighbor_matrix_shifts, num_neighbors) should be
      pre-allocated by caller.

    See Also
    --------
    build_cell_list : Build cell list data structures (call before this)
    _cell_list_build_neighbor_matrix : Kernel that performs the neighbor search
    """
    total_atoms = positions.shape[0]
    wp_cutoff = wp_dtype(cutoff)

    # Build neighbor matrix
    wp.launch(
        _cell_list_build_neighbor_matrix_overload[wp_dtype],
        dim=total_atoms,
        inputs=[
            positions,
            cell,
            pbc,
            wp_cutoff,
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
        ],
        device=device,
    )
