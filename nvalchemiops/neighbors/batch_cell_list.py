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

"""Core warp kernels and launchers for batched cell list neighbor construction.

This module contains warp kernels for batched O(N) cell-based neighbor list computation.
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
    "batch_build_cell_list",
    "batch_query_cell_list",
]

###########################################################################################
########################### Batch Cell List Construction ##################################
###########################################################################################


@wp.kernel(enable_backward=False)
def _batch_estimate_cell_list_sizes(
    cell: wp.array(dtype=Any),
    pbc: wp.array2d(dtype=Any),
    cell_size: Any,
    max_nbins: Any,
    number_of_cells: wp.array(dtype=Any),
    neighbor_search_radius: wp.array(dtype=Any),
) -> None:
    """
    Estimate the number of cells and neighbor search radius for a batch of systems.

    Parameters
    ----------
    cell : wp.array, shape (num_systems, 3, 3), dtype=wp.mat33*
        Unit cell matrices for each system in the batch.
    pbc : wp.array2d, shape (num_systems, 3), dtype=bool
        Periodic boundary condition flags for each system and dimension.
    cell_size : Any
        Size of the cells, usually the neighbor cutoff distance in the simulation box.
    max_nbins : Any
        Maximum number of cells to allocate.
    number_of_cells : wp.array, shape (num_systems,), dtype=wp.int32
        OUTPUT: Number of cells in each system.
    neighbor_search_radius : wp.array, shape (num_systems, 3), dtype=wp.vec3i
        OUTPUT: Radius of neighboring cells to search in each dimension for each system.

    Notes
    -----
    - Thread launch: One thread per system (dim=num_systems)
    - Modifies: number_of_cells, neighbor_search_radius
    - Each thread processes one complete system independently
    - For non-periodic directions with only 1 cell, search radius is set to 0
    """
    system_idx = wp.tid()
    system_cell_matrix = cell[system_idx]
    inverse_cell_transpose = wp.transpose(wp.inverse(system_cell_matrix))

    cells_per_dimension = wp.vec3i(0, 0, 0)
    # Calculate optimal number of cells in each dimension
    for i in range(3):
        # Distance between parallel faces in reciprocal space
        face_distance = type(cell_size)(1.0) / wp.length(inverse_cell_transpose[i])
        cells_per_dimension[i] = max(wp.int32(face_distance / cell_size), 1)

        if cells_per_dimension[i] == 1 and not pbc[system_idx, i]:
            neighbor_search_radius[system_idx][i] = 0
        else:
            neighbor_search_radius[system_idx][i] = wp.int32(
                wp.ceil(
                    cell_size * type(cell_size)(cells_per_dimension[i]) / face_distance
                )
            )

    total_cells_this_system = int(
        cells_per_dimension[0] * cells_per_dimension[1] * cells_per_dimension[2]
    )

    while total_cells_this_system > max_nbins:
        for dim in range(3):
            cells_per_dimension[dim] = max(cells_per_dimension[dim] // 2, 1)
        total_cells_this_system = int(
            cells_per_dimension[0] * cells_per_dimension[1] * cells_per_dimension[2]
        )
    number_of_cells[system_idx] = total_cells_this_system


@wp.kernel(enable_backward=False)
def _batch_cell_list_construct_bin_size(
    cell: wp.array(dtype=Any),
    pbc: wp.array2d(dtype=Any),
    cells_per_dimension: wp.array(dtype=Any),
    target_cell_size: Any,
    max_total_cells: Any,
) -> None:
    """Determine optimal spatial decomposition parameters for batch cell list construction.

    This kernel processes multiple systems simultaneously, calculating
    the optimal number of cells and neighbor search radii for each system based
    on their individual cell geometries and target cell sizes.

    The algorithm for each system:
    1. Computes optimal cell count per dimension based on cell geometry
    2. Reduces cell count if total exceeds maximum allowed per system
    3. Calculates neighbor search radius to ensure neighbor completeness

    Parameters
    ----------
    cell : wp.array, shape (num_systems, 3, 3), dtype=wp.mat33*
        Unit cell matrix defining the simulation box.
    pbc : wp.array2d, shape (num_systems, 3), dtype=bool
        Periodic boundary condition flags for each system and dimension.
    cells_per_dimension : wp.array, shape (num_systems, 3), dtype=wp.vec3i
        OUTPUT: Number of cells to create in x, y, z directions for each system.
    target_cell_size : float
        Desired cell size for each system, typically the neighbor cutoff distance.
    max_total_cells : int
        Maximum total cells allowed (nx * ny * nz ≤ max_total_cells // num_systems).

    Notes
    -----
    - Thread launch: One thread per system (dim=num_systems)
    - Modifies: cells_per_dimension, batch_neighbor_search_radius
    - Each thread processes one complete system independently
    - For non-periodic directions with only 1 cell, search radius is set to 0
    """
    # Thread ID corresponds to system index in the batch
    system_idx = wp.tid()

    # Get cell matrix and target size for this system
    num_systems = cell.shape[0]
    s_cell_matrix = cell[system_idx]
    inverse_cell_transpose = wp.transpose(wp.inverse(s_cell_matrix))

    # Compute optimal number of cells in each dimension for this system
    for dim in range(3):
        # Distance between parallel faces in reciprocal space
        face_distance = type(target_cell_size)(1.0) / wp.length(
            inverse_cell_transpose[dim]
        )
        cells_per_dimension[system_idx][dim] = max(
            wp.int32(face_distance / target_cell_size), 1
        )

    # Check if total cell count exceeds maximum allowed for this system
    total_cells_this_system = int(
        cells_per_dimension[system_idx][0]
        * cells_per_dimension[system_idx][1]
        * cells_per_dimension[system_idx][2]
    )

    # Reduce cell count if necessary by halving dimensions iteratively
    while (total_cells_this_system * num_systems) > max_total_cells:
        for dim in range(3):
            cells_per_dimension[system_idx][dim] = max(
                cells_per_dimension[system_idx][dim] // 2, 1
            )
        total_cells_this_system = int(
            cells_per_dimension[system_idx][0]
            * cells_per_dimension[system_idx][1]
            * cells_per_dimension[system_idx][2]
        )


@wp.kernel(enable_backward=False)
def _batch_cell_list_count_atoms_per_bin(
    positions: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    pbc: wp.array2d(dtype=Any),
    batch_idx: wp.array(dtype=Any),
    cells_per_dimension: wp.array(dtype=Any),
    cell_offsets: wp.array(dtype=Any),
    atoms_per_cell_count: wp.array(dtype=Any),
    atom_periodic_shifts: wp.array(dtype=Any),
) -> None:
    """Count atoms in each spatial cell across batch systems and compute periodic shifts.

    This is the first pass of the two-pass batch cell list construction algorithm.
    Each thread processes one atom, determines which system and cell it belongs to,
    handles periodic boundary conditions, and atomically increments the atom count
    for that cell. Supports heterogeneous batches with different system sizes.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Concatenated atomic coordinates for all systems in the batch.
    cell : wp.array, shape (num_systems, 3, 3), dtype=wp.mat33*
        Unit cell matrices for each system in the batch.
    pbc : wp.array2d, shape (num_systems, 3), dtype=bool
        Periodic boundary condition flags for each system and dimension.
    batch_idx : wp.array, shape (total_atoms,), dtype=wp.int32
        System index for each atom.
    cells_per_dimension : wp.array, shape (num_systems, 3), dtype=wp.vec3i
        Number of cells in x, y, z directions for each system.
    cell_offsets : wp.array, shape (num_systems,), dtype=wp.int32
        Starting index in global cell arrays for each system (exclusive scan of cell counts).
    atoms_per_cell_count : wp.array, shape (total_cells,), dtype=wp.int32
        OUTPUT: Number of atoms assigned to each cell across all systems (modified atomically).
    atom_periodic_shifts : wp.array, shape (total_atoms, 3), dtype=wp.vec3i
        OUTPUT: Periodic boundary crossings for each atom.

    Notes
    -----
    - Thread launch: One thread per atom across all systems (dim=total_atoms)
    - Modifies: batch_atoms_per_cell_count, batch_atom_periodic_shifts
    - Uses atomic operations for thread-safe counting across batch
    - Each thread first determines which system it belongs to, then processes normally
    """
    atom_idx = wp.tid()

    # Find which system this atom belongs to using binary-like search
    system_idx = batch_idx[atom_idx]

    # Get system-specific parameters
    s_cell_matrix = cell[system_idx]
    s_cells_per_dimension = cells_per_dimension[system_idx]
    s_cell_offset = cell_offsets[system_idx]

    # Transform to fractional coordinates for this system
    inverse_cell = wp.inverse(s_cell_matrix)
    fractional_position = positions[atom_idx] * inverse_cell

    # Determine which cell this atom belongs to within its system
    cell_coords = wp.vec3i(0, 0, 0)
    for dim in range(3):
        cell_coords[dim] = wp.int32(
            wp.floor(
                fractional_position[dim]
                * type(fractional_position[dim])(s_cells_per_dimension[dim])
            )
        )

        # Handle periodic boundary conditions for this system
        if pbc[system_idx, dim]:
            cell_before_wrap = cell_coords[dim]
            num_cells_this_dim = s_cells_per_dimension[dim]
            quotient, remainder = wpdivmod(cell_before_wrap, num_cells_this_dim)
            atom_periodic_shifts[atom_idx][dim] = quotient
            cell_coords[dim] = remainder
        else:
            # Clamp to valid cell range for non-periodic dimensions
            atom_periodic_shifts[atom_idx][dim] = 0
            cell_coords[dim] = wp.clamp(
                cell_coords[dim], 0, s_cells_per_dimension[dim] - 1
            )

    # Compute linear cell index with system offset for global cell indexing
    global_linear_cell_index = (
        s_cell_offset
        + cell_coords[0]
        + s_cells_per_dimension[0]
        * (cell_coords[1] + s_cells_per_dimension[1] * cell_coords[2])
    )

    # Atomically increment the count for this cell across the entire batch
    wp.atomic_add(atoms_per_cell_count, global_linear_cell_index, 1)


@wp.kernel(enable_backward=False)
def _batch_cell_list_bin_atoms(
    positions: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    pbc: wp.array2d(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    cells_per_dimension: wp.array(dtype=Any),
    cell_offsets: wp.array(dtype=Any),
    atom_to_cell_mapping: wp.array(dtype=Any),
    atoms_per_cell_count: wp.array(dtype=Any),
    cell_atom_start_indices: wp.array(dtype=Any),
    cell_atom_list: wp.array(dtype=Any),
) -> None:
    """Assign atoms to cells and build cell-to-atom mapping for batch systems.

    This is the second pass of the two-pass batch cell list construction algorithm.
    Each thread processes one atom, determines which system and cell it belongs to,
    and adds it to that cell's atom list using atomic operations for thread safety.
    Supports heterogeneous batches with different system sizes.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Concatenated atomic coordinates for all systems in the batch.
    cell : wp.array, shape (num_systems,3, 3), dtype=wp.mat33*
        Unit cell matrices for each system in the batch.
    pbc : wp.array2d, shape (num_systems, 3), dtype=bool
        Periodic boundary condition flags for each system and dimension.
    batch_idx : wp.array, shape (total_atoms,), dtype=wp.int32
        Index of the system for each atom.
    cells_per_dimension : wp.array, shape (num_systems, 3), dtype=wp.vec3i
        Number of cells in x, y, z directions for each system.
    cell_offsets : wp.array, shape (num_systems,), dtype=wp.int32
        Starting index in global cell arrays for each system (exclusive scan of cell counts).
    atom_to_cell_mapping : wp.array, shape (total_atoms, 3), dtype=wp.vec3i
        OUTPUT: 3D cell coordinates assigned to each atom.
    atoms_per_cell_count : wp.array, shape (total_cells,), dtype=wp.int32
        MODIFIED: Running count of atoms added to each cell (reset before use).
    cell_atom_start_indices : wp.array, shape (total_cells,), dtype=wp.int32
        Starting index in cell_atom_list for each cell's atoms.
    cell_atom_list : wp.array, shape (total_cells,), dtype=wp.int32
        OUTPUT: Flattened list of atom indices organized by cell across all systems.

    Notes
    -----
    - Thread launch: One thread per atom across all systems (dim=total_atoms)
    - Modifies: atom_to_cell_mapping, atoms_per_cell_count, cell_atom_list
    - atoms_per_cell_count must be zeroed before calling this kernel
    - Uses atomic operations for thread-safe list building across batch
    """
    atom_idx = wp.tid()

    # Find which system this atom belongs to
    system_idx = batch_idx[atom_idx]

    # Get system-specific parameters
    s_cell_matrix = cell[system_idx]
    s_cells_per_dimension = cells_per_dimension[system_idx]
    s_cell_offset = cell_offsets[system_idx]

    # Transform to fractional coordinates
    inverse_cell = wp.inverse(s_cell_matrix)
    fractional_position = positions[atom_idx] * inverse_cell

    # Determine which cell this atom belongs to within its system
    cell_coords = wp.vec3i(0, 0, 0)
    for dim in range(3):
        cell_coords[dim] = wp.int32(
            wp.floor(
                fractional_position[dim]
                * type(fractional_position[dim])(s_cells_per_dimension[dim])
            )
        )

        # Handle periodic boundary conditions
        if pbc[system_idx, dim]:
            cell_before_wrap = cell_coords[dim]
            num_cells_this_dim = s_cells_per_dimension[dim]
            _, remainder = wpdivmod(cell_before_wrap, num_cells_this_dim)
            cell_coords[dim] = remainder
        else:
            # Clamp to valid cell range for non-periodic dimensions
            cell_coords[dim] = wp.clamp(
                cell_coords[dim], 0, s_cells_per_dimension[dim] - 1
            )

    # Store the cell assignment for this atom
    atom_to_cell_mapping[atom_idx] = cell_coords

    # Compute global linear cell index with system offset
    global_linear_cell_index = (
        s_cell_offset
        + cell_coords[0]
        + s_cells_per_dimension[0]
        * (cell_coords[1] + s_cells_per_dimension[1] * cell_coords[2])
    )

    # Atomically get position in this cell's atom list
    position_in_cell = wp.atomic_add(atoms_per_cell_count, global_linear_cell_index, 1)
    final_list_index = (
        cell_atom_start_indices[global_linear_cell_index] + position_in_cell
    )

    # Store this atom's index in the cell's atom list
    cell_atom_list[final_list_index] = atom_idx


@wp.kernel(enable_backward=False)
def _compute_cells_per_system(
    cells_per_dimension: wp.array(dtype=wp.vec3i),
    cells_per_system: wp.array(dtype=wp.int32),
) -> None:
    """Compute total cells per system from cell dimension vectors.

    For each system, computes the product of cells in x, y, z dimensions.
    Used as input to exclusive scan for computing cell offsets.

    Parameters
    ----------
    cells_per_dimension : wp.array, shape (num_systems, 3), dtype=wp.vec3i
        Number of cells in x, y, z directions for each system.
    cells_per_system : wp.array, shape (num_systems,), dtype=wp.int32
        OUTPUT: Total number of cells for each system.

    Notes
    -----
    - Thread launch: One thread per system (dim=num_systems)
    - Modifies: cells_per_system
    """
    system_idx = wp.tid()
    dims = cells_per_dimension[system_idx]
    cells_per_system[system_idx] = dims[0] * dims[1] * dims[2]


@wp.kernel(enable_backward=False)
def _batch_cell_list_build_neighbor_matrix(
    positions: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    pbc: wp.array2d(dtype=Any),
    batch_idx: wp.array(dtype=Any),
    cutoff: Any,
    cells_per_dimension: wp.array(dtype=Any),
    neighbor_search_radius: wp.array(dtype=Any),
    atom_periodic_shifts: wp.array(dtype=Any),
    atom_to_cell_mapping: wp.array(dtype=Any),
    atoms_per_cell_count: wp.array(dtype=Any),
    cell_atom_start_indices: wp.array(dtype=Any),
    cell_atom_list: wp.array(dtype=Any),
    cell_offsets: wp.array(dtype=Any),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts: wp.array(dtype=Any, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    half_fill: bool,
) -> None:
    """Build batch neighbor matrix with atom pairs and periodic shifts.

    For each atom across all systems in the batch, searches through neighboring
    cells and records all neighbor atoms within the cutoff distance
    into a fixed-size matrix format. Stores neighbor indices and their periodic
    shift vectors. Supports heterogeneous batches with different system parameters.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Concatenated atomic coordinates for all systems in the batch.
    cell : wp.array, shape (num_systems, 3, 3), dtype=wp.mat33*
        Unit cell matrices for each system in the batch.
    pbc : wp.array2d, shape (num_systems, 3), dtype=bool
        Periodic boundary condition flags for each system and dimension.
    batch_idx : wp.array, shape (total_atoms,), dtype=wp.int32
        Index of the system for each atom.
    cutoff : float
        Neighbor search cutoff distance.
    cells_per_dimension : wp.array, shape (num_systems, 3), dtype=wp.vec3i
        Number of cells in x, y, z directions for each system.
    cell_offsets : wp.array, shape (num_systems,), dtype=wp.int32
        Starting index in global cell arrays for each system (exclusive scan of cell counts).
    atom_periodic_shifts : wp.array, shape (total_atoms, 3), dtype=wp.vec3i
        Periodic boundary crossings for each atom.
    neighbor_search_radius : wp.array, shape (num_systems, 3), dtype=wp.vec3i
        Radius of neighboring cells to search for each system and dimension.
    atom_to_cell_mapping : wp.array, shape (total_atoms, 3), dtype=wp.vec3i
        3D cell coordinates for each atom.
    neighbor_matrix : wp.array, shape (total_atoms, max_neighbors), dtype=wp.int32
        OUTPUT: Neighbor matrix to be filled with neighbor atom indices.
    neighbor_matrix_shifts : wp.array, shape (total_atoms, max_neighbors, 3), dtype=wp.vec3i
        OUTPUT: Shift vectors for each neighbor relationship.
    num_neighbors : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Number of neighbors found for each atom.
    half_fill : bool
        If True, only store half of the neighbor relationships (i < j).

    Notes
    -----
    - Thread launch: One thread per atom across all systems (dim=total_atoms)
    - Modifies: neighbor_matrix, neighbor_matrix_shifts, num_neighbors
    - If max_neighbors is exceeded for an atom, extra neighbors are ignored
    - Each atom is only paired with atoms from its own system
    """
    atom_idx = wp.tid()

    # Find which system this atom belongs to
    system_idx = batch_idx[atom_idx]

    # Get system and atom specific parameters
    central_atom_position = positions[atom_idx]
    central_atom_cell_coords = atom_to_cell_mapping[atom_idx]

    s_cell = cell[system_idx]
    s_cells_per_dimension = cells_per_dimension[system_idx]
    s_cell_offset = cell_offsets[system_idx]
    s_neighbor_search_radius = neighbor_search_radius[system_idx]
    s_atom_periodic_shifts = atom_periodic_shifts[atom_idx]
    max_neighbors = neighbor_matrix.shape[1]

    s_pbc = pbc[system_idx]

    cutoff_distance_sq = cutoff * cutoff

    # Search through neighboring cells in this system
    # Use lexicographic ordering to reduce redundant checks:
    # Only search positive half-space of cell directions
    for dz in range(-s_neighbor_search_radius[2], s_neighbor_search_radius[2] + 1):
        for dy in range(-s_neighbor_search_radius[1], s_neighbor_search_radius[1] + 1):
            for dx in range(0, s_neighbor_search_radius[0] + 1):
                # Skip directions in negative half-space (lexicographic ordering)
                if not (
                    dx > 0 or (dx == 0 and dy > 0) or (dx == 0 and dy == 0 and dz >= 0)
                ):
                    continue

                # Calculate absolute cell coordinates
                target_x = central_atom_cell_coords[0] + dx
                target_y = central_atom_cell_coords[1] + dy
                target_z = central_atom_cell_coords[2] + dz

                # For non-PBC dimensions, skip cells outside the valid range
                if not s_pbc[0] and (
                    target_x < 0 or target_x >= s_cells_per_dimension[0]
                ):
                    continue
                if not s_pbc[1] and (
                    target_y < 0 or target_y >= s_cells_per_dimension[1]
                ):
                    continue
                if not s_pbc[2] and (
                    target_z < 0 or target_z >= s_cells_per_dimension[2]
                ):
                    continue

                # Handle periodic wrapping
                cs_x, wc_x = wpdivmod(target_x, s_cells_per_dimension[0])
                cs_y, wc_y = wpdivmod(target_y, s_cells_per_dimension[1])
                cs_z, wc_z = wpdivmod(target_z, s_cells_per_dimension[2])

                # Convert to global linear cell index
                global_linear_cell_index = (
                    s_cell_offset
                    + wc_x
                    + s_cells_per_dimension[0]
                    * (wc_y + s_cells_per_dimension[1] * wc_z)
                )

                # Get atom range for this cell
                cell_start_index = cell_atom_start_indices[global_linear_cell_index]
                num_atoms_in_cell = atoms_per_cell_count[global_linear_cell_index]

                # Check each atom in this neighboring cell
                for cell_atom_idx in range(num_atoms_in_cell):
                    neighbor_atom_idx = cell_atom_list[cell_start_index + cell_atom_idx]

                    # neighbor atom periodic shifts
                    n_atom_periodic_shifts = atom_periodic_shifts[neighbor_atom_idx]

                    # Calculate unit cell shift
                    shift_x = cs_x
                    shift_y = cs_y
                    shift_z = cs_z

                    if s_pbc[0]:
                        shift_x += s_atom_periodic_shifts[0] - n_atom_periodic_shifts[0]
                    else:
                        shift_x = 0
                    if s_pbc[1]:
                        shift_y += s_atom_periodic_shifts[1] - n_atom_periodic_shifts[1]
                    else:
                        shift_y = 0
                    if s_pbc[2]:
                        shift_z += s_atom_periodic_shifts[2] - n_atom_periodic_shifts[2]
                    else:
                        shift_z = 0

                    # For home cell (dx=dy=dz=0), only process j > i
                    # to avoid double counting
                    if dx == 0 and dy == 0 and dz == 0:
                        if neighbor_atom_idx <= atom_idx:
                            continue

                    # Calculate periodic shift vector in fractional coordinates
                    fractional_shift = type(central_atom_position)(
                        type(cutoff)(shift_x),
                        type(cutoff)(shift_y),
                        type(cutoff)(shift_z),
                    )
                    # Convert to Cartesian shift
                    cartesian_shift = fractional_shift * s_cell

                    # Calculate distance with periodic correction
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
_batch_estimate_cell_list_sizes_overload = {}
_batch_cell_list_construct_bin_size_overload = {}
_batch_cell_list_count_atoms_per_bin_overload = {}
_batch_cell_list_bin_atoms_overload = {}
_batch_cell_list_build_neighbor_matrix_overload = {}
for t, v, m in zip(T, V, M):
    _batch_estimate_cell_list_sizes_overload[t] = wp.overload(
        _batch_estimate_cell_list_sizes,
        [
            wp.array(dtype=m),
            wp.array2d(dtype=wp.bool),
            t,
            wp.int32,
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.vec3i),
        ],
    )
    _batch_cell_list_construct_bin_size_overload[t] = wp.overload(
        _batch_cell_list_construct_bin_size,
        [
            wp.array(dtype=m),
            wp.array2d(dtype=wp.bool),
            wp.array(dtype=wp.vec3i),
            t,
            wp.int32,
        ],
    )
    _batch_cell_list_count_atoms_per_bin_overload[t] = wp.overload(
        _batch_cell_list_count_atoms_per_bin,
        [
            wp.array(dtype=v),
            wp.array(dtype=m),
            wp.array2d(dtype=wp.bool),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.vec3i),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.vec3i),
        ],
    )
    _batch_cell_list_bin_atoms_overload[t] = wp.overload(
        _batch_cell_list_bin_atoms,
        [
            wp.array(dtype=v),
            wp.array(dtype=m),
            wp.array2d(dtype=wp.bool),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.vec3i),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.vec3i),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32),
        ],
    )
    _batch_cell_list_build_neighbor_matrix_overload[t] = wp.overload(
        _batch_cell_list_build_neighbor_matrix,
        [
            wp.array(dtype=v),
            wp.array(dtype=m),
            wp.array2d(dtype=wp.bool),
            wp.array(dtype=wp.int32),
            t,
            wp.array(dtype=wp.vec3i),
            wp.array(dtype=wp.vec3i),
            wp.array(dtype=wp.vec3i),
            wp.array(dtype=wp.vec3i),
            wp.array(dtype=wp.int32),
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
########################### Warp Launchers ###############################################
###########################################################################################


def batch_build_cell_list(
    positions: wp.array,
    cell: wp.array,
    pbc: wp.array,
    cutoff: float,
    batch_idx: wp.array,
    cells_per_dimension: wp.array,
    cell_offsets: wp.array,
    atom_periodic_shifts: wp.array,
    atom_to_cell_mapping: wp.array,
    atoms_per_cell_count: wp.array,
    cell_atom_start_indices: wp.array,
    cell_atom_list: wp.array,
    wp_dtype: type,
    device: str,
) -> None:
    """Core warp launcher for building batch spatial cell lists.

    Constructs spatial decomposition data structures for multiple systems using
    pure warp operations. This function launches warp kernels to organize atoms
    into spatial cells across all systems in the batch.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Concatenated atomic coordinates for all systems in the batch.
    cell : wp.array, shape (num_systems, 3, 3), dtype=wp.mat33*
        Unit cell matrices for each system in the batch.
    pbc : wp.array, shape (num_systems, 3), dtype=wp.bool
        Periodic boundary condition flags for each system and dimension.
    cutoff : float
        Neighbor search cutoff distance.
    batch_idx : wp.array, shape (total_atoms,), dtype=wp.int32
        System index for each atom.
    cells_per_dimension : wp.array, shape (num_systems, 3), dtype=wp.vec3i
        OUTPUT: Number of cells in x, y, z directions for each system.
    cell_offsets : wp.array, shape (num_systems,), dtype=wp.int32
        OUTPUT: Starting index in global cell arrays for each system.
        Computed internally via exclusive scan of cells_per_dimension products.
    atom_periodic_shifts : wp.array, shape (total_atoms, 3), dtype=wp.vec3i
        OUTPUT: Periodic boundary crossings for each atom.
    atom_to_cell_mapping : wp.array, shape (total_atoms, 3), dtype=wp.vec3i
        OUTPUT: 3D cell coordinates assigned to each atom.
    atoms_per_cell_count : wp.array, shape (max_total_cells,), dtype=wp.int32
        OUTPUT: Number of atoms in each cell. Must be zeroed by caller before first use.
    cell_atom_start_indices : wp.array, shape (max_total_cells,), dtype=wp.int32
        OUTPUT: Starting index in cell_atom_list for each cell's atoms.
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
    - cell_offsets is computed internally after cells_per_dimension is determined.
    - This function handles the internal cumsum for cell_atom_start_indices using wp.utils.array_scan.
    - For framework bindings, use the torch/jax wrappers instead.

    See Also
    --------
    batch_query_cell_list : Query cell list to build neighbor matrix (call after this)
    _batch_cell_list_construct_bin_size : Kernel for computing cell dimensions
    _batch_cell_list_count_atoms_per_bin : Kernel for counting atoms per cell
    _batch_cell_list_bin_atoms : Kernel for binning atoms into cells
    """
    total_atoms = positions.shape[0]
    num_systems = cell.shape[0]
    max_total_cells = atoms_per_cell_count.shape[0]
    wp_cutoff = wp_dtype(cutoff)

    # Construct cell dimensions
    wp.launch(
        _batch_cell_list_construct_bin_size_overload[wp_dtype],
        dim=num_systems,
        device=device,
        inputs=(
            cell,
            pbc,
            cells_per_dimension,
            wp_cutoff,
            max_total_cells,
        ),
    )

    # Compute cell_offsets from cells_per_dimension (exclusive scan of products)
    # This must happen after construct_bin_size fills cells_per_dimension
    cells_per_system = wp.zeros(num_systems, dtype=wp.int32, device=device)
    wp.launch(
        _compute_cells_per_system,
        dim=num_systems,
        device=device,
        inputs=(cells_per_dimension, cells_per_system),
    )
    wp.utils.array_scan(cells_per_system, cell_offsets, inclusive=False)

    # Count atoms per bin (expects atoms_per_cell_count to be zeroed by caller)
    wp.launch(
        _batch_cell_list_count_atoms_per_bin_overload[wp_dtype],
        dim=total_atoms,
        inputs=(
            positions,
            cell,
            pbc,
            batch_idx,
            cells_per_dimension,
            cell_offsets,
            atoms_per_cell_count,
            atom_periodic_shifts,
        ),
        device=device,
    )

    # Compute exclusive scan to get starting indices for each cell
    # This converts atom counts [3, 5, 2, ...] -> starting indices [0, 3, 8, ...]
    wp.utils.array_scan(atoms_per_cell_count, cell_atom_start_indices, inclusive=False)

    # Zero counts before binning atoms (second pass needs fresh counts)
    zero_array(atoms_per_cell_count, device)

    # Bin atoms (expects atoms_per_cell_count to be zeroed)
    wp.launch(
        _batch_cell_list_bin_atoms_overload[wp_dtype],
        dim=total_atoms,
        inputs=(
            positions,
            cell,
            pbc,
            batch_idx,
            cells_per_dimension,
            cell_offsets,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
        ),
        device=device,
    )


def batch_query_cell_list(
    positions: wp.array,
    cell: wp.array,
    pbc: wp.array,
    cutoff: float,
    batch_idx: wp.array,
    cells_per_dimension: wp.array,
    neighbor_search_radius: wp.array,
    cell_offsets: wp.array,
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
    """Core warp launcher for querying batch spatial cell lists to build neighbor matrices.

    Uses pre-built cell list data structures to efficiently find all atom pairs
    within the specified cutoff distance for multiple systems using pure warp operations.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Concatenated atomic coordinates for all systems in the batch.
    cell : wp.array, shape (num_systems, 3, 3), dtype=wp.mat33*
        Unit cell matrices for each system in the batch.
    pbc : wp.array, shape (num_systems, 3), dtype=wp.bool
        Periodic boundary condition flags for each system and dimension.
    cutoff : float
        Neighbor search cutoff distance.
    batch_idx : wp.array, shape (total_atoms,), dtype=wp.int32
        System index for each atom.
    cells_per_dimension : wp.array, shape (num_systems, 3), dtype=wp.vec3i
        Number of cells in x, y, z directions for each system.
    neighbor_search_radius : wp.array, shape (num_systems, 3), dtype=wp.vec3i
        Radius of neighboring cells to search for each system.
    cell_offsets : wp.array, shape (num_systems,), dtype=wp.int32
        Starting index in global cell arrays for each system.
        Output from batch_build_cell_list.
    atom_periodic_shifts : wp.array, shape (total_atoms, 3), dtype=wp.vec3i
        Periodic boundary crossings for each atom. Output from batch_build_cell_list.
    atom_to_cell_mapping : wp.array, shape (total_atoms, 3), dtype=wp.vec3i
        3D cell coordinates for each atom. Output from batch_build_cell_list.
    atoms_per_cell_count : wp.array, shape (max_total_cells,), dtype=wp.int32
        Number of atoms in each cell. Output from batch_build_cell_list.
    cell_atom_start_indices : wp.array, shape (max_total_cells,), dtype=wp.int32
        Starting index in cell_atom_list for each cell. Output from batch_build_cell_list.
    cell_atom_list : wp.array, shape (total_atoms,), dtype=wp.int32
        Flattened list of atom indices organized by cell. Output from batch_build_cell_list.
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
    - Output arrays must be pre-allocated by caller.

    See Also
    --------
    batch_build_cell_list : Build cell list data structures (call before this)
    _batch_cell_list_build_neighbor_matrix : Kernel that performs the neighbor search
    """
    total_atoms = positions.shape[0]

    # Build neighbor matrix
    wp.launch(
        _batch_cell_list_build_neighbor_matrix_overload[wp_dtype],
        dim=total_atoms,
        inputs=(
            positions,
            cell,
            pbc,
            batch_idx,
            wp_dtype(cutoff),
            cells_per_dimension,
            neighbor_search_radius,
            atom_periodic_shifts,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
            cell_offsets,
            neighbor_matrix,
            neighbor_matrix_shifts,
            num_neighbors,
            half_fill,
        ),
        device=device,
    )
