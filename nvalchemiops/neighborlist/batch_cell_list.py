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
# type: ignore
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
    cell_offsets : wp.array, shape (num_systems+1,), dtype=wp.int32
        Starting index in global cell arrays for each system in CSR format.
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
    cell_offsets : wp.array, shape (num_systems+1,), dtype=wp.int32
        Starting index in global cell arrays for each system in CSR format.
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
    cell_offsets : wp.array, shape (num_systems+1,), dtype=wp.int32
        Starting index in global cell arrays for each system in CSR format.
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
############################ Batch Cell List Pytorch Wrapper ##############################
###########################################################################################


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
    - Estimates assume roughly uniform atomic distribution within each system
    - Cell sizes are determined by the smallest cutoff to ensure neighbor completeness
    - For degenerate cells or empty systems, returns conservative fallback values
    - Memory usage scales as max_total_cells_across_batch * max_atoms_per_cell_any_system
    """
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
    neighbor_search_radius: torch.Tensor,
    atom_periodic_shifts: torch.Tensor,
    atom_to_cell_mapping: torch.Tensor,
    atoms_per_cell_count: torch.Tensor,
    cell_atom_start_indices: torch.Tensor,
    cell_atom_list: torch.Tensor,
) -> None:
    """Internal custom op for building batch spatial cell lists.

    Constructs a batch spatial cell list with fixed allocation sizes for torch.compile compatibility.
    Uses fixed-size memory allocations to prevent dynamic tensor creation that would
    cause graph breaks in torch.compile. Returns individual tensor components rather
    than a structured object for custom operator compatibility.

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
        Number of cells in x, y, z directions for each system.
    neighbor_search_radius : torch.Tensor, shape (num_systems, 3), dtype=int32
        Radius of neighboring cells to search for each system.
    atom_periodic_shifts : torch.Tensor, shape (total_atoms, 3), dtype=int32
        Periodic boundary crossings for each atom across all systems.
    atom_to_cell_mapping : torch.Tensor, shape (total_atoms, 3), dtype=int32
        3D cell coordinates assigned to each atom across all systems.
    atoms_per_cell_count : torch.Tensor, shape (max_total_cells,), dtype=int32
        Number of atoms in each cell across all systems.
    cell_atom_start_indices : torch.Tensor, shape (max_total_cells,), dtype=int32
        Starting index in global cell arrays for each system (CSR format).
    cell_atom_list : torch.Tensor, shape (total_atoms,), dtype=int32
        Flattened list of atom indices organized by cell across all systems.

    Returns
    -------
    None
        This function modifies the input tensors in-place.
    """
    total_atoms = positions.shape[0]
    num_systems = cell.shape[0]
    device = positions.device
    dtype = positions.dtype

    # Handle empty case
    if total_atoms == 0 or cutoff <= 0:
        return

    # Get warp dtype of input tensors
    wp_dtype = get_wp_dtype(dtype)
    wp_vec_dtype = get_wp_vec_dtype(dtype)
    wp_mat_dtype = get_wp_mat_dtype(dtype)
    wp_device = str(device)

    # Get warp arrays of input tensors
    wp_positions = wp.from_torch(positions, dtype=wp_vec_dtype, return_ctype=True)
    wp_cell = wp.from_torch(cell, dtype=wp_mat_dtype, return_ctype=True)
    wp_pbc = wp.from_torch(pbc, dtype=wp.bool, return_ctype=True)
    wp_cutoff = wp_dtype(cutoff)
    wp_batch_idx = wp.from_torch(
        batch_idx.to(dtype=torch.int32), dtype=wp.int32, return_ctype=True
    )
    # Construct cell list with fixed allocations
    max_total_cells = atoms_per_cell_count.shape[0]
    wp_cells_per_dimension = wp.from_torch(
        cells_per_dimension, dtype=wp.vec3i, return_ctype=True
    )
    wp.launch(
        _batch_cell_list_construct_bin_size_overload[wp_dtype],
        dim=num_systems,
        device=wp_device,
        inputs=(
            wp_cell,
            wp_pbc,
            wp_cells_per_dimension,
            wp_cutoff,
            max_total_cells,
        ),
    )

    # Use fixed allocation instead of dynamic calculation
    wp_atoms_per_cell_count = wp.from_torch(
        atoms_per_cell_count, dtype=wp.int32, return_ctype=True
    )

    wp_atom_periodic_shifts = wp.from_torch(
        atom_periodic_shifts, dtype=wp.vec3i, return_ctype=True
    )
    wp_atom_to_cell_mapping = wp.from_torch(
        atom_to_cell_mapping, dtype=wp.vec3i, return_ctype=True
    )

    # Fixed allocation for cell_atom_list
    wp_cell_atom_list = wp.from_torch(cell_atom_list, dtype=wp.int32, return_ctype=True)

    # Pre-allocate with fixed sizes
    wp_cell_atom_start_indices = wp.from_torch(
        cell_atom_start_indices, dtype=wp.int32, return_ctype=True
    )

    # Count atoms per bin
    atoms_per_cell_count.zero_()
    cell_offsets = torch.zeros(num_systems + 1, dtype=torch.int32, device=device)
    torch.cumsum(cells_per_dimension.prod(dim=1), dim=0, out=cell_offsets[1:])
    wp_cell_offsets = wp.from_torch(cell_offsets, dtype=wp.int32, return_ctype=True)
    wp.launch(
        _batch_cell_list_count_atoms_per_bin_overload[wp_dtype],
        dim=total_atoms,
        inputs=(
            wp_positions,
            wp_cell,
            wp_pbc,
            wp_batch_idx,
            wp_cells_per_dimension,
            wp_cell_offsets,
            wp_atoms_per_cell_count,
            wp_atom_periodic_shifts,
        ),
        device=wp_device,
    )

    # Compute cell offsets
    cell_atom_start_indices[0] = 0
    if max_total_cells > 1:
        torch.cumsum(
            atoms_per_cell_count[:-1],
            dim=0,
            out=cell_atom_start_indices[1:],
        )

    # Reset counts and bin atoms
    atoms_per_cell_count.zero_()
    wp.launch(
        _batch_cell_list_bin_atoms_overload[wp_dtype],
        dim=total_atoms,
        inputs=(
            wp_positions,
            wp_cell,
            wp_pbc,
            wp_batch_idx,
            wp_cells_per_dimension,
            wp_cell_offsets,
            wp_atom_to_cell_mapping,
            wp_atoms_per_cell_count,
            wp_cell_atom_start_indices,
            wp_cell_atom_list,
        ),
        device=wp_device,
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

    Uses pre-built cell list data structures to efficiently find all atom pairs
    within the specified cutoff distance. Handles periodic boundary conditions
    and returns neighbor matrix format.

    This function is torch compilable.

    Parameters
    ----------
    positions : torch.Tensor, shape (total_atoms, 3)
        Concatenated atomic coordinates for all systems in the batch.
    cell : torch.Tensor, shape (num_systems, 3, 3)
        Unit cell matrices for each system in the batch.
    pbc : torch.Tensor, shape (num_systems, 3), dtype=bool
        Periodic boundary condition flags for each system and dimension.
    cutoff : float
        Neighbor search cutoff distance.
    batch_idx : torch.Tensor, shape (total_atoms,), dtype=int32
        Index of the system for each atom.
    cells_per_dimension : torch.Tensor, shape (num_systems, 3), dtype=int32
        Number of cells in x, y, z directions for each system.
    neighbor_search_radius : torch.Tensor, shape (num_systems, 3), dtype=int32
        Radius of neighboring cells to search for each system.
    atom_periodic_shifts : torch.Tensor, shape (total_atoms, 3), dtype=int32
        Periodic boundary crossings for each atom.
    atom_to_cell_mapping : torch.Tensor, shape (total_atoms, 3), dtype=int32
        3D cell coordinates for each atom.
    atoms_per_cell_count : torch.Tensor, shape (max_total_cells,), dtype=int32
        Number of atoms in each cell across all systems.
    cell_atom_start_indices : torch.Tensor, shape (max_total_cells,), dtype=int32
        Starting index in batch_cell_atom_list for each cell.
    neighbor_matrix : torch.Tensor, shape (total_atoms, max_neighbors), dtype=int32
        OUTPUT: Neighbor matrix to be filled with neighbor atom indices.
        Must be pre-allocated.
    neighbor_matrix_shifts : torch.Tensor, shape (total_atoms, max_neighbors, 3), dtype=int32
        OUTPUT: Matrix storing shift vectors for each neighbor relationship.
        Must be pre-allocated.
    num_neighbors : torch.Tensor, shape (total_atoms,), dtype=int32
        OUTPUT: Number of neighbors found for each atom.
        Must be pre-allocated.
    half_fill : bool
        If True, only store half of the neighbor relationships (i < j).
    Returns
    -------
    None
        This function modifies the input tensors in-place:

        - neighbor_matrix : Filled with neighbor atom indices
        - neighbor_matrix_shifts : Filled with corresponding shift vectors
        - num_neighbors : Updated with neighbor counts per atom
    """
    device = positions.device
    wp_device = str(device)
    total_atoms = positions.shape[0]
    num_systems = cell.shape[0]

    # Handle empty case
    if total_atoms == 0 or cutoff <= 0:
        return

    # Get warp dtype of input tensors
    wp_dtype = get_wp_dtype(positions.dtype)
    wp_vec_dtype = get_wp_vec_dtype(positions.dtype)
    wp_mat_dtype = get_wp_mat_dtype(positions.dtype)

    # Get warp arrays
    wp_positions = wp.from_torch(positions, dtype=wp_vec_dtype, return_ctype=True)
    wp_cell = wp.from_torch(cell, dtype=wp_mat_dtype, return_ctype=True)
    wp_pbc = wp.from_torch(pbc, dtype=wp.bool, return_ctype=True)
    wp_cutoff = wp_dtype(cutoff)
    wp_batch_idx = wp.from_torch(
        batch_idx.to(dtype=torch.int32), dtype=wp.int32, return_ctype=True
    )

    wp_cells_per_dimension = wp.from_torch(
        cells_per_dimension, dtype=wp.vec3i, return_ctype=True
    )
    wp_neighbor_search_radius = wp.from_torch(
        neighbor_search_radius, dtype=wp.vec3i, return_ctype=True
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
    cell_offsets = torch.zeros(num_systems + 1, dtype=torch.int32, device=device)
    torch.cumsum(cells_per_dimension.prod(dim=1), dim=0, out=cell_offsets[1:])
    wp_cell_offsets = wp.from_torch(cell_offsets, dtype=wp.int32, return_ctype=True)
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
        _batch_cell_list_build_neighbor_matrix_overload[wp_dtype],
        dim=total_atoms,
        inputs=(
            wp_positions,
            wp_cell,
            wp_pbc,
            wp_batch_idx,
            wp_cutoff,
            wp_cells_per_dimension,
            wp_neighbor_search_radius,
            wp_atom_periodic_shifts,
            wp_atom_to_cell_mapping,
            wp_atoms_per_cell_count,
            wp_cell_atom_start_indices,
            wp_cell_atom_list,
            wp_cell_offsets,
            wp_neighbor_matrix,
            wp_neighbor_matrix_shifts,
            wp_num_neighbors,
            half_fill,
        ),
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

    Constructs a batch spatial cell list with fixed allocation sizes for torch.compile compatibility.
    Uses fixed-size memory allocations to prevent dynamic tensor creation that would
    cause graph breaks in torch.compile. Returns individual tensor components rather
    than a structured object for custom operator compatibility.

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
        Number of cells in x, y, z directions for each system.
    neighbor_search_radius : torch.Tensor, shape (num_systems, 3), dtype=int32
        Radius of neighboring cells to search for each system.
    atom_periodic_shifts : torch.Tensor, shape (total_atoms, 3), dtype=int32
        Periodic boundary crossings for each atom across all systems.
    atom_to_cell_mapping : torch.Tensor, shape (total_atoms, 3), dtype=int32
        3D cell coordinates assigned to each atom across all systems.
    atoms_per_cell_count : torch.Tensor, shape (max_total_cells,), dtype=int32
        Number of atoms in each cell across all systems.
    cell_atom_start_indices : torch.Tensor, shape (max_total_cells,), dtype=int32
        Starting index in global cell arrays for each system (CSR format).
    cell_atom_list : torch.Tensor, shape (total_atoms,), dtype=int32
        Flattened list of atom indices organized by cell across all systems.

    Returns
    -------
    None
        This function modifies the input tensors in-place.
    """
    return _batch_build_cell_list_op(
        positions,
        cutoff,
        cell,
        pbc,
        batch_idx,
        cells_per_dimension,
        neighbor_search_radius,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
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

    Uses pre-built cell list data structures to efficiently find all atom pairs
    within the specified cutoff distance. Handles periodic boundary conditions
    and returns neighbor matrix format.

    This function is torch compilable.

    Parameters
    ----------
    positions : torch.Tensor, shape (total_atoms, 3)
        Concatenated atomic coordinates for all systems in the batch.
    cell : torch.Tensor, shape (num_systems, 3, 3)
        Unit cell matrices for each system in the batch.
    pbc : torch.Tensor, shape (num_systems, 3), dtype=bool
        Periodic boundary condition flags for each system and dimension.
    cutoff : float
        Neighbor search cutoff distance.
    batch_idx : torch.Tensor, shape (total_atoms,), dtype=int32
        Index of the system for each atom.
    cells_per_dimension : torch.Tensor, shape (num_systems, 3), dtype=int32
        Number of cells in x, y, z directions for each system.
    neighbor_search_radius : torch.Tensor, shape (num_systems, 3), dtype=int32
        Radius of neighboring cells to search for each system.
    atom_periodic_shifts : torch.Tensor, shape (total_atoms, 3), dtype=int32
        Periodic boundary crossings for each atom.
    atom_to_cell_mapping : torch.Tensor, shape (total_atoms, 3), dtype=int32
        3D cell coordinates for each atom.
    atoms_per_cell_count : torch.Tensor, shape (max_total_cells,), dtype=int32
        Number of atoms in each cell across all systems.
    cell_atom_start_indices : torch.Tensor, shape (max_total_cells,), dtype=int32
        Starting index in batch_cell_atom_list for each cell.
    neighbor_matrix : torch.Tensor, shape (total_atoms, max_neighbors), dtype=int32
        OUTPUT: Neighbor matrix to be filled with neighbor atom indices.
        Must be pre-allocated.
    neighbor_matrix_shifts : torch.Tensor, shape (total_atoms, max_neighbors, 3), dtype=int32
        OUTPUT: Matrix storing shift vectors for each neighbor relationship.
        Must be pre-allocated.
    num_neighbors : torch.Tensor, shape (total_atoms,), dtype=int32
        OUTPUT: Number of neighbors found for each atom.
        Must be pre-allocated.
    half_fill : bool
        If True, only store half of the neighbor relationships (i < j).
    Returns
    -------
    None
        This function modifies the input tensors in-place:

        - neighbor_matrix : Filled with neighbor atom indices
        - neighbor_matrix_shifts : Filled with corresponding shift vectors
        - num_neighbors : Updated with neighbor counts per atom
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
        If True, convert the neighbor matrix to a neighbor list (idx_i, idx_j) format by
        creating a mask over the fill_value, which can incur a performance penalty.
        We recommend using the neighbor matrix format,
        and only convert to a neighbor list format if absolutely necessary.
    cells_per_dimension : torch.Tensor, shape (num_systems, 3), dtype=int32
        Number of cells in x, y, z directions.
        Pass a pre-allocated tensor to avoid reallocation for cell list construction.
        If None, allocated internally to build the cell list.
    neighbor_search_radius : torch.Tensor, shape (num_systems, 3), dtype=int32
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

            - If ``return_neighbor_list=False`` (default): Returns ``neighbor_matrix``
              with shape (total_atoms, max_neighbors), dtype int32. Each row i contains
              indices of atom i's neighbors, padded with fill_value.
            - If ``return_neighbor_list=True``: Returns ``neighbor_list`` with shape
              (2, num_pairs), dtype int32, in COO format [source_atoms, target_atoms].

        - **num_neighbor_data** (tensor): Information about the number of neighbors for each atom,
          format depends on ``return_neighbor_list``:

            - If ``return_neighbor_list=False`` (default): Returns ``num_neighbors`` with shape (total_atoms,), dtype int32.
              Count of neighbors found for each atom.
            - If ``return_neighbor_list=True``: Returns ``neighbor_ptr`` with shape (total_atoms + 1,), dtype int32.
              CSR-style pointer arrays where ``neighbor_ptr_data[i]`` to ``neighbor_ptr_data[i+1]`` gives the range of
              neighbors for atom i in the flattened neighbor list.

        - **neighbor_shift_data** (tensor): Periodic shift vectors for each neighbor,
          format depends on ``return_neighbor_list``:

            - If ``return_neighbor_list=False`` (default): Returns ``neighbor_matrix_shifts`` with
              shape (total_atoms, max_neighbors, 3), dtype int32.
            - If ``return_neighbor_list=True``: Returns ``neighbor_list_shifts`` with shape
              (num_pairs, 3), dtype int32.

    Notes
    -----
    - This is the main user-facing API for batch neighbor list construction
    - Uses automatic memory allocation estimation for torch.compile compatibility
    - Efficiently processes systems with different sizes, cells, PBC, and cutoffs
    - For advanced users who want to cache cell lists, use build_batch_cell_list and query_batch_cell_list separately
    - Returns empty tensors for empty batches
    """
    total_atoms = positions.shape[0]
    device = positions.device

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
