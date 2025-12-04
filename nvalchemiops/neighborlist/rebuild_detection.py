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

"""
Rebuild detection kernels for cell lists and neighbor lists.

This module provides high-performance warp kernels to determine when
cell lists and neighbor lists need to be rebuilt based on atomic positions,
cell changes, and skin distance criteria.
"""

from typing import Any

import torch
import warp as wp

from nvalchemiops.types import get_wp_dtype, get_wp_mat_dtype, get_wp_vec_dtype

###########################################################################################
########################### Cell List Rebuild Detection ###################################
###########################################################################################


@wp.kernel(enable_backward=False)
def _check_atoms_changed_cells(
    current_positions: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    atom_to_cell_mapping: wp.array(dtype=Any),
    cells_per_dimension: wp.array(dtype=Any),
    pbc: wp.array(dtype=Any),
    rebuild_flag: wp.array(dtype=wp.bool),
) -> None:
    """Detect if atoms have moved between spatial cells requiring cell list rebuild.

    This kernel computes current cell assignments for each atom and compares them
    with the stored cell assignments from the existing cell list to determine if
    any atoms have crossed cell boundaries. Uses early termination for efficiency.

    Parameters
    ----------
    current_positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Current atomic coordinates in Cartesian space.
    cell : wp.array, shape (1, 3, 3), dtype=wp.mat33*
        Unit cell matrix for coordinate transformations.
    atom_to_cell_mapping : wp.array, shape (total_atoms, 3), dtype=wp.vec3i
        Previously computed cell coordinates for each atom from existing cell list.
        This is an output from build_cell_list.
    cells_per_dimension : wp.array, shape (3,), dtype=wp.int32
        Number of cells in x, y, z directions.
    pbc : wp.array, shape (3,), dtype=bool
        Periodic boundary condition flags for x, y, z directions.
    rebuild_flag : wp.array, shape (1,), dtype=bool
        OUTPUT: Flag set to True if any atom changed cells (modified atomically).

    Notes
    -----
    - Currently only supports single system.
    - Thread launch: One thread per atom (dim=total_atoms)
    - Modifies: rebuild_flag (atomic write)
    - Early termination: Threads exit if rebuild already flagged
    - Handles periodic boundaries with proper wrapping
    """
    atom_idx = wp.tid()

    if atom_idx >= current_positions.shape[0]:
        return

    # Skip computation if rebuild already flagged by another thread
    if rebuild_flag[0]:
        return

    # Transform current position to fractional coordinates
    inverse_cell_transpose = wp.transpose(wp.inverse(cell[0]))
    fractional_position = inverse_cell_transpose * current_positions[atom_idx]
    current_cell_coords = wp.vec3i(0, 0, 0)

    # Compute current cell coordinates for each dimension
    for dim in range(3):
        current_cell_coords[dim] = wp.int32(
            wp.floor(
                fractional_position[dim]
                * type(fractional_position[dim])(cells_per_dimension[dim])
            )
        )

        # Handle periodic boundary conditions
        if pbc[dim]:
            current_cell_coords[dim] = (
                current_cell_coords[dim] % cells_per_dimension[dim]
            )
            if current_cell_coords[dim] < 0:
                current_cell_coords[dim] += cells_per_dimension[dim]
        else:
            # Clamp to valid cell range for non-periodic dimensions
            current_cell_coords[dim] = wp.clamp(
                current_cell_coords[dim], 0, cells_per_dimension[dim] - 1
            )

    # Compare with stored cell coordinates from existing cell list
    stored_cell_coords = atom_to_cell_mapping[atom_idx]

    # Check if atom has moved to a different cell
    if (
        current_cell_coords[0] != stored_cell_coords[0]
        or current_cell_coords[1] != stored_cell_coords[1]
        or current_cell_coords[2] != stored_cell_coords[2]
    ):
        # Atom crossed cell boundary - flag for rebuild
        rebuild_flag[0] = True


@wp.overload
def _check_atoms_changed_cells(
    current_positions: wp.array(dtype=wp.vec3d),
    cell: wp.array(dtype=wp.mat33d),
    atom_to_cell_mapping: wp.array(dtype=wp.vec3i),
    cells_per_dimension: wp.array(dtype=wp.int32),
    pbc: wp.array(dtype=wp.bool),
    rebuild_flag: wp.array(dtype=wp.bool),
) -> None:  # pragma: no cover
    """Float64 precision overload for atom cell change detection kernel."""
    ...


@wp.overload
def _check_atoms_changed_cells(
    current_positions: wp.array(dtype=wp.vec3f),
    cell: wp.array(dtype=wp.mat33f),
    atom_to_cell_mapping: wp.array(dtype=wp.vec3i),
    cells_per_dimension: wp.array(dtype=wp.int32),
    pbc: wp.array(dtype=wp.bool),
    rebuild_flag: wp.array(dtype=wp.bool),
) -> None:  # pragma: no cover
    """Float32 precision overload for atom cell change detection kernel."""
    ...


@wp.overload
def _check_atoms_changed_cells(
    current_positions: wp.array(dtype=wp.vec3h),
    cell: wp.array(dtype=wp.mat33h),
    atom_to_cell_mapping: wp.array(dtype=wp.vec3i),
    cells_per_dimension: wp.array(dtype=wp.int32),
    pbc: wp.array(dtype=wp.bool),
    rebuild_flag: wp.array(dtype=wp.bool),
) -> None:  # pragma: no cover
    """Float16 precision overload for atom cell change detection kernel."""
    ...


###########################################################################################
########################### Neighbor List Rebuild Detection #############################
###########################################################################################


@wp.kernel(enable_backward=False)
def _check_atoms_moved_beyond_skin(
    reference_positions: wp.array(dtype=Any),
    current_positions: wp.array(dtype=Any),
    skin_distance_threshold: Any,
    rebuild_flag: wp.array(dtype=wp.bool),
) -> None:
    """Detect if atoms have moved beyond skin distance requiring neighbor list rebuild.

    This kernel computes the displacement of each atom from its reference position
    and checks if any atom has moved farther than the skin distance threshold.
    Uses early termination for computational efficiency when rebuild is already flagged.

    Parameters
    ----------
    reference_positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Atomic positions when the neighbor list was last built.
    current_positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Current atomic positions to compare against reference.
    skin_distance_threshold : float*/int*
        Maximum allowed displacement before neighbor list becomes invalid.
        Typically set to (cutoff_radius - cutoff) / 2.
    rebuild_flag : wp.array, shape (1,), dtype=bool
        OUTPUT: Flag set to True if any atom moved beyond skin distance (modified atomically).

    Notes
    -----
    - Currently only supports single system.
    - Thread launch: One thread per atom (dim=total_atoms)
    - Modifies: rebuild_flag (atomic write)
    - Early termination: Threads exit if rebuild already flagged
    - Displacement calculation uses Euclidean distance
    """
    atom_idx = wp.tid()

    if atom_idx >= reference_positions.shape[0]:
        return

    # Skip computation if rebuild already flagged by another thread
    if rebuild_flag[0]:
        return

    # Calculate displacement vector from reference to current position
    displacement_vector = current_positions[atom_idx] - reference_positions[atom_idx]
    displacement_magnitude = wp.length(displacement_vector)

    # Check if atom has moved beyond the skin distance threshold
    if displacement_magnitude > skin_distance_threshold:
        # Neighbor list is no longer valid - flag for rebuild
        rebuild_flag[0] = True


@wp.overload
def _check_atoms_moved_beyond_skin(
    reference_positions: wp.array(dtype=wp.vec3d),
    current_positions: wp.array(dtype=wp.vec3d),
    skin_distance_threshold: wp.float64,
    rebuild_flag: wp.array(dtype=wp.bool),
) -> None:  # pragma: no cover
    """Float64 precision overload for skin distance movement detection kernel."""
    ...


@wp.overload
def _check_atoms_moved_beyond_skin(
    reference_positions: wp.array(dtype=wp.vec3f),
    current_positions: wp.array(dtype=wp.vec3f),
    skin_distance_threshold: wp.float32,
    rebuild_flag: wp.array(dtype=wp.bool),
) -> None:  # pragma: no cover
    """Float32 precision overload for skin distance movement detection kernel."""
    ...


@wp.overload
def _check_atoms_moved_beyond_skin(
    reference_positions: wp.array(dtype=wp.vec3h),
    current_positions: wp.array(dtype=wp.vec3h),
    skin_distance_threshold: wp.float16,
    rebuild_flag: wp.array(dtype=wp.bool),
) -> None:  # pragma: no cover
    """Float16 precision overload for skin distance movement detection kernel."""
    ...


###########################################################################################
########################### PyTorch API Functions #######################################
###########################################################################################


@torch.library.custom_op("nvalchemiops::_cell_list_needs_rebuild", mutates_args=())
def _cell_list_needs_rebuild(
    current_positions: torch.Tensor,
    atom_to_cell_mapping: torch.Tensor,
    cells_per_dimension: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
) -> torch.Tensor:
    """Detect if spatial cell list requires rebuilding due to atomic motion.

    Parameters
    ----------
    current_positions : torch.Tensor, shape (total_atoms, 3)
    atom_to_cell_mapping : torch.Tensor, shape (total_atoms, 3), dtype=int32
    cells_per_dimension : torch.Tensor, shape (3,), dtype=int32
    cell : torch.Tensor, shape (1, 3, 3)
    pbc : torch.Tensor, shape (3,), dtype=bool

    Returns
    -------
    rebuild_needed : torch.Tensor, shape (1,), dtype=bool
    """
    total_atoms = current_positions.shape[0]
    device = current_positions.device
    pbc = pbc.squeeze(0)
    if total_atoms == 0:
        return torch.tensor([False], device=device, dtype=torch.bool)

    # Get warp data types for the input tensor precision
    wp_vec_dtype = get_wp_vec_dtype(current_positions.dtype)
    wp_mat_dtype = get_wp_mat_dtype(current_positions.dtype)
    wp_device = str(device)

    # Convert PyTorch tensors to warp arrays
    wp_current_positions = wp.from_torch(current_positions, dtype=wp_vec_dtype)
    wp_cell = wp.from_torch(cell, dtype=wp_mat_dtype)
    wp_pbc = wp.from_torch(pbc, dtype=wp.bool)
    wp_atom_to_cell_mapping = wp.from_torch(atom_to_cell_mapping, dtype=wp.vec3i)
    wp_cells_per_dimension = wp.from_torch(cells_per_dimension, dtype=wp.int32)

    # Initialize rebuild flag (False = no rebuild needed)
    rebuild_needed = torch.tensor([False], device=device, dtype=torch.bool)
    wp_rebuild_flag = wp.from_torch(rebuild_needed, dtype=wp.bool)

    # Launch GPU kernel to check for cell boundary crossings
    wp.launch(
        _check_atoms_changed_cells,
        dim=total_atoms,
        inputs=[
            wp_current_positions,
            wp_cell,
            wp_atom_to_cell_mapping,
            wp_cells_per_dimension,
            wp_pbc,
            wp_rebuild_flag,
        ],
        device=wp_device,
    )

    return rebuild_needed


@_cell_list_needs_rebuild.register_fake
def _cell_list_needs_rebuild_fake(
    current_positions: torch.Tensor,
    atom_to_cell_mapping: torch.Tensor,
    cells_per_dimension: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
) -> torch.Tensor:
    """Fake implementation for torch.compile compatibility.

    Returns a conservative default (no rebuild needed) for compilation tracing.
    The actual implementation will be called during runtime execution.
    """
    return torch.tensor([False], device=current_positions.device, dtype=torch.bool)


def cell_list_needs_rebuild(
    current_positions: torch.Tensor,
    atom_to_cell_mapping: torch.Tensor,
    cells_per_dimension: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
) -> torch.Tensor:
    """Detect if spatial cell list requires rebuilding due to atomic motion.

    This torch.compile-compatible custom operator efficiently determines if any atoms
    have moved between spatial cells since the last cell list construction. Uses GPU
    acceleration with early termination for optimal performance.

    Parameters
    ----------
    current_positions : torch.Tensor, shape (total_atoms, 3)
        Current atomic coordinates in Cartesian space.
    atom_to_cell_mapping : torch.Tensor, shape (total_atoms, 3), dtype=int32
        3D cell coordinates for each atom from the existing cell list.
        Typically obtained from build_cell_list.
    cells_per_dimension : torch.Tensor, shape (3,), dtype=int32
        Number of spatial cells in x, y, z directions.
    cell : torch.Tensor, shape (1, 3, 3)
        Unit cell matrix for coordinate transformations.
    pbc : torch.Tensor, shape (3,), dtype=bool
        Periodic boundary condition flags for x, y, z directions.

    Returns
    -------
    rebuild_needed : torch.Tensor, shape (1,), dtype=bool
        True if any atom has moved to a different cell requiring rebuild.

    Notes
    -----
    - Currently only supports single system.
    - torch.compile compatible custom operation
    - Uses GPU kernels for parallel cell assignment computation
    - Early termination optimization stops computation once rebuild is detected
    - Handles periodic boundary conditions correctly
    - Returns tensor (not Python bool) for compilation compatibility
    """
    return _cell_list_needs_rebuild(
        current_positions,
        atom_to_cell_mapping,
        cells_per_dimension,
        cell,
        pbc,
    )


@torch.library.custom_op("nvalchemiops::_neighbor_list_needs_rebuild", mutates_args=())
def _neighbor_list_needs_rebuild(
    reference_positions: torch.Tensor,
    current_positions: torch.Tensor,
    skin_distance_threshold: float,
) -> torch.Tensor:
    """Detect if neighbor list requires rebuilding due to excessive atomic motion.

    Parameters
    ----------
    reference_positions : torch.Tensor, shape (total_atoms, 3)
    current_positions : torch.Tensor, shape (total_atoms, 3)
    skin_distance_threshold : float

    Returns
    -------
    rebuild_needed : torch.Tensor, shape (1,), dtype=bool
    """
    # Check for shape compatibility
    if reference_positions.shape != current_positions.shape:
        return torch.tensor([True], device=current_positions.device, dtype=torch.bool)

    total_atoms = reference_positions.shape[0]
    device = reference_positions.device

    if total_atoms == 0:
        return torch.tensor([False], device=device, dtype=torch.bool)

    # Get warp data types for the input tensor precision
    wp_dtype = get_wp_dtype(reference_positions.dtype)
    wp_vec_dtype = get_wp_vec_dtype(reference_positions.dtype)
    wp_device = str(device)

    # Convert PyTorch tensors to warp arrays
    wp_reference_positions = wp.from_torch(reference_positions, dtype=wp_vec_dtype)
    wp_current_positions = wp.from_torch(current_positions, dtype=wp_vec_dtype)

    # Initialize rebuild flag (False = no rebuild needed)
    rebuild_needed = torch.tensor([False], device=device, dtype=torch.bool)
    wp_rebuild_flag = wp.from_torch(rebuild_needed, dtype=wp.bool)

    # Launch GPU kernel to check for excessive atomic displacements
    wp.launch(
        _check_atoms_moved_beyond_skin,
        dim=total_atoms,
        inputs=[
            wp_reference_positions,
            wp_current_positions,
            wp_dtype(skin_distance_threshold),
            wp_rebuild_flag,
        ],
        device=wp_device,
    )

    return rebuild_needed


@_neighbor_list_needs_rebuild.register_fake
def _neighbor_list_needs_rebuild_fake(
    reference_positions: torch.Tensor,
    current_positions: torch.Tensor,
    skin_distance_threshold: float,
) -> torch.Tensor:
    """Fake implementation for torch.compile compatibility.

    Returns a conservative default (no rebuild needed) for compilation tracing.
    The actual implementation will be called during runtime execution.
    """
    return torch.tensor([False], device=current_positions.device, dtype=torch.bool)


def neighbor_list_needs_rebuild(
    reference_positions: torch.Tensor,
    current_positions: torch.Tensor,
    skin_distance_threshold: float,
) -> torch.Tensor:
    """Detect if neighbor list requires rebuilding due to excessive atomic motion.

    This torch.compile-compatible custom operator efficiently determines if any atoms
    have moved beyond the skin distance since the neighbor list was last built. Uses
    GPU acceleration with early termination for optimal performance in MD simulations.

    The skin distance approach allows neighbor lists to remain valid even when atoms
    move slightly, reducing the frequency of expensive neighbor list reconstructions.

    Parameters
    ----------
    reference_positions : torch.Tensor, shape (total_atoms, 3)
        Atomic coordinates when the neighbor list was last constructed.
    current_positions : torch.Tensor, shape (total_atoms, 3)
        Current atomic coordinates to compare against reference.
    skin_distance_threshold : float
        Maximum allowed atomic displacement before neighbor list becomes invalid.
        Typically set to (cutoff_radius - cutoff) / 2 for safety.

    Returns
    -------
    rebuild_needed : torch.Tensor, shape (1,), dtype=bool
        True if any atom has moved beyond skin distance requiring rebuild.

    Notes
    -----
    - Currently only supports single system.
    - torch.compile compatible custom operation
    - Uses GPU kernels for parallel displacement computation
    - Early termination optimization stops computation once rebuild is detected
    - Displacement calculation uses Euclidean distance
    - Returns tensor (not Python bool) for compilation compatibility
    """
    return _neighbor_list_needs_rebuild(
        reference_positions, current_positions, skin_distance_threshold
    )


###########################################################################################
########################### High-level API Functions ####################################
###########################################################################################


def check_cell_list_rebuild_needed(
    cells_per_dimension: torch.Tensor,
    neighbor_search_radius: torch.Tensor,
    atom_periodic_shifts: torch.Tensor,
    atom_to_cell_mapping: torch.Tensor,
    atoms_per_cell_count: torch.Tensor,
    cell_atom_start_indices: torch.Tensor,
    cell_atom_list: torch.Tensor,
    current_positions: torch.Tensor,
    current_cell: torch.Tensor,
    current_pbc: torch.Tensor,
    cutoff: float,
) -> bool:
    """Determine if spatial cell list requires rebuilding based on atomic motion.

    This high-level function provides a comprehensive check to determine if a spatial
    cell list needs to be reconstructed due to atomic movement. It uses GPU acceleration
    to efficiently detect when atoms have moved between spatial cells.

    The function primarily checks if any atoms have moved to different spatial cells
    since the cell list was last built by comparing current positions against the
    stored cell assignments from the existing cell list.

    This function is not torch.compile compatible.

    Parameters
    ----------
    cells_per_dimension : torch.Tensor, shape (3,), dtype=int32
        Number of spatial cells in x, y, z directions from existing cell list.
    neighbor_search_radius : torch.Tensor, shape (3,), dtype=int32
        Search radius for neighboring cells in each dimension from existing cell list.
    atom_periodic_shifts : torch.Tensor, shape (total_atoms, 3), dtype=int32
        Periodic boundary crossings for each atom from existing cell list.
    atom_to_cell_mapping : torch.Tensor, shape (total_atoms, 3), dtype=int32
        3D cell coordinates assigned to each atom from existing cell list.
        This is the key tensor used for comparison with current positions.
    atoms_per_cell_count : torch.Tensor, shape (total_cells,), dtype=int32
        Number of atoms in each cell from existing cell list.
    cell_atom_start_indices : torch.Tensor, shape (total_cells,), dtype=int32
        Starting indices for each cell's atom list from existing cell list.
    cell_atom_list : torch.Tensor, shape (total_atoms,), dtype=int32
        Flattened atom indices organized by cell from existing cell list.
    current_positions : torch.Tensor, shape (total_atoms, 3)
        Current atomic coordinates to check against existing cell assignments.
    current_cell : torch.Tensor, shape (1, 3, 3)
        Current unit cell matrix for coordinate transformations.
    current_pbc : torch.Tensor, shape (3,), dtype=bool
        Current periodic boundary condition flags for x, y, z directions.
    cutoff : float
        Neighbor search cutoff distance (currently unused, kept for API compatibility).

    Returns
    -------
    needs_rebuild : bool
        True if any atom has moved to a different cell requiring cell list rebuild.

    Notes
    -----
    - Currently only supports single system.
    - Uses GPU kernels for efficient parallel computation
    - Primary check: atomic motion between spatial cells
    - Early termination optimization for performance
    """
    rebuild_tensor = cell_list_needs_rebuild(
        current_positions,
        atom_to_cell_mapping,
        cells_per_dimension,
        current_cell,
        current_pbc,
    )

    return rebuild_tensor.item()


def check_neighbor_list_rebuild_needed(
    reference_positions: torch.Tensor,
    current_positions: torch.Tensor,
    skin_distance_threshold: float,
) -> bool:
    """Determine if neighbor list requires rebuilding based on atomic motion.

    This high-level function provides a convenient interface to check if a neighbor
    list needs reconstruction due to excessive atomic movement. Uses the skin distance
    approach to minimize unnecessary neighbor list rebuilds during MD simulations.

    The skin distance technique allows atoms to move slightly without invalidating
    the neighbor list, reducing computational overhead. When any atom moves beyond
    the skin distance, the neighbor list must be rebuilt to maintain accuracy.

    This function is not torch.compile compatible.

    Parameters
    ----------
    reference_positions : torch.Tensor, shape (total_atoms, 3)
        Atomic coordinates when the neighbor list was last constructed.
        Used as the reference point for displacement calculations.
    current_positions : torch.Tensor, shape (total_atoms, 3)
        Current atomic coordinates to compare against reference positions.
        Must have the same shape as reference_positions.
    skin_distance_threshold : float
        Maximum allowed atomic displacement before neighbor list becomes invalid.
        Typically set to (cutoff_radius - cutoff) / 2 for safety.
        Units should match the coordinate system.

    Returns
    -------
    needs_rebuild : bool
        True if any atom has moved beyond skin distance requiring neighbor list rebuild.

    Notes
    -----
    - Currently only supports single system.
    - Uses GPU acceleration for efficient displacement computation
    - Early termination optimization for performance
    - Essential for efficient molecular dynamics simulations
    """
    rebuild_tensor = neighbor_list_needs_rebuild(
        reference_positions, current_positions, skin_distance_threshold
    )

    return rebuild_tensor.item()
