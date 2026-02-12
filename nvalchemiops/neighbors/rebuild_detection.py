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

"""Core warp kernels and launchers for rebuild detection.

This module provides warp kernels to determine when cell lists and neighbor lists
need to be rebuilt based on atomic positions, cell changes, and skin distance criteria.
See `nvalchemiops.torch.neighbors` for PyTorch bindings.
"""

from typing import Any

import warp as wp

__all__ = [
    "check_cell_list_rebuild",
    "check_neighbor_list_rebuild",
]

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


# Generate overload dictionary for cell list rebuild kernel
_T = [wp.float32, wp.float64, wp.float16]
_V = [wp.vec3f, wp.vec3d, wp.vec3h]
_M = [wp.mat33f, wp.mat33d, wp.mat33h]
_check_atoms_changed_cells_overload = {}
for t, v, m in zip(_T, _V, _M):
    _check_atoms_changed_cells_overload[t] = wp.overload(
        _check_atoms_changed_cells,
        [
            wp.array(dtype=v),
            wp.array(dtype=m),
            wp.array(dtype=wp.vec3i),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.bool),
            wp.array(dtype=wp.bool),
        ],
    )


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


# Generate overload dictionary for neighbor list rebuild kernel
_check_atoms_moved_beyond_skin_overload = {}
for t, v in zip(_T, _V):
    _check_atoms_moved_beyond_skin_overload[t] = wp.overload(
        _check_atoms_moved_beyond_skin,
        [
            wp.array(dtype=v),
            wp.array(dtype=v),
            t,
            wp.array(dtype=wp.bool),
        ],
    )


###########################################################################################
########################### Warp Launchers ###############################################
###########################################################################################


def check_cell_list_rebuild(
    current_positions: wp.array,
    atom_to_cell_mapping: wp.array,
    cells_per_dimension: wp.array,
    cell: wp.array,
    pbc: wp.array,
    rebuild_flag: wp.array,
    wp_dtype: type,
    device: str,
) -> None:
    """Core warp launcher for detecting if cell list needs rebuilding.

    Checks if any atoms have moved between spatial cells since the cell list was built.

    Parameters
    ----------
    current_positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Current atomic coordinates in Cartesian space.
    atom_to_cell_mapping : wp.array, shape (total_atoms, 3), dtype=wp.vec3i
        Previously computed cell coordinates for each atom.
    cells_per_dimension : wp.array, shape (3,), dtype=wp.int32
        Number of cells in x, y, z directions.
    cell : wp.array, shape (1, 3, 3), dtype=wp.mat33*
        Unit cell matrix for coordinate transformations.
    pbc : wp.array, shape (3,), dtype=wp.bool
        Periodic boundary condition flags.
    rebuild_flag : wp.array, shape (1,), dtype=wp.bool
        OUTPUT: Flag set to True if rebuild is needed.
    wp_dtype : type
        Warp dtype (wp.float32, wp.float64, or wp.float16).
    device : str
        Warp device string (e.g., 'cuda:0', 'cpu').

    Notes
    -----
    - This is a low-level warp interface. For framework bindings, use torch/jax wrappers.
    - rebuild_flag must be pre-allocated and initialized to False by caller.

    See Also
    --------
    _check_atoms_changed_cells : Kernel that performs the check
    """
    total_atoms = current_positions.shape[0]

    wp.launch(
        kernel=_check_atoms_changed_cells_overload[wp_dtype],
        dim=total_atoms,
        inputs=[
            current_positions,
            cell,
            atom_to_cell_mapping,
            cells_per_dimension,
            pbc,
            rebuild_flag,
        ],
        device=device,
    )


def check_neighbor_list_rebuild(
    reference_positions: wp.array,
    current_positions: wp.array,
    skin_distance_threshold: float,
    rebuild_flag: wp.array,
    wp_dtype: type,
    device: str,
) -> None:
    """Core warp launcher for detecting if neighbor list needs rebuilding.

    Checks if any atoms have moved beyond the skin distance since the neighbor list was built.

    Parameters
    ----------
    reference_positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Atomic positions when the neighbor list was last built.
    current_positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Current atomic positions to compare against reference.
    skin_distance_threshold : float
        Maximum allowed displacement before neighbor list becomes invalid.
    rebuild_flag : wp.array, shape (1,), dtype=wp.bool
        OUTPUT: Flag set to True if rebuild is needed.
    wp_dtype : type
        Warp dtype (wp.float32, wp.float64, or wp.float16).
    device : str
        Warp device string (e.g., 'cuda:0', 'cpu').

    Notes
    -----
    - This is a low-level warp interface. For framework bindings, use torch/jax wrappers.
    - rebuild_flag must be pre-allocated and initialized to False by caller.

    See Also
    --------
    _check_atoms_moved_beyond_skin : Kernel that performs the check
    """
    total_atoms = reference_positions.shape[0]

    wp.launch(
        kernel=_check_atoms_moved_beyond_skin_overload[wp_dtype],
        dim=total_atoms,
        inputs=[
            reference_positions,
            current_positions,
            wp_dtype(skin_distance_threshold),
            rebuild_flag,
        ],
        device=device,
    )
