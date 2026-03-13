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

from nvalchemiops.neighbors.neighbor_utils import (
    update_ref_positions,
    update_ref_positions_batch,
)

__all__ = [
    "check_cell_list_rebuild",
    "check_neighbor_list_rebuild",
    "check_batch_neighbor_list_rebuild",
    "check_batch_cell_list_rebuild",
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
############### PBC Neighbor List Rebuild Detection (Periodic-Aware) #####################
###########################################################################################


@wp.kernel(enable_backward=False)
def _check_atoms_moved_beyond_skin_pbc(
    reference_positions: wp.array(dtype=Any),
    current_positions: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    cell_inv: wp.array(dtype=Any),
    pbc: wp.array(dtype=wp.bool),
    skin_distance_threshold: Any,
    rebuild_flag: wp.array(dtype=wp.bool),
) -> None:
    """Detect if atoms moved beyond skin distance using minimum-image convention.

    Unlike ``_check_atoms_moved_beyond_skin`` which uses raw Euclidean
    displacement, this kernel applies the minimum-image convention (MIC)
    so that atoms crossing periodic boundaries are not spuriously flagged.

    Parameters
    ----------
    reference_positions : wp.array, shape (total_atoms,), dtype=wp.vec3*
        Atomic positions when the neighbor list was last built.
    current_positions : wp.array, shape (total_atoms,), dtype=wp.vec3*
        Current atomic positions to compare against reference.
    cell : wp.array, shape (1,), dtype=wp.mat33*
        Unit cell matrix (basis vectors as rows).
    cell_inv : wp.array, shape (1,), dtype=wp.mat33*
        Precomputed inverse of the cell matrix.
    pbc : wp.array, shape (3,), dtype=bool
        Periodic boundary condition flags for x, y, z directions.
    skin_distance_threshold : float*/int*
        Maximum allowed displacement before neighbor list becomes invalid.
    rebuild_flag : wp.array, shape (1,), dtype=bool
        OUTPUT: Flag set to True if any atom moved beyond skin distance.

    Notes
    -----
    - Thread launch: One thread per atom (dim=total_atoms)
    - Modifies: rebuild_flag (atomic write)
    - Correct for triclinic cells; avoids per-thread matrix inversion
    """
    atom_idx = wp.tid()

    if atom_idx >= reference_positions.shape[0]:
        return

    if rebuild_flag[0]:
        return

    delta = current_positions[atom_idx] - reference_positions[atom_idx]

    # Convert displacement to fractional coordinates (row-vector convention)
    delta_frac = delta * cell_inv[0]

    # Apply minimum-image convention on periodic dimensions
    for dim in range(3):
        if pbc[dim]:
            delta_frac[dim] -= wp.floor(delta_frac[dim] + type(delta_frac[dim])(0.5))

    # Convert back to Cartesian
    delta_cart = delta_frac * cell[0]
    displacement_magnitude = wp.length(delta_cart)

    if displacement_magnitude > skin_distance_threshold:
        rebuild_flag[0] = True


_check_atoms_moved_beyond_skin_pbc_overload = {}
for t, v, m in zip(_T, _V, _M):
    _check_atoms_moved_beyond_skin_pbc_overload[t] = wp.overload(
        _check_atoms_moved_beyond_skin_pbc,
        [
            wp.array(dtype=v),
            wp.array(dtype=v),
            wp.array(dtype=m),
            wp.array(dtype=m),
            wp.array(dtype=wp.bool),
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
    update_reference_positions: bool = False,
    cell: wp.array | None = None,
    cell_inv: wp.array | None = None,
    pbc: wp.array | None = None,
) -> None:
    """Core warp launcher for detecting if neighbor list needs rebuilding.

    Checks if any atoms have moved beyond the skin distance since the neighbor
    list was built.  When ``cell``, ``cell_inv`` and ``pbc`` are all provided
    the check uses minimum-image convention (MIC) so that atoms crossing
    periodic boundaries are not spuriously flagged.

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
    update_reference_positions : bool, optional
        If True, overwrite ``reference_positions`` with ``current_positions``
        for all atoms when a rebuild is detected. The update runs in a second
        kernel launch after the detection kernel, so every atom is guaranteed
        to be updated with no race conditions. Default False.
    cell : wp.array or None, optional
        Unit cell matrix, shape (1,), dtype=wp.mat33*.  Required together with
        ``cell_inv`` and ``pbc`` to enable MIC displacement.
    cell_inv : wp.array or None, optional
        Precomputed inverse of the cell matrix, same shape/dtype as ``cell``.
    pbc : wp.array or None, optional
        Periodic boundary condition flags, shape (3,), dtype=wp.bool.

    Notes
    -----
    - This is a low-level warp interface. For framework bindings, use torch/jax wrappers.
    - rebuild_flag must be pre-allocated and initialized to False by caller.

    Raises
    ------
    ValueError
        If only a subset of ``cell``, ``cell_inv``, and ``pbc`` are provided.
        All three must be supplied together to enable MIC displacement.

    See Also
    --------
    _check_atoms_moved_beyond_skin : Euclidean kernel
    _check_atoms_moved_beyond_skin_pbc : PBC kernel for periodic systems
    update_ref_positions : Standalone reference-position update launcher
    """
    pbc_params = (cell, cell_inv, pbc)
    if any(p is not None for p in pbc_params) and not all(
        p is not None for p in pbc_params
    ):
        raise ValueError(
            "cell, cell_inv, and pbc must all be provided together to enable MIC "
            "displacement checking. Received a partial set."
        )
    total_atoms = reference_positions.shape[0]
    use_pbc = cell is not None
    if use_pbc:
        wp.launch(
            kernel=_check_atoms_moved_beyond_skin_pbc_overload[wp_dtype],
            dim=total_atoms,
            inputs=[
                reference_positions,
                current_positions,
                cell,
                cell_inv,
                pbc,
                wp_dtype(skin_distance_threshold),
                rebuild_flag,
            ],
            device=device,
        )
    else:
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
    if update_reference_positions:
        update_ref_positions(
            current_positions, rebuild_flag, reference_positions, wp_dtype, device
        )


###########################################################################################
########################### Batch Neighbor List Rebuild Detection ########################
###########################################################################################


@wp.kernel(enable_backward=False)
def _check_batch_atoms_moved_beyond_skin(
    reference_positions: wp.array(dtype=Any),
    current_positions: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    skin_distance_threshold: Any,
    rebuild_flags: wp.array(dtype=wp.bool),
) -> None:
    """Detect per-system if atoms moved beyond skin distance requiring neighbor list rebuild.

    Checks each atom's displacement from its reference position against the skin distance
    threshold. When any atom in a system exceeds this threshold, the system's rebuild flag
    is set to True. Uses early termination per system for efficiency.

    Parameters
    ----------
    reference_positions : wp.array, shape (total_atoms,), dtype=wp.vec3*
        Atomic positions when each system's neighbor list was last built.
    current_positions : wp.array, shape (total_atoms,), dtype=wp.vec3*
        Current atomic positions to compare against reference.
    batch_idx : wp.array, shape (total_atoms,), dtype=wp.int32
        System index for each atom.
    skin_distance_threshold : float*
        Maximum allowed displacement before neighbor list becomes invalid.
        Typically set to (cutoff_radius - cutoff) / 2.
    rebuild_flags : wp.array, shape (num_systems,), dtype=bool
        OUTPUT: Per-system flags set to True if any atom in that system moved beyond
        skin distance (modified per system).

    Notes
    -----
    - Thread launch: One thread per atom (dim=total_atoms)
    - Modifies: rebuild_flags
    - Early termination: Threads exit if their system's rebuild flag is already set
    - Displacement calculation uses Euclidean distance
    - No CPU-GPU synchronization required; flags are set entirely on GPU
    """
    atom_idx = wp.tid()

    if atom_idx >= reference_positions.shape[0]:
        return

    isys = batch_idx[atom_idx]

    # Skip computation if rebuild already flagged for this system
    if rebuild_flags[isys]:
        return

    displacement_vector = current_positions[atom_idx] - reference_positions[atom_idx]
    displacement_magnitude = wp.length(displacement_vector)

    if displacement_magnitude > skin_distance_threshold:
        rebuild_flags[isys] = True


# Generate overload dictionary for batch neighbor list rebuild kernel
_check_batch_atoms_moved_beyond_skin_overload = {}
for t, v in zip(_T, _V):
    _check_batch_atoms_moved_beyond_skin_overload[t] = wp.overload(
        _check_batch_atoms_moved_beyond_skin,
        [
            wp.array(dtype=v),
            wp.array(dtype=v),
            wp.array(dtype=wp.int32),
            t,
            wp.array(dtype=wp.bool),
        ],
    )


###########################################################################################
############ PBC Batch Neighbor List Rebuild Detection (Periodic-Aware) ##################
###########################################################################################


@wp.kernel(enable_backward=False)
def _check_batch_atoms_moved_beyond_skin_pbc(
    reference_positions: wp.array(dtype=Any),
    current_positions: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    cell: wp.array(dtype=Any),
    cell_inv: wp.array(dtype=Any),
    pbc: wp.array2d(dtype=wp.bool),
    skin_distance_threshold: Any,
    rebuild_flags: wp.array(dtype=wp.bool),
) -> None:
    """Per-system PBC-aware skin-distance rebuild detection.

    Like ``_check_batch_atoms_moved_beyond_skin`` but applies minimum-image
    convention per system so atoms wrapping across periodic boundaries are
    not spuriously flagged.

    Parameters
    ----------
    reference_positions : wp.array, shape (total_atoms,), dtype=wp.vec3*
        Atomic positions when each system's neighbor list was last built.
    current_positions : wp.array, shape (total_atoms,), dtype=wp.vec3*
        Current atomic positions to compare against reference.
    batch_idx : wp.array, shape (total_atoms,), dtype=wp.int32
        System index for each atom.
    cell : wp.array, shape (num_systems,), dtype=wp.mat33*
        Per-system unit cell matrices (basis vectors as rows).
    cell_inv : wp.array, shape (num_systems,), dtype=wp.mat33*
        Precomputed per-system inverse cell matrices.
    pbc : wp.array2d, shape (num_systems, 3), dtype=bool
        Per-system periodic boundary condition flags.
    skin_distance_threshold : float*
        Maximum allowed displacement before neighbor list becomes invalid.
    rebuild_flags : wp.array, shape (num_systems,), dtype=bool
        OUTPUT: Per-system flags set to True if any atom moved beyond skin
        distance.

    Notes
    -----
    - Thread launch: One thread per atom (dim=total_atoms)
    - Modifies: rebuild_flags
    - Correct for triclinic cells; avoids per-thread matrix inversion
    """
    atom_idx = wp.tid()

    if atom_idx >= reference_positions.shape[0]:
        return

    isys = batch_idx[atom_idx]

    if rebuild_flags[isys]:
        return

    delta = current_positions[atom_idx] - reference_positions[atom_idx]

    # Convert displacement to fractional coordinates (row-vector convention)
    delta_frac = delta * cell_inv[isys]

    # Apply minimum-image convention on periodic dimensions
    for dim in range(3):
        if pbc[isys, dim]:
            delta_frac[dim] -= wp.floor(delta_frac[dim] + type(delta_frac[dim])(0.5))

    # Convert back to Cartesian
    delta_cart = delta_frac * cell[isys]
    displacement_magnitude = wp.length(delta_cart)

    if displacement_magnitude > skin_distance_threshold:
        rebuild_flags[isys] = True


_check_batch_atoms_moved_beyond_skin_pbc_overload = {}
for t, v, m in zip(_T, _V, _M):
    _check_batch_atoms_moved_beyond_skin_pbc_overload[t] = wp.overload(
        _check_batch_atoms_moved_beyond_skin_pbc,
        [
            wp.array(dtype=v),
            wp.array(dtype=v),
            wp.array(dtype=wp.int32),
            wp.array(dtype=m),
            wp.array(dtype=m),
            wp.array2d(dtype=wp.bool),
            t,
            wp.array(dtype=wp.bool),
        ],
    )


###########################################################################################
########################### Batch Cell List Rebuild Detection ############################
###########################################################################################


@wp.kernel(enable_backward=False)
def _check_batch_atoms_changed_cells(
    current_positions: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    atom_to_cell_mapping: wp.array(dtype=wp.vec3i),
    batch_idx: wp.array(dtype=wp.int32),
    cells_per_dimension: wp.array(dtype=wp.vec3i),
    pbc: wp.array2d(dtype=wp.bool),
    rebuild_flags: wp.array(dtype=wp.bool),
) -> None:
    """Detect per-system if atoms moved between cells requiring cell list rebuild.

    Computes current cell assignments for each atom and compares with stored
    cell assignments. When any atom in a system has crossed a cell boundary,
    that system's rebuild flag is set to True. Uses early termination per system.

    Parameters
    ----------
    current_positions : wp.array, shape (total_atoms,), dtype=wp.vec3*
        Current Cartesian coordinates.
    cell : wp.array, shape (num_systems,), dtype=wp.mat33*
        Per-system unit cell matrices for coordinate transformations.
    atom_to_cell_mapping : wp.array, shape (total_atoms,), dtype=wp.vec3i
        Previously computed cell coordinates for each atom from existing cell lists.
    batch_idx : wp.array, shape (total_atoms,), dtype=wp.int32
        System index for each atom.
    cells_per_dimension : wp.array, shape (num_systems,), dtype=wp.vec3i
        Number of cells in x, y, z directions for each system.
    pbc : wp.array2d, shape (num_systems, 3), dtype=bool
        Per-system periodic boundary condition flags.
    rebuild_flags : wp.array, shape (num_systems,), dtype=bool
        OUTPUT: Per-system flags set to True if any atom changed cells.

    Notes
    -----
    - Thread launch: One thread per atom (dim=total_atoms)
    - Modifies: rebuild_flags
    - Early termination: Threads exit if their system's rebuild flag is already set
    - Handles periodic boundaries with proper wrapping per system
    - No CPU-GPU synchronization required; flags are set entirely on GPU
    """
    atom_idx = wp.tid()

    if atom_idx >= current_positions.shape[0]:
        return

    isys = batch_idx[atom_idx]

    # Skip computation if rebuild already flagged for this system
    if rebuild_flags[isys]:
        return

    _cell = cell[isys]
    _cpd = cells_per_dimension[isys]

    # Transform current position to fractional coordinates (row-vector convention)
    _inv_cell = wp.inverse(_cell)
    fractional_position = current_positions[atom_idx] * _inv_cell
    current_cell_coords = wp.vec3i(0, 0, 0)

    # Compute current cell coordinates for each dimension
    for dim in range(3):
        current_cell_coords[dim] = wp.int32(
            wp.floor(
                fractional_position[dim] * type(fractional_position[dim])(_cpd[dim])
            )
        )

        # Handle periodic boundary conditions
        if pbc[isys, dim]:
            current_cell_coords[dim] = current_cell_coords[dim] % _cpd[dim]
            if current_cell_coords[dim] < 0:
                current_cell_coords[dim] += _cpd[dim]
        else:
            current_cell_coords[dim] = wp.clamp(
                current_cell_coords[dim], 0, _cpd[dim] - 1
            )

    # Compare with stored cell coordinates from existing cell list
    stored_cell_coords = atom_to_cell_mapping[atom_idx]

    if (
        current_cell_coords[0] != stored_cell_coords[0]
        or current_cell_coords[1] != stored_cell_coords[1]
        or current_cell_coords[2] != stored_cell_coords[2]
    ):
        rebuild_flags[isys] = True


# Generate overload dictionary for batch cell list rebuild kernel
_check_batch_atoms_changed_cells_overload = {}
for t, v, m in zip(_T, _V, _M):
    _check_batch_atoms_changed_cells_overload[t] = wp.overload(
        _check_batch_atoms_changed_cells,
        [
            wp.array(dtype=v),
            wp.array(dtype=m),
            wp.array(dtype=wp.vec3i),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.vec3i),
            wp.array2d(dtype=wp.bool),
            wp.array(dtype=wp.bool),
        ],
    )


###########################################################################################
########################### Batch Warp Launchers #########################################
###########################################################################################


def check_batch_neighbor_list_rebuild(
    reference_positions: wp.array,
    current_positions: wp.array,
    batch_idx: wp.array,
    skin_distance_threshold: float,
    rebuild_flags: wp.array,
    wp_dtype: type,
    device: str,
    update_reference_positions: bool = False,
    cell: wp.array | None = None,
    cell_inv: wp.array | None = None,
    pbc: wp.array | None = None,
) -> None:
    """Core warp launcher for detecting per-system neighbor list rebuild needs.

    Checks if any atoms in each system have moved beyond the skin distance since
    the neighbor list was built. Sets per-system rebuild flags on GPU without
    requiring CPU synchronization.

    When ``cell``, ``cell_inv`` and ``pbc`` are all provided the check uses
    minimum-image convention (MIC) so that atoms crossing periodic boundaries
    are not spuriously flagged.

    Parameters
    ----------
    reference_positions : wp.array, shape (total_atoms,), dtype=wp.vec3*
        Atomic positions when each system's neighbor list was last built.
    current_positions : wp.array, shape (total_atoms,), dtype=wp.vec3*
        Current atomic positions to compare against reference.
    batch_idx : wp.array, shape (total_atoms,), dtype=wp.int32
        System index for each atom.
    skin_distance_threshold : float
        Maximum allowed displacement before neighbor list becomes invalid.
    rebuild_flags : wp.array, shape (num_systems,), dtype=wp.bool
        OUTPUT: Per-system flags set to True if rebuild is needed.
        Must be pre-allocated and initialized to False by caller.
    wp_dtype : type
        Warp dtype (wp.float32, wp.float64, or wp.float16).
    device : str
        Warp device string (e.g., 'cuda:0', 'cpu').
    update_reference_positions : bool, optional
        If True, overwrite ``reference_positions`` with ``current_positions``
        for all atoms in rebuilt systems when a rebuild is detected. The update
        runs in a second kernel launch after the detection kernel, so every atom
        in each rebuilt system is guaranteed to be updated with no race
        conditions. Default False.
    cell : wp.array or None, optional
        Per-system cell matrices, shape (num_systems,), dtype=wp.mat33*.
        Required together with ``cell_inv`` and ``pbc`` to enable MIC.
    cell_inv : wp.array or None, optional
        Precomputed per-system inverse cell matrices, same shape/dtype as
        ``cell``.
    pbc : wp.array or None, optional
        Per-system PBC flags, shape (num_systems, 3), dtype=wp.bool (2D).

    Notes
    -----
    - This is a low-level warp interface. For framework bindings, use torch/jax wrappers.
    - rebuild_flags must be pre-allocated and initialized to False by caller.
    - No CPU-GPU synchronization required; flags are written entirely on GPU.

    Raises
    ------
    ValueError
        If only a subset of ``cell``, ``cell_inv``, and ``pbc`` are provided.
        All three must be supplied together to enable MIC displacement.

    See Also
    --------
    _check_batch_atoms_moved_beyond_skin : Euclidean kernel
    _check_batch_atoms_moved_beyond_skin_pbc : PBC kernel for periodic systems
    update_ref_positions_batch : Standalone reference-position update launcher
    """
    pbc_params = (cell, cell_inv, pbc)
    if any(p is not None for p in pbc_params) and not all(
        p is not None for p in pbc_params
    ):
        raise ValueError(
            "cell, cell_inv, and pbc must all be provided together to enable MIC "
            "displacement checking. Received a partial set."
        )
    total_atoms = reference_positions.shape[0]
    use_pbc = cell is not None
    if use_pbc:
        wp.launch(
            kernel=_check_batch_atoms_moved_beyond_skin_pbc_overload[wp_dtype],
            dim=total_atoms,
            inputs=[
                reference_positions,
                current_positions,
                batch_idx,
                cell,
                cell_inv,
                pbc,
                wp_dtype(skin_distance_threshold),
                rebuild_flags,
            ],
            device=device,
        )
    else:
        wp.launch(
            kernel=_check_batch_atoms_moved_beyond_skin_overload[wp_dtype],
            dim=total_atoms,
            inputs=[
                reference_positions,
                current_positions,
                batch_idx,
                wp_dtype(skin_distance_threshold),
                rebuild_flags,
            ],
            device=device,
        )
    if update_reference_positions:
        update_ref_positions_batch(
            current_positions,
            rebuild_flags,
            batch_idx,
            reference_positions,
            wp_dtype,
            device,
        )


def check_batch_cell_list_rebuild(
    current_positions: wp.array,
    atom_to_cell_mapping: wp.array,
    batch_idx: wp.array,
    cells_per_dimension: wp.array,
    cell: wp.array,
    pbc: wp.array,
    rebuild_flags: wp.array,
    wp_dtype: type,
    device: str,
) -> None:
    """Core warp launcher for detecting per-system cell list rebuild needs.

    Checks if any atoms in each system have moved between spatial cells since
    the cell list was built. Sets per-system rebuild flags on GPU without
    requiring CPU synchronization.

    Parameters
    ----------
    current_positions : wp.array, shape (total_atoms,), dtype=wp.vec3*
        Current Cartesian coordinates.
    atom_to_cell_mapping : wp.array, shape (total_atoms,), dtype=wp.vec3i
        Previously computed cell coordinates for each atom.
    batch_idx : wp.array, shape (total_atoms,), dtype=wp.int32
        System index for each atom.
    cells_per_dimension : wp.array, shape (num_systems,), dtype=wp.vec3i
        Number of cells in x, y, z directions for each system.
    cell : wp.array, shape (num_systems,), dtype=wp.mat33*
        Per-system unit cell matrices for coordinate transformations.
    pbc : wp.array, shape (num_systems, 3), dtype=wp.bool
        Per-system periodic boundary condition flags (2D array).
    rebuild_flags : wp.array, shape (num_systems,), dtype=wp.bool
        OUTPUT: Per-system flags set to True if rebuild is needed.
        Must be pre-allocated and initialized to False by caller.
    wp_dtype : type
        Warp dtype (wp.float32, wp.float64, or wp.float16).
    device : str
        Warp device string (e.g., 'cuda:0', 'cpu').

    Notes
    -----
    - This is a low-level warp interface. For framework bindings, use torch/jax wrappers.
    - rebuild_flags must be pre-allocated and initialized to False by caller.
    - No CPU-GPU synchronization required; flags are written entirely on GPU.

    See Also
    --------
    _check_batch_atoms_changed_cells : Kernel that performs the check
    """
    total_atoms = current_positions.shape[0]
    wp.launch(
        kernel=_check_batch_atoms_changed_cells_overload[wp_dtype],
        dim=total_atoms,
        inputs=[
            current_positions,
            cell,
            atom_to_cell_mapping,
            batch_idx,
            cells_per_dimension,
            pbc,
            rebuild_flags,
        ],
        device=device,
    )
