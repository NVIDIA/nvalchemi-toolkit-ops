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

"""JAX bindings for rebuild detection.

This module provides JAX functions for detecting when cell lists and neighbor lists
need to be rebuilt.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import warp as wp
from warp.jax_experimental import jax_kernel

from nvalchemiops.neighbors.rebuild_detection import (
    _check_atoms_changed_cells_overload,
    _check_atoms_moved_beyond_skin_overload,
)

__all__ = [
    "cell_list_needs_rebuild",
    "neighbor_list_needs_rebuild",
    "check_cell_list_rebuild_needed",
    "check_neighbor_list_rebuild_needed",
]

# ==============================================================================
# JAX Kernel Wrappers
# ==============================================================================

# Cell list rebuild detection kernel wrappers
_jax_check_cells_f32 = jax_kernel(
    _check_atoms_changed_cells_overload[wp.float32],
    num_outputs=1,
    in_out_argnames=["rebuild_flag"],
    enable_backward=False,
)
_jax_check_cells_f64 = jax_kernel(
    _check_atoms_changed_cells_overload[wp.float64],
    num_outputs=1,
    in_out_argnames=["rebuild_flag"],
    enable_backward=False,
)

# Neighbor list rebuild detection kernel wrappers
_jax_check_skin_f32 = jax_kernel(
    _check_atoms_moved_beyond_skin_overload[wp.float32],
    num_outputs=1,
    in_out_argnames=["rebuild_flag"],
    enable_backward=False,
)
_jax_check_skin_f64 = jax_kernel(
    _check_atoms_moved_beyond_skin_overload[wp.float64],
    num_outputs=1,
    in_out_argnames=["rebuild_flag"],
    enable_backward=False,
)


# ==============================================================================
# Cell List Rebuild Detection
# ==============================================================================


def cell_list_needs_rebuild(
    current_positions: jax.Array,
    atom_to_cell_mapping: jax.Array,
    cells_per_dimension: jax.Array,
    cell: jax.Array,
    pbc: jax.Array,
) -> jax.Array:
    """Detect if spatial cell list requires rebuilding due to atomic motion.

    Parameters
    ----------
    current_positions : jax.Array, shape (total_atoms, 3)
        Current atomic coordinates in Cartesian space.
    atom_to_cell_mapping : jax.Array, shape (total_atoms, 3), dtype=int32
        3D cell coordinates for each atom from the existing cell list.
    cells_per_dimension : jax.Array, shape (3,), dtype=int32
        Number of spatial cells in x, y, z directions.
    cell : jax.Array, shape (1, 3, 3)
        Unit cell matrix for coordinate transformations.
    pbc : jax.Array, shape (3,), dtype=bool
        Periodic boundary condition flags for x, y, z directions.

    Returns
    -------
    rebuild_needed : jax.Array, shape (1,), dtype=bool
        True if any atom has moved to a different cell requiring rebuild.

    Notes
    -----
    This function is not differentiable and should not be used in JAX transformations
    that require gradients.

    See Also
    --------
    nvalchemiops.neighbors.rebuild_detection.check_cell_list_rebuild : Core warp launcher
    check_cell_list_rebuild_needed : Convenience wrapper that returns Python bool
    """
    total_atoms = current_positions.shape[0]

    if total_atoms == 0:
        return jnp.array([False], dtype=jnp.bool_)

    # Ensure cell dtype matches positions dtype so warp overload dispatch is consistent
    if cell.dtype != current_positions.dtype:
        cell = cell.astype(current_positions.dtype)

    # Ensure pbc is bool
    pbc = pbc.astype(jnp.bool_)

    # Squeeze cells_per_dimension to 1D if needed
    cells_1d = (
        cells_per_dimension.squeeze()
        if cells_per_dimension.ndim == 2
        else cells_per_dimension
    )

    # Allocate output
    rebuild_flag = jnp.array([False], dtype=jnp.bool_)

    # Select kernel based on dtype
    if current_positions.dtype == jnp.float64:
        _jax_check = _jax_check_cells_f64
    else:
        _jax_check = _jax_check_cells_f32
        current_positions = current_positions.astype(jnp.float32)

    # Call kernel
    (rebuild_flag,) = _jax_check(
        current_positions,
        cell,
        atom_to_cell_mapping,
        cells_1d,
        pbc,
        rebuild_flag,
        launch_dims=(total_atoms,),
    )

    return rebuild_flag


def neighbor_list_needs_rebuild(
    reference_positions: jax.Array,
    current_positions: jax.Array,
    skin_distance_threshold: float,
) -> jax.Array:
    """Detect if neighbor list requires rebuilding due to excessive atomic motion.

    Parameters
    ----------
    reference_positions : jax.Array, shape (total_atoms, 3)
        Atomic positions when the neighbor list was last built.
    current_positions : jax.Array, shape (total_atoms, 3)
        Current atomic positions to compare against reference.
    skin_distance_threshold : float
        Maximum allowed displacement before neighbor list becomes invalid.

    Returns
    -------
    rebuild_needed : jax.Array, shape (1,), dtype=bool
        True if any atom has moved beyond skin distance.

    Notes
    -----
    The skin distance approach allows neighbor lists to remain valid even when atoms
    move slightly, reducing the frequency of expensive neighbor list reconstructions.

    This function is not differentiable and should not be used in JAX transformations
    that require gradients.

    See Also
    --------
    nvalchemiops.neighbors.rebuild_detection.check_neighbor_list_rebuild : Core warp launcher
    check_neighbor_list_rebuild_needed : Convenience wrapper that returns Python bool
    """
    # Check for shape compatibility
    if reference_positions.shape != current_positions.shape:
        return jnp.array([True], dtype=jnp.bool_)

    total_atoms = reference_positions.shape[0]

    if total_atoms == 0:
        return jnp.array([False], dtype=jnp.bool_)

    # Allocate output
    rebuild_flag = jnp.array([False], dtype=jnp.bool_)

    # Select kernel based on dtype
    if reference_positions.dtype == jnp.float64:
        _jax_check = _jax_check_skin_f64
    else:
        _jax_check = _jax_check_skin_f32
        reference_positions = reference_positions.astype(jnp.float32)
        current_positions = current_positions.astype(jnp.float32)

    # Call kernel
    (rebuild_flag,) = _jax_check(
        reference_positions,
        current_positions,
        float(skin_distance_threshold),
        rebuild_flag,
        launch_dims=(total_atoms,),
    )

    return rebuild_flag


# ==============================================================================
# High-level API Functions
# ==============================================================================


def check_cell_list_rebuild_needed(
    current_positions: jax.Array,
    atom_to_cell_mapping: jax.Array,
    cells_per_dimension: jax.Array,
    cell: jax.Array,
    pbc: jax.Array,
) -> bool:
    """Determine if spatial cell list requires rebuilding based on atomic motion.

    This high-level convenience function determines if a spatial cell list needs to be
    reconstructed due to atomic movement. It uses GPU acceleration to efficiently detect
    when atoms have moved between spatial cells.

    Parameters
    ----------
    current_positions : jax.Array, shape (total_atoms, 3)
        Current atomic coordinates to check against existing cell assignments.
    atom_to_cell_mapping : jax.Array, shape (total_atoms, 3), dtype=int32
        3D cell coordinates assigned to each atom from existing cell list.
    cells_per_dimension : jax.Array, shape (3,), dtype=int32
        Number of spatial cells in x, y, z directions from existing cell list.
    cell : jax.Array, shape (1, 3, 3)
        Current unit cell matrix for coordinate transformations.
    pbc : jax.Array, shape (3,), dtype=bool
        Current periodic boundary condition flags for x, y, z directions.

    Returns
    -------
    needs_rebuild : bool
        True if any atom has moved to a different cell requiring cell list rebuild.

    Notes
    -----
    This function is not differentiable and should not be used in JAX transformations
    that require gradients.

    See Also
    --------
    cell_list_needs_rebuild : Returns jax.Array instead of bool
    """
    rebuild_tensor = cell_list_needs_rebuild(
        current_positions,
        atom_to_cell_mapping,
        cells_per_dimension,
        cell,
        pbc,
    )

    return bool(rebuild_tensor[0])


def check_neighbor_list_rebuild_needed(
    reference_positions: jax.Array,
    current_positions: jax.Array,
    skin_distance_threshold: float,
) -> bool:
    """Determine if neighbor list requires rebuilding based on atomic motion.

    This high-level function provides a convenient interface to check if a neighbor
    list needs reconstruction due to excessive atomic movement. Uses the skin distance
    approach to minimize unnecessary neighbor list rebuilds during MD simulations.

    The skin distance technique allows atoms to move slightly without invalidating
    the neighbor list, reducing computational overhead. When any atom moves beyond
    the skin distance, the neighbor list must be rebuilt to maintain accuracy.

    Parameters
    ----------
    reference_positions : jax.Array, shape (total_atoms, 3)
        Atomic coordinates when the neighbor list was last constructed.
    current_positions : jax.Array, shape (total_atoms, 3)
        Current atomic coordinates to compare against reference positions.
    skin_distance_threshold : float
        Maximum allowed atomic displacement before neighbor list becomes invalid.
        Typically set to (cutoff_radius - cutoff) / 2 for safety.

    Returns
    -------
    needs_rebuild : bool
        True if any atom has moved beyond skin distance requiring neighbor list rebuild.

    Notes
    -----
    This function is not differentiable and should not be used in JAX transformations
    that require gradients.

    See Also
    --------
    neighbor_list_needs_rebuild : Returns jax.Array instead of bool
    """
    rebuild_tensor = neighbor_list_needs_rebuild(
        reference_positions, current_positions, skin_distance_threshold
    )

    return bool(rebuild_tensor[0])
