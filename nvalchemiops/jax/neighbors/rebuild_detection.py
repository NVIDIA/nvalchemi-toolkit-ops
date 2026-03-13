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
    _check_atoms_moved_beyond_skin_pbc_overload,
    _check_batch_atoms_changed_cells_overload,
    _check_batch_atoms_moved_beyond_skin_overload,
    _check_batch_atoms_moved_beyond_skin_pbc_overload,
)

__all__ = [
    "cell_list_needs_rebuild",
    "neighbor_list_needs_rebuild",
    "check_cell_list_rebuild_needed",
    "check_neighbor_list_rebuild_needed",
    "batch_neighbor_list_needs_rebuild",
    "batch_cell_list_needs_rebuild",
    "check_batch_neighbor_list_rebuild_needed",
    "check_batch_cell_list_rebuild_needed",
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

# Batch neighbor list rebuild detection kernel wrappers
_jax_batch_check_skin_f32 = jax_kernel(
    _check_batch_atoms_moved_beyond_skin_overload[wp.float32],
    num_outputs=1,
    in_out_argnames=["rebuild_flags"],
    enable_backward=False,
)
_jax_batch_check_skin_f64 = jax_kernel(
    _check_batch_atoms_moved_beyond_skin_overload[wp.float64],
    num_outputs=1,
    in_out_argnames=["rebuild_flags"],
    enable_backward=False,
)

# MIC neighbor list rebuild detection kernel wrappers
_jax_check_skin_pbc_f32 = jax_kernel(
    _check_atoms_moved_beyond_skin_pbc_overload[wp.float32],
    num_outputs=1,
    in_out_argnames=["rebuild_flag"],
    enable_backward=False,
)
_jax_check_skin_pbc_f64 = jax_kernel(
    _check_atoms_moved_beyond_skin_pbc_overload[wp.float64],
    num_outputs=1,
    in_out_argnames=["rebuild_flag"],
    enable_backward=False,
)

# MIC batch neighbor list rebuild detection kernel wrappers
_jax_batch_check_skin_pbc_f32 = jax_kernel(
    _check_batch_atoms_moved_beyond_skin_pbc_overload[wp.float32],
    num_outputs=1,
    in_out_argnames=["rebuild_flags"],
    enable_backward=False,
)
_jax_batch_check_skin_pbc_f64 = jax_kernel(
    _check_batch_atoms_moved_beyond_skin_pbc_overload[wp.float64],
    num_outputs=1,
    in_out_argnames=["rebuild_flags"],
    enable_backward=False,
)

# Batch cell list rebuild detection kernel wrappers
_jax_batch_check_cells_f32 = jax_kernel(
    _check_batch_atoms_changed_cells_overload[wp.float32],
    num_outputs=1,
    in_out_argnames=["rebuild_flags"],
    enable_backward=False,
)
_jax_batch_check_cells_f64 = jax_kernel(
    _check_batch_atoms_changed_cells_overload[wp.float64],
    num_outputs=1,
    in_out_argnames=["rebuild_flags"],
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
    cell: jax.Array | None = None,
    cell_inv: jax.Array | None = None,
    pbc: jax.Array | None = None,
) -> jax.Array:
    """Detect if neighbor list requires rebuilding due to excessive atomic motion.

    When ``cell``, ``cell_inv`` and ``pbc`` are all provided, uses minimum-image
    convention (MIC) so atoms crossing periodic boundaries are not spuriously
    flagged.

    Parameters
    ----------
    reference_positions : jax.Array, shape (total_atoms, 3)
        Atomic positions when the neighbor list was last built.
    current_positions : jax.Array, shape (total_atoms, 3)
        Current atomic positions to compare against reference.
    skin_distance_threshold : float
        Maximum allowed displacement before neighbor list becomes invalid.
    cell : jax.Array or None, optional
        Unit cell matrix, shape (1, 3, 3).
    cell_inv : jax.Array or None, optional
        Inverse cell matrix, same shape as ``cell``.
    pbc : jax.Array or None, optional
        PBC flags, shape (3,), dtype=bool.

    Returns
    -------
    rebuild_needed : jax.Array, shape (1,), dtype=bool
        True if any atom has moved beyond skin distance.

    Notes
    -----
    This function is not differentiable and should not be used in JAX
    transformations that require gradients.

    See Also
    --------
    nvalchemiops.neighbors.rebuild_detection.check_neighbor_list_rebuild : Core warp launcher
    check_neighbor_list_rebuild_needed : Convenience wrapper that returns Python bool
    """
    if reference_positions.shape != current_positions.shape:
        return jnp.array([True], dtype=jnp.bool_)

    total_atoms = reference_positions.shape[0]

    if total_atoms == 0:
        return jnp.array([False], dtype=jnp.bool_)

    rebuild_flag = jnp.array([False], dtype=jnp.bool_)

    use_pbc = cell is not None and cell_inv is not None and pbc is not None

    if use_pbc:
        if cell.dtype != reference_positions.dtype:
            cell = cell.astype(reference_positions.dtype)
        if cell_inv.dtype != reference_positions.dtype:
            cell_inv = cell_inv.astype(reference_positions.dtype)
        pbc = pbc.astype(jnp.bool_)

        if reference_positions.dtype == jnp.float64:
            _jax_check = _jax_check_skin_pbc_f64
        else:
            _jax_check = _jax_check_skin_pbc_f32
            reference_positions = reference_positions.astype(jnp.float32)
            current_positions = current_positions.astype(jnp.float32)

        (rebuild_flag,) = _jax_check(
            reference_positions,
            current_positions,
            cell,
            cell_inv,
            pbc,
            float(skin_distance_threshold),
            rebuild_flag,
            launch_dims=(total_atoms,),
        )
    else:
        if reference_positions.dtype == jnp.float64:
            _jax_check = _jax_check_skin_f64
        else:
            _jax_check = _jax_check_skin_f32
            reference_positions = reference_positions.astype(jnp.float32)
            current_positions = current_positions.astype(jnp.float32)

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
    cell: jax.Array | None = None,
    cell_inv: jax.Array | None = None,
    pbc: jax.Array | None = None,
) -> bool:
    """Determine if neighbor list requires rebuilding based on atomic motion.

    When ``cell``, ``cell_inv`` and ``pbc`` are all provided, uses MIC
    displacement so periodic boundary crossings are handled correctly.

    Parameters
    ----------
    reference_positions : jax.Array, shape (total_atoms, 3)
        Atomic coordinates when the neighbor list was last constructed.
    current_positions : jax.Array, shape (total_atoms, 3)
        Current atomic coordinates to compare against reference positions.
    skin_distance_threshold : float
        Maximum allowed atomic displacement before neighbor list becomes invalid.
    cell : jax.Array or None, optional
        Unit cell matrix, shape (1, 3, 3).
    cell_inv : jax.Array or None, optional
        Inverse cell matrix, same shape as ``cell``.
    pbc : jax.Array or None, optional
        PBC flags, shape (3,), dtype=bool.

    Returns
    -------
    needs_rebuild : bool
        True if any atom has moved beyond skin distance requiring rebuild.

    See Also
    --------
    neighbor_list_needs_rebuild : Returns jax.Array instead of bool
    """
    rebuild_tensor = neighbor_list_needs_rebuild(
        reference_positions,
        current_positions,
        skin_distance_threshold,
        cell,
        cell_inv,
        pbc,
    )

    return bool(rebuild_tensor[0])


# ==============================================================================
# Batch Rebuild Detection
# ==============================================================================


def batch_neighbor_list_needs_rebuild(
    reference_positions: jax.Array,
    current_positions: jax.Array,
    batch_idx: jax.Array,
    skin_distance_threshold: float,
    num_systems: int,
    cell: jax.Array | None = None,
    cell_inv: jax.Array | None = None,
    pbc: jax.Array | None = None,
) -> jax.Array:
    """Detect per-system if neighbor lists require rebuilding due to atomic motion.

    When ``cell``, ``cell_inv`` and ``pbc`` are all provided, uses MIC
    displacement so periodic boundary crossings are handled correctly.

    Parameters
    ----------
    reference_positions : jax.Array, shape (total_atoms, 3)
        Atomic positions when each system's neighbor list was last built.
    current_positions : jax.Array, shape (total_atoms, 3)
        Current atomic positions to compare against reference.
    batch_idx : jax.Array, shape (total_atoms,), dtype=int32
        System index for each atom.
    skin_distance_threshold : float
        Maximum allowed displacement before neighbor list becomes invalid.
    num_systems : int
        Number of systems in the batch.
    cell : jax.Array or None, optional
        Per-system cell matrices, shape (num_systems, 3, 3).
    cell_inv : jax.Array or None, optional
        Inverse cell matrices, same shape as ``cell``.
    pbc : jax.Array or None, optional
        PBC flags, shape (num_systems, 3), dtype=bool.

    Returns
    -------
    rebuild_flags : jax.Array, shape (num_systems,), dtype=bool
        Per-system flags; True if any atom in that system moved beyond skin
        distance.

    Notes
    -----
    This function is not differentiable and should not be used in JAX
    transformations that require gradients.

    See Also
    --------
    neighbor_list_needs_rebuild : Single-system version
    check_batch_neighbor_list_rebuild_needed : Convenience wrapper
    """
    total_atoms = reference_positions.shape[0]

    if total_atoms == 0:
        return jnp.zeros(num_systems, dtype=jnp.bool_)

    rebuild_flags = jnp.zeros(num_systems, dtype=jnp.bool_)
    use_pbc = cell is not None and cell_inv is not None and pbc is not None

    if use_pbc:
        if cell.dtype != reference_positions.dtype:
            cell = cell.astype(reference_positions.dtype)
        if cell_inv.dtype != reference_positions.dtype:
            cell_inv = cell_inv.astype(reference_positions.dtype)
        pbc = pbc.astype(jnp.bool_)

        if reference_positions.dtype == jnp.float64:
            _jax_check = _jax_batch_check_skin_pbc_f64
        else:
            _jax_check = _jax_batch_check_skin_pbc_f32
            reference_positions = reference_positions.astype(jnp.float32)
            current_positions = current_positions.astype(jnp.float32)

        (rebuild_flags,) = _jax_check(
            reference_positions,
            current_positions,
            batch_idx,
            cell,
            cell_inv,
            pbc,
            float(skin_distance_threshold),
            rebuild_flags,
            launch_dims=(total_atoms,),
        )
    else:
        if reference_positions.dtype == jnp.float64:
            _jax_check = _jax_batch_check_skin_f64
        else:
            _jax_check = _jax_batch_check_skin_f32
            reference_positions = reference_positions.astype(jnp.float32)
            current_positions = current_positions.astype(jnp.float32)

        (rebuild_flags,) = _jax_check(
            reference_positions,
            current_positions,
            batch_idx,
            float(skin_distance_threshold),
            rebuild_flags,
            launch_dims=(total_atoms,),
        )

    return rebuild_flags


def batch_cell_list_needs_rebuild(
    current_positions: jax.Array,
    atom_to_cell_mapping: jax.Array,
    batch_idx: jax.Array,
    cells_per_dimension: jax.Array,
    cell: jax.Array,
    pbc: jax.Array,
) -> jax.Array:
    """Detect per-system if cell lists require rebuilding due to atomic motion.

    Parameters
    ----------
    current_positions : jax.Array, shape (total_atoms, 3)
        Current atomic coordinates in Cartesian space.
    atom_to_cell_mapping : jax.Array, shape (total_atoms, 3), dtype=int32
        3D cell coordinates for each atom from the existing cell lists.
    batch_idx : jax.Array, shape (total_atoms,), dtype=int32
        System index for each atom.
    cells_per_dimension : jax.Array, shape (num_systems, 3), dtype=int32
        Number of spatial cells in x, y, z directions per system.
    cell : jax.Array, shape (num_systems, 3, 3)
        Per-system unit cell matrices for coordinate transformations.
    pbc : jax.Array, shape (num_systems, 3), dtype=bool
        Per-system periodic boundary condition flags.

    Returns
    -------
    rebuild_flags : jax.Array, shape (num_systems,), dtype=bool
        Per-system flags; True if any atom in that system changed cells.

    Notes
    -----
    This function is not differentiable and should not be used in JAX transformations
    that require gradients.

    See Also
    --------
    cell_list_needs_rebuild : Single-system version
    check_batch_cell_list_rebuild_needed : Convenience wrapper returning list[bool]
    """
    total_atoms = current_positions.shape[0]
    num_systems = cell.shape[0]

    if total_atoms == 0:
        return jnp.zeros(num_systems, dtype=jnp.bool_)

    if cell.dtype != current_positions.dtype:
        cell = cell.astype(current_positions.dtype)

    pbc = pbc.astype(jnp.bool_)

    rebuild_flags = jnp.zeros(num_systems, dtype=jnp.bool_)

    if current_positions.dtype == jnp.float64:
        _jax_check = _jax_batch_check_cells_f64
    else:
        _jax_check = _jax_batch_check_cells_f32
        current_positions = current_positions.astype(jnp.float32)

    (rebuild_flags,) = _jax_check(
        current_positions,
        cell,
        atom_to_cell_mapping,
        batch_idx,
        cells_per_dimension,
        pbc,
        rebuild_flags,
        launch_dims=(total_atoms,),
    )

    return rebuild_flags


# ==============================================================================
# High-level Batch API Functions
# ==============================================================================


def check_batch_neighbor_list_rebuild_needed(
    reference_positions: jax.Array,
    current_positions: jax.Array,
    batch_idx: jax.Array,
    skin_distance_threshold: float,
    num_systems: int,
    cell: jax.Array | None = None,
    cell_inv: jax.Array | None = None,
    pbc: jax.Array | None = None,
) -> list[bool]:
    """Determine per-system if neighbor lists require rebuilding.

    When ``cell``, ``cell_inv`` and ``pbc`` are all provided, uses MIC
    displacement so periodic boundary crossings are handled correctly.

    Parameters
    ----------
    reference_positions : jax.Array, shape (total_atoms, 3)
        Atomic positions when each system's neighbor list was last built.
    current_positions : jax.Array, shape (total_atoms, 3)
        Current atomic positions to compare against reference.
    batch_idx : jax.Array, shape (total_atoms,), dtype=int32
        System index for each atom.
    skin_distance_threshold : float
        Maximum allowed displacement before neighbor list becomes invalid.
    num_systems : int
        Number of systems in the batch.
    cell : jax.Array or None, optional
        Per-system cell matrices, shape (num_systems, 3, 3).
    cell_inv : jax.Array or None, optional
        Inverse cell matrices, same shape as ``cell``.
    pbc : jax.Array or None, optional
        PBC flags, shape (num_systems, 3), dtype=bool.

    Returns
    -------
    needs_rebuild : list[bool]
        Per-system flags; True if neighbor list for that system needs rebuilding.

    See Also
    --------
    batch_neighbor_list_needs_rebuild : Returns jax.Array instead of list[bool]
    """
    rebuild_flags = batch_neighbor_list_needs_rebuild(
        reference_positions,
        current_positions,
        batch_idx,
        skin_distance_threshold,
        num_systems,
        cell,
        cell_inv,
        pbc,
    )

    return [bool(flag) for flag in rebuild_flags]


def check_batch_cell_list_rebuild_needed(
    current_positions: jax.Array,
    atom_to_cell_mapping: jax.Array,
    batch_idx: jax.Array,
    cells_per_dimension: jax.Array,
    cell: jax.Array,
    pbc: jax.Array,
) -> list[bool]:
    """Determine per-system if cell lists require rebuilding based on atomic motion.

    Parameters
    ----------
    current_positions : jax.Array, shape (total_atoms, 3)
        Current atomic coordinates in Cartesian space.
    atom_to_cell_mapping : jax.Array, shape (total_atoms, 3), dtype=int32
        3D cell coordinates for each atom from the existing cell lists.
    batch_idx : jax.Array, shape (total_atoms,), dtype=int32
        System index for each atom.
    cells_per_dimension : jax.Array, shape (num_systems, 3), dtype=int32
        Number of spatial cells in x, y, z directions per system.
    cell : jax.Array, shape (num_systems, 3, 3)
        Per-system unit cell matrices for coordinate transformations.
    pbc : jax.Array, shape (num_systems, 3), dtype=bool
        Per-system periodic boundary condition flags.

    Returns
    -------
    needs_rebuild : list[bool]
        Per-system flags; True if cell list for that system needs rebuilding.

    Notes
    -----
    This function is not differentiable and should not be used in JAX transformations
    that require gradients.

    See Also
    --------
    batch_cell_list_needs_rebuild : Returns jax.Array instead of list[bool]
    """
    rebuild_flags = batch_cell_list_needs_rebuild(
        current_positions,
        atom_to_cell_mapping,
        batch_idx,
        cells_per_dimension,
        cell,
        pbc,
    )

    return [bool(flag) for flag in rebuild_flags]
