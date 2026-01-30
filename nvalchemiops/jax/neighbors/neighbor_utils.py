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

"""JAX utilities for neighbor list construction.

This module contains JAX-specific helper functions for neighbor list operations.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import warp as wp

from nvalchemiops.jax.types import get_wp_dtype, jax_to_warp
from nvalchemiops.neighbors.neighbor_utils import (
    NeighborOverflowError,
    estimate_max_neighbors,
)
from nvalchemiops.neighbors.neighbor_utils import (
    compute_naive_num_shifts as wp_compute_naive_num_shifts,
)

__all__ = [
    "compute_naive_num_shifts",
    "get_neighbor_list_from_neighbor_matrix",
    "prepare_batch_idx_ptr",
    "allocate_cell_list",
    "estimate_max_neighbors",
    "NeighborOverflowError",
]


def compute_naive_num_shifts(
    cell: jax.Array,
    cutoff: float,
    pbc: jax.Array,
) -> tuple[jax.Array, jax.Array, int]:
    """Compute periodic image shifts needed for neighbor searching.

    Parameters
    ----------
    cell : jax.Array, shape (num_systems, 3, 3)
        Cell matrices defining lattice vectors in Cartesian coordinates.
        Each 3x3 matrix represents one system's periodic cell.
    cutoff : float
        Cutoff distance for neighbor searching in Cartesian units.
        Must be positive and typically less than half the minimum cell dimension.
    pbc : jax.Array, shape (num_systems, 3), dtype=bool
        Periodic boundary condition flags for each dimension.
        True enables periodicity in that direction.

    Returns
    -------
    shift_range : jax.Array, shape (num_systems, 3), dtype=int32
        Maximum shift indices in each dimension for each system.
    shift_offset : jax.Array, shape (num_systems + 1,), dtype=int32
        Cumulative sum of number of shifts for each system.
    total_shifts : int
        Total number of periodic shifts needed across all systems.

    See Also
    --------
    nvalchemiops.neighbors.neighbor_utils.compute_naive_num_shifts : Core warp launcher
    """
    num_systems = cell.shape[0]
    jax_device = cell.devices().pop()

    # Allocate on JAX device
    num_shifts = jax.device_put(jnp.empty(num_systems, dtype=jnp.int32), jax_device)
    shift_range = jax.device_put(
        jnp.empty((num_systems, 3), dtype=jnp.int32), jax_device
    )

    wp_dtype = get_wp_dtype(cell.dtype)

    # Get device string from JAX device
    if jax_device.platform == "gpu":
        device_str = "cuda:0"
    else:
        device_str = "cpu"

    # Convert JAX arrays to Warp via dlpack (zero-copy)
    wp_cell = wp.from_dlpack(cell.astype(jnp.float32), dtype=wp.mat33f)
    # Note: DLPack doesn't support bool, so we use jax_to_warp
    wp_pbc = jax_to_warp(pbc.astype(jnp.bool_), dtype=wp.bool)
    wp_num_shifts = wp.from_dlpack(num_shifts, dtype=wp.int32)
    wp_shift_range = wp.from_dlpack(shift_range, dtype=wp.vec3i)

    wp_compute_naive_num_shifts(
        cell=wp_cell,
        cutoff=cutoff,
        pbc=wp_pbc,
        num_shifts=wp_num_shifts,
        shift_range=wp_shift_range,
        wp_dtype=wp_dtype,
        device=device_str,
    )

    # Synchronize after kernel launch
    wp.synchronize_device(device_str)

    # Compute cumulative sum for shift offsets
    shift_offset = jax.device_put(
        jnp.zeros(num_systems + 1, dtype=jnp.int32), jax_device
    )
    shift_offset = shift_offset.at[1:].set(jnp.cumsum(num_shifts))
    total_shifts_value = int(shift_offset[-1])

    return shift_range, shift_offset, total_shifts_value


def get_neighbor_list_from_neighbor_matrix(
    neighbor_matrix: jax.Array,
    num_neighbors: jax.Array,
    neighbor_shift_matrix: jax.Array | None = None,
    fill_value: int = -1,
) -> tuple[jax.Array, jax.Array] | tuple[jax.Array, jax.Array, jax.Array]:
    """Convert neighbor matrix format to neighbor list format.

    Parameters
    ----------
    neighbor_matrix : jax.Array, shape (total_atoms, max_neighbors), dtype=int32
        The neighbor matrix with neighbor atom indices.
    num_neighbors : jax.Array, shape (total_atoms,), dtype=int32
        The number of neighbors for each atom.
    neighbor_shift_matrix : jax.Array | None, shape (total_atoms, max_neighbors, 3), dtype=int32
        Optional neighbor shift matrix with periodic shift vectors.
    fill_value : int, default=-1
        The fill value used in the neighbor matrix to indicate empty slots.
        This is used to create a mask from the neighbor matrix.

    Returns
    -------
    neighbor_list : jax.Array, shape (2, num_pairs), dtype=int32
        The neighbor list in COO format [source_atoms, target_atoms].
    neighbor_ptr : jax.Array, shape (total_atoms + 1,), dtype=int32
        CSR-style pointer array where neighbor_ptr[i]:neighbor_ptr[i+1] gives the range of
        neighbors for atom i in the flattened neighbor list.
    neighbor_list_shifts : jax.Array, shape (num_pairs, 3), dtype=int32
        The neighbor shift vectors (only returned if neighbor_shift_matrix is not None).

    Raises
    ------
    ValueError
        If the max number of neighbors is larger than the neighbor matrix width.

    Notes
    -----
    This is a pure JAX utility function with no warp dependencies. It converts
    from the fixed-width matrix format to the variable-width list format by masking
    out fill values and flattening the result.

    See Also
    --------
    nvalchemiops.jax.neighbors.unbatched.naive_neighbor_list : Uses this for format conversion
    nvalchemiops.jax.neighbors.unbatched.cell_list : Uses this for format conversion
    """
    # Handle empty case
    if neighbor_matrix.shape[0] == 0:
        jax_device = neighbor_matrix.devices().pop()
        neighbor_list = jax.device_put(
            jnp.zeros((2, 0), dtype=neighbor_matrix.dtype),
            jax_device,
        )
        neighbor_ptr = jax.device_put(
            jnp.zeros(1, dtype=jnp.int32),
            jax_device,
        )
        if neighbor_shift_matrix is not None:
            neighbor_shift_list = jax.device_put(
                jnp.empty((0, 3), dtype=neighbor_shift_matrix.dtype),
                jax_device,
            )
            return neighbor_list, neighbor_ptr, neighbor_shift_list
        else:
            return neighbor_list, neighbor_ptr

    # Validate that the neighbor matrix is large enough
    max_found = jnp.max(num_neighbors)
    if max_found > neighbor_matrix.shape[1]:
        raise NeighborOverflowError(
            neighbor_matrix.shape[1],
            int(max_found),
        )

    # Create mask and extract neighbor pairs
    mask = neighbor_matrix != fill_value
    dtype = neighbor_matrix.dtype
    i_idx = jnp.where(mask)[0].astype(dtype)
    j_idx = neighbor_matrix[mask].astype(dtype)
    neighbor_list = jnp.stack([i_idx, j_idx], axis=0)

    # Create CSR-style pointer array
    neighbor_ptr = jnp.zeros(num_neighbors.shape[0] + 1, dtype=jnp.int32)
    neighbor_ptr = neighbor_ptr.at[1:].set(jnp.cumsum(num_neighbors))

    if neighbor_shift_matrix is not None:
        neighbor_list_shifts = neighbor_shift_matrix[mask]
        return neighbor_list, neighbor_ptr, neighbor_list_shifts
    else:
        return neighbor_list, neighbor_ptr


def prepare_batch_idx_ptr(
    batch_idx: jax.Array | None,
    batch_ptr: jax.Array | None,
    num_atoms: int,
    jax_device: jax.Device,
) -> tuple[jax.Array, jax.Array]:
    """Prepare batch index and pointer tensors from either representation.

    Utility function to ensure both batch_idx and batch_ptr are available,
    computing one from the other if needed.

    Parameters
    ----------
    batch_idx : jax.Array | None, shape (total_atoms,), dtype=int32
        Tensor indicating the batch index for each atom.
    batch_ptr : jax.Array | None, shape (num_systems + 1,), dtype=int32
        Tensor indicating the start index of each batch in the atom list.
    num_atoms : int
        Total number of atoms across all systems.
    jax_device : jax.Device
        Device on which to create tensors if needed.

    Returns
    -------
    batch_idx : jax.Array, shape (total_atoms,), dtype=int32
        Prepared batch index tensor.
    batch_ptr : jax.Array, shape (num_systems + 1,), dtype=int32
        Prepared batch pointer tensor.

    Raises
    ------
    ValueError
        If both batch_idx and batch_ptr are None.

    Notes
    -----
    This is a pure JAX utility function with no warp dependencies. It provides
    convenience for batch operations by converting between dense (batch_idx) and
    sparse (batch_ptr) batch representations.

    See Also
    --------
    nvalchemiops.jax.neighbors.batched.batch_naive_neighbor_list : Uses this for batch setup
    nvalchemiops.jax.neighbors.batched.batch_cell_list : Uses this for batch setup
    """
    if batch_idx is None and batch_ptr is None:
        raise ValueError("Either batch_idx or batch_ptr must be provided.")

    if batch_idx is None:
        num_systems = batch_ptr.shape[0] - 1
        num_atoms_per_system = batch_ptr[1:] - batch_ptr[:-1]
        batch_idx = jax.device_put(
            jnp.repeat(
                jnp.arange(num_systems, dtype=jnp.int32),
                num_atoms_per_system,
            ),
            jax_device,
        )

    elif batch_ptr is None:
        num_systems = int(jnp.max(batch_idx)) + 1
        # Use bincount to compute atoms per system
        num_atoms_per_system = jnp.bincount(
            batch_idx, minlength=num_systems, length=num_systems
        )
        batch_ptr = jnp.zeros(num_systems + 1, dtype=jnp.int32)
        batch_ptr = batch_ptr.at[1:].set(jnp.cumsum(num_atoms_per_system))

    return batch_idx, batch_ptr


def allocate_cell_list(
    total_atoms: int,
    max_total_cells: int,
    neighbor_search_radius: jax.Array,
    jax_device: jax.Device,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Allocate memory tensors for cell list data structures.

    Parameters
    ----------
    total_atoms : int
        Total number of atoms across all systems.
    max_total_cells : int
        Maximum number of cells to allocate.
    neighbor_search_radius : jax.Array, shape (3,) or (num_systems, 3), dtype=int32
        Radius of neighboring cells to search in each dimension.
    jax_device : jax.Device
        Device on which to create tensors.

    Returns
    -------
    cells_per_dimension : jax.Array, shape (3,) or (num_systems, 3), dtype=int32
        Number of cells in x, y, z directions (to be filled by build_cell_list).
    neighbor_search_radius : jax.Array, shape (3,) or (num_systems, 3), dtype=int32
        Radius of neighboring cells to search (passed through for convenience).
    atom_periodic_shifts : jax.Array, shape (total_atoms, 3), dtype=int32
        Periodic boundary crossings for each atom (to be filled by build_cell_list).
    atom_to_cell_mapping : jax.Array, shape (total_atoms, 3), dtype=int32
        3D cell coordinates for each atom (to be filled by build_cell_list).
    atoms_per_cell_count : jax.Array, shape (max_total_cells,), dtype=int32
        Number of atoms in each cell (to be filled by build_cell_list).
    cell_atom_start_indices : jax.Array, shape (max_total_cells,), dtype=int32
        Starting index in cell_atom_list for each cell (to be filled by build_cell_list).
    cell_atom_list : jax.Array, shape (total_atoms,), dtype=int32
        Flattened list of atom indices organized by cell (to be filled by build_cell_list).

    Notes
    -----
    This is a pure JAX utility function with no warp dependencies. It pre-allocates
    all tensors needed for cell list construction, supporting both single-system and
    batched operations based on the shape of neighbor_search_radius.

    See Also
    --------
    nvalchemiops.neighbors.cell_list.build_cell_list : Warp launcher that uses these tensors
    nvalchemiops.jax.neighbors.unbatched.build_cell_list : High-level JAX wrapper
    nvalchemiops.jax.neighbors.batched.batch_build_cell_list : Batched version
    """
    # Detect number of systems from neighbor_search_radius shape
    is_batched = neighbor_search_radius.ndim == 2
    num_systems = neighbor_search_radius.shape[0] if is_batched else 1

    cells_per_dimension = jax.device_put(
        jnp.zeros(
            (3,) if not is_batched else (num_systems, 3),
            dtype=jnp.int32,
        ),
        jax_device,
    )

    atom_periodic_shifts = jax.device_put(
        jnp.zeros((total_atoms, 3), dtype=jnp.int32),
        jax_device,
    )
    atom_to_cell_mapping = jax.device_put(
        jnp.zeros((total_atoms, 3), dtype=jnp.int32),
        jax_device,
    )
    atoms_per_cell_count = jax.device_put(
        jnp.zeros((max_total_cells,), dtype=jnp.int32),
        jax_device,
    )
    cell_atom_start_indices = jax.device_put(
        jnp.zeros((max_total_cells,), dtype=jnp.int32),
        jax_device,
    )
    cell_atom_list = jax.device_put(
        jnp.zeros((total_atoms,), dtype=jnp.int32),
        jax_device,
    )
    return (
        cells_per_dimension,
        neighbor_search_radius,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
    )
