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

"""JAX bindings for unbatched cell list O(N) neighbor list construction."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import warp as wp

from nvalchemiops.jax.neighbors.neighbor_utils import (
    get_neighbor_list_from_neighbor_matrix,
)
from nvalchemiops.jax.types import (
    get_warp_device_from_array,
    get_wp_dtype,
    get_wp_mat_dtype,
    get_wp_vec_dtype,
    jax_to_warp,
)
from nvalchemiops.neighbors.cell_list import (
    build_cell_list as wp_build_cell_list,
)
from nvalchemiops.neighbors.cell_list import (
    query_cell_list as wp_query_cell_list,
)
from nvalchemiops.neighbors.neighbor_utils import estimate_max_neighbors

__all__ = [
    "cell_list",
    "build_cell_list",
    "query_cell_list",
    "estimate_cell_list_sizes",
]


def estimate_cell_list_sizes(
    positions: jax.Array,
    cell: jax.Array,
    cutoff: float,
    pbc: jax.Array | None = None,
    buffer_factor: float = 1.5,
) -> tuple[int, jax.Array, jax.Array]:
    """Estimate required cell list sizes based on atomic density.

    Parameters
    ----------
    positions : jax.Array, shape (total_atoms, 3), dtype=float32 or float64
        Atomic coordinates in Cartesian space.
    cell : jax.Array, shape (1, 3, 3), dtype=float32 or float64
        Cell matrix defining lattice vectors.
    cutoff : float
        Cutoff distance for neighbor searching.
    pbc : jax.Array, shape (1, 3), dtype=bool, optional
        Periodic boundary condition flags. Default is all True.
    buffer_factor : float, optional
        Buffer multiplier for cell count estimation. Default is 1.5.

    Returns
    -------
    max_total_cells : int
        Maximum total number of cells to allocate.
    cells_per_dimension : jax.Array, shape (3,) or (1, 3), dtype=int32
        Estimated number of cells in each dimension.
    neighbor_search_radius : jax.Array, shape (3,), dtype=int32
        Estimated search radius in neighboring cells.

    Notes
    -----
    This function estimates cell list parameters based on atomic positions and
    density. The actual number of cells used will be determined during cell
    list construction.
    """
    if cell.ndim == 2:
        cell = cell[jnp.newaxis, :, :]
    if pbc is None:
        pbc = jnp.ones((1, 3), dtype=jnp.bool_)
    if pbc.ndim == 1:
        pbc = pbc[jnp.newaxis, :]

    # Simple estimation: compute total volume and estimate cell volume
    # Cell volume = det(cell_matrix)
    det = jnp.linalg.det(cell[0])
    volume = jnp.abs(det)
    cell_volume = cutoff**3
    num_cells_est = int(volume / cell_volume * buffer_factor)
    max_total_cells = max(num_cells_est, 8)  # Minimum 8 cells

    # Estimate cells per dimension
    cells_per_dimension = jnp.ceil(jnp.ones(3) * (max_total_cells ** (1 / 3))).astype(
        jnp.int32
    )

    # Search radius estimate
    neighbor_search_radius = jnp.ones(3, dtype=jnp.int32) * 1

    return max_total_cells, cells_per_dimension, neighbor_search_radius


def build_cell_list(
    positions: jax.Array,
    cutoff: float,
    cell: jax.Array,
    pbc: jax.Array,
    cells_per_dimension: jax.Array | None = None,
    neighbor_search_radius: jax.Array | None = None,
    atom_periodic_shifts: jax.Array | None = None,
    atom_to_cell_mapping: jax.Array | None = None,
    atoms_per_cell_count: jax.Array | None = None,
    cell_atom_start_indices: jax.Array | None = None,
    cell_atom_list: jax.Array | None = None,
    max_total_cells: int | None = None,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Build spatial cell list for efficient neighbor searching.

    Parameters
    ----------
    positions : jax.Array, shape (total_atoms, 3), dtype=float32 or float64
        Atomic coordinates in Cartesian space.
    cutoff : float
        Cutoff distance for neighbor searching. Must be positive.
    cell : jax.Array, shape (1, 3, 3), dtype=float32 or float64
        Cell matrix defining lattice vectors.
    pbc : jax.Array, shape (1, 3), dtype=bool
        Periodic boundary condition flags.
    cells_per_dimension : jax.Array, shape (3,), dtype=int32, optional
        OUTPUT: Number of cells in x, y, z directions. If None, allocated.
    neighbor_search_radius : jax.Array, shape (3,), dtype=int32, optional
        Search radius in neighboring cells. If None, allocated.
    atom_periodic_shifts : jax.Array, shape (total_atoms, 3), dtype=int32, optional
        OUTPUT: Periodic boundary crossings for each atom. If None, allocated.
    atom_to_cell_mapping : jax.Array, shape (total_atoms, 3), dtype=int32, optional
        OUTPUT: 3D cell coordinates for each atom. If None, allocated.
    atoms_per_cell_count : jax.Array, shape (max_total_cells,), dtype=int32, optional
        OUTPUT: Number of atoms in each cell. If None, allocated.
    cell_atom_start_indices : jax.Array, shape (max_total_cells,), dtype=int32, optional
        OUTPUT: Starting index in cell_atom_list for each cell. If None, allocated.
    cell_atom_list : jax.Array, shape (total_atoms,), dtype=int32, optional
        OUTPUT: Flattened list of atom indices organized by cell. If None, allocated.
    max_total_cells : int, optional
        Maximum number of cells to allocate. If None, will be estimated.

    Returns
    -------
    cells_per_dimension : jax.Array, shape (3,), dtype=int32
        Number of cells in x, y, z directions.
    atom_periodic_shifts : jax.Array, shape (total_atoms, 3), dtype=int32
        Periodic boundary crossings for each atom.
    atom_to_cell_mapping : jax.Array, shape (total_atoms, 3), dtype=int32
        3D cell coordinates for each atom.
    atoms_per_cell_count : jax.Array, shape (max_total_cells,), dtype=int32
        Number of atoms in each cell.
    cell_atom_start_indices : jax.Array, shape (max_total_cells,), dtype=int32
        Starting index in cell_atom_list for each cell.
    cell_atom_list : jax.Array, shape (total_atoms,), dtype=int32
        Flattened list of atom indices organized by cell.
    neighbor_search_radius : jax.Array, shape (3,), dtype=int32
        Search radius in neighboring cells.

    See Also
    --------
    query_cell_list : Query the built cell list for neighbors
    """
    if cell.ndim == 2:
        cell = cell[jnp.newaxis, :, :]
    if pbc.ndim == 1:
        pbc = pbc[jnp.newaxis, :]
    # Ensure cell dtype matches positions dtype so warp overload dispatch is consistent
    if cell.dtype != positions.dtype:
        cell = cell.astype(positions.dtype)

    if max_total_cells is None:
        max_total_cells, _, neighbor_search_radius_est = estimate_cell_list_sizes(
            positions, cell, cutoff, pbc
        )
        if neighbor_search_radius is None:
            neighbor_search_radius = neighbor_search_radius_est
    else:
        if neighbor_search_radius is None:
            neighbor_search_radius = jnp.ones(3, dtype=jnp.int32)

    # Allocate cell list tensors if not provided
    if cells_per_dimension is None:
        cells_per_dimension = jnp.ones(3, dtype=jnp.int32)
    if atom_periodic_shifts is None:
        atom_periodic_shifts = jnp.zeros((positions.shape[0], 3), dtype=jnp.int32)
    if atom_to_cell_mapping is None:
        atom_to_cell_mapping = jnp.zeros((positions.shape[0], 3), dtype=jnp.int32)
    if atoms_per_cell_count is None:
        atoms_per_cell_count = jnp.zeros(max_total_cells, dtype=jnp.int32)
    if cell_atom_start_indices is None:
        cell_atom_start_indices = jnp.zeros(max_total_cells, dtype=jnp.int32)
    if cell_atom_list is None:
        cell_atom_list = jnp.zeros(positions.shape[0], dtype=jnp.int32)

    # Set device string
    device_str = get_warp_device_from_array(positions)

    wp_dtype = get_wp_dtype(positions.dtype)
    wp_vec_dtype = get_wp_vec_dtype(positions.dtype)
    wp_mat_dtype = get_wp_mat_dtype(cell.dtype)

    # Convert to warp arrays
    positions_wp = wp.from_dlpack(positions, dtype=wp_vec_dtype)
    cell_wp = wp.from_dlpack(cell, dtype=wp_mat_dtype)
    # Squeeze pbc to 1D if it's 2D with shape (1, 3)
    pbc_1d = pbc.squeeze() if pbc.ndim == 2 else pbc
    pbc_wp = jax_to_warp(pbc_1d, dtype=wp.bool)
    cells_per_dimension_wp = wp.from_dlpack(cells_per_dimension, dtype=wp.int32)
    atom_periodic_shifts_wp = wp.from_dlpack(atom_periodic_shifts, dtype=wp.vec3i)
    atom_to_cell_mapping_wp = wp.from_dlpack(atom_to_cell_mapping, dtype=wp.vec3i)
    atoms_per_cell_count_wp = wp.from_dlpack(atoms_per_cell_count, dtype=wp.int32)
    cell_atom_start_indices_wp = wp.from_dlpack(cell_atom_start_indices, dtype=wp.int32)
    cell_atom_list_wp = wp.from_dlpack(cell_atom_list, dtype=wp.int32)

    # Call warp kernel
    wp_build_cell_list(
        positions=positions_wp,
        cell=cell_wp,
        pbc=pbc_wp,
        cutoff=cutoff,
        cells_per_dimension=cells_per_dimension_wp,
        atom_periodic_shifts=atom_periodic_shifts_wp,
        atom_to_cell_mapping=atom_to_cell_mapping_wp,
        atoms_per_cell_count=atoms_per_cell_count_wp,
        cell_atom_start_indices=cell_atom_start_indices_wp,
        cell_atom_list=cell_atom_list_wp,
        wp_dtype=wp_dtype,
        device=device_str,
    )

    # Synchronize
    wp.synchronize_device(device_str)

    return (
        cells_per_dimension,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
        neighbor_search_radius,
    )


def query_cell_list(
    positions: jax.Array,
    cutoff: float,
    cell: jax.Array,
    pbc: jax.Array,
    cells_per_dimension: jax.Array,
    atom_periodic_shifts: jax.Array,
    atom_to_cell_mapping: jax.Array,
    atoms_per_cell_count: jax.Array,
    cell_atom_start_indices: jax.Array,
    cell_atom_list: jax.Array,
    neighbor_search_radius: jax.Array,
    max_neighbors: int | None = None,
    neighbor_matrix: jax.Array | None = None,
    neighbor_matrix_shifts: jax.Array | None = None,
    num_neighbors: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Query cell list to find neighbors within cutoff.

    Parameters
    ----------
    positions : jax.Array, shape (total_atoms, 3), dtype=float32 or float64
        Atomic coordinates in Cartesian space.
    cutoff : float
        Cutoff distance for neighbor detection.
    cell : jax.Array, shape (1, 3, 3), dtype=float32 or float64
        Cell matrix defining lattice vectors.
    pbc : jax.Array, shape (1, 3), dtype=bool
        Periodic boundary condition flags.
    cells_per_dimension : jax.Array, shape (3,), dtype=int32
        Number of cells in each dimension.
    atom_periodic_shifts : jax.Array, shape (total_atoms, 3), dtype=int32
        Periodic boundary crossings for each atom (output from ``build_cell_list``).
    atom_to_cell_mapping : jax.Array, shape (total_atoms, 3), dtype=int32
        3D cell coordinates for each atom.
    atoms_per_cell_count : jax.Array, shape (max_total_cells,), dtype=int32
        Number of atoms in each cell (output from ``build_cell_list``).
    cell_atom_start_indices : jax.Array, shape (max_total_cells,), dtype=int32
        Starting index in cell_atom_list for each cell.
    cell_atom_list : jax.Array, shape (total_atoms,), dtype=int32
        Flattened list of atom indices organized by cell.
    neighbor_search_radius : jax.Array, shape (3,), dtype=int32
        Search radius in neighboring cells.
    max_neighbors : int, optional
        Maximum number of neighbors per atom.
    neighbor_matrix : jax.Array, optional
        Pre-allocated neighbor matrix.
    num_neighbors : jax.Array, optional
        Pre-allocated neighbors count array.

    Returns
    -------
    neighbor_matrix : jax.Array, shape (total_atoms, max_neighbors), dtype=int32
        Neighbor matrix with neighbor atom indices.
    num_neighbors : jax.Array, shape (total_atoms,), dtype=int32
        Number of neighbors found for each atom.
    neighbor_matrix_shifts : jax.Array, shape (total_atoms, max_neighbors, 3), dtype=int32
        Periodic shift vectors for each neighbor relationship.

    See Also
    --------
    build_cell_list : Build cell list before querying
    cell_list : Combined build and query operation
    """
    if max_neighbors is None:
        max_neighbors = estimate_max_neighbors(cutoff)

    # Ensure cell dtype matches positions dtype so warp overload dispatch is consistent
    if cell.dtype != positions.dtype:
        cell = cell.astype(positions.dtype)

    if neighbor_matrix is None:
        neighbor_matrix = jnp.full(
            (positions.shape[0], max_neighbors),
            positions.shape[0],
            dtype=jnp.int32,
        )
    else:
        neighbor_matrix = neighbor_matrix.at[:].set(positions.shape[0])

    if num_neighbors is None:
        num_neighbors = jnp.zeros(positions.shape[0], dtype=jnp.int32)
    else:
        num_neighbors = num_neighbors.at[:].set(0)

    if neighbor_matrix_shifts is None:
        neighbor_matrix_shifts = jnp.zeros(
            (positions.shape[0], max_neighbors, 3),
            dtype=jnp.int32,
        )
    else:
        neighbor_matrix_shifts = neighbor_matrix_shifts.at[:].set(0)

    device_str = get_warp_device_from_array(positions)

    wp_dtype = get_wp_dtype(positions.dtype)
    wp_vec_dtype = get_wp_vec_dtype(positions.dtype)
    wp_mat_dtype = get_wp_mat_dtype(cell.dtype)

    # Convert to warp arrays
    positions_wp = wp.from_dlpack(positions, dtype=wp_vec_dtype)
    cell_wp = wp.from_dlpack(cell, dtype=wp_mat_dtype)
    # Squeeze pbc to 1D if it's 2D with shape (1, 3)
    pbc_1d = pbc.squeeze() if pbc.ndim == 2 else pbc
    pbc_wp = jax_to_warp(pbc_1d, dtype=wp.bool)
    cells_per_dimension_wp = wp.from_dlpack(cells_per_dimension, dtype=wp.int32)
    atom_periodic_shifts_wp = wp.from_dlpack(atom_periodic_shifts, dtype=wp.vec3i)
    atom_to_cell_mapping_wp = wp.from_dlpack(atom_to_cell_mapping, dtype=wp.vec3i)
    atoms_per_cell_count_wp = wp.from_dlpack(atoms_per_cell_count, dtype=wp.int32)
    cell_atom_start_indices_wp = wp.from_dlpack(cell_atom_start_indices, dtype=wp.int32)
    cell_atom_list_wp = wp.from_dlpack(cell_atom_list, dtype=wp.int32)
    neighbor_search_radius_wp = wp.from_dlpack(neighbor_search_radius, dtype=wp.int32)
    neighbor_matrix_wp = wp.from_dlpack(neighbor_matrix, dtype=wp.int32)
    neighbor_matrix_shifts_wp = wp.from_dlpack(neighbor_matrix_shifts, dtype=wp.vec3i)
    num_neighbors_wp = wp.from_dlpack(num_neighbors, dtype=wp.int32)

    # Call warp kernel
    wp_query_cell_list(
        positions=positions_wp,
        cell=cell_wp,
        pbc=pbc_wp,
        cutoff=cutoff,
        cells_per_dimension=cells_per_dimension_wp,
        neighbor_search_radius=neighbor_search_radius_wp,
        atom_periodic_shifts=atom_periodic_shifts_wp,
        atom_to_cell_mapping=atom_to_cell_mapping_wp,
        atoms_per_cell_count=atoms_per_cell_count_wp,
        cell_atom_start_indices=cell_atom_start_indices_wp,
        cell_atom_list=cell_atom_list_wp,
        neighbor_matrix=neighbor_matrix_wp,
        neighbor_matrix_shifts=neighbor_matrix_shifts_wp,
        num_neighbors=num_neighbors_wp,
        wp_dtype=wp_dtype,
        device=device_str,
    )

    # Synchronize
    wp.synchronize_device(device_str)

    return neighbor_matrix, num_neighbors, neighbor_matrix_shifts


def cell_list(
    positions: jax.Array,
    cutoff: float,
    cell: jax.Array | None = None,
    pbc: jax.Array | None = None,
    max_neighbors: int | None = None,
    max_total_cells: int | None = None,
    return_neighbor_list: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Build and query spatial cell list for efficient neighbor finding.

    This is a convenience function that combines build_cell_list and query_cell_list
    in a single call.

    Parameters
    ----------
    positions : jax.Array, shape (total_atoms, 3), dtype=float32 or float64
        Atomic coordinates in Cartesian space.
    cutoff : float
        Cutoff distance for neighbor detection.
    cell : jax.Array, shape (1, 3, 3), dtype=float32 or float64, optional
        Cell matrix defining lattice vectors. Default is identity matrix.
    pbc : jax.Array, shape (1, 3), dtype=bool, optional
        Periodic boundary condition flags. Default is all True.
    max_neighbors : int, optional
        Maximum number of neighbors per atom. If None, will be estimated.
    max_total_cells : int, optional
        Maximum number of cells to allocate. If None, will be estimated.
    return_neighbor_list : bool, optional
        If True, convert result to COO neighbor list format. Default is False.

    Returns
    -------
    neighbor_data : jax.Array
        If ``return_neighbor_list=False`` (default): ``neighbor_matrix`` with shape
        (total_atoms, max_neighbors), dtype int32.
        If ``return_neighbor_list=True``: ``neighbor_list`` with shape
        (2, num_pairs), dtype int32, in COO format.
    neighbor_count : jax.Array
        If ``return_neighbor_list=False``: ``num_neighbors`` with shape
        (total_atoms,), dtype int32.
        If ``return_neighbor_list=True``: ``neighbor_ptr`` with shape
        (total_atoms + 1,), dtype int32.
    shift_data : jax.Array
        If ``return_neighbor_list=False``: ``neighbor_matrix_shifts`` with shape
        (total_atoms, max_neighbors, 3), dtype int32.
        If ``return_neighbor_list=True``: ``neighbor_list_shifts`` with shape
        (num_pairs, 3), dtype int32.

    See Also
    --------
    build_cell_list : Build cell list separately
    query_cell_list : Query cell list separately
    naive_neighbor_list : Naive O(N^2) method
    """
    if cell is None:
        cell = jnp.eye(3, dtype=jnp.float32)[jnp.newaxis, :, :]
    if pbc is None:
        pbc = jnp.ones((1, 3), dtype=jnp.bool_)

    # Build cell list
    (
        cells_per_dimension,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
        neighbor_search_radius,
    ) = build_cell_list(
        positions,
        cutoff,
        cell,
        pbc,
        max_total_cells=max_total_cells,
    )

    # Query cell list
    neighbor_matrix, num_neighbors, neighbor_matrix_shifts = query_cell_list(
        positions=positions,
        cutoff=cutoff,
        cell=cell,
        pbc=pbc,
        cells_per_dimension=cells_per_dimension,
        atom_periodic_shifts=atom_periodic_shifts,
        atom_to_cell_mapping=atom_to_cell_mapping,
        atoms_per_cell_count=atoms_per_cell_count,
        cell_atom_start_indices=cell_atom_start_indices,
        cell_atom_list=cell_atom_list,
        neighbor_search_radius=neighbor_search_radius,
        max_neighbors=max_neighbors,
    )

    if return_neighbor_list:
        neighbor_list, neighbor_ptr, neighbor_list_shifts = (
            get_neighbor_list_from_neighbor_matrix(
                neighbor_matrix,
                num_neighbors=num_neighbors,
                neighbor_shift_matrix=neighbor_matrix_shifts,
                fill_value=positions.shape[0],
            )
        )
        return (
            neighbor_list,
            neighbor_ptr,
            neighbor_list_shifts,
        )
    else:
        return (
            neighbor_matrix,
            num_neighbors,
            neighbor_matrix_shifts,
        )
