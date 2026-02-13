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

"""JAX bindings for batched cell list O(N) neighbor list construction."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import warp as wp

from nvalchemiops.jax.neighbors.neighbor_utils import (
    allocate_cell_list,
    get_neighbor_list_from_neighbor_matrix,
    prepare_batch_idx_ptr,
)
from nvalchemiops.jax.types import (
    get_warp_device_from_array,
    get_wp_dtype,
    get_wp_mat_dtype,
    get_wp_vec_dtype,
    jax_to_warp,
)
from nvalchemiops.neighbors.batch_cell_list import (
    batch_build_cell_list as wp_batch_build_cell_list,
)
from nvalchemiops.neighbors.batch_cell_list import (
    batch_query_cell_list as wp_batch_query_cell_list,
)
from nvalchemiops.neighbors.neighbor_utils import estimate_max_neighbors

__all__ = [
    "batch_cell_list",
    "batch_build_cell_list",
    "batch_query_cell_list",
    "estimate_batch_cell_list_sizes",
]


def estimate_batch_cell_list_sizes(
    positions: jax.Array,
    batch_ptr: jax.Array | None = None,
    batch_idx: jax.Array | None = None,
    cell: jax.Array | None = None,
    cutoff: float = 5.0,
    pbc: jax.Array | None = None,
    buffer_factor: float = 1.5,
) -> tuple[int, jax.Array, jax.Array]:
    """Estimate required batch cell list sizes.

    Parameters
    ----------
    positions : jax.Array, shape (total_atoms, 3), dtype=float32 or float64
        Atomic coordinates.
    batch_ptr : jax.Array, shape (num_systems + 1,), dtype=int32, optional
        Cumulative atom counts.
    batch_idx : jax.Array, shape (total_atoms,), dtype=int32, optional
        Batch indices for each atom.
    cell : jax.Array, shape (num_systems, 3, 3), dtype=float32 or float64, optional
        Cell matrices for each system.
    cutoff : float, optional
        Cutoff distance. Default is 5.0.
    pbc : jax.Array, shape (num_systems, 3), dtype=bool, optional
        PBC flags.
    buffer_factor : float, optional
        Buffer multiplier. Default is 1.5.

    Returns
    -------
    max_total_cells : int
        Maximum total cells to allocate.
    cells_per_dimension : jax.Array, shape (num_systems, 3)
        Cells per dimension for each system.
    neighbor_search_radius : jax.Array, shape (num_systems, 3)
        Search radius for each system.
    """
    jax_device = positions.devices().pop()

    # Prepare batch info
    batch_idx, batch_ptr = prepare_batch_idx_ptr(
        batch_idx, batch_ptr, positions.shape[0], jax_device
    )
    num_systems = batch_ptr.shape[0] - 1

    # Simple estimation per system
    max_total_cells = 0
    cells_per_dim_list = []
    search_radius_list = []

    for sys_idx in range(num_systems):
        start_idx = batch_ptr[sys_idx]
        end_idx = batch_ptr[sys_idx + 1]
        num_atoms_in_sys = end_idx - start_idx

        if num_atoms_in_sys == 0:
            cells_per_dim_list.append(jnp.ones(3, dtype=jnp.int32))
            search_radius_list.append(jnp.ones(3, dtype=jnp.int32))
            continue

        # Volume estimation
        if cell is not None:
            det = jnp.linalg.det(cell[sys_idx])
            volume = jnp.abs(det)
        else:
            volume = 1000.0  # Default assumption

        cell_volume = cutoff**3
        num_cells_est = max(int(volume / cell_volume * buffer_factor), 8)
        max_total_cells += num_cells_est

        cells_per_dim = jnp.ceil(num_cells_est ** (1 / 3)).astype(jnp.int32)
        cells_per_dim_list.append(cells_per_dim * jnp.ones(3, dtype=jnp.int32))
        search_radius_list.append(jnp.ones(3, dtype=jnp.int32))

    cells_per_dimension = jnp.stack(cells_per_dim_list, axis=0)
    neighbor_search_radius = jnp.stack(search_radius_list, axis=0)

    return max_total_cells, cells_per_dimension, neighbor_search_radius


def batch_build_cell_list(
    positions: jax.Array,
    batch_idx: jax.Array | None = None,
    batch_ptr: jax.Array | None = None,
    cell: jax.Array | None = None,
    pbc: jax.Array | None = None,
    cutoff: float = 5.0,
    max_total_cells: int | None = None,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Build spatial cell lists for batch of systems.

    Parameters
    ----------
    positions : jax.Array, shape (total_atoms, 3), dtype=float32 or float64
        Atomic coordinates.
    batch_idx : jax.Array, shape (total_atoms,), dtype=int32, optional
        Batch indices.
    batch_ptr : jax.Array, shape (num_systems + 1,), dtype=int32, optional
        Cumulative atom counts.
    cell : jax.Array, shape (num_systems, 3, 3), dtype=float32 or float64, optional
        Cell matrices.
    pbc : jax.Array, shape (num_systems, 3), dtype=bool, optional
        PBC flags.
    cutoff : float, optional
        Cutoff distance. Default is 5.0.
    max_total_cells : int, optional
        Maximum cells. If None, will be estimated.

    Returns
    -------
    cells_per_dimension : jax.Array, shape (num_systems, 3), dtype=int32
        Number of cells in x, y, z directions for each system.
    atom_periodic_shifts : jax.Array, shape (total_atoms, 3), dtype=int32
        Periodic boundary crossings for each atom.
    atom_to_cell_mapping : jax.Array, shape (total_atoms, 3), dtype=int32
        3D cell coordinates for each atom.
    atoms_per_cell_count : jax.Array, shape (max_total_cells,), dtype=int32
        Number of atoms in each cell.
    cell_atom_start_indices : jax.Array, shape (max_total_cells,), dtype=int32
        Starting index in ``cell_atom_list`` for each cell.
    cell_atom_list : jax.Array, shape (total_atoms,), dtype=int32
        Flattened list of atom indices organized by cell.
    neighbor_search_radius : jax.Array, shape (num_systems, 3), dtype=int32
        Search radius in neighboring cells for each system.
    cell_origin : jax.Array, shape (3,), dtype same as positions
        Cell origin point (currently zeros).
    """
    jax_device = positions.devices().pop()

    # Prepare batch info
    batch_idx, batch_ptr = prepare_batch_idx_ptr(
        batch_idx, batch_ptr, positions.shape[0], jax_device
    )
    num_systems = batch_ptr.shape[0] - 1

    if max_total_cells is None:
        max_total_cells, cells_per_dim_est, neighbor_search_radius = (
            estimate_batch_cell_list_sizes(
                positions, batch_ptr, batch_idx, cell, cutoff, pbc
            )
        )
        # Ensure neighbor_search_radius is on the correct device
        neighbor_search_radius = jax.device_put(neighbor_search_radius, jax_device)
    else:
        neighbor_search_radius = jax.device_put(
            jnp.ones((num_systems, 3), dtype=jnp.int32),
            jax_device,
        )

    # Allocate cell list tensors
    (
        cells_per_dimension,
        neighbor_search_radius,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
    ) = allocate_cell_list(
        positions.shape[0],
        max_total_cells,
        neighbor_search_radius,
        jax_device,
    )

    # Set device string
    device_str = get_warp_device_from_array(positions)

    wp_dtype = get_wp_dtype(positions.dtype)
    wp_vec_dtype = get_wp_vec_dtype(positions.dtype)
    wp_mat_dtype = get_wp_mat_dtype(cell.dtype) if cell is not None else None

    # Convert to warp arrays
    positions_wp = wp.from_dlpack(positions, dtype=wp_vec_dtype)
    batch_idx_wp = wp.from_dlpack(batch_idx, dtype=wp.int32)
    cells_per_dimension_wp = wp.from_dlpack(cells_per_dimension, dtype=wp.vec3i)
    atom_periodic_shifts_wp = wp.from_dlpack(atom_periodic_shifts, dtype=wp.vec3i)
    atom_to_cell_mapping_wp = wp.from_dlpack(atom_to_cell_mapping, dtype=wp.vec3i)
    atoms_per_cell_count_wp = wp.from_dlpack(atoms_per_cell_count, dtype=wp.int32)
    cell_atom_start_indices_wp = wp.from_dlpack(cell_atom_start_indices, dtype=wp.int32)
    cell_atom_list_wp = wp.from_dlpack(cell_atom_list, dtype=wp.int32)

    if cell is not None:
        # Ensure cell dtype matches positions dtype so warp overload dispatch is consistent
        if cell.dtype != positions.dtype:
            cell = cell.astype(positions.dtype)
            wp_mat_dtype = get_wp_mat_dtype(cell.dtype)
        cell_wp = wp.from_dlpack(cell, dtype=wp_mat_dtype)
    else:
        cell_wp = None

    if pbc is not None:
        pbc_wp = jax_to_warp(pbc, dtype=wp.bool)
    else:
        pbc_wp = None

    # Allocate cell_offsets array (shape num_systems, not num_systems+1)
    cell_offsets = jax.device_put(
        jnp.zeros(num_systems, dtype=jnp.int32),
        jax_device,
    )
    cell_offsets_wp = wp.from_dlpack(cell_offsets, dtype=wp.int32)

    # Zero atoms_per_cell_count before building
    atoms_per_cell_count = atoms_per_cell_count.at[:].set(0)
    atoms_per_cell_count_wp = wp.from_dlpack(atoms_per_cell_count, dtype=wp.int32)

    # Call warp kernel
    wp_batch_build_cell_list(
        positions=positions_wp,
        cell=cell_wp,
        pbc=pbc_wp,
        cutoff=cutoff,
        batch_idx=batch_idx_wp,
        cells_per_dimension=cells_per_dimension_wp,
        cell_offsets=cell_offsets_wp,
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

    # Create cell origin
    cell_origin = jax.device_put(
        jnp.zeros(3, dtype=positions.dtype),
        jax_device,
    )

    return (
        cells_per_dimension,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
        neighbor_search_radius,
        cell_origin,
    )


def batch_query_cell_list(
    positions: jax.Array,
    batch_idx: jax.Array | None = None,
    batch_ptr: jax.Array | None = None,
    cutoff: float = 5.0,
    cell: jax.Array | None = None,
    pbc: jax.Array | None = None,
    cells_per_dimension: jax.Array | None = None,
    atom_periodic_shifts: jax.Array | None = None,
    atom_to_cell_mapping: jax.Array | None = None,
    cell_atom_start_indices: jax.Array | None = None,
    cell_atom_list: jax.Array | None = None,
    neighbor_search_radius: jax.Array | None = None,
    max_neighbors: int | None = None,
    neighbor_matrix: jax.Array | None = None,
    num_neighbors: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Query batch cell lists to find neighbors.

    Parameters
    ----------
    positions : jax.Array, shape (total_atoms, 3), dtype=float32 or float64
        Atomic coordinates.
    batch_idx : jax.Array, shape (total_atoms,), dtype=int32, optional
        Batch indices.
    batch_ptr : jax.Array, shape (num_systems + 1,), dtype=int32, optional
        Cumulative atom counts.
    cutoff : float, optional
        Cutoff distance.
    cell : jax.Array, shape (num_systems, 3, 3), dtype=float32 or float64, optional
        Cell matrices.
    pbc : jax.Array, shape (num_systems, 3), dtype=bool, optional
        PBC flags.
    cells_per_dimension : jax.Array, shape (num_systems, 3), dtype=int32, optional
        Cells per dimension.
    atom_periodic_shifts : jax.Array, shape (total_atoms, 3), dtype=int32, optional
        Periodic shifts for each atom (output from ``batch_build_cell_list``).
    atom_to_cell_mapping : jax.Array, shape (total_atoms, 3), dtype=int32, optional
        Cell mappings.
    cell_atom_start_indices : jax.Array, shape (max_total_cells,), dtype=int32, optional
        Start indices.
    cell_atom_list : jax.Array, shape (total_atoms,), dtype=int32, optional
        Cell atom list.
    neighbor_search_radius : jax.Array, shape (num_systems, 3), dtype=int32, optional
        Search radius.
    max_neighbors : int, optional
        Maximum neighbors per atom.
    neighbor_matrix : jax.Array, shape (total_atoms, max_neighbors), dtype=int32, optional
        Pre-allocated neighbor matrix.
    num_neighbors : jax.Array, shape (total_atoms,), dtype=int32, optional
        Pre-allocated neighbors count array.

    Returns
    -------
    neighbor_matrix : jax.Array, shape (total_atoms, max_neighbors), dtype=int32
        Neighbor matrix.
    num_neighbors : jax.Array, shape (total_atoms,), dtype=int32
        Neighbors count.
    neighbor_matrix_shifts : jax.Array, shape (total_atoms, max_neighbors, 3), dtype=int32
        Periodic shifts for each neighbor relationship.
    """
    if max_neighbors is None:
        max_neighbors = estimate_max_neighbors(cutoff)

    jax_device = positions.devices().pop()

    # Prepare batch info
    batch_idx, batch_ptr = prepare_batch_idx_ptr(
        batch_idx, batch_ptr, positions.shape[0], jax_device
    )

    if neighbor_matrix is None:
        neighbor_matrix = jax.device_put(
            jnp.full(
                (positions.shape[0], max_neighbors),
                positions.shape[0],
                dtype=jnp.int32,
            ),
            jax_device,
        )
    else:
        neighbor_matrix = neighbor_matrix.at[:].set(positions.shape[0])

    if num_neighbors is None:
        num_neighbors = jax.device_put(
            jnp.zeros(positions.shape[0], dtype=jnp.int32),
            jax_device,
        )
    else:
        num_neighbors = num_neighbors.at[:].set(0)

    device_str = get_warp_device_from_array(positions)

    wp_dtype = get_wp_dtype(positions.dtype)
    wp_vec_dtype = get_wp_vec_dtype(positions.dtype)

    # Convert to warp arrays
    positions_wp = wp.from_dlpack(positions, dtype=wp_vec_dtype)
    batch_idx_wp = wp.from_dlpack(batch_idx, dtype=wp.int32)
    cells_per_dimension_wp = wp.from_dlpack(cells_per_dimension, dtype=wp.vec3i)
    atom_to_cell_mapping_wp = wp.from_dlpack(atom_to_cell_mapping, dtype=wp.vec3i)
    cell_atom_start_indices_wp = wp.from_dlpack(cell_atom_start_indices, dtype=wp.int32)
    cell_atom_list_wp = wp.from_dlpack(cell_atom_list, dtype=wp.int32)
    neighbor_search_radius_wp = wp.from_dlpack(neighbor_search_radius, dtype=wp.vec3i)
    neighbor_matrix_wp = wp.from_dlpack(neighbor_matrix, dtype=wp.int32)
    num_neighbors_wp = wp.from_dlpack(num_neighbors, dtype=wp.int32)
    atom_periodic_shifts_wp = wp.from_dlpack(atom_periodic_shifts, dtype=wp.vec3i)

    if cell is not None:
        # Ensure cell dtype matches positions dtype so warp overload dispatch is consistent
        if cell.dtype != positions.dtype:
            cell = cell.astype(positions.dtype)
        wp_mat_dtype = get_wp_mat_dtype(cell.dtype)
        cell_wp = wp.from_dlpack(cell, dtype=wp_mat_dtype)
    else:
        cell_wp = None

    if pbc is not None:
        pbc_wp = jax_to_warp(pbc, dtype=wp.bool)
    else:
        pbc_wp = None

    # Allocate neighbor_matrix_shifts
    neighbor_matrix_shifts = jax.device_put(
        jnp.zeros(
            (positions.shape[0], max_neighbors, 3),
            dtype=jnp.int32,
        ),
        jax_device,
    )
    neighbor_matrix_shifts_wp = wp.from_dlpack(neighbor_matrix_shifts, dtype=wp.vec3i)

    # Compute atoms_per_cell_count from cell_atom_start_indices and cell_atom_list
    # This needs to be reconstructed from the output of batch_build_cell_list
    max_total_cells = cell_atom_start_indices.shape[0]
    atoms_per_cell_count = jax.device_put(
        jnp.zeros(max_total_cells, dtype=jnp.int32),
        jax_device,
    )
    atoms_per_cell_count_wp = wp.from_dlpack(atoms_per_cell_count, dtype=wp.int32)

    # Allocate cell_offsets array (shape num_systems)
    # Compute cell_offsets from cells_per_dimension using cumsum
    cells_per_system = jnp.prod(cells_per_dimension, axis=1)  # (num_systems,)
    cell_offsets = jax.device_put(
        jnp.concatenate(
            [
                jnp.array([0], dtype=jnp.int32),
                jnp.cumsum(cells_per_system[:-1], dtype=jnp.int32),
            ]
        ),
        jax_device,
    )
    cell_offsets_wp = wp.from_dlpack(cell_offsets, dtype=wp.int32)

    # Call warp kernel
    wp_batch_query_cell_list(
        positions=positions_wp,
        cell=cell_wp,
        pbc=pbc_wp,
        cutoff=cutoff,
        batch_idx=batch_idx_wp,
        cells_per_dimension=cells_per_dimension_wp,
        neighbor_search_radius=neighbor_search_radius_wp,
        cell_offsets=cell_offsets_wp,
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


def batch_cell_list(
    positions: jax.Array,
    cutoff: float,
    cell: jax.Array | None = None,
    pbc: jax.Array | None = None,
    batch_idx: jax.Array | None = None,
    batch_ptr: jax.Array | None = None,
    max_neighbors: int | None = None,
    max_total_cells: int | None = None,
    return_neighbor_list: bool = False,
) -> tuple[jax.Array, jax.Array] | tuple[jax.Array, jax.Array, tuple]:
    """Build and query spatial cell lists for batch of systems.

    Parameters
    ----------
    positions : jax.Array, shape (total_atoms, 3), dtype=float32 or float64
        Atomic coordinates.
    cutoff : float
        Cutoff distance for neighbor detection.
    cell : jax.Array, shape (num_systems, 3, 3), dtype=float32 or float64, optional
        Cell matrices defining lattice vectors. Default is identity matrix.
    pbc : jax.Array, shape (num_systems, 3), dtype=bool, optional
        Periodic boundary condition flags. Default is all True.
    batch_idx : jax.Array, shape (total_atoms,), dtype=int32, optional
        Batch indices for each atom.
    batch_ptr : jax.Array, shape (num_systems + 1,), dtype=int32, optional
        Cumulative atom counts defining system boundaries.
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
        ``neighbor_matrix_shifts`` with shape (total_atoms, max_neighbors, 3), dtype int32.
        Periodic shift vectors for each neighbor relationship.

    See Also
    --------
    batch_build_cell_list : Build cell list separately
    batch_query_cell_list : Query cell list separately
    batch_naive_neighbor_list : Naive O(N^2) method
    """
    jax_device = positions.devices().pop()

    # Prepare batch info
    batch_idx, batch_ptr = prepare_batch_idx_ptr(
        batch_idx, batch_ptr, positions.shape[0], jax_device
    )

    # Build cell list
    (
        cells_per_dimension,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
        neighbor_search_radius,
        cell_origin,
    ) = batch_build_cell_list(
        positions,
        batch_idx=batch_idx,
        batch_ptr=batch_ptr,
        cell=cell,
        pbc=pbc,
        cutoff=cutoff,
        max_total_cells=max_total_cells,
    )

    # Query cell list
    neighbor_matrix, num_neighbors, neighbor_matrix_shifts = batch_query_cell_list(
        positions=positions,
        batch_idx=batch_idx,
        batch_ptr=batch_ptr,
        cutoff=cutoff,
        cell=cell,
        pbc=pbc,
        cells_per_dimension=cells_per_dimension,
        atom_periodic_shifts=atom_periodic_shifts,
        atom_to_cell_mapping=atom_to_cell_mapping,
        cell_atom_start_indices=cell_atom_start_indices,
        cell_atom_list=cell_atom_list,
        neighbor_search_radius=neighbor_search_radius,
        max_neighbors=max_neighbors,
    )

    if return_neighbor_list:
        neighbor_list, neighbor_ptr = get_neighbor_list_from_neighbor_matrix(
            neighbor_matrix,
            num_neighbors=num_neighbors,
            fill_value=positions.shape[0],
        )
        return neighbor_list, neighbor_ptr, neighbor_matrix_shifts
    else:
        return neighbor_matrix, num_neighbors, neighbor_matrix_shifts
