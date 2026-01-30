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

"""JAX bindings for batched (multi-system) neighbor list construction.

This module provides JAX functions for batched naive and cell list methods for
building neighbor lists for multiple systems in parallel.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import warp as wp

from nvalchemiops.jax.neighbors.neighbor_utils import (
    allocate_cell_list,
    compute_naive_num_shifts,
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
from nvalchemiops.neighbors.batch_naive import (
    batch_naive_neighbor_matrix,
    batch_naive_neighbor_matrix_pbc,
)
from nvalchemiops.neighbors.neighbor_utils import (
    _expand_naive_shifts,
    estimate_max_neighbors,
)

__all__ = [
    "batch_naive_neighbor_list",
    "batch_cell_list",
    "batch_build_cell_list",
    "batch_query_cell_list",
    "estimate_batch_cell_list_sizes",
]


# ==============================================================================
# Batch Naive Neighbor List JAX Bindings
# ==============================================================================


def batch_naive_neighbor_list(
    positions: jax.Array,
    cutoff: float,
    batch_idx: jax.Array | None = None,
    batch_ptr: jax.Array | None = None,
    pbc: jax.Array | None = None,
    cell: jax.Array | None = None,
    max_neighbors: int | None = None,
    half_fill: bool = False,
    fill_value: int | None = None,
    return_neighbor_list: bool = False,
    neighbor_matrix: jax.Array | None = None,
    neighbor_matrix_shifts: jax.Array | None = None,
    num_neighbors: jax.Array | None = None,
    shift_range_per_dimension: jax.Array | None = None,
    shift_offset: jax.Array | None = None,
    total_shifts: int | None = None,
    max_atoms_per_system: int | None = None,
) -> (
    tuple[jax.Array, jax.Array, jax.Array, jax.Array]
    | tuple[jax.Array, jax.Array, jax.Array]
    | tuple[jax.Array, jax.Array]
):
    """Compute neighbor list for batch of systems using naive O(N^2) algorithm.

    Identifies all atom pairs within a specified cutoff distance for each system
    independently using a brute-force pairwise distance calculation. Supports both
    non-periodic and periodic boundary conditions.

    Parameters
    ----------
    positions : jax.Array, shape (total_atoms, 3), dtype=float32 or float64
        Concatenated atomic coordinates for all systems in Cartesian space.
    cutoff : float
        Cutoff distance for neighbor detection in Cartesian units.
        Must be positive. Atoms within this distance are considered neighbors.
    batch_idx : jax.Array, shape (total_atoms,), dtype=int32, optional
        System index for each atom. If None, batch_ptr must be provided.
    batch_ptr : jax.Array, shape (num_systems + 1,), dtype=int32, optional
        Cumulative atom counts defining system boundaries. If None, batch_idx must be provided.
    pbc : jax.Array, shape (num_systems, 3), dtype=bool, optional
        Periodic boundary condition flags for each system and dimension.
        True enables periodicity in that direction. Default is None (no PBC).
    cell : jax.Array, shape (num_systems, 3, 3), dtype=float32 or float64, optional
        Cell matrices defining lattice vectors. Required if pbc is provided.
    max_neighbors : int, optional
        Maximum number of neighbors per atom.
    half_fill : bool, optional
        If True, only store relationships where i < j. Default is False.
    fill_value : int, optional
        Value to fill the neighbor matrix with. Default is total_atoms.
    neighbor_matrix : jax.Array, optional
        Pre-allocated neighbor matrix.
    neighbor_matrix_shifts : jax.Array, optional
        Pre-allocated shift matrix for PBC.
    num_neighbors : jax.Array, optional
        Pre-allocated neighbors count array.
    shift_range_per_dimension : jax.Array, optional
        Pre-computed shift range for PBC systems.
    shift_offset : jax.Array, optional
        Pre-computed shift offsets for PBC systems.
    total_shifts : int, optional
        Total number of shifts for PBC.
    max_atoms_per_system : int, optional
        Maximum atoms in any system.

    Returns
    -------
    results : tuple of jax.Array
        Variable-length tuple depending on input parameters.

    Examples
    --------
    Basic usage with batch_ptr:

    >>> import jax.numpy as jnp
    >>> from nvalchemiops.jax.neighbors import batch_naive_neighbor_list
    >>> positions = jnp.zeros((200, 3), dtype=jnp.float32)
    >>> batch_ptr = jnp.array([0, 100, 200], dtype=jnp.int32)  # 2 systems
    >>> cutoff = 2.5
    >>> max_neighbors = 50
    >>> neighbor_matrix, num_neighbors = batch_naive_neighbor_list(
    ...     positions, cutoff, batch_ptr=batch_ptr, max_neighbors=max_neighbors
    ... )

    With PBC:

    >>> cell = jnp.eye(3, dtype=jnp.float32)[jnp.newaxis, :, :] * 10.0
    >>> cell = jnp.repeat(cell, 2, axis=0)
    >>> pbc = jnp.ones((2, 3), dtype=jnp.bool_)
    >>> neighbor_matrix, num_neighbors, shifts = batch_naive_neighbor_list(
    ...     positions, cutoff, batch_ptr=batch_ptr, max_neighbors=max_neighbors,
    ...     pbc=pbc, cell=cell
    ... )

    See Also
    --------
    nvalchemiops.neighbors.batch_naive.batch_naive_neighbor_matrix : Core warp launcher
    nvalchemiops.jax.neighbors.unbatched.naive_neighbor_list : Non-batched version
    batch_cell_list : Cell list method for large systems
    """
    if pbc is None and cell is not None:
        raise ValueError("If cell is provided, pbc must also be provided")
    if pbc is not None and cell is None:
        raise ValueError("If pbc is provided, cell must also be provided")

    jax_device = positions.devices().pop()

    # Prepare batch indices and pointers
    batch_idx, batch_ptr = prepare_batch_idx_ptr(
        batch_idx, batch_ptr, positions.shape[0], jax_device
    )
    num_systems = batch_ptr.shape[0] - 1

    if cell is not None:
        cell = cell if cell.ndim == 3 else cell[jnp.newaxis, :, :]
    if pbc is not None:
        pbc = pbc if pbc.ndim == 2 else pbc[jnp.newaxis, :]

    if max_neighbors is None:
        max_neighbors = estimate_max_neighbors(cutoff)

    if fill_value is None:
        fill_value = positions.shape[0]

    if neighbor_matrix is None:
        neighbor_matrix = jax.device_put(
            jnp.full(
                (positions.shape[0], max_neighbors),
                fill_value,
                dtype=jnp.int32,
            ),
            jax_device,
        )
    else:
        neighbor_matrix = neighbor_matrix.at[:].set(fill_value)

    if num_neighbors is None:
        num_neighbors = jax.device_put(
            jnp.zeros(positions.shape[0], dtype=jnp.int32),
            jax_device,
        )
    else:
        num_neighbors = num_neighbors.at[:].set(0)

    if pbc is not None:
        if neighbor_matrix_shifts is None:
            neighbor_matrix_shifts = jax.device_put(
                jnp.zeros(
                    (positions.shape[0], max_neighbors, 3),
                    dtype=jnp.int32,
                ),
                jax_device,
            )
        else:
            neighbor_matrix_shifts = neighbor_matrix_shifts.at[:].set(0)
        if (
            total_shifts is None
            or shift_offset is None
            or shift_range_per_dimension is None
        ):
            shift_range_per_dimension, shift_offset, total_shifts = (
                compute_naive_num_shifts(cell, cutoff, pbc)
            )

    if cutoff <= 0:
        if return_neighbor_list:
            if pbc is not None:
                return (
                    jnp.zeros((2, 0), dtype=jnp.int32),
                    jnp.zeros((positions.shape[0] + 1,), dtype=jnp.int32),
                    jnp.zeros((0, 3), dtype=jnp.int32),
                )
            else:
                return (
                    jnp.zeros((2, 0), dtype=jnp.int32),
                    jnp.zeros((positions.shape[0] + 1,), dtype=jnp.int32),
                )
        else:
            if pbc is not None:
                return neighbor_matrix, num_neighbors, neighbor_matrix_shifts
            else:
                return neighbor_matrix, num_neighbors

    # Get device and dtype info
    device_str = get_warp_device_from_array(positions)

    wp_dtype = get_wp_dtype(positions.dtype)
    wp_vec_dtype = get_wp_vec_dtype(positions.dtype)
    wp_mat_dtype = get_wp_mat_dtype(cell.dtype) if cell is not None else None

    # Convert JAX arrays to Warp via dlpack (zero-copy)
    positions_wp = wp.from_dlpack(positions, dtype=wp_vec_dtype)
    batch_idx_wp = wp.from_dlpack(batch_idx, dtype=wp.int32)
    batch_ptr_wp = wp.from_dlpack(batch_ptr, dtype=wp.int32)
    neighbor_matrix_wp = wp.from_dlpack(neighbor_matrix, dtype=wp.int32)
    num_neighbors_wp = wp.from_dlpack(num_neighbors, dtype=wp.int32)

    if pbc is None:
        # No PBC case
        batch_naive_neighbor_matrix(
            positions=positions_wp,
            cutoff=cutoff,
            batch_idx=batch_idx_wp,
            batch_ptr=batch_ptr_wp,
            neighbor_matrix=neighbor_matrix_wp,
            num_neighbors=num_neighbors_wp,
            wp_dtype=wp_dtype,
            device=device_str,
            half_fill=half_fill,
        )
    else:
        # PBC case - expand shifts and call kernel
        shifts = jax.device_put(
            jnp.empty((total_shifts, 3), dtype=jnp.int32),
            jax_device,
        )
        shift_system_idx = jax.device_put(
            jnp.empty((total_shifts,), dtype=jnp.int32),
            jax_device,
        )
        shifts_wp = wp.from_dlpack(shifts, dtype=wp.vec3i)
        shift_system_idx_wp = wp.from_dlpack(shift_system_idx, dtype=wp.int32)
        shift_range_per_dimension_wp = wp.from_dlpack(
            shift_range_per_dimension, dtype=wp.vec3i
        )
        shift_offset_wp = wp.from_dlpack(shift_offset, dtype=wp.int32)

        wp_device_obj = wp.device_from_jax(jax_device)
        wp.launch(
            kernel=_expand_naive_shifts,
            dim=num_systems,
            inputs=[
                shift_range_per_dimension_wp,
                shift_offset_wp,
                shifts_wp,
                shift_system_idx_wp,
            ],
            device=wp_device_obj,
        )

        cell_wp = wp.from_dlpack(cell, dtype=wp_mat_dtype)
        neighbor_matrix_shifts_wp = wp.from_dlpack(
            neighbor_matrix_shifts, dtype=wp.vec3i
        )

        if max_atoms_per_system is None:
            max_atoms_per_system = int(jnp.max(batch_ptr[1:] - batch_ptr[:-1]))

        batch_naive_neighbor_matrix_pbc(
            positions=positions_wp,
            cell=cell_wp,
            cutoff=cutoff,
            batch_ptr=batch_ptr_wp,
            shifts=shifts_wp,
            shift_system_idx=shift_system_idx_wp,
            neighbor_matrix=neighbor_matrix_wp,
            neighbor_matrix_shifts=neighbor_matrix_shifts_wp,
            num_neighbors=num_neighbors_wp,
            wp_dtype=wp_dtype,
            device=device_str,
            max_atoms_per_system=max_atoms_per_system,
            half_fill=half_fill,
        )

    # Synchronize device
    wp.synchronize_device(device_str)

    if return_neighbor_list:
        if pbc is not None:
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
            neighbor_list, neighbor_ptr = get_neighbor_list_from_neighbor_matrix(
                neighbor_matrix,
                num_neighbors=num_neighbors,
                fill_value=fill_value,
            )
            return neighbor_list, neighbor_ptr
    else:
        if pbc is not None:
            return neighbor_matrix, num_neighbors, neighbor_matrix_shifts
        else:
            return neighbor_matrix, num_neighbors


# ==============================================================================
# Batch Cell List Methods
# ==============================================================================


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
    positions : jax.Array, shape (total_atoms, 3)
        Atomic coordinates.
    batch_ptr : jax.Array, shape (num_systems + 1,), optional
        Cumulative atom counts.
    batch_idx : jax.Array, shape (total_atoms,), optional
        Batch indices for each atom.
    cell : jax.Array, shape (num_systems, 3, 3), optional
        Cell matrices for each system.
    cutoff : float, optional
        Cutoff distance. Default is 5.0.
    pbc : jax.Array, shape (num_systems, 3), optional
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
    positions : jax.Array, shape (total_atoms, 3)
        Atomic coordinates.
    batch_idx : jax.Array, shape (total_atoms,), optional
        Batch indices.
    batch_ptr : jax.Array, shape (num_systems + 1,), optional
        Cumulative atom counts.
    cell : jax.Array, shape (num_systems, 3, 3), optional
        Cell matrices.
    pbc : jax.Array, shape (num_systems, 3), optional
        PBC flags.
    cutoff : float, optional
        Cutoff distance. Default is 5.0.
    max_total_cells : int, optional
        Maximum cells. If None, will be estimated.

    Returns
    -------
    cells_per_dimension : jax.Array
        Cells per dimension.
    atom_periodic_shifts : jax.Array
        Periodic shifts.
    atom_to_cell_mapping : jax.Array
        Cell mappings.
    atoms_per_cell_count : jax.Array
        Atoms per cell.
    cell_atom_start_indices : jax.Array
        Start indices.
    cell_atom_list : jax.Array
        Cell atom list.
    neighbor_search_radius : jax.Array
        Search radius.
    cell_origin : jax.Array
        Cell origin.
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
    positions : jax.Array, shape (total_atoms, 3)
        Atomic coordinates.
    batch_idx : jax.Array, shape (total_atoms,), optional
        Batch indices.
    batch_ptr : jax.Array, shape (num_systems + 1,), optional
        Cumulative atom counts.
    cutoff : float, optional
        Cutoff distance.
    cell : jax.Array, shape (num_systems, 3, 3), optional
        Cell matrices.
    pbc : jax.Array, shape (num_systems, 3), optional
        PBC flags.
    cells_per_dimension : jax.Array, optional
        Cells per dimension.
    atom_periodic_shifts : jax.Array, optional
        Periodic shifts for each atom (output from batch_build_cell_list).
    atom_to_cell_mapping : jax.Array, optional
        Cell mappings.
    cell_atom_start_indices : jax.Array, optional
        Start indices.
    cell_atom_list : jax.Array, optional
        Cell atom list.
    neighbor_search_radius : jax.Array, optional
        Search radius.
    max_neighbors : int, optional
        Maximum neighbors per atom.
    neighbor_matrix : jax.Array, optional
        Pre-allocated neighbor matrix.
    num_neighbors : jax.Array, optional
        Pre-allocated neighbors count array.

    Returns
    -------
    neighbor_matrix : jax.Array, shape (total_atoms, max_neighbors)
        Neighbor matrix.
    num_neighbors : jax.Array, shape (total_atoms,)
        Neighbors count.
    neighbor_matrix_shifts : jax.Array, shape (total_atoms, max_neighbors, 3)
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
    wp_mat_dtype = get_wp_mat_dtype(cell.dtype) if cell is not None else None

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
    positions : jax.Array, shape (total_atoms, 3)
        Atomic coordinates.
    batch_idx : jax.Array, shape (total_atoms,), optional
        Batch indices.
    batch_ptr : jax.Array, shape (num_systems + 1,), optional
        Cumulative atom counts.
    cutoff : float, optional
        Cutoff distance. Default is 5.0.
    cell : jax.Array, shape (num_systems, 3, 3), optional
        Cell matrices.
    pbc : jax.Array, shape (num_systems, 3), optional
        PBC flags.
    max_neighbors : int, optional
        Maximum neighbors per atom.
    max_total_cells : int, optional
        Maximum cells to allocate.
    return_neighbor_list : bool, optional
        If True, return COO neighbor list format. Default is False.

    Returns
    -------
    neighbor_data : jax.Array
        Neighbor information.
    neighbor_ptr_or_count : jax.Array
        Neighbor pointer or count.
    cell_data : tuple, optional
        Cell list construction info.

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
