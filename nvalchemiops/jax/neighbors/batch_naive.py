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

"""JAX bindings for batched naive O(N^2) neighbor list construction."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import warp as wp

from nvalchemiops.jax.neighbors.neighbor_utils import (
    compute_naive_num_shifts,
    get_neighbor_list_from_neighbor_matrix,
    prepare_batch_idx_ptr,
)
from nvalchemiops.jax.types import (
    get_warp_device_from_array,
    get_wp_dtype,
    get_wp_mat_dtype,
    get_wp_vec_dtype,
)
from nvalchemiops.neighbors.batch_naive import (
    batch_naive_neighbor_matrix,
    batch_naive_neighbor_matrix_pbc,
)
from nvalchemiops.neighbors.neighbor_utils import (
    _expand_naive_shifts,
    estimate_max_neighbors,
)

__all__ = ["batch_naive_neighbor_list"]


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
    nvalchemiops.jax.neighbors.naive.naive_neighbor_list : Non-batched version
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
        # Ensure cell dtype matches positions dtype so warp overload dispatch is consistent
        if cell.dtype != positions.dtype:
            cell = cell.astype(positions.dtype)
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
