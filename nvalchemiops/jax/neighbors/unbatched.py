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

"""JAX bindings for unbatched (single-system) neighbor list construction.

This module provides JAX functions for naive and cell list methods for building
neighbor lists for single systems.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import warp as wp

from nvalchemiops.jax.neighbors.neighbor_utils import (
    compute_naive_num_shifts,
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
from nvalchemiops.neighbors.naive import (
    naive_neighbor_matrix,
    naive_neighbor_matrix_pbc,
)
from nvalchemiops.neighbors.naive_dual_cutoff import (
    naive_neighbor_matrix_dual_cutoff,
    naive_neighbor_matrix_pbc_dual_cutoff,
)
from nvalchemiops.neighbors.neighbor_utils import (
    _expand_naive_shifts,
    estimate_max_neighbors,
)

__all__ = [
    "naive_neighbor_list",
    "naive_neighbor_list_dual_cutoff",
    "cell_list",
    "build_cell_list",
    "query_cell_list",
    "estimate_cell_list_sizes",
]


# ==============================================================================
# Naive Neighbor List JAX Bindings
# ==============================================================================


def naive_neighbor_list(
    positions: jax.Array,
    cutoff: float,
    cell: jax.Array | None = None,
    pbc: jax.Array | None = None,
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
) -> (
    tuple[jax.Array, jax.Array, jax.Array, jax.Array]
    | tuple[jax.Array, jax.Array, jax.Array]
    | tuple[jax.Array, jax.Array]
):
    """Compute neighbor list using naive O(N^2) algorithm.

    Identifies all atom pairs within a specified cutoff distance using a
    brute-force pairwise distance calculation. Supports both non-periodic
    and periodic boundary conditions.

    Parameters
    ----------
    positions : jax.Array, shape (total_atoms, 3), dtype=float32 or float64
        Atomic coordinates in Cartesian space. Each row represents one atom's
        (x, y, z) position.
    cutoff : float
        Cutoff distance for neighbor detection in Cartesian units.
        Must be positive. Atoms within this distance are considered neighbors.
    pbc : jax.Array, shape (1, 3), dtype=bool, optional
        Periodic boundary condition flags for each dimension.
        True enables periodicity in that direction. Default is None (no PBC).
    cell : jax.Array, shape (1, 3, 3), dtype=float32 or float64, optional
        Cell matrices defining lattice vectors in Cartesian coordinates.
        Required if pbc is provided. Default is None.
    max_neighbors : int, optional
        Maximum number of neighbors per atom. Must be positive.
        If exceeded, excess neighbors are ignored.
        Must be provided if neighbor_matrix is not provided.
    half_fill : bool, optional
        If True, only store relationships where i < j to avoid double counting.
        If False, store all neighbor relationships symmetrically. Default is False.
    fill_value : int, optional
        Value to fill the neighbor matrix with. Default is total_atoms.
    neighbor_matrix : jax.Array, shape (total_atoms, max_neighbors), dtype=int32, optional
        Neighbor matrix to be filled. Pass in a pre-allocated tensor to avoid reallocation.
        Must be provided if max_neighbors is not provided.
    neighbor_matrix_shifts : jax.Array, shape (total_atoms, max_neighbors, 3), dtype=int32, optional
        Shift vectors for each neighbor relationship. Pass in a pre-allocated tensor to avoid reallocation.
        Must be provided if max_neighbors is not provided.
    num_neighbors : jax.Array, shape (total_atoms,), dtype=int32, optional
        Number of neighbors found for each atom. Pass in a pre-allocated tensor to avoid reallocation.
        Must be provided if max_neighbors is not provided.
    shift_range_per_dimension : jax.Array, shape (1, 3), dtype=int32, optional
        Shift range in each dimension for each system.
        Pass in a pre-allocated tensor to avoid reallocation for pbc systems.
    shift_offset : jax.Array, shape (2,), dtype=int32, optional
        Cumulative sum of number of shifts for each system.
        Pass in a pre-allocated tensor to avoid reallocation for pbc systems.
    total_shifts : int, optional
        Total number of shifts.
        Pass in a pre-allocated tensor to avoid reallocation for pbc systems.
    return_neighbor_list : bool, optional - default = False
        If True, convert the neighbor matrix to a neighbor list (idx_i, idx_j) format by
        creating a mask over the fill_value, which can incur a performance penalty.

    Returns
    -------
    results : tuple of jax.Array
        Variable-length tuple depending on input parameters. The return pattern follows:

        - No PBC, matrix format: ``(neighbor_matrix, num_neighbors)``
        - No PBC, list format: ``(neighbor_list, neighbor_ptr)``
        - With PBC, matrix format: ``(neighbor_matrix, num_neighbors, neighbor_matrix_shifts)``
        - With PBC, list format: ``(neighbor_list, neighbor_ptr, neighbor_list_shifts)``

    Examples
    --------
    Basic usage without periodic boundary conditions:

    >>> import jax.numpy as jnp
    >>> from nvalchemiops.jax.neighbors import naive_neighbor_list
    >>> positions = jnp.zeros((100, 3), dtype=jnp.float32)
    >>> cutoff = 2.5
    >>> max_neighbors = 50
    >>> neighbor_matrix, num_neighbors = naive_neighbor_list(
    ...     positions, cutoff, max_neighbors=max_neighbors
    ... )

    With periodic boundary conditions:

    >>> cell = jnp.eye(3, dtype=jnp.float32).reshape(1, 3, 3) * 10.0
    >>> pbc = jnp.array([[True, True, True]])
    >>> neighbor_matrix, num_neighbors, shifts = naive_neighbor_list(
    ...     positions, cutoff, max_neighbors=max_neighbors, pbc=pbc, cell=cell
    ... )

    See Also
    --------
    nvalchemiops.neighbors.naive.naive_neighbor_matrix : Core warp launcher (no PBC)
    nvalchemiops.neighbors.naive.naive_neighbor_matrix_pbc : Core warp launcher (with PBC)
    cell_list : O(N) cell list method for larger systems
    """
    if pbc is None and cell is not None:
        raise ValueError("If cell is provided, pbc must also be provided")
    if pbc is not None and cell is None:
        raise ValueError("If pbc is provided, cell must also be provided")

    if cell is not None:
        cell = cell if cell.ndim == 3 else cell[jnp.newaxis, :, :]
        # Ensure cell dtype matches positions dtype so warp overload dispatch is consistent
        if cell.dtype != positions.dtype:
            cell = cell.astype(positions.dtype)
    if pbc is not None:
        pbc = pbc if pbc.ndim == 2 else pbc[jnp.newaxis, :]

    if max_neighbors is None and (
        neighbor_matrix is None
        or (neighbor_matrix_shifts is None and pbc is not None)
        or num_neighbors is None
    ):
        max_neighbors = estimate_max_neighbors(cutoff)

    if fill_value is None:
        fill_value = positions.shape[0]

    jax_device = positions.devices().pop()

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
                    jnp.zeros(
                        (positions.shape[0] + 1,),
                        dtype=jnp.int32,
                    ),
                    jnp.zeros((0, 3), dtype=jnp.int32),
                )
            else:
                return (
                    jnp.zeros((2, 0), dtype=jnp.int32),
                    jnp.zeros(
                        (positions.shape[0] + 1,),
                        dtype=jnp.int32,
                    ),
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
    neighbor_matrix_wp = wp.from_dlpack(neighbor_matrix, dtype=wp.int32)
    num_neighbors_wp = wp.from_dlpack(num_neighbors, dtype=wp.int32)

    if pbc is None:
        # No PBC case
        naive_neighbor_matrix(
            positions=positions_wp,
            cutoff=cutoff,
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
            dim=1,
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

        naive_neighbor_matrix_pbc(
            positions=positions_wp,
            cutoff=cutoff,
            cell=cell_wp,
            shifts=shifts_wp,
            neighbor_matrix=neighbor_matrix_wp,
            neighbor_matrix_shifts=neighbor_matrix_shifts_wp,
            num_neighbors=num_neighbors_wp,
            wp_dtype=wp_dtype,
            device=device_str,
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


def naive_neighbor_list_dual_cutoff(
    positions: jax.Array,
    cutoff1: float,
    cutoff2: float,
    pbc: jax.Array | None = None,
    cell: jax.Array | None = None,
    max_neighbors1: int | None = None,
    max_neighbors2: int | None = None,
    half_fill: bool = False,
    fill_value: int | None = None,
    return_neighbor_list: bool = False,
    neighbor_matrix1: jax.Array | None = None,
    neighbor_matrix2: jax.Array | None = None,
    neighbor_matrix_shifts1: jax.Array | None = None,
    neighbor_matrix_shifts2: jax.Array | None = None,
    num_neighbors1: jax.Array | None = None,
    num_neighbors2: jax.Array | None = None,
    shift_range_per_dimension: jax.Array | None = None,
    shift_offset: jax.Array | None = None,
    total_shifts: int | None = None,
) -> (
    tuple[
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
    ]
    | tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]
    | tuple[jax.Array, jax.Array, jax.Array, jax.Array]
):
    """Compute neighbor lists for two cutoff distances using naive O(N^2) algorithm.

    This function builds two neighbor matrices simultaneously for different cutoff
    distances, which is more efficient than calling the single-cutoff function twice.

    Parameters
    ----------
    positions : jax.Array, shape (total_atoms, 3), dtype=float32 or float64
        Atomic coordinates in Cartesian space.
    cutoff1 : float
        First cutoff distance (typically smaller).
    cutoff2 : float
        Second cutoff distance (typically larger).
    pbc : jax.Array, shape (1, 3) or (3,), dtype=bool, optional
        Periodic boundary condition flags for each dimension.
    cell : jax.Array, shape (1, 3, 3) or (3, 3), dtype=float32 or float64, optional
        Cell matrix defining lattice vectors in Cartesian coordinates.
    max_neighbors1 : int, optional
        Maximum number of neighbors per atom for cutoff1.
    max_neighbors2 : int, optional
        Maximum number of neighbors per atom for cutoff2.
    half_fill : bool, optional - default = False
        If True, only store relationships where i < j to avoid double counting.
    fill_value : int, optional
        Value to use for padding in neighbor matrices. Default is total_atoms.
    return_neighbor_list : bool, optional - default = False
        If True, convert neighbor matrices to neighbor list (idx_i, idx_j) format.
    neighbor_matrix1 : jax.Array, shape (total_atoms, max_neighbors1), dtype=int32, optional
        Pre-allocated first neighbor matrix.
    neighbor_matrix2 : jax.Array, shape (total_atoms, max_neighbors2), dtype=int32, optional
        Pre-allocated second neighbor matrix.
    neighbor_matrix_shifts1 : jax.Array, shape (total_atoms, max_neighbors1, 3), dtype=int32, optional
        Pre-allocated first shift matrix for PBC.
    neighbor_matrix_shifts2 : jax.Array, shape (total_atoms, max_neighbors2, 3), dtype=int32, optional
        Pre-allocated second shift matrix for PBC.
    num_neighbors1 : jax.Array, shape (total_atoms,), dtype=int32, optional
        Pre-allocated first neighbor count array.
    num_neighbors2 : jax.Array, shape (total_atoms,), dtype=int32, optional
        Pre-allocated second neighbor count array.
    shift_range_per_dimension : jax.Array, shape (3,), dtype=int32, optional
        Pre-computed shift ranges for PBC.
    shift_offset : jax.Array, shape (1,), dtype=int32, optional
        Pre-computed shift offset for PBC.
    total_shifts : int, optional
        Total number of shifts for PBC.

    Returns
    -------
    results : tuple of jax.Array
        Variable-length tuple depending on input parameters:

        - No PBC, matrix format: ``(neighbor_matrix1, num_neighbors1, neighbor_matrix2, num_neighbors2)``
        - No PBC, list format: ``(neighbor_list1, neighbor_ptr1, neighbor_list2, neighbor_ptr2)``
        - With PBC, matrix format: ``(neighbor_matrix1, num_neighbors1, neighbor_matrix_shifts1, neighbor_matrix2, num_neighbors2, neighbor_matrix_shifts2)``
        - With PBC, list format: ``(neighbor_list1, neighbor_ptr1, unit_shifts1, neighbor_list2, neighbor_ptr2, unit_shifts2)``

    See Also
    --------
    nvalchemiops.neighbors.naive_dual_cutoff.naive_neighbor_matrix_dual_cutoff : Core warp launcher (no PBC)
    nvalchemiops.neighbors.naive_dual_cutoff.naive_neighbor_matrix_pbc_dual_cutoff : Core warp launcher (with PBC)
    naive_neighbor_list : Single cutoff version
    """
    if pbc is None and cell is not None:
        raise ValueError("If cell is provided, pbc must also be provided")
    if pbc is not None and cell is None:
        raise ValueError("If pbc is provided, cell must also be provided")

    if cell is not None:
        cell = cell if cell.ndim == 3 else cell[jnp.newaxis, :, :]
        # Ensure cell dtype matches positions dtype so warp overload dispatch is consistent
        if cell.dtype != positions.dtype:
            cell = cell.astype(positions.dtype)
    if pbc is not None:
        pbc = pbc if pbc.ndim == 2 else pbc[jnp.newaxis, :]

    # Estimate max_neighbors if not provided - use larger cutoff for estimation
    if max_neighbors1 is None and (
        neighbor_matrix1 is None
        or (neighbor_matrix_shifts1 is None and pbc is not None)
        or num_neighbors1 is None
    ):
        max_neighbors1 = estimate_max_neighbors(cutoff2)  # Use larger cutoff
    if max_neighbors2 is None and (
        neighbor_matrix2 is None
        or (neighbor_matrix_shifts2 is None and pbc is not None)
        or num_neighbors2 is None
    ):
        max_neighbors2 = estimate_max_neighbors(cutoff2)  # Use larger cutoff

    if fill_value is None:
        fill_value = positions.shape[0]

    jax_device = positions.devices().pop()

    # Allocate first neighbor matrix
    if neighbor_matrix1 is None:
        neighbor_matrix1 = jax.device_put(
            jnp.full(
                (positions.shape[0], max_neighbors1),
                fill_value,
                dtype=jnp.int32,
            ),
            jax_device,
        )
    else:
        neighbor_matrix1 = neighbor_matrix1.at[:].set(fill_value)

    # Allocate second neighbor matrix
    if neighbor_matrix2 is None:
        neighbor_matrix2 = jax.device_put(
            jnp.full(
                (positions.shape[0], max_neighbors2),
                fill_value,
                dtype=jnp.int32,
            ),
            jax_device,
        )
    else:
        neighbor_matrix2 = neighbor_matrix2.at[:].set(fill_value)

    # Allocate first num_neighbors
    if num_neighbors1 is None:
        num_neighbors1 = jax.device_put(
            jnp.zeros(positions.shape[0], dtype=jnp.int32),
            jax_device,
        )
    else:
        num_neighbors1 = num_neighbors1.at[:].set(0)

    # Allocate second num_neighbors
    if num_neighbors2 is None:
        num_neighbors2 = jax.device_put(
            jnp.zeros(positions.shape[0], dtype=jnp.int32),
            jax_device,
        )
    else:
        num_neighbors2 = num_neighbors2.at[:].set(0)

    if pbc is not None:
        # Allocate shift matrices
        if neighbor_matrix_shifts1 is None:
            neighbor_matrix_shifts1 = jax.device_put(
                jnp.zeros(
                    (positions.shape[0], max_neighbors1, 3),
                    dtype=jnp.int32,
                ),
                jax_device,
            )
        else:
            neighbor_matrix_shifts1 = neighbor_matrix_shifts1.at[:].set(0)

        if neighbor_matrix_shifts2 is None:
            neighbor_matrix_shifts2 = jax.device_put(
                jnp.zeros(
                    (positions.shape[0], max_neighbors2, 3),
                    dtype=jnp.int32,
                ),
                jax_device,
            )
        else:
            neighbor_matrix_shifts2 = neighbor_matrix_shifts2.at[:].set(0)

        if (
            total_shifts is None
            or shift_offset is None
            or shift_range_per_dimension is None
        ):
            shift_range_per_dimension, shift_offset, total_shifts = (
                compute_naive_num_shifts(cell, cutoff2, pbc)  # Use larger cutoff
            )

    if cutoff1 <= 0 and cutoff2 <= 0:
        if return_neighbor_list:
            if pbc is not None:
                return (
                    jnp.zeros((2, 0), dtype=jnp.int32),
                    jnp.zeros((positions.shape[0] + 1,), dtype=jnp.int32),
                    jnp.zeros((0, 3), dtype=jnp.int32),
                    jnp.zeros((2, 0), dtype=jnp.int32),
                    jnp.zeros((positions.shape[0] + 1,), dtype=jnp.int32),
                    jnp.zeros((0, 3), dtype=jnp.int32),
                )
            else:
                return (
                    jnp.zeros((2, 0), dtype=jnp.int32),
                    jnp.zeros((positions.shape[0] + 1,), dtype=jnp.int32),
                    jnp.zeros((2, 0), dtype=jnp.int32),
                    jnp.zeros((positions.shape[0] + 1,), dtype=jnp.int32),
                )
        else:
            if pbc is not None:
                return (
                    neighbor_matrix1,
                    num_neighbors1,
                    neighbor_matrix_shifts1,
                    neighbor_matrix2,
                    num_neighbors2,
                    neighbor_matrix_shifts2,
                )
            else:
                return (
                    neighbor_matrix1,
                    num_neighbors1,
                    neighbor_matrix2,
                    num_neighbors2,
                )

    # Get device and dtype info
    device_str = get_warp_device_from_array(positions)

    wp_dtype = get_wp_dtype(positions.dtype)
    wp_vec_dtype = get_wp_vec_dtype(positions.dtype)
    wp_mat_dtype = get_wp_mat_dtype(cell.dtype) if cell is not None else None

    # Convert JAX arrays to Warp via dlpack (zero-copy)
    positions_wp = wp.from_dlpack(positions, dtype=wp_vec_dtype)
    neighbor_matrix1_wp = wp.from_dlpack(neighbor_matrix1, dtype=wp.int32)
    neighbor_matrix2_wp = wp.from_dlpack(neighbor_matrix2, dtype=wp.int32)
    num_neighbors1_wp = wp.from_dlpack(num_neighbors1, dtype=wp.int32)
    num_neighbors2_wp = wp.from_dlpack(num_neighbors2, dtype=wp.int32)

    if pbc is None:
        # No PBC case
        naive_neighbor_matrix_dual_cutoff(
            positions=positions_wp,
            cutoff1=cutoff1,
            cutoff2=cutoff2,
            neighbor_matrix1=neighbor_matrix1_wp,
            num_neighbors1=num_neighbors1_wp,
            neighbor_matrix2=neighbor_matrix2_wp,
            num_neighbors2=num_neighbors2_wp,
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
            dim=1,
            inputs=[
                shift_range_per_dimension_wp,
                shift_offset_wp,
                shifts_wp,
                shift_system_idx_wp,
            ],
            device=wp_device_obj,
        )

        cell_wp = wp.from_dlpack(cell, dtype=wp_mat_dtype)
        neighbor_matrix_shifts1_wp = wp.from_dlpack(
            neighbor_matrix_shifts1, dtype=wp.vec3i
        )
        neighbor_matrix_shifts2_wp = wp.from_dlpack(
            neighbor_matrix_shifts2, dtype=wp.vec3i
        )

        naive_neighbor_matrix_pbc_dual_cutoff(
            positions=positions_wp,
            cutoff1=cutoff1,
            cutoff2=cutoff2,
            cell=cell_wp,
            shifts=shifts_wp,
            neighbor_matrix1=neighbor_matrix1_wp,
            neighbor_matrix2=neighbor_matrix2_wp,
            neighbor_matrix_shifts1=neighbor_matrix_shifts1_wp,
            neighbor_matrix_shifts2=neighbor_matrix_shifts2_wp,
            num_neighbors1=num_neighbors1_wp,
            num_neighbors2=num_neighbors2_wp,
            wp_dtype=wp_dtype,
            device=device_str,
            half_fill=half_fill,
        )

    # Synchronize device
    wp.synchronize_device(device_str)

    if return_neighbor_list:
        if pbc is not None:
            neighbor_list1, neighbor_ptr1, neighbor_list_shifts1 = (
                get_neighbor_list_from_neighbor_matrix(
                    neighbor_matrix1,
                    num_neighbors=num_neighbors1,
                    neighbor_shift_matrix=neighbor_matrix_shifts1,
                    fill_value=fill_value,
                )
            )
            neighbor_list2, neighbor_ptr2, neighbor_list_shifts2 = (
                get_neighbor_list_from_neighbor_matrix(
                    neighbor_matrix2,
                    num_neighbors=num_neighbors2,
                    neighbor_shift_matrix=neighbor_matrix_shifts2,
                    fill_value=fill_value,
                )
            )
            return (
                neighbor_list1,
                neighbor_ptr1,
                neighbor_list_shifts1,
                neighbor_list2,
                neighbor_ptr2,
                neighbor_list_shifts2,
            )
        else:
            neighbor_list1, neighbor_ptr1 = get_neighbor_list_from_neighbor_matrix(
                neighbor_matrix1,
                num_neighbors=num_neighbors1,
                fill_value=fill_value,
            )
            neighbor_list2, neighbor_ptr2 = get_neighbor_list_from_neighbor_matrix(
                neighbor_matrix2,
                num_neighbors=num_neighbors2,
                fill_value=fill_value,
            )
            return neighbor_list1, neighbor_ptr1, neighbor_list2, neighbor_ptr2
    else:
        if pbc is not None:
            return (
                neighbor_matrix1,
                num_neighbors1,
                neighbor_matrix_shifts1,
                neighbor_matrix2,
                num_neighbors2,
                neighbor_matrix_shifts2,
            )
        else:
            return neighbor_matrix1, num_neighbors1, neighbor_matrix2, num_neighbors2


# ==============================================================================
# Cell List Methods
# ==============================================================================


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

    jax_device = positions.devices().pop()

    # Simple estimation: compute total volume and estimate cell volume
    # Cell volume = det(cell_matrix)
    det = jnp.linalg.det(cell[0])
    volume = jnp.abs(det)
    cell_volume = cutoff**3
    num_cells_est = int(volume / cell_volume * buffer_factor)
    max_total_cells = max(num_cells_est, 8)  # Minimum 8 cells

    # Estimate cells per dimension
    cells_per_dimension = jax.device_put(
        jnp.ceil(jnp.ones(3) * (max_total_cells ** (1 / 3))).astype(jnp.int32),
        jax_device,
    )

    # Search radius estimate
    neighbor_search_radius = jax.device_put(
        jnp.ones(3, dtype=jnp.int32) * 1,
        jax_device,
    )

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

    jax_device = positions.devices().pop()

    if max_total_cells is None:
        max_total_cells, _, neighbor_search_radius_est = estimate_cell_list_sizes(
            positions, cell, cutoff, pbc
        )
        if neighbor_search_radius is None:
            neighbor_search_radius = neighbor_search_radius_est
    else:
        if neighbor_search_radius is None:
            neighbor_search_radius = jax.device_put(
                jnp.ones(3, dtype=jnp.int32),
                jax_device,
            )

    # Allocate cell list tensors if not provided
    if cells_per_dimension is None:
        cells_per_dimension = jax.device_put(
            jnp.ones(3, dtype=jnp.int32),
            jax_device,
        )
    if atom_periodic_shifts is None:
        atom_periodic_shifts = jax.device_put(
            jnp.zeros((positions.shape[0], 3), dtype=jnp.int32),
            jax_device,
        )
    if atom_to_cell_mapping is None:
        atom_to_cell_mapping = jax.device_put(
            jnp.zeros((positions.shape[0], 3), dtype=jnp.int32),
            jax_device,
        )
    if atoms_per_cell_count is None:
        atoms_per_cell_count = jax.device_put(
            jnp.zeros(max_total_cells, dtype=jnp.int32),
            jax_device,
        )
    if cell_atom_start_indices is None:
        cell_atom_start_indices = jax.device_put(
            jnp.zeros(max_total_cells, dtype=jnp.int32),
            jax_device,
        )
    if cell_atom_list is None:
        cell_atom_list = jax.device_put(
            jnp.zeros(positions.shape[0], dtype=jnp.int32),
            jax_device,
        )

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
    atom_to_cell_mapping : jax.Array, shape (total_atoms, 3), dtype=int32
        3D cell coordinates for each atom.
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

    jax_device = positions.devices().pop()

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
        Neighbor information. If return_neighbor_list=False, returns neighbor_matrix
        with shape (total_atoms, max_neighbors) and dtype int32. If True, returns
        neighbor_list with shape (2, num_pairs) and dtype int32.
    neighbor_ptr_or_count : jax.Array
        Additional neighbor information. If return_neighbor_list=False, returns
        num_neighbors with shape (total_atoms,). If True, returns neighbor_ptr
        with shape (total_atoms + 1,).
    cell_data : jax.Array, optional
        Cell list construction info tuple (cells_per_dimension, atom_to_cell_mapping,
        atoms_per_cell_count, cell_atom_start_indices, cell_atom_list).

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
