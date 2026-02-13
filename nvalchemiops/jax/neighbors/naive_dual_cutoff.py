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

"""JAX bindings for unbatched naive O(N^2) dual cutoff neighbor list construction."""

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
)
from nvalchemiops.neighbors.naive_dual_cutoff import (
    naive_neighbor_matrix_dual_cutoff,
    naive_neighbor_matrix_pbc_dual_cutoff,
)
from nvalchemiops.neighbors.neighbor_utils import (
    _expand_naive_shifts,
    estimate_max_neighbors,
)

__all__ = ["naive_neighbor_list_dual_cutoff"]


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

    # Allocate first neighbor matrix
    if neighbor_matrix1 is None:
        neighbor_matrix1 = jnp.full(
            (positions.shape[0], max_neighbors1),
            fill_value,
            dtype=jnp.int32,
        )
    else:
        neighbor_matrix1 = neighbor_matrix1.at[:].set(fill_value)

    # Allocate second neighbor matrix
    if neighbor_matrix2 is None:
        neighbor_matrix2 = jnp.full(
            (positions.shape[0], max_neighbors2),
            fill_value,
            dtype=jnp.int32,
        )
    else:
        neighbor_matrix2 = neighbor_matrix2.at[:].set(fill_value)

    # Allocate first num_neighbors
    if num_neighbors1 is None:
        num_neighbors1 = jnp.zeros(positions.shape[0], dtype=jnp.int32)
    else:
        num_neighbors1 = num_neighbors1.at[:].set(0)

    # Allocate second num_neighbors
    if num_neighbors2 is None:
        num_neighbors2 = jnp.zeros(positions.shape[0], dtype=jnp.int32)
    else:
        num_neighbors2 = num_neighbors2.at[:].set(0)

    if pbc is not None:
        # Allocate shift matrices
        if neighbor_matrix_shifts1 is None:
            neighbor_matrix_shifts1 = jnp.zeros(
                (positions.shape[0], max_neighbors1, 3),
                dtype=jnp.int32,
            )
        else:
            neighbor_matrix_shifts1 = neighbor_matrix_shifts1.at[:].set(0)

        if neighbor_matrix_shifts2 is None:
            neighbor_matrix_shifts2 = jnp.zeros(
                (positions.shape[0], max_neighbors2, 3),
                dtype=jnp.int32,
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
        shifts = jnp.empty((total_shifts, 3), dtype=jnp.int32)
        shift_system_idx = jnp.empty((total_shifts,), dtype=jnp.int32)
        shifts_wp = wp.from_dlpack(shifts, dtype=wp.vec3i)
        shift_system_idx_wp = wp.from_dlpack(shift_system_idx, dtype=wp.int32)
        shift_range_per_dimension_wp = wp.from_dlpack(
            shift_range_per_dimension, dtype=wp.vec3i
        )
        shift_offset_wp = wp.from_dlpack(shift_offset, dtype=wp.int32)

        wp_device_obj = wp.get_device(device_str)
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
