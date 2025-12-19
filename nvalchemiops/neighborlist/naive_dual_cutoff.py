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

from typing import Any

import torch
import warp as wp

from nvalchemiops.neighborlist.neighbor_utils import (
    _expand_naive_shifts,
    _update_neighbor_matrix,
    _update_neighbor_matrix_pbc,
    compute_naive_num_shifts,
    estimate_max_neighbors,
    get_neighbor_list_from_neighbor_matrix,
)
from nvalchemiops.types import get_wp_dtype, get_wp_mat_dtype, get_wp_vec_dtype

###########################################################################################
########################### Naive Neighbor List Kernels ###################################
###########################################################################################


@wp.kernel(enable_backward=False)
def _fill_naive_neighbor_matrix_dual_cutoff(
    positions: wp.array(dtype=Any),
    cutoff1_sq: Any,
    cutoff2_sq: Any,
    neighbor_matrix1: wp.array2d(dtype=wp.int32, ndim=2),
    num_neighbors1: wp.array(dtype=wp.int32),
    neighbor_matrix2: wp.array2d(dtype=wp.int32, ndim=2),
    num_neighbors2: wp.array(dtype=wp.int32),
    half_fill: wp.bool,
) -> None:
    """Calculate two neighbor matrices using dual cutoffs with naive O(N^2) algorithm.

    Computes pairwise distances between all atoms and identifies neighbors
    within two different cutoff distances simultaneously. This is more efficient
    than running two separate neighbor calculations when both neighbor lists are needed.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Atomic coordinates in Cartesian space. Each row represents one atom's
        (x, y, z) position.
    cutoff1_sq : float
        Squared first cutoff distance in Cartesian units (typically the smaller cutoff).
        Must be positive. Atoms within this distance are considered neighbors.
    cutoff2_sq : float
        Squared second cutoff distance in Cartesian units (typically the larger cutoff).
        Must be positive and should be >= cutoff1_sq for optimal performance.
    neighbor_matrix1 : wp.array, shape (total_atoms, max_neighbors1), dtype=wp.int32
        OUTPUT: First neighbor matrix for cutoff1 to be filled with atom indices.
        Entries are filled with atom indices, remaining entries stay as initialized.
    num_neighbors1 : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Number of neighbors found for each atom within cutoff1.
        Updated in-place with actual neighbor counts.
    neighbor_matrix2 : wp.array, shape (total_atoms, max_neighbors2), dtype=wp.int32
        OUTPUT: Second neighbor matrix for cutoff2 to be filled with atom indices.
        Entries are filled with atom indices, remaining entries stay as initialized.
    num_neighbors2 : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Number of neighbors found for each atom within cutoff2.
        Updated in-place with actual neighbor counts.
    half_fill : wp.bool
        If True, only store relationships where i < j to avoid double counting.
        If False, store all neighbor relationships symmetrically.

    Returns
    -------
    None
        This function modifies the input arrays in-place:

        - neighbor_matrix1 : Filled with neighbor atom indices within cutoff1
        - num_neighbors1 : Updated with neighbor counts per atom for cutoff1
        - neighbor_matrix2 : Filled with neighbor atom indices within cutoff2
        - num_neighbors2 : Updated with neighbor counts per atom for cutoff2

    See Also
    --------
    _fill_naive_neighbor_matrix : Single cutoff version
    _fill_naive_neighbor_matrix_pbc_dual_cutoff : Version with periodic boundaries
    """
    tid = wp.tid()
    i = tid
    j_end = positions.shape[0]

    positions_i = positions[i]
    maxnb1 = neighbor_matrix1.shape[1]
    maxnb2 = neighbor_matrix2.shape[1]
    for j in range(i + 1, j_end):
        diff = positions_i - positions[j]
        dist_sq = wp.length_sq(diff)
        if dist_sq < cutoff2_sq:
            _update_neighbor_matrix(
                i, j, neighbor_matrix2, num_neighbors2, maxnb2, half_fill
            )
            if dist_sq < cutoff1_sq:
                _update_neighbor_matrix(
                    i, j, neighbor_matrix1, num_neighbors1, maxnb1, half_fill
                )


@wp.kernel(enable_backward=False)
def _fill_naive_neighbor_matrix_pbc_dual_cutoff(
    positions: wp.array(dtype=Any),
    cutoff1_sq: Any,
    cutoff2_sq: Any,
    cell: wp.array(dtype=Any),
    shifts: wp.array(dtype=wp.vec3i),
    neighbor_matrix1: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix2: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts1: wp.array(dtype=wp.vec3i, ndim=2),
    neighbor_matrix_shifts2: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors1: wp.array(dtype=wp.int32),
    num_neighbors2: wp.array(dtype=wp.int32),
    half_fill: wp.bool,
) -> None:
    """Calculate two neighbor matrices with periodic boundary conditions using dual cutoffs and naive O(N^2) algorithm.

    Computes neighbor relationships between atoms across periodic boundaries by
    considering all periodic images within two different cutoff distances simultaneously.
    Uses a 2D launch pattern to parallelize over both atoms and periodic shifts.
    This is more efficient than running two separate PBC neighbor calculations.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Atomic coordinates in Cartesian space. Each row represents one atom's
        (x, y, z) position.
    cutoff1_sq : float
        Squared first cutoff distance in Cartesian units (typically the smaller cutoff).
        Must be positive. Atoms within this distance are considered neighbors.
    cutoff2_sq : float
        Squared second cutoff distance in Cartesian units (typically the larger cutoff).
        Must be positive and should be >= cutoff1_sq for optimal performance.
    cell : wp.array, shape (1, 3, 3), dtype=wp.mat33*
        Cell matrix defining lattice vectors in Cartesian coordinates.
        Single 3x3 matrix representing the periodic cell.
    shifts : wp.array, shape (total_shifts, 3), dtype=wp.vec3i
        Integer shift vectors for periodic images.
        Each row represents (nx, ny, nz) multiples of the cell vectors.
    neighbor_matrix1 : wp.array, shape (total_atoms, max_neighbors1), dtype=wp.int32
        OUTPUT: First neighbor matrix for cutoff1 to be filled with atom indices.
        Entries are filled with atom indices, remaining entries stay as initialized.
    neighbor_matrix2 : wp.array, shape (total_atoms, max_neighbors2), dtype=wp.int32
        OUTPUT: Second neighbor matrix for cutoff2 to be filled with atom indices.
        Entries are filled with atom indices, remaining entries stay as initialized.
    neighbor_matrix_shifts1 : wp.array, shape (total_atoms, max_neighbors1), dtype=wp.vec3i
        OUTPUT: Matrix storing shift vectors for each neighbor relationship in matrix1.
        Each entry corresponds to the shift used for the neighbor in neighbor_matrix1.
    neighbor_matrix_shifts2 : wp.array, shape (total_atoms, max_neighbors2), dtype=wp.vec3i
        OUTPUT: Matrix storing shift vectors for each neighbor relationship in matrix2.
        Each entry corresponds to the shift used for the neighbor in neighbor_matrix2.
    num_neighbors1 : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Number of neighbors found for each atom within cutoff1.
        Updated in-place with actual neighbor counts.
    num_neighbors2 : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Number of neighbors found for each atom within cutoff2.
        Updated in-place with actual neighbor counts.
    half_fill : wp.bool
        If True, only store relationships where i < j to avoid double counting.
        If False, store all neighbor relationships symmetrically.

    Returns
    -------
    None
        This function modifies the input arrays in-place:

        - neighbor_matrix1 : Filled with neighbor atom indices within cutoff1
        - neighbor_matrix_shifts1 : Filled with corresponding shift vectors for cutoff1
        - num_neighbors1 : Updated with neighbor counts per atom for cutoff1
        - neighbor_matrix2 : Filled with neighbor atom indices within cutoff2
        - neighbor_matrix_shifts2 : Filled with corresponding shift vectors for cutoff2
        - num_neighbors2 : Updated with neighbor counts per atom for cutoff2

    See Also
    --------
    _fill_naive_neighbor_matrix_dual_cutoff : Version without periodic boundary conditions
    _fill_naive_neighbor_matrix_pbc : Single cutoff PBC version
    """
    ishift, iatom = wp.tid()

    jatom_start = 0
    jatom_end = positions.shape[0]

    maxnb1 = neighbor_matrix1.shape[1]
    maxnb2 = neighbor_matrix2.shape[1]

    # Get the atom coordinates and shift vector
    _positions = positions[iatom]
    _cell = cell[0]
    _shift = shifts[ishift]

    positions_shifted = wp.transpose(_cell) * type(_cell[0])(_shift) + _positions

    _zero_shift = _shift[0] == 0 and _shift[1] == 0 and _shift[2] == 0
    if _zero_shift:
        jatom_end = iatom

    for jatom in range(jatom_start, jatom_end):
        diff = positions_shifted - positions[jatom]
        dist_sq = wp.length_sq(diff)
        if dist_sq < cutoff2_sq:
            _update_neighbor_matrix_pbc(
                jatom,
                iatom,
                neighbor_matrix2,
                neighbor_matrix_shifts2,
                num_neighbors2,
                _shift,
                maxnb2,
                half_fill,
            )
            if dist_sq < cutoff1_sq:
                _update_neighbor_matrix_pbc(
                    jatom,
                    iatom,
                    neighbor_matrix1,
                    neighbor_matrix_shifts1,
                    num_neighbors1,
                    _shift,
                    maxnb1,
                    half_fill,
                )


## Generate overloads for all kernels
T = [wp.float32, wp.float64, wp.float16]
V = [wp.vec3f, wp.vec3d, wp.vec3h]
M = [wp.mat33f, wp.mat33d, wp.mat33h]
_fill_naive_neighbor_matrix_dual_cutoff_overload = {}
_fill_naive_neighbor_matrix_pbc_dual_cutoff_overload = {}
for t, v, m in zip(T, V, M):
    _fill_naive_neighbor_matrix_dual_cutoff_overload[t] = wp.overload(
        _fill_naive_neighbor_matrix_dual_cutoff,
        [
            wp.array(dtype=v),
            t,
            t,
            wp.array2d(dtype=wp.int32),
            wp.array(dtype=wp.int32),
            wp.array2d(dtype=wp.int32),
            wp.array(dtype=wp.int32),
            wp.bool,
        ],
    )

    _fill_naive_neighbor_matrix_pbc_dual_cutoff_overload[t] = wp.overload(
        _fill_naive_neighbor_matrix_pbc_dual_cutoff,
        [
            wp.array(dtype=v),
            t,
            t,
            wp.array(dtype=m),
            wp.array(dtype=wp.vec3i),
            wp.array2d(dtype=wp.int32),
            wp.array2d(dtype=wp.int32),
            wp.array2d(dtype=wp.vec3i),
            wp.array2d(dtype=wp.vec3i),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32),
            wp.bool,
        ],
    )

################################################################################
########################### Dual cutoff ########################################
################################################################################


@torch.library.custom_op(
    "nvalchemiops::_naive_neighbor_matrix_no_pbc_no_alloc_dual_cutoff",
    mutates_args=(
        "neighbor_matrix1",
        "num_neighbors1",
        "neighbor_matrix2",
        "num_neighbors2",
    ),
)
def _naive_neighbor_matrix_no_pbc_dual_cutoff(
    positions: torch.Tensor,
    cutoff1: float,
    cutoff2: float,
    neighbor_matrix1: torch.Tensor,
    num_neighbors1: torch.Tensor,
    neighbor_matrix2: torch.Tensor,
    num_neighbors2: torch.Tensor,
    half_fill: bool = False,
) -> None:
    """Fill two neighbor matrices for atoms using dual cutoffs with naive O(N^2) algorithm.

    Custom PyTorch operator that computes pairwise distances and fills
    two neighbor matrices with atom indices within different cutoff distances
    simultaneously. This is more efficient than running two separate neighbor
    calculations when both neighbor lists are needed. No periodic boundary
    conditions are applied.

    This function is torch compilable.

    This function does not allocate any tensors.

    Parameters
    ----------
    positions : torch.Tensor, shape (total_atoms, 3), dtype=torch.float32 or torch.float64
        Atomic coordinates in Cartesian space. Each row represents one atom's
        (x, y, z) position.
    cutoff1 : float
        First cutoff distance in Cartesian units (typically the smaller cutoff).
        Must be positive. Atoms within this distance are considered neighbors.
    cutoff2 : float
        Second cutoff distance in Cartesian units (typically the larger cutoff).
        Must be positive and should be >= cutoff1 for optimal performance.
    max_neighbors1 : int
        Maximum number of neighbors per atom for the first neighbor matrix.
        Must be positive. If exceeded, excess neighbors are ignored.
    max_neighbors2 : int
        Maximum number of neighbors per atom for the second neighbor matrix.
        Must be positive. If exceeded, excess neighbors are ignored.
    neighbor_matrix1 : torch.Tensor, shape (total_atoms, max_neighbors1), dtype=torch.int32
        OUTPUT: First neighbor matrix for cutoff1 to be filled with atom indices.
        Must be pre-allocated. Entries are filled with atom indices.
    num_neighbors1 : torch.Tensor, shape (total_atoms,), dtype=torch.int32
        OUTPUT: Number of neighbors found for each atom within cutoff1.
        Must be pre-allocated. Updated in-place with actual neighbor counts.
    neighbor_matrix2 : torch.Tensor, shape (total_atoms, max_neighbors2), dtype=torch.int32
        OUTPUT: Second neighbor matrix for cutoff2 to be filled with atom indices.
        Must be pre-allocated. Entries are filled with atom indices.
    num_neighbors2 : torch.Tensor, shape (total_atoms,), dtype=torch.int32
        OUTPUT: Number of neighbors found for each atom within cutoff2.
        Must be pre-allocated. Updated in-place with actual neighbor counts.
    half_fill : bool
        If True, only store relationships where i < j to avoid double counting.
        If False, store all neighbor relationships symmetrically.
        Default is False.

    Returns
    -------
    None
        This function modifies the input tensors in-place:

        - neighbor_matrix1 : Filled with neighbor atom indices within cutoff1
        - num_neighbors1 : Updated with neighbor counts per atom for cutoff1
        - neighbor_matrix2 : Filled with neighbor atom indices within cutoff2
        - num_neighbors2 : Updated with neighbor counts per atom for cutoff2

    See Also
    --------
    _naive_neighbor_matrix_dual_cutoff_no_pbc : Higher-level wrapper function
    _naive_neighbor_matrix_no_pbc_no_alloc : Single cutoff version
    """
    device = positions.device
    wp_vec_dtype = get_wp_vec_dtype(positions.dtype)
    wp_dtype = get_wp_dtype(positions.dtype)
    wp_positions = wp.from_torch(positions, dtype=wp_vec_dtype, return_ctype=True)
    wp_neighbor_matrix1 = wp.from_torch(
        neighbor_matrix1, dtype=wp.int32, return_ctype=True
    )
    wp_num_neighbors1 = wp.from_torch(num_neighbors1, dtype=wp.int32, return_ctype=True)
    wp_neighbor_matrix2 = wp.from_torch(
        neighbor_matrix2, dtype=wp.int32, return_ctype=True
    )
    wp_num_neighbors2 = wp.from_torch(num_neighbors2, dtype=wp.int32, return_ctype=True)

    wp.launch(
        kernel=_fill_naive_neighbor_matrix_dual_cutoff_overload[wp_dtype],
        dim=positions.shape[0],
        inputs=[
            wp_positions,
            wp_dtype(cutoff1 * cutoff1),
            wp_dtype(cutoff2 * cutoff2),
            wp_neighbor_matrix1,
            wp_num_neighbors1,
            wp_neighbor_matrix2,
            wp_num_neighbors2,
            half_fill,
        ],
        device=wp.device_from_torch(device),
    )


@torch.library.custom_op(
    "nvalchemiops::_naive_neighbor_matrix_pbc_dual_cutoff",
    mutates_args=(
        "neighbor_matrix1",
        "neighbor_matrix2",
        "neighbor_matrix_shifts1",
        "neighbor_matrix_shifts2",
        "num_neighbors1",
        "num_neighbors2",
    ),
)
def _naive_neighbor_matrix_pbc_dual_cutoff(
    positions: torch.Tensor,
    cutoff1: float,
    cutoff2: float,
    cell: torch.Tensor,
    neighbor_matrix1: torch.Tensor,
    neighbor_matrix2: torch.Tensor,
    neighbor_matrix_shifts1: torch.Tensor,
    neighbor_matrix_shifts2: torch.Tensor,
    num_neighbors1: torch.Tensor,
    num_neighbors2: torch.Tensor,
    shift_range_per_dimension: torch.Tensor,
    shift_offset: torch.Tensor,
    total_shifts: int,
    half_fill: bool = False,
) -> None:
    """Compute two neighbor matrices with periodic boundary conditions using dual cutoffs and naive O(N^2) algorithm.

    Custom PyTorch operator that computes neighbor relationships between atoms
    across periodic boundaries for two different cutoff distances simultaneously.
    Uses pre-computed shift vectors for torch compilation compatibility. This is
    more efficient than running two separate PBC neighbor calculations.

    This function is torch compilable.

    Parameters
    ----------
    positions : torch.Tensor, shape (total_atoms, 3), dtype=torch.float32 or torch.float64
        Atomic coordinates in Cartesian space. Each row represents one atom's
        (x, y, z) position.
    cutoff1 : float
        First cutoff distance in Cartesian units (typically the smaller cutoff).
        Must be positive. Atoms within this distance are considered neighbors.
    cutoff2 : float
        Second cutoff distance in Cartesian units (typically the larger cutoff).
        Must be positive and should be >= cutoff1 for optimal performance.
    cell : torch.Tensor, shape (1, 3, 3), dtype=torch.float32 or torch.float64
        Cell matrix defining lattice vectors in Cartesian coordinates.
        Single 3x3 matrix representing the periodic cell.
    pbc : torch.Tensor, shape (1, 3), dtype=torch.bool
        Periodic boundary condition flags for each dimension.
        True enables periodicity in that direction.
    neighbor_matrix1 : torch.Tensor, shape (total_atoms, max_neighbors1), dtype=torch.int32
        OUTPUT: First neighbor matrix for cutoff1 to be filled with atom indices.
        Must be pre-allocated. Entries are filled with atom indices.
    neighbor_matrix2 : torch.Tensor, shape (total_atoms, max_neighbors2), dtype=torch.int32
        OUTPUT: Second neighbor matrix for cutoff2 to be filled with atom indices.
        Must be pre-allocated. Entries are filled with atom indices.
    neighbor_matrix_shifts1 : torch.Tensor, shape (total_atoms, max_neighbors1, 3), dtype=torch.int32
        OUTPUT: Shift vectors for each neighbor relationship in matrix1.
        Must be pre-allocated. Entries are filled with shift vectors.
    neighbor_matrix_shifts2 : torch.Tensor, shape (total_atoms, max_neighbors2, 3), dtype=torch.int32
        OUTPUT: Shift vectors for each neighbor relationship in matrix2.
        Must be pre-allocated. Entries are filled with shift vectors.
    num_neighbors1 : torch.Tensor, shape (total_atoms,), dtype=torch.int32
        OUTPUT: Number of neighbors found for each atom within cutoff1.
        Must be pre-allocated. Updated in-place with actual neighbor counts.
    num_neighbors2 : torch.Tensor, shape (total_atoms,), dtype=torch.int32
        OUTPUT: Number of neighbors found for each atom within cutoff2.
        Must be pre-allocated. Updated in-place with actual neighbor counts.
    shift_range_per_dimension : torch.Tensor, shape (1, 3), dtype=torch.int32
        Shift range in each dimension for each system.
    shift_offset : torch.Tensor, shape (2,), dtype=torch.int32
        Cumulative sum of number of shifts for each system.
    total_shifts : int
        Total number of shifts.
    half_fill : bool
        If True, only store relationships where i < j to avoid double counting.
        If False, store all neighbor relationships symmetrically.
        Default is False.

    See Also
    --------
    _naive_neighbor_matrix_dual_cutoff_pbc : Higher-level wrapper function
    _compute_total_shifts : Computes the required shift vectors
    """
    total_atoms = positions.shape[0]
    device = positions.device
    wp_vec_dtype = get_wp_vec_dtype(positions.dtype)
    wp_mat_dtype = get_wp_mat_dtype(positions.dtype)
    wp_dtype = get_wp_dtype(positions.dtype)
    wp_positions = wp.from_torch(positions, dtype=wp_vec_dtype)
    wp_cell = wp.from_torch(cell, dtype=wp_mat_dtype)

    shifts = torch.empty((total_shifts, 3), dtype=torch.int32, device=device)
    wp_shifts = wp.from_torch(shifts, dtype=wp.vec3i, return_ctype=True)
    shift_system_idx = torch.empty((total_shifts,), dtype=torch.int32, device=device)
    wp_shift_system_idx = wp.from_torch(
        shift_system_idx, dtype=wp.int32, return_ctype=True
    )

    wp.launch(
        kernel=_expand_naive_shifts,
        dim=1,
        inputs=[
            wp.from_torch(shift_range_per_dimension, dtype=wp.vec3i, return_ctype=True),
            wp.from_torch(shift_offset, dtype=wp.int32, return_ctype=True),
            wp_shifts,
            wp_shift_system_idx,
        ],
        device=wp.device_from_torch(device),
    )

    # Initialize neighbor matrices
    wp_neighbor_matrix1 = wp.from_torch(
        neighbor_matrix1, dtype=wp.int32, return_ctype=True
    )
    wp_neighbor_matrix2 = wp.from_torch(
        neighbor_matrix2, dtype=wp.int32, return_ctype=True
    )

    wp_num_neighbors1 = wp.from_torch(num_neighbors1, dtype=wp.int32, return_ctype=True)
    wp_num_neighbors2 = wp.from_torch(num_neighbors2, dtype=wp.int32, return_ctype=True)
    wp_neighbor_matrix_shifts1 = wp.from_torch(
        neighbor_matrix_shifts1, dtype=wp.vec3i, return_ctype=True
    )
    wp_neighbor_matrix_shifts2 = wp.from_torch(
        neighbor_matrix_shifts2, dtype=wp.vec3i, return_ctype=True
    )
    wp.launch(
        kernel=_fill_naive_neighbor_matrix_pbc_dual_cutoff_overload[wp_dtype],
        dim=(total_shifts, total_atoms),
        inputs=[
            wp_positions,
            wp_dtype(cutoff1 * cutoff1),
            wp_dtype(cutoff2 * cutoff2),
            wp_cell,
            wp_shifts,
            wp_neighbor_matrix1,
            wp_neighbor_matrix2,
            wp_neighbor_matrix_shifts1,
            wp_neighbor_matrix_shifts2,
            wp_num_neighbors1,
            wp_num_neighbors2,
            half_fill,
        ],
        device=wp.device_from_torch(device),
    )


def naive_neighbor_list_dual_cutoff(
    positions: torch.Tensor,
    cutoff1: float,
    cutoff2: float,
    pbc: torch.Tensor | None = None,
    cell: torch.Tensor | None = None,
    max_neighbors1: int | None = None,
    max_neighbors2: int | None = None,
    half_fill: bool = False,
    fill_value: int | None = None,
    return_neighbor_list: bool = False,
    neighbor_matrix1: torch.Tensor | None = None,
    neighbor_matrix2: torch.Tensor | None = None,
    neighbor_matrix_shifts1: torch.Tensor | None = None,
    neighbor_matrix_shifts2: torch.Tensor | None = None,
    num_neighbors1: torch.Tensor | None = None,
    num_neighbors2: torch.Tensor | None = None,
    shift_range_per_dimension: torch.Tensor | None = None,
    shift_offset: torch.Tensor | None = None,
    total_shifts: int | None = None,
) -> (
    tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]
    | tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
):
    """Compute neighbor list using naive O(N^2) algorithm with dual cutoffs.

    Identifies all atom pairs within two different cutoff distances using a
    single brute-force pairwise distance calculation. This is more efficient
    than running two separate neighbor calculations when both neighbor lists are needed.
    Supports both non-periodic and periodic boundary conditions.

    For non-pbc systems, this function is torch compilable. For pbc systems,
    precompute the shift vectors using _compute_total_shifts to maintain torch compilability.

    Parameters
    ----------
    positions : torch.Tensor, shape (total_atoms, 3), dtype=torch.float32 or torch.float64
        Atomic coordinates in Cartesian space. Each row represents one atom's
        (x, y, z) position.
    cutoff1 : float
        First cutoff distance in Cartesian units (typically the smaller cutoff).
        Must be positive. Atoms within this distance are considered neighbors.
    cutoff2 : float
        Second cutoff distance in Cartesian units (typically the larger cutoff).
        Must be positive and should be >= cutoff1 for optimal performance.
    pbc : torch.Tensor, shape (num_systems, 3), dtype=torch.bool, optional
        Periodic boundary condition flags for each dimension.
        True enables periodicity in that direction. Default is None (no PBC).
    cell : torch.Tensor, shape (num_systems, 3, 3), dtype=torch.float32 or torch.float64, optional
        Cell matrices defining lattice vectors in Cartesian coordinates.
        Required if pbc is provided. Default is None.
    max_neighbors1 : int
        Maximum number of neighbors per atom for the first neighbor matrix.
        Must be positive. If exceeded, excess neighbors are ignored.
    max_neighbors2 : int, optional
        Maximum number of neighbors per atom for the second neighbor matrix.
        If None, defaults to max_neighbors1. Must be positive if provided.
    half_fill : bool, optional
        If True, only store relationships where i < j to avoid double counting.
        If False, store all neighbor relationships symmetrically. Default is False.
    fill_value : int | None, optional
        Value to fill the neighbor matrices with. Default is total_atoms.
    return_neighbor_list : bool, optional - default = False
        If True, convert the neighbor matrix to a neighbor list (idx_i, idx_j) format by
        creating a mask over the fill_value, which can incur a performance penalty.
        We recommend using the neighbor matrix format,
        and only convert to a neighbor list format if absolutely necessary.
    neighbor_matrix1 : torch.Tensor, shape (total_atoms, max_neighbors1), dtype=torch.int32, optional
        First neighbor matrix for cutoff1 to be filled with fill_value.
        Must be pre-allocated. Entries are filled with fill_value.
    neighbor_matrix2 : torch.Tensor, shape (total_atoms, max_neighbors2), dtype=torch.int32, optional
        Second neighbor matrix for cutoff2 to be filled with fill_value.
        Must be pre-allocated. Entries are filled with fill_value.
    neighbor_matrix_shifts1 : torch.Tensor, shape (total_atoms, max_neighbors1, 3), dtype=torch.int32, optional
        Shift vectors for each neighbor relationship in the first matrix.
        Must be pre-allocated. Entries are filled with shift vectors.
    neighbor_matrix_shifts2 : torch.Tensor, shape (total_atoms, max_neighbors2, 3), dtype=torch.int32, optional
        Shift vectors for each neighbor relationship in the second matrix.
        Must be pre-allocated. Entries are filled with shift vectors.
    num_neighbors1 : torch.Tensor, shape (total_atoms,), dtype=torch.int32, optional
        Number of neighbors found for each atom within cutoff1.
        Must be pre-allocated. Updated in-place with actual neighbor counts.
    num_neighbors2 : torch.Tensor, shape (total_atoms,), dtype=torch.int32, optional
        Number of neighbors found for each atom within cutoff2.
        Must be pre-allocated. Updated in-place with actual neighbor counts.
    shift_range_per_dimension : torch.Tensor, shape (1, 3), dtype=torch.int32, optional
        Shift range in each dimension for each system.
    shift_offset : torch.Tensor, shape (2,), dtype=torch.int32, optional
        Cumulative sum of number of shifts for each system.
    total_shifts : int, optional
        Total number of shifts.

    Returns
    -------
    results : tuple of torch.Tensor
        Variable-length tuple with interleaved results for cutoff1 and cutoff2. The return pattern follows:

        - No PBC, matrix format: ``(neighbor_matrix1, num_neighbors1, neighbor_matrix2, num_neighbors2)``
        - No PBC, list format: ``(neighbor_list1, neighbor_ptr1, neighbor_list2, neighbor_ptr2)``
        - With PBC, matrix format: ``(neighbor_matrix1, num_neighbors1, neighbor_matrix_shifts1, neighbor_matrix2, num_neighbors2, neighbor_matrix_shifts2)``
        - With PBC, list format: ``(neighbor_list1, neighbor_ptr1, neighbor_list_shifts1, neighbor_list2, neighbor_ptr2, neighbor_list_shifts2)``

        **Components returned (interleaved for each cutoff):**

        - **neighbor_data1, neighbor_data2** (tensors): Neighbor indices, format depends on ``return_neighbor_list``:

            * If ``return_neighbor_list=False`` (default): Returns ``neighbor_matrix1`` and ``neighbor_matrix2``
              with shapes (total_atoms, max_neighbors1) and (total_atoms, max_neighbors2), dtype int32.
              Each row i contains indices of atom i's neighbors within the respective cutoff.
            * If ``return_neighbor_list=True``: Returns ``neighbor_list1`` and ``neighbor_list2`` with shapes
              (2, num_pairs1) and (2, num_pairs2), dtype int32, in COO format [source_atoms, target_atoms].

        - **num_neighbor_data1, num_neighbor_data2** (tensors): Information about the number of neighbors for each atom,
          format depends on ``return_neighbor_list``:

            * If ``return_neighbor_list=False`` (default): Returns ``num_neighbors1`` and ``num_neighbors2`` with shape (total_atoms,), dtype int32.
              Count of neighbors found for each atom within cutoff1 and cutoff2 respectively.
            * If ``return_neighbor_list=True``: Returns ``neighbor_ptr1`` and ``neighbor_ptr2`` with shape (total_atoms + 1,), dtype int32.
              CSR-style pointer arrays where ``neighbor_ptr_data[i]`` to ``neighbor_ptr_data[i+1]`` gives the range of
              neighbors for atom i in the flattened neighbor list.

        - **neighbor_shift_data1, neighbor_shift_data2** (tensors, optional): Periodic shift vectors, only when ``pbc`` is provided:
          format depends on ``return_neighbor_list``:

            * If ``return_neighbor_list=False`` (default): Returns ``neighbor_matrix_shifts1`` and ``neighbor_matrix_shifts2`` with
              shape (total_atoms, max_neighbors1, 3) and (total_atoms, max_neighbors2, 3), dtype int32.
            * If ``return_neighbor_list=True``: Returns ``unit_shifts1`` and ``unit_shifts2`` with shapes
              (num_pairs1, 3) and (num_pairs2, 3), dtype int32.

    Examples
    --------
    Basic usage with dual cutoffs:

    >>> import torch
    >>> positions = torch.rand(100, 3) * 10.0  # 100 atoms in 10x10x10 box
    >>> cutoff1 = 2.0  # Short-range interactions
    >>> cutoff2 = 4.0  # Long-range interactions
    >>> max_neighbors1, max_neighbors2 = 20, 50
    >>>
    >>> results = naive_neighbor_list_dual_cutoff(
    ...     positions, cutoff1, cutoff2, max_neighbors1=max_neighbors1, max_neighbors2=max_neighbors2
    ... )
    >>> neighbor_matrix1, num_neighbors1, neighbor_matrix2, num_neighbors2 = results
    >>> print(f"Short-range pairs: {num_neighbors1.sum()}")
    >>> print(f"Long-range pairs: {num_neighbors2.sum()}")

    With periodic boundary conditions:

    >>> cell = torch.eye(3).unsqueeze(0) * 10.0  # 10x10x10 cubic cell
    >>> pbc = torch.tensor([[True, True, True]])  # Periodic in all directions
    >>> results = naive_neighbor_list_dual_cutoff(
    ...     positions, cutoff1, cutoff2, max_neighbors1=max_neighbors1, max_neighbors2=max_neighbors2,
    ...     pbc=pbc, cell=cell
    ... )
    >>> (neighbor_matrix1, num_neighbors1, shifts1,
    ...  neighbor_matrix2, num_neighbors2, shifts2) = results

    Preallocate tensors for non-pbc systems:
    >>> max_neighbors1 = 100
    >>> neighbor_matrix1 = torch.zeros((positions.shape[0], max_neighbors1), dtype=torch.int32, device=positions.device)
    >>> neighbor_matrix2 = torch.zeros((positions.shape[0], max_neighbors2), dtype=torch.int32, device=positions.device)
    >>> num_neighbors1 = torch.zeros(positions.shape[0], dtype=torch.int32, device=positions.device)
    >>> num_neighbors2 = torch.zeros(positions.shape[0], dtype=torch.int32, device=positions.device)
    >>> naive_neighbor_list_dual_cutoff(
    ...     positions, cutoff1, cutoff2,
    ...     max_neighbors1=max_neighbors1, max_neighbors2=max_neighbors2,
    ...     neighbor_matrix1=neighbor_matrix1, neighbor_matrix2=neighbor_matrix2,
    ...     num_neighbors1=num_neighbors1, num_neighbors2=num_neighbors2
    ... )
    >>> print(f"Short-range pairs: {num_neighbors1.sum()}")
    >>> print(f"Long-range pairs: {num_neighbors2.sum()}")

    Preallocate tensors for pbc systems:
    >>> shift_range_per_dimension, shift_offset, total_shifts = compute_naive_num_shifts(
    ...     cell, cutoff1, pbc
    ... )
    >>> naive_neighbor_list_dual_cutoff(
    ...     positions, cutoff1, cutoff2,
    ...     max_neighbors1=max_neighbors1, max_neighbors2=max_neighbors2,
    ...     shift_range_per_dimension=shift_range_per_dimension, shift_offset=shift_offset, total_shifts=total_shifts
    ...     neighbor_matrix1=neighbor_matrix1, neighbor_matrix2=neighbor_matrix2,
    ...     num_neighbors1=num_neighbors1, num_neighbors2=num_neighbors2
    ... )
    >>> print(f"Short-range pairs: {num_neighbors1.sum()}")
    >>> print(f"Long-range pairs: {num_neighbors2.sum()}")

    See Also
    --------
    naive_neighbor_list : Single cutoff version
    batch_neighbor_list_dual_cutoff : Batch version for multiple systems
    """
    if pbc is None and cell is not None:
        raise ValueError("If cell is provided, pbc must also be provided")
    if pbc is not None and cell is None:
        raise ValueError("If pbc is provided, cell must also be provided")

    if cell is not None:
        cell = cell if cell.ndim == 3 else cell.unsqueeze(0)
    if pbc is not None:
        pbc = pbc if pbc.ndim == 2 else pbc.unsqueeze(0)

    if fill_value is None:
        fill_value = positions.shape[0]

    if max_neighbors1 is None and (
        neighbor_matrix1 is None
        or neighbor_matrix2 is None
        or (neighbor_matrix_shifts1 is None and pbc is not None)
        or (neighbor_matrix_shifts2 is None and pbc is not None)
        or num_neighbors1 is None
        or num_neighbors2 is None
    ):
        max_neighbors2 = estimate_max_neighbors(cutoff2)
        max_neighbors1 = max_neighbors2

    if max_neighbors2 is None:
        max_neighbors2 = max_neighbors1

    if neighbor_matrix1 is None:
        neighbor_matrix1 = torch.full(
            (positions.shape[0], max_neighbors1),
            fill_value,
            dtype=torch.int32,
            device=positions.device,
        )
    else:
        neighbor_matrix1.fill_(fill_value)

    if num_neighbors1 is None:
        num_neighbors1 = torch.zeros(
            positions.shape[0], dtype=torch.int32, device=positions.device
        )
    else:
        num_neighbors1.zero_()
    if neighbor_matrix2 is None:
        neighbor_matrix2 = torch.full(
            (positions.shape[0], max_neighbors2),
            fill_value,
            dtype=torch.int32,
            device=positions.device,
        )
    else:
        neighbor_matrix2.fill_(fill_value)

    if num_neighbors2 is None:
        num_neighbors2 = torch.zeros(
            positions.shape[0], dtype=torch.int32, device=positions.device
        )
    else:
        num_neighbors2.zero_()

    if pbc is not None:
        if neighbor_matrix_shifts1 is None:
            neighbor_matrix_shifts1 = torch.zeros(
                (positions.shape[0], max_neighbors1, 3),
                dtype=torch.int32,
                device=positions.device,
            )
        else:
            neighbor_matrix_shifts1.zero_()
        if neighbor_matrix_shifts2 is None:
            neighbor_matrix_shifts2 = torch.zeros(
                (positions.shape[0], max_neighbors2, 3),
                dtype=torch.int32,
                device=positions.device,
            )
        else:
            neighbor_matrix_shifts2.zero_()
        if (
            total_shifts is None
            or shift_offset is None
            or shift_range_per_dimension is None
        ):
            shift_range_per_dimension, shift_offset, total_shifts = (
                compute_naive_num_shifts(cell, cutoff2, pbc)
            )

    if pbc is None:
        _naive_neighbor_matrix_no_pbc_dual_cutoff(
            positions=positions,
            cutoff1=cutoff1,
            cutoff2=cutoff2,
            neighbor_matrix1=neighbor_matrix1,
            num_neighbors1=num_neighbors1,
            neighbor_matrix2=neighbor_matrix2,
            num_neighbors2=num_neighbors2,
            half_fill=half_fill,
        )
        if return_neighbor_list:
            neighbor_list1, neighbor_ptr1 = get_neighbor_list_from_neighbor_matrix(
                neighbor_matrix1, num_neighbors=num_neighbors1, fill_value=fill_value
            )
            neighbor_list2, neighbor_ptr2 = get_neighbor_list_from_neighbor_matrix(
                neighbor_matrix2, num_neighbors=num_neighbors2, fill_value=fill_value
            )
            return (
                neighbor_list1,
                neighbor_ptr1,
                neighbor_list2,
                neighbor_ptr2,
            )
        else:
            return (
                neighbor_matrix1,
                num_neighbors1,
                neighbor_matrix2,
                num_neighbors2,
            )
    else:
        _naive_neighbor_matrix_pbc_dual_cutoff(
            positions=positions,
            cutoff1=cutoff1,
            cutoff2=cutoff2,
            cell=cell,
            neighbor_matrix1=neighbor_matrix1,
            neighbor_matrix2=neighbor_matrix2,
            neighbor_matrix_shifts1=neighbor_matrix_shifts1,
            neighbor_matrix_shifts2=neighbor_matrix_shifts2,
            num_neighbors1=num_neighbors1,
            num_neighbors2=num_neighbors2,
            shift_range_per_dimension=shift_range_per_dimension,
            shift_offset=shift_offset,
            total_shifts=total_shifts,
            half_fill=half_fill,
        )
        if return_neighbor_list:
            neighbor_list1, neighbor_ptr1, unit_shifts1 = (
                get_neighbor_list_from_neighbor_matrix(
                    neighbor_matrix1,
                    num_neighbors=num_neighbors1,
                    neighbor_shift_matrix=neighbor_matrix_shifts1,
                    fill_value=fill_value,
                )
            )
            neighbor_list2, neighbor_ptr2, unit_shifts2 = (
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
                unit_shifts1,
                neighbor_list2,
                neighbor_ptr2,
                unit_shifts2,
            )
        else:
            return (
                neighbor_matrix1,
                num_neighbors1,
                neighbor_matrix_shifts1,
                neighbor_matrix2,
                num_neighbors2,
                neighbor_matrix_shifts2,
            )
