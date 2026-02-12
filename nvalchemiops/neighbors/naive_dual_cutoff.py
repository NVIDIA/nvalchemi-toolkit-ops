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

"""Core warp kernels and launchers for naive dual cutoff neighbor list construction.

This module contains warp kernels for O(N²) neighbor list computation with two cutoffs.
See `nvalchemiops.torch.neighbors` for PyTorch bindings.
"""

from typing import Any

import warp as wp

from nvalchemiops.neighbors.neighbor_utils import (
    _update_neighbor_matrix,
    _update_neighbor_matrix_pbc,
)

__all__ = [
    "naive_neighbor_matrix_dual_cutoff",
    "naive_neighbor_matrix_pbc_dual_cutoff",
]

###########################################################################################
########################### Naive Dual Cutoff Kernels #####################################
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

###########################################################################################
########################### Warp Launchers ###############################################
###########################################################################################


def naive_neighbor_matrix_dual_cutoff(
    positions: wp.array,
    cutoff1: float,
    cutoff2: float,
    neighbor_matrix1: wp.array,
    num_neighbors1: wp.array,
    neighbor_matrix2: wp.array,
    num_neighbors2: wp.array,
    wp_dtype: type,
    device: str,
    half_fill: bool = False,
) -> None:
    """Core warp launcher for naive dual cutoff neighbor matrix construction (no PBC).

    Computes pairwise distances and fills two neighbor matrices with atom indices
    within different cutoff distances using pure warp operations. No periodic boundary
    conditions are applied.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Atomic coordinates in Cartesian space.
    cutoff1 : float
        First cutoff distance (typically smaller).
    cutoff2 : float
        Second cutoff distance (typically larger).
    neighbor_matrix1 : wp.array, shape (total_atoms, max_neighbors1), dtype=wp.int32
        OUTPUT: First neighbor matrix to be filled.
    num_neighbors1 : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Number of neighbors found for each atom within cutoff1.
    neighbor_matrix2 : wp.array, shape (total_atoms, max_neighbors2), dtype=wp.int32
        OUTPUT: Second neighbor matrix to be filled.
    num_neighbors2 : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Number of neighbors found for each atom within cutoff2.
    wp_dtype : type
        Warp dtype (wp.float32, wp.float64, or wp.float16).
    device : str
        Warp device string (e.g., 'cuda:0', 'cpu').
    half_fill : bool, default=False
        If True, only store relationships where i < j to avoid double counting.

    Notes
    -----
    - This is a low-level warp interface. For framework bindings, use torch/jax wrappers.
    - Output arrays must be pre-allocated by caller.

    See Also
    --------
    naive_neighbor_matrix_pbc_dual_cutoff : Version with periodic boundary conditions
    _fill_naive_neighbor_matrix_dual_cutoff : Kernel that performs the computation
    """
    total_atoms = positions.shape[0]

    wp.launch(
        kernel=_fill_naive_neighbor_matrix_dual_cutoff_overload[wp_dtype],
        dim=total_atoms,
        inputs=[
            positions,
            wp_dtype(cutoff1 * cutoff1),
            wp_dtype(cutoff2 * cutoff2),
            neighbor_matrix1,
            num_neighbors1,
            neighbor_matrix2,
            num_neighbors2,
            half_fill,
        ],
        device=device,
    )


def naive_neighbor_matrix_pbc_dual_cutoff(
    positions: wp.array,
    cutoff1: float,
    cutoff2: float,
    cell: wp.array,
    shifts: wp.array,
    neighbor_matrix1: wp.array,
    neighbor_matrix2: wp.array,
    neighbor_matrix_shifts1: wp.array,
    neighbor_matrix_shifts2: wp.array,
    num_neighbors1: wp.array,
    num_neighbors2: wp.array,
    wp_dtype: type,
    device: str,
    half_fill: bool = False,
) -> None:
    """Core warp launcher for naive dual cutoff neighbor matrix construction with PBC.

    Computes neighbor relationships between atoms across periodic boundaries for
    two different cutoff distances using pure warp operations. Assumes shift vectors
    have been pre-computed.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Atomic coordinates in Cartesian space.
    cutoff1 : float
        First cutoff distance (typically smaller).
    cutoff2 : float
        Second cutoff distance (typically larger).
    cell : wp.array, shape (1, 3, 3), dtype=wp.mat33*
        Cell matrix defining lattice vectors in Cartesian coordinates.
    shifts : wp.array, shape (total_shifts, 3), dtype=wp.vec3i
        Integer shift vectors for periodic images.
    neighbor_matrix1 : wp.array, shape (total_atoms, max_neighbors1), dtype=wp.int32
        OUTPUT: First neighbor matrix to be filled.
    neighbor_matrix2 : wp.array, shape (total_atoms, max_neighbors2), dtype=wp.int32
        OUTPUT: Second neighbor matrix to be filled.
    neighbor_matrix_shifts1 : wp.array, shape (total_atoms, max_neighbors1, 3), dtype=wp.vec3i
        OUTPUT: Shift vectors for first neighbor matrix.
    neighbor_matrix_shifts2 : wp.array, shape (total_atoms, max_neighbors2, 3), dtype=wp.vec3i
        OUTPUT: Shift vectors for second neighbor matrix.
    num_neighbors1 : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Number of neighbors found for each atom within cutoff1.
    num_neighbors2 : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Number of neighbors found for each atom within cutoff2.
    wp_dtype : type
        Warp dtype (wp.float32, wp.float64, or wp.float16).
    device : str
        Warp device string (e.g., 'cuda:0', 'cpu').
    half_fill : bool, default=False
        If True, only store relationships where i < j to avoid double counting.

    Notes
    -----
    - This is a low-level warp interface. For framework bindings, use torch/jax wrappers.
    - Output arrays must be pre-allocated by caller.
    - Shift vectors must be pre-computed using compute_naive_num_shifts and _expand_naive_shifts.

    See Also
    --------
    naive_neighbor_matrix_dual_cutoff : Version without periodic boundary conditions
    _fill_naive_neighbor_matrix_pbc_dual_cutoff : Kernel that performs the computation
    """
    total_atoms = positions.shape[0]
    total_shifts = shifts.shape[0]

    wp.launch(
        kernel=_fill_naive_neighbor_matrix_pbc_dual_cutoff_overload[wp_dtype],
        dim=(total_shifts, total_atoms),
        inputs=[
            positions,
            wp_dtype(cutoff1 * cutoff1),
            wp_dtype(cutoff2 * cutoff2),
            cell,
            shifts,
            neighbor_matrix1,
            neighbor_matrix2,
            neighbor_matrix_shifts1,
            neighbor_matrix_shifts2,
            num_neighbors1,
            num_neighbors2,
            half_fill,
        ],
        device=device,
    )
