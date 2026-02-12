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

"""Core warp kernels and launchers for batched naive dual cutoff neighbor list construction.

This module contains warp kernels for batched O(N²) neighbor list computation with two cutoffs.
See `nvalchemiops.torch.neighbors` for PyTorch bindings.
"""

from typing import Any

import warp as wp

from nvalchemiops.neighbors.neighbor_utils import (
    _update_neighbor_matrix,
    _update_neighbor_matrix_pbc,
)

__all__ = [
    "batch_naive_neighbor_matrix_dual_cutoff",
    "batch_naive_neighbor_matrix_pbc_dual_cutoff",
]

###########################################################################################
########################### Batch Naive Dual Cutoff Kernels ###############################
###########################################################################################


@wp.kernel(enable_backward=False)
def _fill_batch_naive_neighbor_matrix_dual_cutoff(
    positions: wp.array(dtype=Any),
    cutoff1_sq: Any,
    cutoff2_sq: Any,
    batch_idx: wp.array(dtype=wp.int32),
    batch_ptr: wp.array(dtype=wp.int32),
    neighbor_matrix1: wp.array2d(dtype=wp.int32, ndim=2),
    num_neighbors1: wp.array(dtype=wp.int32),
    neighbor_matrix2: wp.array2d(dtype=wp.int32, ndim=2),
    num_neighbors2: wp.array(dtype=wp.int32),
    half_fill: wp.bool,
) -> None:
    """Calculate two neighbor matrices using dual cutoffs with naive O(N^2) algorithm.

    Computes pairwise distances between atoms within each system in a batch
    and identifies neighbors within two different cutoff distances simultaneously.
    This is more efficient than running two separate neighbor calculations when
    both neighbor lists are needed. Atoms from different systems do not interact.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Concatenated atomic coordinates for all systems in Cartesian space.
        Each row represents one atom's (x, y, z) position.
    cutoff1_sq : float
        Squared short-range cutoff distance in Cartesian units.
        Atoms within this distance are considered neighbors.
    cutoff2_sq : float
        Squared long-range cutoff distance in Cartesian units.
        Must be larger than cutoff1_sq. Atoms within this distance are considered neighbors.
    batch_idx : wp.array, shape (total_atoms,), dtype=wp.int32
        System index for each atom. Atoms with the same index belong to
        the same system and can be neighbors.
    batch_ptr : wp.array, shape (num_systems + 1,), dtype=wp.int32
        Cumulative atom counts defining system boundaries.
        System i contains atoms from batch_ptr[i] to batch_ptr[i+1]-1.
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

    See Also
    --------
    _fill_naive_neighbor_matrix_dual_cutoff : Single system version
    _fill_batch_naive_neighbor_matrix_pbc_dual_cutoff : Version with periodic boundaries
    """
    tid = wp.tid()
    i = tid
    isys = batch_idx[i]
    j_end = batch_ptr[isys + 1]

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
def _fill_batch_naive_neighbor_matrix_pbc_dual_cutoff(
    positions: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    cutoff1_sq: Any,
    cutoff2_sq: Any,
    batch_ptr: wp.array(dtype=wp.int32),
    shifts: wp.array(dtype=wp.vec3i),
    shift_system_idx: wp.array(dtype=wp.int32),
    neighbor_matrix1: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix2: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts1: wp.array(dtype=wp.vec3i, ndim=2),
    neighbor_matrix_shifts2: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors1: wp.array(dtype=wp.int32),
    num_neighbors2: wp.array(dtype=wp.int32),
    half_fill: wp.bool,
) -> None:
    """Calculate two neighbor matrices with periodic boundary conditions using naive O(N^2) algorithm.

    Computes neighbor relationships between atoms across periodic boundaries by
    considering all periodic images within the cutoff distance. Uses a 2D launch
    pattern to parallelize over both atoms and periodic shifts.

    This function operates on a batch of systems.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Concatenated atomic coordinates in Cartesian space.
        Must be wrapped into the unit cell.
    cell : wp.array, shape (num_systems, 3, 3), dtype=wp.mat33*
        Array of cell matrices for each system in the batch. Each matrix
        defines the lattice vectors in Cartesian coordinates.
    cutoff1_sq : float
        Squared short-range cutoff distance in Cartesian units.
        Atoms within this distance are considered neighbors.
    cutoff2_sq : float
        Squared long-range cutoff distance in Cartesian units.
        Must be larger than cutoff1_sq. Atoms within this distance are considered neighbors.
    batch_ptr : wp.array, shape (num_systems + 1,), dtype=wp.int32
        Cumulative sum of number of atoms per system in the batch.
    shifts : wp.array, shape (total_shifts, 3), dtype=wp.vec3i
        Array of integer shift vectors for periodic images.
    shift_system_idx : wp.array, shape (total_shifts,), dtype=wp.int32
        Array mapping each shift to its system index in the batch.
    neighbor_matrix1 : wp.array, shape (total_atoms, max_neighbors1), dtype=wp.int32
        OUTPUT: First neighbor matrix to be filled with neighbor atom indices.
    neighbor_matrix2 : wp.array, shape (total_atoms, max_neighbors2), dtype=wp.int32
        OUTPUT: Second neighbor matrix to be filled with neighbor atom indices.
    neighbor_matrix_shifts1 : wp.array, shape (total_atoms, max_neighbors1), dtype=wp.vec3i
        OUTPUT: Matrix storing shift vectors for each neighbor relationship.
    neighbor_matrix_shifts2 : wp.array, shape (total_atoms, max_neighbors2), dtype=wp.vec3i
        OUTPUT: Matrix storing shift vectors for each neighbor relationship.
    num_neighbors1 : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Array storing the number of neighbors for each atom.
    num_neighbors2 : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Array storing the number of neighbors for each atom.
    half_fill : wp.bool
        If True, only store half of the neighbor relationships (i < j).

    See Also
    --------
    _fill_batch_naive_neighbor_matrix_dual_cutoff : Version without periodic boundaries
    _fill_naive_neighbor_matrix_pbc_dual_cutoff : Single system version
    """
    ishift, iatom = wp.tid()
    isys = shift_system_idx[ishift]

    _natom = batch_ptr[isys + 1] - batch_ptr[isys]

    if iatom >= _natom:
        return

    start = batch_ptr[isys]
    iatom = iatom + start
    jatom_start = start
    jatom_end = batch_ptr[isys + 1]

    maxnb1 = neighbor_matrix1.shape[1]
    maxnb2 = neighbor_matrix2.shape[1]

    # Get the atom coordinates and shift vector
    _positions = positions[iatom]
    _cell = cell[isys]
    _shift = shifts[ishift]

    positions_shifted = type(_cell[0])(_shift) * _cell + _positions

    _zero_shift = _shift[0] == 0 and _shift[1] == 0 and _shift[2] == 0
    if _zero_shift:
        jatom_end = iatom

    for jatom in range(jatom_start, jatom_end):
        diff = positions_shifted - positions[jatom]
        dist_sq = wp.length_sq(diff)
        if dist_sq < cutoff2_sq:
            # Since we only generate half the shifts (lexicographically ordered),
            # we need to add both directions for non-zero shifts to get all pairs.
            # For zero shift, we already only process i < j, so use half_fill as-is.
            # For non-zero shifts, always add reciprocal (unless user wants half_fill).
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


T = [wp.float32, wp.float64, wp.float16]
V = [wp.vec3f, wp.vec3d, wp.vec3h]
M = [wp.mat33f, wp.mat33d, wp.mat33h]
_fill_batch_naive_neighbor_matrix_dual_cutoff_overload = {}
_fill_batch_naive_neighbor_matrix_pbc_dual_cutoff_overload = {}
for t, v, m in zip(T, V, M):
    _fill_batch_naive_neighbor_matrix_dual_cutoff_overload[t] = wp.overload(
        _fill_batch_naive_neighbor_matrix_dual_cutoff,
        [
            wp.array(dtype=v),
            t,
            t,
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32, ndim=2),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32, ndim=2),
            wp.array(dtype=wp.int32),
            wp.bool,
        ],
    )
    _fill_batch_naive_neighbor_matrix_pbc_dual_cutoff_overload[t] = wp.overload(
        _fill_batch_naive_neighbor_matrix_pbc_dual_cutoff,
        [
            wp.array(dtype=v),
            wp.array(dtype=m),
            t,
            t,
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.vec3i),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32, ndim=2),
            wp.array(dtype=wp.int32, ndim=2),
            wp.array(dtype=wp.vec3i, ndim=2),
            wp.array(dtype=wp.vec3i, ndim=2),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32),
            wp.bool,
        ],
    )

###########################################################################################
########################### Warp Launchers ###############################################
###########################################################################################


def batch_naive_neighbor_matrix_dual_cutoff(
    positions: wp.array,
    cutoff1: float,
    cutoff2: float,
    batch_idx: wp.array,
    batch_ptr: wp.array,
    neighbor_matrix1: wp.array,
    num_neighbors1: wp.array,
    neighbor_matrix2: wp.array,
    num_neighbors2: wp.array,
    wp_dtype: type,
    device: str,
    half_fill: bool = False,
) -> None:
    """Core warp launcher for batched naive dual cutoff neighbor matrix construction (no PBC).

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Concatenated atomic coordinates for all systems.
    cutoff1 : float
        First cutoff distance (typically smaller).
    cutoff2 : float
        Second cutoff distance (typically larger).
    batch_idx : wp.array, shape (total_atoms,), dtype=wp.int32
        System index for each atom.
    batch_ptr : wp.array, shape (num_systems + 1,), dtype=wp.int32
        Cumulative atom counts defining system boundaries.
    neighbor_matrix1 : wp.array, shape (total_atoms, max_neighbors1), dtype=wp.int32
        OUTPUT: First neighbor matrix.
    num_neighbors1 : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Neighbor counts for cutoff1.
    neighbor_matrix2 : wp.array, shape (total_atoms, max_neighbors2), dtype=wp.int32
        OUTPUT: Second neighbor matrix.
    num_neighbors2 : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Neighbor counts for cutoff2.
    wp_dtype : type
        Warp dtype (wp.float32, wp.float64, or wp.float16).
    device : str
        Warp device string (e.g., 'cuda:0', 'cpu').
    half_fill : bool, default=False
        If True, only store relationships where i < j.

    See Also
    --------
    batch_naive_neighbor_matrix_pbc_dual_cutoff : Version with PBC
    _fill_batch_naive_neighbor_matrix_dual_cutoff : Kernel that performs computation
    """
    total_atoms = positions.shape[0]

    wp.launch(
        kernel=_fill_batch_naive_neighbor_matrix_dual_cutoff_overload[wp_dtype],
        dim=total_atoms,
        inputs=[
            positions,
            wp_dtype(cutoff1 * cutoff1),
            wp_dtype(cutoff2 * cutoff2),
            batch_idx,
            batch_ptr,
            neighbor_matrix1,
            num_neighbors1,
            neighbor_matrix2,
            num_neighbors2,
            half_fill,
        ],
        device=device,
    )


def batch_naive_neighbor_matrix_pbc_dual_cutoff(
    positions: wp.array,
    cell: wp.array,
    cutoff1: float,
    cutoff2: float,
    batch_ptr: wp.array,
    shifts: wp.array,
    shift_system_idx: wp.array,
    neighbor_matrix1: wp.array,
    neighbor_matrix2: wp.array,
    neighbor_matrix_shifts1: wp.array,
    neighbor_matrix_shifts2: wp.array,
    num_neighbors1: wp.array,
    num_neighbors2: wp.array,
    wp_dtype: type,
    device: str,
    max_atoms_per_system: int,
    half_fill: bool = False,
) -> None:
    """Core warp launcher for batched naive dual cutoff neighbor matrix construction with PBC.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Concatenated atomic coordinates for all systems.
    cell : wp.array, shape (num_systems, 3, 3), dtype=wp.mat33*
        Cell matrices for each system in the batch.
    cutoff1 : float
        First cutoff distance (typically smaller).
    cutoff2 : float
        Second cutoff distance (typically larger).
    batch_ptr : wp.array, shape (num_systems + 1,), dtype=wp.int32
        Cumulative atom counts defining system boundaries.
    shifts : wp.array, shape (total_shifts, 3), dtype=wp.vec3i
        Integer shift vectors for periodic images.
    shift_system_idx : wp.array, shape (total_shifts,), dtype=wp.int32
        System index for each shift.
    neighbor_matrix1 : wp.array, shape (total_atoms, max_neighbors1), dtype=wp.int32
        OUTPUT: First neighbor matrix.
    neighbor_matrix2 : wp.array, shape (total_atoms, max_neighbors2), dtype=wp.int32
        OUTPUT: Second neighbor matrix.
    neighbor_matrix_shifts1 : wp.array, shape (total_atoms, max_neighbors1, 3), dtype=wp.vec3i
        OUTPUT: Shift vectors for first neighbor matrix.
    neighbor_matrix_shifts2 : wp.array, shape (total_atoms, max_neighbors2, 3), dtype=wp.vec3i
        OUTPUT: Shift vectors for second neighbor matrix.
    num_neighbors1 : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Neighbor counts for cutoff1.
    num_neighbors2 : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Neighbor counts for cutoff2.
    wp_dtype : type
        Warp dtype (wp.float32, wp.float64, or wp.float16).
    device : str
        Warp device string (e.g., 'cuda:0', 'cpu').
    max_atoms_per_system : int
        Maximum number of atoms in any single system.
    half_fill : bool, default=False
        If True, only store half of the neighbor relationships.

    See Also
    --------
    batch_naive_neighbor_matrix_dual_cutoff : Version without PBC
    _fill_batch_naive_neighbor_matrix_pbc_dual_cutoff : Kernel that performs computation
    """
    total_shifts = shifts.shape[0]

    wp.launch(
        kernel=_fill_batch_naive_neighbor_matrix_pbc_dual_cutoff_overload[wp_dtype],
        dim=(total_shifts, max_atoms_per_system),
        inputs=[
            positions,
            cell,
            wp_dtype(cutoff1 * cutoff1),
            wp_dtype(cutoff2 * cutoff2),
            batch_ptr,
            shifts,
            shift_system_idx,
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
