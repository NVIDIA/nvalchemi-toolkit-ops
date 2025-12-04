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
    _prepare_batch_idx_ptr,
    _update_neighbor_matrix,
    _update_neighbor_matrix_pbc,
    compute_naive_num_shifts,
    get_neighbor_list_from_neighbor_matrix,
)
from nvalchemiops.types import get_wp_dtype, get_wp_mat_dtype, get_wp_vec_dtype

###########################################################################################
########################### Naive Neighbor List Kernels ###################################
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

    Notes
    -----
    - Thread launch: 2D launch over (total_shifts, total_atoms)
    - Modifies: neighbor_matrix1, neighbor_matrix_shifts1, num_neighbors1, neighbor_matrix2, neighbor_matrix_shifts2, num_neighbors2
    - Maximum neighbors per atom limited by nbmat.shape[1]
    - If maximum exceeded, excess neighbors are ignored
    - Applies periodic boundary conditions using provided shift vectors
    - Uses atomic operations for thread-safe neighbor list construction
    - Handles minimum image convention automatically
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
################################################################################
########################### Dual cutoff ########################################
################################################################################


@torch.library.custom_op(
    "nvalchemiops::_batch_naive_neighbor_matrix_no_pbc_dual_cutoff",
    mutates_args=(
        "neighbor_matrix1",
        "num_neighbors1",
        "neighbor_matrix2",
        "num_neighbors2",
    ),
)
def _batch_naive_neighbor_matrix_no_pbc_dual_cutoff(
    positions: torch.Tensor,
    cutoff1: float,
    cutoff2: float,
    batch_idx: torch.Tensor,
    batch_ptr: torch.Tensor,
    neighbor_matrix1: torch.Tensor,
    num_neighbors1: torch.Tensor,
    neighbor_matrix2: torch.Tensor,
    num_neighbors2: torch.Tensor,
    half_fill: bool,
) -> None:
    """Fill two neighbor matrices for batch of atoms using dual cutoffs with naive O(N^2) algorithm.

    Custom PyTorch operator that computes pairwise distances and fills
    two neighbor matrices with atom indices within different cutoff distances
    simultaneously. Processes multiple systems in a batch where atoms from
    different systems do not interact. This is more efficient than running
    two separate batch neighbor calculations. No periodic boundary conditions are applied.

    This function does not allocate any tensors.

    This function is torch compilable.

    Parameters
    ----------
    positions : torch.Tensor, shape (total_atoms, 3), dtype=torch.float32 or torch.float64
        Concatenated atomic coordinates for all systems in Cartesian space.
        Each row represents one atom's (x, y, z) position.
    cutoff1 : float
        First cutoff distance in Cartesian units (typically the smaller cutoff).
        Must be positive. Atoms within this distance are considered neighbors.
    cutoff2 : float
        Second cutoff distance in Cartesian units (typically the larger cutoff).
        Must be positive and should be >= cutoff1 for optimal performance.
    batch_idx : torch.Tensor, shape (total_atoms,), dtype=torch.int32
        System index for each atom. Atoms with the same index belong to
        the same system and can be neighbors.
    batch_ptr : torch.Tensor, shape (num_systems + 1,), dtype=torch.int32
        Cumulative atom counts defining system boundaries.
        System i contains atoms from batch_ptr[i] to batch_ptr[i+1]-1.
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
    batch_naive_neighbor_list_dual_cutoff : Higher-level wrapper function
    _naive_neighbor_matrix_no_pbc_dual_cutoff : Single system version
    """
    device = positions.device
    wp_dtype = get_wp_dtype(positions.dtype)
    wp_vec_dtype = get_wp_vec_dtype(positions.dtype)
    wp_positions = wp.from_torch(positions, dtype=wp_vec_dtype, return_ctype=True)
    wp_batch_idx = wp.from_torch(batch_idx, dtype=wp.int32, return_ctype=True)
    wp_batch_ptr = wp.from_torch(batch_ptr, dtype=wp.int32, return_ctype=True)
    wp_neighbor_matrix1 = wp.from_torch(
        neighbor_matrix1, dtype=wp.int32, return_ctype=True
    )
    wp_num_neighbors1 = wp.from_torch(num_neighbors1, dtype=wp.int32, return_ctype=True)
    wp_neighbor_matrix2 = wp.from_torch(
        neighbor_matrix2, dtype=wp.int32, return_ctype=True
    )
    wp_num_neighbors2 = wp.from_torch(num_neighbors2, dtype=wp.int32, return_ctype=True)
    wp.launch(
        kernel=_fill_batch_naive_neighbor_matrix_dual_cutoff_overload[wp_dtype],
        dim=positions.shape[0],
        inputs=[
            wp_positions,
            wp_dtype(cutoff1 * cutoff1),
            wp_dtype(cutoff2 * cutoff2),
            wp_batch_idx,
            wp_batch_ptr,
            wp_neighbor_matrix1,
            wp_num_neighbors1,
            wp_neighbor_matrix2,
            wp_num_neighbors2,
            half_fill,
        ],
        device=wp.device_from_torch(device),
    )


@torch.library.custom_op(
    "nvalchemiops::_batch_naive_neighbor_matrix_pbc_dual_cutoff",
    mutates_args=(
        "neighbor_matrix1",
        "neighbor_matrix2",
        "neighbor_matrix_shifts1",
        "neighbor_matrix_shifts2",
        "num_neighbors1",
        "num_neighbors2",
    ),
)
def _batch_naive_neighbor_matrix_pbc_dual_cutoff(
    positions: torch.Tensor,
    cell: torch.Tensor,
    cutoff1: float,
    cutoff2: float,
    batch_ptr: torch.Tensor,
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
    max_atoms_per_system: int | None = None,
) -> None:
    """Compute two batch neighbor matrices with periodic boundary conditions using dual cutoffs and naive O(N^2) algorithm.

    Custom PyTorch operator that computes neighbor relationships between atoms
    across periodic boundaries for two different cutoff distances simultaneously
    for multiple systems in a batch. Uses pre-computed shift vectors for
    efficiency. Each system can have different periodic cells
    and boundary conditions. This is more efficient than running two separate
    batch PBC neighbor calculations.

    This function does not allocate any tensors.

    This function is torch compilable.

    Parameters
    ----------
    positions : torch.Tensor, shape (total_atoms, 3), dtype=torch.float32 or torch.float64
        Concatenated atomic coordinates for all systems in Cartesian space.
        Each row represents one atom's (x, y, z) position.
        Must be wrapped into the unit cell.
    cell : torch.Tensor, shape (num_systems, 3, 3), dtype=torch.float32 or torch.float64
        Cell matrices defining lattice vectors in Cartesian coordinates.
        Each 3x3 matrix represents one system's periodic cell.
    cutoff1 : float
        First cutoff distance in Cartesian units (typically the smaller cutoff).
        Must be positive. Atoms within this distance are considered neighbors.
    cutoff2 : float
        Second cutoff distance in Cartesian units (typically the larger cutoff).
        Must be positive and should be >= cutoff1 for optimal performance.
    batch_ptr : torch.Tensor, shape (num_systems + 1,), dtype=torch.int32
        Cumulative atom counts defining system boundaries.
        System i contains atoms from batch_ptr[i] to batch_ptr[i+1]-1.
    neighbor_matrix1 : torch.Tensor, shape (total_atoms, max_neighbors1), dtype=torch.int32
        OUTPUT: First (short-range) neighbor matrix for cutoff1 to be filled with atom indices.
        Must be pre-allocated. Entries are filled with atom indices.
    neighbor_matrix2 : torch.Tensor, shape (total_atoms, max_neighbors2), dtype=torch.int32
        OUTPUT: Second (long-range) neighbor matrix for cutoff2 to be filled with atom indices.
        Must be pre-allocated. Entries are filled with atom indices.
    neighbor_matrix_shifts1 : torch.Tensor, shape (total_atoms, max_neighbors1, 3), dtype=torch.int32
        OUTPUT: Matrix storing shift vectors for each neighbor relationship in matrix1.
        Must be pre-allocated. Each entry corresponds to the shift used for the neighbor in neighbor_matrix1.
    neighbor_matrix_shifts2 : torch.Tensor, shape (total_atoms, max_neighbors2, 3), dtype=torch.int32
        OUTPUT: Matrix storing shift vectors for each neighbor relationship in matrix2.
        Must be pre-allocated. Each entry corresponds to the shift used for the neighbor in neighbor_matrix2.
    num_neighbors1 : torch.Tensor, shape (total_atoms,), dtype=torch.int32
        OUTPUT: Number of neighbors found for each atom within cutoff1.
        Must be pre-allocated. Updated in-place with actual neighbor counts.
    num_neighbors2 : torch.Tensor, shape (total_atoms,), dtype=torch.int32
        OUTPUT: Number of neighbors found for each atom within cutoff2.
        Must be pre-allocated. Updated in-place with actual neighbor counts.
    shift_range_per_dimension : torch.Tensor, shape (num_systems, 3), dtype=torch.int32
        Shift range in each dimension for each system for cutoff2.
    shift_offset : torch.Tensor, shape (num_systems + 1,), dtype=torch.int32
        Cumulative sum of shift counts for the cutoff2, defining shift boundaries.
        System i uses shifts from shift_offset[i] to shift_offset[i+1]-1.
    total_shifts : int
        Total number of periodic shifts for the cutoff2 across all systems.
        Must match the sum of shifts for cutoff2 across all systems.
    half_fill : bool, optional
        If True, only store relationships where i < j to avoid double counting.
        If False, store all neighbor relationships symmetrically (default).
    max_atoms_per_system : int, optional
        Maximum number of atoms per system.
        If not provided, it will be computed automaticaly.
        Can be provided to avoid CUDA synchronization.

    Returns
    -------
    None
        This function modifies the input tensors in-place:

        - neighbor_matrix1 : Filled with neighbor atom indices within cutoff1
        - neighbor_matrix2 : Filled with neighbor atom indices within cutoff2
        - neighbor_matrix_shifts1 : Filled with corresponding shift vectors for cutoff1
        - neighbor_matrix_shifts2 : Filled with corresponding shift vectors for cutoff2
        - num_neighbors1 : Updated with neighbor counts per atom for cutoff1
        - num_neighbors2 : Updated with neighbor counts per atom for cutoff2

    See Also
    --------
    batch_naive_neighbor_list_dual_cutoff : Higher-level wrapper function
    _batch_compute_total_shifts : Computes the required shift vectors
    _naive_neighbor_matrix_pbc_dual_cutoff : Single system version
    """
    num_systems = cell.shape[0]
    device = positions.device
    wp_device = wp.device_from_torch(device)
    wp_dtype = get_wp_dtype(positions.dtype)
    wp_vec_dtype = get_wp_vec_dtype(positions.dtype)
    wp_mat_dtype = get_wp_mat_dtype(positions.dtype)

    wp_positions = wp.from_torch(positions, dtype=wp_vec_dtype)
    wp_cell = wp.from_torch(cell, dtype=wp_mat_dtype)

    shifts = torch.empty((total_shifts, 3), dtype=torch.int32, device=device)
    shift_system_idx = torch.empty((total_shifts,), dtype=torch.int32, device=device)
    wp_shifts = wp.from_torch(shifts, dtype=wp.vec3i)
    wp_shift_system_idx = wp.from_torch(shift_system_idx, dtype=wp.int32)

    wp.launch(
        kernel=_expand_naive_shifts,
        dim=num_systems,
        inputs=[
            wp.from_torch(shift_range_per_dimension, dtype=wp.vec3i, return_ctype=True),
            wp.from_torch(shift_offset, dtype=wp.int32, return_ctype=True),
            wp_shifts,
            wp_shift_system_idx,
        ],
        device=wp_device,
    )
    wp_neighbor_matrix1 = wp.from_torch(
        neighbor_matrix1, dtype=wp.int32, return_ctype=True
    )
    wp_neighbor_matrix2 = wp.from_torch(
        neighbor_matrix2, dtype=wp.int32, return_ctype=True
    )
    wp_neighbor_matrix_shifts1 = wp.from_torch(
        neighbor_matrix_shifts1, dtype=wp.vec3i, return_ctype=True
    )
    wp_neighbor_matrix_shifts2 = wp.from_torch(
        neighbor_matrix_shifts2, dtype=wp.vec3i, return_ctype=True
    )
    wp_num_neighbors1 = wp.from_torch(num_neighbors1, dtype=wp.int32, return_ctype=True)
    wp_num_neighbors2 = wp.from_torch(num_neighbors2, dtype=wp.int32, return_ctype=True)
    wp_batch_ptr = wp.from_torch(batch_ptr, dtype=wp.int32, return_ctype=True)

    if max_atoms_per_system is None:
        max_atoms_per_system = (batch_ptr[1:] - batch_ptr[:-1]).max().item()

    wp.launch(
        kernel=_fill_batch_naive_neighbor_matrix_pbc_dual_cutoff_overload[wp_dtype],
        dim=(total_shifts, max_atoms_per_system),
        inputs=[
            wp_positions,
            wp_cell,
            wp_dtype(cutoff1 * cutoff1),
            wp_dtype(cutoff2 * cutoff2),
            wp_batch_ptr,
            wp_shifts,
            wp_shift_system_idx,
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


def batch_naive_neighbor_list_dual_cutoff(
    positions: torch.Tensor,
    cutoff1: float,
    cutoff2: float,
    batch_idx: torch.Tensor | None = None,
    batch_ptr: torch.Tensor | None = None,
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
    max_atoms_per_system: int | None = None,
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
    """Compute two batch neighbor matrices using dual cutoffs with naive O(N^2) algorithm.

    Identifies all atom pairs within two different cutoff distances for multiple
    systems processed in a batch. Each system is processed independently with
    dual cutoffs, supporting both non-periodic and periodic boundary conditions.
    This is more efficient than running batch neighbor calculations for each cutoff separately.

    For efficiency, this function supports in-place modification of the pre-allocated tensors.
    If not provided, the resulting tensors will be allocated.
    This function does not introduce CUDA graph breaks for non-PBC systems.
    For PBC systems, pre-compute unit shifts to avoid CUDA graph breaks:
    `shift_range_per_dimension, shift_offset, total_shifts = compute_naive_num_shifts(cell, cutoff2, pbc)`

    Parameters
    ----------
    positions : torch.Tensor, shape (total_atoms, 3), dtype=torch.float32 or torch.float64
        Concatenated atomic coordinates for all systems in Cartesian space.
        Each row represents one atom's (x, y, z) position.
        Must be wrapped into the unit cell if PBC is used.
    cutoff1 : float
        First (short range) cutoff distance in Cartesian units (typically the smaller cutoff).
        Must be positive. Atoms within this distance are considered neighbors.
    cutoff2 : float
        Second cutoff distance in Cartesian units (typically the larger cutoff).
        Must be positive and be >= cutoff1.
    batch_idx : torch.Tensor, shape (total_atoms,), dtype=torch.int32, optional
        System index for each atom. Atoms with the same index belong to
        the same system and can be neighbors. Must be in sorted order.
        If not provided, assumes all atoms belong to a single system.
    batch_ptr : torch.Tensor, shape (num_systems + 1,), dtype=torch.int32, optional
        Cumulative atom counts defining system boundaries.
        System i contains atoms from batch_ptr[i] to batch_ptr[i+1]-1.
        If not provided and batch_idx is provided, it will be computed automatically.
    pbc : torch.Tensor, shape (num_systems, 3), dtype=torch.bool, optional
        Periodic boundary condition flags for each dimension of each system.
        True enables periodicity in that direction. Default is None (no PBC).
    cell : torch.Tensor, shape (num_systems, 3, 3), dtype=torch.float32 or torch.float64, optional
        Cell matrices defining lattice vectors in Cartesian coordinates.
        Required if pbc is provided. Default is None.
    max_neighbors1 : int, optional
        Maximum number of neighbors per atom for the first neighbor matrix.
        Must be positive. If exceeded, excess neighbors are ignored.
        Must be provided if neighbor_matrix1 is not provided.
    max_neighbors2 : int, optional
        Maximum number of neighbors per atom for the second neighbor matrix.
        Must be positive. If exceeded, excess neighbors are ignored.
        Must be provided if neighbor_matrix2 is not provided.
    half_fill : bool, optional
        If True, only store half of the neighbor relationships to avoid double counting.
        Another half could be reconstructed by swapping source and target indices and inverting unit shifts.
        If False, store all neighbor relationships. Default is False.
    fill_value : int | None, optional
        Value to fill the neighbor matrices with. Default is total_atoms.
    return_neighbor_list : bool, optional - default = False
        If True, convert the neighbor matrix to a neighbor list (idx_i, idx_j) format by
        creating a mask over the fill_value, which can incur a performance penalty.
        We recommend using the neighbor matrix format,
        and only convert to a neighbor list format if absolutely necessary.
    neighbor_matrix1 : torch.Tensor, shape (total_atoms, max_neighbors1), dtype=torch.int32, optional
        Optional pre-allocated tensor for the first (short-range) neighbor matrix.
        Must be provided if max_neighbors1 is not provided.
        If provided, return_neighbor_list must be False.
    neighbor_matrix2 : torch.Tensor, shape (total_atoms, max_neighbors2), dtype=torch.int32, optional
        Optional pre-allocated tensor for the second (long-range) neighbor matrix.
        Must be provided if max_neighbors2 is not provided.
        If provided, return_neighbor_list must be False.
    neighbor_matrix_shifts1 : torch.Tensor, shape (total_atoms, max_neighbors1, 3), dtype=torch.int32, optional
        Optional pre-allocated tensor for the shift vectors of the first (short-range) neighbor matrix.
        Must be provided if max_neighbors1 is not provided and pbc is not None.
        If provided, return_neighbor_list must be False.
    neighbor_matrix_shifts2 : torch.Tensor, shape (total_atoms, max_neighbors2, 3), dtype=torch.int32, optional
        Optional pre-allocated tensor for the shift vectors of the second (long-range) neighbor matrix.
        Must be provided if max_neighbors2 is not provided and pbc is not None.
        If provided, return_neighbor_list must be False.
    num_neighbors1 : torch.Tensor, shape (total_atoms,), dtype=torch.int32, optional
        Optional pre-allocated tensor for the number of neighbors in the first (short-range) neighbor matrix.
        Must be provided if max_neighbors1 is not provided.
    num_neighbors2 : torch.Tensor, shape (total_atoms,), dtype=torch.int32, optional
        Optional pre-allocated tensor for the number of neighbors in the second (long-range) neighbor matrix.
        Must be provided if max_neighbors2 is not provided.
    shift_range_per_dimension : torch.Tensor, shape (num_systems, 3), dtype=torch.int32, optional
        Optional pre-allocated tensor for the shift range in each dimension for each system for cutoff2.
    shift_offset : torch.Tensor, shape (num_systems + 1,), dtype=torch.int32, optional
        Optional pre-allocated tensor for the cumulative sum of number of shifts for each system for cutoff2.
    total_shifts : int, optional
        Total number of shifts for cutoff2.
        Pass in to avoid reallocation for pbc systems.
    max_atoms_per_system : int, optional
        Maximum number of atoms per system.
        If not provided, it will be computed automaticaly.
        Can be provided to avoid CUDA synchronization.

    Returns
    -------
    results : tuple of torch.Tensor
        Variable-length tuple with interleaved results for cutoff1 and cutoff2. The return pattern follows:

        - No PBC, matrix format: ``(neighbor_matrix1, num_neighbors1, neighbor_matrix2, num_neighbors2)``
        - No PBC, list format: ``(neighbor_list1, neighbor_ptr1, neighbor_list2, neighbor_ptr2)``
        - With PBC, matrix format: ``(neighbor_matrix1, num_neighbors1, neighbor_matrix_shifts1, neighbor_matrix2, num_neighbors2, neighbor_matrix_shifts2)``
        - With PBC, list format: ``(neighbor_list1, neighbor_ptr1, shifts1, neighbor_list2, neighbor_ptr2, shifts2)``

        **Components returned (interleaved for each cutoff):**

        - **neighbor_data1, neighbor_data2** (tensors): Neighbor indices, format depends on ``return_neighbor_list``:

            - If ``return_neighbor_list=False`` (default): Returns ``neighbor_matrix1`` and ``neighbor_matrix2``
              with shapes (total_atoms, max_neighbors1) and (total_atoms, max_neighbors2), dtype int32.
              Each row i contains indices of atom i's neighbors within the respective cutoff.
            - If ``return_neighbor_list=True``: Returns ``neighbor_list1`` and ``neighbor_list2`` with shapes
              (2, num_pairs1) and (2, num_pairs2), dtype int32, in COO format [source_atoms, target_atoms].

        - **num_neighbor_data1, num_neighbor_data2** (tensor): Information about the number of neighbors for each atom,
          format depends on ``return_neighbor_list``:

            - If ``return_neighbor_list=False`` (default): Returns ``num_neighbors`` with shape (total_atoms,), dtype int32.
              Count of neighbors found for each atom.
            - If ``return_neighbor_list=True``: Returns ``neighbor_ptr`` with shape (total_atoms + 1,), dtype int32.
              CSR-style pointer arrays where ``neighbor_ptr_data[i]`` to ``neighbor_ptr_data[i+1]`` gives the range of
              neighbors for atom i in the flattened neighbor list.

        - **neighbor_shift_data1, neighbor_shift_data2** (tensor): Periodic shift vectors for each neighbor,
          format depends on ``return_neighbor_list`` and only returned when ``pbc`` is provided:

            - If ``return_neighbor_list=False`` (default): Returns ``neighbor_matrix_shifts1`` and ``neighbor_matrix_shifts2`` with
              shape (total_atoms, max_neighbors1, 3) and (total_atoms, max_neighbors2, 3), dtype int32.
            - If ``return_neighbor_list=True``: Returns ``neighbor_list_shifts1`` and ``neighbor_list_shifts2`` with shape
              (num_pairs1, 3) and (num_pairs2, 3), dtype int32.
    Examples
    --------
    Basic batch processing with dual cutoffs:

    >>> import torch
    >>> # Create batch with 2 systems: 50 and 30 atoms
    >>> coord1 = torch.rand(50, 3) * 5.0
    >>> coord2 = torch.rand(30, 3) * 8.0
    >>> positions = torch.cat([coord1, coord2], dim=0)
    >>> batch_idx = torch.cat([torch.zeros(50), torch.ones(30)]).int()
    >>> batch_ptr = torch.tensor([0, 50, 80], dtype=torch.int32)
    >>>
    >>> cutoff1, cutoff2 = 2.0, 4.0  # Short and long range
    >>> max_neighbors1, max_neighbors2 = 20, 50
    >>> (neighbor_matrix1, num_neighbors1, neighbor_matrix2, num_neighbors2) = (
    ...     batch_naive_neighbor_list_dual_cutoff(
    ...         positions, cutoff1, cutoff2, batch_idx, batch_ptr,
    ...         max_neighbors1=max_neighbors1, max_neighbors2=max_neighbors2
    ...     )
    ... )
    >>> # neighbor_matrix_shifts will be empty tensors for non-PBC systems

    With periodic boundary conditions:

    >>> # Different cells for each system
    >>> cell = torch.stack([
    ...     torch.eye(3) * 5.0,  # System 0: 5x5x5 cubic cell
    ...     torch.eye(3) * 8.0   # System 1: 8x8x8 cubic cell
    ... ])
    >>> pbc = torch.tensor([[True, True, True], [True, True, False]])
    >>> (neighbor_matrix1, num_neighbors1, neighbor_matrix_shifts1,
    ...  neighbor_matrix2, num_neighbors2, neighbor_matrix_shifts2) = (
    ...     batch_naive_neighbor_list_dual_cutoff(
    ...         positions, cutoff1, cutoff2, batch_idx, batch_ptr,
    ...         pbc=pbc, cell=cell,
    ...         max_neighbors1=max_neighbors1, max_neighbors2=max_neighbors2
    ...     )
    ... )

    See Also
    --------
    batch_naive_neighbor_list : Single cutoff version
    """
    if pbc is None and cell is not None:
        raise ValueError("If cell is provided, pbc must also be provided")
    if pbc is not None and cell is None:
        raise ValueError("If pbc is provided, cell must also be provided")

    if cell is not None:
        cell = cell if cell.ndim == 3 else cell.unsqueeze(0)
    if pbc is not None:
        pbc = pbc if pbc.ndim == 2 else pbc.unsqueeze(0)

    if max_neighbors1 is None and (
        neighbor_matrix1 is None
        or (neighbor_matrix_shifts1 is None and pbc is not None)
        or num_neighbors1 is None
    ):
        raise ValueError(
            "max_neighbors1 must be provided if neighbor_matrix1, neighbor_matrix_shifts1, or num_neighbors1 are not provided"
        )

    # Default max_neighbors2 to max_neighbors1 if not provided
    if max_neighbors2 is None:
        max_neighbors2 = max_neighbors1 * (cutoff2 / cutoff1) ** 3

    if max_neighbors2 is None and (
        neighbor_matrix2 is None
        or (neighbor_matrix_shifts2 is None and pbc is not None)
        or num_neighbors2 is None
    ):
        raise ValueError(
            "max_neighbors2 must be provided if neighbor_matrix2, neighbor_matrix_shifts2, or num_neighbors2 are not provided"
        )

    if fill_value is None:
        fill_value = positions.shape[0]

    # Allocate or zero neighbor_matrix1
    if neighbor_matrix1 is None:
        neighbor_matrix1 = torch.full(
            (positions.shape[0], max_neighbors1),
            fill_value,
            dtype=torch.int32,
            device=positions.device,
        )
    else:
        neighbor_matrix1.fill_(fill_value)

    # Allocate or zero neighbor_matrix2
    if neighbor_matrix2 is None:
        neighbor_matrix2 = torch.full(
            (positions.shape[0], max_neighbors2),
            fill_value,
            dtype=torch.int32,
            device=positions.device,
        )
    else:
        neighbor_matrix2.fill_(fill_value)

    # Allocate or zero num_neighbors1
    if num_neighbors1 is None:
        num_neighbors1 = torch.zeros(
            positions.shape[0], dtype=torch.int32, device=positions.device
        )
    else:
        num_neighbors1.zero_()

    # Allocate or zero num_neighbors2
    if num_neighbors2 is None:
        num_neighbors2 = torch.zeros(
            positions.shape[0], dtype=torch.int32, device=positions.device
        )
    else:
        num_neighbors2.zero_()

    if pbc is not None:
        # Allocate or zero neighbor_matrix_shifts1
        if neighbor_matrix_shifts1 is None:
            neighbor_matrix_shifts1 = torch.zeros(
                (positions.shape[0], max_neighbors1, 3),
                dtype=torch.int32,
                device=positions.device,
            )
        else:
            neighbor_matrix_shifts1.zero_()

        # Allocate or zero neighbor_matrix_shifts2
        if neighbor_matrix_shifts2 is None:
            neighbor_matrix_shifts2 = torch.zeros(
                (positions.shape[0], max_neighbors2, 3),
                dtype=torch.int32,
                device=positions.device,
            )
        else:
            neighbor_matrix_shifts2.zero_()

        # Compute shifts for cutoff1 if not provided
        if (
            total_shifts is None
            or shift_offset is None
            or shift_range_per_dimension is None
        ):
            shift_range_per_dimension, shift_offset, total_shifts = (
                compute_naive_num_shifts(cell, cutoff2, pbc)
            )
            total_shifts = shift_offset[-1].item()

    # check batch_idx and batch_ptr
    batch_idx, batch_ptr = _prepare_batch_idx_ptr(
        batch_idx=batch_idx,
        batch_ptr=batch_ptr,
        num_atoms=positions.shape[0],
        device=positions.device,
    )

    if pbc is None:
        _batch_naive_neighbor_matrix_no_pbc_dual_cutoff(
            positions=positions,
            cutoff1=cutoff1,
            cutoff2=cutoff2,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
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
            return neighbor_matrix1, num_neighbors1, neighbor_matrix2, num_neighbors2
    else:
        _batch_naive_neighbor_matrix_pbc_dual_cutoff(
            positions=positions,
            cell=cell,
            cutoff1=cutoff1,
            cutoff2=cutoff2,
            batch_ptr=batch_ptr,
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
            max_atoms_per_system=max_atoms_per_system,
        )
        if return_neighbor_list:
            neighbor_list1, neighbor_ptr1, neighbor_list_shifts1 = (
                get_neighbor_list_from_neighbor_matrix(
                    neighbor_matrix1,
                    num_neighbors=num_neighbors1,
                    fill_value=fill_value,
                    neighbor_shift_matrix=neighbor_matrix_shifts1,
                )
            )
            neighbor_list2, neighbor_ptr2, neighbor_list_shifts2 = (
                get_neighbor_list_from_neighbor_matrix(
                    neighbor_matrix2,
                    num_neighbors=num_neighbors2,
                    fill_value=fill_value,
                    neighbor_shift_matrix=neighbor_matrix_shifts2,
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
            return (
                neighbor_matrix1,
                num_neighbors1,
                neighbor_matrix_shifts1,
                neighbor_matrix2,
                num_neighbors2,
                neighbor_matrix_shifts2,
            )
