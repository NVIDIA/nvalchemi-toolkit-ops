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

"""Core warp kernels and launchers for batched naive neighbor list construction.

This module contains warp kernels for batched O(N²) neighbor list computation.
See `nvalchemiops.torch.neighbors` for PyTorch bindings.
"""

from typing import Any

import warp as wp

from nvalchemiops.neighbors.neighbor_utils import (
    _update_neighbor_matrix,
    _update_neighbor_matrix_pbc,
)

__all__ = [
    "batch_naive_neighbor_matrix",
    "batch_naive_neighbor_matrix_pbc",
]

###########################################################################################
########################### Batch Naive Neighbor List Kernels ############################
###########################################################################################


@wp.kernel(enable_backward=False)
def _fill_batch_naive_neighbor_matrix(
    positions: wp.array(dtype=Any),
    cutoff_sq: Any,
    batch_idx: wp.array(dtype=wp.int32),
    batch_ptr: wp.array(dtype=wp.int32),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    half_fill: wp.bool,
) -> None:
    """Calculate batch neighbor matrix using naive O(N^2) algorithm.

    Computes pairwise distances between atoms within each system in a batch
    and identifies neighbors within the specified cutoff distance. Atoms from
    different systems do not interact. No periodic boundary conditions are applied.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Concatenated atomic coordinates for all systems in Cartesian space.
        Each row represents one atom's (x, y, z) position.
    cutoff_sq : float
        Squared cutoff distance for neighbor detection in Cartesian units.
        Atoms within this distance are considered neighbors.
    batch_idx : wp.array, shape (total_atoms,), dtype=wp.int32
        System index for each atom. Atoms with the same index belong to
        the same system and can be neighbors.
    batch_ptr : wp.array, shape (num_systems + 1,), dtype=wp.int32
        Cumulative atom counts defining system boundaries.
        System i contains atoms from batch_ptr[i] to batch_ptr[i+1]-1.
    neighbor_matrix : wp.array, shape (total_atoms, max_neighbors), dtype=wp.int32
        OUTPUT: Neighbor matrix to be filled with neighbor atom indices.
        Entries are filled with atom indices, remaining entries stay as initialized.
    num_neighbors : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Number of neighbors found for each atom.
        Updated in-place with actual neighbor counts.
    half_fill : wp.bool
        If True, only store relationships where i < j to avoid double counting.
        If False, store all neighbor relationships symmetrically.

    Returns
    -------
    None
        This function modifies the input arrays in-place:

        - neighbor_matrix : Filled with neighbor atom indices
        - num_neighbors : Updated with neighbor counts per atom

    See Also
    --------
    _fill_naive_neighbor_matrix : Single system version
    _fill_batch_naive_neighbor_matrix_pbc : Version with periodic boundary conditions
    """
    tid = wp.tid()
    isys = batch_idx[tid]
    j_end = batch_ptr[isys + 1]

    positions_i = positions[tid]
    max_neighbors = neighbor_matrix.shape[1]
    for j in range(tid + 1, j_end):
        diff = positions_i - positions[j]
        dist_sq = wp.length_sq(diff)
        if dist_sq < cutoff_sq:
            _update_neighbor_matrix(
                tid, j, neighbor_matrix, num_neighbors, max_neighbors, half_fill
            )


@wp.kernel(enable_backward=False)
def _fill_batch_naive_neighbor_matrix_pbc(
    positions: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    cutoff_sq: Any,
    batch_ptr: wp.array(dtype=wp.int32),
    shifts: wp.array(dtype=wp.vec3i),
    shift_system_idx: wp.array(dtype=wp.int32),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    half_fill: wp.bool,
) -> None:
    """Calculate batch neighbor matrix with periodic boundary conditions using naive O(N^2) algorithm.

    Computes neighbor relationships between atoms across periodic boundaries by
    considering all periodic images within the cutoff distance. Processes multiple
    systems in a batch, where each system can have different periodic cells.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Concatenated atomic coordinates for all systems in Cartesian space.
        Each row represents one atom's (x, y, z) position.
    cell : wp.array, shape (num_systems, 3, 3), dtype=wp.mat33*
        Array of cell matrices for each system in the batch. Each matrix
        defines the lattice vectors in Cartesian coordinates.
    cutoff_sq : float
        Squared cutoff distance for neighbor detection in Cartesian units.
        Must be positive. Atoms within this distance are considered neighbors.
    batch_ptr : wp.array, shape (num_systems + 1,), dtype=wp.int32
        Cumulative sum of number of atoms per system in the batch.
        System i contains atoms from batch_ptr[i] to batch_ptr[i+1]-1.
    shifts : wp.array, shape (total_shifts, 3), dtype=wp.vec3i
        Array of integer shift vectors for periodic images.
    shift_system_idx : wp.array, shape (total_shifts,), dtype=wp.int32
        Array mapping each shift to its system index in the batch.
    neighbor_matrix : wp.array, shape (total_atoms, max_neighbors), dtype=wp.int32
        OUTPUT: Neighbor matrix to be filled with neighbor atom indices.
        Entries are filled with atom indices, remaining entries stay as initialized.
    neighbor_matrix_shifts : wp.array, shape (total_atoms, max_neighbors), dtype=wp.vec3i
        OUTPUT: Matrix storing shift vectors for each neighbor relationship.
        Each entry corresponds to the shift used for the neighbor in neighbor_matrix.
    num_neighbors : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Number of neighbors found for each atom.
        Updated in-place with actual neighbor counts.
    half_fill : wp.bool
        If True, only store half of the neighbor relationships.
        The other half can be reconstructed by swapping indices and inverting shifts.
        If False, store all neighbor relationships symmetrically.

    Returns
    -------
    None
        This function modifies the input arrays in-place:

        - neighbor_matrix : Filled with neighbor atom indices
        - neighbor_matrix_shifts : Filled with corresponding shift vectors
        - num_neighbors : Updated with neighbor counts per atom

    See Also
    --------
    _fill_batch_naive_neighbor_matrix : Version without periodic boundary conditions
    _fill_naive_neighbor_matrix_pbc : Single system version
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

    maxnb = neighbor_matrix.shape[1]
    _positions = positions[iatom]
    _shift = shifts[ishift]
    _cell = cell[isys]

    positions_shifted = type(_cell[0])(_shift) * _cell + _positions

    _zero_shift = _shift[0] == 0 and _shift[1] == 0 and _shift[2] == 0
    if _zero_shift:
        jatom_end = iatom
    for jatom in range(jatom_start, jatom_end):
        diff = positions_shifted - positions[jatom]
        dist_sq = wp.length_sq(diff)
        if dist_sq < cutoff_sq:
            _update_neighbor_matrix_pbc(
                jatom,
                iatom,
                neighbor_matrix,
                neighbor_matrix_shifts,
                num_neighbors,
                _shift,
                maxnb,
                half_fill,
            )


T = [wp.float32, wp.float64, wp.float16]
V = [wp.vec3f, wp.vec3d, wp.vec3h]
M = [wp.mat33f, wp.mat33d, wp.mat33h]
_fill_batch_naive_neighbor_matrix_overload = {}
_fill_batch_naive_neighbor_matrix_pbc_overload = {}
for t, v, m in zip(T, V, M):
    _fill_batch_naive_neighbor_matrix_overload[t] = wp.overload(
        _fill_batch_naive_neighbor_matrix,
        [
            wp.array(dtype=v),
            t,
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32, ndim=2),
            wp.array(dtype=wp.int32),
            wp.bool,
        ],
    )
    _fill_batch_naive_neighbor_matrix_pbc_overload[t] = wp.overload(
        _fill_batch_naive_neighbor_matrix_pbc,
        [
            wp.array(dtype=v),
            wp.array(dtype=m),
            t,
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.vec3i),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32, ndim=2),
            wp.array(dtype=wp.vec3i, ndim=2),
            wp.array(dtype=wp.int32),
            wp.bool,
        ],
    )

###########################################################################################
########################### Warp Launchers ###############################################
###########################################################################################


def batch_naive_neighbor_matrix(
    positions: wp.array,
    cutoff: float,
    batch_idx: wp.array,
    batch_ptr: wp.array,
    neighbor_matrix: wp.array,
    num_neighbors: wp.array,
    wp_dtype: type,
    device: str,
    half_fill: bool = False,
) -> None:
    """Core warp launcher for batched naive neighbor matrix construction (no PBC).

    Computes pairwise distances and fills the neighbor matrix for multiple systems
    in a batch using pure warp operations. No periodic boundary conditions are applied.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Concatenated atomic coordinates for all systems in Cartesian space.
    cutoff : float
        Cutoff distance for neighbor detection in Cartesian units.
    batch_idx : wp.array, shape (total_atoms,), dtype=wp.int32
        System index for each atom.
    batch_ptr : wp.array, shape (num_systems + 1,), dtype=wp.int32
        Cumulative atom counts defining system boundaries.
    neighbor_matrix : wp.array, shape (total_atoms, max_neighbors), dtype=wp.int32
        OUTPUT: Neighbor matrix to be filled with neighbor atom indices.
    num_neighbors : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Number of neighbors found for each atom.
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
    batch_naive_neighbor_matrix_pbc : Version with periodic boundary conditions
    _fill_batch_naive_neighbor_matrix : Kernel that performs the computation
    """
    total_atoms = positions.shape[0]

    wp.launch(
        kernel=_fill_batch_naive_neighbor_matrix_overload[wp_dtype],
        dim=total_atoms,
        inputs=[
            positions,
            wp_dtype(cutoff * cutoff),
            batch_idx,
            batch_ptr,
            neighbor_matrix,
            num_neighbors,
            half_fill,
        ],
        device=device,
    )


def batch_naive_neighbor_matrix_pbc(
    positions: wp.array,
    cell: wp.array,
    cutoff: float,
    batch_ptr: wp.array,
    shifts: wp.array,
    shift_system_idx: wp.array,
    neighbor_matrix: wp.array,
    neighbor_matrix_shifts: wp.array,
    num_neighbors: wp.array,
    wp_dtype: type,
    device: str,
    max_atoms_per_system: int,
    half_fill: bool = False,
) -> None:
    """Core warp launcher for batched naive neighbor matrix construction with PBC.

    Computes neighbor relationships between atoms across periodic boundaries for
    multiple systems in a batch using pure warp operations. Assumes shift vectors
    have been pre-computed.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Concatenated atomic coordinates for all systems in Cartesian space.
    cell : wp.array, shape (num_systems, 3, 3), dtype=wp.mat33*
        Array of cell matrices for each system in the batch.
    cutoff : float
        Cutoff distance for neighbor detection in Cartesian units.
    batch_ptr : wp.array, shape (num_systems + 1,), dtype=wp.int32
        Cumulative atom counts defining system boundaries.
    shifts : wp.array, shape (total_shifts, 3), dtype=wp.vec3i
        Array of integer shift vectors for periodic images.
    shift_system_idx : wp.array, shape (total_shifts,), dtype=wp.int32
        Array mapping each shift to its system index in the batch.
    neighbor_matrix : wp.array, shape (total_atoms, max_neighbors), dtype=wp.int32
        OUTPUT: Neighbor matrix to be filled with neighbor atom indices.
    neighbor_matrix_shifts : wp.array, shape (total_atoms, max_neighbors, 3), dtype=wp.vec3i
        OUTPUT: Matrix storing shift vectors for each neighbor relationship.
    num_neighbors : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Number of neighbors found for each atom.
    wp_dtype : type
        Warp dtype (wp.float32, wp.float64, or wp.float16).
    device : str
        Warp device string (e.g., 'cuda:0', 'cpu').
    max_atoms_per_system : int
        Maximum number of atoms in any single system in the batch.
    half_fill : bool, default=False
        If True, only store half of the neighbor relationships.

    Notes
    -----
    - This is a low-level warp interface. For framework bindings, use torch/jax wrappers.
    - Output arrays must be pre-allocated by caller.
    - Shift vectors must be pre-computed using compute_naive_num_shifts and _expand_naive_shifts.

    See Also
    --------
    batch_naive_neighbor_matrix : Version without periodic boundary conditions
    _fill_batch_naive_neighbor_matrix_pbc : Kernel that performs the computation
    """
    total_shifts = shifts.shape[0]

    wp.launch(
        kernel=_fill_batch_naive_neighbor_matrix_pbc_overload[wp_dtype],
        dim=(total_shifts, max_atoms_per_system),
        inputs=[
            positions,
            cell,
            wp_dtype(cutoff * cutoff),
            batch_ptr,
            shifts,
            shift_system_idx,
            neighbor_matrix,
            neighbor_matrix_shifts,
            num_neighbors,
            half_fill,
        ],
        device=device,
    )
