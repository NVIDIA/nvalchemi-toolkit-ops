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
    _decode_shift_index,
    _update_neighbor_matrix,
    _update_neighbor_matrix_pbc,
    compute_inv_cells,
    selective_zero_num_neighbors,
    wrap_positions_batch,
)

__all__ = [
    "batch_naive_neighbor_matrix_dual_cutoff",
    "batch_naive_neighbor_matrix_pbc_dual_cutoff",
]

###########################################################################################
########################### Batch Naive Dual Cutoff Kernels ###############################
###########################################################################################


@wp.func
def _batch_naive_dual_cutoff_body(
    tid: int,
    positions: wp.array(dtype=Any),
    cutoff1_sq: Any,
    cutoff2_sq: Any,
    batch_idx: wp.array(dtype=wp.int32),
    batch_ptr: wp.array(dtype=wp.int32),
    neighbor_matrix1: wp.array2d(dtype=wp.int32),
    num_neighbors1: wp.array(dtype=wp.int32),
    neighbor_matrix2: wp.array2d(dtype=wp.int32),
    num_neighbors2: wp.array(dtype=wp.int32),
    half_fill: wp.bool,
):
    """Body function for batched naive dual cutoff neighbor search (no PBC).

    Parameters
    ----------
    tid : int
        Thread index (atom index i in the global atom array).
    positions : wp.array, shape (total_atoms,), dtype=wp.vec3*
        Concatenated Cartesian coordinates for all systems.
    cutoff1_sq : float
        Squared first cutoff distance (typically smaller).
    cutoff2_sq : float
        Squared second cutoff distance (typically larger).
    batch_idx : wp.array, shape (total_atoms,), dtype=wp.int32
        System index for each atom.
    batch_ptr : wp.array, shape (num_systems + 1,), dtype=wp.int32
        Cumulative atom counts defining system boundaries.
    neighbor_matrix1 : wp.array2d, shape (total_atoms, max_neighbors1), dtype=wp.int32
        OUTPUT: First neighbor matrix for cutoff1.
    num_neighbors1 : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Neighbor counts for cutoff1.
    neighbor_matrix2 : wp.array2d, shape (total_atoms, max_neighbors2), dtype=wp.int32
        OUTPUT: Second neighbor matrix for cutoff2.
    num_neighbors2 : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Neighbor counts for cutoff2.
    half_fill : wp.bool
        If True, only store relationships where i < j.
    """
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


@wp.func
def _batch_naive_dual_cutoff_pbc_body(
    shift: wp.vec3i,
    iatom_global: int,
    isys: int,
    positions: wp.array(dtype=Any),
    per_atom_cell_offsets: wp.array(dtype=wp.vec3i),
    cell: wp.array(dtype=Any),
    cutoff1_sq: Any,
    cutoff2_sq: Any,
    batch_ptr: wp.array(dtype=wp.int32),
    neighbor_matrix1: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix2: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts1: wp.array(dtype=wp.vec3i, ndim=2),
    neighbor_matrix_shifts2: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors1: wp.array(dtype=wp.int32),
    num_neighbors2: wp.array(dtype=wp.int32),
    half_fill: wp.bool,
):
    """Body function for batched naive dual cutoff neighbor search with PBC.

    Parameters
    ----------
    shift : wp.vec3i
        Integer shift vector for the current periodic image.
    iatom_global : int
        Global atom index (already offset by system start).
    isys : int
        System index for this atom.
    positions : wp.array, shape (total_atoms,), dtype=wp.vec3*
        Wrapped concatenated Cartesian coordinates for all systems.
    per_atom_cell_offsets : wp.array, shape (total_atoms,), dtype=wp.vec3i
        Integer cell offsets for each atom.
    cell : wp.array, shape (num_systems,), dtype=wp.mat33*
        Cell matrices for each system.
    cutoff1_sq : float
        Squared first cutoff distance (typically smaller).
    cutoff2_sq : float
        Squared second cutoff distance (typically larger).
    batch_ptr : wp.array, shape (num_systems + 1,), dtype=wp.int32
        Cumulative atom counts defining system boundaries.
    neighbor_matrix1 : wp.array, ndim=2, dtype=wp.int32
        OUTPUT: First neighbor matrix for cutoff1.
    neighbor_matrix2 : wp.array, ndim=2, dtype=wp.int32
        OUTPUT: Second neighbor matrix for cutoff2.
    neighbor_matrix_shifts1 : wp.array, ndim=2, dtype=wp.vec3i
        OUTPUT: Shift vectors for first neighbor matrix.
    neighbor_matrix_shifts2 : wp.array, ndim=2, dtype=wp.vec3i
        OUTPUT: Shift vectors for second neighbor matrix.
    num_neighbors1 : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Neighbor counts for cutoff1.
    num_neighbors2 : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Neighbor counts for cutoff2.
    half_fill : wp.bool
        If True, only store relationships where i < j.
    """
    jatom_start = batch_ptr[isys]
    jatom_end = batch_ptr[isys + 1]
    maxnb1 = neighbor_matrix1.shape[1]
    maxnb2 = neighbor_matrix2.shape[1]
    _cell = cell[isys]
    _pos_i = positions[iatom_global]
    _int_i = per_atom_cell_offsets[iatom_global]
    positions_shifted = type(_cell[0])(shift) * _cell + _pos_i
    _zero_shift = shift[0] == 0 and shift[1] == 0 and shift[2] == 0
    if _zero_shift:
        jatom_end = iatom_global
    for jatom in range(jatom_start, jatom_end):
        _pos_j = positions[jatom]
        diff = positions_shifted - _pos_j
        dist_sq = wp.length_sq(diff)
        if dist_sq < cutoff2_sq:
            _int_j = per_atom_cell_offsets[jatom]
            _corrected_shift = wp.vec3i(
                shift[0] - _int_i[0] + _int_j[0],
                shift[1] - _int_i[1] + _int_j[1],
                shift[2] - _int_i[2] + _int_j[2],
            )
            _update_neighbor_matrix_pbc(
                jatom,
                iatom_global,
                neighbor_matrix2,
                neighbor_matrix_shifts2,
                num_neighbors2,
                _corrected_shift,
                maxnb2,
                half_fill,
            )
            if dist_sq < cutoff1_sq:
                _update_neighbor_matrix_pbc(
                    jatom,
                    iatom_global,
                    neighbor_matrix1,
                    neighbor_matrix_shifts1,
                    num_neighbors1,
                    _corrected_shift,
                    maxnb1,
                    half_fill,
                )


@wp.func
def _batch_naive_dual_cutoff_pbc_body_prewrapped(
    shift: wp.vec3i,
    iatom_global: int,
    isys: int,
    positions: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    cutoff1_sq: Any,
    cutoff2_sq: Any,
    batch_ptr: wp.array(dtype=wp.int32),
    neighbor_matrix1: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix2: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts1: wp.array(dtype=wp.vec3i, ndim=2),
    neighbor_matrix_shifts2: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors1: wp.array(dtype=wp.int32),
    num_neighbors2: wp.array(dtype=wp.int32),
    half_fill: wp.bool,
):
    jatom_start = batch_ptr[isys]
    jatom_end = batch_ptr[isys + 1]
    maxnb1 = neighbor_matrix1.shape[1]
    maxnb2 = neighbor_matrix2.shape[1]
    _cell = cell[isys]
    _pos_i = positions[iatom_global]
    positions_shifted = type(_cell[0])(shift) * _cell + _pos_i
    _zero_shift = shift[0] == 0 and shift[1] == 0 and shift[2] == 0
    if _zero_shift:
        jatom_end = iatom_global
    for jatom in range(jatom_start, jatom_end):
        _pos_j = positions[jatom]
        diff = positions_shifted - _pos_j
        dist_sq = wp.length_sq(diff)
        if dist_sq < cutoff2_sq:
            _update_neighbor_matrix_pbc(
                jatom,
                iatom_global,
                neighbor_matrix2,
                neighbor_matrix_shifts2,
                num_neighbors2,
                shift,
                maxnb2,
                half_fill,
            )
            if dist_sq < cutoff1_sq:
                _update_neighbor_matrix_pbc(
                    jatom,
                    iatom_global,
                    neighbor_matrix1,
                    neighbor_matrix_shifts1,
                    num_neighbors1,
                    shift,
                    maxnb1,
                    half_fill,
                )


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
        Concatenated Cartesian coordinates for all systems.
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
    _batch_naive_dual_cutoff_body(
        tid,
        positions,
        cutoff1_sq,
        cutoff2_sq,
        batch_idx,
        batch_ptr,
        neighbor_matrix1,
        num_neighbors1,
        neighbor_matrix2,
        num_neighbors2,
        half_fill,
    )


@wp.kernel(enable_backward=False)
def _fill_batch_naive_neighbor_matrix_pbc_dual_cutoff(
    positions: wp.array(dtype=Any),
    per_atom_cell_offsets: wp.array(dtype=wp.vec3i),
    cell: wp.array(dtype=Any),
    cutoff1_sq: Any,
    cutoff2_sq: Any,
    batch_ptr: wp.array(dtype=wp.int32),
    shift_range: wp.array(dtype=wp.vec3i),
    num_shifts_arr: wp.array(dtype=wp.int32),
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
    considering all periodic images within the cutoff distance. Uses a 3D launch
    pattern to parallelize over systems, shifts, and atoms.

    This function operates on a batch of systems.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Concatenated Cartesian coordinates for all systems.
        Assumed to be wrapped into the primary cell before calling this kernel via wrap_positions_batch.
    per_atom_cell_offsets : wp.array, shape (total_atoms,), dtype=wp.vec3i
        Integer cell offsets for each atom (floor of fractional coordinates).
        Used to reconstruct corrected shift vectors for the original positions.
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
    shift_range : wp.array, shape (num_systems, 3), dtype=wp.vec3i
        Shift range per dimension per system.
    num_shifts_arr : wp.array, shape (num_systems,), dtype=wp.int32
        Number of shifts per system (for bounds checking).
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
    - Thread launch: 3D (num_systems, max_shifts_per_system, max_atoms_per_system)

    See Also
    --------
    _fill_batch_naive_neighbor_matrix_dual_cutoff : Version without periodic boundaries
    _fill_naive_neighbor_matrix_pbc_dual_cutoff : Single system version
    """
    isys, ishift_local, iatom = wp.tid()

    if ishift_local >= num_shifts_arr[isys]:
        return

    _natom = batch_ptr[isys + 1] - batch_ptr[isys]

    if iatom >= _natom:
        return

    iatom_global = iatom + batch_ptr[isys]
    shift = _decode_shift_index(ishift_local, shift_range[isys])
    _batch_naive_dual_cutoff_pbc_body(
        shift,
        iatom_global,
        isys,
        positions,
        per_atom_cell_offsets,
        cell,
        cutoff1_sq,
        cutoff2_sq,
        batch_ptr,
        neighbor_matrix1,
        neighbor_matrix2,
        neighbor_matrix_shifts1,
        neighbor_matrix_shifts2,
        num_neighbors1,
        num_neighbors2,
        half_fill,
    )


@wp.kernel(enable_backward=False)
def _fill_batch_naive_neighbor_matrix_dual_cutoff_selective(
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
    rebuild_flags: wp.array(dtype=wp.bool),
) -> None:
    """Selective batched naive dual cutoff kernel — skips systems where rebuild_flags[isys] is False.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Concatenated Cartesian coordinates for all systems.
    cutoff1_sq : float
        Squared first cutoff distance (typically smaller).
    cutoff2_sq : float
        Squared second cutoff distance (typically larger).
    batch_idx : wp.array, shape (total_atoms,), dtype=wp.int32
        System index for each atom.
    batch_ptr : wp.array, shape (num_systems + 1,), dtype=wp.int32
        Cumulative atom counts defining system boundaries.
    neighbor_matrix1 : wp.array2d, shape (total_atoms, max_neighbors1), dtype=wp.int32
        OUTPUT: First neighbor matrix for cutoff1.
    num_neighbors1 : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Neighbor counts for cutoff1.
    neighbor_matrix2 : wp.array2d, shape (total_atoms, max_neighbors2), dtype=wp.int32
        OUTPUT: Second neighbor matrix for cutoff2.
    num_neighbors2 : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Neighbor counts for cutoff2.
    half_fill : wp.bool
        If True, only store relationships where i < j.
    rebuild_flags : wp.array, shape (num_systems,), dtype=wp.bool
        Per-system rebuild flags. Systems where rebuild_flags[i] is False are skipped.

    Notes
    -----
    - Thread launch: One thread per atom (dim=total_atoms)
    - GPU-side conditional: no CPU-GPU synchronization occurs
    """
    tid = wp.tid()
    i = tid
    isys = batch_idx[i]
    if not rebuild_flags[isys]:
        return
    _batch_naive_dual_cutoff_body(
        i,
        positions,
        cutoff1_sq,
        cutoff2_sq,
        batch_idx,
        batch_ptr,
        neighbor_matrix1,
        num_neighbors1,
        neighbor_matrix2,
        num_neighbors2,
        half_fill,
    )


@wp.kernel(enable_backward=False)
def _fill_batch_naive_neighbor_matrix_pbc_dual_cutoff_selective(
    positions: wp.array(dtype=Any),
    per_atom_cell_offsets: wp.array(dtype=wp.vec3i),
    cell: wp.array(dtype=Any),
    cutoff1_sq: Any,
    cutoff2_sq: Any,
    batch_ptr: wp.array(dtype=wp.int32),
    shift_range: wp.array(dtype=wp.vec3i),
    num_shifts_arr: wp.array(dtype=wp.int32),
    neighbor_matrix1: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix2: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts1: wp.array(dtype=wp.vec3i, ndim=2),
    neighbor_matrix_shifts2: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors1: wp.array(dtype=wp.int32),
    num_neighbors2: wp.array(dtype=wp.int32),
    half_fill: wp.bool,
    rebuild_flags: wp.array(dtype=wp.bool),
) -> None:
    """Selective batched PBC naive dual cutoff kernel — skips systems where rebuild_flags[isys] is False.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Wrapped concatenated Cartesian coordinates for all systems.
    per_atom_cell_offsets : wp.array, shape (total_atoms,), dtype=wp.vec3i
        Integer cell offsets for each atom.
    cell : wp.array, shape (num_systems, 3, 3), dtype=wp.mat33*
        Array of cell matrices for each system in the batch.
    cutoff1_sq : float
        Squared first cutoff distance (typically smaller).
    cutoff2_sq : float
        Squared second cutoff distance (typically larger).
    batch_ptr : wp.array, shape (num_systems + 1,), dtype=wp.int32
        Cumulative sum of number of atoms per system.
    shift_range : wp.array, shape (num_systems, 3), dtype=wp.vec3i
        Shift range per dimension per system.
    num_shifts_arr : wp.array, shape (num_systems,), dtype=wp.int32
        Number of shifts per system (for bounds checking).
    neighbor_matrix1 : wp.array, ndim=2, dtype=wp.int32
        OUTPUT: First neighbor matrix for cutoff1.
    neighbor_matrix2 : wp.array, ndim=2, dtype=wp.int32
        OUTPUT: Second neighbor matrix for cutoff2.
    neighbor_matrix_shifts1 : wp.array, ndim=2, dtype=wp.vec3i
        OUTPUT: Shift vectors for first neighbor matrix.
    neighbor_matrix_shifts2 : wp.array, ndim=2, dtype=wp.vec3i
        OUTPUT: Shift vectors for second neighbor matrix.
    num_neighbors1 : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Neighbor counts for cutoff1.
    num_neighbors2 : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Neighbor counts for cutoff2.
    half_fill : wp.bool
        If True, only store relationships where i < j.
    rebuild_flags : wp.array, shape (num_systems,), dtype=wp.bool
        Per-system rebuild flags. Systems where rebuild_flags[i] is False are skipped.

    Notes
    -----
    - Thread launch: 3D (num_systems, max_shifts_per_system, max_atoms_per_system)
    - GPU-side conditional: no CPU-GPU synchronization occurs
    """
    isys, ishift_local, iatom = wp.tid()

    if not rebuild_flags[isys]:
        return

    if ishift_local >= num_shifts_arr[isys]:
        return

    _natom = batch_ptr[isys + 1] - batch_ptr[isys]

    if iatom >= _natom:
        return

    iatom_global = iatom + batch_ptr[isys]
    shift = _decode_shift_index(ishift_local, shift_range[isys])
    _batch_naive_dual_cutoff_pbc_body(
        shift,
        iatom_global,
        isys,
        positions,
        per_atom_cell_offsets,
        cell,
        cutoff1_sq,
        cutoff2_sq,
        batch_ptr,
        neighbor_matrix1,
        neighbor_matrix2,
        neighbor_matrix_shifts1,
        neighbor_matrix_shifts2,
        num_neighbors1,
        num_neighbors2,
        half_fill,
    )


@wp.kernel(enable_backward=False)
def _fill_batch_naive_neighbor_matrix_pbc_dual_cutoff_prewrapped(
    positions: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    cutoff1_sq: Any,
    cutoff2_sq: Any,
    batch_ptr: wp.array(dtype=wp.int32),
    shift_range: wp.array(dtype=wp.vec3i),
    num_shifts_arr: wp.array(dtype=wp.int32),
    neighbor_matrix1: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix2: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts1: wp.array(dtype=wp.vec3i, ndim=2),
    neighbor_matrix_shifts2: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors1: wp.array(dtype=wp.int32),
    num_neighbors2: wp.array(dtype=wp.int32),
    half_fill: wp.bool,
) -> None:
    """Batch PBC dual cutoff kernel for pre-wrapped positions (no cell-offset correction).

    Notes
    -----
    - Thread launch: 3D (num_systems, max_shifts_per_system, max_atoms_per_system)
    """
    isys, ishift_local, iatom = wp.tid()

    if ishift_local >= num_shifts_arr[isys]:
        return

    _natom = batch_ptr[isys + 1] - batch_ptr[isys]

    if iatom >= _natom:
        return

    iatom_global = iatom + batch_ptr[isys]
    shift = _decode_shift_index(ishift_local, shift_range[isys])
    _batch_naive_dual_cutoff_pbc_body_prewrapped(
        shift,
        iatom_global,
        isys,
        positions,
        cell,
        cutoff1_sq,
        cutoff2_sq,
        batch_ptr,
        neighbor_matrix1,
        neighbor_matrix2,
        neighbor_matrix_shifts1,
        neighbor_matrix_shifts2,
        num_neighbors1,
        num_neighbors2,
        half_fill,
    )


@wp.kernel(enable_backward=False)
def _fill_batch_naive_neighbor_matrix_pbc_dual_cutoff_prewrapped_selective(
    positions: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    cutoff1_sq: Any,
    cutoff2_sq: Any,
    batch_ptr: wp.array(dtype=wp.int32),
    shift_range: wp.array(dtype=wp.vec3i),
    num_shifts_arr: wp.array(dtype=wp.int32),
    neighbor_matrix1: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix2: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts1: wp.array(dtype=wp.vec3i, ndim=2),
    neighbor_matrix_shifts2: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors1: wp.array(dtype=wp.int32),
    num_neighbors2: wp.array(dtype=wp.int32),
    half_fill: wp.bool,
    rebuild_flags: wp.array(dtype=wp.bool),
) -> None:
    """Selective batch PBC dual cutoff kernel for pre-wrapped positions.

    Notes
    -----
    - Thread launch: 3D (num_systems, max_shifts_per_system, max_atoms_per_system)
    """
    isys, ishift_local, iatom = wp.tid()

    if not rebuild_flags[isys]:
        return

    if ishift_local >= num_shifts_arr[isys]:
        return

    _natom = batch_ptr[isys + 1] - batch_ptr[isys]

    if iatom >= _natom:
        return

    iatom_global = iatom + batch_ptr[isys]
    shift = _decode_shift_index(ishift_local, shift_range[isys])
    _batch_naive_dual_cutoff_pbc_body_prewrapped(
        shift,
        iatom_global,
        isys,
        positions,
        cell,
        cutoff1_sq,
        cutoff2_sq,
        batch_ptr,
        neighbor_matrix1,
        neighbor_matrix2,
        neighbor_matrix_shifts1,
        neighbor_matrix_shifts2,
        num_neighbors1,
        num_neighbors2,
        half_fill,
    )


T = [wp.float32, wp.float64, wp.float16]
V = [wp.vec3f, wp.vec3d, wp.vec3h]
M = [wp.mat33f, wp.mat33d, wp.mat33h]
_fill_batch_naive_neighbor_matrix_dual_cutoff_overload = {}
_fill_batch_naive_neighbor_matrix_pbc_dual_cutoff_overload = {}
_fill_batch_naive_neighbor_matrix_pbc_dual_cutoff_prewrapped_overload = {}
_fill_batch_naive_neighbor_matrix_dual_cutoff_selective_overload = {}
_fill_batch_naive_neighbor_matrix_pbc_dual_cutoff_selective_overload = {}
_fill_batch_naive_neighbor_matrix_pbc_dual_cutoff_prewrapped_selective_overload = {}
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
            wp.array(dtype=wp.vec3i),
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
    _fill_batch_naive_neighbor_matrix_dual_cutoff_selective_overload[t] = wp.overload(
        _fill_batch_naive_neighbor_matrix_dual_cutoff_selective,
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
            wp.array(dtype=wp.bool),
        ],
    )
    _fill_batch_naive_neighbor_matrix_pbc_dual_cutoff_selective_overload[t] = (
        wp.overload(
            _fill_batch_naive_neighbor_matrix_pbc_dual_cutoff_selective,
            [
                wp.array(dtype=v),
                wp.array(dtype=wp.vec3i),
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
                wp.array(dtype=wp.bool),
            ],
        )
    )
    _fill_batch_naive_neighbor_matrix_pbc_dual_cutoff_prewrapped_overload[t] = (
        wp.overload(
            _fill_batch_naive_neighbor_matrix_pbc_dual_cutoff_prewrapped,
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
    )
    _fill_batch_naive_neighbor_matrix_pbc_dual_cutoff_prewrapped_selective_overload[
        t
    ] = wp.overload(
        _fill_batch_naive_neighbor_matrix_pbc_dual_cutoff_prewrapped_selective,
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
            wp.array(dtype=wp.bool),
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
    rebuild_flags: wp.array | None = None,
) -> None:
    """Core warp launcher for batched naive dual cutoff neighbor matrix construction (no PBC).

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Concatenated Cartesian coordinates for all systems.
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
    rebuild_flags : wp.array, shape (num_systems,), dtype=wp.bool, optional
        Per-system rebuild flags. If provided, only systems where rebuild_flags[i]
        is True are processed; others are skipped on the GPU without CPU sync.
        Call selective_zero_num_neighbors before this launcher to reset counts.

    See Also
    --------
    batch_naive_neighbor_matrix_pbc_dual_cutoff : Version with PBC
    _fill_batch_naive_neighbor_matrix_dual_cutoff : Kernel that performs computation
    _fill_batch_naive_neighbor_matrix_dual_cutoff_selective : Selective-skip kernel variant
    """
    total_atoms = positions.shape[0]

    if rebuild_flags is not None:
        selective_zero_num_neighbors(num_neighbors1, batch_idx, rebuild_flags, device)
        selective_zero_num_neighbors(num_neighbors2, batch_idx, rebuild_flags, device)
        wp.launch(
            kernel=_fill_batch_naive_neighbor_matrix_dual_cutoff_selective_overload[
                wp_dtype
            ],
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
                rebuild_flags,
            ],
            device=device,
        )
    else:
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
    batch_idx: wp.array,
    shift_range: wp.array,
    num_shifts_arr: wp.array,
    max_shifts_per_system: int,
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
    rebuild_flags: wp.array | None = None,
    wrap_positions: bool = True,
) -> None:
    """Core warp launcher for batched naive dual cutoff neighbor matrix construction with PBC.

    Computes neighbor relationships between atoms across periodic boundaries for
    two different cutoff distances using pure warp operations.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Concatenated Cartesian coordinates for all systems.
    cell : wp.array, shape (num_systems, 3, 3), dtype=wp.mat33*
        Cell matrices for each system in the batch.
    cutoff1 : float
        First cutoff distance (typically smaller).
    cutoff2 : float
        Second cutoff distance (typically larger).
    batch_ptr : wp.array, shape (num_systems + 1,), dtype=wp.int32
        Cumulative atom counts defining system boundaries.
    batch_idx : wp.array, shape (total_atoms,), dtype=wp.int32
        System index for each atom. Required for the position-wrapping
        preprocessing step that maps atoms to their system's cell.
    shift_range : wp.array, shape (num_systems, 3), dtype=wp.vec3i
        Shift range per dimension per system.
    num_shifts_arr : wp.array, shape (num_systems,), dtype=wp.int32
        Number of shifts per system.
    max_shifts_per_system : int
        Maximum per-system shift count (launch dimension).
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
    rebuild_flags : wp.array, shape (num_systems,), dtype=wp.bool, optional
        Per-system rebuild flags. If provided, only systems where rebuild_flags[i]
        is True are processed; others are skipped on the GPU without CPU sync.
        Call selective_zero_num_neighbors before this launcher to reset counts.
    wrap_positions : bool, default=True
        If True, wrap input positions into the primary cell before
        neighbor search. Set to False when positions are already
        wrapped (e.g. by a preceding integration step) to save two
        GPU kernel launches per call.

    Notes
    -----
    - This is a low-level warp interface. For framework bindings, use torch/jax wrappers.
    - Output arrays must be pre-allocated by caller.
    - When ``wrap_positions`` is True, positions are wrapped into the primary cell in a
      preprocessing step before the neighbor search kernel.

    See Also
    --------
    batch_naive_neighbor_matrix_dual_cutoff : Version without PBC
    _fill_batch_naive_neighbor_matrix_pbc_dual_cutoff : Kernel that performs computation
    _fill_batch_naive_neighbor_matrix_pbc_dual_cutoff_selective : Selective-skip kernel variant
    wrap_positions_batch : Preprocessing step that wraps positions
    """
    total_atoms = positions.shape[0]
    num_systems = cell.shape[0]

    if wrap_positions:
        wp_mat_dtype = (
            wp.mat33f
            if wp_dtype == wp.float32
            else wp.mat33d
            if wp_dtype == wp.float64
            else wp.mat33h
            if wp_dtype == wp.float16
            else None
        )
        wp_vec_dtype = (
            wp.vec3f
            if wp_dtype == wp.float32
            else wp.vec3d
            if wp_dtype == wp.float64
            else wp.vec3h
            if wp_dtype == wp.float16
            else None
        )
        inv_cell = wp.empty((cell.shape[0],), dtype=wp_mat_dtype, device=device)
        compute_inv_cells(cell, inv_cell, wp_dtype, device)
        positions_wrapped = wp.empty((total_atoms,), dtype=wp_vec_dtype, device=device)
        per_atom_cell_offsets = wp.empty(total_atoms, dtype=wp.vec3i, device=device)
        wrap_positions_batch(
            positions,
            cell,
            inv_cell,
            batch_idx,
            positions_wrapped,
            per_atom_cell_offsets,
            wp_dtype,
            device,
        )

        if rebuild_flags is not None:
            selective_zero_num_neighbors(
                num_neighbors1, batch_idx, rebuild_flags, device
            )
            selective_zero_num_neighbors(
                num_neighbors2, batch_idx, rebuild_flags, device
            )
            wp.launch(
                kernel=_fill_batch_naive_neighbor_matrix_pbc_dual_cutoff_selective_overload[
                    wp_dtype
                ],
                dim=(num_systems, max_shifts_per_system, max_atoms_per_system),
                inputs=[
                    positions_wrapped,
                    per_atom_cell_offsets,
                    cell,
                    wp_dtype(cutoff1 * cutoff1),
                    wp_dtype(cutoff2 * cutoff2),
                    batch_ptr,
                    shift_range,
                    num_shifts_arr,
                    neighbor_matrix1,
                    neighbor_matrix2,
                    neighbor_matrix_shifts1,
                    neighbor_matrix_shifts2,
                    num_neighbors1,
                    num_neighbors2,
                    half_fill,
                    rebuild_flags,
                ],
                device=device,
            )
        else:
            wp.launch(
                kernel=_fill_batch_naive_neighbor_matrix_pbc_dual_cutoff_overload[
                    wp_dtype
                ],
                dim=(num_systems, max_shifts_per_system, max_atoms_per_system),
                inputs=[
                    positions_wrapped,
                    per_atom_cell_offsets,
                    cell,
                    wp_dtype(cutoff1 * cutoff1),
                    wp_dtype(cutoff2 * cutoff2),
                    batch_ptr,
                    shift_range,
                    num_shifts_arr,
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
    else:
        if rebuild_flags is not None:
            selective_zero_num_neighbors(
                num_neighbors1, batch_idx, rebuild_flags, device
            )
            selective_zero_num_neighbors(
                num_neighbors2, batch_idx, rebuild_flags, device
            )
            wp.launch(
                kernel=_fill_batch_naive_neighbor_matrix_pbc_dual_cutoff_prewrapped_selective_overload[
                    wp_dtype
                ],
                dim=(num_systems, max_shifts_per_system, max_atoms_per_system),
                inputs=[
                    positions,
                    cell,
                    wp_dtype(cutoff1 * cutoff1),
                    wp_dtype(cutoff2 * cutoff2),
                    batch_ptr,
                    shift_range,
                    num_shifts_arr,
                    neighbor_matrix1,
                    neighbor_matrix2,
                    neighbor_matrix_shifts1,
                    neighbor_matrix_shifts2,
                    num_neighbors1,
                    num_neighbors2,
                    half_fill,
                    rebuild_flags,
                ],
                device=device,
            )
        else:
            wp.launch(
                kernel=_fill_batch_naive_neighbor_matrix_pbc_dual_cutoff_prewrapped_overload[
                    wp_dtype
                ],
                dim=(num_systems, max_shifts_per_system, max_atoms_per_system),
                inputs=[
                    positions,
                    cell,
                    wp_dtype(cutoff1 * cutoff1),
                    wp_dtype(cutoff2 * cutoff2),
                    batch_ptr,
                    shift_range,
                    num_shifts_arr,
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
