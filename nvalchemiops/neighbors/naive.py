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

"""Core warp kernels and launchers for naive neighbor list construction.

This module contains warp kernels for O(N²) neighbor list computation.
See `nvalchemiops.torch.neighbors` for PyTorch bindings.
"""

from typing import Any

import warp as wp

from nvalchemiops.neighbors.neighbor_utils import (
    _decode_shift_index,
    _update_neighbor_matrix,
    _update_neighbor_matrix_pbc,
    compute_inv_cells,
    wrap_positions_single,
)


@wp.func
def _naive_neighbor_body(
    tid: int,
    positions: wp.array(dtype=Any),
    cutoff_sq: Any,
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    half_fill: wp.bool,
):
    j_end = positions.shape[0]
    positions_i = positions[tid]
    max_neighbors = neighbor_matrix.shape[1]
    for j in range(tid + 1, j_end):
        diff = positions_i - positions[j]
        dist_sq = wp.length_sq(diff)
        if dist_sq < cutoff_sq:
            _update_neighbor_matrix(
                tid, j, neighbor_matrix, num_neighbors, max_neighbors, half_fill
            )


@wp.func
def _naive_neighbor_pbc_body(
    shift: wp.vec3i,
    iatom: int,
    positions: wp.array(dtype=Any),
    per_atom_cell_offsets: wp.array(dtype=wp.vec3i),
    cutoff_sq: Any,
    cell: wp.array(dtype=Any),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    half_fill: wp.bool,
):
    jatom_start = wp.int32(0)
    jatom_end = positions.shape[0]
    maxnb = neighbor_matrix.shape[1]
    _cell = cell[0]
    _pos_i = positions[iatom]
    _int_i = per_atom_cell_offsets[iatom]
    positions_shifted = type(_cell[0])(shift) * _cell + _pos_i
    _zero_shift = shift[0] == 0 and shift[1] == 0 and shift[2] == 0
    if _zero_shift:
        jatom_end = iatom
    for jatom in range(jatom_start, jatom_end):
        _pos_j = positions[jatom]
        diff = positions_shifted - _pos_j
        dist_sq = wp.length_sq(diff)
        if dist_sq < cutoff_sq:
            _int_j = per_atom_cell_offsets[jatom]
            _corrected_shift = wp.vec3i(
                shift[0] - _int_i[0] + _int_j[0],
                shift[1] - _int_i[1] + _int_j[1],
                shift[2] - _int_i[2] + _int_j[2],
            )
            _update_neighbor_matrix_pbc(
                jatom,
                iatom,
                neighbor_matrix,
                neighbor_matrix_shifts,
                num_neighbors,
                _corrected_shift,
                maxnb,
                half_fill,
            )


@wp.func
def _naive_neighbor_pbc_body_prewrapped(
    shift: wp.vec3i,
    iatom: int,
    positions: wp.array(dtype=Any),
    cutoff_sq: Any,
    cell: wp.array(dtype=Any),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    half_fill: wp.bool,
):
    jatom_start = wp.int32(0)
    jatom_end = positions.shape[0]
    maxnb = neighbor_matrix.shape[1]
    _cell = cell[0]
    _pos_i = positions[iatom]
    positions_shifted = type(_cell[0])(shift) * _cell + _pos_i
    _zero_shift = shift[0] == 0 and shift[1] == 0 and shift[2] == 0
    if _zero_shift:
        jatom_end = iatom
    for jatom in range(jatom_start, jatom_end):
        _pos_j = positions[jatom]
        diff = positions_shifted - _pos_j
        dist_sq = wp.length_sq(diff)
        if dist_sq < cutoff_sq:
            _update_neighbor_matrix_pbc(
                jatom,
                iatom,
                neighbor_matrix,
                neighbor_matrix_shifts,
                num_neighbors,
                shift,
                maxnb,
                half_fill,
            )


@wp.kernel(enable_backward=False)
def _fill_naive_neighbor_matrix(
    positions: wp.array(dtype=Any),
    cutoff_sq: Any,
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    half_fill: wp.bool,
) -> None:
    """Calculate neighbor matrix using naive O(N^2) algorithm.

    Computes pairwise distances between all atoms and identifies neighbors
    within the specified cutoff distance. No periodic boundary conditions
    are applied.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Atomic coordinates in Cartesian space. Each row represents one atom's
        (x, y, z) position.
    cutoff_sq : float
        Squared cutoff distance for neighbor detection in Cartesian units.
        Must be positive. Atoms within this distance are considered neighbors.
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
    _fill_naive_neighbor_matrix_pbc : Version with periodic boundary conditions
    _fill_batch_naive_neighbor_matrix : Batch version for multiple systems
    """
    tid = wp.tid()
    _naive_neighbor_body(
        tid, positions, cutoff_sq, neighbor_matrix, num_neighbors, half_fill
    )


@wp.kernel(enable_backward=False)
def _fill_naive_neighbor_matrix_pbc(
    positions: wp.array(dtype=Any),
    per_atom_cell_offsets: wp.array(dtype=wp.vec3i),
    cutoff_sq: Any,
    cell: wp.array(dtype=Any),
    shift_range: wp.array(dtype=wp.vec3i),
    neighbor_matrix: wp.array2d(dtype=wp.int32),
    neighbor_matrix_shifts: wp.array2d(dtype=wp.vec3i),
    num_neighbors: wp.array(dtype=wp.int32),
    half_fill: wp.bool,
) -> None:
    """Calculate neighbor matrix with periodic boundary conditions using naive O(N^2) algorithm.

    Computes neighbor relationships between atoms across periodic boundaries by
    considering all periodic images within the cutoff distance. Uses a 2D launch
    pattern to parallelize over both atoms and periodic shifts.

    Positions must be pre-wrapped and per-atom cell offsets pre-computed via
    ``wrap_positions_single`` before calling this kernel.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Assumed to be wrapped into the primary cell before calling this kernel.
    per_atom_cell_offsets : wp.array, shape (total_atoms,), dtype=wp.vec3i
        Integer cell offsets for each atom (floor of fractional coordinates).
    cutoff_sq : float
        Squared cutoff distance for neighbor detection.
    cell : wp.array, shape (1, 3, 3), dtype=wp.mat33*
        Cell matrix defining lattice vectors.
    shift_range : wp.array, shape (1, 3), dtype=wp.vec3i
        Shift range per dimension for the single system.
    neighbor_matrix : wp.array, shape (total_atoms, max_neighbors), dtype=wp.int32
        OUTPUT: Neighbor matrix to be filled with neighbor atom indices.
    neighbor_matrix_shifts : wp.array, shape (total_atoms, max_neighbors), dtype=wp.vec3i
        OUTPUT: Matrix storing shift vectors for each neighbor relationship.
    num_neighbors : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Number of neighbors found for each atom.
    half_fill : wp.bool
        If True, only store relationships where i < j.

    Returns
    -------
    None
        This function modifies the input arrays in-place:

        - neighbor_matrix : Filled with neighbor atom indices
        - neighbor_matrix_shifts : Filled with corresponding shift vectors
        - num_neighbors : Updated with neighbor counts per atom

    Notes
    -----
    - Thread launch: 2D (num_shifts, total_atoms)
    - Shift vectors are decoded on-the-fly from the thread index via ``_decode_shift_index``

    See Also
    --------
    _fill_naive_neighbor_matrix : Version without periodic boundary conditions
    _fill_batch_naive_neighbor_matrix_pbc : Batch version for multiple systems
    """
    ishift, iatom = wp.tid()
    shift = _decode_shift_index(ishift, shift_range[0])
    _naive_neighbor_pbc_body(
        shift,
        iatom,
        positions,
        per_atom_cell_offsets,
        cutoff_sq,
        cell,
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
        half_fill,
    )


@wp.kernel(enable_backward=False)
def _fill_naive_neighbor_matrix_selective(
    positions: wp.array(dtype=Any),
    cutoff_sq: Any,
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    half_fill: wp.bool,
    rebuild_flags: wp.array(dtype=wp.bool),
) -> None:
    """Selective single-system naive neighbor matrix kernel — skips when not rebuilding.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Cartesian coordinates.
    cutoff_sq : float
        Squared cutoff distance for neighbor detection.
    neighbor_matrix : wp.array, shape (total_atoms, max_neighbors), dtype=wp.int32
        OUTPUT: Neighbor matrix to be filled with neighbor atom indices.
    num_neighbors : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Number of neighbors found for each atom.
    half_fill : wp.bool
        If True, only store relationships where i < j.
    rebuild_flags : wp.array, shape (1,), dtype=wp.bool
        When False the kernel returns immediately — no recomputation.

    Notes
    -----
    - Thread launch: One thread per atom (dim=total_atoms)
    - GPU-side conditional: no CPU-GPU synchronization occurs
    """
    tid = wp.tid()
    if not rebuild_flags[0]:
        return
    _naive_neighbor_body(
        tid, positions, cutoff_sq, neighbor_matrix, num_neighbors, half_fill
    )


@wp.kernel(enable_backward=False)
def _fill_naive_neighbor_matrix_pbc_selective(
    positions: wp.array(dtype=Any),
    per_atom_cell_offsets: wp.array(dtype=wp.vec3i),
    cutoff_sq: Any,
    cell: wp.array(dtype=Any),
    shift_range: wp.array(dtype=wp.vec3i),
    neighbor_matrix: wp.array2d(dtype=wp.int32),
    neighbor_matrix_shifts: wp.array2d(dtype=wp.vec3i),
    num_neighbors: wp.array(dtype=wp.int32),
    half_fill: wp.bool,
    rebuild_flags: wp.array(dtype=wp.bool),
) -> None:
    """Selective single-system PBC naive neighbor matrix kernel — skips when not rebuilding.

    Positions must be pre-wrapped and per-atom cell offsets pre-computed via
    ``wrap_positions_single`` before calling this kernel.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Assumed to be wrapped into the primary cell.
    per_atom_cell_offsets : wp.array, shape (total_atoms,), dtype=wp.vec3i
        Integer cell offsets for each atom.
    cutoff_sq : float
        Squared cutoff distance for neighbor detection.
    cell : wp.array, shape (1, 3, 3), dtype=wp.mat33*
        Cell matrix defining lattice vectors.
    shift_range : wp.array, shape (1, 3), dtype=wp.vec3i
        Shift range per dimension for the single system.
    neighbor_matrix : wp.array, shape (total_atoms, max_neighbors), dtype=wp.int32
        OUTPUT: Neighbor matrix to be filled with neighbor atom indices.
    neighbor_matrix_shifts : wp.array, shape (total_atoms, max_neighbors), dtype=wp.vec3i
        OUTPUT: Shift vectors for each neighbor relationship.
    num_neighbors : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Number of neighbors found for each atom.
    half_fill : wp.bool
        If True, only store relationships where i < j.
    rebuild_flags : wp.array, shape (1,), dtype=wp.bool
        When False the kernel returns immediately.

    Notes
    -----
    - Thread launch: 2D (num_shifts, total_atoms)
    - GPU-side conditional: no CPU-GPU synchronization occurs
    """
    ishift, iatom = wp.tid()
    if not rebuild_flags[0]:
        return
    shift = _decode_shift_index(ishift, shift_range[0])
    _naive_neighbor_pbc_body(
        shift,
        iatom,
        positions,
        per_atom_cell_offsets,
        cutoff_sq,
        cell,
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
        half_fill,
    )


@wp.kernel(enable_backward=False)
def _fill_naive_neighbor_matrix_pbc_prewrapped(
    positions: wp.array(dtype=Any),
    cutoff_sq: Any,
    cell: wp.array(dtype=Any),
    shift_range: wp.array(dtype=wp.vec3i),
    neighbor_matrix: wp.array2d(dtype=wp.int32),
    neighbor_matrix_shifts: wp.array2d(dtype=wp.vec3i),
    num_neighbors: wp.array(dtype=wp.int32),
    half_fill: wp.bool,
) -> None:
    """PBC neighbor matrix for pre-wrapped positions (no cell-offset correction).

    Notes
    -----
    - Thread launch: 2D (num_shifts, total_atoms)
    """
    ishift, iatom = wp.tid()
    shift = _decode_shift_index(ishift, shift_range[0])
    _naive_neighbor_pbc_body_prewrapped(
        shift,
        iatom,
        positions,
        cutoff_sq,
        cell,
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
        half_fill,
    )


@wp.kernel(enable_backward=False)
def _fill_naive_neighbor_matrix_pbc_prewrapped_selective(
    positions: wp.array(dtype=Any),
    cutoff_sq: Any,
    cell: wp.array(dtype=Any),
    shift_range: wp.array(dtype=wp.vec3i),
    neighbor_matrix: wp.array2d(dtype=wp.int32),
    neighbor_matrix_shifts: wp.array2d(dtype=wp.vec3i),
    num_neighbors: wp.array(dtype=wp.int32),
    half_fill: wp.bool,
    rebuild_flags: wp.array(dtype=wp.bool),
) -> None:
    """Selective PBC kernel for pre-wrapped positions - skips when not rebuilding.

    Notes
    -----
    - Thread launch: 2D (num_shifts, total_atoms)
    """
    ishift, iatom = wp.tid()
    if not rebuild_flags[0]:
        return
    shift = _decode_shift_index(ishift, shift_range[0])
    _naive_neighbor_pbc_body_prewrapped(
        shift,
        iatom,
        positions,
        cutoff_sq,
        cell,
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
        half_fill,
    )


## Generate overloads for all kernels
T = [wp.float32, wp.float64, wp.float16]
V = [wp.vec3f, wp.vec3d, wp.vec3h]
M = [wp.mat33f, wp.mat33d, wp.mat33h]
_fill_naive_neighbor_matrix_overload = {}
_fill_naive_neighbor_matrix_pbc_overload = {}
_fill_naive_neighbor_matrix_pbc_prewrapped_overload = {}
_fill_naive_neighbor_matrix_selective_overload = {}
_fill_naive_neighbor_matrix_pbc_selective_overload = {}
_fill_naive_neighbor_matrix_pbc_prewrapped_selective_overload = {}
for t, v, m in zip(T, V, M):
    _fill_naive_neighbor_matrix_overload[t] = wp.overload(
        _fill_naive_neighbor_matrix,
        [
            wp.array(dtype=v),
            t,
            wp.array2d(dtype=wp.int32),
            wp.array(dtype=wp.int32),
            wp.bool,
        ],
    )
    _fill_naive_neighbor_matrix_pbc_overload[t] = wp.overload(
        _fill_naive_neighbor_matrix_pbc,
        [
            wp.array(dtype=v),
            wp.array(dtype=wp.vec3i),
            t,
            wp.array(dtype=m),
            wp.array(dtype=wp.vec3i),
            wp.array2d(dtype=wp.int32),
            wp.array2d(dtype=wp.vec3i),
            wp.array(dtype=wp.int32),
            wp.bool,
        ],
    )
    _fill_naive_neighbor_matrix_selective_overload[t] = wp.overload(
        _fill_naive_neighbor_matrix_selective,
        [
            wp.array(dtype=v),
            t,
            wp.array2d(dtype=wp.int32),
            wp.array(dtype=wp.int32),
            wp.bool,
            wp.array(dtype=wp.bool),
        ],
    )
    _fill_naive_neighbor_matrix_pbc_selective_overload[t] = wp.overload(
        _fill_naive_neighbor_matrix_pbc_selective,
        [
            wp.array(dtype=v),
            wp.array(dtype=wp.vec3i),
            t,
            wp.array(dtype=m),
            wp.array(dtype=wp.vec3i),
            wp.array2d(dtype=wp.int32),
            wp.array2d(dtype=wp.vec3i),
            wp.array(dtype=wp.int32),
            wp.bool,
            wp.array(dtype=wp.bool),
        ],
    )
    _fill_naive_neighbor_matrix_pbc_prewrapped_overload[t] = wp.overload(
        _fill_naive_neighbor_matrix_pbc_prewrapped,
        [
            wp.array(dtype=v),
            t,
            wp.array(dtype=m),
            wp.array(dtype=wp.vec3i),
            wp.array2d(dtype=wp.int32),
            wp.array2d(dtype=wp.vec3i),
            wp.array(dtype=wp.int32),
            wp.bool,
        ],
    )
    _fill_naive_neighbor_matrix_pbc_prewrapped_selective_overload[t] = wp.overload(
        _fill_naive_neighbor_matrix_pbc_prewrapped_selective,
        [
            wp.array(dtype=v),
            t,
            wp.array(dtype=m),
            wp.array(dtype=wp.vec3i),
            wp.array2d(dtype=wp.int32),
            wp.array2d(dtype=wp.vec3i),
            wp.array(dtype=wp.int32),
            wp.bool,
            wp.array(dtype=wp.bool),
        ],
    )

###########################################################################################
########################### Warp Launchers ###############################################
###########################################################################################


def naive_neighbor_matrix(
    positions: wp.array,
    cutoff: float,
    neighbor_matrix: wp.array,
    num_neighbors: wp.array,
    wp_dtype: type,
    device: str,
    half_fill: bool = False,
    rebuild_flags: wp.array | None = None,
) -> None:
    """Core warp launcher for naive neighbor matrix construction (no PBC).

    Computes pairwise distances and fills the neighbor matrix with atom indices
    within the cutoff distance using pure warp operations. No periodic boundary
    conditions are applied.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Atomic coordinates in Cartesian space.
    cutoff : float
        Cutoff distance for neighbor detection in Cartesian units.
        Must be positive. Atoms within this distance are considered neighbors.
    neighbor_matrix : wp.array, shape (total_atoms, max_neighbors), dtype=wp.int32
        OUTPUT: Neighbor matrix to be filled with neighbor atom indices.
        Must be pre-allocated. Entries are filled with atom indices.
    num_neighbors : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Number of neighbors found for each atom.
        Must be pre-allocated. Updated in-place with actual neighbor counts.
    wp_dtype : type
        Warp dtype (wp.float32, wp.float64, or wp.float16).
    device : str
        Warp device string (e.g., 'cuda:0', 'cpu').
    half_fill : bool, default=False
        If True, only store relationships where i < j to avoid double counting.
        If False, store all neighbor relationships symmetrically.

    Notes
    -----
    - This is a low-level warp interface. For framework bindings, use torch/jax wrappers.
    - Output arrays must be pre-allocated by caller.

    See Also
    --------
    naive_neighbor_matrix_pbc : Version with periodic boundary conditions
    _fill_naive_neighbor_matrix : Kernel that performs the computation
    _fill_naive_neighbor_matrix_selective : Selective-skip kernel variant
    wrap_positions_single : Preprocessing step that wraps positions
    """
    total_atoms = positions.shape[0]

    if rebuild_flags is None:
        wp.launch(
            kernel=_fill_naive_neighbor_matrix_overload[wp_dtype],
            dim=total_atoms,
            inputs=[
                positions,
                wp_dtype(cutoff * cutoff),
                neighbor_matrix,
                num_neighbors,
                half_fill,
            ],
            device=device,
        )
    else:
        wp.launch(
            kernel=_fill_naive_neighbor_matrix_selective_overload[wp_dtype],
            dim=total_atoms,
            inputs=[
                positions,
                wp_dtype(cutoff * cutoff),
                neighbor_matrix,
                num_neighbors,
                half_fill,
                rebuild_flags,
            ],
            device=device,
        )


def naive_neighbor_matrix_pbc(
    positions: wp.array,
    cutoff: float,
    cell: wp.array,
    shift_range: wp.array,
    num_shifts: int,
    neighbor_matrix: wp.array,
    neighbor_matrix_shifts: wp.array,
    num_neighbors: wp.array,
    wp_dtype: type,
    device: str,
    half_fill: bool = False,
    rebuild_flags: wp.array | None = None,
    wrap_positions: bool = True,
) -> None:
    """Core warp launcher for naive neighbor matrix construction with PBC.

    Computes neighbor relationships between atoms across periodic boundaries using
    pure warp operations.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Atomic coordinates in Cartesian space.
    cutoff : float
        Cutoff distance for neighbor detection in Cartesian units.
    cell : wp.array, shape (1, 3, 3), dtype=wp.mat33*
        Cell matrix defining lattice vectors in Cartesian coordinates.
    shift_range : wp.array, shape (1, 3), dtype=wp.vec3i
        Shift range per dimension for the single system.
    num_shifts : int
        Number of periodic shifts for the single system.
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
    half_fill : bool, default=False
        If True, only store relationships where i < j.
    rebuild_flags : wp.array, shape (1,), dtype=wp.bool, optional
        When provided, the kernel checks this flag on the GPU and skips
        work when False (no CPU-GPU sync).
    wrap_positions : bool, default=True
        If True, wrap input positions into the primary cell before
        neighbor search.

    Notes
    -----
    - This is a low-level warp interface. For framework bindings, use torch/jax wrappers.
    - Output arrays must be pre-allocated by caller.
    - When ``wrap_positions`` is True, positions are wrapped into the primary cell in a
      preprocessing step before the neighbor search kernel.

    See Also
    --------
    naive_neighbor_matrix : Version without periodic boundary conditions
    _fill_naive_neighbor_matrix_pbc : Kernel that performs the computation
    _fill_naive_neighbor_matrix_pbc_selective : Selective-skip kernel variant
    wrap_positions_single : Preprocessing step that wraps positions
    """
    total_atoms = positions.shape[0]

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
        per_atom_cell_offsets = wp.empty((total_atoms,), dtype=wp.vec3i, device=device)
        wrap_positions_single(
            positions,
            cell,
            inv_cell,
            positions_wrapped,
            per_atom_cell_offsets,
            wp_dtype,
            device,
        )

        if rebuild_flags is None:
            wp.launch(
                kernel=_fill_naive_neighbor_matrix_pbc_overload[wp_dtype],
                dim=(num_shifts, total_atoms),
                inputs=[
                    positions_wrapped,
                    per_atom_cell_offsets,
                    wp_dtype(cutoff * cutoff),
                    cell,
                    shift_range,
                    neighbor_matrix,
                    neighbor_matrix_shifts,
                    num_neighbors,
                    half_fill,
                ],
                device=device,
            )
        else:
            wp.launch(
                kernel=_fill_naive_neighbor_matrix_pbc_selective_overload[wp_dtype],
                dim=(num_shifts, total_atoms),
                inputs=[
                    positions_wrapped,
                    per_atom_cell_offsets,
                    wp_dtype(cutoff * cutoff),
                    cell,
                    shift_range,
                    neighbor_matrix,
                    neighbor_matrix_shifts,
                    num_neighbors,
                    half_fill,
                    rebuild_flags,
                ],
                device=device,
            )
    else:
        if rebuild_flags is None:
            wp.launch(
                kernel=_fill_naive_neighbor_matrix_pbc_prewrapped_overload[wp_dtype],
                dim=(num_shifts, total_atoms),
                inputs=[
                    positions,
                    wp_dtype(cutoff * cutoff),
                    cell,
                    shift_range,
                    neighbor_matrix,
                    neighbor_matrix_shifts,
                    num_neighbors,
                    half_fill,
                ],
                device=device,
            )
        else:
            wp.launch(
                kernel=_fill_naive_neighbor_matrix_pbc_prewrapped_selective_overload[
                    wp_dtype
                ],
                dim=(num_shifts, total_atoms),
                inputs=[
                    positions,
                    wp_dtype(cutoff * cutoff),
                    cell,
                    shift_range,
                    neighbor_matrix,
                    neighbor_matrix_shifts,
                    num_neighbors,
                    half_fill,
                    rebuild_flags,
                ],
                device=device,
            )
