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

"""Generated Warp kernels and factories for naive neighbor lists."""

from functools import lru_cache
from typing import Any, Literal

import warp as wp

from nvalchemiops.neighbors.naive.dispatch import (
    _NaiveStrategy,
    _parse_pbc_mode,
    _parse_strategy,
    _PBCMode,
)
from nvalchemiops.neighbors.neighbor_utils import (
    DTYPE_INFO_ALL,
    _append_specialization_doc,
    _correct_shift,
    _decode_full_shift_index,
    _decode_shift_index,
    _shifted_position,
    _update_dual_neighbor_matrix,
    _update_neighbor_matrix,
    kernel_specialization_name,
    require_supported_dtype,
    set_fn_doc,
    set_fn_name,
)

__all__ = [
    "get_naive_neighbor_matrix_dual_cutoff_kernel",
    "get_naive_neighbor_matrix_kernel",
]

_SUPPORTED_DTYPES = (wp.float16, wp.float32, wp.float64)
_DTYPE_INFO: dict[type, tuple[type, type]] = {
    dtype: DTYPE_INFO_ALL[dtype] for dtype in _SUPPORTED_DTYPES
}

BLOCK_DIM = 64


def _require_supported_dtype(wp_dtype: type) -> None:
    """Validate a native scalar dtype supported by naive kernels."""
    require_supported_dtype(wp_dtype, _SUPPORTED_DTYPES)


def _pair_output_features(
    *,
    partial: bool = False,
    return_vectors: bool = False,
    return_distances: bool = False,
    pair_fn: wp.Function | None = None,
) -> tuple[str, ...]:
    """Return specialization-name tokens for pair-output options."""
    return tuple(
        feature
        for feature in (
            "partial" if partial else "",
            "vectors" if return_vectors else "",
            "distances" if return_distances else "",
            "pair_fn" if pair_fn is not None else "",
        )
        if feature
    )


def _naive_kernel_base_name(
    operation: Literal["single_cutoff", "dual_cutoff"],
    *,
    pbc_mode: _PBCMode,
    batched: bool,
    selective: bool,
) -> str:
    """Return the 0.3.1-style base name for a naive specialization."""
    base = (
        "_fill_batch_naive_neighbor_matrix"
        if batched
        else "_fill_naive_neighbor_matrix"
    )
    if pbc_mode is not _PBCMode.NONE:
        base = f"{base}_pbc"
    if operation == "dual_cutoff":
        base = f"{base}_dual_cutoff"
    if pbc_mode is _PBCMode.PREWRAPPED:
        base = f"{base}_prewrapped"
    if selective:
        base = f"{base}_selective"
    return base


@wp.func
def _write_neighbor_slot(
    row: int,
    nbr: int,
    shift: wp.vec3i,
    displacement: Any,
    distance: Any,
    pair_energy: Any,
    pair_force: Any,
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    neighbor_vectors: wp.array(dtype=Any, ndim=2),
    neighbor_distances: wp.array(dtype=Any, ndim=2),
    pair_energies: wp.array(dtype=Any, ndim=2),
    pair_forces: wp.array(dtype=Any, ndim=2),
    max_neighbors: int,
    pbc: bool,
    return_vectors: bool,
    return_distances: bool,
    has_pair_fn: bool,
):
    """Store one accepted naive neighbor slot

    Parameters
    ----------
    row : int
        Output row to update.
    nbr : int
        Neighbor atom index written to ``neighbor_matrix``.
    shift : wp.vec3i
        Periodic unit shift stored in PBC mode.
    displacement : wp.vec3*
        Pair displacement vector.
    distance : float
        Pair distance.
    pair_energy : float
        Pair-function energy to store when ``has_pair_fn`` is true.
    pair_force : wp.vec3*
        Pair-function force to store when ``has_pair_fn`` is true.
    neighbor_matrix : wp.array, shape (rows, max_neighbors), dtype=wp.int32
        OUTPUT: Neighbor atom indices.
    neighbor_matrix_shifts : wp.array, shape (rows, max_neighbors), dtype=wp.vec3i
        OUTPUT: Periodic shift vectors in PBC mode.
    num_neighbors : wp.array, shape (rows,), dtype=wp.int32
        MODIFIED: Per-row neighbor counts.
    neighbor_vectors : wp.array, shape (rows, max_neighbors), dtype=wp.vec3*
        OUTPUT: Optional displacement vectors.
    neighbor_distances : wp.array, shape (rows, max_neighbors), dtype=wp.float*
        OUTPUT: Optional pair distances.
    pair_energies : wp.array, shape (rows, max_neighbors), dtype=wp.float*
        OUTPUT: Optional pair-function energies.
    pair_forces : wp.array, shape (rows, max_neighbors), dtype=wp.vec3*
        OUTPUT: Optional pair-function forces.
    max_neighbors : int
        Maximum writable slots per row.
    pbc : bool
        If True, write ``neighbor_matrix_shifts``.
    return_vectors : bool
        If True, write ``neighbor_vectors``.
    return_distances : bool
        If True, write ``neighbor_distances``.
    has_pair_fn : bool
        If True, write ``pair_energies`` and ``pair_forces``.

    Returns
    -------
    None
        This function modifies the input arrays in-place.

    Notes
    -----
    - Modifies: neighbor_matrix, num_neighbors, and enabled optional output buffers.
    """
    pos = wp.atomic_add(num_neighbors, row, 1)
    if pos < max_neighbors:
        neighbor_matrix[row, pos] = nbr
        if pbc:
            neighbor_matrix_shifts[row, pos] = shift
        if return_vectors:
            neighbor_vectors[row, pos] = displacement
        if return_distances:
            neighbor_distances[row, pos] = distance
        if has_pair_fn:
            pair_energies[row, pos] = pair_energy
            pair_forces[row, pos] = pair_force


@wp.func
def _write_neighbor_pair(
    row: int,
    nbr: int,
    symmetric_row: int,
    symmetric_nbr: int,
    shift: wp.vec3i,
    displacement: Any,
    distance: Any,
    pair_energy: Any,
    pair_force: Any,
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    neighbor_vectors: wp.array(dtype=Any, ndim=2),
    neighbor_distances: wp.array(dtype=Any, ndim=2),
    pair_energies: wp.array(dtype=Any, ndim=2),
    pair_forces: wp.array(dtype=Any, ndim=2),
    max_neighbors: int,
    pbc: bool,
    half_fill: bool,
    partial: bool,
    return_vectors: bool,
    return_distances: bool,
    has_pair_fn: bool,
):
    """Store one accepted naive neighbor pair

    Parameters
    ----------
    row : int
        Primary output row to update.
    nbr : int
        Neighbor atom index for the primary row.
    symmetric_row : int
        Symmetric output row for full-fill mode.
    symmetric_nbr : int
        Neighbor atom index for the symmetric row.
    shift : wp.vec3i
        Periodic unit shift stored with the primary pair.
    displacement : wp.vec3*
        Pair displacement vector for the primary pair.
    distance : float
        Pair distance.
    pair_energy : float
        Pair-function energy to store when ``has_pair_fn`` is true.
    pair_force : wp.vec3*
        Pair-function force for the primary pair.
    neighbor_matrix : wp.array, shape (rows, max_neighbors), dtype=wp.int32
        OUTPUT: Neighbor atom indices.
    neighbor_matrix_shifts : wp.array, shape (rows, max_neighbors), dtype=wp.vec3i
        OUTPUT: Periodic shift vectors in PBC mode.
    num_neighbors : wp.array, shape (rows,), dtype=wp.int32
        MODIFIED: Per-row neighbor counts.
    neighbor_vectors : wp.array, shape (rows, max_neighbors), dtype=wp.vec3*
        OUTPUT: Optional displacement vectors.
    neighbor_distances : wp.array, shape (rows, max_neighbors), dtype=wp.float*
        OUTPUT: Optional pair distances.
    pair_energies : wp.array, shape (rows, max_neighbors), dtype=wp.float*
        OUTPUT: Optional pair-function energies.
    pair_forces : wp.array, shape (rows, max_neighbors), dtype=wp.vec3*
        OUTPUT: Optional pair-function forces.
    max_neighbors : int
        Maximum writable slots per row.
    pbc : bool
        If True, write ``neighbor_matrix_shifts``.
    half_fill : bool
        If True, suppress the symmetric write.
    partial : bool
        If True, suppress non-target symmetric rows.
    return_vectors : bool
        If True, write ``neighbor_vectors``.
    return_distances : bool
        If True, write ``neighbor_distances``.
    has_pair_fn : bool
        If True, write ``pair_energies`` and ``pair_forces``.

    Returns
    -------
    None
        This function modifies the input arrays in-place.

    Notes
    -----
    - Modifies: neighbor_matrix, num_neighbors, and enabled optional output buffers.
    - In full-fill mode the symmetric slot stores negated shift, displacement, and force.
    """
    _write_neighbor_slot(
        row,
        nbr,
        shift,
        displacement,
        distance,
        pair_energy,
        pair_force,
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
        neighbor_vectors,
        neighbor_distances,
        pair_energies,
        pair_forces,
        max_neighbors,
        pbc,
        return_vectors,
        return_distances,
        has_pair_fn,
    )
    if not partial and not half_fill:
        _write_neighbor_slot(
            symmetric_row,
            symmetric_nbr,
            -shift,
            -displacement,
            distance,
            pair_energy,
            -pair_force,
            neighbor_matrix,
            neighbor_matrix_shifts,
            num_neighbors,
            neighbor_vectors,
            neighbor_distances,
            pair_energies,
            pair_forces,
            max_neighbors,
            pbc,
            return_vectors,
            return_distances,
            has_pair_fn,
        )


@wp.func
def _store_cutoff_pair(
    atom_i: int,
    atom_j: int,
    dist_sq: Any,
    cutoff1_sq: Any,
    cutoff2_sq: Any,
    shift: wp.vec3i,
    neighbor_matrix1: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts1: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors1: wp.array(dtype=wp.int32),
    neighbor_matrix2: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts2: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors2: wp.array(dtype=wp.int32),
    half_fill: bool,
    dual_cutoff: bool,
    pbc: bool,
):
    """Store one accepted pair for single- or dual-cutoff matrix outputs

    Parameters
    ----------
    atom_i : int
        Source atom row.
    atom_j : int
        Neighbor atom index.
    dist_sq : float
        Squared pair distance.
    cutoff1_sq : float
        Squared primary cutoff distance.
    cutoff2_sq : float
        Squared secondary cutoff distance for dual-cutoff mode.
    shift : wp.vec3i
        Periodic unit shift stored in PBC mode.
    neighbor_matrix1 : wp.array, shape (rows, max_neighbors), dtype=wp.int32
        OUTPUT: Primary neighbor matrix.
    neighbor_matrix_shifts1 : wp.array, shape (rows, max_neighbors), dtype=wp.vec3i
        OUTPUT: Primary shift matrix for PBC mode.
    num_neighbors1 : wp.array, shape (rows,), dtype=wp.int32
        MODIFIED: Primary neighbor counts.
    neighbor_matrix2 : wp.array, shape (rows, max_neighbors), dtype=wp.int32
        OUTPUT: Secondary neighbor matrix for dual-cutoff mode.
    neighbor_matrix_shifts2 : wp.array, shape (rows, max_neighbors), dtype=wp.vec3i
        OUTPUT: Secondary shift matrix for dual-cutoff PBC mode.
    num_neighbors2 : wp.array, shape (rows,), dtype=wp.int32
        MODIFIED: Secondary neighbor counts.
    half_fill : bool
        If True, store only one direction for each unordered pair.
    dual_cutoff : bool
        If True, write secondary cutoff outputs and maybe primary cutoff outputs.
    pbc : bool
        If True, write shift matrices alongside atom indices.

    Returns
    -------
    None
        This function modifies the input arrays in-place.

    Notes
    -----
    - Modifies: primary outputs and, in dual-cutoff mode, secondary outputs.
    """
    max_neighbors1 = neighbor_matrix1.shape[1]
    max_neighbors2 = neighbor_matrix2.shape[1]
    if dual_cutoff:
        _update_dual_neighbor_matrix(
            atom_i,
            atom_j,
            dist_sq,
            cutoff1_sq,
            cutoff2_sq,
            neighbor_matrix1,
            neighbor_matrix_shifts1,
            num_neighbors1,
            max_neighbors1,
            neighbor_matrix2,
            neighbor_matrix_shifts2,
            num_neighbors2,
            max_neighbors2,
            shift,
            half_fill,
            pbc,
        )
    elif dist_sq < cutoff1_sq:
        _update_neighbor_matrix(
            atom_i,
            atom_j,
            neighbor_matrix1,
            neighbor_matrix_shifts1,
            num_neighbors1,
            shift,
            max_neighbors1,
            half_fill,
            pbc,
        )


@lru_cache(maxsize=None)
def _make_scalar_kernel(
    operation: Literal["single_cutoff", "dual_cutoff"],
    wp_dtype: type,
    *,
    pbc_mode: _PBCMode,
    batched: bool,
    half_fill: bool,
    selective: bool,
    partial: bool,
    return_vectors: bool,
    return_distances: bool,
    pair_fn: wp.Function | None,
) -> wp.Kernel:
    """Return a cached scalar kernel for the selected static options."""
    _require_supported_dtype(wp_dtype)
    has_pair_outputs = (
        partial or return_vectors or return_distances or pair_fn is not None
    )
    if has_pair_outputs and operation != "single_cutoff":
        raise ValueError("pair outputs are only valid for single_cutoff")

    vec_dtype, mat_dtype = _DTYPE_INFO[wp_dtype]
    DUAL_CUTOFF = operation == "dual_cutoff"
    PBC = pbc_mode is not _PBCMode.NONE
    BATCHED = bool(batched)
    SELECTIVE = bool(selective)
    PARTIAL = bool(partial)
    HALF_FILL = wp.constant(bool(half_fill))
    WRAP_ON_ENTRY = pbc_mode is _PBCMode.WRAP_ON_ENTRY
    RETURN_VECTORS = bool(return_vectors)
    RETURN_DISTANCES = bool(return_distances)
    HAS_PAIR_FN = pair_fn is not None
    HAS_PAIR_OUTPUTS = bool(has_pair_outputs)

    @wp.kernel(enable_backward=False, module="naive_scalar")
    def _kernel(
        positions: wp.array(dtype=vec_dtype),
        per_atom_cell_offsets: wp.array(dtype=wp.vec3i),
        cutoff1_sq: wp_dtype,
        cutoff2_sq: wp_dtype,
        cell: wp.array(dtype=mat_dtype),
        shift_range: wp.array(dtype=wp.vec3i),
        num_shifts_arr: wp.array(dtype=wp.int32),
        batch_idx: wp.array(dtype=wp.int32),
        batch_ptr: wp.array(dtype=wp.int32),
        target_indices: wp.array(dtype=wp.int32),
        neighbor_matrix1: wp.array(dtype=wp.int32, ndim=2),
        neighbor_matrix_shifts1: wp.array(dtype=wp.vec3i, ndim=2),
        num_neighbors1: wp.array(dtype=wp.int32),
        neighbor_matrix2: wp.array(dtype=wp.int32, ndim=2),
        neighbor_matrix_shifts2: wp.array(dtype=wp.vec3i, ndim=2),
        num_neighbors2: wp.array(dtype=wp.int32),
        neighbor_vectors: wp.array(dtype=vec_dtype, ndim=2),
        neighbor_distances: wp.array(dtype=wp_dtype, ndim=2),
        pair_params: wp.array(dtype=wp_dtype, ndim=2),
        pair_energies: wp.array(dtype=wp_dtype, ndim=2),
        pair_forces: wp.array(dtype=vec_dtype, ndim=2),
        rebuild_flags: wp.array(dtype=wp.bool),
    ) -> None:
        """Calculate neighbor matrix using naive O(N^2) algorithm

        Parameters
        ----------
        positions : wp.array, shape (total_atoms,), dtype=wp.vec3*
            Atomic coordinates in Cartesian space.
        per_atom_cell_offsets : wp.array, shape (total_atoms,), dtype=wp.vec3i
            Integer cell offsets for wrap-on-entry PBC. Zero-size sentinel for
            no-PBC and prewrapped PBC specializations.
        cutoff1_sq : float
            Squared primary cutoff distance.
        cutoff2_sq : float
            Squared secondary cutoff distance for dual-cutoff specializations.
            Sentinel value for single-cutoff specializations.
        cell : wp.array, shape (num_systems,), dtype=wp.mat33*
            Cell matrices for PBC specializations. Zero-size sentinel for no-PBC.
        shift_range : wp.array, shape (num_systems,), dtype=wp.vec3i
            Encoded PBC shift ranges per system. Zero-size sentinel for no-PBC.
        num_shifts_arr : wp.array, shape (num_systems,), dtype=wp.int32
            Number of active PBC shifts per system in batched PBC mode.
        batch_idx : wp.array, shape (total_atoms,), dtype=wp.int32
            System index for each atom. Zero-size sentinel for single-system modes.
        batch_ptr : wp.array, shape (num_systems + 1,), dtype=wp.int32
            Prefix offsets delimiting atoms per system in batched modes.
        target_indices : wp.array, shape (n_targets,), dtype=wp.int32
            Compact target rows for partial neighbor lists. Zero-size sentinel
            for full neighbor-list specializations.
        neighbor_matrix1 : wp.array, shape (rows, max_neighbors), dtype=wp.int32
            OUTPUT: Primary neighbor matrix.
        neighbor_matrix_shifts1 : wp.array, shape (rows, max_neighbors), dtype=wp.vec3i
            OUTPUT: Primary periodic shift matrix for PBC modes.
        num_neighbors1 : wp.array, shape (rows,), dtype=wp.int32
            OUTPUT: Primary neighbor counts.
        neighbor_matrix2 : wp.array, shape (rows, max_neighbors), dtype=wp.int32
            OUTPUT: Secondary neighbor matrix for dual-cutoff modes.
        neighbor_matrix_shifts2 : wp.array, shape (rows, max_neighbors), dtype=wp.vec3i
            OUTPUT: Secondary periodic shift matrix for dual-cutoff PBC modes.
        num_neighbors2 : wp.array, shape (rows,), dtype=wp.int32
            OUTPUT: Secondary neighbor counts for dual-cutoff modes.
        neighbor_vectors : wp.array, shape (rows, max_neighbors), dtype=wp.vec3*
            OUTPUT: Pair displacement vectors when requested; sentinel otherwise.
        neighbor_distances : wp.array, shape (rows, max_neighbors), dtype=wp.float*
            OUTPUT: Pair distances when requested; sentinel otherwise.
        pair_params : wp.array, shape (total_atoms, K), dtype=wp.float*
            Pair-function parameters. Zero-size sentinel when no pair function is active.
        pair_energies : wp.array, shape (rows, max_neighbors), dtype=wp.float*
            OUTPUT: Pair-function energies. Sentinel when no pair function is active.
        pair_forces : wp.array, shape (rows, max_neighbors), dtype=wp.vec3*
            OUTPUT: Pair-function forces. Sentinel when no pair function is active.
        rebuild_flags : wp.array, shape (num_systems,), dtype=wp.bool
            Selective rebuild flags. Sentinel for non-selective specializations.

        Returns
        -------
        None
            This function modifies the input arrays in-place.

        Notes
        -----
        - Thread launch: Uniform 3D launch ``(system, shift, row)``. Inactive dimensions use sentinel extent 1.
        - Modifies: ``neighbor_matrix1``, ``neighbor_matrix_shifts1``, ``num_neighbors1``, optional dual-cutoff outputs, and enabled pair-output buffers.
        PBC, batching, selective rebuild, partial rows, dual cutoff, and pair
        outputs are static factory specializations. Inactive inputs are
        zero-size sentinels and are not read by the matching specialization.

        See Also
        --------
        get_naive_neighbor_matrix_kernel : Return the specialized naive neighbor-matrix kernel.
        """
        isys_tid, ishift, row_tid = wp.tid()

        if HAS_PAIR_OUTPUTS:
            row = row_tid
            atom_i = row
            if PARTIAL:
                atom_i = target_indices[row]

            isys = wp.int32(0)
            j_start = wp.int32(0)
            j_end = positions.shape[0]
            if BATCHED:
                if PBC and not PARTIAL:
                    # BATCHED+PBC launch_dim is
                    # ``(num_systems, max_shifts, max_atoms_per_system)``
                    # so ``row_tid`` is per-system; derive ``isys`` from
                    # the system launch dim, not from ``batch_idx[row_tid]``
                    # (which would alias to system 0 for every isys_tid).
                    isys = isys_tid
                else:
                    # BATCHED+no-PBC launch is ``(1, 1, total_atoms)``
                    # (so ``row_tid`` is global), and BATCHED+PARTIAL
                    # treats ``atom_i = target_indices[row]`` as a global
                    # atom index — both safe to look up via batch_idx.
                    isys = batch_idx[atom_i]
                j_start = batch_ptr[isys]
                j_end = batch_ptr[isys + 1]
            if SELECTIVE and not rebuild_flags[isys]:
                return

            max_neighbors = neighbor_matrix1.shape[1]
            if PBC:
                if BATCHED:
                    active_shift_count = num_shifts_arr[isys]
                    if PARTIAL and not HALF_FILL:
                        active_shift_count = 2 * active_shift_count - 1
                    if ishift >= active_shift_count:
                        return

                atom_nbr = row_tid
                if not PARTIAL and BATCHED:
                    natom = batch_ptr[isys + 1] - batch_ptr[isys]
                    if row_tid >= natom:
                        return
                    atom_nbr = row_tid + batch_ptr[isys]
                    j_start = batch_ptr[isys]
                    j_end = batch_ptr[isys + 1]

                shift = wp.vec3i(0, 0, 0)
                if PARTIAL and not HALF_FILL:
                    if ishift > 0:
                        shift = _decode_full_shift_index(ishift - 1, shift_range[isys])
                else:
                    shift = _decode_shift_index(ishift, shift_range[isys])

                zero_shift = shift[0] == 0 and shift[1] == 0 and shift[2] == 0
                current_cell = cell[isys]
                if PARTIAL:
                    row_pos = positions[atom_i]
                    for atom_nbr in range(j_start, j_end):
                        if zero_shift and atom_nbr == atom_i:
                            continue
                        if HALF_FILL and zero_shift and atom_nbr <= atom_i:
                            continue
                        nbr_image = _shifted_position(
                            shift, current_cell, positions[atom_nbr]
                        )
                        displacement = nbr_image - row_pos
                        dist_sq = wp.length_sq(displacement)
                        if dist_sq < cutoff1_sq:
                            distance = wp_dtype(0.0)
                            if RETURN_DISTANCES or HAS_PAIR_FN:
                                distance = wp.sqrt(dist_sq)
                            corrected_shift = shift
                            if WRAP_ON_ENTRY:
                                row_offset = per_atom_cell_offsets[atom_i]
                                nbr_offset = per_atom_cell_offsets[atom_nbr]
                                corrected_shift = _correct_shift(
                                    shift, row_offset, nbr_offset
                                )
                            pair_energy = wp_dtype(0.0)
                            pair_force = vec_dtype()
                            if HAS_PAIR_FN:
                                pair_energy, pair_force = pair_fn(
                                    displacement,
                                    distance,
                                    pair_params,
                                    atom_i,
                                    atom_nbr,
                                )
                            _write_neighbor_pair(
                                row,
                                atom_nbr,
                                atom_nbr,
                                atom_i,
                                corrected_shift,
                                displacement,
                                distance,
                                pair_energy,
                                pair_force,
                                neighbor_matrix1,
                                neighbor_matrix_shifts1,
                                num_neighbors1,
                                neighbor_vectors,
                                neighbor_distances,
                                pair_energies,
                                pair_forces,
                                max_neighbors,
                                True,
                                True,
                                True,
                                RETURN_VECTORS,
                                RETURN_DISTANCES,
                                HAS_PAIR_FN,
                            )
                    return

                if zero_shift:
                    j_end = atom_nbr
                nbr_image = _shifted_position(shift, current_cell, positions[atom_nbr])
                for atom_row in range(j_start, j_end):
                    displacement = nbr_image - positions[atom_row]
                    dist_sq = wp.length_sq(displacement)
                    if dist_sq < cutoff1_sq:
                        distance = wp_dtype(0.0)
                        if RETURN_DISTANCES or HAS_PAIR_FN:
                            distance = wp.sqrt(dist_sq)
                        corrected_shift = shift
                        if WRAP_ON_ENTRY:
                            nbr_offset = per_atom_cell_offsets[atom_nbr]
                            row_offset = per_atom_cell_offsets[atom_row]
                            corrected_shift = _correct_shift(
                                shift, row_offset, nbr_offset
                            )
                        pair_energy = wp_dtype(0.0)
                        pair_force = vec_dtype()
                        if HAS_PAIR_FN:
                            pair_energy, pair_force = pair_fn(
                                displacement,
                                distance,
                                pair_params,
                                atom_row,
                                atom_nbr,
                            )
                        _write_neighbor_pair(
                            atom_row,
                            atom_nbr,
                            atom_nbr,
                            atom_row,
                            corrected_shift,
                            displacement,
                            distance,
                            pair_energy,
                            pair_force,
                            neighbor_matrix1,
                            neighbor_matrix_shifts1,
                            num_neighbors1,
                            neighbor_vectors,
                            neighbor_distances,
                            pair_energies,
                            pair_forces,
                            max_neighbors,
                            True,
                            HALF_FILL,
                            False,
                            RETURN_VECTORS,
                            RETURN_DISTANCES,
                            HAS_PAIR_FN,
                        )
                return

            if not PARTIAL:
                j_start = atom_i + 1
            pos_i = positions[atom_i]
            for atom_j in range(j_start, j_end):
                if atom_j == atom_i:
                    continue
                if PARTIAL and HALF_FILL and atom_j <= atom_i:
                    continue
                displacement = positions[atom_j] - pos_i
                dist_sq = wp.length_sq(displacement)
                if dist_sq < cutoff1_sq:
                    distance = wp_dtype(0.0)
                    if RETURN_DISTANCES or HAS_PAIR_FN:
                        distance = wp.sqrt(dist_sq)
                    pair_energy = wp_dtype(0.0)
                    pair_force = vec_dtype()
                    if HAS_PAIR_FN:
                        pair_energy, pair_force = pair_fn(
                            displacement,
                            distance,
                            pair_params,
                            atom_i,
                            atom_j,
                        )
                    _write_neighbor_pair(
                        row,
                        atom_j,
                        atom_j,
                        atom_i,
                        wp.vec3i(0, 0, 0),
                        displacement,
                        distance,
                        pair_energy,
                        pair_force,
                        neighbor_matrix1,
                        neighbor_matrix_shifts1,
                        num_neighbors1,
                        neighbor_vectors,
                        neighbor_distances,
                        pair_energies,
                        pair_forces,
                        max_neighbors,
                        False,
                        HALF_FILL,
                        PARTIAL,
                        RETURN_VECTORS,
                        RETURN_DISTANCES,
                        HAS_PAIR_FN,
                    )
            return

        isys = wp.int32(0)
        atom_i = row_tid
        j_start = wp.int32(0)
        j_end = positions.shape[0]

        if PBC:
            if BATCHED:
                isys = isys_tid
                if ishift >= num_shifts_arr[isys]:
                    return
                natom = batch_ptr[isys + 1] - batch_ptr[isys]
                if row_tid >= natom:
                    return
                atom_i = batch_ptr[isys] + row_tid
                j_start = batch_ptr[isys]
                j_end = batch_ptr[isys + 1]
            elif row_tid >= positions.shape[0]:
                return
        elif BATCHED:
            isys = batch_idx[atom_i]
            j_end = batch_ptr[isys + 1]

        if SELECTIVE and not rebuild_flags[isys]:
            return

        if PBC:
            shift = _decode_shift_index(ishift, shift_range[isys])
            zero_shift = shift[0] == 0 and shift[1] == 0 and shift[2] == 0
            if zero_shift:
                j_end = atom_i

            current_cell = cell[isys]
            atom_i_image = _shifted_position(shift, current_cell, positions[atom_i])
            max_neighbors1 = neighbor_matrix1.shape[1]
            max_neighbors2 = wp.int32(0)
            if DUAL_CUTOFF:
                max_neighbors2 = neighbor_matrix2.shape[1]
            if WRAP_ON_ENTRY:
                atom_i_offset = per_atom_cell_offsets[atom_i]

            for atom_j in range(j_start, j_end):
                dist_sq = wp.length_sq(atom_i_image - positions[atom_j])
                if DUAL_CUTOFF:
                    if dist_sq < cutoff2_sq:
                        stored_shift = shift
                        if WRAP_ON_ENTRY:
                            atom_j_offset = per_atom_cell_offsets[atom_j]
                            stored_shift = _correct_shift(
                                shift, atom_j_offset, atom_i_offset
                            )
                        _update_neighbor_matrix(
                            atom_j,
                            atom_i,
                            neighbor_matrix2,
                            neighbor_matrix_shifts2,
                            num_neighbors2,
                            stored_shift,
                            max_neighbors2,
                            HALF_FILL,
                            True,
                        )
                        if dist_sq < cutoff1_sq:
                            _update_neighbor_matrix(
                                atom_j,
                                atom_i,
                                neighbor_matrix1,
                                neighbor_matrix_shifts1,
                                num_neighbors1,
                                stored_shift,
                                max_neighbors1,
                                HALF_FILL,
                                True,
                            )
                elif dist_sq < cutoff1_sq:
                    stored_shift = shift
                    if WRAP_ON_ENTRY:
                        atom_j_offset = per_atom_cell_offsets[atom_j]
                        stored_shift = _correct_shift(
                            shift, atom_j_offset, atom_i_offset
                        )
                    _update_neighbor_matrix(
                        atom_j,
                        atom_i,
                        neighbor_matrix1,
                        neighbor_matrix_shifts1,
                        num_neighbors1,
                        stored_shift,
                        max_neighbors1,
                        HALF_FILL,
                        True,
                    )
            return

        pos_i = positions[atom_i]
        for atom_j in range(atom_i + 1, j_end):
            dist_sq = wp.length_sq(pos_i - positions[atom_j])
            _store_cutoff_pair(
                atom_i,
                atom_j,
                dist_sq,
                cutoff1_sq,
                cutoff2_sq,
                wp.vec3i(0, 0, 0),
                neighbor_matrix1,
                neighbor_matrix_shifts1,
                num_neighbors1,
                neighbor_matrix2,
                neighbor_matrix_shifts2,
                num_neighbors2,
                HALF_FILL,
                DUAL_CUTOFF,
                False,
            )

    name = kernel_specialization_name(
        _naive_kernel_base_name(
            operation,
            pbc_mode=pbc_mode,
            batched=BATCHED,
            selective=SELECTIVE,
        ),
        wp_dtype=wp_dtype,
        features=(
            "half" if bool(half_fill) else "",
            *_pair_output_features(
                partial=PARTIAL,
                return_vectors=RETURN_VECTORS,
                return_distances=RETURN_DISTANCES,
                pair_fn=pair_fn,
            ),
        ),
    )
    return set_fn_doc(
        set_fn_name(_kernel, name),
        _append_specialization_doc(
            _kernel.__doc__,
            dtype=wp_dtype,
            entries=(
                ("operation", operation),
                ("pbc_mode", pbc_mode.value),
                ("batched", BATCHED),
                ("selective", SELECTIVE),
                ("partial", PARTIAL),
                ("half_fill", bool(half_fill)),
                ("dual_cutoff", DUAL_CUTOFF),
                ("return_vectors", RETURN_VECTORS),
                ("return_distances", RETURN_DISTANCES),
                ("pair_fn", HAS_PAIR_FN),
            ),
        ),
    )


@lru_cache(maxsize=None)
def _make_tile_kernel(
    wp_dtype: type,
    *,
    pbc_mode: _PBCMode,
    batched: bool,
    half_fill: bool,
    selective: bool,
) -> wp.Kernel:
    """Return a cached CUDA tile-cooperative single-cutoff kernel."""
    _require_supported_dtype(wp_dtype)
    if batched and pbc_mode is _PBCMode.PREWRAPPED:
        raise NotImplementedError("batched prewrapped PBC has no tiled kernel")

    vec_dtype, mat_dtype = _DTYPE_INFO[wp_dtype]
    BATCHED = bool(batched)
    SELECTIVE = bool(selective)
    HALF_FILL = wp.constant(bool(half_fill))
    PBC = pbc_mode is not _PBCMode.NONE
    WRAP_ON_ENTRY = pbc_mode is _PBCMode.WRAP_ON_ENTRY

    @wp.kernel(enable_backward=False, module="naive_tile")
    def _kernel(
        positions: wp.array(dtype=vec_dtype),
        per_atom_cell_offsets: wp.array(dtype=wp.vec3i),
        cutoff_sq: wp_dtype,
        cell: wp.array(dtype=mat_dtype),
        shift_range: wp.array(dtype=wp.vec3i),
        num_shifts_arr: wp.array(dtype=wp.int32),
        batch_idx: wp.array(dtype=wp.int32),
        batch_ptr: wp.array(dtype=wp.int32),
        neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
        neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
        num_neighbors: wp.array(dtype=wp.int32),
        rebuild_flags: wp.array(dtype=wp.bool),
    ) -> None:
        """Calculate neighbor matrix using tile-cooperative naive O(N^2) algorithm

        Parameters
        ----------
        positions : wp.array, shape (total_atoms,), dtype=wp.vec3*
            Atomic coordinates in Cartesian space.
        per_atom_cell_offsets : wp.array, shape (total_atoms,), dtype=wp.vec3i
            Integer cell offsets for wrap-on-entry PBC. Zero-size sentinel for
            no-PBC and prewrapped PBC specializations.
        cutoff_sq : float
            Squared cutoff distance.
        cell : wp.array, shape (num_systems,), dtype=wp.mat33*
            Cell matrices for PBC specializations. Zero-size sentinel for no-PBC.
        shift_range : wp.array, shape (num_systems,), dtype=wp.vec3i
            Encoded PBC shift ranges per system. Zero-size sentinel for no-PBC.
        num_shifts_arr : wp.array, shape (num_systems,), dtype=wp.int32
            Number of active PBC shifts per system in batched PBC mode.
        batch_idx : wp.array, shape (total_atoms,), dtype=wp.int32
            System index for each atom. Zero-size sentinel for single-system modes.
        batch_ptr : wp.array, shape (num_systems + 1,), dtype=wp.int32
            Prefix offsets delimiting atoms per system in batched modes.
        neighbor_matrix : wp.array, shape (total_atoms, max_neighbors), dtype=wp.int32
            OUTPUT: Neighbor matrix.
        neighbor_matrix_shifts : wp.array, shape (total_atoms, max_neighbors), dtype=wp.vec3i
            OUTPUT: Periodic shift matrix for PBC modes.
        num_neighbors : wp.array, shape (total_atoms,), dtype=wp.int32
            OUTPUT: Per-atom neighbor counts.
        rebuild_flags : wp.array, shape (num_systems,), dtype=wp.bool
            Selective rebuild flags. Sentinel for non-selective specializations.

        Returns
        -------
        None
            This function modifies the input arrays in-place.

        Notes
        -----
        - Thread launch: Tiled launch with one tile block per ``(shift, source atom)`` pair. No-PBC launchers use a single sentinel shift.
        - Modifies: ``neighbor_matrix``, ``neighbor_matrix_shifts`` in PBC modes, and ``num_neighbors``.
        PBC, batching, wrap-on-entry, and selective rebuild are static factory
        specializations. Inactive inputs are zero-size sentinels and are not read.

        See Also
        --------
        get_naive_neighbor_matrix_kernel : Return the specialized naive tile-output kernel.
        """
        ishift, atom_i = wp.tid()
        isys = wp.int32(0)
        j_start = wp.int32(0)
        j_end = positions.shape[0]

        if BATCHED:
            isys = batch_idx[atom_i]
            j_start = batch_ptr[isys]
            j_end = batch_ptr[isys + 1]
        if SELECTIVE and not rebuild_flags[isys]:
            return

        max_neighbors = neighbor_matrix.shape[1]
        lane = wp.untile(wp.tile_arange(BLOCK_DIM, dtype=wp.int32))

        if PBC:
            if BATCHED and ishift >= num_shifts_arr[isys]:
                return

            shift = _decode_shift_index(ishift, shift_range[isys])
            current_cell = cell[isys]
            pos_i = positions[atom_i]
            pos_i_image = _shifted_position(shift, current_cell, pos_i)
            atom_i_offset = wp.vec3i(0, 0, 0)
            if WRAP_ON_ENTRY:
                atom_i_offset = per_atom_cell_offsets[atom_i]

            if shift[0] == 0 and shift[1] == 0 and shift[2] == 0:
                j_end = atom_i

            for chunk_start in range(j_start, j_end, BLOCK_DIM):
                atom_j = chunk_start + lane
                safe_j = wp.min(atom_j, positions.shape[0] - 1)
                diff = pos_i_image - positions[safe_j]
                dist_sq = wp.dot(diff, diff)
                if atom_j < j_end and dist_sq < cutoff_sq:
                    # Tile traversal writes the shifted atom as the row;
                    # match the scalar row-local shift convention.
                    stored_shift = -shift
                    if WRAP_ON_ENTRY:
                        atom_j_offset = per_atom_cell_offsets[atom_j]
                        corrected_shift = _correct_shift(
                            shift, atom_j_offset, atom_i_offset
                        )
                        stored_shift = -corrected_shift
                    _update_neighbor_matrix(
                        atom_i,
                        atom_j,
                        neighbor_matrix,
                        neighbor_matrix_shifts,
                        num_neighbors,
                        stored_shift,
                        max_neighbors,
                        HALF_FILL,
                        True,
                    )
            return

        pos_i = positions[atom_i]
        for chunk_start in range(atom_i + 1, j_end, BLOCK_DIM):
            atom_j = chunk_start + lane
            safe_j = wp.min(atom_j, positions.shape[0] - 1)
            diff = pos_i - positions[safe_j]
            dist_sq = wp.dot(diff, diff)
            if atom_j < j_end and dist_sq < cutoff_sq:
                _update_neighbor_matrix(
                    atom_i,
                    atom_j,
                    neighbor_matrix,
                    neighbor_matrix_shifts,
                    num_neighbors,
                    wp.vec3i(0, 0, 0),
                    max_neighbors,
                    HALF_FILL,
                    False,
                )

    name = kernel_specialization_name(
        _naive_kernel_base_name(
            "single_cutoff",
            pbc_mode=pbc_mode,
            batched=BATCHED,
            selective=SELECTIVE,
        ),
        wp_dtype=wp_dtype,
        features=("tile", "half" if bool(half_fill) else ""),
    )
    return set_fn_doc(
        set_fn_name(_kernel, name),
        _append_specialization_doc(
            _kernel.__doc__,
            dtype=wp_dtype,
            entries=(
                ("operation", "single_cutoff"),
                ("strategy", "tile"),
                ("pbc_mode", pbc_mode.value),
                ("batched", BATCHED),
                ("half_fill", bool(half_fill)),
                ("selective", SELECTIVE),
            ),
        ),
    )


def get_naive_neighbor_matrix_kernel(
    wp_dtype: type,
    *,
    pbc_mode: Literal["none", "prewrapped", "wrap_on_entry"] = "none",
    batched: bool = False,
    half_fill: bool = False,
    selective: bool = False,
    partial: bool = False,
    return_vectors: bool = False,
    return_distances: bool = False,
    pair_fn: wp.Function | None = None,
    strategy: Literal["scalar", "tile"] = "scalar",
) -> wp.Kernel:
    """Return a cached single-cutoff naive neighbor-list kernel.

    The cache key stores the ``@wp.func`` object directly. Warp function
    objects are hashable by identity, so callbacks should be module-scope
    singleton objects to keep specializations reusable.
    """
    _require_supported_dtype(wp_dtype)
    mode = _parse_pbc_mode(pbc_mode)
    strat = _parse_strategy(strategy)
    if strat is _NaiveStrategy.TILE:
        if partial or return_vectors or return_distances or pair_fn is not None:
            raise NotImplementedError("pair outputs currently use strategy='scalar'")
        return _make_tile_kernel(
            wp_dtype,
            pbc_mode=mode,
            batched=batched,
            half_fill=half_fill,
            selective=selective,
        )
    return _make_scalar_kernel(
        "single_cutoff",
        wp_dtype,
        pbc_mode=mode,
        batched=batched,
        half_fill=half_fill,
        selective=selective,
        partial=partial,
        return_vectors=return_vectors,
        return_distances=return_distances,
        pair_fn=pair_fn,
    )


def get_naive_neighbor_matrix_dual_cutoff_kernel(
    wp_dtype: type,
    *,
    pbc_mode: Literal["none", "prewrapped", "wrap_on_entry"] = "none",
    batched: bool = False,
    half_fill: bool = False,
    selective: bool = False,
) -> wp.Kernel:
    """Return a cached dual-cutoff naive neighbor-list kernel."""
    _require_supported_dtype(wp_dtype)
    mode = _parse_pbc_mode(pbc_mode)
    return _make_scalar_kernel(
        "dual_cutoff",
        wp_dtype,
        pbc_mode=mode,
        batched=batched,
        half_fill=half_fill,
        selective=selective,
        partial=False,
        return_vectors=False,
        return_distances=False,
        pair_fn=None,
    )
