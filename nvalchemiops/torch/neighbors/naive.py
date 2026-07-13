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

"""PyTorch bindings for unbatched naive neighbor list construction."""

from __future__ import annotations

import torch
import warp as wp

from nvalchemiops.neighbors.naive import (
    naive_neighbor_matrix,
    naive_neighbor_matrix_pbc,
)
from nvalchemiops.neighbors.neighbor_utils import (
    estimate_max_neighbors,
    selective_zero_num_neighbors_single,
)
from nvalchemiops.torch._warp_op_helpers import register_noop_fake, scoped_warp_stream
from nvalchemiops.torch.neighbors._autograd import (
    _flatten_active_pairs,
    _NeighborForwardOutput,
    _route_pair_outputs,
)
from nvalchemiops.torch.neighbors._compiled_pair_fn import (
    CompiledPairFn,
    is_compiled_pair_fn,
)
from nvalchemiops.torch.neighbors.neighbor_utils import (
    _validate_pair_params_present,
    compute_naive_num_shifts,
    coo_pack_pair_geometry,
    get_neighbor_list_from_neighbor_matrix,
)
from nvalchemiops.torch.types import get_wp_dtype, get_wp_mat_dtype, get_wp_vec_dtype

__all__ = ["naive_neighbor_list"]


@torch.library.custom_op(
    "nvalchemiops::_naive_neighbor_matrix_no_pbc",
    mutates_args=("neighbor_matrix", "num_neighbors"),
)
def _naive_neighbor_matrix_no_pbc(
    positions: torch.Tensor,
    cutoff: float,
    neighbor_matrix: torch.Tensor,
    num_neighbors: torch.Tensor,
    half_fill: bool = False,
    rebuild_flags: torch.Tensor | None = None,
    strategy: str = "auto",
) -> None:
    """Fill neighbor matrix for atoms using naive O(N^2) algorithm.

    Custom PyTorch operator that computes pairwise distances and fills
    the neighbor matrix with atom indices within the cutoff distance.
    No periodic boundary conditions are applied.

    This function does not allocate any tensors.

    This function is torch compilable.

    Parameters
    ----------
    positions : torch.Tensor, shape (total_atoms, 3), dtype=torch.float32 or torch.float64
        Atomic coordinates in Cartesian space. Each row represents one atom's
        (x, y, z) position.
    cutoff : float
        Cutoff distance for neighbor detection in Cartesian units.
        Must be positive. Atoms within this distance are considered neighbors.
    neighbor_matrix : torch.Tensor, shape (total_atoms, max_neighbors), dtype=torch.int32
        OUTPUT: Neighbor matrix to be filled with neighbor atom indices.
        Must be pre-allocated. Entries are filled with atom indices.
    num_neighbors : torch.Tensor, shape (total_atoms,), dtype=torch.int32
        OUTPUT: Number of neighbors found for each atom.
        Must be pre-allocated. Updated in-place with actual neighbor counts.
    half_fill : bool
        If True, only store relationships where i < j to avoid double counting.
        If False, store all neighbor relationships symmetrically.
    rebuild_flags : torch.Tensor, shape (1,), dtype=torch.bool, optional
        Per-system rebuild flags. If provided, only systems where rebuild_flags[i]
        is True are processed; others are skipped on the GPU without CPU sync.
        Call selective_zero_num_neighbors before this launcher to reset counts.
    See Also
    --------
    nvalchemiops.neighbors.naive.naive_neighbor_matrix : Core warp launcher
    naive_neighbor_list : High-level wrapper function
    """
    device = positions.device
    wp_dtype = get_wp_dtype(positions.dtype)
    wp_vec_dtype = get_wp_vec_dtype(positions.dtype)

    wp_positions = wp.from_torch(
        positions, dtype=wp_vec_dtype, requires_grad=False, return_ctype=True
    )
    wp_neighbor_matrix = wp.from_torch(
        neighbor_matrix, dtype=wp.int32, requires_grad=False, return_ctype=True
    )
    wp_num_neighbors = wp.from_torch(
        num_neighbors, dtype=wp.int32, requires_grad=False, return_ctype=True
    )
    with scoped_warp_stream(device):
        if rebuild_flags is not None:
            wp_rebuild_flags = wp.from_torch(
                rebuild_flags, dtype=wp.bool, requires_grad=False, return_ctype=True
            )
            selective_zero_num_neighbors_single(
                wp_num_neighbors, wp_rebuild_flags, str(device)
            )
        else:
            wp_rebuild_flags = None

        naive_neighbor_matrix(
            positions=wp_positions,
            cutoff=cutoff,
            neighbor_matrix=wp_neighbor_matrix,
            num_neighbors=wp_num_neighbors,
            wp_dtype=wp_dtype,
            device=str(device),
            half_fill=half_fill,
            rebuild_flags=wp_rebuild_flags,
            strategy=strategy,
        )


@torch.library.custom_op(
    "nvalchemiops::_naive_neighbor_matrix_pbc",
    mutates_args=("neighbor_matrix", "neighbor_matrix_shifts", "num_neighbors"),
)
def _naive_neighbor_matrix_pbc(
    positions: torch.Tensor,
    cutoff: float,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    num_neighbors: torch.Tensor,
    shift_range_per_dimension: torch.Tensor,
    num_shifts_per_system: torch.Tensor,
    max_shifts_per_system: int,
    half_fill: bool = False,
    rebuild_flags: torch.Tensor | None = None,
    wrap_positions: bool = True,
    positions_wrapped_buffer: torch.Tensor | None = None,
    per_atom_cell_offsets_buffer: torch.Tensor | None = None,
    inv_cell_buffer: torch.Tensor | None = None,
    strategy: str = "auto",
) -> None:
    """Compute neighbor matrix with periodic boundary conditions using naive O(N^2) algorithm.

    This function is torch compilable.

    Parameters
    ----------
    positions : torch.Tensor, shape (total_atoms, 3)
        Atomic coordinates in Cartesian space.
    cutoff : float
        Cutoff distance for neighbor detection.
    cell : torch.Tensor, shape (1, 3, 3)
        Cell matrix defining lattice vectors.
    neighbor_matrix : torch.Tensor, shape (total_atoms, max_neighbors), dtype=torch.int32
        OUTPUT: Neighbor matrix to be filled.
    neighbor_matrix_shifts : torch.Tensor, shape (total_atoms, max_neighbors, 3), dtype=torch.int32
        OUTPUT: Shift vectors for each neighbor relationship.
    num_neighbors : torch.Tensor, shape (total_atoms,), dtype=torch.int32
        OUTPUT: Number of neighbors found for each atom.
    shift_range_per_dimension : torch.Tensor, shape (1, 3), dtype=torch.int32
        Shift range in each dimension.
    num_shifts_per_system : torch.Tensor, shape (1,), dtype=torch.int32
        Number of periodic shifts for the system.
    max_shifts_per_system : int
        Maximum shift count (used as launch dimension).
    half_fill : bool, optional
        If True, only store relationships where i < j. Default is False.
    rebuild_flags : torch.Tensor, shape (1,), dtype=torch.bool, optional
        When False the kernel skips work (no CPU-GPU sync).
    wrap_positions : bool, default=True
        If True, wrap positions into the primary cell before neighbor search.
    positions_wrapped_buffer : torch.Tensor, shape (total_atoms, 3), optional
        Caller-supplied scratch buffer for wrapped positions
        (used when ``wrap_positions=True``).  When omitted, the warp launcher
        allocates internally; supply to avoid per-call allocation.
    per_atom_cell_offsets_buffer : torch.Tensor, shape (total_atoms, 3), dtype=torch.int32, optional
        Caller-supplied scratch buffer for per-atom cell offsets.
    inv_cell_buffer : torch.Tensor, shape (num_systems, 3, 3), optional
        Caller-supplied scratch buffer for inverse cell matrices.

    See Also
    --------
    nvalchemiops.neighbors.naive.naive_neighbor_matrix_pbc : Core warp launcher
    naive_neighbor_list : High-level wrapper function
    """
    device = positions.device
    wp_dtype = get_wp_dtype(positions.dtype)
    wp_vec_dtype = get_wp_vec_dtype(positions.dtype)
    wp_mat_dtype = get_wp_mat_dtype(cell.dtype)

    wp_positions = wp.from_torch(
        positions, dtype=wp_vec_dtype, requires_grad=False, return_ctype=True
    )
    wp_cell = wp.from_torch(
        cell, dtype=wp_mat_dtype, requires_grad=False, return_ctype=True
    )
    wp_pbc = wp.from_torch(pbc, dtype=wp.bool, requires_grad=False, return_ctype=True)
    wp_shift_range = wp.from_torch(
        shift_range_per_dimension,
        dtype=wp.vec3i,
        requires_grad=False,
        return_ctype=True,
    )
    wp_neighbor_matrix = wp.from_torch(
        neighbor_matrix, dtype=wp.int32, requires_grad=False, return_ctype=True
    )
    wp_neighbor_matrix_shifts = wp.from_torch(
        neighbor_matrix_shifts, dtype=wp.vec3i, requires_grad=False, return_ctype=True
    )
    wp_num_neighbors = wp.from_torch(
        num_neighbors, dtype=wp.int32, requires_grad=False, return_ctype=True
    )

    with scoped_warp_stream(device):
        if rebuild_flags is not None:
            wp_rebuild_flags = wp.from_torch(
                rebuild_flags, dtype=wp.bool, requires_grad=False, return_ctype=True
            )
            selective_zero_num_neighbors_single(
                wp_num_neighbors, wp_rebuild_flags, str(device)
            )
        else:
            wp_rebuild_flags = None

        wp_positions_wrapped = (
            wp.from_torch(
                positions_wrapped_buffer,
                dtype=wp_vec_dtype,
                requires_grad=False,
                return_ctype=True,
            )
            if positions_wrapped_buffer is not None
            else None
        )
        wp_per_atom_cell_offsets = (
            wp.from_torch(
                per_atom_cell_offsets_buffer,
                dtype=wp.vec3i,
                requires_grad=False,
                return_ctype=True,
            )
            if per_atom_cell_offsets_buffer is not None
            else None
        )
        wp_inv_cell = (
            wp.from_torch(
                inv_cell_buffer,
                dtype=wp_mat_dtype,
                requires_grad=False,
                return_ctype=True,
            )
            if inv_cell_buffer is not None
            else None
        )

        naive_neighbor_matrix_pbc(
            positions=wp_positions,
            cutoff=cutoff,
            cell=wp_cell,
            pbc=wp_pbc,
            shift_range=wp_shift_range,
            num_shifts=max_shifts_per_system,
            neighbor_matrix=wp_neighbor_matrix,
            neighbor_matrix_shifts=wp_neighbor_matrix_shifts,
            num_neighbors=wp_num_neighbors,
            wp_dtype=wp_dtype,
            device=str(device),
            half_fill=half_fill,
            rebuild_flags=wp_rebuild_flags,
            wrap_positions=wrap_positions,
            positions_wrapped_buffer=wp_positions_wrapped,
            per_atom_cell_offsets_buffer=wp_per_atom_cell_offsets,
            inv_cell_buffer=wp_inv_cell,
            strategy=strategy,
        )


@torch.library.custom_op(
    "nvalchemiops::_naive_neighbor_matrix_no_pbc_pair",
    mutates_args=(
        "neighbor_matrix",
        "num_neighbors",
        "neighbor_vectors",
        "neighbor_distances",
    ),
)
def _naive_neighbor_matrix_no_pbc_pair(
    positions: torch.Tensor,
    cutoff: float,
    neighbor_matrix: torch.Tensor,
    num_neighbors: torch.Tensor,
    neighbor_vectors: torch.Tensor,
    neighbor_distances: torch.Tensor,
    half_fill: bool = False,
) -> None:
    """No-PBC naive neighbor matrix kernel with pair outputs."""
    device = positions.device
    wp_dtype = get_wp_dtype(positions.dtype)
    wp_vec_dtype = get_wp_vec_dtype(positions.dtype)
    wp_positions = wp.from_torch(
        positions, dtype=wp_vec_dtype, requires_grad=False, return_ctype=True
    )
    wp_neighbor_matrix = wp.from_torch(
        neighbor_matrix, dtype=wp.int32, requires_grad=False, return_ctype=True
    )
    wp_num_neighbors = wp.from_torch(
        num_neighbors, dtype=wp.int32, requires_grad=False, return_ctype=True
    )
    # Validated by ``_prepare_pair_output_args`` in the launcher -> pass real
    # Warp arrays (zero-copy views of the torch tensors), not ctype structs.
    wp_neighbor_vectors = wp.from_torch(
        neighbor_vectors, dtype=wp_vec_dtype, requires_grad=False
    )
    wp_neighbor_distances = wp.from_torch(
        neighbor_distances, dtype=wp_dtype, requires_grad=False
    )
    naive_neighbor_matrix(
        positions=wp_positions,
        cutoff=cutoff,
        neighbor_matrix=wp_neighbor_matrix,
        num_neighbors=wp_num_neighbors,
        wp_dtype=wp_dtype,
        device=str(device),
        half_fill=half_fill,
        rebuild_flags=None,
        return_vectors=True,
        return_distances=True,
        neighbor_vectors=wp_neighbor_vectors,
        neighbor_distances=wp_neighbor_distances,
    )


@torch.library.custom_op(
    "nvalchemiops::_naive_neighbor_matrix_pbc_pair",
    mutates_args=(
        "neighbor_matrix",
        "neighbor_matrix_shifts",
        "num_neighbors",
        "neighbor_vectors",
        "neighbor_distances",
    ),
)
def _naive_neighbor_matrix_pbc_pair(
    positions: torch.Tensor,
    cutoff: float,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    num_neighbors: torch.Tensor,
    neighbor_vectors: torch.Tensor,
    neighbor_distances: torch.Tensor,
    shift_range_per_dimension: torch.Tensor,
    num_shifts_per_system: torch.Tensor,
    max_shifts_per_system: int,
    half_fill: bool = False,
    wrap_positions: bool = True,
) -> None:
    """PBC naive neighbor matrix kernel with pair outputs.

    The warp launcher picks between ``wrap_on_entry`` and ``prewrapped``
    kernel specializations based on ``wrap_positions``; both produce
    shifts consistent with the input positions, so the autograd
    primitive's reconstruction ``r = pos[j] - pos[i] + shifts @ cell``
    is correct for either path.
    """
    device = positions.device
    wp_dtype = get_wp_dtype(positions.dtype)
    wp_vec_dtype = get_wp_vec_dtype(positions.dtype)
    wp_mat_dtype = get_wp_mat_dtype(cell.dtype)

    wp_positions = wp.from_torch(
        positions, dtype=wp_vec_dtype, requires_grad=False, return_ctype=True
    )
    wp_cell = wp.from_torch(
        cell, dtype=wp_mat_dtype, requires_grad=False, return_ctype=True
    )
    wp_pbc = wp.from_torch(pbc, dtype=wp.bool, requires_grad=False, return_ctype=True)
    wp_shift_range = wp.from_torch(
        shift_range_per_dimension,
        dtype=wp.vec3i,
        requires_grad=False,
        return_ctype=True,
    )
    wp_neighbor_matrix = wp.from_torch(
        neighbor_matrix, dtype=wp.int32, requires_grad=False, return_ctype=True
    )
    wp_neighbor_matrix_shifts = wp.from_torch(
        neighbor_matrix_shifts, dtype=wp.vec3i, requires_grad=False, return_ctype=True
    )
    wp_num_neighbors = wp.from_torch(
        num_neighbors, dtype=wp.int32, requires_grad=False, return_ctype=True
    )
    # Validated by ``_prepare_pair_output_args`` in the launcher -> pass real
    # Warp arrays (zero-copy views of the torch tensors), not ctype structs.
    wp_neighbor_vectors = wp.from_torch(
        neighbor_vectors, dtype=wp_vec_dtype, requires_grad=False
    )
    wp_neighbor_distances = wp.from_torch(
        neighbor_distances, dtype=wp_dtype, requires_grad=False
    )

    naive_neighbor_matrix_pbc(
        positions=wp_positions,
        cutoff=cutoff,
        cell=wp_cell,
        pbc=wp_pbc,
        shift_range=wp_shift_range,
        num_shifts=max_shifts_per_system,
        neighbor_matrix=wp_neighbor_matrix,
        neighbor_matrix_shifts=wp_neighbor_matrix_shifts,
        num_neighbors=wp_num_neighbors,
        wp_dtype=wp_dtype,
        device=str(device),
        half_fill=half_fill,
        rebuild_flags=None,
        wrap_positions=wrap_positions,
        return_vectors=True,
        return_distances=True,
        neighbor_vectors=wp_neighbor_vectors,
        neighbor_distances=wp_neighbor_distances,
    )


@torch.library.custom_op(
    "nvalchemiops::_naive_neighbor_matrix_no_pbc_pair_target",
    mutates_args=(
        "neighbor_matrix",
        "num_neighbors",
        "neighbor_vectors",
        "neighbor_distances",
    ),
)
def _naive_neighbor_matrix_no_pbc_pair_target(
    positions: torch.Tensor,
    cutoff: float,
    target_indices: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    num_neighbors: torch.Tensor,
    neighbor_vectors: torch.Tensor,
    neighbor_distances: torch.Tensor,
    half_fill: bool = False,
) -> None:
    """No-PBC naive pair-output kernel for compact target rows."""
    device = positions.device
    wp_dtype = get_wp_dtype(positions.dtype)
    wp_vec_dtype = get_wp_vec_dtype(positions.dtype)
    naive_neighbor_matrix(
        positions=wp.from_torch(
            positions, dtype=wp_vec_dtype, requires_grad=False, return_ctype=True
        ),
        cutoff=cutoff,
        neighbor_matrix=wp.from_torch(
            neighbor_matrix, dtype=wp.int32, requires_grad=False, return_ctype=True
        ),
        num_neighbors=wp.from_torch(
            num_neighbors, dtype=wp.int32, requires_grad=False, return_ctype=True
        ),
        wp_dtype=wp_dtype,
        device=str(device),
        half_fill=half_fill,
        rebuild_flags=None,
        target_indices=wp.from_torch(
            target_indices, dtype=wp.int32, requires_grad=False, return_ctype=True
        ),
        return_vectors=True,
        return_distances=True,
        neighbor_vectors=wp.from_torch(
            neighbor_vectors, dtype=wp_vec_dtype, requires_grad=False
        ),
        neighbor_distances=wp.from_torch(
            neighbor_distances, dtype=wp_dtype, requires_grad=False
        ),
    )


@torch.library.custom_op(
    "nvalchemiops::_naive_neighbor_matrix_pbc_pair_target",
    mutates_args=(
        "neighbor_matrix",
        "neighbor_matrix_shifts",
        "num_neighbors",
        "neighbor_vectors",
        "neighbor_distances",
    ),
)
def _naive_neighbor_matrix_pbc_pair_target(
    positions: torch.Tensor,
    cutoff: float,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    target_indices: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    num_neighbors: torch.Tensor,
    neighbor_vectors: torch.Tensor,
    neighbor_distances: torch.Tensor,
    shift_range_per_dimension: torch.Tensor,
    num_shifts_per_system: torch.Tensor,
    max_shifts_per_system: int,
    half_fill: bool = False,
    wrap_positions: bool = True,
) -> None:
    """PBC naive pair-output kernel for compact target rows."""
    device = positions.device
    wp_dtype = get_wp_dtype(positions.dtype)
    wp_vec_dtype = get_wp_vec_dtype(positions.dtype)
    wp_mat_dtype = get_wp_mat_dtype(cell.dtype)
    naive_neighbor_matrix_pbc(
        positions=wp.from_torch(
            positions, dtype=wp_vec_dtype, requires_grad=False, return_ctype=True
        ),
        cutoff=cutoff,
        cell=wp.from_torch(
            cell, dtype=wp_mat_dtype, requires_grad=False, return_ctype=True
        ),
        pbc=wp.from_torch(pbc, dtype=wp.bool, requires_grad=False, return_ctype=True),
        shift_range=wp.from_torch(
            shift_range_per_dimension,
            dtype=wp.vec3i,
            requires_grad=False,
            return_ctype=True,
        ),
        num_shifts=max_shifts_per_system,
        neighbor_matrix=wp.from_torch(
            neighbor_matrix, dtype=wp.int32, requires_grad=False, return_ctype=True
        ),
        neighbor_matrix_shifts=wp.from_torch(
            neighbor_matrix_shifts,
            dtype=wp.vec3i,
            requires_grad=False,
            return_ctype=True,
        ),
        num_neighbors=wp.from_torch(
            num_neighbors, dtype=wp.int32, requires_grad=False, return_ctype=True
        ),
        wp_dtype=wp_dtype,
        device=str(device),
        half_fill=half_fill,
        rebuild_flags=None,
        wrap_positions=wrap_positions,
        target_indices=wp.from_torch(
            target_indices, dtype=wp.int32, requires_grad=False, return_ctype=True
        ),
        return_vectors=True,
        return_distances=True,
        neighbor_vectors=wp.from_torch(
            neighbor_vectors, dtype=wp_vec_dtype, requires_grad=False
        ),
        neighbor_distances=wp.from_torch(
            neighbor_distances, dtype=wp_dtype, requires_grad=False
        ),
    )


register_noop_fake(_naive_neighbor_matrix_no_pbc)
register_noop_fake(_naive_neighbor_matrix_pbc)
register_noop_fake(_naive_neighbor_matrix_no_pbc_pair)
register_noop_fake(_naive_neighbor_matrix_pbc_pair)
register_noop_fake(_naive_neighbor_matrix_no_pbc_pair_target)
register_noop_fake(_naive_neighbor_matrix_pbc_pair_target)


def _register_compiled_naive_no_pbc_pair_op(compiled: CompiledPairFn):
    """Register a pair_fn-specialized no-PBC naive custom op."""

    @torch.library.custom_op(
        f"nvalchemiops::{compiled.op_name('naive_no_pbc_pair')}",
        mutates_args=(
            "neighbor_matrix",
            "num_neighbors",
            "neighbor_vectors",
            "neighbor_distances",
            "pair_energies",
            "pair_forces",
        ),
    )
    def _compiled_naive_no_pbc_pair(
        positions: torch.Tensor,
        cutoff: float,
        target_indices: torch.Tensor | None,
        neighbor_matrix: torch.Tensor,
        num_neighbors: torch.Tensor,
        neighbor_vectors: torch.Tensor,
        neighbor_distances: torch.Tensor,
        pair_params: torch.Tensor,
        pair_energies: torch.Tensor,
        pair_forces: torch.Tensor,
        half_fill: bool,
    ) -> None:
        device = positions.device
        wp_dtype = get_wp_dtype(positions.dtype)
        wp_vec_dtype = get_wp_vec_dtype(positions.dtype)
        wp_target_indices = (
            wp.from_torch(
                target_indices,
                dtype=wp.int32,
                requires_grad=False,
                return_ctype=True,
            )
            if target_indices is not None
            else None
        )
        naive_neighbor_matrix(
            positions=wp.from_torch(
                positions,
                dtype=wp_vec_dtype,
                requires_grad=False,
                return_ctype=True,
            ),
            cutoff=cutoff,
            neighbor_matrix=wp.from_torch(
                neighbor_matrix,
                dtype=wp.int32,
                requires_grad=False,
                return_ctype=True,
            ),
            num_neighbors=wp.from_torch(
                num_neighbors,
                dtype=wp.int32,
                requires_grad=False,
                return_ctype=True,
            ),
            wp_dtype=wp_dtype,
            device=str(device),
            half_fill=half_fill,
            rebuild_flags=None,
            target_indices=wp_target_indices,
            return_vectors=True,
            return_distances=True,
            neighbor_vectors=wp.from_torch(
                neighbor_vectors, dtype=wp_vec_dtype, requires_grad=False
            ),
            neighbor_distances=wp.from_torch(
                neighbor_distances, dtype=wp_dtype, requires_grad=False
            ),
            pair_fn=compiled.pair_fn,
            pair_params=wp.from_torch(pair_params, dtype=wp_dtype, requires_grad=False),
            pair_energies=wp.from_torch(
                pair_energies, dtype=wp_dtype, requires_grad=False
            ),
            pair_forces=wp.from_torch(
                pair_forces, dtype=wp_vec_dtype, requires_grad=False
            ),
        )

    register_noop_fake(_compiled_naive_no_pbc_pair)
    return _compiled_naive_no_pbc_pair


def _register_compiled_naive_pbc_pair_op(compiled: CompiledPairFn):
    """Register a pair_fn-specialized PBC naive custom op."""

    @torch.library.custom_op(
        f"nvalchemiops::{compiled.op_name('naive_pbc_pair')}",
        mutates_args=(
            "neighbor_matrix",
            "neighbor_matrix_shifts",
            "num_neighbors",
            "neighbor_vectors",
            "neighbor_distances",
            "pair_energies",
            "pair_forces",
        ),
    )
    def _compiled_naive_pbc_pair(
        positions: torch.Tensor,
        cutoff: float,
        cell: torch.Tensor,
        pbc: torch.Tensor,
        target_indices: torch.Tensor | None,
        neighbor_matrix: torch.Tensor,
        neighbor_matrix_shifts: torch.Tensor,
        num_neighbors: torch.Tensor,
        neighbor_vectors: torch.Tensor,
        neighbor_distances: torch.Tensor,
        pair_params: torch.Tensor,
        pair_energies: torch.Tensor,
        pair_forces: torch.Tensor,
        shift_range_per_dimension: torch.Tensor,
        num_shifts_per_system: torch.Tensor,
        max_shifts_per_system: int,
        half_fill: bool,
        wrap_positions: bool,
    ) -> None:
        device = positions.device
        wp_dtype = get_wp_dtype(positions.dtype)
        wp_vec_dtype = get_wp_vec_dtype(positions.dtype)
        wp_mat_dtype = get_wp_mat_dtype(cell.dtype)
        wp_target_indices = (
            wp.from_torch(
                target_indices,
                dtype=wp.int32,
                requires_grad=False,
                return_ctype=True,
            )
            if target_indices is not None
            else None
        )
        naive_neighbor_matrix_pbc(
            positions=wp.from_torch(
                positions,
                dtype=wp_vec_dtype,
                requires_grad=False,
                return_ctype=True,
            ),
            cutoff=cutoff,
            cell=wp.from_torch(
                cell,
                dtype=wp_mat_dtype,
                requires_grad=False,
                return_ctype=True,
            ),
            pbc=wp.from_torch(
                pbc, dtype=wp.bool, requires_grad=False, return_ctype=True
            ),
            shift_range=wp.from_torch(
                shift_range_per_dimension,
                dtype=wp.vec3i,
                requires_grad=False,
                return_ctype=True,
            ),
            num_shifts=max_shifts_per_system,
            neighbor_matrix=wp.from_torch(
                neighbor_matrix,
                dtype=wp.int32,
                requires_grad=False,
                return_ctype=True,
            ),
            neighbor_matrix_shifts=wp.from_torch(
                neighbor_matrix_shifts,
                dtype=wp.vec3i,
                requires_grad=False,
                return_ctype=True,
            ),
            num_neighbors=wp.from_torch(
                num_neighbors,
                dtype=wp.int32,
                requires_grad=False,
                return_ctype=True,
            ),
            wp_dtype=wp_dtype,
            device=str(device),
            half_fill=half_fill,
            rebuild_flags=None,
            wrap_positions=wrap_positions,
            target_indices=wp_target_indices,
            return_vectors=True,
            return_distances=True,
            neighbor_vectors=wp.from_torch(
                neighbor_vectors, dtype=wp_vec_dtype, requires_grad=False
            ),
            neighbor_distances=wp.from_torch(
                neighbor_distances, dtype=wp_dtype, requires_grad=False
            ),
            pair_fn=compiled.pair_fn,
            pair_params=wp.from_torch(pair_params, dtype=wp_dtype, requires_grad=False),
            pair_energies=wp.from_torch(
                pair_energies, dtype=wp_dtype, requires_grad=False
            ),
            pair_forces=wp.from_torch(
                pair_forces, dtype=wp_vec_dtype, requires_grad=False
            ),
        )

    register_noop_fake(_compiled_naive_pbc_pair)
    return _compiled_naive_pbc_pair


def _validate_output_buffer(
    name: str,
    tensor: torch.Tensor | None,
    expected_shape: tuple[int, ...],
    expected_dtype: torch.dtype | None = None,
) -> None:
    """Validate optional compact-row output buffers."""
    if tensor is None:
        return
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}; got {tuple(tensor.shape)}.",
        )
    if expected_dtype is not None and tensor.dtype != expected_dtype:
        raise ValueError(
            f"{name} dtype must be {expected_dtype}; got {tensor.dtype}.",
        )


def _naive_pair_outputs_forward(
    positions: torch.Tensor,
    cell: torch.Tensor | None,
    *,
    cutoff: float,
    pbc: torch.Tensor | None,
    neighbor_matrix: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor | None,
    num_neighbors: torch.Tensor,
    neighbor_vectors: torch.Tensor,
    neighbor_distances: torch.Tensor,
    half_fill: bool,
    shift_range_per_dimension: torch.Tensor | None,
    num_shifts_per_system: torch.Tensor | None,
    max_shifts_per_system: int | None,
    wrap_positions: bool,
    target_indices: torch.Tensor | None = None,
    pair_fn=None,
    pair_params: torch.Tensor | None = None,
    pair_energies: torch.Tensor | None = None,
    pair_forces: torch.Tensor | None = None,
) -> _NeighborForwardOutput:
    """Forward closure for the torch naive autograd path.

    Without ``pair_fn`` this runs the torch custom ops (``torch.compile``
    friendly).  With ``pair_fn`` set it calls the Warp launcher directly,
    because a callable cannot cross a torch custom-op boundary; the kernel does
    not participate in autograd, so ``pair_energies`` / ``pair_forces`` are
    forward-only (matching the cell-list binding).
    """
    is_partial = target_indices is not None
    if is_compiled_pair_fn(pair_fn):
        if pbc is None:
            op = pair_fn.get_or_register(
                "naive_no_pbc_pair",
                _register_compiled_naive_no_pbc_pair_op,
            )
            op(
                positions=positions.detach(),
                cutoff=cutoff,
                target_indices=target_indices,
                neighbor_matrix=neighbor_matrix,
                num_neighbors=num_neighbors,
                neighbor_vectors=neighbor_vectors,
                neighbor_distances=neighbor_distances,
                pair_params=pair_params,
                pair_energies=pair_energies,
                pair_forces=pair_forces,
                half_fill=half_fill,
            )
        else:
            op = pair_fn.get_or_register(
                "naive_pbc_pair",
                _register_compiled_naive_pbc_pair_op,
            )
            op(
                positions=positions.detach(),
                cutoff=cutoff,
                cell=cell.detach(),
                pbc=pbc,
                target_indices=target_indices,
                neighbor_matrix=neighbor_matrix,
                neighbor_matrix_shifts=neighbor_matrix_shifts,
                num_neighbors=num_neighbors,
                neighbor_vectors=neighbor_vectors,
                neighbor_distances=neighbor_distances,
                pair_params=pair_params,
                pair_energies=pair_energies,
                pair_forces=pair_forces,
                shift_range_per_dimension=shift_range_per_dimension,
                num_shifts_per_system=num_shifts_per_system,
                max_shifts_per_system=int(max_shifts_per_system),
                half_fill=half_fill,
                wrap_positions=wrap_positions,
            )
    elif pair_fn is None and not is_partial and pbc is None:
        _naive_neighbor_matrix_no_pbc_pair(
            positions=positions.detach(),
            cutoff=cutoff,
            neighbor_matrix=neighbor_matrix,
            num_neighbors=num_neighbors,
            neighbor_vectors=neighbor_vectors,
            neighbor_distances=neighbor_distances,
            half_fill=half_fill,
        )
    elif pair_fn is None and is_partial and pbc is None:
        _naive_neighbor_matrix_no_pbc_pair_target(
            positions=positions.detach(),
            cutoff=cutoff,
            target_indices=target_indices,
            neighbor_matrix=neighbor_matrix,
            num_neighbors=num_neighbors,
            neighbor_vectors=neighbor_vectors,
            neighbor_distances=neighbor_distances,
            half_fill=half_fill,
        )
    elif pair_fn is None and not is_partial:
        _naive_neighbor_matrix_pbc_pair(
            positions=positions.detach(),
            cutoff=cutoff,
            cell=cell.detach(),
            pbc=pbc,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            num_neighbors=num_neighbors,
            neighbor_vectors=neighbor_vectors,
            neighbor_distances=neighbor_distances,
            shift_range_per_dimension=shift_range_per_dimension,
            num_shifts_per_system=num_shifts_per_system,
            max_shifts_per_system=int(max_shifts_per_system),
            half_fill=half_fill,
            wrap_positions=wrap_positions,
        )
    elif pair_fn is None:
        _naive_neighbor_matrix_pbc_pair_target(
            positions=positions.detach(),
            cutoff=cutoff,
            cell=cell.detach(),
            pbc=pbc,
            target_indices=target_indices,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            num_neighbors=num_neighbors,
            neighbor_vectors=neighbor_vectors,
            neighbor_distances=neighbor_distances,
            shift_range_per_dimension=shift_range_per_dimension,
            num_shifts_per_system=num_shifts_per_system,
            max_shifts_per_system=int(max_shifts_per_system),
            half_fill=half_fill,
            wrap_positions=wrap_positions,
        )
    else:
        if torch.compiler.is_compiling():
            raise NotImplementedError(
                "naive pair_fn kernels are eager-only because Python callables "
                "cannot cross a torch custom-op boundary.",
            )
        # ``pair_fn`` and/or ``target_indices`` bypass the custom op and call
        # the Warp scalar launcher directly. Torch custom ops cannot carry a
        # Python pair_fn, and they do not expose compact partial rows.
        wp_dtype = get_wp_dtype(positions.dtype)
        wp_vec_dtype = get_wp_vec_dtype(positions.dtype)
        pair_kwargs = {}
        if pair_fn is not None:
            pair_kwargs = {
                "pair_fn": pair_fn,
                "pair_params": wp.from_torch(
                    pair_params, dtype=wp_dtype, requires_grad=False
                ),
                "pair_energies": wp.from_torch(
                    pair_energies, dtype=wp_dtype, requires_grad=False
                ),
                "pair_forces": wp.from_torch(
                    pair_forces, dtype=wp_vec_dtype, requires_grad=False
                ),
            }
        wp_target_indices = (
            wp.from_torch(
                target_indices, dtype=wp.int32, requires_grad=False, return_ctype=True
            )
            if is_partial
            else None
        )
        if pbc is None:
            naive_neighbor_matrix(
                positions=wp.from_torch(
                    positions.detach(),
                    dtype=wp_vec_dtype,
                    requires_grad=False,
                    return_ctype=True,
                ),
                cutoff=cutoff,
                neighbor_matrix=wp.from_torch(
                    neighbor_matrix,
                    dtype=wp.int32,
                    requires_grad=False,
                    return_ctype=True,
                ),
                num_neighbors=wp.from_torch(
                    num_neighbors,
                    dtype=wp.int32,
                    requires_grad=False,
                    return_ctype=True,
                ),
                wp_dtype=wp_dtype,
                device=str(positions.device),
                half_fill=half_fill,
                rebuild_flags=None,
                target_indices=wp_target_indices,
                return_vectors=True,
                return_distances=True,
                neighbor_vectors=wp.from_torch(
                    neighbor_vectors, dtype=wp_vec_dtype, requires_grad=False
                ),
                neighbor_distances=wp.from_torch(
                    neighbor_distances, dtype=wp_dtype, requires_grad=False
                ),
                **pair_kwargs,
            )
        else:
            wp_mat_dtype = get_wp_mat_dtype(cell.dtype)
            naive_neighbor_matrix_pbc(
                positions=wp.from_torch(
                    positions.detach(),
                    dtype=wp_vec_dtype,
                    requires_grad=False,
                    return_ctype=True,
                ),
                cutoff=cutoff,
                cell=wp.from_torch(
                    cell.detach(),
                    dtype=wp_mat_dtype,
                    requires_grad=False,
                    return_ctype=True,
                ),
                pbc=wp.from_torch(
                    pbc, dtype=wp.bool, requires_grad=False, return_ctype=True
                ),
                shift_range=wp.from_torch(
                    shift_range_per_dimension,
                    dtype=wp.vec3i,
                    requires_grad=False,
                    return_ctype=True,
                ),
                num_shifts=int(max_shifts_per_system),
                neighbor_matrix=wp.from_torch(
                    neighbor_matrix,
                    dtype=wp.int32,
                    requires_grad=False,
                    return_ctype=True,
                ),
                neighbor_matrix_shifts=wp.from_torch(
                    neighbor_matrix_shifts,
                    dtype=wp.vec3i,
                    requires_grad=False,
                    return_ctype=True,
                ),
                num_neighbors=wp.from_torch(
                    num_neighbors,
                    dtype=wp.int32,
                    requires_grad=False,
                    return_ctype=True,
                ),
                wp_dtype=wp_dtype,
                device=str(positions.device),
                half_fill=half_fill,
                rebuild_flags=None,
                wrap_positions=wrap_positions,
                target_indices=wp_target_indices,
                return_vectors=True,
                return_distances=True,
                neighbor_vectors=wp.from_torch(
                    neighbor_vectors, dtype=wp_vec_dtype, requires_grad=False
                ),
                neighbor_distances=wp.from_torch(
                    neighbor_distances, dtype=wp_dtype, requires_grad=False
                ),
                **pair_kwargs,
            )
    shifts_arg = (
        neighbor_matrix_shifts
        if neighbor_matrix_shifts is not None
        else torch.zeros(
            (*neighbor_matrix.shape, 3),
            dtype=torch.int32,
            device=neighbor_matrix.device,
        )
    )
    i_idx, j_idx, shifts_flat, batch_idx_flat, mask = _flatten_active_pairs(
        neighbor_matrix,
        num_neighbors,
        shifts_arg,
        target_indices=target_indices,
    )
    K, M = neighbor_matrix.shape
    return _NeighborForwardOutput(
        distances=neighbor_distances,
        vectors=neighbor_vectors,
        extra_outputs=(neighbor_matrix, num_neighbors, shifts_arg),
        i_idx_flat=i_idx,
        j_idx_flat=j_idx,
        shifts_flat=shifts_flat,
        batch_idx_flat=batch_idx_flat,
        active_mask=mask,
        matrix_shape=(K, M),
    )


def naive_neighbor_list(
    positions: torch.Tensor,
    cutoff: float,
    cell: torch.Tensor | None = None,
    pbc: torch.Tensor | None = None,
    max_neighbors: int | None = None,
    half_fill: bool = False,
    fill_value: int | None = None,
    return_neighbor_list: bool = False,
    neighbor_matrix: torch.Tensor | None = None,
    neighbor_matrix_shifts: torch.Tensor | None = None,
    num_neighbors: torch.Tensor | None = None,
    shift_range_per_dimension: torch.Tensor | None = None,
    num_shifts_per_system: torch.Tensor | None = None,
    max_shifts_per_system: int | None = None,
    rebuild_flags: torch.Tensor | None = None,
    wrap_positions: bool = True,
    positions_wrapped_buffer: torch.Tensor | None = None,
    per_atom_cell_offsets_buffer: torch.Tensor | None = None,
    inv_cell_buffer: torch.Tensor | None = None,
    *,
    return_distances: bool = False,
    return_vectors: bool = False,
    neighbor_vectors: torch.Tensor | None = None,
    neighbor_distances: torch.Tensor | None = None,
    pair_fn: wp.Function | CompiledPairFn | None = None,
    pair_params: torch.Tensor | None = None,
    pair_energies: torch.Tensor | None = None,
    pair_forces: torch.Tensor | None = None,
    strategy: str = "auto",
    target_indices: torch.Tensor | None = None,
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor]
):
    """Compute neighbor list using naive O(N^2) algorithm.

    Identifies all atom pairs within a specified cutoff distance using a
    brute-force pairwise distance calculation. Supports both non-periodic
    and periodic boundary conditions.

    For non-pbc systems, this function is torch compilable. For pbc systems,
    precompute the shift metadata using compute_naive_num_shifts.

    Parameters
    ----------
    positions : torch.Tensor, shape (total_atoms, 3), dtype=torch.float32 or torch.float64
        Atomic coordinates in Cartesian space. Each row represents one atom's
        (x, y, z) position.
    cutoff : float
        Cutoff distance for neighbor detection in Cartesian units.
        Must be positive. Atoms within this distance are considered neighbors.
    pbc : torch.Tensor, shape (1, 3), dtype=torch.bool, optional
        Periodic boundary condition flags for each dimension.
        True enables periodicity in that direction. Default is None (no PBC).
    cell : torch.Tensor, shape (1, 3, 3), dtype=torch.float32 or torch.float64, optional
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
    neighbor_matrix : torch.Tensor, shape (num_rows, max_neighbors), dtype=torch.int32, optional
        Neighbor matrix to be filled. Pass in a pre-allocated tensor to avoid reallocation.
        ``num_rows`` is ``total_atoms`` normally and ``len(target_indices)`` when
        partial rows are requested.
        Must be provided if max_neighbors is not provided.
    neighbor_matrix_shifts : torch.Tensor, shape (num_rows, max_neighbors, 3), dtype=torch.int32, optional
        Shift vectors for each neighbor relationship. Pass in a pre-allocated tensor to avoid reallocation.
        Must be provided if max_neighbors is not provided.
    num_neighbors : torch.Tensor, shape (num_rows,), dtype=torch.int32, optional
        Number of neighbors found for each atom. Pass in a pre-allocated tensor to avoid reallocation.
        Must be provided if max_neighbors is not provided.
    shift_range_per_dimension : torch.Tensor, shape (1, 3), dtype=torch.int32, optional
        Shift range in each dimension for each system.
        Pass in a pre-allocated tensor to avoid reallocation for pbc systems.
    num_shifts_per_system : torch.Tensor, shape (1,), dtype=torch.int32, optional
        Number of periodic shifts for the system.
        Pass in to avoid recomputation for pbc systems.
    max_shifts_per_system : int, optional
        Maximum shift count across all systems.
        Pass in to avoid recomputation for pbc systems.
    rebuild_flags : torch.Tensor, shape () or (1,), dtype=torch.bool, optional
        If provided, controls whether the neighbor list is recomputed.
        When the flag is False the existing ``neighbor_matrix``, ``num_neighbors``,
        and ``neighbor_matrix_shifts`` tensors are returned unchanged and all
        kernel launches are skipped.  When the flag is True (or when this argument
        is None) the neighbor list is recomputed as normal.
        Note: providing this argument disables torch.compile compatibility.
    wrap_positions : bool, default=True
        If True, wrap input positions into the primary cell before
        neighbor search. Set to False when positions are already
        wrapped (e.g. by a preceding integration step) to save two
        GPU kernel launches per call.
    return_neighbor_list : bool, optional - default = False
        If True, convert the neighbor matrix to a neighbor list (idx_i, idx_j) format by
        creating a mask over the fill_value, which can incur a performance penalty.
        We recommend using the neighbor matrix format,
        and only convert to a neighbor list format if absolutely necessary.
    target_indices : torch.Tensor, shape (num_targets,), dtype=torch.int32, optional
        Compact partial-list source rows. Output row ``r`` maps to atom
        ``target_indices[r]``; COO source rows remain compact row ids. User
        buffers must be compact-row shaped, not full atom-row shaped.

    Returns
    -------
    results : tuple of torch.Tensor
        Variable-length tuple depending on input parameters. The return pattern follows:

        - No PBC, matrix format: ``(neighbor_matrix, num_neighbors)``
        - No PBC, list format: ``(neighbor_list, neighbor_ptr)``
        - With PBC, matrix format: ``(neighbor_matrix, num_neighbors, neighbor_matrix_shifts)``
        - With PBC, list format: ``(neighbor_list, neighbor_ptr, neighbor_list_shifts)``

        **Components returned:**

        - **neighbor_data** (tensor): Neighbor indices, format depends on ``return_neighbor_list``:

            * If ``return_neighbor_list=False`` (default): Returns ``neighbor_matrix``
              with shape (num_rows, max_neighbors), dtype int32. Row ``r`` contains
              neighbors for atom ``r`` or ``target_indices[r]`` when partial rows
              are requested.
            * If ``return_neighbor_list=True``: Returns ``neighbor_list`` with shape
              (2, num_pairs), dtype int32, in COO format [source_rows, target_atoms].
              With ``target_indices``, source rows are compact row ids.

        - **num_neighbor_data** (tensor): Information about the number of neighbors for each atom,
          format depends on ``return_neighbor_list``:

            * If ``return_neighbor_list=False`` (default): Returns ``num_neighbors`` with shape (num_rows,), dtype int32.
              Count of neighbors found for each atom. Always returned.
            * If ``return_neighbor_list=True``: Returns ``neighbor_ptr`` with shape (num_rows + 1,), dtype int32.
              CSR-style pointer arrays where ``neighbor_ptr_data[i]`` to ``neighbor_ptr_data[i+1]`` gives the range of
              neighbors for row i in the flattened neighbor list.

        - **neighbor_shift_data** (tensor, optional): Periodic shift vectors, only when ``pbc`` is provided:
          format depends on ``return_neighbor_list``:

            * If ``return_neighbor_list=False`` (default): Returns ``neighbor_matrix_shifts`` with
              shape (num_rows, max_neighbors, 3), dtype int32.
            * If ``return_neighbor_list=True``: Returns ``unit_shifts`` with shape
              (num_pairs, 3), dtype int32.

    Examples
    --------
    Basic usage without periodic boundary conditions:

    >>> import torch
    >>> positions = torch.rand(100, 3) * 10.0  # 100 atoms in 10x10x10 box
    >>> cutoff = 2.5
    >>> max_neighbors = 50
    >>> neighbor_matrix, num_neighbors = naive_neighbor_list(
    ...     positions, cutoff, max_neighbors
    ... )
    >>> print(f"Found {num_neighbors.sum()} total neighbor pairs")

    With periodic boundary conditions:

    >>> cell = torch.eye(3).unsqueeze(0) * 10.0  # 10x10x10 cubic cell
    >>> pbc = torch.tensor([[True, True, True]])  # Periodic in all directions
    >>> neighbor_matrix, num_neighbors, shifts = naive_neighbor_list(
    ...     positions, cutoff, max_neighbors, pbc=pbc, cell=cell
    ... )

    Return as neighbor list instead of matrix:

    >>> neighbor_list, neighbor_ptr = naive_neighbor_list(
    ...     positions, cutoff, max_neighbors, return_neighbor_list=True
    ... )
    >>> source_atoms, target_atoms = neighbor_list[0], neighbor_list[1]

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
        cell = cell if cell.ndim == 3 else cell.unsqueeze(0)
    if pbc is not None:
        pbc = pbc if pbc.ndim == 2 else pbc.unsqueeze(0)

    has_pair_outputs = (
        bool(return_distances)
        or bool(return_vectors)
        or neighbor_vectors is not None
        or neighbor_distances is not None
        or pair_fn is not None
        or pair_energies is not None
        or pair_forces is not None
        or target_indices is not None
    )
    if strategy == "tile" and target_indices is not None:
        raise NotImplementedError(
            "strategy='tile' has no target_indices (partial "
            "neighbor-list) variant; use strategy='scalar'.",
        )
    if has_pair_outputs:
        if is_compiled_pair_fn(pair_fn) and torch.compiler.is_compiling():
            if return_neighbor_list:
                raise NotImplementedError(
                    "CompiledPairFn supports torch.compile(fullgraph=True) for "
                    "matrix neighbor-list output only; use return_neighbor_list=False.",
                )
            missing = [
                name
                for name, value in (
                    ("neighbor_matrix", neighbor_matrix),
                    ("num_neighbors", num_neighbors),
                    ("neighbor_vectors", neighbor_vectors),
                    ("neighbor_distances", neighbor_distances),
                    ("pair_params", pair_params),
                    ("pair_energies", pair_energies),
                    ("pair_forces", pair_forces),
                )
                if value is None
            ]
            if pbc is not None:
                missing.extend(
                    name
                    for name, value in (
                        ("neighbor_matrix_shifts", neighbor_matrix_shifts),
                        ("shift_range_per_dimension", shift_range_per_dimension),
                        ("num_shifts_per_system", num_shifts_per_system),
                        ("max_shifts_per_system", max_shifts_per_system),
                    )
                    if value is None
                )
            if missing:
                raise ValueError(
                    "CompiledPairFn under torch.compile(fullgraph=True) requires "
                    "fixed-shape caller-provided buffers/metadata; missing "
                    f"{', '.join(missing)}.",
                )
        _validate_pair_params_present(pair_fn, pair_params)
        if rebuild_flags is not None:
            raise NotImplementedError(
                "Pair outputs are not supported with rebuild_flags.",
            )
        if max_neighbors is None and neighbor_matrix is not None:
            max_neighbors = int(neighbor_matrix.shape[1])
        if max_neighbors is None:
            max_neighbors = estimate_max_neighbors(cutoff)
        if fill_value is None:
            fill_value = positions.shape[0]
        num_rows = (
            int(target_indices.shape[0])
            if target_indices is not None
            else positions.shape[0]
        )
        if target_indices is not None:
            _validate_output_buffer(
                "neighbor_matrix",
                neighbor_matrix,
                (num_rows, max_neighbors),
                torch.int32,
            )
            _validate_output_buffer(
                "num_neighbors",
                num_neighbors,
                (num_rows,),
                torch.int32,
            )
            if pbc is not None:
                _validate_output_buffer(
                    "neighbor_matrix_shifts",
                    neighbor_matrix_shifts,
                    (num_rows, max_neighbors, 3),
                    torch.int32,
                )
            _validate_output_buffer(
                "neighbor_distances",
                neighbor_distances,
                (num_rows, max_neighbors),
                positions.dtype,
            )
            _validate_output_buffer(
                "neighbor_vectors",
                neighbor_vectors,
                (num_rows, max_neighbors, 3),
                positions.dtype,
            )
            _validate_output_buffer(
                "pair_energies",
                pair_energies,
                (num_rows, max_neighbors),
                positions.dtype,
            )
            _validate_output_buffer(
                "pair_forces",
                pair_forces,
                (num_rows, max_neighbors, 3),
                positions.dtype,
            )
        if neighbor_matrix is None:
            neighbor_matrix = torch.full(
                (num_rows, max_neighbors),
                fill_value,
                dtype=torch.int32,
                device=positions.device,
            )
        else:
            neighbor_matrix.fill_(fill_value)
        if num_neighbors is None:
            num_neighbors = torch.zeros(
                num_rows,
                dtype=torch.int32,
                device=positions.device,
            )
        else:
            num_neighbors.zero_()
        if pbc is not None:
            if neighbor_matrix_shifts is None:
                neighbor_matrix_shifts = torch.zeros(
                    (num_rows, max_neighbors, 3),
                    dtype=torch.int32,
                    device=positions.device,
                )
            else:
                neighbor_matrix_shifts.zero_()
            if (
                max_shifts_per_system is None
                or num_shifts_per_system is None
                or shift_range_per_dimension is None
            ):
                (
                    shift_range_per_dimension,
                    num_shifts_per_system,
                    max_shifts_per_system,
                ) = compute_naive_num_shifts(cell.detach(), cutoff, pbc)
        if neighbor_distances is None:
            neighbor_distances = torch.zeros(
                (num_rows, max_neighbors),
                dtype=positions.dtype,
                device=positions.device,
            )
        if neighbor_vectors is None:
            neighbor_vectors = torch.zeros(
                (num_rows, max_neighbors, 3),
                dtype=positions.dtype,
                device=positions.device,
            )
        # ``pair_fn`` energy/force buffers are optional: allocate them like the
        # neighbor matrix when not supplied, so they can be returned.
        if pair_fn is not None and pair_energies is None:
            pair_energies = torch.zeros(
                (num_rows, max_neighbors),
                dtype=positions.dtype,
                device=positions.device,
            )
        if pair_fn is not None and pair_forces is None:
            pair_forces = torch.zeros(
                (num_rows, max_neighbors, 3),
                dtype=positions.dtype,
                device=positions.device,
            )
        forward_kwargs = {
            "cutoff": cutoff,
            "pbc": pbc,
            "neighbor_matrix": neighbor_matrix,
            "neighbor_matrix_shifts": neighbor_matrix_shifts,
            "num_neighbors": num_neighbors,
            "neighbor_vectors": neighbor_vectors,
            "neighbor_distances": neighbor_distances,
            "half_fill": half_fill,
            "shift_range_per_dimension": shift_range_per_dimension,
            "num_shifts_per_system": num_shifts_per_system,
            "max_shifts_per_system": max_shifts_per_system,
            "wrap_positions": wrap_positions,
            "target_indices": target_indices,
            "pair_fn": pair_fn,
            "pair_params": pair_params,
            "pair_energies": pair_energies,
            "pair_forces": pair_forces,
        }
        distances_out, vectors_out, nm_out, nn_out, shifts_out = _route_pair_outputs(
            positions,
            cell,
            _naive_pair_outputs_forward,
            forward_kwargs,
        )
        if return_neighbor_list:
            if pbc is not None:
                nl, nptr, nl_shifts = get_neighbor_list_from_neighbor_matrix(
                    nm_out,
                    num_neighbors=nn_out,
                    neighbor_shift_matrix=shifts_out,
                    fill_value=fill_value,
                )
                base = (nl, nptr, nl_shifts)
            else:
                nl, nptr = get_neighbor_list_from_neighbor_matrix(
                    nm_out,
                    num_neighbors=nn_out,
                    fill_value=fill_value,
                )
                base = (nl, nptr)
            # Repack the per-pair outputs into the same COO order as the
            # neighbor list so they index-align with it; ``index_select`` keeps
            # the autograd link.
            active = nm_out != fill_value
            distances_out, vectors_out = coo_pack_pair_geometry(
                active, distances_out, vectors_out
            )
            pe_out, pf_out = coo_pack_pair_geometry(active, pair_energies, pair_forces)
        elif pbc is not None:
            base = (nm_out, nn_out, shifts_out)
            pe_out, pf_out = pair_energies, pair_forces
        else:
            base = (nm_out, nn_out)
            pe_out, pf_out = pair_energies, pair_forces

        tail: list[torch.Tensor] = []
        if return_distances:
            tail.append(distances_out)
        if return_vectors:
            tail.append(vectors_out)
        if pair_fn is not None:
            tail.extend((pe_out, pf_out))
        return (*base, *tail)

    if max_neighbors is None and (
        neighbor_matrix is None
        or (neighbor_matrix_shifts is None and pbc is not None)
        or num_neighbors is None
    ):
        max_neighbors = estimate_max_neighbors(cutoff)

    if fill_value is None:
        fill_value = positions.shape[0]

    if neighbor_matrix is None:
        neighbor_matrix = torch.full(
            (positions.shape[0], max_neighbors),
            fill_value,
            dtype=torch.int32,
            device=positions.device,
        )
    elif rebuild_flags is None:
        neighbor_matrix.fill_(fill_value)

    if num_neighbors is None:
        num_neighbors = torch.zeros(
            positions.shape[0], dtype=torch.int32, device=positions.device
        )
    elif rebuild_flags is None:
        num_neighbors.zero_()

    if pbc is not None:
        if neighbor_matrix_shifts is None:
            neighbor_matrix_shifts = torch.zeros(
                (positions.shape[0], max_neighbors, 3),
                dtype=torch.int32,
                device=positions.device,
            )
        elif rebuild_flags is None:
            neighbor_matrix_shifts.zero_()
        if (
            max_shifts_per_system is None
            or num_shifts_per_system is None
            or shift_range_per_dimension is None
        ):
            shift_range_per_dimension, num_shifts_per_system, max_shifts_per_system = (
                compute_naive_num_shifts(cell, cutoff, pbc)
            )

    if cutoff <= 0:
        if return_neighbor_list:
            if pbc is not None:
                return (
                    torch.zeros((2, 0), dtype=torch.int32, device=positions.device),
                    torch.zeros(
                        (positions.shape[0] + 1,),
                        dtype=torch.int32,
                        device=positions.device,
                    ),
                    torch.zeros((0, 3), dtype=torch.int32, device=positions.device),
                )
            else:
                return (
                    torch.zeros((2, 0), dtype=torch.int32, device=positions.device),
                    torch.zeros(
                        (positions.shape[0] + 1,),
                        dtype=torch.int32,
                        device=positions.device,
                    ),
                )
        else:
            if pbc is not None:
                return neighbor_matrix, num_neighbors, neighbor_matrix_shifts
            else:
                return neighbor_matrix, num_neighbors

    if pbc is None:
        _naive_neighbor_matrix_no_pbc(
            positions=positions,
            cutoff=cutoff,
            neighbor_matrix=neighbor_matrix,
            num_neighbors=num_neighbors,
            half_fill=half_fill,
            rebuild_flags=rebuild_flags,
            strategy=strategy,
        )
        if return_neighbor_list:
            neighbor_list, neighbor_ptr = get_neighbor_list_from_neighbor_matrix(
                neighbor_matrix,
                num_neighbors=num_neighbors,
                fill_value=fill_value,
            )
            return neighbor_list, neighbor_ptr
        else:
            return neighbor_matrix, num_neighbors
    else:
        _naive_neighbor_matrix_pbc(
            positions=positions,
            cutoff=cutoff,
            cell=cell,
            pbc=pbc,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            num_neighbors=num_neighbors,
            shift_range_per_dimension=shift_range_per_dimension,
            num_shifts_per_system=num_shifts_per_system,
            max_shifts_per_system=max_shifts_per_system,
            half_fill=half_fill,
            rebuild_flags=rebuild_flags,
            wrap_positions=wrap_positions,
            positions_wrapped_buffer=positions_wrapped_buffer,
            per_atom_cell_offsets_buffer=per_atom_cell_offsets_buffer,
            inv_cell_buffer=inv_cell_buffer,
            strategy=strategy,
        )
        if return_neighbor_list:
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
            return neighbor_matrix, num_neighbors, neighbor_matrix_shifts
