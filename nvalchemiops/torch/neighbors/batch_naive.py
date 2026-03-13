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

"""PyTorch bindings for batched naive neighbor list construction."""

from __future__ import annotations

import torch
import warp as wp

from nvalchemiops.neighbors.batch_naive import (
    batch_naive_neighbor_matrix,
    batch_naive_neighbor_matrix_pbc,
)
from nvalchemiops.neighbors.neighbor_utils import (
    estimate_max_neighbors,
    selective_zero_num_neighbors,
)
from nvalchemiops.torch.neighbors.neighbor_utils import (
    compute_naive_num_shifts,
    get_neighbor_list_from_neighbor_matrix,
    prepare_batch_idx_ptr,
)
from nvalchemiops.torch.types import get_wp_dtype, get_wp_mat_dtype, get_wp_vec_dtype

__all__ = ["batch_naive_neighbor_list"]


@torch.library.custom_op(
    "nvalchemiops::_naive_batch_neighbor_matrix_no_pbc",
    mutates_args=("neighbor_matrix", "num_neighbors"),
)
def _batch_naive_neighbor_matrix_no_pbc(
    positions: torch.Tensor,
    cutoff: float,
    batch_idx: torch.Tensor,
    batch_ptr: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    num_neighbors: torch.Tensor,
    half_fill: bool,
    rebuild_flags: torch.Tensor | None = None,
) -> None:
    """Fill neighbor matrix for batch of atoms using naive O(N^2) algorithm.

    Custom PyTorch operator that computes pairwise distances and fills
    the neighbor matrix with atom indices within the cutoff distance.
    Processes multiple systems in a batch where atoms from different systems
    do not interact. No periodic boundary conditions are applied.

    This function does not allocate any tensors.

    This function is torch compilable.

    Parameters
    ----------
    positions : torch.Tensor, shape (total_atoms, 3), dtype=torch.float32 or torch.float64
        Concatenated Cartesian coordinates for all systems.
        Each row represents one atom's (x, y, z) position.
    cutoff : float
        Cutoff distance for neighbor detection in Cartesian units.
        Must be positive. Atoms within this distance are considered neighbors.
    batch_idx : torch.Tensor, shape (total_atoms,), dtype=torch.int32
        System index for each atom. Atoms with the same index belong to
        the same system and can be neighbors.
    batch_ptr : torch.Tensor, shape (num_systems + 1,), dtype=torch.int32
        Cumulative atom counts defining system boundaries.
        System i contains atoms from batch_ptr[i] to batch_ptr[i+1]-1.
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
    nvalchemiops.neighbors.batch_naive.batch_naive_neighbor_matrix : Core warp launcher
    batch_naive_neighbor_list : Higher-level wrapper function
    """
    device = positions.device
    wp_dtype = get_wp_dtype(positions.dtype)
    wp_vec_dtype = get_wp_vec_dtype(positions.dtype)

    wp_positions = wp.from_torch(positions, dtype=wp_vec_dtype, return_ctype=True)
    wp_batch_idx = wp.from_torch(batch_idx, dtype=wp.int32, return_ctype=True)
    wp_batch_ptr = wp.from_torch(batch_ptr, dtype=wp.int32, return_ctype=True)
    wp_neighbor_matrix = wp.from_torch(
        neighbor_matrix, dtype=wp.int32, return_ctype=True
    )
    wp_num_neighbors = wp.from_torch(num_neighbors, dtype=wp.int32, return_ctype=True)
    if rebuild_flags is not None:
        wp_rebuild_flags = wp.from_torch(
            rebuild_flags, dtype=wp.bool, return_ctype=True
        )
        selective_zero_num_neighbors(
            wp_num_neighbors, wp_batch_idx, wp_rebuild_flags, str(device)
        )
    else:
        wp_rebuild_flags = None
    batch_naive_neighbor_matrix(
        positions=wp_positions,
        cutoff=cutoff,
        batch_idx=wp_batch_idx,
        batch_ptr=wp_batch_ptr,
        neighbor_matrix=wp_neighbor_matrix,
        num_neighbors=wp_num_neighbors,
        wp_dtype=wp_dtype,
        device=str(device),
        half_fill=half_fill,
        rebuild_flags=wp_rebuild_flags,
    )


@torch.library.custom_op(
    "nvalchemiops::_batch_naive_neighbor_matrix_pbc",
    mutates_args=("neighbor_matrix", "neighbor_matrix_shifts", "num_neighbors"),
)
def _batch_naive_neighbor_matrix_pbc(
    positions: torch.Tensor,
    cell: torch.Tensor,
    cutoff: float,
    batch_idx: torch.Tensor,
    batch_ptr: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    num_neighbors: torch.Tensor,
    shift_range_per_dimension: torch.Tensor,
    num_shifts_per_system: torch.Tensor,
    max_shifts_per_system: int,
    half_fill: bool = False,
    max_atoms_per_system: int | None = None,
    rebuild_flags: torch.Tensor | None = None,
    wrap_positions: bool = True,
) -> None:
    """Compute batch neighbor matrix with PBC using naive O(N^2) algorithm.

    This function is torch compilable.

    Parameters
    ----------
    positions : torch.Tensor, shape (total_atoms, 3)
        Concatenated Cartesian coordinates for all systems.
    cell : torch.Tensor, shape (num_systems, 3, 3)
        Cell matrices defining lattice vectors.
    cutoff : float
        Cutoff distance for neighbor detection.
    batch_idx : torch.Tensor, shape (total_atoms,), dtype=torch.int32
        System index for each atom.
    batch_ptr : torch.Tensor, shape (num_systems + 1,), dtype=torch.int32
        Cumulative atom counts defining system boundaries.
    neighbor_matrix : torch.Tensor, shape (total_atoms, max_neighbors), dtype=torch.int32
        OUTPUT: Neighbor matrix.
    neighbor_matrix_shifts : torch.Tensor, shape (total_atoms, max_neighbors, 3), dtype=torch.int32
        OUTPUT: Shift vectors for each neighbor.
    num_neighbors : torch.Tensor, shape (total_atoms,), dtype=torch.int32
        OUTPUT: Number of neighbors per atom.
    shift_range_per_dimension : torch.Tensor, shape (num_systems, 3), dtype=torch.int32
        Shift range in each dimension for each system.
    num_shifts_per_system : torch.Tensor, shape (num_systems,), dtype=torch.int32
        Number of periodic shifts per system.
    max_shifts_per_system : int
        Maximum per-system shift count (launch dimension).
    half_fill : bool, optional
        If True, only store relationships where i < j. Default is False.
    max_atoms_per_system : int, optional
        Maximum atoms per system. Computed automatically if not provided.
    rebuild_flags : torch.Tensor, shape (num_systems,), dtype=torch.bool, optional
        Per-system rebuild flags. Non-rebuilt systems are skipped on GPU.
    wrap_positions : bool, default=True
        If True, wrap positions into the primary cell before neighbor search.

    See Also
    --------
    nvalchemiops.neighbors.batch_naive.batch_naive_neighbor_matrix_pbc : Core warp launcher
    batch_naive_neighbor_list : Higher-level wrapper function
    """
    device = positions.device
    wp_device = wp.device_from_torch(device)
    wp_vec_dtype = get_wp_vec_dtype(positions.dtype)
    wp_mat_dtype = get_wp_mat_dtype(positions.dtype)
    wp_dtype = get_wp_dtype(positions.dtype)

    wp_positions = wp.from_torch(positions, dtype=wp_vec_dtype, return_ctype=True)
    wp_cell = wp.from_torch(cell, dtype=wp_mat_dtype, return_ctype=True)
    wp_shift_range = wp.from_torch(
        shift_range_per_dimension, dtype=wp.vec3i, return_ctype=True
    )
    wp_num_shifts_arr = wp.from_torch(
        num_shifts_per_system, dtype=wp.int32, return_ctype=True
    )
    wp_neighbor_matrix = wp.from_torch(
        neighbor_matrix, dtype=wp.int32, return_ctype=True
    )
    wp_neighbor_matrix_shifts = wp.from_torch(
        neighbor_matrix_shifts, dtype=wp.vec3i, return_ctype=True
    )
    wp_num_neighbors = wp.from_torch(num_neighbors, dtype=wp.int32, return_ctype=True)
    wp_batch_idx = wp.from_torch(batch_idx, dtype=wp.int32, return_ctype=True)
    wp_batch_ptr = wp.from_torch(batch_ptr, dtype=wp.int32, return_ctype=True)

    if max_atoms_per_system is None:
        max_atoms_per_system = (batch_ptr[1:] - batch_ptr[:-1]).max().item()

    if rebuild_flags is not None:
        wp_rebuild_flags = wp.from_torch(
            rebuild_flags, dtype=wp.bool, return_ctype=True
        )
        selective_zero_num_neighbors(
            wp_num_neighbors, wp_batch_idx, wp_rebuild_flags, str(wp_device)
        )
    else:
        wp_rebuild_flags = None

    batch_naive_neighbor_matrix_pbc(
        positions=wp_positions,
        cell=wp_cell,
        cutoff=cutoff,
        batch_ptr=wp_batch_ptr,
        batch_idx=wp_batch_idx,
        shift_range=wp_shift_range,
        num_shifts_arr=wp_num_shifts_arr,
        max_shifts_per_system=max_shifts_per_system,
        neighbor_matrix=wp_neighbor_matrix,
        neighbor_matrix_shifts=wp_neighbor_matrix_shifts,
        num_neighbors=wp_num_neighbors,
        wp_dtype=wp_dtype,
        device=str(device),
        max_atoms_per_system=max_atoms_per_system,
        half_fill=half_fill,
        rebuild_flags=wp_rebuild_flags,
        wrap_positions=wrap_positions,
    )


def batch_naive_neighbor_list(
    positions: torch.Tensor,
    cutoff: float,
    batch_idx: torch.Tensor | None = None,
    batch_ptr: torch.Tensor | None = None,
    pbc: torch.Tensor | None = None,
    cell: torch.Tensor | None = None,
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
    max_atoms_per_system: int | None = None,
    rebuild_flags: torch.Tensor | None = None,
    wrap_positions: bool = True,
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor]
):
    """Compute batch neighbor matrix using naive O(N^2) algorithm.

    Identifies all atom pairs within a specified cutoff distance for multiple
    systems processed in a batch. Each system is processed independently,
    supporting both non-periodic and periodic boundary conditions.

    For efficiency, this function supports in-place modification of pre-allocated tensors.
    If not provided, the resulting tensors will be allocated.
    This function does not introduce CUDA graph breaks for non-PBC systems.
    For PBC systems, pre-compute unit shifts to avoid CUDA graph breaks.

    Parameters
    ----------
    positions : torch.Tensor, shape (total_atoms, 3), dtype=torch.float32 or torch.float64
        Concatenated Cartesian coordinates for all systems.
        Each row represents one atom's (x, y, z) position.
        Unwrapped (box-crossing) coordinates are supported when PBC is used;
        the kernel wraps positions internally.
    cutoff : float
        Cutoff distance for neighbor detection in Cartesian units.
        Must be positive. Atoms within this distance are considered neighbors.
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
    max_neighbors : int, optional
        Maximum number of neighbors per atom. Must be positive.
        If exceeded, excess neighbors are ignored.
        Must be provided if neighbor_matrix is not provided.
    half_fill : bool, optional
        If True, only store half of the neighbor relationships to avoid double counting.
        Another half could be reconstructed by swapping source and target indices and inverting unit shifts.
        If False, store all neighbor relationships. Default is False.
    fill_value : int | None, optional
        Value to fill the neighbor matrix with. Default is total_atoms.
    return_neighbor_list : bool, optional - default = False
        If True, convert the neighbor matrix to a neighbor list (idx_i, idx_j) format by
        creating a mask over the fill_value, which can incur a performance penalty.
        We recommend using the neighbor matrix format,
        and only convert to a neighbor list format if absolutely necessary.
    neighbor_matrix : torch.Tensor, shape (total_atoms, max_neighbors), dtype=torch.int32, optional
        Optional pre-allocated tensor for the neighbor matrix.
        Must be provided if max_neighbors is not provided.
    neighbor_matrix_shifts : torch.Tensor, shape (total_atoms, max_neighbors, 3), dtype=torch.int32, optional
        Optional pre-allocated tensor for the shift vectors of the neighbor matrix.
        Must be provided if max_neighbors is not provided and pbc is not None.
    num_neighbors : torch.Tensor, shape (total_atoms,), dtype=torch.int32, optional
        Optional pre-allocated tensor for the number of neighbors in the neighbor matrix.
        Must be provided if max_neighbors is not provided.
    shift_range_per_dimension : torch.Tensor, shape (num_systems, 3), dtype=torch.int32, optional
        Optional pre-allocated tensor for the shift range in each dimension for each system.
    num_shifts_per_system : torch.Tensor, shape (num_systems,), dtype=torch.int32, optional
        Number of periodic shifts per system.
        Pass in to avoid recomputation for pbc systems.
    max_shifts_per_system : int, optional
        Maximum per-system shift count.
        Pass in to avoid recomputation for pbc systems.
    max_atoms_per_system : int, optional
        Maximum number of atoms per system.
        If not provided, it will be computed automatically. Can be provided to avoid CUDA synchronization.
    rebuild_flags : torch.Tensor, shape (num_systems,), dtype=torch.bool, optional
        Per-system rebuild flags produced by ``batch_neighbor_list_needs_rebuild``.
        If provided, only systems where rebuild_flags[i] is True are recomputed;
        existing data in ``neighbor_matrix`` and ``num_neighbors`` is preserved for
        non-rebuilt systems entirely on the GPU (no CPU-GPU sync). When this is used,
        pre-allocated ``neighbor_matrix`` and ``num_neighbors`` tensors must be provided
        and will not be globally zeroed — only rebuilt-system entries are reset.
    wrap_positions : bool, default=True
        If True, wrap input positions into the primary cell before
        neighbor search. Set to False when positions are already
        wrapped (e.g. by a preceding integration step) to save two
        GPU kernel launches per call.

    Returns
    -------
    results : tuple of torch.Tensor
        Variable-length tuple depending on input parameters. The return pattern follows:

        - No PBC, matrix format: ``(neighbor_matrix, num_neighbors)``
        - No PBC, list format: ``(neighbor_list, neighbor_ptr)``
        - With PBC, matrix format: ``(neighbor_matrix, num_neighbors, neighbor_matrix_shifts)``
        - With PBC, list format: ``(neighbor_list, neighbor_ptr, neighbor_list_shifts)``

    See Also
    --------
    nvalchemiops.neighbors.batch_naive.batch_naive_neighbor_matrix : Core warp launcher (no PBC)
    nvalchemiops.neighbors.batch_naive.batch_naive_neighbor_matrix_pbc : Core warp launcher (with PBC)
    batch_cell_list : O(N) cell list method for larger systems
    """
    if pbc is None and cell is not None:
        raise ValueError("If cell is provided, pbc must also be provided")
    if pbc is not None and cell is None:
        raise ValueError("If pbc is provided, cell must also be provided")

    if cell is not None:
        cell = cell if cell.ndim == 3 else cell.unsqueeze(0)
    if pbc is not None:
        pbc = pbc if pbc.ndim == 2 else pbc.unsqueeze(0)

    if max_neighbors is None and (
        neighbor_matrix is None
        or (neighbor_matrix_shifts is None and pbc is not None)
        or num_neighbors is None
    ):
        max_neighbors = estimate_max_neighbors(cutoff)

    total_atoms = positions.shape[0]
    if fill_value is None:
        fill_value = total_atoms

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

    batch_idx, batch_ptr = prepare_batch_idx_ptr(
        batch_idx=batch_idx,
        batch_ptr=batch_ptr,
        num_atoms=total_atoms,
        device=positions.device,
    )

    # Validate batch_idx size matches total_atoms (check here since prepare_batch_idx_ptr
    # is @torch.compile decorated and the check would be skipped during tracing)
    if batch_idx.shape[0] != total_atoms:
        raise RuntimeError(
            f"batch_idx length ({batch_idx.shape[0]}) does not match "
            f"num_atoms ({total_atoms}). batch_idx must have one entry per atom."
        )

    if pbc is None:
        _batch_naive_neighbor_matrix_no_pbc(
            positions=positions,
            cutoff=cutoff,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            neighbor_matrix=neighbor_matrix,
            num_neighbors=num_neighbors,
            half_fill=half_fill,
            rebuild_flags=rebuild_flags,
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
        _batch_naive_neighbor_matrix_pbc(
            positions=positions,
            cell=cell,
            cutoff=cutoff,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            num_neighbors=num_neighbors,
            shift_range_per_dimension=shift_range_per_dimension,
            num_shifts_per_system=num_shifts_per_system,
            max_shifts_per_system=max_shifts_per_system,
            half_fill=half_fill,
            max_atoms_per_system=max_atoms_per_system,
            rebuild_flags=rebuild_flags,
            wrap_positions=wrap_positions,
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
