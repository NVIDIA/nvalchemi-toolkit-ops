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
from nvalchemiops.torch.neighbors.neighbor_utils import (
    compute_naive_num_shifts,
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

    wp_positions = wp.from_torch(positions, dtype=wp_vec_dtype, return_ctype=True)
    wp_neighbor_matrix = wp.from_torch(
        neighbor_matrix, dtype=wp.int32, return_ctype=True
    )
    wp_num_neighbors = wp.from_torch(num_neighbors, dtype=wp.int32, return_ctype=True)
    if rebuild_flags is not None:
        wp_rebuild_flags = wp.from_torch(
            rebuild_flags, dtype=wp.bool, return_ctype=True
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
    )


@torch.library.custom_op(
    "nvalchemiops::_naive_neighbor_matrix_pbc",
    mutates_args=("neighbor_matrix", "neighbor_matrix_shifts", "num_neighbors"),
)
def _naive_neighbor_matrix_pbc(
    positions: torch.Tensor,
    cutoff: float,
    cell: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    num_neighbors: torch.Tensor,
    shift_range_per_dimension: torch.Tensor,
    num_shifts_per_system: torch.Tensor,
    max_shifts_per_system: int,
    half_fill: bool = False,
    rebuild_flags: torch.Tensor | None = None,
    wrap_positions: bool = True,
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

    See Also
    --------
    nvalchemiops.neighbors.naive.naive_neighbor_matrix_pbc : Core warp launcher
    naive_neighbor_list : High-level wrapper function
    """
    device = positions.device
    wp_dtype = get_wp_dtype(positions.dtype)
    wp_vec_dtype = get_wp_vec_dtype(positions.dtype)
    wp_mat_dtype = get_wp_mat_dtype(cell.dtype)

    wp_positions = wp.from_torch(positions, dtype=wp_vec_dtype, return_ctype=True)
    wp_cell = wp.from_torch(cell, dtype=wp_mat_dtype, return_ctype=True)
    wp_shift_range = wp.from_torch(
        shift_range_per_dimension, dtype=wp.vec3i, return_ctype=True
    )
    wp_neighbor_matrix = wp.from_torch(
        neighbor_matrix, dtype=wp.int32, return_ctype=True
    )
    wp_neighbor_matrix_shifts = wp.from_torch(
        neighbor_matrix_shifts, dtype=wp.vec3i, return_ctype=True
    )
    wp_num_neighbors = wp.from_torch(num_neighbors, dtype=wp.int32, return_ctype=True)

    if rebuild_flags is not None:
        wp_rebuild_flags = wp.from_torch(
            rebuild_flags, dtype=wp.bool, return_ctype=True
        )
        selective_zero_num_neighbors_single(
            wp_num_neighbors, wp_rebuild_flags, str(device)
        )
    else:
        wp_rebuild_flags = None

    naive_neighbor_matrix_pbc(
        positions=wp_positions,
        cutoff=cutoff,
        cell=wp_cell,
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
    neighbor_matrix : torch.Tensor, shape (total_atoms, max_neighbors), dtype=torch.int32, optional
        Neighbor matrix to be filled. Pass in a pre-allocated tensor to avoid reallocation.
        Must be provided if max_neighbors is not provided.
    neighbor_matrix_shifts : torch.Tensor, shape (total_atoms, max_neighbors, 3), dtype=torch.int32, optional
        Shift vectors for each neighbor relationship. Pass in a pre-allocated tensor to avoid reallocation.
        Must be provided if max_neighbors is not provided.
    num_neighbors : torch.Tensor, shape (total_atoms,), dtype=torch.int32, optional
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
              with shape (total_atoms, max_neighbors), dtype int32. Each row i contains
              indices of atom i's neighbors.
            * If ``return_neighbor_list=True``: Returns ``neighbor_list`` with shape
              (2, num_pairs), dtype int32, in COO format [source_atoms, target_atoms].

        - **num_neighbor_data** (tensor): Information about the number of neighbors for each atom,
          format depends on ``return_neighbor_list``:

            * If ``return_neighbor_list=False`` (default): Returns ``num_neighbors`` with shape (total_atoms,), dtype int32.
              Count of neighbors found for each atom. Always returned.
            * If ``return_neighbor_list=True``: Returns ``neighbor_ptr`` with shape (total_atoms + 1,), dtype int32.
              CSR-style pointer arrays where ``neighbor_ptr_data[i]`` to ``neighbor_ptr_data[i+1]`` gives the range of
              neighbors for atom i in the flattened neighbor list.

        - **neighbor_shift_data** (tensor, optional): Periodic shift vectors, only when ``pbc`` is provided:
          format depends on ``return_neighbor_list``:

            * If ``return_neighbor_list=False`` (default): Returns ``neighbor_matrix_shifts`` with
              shape (total_atoms, max_neighbors, 3), dtype int32.
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
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            num_neighbors=num_neighbors,
            shift_range_per_dimension=shift_range_per_dimension,
            num_shifts_per_system=num_shifts_per_system,
            max_shifts_per_system=max_shifts_per_system,
            half_fill=half_fill,
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
