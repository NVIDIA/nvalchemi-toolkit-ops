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
    _update_neighbor_matrix,
    _update_neighbor_matrix_pbc,
    compute_naive_num_shifts,
    estimate_max_neighbors,
    get_neighbor_list_from_neighbor_matrix,
)
from nvalchemiops.types import get_wp_dtype, get_wp_mat_dtype, get_wp_vec_dtype

###########################################################################################
########################### Naive Neighbor List Kernels ################################
###########################################################################################


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


@wp.kernel(enable_backward=False)
def _fill_naive_neighbor_matrix_pbc(
    positions: wp.array(dtype=Any),
    cutoff_sq: Any,
    cell: wp.array(dtype=Any),
    shifts: wp.array(dtype=wp.vec3i),
    neighbor_matrix: wp.array2d(dtype=wp.int32),
    neighbor_matrix_shifts: wp.array2d(dtype=wp.vec3i),
    num_neighbors: wp.array(dtype=wp.int32),
    half_fill: wp.bool,
) -> None:
    """Calculate neighbor matrix with periodic boundary conditions using naive O(N^2) algorithm.

    Computes neighbor relationships between atoms across periodic boundaries by
    considering all periodic images within the cutoff distance. Uses a 2D launch
    pattern to parallelize over both atoms and periodic shifts.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Atomic coordinates in Cartesian space. Each row represents one atom's
        (x, y, z) position.
    cutoff_sq : float
        Squared cutoff distance for neighbor detection in Cartesian units.
        Must be positive. Atoms within this distance are considered neighbors.
    cell : wp.array, shape (num_systems, 3, 3), dtype=wp.mat33*
        Cell matrices defining lattice vectors in Cartesian coordinates.
        Each 3x3 matrix represents one system's periodic cell.
    shifts : wp.array, shape (total_shifts, 3), dtype=wp.vec3i
        Integer shift vectors for periodic images. Each row represents
        (nx, ny, nz) multiples of the cell vectors.
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
        If True, only store relationships where i < j to avoid double counting.
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
    _fill_naive_neighbor_matrix : Version without periodic boundary conditions
    _fill_batch_naive_neighbor_matrix_pbc : Batch version for multiple systems
    """
    ishift, iatom = wp.tid()

    jatom_start = 0
    jatom_end = positions.shape[0]

    maxnb = neighbor_matrix.shape[1]
    _positions = positions[iatom]
    _shift = shifts[ishift]
    _cell = cell[0]

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


## Generate overloads for all kernels
T = [wp.float32, wp.float64, wp.float16]
V = [wp.vec3f, wp.vec3d, wp.vec3h]
M = [wp.mat33f, wp.mat33d, wp.mat33h]
_fill_naive_neighbor_matrix_overload = {}
_fill_naive_neighbor_matrix_pbc_overload = {}
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
            t,
            wp.array(dtype=m),
            wp.array(dtype=wp.vec3i),
            wp.array2d(dtype=wp.int32),
            wp.array2d(dtype=wp.vec3i),
            wp.array(dtype=wp.int32),
            wp.bool,
        ],
    )

###########################################################################################
###################### Naive Neighbor List Python Wrapper ##############################
###########################################################################################


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
    max_neighbors : int
        Maximum number of neighbors per atom. Must be positive.
        If exceeded, excess neighbors are ignored.
    neighbor_matrix : torch.Tensor, shape (total_atoms, max_neighbors), dtype=torch.int32
        OUTPUT: Neighbor matrix to be filled with neighbor atom indices.
        Must be pre-allocated. Entries are filled with atom indices.
    num_neighbors : torch.Tensor, shape (total_atoms,), dtype=torch.int32
        OUTPUT: Number of neighbors found for each atom.
        Must be pre-allocated. Updated in-place with actual neighbor counts.
    half_fill : bool
        If True, only store relationships where i < j to avoid double counting.
        If False, store all neighbor relationships symmetrically.

    Returns
    -------
    None
        This function modifies the input tensors in-place:

        - neighbor_matrix : Filled with neighbor atom indices
        - num_neighbors : Updated with neighbor counts per atom

    See Also
    --------
    _naive_neighbor_matrix_no_pbc : Higher-level wrapper function
    _naive_neighbor_matrix_pbc : Version with periodic boundaries
    """
    device = positions.device
    wp_dtype = get_wp_dtype(positions.dtype)
    wp_vec_dtype = get_wp_vec_dtype(positions.dtype)
    wp_positions = wp.from_torch(positions, dtype=wp_vec_dtype, return_ctype=True)
    wp_neighbor_matrix = wp.from_torch(
        neighbor_matrix, dtype=wp.int32, return_ctype=True
    )
    wp_num_neighbors = wp.from_torch(num_neighbors, dtype=wp.int32, return_ctype=True)

    wp.launch(
        kernel=_fill_naive_neighbor_matrix_overload[wp_dtype],
        dim=positions.shape[0],
        inputs=[
            wp_positions,
            wp_dtype(cutoff * cutoff),
            wp_neighbor_matrix,
            wp_num_neighbors,
            half_fill,
        ],
        device=wp.device_from_torch(device),
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
    shift_offset: torch.Tensor,
    total_shifts: int,
    half_fill: bool = False,
) -> None:
    """
    Compute neighbor matrix with periodic boundary conditions using a naive O(N^2) algorithm.

    This function assumes that the number of shifts has been computed and the shifts have been
    expanded into a single array of shift vectors.

    This function is torch compilable.

    Parameters:
    ----------
    positions : torch.Tensor, shape (total_atoms, 3), dtype=torch.float32 or torch.float64
        Atomic coordinates in Cartesian space. Each row represents one atom's
        (x, y, z) position.
    cutoff : float
        Cutoff distance for neighbor detection in Cartesian units.
        Must be positive. Atoms within this distance are considered neighbors.
    cell : torch.Tensor, shape (1, 3, 3), dtype=torch.float32 or torch.float64
        Cell matrices defining lattice vectors in Cartesian coordinates.
    pbc : torch.Tensor, shape (1, 3), dtype=torch.bool
        Periodic boundary condition flags for each dimension.
        True enables periodicity in that direction.
    neighbor_matrix : torch.Tensor, shape (total_atoms, max_neighbors), dtype=torch.int32
        OUTPUT: Neighbor matrix to be filled.
    neighbor_matrix_shifts : torch.Tensor, shape (total_atoms, max_neighbors, 3), dtype=torch.int32, optional
        OUTPUT: Shift vectors for each neighbor relationship.
    num_neighbors : torch.Tensor, shape (total_atoms,), dtype=torch.int32, optional
        OUTPUT: Number of neighbors found for each atom.
    shift_range_per_dimension : torch.Tensor, shape (1, 3), dtype=torch.int32, optional
        Shift range in each dimension for each system.
    shift_offset : torch.Tensor, shape (2,), dtype=torch.int32, optional
        Cumulative sum of number of shifts for each system.
    total_shifts : int, optional
        Total number of shifts.
    half_fill : bool, optional
        If True, only store relationships where i < j to avoid double counting.
        If False, store all neighbor relationships symmetrically. Default is False.
    """
    total_atoms = positions.shape[0]
    device = positions.device
    wp_device = wp.device_from_torch(device)
    wp_dtype = get_wp_dtype(positions.dtype)
    wp_vec_dtype = get_wp_vec_dtype(positions.dtype)
    wp_mat_dtype = get_wp_mat_dtype(cell.dtype)

    shifts = torch.empty((total_shifts, 3), dtype=torch.int32, device=device)
    shift_system_idx = torch.empty((total_shifts,), dtype=torch.int32, device=device)
    wp_shifts = wp.from_torch(shifts, dtype=wp.vec3i, return_ctype=True)
    wp_shift_system_idx = wp.from_torch(
        shift_system_idx, dtype=wp.int32, return_ctype=True
    )
    wp_shift_range_per_dimension = wp.from_torch(
        shift_range_per_dimension, dtype=wp.vec3i, return_ctype=True
    )
    wp_shift_offset = wp.from_torch(shift_offset, dtype=wp.int32, return_ctype=True)
    wp.launch(
        kernel=_expand_naive_shifts,
        dim=1,
        inputs=[
            wp_shift_range_per_dimension,
            wp_shift_offset,
            wp_shifts,
            wp_shift_system_idx,
        ],
        device=wp_device,
    )

    # Launch neighbor computation kernel
    wp.launch(
        kernel=_fill_naive_neighbor_matrix_pbc_overload[wp_dtype],
        dim=(total_shifts, total_atoms),
        inputs=[
            wp.from_torch(positions, dtype=wp_vec_dtype, return_ctype=True),
            wp_dtype(cutoff * cutoff),
            wp.from_torch(cell, dtype=wp_mat_dtype, return_ctype=True),
            wp.from_torch(shifts, dtype=wp.vec3i, return_ctype=True),
            wp.from_torch(neighbor_matrix, dtype=wp.int32, return_ctype=True),
            wp.from_torch(neighbor_matrix_shifts, dtype=wp.vec3i, return_ctype=True),
            wp.from_torch(num_neighbors, dtype=wp.int32, return_ctype=True),
            half_fill,
        ],
        device=wp_device,
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
    shift_offset: torch.Tensor | None = None,
    total_shifts: int | None = None,
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
    precompute the shift vectors using compute_naive_num_shifts.

    .. code-block:: python

        >> from nvalchemiops.neighborlist import compute_naive_num_shifts
        >> shift_range_per_dimension, shift_offset, total_shifts = compute_naive_num_shifts(
        ...     cell, cutoff, pbc
        ... )

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
    shift_offset : torch.Tensor, shape (2,), dtype=torch.int32, optional
        Cumulative sum of number of shifts for each system.
        Pass in a pre-allocated tensor to avoid reallocation for pbc systems.
    total_shifts : int, optional
        Total number of shifts.
        Pass in a pre-allocated tensor to avoid reallocation for pbc systems.
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
    >>> neighbor_matrix, shifts, num_neighbors = naive_neighbor_list(
    ...     positions, cutoff, max_neighbors, pbc=pbc, cell=cell
    ... )

    Return as neighbor list instead of matrix:

    >>> neighbor_list, num_neighbors = naive_neighbor_list(
    ...     positions, cutoff, max_neighbors, return_neighbor_list=True
    ... )
    >>> source_atoms, target_atoms = neighbor_list[0], neighbor_list[1]

    Preallocate tensors for non-pbc systems:

    >>> max_neighbors = 100
    >>> neighbor_matrix = torch.zeros((positions.shape[0], max_neighbors), dtype=torch.int32, device=positions.device)
    >>> neighbor_matrix_shifts = torch.zeros((positions.shape[0], max_neighbors, 3), dtype=torch.int32, device=positions.device)
    >>> num_neighbors = torch.zeros(positions.shape[0], dtype=torch.int32, device=positions.device)
    >>> naive_neighbor_list(
    ...     positions, cutoff, max_neighbors, neighbor_matrix=neighbor_matrix, neighbor_matrix_shifts=neighbor_matrix_shifts, num_neighbors=num_neighbors
    ... )

    Preallocate tensors for pbc systems:

    >>> shift_range_per_dimension, shift_offset, total_shifts = _compute_total_shifts(
    ...     cell, cutoff, pbc
    ... )
    >>> naive_neighbor_list(
    ...     positions, cutoff, max_neighbors, shift_range_per_dimension=shift_range_per_dimension, shift_offset=shift_offset, total_shifts=total_shifts
    ... )

    See Also
    --------
    batch_neighbor_list : Batch version for multiple systems
    naive_neighbor_list_dual_cutoff : Version with two cutoff distances
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
    else:
        neighbor_matrix.fill_(fill_value)

    if num_neighbors is None:
        num_neighbors = torch.zeros(
            positions.shape[0], dtype=torch.int32, device=positions.device
        )
    else:
        num_neighbors.zero_()

    if pbc is not None:
        if neighbor_matrix_shifts is None:
            neighbor_matrix_shifts = torch.zeros(
                (positions.shape[0], max_neighbors, 3),
                dtype=torch.int32,
                device=positions.device,
            )
        else:
            neighbor_matrix_shifts.zero_()
        if (
            total_shifts is None
            or shift_offset is None
            or shift_range_per_dimension is None
        ):
            shift_range_per_dimension, shift_offset, total_shifts = (
                compute_naive_num_shifts(cell, cutoff, pbc)
            )

    if cutoff <= 0:
        if return_neighbor_list:
            if pbc is not None:
                return (
                    torch.zeros((2, 0), dtype=torch.int32, device=positions.device),
                    torch.zeros(
                        (positions.shape[0],),
                        dtype=torch.int32,
                        device=positions.device,
                    ),
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
                        (positions.shape[0],),
                        dtype=torch.int32,
                        device=positions.device,
                    ),
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
            shift_offset=shift_offset,
            total_shifts=total_shifts,
            half_fill=half_fill,
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
