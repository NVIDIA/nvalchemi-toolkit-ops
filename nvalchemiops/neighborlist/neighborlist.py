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

import torch

from nvalchemiops.neighborlist.batch_cell_list import (
    batch_cell_list,
)
from nvalchemiops.neighborlist.batch_naive import (
    batch_naive_neighbor_list,
)
from nvalchemiops.neighborlist.batch_naive_dual_cutoff import (
    batch_naive_neighbor_list_dual_cutoff,
)
from nvalchemiops.neighborlist.cell_list import (
    cell_list,
)
from nvalchemiops.neighborlist.naive import (
    naive_neighbor_list,
)
from nvalchemiops.neighborlist.naive_dual_cutoff import (
    naive_neighbor_list_dual_cutoff,
)
from nvalchemiops.neighborlist.neighbor_utils import (
    _prepare_batch_idx_ptr,
)


def neighbor_list(
    positions: torch.Tensor,
    cutoff: float,
    cell: torch.Tensor | None = None,
    pbc: torch.Tensor | None = None,
    batch_idx: torch.Tensor | None = None,
    batch_ptr: torch.Tensor | None = None,
    cutoff2: float | None = None,
    half_fill: bool = False,
    fill_value: int | None = None,
    return_neighbor_list: bool = False,
    method: str | None = None,
    **kwargs: dict,
):
    """Compute neighbor list using the appropriate method based on the provided parameters.

    Parameters
    ----------
    positions : torch.Tensor, shape (total_atoms, 3)
        Concatenated atomic coordinates for all systems in Cartesian space.
        Each row represents one atom's (x, y, z) position.
        Must be wrapped into the unit cell if PBC is used.
    cutoff : float
        Cutoff distance for neighbor detection in Cartesian units.
        Must be positive. Atoms within this distance are considered neighbors.
    cell : torch.Tensor, shape (3, 3), dtype=torch.float32 or torch.float64, optional
        Cell matrix defining the simulation box.
    pbc : torch.Tensor, shape (3,), dtype=torch.bool, optional
        Periodic boundary condition flags for each dimension.
    batch_idx : torch.Tensor, shape (total_atoms,), dtype=torch.int32, optional
        System index for each atom.
    batch_ptr : torch.Tensor, shape (num_systems + 1,), dtype=torch.int32, optional
        Cumulative atom counts defining system boundaries.
    cutoff2 : float, optional
        Second cutoff distance for neighbor detection in Cartesian units.
        Must be positive. Atoms within this distance are considered neighbors.
    half_fill : bool, optional
        If True, only store half of the neighbor relationships to avoid double counting.
        Another half could be reconstructed by swapping source and target indices and inverting unit shifts.
    fill_value : int | None, optional
        Value to fill the neighbor matrix with. Default is total_atoms.
    return_neighbor_list : bool, optional - default = False
        If True, convert the neighbor matrix to a neighbor list (idx_i, idx_j) format by
        creating a mask over the fill_value, which can incur a performance penalty.
        We recommend using the neighbor matrix format,
        and only convert to a neighbor list format if absolutely necessary.
    method : str | None, optional
        Method to use for neighbor list computation.
        Choices: "naive", "cell_list", "batch_naive", "batch_cell_list", "naive_dual_cutoff", "batch_naive_dual_cutoff".
        If None, a default method will be chosen based on the number of atoms.
    **kwargs : dict, optional
        Additional keyword arguments to pass to the method.

        - max_neighbors: int, optional
            Maximum number of neighbors per atom.
            Can be provided to aid in allocation for both naive and cell list methods.
        - max_neighbors2: int, optional
            Maximum number of neighbors per atom within cutoff2.
            Can be provided to aid in allocation for naive dual cutoff method.
        - neighbor_matrix: torch.Tensor, optional
            Pre-allocated tensor of shape (total_atoms, max_neighbors) for neighbor indices.
            Can be provided to avoid reallocation for both naive and cell list methods.
        - neighbor_matrix_shifts: torch.Tensor, optional
            Pre-allocated tensor of shape (total_atoms, max_neighbors, 3) for shift vectors.
            Can be provided to avoid reallocation for both naive and cell list methods.
        - num_neighbors: torch.Tensor, optional
            Pre-allocated tensor of shape (total_atoms,) for neighbor counts.
            Can be provided to avoid reallocation for both naive and cell list methods.
        - shift_range_per_dimension: torch.Tensor, optional
            Pre-allocated tensor of shape (1, 3) for shift range in each dimension.
            Can be provided to avoid reallocation for naive methods.
        - shift_offset: torch.Tensor, optional
            Pre-allocated tensor of shape (2,) for cumulative sum of number of shifts for each system.
            Can be provided to avoid reallocation for naive methods.
        - total_shifts: int, optional
            Total number of shifts.
            Can be provided to avoid reallocation for naive methods.
        - cells_per_dimension: torch.Tensor, optional
            Pre-allocated tensor of shape (3,) for number of cells in x, y, z directions.
            Can be provided to avoid reallocation for cell list construction.
        - neighbor_search_radius: torch.Tensor, optional
            Pre-allocated tensor of shape (3,) for radius of neighboring cells to search in each dimension.
            Can be provided to avoid reallocation for cell list construction.
        - atom_periodic_shifts: torch.Tensor, optional
            Pre-allocated tensor of shape (total_atoms, 3) for periodic boundary crossings for each atom.
            Can be provided to avoid reallocation for cell list construction.
        - atom_to_cell_mapping: torch.Tensor, optional
            Pre-allocated tensor of shape (total_atoms, 3) for cell coordinates for each atom.
            Can be provided to avoid reallocation for cell list construction.
        - atoms_per_cell_count: torch.Tensor, optional
            Pre-allocated tensor of shape (max_total_cells,) for number of atoms in each cell.
            Can be provided to avoid reallocation for cell list construction.
        - cell_atom_start_indices: torch.Tensor, optional
            Pre-allocated tensor of shape (max_total_cells,) for starting index in cell_atom_list for each cell.
            Can be provided to avoid reallocation for cell list construction.
        - cell_atom_list: torch.Tensor, optional
            Pre-allocated tensor of shape (total_atoms,) for flattened list of atom indices organized by cell.
            Can be provided to avoid reallocation for cell list construction.
        - max_atoms_per_system: int, optional
            Maximum number of atoms per system.
            Used in batch naive implementation with PBC. If not provided, it will be computed automaticaly.
            Can be provided to avoid CUDA synchronization.

    Returns
    -------
    results : tuple of torch.Tensor
        Variable-length tuple depending on input parameters. The return pattern follows:

        **Single cutoff:**
          - No PBC, matrix format: ``(neighbor_matrix, num_neighbors)``
          - No PBC, list format: ``(neighbor_list, neighbor_ptr)``
          - With PBC, matrix format: ``(neighbor_matrix, num_neighbors, neighbor_matrix_shifts)``
          - With PBC, list format: ``(neighbor_list, neighbor_ptr, neighbor_list_shifts)``

        **Dual cutoff:**
          - No PBC, matrix format: ``(neighbor_matrix1, num_neighbors1, neighbor_matrix2, num_neighbors2)``
          - No PBC, list format: ``(neighbor_list1, neighbor_ptr1, neighbor_list2, neighbor_ptr2)``
          - With PBC, matrix format: ``(neighbor_matrix1, num_neighbors1, neighbor_matrix_shifts1, neighbor_matrix2, num_neighbors2, neighbor_matrix_shifts2)``
          - With PBC, list format: ``(neighbor_list1, neighbor_ptr1, neighbor_list_shifts1, neighbor_list2, neighbor_ptr2, neighbor_list_shifts2)``

        **Components returned:**

        - **neighbor_data** (tensor): Neighbor indices, format depends on ``return_neighbor_list``:

            - If ``return_neighbor_list=False`` (default): Returns ``neighbor_matrix``
              with shape (total_atoms, max_neighbors), dtype int32. Each row i contains
              indices of atom i's neighbors.
            - If ``return_neighbor_list=True``: Returns ``neighbor_list`` with shape
              (2, num_pairs), dtype int32, in COO format [source_atoms, target_atoms].

        - **num_neighbor_data** (tensor): Information about the number of neighbors for each atom,
          format depends on ``return_neighbor_list``:

            - If ``return_neighbor_list=False`` (default): Returns ``num_neighbors`` with shape (total_atoms,), dtype int32.
              Count of neighbors found for each atom.
            - If ``return_neighbor_list=True``: Returns ``neighbor_ptr`` with shape (total_atoms + 1,), dtype int32.
            CSR-style pointer arrays where ``neighbor_ptr_data[i]`` to ``neighbor_ptr_data[i+1]`` gives the range of
            neighbors for atom i in the flattened neighbor list.

        - **neighbor_shift_data** (tensor, optional): Periodic shift vectors, only when ``pbc`` is provided:
          format depends on ``return_neighbor_list``:

            - If ``return_neighbor_list=False`` (default): Returns ``neighbor_matrix_shifts`` with
              shape (total_atoms, max_neighbors, 3), dtype int32.
            - If ``return_neighbor_list=True``: Returns ``unit_shifts`` with shape
            (num_pairs, 3), dtype int32.

        When ``cutoff2`` is provided, the pattern repeats for the second cutoff with interleaved
        components (neighbor_data2, num_neighbor_data2, neighbor_shift_data2) appended to the tuple.


    Examples
    --------
    Single cutoff, matrix format, with PBC::

        >>> nm, num, shifts = neighbor_list(pos, 5.0, cell=cell, pbc=pbc)

    Single cutoff, list format, no PBC::

        >>> nlist, ptr = neighbor_list(pos, 5.0, return_neighbor_list=True)

    Dual cutoff, matrix format, with PBC::

        >>> nm1, num1, sh1, nm2, num2, sh2 = neighbor_list(
        ...     pos, 2.5, cutoff2=5.0, cell=cell, pbc=pbc
        ... )

    See Also
    --------
    naive_neighbor_list : Direct access to naive O(N²) algorithm
    cell_list : Direct access to cell list O(N) algorithm
    """
    if method is None:
        total_atoms = positions.shape[0]
        if cutoff2 is not None:
            method = "naive_dual_cutoff"

        elif total_atoms >= 5000:
            method = "cell_list"
            if cell is None or pbc is None:
                cell = torch.eye(
                    3, dtype=positions.dtype, device=positions.device
                ).reshape(1, 3, 3)
                pbc = torch.tensor(
                    [False, False, False], dtype=torch.bool, device=positions.device
                )
        else:
            method = "naive"

        if batch_idx is not None or batch_ptr is not None:
            method = "batch_" + method
            batch_idx, batch_ptr = _prepare_batch_idx_ptr(
                batch_idx, batch_ptr, total_atoms, positions.device
            )
    match method:
        case "naive":
            return naive_neighbor_list(
                positions,
                cutoff,
                pbc=pbc,
                cell=cell,
                half_fill=half_fill,
                fill_value=fill_value,
                return_neighbor_list=return_neighbor_list,
                **kwargs,
            )
        case "cell_list":
            return cell_list(
                positions,
                cutoff,
                cell,
                pbc,
                half_fill=half_fill,
                fill_value=fill_value,
                return_neighbor_list=return_neighbor_list,
                **kwargs,
            )
        case "batch_naive":
            return batch_naive_neighbor_list(
                positions,
                cutoff,
                pbc=pbc,
                cell=cell,
                batch_idx=batch_idx,
                batch_ptr=batch_ptr,
                half_fill=half_fill,
                fill_value=fill_value,
                return_neighbor_list=return_neighbor_list,
                **kwargs,
            )
        case "batch_cell_list":
            return batch_cell_list(
                positions,
                cutoff,
                cell,
                pbc,
                batch_idx,
                half_fill=half_fill,
                fill_value=fill_value,
                return_neighbor_list=return_neighbor_list,
                **kwargs,
            )
        case "naive_dual_cutoff":
            return naive_neighbor_list_dual_cutoff(
                positions,
                cutoff,
                cutoff2,
                pbc=pbc,
                cell=cell,
                half_fill=half_fill,
                fill_value=fill_value,
                return_neighbor_list=return_neighbor_list,
                **kwargs,
            )
        case "batch_naive_dual_cutoff":
            return batch_naive_neighbor_list_dual_cutoff(
                positions,
                cutoff,
                cutoff2,
                pbc=pbc,
                cell=cell,
                batch_idx=batch_idx,
                batch_ptr=batch_ptr,
                half_fill=half_fill,
                fill_value=fill_value,
                return_neighbor_list=return_neighbor_list,
                **kwargs,
            )
        case _:
            raise ValueError(f"Invalid method: {method}")
