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
# type: ignore

import math
from typing import Any

import torch
import warp as wp

from nvalchemiops.types import get_wp_dtype, get_wp_mat_dtype


@wp.kernel(enable_backward=False)
def _expand_naive_shifts(
    shift_range: wp.array(dtype=wp.vec3i),
    shift_offset: wp.array(dtype=int),
    shifts: wp.array(dtype=wp.vec3i),
    shift_system_idx: wp.array(dtype=int),
) -> None:
    """Expand shift ranges into actual shift vectors for all systems in the batch.

    Converts the compact shift range representation into a flattened array
    of explicit shift vectors, maintaining proper indexing to avoid double
    counting of periodic images.

    Parameters
    ----------
    shift_range : wp.array, shape (num_systems, 3), dtype=wp.vec3i
        Array of shift ranges in each dimension for each system.
    shift_offset : wp.array, shape (num_systems+1,), dtype=wp.int32
        Cumulative sum of number of shifts for each system.
    shifts : wp.array, shape (total_shifts, 3), dtype=wp.vec3i
        OUTPUT: Flattened array to store the shift vectors.
    shift_system_idx : wp.array, shape (total_shifts,), dtype=wp.int32
        OUTPUT: System index mapping for each shift vector.

    Notes
    -----
    - Thread launch: One thread per system in the batch (dim=num_systems)
    - Modifies: shifts, shift_system_idx
    - total_shifts = shift_offset[-1]
    - Shift vectors generated in order k0, k1, k2 (increasing)
    - All shift vectors are integer lattice coordinates
    """
    tid = wp.tid()
    pos = shift_offset[tid]
    _shift_range = shift_range[tid]
    for k0 in range(0, _shift_range[0] + 1):
        for k1 in range(-_shift_range[1], _shift_range[1] + 1):
            for k2 in range(-_shift_range[2], _shift_range[2] + 1):
                if k0 > 0 or (k0 == 0 and k1 > 0) or (k0 == 0 and k1 == 0 and k2 >= 0):
                    shifts[pos] = wp.vec3i(k0, k1, k2)
                    shift_system_idx[pos] = tid
                    pos += 1


@wp.func
def _update_neighbor_matrix(
    i: int,
    j: int,
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    max_neighbors: int,
    half_fill: bool,
):
    """
    Update the neighbor matrix with the given atom indices.

    Parameters
    ----------
    i: int
        The index of the source atom.
    j: int
        The index of the target atom.
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2)
        OUTPUT: The neighbor matrix to be updated.
    num_neighbors: wp.array(dtype=wp.int32)
        OUTPUT: The number of neighbors for each atom.
    max_neighbors: int
        The maximum number of neighbors for each atom.
    half_fill: bool
        If True, only fill half of the neighbor matrix.
    """
    pos = wp.atomic_add(num_neighbors, i, 1)
    if pos < max_neighbors:
        neighbor_matrix[i, pos] = j
    if not half_fill and i < j:
        pos = wp.atomic_add(num_neighbors, j, 1)
        if pos < max_neighbors:
            neighbor_matrix[j, pos] = i


@wp.func
def _update_neighbor_matrix_pbc(
    i: int,
    j: int,
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    unit_shift: wp.vec3i,
    max_neighbors: int,
    half_fill: bool,
):
    """
    Update the neighbor matrix with the given atom indices and periodic shift.

    Parameters
    ----------
    i: int
        The index of the source atom.
    j: int
        The index of the target atom.
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2)
        OUTPUT: The neighbor matrix to be updated.
    neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2)
        OUTPUT: The neighbor matrix shifts to be updated.
    num_neighbors: wp.array(dtype=wp.int32)
        OUTPUT: The number of neighbors for each atom.
    unit_shift: wp.vec3i
        The unit shift vector for the periodic boundary.
    max_neighbors: int
        The maximum number of neighbors for each atom.
    half_fill: bool
        If True, only fill half of the neighbor matrix.
    """
    pos = wp.atomic_add(num_neighbors, i, 1)
    if pos < max_neighbors:
        neighbor_matrix[i, pos] = j
        neighbor_matrix_shifts[i, pos] = unit_shift
    if not half_fill:
        pos = wp.atomic_add(num_neighbors, j, 1)
        if pos < max_neighbors:
            neighbor_matrix[j, pos] = i
            neighbor_matrix_shifts[j, pos] = -unit_shift


@wp.kernel(enable_backward=False)
def _compute_naive_num_shifts(
    cell: wp.array(dtype=Any),
    cutoff: Any,
    pbc: wp.array2d(dtype=wp.bool),
    num_shifts: wp.array(dtype=int),
    shift_range: wp.array(dtype=wp.vec3i),
) -> None:
    """Compute periodic image shifts needed for neighbor searching.

    Calculates the number and range of periodic boundary shifts required
    to ensure all atoms within the cutoff distance are found, taking into
    account the geometry of the simulation cell and minimum image convention.

    Parameters
    ----------
    cell : wp.array, shape (num_systems, 3, 3), dtype=wp.mat33*
        Cell matrices defining lattice vectors in Cartesian coordinates.
        Each 3x3 matrix represents one system's periodic cell.
    cutoff : float
        Cutoff distance for neighbor searching in Cartesian units.
        Must be positive and typically less than half the minimum cell dimension.
    pbc : wp.array, shape (num_systems, 3), dtype=wp.bool
        Periodic boundary condition flags for each dimension.
        True enables periodicity in that direction.
    num_shifts : wp.array, shape (num_systems,), dtype=int
        OUTPUT: Total number of periodic shifts needed for each system.
        Updated with calculated shift counts.
    shift_range : wp.array, shape (num_systems, 3), dtype=wp.vec3i
        OUTPUT: Maximum shift indices in each dimension for each system.
        Updated with calculated shift ranges.

    Returns
    -------
    None
        This function modifies the input arrays in-place:

        - num_shifts : Updated with total shift counts per system
        - shift_range : Updated with shift ranges per dimension

    See Also
    --------
    _expand_naive_shifts : Expands shift ranges into explicit shift vectors
    """
    tid = wp.tid()

    _cell = cell[tid]
    _pbc = pbc[tid]

    _cell_inv = wp.transpose(wp.inverse(_cell))
    _d_inv_0 = wp.length(_cell_inv[0]) if _pbc[0] else type(_cell_inv[0, 0])(0.0)
    _d_inv_1 = wp.length(_cell_inv[1]) if _pbc[1] else type(_cell_inv[1, 0])(0.0)
    _d_inv_2 = wp.length(_cell_inv[2]) if _pbc[2] else type(_cell_inv[2, 0])(0.0)
    _s = wp.vec3i(
        wp.int32(wp.ceil(_d_inv_0 * type(_d_inv_0)(cutoff))),
        wp.int32(wp.ceil(_d_inv_1 * type(_d_inv_1)(cutoff))),
        wp.int32(wp.ceil(_d_inv_2 * type(_d_inv_2)(cutoff))),
    )
    k1 = 2 * _s[1] + 1
    k2 = 2 * _s[2] + 1
    shift_range[tid] = _s
    num_shifts[tid] = _s[0] * k1 * k2 + _s[1] * k2 + _s[2] + 1


## Generate overloads
T = [wp.float32, wp.float64, wp.float16]
V = [wp.vec3f, wp.vec3d, wp.vec3h]
M = [wp.mat33f, wp.mat33d, wp.mat33h]
_compute_naive_num_shifts_overload = {}
for t, v, m in zip(T, V, M):
    _compute_naive_num_shifts_overload[t] = wp.overload(
        _compute_naive_num_shifts,
        [
            wp.array(dtype=m),
            t,
            wp.array2d(dtype=wp.bool),
            wp.array(dtype=int),
            wp.array(dtype=wp.vec3i),
        ],
    )


# interface
def compute_naive_num_shifts(
    cell: torch.Tensor,
    cutoff: float,
    pbc: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Compute periodic image shifts needed for neighbor searching.

    Parameters
    ----------
    cell: torch.Tensor
        Cell matrices defining lattice vectors in Cartesian coordinates.
        Each 3x3 matrix represents one system's periodic cell.
    cutoff: float
        Cutoff distance for neighbor searching in Cartesian units.
        Must be positive and typically less than half the minimum cell dimension.
    pbc: torch.Tensor
        Periodic boundary condition flags for each dimension.
        True enables periodicity in that direction.

    Returns
    -------
    shift_range: torch.Tensor
        Maximum shift indices in each dimension for each system.
    shift_offset: torch.Tensor
        Cumulative sum of number of shifts for each system.
    total_shifts: int
        Total number of periodic shifts needed for each system.
    """
    num_systems = cell.shape[0]
    device = cell.device

    num_shifts = torch.empty(num_systems, dtype=torch.int32, device=device)
    shift_range = torch.empty((num_systems, 3), dtype=torch.int32, device=device)

    wp_dtype = get_wp_dtype(cell.dtype)
    wp_mat_dtype = get_wp_mat_dtype(cell.dtype)
    wp_device = wp.device_from_torch(device)

    wp_cell = wp.from_torch(cell, dtype=wp_mat_dtype)
    wp_pbc = wp.from_torch(pbc, dtype=wp.bool)
    wp_num_shifts = wp.from_torch(num_shifts)
    wp_shift_range = wp.from_torch(shift_range, dtype=wp.vec3i)

    wp.launch(
        kernel=_compute_naive_num_shifts,
        dim=num_systems,
        inputs=[
            wp_cell,
            wp_dtype(cutoff),
            wp_pbc,
            wp_num_shifts,
            wp_shift_range,
        ],
        device=wp_device,
    )

    shift_offset = torch.empty((num_systems + 1,), dtype=torch.int32, device=device)
    shift_offset[0] = 0
    torch.cumsum(num_shifts, dim=0, out=shift_offset[1:])
    return shift_range, shift_offset, shift_offset[-1].item()


def estimate_max_neighbors(
    cutoff: float,
    atomic_density: float = 0.35,
    safety_factor: float = 5.0,
) -> int:
    r"""Estimate maximum neighbors per atom based on volume calculations.

    Uses atomic density and cutoff volume to estimate a conservative upper bound
    on the number of neighbors any atom could have. This maintains torch.compile
    compatibility by using only tensor operations without dynamic control flow.

    Parameters
    ----------
    cutoff : float
        Maximum distance for considering atoms as neighbors.
    atomic_density : float, optional
        Atomic density in atoms per unit volume. Default is 1.0.
    safety_factor : float
        Safety factor to multiply the estimated number of neighbors.

    Returns
    -------
    max_neighbors_estimate : torch.Tensor
        Conservative estimate of maximum neighbors per atom. Returns 0 for
        empty systems, total atom count for degenerate cells.

    Notes
    -----
    The estimation uses the formula:
    neighbors = safety_factor * density × cutoff_sphere_volume
    where density = N_atoms / cell_volume and cutoff_sphere_volume = (4/3)\pi r³

    The result is rounded up to the multiple of 16 for memory alignment.
    """
    if cutoff <= 0:
        return 0
    # Calculate volume of cutoff sphere: V_sphere = (4/3) * \pi * r³
    cutoff_sphere_volume = atomic_density * (4.0 / 3.0) * math.pi * (cutoff**3)

    # Estimate neighbors based on density and cutoff volume
    expected_neighbors = max(1, safety_factor * cutoff_sphere_volume)

    # Round up to next power of 2 for memory alignment and safety
    max_neighbors_estimate = int(math.ceil(expected_neighbors / 16)) * 16
    return max_neighbors_estimate


class NeighborOverflowError(Exception):
    """Exception raised when the number of neighbors larger than the maximum allowed."""

    def __init__(self, max_neighbors: int, num_neighbors: int):
        super().__init__(
            f"The number of neighbors is larger than the maximum allowed: {num_neighbors} > {max_neighbors}."
        )


def assert_max_neighbors(neighbor_matrix: torch.Tensor, num_neighbors: torch.Tensor):
    """Assert that the number of neighbors is not larger than size of the neighbor matrix."""
    max_neighbors = 0 if num_neighbors.numel() == 0 else num_neighbors.max()
    if max_neighbors > neighbor_matrix.shape[1]:
        raise NeighborOverflowError(
            neighbor_matrix.shape[1],
            max_neighbors if isinstance(max_neighbors, int) else max_neighbors.item(),
        )


def get_neighbor_list_from_neighbor_matrix(
    neighbor_matrix: torch.Tensor,
    num_neighbors: torch.Tensor,
    neighbor_shift_matrix: torch.Tensor | None = None,
    fill_value: int = -1,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Get the neighbor list from the neighbor matrix.

    Parameters
    ----------
    neighbor_matrix: torch.Tensor
        The neighbor matrix with shape (total_atoms, max_neighbors), dtype int32.
    num_neighbors: torch.Tensor
        The number of neighbors for each atom with shape (total_atoms,), dtype int32.
    neighbor_shift_matrix: torch.Tensor | None
        Optional neighbor shift matrix with shape (total_atoms, max_neighbors, 3), dtype int32.
    fill_value: int
        The fill value for the neighbor matrix.
        This is used to create a mask from the neighbor matrix.

    Returns
    -------
    neighbor_list: torch.Tensor
        The neighbor list with shape (2, num_pairs), dtype int32, in COO format [source_atoms, target_atoms].
    neighbor_ptr: torch.Tensor
        The neighbor pointer with shape (total_atoms + 1,), dtype int32.
        CSR-style pointer arrays where ``neighbor_ptr_data[i]`` to ``neighbor_ptr_data[i+1]`` gives the range of
        neighbors for atom i in the flattened neighbor list.
    neighbor_shift_matrix: torch.Tensor | None
        The neighbor shift matrix with shape (total_atoms, max_neighbors, 3), dtype int32.
        If input neighbor_shift_matrix is None, returns None.

    Raises
    ------
    ValueError
        If the max number of neighbors is larger than the neighbor matrix.
    """

    # Raise ValueError if the max number of neighbors is larger than the neighbor matrix
    if num_neighbors.shape[0] == 0:
        neighbor_list = torch.zeros(
            2, 0, dtype=neighbor_matrix.dtype, device=neighbor_matrix.device
        )
        neighbor_ptr = torch.zeros(1, dtype=torch.int32, device=neighbor_matrix.device)
        neighbor_shift_matrix = (
            None
            if neighbor_shift_matrix is None
            else torch.empty(
                0,
                2,
                3,
                dtype=neighbor_shift_matrix.dtype,
                device=neighbor_shift_matrix.device,
            )
        )
        returns = (
            (neighbor_list, neighbor_ptr, neighbor_shift_matrix)
            if neighbor_shift_matrix is not None
            else (neighbor_list, neighbor_ptr)
        )
        return returns

    # Raise NeighborOverflowError if the number of neighbors is larger than the neighbor matrix
    assert_max_neighbors(neighbor_matrix, num_neighbors)

    mask = neighbor_matrix != fill_value
    dtype = neighbor_matrix.dtype
    i_idx = torch.where(mask)[0].to(dtype)
    j_idx = neighbor_matrix[mask].to(dtype)
    neighbor_list = torch.stack([i_idx, j_idx], dim=0)
    neighbor_ptr = torch.zeros(
        num_neighbors.shape[0] + 1, dtype=torch.int32, device=neighbor_matrix.device
    )
    torch.cumsum(num_neighbors, dim=0, out=neighbor_ptr[1:])
    if neighbor_shift_matrix is not None:
        neighbor_list_shifts = neighbor_shift_matrix[mask]
        return neighbor_list, neighbor_ptr, neighbor_list_shifts
    else:
        return neighbor_list, neighbor_ptr


@torch.compile
def _prepare_batch_idx_ptr(
    batch_idx: torch.Tensor | None,
    batch_ptr: torch.Tensor | None,
    num_atoms: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Utility function to prepare batch index and pointer tensors.

    Parameters
    ----------
    batch_idx: torch.Tensor | None
        Tensor indicating the batch index for each atom.
    batch_ptr: torch.Tensor | None
        Tensor indicating the start index of each batch in the atom list.
    num_atoms: int
        Total number of atoms.
    num_systems: int | None
        Total number of systems.
    device: torch.device
        Device on which to create tensors if needed.

    Returns
    -------
    batch_idx: torch.Tensor
        Prepared batch index tensor.
    batch_ptr: torch.Tensor
        Prepared batch pointer tensor.
    """
    if batch_idx is None and batch_ptr is None:
        raise ValueError("Either batch_idx or batch_ptr must be provided.")

    if batch_idx is None:
        num_systems = batch_ptr.shape[0] - 1
        num_atoms_per_system = batch_ptr[1:] - batch_ptr[:-1]
        batch_idx = torch.repeat_interleave(
            torch.arange(num_systems, dtype=torch.int32, device=device),
            num_atoms_per_system,
        )

    elif batch_ptr is None:
        num_systems = batch_idx.max() + 1
        num_atoms_per_system = torch.bincount(batch_idx, minlength=num_systems)
        batch_ptr = torch.zeros(num_systems + 1, dtype=torch.int32, device=device)
        torch.cumsum(num_atoms_per_system, dim=0, out=batch_ptr[1:])

    return batch_idx, batch_ptr


def allocate_cell_list(
    total_atoms: int,
    max_total_cells: int,
    neighbor_search_radius: torch.Tensor,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Allocate memory for the cell list."""
    # detect number of systems from neighbor_search_radius
    cells_per_dimension = torch.zeros(
        (3,)
        if neighbor_search_radius.ndim == 1
        else (neighbor_search_radius.shape[0], 3),
        dtype=torch.int32,
        device=device,
    )

    atom_periodic_shifts = torch.zeros(
        (total_atoms, 3), dtype=torch.int32, device=device
    )
    atom_to_cell_mapping = torch.zeros(
        (total_atoms, 3), dtype=torch.int32, device=device
    )
    atoms_per_cell_count = torch.zeros(
        (max_total_cells,), dtype=torch.int32, device=device
    )
    cell_atom_start_indices = torch.zeros(
        (max_total_cells,), dtype=torch.int32, device=device
    )
    cell_atom_list = torch.zeros((total_atoms,), dtype=torch.int32, device=device)
    return (
        cells_per_dimension,
        neighbor_search_radius,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
    )
