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

"""Core warp utilities for neighbor list construction.

This module contains warp kernels and launchers for neighbor list operations.
See `nvalchemiops.torch.neighbors` for PyTorch bindings.
"""

import math
from typing import Any

import warp as wp


class NeighborOverflowError(Exception):
    """Exception raised when the number of neighbors exceeds the maximum allowed.

    This error indicates that the pre-allocated neighbor matrix is too small
    to hold all discovered neighbors. Users should increase `max_neighbors`
    parameter or use a larger pre-allocated tensor.

    Parameters
    ----------
    max_neighbors : int
        The maximum number of neighbors the matrix can hold.
    num_neighbors : int
        The actual number of neighbors found.
    """

    def __init__(self, max_neighbors: int, num_neighbors: int):
        super().__init__(
            f"The number of neighbors is larger than the maximum allowed: "
            f"{num_neighbors} > {max_neighbors}."
        )
        self.max_neighbors = max_neighbors
        self.num_neighbors = num_neighbors


__all__ = [
    "NeighborOverflowError",
    "compute_naive_num_shifts",
    "zero_array",
    "estimate_max_neighbors",
]


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


@wp.kernel(enable_backward=False)
def _zero_int32_array_kernel(
    array: wp.array(dtype=wp.int32),
) -> None:
    """Zero an int32 array in parallel.

    Parameters
    ----------
    array : wp.array, dtype=wp.int32
        OUTPUT: Array to be zeroed.

    Notes
    -----
    - Thread launch: One thread per element (dim=array.shape[0])
    - Modifies: array (sets all elements to 0)
    """
    tid = wp.tid()
    array[tid] = 0


def zero_array(
    array: wp.array,
    device: str,
) -> None:
    """Core warp launcher for zeroing an int32 array.

    Zeros all elements of an int32 array in parallel using pure warp operations.

    Parameters
    ----------
    array : wp.array, dtype=wp.int32
        OUTPUT: Array to be zeroed.
    device : str
        Warp device string (e.g., 'cuda:0', 'cpu').

    Notes
    -----
    - This is a low-level warp interface.
    - Operates on int32 arrays only.

    See Also
    --------
    _zero_int32_array_kernel : Kernel that performs the zeroing
    """
    n = array.shape[0]

    wp.launch(
        kernel=_zero_int32_array_kernel,
        dim=n,
        inputs=[array],
        device=device,
    )


def compute_naive_num_shifts(
    cell: wp.array,
    cutoff: float,
    pbc: wp.array,
    num_shifts: wp.array,
    shift_range: wp.array,
    wp_dtype: type,
    device: str,
) -> None:
    """Core warp launcher for computing periodic image shifts.

    Calculates the number and range of periodic boundary shifts required
    to ensure all atoms within the cutoff distance are found, using pure
    warp operations.

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
    num_shifts : wp.array, shape (num_systems,), dtype=wp.int32
        OUTPUT: Total number of periodic shifts needed for each system.
        Updated with calculated shift counts.
    shift_range : wp.array, shape (num_systems, 3), dtype=wp.vec3i
        OUTPUT: Maximum shift indices in each dimension for each system.
        Updated with calculated shift ranges.
    wp_dtype : type
        Warp dtype (wp.float32, wp.float64, or wp.float16).
    device : str
        Warp device string (e.g., 'cuda:0', 'cpu').

    Notes
    -----
    - This is a low-level warp interface. For framework bindings, use torch/jax wrappers.
    - Output arrays (num_shifts, shift_range) must be pre-allocated by caller.

    See Also
    --------
    _compute_naive_num_shifts : Kernel that performs the computation
    _expand_naive_shifts : Expands shift ranges into explicit shift vectors
    """
    num_systems = cell.shape[0]

    wp.launch(
        kernel=_compute_naive_num_shifts,
        dim=num_systems,
        inputs=[
            cell,
            wp_dtype(cutoff),
            pbc,
            num_shifts,
            shift_range,
        ],
        device=device,
    )


def estimate_max_neighbors(
    cutoff: float,
    atomic_density: float = 0.35,
    safety_factor: float = 5.0,
) -> int:
    r"""Estimate maximum neighbors per atom based on volume calculations.

    Uses atomic density and cutoff volume to estimate a conservative upper bound
    on the number of neighbors any atom could have. This is a pure Python function
    with no framework dependencies.

    Parameters
    ----------
    cutoff : float
        Maximum distance for considering atoms as neighbors.
    atomic_density : float, optional
        Atomic density in atoms per unit volume. Default is 0.35.
    safety_factor : float
        Safety factor to multiply the estimated number of neighbors. Default is 5.0.

    Returns
    -------
    max_neighbors_estimate : int
        Conservative estimate of maximum neighbors per atom. Returns 0 for
        empty systems.

    Notes
    -----
    The estimation uses the formula:

    .. math::

        \text{neighbors} = \text{safety\_factor} \times \text{density} \times V_{\text{sphere}}

    where the cutoff sphere volume is:

    .. math::

        V_{\text{sphere}} = \frac{4}{3}\pi r^3

    The result is rounded up to the multiple of 16 for memory alignment.
    """
    if cutoff <= 0:
        return 0
    cutoff_sphere_volume = atomic_density * (4.0 / 3.0) * math.pi * (cutoff**3)

    # Estimate neighbors based on density and cutoff volume
    expected_neighbors = max(1, safety_factor * cutoff_sphere_volume)

    # Round up to multiple of 16 for memory alignment and safety
    max_neighbors_estimate = int(math.ceil(expected_neighbors / 16)) * 16
    return max_neighbors_estimate
