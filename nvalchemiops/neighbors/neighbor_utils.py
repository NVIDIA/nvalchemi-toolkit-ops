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
    "compute_inv_cells",
    "zero_array",
    "selective_zero_num_neighbors",
    "selective_zero_num_neighbors_single",
    "estimate_max_neighbors",
    "wrap_positions_single",
    "wrap_positions_batch",
    "_expand_naive_shifts_selective",
    "update_ref_positions",
    "update_ref_positions_batch",
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


@wp.kernel(enable_backward=False)
def _expand_naive_shifts_selective(
    shift_range: wp.array(dtype=wp.vec3i),
    shift_offset: wp.array(dtype=int),
    shifts: wp.array(dtype=wp.vec3i),
    shift_system_idx: wp.array(dtype=int),
    rebuild_flags: wp.array(dtype=wp.bool),
) -> None:
    """Expand shift ranges into actual shift vectors, skipping non-rebuilt systems.

    Identical to ``_expand_naive_shifts`` but checks ``rebuild_flags[tid]``
    on the GPU and exits immediately for systems that do not need rebuilding.
    No CPU-GPU synchronisation occurs.

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
    rebuild_flags : wp.array, shape (num_systems,), dtype=wp.bool
        Per-system rebuild flags. False → kernel returns immediately for that system.

    Notes
    -----
    - Thread launch: One thread per system in the batch (dim=num_systems)
    - Modifies: shifts, shift_system_idx (only for rebuilt systems)
    - total_shifts = shift_offset[-1]
    """
    tid = wp.tid()
    if not rebuild_flags[tid]:
        return
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
def _decode_shift_index(local_idx: int, shift_range: wp.vec3i) -> wp.vec3i:
    """Decode a flat shift index into (kx, ky, kz) lattice shift vector.

    Reverses the enumeration order used by ``_expand_naive_shifts`` so that
    shift vectors can be computed on-the-fly from a thread index without
    materialising the full shifts array.

    Parameters
    ----------
    local_idx : int
        Zero-based index into the per-system shift enumeration.
    shift_range : wp.vec3i
        Shift range in each dimension (from ``_compute_naive_num_shifts``).

    Returns
    -------
    wp.vec3i
        The integer lattice shift vector ``(kx, ky, kz)``.
    """
    k2_size = 2 * shift_range[2] + 1
    k1_size = 2 * shift_range[1] + 1
    group0_size = shift_range[1] * k2_size + shift_range[2] + 1

    k0 = wp.int32(0)
    k1 = wp.int32(0)
    k2 = wp.int32(0)

    if local_idx < group0_size:
        if local_idx <= shift_range[2]:
            k2 = local_idx
        else:
            rem = local_idx - (shift_range[2] + 1)
            k1 = rem / k2_size + 1
            k2 = rem % k2_size - shift_range[2]
    else:
        rem = local_idx - group0_size
        k0 = rem / (k1_size * k2_size) + 1
        rem2 = rem % (k1_size * k2_size)
        k1 = rem2 / k2_size - shift_range[1]
        k2 = rem2 % k2_size - shift_range[2]

    return wp.vec3i(k0, k1, k2)


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
def _zero_array_kernel(
    array: wp.array(dtype=Any),
) -> None:
    """Zero an array in parallel.

    Parameters
    ----------
    array : wp.array, dtype=Any
        OUTPUT: Array to be zeroed.

    Notes
    -----
    - Thread launch: One thread per element (dim=array.shape[0])
    - Modifies: array (sets all elements to 0)
    """
    tid = wp.tid()
    array[tid] = type(array[tid])(0)


def zero_array(
    array: wp.array,
    device: str,
) -> None:
    """Core warp launcher for zeroing an array.

    Zeros all elements of an array in parallel using pure warp operations.

    Parameters
    ----------
    array : wp.array, dtype=Any
        OUTPUT: Array to be zeroed.
    device : str
        Warp device string (e.g., 'cuda:0', 'cpu').

    Notes
    -----
    - This is a low-level warp interface.
    - Operates on arrays of any dtype.

    See Also
    --------
    _zero_array_kernel : Kernel that performs the zeroing
    """
    n = array.shape[0]

    wp.launch(
        kernel=_zero_array_kernel,
        dim=n,
        inputs=[array],
        device=device,
    )


@wp.kernel(enable_backward=False)
def _selective_zero_num_neighbors(
    num_neighbors: wp.array(dtype=wp.int32),
    batch_idx: wp.array(dtype=wp.int32),
    rebuild_flags: wp.array(dtype=wp.bool),
) -> None:
    """Zero num_neighbors entries for atoms in systems that need rebuilding.

    Parameters
    ----------
    num_neighbors : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Number of neighbors; zeroed for atoms in rebuilt systems.
    batch_idx : wp.array, shape (total_atoms,), dtype=wp.int32
        System index for each atom.
    rebuild_flags : wp.array, shape (num_systems,), dtype=wp.bool
        Per-system rebuild flags. True means this system needs rebuilding.

    Notes
    -----
    - Thread launch: One thread per atom (dim=total_atoms)
    - Modifies: num_neighbors (selective zero for rebuilt systems)
    """
    tid = wp.tid()
    isys = batch_idx[tid]
    if rebuild_flags[isys]:
        num_neighbors[tid] = 0


def selective_zero_num_neighbors(
    num_neighbors: wp.array,
    batch_idx: wp.array,
    rebuild_flags: wp.array,
    device: str,
) -> None:
    """Core warp launcher for selectively zeroing num_neighbors.

    Zeros the num_neighbors count for atoms belonging to systems where
    rebuild_flags is True, preserving counts for non-rebuilt systems.

    Parameters
    ----------
    num_neighbors : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Per-atom neighbor counts; selectively zeroed.
    batch_idx : wp.array, shape (total_atoms,), dtype=wp.int32
        System index for each atom.
    rebuild_flags : wp.array, shape (num_systems,), dtype=wp.bool
        Per-system flags indicating which systems need rebuilding.
    device : str
        Warp device string (e.g., 'cuda:0', 'cpu').

    See Also
    --------
    _selective_zero_num_neighbors : Kernel that performs the selective zeroing
    """
    total_atoms = num_neighbors.shape[0]
    wp.launch(
        kernel=_selective_zero_num_neighbors,
        dim=total_atoms,
        inputs=[num_neighbors, batch_idx, rebuild_flags],
        device=device,
    )


@wp.kernel(enable_backward=False)
def _selective_zero_num_neighbors_single(
    num_neighbors: wp.array(dtype=wp.int32),
    rebuild_flags: wp.array(dtype=wp.bool),
) -> None:
    """Zero num_neighbors entries when the single-system rebuild flag is set.

    Parameters
    ----------
    num_neighbors : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Number of neighbors; zeroed for all atoms when rebuild_flags[0] is True.
    rebuild_flags : wp.array, shape (1,) or shape (), dtype=wp.bool
        Single-system flag. When True, all entries of num_neighbors are zeroed.

    Notes
    -----
    - Thread launch: One thread per atom (dim=total_atoms)
    - Modifies: num_neighbors (only when rebuild_flags[0] is True)
    """
    tid = wp.tid()
    if rebuild_flags[0]:
        num_neighbors[tid] = 0


def selective_zero_num_neighbors_single(
    num_neighbors: wp.array,
    rebuild_flags: wp.array,
    device: str,
) -> None:
    """Core warp launcher for selectively zeroing num_neighbors for a single system.

    Zeros all num_neighbors entries when rebuild_flags[0] is True.  When False
    the kernel returns immediately — no CPU-GPU synchronization occurs.

    Parameters
    ----------
    num_neighbors : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Per-atom neighbor counts; zeroed when rebuild is needed.
    rebuild_flags : wp.array, shape (1,) or shape (), dtype=wp.bool
        Single-system rebuild flag.
    device : str
        Warp device string (e.g., 'cuda:0', 'cpu').

    See Also
    --------
    _selective_zero_num_neighbors_single : Kernel that performs the selective zeroing
    selective_zero_num_neighbors : Batch variant using per-atom batch_idx
    """
    total_atoms = num_neighbors.shape[0]
    wp.launch(
        kernel=_selective_zero_num_neighbors_single,
        dim=total_atoms,
        inputs=[num_neighbors, rebuild_flags],
        device=device,
    )


@wp.kernel(enable_backward=False)
def _compute_inv_cells_kernel(
    cell: wp.array(dtype=Any),
    inv_cell: wp.array(dtype=Any),
) -> None:
    """Compute the inverse of each cell matrix.

    Parameters
    ----------
    cell : wp.array, shape (num_systems,), dtype=wp.mat33*
        Input cell matrices.
    inv_cell : wp.array, shape (num_systems,), dtype=wp.mat33*
        OUTPUT: Inverse of each cell matrix.

    Notes
    -----
    - Thread launch: One thread per system (dim=num_systems)
    """
    tid = wp.tid()
    inv_cell[tid] = wp.inverse(cell[tid])


_compute_inv_cells_overload = {}
for _t, _m in zip(
    [wp.float32, wp.float64, wp.float16],
    [wp.mat33f, wp.mat33d, wp.mat33h],
):
    _compute_inv_cells_overload[_t] = wp.overload(
        _compute_inv_cells_kernel,
        [wp.array(dtype=_m), wp.array(dtype=_m)],
    )


def compute_inv_cells(
    cell: wp.array,
    inv_cell: wp.array,
    wp_dtype: type,
    device: str,
) -> None:
    """Core warp launcher for computing inverse cell matrices.

    Inverts each cell matrix in the batch using pure warp operations.
    Call this once before launching naive PBC neighbor-list kernels to
    avoid redundant per-thread inversions inside those kernels.

    Parameters
    ----------
    cell : wp.array, shape (num_systems,), dtype=wp.mat33*
        Input cell matrices.
    inv_cell : wp.array, shape (num_systems,), dtype=wp.mat33*
        OUTPUT: Inverse of each cell matrix. Must be pre-allocated
        with the same shape and dtype as *cell*.
    wp_dtype : type
        Warp scalar dtype (wp.float32, wp.float64, or wp.float16).
    device : str
        Warp device string (e.g., ``'cuda:0'``, ``'cpu'``).

    See Also
    --------
    _compute_inv_cells_kernel : Underlying warp kernel
    """
    num_systems = cell.shape[0]
    wp.launch(
        kernel=_compute_inv_cells_overload[wp_dtype],
        dim=num_systems,
        inputs=[cell, inv_cell],
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
    atomic_density: float = 0.2,
    safety_factor: float = 1.0,
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
        Atomic density in atoms per unit volume. Default is 0.2.
    safety_factor : float
        Safety factor to multiply the estimated number of neighbors. Default is 1.0.

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


###########################################################################################
########################### Position Wrapping Kernels ####################################
###########################################################################################


@wp.kernel(enable_backward=False)
def _wrap_positions_single_kernel(
    positions: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    inv_cell: wp.array(dtype=Any),
    positions_wrapped: wp.array(dtype=Any),
    per_atom_cell_offsets: wp.array(dtype=wp.vec3i),
) -> None:
    """Wrap positions into the primary cell for a single system.

    Computes fractional coordinates to determine integer cell offsets, then
    shifts each atom back into the primary cell. The integer offsets are stored
    so that corrected shift vectors can be recovered for the original (unwrapped)
    positions.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms,), dtype=wp.vec3*
        Atomic coordinates in Cartesian space. May be unwrapped.
    cell : wp.array, shape (1,), dtype=wp.mat33*
        Cell matrix defining lattice vectors in Cartesian coordinates.
    inv_cell : wp.array, shape (1,), dtype=wp.mat33*
        Pre-computed inverse of the cell matrix.
    positions_wrapped : wp.array, shape (total_atoms,), dtype=wp.vec3*
        OUTPUT: Wrapped positions in Cartesian space.
    per_atom_cell_offsets : wp.array, shape (total_atoms,), dtype=wp.vec3i
        OUTPUT: Integer cell offsets for each atom (floor of fractional coordinates).

    Notes
    -----
    - Thread launch: One thread per atom (dim=total_atoms)
    - Modifies: positions_wrapped, per_atom_cell_offsets
    """
    i = wp.tid()
    _cell = cell[0]
    _inv_cell = inv_cell[0]
    _pos = positions[i]
    _frac = _pos * _inv_cell
    _int = wp.vec3i(
        wp.int32(wp.floor(_frac[0])),
        wp.int32(wp.floor(_frac[1])),
        wp.int32(wp.floor(_frac[2])),
    )
    positions_wrapped[i] = _pos - type(_pos)(_int) * _cell
    per_atom_cell_offsets[i] = _int


_wrap_positions_single_overload = {}
for _t, _v, _m in zip(
    [wp.float32, wp.float64, wp.float16],
    [wp.vec3f, wp.vec3d, wp.vec3h],
    [wp.mat33f, wp.mat33d, wp.mat33h],
):
    _wrap_positions_single_overload[_t] = wp.overload(
        _wrap_positions_single_kernel,
        [
            wp.array(dtype=_v),
            wp.array(dtype=_m),
            wp.array(dtype=_m),
            wp.array(dtype=_v),
            wp.array(dtype=wp.vec3i),
        ],
    )


def wrap_positions_single(
    positions: wp.array,
    cell: wp.array,
    inv_cell: wp.array,
    positions_wrapped: wp.array,
    per_atom_cell_offsets: wp.array,
    wp_dtype: type,
    device: str,
) -> None:
    """Core warp launcher for wrapping positions into the primary cell (single system).

    Computes per-atom integer cell offsets and wrapped positions in a single
    GPU pass. Call this before naive PBC neighbor-list kernels to move the
    wrapping out of the hot ishift × iatom loop.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms,), dtype=wp.vec3*
        Atomic coordinates in Cartesian space. May be unwrapped.
    cell : wp.array, shape (1,), dtype=wp.mat33*
        Cell matrix defining lattice vectors.
    inv_cell : wp.array, shape (1,), dtype=wp.mat33*
        Pre-computed inverse cell matrix. Must be pre-allocated with the
        same shape and dtype as *cell*.
    positions_wrapped : wp.array, shape (total_atoms,), dtype=wp.vec3*
        OUTPUT: Wrapped positions. Must be pre-allocated with the same shape
        and dtype as *positions*.
    per_atom_cell_offsets : wp.array, shape (total_atoms,), dtype=wp.vec3i
        OUTPUT: Integer cell offsets per atom. Must be pre-allocated.
    wp_dtype : type
        Warp scalar dtype (wp.float32, wp.float64, or wp.float16).
    device : str
        Warp device string (e.g., ``'cuda:0'``, ``'cpu'``).

    See Also
    --------
    _wrap_positions_single_kernel : Underlying warp kernel
    wrap_positions_batch : Batch variant for multiple systems
    """
    total_atoms = positions.shape[0]
    wp.launch(
        kernel=_wrap_positions_single_overload[wp_dtype],
        dim=total_atoms,
        inputs=[positions, cell, inv_cell, positions_wrapped, per_atom_cell_offsets],
        device=device,
    )


@wp.kernel(enable_backward=False)
def _wrap_positions_batch_kernel(
    positions: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    inv_cell: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    positions_wrapped: wp.array(dtype=Any),
    per_atom_cell_offsets: wp.array(dtype=wp.vec3i),
) -> None:
    """Wrap positions into the primary cell for a batch of systems.

    Each atom uses the cell matrix of its system (indexed via batch_idx).
    Computes fractional coordinates to determine integer cell offsets, then
    shifts each atom back into the primary cell.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms,), dtype=wp.vec3*
        Concatenated atomic coordinates for all systems. May be unwrapped.
    cell : wp.array, shape (num_systems,), dtype=wp.mat33*
        Cell matrices for each system.
    inv_cell : wp.array, shape (num_systems,), dtype=wp.mat33*
        Pre-computed inverse cell matrices.
    batch_idx : wp.array, shape (total_atoms,), dtype=wp.int32
        System index for each atom.
    positions_wrapped : wp.array, shape (total_atoms,), dtype=wp.vec3*
        OUTPUT: Wrapped positions in Cartesian space.
    per_atom_cell_offsets : wp.array, shape (total_atoms,), dtype=wp.vec3i
        OUTPUT: Integer cell offsets for each atom (floor of fractional coordinates).

    Notes
    -----
    - Thread launch: One thread per atom (dim=total_atoms)
    - Modifies: positions_wrapped, per_atom_cell_offsets
    """
    i = wp.tid()
    isys = batch_idx[i]
    _cell = cell[isys]
    _inv_cell = inv_cell[isys]
    _pos = positions[i]
    _frac = _pos * _inv_cell
    _int = wp.vec3i(
        wp.int32(wp.floor(_frac[0])),
        wp.int32(wp.floor(_frac[1])),
        wp.int32(wp.floor(_frac[2])),
    )
    positions_wrapped[i] = _pos - type(_pos)(_int) * _cell
    per_atom_cell_offsets[i] = _int


_wrap_positions_batch_overload = {}
for _t, _v, _m in zip(
    [wp.float32, wp.float64, wp.float16],
    [wp.vec3f, wp.vec3d, wp.vec3h],
    [wp.mat33f, wp.mat33d, wp.mat33h],
):
    _wrap_positions_batch_overload[_t] = wp.overload(
        _wrap_positions_batch_kernel,
        [
            wp.array(dtype=_v),
            wp.array(dtype=_m),
            wp.array(dtype=_m),
            wp.array(dtype=wp.int32),
            wp.array(dtype=_v),
            wp.array(dtype=wp.vec3i),
        ],
    )


def wrap_positions_batch(
    positions: wp.array,
    cell: wp.array,
    inv_cell: wp.array,
    batch_idx: wp.array,
    positions_wrapped: wp.array,
    per_atom_cell_offsets: wp.array,
    wp_dtype: type,
    device: str,
) -> None:
    """Core warp launcher for wrapping positions into the primary cell (batch of systems).

    Each atom uses the cell matrix of its system (indexed via batch_idx).
    Computes per-atom integer cell offsets and wrapped positions in a single
    GPU pass. Call this before batch naive PBC neighbor-list kernels to move
    the wrapping out of the hot ishift × iatom loop.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms,), dtype=wp.vec3*
        Concatenated atomic coordinates for all systems. May be unwrapped.
    cell : wp.array, shape (num_systems,), dtype=wp.mat33*
        Cell matrices for each system.
    inv_cell : wp.array, shape (num_systems,), dtype=wp.mat33*
        Pre-computed inverse cell matrices. Must be pre-allocated with the
        same shape and dtype as *cell*.
    batch_idx : wp.array, shape (total_atoms,), dtype=wp.int32
        System index for each atom.
    positions_wrapped : wp.array, shape (total_atoms,), dtype=wp.vec3*
        OUTPUT: Wrapped positions. Must be pre-allocated with the same shape
        and dtype as *positions*.
    per_atom_cell_offsets : wp.array, shape (total_atoms,), dtype=wp.vec3i
        OUTPUT: Integer cell offsets per atom. Must be pre-allocated.
    wp_dtype : type
        Warp scalar dtype (wp.float32, wp.float64, or wp.float16).
    device : str
        Warp device string (e.g., ``'cuda:0'``, ``'cpu'``).

    See Also
    --------
    _wrap_positions_batch_kernel : Underlying warp kernel
    wrap_positions_single : Single-system variant
    """
    total_atoms = positions.shape[0]
    wp.launch(
        kernel=_wrap_positions_batch_overload[wp_dtype],
        dim=total_atoms,
        inputs=[
            positions,
            cell,
            inv_cell,
            batch_idx,
            positions_wrapped,
            per_atom_cell_offsets,
        ],
        device=device,
    )


###########################################################################################
########################### Reference Position Update Kernels ############################
###########################################################################################


@wp.kernel(enable_backward=False)
def _update_ref_positions_kernel(
    positions: wp.array(dtype=Any),
    rebuild_flag: wp.array(dtype=wp.bool),
    ref_positions: wp.array(dtype=Any),
) -> None:
    """Conditionally copy positions to ref_positions when rebuild_flag[0] is True.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms,), dtype=wp.vec3*
        Current atomic coordinates.
    rebuild_flag : wp.array, shape (1,), dtype=wp.bool
        Single-system rebuild flag. When True, ref_positions is updated.
    ref_positions : wp.array, shape (total_atoms,), dtype=wp.vec3*
        OUTPUT: Reference positions updated when rebuild_flag[0] is True.

    Notes
    -----
    - Thread launch: One thread per atom (dim=total_atoms)
    - Modifies: ref_positions (only when rebuild_flag[0] is True)
    """
    i = wp.tid()
    if rebuild_flag[0]:
        ref_positions[i] = positions[i]


_update_ref_positions_overload = {}
for _t, _v in zip([wp.float32, wp.float64], [wp.vec3f, wp.vec3d]):
    _update_ref_positions_overload[_t] = wp.overload(
        _update_ref_positions_kernel,
        [wp.array(dtype=_v), wp.array(dtype=wp.bool), wp.array(dtype=_v)],
    )


def update_ref_positions(
    positions: wp.array,
    rebuild_flag: wp.array,
    ref_positions: wp.array,
    wp_dtype: type,
    device: str,
) -> None:
    """Core warp launcher for conditionally updating reference positions (single system).

    Copies current positions into reference positions only when rebuild_flag[0] is True.
    No CPU-GPU synchronization required.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms,), dtype=wp.vec3*
        Current atomic coordinates.
    rebuild_flag : wp.array, shape (1,), dtype=wp.bool
        Single-system rebuild flag.
    ref_positions : wp.array, shape (total_atoms,), dtype=wp.vec3*
        OUTPUT: Reference positions to update selectively.
    wp_dtype : type
        Warp scalar dtype (wp.float32 or wp.float64).
    device : str
        Warp device string (e.g., 'cuda:0', 'cpu').

    See Also
    --------
    _update_ref_positions_kernel : Underlying warp kernel
    update_ref_positions_batch : Batch variant
    """
    total_atoms = positions.shape[0]
    wp.launch(
        kernel=_update_ref_positions_overload[wp_dtype],
        dim=total_atoms,
        inputs=[positions, rebuild_flag, ref_positions],
        device=device,
    )


@wp.kernel(enable_backward=False)
def _update_ref_positions_batch_kernel(
    positions: wp.array(dtype=Any),
    rebuild_flags: wp.array(dtype=wp.bool),
    batch_idx: wp.array(dtype=wp.int32),
    ref_positions: wp.array(dtype=Any),
) -> None:
    """Conditionally copy positions to ref_positions per-system (batch, no CPU sync).

    Parameters
    ----------
    positions : wp.array, shape (total_atoms,), dtype=wp.vec3*
        Current atomic coordinates for all systems.
    rebuild_flags : wp.array, shape (num_systems,), dtype=wp.bool
        Per-system rebuild flags.
    batch_idx : wp.array, shape (total_atoms,), dtype=wp.int32
        System index for each atom.
    ref_positions : wp.array, shape (total_atoms,), dtype=wp.vec3*
        OUTPUT: Reference positions; updated for atoms in rebuilt systems.

    Notes
    -----
    - Thread launch: One thread per atom (dim=total_atoms)
    - Modifies: ref_positions (only for atoms in rebuilt systems)
    """
    i = wp.tid()
    if rebuild_flags[batch_idx[i]]:
        ref_positions[i] = positions[i]


_update_ref_positions_batch_overload = {}
for _t, _v in zip([wp.float32, wp.float64], [wp.vec3f, wp.vec3d]):
    _update_ref_positions_batch_overload[_t] = wp.overload(
        _update_ref_positions_batch_kernel,
        [
            wp.array(dtype=_v),
            wp.array(dtype=wp.bool),
            wp.array(dtype=wp.int32),
            wp.array(dtype=_v),
        ],
    )


def update_ref_positions_batch(
    positions: wp.array,
    rebuild_flags: wp.array,
    batch_idx: wp.array,
    ref_positions: wp.array,
    wp_dtype: type,
    device: str,
) -> None:
    """Core warp launcher for conditionally updating reference positions (batch).

    Updates reference positions only for atoms in systems where rebuild_flags is True.
    No CPU-GPU synchronization required.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms,), dtype=wp.vec3*
        Current atomic coordinates for all systems.
    rebuild_flags : wp.array, shape (num_systems,), dtype=wp.bool
        Per-system rebuild flags.
    batch_idx : wp.array, shape (total_atoms,), dtype=wp.int32
        System index for each atom.
    ref_positions : wp.array, shape (total_atoms,), dtype=wp.vec3*
        OUTPUT: Reference positions to update selectively.
    wp_dtype : type
        Warp scalar dtype (wp.float32 or wp.float64).
    device : str
        Warp device string (e.g., 'cuda:0', 'cpu').

    See Also
    --------
    _update_ref_positions_batch_kernel : Underlying warp kernel
    update_ref_positions : Single-system variant
    """
    total_atoms = positions.shape[0]
    wp.launch(
        kernel=_update_ref_positions_batch_overload[wp_dtype],
        dim=total_atoms,
        inputs=[positions, rebuild_flags, batch_idx, ref_positions],
        device=device,
    )
