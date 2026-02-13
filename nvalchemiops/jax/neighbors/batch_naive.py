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

"""JAX bindings for batched naive O(N^2) neighbor list construction."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import warp as wp
from warp.jax_experimental import jax_kernel

from nvalchemiops.jax.neighbors.neighbor_utils import (
    compute_naive_num_shifts,
    get_neighbor_list_from_neighbor_matrix,
    prepare_batch_idx_ptr,
)
from nvalchemiops.neighbors.batch_naive import (
    _fill_batch_naive_neighbor_matrix_overload,
    _fill_batch_naive_neighbor_matrix_pbc_overload,
)
from nvalchemiops.neighbors.neighbor_utils import (
    _expand_naive_shifts,
    estimate_max_neighbors,
)

__all__ = ["batch_naive_neighbor_list"]

# ==============================================================================
# JAX Kernel Wrappers
# ==============================================================================

# No-PBC batch naive neighbor matrix kernel wrappers
_jax_fill_batch_naive_f32 = jax_kernel(
    _fill_batch_naive_neighbor_matrix_overload[wp.float32],
    num_outputs=2,
    in_out_argnames=["neighbor_matrix", "num_neighbors"],
    enable_backward=False,
)
_jax_fill_batch_naive_f64 = jax_kernel(
    _fill_batch_naive_neighbor_matrix_overload[wp.float64],
    num_outputs=2,
    in_out_argnames=["neighbor_matrix", "num_neighbors"],
    enable_backward=False,
)

# PBC batch naive neighbor matrix kernel wrappers
_jax_fill_batch_naive_pbc_f32 = jax_kernel(
    _fill_batch_naive_neighbor_matrix_pbc_overload[wp.float32],
    num_outputs=3,
    in_out_argnames=["neighbor_matrix", "neighbor_matrix_shifts", "num_neighbors"],
    enable_backward=False,
)
_jax_fill_batch_naive_pbc_f64 = jax_kernel(
    _fill_batch_naive_neighbor_matrix_pbc_overload[wp.float64],
    num_outputs=3,
    in_out_argnames=["neighbor_matrix", "neighbor_matrix_shifts", "num_neighbors"],
    enable_backward=False,
)

# Expand shifts kernel wrapper (no overloads — single kernel)
_jax_expand_naive_shifts = jax_kernel(
    _expand_naive_shifts,
    num_outputs=2,
    in_out_argnames=["shifts", "shift_system_idx"],
    enable_backward=False,
)


def batch_naive_neighbor_list(
    positions: jax.Array,
    cutoff: float,
    batch_idx: jax.Array | None = None,
    batch_ptr: jax.Array | None = None,
    pbc: jax.Array | None = None,
    cell: jax.Array | None = None,
    max_neighbors: int | None = None,
    half_fill: bool = False,
    fill_value: int | None = None,
    return_neighbor_list: bool = False,
    neighbor_matrix: jax.Array | None = None,
    neighbor_matrix_shifts: jax.Array | None = None,
    num_neighbors: jax.Array | None = None,
    shift_range_per_dimension: jax.Array | None = None,
    shift_offset: jax.Array | None = None,
    total_shifts: int | None = None,
    max_atoms_per_system: int | None = None,
) -> (
    tuple[jax.Array, jax.Array, jax.Array, jax.Array]
    | tuple[jax.Array, jax.Array, jax.Array]
    | tuple[jax.Array, jax.Array]
):
    """Compute neighbor list for batch of systems using naive O(N^2) algorithm.

    Identifies all atom pairs within a specified cutoff distance for each system
    independently using a brute-force pairwise distance calculation. Supports both
    non-periodic and periodic boundary conditions.

    Parameters
    ----------
    positions : jax.Array, shape (total_atoms, 3), dtype=float32 or float64
        Concatenated atomic coordinates for all systems in Cartesian space.
    cutoff : float
        Cutoff distance for neighbor detection in Cartesian units.
        Must be positive. Atoms within this distance are considered neighbors.
    batch_idx : jax.Array, shape (total_atoms,), dtype=int32, optional
        System index for each atom. If None, batch_ptr must be provided.
    batch_ptr : jax.Array, shape (num_systems + 1,), dtype=int32, optional
        Cumulative atom counts defining system boundaries. If None, batch_idx must be provided.
    pbc : jax.Array, shape (num_systems, 3), dtype=bool, optional
        Periodic boundary condition flags for each system and dimension.
        True enables periodicity in that direction. Default is None (no PBC).
    cell : jax.Array, shape (num_systems, 3, 3), dtype=float32 or float64, optional
        Cell matrices defining lattice vectors. Required if pbc is provided.
    max_neighbors : int, optional
        Maximum number of neighbors per atom.
    half_fill : bool, optional
        If True, only store relationships where i < j. Default is False.
    fill_value : int, optional
        Value to fill the neighbor matrix with. Default is total_atoms.
    neighbor_matrix : jax.Array, optional
        Pre-allocated neighbor matrix.
    neighbor_matrix_shifts : jax.Array, optional
        Pre-allocated shift matrix for PBC.
    num_neighbors : jax.Array, optional
        Pre-allocated neighbors count array.
    shift_range_per_dimension : jax.Array, optional
        Pre-computed shift range for PBC systems.
    shift_offset : jax.Array, optional
        Pre-computed shift offsets for PBC systems.
    total_shifts : int, optional
        Total number of shifts for PBC.
    max_atoms_per_system : int, optional
        Maximum atoms in any system.

    Returns
    -------
    results : tuple of jax.Array
        Variable-length tuple depending on input parameters.

    Examples
    --------
    Basic usage with batch_ptr:

    >>> import jax.numpy as jnp
    >>> from nvalchemiops.jax.neighbors import batch_naive_neighbor_list
    >>> positions = jnp.zeros((200, 3), dtype=jnp.float32)
    >>> batch_ptr = jnp.array([0, 100, 200], dtype=jnp.int32)  # 2 systems
    >>> cutoff = 2.5
    >>> max_neighbors = 50
    >>> neighbor_matrix, num_neighbors = batch_naive_neighbor_list(
    ...     positions, cutoff, batch_ptr=batch_ptr, max_neighbors=max_neighbors
    ... )

    With PBC:

    >>> cell = jnp.eye(3, dtype=jnp.float32)[jnp.newaxis, :, :] * 10.0
    >>> cell = jnp.repeat(cell, 2, axis=0)
    >>> pbc = jnp.ones((2, 3), dtype=jnp.bool_)
    >>> neighbor_matrix, num_neighbors, shifts = batch_naive_neighbor_list(
    ...     positions, cutoff, batch_ptr=batch_ptr, max_neighbors=max_neighbors,
    ...     pbc=pbc, cell=cell
    ... )

    See Also
    --------
    nvalchemiops.neighbors.batch_naive.batch_naive_neighbor_matrix : Core warp launcher
    nvalchemiops.jax.neighbors.naive.naive_neighbor_list : Non-batched version
    batch_cell_list : Cell list method for large systems
    """
    if pbc is None and cell is not None:
        raise ValueError("If cell is provided, pbc must also be provided")
    if pbc is not None and cell is None:
        raise ValueError("If pbc is provided, cell must also be provided")

    # Prepare batch indices and pointers
    batch_idx, batch_ptr = prepare_batch_idx_ptr(
        batch_idx, batch_ptr, positions.shape[0]
    )
    num_systems = batch_ptr.shape[0] - 1

    if cell is not None:
        cell = cell if cell.ndim == 3 else cell[jnp.newaxis, :, :]
        # Ensure cell dtype matches positions dtype so warp overload dispatch is consistent
        if cell.dtype != positions.dtype:
            cell = cell.astype(positions.dtype)
    if pbc is not None:
        pbc = pbc if pbc.ndim == 2 else pbc[jnp.newaxis, :]

    if max_neighbors is None:
        max_neighbors = estimate_max_neighbors(cutoff)

    if fill_value is None:
        fill_value = positions.shape[0]

    if neighbor_matrix is None:
        neighbor_matrix = jnp.full(
            (positions.shape[0], max_neighbors),
            fill_value,
            dtype=jnp.int32,
        )
    else:
        neighbor_matrix = neighbor_matrix.at[:].set(fill_value)

    if num_neighbors is None:
        num_neighbors = jnp.zeros(positions.shape[0], dtype=jnp.int32)
    else:
        num_neighbors = num_neighbors.at[:].set(0)

    if pbc is not None:
        if neighbor_matrix_shifts is None:
            neighbor_matrix_shifts = jnp.zeros(
                (positions.shape[0], max_neighbors, 3),
                dtype=jnp.int32,
            )
        else:
            neighbor_matrix_shifts = neighbor_matrix_shifts.at[:].set(0)
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
                    jnp.zeros((2, 0), dtype=jnp.int32),
                    jnp.zeros((positions.shape[0] + 1,), dtype=jnp.int32),
                    jnp.zeros((0, 3), dtype=jnp.int32),
                )
            else:
                return (
                    jnp.zeros((2, 0), dtype=jnp.int32),
                    jnp.zeros((positions.shape[0] + 1,), dtype=jnp.int32),
                )
        else:
            if pbc is not None:
                return neighbor_matrix, num_neighbors, neighbor_matrix_shifts
            else:
                return neighbor_matrix, num_neighbors

    # Select kernel based on dtype
    if positions.dtype == jnp.float64:
        _jax_fill = _jax_fill_batch_naive_f64
        _jax_fill_pbc = _jax_fill_batch_naive_pbc_f64
    else:
        _jax_fill = _jax_fill_batch_naive_f32
        _jax_fill_pbc = _jax_fill_batch_naive_pbc_f32
        positions = positions.astype(jnp.float32)

    total_atoms = positions.shape[0]

    if pbc is None:
        # No PBC case
        neighbor_matrix, num_neighbors = _jax_fill(
            positions,
            float(cutoff * cutoff),
            batch_idx.astype(jnp.int32),
            batch_ptr.astype(jnp.int32),
            neighbor_matrix,
            num_neighbors,
            half_fill,
            launch_dims=(total_atoms,),
        )
    else:
        # PBC case - expand shifts first
        shifts = jnp.zeros((total_shifts, 3), dtype=jnp.int32)
        shift_system_idx = jnp.zeros(total_shifts, dtype=jnp.int32)

        shifts, shift_system_idx = _jax_expand_naive_shifts(
            shift_range_per_dimension,
            shift_offset,
            shifts,
            shift_system_idx,
            launch_dims=(num_systems,),  # batched — one thread per system
        )

        if cell.dtype != positions.dtype:
            cell = cell.astype(positions.dtype)

        if max_atoms_per_system is None:
            try:
                max_atoms_per_system = int(jnp.max(batch_ptr[1:] - batch_ptr[:-1]))
            except (
                jax.errors.ConcretizationTypeError,
                jax.errors.TracerIntegerConversionError,
            ):
                raise ValueError(
                    "Cannot infer max_atoms_per_system inside jax.jit. "
                    "Please provide max_atoms_per_system explicitly when using jax.jit."
                ) from None

        neighbor_matrix, neighbor_matrix_shifts, num_neighbors = _jax_fill_pbc(
            positions,
            cell,
            float(cutoff * cutoff),
            batch_ptr.astype(jnp.int32),
            shifts,
            shift_system_idx,
            neighbor_matrix,
            neighbor_matrix_shifts,
            num_neighbors,
            half_fill,
            launch_dims=(total_shifts, max_atoms_per_system),
        )

    if return_neighbor_list:
        if pbc is not None:
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
            neighbor_list, neighbor_ptr = get_neighbor_list_from_neighbor_matrix(
                neighbor_matrix,
                num_neighbors=num_neighbors,
                fill_value=fill_value,
            )
            return neighbor_list, neighbor_ptr
    else:
        if pbc is not None:
            return neighbor_matrix, num_neighbors, neighbor_matrix_shifts
        else:
            return neighbor_matrix, num_neighbors
