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

"""JAX bindings for unbatched naive O(N^2) neighbor list construction."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import warp as wp
from warp.jax_experimental import jax_kernel

from nvalchemiops.jax.neighbors.neighbor_utils import (
    compute_naive_num_shifts,
    get_neighbor_list_from_neighbor_matrix,
)
from nvalchemiops.neighbors.naive import (
    _fill_naive_neighbor_matrix_overload,
    _fill_naive_neighbor_matrix_pbc_overload,
    _fill_naive_neighbor_matrix_pbc_prewrapped_overload,
    _fill_naive_neighbor_matrix_pbc_prewrapped_selective_overload,
    _fill_naive_neighbor_matrix_pbc_selective_overload,
    _fill_naive_neighbor_matrix_selective_overload,
)
from nvalchemiops.neighbors.neighbor_utils import (
    _wrap_positions_single_overload,
    estimate_max_neighbors,
)

__all__ = ["naive_neighbor_list"]

# ==============================================================================
# JAX Kernel Wrappers
# ==============================================================================

# No-PBC naive neighbor matrix kernel wrappers
_jax_fill_naive_f32 = jax_kernel(
    _fill_naive_neighbor_matrix_overload[wp.float32],
    num_outputs=2,
    in_out_argnames=["neighbor_matrix", "num_neighbors"],
    enable_backward=False,
)
_jax_fill_naive_f64 = jax_kernel(
    _fill_naive_neighbor_matrix_overload[wp.float64],
    num_outputs=2,
    in_out_argnames=["neighbor_matrix", "num_neighbors"],
    enable_backward=False,
)

# PBC naive neighbor matrix kernel wrappers
_jax_fill_naive_pbc_f32 = jax_kernel(
    _fill_naive_neighbor_matrix_pbc_overload[wp.float32],
    num_outputs=3,
    in_out_argnames=["neighbor_matrix", "neighbor_matrix_shifts", "num_neighbors"],
    enable_backward=False,
)
_jax_fill_naive_pbc_f64 = jax_kernel(
    _fill_naive_neighbor_matrix_pbc_overload[wp.float64],
    num_outputs=3,
    in_out_argnames=["neighbor_matrix", "neighbor_matrix_shifts", "num_neighbors"],
    enable_backward=False,
)

# Selective no-PBC naive neighbor matrix kernel wrappers
_jax_fill_naive_selective_f32 = jax_kernel(
    _fill_naive_neighbor_matrix_selective_overload[wp.float32],
    num_outputs=2,
    in_out_argnames=["neighbor_matrix", "num_neighbors"],
    enable_backward=False,
)
_jax_fill_naive_selective_f64 = jax_kernel(
    _fill_naive_neighbor_matrix_selective_overload[wp.float64],
    num_outputs=2,
    in_out_argnames=["neighbor_matrix", "num_neighbors"],
    enable_backward=False,
)

# Selective PBC naive neighbor matrix kernel wrappers
_jax_fill_naive_pbc_selective_f32 = jax_kernel(
    _fill_naive_neighbor_matrix_pbc_selective_overload[wp.float32],
    num_outputs=3,
    in_out_argnames=["neighbor_matrix", "neighbor_matrix_shifts", "num_neighbors"],
    enable_backward=False,
)
_jax_fill_naive_pbc_selective_f64 = jax_kernel(
    _fill_naive_neighbor_matrix_pbc_selective_overload[wp.float64],
    num_outputs=3,
    in_out_argnames=["neighbor_matrix", "neighbor_matrix_shifts", "num_neighbors"],
    enable_backward=False,
)

# PBC prewrapped naive neighbor matrix kernel wrappers
_jax_fill_naive_pbc_prewrapped_f32 = jax_kernel(
    _fill_naive_neighbor_matrix_pbc_prewrapped_overload[wp.float32],
    num_outputs=3,
    in_out_argnames=["neighbor_matrix", "neighbor_matrix_shifts", "num_neighbors"],
    enable_backward=False,
)
_jax_fill_naive_pbc_prewrapped_f64 = jax_kernel(
    _fill_naive_neighbor_matrix_pbc_prewrapped_overload[wp.float64],
    num_outputs=3,
    in_out_argnames=["neighbor_matrix", "neighbor_matrix_shifts", "num_neighbors"],
    enable_backward=False,
)

# Selective PBC prewrapped naive neighbor matrix kernel wrappers
_jax_fill_naive_pbc_prewrapped_selective_f32 = jax_kernel(
    _fill_naive_neighbor_matrix_pbc_prewrapped_selective_overload[wp.float32],
    num_outputs=3,
    in_out_argnames=["neighbor_matrix", "neighbor_matrix_shifts", "num_neighbors"],
    enable_backward=False,
)
_jax_fill_naive_pbc_prewrapped_selective_f64 = jax_kernel(
    _fill_naive_neighbor_matrix_pbc_prewrapped_selective_overload[wp.float64],
    num_outputs=3,
    in_out_argnames=["neighbor_matrix", "neighbor_matrix_shifts", "num_neighbors"],
    enable_backward=False,
)

# Wrap positions single kernel wrappers
_jax_wrap_positions_single_f32 = jax_kernel(
    _wrap_positions_single_overload[wp.float32],
    num_outputs=2,
    in_out_argnames=["positions_wrapped", "per_atom_cell_offsets"],
    enable_backward=False,
)
_jax_wrap_positions_single_f64 = jax_kernel(
    _wrap_positions_single_overload[wp.float64],
    num_outputs=2,
    in_out_argnames=["positions_wrapped", "per_atom_cell_offsets"],
    enable_backward=False,
)


def naive_neighbor_list(
    positions: jax.Array,
    cutoff: float,
    cell: jax.Array | None = None,
    pbc: jax.Array | None = None,
    max_neighbors: int | None = None,
    half_fill: bool = False,
    fill_value: int | None = None,
    return_neighbor_list: bool = False,
    neighbor_matrix: jax.Array | None = None,
    neighbor_matrix_shifts: jax.Array | None = None,
    num_neighbors: jax.Array | None = None,
    shift_range_per_dimension: jax.Array | None = None,
    num_shifts_per_system: jax.Array | None = None,
    max_shifts_per_system: int | None = None,
    rebuild_flags: jax.Array | None = None,
    wrap_positions: bool = True,
) -> (
    tuple[jax.Array, jax.Array, jax.Array, jax.Array]
    | tuple[jax.Array, jax.Array, jax.Array]
    | tuple[jax.Array, jax.Array]
):
    """Compute neighbor list using naive O(N^2) algorithm.

    Identifies all atom pairs within a specified cutoff distance using a
    brute-force pairwise distance calculation. Supports both non-periodic
    and periodic boundary conditions.

    Parameters
    ----------
    positions : jax.Array, shape (total_atoms, 3), dtype=float32 or float64
        Atomic coordinates in Cartesian space. Each row represents one atom's
        (x, y, z) position.
    cutoff : float
        Cutoff distance for neighbor detection in Cartesian units.
        Must be positive. Atoms within this distance are considered neighbors.
    pbc : jax.Array, shape (3,) or (1, 3), dtype=bool, optional
        Periodic boundary condition flags for each dimension.
        True enables periodicity in that direction. Default is None (no PBC).
    cell : jax.Array, shape (1, 3, 3), dtype=float32 or float64, optional
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
    neighbor_matrix : jax.Array, shape (total_atoms, max_neighbors), dtype=int32, optional
        Neighbor matrix to be filled. Pass in a pre-shaped array to hint buffer reuse
        to XLA; note that JAX returns a new array rather than mutating the input.
        Must be provided if max_neighbors is not provided.
    neighbor_matrix_shifts : jax.Array, shape (total_atoms, max_neighbors, 3), dtype=int32, optional
        Shift vectors for each neighbor relationship. Pass in a pre-shaped array to hint
        buffer reuse to XLA; note that JAX returns a new array rather than mutating the input.
        Must be provided if max_neighbors is not provided.
    num_neighbors : jax.Array, shape (total_atoms,), dtype=int32, optional
        Number of neighbors found for each atom. Pass in a pre-shaped array to hint buffer
        reuse to XLA; note that JAX returns a new array rather than mutating the input.
        Must be provided if max_neighbors is not provided.
    shift_range_per_dimension : jax.Array, shape (1, 3), dtype=int32, optional
        Shift range in each dimension for each system.
        Pass in a pre-computed value to avoid recomputation for PBC systems.
    num_shifts_per_system : jax.Array, shape (1,), dtype=int32, optional
        Number of periodic shifts for the system.
        Pass in a pre-computed value to avoid recomputation for PBC systems.
    max_shifts_per_system : int, optional
        Maximum per-system shift count.
        Pass in a pre-computed value to avoid recomputation for PBC systems.
    return_neighbor_list : bool, optional - default = False
        If True, convert the neighbor matrix to a neighbor list (idx_i, idx_j) format by
        creating a mask over the fill_value, which can incur a performance penalty.
    wrap_positions : bool, default=True
        If True, wrap input positions into the primary cell before
        neighbor search. Set to False when positions are already
        wrapped (e.g. by a preceding integration step) to save two
        GPU kernel launches per call.

    Returns
    -------
    results : tuple of jax.Array
        Variable-length tuple depending on input parameters. The return pattern follows:

        - No PBC, matrix format: ``(neighbor_matrix, num_neighbors)``
        - No PBC, list format: ``(neighbor_list, neighbor_ptr)``
        - With PBC, matrix format: ``(neighbor_matrix, num_neighbors, neighbor_matrix_shifts)``
        - With PBC, list format: ``(neighbor_list, neighbor_ptr, neighbor_list_shifts)``

        **Components returned:**

        - **neighbor_data** (array): Neighbor indices, format depends on ``return_neighbor_list``:

            * If ``return_neighbor_list=False`` (default): Returns ``neighbor_matrix``
              with shape (total_atoms, max_neighbors), dtype int32. Each row i contains
              indices of atom i's neighbors.
            * If ``return_neighbor_list=True``: Returns ``neighbor_list`` with shape
              (2, num_pairs), dtype int32, in COO format [source_atoms, target_atoms].

        - **num_neighbor_data** (array): Information about the number of neighbors for each atom,
          format depends on ``return_neighbor_list``:

            * If ``return_neighbor_list=False`` (default): Returns ``num_neighbors`` with shape (total_atoms,), dtype int32.
              Count of neighbors found for each atom. Always returned.
            * If ``return_neighbor_list=True``: Returns ``neighbor_ptr`` with shape (total_atoms + 1,), dtype int32.
              CSR-style pointer arrays where ``neighbor_ptr_data[i]`` to ``neighbor_ptr_data[i+1]`` gives the range of
              neighbors for atom i in the flattened neighbor list.

        - **neighbor_shift_data** (array, optional): Periodic shift vectors, only when ``pbc`` is provided:
          format depends on ``return_neighbor_list``:

            * If ``return_neighbor_list=False`` (default): Returns ``neighbor_matrix_shifts`` with
              shape (total_atoms, max_neighbors, 3), dtype int32.
            * If ``return_neighbor_list=True``: Returns ``unit_shifts`` with shape
              (num_pairs, 3), dtype int32.

    Examples
    --------
    Basic usage without periodic boundary conditions:

    >>> import jax.numpy as jnp
    >>> from nvalchemiops.jax.neighbors import naive_neighbor_list
    >>> positions = jnp.zeros((100, 3), dtype=jnp.float32)
    >>> cutoff = 2.5
    >>> max_neighbors = 50
    >>> neighbor_matrix, num_neighbors = naive_neighbor_list(
    ...     positions, cutoff, max_neighbors=max_neighbors
    ... )

    With periodic boundary conditions:

    >>> cell = jnp.eye(3, dtype=jnp.float32).reshape(1, 3, 3) * 10.0
    >>> pbc = jnp.array([[True, True, True]])
    >>> neighbor_matrix, num_neighbors, shifts = naive_neighbor_list(
    ...     positions, cutoff, max_neighbors=max_neighbors, pbc=pbc, cell=cell
    ... )

    Return as neighbor list instead of matrix:

    >>> neighbor_list, neighbor_ptr = naive_neighbor_list(
    ...     positions, cutoff, max_neighbors=max_neighbors, return_neighbor_list=True
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
        cell = cell if cell.ndim == 3 else cell[jnp.newaxis, :, :]
        # Ensure cell dtype matches positions dtype so warp overload dispatch is consistent
        if cell.dtype != positions.dtype:
            cell = cell.astype(positions.dtype)
    if pbc is not None:
        pbc = pbc if pbc.ndim == 2 else pbc[jnp.newaxis, :]

    if max_neighbors is None and (
        neighbor_matrix is None
        or (neighbor_matrix_shifts is None and pbc is not None)
        or num_neighbors is None
    ):
        max_neighbors = estimate_max_neighbors(cutoff)

    if fill_value is None:
        fill_value = jnp.int32(positions.shape[0])

    if neighbor_matrix is None:
        neighbor_matrix = jnp.full(
            (positions.shape[0], max_neighbors),
            fill_value,
            dtype=jnp.int32,
        )
    elif rebuild_flags is None:
        neighbor_matrix = neighbor_matrix.at[:].set(fill_value)

    if num_neighbors is None:
        num_neighbors = jnp.zeros(positions.shape[0], dtype=jnp.int32)
    elif rebuild_flags is None:
        num_neighbors = num_neighbors.at[:].set(jnp.int32(0))

    if pbc is not None:
        if neighbor_matrix_shifts is None:
            neighbor_matrix_shifts = jnp.zeros(
                (positions.shape[0], max_neighbors, 3),
                dtype=jnp.int32,
            )
        elif rebuild_flags is None:
            neighbor_matrix_shifts = neighbor_matrix_shifts.at[:].set(jnp.int32(0))
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
                    jnp.zeros((2, 0), dtype=jnp.int32),
                    jnp.zeros(
                        (positions.shape[0] + 1,),
                        dtype=jnp.int32,
                    ),
                    jnp.zeros((0, 3), dtype=jnp.int32),
                )
            else:
                return (
                    jnp.zeros((2, 0), dtype=jnp.int32),
                    jnp.zeros(
                        (positions.shape[0] + 1,),
                        dtype=jnp.int32,
                    ),
                )
        else:
            if pbc is not None:
                return neighbor_matrix, num_neighbors, neighbor_matrix_shifts
            else:
                return neighbor_matrix, num_neighbors

    # Select kernel based on dtype
    if positions.dtype == jnp.float64:
        _jax_fill = _jax_fill_naive_f64
        _jax_fill_pbc = _jax_fill_naive_pbc_f64
        _jax_fill_pbc_prewrapped = _jax_fill_naive_pbc_prewrapped_f64
        _jax_fill_selective = _jax_fill_naive_selective_f64
        _jax_fill_pbc_selective = _jax_fill_naive_pbc_selective_f64
        _jax_fill_pbc_prewrapped_selective = (
            _jax_fill_naive_pbc_prewrapped_selective_f64
        )
        _jax_wrap_single = _jax_wrap_positions_single_f64
    else:
        _jax_fill = _jax_fill_naive_f32
        _jax_fill_pbc = _jax_fill_naive_pbc_f32
        _jax_fill_pbc_prewrapped = _jax_fill_naive_pbc_prewrapped_f32
        _jax_fill_selective = _jax_fill_naive_selective_f32
        _jax_fill_pbc_selective = _jax_fill_naive_pbc_selective_f32
        _jax_fill_pbc_prewrapped_selective = (
            _jax_fill_naive_pbc_prewrapped_selective_f32
        )
        _jax_wrap_single = _jax_wrap_positions_single_f32
        positions = positions.astype(jnp.float32)

    total_atoms = positions.shape[0]

    if pbc is None:
        # No PBC case
        if rebuild_flags is not None:
            rf = rebuild_flags.flatten()[:1].astype(jnp.bool_)
            num_neighbors = jnp.where(
                rf[0], jnp.zeros_like(num_neighbors), num_neighbors
            )
            neighbor_matrix, num_neighbors = _jax_fill_selective(
                positions,
                float(cutoff * cutoff),
                neighbor_matrix,
                num_neighbors,
                half_fill,
                rf,
                launch_dims=(total_atoms,),
            )
        else:
            neighbor_matrix, num_neighbors = _jax_fill(
                positions,
                float(cutoff * cutoff),
                neighbor_matrix,
                num_neighbors,
                half_fill,
                launch_dims=(total_atoms,),
            )
    else:
        if cell.dtype != positions.dtype:
            cell = cell.astype(positions.dtype)

        if wrap_positions:
            inv_cell = jnp.linalg.inv(cell)
            positions_wrapped = jnp.zeros_like(positions)
            per_atom_cell_offsets = jnp.zeros((total_atoms, 3), dtype=jnp.int32)
            positions_wrapped, per_atom_cell_offsets = _jax_wrap_single(
                positions,
                cell,
                inv_cell,
                positions_wrapped,
                per_atom_cell_offsets,
                launch_dims=(total_atoms,),
            )

            if rebuild_flags is not None:
                rf = rebuild_flags.flatten()[:1].astype(jnp.bool_)
                num_neighbors = jnp.where(
                    rf[0], jnp.zeros_like(num_neighbors), num_neighbors
                )
                neighbor_matrix, neighbor_matrix_shifts, num_neighbors = (
                    _jax_fill_pbc_selective(
                        positions_wrapped,
                        per_atom_cell_offsets,
                        float(cutoff * cutoff),
                        cell,
                        shift_range_per_dimension,
                        neighbor_matrix,
                        neighbor_matrix_shifts,
                        num_neighbors,
                        half_fill,
                        rf,
                        launch_dims=(max_shifts_per_system, total_atoms),
                    )
                )
            else:
                neighbor_matrix, neighbor_matrix_shifts, num_neighbors = _jax_fill_pbc(
                    positions_wrapped,
                    per_atom_cell_offsets,
                    float(cutoff * cutoff),
                    cell,
                    shift_range_per_dimension,
                    neighbor_matrix,
                    neighbor_matrix_shifts,
                    num_neighbors,
                    half_fill,
                    launch_dims=(max_shifts_per_system, total_atoms),
                )
        else:
            if rebuild_flags is not None:
                rf = rebuild_flags.flatten()[:1].astype(jnp.bool_)
                num_neighbors = jnp.where(
                    rf[0], jnp.zeros_like(num_neighbors), num_neighbors
                )
                neighbor_matrix, neighbor_matrix_shifts, num_neighbors = (
                    _jax_fill_pbc_prewrapped_selective(
                        positions,
                        float(cutoff * cutoff),
                        cell,
                        shift_range_per_dimension,
                        neighbor_matrix,
                        neighbor_matrix_shifts,
                        num_neighbors,
                        half_fill,
                        rf,
                        launch_dims=(max_shifts_per_system, total_atoms),
                    )
                )
            else:
                neighbor_matrix, neighbor_matrix_shifts, num_neighbors = (
                    _jax_fill_pbc_prewrapped(
                        positions,
                        float(cutoff * cutoff),
                        cell,
                        shift_range_per_dimension,
                        neighbor_matrix,
                        neighbor_matrix_shifts,
                        num_neighbors,
                        half_fill,
                        launch_dims=(max_shifts_per_system, total_atoms),
                    )
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
