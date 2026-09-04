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

"""JAX utilities for neighbor list construction.

This module contains JAX-specific helper functions for neighbor list operations.
"""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
import warp as wp
from warp import jax_kernel

from nvalchemiops.neighbors.neighbor_utils import (
    NeighborOverflowError,
    estimate_max_neighbors,
    get_compute_naive_num_shifts_kernel,
)

_INT32_MAX = 2**31 - 1
_INT32_HALF_MAX = (_INT32_MAX - 1) // 2

__all__ = [
    "compute_naive_num_shifts",
    "get_fixed_capacity_neighbor_list_from_neighbor_matrix",
    "get_neighbor_list_from_neighbor_matrix",
    "prepare_batch_idx_ptr",
    "allocate_cell_list",
    "estimate_max_neighbors",
    "NeighborOverflowError",
]


def _fixed_capacity_flat_indices(
    active_mask: jax.Array,
    capacity: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return fixed-size flat active indices, a valid-slot mask, and count."""
    capacity = int(capacity)
    if capacity < 0:
        raise ValueError(f"capacity must be non-negative, got {capacity}")
    active_count = jnp.sum(active_mask, dtype=jnp.int32)
    flat_indices = jnp.nonzero(
        active_mask.reshape(-1),
        size=capacity,
        fill_value=0,
    )[0]
    valid_slots = jnp.arange(capacity, dtype=jnp.int32) < active_count
    return flat_indices, valid_slots, active_count


def build_naive_kernel_tables(
    operation: Literal["single_cutoff", "dual_cutoff"],
    *,
    batched: bool,
    dtypes: tuple[type, ...],
    half_fill: bool = False,
) -> tuple[dict, dict, dict, dict, dict, dict]:
    """Build the six naive Warp kernel tables the JAX wrappers need.

    Returns per-dtype kernel tables for, in order:

    1. no-PBC,
    2. no-PBC selective-rebuild,
    3. wrap-on-entry PBC,
    4. wrap-on-entry PBC selective-rebuild,
    5. prewrapped PBC,
    6. prewrapped PBC selective-rebuild.

    Parameters
    ----------
    operation : {"single_cutoff", "dual_cutoff"}
        Whether to build tables for single-cutoff or dual-cutoff neighbor search.
    batched : bool
        If ``True``, build kernels for batched (multi-system) neighbor search.
    dtypes : tuple of type
        Floating-point dtypes (e.g. ``(wp.float32,)``) for which to instantiate
        each kernel variant.
    half_fill : bool, optional
        If ``True``, only fill the upper triangle of the neighbor matrix.
        Default is ``False``.

    Returns
    -------
    tuple of six dict
        Six ``{dtype: kernel}`` lookup dictionaries, one per (PBC mode, selective)
        combination in the order listed above.

    Raises
    ------
    ValueError
        If ``operation`` is not ``"single_cutoff"`` or ``"dual_cutoff"``.

    See Also
    --------
    :func:`nvalchemiops.neighbors.naive.get_naive_neighbor_matrix_kernel` : Single-cutoff kernel factory.
    :func:`nvalchemiops.neighbors.naive.get_naive_neighbor_matrix_dual_cutoff_kernel` : Dual-cutoff kernel factory.
    """
    from nvalchemiops.neighbors.naive import (
        get_naive_neighbor_matrix_dual_cutoff_kernel,
        get_naive_neighbor_matrix_kernel,
    )

    if operation == "single_cutoff":
        getter = get_naive_neighbor_matrix_kernel
    elif operation == "dual_cutoff":
        getter = get_naive_neighbor_matrix_dual_cutoff_kernel
    else:
        raise ValueError("operation must be 'single_cutoff' or 'dual_cutoff'")

    def _table(pbc_mode: str, selective: bool) -> dict:
        return {
            t: getter(
                t,
                pbc_mode=pbc_mode,
                batched=batched,
                half_fill=bool(half_fill),
                selective=selective,
            )
            for t in dtypes
        }

    return (
        _table("none", False),
        _table("none", True),
        _table("wrap_on_entry", False),
        _table("wrap_on_entry", True),
        _table("prewrapped", False),
        _table("prewrapped", True),
    )


def _validate_graph_mode(graph_mode: str) -> Literal["none", "warp"]:
    """Validate the public ``graph_mode`` argument used by neighbor-list APIs.

    Parameters
    ----------
    graph_mode : str
        User-supplied mode string. Must be one of ``{"none", "warp"}``.

    Returns
    -------
    Literal["none", "warp"]
        The validated mode.

    Raises
    ------
    ValueError
        If ``graph_mode`` is not a recognized mode.
    """
    if graph_mode not in {"none", "warp"}:
        raise ValueError("graph_mode must be one of {'none', 'warp'}")
    return graph_mode


# ==============================================================================
# JAX Kernel Wrappers
# ==============================================================================

# Wrap the original kernels with jax_kernel
# jax_kernel handles the bool-to-int conversion internally
_jax_compute_naive_num_shifts_f32 = jax_kernel(
    get_compute_naive_num_shifts_kernel(wp.float32),
    num_outputs=2,
    in_out_argnames=["num_shifts", "shift_range"],
    enable_backward=False,
)

_jax_compute_naive_num_shifts_f64 = jax_kernel(
    get_compute_naive_num_shifts_kernel(wp.float64),
    num_outputs=2,
    in_out_argnames=["num_shifts", "shift_range"],
    enable_backward=False,
)


def _num_shifts_from_shift_range(
    shift_range: jax.Array,
) -> tuple[jax.Array, int]:
    """Return int32 shift counts after checking the count formula for overflow."""
    num_systems = shift_range.shape[0]
    if num_systems == 0:
        return jnp.zeros((0,), dtype=jnp.int32), 0

    shift_range_i32 = shift_range.astype(jnp.int32)
    rx = shift_range_i32[:, 0]
    ry = shift_range_i32[:, 1]
    rz = shift_range_i32[:, 2]

    int32_max = jnp.array(_INT32_MAX, dtype=jnp.int32)
    int32_half_max = jnp.array(_INT32_HALF_MAX, dtype=jnp.int32)

    k1_overflows = ry > int32_half_max
    k2_overflows = rz > int32_half_max
    k1 = 2 * jnp.minimum(ry, int32_half_max) + 1
    k2 = 2 * jnp.minimum(rz, int32_half_max) + 1

    tail_limit = int32_max - rz - 1
    base_limit = tail_limit // k2
    base_room = jnp.maximum(base_limit - ry, 0)
    count_overflows = (
        (base_limit < 0)
        | (ry > base_limit)
        | (rx > (base_room // k1))
        | (k1_overflows & (rx > 0))
        | (k2_overflows & ((rx > 0) | (ry > 0)))
    )
    num_shifts = ((rx * k1 + ry) * k2 + rz + 1).astype(jnp.int32)
    max_shifts = int(num_shifts.max())
    if bool(jnp.any(count_overflows)):
        raise ValueError(
            "Per-system shift count exceeds int32 max "
            "(2^31 - 1). Reduce the cutoff, increase cell size, or use a "
            "cell-list method for very small cells."
        )

    return num_shifts, max_shifts


# ==============================================================================
# Public API
# ==============================================================================


def compute_naive_num_shifts(
    cell: jax.Array,
    cutoff: float,
    pbc: jax.Array,
) -> tuple[jax.Array, jax.Array, int]:
    """Compute periodic image shifts needed for neighbor searching.

    Parameters
    ----------
    cell : jax.Array, shape (num_systems, 3, 3)
        Cell matrices defining lattice vectors in Cartesian coordinates.
        Each 3x3 matrix represents one system's periodic cell.
    cutoff : float
        Cutoff distance for neighbor searching in Cartesian units.
        Must be positive and typically less than half the minimum cell dimension.
    pbc : jax.Array, shape (num_systems, 3), dtype=bool
        Periodic boundary condition flags for each dimension.
        True enables periodicity in that direction.

    Returns
    -------
    shift_range : jax.Array, shape (num_systems, 3), dtype=int32
        Maximum shift indices in each dimension for each system.
    num_shifts : jax.Array, shape (num_systems,), dtype=int32
        Number of periodic shifts for each system.
    max_shifts : int
        Maximum per-system shift count across all systems.

    Raises
    ------
    ValueError
        If any per-system shift count exceeds int32 range.

    See Also
    --------
    nvalchemiops.neighbors.neighbor_utils.get_compute_naive_num_shifts_kernel : Warp kernel factory

    Notes
    -----
    This function must be called outside ``jax.jit`` scope. The returned
    ``max_shifts`` is a Python int needed for determining launch dimensions,
    which cannot be traced. This is an inherent limitation: array shapes must
    be known at trace time in JAX.
    """
    num_systems = cell.shape[0]

    # Allocate outputs as JAX arrays
    num_shifts_i32 = jnp.zeros(num_systems, dtype=jnp.int32)
    shift_range = jnp.zeros((num_systems, 3), dtype=jnp.int32)

    # Ensure pbc is bool dtype (jax_kernel handles bool arrays directly)
    pbc_bool = pbc.astype(jnp.bool_)

    # Select the appropriate kernel based on input dtype
    if cell.dtype == jnp.float64 and jax.config.jax_enable_x64:
        cell_f64 = cell.astype(jnp.float64)
        num_shifts_i32, shift_range = _jax_compute_naive_num_shifts_f64(
            cell_f64,
            float(cutoff),
            pbc_bool,
            num_shifts_i32,
            shift_range,
            launch_dims=(num_systems,),
        )
    else:
        cell_f32 = cell.astype(jnp.float32)
        num_shifts_i32, shift_range = _jax_compute_naive_num_shifts_f32(
            cell_f32,
            float(cutoff),
            pbc_bool,
            num_shifts_i32,
            shift_range,
            launch_dims=(num_systems,),
        )

    num_shifts, max_shifts = _num_shifts_from_shift_range(shift_range)
    return shift_range, num_shifts, max_shifts


def get_neighbor_list_from_neighbor_matrix(
    neighbor_matrix: jax.Array,
    num_neighbors: jax.Array,
    neighbor_shift_matrix: jax.Array | None = None,
    fill_value: int = -1,
) -> tuple[jax.Array, jax.Array] | tuple[jax.Array, jax.Array, jax.Array]:
    """Convert neighbor matrix format to neighbor list format.

    Parameters
    ----------
    neighbor_matrix : jax.Array, shape (total_atoms, max_neighbors), dtype=int32
        The neighbor matrix with neighbor atom indices.
    num_neighbors : jax.Array, shape (total_atoms,), dtype=int32
        The number of neighbors for each atom.
    neighbor_shift_matrix : jax.Array | None, shape (total_atoms, max_neighbors, 3), dtype=int32
        Optional neighbor shift matrix with periodic shift vectors.
    fill_value : int, default=-1
        The fill value used in the neighbor matrix to indicate empty slots.
        This is used to create a mask from the neighbor matrix.

    Returns
    -------
    neighbor_list : jax.Array, shape (2, num_pairs), dtype=int32
        The neighbor list in COO format [source_atoms, target_atoms].
    neighbor_ptr : jax.Array, shape (total_atoms + 1,), dtype=int32
        CSR-style pointer array where neighbor_ptr[i]:neighbor_ptr[i+1] gives the range of
        neighbors for atom i in the flattened neighbor list.
    neighbor_list_shifts : jax.Array, shape (num_pairs, 3), dtype=int32
        The neighbor shift vectors (only returned if neighbor_shift_matrix is not None).

    Raises
    ------
    ValueError
        If the max number of neighbors is larger than the neighbor matrix width.

    Notes
    -----
    This is a pure JAX utility function with no warp dependencies. It converts
    from the fixed-width matrix format to the variable-width list format by masking
    out fill values and flattening the result.

    See Also
    --------
    nvalchemiops.jax.neighbors.naive.naive_neighbor_list : Uses this for format conversion
    nvalchemiops.jax.neighbors.cell_list.cell_list : Uses this for format conversion
    """
    # Handle empty case
    if neighbor_matrix.shape[0] == 0:
        neighbor_list = jnp.zeros((2, 0), dtype=neighbor_matrix.dtype)
        neighbor_ptr = jnp.zeros(1, dtype=jnp.int32)
        if neighbor_shift_matrix is not None:
            neighbor_shift_list = jnp.empty((0, 3), dtype=neighbor_shift_matrix.dtype)
            return neighbor_list, neighbor_ptr, neighbor_shift_list
        else:
            return neighbor_list, neighbor_ptr

    # Validate that the neighbor matrix is large enough
    # Note: This check only works outside jax.jit scope; inside jit it's skipped
    # because max_found would be a tracer and int() conversion fails.
    max_found = jnp.max(num_neighbors)
    try:
        if int(max_found) > neighbor_matrix.shape[1]:
            raise NeighborOverflowError(
                neighbor_matrix.shape[1],
                int(max_found),
            )
    except (
        jax.errors.ConcretizationTypeError,
        jax.errors.TracerIntegerConversionError,
    ):
        pass  # Skip validation during jax.jit tracing

    # Create mask and extract neighbor pairs
    mask = neighbor_matrix != fill_value
    dtype = neighbor_matrix.dtype
    i_idx = jnp.where(mask)[0].astype(dtype)
    j_idx = neighbor_matrix[mask].astype(dtype)
    neighbor_list = jnp.stack([i_idx, j_idx], axis=0)

    # Create CSR-style pointer array
    neighbor_ptr = jnp.zeros(num_neighbors.shape[0] + 1, dtype=jnp.int32)
    neighbor_ptr = neighbor_ptr.at[1:].set(jnp.cumsum(num_neighbors, dtype=jnp.int32))

    if neighbor_shift_matrix is not None:
        neighbor_list_shifts = neighbor_shift_matrix[mask]
        return neighbor_list, neighbor_ptr, neighbor_list_shifts
    else:
        return neighbor_list, neighbor_ptr


def get_fixed_capacity_neighbor_list_from_neighbor_matrix(
    neighbor_matrix: jax.Array,
    num_neighbors: jax.Array,
    capacity: int,
    neighbor_shift_matrix: jax.Array | None = None,
    fill_value: int = -1,
) -> (
    tuple[jax.Array, jax.Array, jax.Array]
    | tuple[jax.Array, jax.Array, jax.Array, jax.Array]
):
    """Convert a neighbor matrix to fixed-capacity COO form under ``jax.jit``.

    Parameters
    ----------
    neighbor_matrix : jax.Array, shape (num_rows, max_neighbors), dtype=int32
        Fixed-width neighbor indices.
    num_neighbors : jax.Array, shape (num_rows,), dtype=int32
        Unclipped neighbor counts produced by the neighbor query.
    capacity : int
        Static number of COO columns to return.
    neighbor_shift_matrix : jax.Array, shape (num_rows, max_neighbors, 3), optional
        Shift vectors aligned with ``neighbor_matrix``.
    fill_value : int, default=-1
        Matrix sentinel and padding value for unused COO columns.

    Returns
    -------
    neighbor_list : jax.Array, shape (2, capacity), dtype=int32
        Row-major COO pairs padded with ``fill_value``.
    neighbor_ptr : jax.Array, shape (num_rows + 1,), dtype=int32
        Pointers into the stored prefix. Values are clipped to ``capacity`` so
        they are safe to consume even when ``overflow`` is true.
    neighbor_list_shifts : jax.Array, shape (capacity, 3), dtype=int32
        Shift vectors aligned with ``neighbor_list``. Returned before
        ``overflow`` when ``neighbor_shift_matrix`` is supplied.
    overflow : jax.Array, shape (), dtype=bool
        True when either a matrix row exceeds ``max_neighbors`` or the stored
        COO pairs exceed ``capacity``.

    Notes
    -----
    ``capacity`` determines every output shape, while pair counts stay on the
    device. This makes the conversion compatible with ``jax.jit``. An eager
    caller should inspect ``overflow`` and retry with larger matrix or COO
    capacities before consuming a truncated result.

    See Also
    --------
    get_neighbor_list_from_neighbor_matrix : Compact eager conversion.
    """
    capacity = int(capacity)
    if capacity < 0:
        raise ValueError(f"capacity must be non-negative, got {capacity}")
    if neighbor_matrix.size == 0:
        neighbor_list = jnp.full(
            (2, capacity),
            fill_value,
            dtype=neighbor_matrix.dtype,
        )
        neighbor_ptr = jnp.zeros(num_neighbors.shape[0] + 1, dtype=jnp.int32)
        overflow = jnp.any(num_neighbors > neighbor_matrix.shape[1])
        if neighbor_shift_matrix is not None:
            neighbor_list_shifts = jnp.zeros(
                (capacity, 3),
                dtype=neighbor_shift_matrix.dtype,
            )
            return neighbor_list, neighbor_ptr, neighbor_list_shifts, overflow
        return neighbor_list, neighbor_ptr, overflow

    active_mask = neighbor_matrix != fill_value
    flat_indices, valid_slots, active_count = _fixed_capacity_flat_indices(
        active_mask,
        capacity,
    )
    _, matrix_width = neighbor_matrix.shape
    source_indices = flat_indices // matrix_width
    target_indices = neighbor_matrix.reshape(-1)[flat_indices]
    pad = jnp.asarray(fill_value, dtype=neighbor_matrix.dtype)
    source_indices = jnp.where(valid_slots, source_indices, pad)
    target_indices = jnp.where(valid_slots, target_indices, pad)
    neighbor_list = jnp.stack(
        [
            source_indices.astype(neighbor_matrix.dtype),
            target_indices.astype(neighbor_matrix.dtype),
        ],
        axis=0,
    )

    stored_per_row = jnp.sum(active_mask, axis=1, dtype=jnp.int32)
    neighbor_ptr = jnp.concatenate(
        [
            jnp.zeros(1, dtype=jnp.int32),
            jnp.minimum(
                jnp.cumsum(stored_per_row, dtype=jnp.int32),
                jnp.int32(capacity),
            ),
        ]
    )
    overflow = (active_count > capacity) | jnp.any(num_neighbors > matrix_width)

    if neighbor_shift_matrix is not None:
        flat_shifts = neighbor_shift_matrix.reshape(-1, 3)[flat_indices]
        neighbor_list_shifts = jnp.where(
            valid_slots[:, None],
            flat_shifts,
            jnp.zeros_like(flat_shifts),
        )
        return neighbor_list, neighbor_ptr, neighbor_list_shifts, overflow
    return neighbor_list, neighbor_ptr, overflow


def coo_pack_pair_geometry(
    active_mask: jax.Array,
    distances: jax.Array | None = None,
    vectors: jax.Array | None = None,
    capacity: int | None = None,
) -> tuple[jax.Array | None, jax.Array | None]:
    """Repack matrix-layout per-pair geometry into COO order.

    ``active_mask`` is ``neighbor_matrix != fill_value``.  Flattening it in
    row-major order yields the active-slot indices in the same order
    :func:`get_neighbor_list_from_neighbor_matrix` uses, so the gathered
    distances ``(num_pairs,)`` and vectors ``(num_pairs, 3)`` index-align with
    the returned neighbor list. With ``capacity=None`` the pair count is
    data-dependent and the conversion is eager. A static ``capacity`` returns
    padded fixed-size arrays compatible with ``jax.jit``.

    Parameters
    ----------
    active_mask : jax.Array, shape (total_atoms, max_neighbors), dtype=bool
        Mask of active neighbor-matrix slots.
    distances : jax.Array | None, shape (total_atoms, max_neighbors)
        Per-pair distances in matrix layout, or ``None``.
    vectors : jax.Array | None, shape (total_atoms, max_neighbors, 3)
        Per-pair displacement vectors in matrix layout, or ``None``.
    capacity : int, optional
        Static number of output pairs. Unused tail entries are zero.

    Returns
    -------
    tuple of (jax.Array | None, jax.Array | None)
        ``(distances, vectors)`` in COO layout, each unchanged if ``None``.
    """
    if capacity is None:
        flat_active = jnp.nonzero(active_mask.reshape(-1))[0]
        valid_slots = None
    else:
        flat_active, valid_slots, _ = _fixed_capacity_flat_indices(
            active_mask,
            capacity,
        )
    if distances is not None:
        distances = jnp.take(distances.reshape(-1), flat_active, axis=0)
        if valid_slots is not None:
            distances = jnp.where(valid_slots, distances, jnp.zeros_like(distances))
    if vectors is not None:
        vectors = jnp.take(vectors.reshape(-1, vectors.shape[-1]), flat_active, axis=0)
        if valid_slots is not None:
            vectors = jnp.where(
                valid_slots[:, None],
                vectors,
                jnp.zeros_like(vectors),
            )
    return distances, vectors


def prepare_batch_idx_ptr(
    batch_idx: jax.Array | None,
    batch_ptr: jax.Array | None,
    num_atoms: int,
) -> tuple[jax.Array, jax.Array]:
    """Prepare batch index and pointer tensors from either representation.

    Utility function to ensure both batch_idx and batch_ptr are available,
    computing one from the other if needed.

    Parameters
    ----------
    batch_idx : jax.Array | None, shape (total_atoms,), dtype=int32
        Array indicating the batch index for each atom.
    batch_ptr : jax.Array | None, shape (num_systems + 1,), dtype=int32
        Array indicating the start index of each batch in the atom list.
    num_atoms : int
        Total number of atoms across all systems.

    Returns
    -------
    batch_idx : jax.Array, shape (total_atoms,), dtype=int32
        Prepared batch index tensor.
    batch_ptr : jax.Array, shape (num_systems + 1,), dtype=int32
        Prepared batch pointer tensor.

    Raises
    ------
    ValueError
        If both batch_idx and batch_ptr are None.

    Notes
    -----
    This is a pure JAX utility function with no warp dependencies. It provides
    convenience for batch operations by converting between dense (batch_idx) and
    sparse (batch_ptr) batch representations.

    See Also
    --------
    nvalchemiops.jax.neighbors.batch_naive.batch_naive_neighbor_list : Uses this for batch setup
    nvalchemiops.jax.neighbors.batch_cell_list.batch_cell_list : Uses this for batch setup
    """
    if batch_idx is None and batch_ptr is None:
        raise ValueError("Either batch_idx or batch_ptr must be provided.")

    if batch_ptr is not None and int(batch_ptr.shape[0]) < 2:
        raise ValueError("batch_ptr must have length at least 2")

    if batch_idx is None:
        num_systems = batch_ptr.shape[0] - 1
        num_atoms_per_system = batch_ptr[1:] - batch_ptr[:-1]
        batch_idx = jnp.repeat(
            jnp.arange(num_systems, dtype=jnp.int32),
            num_atoms_per_system,
        )

    elif batch_ptr is None:
        try:
            num_systems = int(jnp.max(batch_idx)) + 1
        except (
            jax.errors.ConcretizationTypeError,
            jax.errors.TracerIntegerConversionError,
        ):
            raise ValueError(
                "Cannot infer num_systems from batch_idx inside jax.jit. "
                "Please provide batch_ptr explicitly when using jax.jit."
            ) from None
        # Use bincount to compute atoms per system
        num_atoms_per_system = jnp.bincount(
            batch_idx, minlength=num_systems, length=num_systems
        )
        batch_ptr = jnp.zeros(num_systems + 1, dtype=jnp.int32)
        batch_ptr = batch_ptr.at[1:].set(
            jnp.cumsum(num_atoms_per_system, dtype=jnp.int32)
        )

    return batch_idx, batch_ptr


def allocate_cell_list(
    total_atoms: int,
    max_total_cells: int,
    neighbor_search_radius: jax.Array,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Allocate memory tensors for cell list data structures.

    Parameters
    ----------
    total_atoms : int
        Total number of atoms across all systems.
    max_total_cells : int
        Maximum number of cells to allocate.
    neighbor_search_radius : jax.Array, shape (3,) or (num_systems, 3), dtype=int32
        Radius of neighboring cells to search in each dimension.

    Returns
    -------
    cells_per_dimension : jax.Array, shape (3,) or (num_systems, 3), dtype=int32
        Number of cells in x, y, z directions (to be filled by build_cell_list).
    neighbor_search_radius : jax.Array, shape (3,) or (num_systems, 3), dtype=int32
        Radius of neighboring cells to search (passed through for convenience).
    atom_periodic_shifts : jax.Array, shape (total_atoms, 3), dtype=int32
        Periodic boundary crossings for each atom (to be filled by build_cell_list).
    atom_to_cell_mapping : jax.Array, shape (total_atoms, 3), dtype=int32
        3D cell coordinates for each atom (to be filled by build_cell_list).
    atoms_per_cell_count : jax.Array, shape (max_total_cells,), dtype=int32
        Number of atoms in each cell (to be filled by build_cell_list).
    cell_atom_start_indices : jax.Array, shape (max_total_cells,), dtype=int32
        Starting index in cell_atom_list for each cell (to be filled by build_cell_list).
    cell_atom_list : jax.Array, shape (total_atoms,), dtype=int32
        Flattened list of atom indices organized by cell (to be filled by build_cell_list).

    Notes
    -----
    This is a pure JAX utility function with no warp dependencies. It pre-allocates
    all tensors needed for cell list construction, supporting both single-system and
    batched operations based on the shape of neighbor_search_radius.

    See Also
    --------
    nvalchemiops.neighbors.cell_list.build_cell_list : Warp launcher that uses these tensors
    nvalchemiops.jax.neighbors.cell_list.build_cell_list : High-level JAX wrapper
    nvalchemiops.jax.neighbors.batch_cell_list.batch_build_cell_list : Batched version
    """
    if max_total_cells < 0:
        raise ValueError(
            f"allocate_cell_list: max_total_cells={max_total_cells} < 0 "
            "(cell-count overflow or bad estimate)."
        )
    # Detect number of systems from neighbor_search_radius shape
    is_batched = neighbor_search_radius.ndim == 2
    num_systems = neighbor_search_radius.shape[0] if is_batched else 1

    cells_per_dimension = jnp.zeros(
        (3,) if not is_batched else (num_systems, 3),
        dtype=jnp.int32,
    )

    atom_periodic_shifts = jnp.zeros((total_atoms, 3), dtype=jnp.int32)
    atom_to_cell_mapping = jnp.zeros((total_atoms, 3), dtype=jnp.int32)
    atoms_per_cell_count = jnp.zeros((max_total_cells,), dtype=jnp.int32)
    cell_atom_start_indices = jnp.zeros((max_total_cells,), dtype=jnp.int32)
    cell_atom_list = jnp.zeros((total_atoms,), dtype=jnp.int32)
    return (
        cells_per_dimension,
        neighbor_search_radius,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
    )
