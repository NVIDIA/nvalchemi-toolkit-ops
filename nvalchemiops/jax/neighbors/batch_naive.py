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

from nvalchemiops.jax.neighbors._autograd import (
    _build_index_residuals,
    _NeighborForwardOutput,
    _route_pair_outputs,
)
from nvalchemiops.jax.neighbors.neighbor_utils import (
    build_naive_kernel_tables,
    compute_naive_num_shifts,
    get_neighbor_list_from_neighbor_matrix,
    prepare_batch_idx_ptr,
)
from nvalchemiops.neighbors.naive import (
    get_naive_neighbor_matrix_kernel as _get_naive_kernel,
)
from nvalchemiops.neighbors.neighbor_utils import (
    estimate_max_neighbors,
    get_wrap_positions_kernel,
)

_DTYPE_TO_BATCH_NAIVE_KERNELS = (wp.float32, wp.float64)
(
    _fill_batch_naive_neighbor_matrix_kernels,
    _fill_batch_naive_neighbor_matrix_selective_kernels,
    _fill_batch_naive_neighbor_matrix_pbc_kernels,
    _fill_batch_naive_neighbor_matrix_pbc_selective_kernels,
    _fill_batch_naive_neighbor_matrix_pbc_prewrapped_kernels,
    _fill_batch_naive_neighbor_matrix_pbc_prewrapped_selective_kernels,
) = build_naive_kernel_tables(
    "single_cutoff", batched=True, dtypes=_DTYPE_TO_BATCH_NAIVE_KERNELS
)

# Pair-output kernel tables (autograd path).  Same factory with
# return_vectors / return_distances flipped on.
#
# The PBC variant is hard-wired to ``pbc_mode='wrap_on_entry'``; the
# autograd path silently ignores the public ``wrap_positions`` kwarg
# (the kernel is idempotent on already-wrapped positions and correct on
# raw positions).

_fill_batch_naive_pair_kernels = {
    t: _get_naive_kernel(
        t,
        pbc_mode="none",
        batched=True,
        selective=False,
        return_vectors=True,
        return_distances=True,
    )
    for t in _DTYPE_TO_BATCH_NAIVE_KERNELS
}
_fill_batch_naive_pbc_pair_kernels = {
    t: _get_naive_kernel(
        t,
        pbc_mode="wrap_on_entry",
        batched=True,
        selective=False,
        return_vectors=True,
        return_distances=True,
    )
    for t in _DTYPE_TO_BATCH_NAIVE_KERNELS
}


__all__ = ["batch_naive_neighbor_list"]

# ==============================================================================
# JAX Kernel Wrappers
# ==============================================================================

# No-PBC batch naive neighbor matrix kernel wrappers
_jax_fill_batch_naive_f32 = jax_kernel(
    _fill_batch_naive_neighbor_matrix_kernels[wp.float32],
    num_outputs=2,
    in_out_argnames=["neighbor_matrix1", "num_neighbors1"],
    enable_backward=False,
)
_jax_fill_batch_naive_f64 = jax_kernel(
    _fill_batch_naive_neighbor_matrix_kernels[wp.float64],
    num_outputs=2,
    in_out_argnames=["neighbor_matrix1", "num_neighbors1"],
    enable_backward=False,
)

# PBC batch naive neighbor matrix kernel wrappers
_jax_fill_batch_naive_pbc_f32 = jax_kernel(
    _fill_batch_naive_neighbor_matrix_pbc_kernels[wp.float32],
    num_outputs=3,
    in_out_argnames=["neighbor_matrix1", "neighbor_matrix_shifts1", "num_neighbors1"],
    enable_backward=False,
)
_jax_fill_batch_naive_pbc_f64 = jax_kernel(
    _fill_batch_naive_neighbor_matrix_pbc_kernels[wp.float64],
    num_outputs=3,
    in_out_argnames=["neighbor_matrix1", "neighbor_matrix_shifts1", "num_neighbors1"],
    enable_backward=False,
)

# Selective no-PBC batch naive neighbor matrix kernel wrappers
_jax_fill_batch_naive_selective_f32 = jax_kernel(
    _fill_batch_naive_neighbor_matrix_selective_kernels[wp.float32],
    num_outputs=2,
    in_out_argnames=["neighbor_matrix1", "num_neighbors1"],
    enable_backward=False,
)
_jax_fill_batch_naive_selective_f64 = jax_kernel(
    _fill_batch_naive_neighbor_matrix_selective_kernels[wp.float64],
    num_outputs=2,
    in_out_argnames=["neighbor_matrix1", "num_neighbors1"],
    enable_backward=False,
)

# Selective PBC batch naive neighbor matrix kernel wrappers
_jax_fill_batch_naive_pbc_selective_f32 = jax_kernel(
    _fill_batch_naive_neighbor_matrix_pbc_selective_kernels[wp.float32],
    num_outputs=3,
    in_out_argnames=["neighbor_matrix1", "neighbor_matrix_shifts1", "num_neighbors1"],
    enable_backward=False,
)
_jax_fill_batch_naive_pbc_selective_f64 = jax_kernel(
    _fill_batch_naive_neighbor_matrix_pbc_selective_kernels[wp.float64],
    num_outputs=3,
    in_out_argnames=["neighbor_matrix1", "neighbor_matrix_shifts1", "num_neighbors1"],
    enable_backward=False,
)

# Prewrapped PBC batch naive neighbor matrix kernel wrappers
_jax_fill_batch_naive_pbc_prewrapped_f32 = jax_kernel(
    _fill_batch_naive_neighbor_matrix_pbc_prewrapped_kernels[wp.float32],
    num_outputs=3,
    in_out_argnames=["neighbor_matrix1", "neighbor_matrix_shifts1", "num_neighbors1"],
    enable_backward=False,
)
_jax_fill_batch_naive_pbc_prewrapped_f64 = jax_kernel(
    _fill_batch_naive_neighbor_matrix_pbc_prewrapped_kernels[wp.float64],
    num_outputs=3,
    in_out_argnames=["neighbor_matrix1", "neighbor_matrix_shifts1", "num_neighbors1"],
    enable_backward=False,
)
_jax_fill_batch_naive_pbc_prewrapped_selective_f32 = jax_kernel(
    _fill_batch_naive_neighbor_matrix_pbc_prewrapped_selective_kernels[wp.float32],
    num_outputs=3,
    in_out_argnames=["neighbor_matrix1", "neighbor_matrix_shifts1", "num_neighbors1"],
    enable_backward=False,
)
_jax_fill_batch_naive_pbc_prewrapped_selective_f64 = jax_kernel(
    _fill_batch_naive_neighbor_matrix_pbc_prewrapped_selective_kernels[wp.float64],
    num_outputs=3,
    in_out_argnames=["neighbor_matrix1", "neighbor_matrix_shifts1", "num_neighbors1"],
    enable_backward=False,
)

# Pair-output variants (autograd path).
_jax_fill_batch_naive_pair_f32 = jax_kernel(
    _fill_batch_naive_pair_kernels[wp.float32],
    num_outputs=4,
    in_out_argnames=[
        "neighbor_matrix1",
        "num_neighbors1",
        "neighbor_vectors",
        "neighbor_distances",
    ],
    enable_backward=False,
)
_jax_fill_batch_naive_pair_f64 = jax_kernel(
    _fill_batch_naive_pair_kernels[wp.float64],
    num_outputs=4,
    in_out_argnames=[
        "neighbor_matrix1",
        "num_neighbors1",
        "neighbor_vectors",
        "neighbor_distances",
    ],
    enable_backward=False,
)
_jax_fill_batch_naive_pbc_pair_f32 = jax_kernel(
    _fill_batch_naive_pbc_pair_kernels[wp.float32],
    num_outputs=5,
    in_out_argnames=[
        "neighbor_matrix1",
        "neighbor_matrix_shifts1",
        "num_neighbors1",
        "neighbor_vectors",
        "neighbor_distances",
    ],
    enable_backward=False,
)
_jax_fill_batch_naive_pbc_pair_f64 = jax_kernel(
    _fill_batch_naive_pbc_pair_kernels[wp.float64],
    num_outputs=5,
    in_out_argnames=[
        "neighbor_matrix1",
        "neighbor_matrix_shifts1",
        "num_neighbors1",
        "neighbor_vectors",
        "neighbor_distances",
    ],
    enable_backward=False,
)

# Wrap positions batch kernel wrappers
_jax_wrap_positions_batch_f32 = jax_kernel(
    get_wrap_positions_kernel(wp.float32, batched=True),
    num_outputs=2,
    in_out_argnames=["positions_wrapped", "per_atom_cell_offsets"],
    enable_backward=False,
)
_jax_wrap_positions_batch_f64 = jax_kernel(
    get_wrap_positions_kernel(wp.float64, batched=True),
    num_outputs=2,
    in_out_argnames=["positions_wrapped", "per_atom_cell_offsets"],
    enable_backward=False,
)


def _jax_scalar_sentinels(dtype):
    """Return JAX zero-size placeholders for inactive naive scalar inputs."""
    return (
        jnp.empty((0, 3), dtype=jnp.int32),
        jnp.empty((0, 3, 3), dtype=dtype),
        jnp.empty((0, 3), dtype=jnp.int32),
        jnp.empty((0,), dtype=jnp.int32),
        jnp.empty((0,), dtype=jnp.int32),
        jnp.empty((0,), dtype=jnp.int32),
        jnp.empty((0,), dtype=jnp.int32),
        jnp.empty((0, 0), dtype=jnp.int32),
        jnp.empty((0, 0, 3), dtype=jnp.int32),
        jnp.empty((0,), dtype=jnp.int32),
        jnp.empty((0, 0, 3), dtype=dtype),
        jnp.empty((0, 0), dtype=dtype),
        jnp.empty((0, 0), dtype=dtype),
        jnp.empty((0, 0), dtype=dtype),
        jnp.empty((0, 0, 3), dtype=dtype),
        jnp.empty((0,), dtype=jnp.bool_),
    )


def _batch_naive_pair_outputs_forward(
    positions: jax.Array,
    cell: jax.Array | None,
    *,
    pbc: jax.Array | None,
    batch_idx_i32: jax.Array,
    batch_ptr_i32: jax.Array,
    cutoff: float,
    max_neighbors: int,
    fill_value: int,
    max_shifts_per_system: int,
    max_atoms_per_system: int,
    num_systems: int,
) -> _NeighborForwardOutput:
    """Forward closure for the batch_naive autograd path."""
    positions = jax.lax.stop_gradient(positions)
    if cell is not None:
        cell = jax.lax.stop_gradient(cell)
    total_atoms = positions.shape[0]
    f64 = positions.dtype == jnp.float64
    cutoff_sq = float(cutoff * cutoff)
    (
        empty_offsets,
        empty_cell,
        empty_shift_range,
        empty_num_shifts,
        empty_batch_idx,
        empty_batch_ptr,
        empty_target_indices,
        empty_matrix,
        empty_shifts,
        empty_num_neighbors,
        empty_vectors,
        empty_distances,
        empty_pair_params,
        empty_energies,
        empty_forces,
        empty_rebuild_flags,
    ) = _jax_scalar_sentinels(positions.dtype)

    nm = jnp.full((total_atoms, max_neighbors), fill_value, dtype=jnp.int32)
    nn = jnp.zeros(total_atoms, dtype=jnp.int32)
    nv = jnp.zeros((total_atoms, max_neighbors, 3), dtype=positions.dtype)
    nd = jnp.zeros((total_atoms, max_neighbors), dtype=positions.dtype)

    if pbc is None:
        kernel = (
            _jax_fill_batch_naive_pair_f64 if f64 else _jax_fill_batch_naive_pair_f32
        )
        nm, nn, nv, nd = kernel(
            positions,
            empty_offsets,
            cutoff_sq,
            0.0,
            empty_cell,
            empty_shift_range,
            empty_num_shifts,
            batch_idx_i32,
            batch_ptr_i32,
            empty_target_indices,
            nm,
            empty_shifts,
            nn,
            empty_matrix,
            empty_shifts,
            empty_num_neighbors,
            nv,
            nd,
            empty_pair_params,
            empty_energies,
            empty_forces,
            False,  # half_fill
            empty_rebuild_flags,
            launch_dims=(1, 1, total_atoms),
        )
        nms = jnp.zeros((total_atoms, max_neighbors, 3), dtype=jnp.int32)
    else:
        kernel = (
            _jax_fill_batch_naive_pbc_pair_f64
            if f64
            else _jax_fill_batch_naive_pbc_pair_f32
        )
        nms = jnp.zeros((total_atoms, max_neighbors, 3), dtype=jnp.int32)
        shift_range, num_shifts_arr, _ = compute_naive_num_shifts(cell, cutoff, pbc)
        inv_cell = jnp.linalg.inv(cell)
        positions_wrapped = jnp.zeros_like(positions)
        per_atom_cell_offsets = jnp.zeros((total_atoms, 3), dtype=jnp.int32)
        if f64:
            _wrap_kernel = _jax_wrap_positions_batch_f64
        else:
            _wrap_kernel = _jax_wrap_positions_batch_f32
        positions_wrapped, per_atom_cell_offsets = _wrap_kernel(
            positions,
            cell,
            inv_cell,
            batch_idx_i32,
            positions_wrapped,
            per_atom_cell_offsets,
            launch_dims=(total_atoms,),
        )
        nm, nms, nn, nv, nd = kernel(
            positions_wrapped,
            per_atom_cell_offsets,
            cutoff_sq,
            0.0,
            cell,
            shift_range,
            num_shifts_arr,
            batch_idx_i32,
            batch_ptr_i32,
            empty_target_indices,
            nm,
            nms,
            nn,
            empty_matrix,
            empty_shifts,
            empty_num_neighbors,
            nv,
            nd,
            empty_pair_params,
            empty_energies,
            empty_forces,
            False,  # half_fill
            empty_rebuild_flags,
            launch_dims=(
                num_systems,
                max_shifts_per_system,
                max_atoms_per_system,
            ),
        )

    i_idx, j_idx, shifts_ret, _, mask_ = _build_index_residuals(nm, nn, nms)
    K, M = nm.shape
    return _NeighborForwardOutput(
        distances=nd,
        vectors=nv,
        extra_outputs=(nm, nn, nms),
        i_idx=i_idx,
        j_idx=j_idx,
        shifts=shifts_ret,
        batch_idx=batch_idx_i32,
        active_mask=mask_,
        matrix_shape=(K, M),
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
    num_shifts_per_system: jax.Array | None = None,
    max_shifts_per_system: int | None = None,
    max_atoms_per_system: int | None = None,
    rebuild_flags: jax.Array | None = None,
    wrap_positions: bool = True,
    positions_wrapped_buffer: jax.Array | None = None,
    per_atom_cell_offsets_buffer: jax.Array | None = None,
    inv_cell_buffer: jax.Array | None = None,
    *,
    return_distances: bool = False,
    return_vectors: bool = False,
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
        Concatenated Cartesian coordinates for all systems.
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
    num_shifts_per_system : jax.Array, optional
        Number of periodic shifts per system.
    max_shifts_per_system : int, optional
        Maximum per-system shift count (launch dimension).
    max_atoms_per_system : int, optional
        Maximum atoms in any system.
    wrap_positions : bool, default=True
        If True, wrap input positions into the primary cell before
        neighbor search. Set to False when positions are already
        wrapped (e.g. by a preceding integration step) to save two
        GPU kernel launches per call.

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

    has_pair_outputs = bool(return_distances) or bool(return_vectors)
    if has_pair_outputs:
        if half_fill or rebuild_flags is not None or return_neighbor_list:
            raise NotImplementedError(
                "return_distances / return_vectors on the JAX batch_naive "
                "binding require half_fill=False, no rebuild_flags, and "
                "return_neighbor_list=False.",
            )
        if max_neighbors is None:
            max_neighbors = estimate_max_neighbors(cutoff)
        if fill_value is None:
            fill_value = positions.shape[0]
        cell_norm = cell
        if cell_norm is not None:
            cell_norm = (
                cell_norm if cell_norm.ndim == 3 else cell_norm[jnp.newaxis, :, :]
            )
            if cell_norm.dtype != positions.dtype:
                cell_norm = cell_norm.astype(positions.dtype)
        pbc_norm = pbc
        if pbc_norm is not None:
            pbc_norm = pbc_norm if pbc_norm.ndim == 2 else pbc_norm[jnp.newaxis, :]
        batch_idx_i32 = batch_idx.astype(jnp.int32)
        batch_ptr_i32 = batch_ptr.astype(jnp.int32)
        if pbc_norm is not None:
            if max_shifts_per_system is None or num_shifts_per_system is None:
                _, _, max_shifts_per_system = compute_naive_num_shifts(
                    jax.lax.stop_gradient(cell_norm), cutoff, pbc_norm
                )
            if max_atoms_per_system is None:
                try:
                    max_atoms_per_system = int(jnp.max(batch_ptr[1:] - batch_ptr[:-1]))
                except (
                    jax.errors.ConcretizationTypeError,
                    jax.errors.TracerIntegerConversionError,
                ):
                    raise ValueError(
                        "max_atoms_per_system must be passed explicitly when "
                        "calling batch_naive_neighbor_list under jax.jit with "
                        "return_distances / return_vectors set.  The autograd "
                        "path needs a concrete launch dimension and cannot "
                        "infer it from a traced batch_ptr."
                    ) from None
        else:
            max_shifts_per_system = 1
            max_atoms_per_system = positions.shape[0]

        forward_kwargs = {
            "pbc": pbc_norm,
            "batch_idx_i32": batch_idx_i32,
            "batch_ptr_i32": batch_ptr_i32,
            "cutoff": float(cutoff),
            "max_neighbors": int(max_neighbors),
            "fill_value": int(fill_value),
            "max_shifts_per_system": int(max_shifts_per_system),
            "max_atoms_per_system": int(max_atoms_per_system),
            "num_systems": int(num_systems),
        }
        distances_out, vectors_out, nm_out, nn_out, shifts_out = _route_pair_outputs(
            positions,
            cell_norm,
            _batch_naive_pair_outputs_forward,
            forward_kwargs,
        )
        if pbc is not None:
            base = (nm_out, nn_out, shifts_out)
        else:
            base = (nm_out, nn_out)
        if return_distances and return_vectors:
            return (*base, distances_out, vectors_out)
        if return_distances:
            return (*base, distances_out)
        return (*base, vectors_out)

    if cell is not None:
        cell = cell if cell.ndim == 3 else cell[jnp.newaxis, :, :]
        # Ensure cell dtype matches positions dtype so Warp kernel dispatch is consistent
        if cell.dtype != positions.dtype:
            cell = cell.astype(positions.dtype)
    if pbc is not None:
        pbc = pbc if pbc.ndim == 2 else pbc[jnp.newaxis, :]

    if max_neighbors is None:
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
        _jax_fill_selective = _jax_fill_batch_naive_selective_f64
        _jax_fill_pbc_selective = _jax_fill_batch_naive_pbc_selective_f64
        _jax_fill_pbc_prewrapped = _jax_fill_batch_naive_pbc_prewrapped_f64
        _jax_fill_pbc_prewrapped_selective = (
            _jax_fill_batch_naive_pbc_prewrapped_selective_f64
        )
        _jax_wrap_batch = _jax_wrap_positions_batch_f64
    else:
        _jax_fill = _jax_fill_batch_naive_f32
        _jax_fill_pbc = _jax_fill_batch_naive_pbc_f32
        _jax_fill_selective = _jax_fill_batch_naive_selective_f32
        _jax_fill_pbc_selective = _jax_fill_batch_naive_pbc_selective_f32
        _jax_fill_pbc_prewrapped = _jax_fill_batch_naive_pbc_prewrapped_f32
        _jax_fill_pbc_prewrapped_selective = (
            _jax_fill_batch_naive_pbc_prewrapped_selective_f32
        )
        _jax_wrap_batch = _jax_wrap_positions_batch_f32
        positions = positions.astype(jnp.float32)

    total_atoms = positions.shape[0]

    batch_idx_i32 = batch_idx.astype(jnp.int32)
    batch_ptr_i32 = batch_ptr.astype(jnp.int32)
    (
        empty_offsets,
        empty_cell,
        empty_shift_range,
        empty_num_shifts,
        empty_batch_idx,
        empty_batch_ptr,
        empty_target_indices,
        empty_matrix,
        empty_shifts,
        empty_num_neighbors,
        empty_vectors,
        empty_distances,
        empty_pair_params,
        empty_energies,
        empty_forces,
        empty_rebuild_flags,
    ) = _jax_scalar_sentinels(positions.dtype)

    if pbc is None:
        # No PBC case
        if rebuild_flags is not None:
            rf = rebuild_flags.astype(jnp.bool_)
            atom_rebuild = rf[batch_idx_i32]
            num_neighbors = jnp.where(
                atom_rebuild, jnp.zeros_like(num_neighbors), num_neighbors
            )
            neighbor_matrix, num_neighbors = _jax_fill_selective(
                positions,
                empty_offsets,
                float(cutoff * cutoff),
                0.0,
                empty_cell,
                empty_shift_range,
                empty_num_shifts,
                batch_idx_i32,
                batch_ptr_i32,
                empty_target_indices,
                neighbor_matrix,
                empty_shifts,
                num_neighbors,
                empty_matrix,
                empty_shifts,
                empty_num_neighbors,
                empty_vectors,
                empty_distances,
                empty_pair_params,
                empty_energies,
                empty_forces,
                half_fill,
                rf,
                launch_dims=(1, 1, total_atoms),
            )
        else:
            neighbor_matrix, num_neighbors = _jax_fill(
                positions,
                empty_offsets,
                float(cutoff * cutoff),
                0.0,
                empty_cell,
                empty_shift_range,
                empty_num_shifts,
                batch_idx_i32,
                batch_ptr_i32,
                empty_target_indices,
                neighbor_matrix,
                empty_shifts,
                num_neighbors,
                empty_matrix,
                empty_shifts,
                empty_num_neighbors,
                empty_vectors,
                empty_distances,
                empty_pair_params,
                empty_energies,
                empty_forces,
                half_fill,
                empty_rebuild_flags,
                launch_dims=(1, 1, total_atoms),
            )
    else:
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

        if wrap_positions:
            inv_cell = (
                inv_cell_buffer if inv_cell_buffer is not None else jnp.linalg.inv(cell)
            )
            positions_wrapped = (
                positions_wrapped_buffer
                if positions_wrapped_buffer is not None
                else jnp.zeros_like(positions)
            )
            per_atom_cell_offsets = (
                per_atom_cell_offsets_buffer
                if per_atom_cell_offsets_buffer is not None
                else jnp.zeros((total_atoms, 3), dtype=jnp.int32)
            )
            positions_wrapped, per_atom_cell_offsets = _jax_wrap_batch(
                positions,
                cell,
                inv_cell,
                batch_idx_i32,
                positions_wrapped,
                per_atom_cell_offsets,
                launch_dims=(total_atoms,),
            )

            if rebuild_flags is not None:
                rf = rebuild_flags.astype(jnp.bool_)
                atom_rebuild = rf[batch_idx_i32]
                num_neighbors = jnp.where(
                    atom_rebuild, jnp.zeros_like(num_neighbors), num_neighbors
                )
                neighbor_matrix, neighbor_matrix_shifts, num_neighbors = (
                    _jax_fill_pbc_selective(
                        positions_wrapped,
                        per_atom_cell_offsets,
                        float(cutoff * cutoff),
                        0.0,
                        cell,
                        shift_range_per_dimension,
                        num_shifts_per_system,
                        batch_idx_i32,
                        batch_ptr_i32,
                        empty_target_indices,
                        neighbor_matrix,
                        neighbor_matrix_shifts,
                        num_neighbors,
                        empty_matrix,
                        empty_shifts,
                        empty_num_neighbors,
                        empty_vectors,
                        empty_distances,
                        empty_pair_params,
                        empty_energies,
                        empty_forces,
                        half_fill,
                        rf,
                        launch_dims=(
                            num_systems,
                            max_shifts_per_system,
                            max_atoms_per_system,
                        ),
                    )
                )
            else:
                neighbor_matrix, neighbor_matrix_shifts, num_neighbors = _jax_fill_pbc(
                    positions_wrapped,
                    per_atom_cell_offsets,
                    float(cutoff * cutoff),
                    0.0,
                    cell,
                    shift_range_per_dimension,
                    num_shifts_per_system,
                    batch_idx_i32,
                    batch_ptr_i32,
                    empty_target_indices,
                    neighbor_matrix,
                    neighbor_matrix_shifts,
                    num_neighbors,
                    empty_matrix,
                    empty_shifts,
                    empty_num_neighbors,
                    empty_vectors,
                    empty_distances,
                    empty_pair_params,
                    empty_energies,
                    empty_forces,
                    half_fill,
                    empty_rebuild_flags,
                    launch_dims=(
                        num_systems,
                        max_shifts_per_system,
                        max_atoms_per_system,
                    ),
                )
        else:
            if rebuild_flags is not None:
                rf = rebuild_flags.astype(jnp.bool_)
                atom_rebuild = rf[batch_idx_i32]
                num_neighbors = jnp.where(
                    atom_rebuild, jnp.zeros_like(num_neighbors), num_neighbors
                )
                neighbor_matrix, neighbor_matrix_shifts, num_neighbors = (
                    _jax_fill_pbc_prewrapped_selective(
                        positions,
                        empty_offsets,
                        float(cutoff * cutoff),
                        0.0,
                        cell,
                        shift_range_per_dimension,
                        num_shifts_per_system,
                        batch_idx_i32,
                        batch_ptr_i32,
                        empty_target_indices,
                        neighbor_matrix,
                        neighbor_matrix_shifts,
                        num_neighbors,
                        empty_matrix,
                        empty_shifts,
                        empty_num_neighbors,
                        empty_vectors,
                        empty_distances,
                        empty_pair_params,
                        empty_energies,
                        empty_forces,
                        half_fill,
                        rf,
                        launch_dims=(
                            num_systems,
                            max_shifts_per_system,
                            max_atoms_per_system,
                        ),
                    )
                )
            else:
                neighbor_matrix, neighbor_matrix_shifts, num_neighbors = (
                    _jax_fill_pbc_prewrapped(
                        positions,
                        empty_offsets,
                        float(cutoff * cutoff),
                        0.0,
                        cell,
                        shift_range_per_dimension,
                        num_shifts_per_system,
                        batch_idx_i32,
                        batch_ptr_i32,
                        empty_target_indices,
                        neighbor_matrix,
                        neighbor_matrix_shifts,
                        num_neighbors,
                        empty_matrix,
                        empty_shifts,
                        empty_num_neighbors,
                        empty_vectors,
                        empty_distances,
                        empty_pair_params,
                        empty_energies,
                        empty_forces,
                        half_fill,
                        empty_rebuild_flags,
                        launch_dims=(
                            num_systems,
                            max_shifts_per_system,
                            max_atoms_per_system,
                        ),
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
