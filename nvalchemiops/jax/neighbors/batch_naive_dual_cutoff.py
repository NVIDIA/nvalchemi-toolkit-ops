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

"""JAX bindings for batched naive O(N^2) dual cutoff neighbor list construction."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import warp as wp
from warp.jax_experimental import jax_kernel

from nvalchemiops.jax.neighbors.neighbor_utils import (
    build_naive_kernel_tables,
    compute_naive_num_shifts,
    get_neighbor_list_from_neighbor_matrix,
    prepare_batch_idx_ptr,
)
from nvalchemiops.neighbors.neighbor_utils import (
    estimate_max_neighbors,
    get_wrap_positions_kernel,
)

_DTYPE_TO_BATCH_NAIVE_DUAL_KERNELS = (wp.float32, wp.float64)
(
    _fill_batch_naive_neighbor_matrix_dual_cutoff_kernels,
    _fill_batch_naive_neighbor_matrix_dual_cutoff_selective_kernels,
    _fill_batch_naive_neighbor_matrix_pbc_dual_cutoff_kernels,
    _fill_batch_naive_neighbor_matrix_pbc_dual_cutoff_selective_kernels,
    _fill_batch_naive_neighbor_matrix_pbc_dual_cutoff_prewrapped_kernels,
    _fill_batch_naive_neighbor_matrix_pbc_dual_cutoff_prewrapped_selective_kernels,
) = build_naive_kernel_tables(
    "dual_cutoff", batched=True, dtypes=_DTYPE_TO_BATCH_NAIVE_DUAL_KERNELS
)

__all__ = ["batch_naive_neighbor_list_dual_cutoff"]

# ==============================================================================
# JAX Kernel Wrappers
# ==============================================================================

# No-PBC batch naive dual cutoff neighbor matrix kernel wrappers
_jax_fill_batch_dual_f32 = jax_kernel(
    _fill_batch_naive_neighbor_matrix_dual_cutoff_kernels[wp.float32],
    num_outputs=4,
    in_out_argnames=[
        "neighbor_matrix1",
        "num_neighbors1",
        "neighbor_matrix2",
        "num_neighbors2",
    ],
    enable_backward=False,
)
_jax_fill_batch_dual_f64 = jax_kernel(
    _fill_batch_naive_neighbor_matrix_dual_cutoff_kernels[wp.float64],
    num_outputs=4,
    in_out_argnames=[
        "neighbor_matrix1",
        "num_neighbors1",
        "neighbor_matrix2",
        "num_neighbors2",
    ],
    enable_backward=False,
)

# PBC batch naive dual cutoff neighbor matrix kernel wrappers
_jax_fill_batch_dual_pbc_f32 = jax_kernel(
    _fill_batch_naive_neighbor_matrix_pbc_dual_cutoff_kernels[wp.float32],
    num_outputs=6,
    in_out_argnames=[
        "neighbor_matrix1",
        "neighbor_matrix_shifts1",
        "num_neighbors1",
        "neighbor_matrix2",
        "neighbor_matrix_shifts2",
        "num_neighbors2",
    ],
    enable_backward=False,
)
_jax_fill_batch_dual_pbc_f64 = jax_kernel(
    _fill_batch_naive_neighbor_matrix_pbc_dual_cutoff_kernels[wp.float64],
    num_outputs=6,
    in_out_argnames=[
        "neighbor_matrix1",
        "neighbor_matrix_shifts1",
        "num_neighbors1",
        "neighbor_matrix2",
        "neighbor_matrix_shifts2",
        "num_neighbors2",
    ],
    enable_backward=False,
)

# Selective batch dual cutoff neighbor matrix kernel wrappers
_jax_fill_batch_dual_selective_f32 = jax_kernel(
    _fill_batch_naive_neighbor_matrix_dual_cutoff_selective_kernels[wp.float32],
    num_outputs=4,
    in_out_argnames=[
        "neighbor_matrix1",
        "num_neighbors1",
        "neighbor_matrix2",
        "num_neighbors2",
    ],
    enable_backward=False,
)
_jax_fill_batch_dual_selective_f64 = jax_kernel(
    _fill_batch_naive_neighbor_matrix_dual_cutoff_selective_kernels[wp.float64],
    num_outputs=4,
    in_out_argnames=[
        "neighbor_matrix1",
        "num_neighbors1",
        "neighbor_matrix2",
        "num_neighbors2",
    ],
    enable_backward=False,
)

# Selective PBC batch dual cutoff neighbor matrix kernel wrappers
_jax_fill_batch_dual_pbc_selective_f32 = jax_kernel(
    _fill_batch_naive_neighbor_matrix_pbc_dual_cutoff_selective_kernels[wp.float32],
    num_outputs=6,
    in_out_argnames=[
        "neighbor_matrix1",
        "neighbor_matrix_shifts1",
        "num_neighbors1",
        "neighbor_matrix2",
        "neighbor_matrix_shifts2",
        "num_neighbors2",
    ],
    enable_backward=False,
)
_jax_fill_batch_dual_pbc_selective_f64 = jax_kernel(
    _fill_batch_naive_neighbor_matrix_pbc_dual_cutoff_selective_kernels[wp.float64],
    num_outputs=6,
    in_out_argnames=[
        "neighbor_matrix1",
        "neighbor_matrix_shifts1",
        "num_neighbors1",
        "neighbor_matrix2",
        "neighbor_matrix_shifts2",
        "num_neighbors2",
    ],
    enable_backward=False,
)

# Prewrapped PBC batch dual cutoff neighbor matrix kernel wrappers
_jax_fill_batch_dual_pbc_prewrapped_f32 = jax_kernel(
    _fill_batch_naive_neighbor_matrix_pbc_dual_cutoff_prewrapped_kernels[wp.float32],
    num_outputs=6,
    in_out_argnames=[
        "neighbor_matrix1",
        "neighbor_matrix_shifts1",
        "num_neighbors1",
        "neighbor_matrix2",
        "neighbor_matrix_shifts2",
        "num_neighbors2",
    ],
    enable_backward=False,
)
_jax_fill_batch_dual_pbc_prewrapped_f64 = jax_kernel(
    _fill_batch_naive_neighbor_matrix_pbc_dual_cutoff_prewrapped_kernels[wp.float64],
    num_outputs=6,
    in_out_argnames=[
        "neighbor_matrix1",
        "neighbor_matrix_shifts1",
        "num_neighbors1",
        "neighbor_matrix2",
        "neighbor_matrix_shifts2",
        "num_neighbors2",
    ],
    enable_backward=False,
)
_jax_fill_batch_dual_pbc_prewrapped_selective_f32 = jax_kernel(
    _fill_batch_naive_neighbor_matrix_pbc_dual_cutoff_prewrapped_selective_kernels[
        wp.float32
    ],
    num_outputs=6,
    in_out_argnames=[
        "neighbor_matrix1",
        "neighbor_matrix_shifts1",
        "num_neighbors1",
        "neighbor_matrix2",
        "neighbor_matrix_shifts2",
        "num_neighbors2",
    ],
    enable_backward=False,
)
_jax_fill_batch_dual_pbc_prewrapped_selective_f64 = jax_kernel(
    _fill_batch_naive_neighbor_matrix_pbc_dual_cutoff_prewrapped_selective_kernels[
        wp.float64
    ],
    num_outputs=6,
    in_out_argnames=[
        "neighbor_matrix1",
        "neighbor_matrix_shifts1",
        "num_neighbors1",
        "neighbor_matrix2",
        "neighbor_matrix_shifts2",
        "num_neighbors2",
    ],
    enable_backward=False,
)

# Wrap positions batch kernel wrappers
_jax_wrap_positions_batch_f32 = jax_kernel(
    get_wrap_positions_kernel(wp.float32, batched=True, pbc_aware=True),
    num_outputs=2,
    in_out_argnames=["positions_wrapped", "per_atom_cell_offsets"],
    enable_backward=False,
)
_jax_wrap_positions_batch_f64 = jax_kernel(
    get_wrap_positions_kernel(wp.float64, batched=True, pbc_aware=True),
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


def batch_naive_neighbor_list_dual_cutoff(
    positions: jax.Array,
    cutoff1: float,
    cutoff2: float,
    batch_idx: jax.Array | None = None,
    batch_ptr: jax.Array | None = None,
    pbc: jax.Array | None = None,
    cell: jax.Array | None = None,
    max_neighbors1: int | None = None,
    max_neighbors2: int | None = None,
    half_fill: bool = False,
    fill_value: int | None = None,
    return_neighbor_list: bool = False,
    neighbor_matrix1: jax.Array | None = None,
    neighbor_matrix2: jax.Array | None = None,
    neighbor_matrix_shifts1: jax.Array | None = None,
    neighbor_matrix_shifts2: jax.Array | None = None,
    num_neighbors1: jax.Array | None = None,
    num_neighbors2: jax.Array | None = None,
    shift_range_per_dimension: jax.Array | None = None,
    num_shifts_per_system: jax.Array | None = None,
    max_shifts_per_system: int | None = None,
    max_atoms_per_system: int | None = None,
    rebuild_flags: jax.Array | None = None,
    wrap_positions: bool = True,
    positions_wrapped_buffer: jax.Array | None = None,
    per_atom_cell_offsets_buffer: jax.Array | None = None,
    inv_cell_buffer: jax.Array | None = None,
) -> (
    tuple[
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
    ]
    | tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]
    | tuple[jax.Array, jax.Array, jax.Array, jax.Array]
):
    """Compute batched neighbor lists for two cutoff distances using naive O(N^2) algorithm.

    This function builds two neighbor matrices simultaneously for different cutoff
    distances in a batched manner, which is more efficient than calling the
    single-cutoff function twice.

    Parameters
    ----------
    positions : jax.Array, shape (total_atoms, 3), dtype=float32 or float64
        Concatenated Cartesian coordinates for all systems.
    cutoff1 : float
        First cutoff distance (typically smaller).
    cutoff2 : float
        Second cutoff distance (typically larger).
    batch_idx : jax.Array, shape (total_atoms,), dtype=int32, optional
        System index for each atom.
    batch_ptr : jax.Array, shape (num_systems + 1,), dtype=int32, optional
        Cumulative atom counts defining system boundaries.
    pbc : jax.Array, shape (num_systems, 3) or (1, 3), dtype=bool, optional
        Periodic boundary condition flags for each dimension.
    cell : jax.Array, shape (num_systems, 3, 3) or (1, 3, 3), dtype=float32 or float64, optional
        Cell matrices defining lattice vectors in Cartesian coordinates.
    max_neighbors1 : int, optional
        Maximum number of neighbors per atom for cutoff1.
    max_neighbors2 : int, optional
        Maximum number of neighbors per atom for cutoff2.
    half_fill : bool, optional - default = False
        If True, only store relationships where i < j to avoid double counting.
    fill_value : int, optional
        Value to use for padding in neighbor matrices. Default is total_atoms.
    return_neighbor_list : bool, optional - default = False
        If True, convert neighbor matrices to neighbor list (idx_i, idx_j) format.
    neighbor_matrix1 : jax.Array, shape (total_atoms, max_neighbors1), dtype=int32, optional
        Pre-allocated first neighbor matrix.
    neighbor_matrix2 : jax.Array, shape (total_atoms, max_neighbors2), dtype=int32, optional
        Pre-allocated second neighbor matrix.
    neighbor_matrix_shifts1 : jax.Array, shape (total_atoms, max_neighbors1, 3), dtype=int32, optional
        Pre-allocated first shift matrix for PBC.
    neighbor_matrix_shifts2 : jax.Array, shape (total_atoms, max_neighbors2, 3), dtype=int32, optional
        Pre-allocated second shift matrix for PBC.
    num_neighbors1 : jax.Array, shape (total_atoms,), dtype=int32, optional
        Pre-allocated first neighbor count array.
    num_neighbors2 : jax.Array, shape (total_atoms,), dtype=int32, optional
        Pre-allocated second neighbor count array.
    shift_range_per_dimension : jax.Array, shape (num_systems, 3), dtype=int32, optional
        Pre-computed shift ranges for PBC.
    num_shifts_per_system : jax.Array, shape (num_systems,), dtype=int32, optional
        Number of periodic shifts per system.
    max_shifts_per_system : int, optional
        Maximum per-system shift count (launch dimension).
    max_atoms_per_system : int, optional
        Maximum number of atoms in any system (for PBC batched dispatch).
    wrap_positions : bool, default=True
        If True, wrap input positions into the primary cell before
        neighbor search. Set to False when positions are already
        wrapped (e.g. by a preceding integration step) to save two
        GPU kernel launches per call.

    Returns
    -------
    results : tuple of jax.Array
        Variable-length tuple depending on input parameters:

        - No PBC, matrix format: ``(neighbor_matrix1, num_neighbors1, neighbor_matrix2, num_neighbors2)``
        - No PBC, list format: ``(neighbor_list1, neighbor_ptr1, neighbor_list2, neighbor_ptr2)``
        - With PBC, matrix format: ``(neighbor_matrix1, num_neighbors1, neighbor_matrix_shifts1, neighbor_matrix2, num_neighbors2, neighbor_matrix_shifts2)``
        - With PBC, list format: ``(neighbor_list1, neighbor_ptr1, unit_shifts1, neighbor_list2, neighbor_ptr2, unit_shifts2)``

    See Also
    --------
    nvalchemiops.neighbors.batch_naive_dual_cutoff.batch_naive_neighbor_matrix_dual_cutoff : Core warp launcher (no PBC)
    nvalchemiops.neighbors.batch_naive_dual_cutoff.batch_naive_neighbor_matrix_pbc_dual_cutoff : Core warp launcher (with PBC)
    batch_naive_neighbor_list : Single cutoff version
    """
    if pbc is None and cell is not None:
        raise ValueError("If cell is provided, pbc must also be provided")
    if pbc is not None and cell is None:
        raise ValueError("If pbc is provided, cell must also be provided")

    # Prepare batch_idx and batch_ptr
    batch_idx, batch_ptr = prepare_batch_idx_ptr(
        batch_idx, batch_ptr, positions.shape[0]
    )

    if cell is not None:
        cell = cell if cell.ndim == 3 else cell[jnp.newaxis, :, :]
        # Ensure cell dtype matches positions dtype so Warp kernel dispatch is consistent
        if cell.dtype != positions.dtype:
            cell = cell.astype(positions.dtype)
    if pbc is not None:
        pbc = pbc if pbc.ndim == 2 else pbc[jnp.newaxis, :]

    # Estimate max_neighbors if not provided - use larger cutoff for estimation
    if max_neighbors1 is None and (
        neighbor_matrix1 is None
        or (neighbor_matrix_shifts1 is None and pbc is not None)
        or num_neighbors1 is None
    ):
        max_neighbors1 = estimate_max_neighbors(cutoff2)  # Use larger cutoff
    if max_neighbors2 is None and (
        neighbor_matrix2 is None
        or (neighbor_matrix_shifts2 is None and pbc is not None)
        or num_neighbors2 is None
    ):
        max_neighbors2 = estimate_max_neighbors(cutoff2)  # Use larger cutoff

    if fill_value is None:
        fill_value = jnp.int32(positions.shape[0])

    # Allocate first neighbor matrix
    if neighbor_matrix1 is None:
        neighbor_matrix1 = jnp.full(
            (positions.shape[0], max_neighbors1),
            fill_value,
            dtype=jnp.int32,
        )
    elif rebuild_flags is None:
        neighbor_matrix1 = neighbor_matrix1.at[:].set(fill_value)

    # Allocate second neighbor matrix
    if neighbor_matrix2 is None:
        neighbor_matrix2 = jnp.full(
            (positions.shape[0], max_neighbors2),
            fill_value,
            dtype=jnp.int32,
        )
    elif rebuild_flags is None:
        neighbor_matrix2 = neighbor_matrix2.at[:].set(fill_value)

    # Allocate first num_neighbors
    if num_neighbors1 is None:
        num_neighbors1 = jnp.zeros(positions.shape[0], dtype=jnp.int32)
    elif rebuild_flags is None:
        num_neighbors1 = num_neighbors1.at[:].set(jnp.int32(0))

    # Allocate second num_neighbors
    if num_neighbors2 is None:
        num_neighbors2 = jnp.zeros(positions.shape[0], dtype=jnp.int32)
    elif rebuild_flags is None:
        num_neighbors2 = num_neighbors2.at[:].set(jnp.int32(0))

    if pbc is not None:
        # Allocate shift matrices
        if neighbor_matrix_shifts1 is None:
            neighbor_matrix_shifts1 = jnp.zeros(
                (positions.shape[0], max_neighbors1, 3),
                dtype=jnp.int32,
            )
        elif rebuild_flags is None:
            neighbor_matrix_shifts1 = neighbor_matrix_shifts1.at[:].set(jnp.int32(0))

        if neighbor_matrix_shifts2 is None:
            neighbor_matrix_shifts2 = jnp.zeros(
                (positions.shape[0], max_neighbors2, 3),
                dtype=jnp.int32,
            )
        elif rebuild_flags is None:
            neighbor_matrix_shifts2 = neighbor_matrix_shifts2.at[:].set(jnp.int32(0))

        if (
            max_shifts_per_system is None
            or num_shifts_per_system is None
            or shift_range_per_dimension is None
        ):
            shift_range_per_dimension, num_shifts_per_system, max_shifts_per_system = (
                compute_naive_num_shifts(cell, cutoff2, pbc)  # Use larger cutoff
            )

        # Compute max_atoms_per_system if needed
        if max_atoms_per_system is None:
            try:
                atoms_per_system = batch_ptr[1:] - batch_ptr[:-1]
                max_atoms_per_system = int(jnp.max(atoms_per_system))
            except (
                jax.errors.ConcretizationTypeError,
                jax.errors.TracerIntegerConversionError,
            ):
                raise ValueError(
                    "Cannot infer max_atoms_per_system inside jax.jit. "
                    "Please provide max_atoms_per_system explicitly when using jax.jit."
                ) from None

    if cutoff1 <= 0 and cutoff2 <= 0:
        if return_neighbor_list:
            if pbc is not None:
                return (
                    jnp.zeros((2, 0), dtype=jnp.int32),
                    jnp.zeros((positions.shape[0] + 1,), dtype=jnp.int32),
                    jnp.zeros((0, 3), dtype=jnp.int32),
                    jnp.zeros((2, 0), dtype=jnp.int32),
                    jnp.zeros((positions.shape[0] + 1,), dtype=jnp.int32),
                    jnp.zeros((0, 3), dtype=jnp.int32),
                )
            else:
                return (
                    jnp.zeros((2, 0), dtype=jnp.int32),
                    jnp.zeros((positions.shape[0] + 1,), dtype=jnp.int32),
                    jnp.zeros((2, 0), dtype=jnp.int32),
                    jnp.zeros((positions.shape[0] + 1,), dtype=jnp.int32),
                )
        else:
            if pbc is not None:
                return (
                    neighbor_matrix1,
                    num_neighbors1,
                    neighbor_matrix_shifts1,
                    neighbor_matrix2,
                    num_neighbors2,
                    neighbor_matrix_shifts2,
                )
            else:
                return (
                    neighbor_matrix1,
                    num_neighbors1,
                    neighbor_matrix2,
                    num_neighbors2,
                )

    # Select kernel based on dtype
    if positions.dtype == jnp.float64:
        _jax_fill = _jax_fill_batch_dual_f64
        _jax_fill_pbc = _jax_fill_batch_dual_pbc_f64
        _jax_fill_selective = _jax_fill_batch_dual_selective_f64
        _jax_fill_pbc_selective = _jax_fill_batch_dual_pbc_selective_f64
        _jax_fill_pbc_prewrapped = _jax_fill_batch_dual_pbc_prewrapped_f64
        _jax_fill_pbc_prewrapped_selective = (
            _jax_fill_batch_dual_pbc_prewrapped_selective_f64
        )
        _jax_wrap_batch = _jax_wrap_positions_batch_f64
    else:
        _jax_fill = _jax_fill_batch_dual_f32
        _jax_fill_pbc = _jax_fill_batch_dual_pbc_f32
        _jax_fill_selective = _jax_fill_batch_dual_selective_f32
        _jax_fill_pbc_selective = _jax_fill_batch_dual_pbc_selective_f32
        _jax_fill_pbc_prewrapped = _jax_fill_batch_dual_pbc_prewrapped_f32
        _jax_fill_pbc_prewrapped_selective = (
            _jax_fill_batch_dual_pbc_prewrapped_selective_f32
        )
        _jax_wrap_batch = _jax_wrap_positions_batch_f32
        positions = positions.astype(jnp.float32)

    total_atoms = positions.shape[0]
    num_systems = batch_ptr.shape[0] - 1
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
        if rebuild_flags is not None:
            rf = rebuild_flags.astype(jnp.bool_)
            atom_rebuild = rf[batch_idx_i32]
            num_neighbors1 = jnp.where(
                atom_rebuild, jnp.zeros_like(num_neighbors1), num_neighbors1
            )
            num_neighbors2 = jnp.where(
                atom_rebuild, jnp.zeros_like(num_neighbors2), num_neighbors2
            )
            neighbor_matrix1, num_neighbors1, neighbor_matrix2, num_neighbors2 = (
                _jax_fill_selective(
                    positions,
                    empty_offsets,
                    float(cutoff1 * cutoff1),
                    float(cutoff2 * cutoff2),
                    empty_cell,
                    empty_shift_range,
                    empty_num_shifts,
                    batch_idx_i32,
                    batch_ptr_i32,
                    empty_target_indices,
                    neighbor_matrix1,
                    empty_shifts,
                    num_neighbors1,
                    neighbor_matrix2,
                    empty_shifts,
                    num_neighbors2,
                    empty_vectors,
                    empty_distances,
                    empty_pair_params,
                    empty_energies,
                    empty_forces,
                    rf,
                    launch_dims=(1, 1, total_atoms),
                )
            )
        else:
            neighbor_matrix1, num_neighbors1, neighbor_matrix2, num_neighbors2 = (
                _jax_fill(
                    positions,
                    empty_offsets,
                    float(cutoff1 * cutoff1),
                    float(cutoff2 * cutoff2),
                    empty_cell,
                    empty_shift_range,
                    empty_num_shifts,
                    batch_idx_i32,
                    batch_ptr_i32,
                    empty_target_indices,
                    neighbor_matrix1,
                    empty_shifts,
                    num_neighbors1,
                    neighbor_matrix2,
                    empty_shifts,
                    num_neighbors2,
                    empty_vectors,
                    empty_distances,
                    empty_pair_params,
                    empty_energies,
                    empty_forces,
                    empty_rebuild_flags,
                    launch_dims=(1, 1, total_atoms),
                )
            )
    else:
        if cell.dtype != positions.dtype:
            cell = cell.astype(positions.dtype)

        if max_atoms_per_system is None:
            try:
                atoms_per_system = batch_ptr[1:] - batch_ptr[:-1]
                max_atoms_per_system = int(jnp.max(atoms_per_system))
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
                pbc,
                batch_idx_i32,
                positions_wrapped,
                per_atom_cell_offsets,
                launch_dims=(total_atoms,),
            )

            if rebuild_flags is not None:
                rf = rebuild_flags.astype(jnp.bool_)
                atom_rebuild = rf[batch_idx_i32]
                num_neighbors1 = jnp.where(
                    atom_rebuild, jnp.zeros_like(num_neighbors1), num_neighbors1
                )
                num_neighbors2 = jnp.where(
                    atom_rebuild, jnp.zeros_like(num_neighbors2), num_neighbors2
                )
                (
                    neighbor_matrix1,
                    neighbor_matrix_shifts1,
                    num_neighbors1,
                    neighbor_matrix2,
                    neighbor_matrix_shifts2,
                    num_neighbors2,
                ) = _jax_fill_pbc_selective(
                    positions_wrapped,
                    per_atom_cell_offsets,
                    float(cutoff1 * cutoff1),
                    float(cutoff2 * cutoff2),
                    cell,
                    shift_range_per_dimension,
                    num_shifts_per_system,
                    batch_idx_i32,
                    batch_ptr_i32,
                    empty_target_indices,
                    neighbor_matrix1,
                    neighbor_matrix_shifts1,
                    num_neighbors1,
                    neighbor_matrix2,
                    neighbor_matrix_shifts2,
                    num_neighbors2,
                    empty_vectors,
                    empty_distances,
                    empty_pair_params,
                    empty_energies,
                    empty_forces,
                    rf,
                    launch_dims=(
                        num_systems,
                        max_shifts_per_system,
                        max_atoms_per_system,
                    ),
                )
            else:
                (
                    neighbor_matrix1,
                    neighbor_matrix_shifts1,
                    num_neighbors1,
                    neighbor_matrix2,
                    neighbor_matrix_shifts2,
                    num_neighbors2,
                ) = _jax_fill_pbc(
                    positions_wrapped,
                    per_atom_cell_offsets,
                    float(cutoff1 * cutoff1),
                    float(cutoff2 * cutoff2),
                    cell,
                    shift_range_per_dimension,
                    num_shifts_per_system,
                    batch_idx_i32,
                    batch_ptr_i32,
                    empty_target_indices,
                    neighbor_matrix1,
                    neighbor_matrix_shifts1,
                    num_neighbors1,
                    neighbor_matrix2,
                    neighbor_matrix_shifts2,
                    num_neighbors2,
                    empty_vectors,
                    empty_distances,
                    empty_pair_params,
                    empty_energies,
                    empty_forces,
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
                num_neighbors1 = jnp.where(
                    atom_rebuild, jnp.zeros_like(num_neighbors1), num_neighbors1
                )
                num_neighbors2 = jnp.where(
                    atom_rebuild, jnp.zeros_like(num_neighbors2), num_neighbors2
                )
                (
                    neighbor_matrix1,
                    neighbor_matrix_shifts1,
                    num_neighbors1,
                    neighbor_matrix2,
                    neighbor_matrix_shifts2,
                    num_neighbors2,
                ) = _jax_fill_pbc_prewrapped_selective(
                    positions,
                    empty_offsets,
                    float(cutoff1 * cutoff1),
                    float(cutoff2 * cutoff2),
                    cell,
                    shift_range_per_dimension,
                    num_shifts_per_system,
                    batch_idx_i32,
                    batch_ptr_i32,
                    empty_target_indices,
                    neighbor_matrix1,
                    neighbor_matrix_shifts1,
                    num_neighbors1,
                    neighbor_matrix2,
                    neighbor_matrix_shifts2,
                    num_neighbors2,
                    empty_vectors,
                    empty_distances,
                    empty_pair_params,
                    empty_energies,
                    empty_forces,
                    rf,
                    launch_dims=(
                        num_systems,
                        max_shifts_per_system,
                        max_atoms_per_system,
                    ),
                )
            else:
                (
                    neighbor_matrix1,
                    neighbor_matrix_shifts1,
                    num_neighbors1,
                    neighbor_matrix2,
                    neighbor_matrix_shifts2,
                    num_neighbors2,
                ) = _jax_fill_pbc_prewrapped(
                    positions,
                    empty_offsets,
                    float(cutoff1 * cutoff1),
                    float(cutoff2 * cutoff2),
                    cell,
                    shift_range_per_dimension,
                    num_shifts_per_system,
                    batch_idx_i32,
                    batch_ptr_i32,
                    empty_target_indices,
                    neighbor_matrix1,
                    neighbor_matrix_shifts1,
                    num_neighbors1,
                    neighbor_matrix2,
                    neighbor_matrix_shifts2,
                    num_neighbors2,
                    empty_vectors,
                    empty_distances,
                    empty_pair_params,
                    empty_energies,
                    empty_forces,
                    empty_rebuild_flags,
                    launch_dims=(
                        num_systems,
                        max_shifts_per_system,
                        max_atoms_per_system,
                    ),
                )

    if return_neighbor_list:
        if pbc is not None:
            neighbor_list1, neighbor_ptr1, neighbor_list_shifts1 = (
                get_neighbor_list_from_neighbor_matrix(
                    neighbor_matrix1,
                    num_neighbors=num_neighbors1,
                    neighbor_shift_matrix=neighbor_matrix_shifts1,
                    fill_value=fill_value,
                )
            )
            neighbor_list2, neighbor_ptr2, neighbor_list_shifts2 = (
                get_neighbor_list_from_neighbor_matrix(
                    neighbor_matrix2,
                    num_neighbors=num_neighbors2,
                    neighbor_shift_matrix=neighbor_matrix_shifts2,
                    fill_value=fill_value,
                )
            )
            return (
                neighbor_list1,
                neighbor_ptr1,
                neighbor_list_shifts1,
                neighbor_list2,
                neighbor_ptr2,
                neighbor_list_shifts2,
            )
        else:
            neighbor_list1, neighbor_ptr1 = get_neighbor_list_from_neighbor_matrix(
                neighbor_matrix1,
                num_neighbors=num_neighbors1,
                fill_value=fill_value,
            )
            neighbor_list2, neighbor_ptr2 = get_neighbor_list_from_neighbor_matrix(
                neighbor_matrix2,
                num_neighbors=num_neighbors2,
                fill_value=fill_value,
            )
            return neighbor_list1, neighbor_ptr1, neighbor_list2, neighbor_ptr2
    else:
        if pbc is not None:
            return (
                neighbor_matrix1,
                num_neighbors1,
                neighbor_matrix_shifts1,
                neighbor_matrix2,
                num_neighbors2,
                neighbor_matrix_shifts2,
            )
        else:
            return neighbor_matrix1, num_neighbors1, neighbor_matrix2, num_neighbors2
