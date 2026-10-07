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

import functools
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import warp as wp
from warp import JaxCallableGraphMode, jax_callable, jax_kernel

from nvalchemiops.jax.neighbors._autograd import (
    _build_index_residuals,
    _NeighborForwardOutput,
    _route_pair_outputs,
)
from nvalchemiops.jax.neighbors._dispatch import (
    _jax_array_device_kind,
)
from nvalchemiops.jax.neighbors._registration import (
    _lazy_cell_inverse_kernel,
    _lazy_naive_kernel,
)
from nvalchemiops.jax.neighbors.neighbor_utils import (
    _pack_fixed_capacity_neighbor_list_from_neighbor_matrix,
    _validate_coo_capacity,
    _validate_graph_mode,
    build_naive_kernel_tables,
    compute_naive_num_shifts,
    coo_pack_pair_geometry,
    get_fixed_capacity_neighbor_list_from_neighbor_matrix,
    get_neighbor_list_from_neighbor_matrix,
)
from nvalchemiops.neighbors.naive.dispatch import (
    _NaiveWorkload,
    _require_cuda_tile_device,
    _resolve_naive_strategy,
)
from nvalchemiops.neighbors.naive.launchers import (
    _launch_naive_neighbor_matrix_no_pbc,
    _launch_naive_neighbor_matrix_pbc,
)
from nvalchemiops.neighbors.neighbor_utils import (
    DTYPE_INFO_ALL,
    empty_sentinel,
    estimate_max_neighbors,
    get_wrap_positions_kernel,
    resolve_buffer_alias,
    selective_zero_num_neighbors_single,
)

_DTYPE_TO_NAIVE_KERNELS = (wp.float32, wp.float64)
_CELL_INVERSE_REGISTRATION = _lazy_cell_inverse_kernel()


def _prepare_wrapped_tile_operands(
    positions: jax.Array,
    cell: jax.Array,
    *,
    inv_cell_buffer: jax.Array | None,
    positions_wrapped_buffer: jax.Array | None,
    per_atom_cell_offsets_buffer: jax.Array | None,
    fixed_cell_geometry: tuple[jax.Array | None, ...] | None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Validate and prepare wrapped-PBC tile operands for one system.

    Parameters
    ----------
    positions : jax.Array, shape (N, 3)
        Positions whose wrapped coordinates and cell offsets are needed.
    cell : jax.Array, shape (1, 3, 3)
        Runtime cell used to compute an inverse when no fixed inverse is cached.
    inv_cell_buffer : jax.Array, shape matching ``cell``, optional
        Caller-supplied inverse scratch with dtype matching ``positions`` for
        the default-cell path.
    positions_wrapped_buffer : jax.Array, shape (N, 3), optional
        Caller-supplied wrapped-position output operand with dtype matching
        ``positions``.
    per_atom_cell_offsets_buffer : jax.Array, shape (N, 3), dtype=int32, optional
        Caller-supplied integer offset output operand.
    fixed_cell_geometry : tuple or None
        Prepared fixed-cell metadata. Its first element is the cached inverse
        when that cache is present.

    Returns
    -------
    inverse_cell : jax.Array, shape matching ``cell``
        Warp-precision inverse for the tile callback. A prepared cached inverse
        is returned read-only; otherwise this is the supplied or allocated
        inverse scratch.
    wrapped_positions : jax.Array, shape (N, 3)
        Supplied or allocated position scratch, with dtype matching ``positions``.
    cell_offsets : jax.Array, shape (N, 3), dtype=int32
        Supplied or allocated per-atom wrapping-offset scratch.

    Raises
    ------
    ValueError
        If a supplied scratch buffer has an incompatible shape or dtype.

    Notes
    -----
    A cached fixed-cell inverse is reused directly. Otherwise, the registered
    Warp inverse kernel computes the inverse of the current cell into the
    supplied buffer or newly allocated scratch. Wrapped positions and offsets
    are returned as callback operands so each execution uses current positions.
    """
    if inv_cell_buffer is not None:
        if tuple(inv_cell_buffer.shape) != tuple(cell.shape):
            raise ValueError(
                f"inv_cell_buffer must have shape {tuple(cell.shape)}; "
                f"got {tuple(inv_cell_buffer.shape)}."
            )
        if inv_cell_buffer.dtype != positions.dtype:
            raise ValueError(
                f"inv_cell_buffer dtype must match positions dtype "
                f"({positions.dtype}); got {inv_cell_buffer.dtype}."
            )
    if positions_wrapped_buffer is not None:
        expected_shape = (positions.shape[0], 3)
        if tuple(positions_wrapped_buffer.shape) != expected_shape:
            raise ValueError(
                f"positions_wrapped_buffer must have shape {expected_shape}; "
                f"got {tuple(positions_wrapped_buffer.shape)}."
            )
        if positions_wrapped_buffer.dtype != positions.dtype:
            raise ValueError(
                f"positions_wrapped_buffer dtype must match positions dtype "
                f"({positions.dtype}); got {positions_wrapped_buffer.dtype}."
            )
    if per_atom_cell_offsets_buffer is not None:
        expected_shape = (positions.shape[0], 3)
        if tuple(per_atom_cell_offsets_buffer.shape) != expected_shape:
            raise ValueError(
                f"per_atom_cell_offsets_buffer must have shape {expected_shape}; "
                f"got {tuple(per_atom_cell_offsets_buffer.shape)}."
            )
        if per_atom_cell_offsets_buffer.dtype != jnp.int32:
            raise ValueError(
                "per_atom_cell_offsets_buffer dtype must be int32; "
                f"got {per_atom_cell_offsets_buffer.dtype}."
            )
    fixed_inverse = None if fixed_cell_geometry is None else fixed_cell_geometry[0]
    if fixed_inverse is not None:
        inverse = fixed_inverse
    else:
        inverse = (
            inv_cell_buffer if inv_cell_buffer is not None else jnp.empty_like(cell)
        )
        (inverse,) = _CELL_INVERSE_REGISTRATION[positions.dtype](
            cell,
            inverse,
            launch_dims=(1,),
        )
    positions_wrapped = (
        positions_wrapped_buffer
        if positions_wrapped_buffer is not None
        else jnp.empty_like(positions)
    )
    per_atom_cell_offsets = (
        per_atom_cell_offsets_buffer
        if per_atom_cell_offsets_buffer is not None
        else jnp.empty((positions.shape[0], 3), dtype=jnp.int32)
    )
    return inverse, positions_wrapped, per_atom_cell_offsets


(
    _fill_naive_neighbor_matrix_kernels,
    _fill_naive_neighbor_matrix_selective_kernels,
    _fill_naive_neighbor_matrix_pbc_kernels,
    _fill_naive_neighbor_matrix_pbc_selective_kernels,
    _fill_naive_neighbor_matrix_pbc_prewrapped_kernels,
    _fill_naive_neighbor_matrix_pbc_prewrapped_selective_kernels,
) = build_naive_kernel_tables(
    "single_cutoff", batched=False, dtypes=_DTYPE_TO_NAIVE_KERNELS
)

(
    _fill_naive_neighbor_matrix_half_kernels,
    _fill_naive_neighbor_matrix_selective_half_kernels,
    _fill_naive_neighbor_matrix_pbc_half_kernels,
    _fill_naive_neighbor_matrix_pbc_selective_half_kernels,
    _fill_naive_neighbor_matrix_pbc_prewrapped_half_kernels,
    _fill_naive_neighbor_matrix_pbc_prewrapped_selective_half_kernels,
) = build_naive_kernel_tables(
    "single_cutoff",
    batched=False,
    dtypes=_DTYPE_TO_NAIVE_KERNELS,
    half_fill=True,
)

# Direct jax_kernel registrations are constructed lazily.  The retained
# build_naive_kernel_tables tables below remain for JaxCallableGraphMode.WARP callbacks.


_DIRECT_NAIVE_KERNELS = {
    (pbc_mode, selective, half_fill): _lazy_naive_kernel(
        operation="single_cutoff",
        batched=False,
        pbc_mode=pbc_mode,
        selective=selective,
        half_fill=half_fill,
    )
    for pbc_mode in ("none", "wrap_on_entry", "prewrapped")
    for selective in (False, True)
    for half_fill in (False, True)
}
_DIRECT_NAIVE_GEOMETRY_KERNELS = {
    (pbc_mode, half_fill): _lazy_naive_kernel(
        operation="single_cutoff",
        batched=False,
        pbc_mode=pbc_mode,
        half_fill=half_fill,
        geometry=True,
    )
    for pbc_mode in ("none", "wrap_on_entry")
    for half_fill in (False, True)
}


@functools.cache
def _get_jax_naive_pair_kernel(
    wp_dtype, pbc_mode: str, half_fill: bool = False, partial: bool = False
):
    """Return a cached direct geometry registration for optional partial rows."""
    jax_dtype = jnp.float64 if wp_dtype == wp.float64 else jnp.float32
    return _lazy_naive_kernel(
        operation="single_cutoff",
        batched=False,
        pbc_mode=pbc_mode,
        selective=False,
        partial=partial,
        half_fill=half_fill,
        geometry=True,
        pair_fn=None,
    )[jax_dtype]


@functools.cache
def _get_jax_naive_pair_fn_kernel(
    pair_fn,
    wp_dtype,
    pbc_mode: str,
    half_fill: bool = False,
    partial: bool = False,
):
    """Return a cached pair-function direct naive registration."""
    jax_dtype = jnp.float64 if wp_dtype == wp.float64 else jnp.float32
    return _lazy_naive_kernel(
        operation="single_cutoff",
        batched=False,
        pbc_mode=pbc_mode,
        selective=False,
        partial=partial,
        half_fill=half_fill,
        geometry=True,
        pair_fn=pair_fn,
    )[jax_dtype]


__all__ = ["naive_neighbor_list"]

# ==============================================================================
# JAX Kernel Wrappers
# ==============================================================================


# Wrap positions single kernel wrappers
_jax_wrap_positions_single_f32 = jax_kernel(
    get_wrap_positions_kernel(wp.float32, pbc_aware=True),
    num_outputs=2,
    in_out_argnames=["positions_wrapped", "per_atom_cell_offsets"],
    enable_backward=False,
)
_jax_wrap_positions_single_f64 = jax_kernel(
    get_wrap_positions_kernel(wp.float64, pbc_aware=True),
    num_outputs=2,
    in_out_argnames=["positions_wrapped", "per_atom_cell_offsets"],
    enable_backward=False,
)


def _reset_graph_neighbor_outputs(
    neighbor_matrix,
    num_neighbors,
    fill_value,
    neighbor_matrix_shifts=None,
) -> None:
    """Reset neighbor outputs inside the Warp callback to keep buffers stable."""
    neighbor_matrix.fill_(fill_value)
    num_neighbors.zero_()
    if neighbor_matrix_shifts is not None:
        neighbor_matrix_shifts.zero_()


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


def _wp_scalar_sentinels(wp_dtype: type, device):
    """Return Warp zero-size placeholders for inactive naive scalar inputs."""
    vec_dtype, mat_dtype = DTYPE_INFO_ALL[wp_dtype]
    return (
        empty_sentinel(1, wp.vec3i, device),
        empty_sentinel(1, mat_dtype, device),
        empty_sentinel(1, wp.vec3i, device),
        empty_sentinel(1, wp.int32, device),
        empty_sentinel(1, wp.int32, device),
        empty_sentinel(1, wp.int32, device),
        empty_sentinel(1, wp.int32, device),
        empty_sentinel(2, wp.int32, device),
        empty_sentinel(2, wp.vec3i, device),
        empty_sentinel(1, wp.int32, device),
        empty_sentinel(2, vec_dtype, device),
        empty_sentinel(2, wp_dtype, device),
        empty_sentinel(2, wp_dtype, device),
        empty_sentinel(2, wp_dtype, device),
        empty_sentinel(2, vec_dtype, device),
        empty_sentinel(1, wp.bool, device),
    )


def _run_graph_naive_no_pbc(
    positions,
    neighbor_matrix,
    num_neighbors,
    cutoff_sq,
    fill_value,
    half_fill,
    wp_dtype,
    fill_kernel,
    selective_kernel=None,
    rebuild_flags=None,
) -> None:
    """Execute the no-PBC graph-mode body."""
    total_atoms = positions.shape[0]
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
    ) = _wp_scalar_sentinels(wp_dtype, num_neighbors.device)
    if rebuild_flags is None:
        _reset_graph_neighbor_outputs(neighbor_matrix, num_neighbors, fill_value)
        active_kernel = fill_kernel
        rebuild_flags_arg = empty_rebuild_flags
    else:
        selective_zero_num_neighbors_single(
            num_neighbors, rebuild_flags, str(num_neighbors.device)
        )
        active_kernel = selective_kernel
        rebuild_flags_arg = rebuild_flags
    wp.launch(
        kernel=active_kernel,
        dim=(1, 1, total_atoms),
        inputs=[
            positions,
            empty_offsets,
            cutoff_sq,
            wp_dtype(0.0),
            empty_cell,
            empty_shift_range,
            empty_num_shifts,
            empty_batch_idx,
            empty_batch_ptr,
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
            rebuild_flags_arg,
        ],
    )


def _run_graph_naive_pbc_prewrapped(
    positions,
    cell,
    shift_range,
    neighbor_matrix,
    neighbor_matrix_shifts,
    num_neighbors,
    cutoff_sq,
    num_shifts,
    fill_value,
    half_fill,
    wp_dtype,
    fill_kernel,
    selective_kernel=None,
    rebuild_flags=None,
) -> None:
    """Execute the prewrapped-PBC graph-mode body."""
    launch_dims = (1, num_shifts, positions.shape[0])
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
    ) = _wp_scalar_sentinels(wp_dtype, num_neighbors.device)
    if rebuild_flags is None:
        _reset_graph_neighbor_outputs(
            neighbor_matrix,
            num_neighbors,
            fill_value,
            neighbor_matrix_shifts,
        )
        active_kernel = fill_kernel
        rebuild_flags_arg = empty_rebuild_flags
    else:
        selective_zero_num_neighbors_single(
            num_neighbors, rebuild_flags, str(num_neighbors.device)
        )
        active_kernel = selective_kernel
        rebuild_flags_arg = rebuild_flags

    wp.launch(
        kernel=active_kernel,
        dim=launch_dims,
        inputs=[
            positions,
            empty_offsets,
            cutoff_sq,
            wp_dtype(0.0),
            cell,
            shift_range,
            empty_num_shifts,
            empty_batch_idx,
            empty_batch_ptr,
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
            rebuild_flags_arg,
        ],
    )


def _run_graph_naive_pbc_wrapped(
    positions,
    cell,
    inv_cell,
    pbc,
    shift_range,
    positions_wrapped,
    per_atom_cell_offsets,
    neighbor_matrix,
    neighbor_matrix_shifts,
    num_neighbors,
    cutoff_sq,
    num_shifts,
    fill_value,
    half_fill,
    wp_dtype,
    wrap_kernel,
    fill_kernel,
    selective_kernel=None,
    rebuild_flags=None,
) -> None:
    """Execute the wrapped-PBC graph-mode body."""
    total_atoms = positions.shape[0]
    launch_dims = (1, num_shifts, total_atoms)
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
    ) = _wp_scalar_sentinels(wp_dtype, num_neighbors.device)
    if rebuild_flags is None:
        _reset_graph_neighbor_outputs(
            neighbor_matrix,
            num_neighbors,
            fill_value,
            neighbor_matrix_shifts,
        )
        active_kernel = fill_kernel
        rebuild_flags_arg = empty_rebuild_flags
    else:
        selective_zero_num_neighbors_single(
            num_neighbors, rebuild_flags, str(num_neighbors.device)
        )
        active_kernel = selective_kernel
        rebuild_flags_arg = rebuild_flags

    wp.launch(
        kernel=wrap_kernel,
        dim=total_atoms,
        inputs=[positions, cell, inv_cell, pbc, wp.empty((0,), dtype=wp.int32)],
        outputs=[positions_wrapped, per_atom_cell_offsets],
    )

    wp.launch(
        kernel=active_kernel,
        dim=launch_dims,
        inputs=[
            positions_wrapped,
            per_atom_cell_offsets,
            cutoff_sq,
            wp_dtype(0.0),
            cell,
            shift_range,
            empty_num_shifts,
            empty_batch_idx,
            empty_batch_ptr,
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
            rebuild_flags_arg,
        ],
    )


def _graph_naive_no_pbc_f32(
    positions: wp.array(dtype=wp.vec3f),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    cutoff_sq: wp.float32,
    fill_value: wp.int32,
    half_fill: wp.bool,
) -> None:
    _run_graph_naive_no_pbc(
        positions,
        neighbor_matrix,
        num_neighbors,
        cutoff_sq,
        fill_value,
        half_fill,
        wp.float32,
        _fill_naive_neighbor_matrix_kernels[wp.float32],
    )


def _graph_naive_no_pbc_f64(
    positions: wp.array(dtype=wp.vec3d),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    cutoff_sq: wp.float64,
    fill_value: wp.int32,
    half_fill: wp.bool,
) -> None:
    _run_graph_naive_no_pbc(
        positions,
        neighbor_matrix,
        num_neighbors,
        cutoff_sq,
        fill_value,
        half_fill,
        wp.float64,
        _fill_naive_neighbor_matrix_kernels[wp.float64],
    )


def _graph_naive_no_pbc_selective_f32(
    positions: wp.array(dtype=wp.vec3f),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    cutoff_sq: wp.float32,
    fill_value: wp.int32,
    half_fill: wp.bool,
    rebuild_flags: wp.array(dtype=wp.bool),
) -> None:
    _run_graph_naive_no_pbc(
        positions,
        neighbor_matrix,
        num_neighbors,
        cutoff_sq,
        fill_value,
        half_fill,
        wp.float32,
        _fill_naive_neighbor_matrix_kernels[wp.float32],
        selective_kernel=_fill_naive_neighbor_matrix_selective_kernels[wp.float32],
        rebuild_flags=rebuild_flags,
    )


def _graph_naive_no_pbc_selective_f64(
    positions: wp.array(dtype=wp.vec3d),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    cutoff_sq: wp.float64,
    fill_value: wp.int32,
    half_fill: wp.bool,
    rebuild_flags: wp.array(dtype=wp.bool),
) -> None:
    _run_graph_naive_no_pbc(
        positions,
        neighbor_matrix,
        num_neighbors,
        cutoff_sq,
        fill_value,
        half_fill,
        wp.float64,
        _fill_naive_neighbor_matrix_kernels[wp.float64],
        selective_kernel=_fill_naive_neighbor_matrix_selective_kernels[wp.float64],
        rebuild_flags=rebuild_flags,
    )


def _graph_naive_pbc_prewrapped_f32(
    positions: wp.array(dtype=wp.vec3f),
    cell: wp.array(dtype=wp.mat33f),
    shift_range: wp.array(dtype=wp.vec3i),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    cutoff_sq: wp.float32,
    num_shifts: wp.int32,
    fill_value: wp.int32,
    half_fill: wp.bool,
) -> None:
    _run_graph_naive_pbc_prewrapped(
        positions,
        cell,
        shift_range,
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
        cutoff_sq,
        num_shifts,
        fill_value,
        half_fill,
        wp.float32,
        _fill_naive_neighbor_matrix_pbc_prewrapped_kernels[wp.float32],
    )


def _graph_naive_pbc_prewrapped_f64(
    positions: wp.array(dtype=wp.vec3d),
    cell: wp.array(dtype=wp.mat33d),
    shift_range: wp.array(dtype=wp.vec3i),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    cutoff_sq: wp.float64,
    num_shifts: wp.int32,
    fill_value: wp.int32,
    half_fill: wp.bool,
) -> None:
    _run_graph_naive_pbc_prewrapped(
        positions,
        cell,
        shift_range,
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
        cutoff_sq,
        num_shifts,
        fill_value,
        half_fill,
        wp.float64,
        _fill_naive_neighbor_matrix_pbc_prewrapped_kernels[wp.float64],
    )


def _graph_naive_pbc_prewrapped_selective_f32(
    positions: wp.array(dtype=wp.vec3f),
    cell: wp.array(dtype=wp.mat33f),
    shift_range: wp.array(dtype=wp.vec3i),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    cutoff_sq: wp.float32,
    num_shifts: wp.int32,
    fill_value: wp.int32,
    half_fill: wp.bool,
    rebuild_flags: wp.array(dtype=wp.bool),
) -> None:
    _run_graph_naive_pbc_prewrapped(
        positions,
        cell,
        shift_range,
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
        cutoff_sq,
        num_shifts,
        fill_value,
        half_fill,
        wp.float32,
        _fill_naive_neighbor_matrix_pbc_prewrapped_kernels[wp.float32],
        selective_kernel=_fill_naive_neighbor_matrix_pbc_prewrapped_selective_kernels[
            wp.float32
        ],
        rebuild_flags=rebuild_flags,
    )


def _graph_naive_pbc_prewrapped_selective_f64(
    positions: wp.array(dtype=wp.vec3d),
    cell: wp.array(dtype=wp.mat33d),
    shift_range: wp.array(dtype=wp.vec3i),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    cutoff_sq: wp.float64,
    num_shifts: wp.int32,
    fill_value: wp.int32,
    half_fill: wp.bool,
    rebuild_flags: wp.array(dtype=wp.bool),
) -> None:
    _run_graph_naive_pbc_prewrapped(
        positions,
        cell,
        shift_range,
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
        cutoff_sq,
        num_shifts,
        fill_value,
        half_fill,
        wp.float64,
        _fill_naive_neighbor_matrix_pbc_prewrapped_kernels[wp.float64],
        selective_kernel=_fill_naive_neighbor_matrix_pbc_prewrapped_selective_kernels[
            wp.float64
        ],
        rebuild_flags=rebuild_flags,
    )


def _graph_naive_pbc_wrapped_f32(
    positions: wp.array(dtype=wp.vec3f),
    cell: wp.array(dtype=wp.mat33f),
    inv_cell: wp.array(dtype=wp.mat33f),
    pbc: wp.array2d(dtype=wp.bool),
    shift_range: wp.array(dtype=wp.vec3i),
    positions_wrapped: wp.array(dtype=wp.vec3f),
    per_atom_cell_offsets: wp.array(dtype=wp.vec3i),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    cutoff_sq: wp.float32,
    num_shifts: wp.int32,
    fill_value: wp.int32,
    half_fill: wp.bool,
) -> None:
    _run_graph_naive_pbc_wrapped(
        positions,
        cell,
        inv_cell,
        pbc,
        shift_range,
        positions_wrapped,
        per_atom_cell_offsets,
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
        cutoff_sq,
        num_shifts,
        fill_value,
        half_fill,
        wp.float32,
        get_wrap_positions_kernel(wp.float32, pbc_aware=True),
        _fill_naive_neighbor_matrix_pbc_kernels[wp.float32],
    )


def _graph_naive_pbc_wrapped_f64(
    positions: wp.array(dtype=wp.vec3d),
    cell: wp.array(dtype=wp.mat33d),
    inv_cell: wp.array(dtype=wp.mat33d),
    pbc: wp.array2d(dtype=wp.bool),
    shift_range: wp.array(dtype=wp.vec3i),
    positions_wrapped: wp.array(dtype=wp.vec3d),
    per_atom_cell_offsets: wp.array(dtype=wp.vec3i),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    cutoff_sq: wp.float64,
    num_shifts: wp.int32,
    fill_value: wp.int32,
    half_fill: wp.bool,
) -> None:
    _run_graph_naive_pbc_wrapped(
        positions,
        cell,
        inv_cell,
        pbc,
        shift_range,
        positions_wrapped,
        per_atom_cell_offsets,
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
        cutoff_sq,
        num_shifts,
        fill_value,
        half_fill,
        wp.float64,
        get_wrap_positions_kernel(wp.float64, pbc_aware=True),
        _fill_naive_neighbor_matrix_pbc_kernels[wp.float64],
    )


def _graph_naive_pbc_wrapped_selective_f32(
    positions: wp.array(dtype=wp.vec3f),
    cell: wp.array(dtype=wp.mat33f),
    inv_cell: wp.array(dtype=wp.mat33f),
    pbc: wp.array2d(dtype=wp.bool),
    shift_range: wp.array(dtype=wp.vec3i),
    positions_wrapped: wp.array(dtype=wp.vec3f),
    per_atom_cell_offsets: wp.array(dtype=wp.vec3i),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    cutoff_sq: wp.float32,
    num_shifts: wp.int32,
    fill_value: wp.int32,
    half_fill: wp.bool,
    rebuild_flags: wp.array(dtype=wp.bool),
) -> None:
    _run_graph_naive_pbc_wrapped(
        positions,
        cell,
        inv_cell,
        pbc,
        shift_range,
        positions_wrapped,
        per_atom_cell_offsets,
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
        cutoff_sq,
        num_shifts,
        fill_value,
        half_fill,
        wp.float32,
        get_wrap_positions_kernel(wp.float32, pbc_aware=True),
        _fill_naive_neighbor_matrix_pbc_kernels[wp.float32],
        selective_kernel=_fill_naive_neighbor_matrix_pbc_selective_kernels[wp.float32],
        rebuild_flags=rebuild_flags,
    )


def _graph_naive_pbc_wrapped_selective_f64(
    positions: wp.array(dtype=wp.vec3d),
    cell: wp.array(dtype=wp.mat33d),
    inv_cell: wp.array(dtype=wp.mat33d),
    pbc: wp.array2d(dtype=wp.bool),
    shift_range: wp.array(dtype=wp.vec3i),
    positions_wrapped: wp.array(dtype=wp.vec3d),
    per_atom_cell_offsets: wp.array(dtype=wp.vec3i),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    cutoff_sq: wp.float64,
    num_shifts: wp.int32,
    fill_value: wp.int32,
    half_fill: wp.bool,
    rebuild_flags: wp.array(dtype=wp.bool),
) -> None:
    _run_graph_naive_pbc_wrapped(
        positions,
        cell,
        inv_cell,
        pbc,
        shift_range,
        positions_wrapped,
        per_atom_cell_offsets,
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
        cutoff_sq,
        num_shifts,
        fill_value,
        half_fill,
        wp.float64,
        get_wrap_positions_kernel(wp.float64, pbc_aware=True),
        _fill_naive_neighbor_matrix_pbc_kernels[wp.float64],
        selective_kernel=_fill_naive_neighbor_matrix_pbc_selective_kernels[wp.float64],
        rebuild_flags=rebuild_flags,
    )


_GRAPH_NAIVE_NO_PBC_IN_OUT_ARGS = ("neighbor_matrix", "num_neighbors")
_GRAPH_NAIVE_PBC_IN_OUT_ARGS = (
    "neighbor_matrix",
    "neighbor_matrix_shifts",
    "num_neighbors",
)
_GRAPH_NAIVE_PBC_WRAPPED_IN_OUT_ARGS = (
    "positions_wrapped",
    "per_atom_cell_offsets",
    "neighbor_matrix",
    "neighbor_matrix_shifts",
    "num_neighbors",
)
_GRAPH_NAIVE_DTYPE_TO_WARP_CALLABLES = {
    (False, False): {
        "num_outputs": 2,
        "in_out_argnames": _GRAPH_NAIVE_NO_PBC_IN_OUT_ARGS,
        jnp.dtype(jnp.float32): _graph_naive_no_pbc_f32,
        jnp.dtype(jnp.float64): _graph_naive_no_pbc_f64,
    },
    (False, True): {
        "num_outputs": 2,
        "in_out_argnames": _GRAPH_NAIVE_NO_PBC_IN_OUT_ARGS,
        jnp.dtype(jnp.float32): _graph_naive_no_pbc_selective_f32,
        jnp.dtype(jnp.float64): _graph_naive_no_pbc_selective_f64,
    },
    (True, False, False): {
        "num_outputs": 3,
        "in_out_argnames": _GRAPH_NAIVE_PBC_IN_OUT_ARGS,
        jnp.dtype(jnp.float32): _graph_naive_pbc_prewrapped_f32,
        jnp.dtype(jnp.float64): _graph_naive_pbc_prewrapped_f64,
    },
    (True, False, True): {
        "num_outputs": 3,
        "in_out_argnames": _GRAPH_NAIVE_PBC_IN_OUT_ARGS,
        jnp.dtype(jnp.float32): _graph_naive_pbc_prewrapped_selective_f32,
        jnp.dtype(jnp.float64): _graph_naive_pbc_prewrapped_selective_f64,
    },
    (True, True, False): {
        "num_outputs": 5,
        "in_out_argnames": _GRAPH_NAIVE_PBC_WRAPPED_IN_OUT_ARGS,
        jnp.dtype(jnp.float32): _graph_naive_pbc_wrapped_f32,
        jnp.dtype(jnp.float64): _graph_naive_pbc_wrapped_f64,
    },
    (True, True, True): {
        "num_outputs": 5,
        "in_out_argnames": _GRAPH_NAIVE_PBC_WRAPPED_IN_OUT_ARGS,
        jnp.dtype(jnp.float32): _graph_naive_pbc_wrapped_selective_f32,
        jnp.dtype(jnp.float64): _graph_naive_pbc_wrapped_selective_f64,
    },
}


def _register_graph_naive_callables() -> dict[
    tuple[bool, bool, bool, jnp.dtype], object
]:
    """Register JaxCallableGraphMode.WARP callables for all naive graph-mode paths."""
    registered: dict[tuple[bool, bool, bool, jnp.dtype], object] = {}

    for key, spec in _GRAPH_NAIVE_DTYPE_TO_WARP_CALLABLES.items():
        if len(key) == 2:
            has_pbc, selective = key
            wrap_positions_values = (False, True)
        else:
            has_pbc, wrap_positions, selective = key
            wrap_positions_values = (wrap_positions,)

        for dtype in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
            callable_obj = jax_callable(
                spec[dtype],
                num_outputs=spec["num_outputs"],
                in_out_argnames=spec["in_out_argnames"],
                graph_mode=JaxCallableGraphMode.WARP,
            )
            for wrap_positions in wrap_positions_values:
                registered[(has_pbc, wrap_positions, selective, dtype)] = callable_obj

    return registered


_GRAPH_NAIVE_WARP_CALLABLES = _register_graph_naive_callables()


# ==============================================================================
# Tiled-kernel callables (``strategy="tile"``, CUDA-only)
# ==============================================================================
#
# These force ``strategy="tile"`` in the inner Warp launchers. Eager calls
# pre-fill topology outputs.
# Cutoff, ``half_fill``, ``partial``, and PBC shift count are static.


def _graph_naive_tile_no_pbc_f32(
    positions: wp.array(dtype=wp.vec3f),
    target_indices: wp.array(dtype=wp.int32),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    cutoff: wp.float32,
    half_fill: wp.bool,
    partial: wp.bool,
) -> None:
    _launch_naive_neighbor_matrix_no_pbc(
        positions,
        float(cutoff),
        neighbor_matrix,
        num_neighbors,
        wp.float32,
        str(positions.device),
        batched=False,
        half_fill=bool(half_fill),
        target_indices=target_indices if bool(partial) else None,
        strategy="tile",
    )


def _graph_naive_tile_no_pbc_f64(
    positions: wp.array(dtype=wp.vec3d),
    target_indices: wp.array(dtype=wp.int32),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    cutoff: wp.float64,
    half_fill: wp.bool,
    partial: wp.bool,
) -> None:
    _launch_naive_neighbor_matrix_no_pbc(
        positions,
        float(cutoff),
        neighbor_matrix,
        num_neighbors,
        wp.float64,
        str(positions.device),
        batched=False,
        half_fill=bool(half_fill),
        target_indices=target_indices if bool(partial) else None,
        strategy="tile",
    )


# Prewrapped PBC kernels consume precomputed shift ranges; ``pbc`` only affects
# position wrapping, so these callables intentionally omit it.
def _graph_naive_tile_pbc_prewrapped_f32(
    positions: wp.array(dtype=wp.vec3f),
    target_indices: wp.array(dtype=wp.int32),
    cell: wp.array(dtype=wp.mat33f),
    shift_range: wp.array(dtype=wp.vec3i),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    cutoff: wp.float32,
    num_shifts: wp.int32,
    half_fill: wp.bool,
    partial: wp.bool,
) -> None:
    _launch_naive_neighbor_matrix_pbc(
        positions,
        float(cutoff),
        cell,
        None,
        shift_range,
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
        wp.float32,
        str(positions.device),
        batched=False,
        num_shifts=int(num_shifts),
        half_fill=bool(half_fill),
        wrap_positions=False,
        target_indices=target_indices if bool(partial) else None,
        strategy="tile",
    )


def _graph_naive_tile_pbc_prewrapped_f64(
    positions: wp.array(dtype=wp.vec3d),
    target_indices: wp.array(dtype=wp.int32),
    cell: wp.array(dtype=wp.mat33d),
    shift_range: wp.array(dtype=wp.vec3i),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    cutoff: wp.float64,
    num_shifts: wp.int32,
    half_fill: wp.bool,
    partial: wp.bool,
) -> None:
    _launch_naive_neighbor_matrix_pbc(
        positions,
        float(cutoff),
        cell,
        None,
        shift_range,
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
        wp.float64,
        str(positions.device),
        batched=False,
        num_shifts=int(num_shifts),
        half_fill=bool(half_fill),
        wrap_positions=False,
        target_indices=target_indices if bool(partial) else None,
        strategy="tile",
    )


def _graph_naive_tile_pbc_wrapped_f32(
    positions: wp.array(dtype=wp.vec3f),
    target_indices: wp.array(dtype=wp.int32),
    cell: wp.array(dtype=wp.mat33f),
    pbc: wp.array2d(dtype=wp.bool),
    inv_cell: wp.array(dtype=wp.mat33f),
    shift_range: wp.array(dtype=wp.vec3i),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    positions_wrapped: wp.array(dtype=wp.vec3f),
    per_atom_cell_offsets: wp.array(dtype=wp.vec3i),
    cutoff: wp.float32,
    num_shifts: wp.int32,
    half_fill: wp.bool,
    partial: wp.bool,
) -> None:
    _launch_naive_neighbor_matrix_pbc(
        positions,
        float(cutoff),
        cell,
        pbc,
        shift_range,
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
        wp.float32,
        str(positions.device),
        batched=False,
        num_shifts=int(num_shifts),
        half_fill=bool(half_fill),
        wrap_positions=True,
        target_indices=target_indices if bool(partial) else None,
        strategy="tile",
        positions_wrapped_buffer=positions_wrapped,
        per_atom_cell_offsets_buffer=per_atom_cell_offsets,
        inv_cell_buffer=inv_cell,
        reuse_inv_cell=True,
    )


def _graph_naive_tile_pbc_wrapped_f64(
    positions: wp.array(dtype=wp.vec3d),
    target_indices: wp.array(dtype=wp.int32),
    cell: wp.array(dtype=wp.mat33d),
    pbc: wp.array2d(dtype=wp.bool),
    inv_cell: wp.array(dtype=wp.mat33d),
    shift_range: wp.array(dtype=wp.vec3i),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    positions_wrapped: wp.array(dtype=wp.vec3d),
    per_atom_cell_offsets: wp.array(dtype=wp.vec3i),
    cutoff: wp.float64,
    num_shifts: wp.int32,
    half_fill: wp.bool,
    partial: wp.bool,
) -> None:
    _launch_naive_neighbor_matrix_pbc(
        positions,
        float(cutoff),
        cell,
        pbc,
        shift_range,
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
        wp.float64,
        str(positions.device),
        batched=False,
        num_shifts=int(num_shifts),
        half_fill=bool(half_fill),
        wrap_positions=True,
        target_indices=target_indices if bool(partial) else None,
        strategy="tile",
        positions_wrapped_buffer=positions_wrapped,
        per_atom_cell_offsets_buffer=per_atom_cell_offsets,
        inv_cell_buffer=inv_cell,
        reuse_inv_cell=True,
    )


# Keyed by ``(has_pbc, wrap_positions)``.  Tile has no selective variant, so the
# selective axis is omitted here; ``strategy="tile"`` rejects
# ``rebuild_flags`` at the dispatch site.
_GRAPH_NAIVE_TILE_NO_PBC_IN_OUT_ARGS = ("neighbor_matrix", "num_neighbors")
_GRAPH_NAIVE_TILE_PBC_IN_OUT_ARGS = (
    "neighbor_matrix",
    "neighbor_matrix_shifts",
    "num_neighbors",
)
_GRAPH_NAIVE_TILE_PBC_WRAPPED_IN_OUT_ARGS = (
    *_GRAPH_NAIVE_TILE_PBC_IN_OUT_ARGS,
    "positions_wrapped",
    "per_atom_cell_offsets",
)
_GRAPH_NAIVE_TILE_SPECS = {
    (False, False): {
        "num_outputs": 2,
        "in_out_argnames": _GRAPH_NAIVE_TILE_NO_PBC_IN_OUT_ARGS,
        jnp.dtype(jnp.float32): _graph_naive_tile_no_pbc_f32,
        jnp.dtype(jnp.float64): _graph_naive_tile_no_pbc_f64,
    },
    (True, False): {
        "num_outputs": 3,
        "in_out_argnames": _GRAPH_NAIVE_TILE_PBC_IN_OUT_ARGS,
        jnp.dtype(jnp.float32): _graph_naive_tile_pbc_prewrapped_f32,
        jnp.dtype(jnp.float64): _graph_naive_tile_pbc_prewrapped_f64,
    },
    (True, True): {
        "num_outputs": 5,
        "in_out_argnames": _GRAPH_NAIVE_TILE_PBC_WRAPPED_IN_OUT_ARGS,
        jnp.dtype(jnp.float32): _graph_naive_tile_pbc_wrapped_f32,
        jnp.dtype(jnp.float64): _graph_naive_tile_pbc_wrapped_f64,
    },
}


def _register_graph_naive_tile_callables() -> dict[
    tuple[bool, bool, jnp.dtype], object
]:
    """Register JaxCallableGraphMode.NONE tile callables for the naive eager path.

    ``JaxCallableGraphMode.NONE`` (not WARP): the tile bodies assume the caller has
    already pre-filled the output buffers, which only the eager
    (``graph_mode="none"``) path of ``naive_neighbor_list`` does.
    """
    registered: dict[tuple[bool, bool, jnp.dtype], object] = {}
    for (has_pbc, wrap_positions), spec in _GRAPH_NAIVE_TILE_SPECS.items():
        for dtype in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
            registered[(has_pbc, wrap_positions, dtype)] = jax_callable(
                spec[dtype],
                num_outputs=spec["num_outputs"],
                in_out_argnames=spec["in_out_argnames"],
                graph_mode=JaxCallableGraphMode.NONE,
            )
    return registered


_GRAPH_NAIVE_TILE_CALLABLES = _register_graph_naive_tile_callables()


def _naive_pair_outputs_forward(
    positions: jax.Array,
    cell: jax.Array | None,
    *,
    pbc: jax.Array | None,
    cutoff: float,
    max_neighbors: int,
    fill_value: int,
    neighbor_matrix: jax.Array | None = None,
    neighbor_matrix_shifts: jax.Array | None = None,
    num_neighbors: jax.Array | None = None,
    neighbor_vectors: jax.Array | None = None,
    neighbor_distances: jax.Array | None = None,
    shift_range_per_dimension: jax.Array | None = None,
    num_shifts_per_system: jax.Array | None = None,
    max_shifts_per_system: int | None = None,
    pair_fn=None,
    pair_params: jax.Array | None = None,
    target_indices: jax.Array | None = None,
    half_fill: bool = False,
    strategy: str = "scalar",
    wrap_positions: bool = True,
    graph_mode: str = "none",
    inv_cell: jax.Array | None = None,
    positions_wrapped: jax.Array | None = None,
    per_atom_cell_offsets: jax.Array | None = None,
    _fixed_cell_geometry: tuple[jax.Array | None, ...] | None = None,
) -> _NeighborForwardOutput:
    """Forward closure for the naive autograd path.

    Detaches positions/cell, runs the pair-output naive kernel, and
    packs the indices the autograd primitive needs for the reconstruction
    backward.

    When ``pair_fn`` is set, a ``pair_fn``-specialized kernel is launched and the
    per-pair ``pair_energies`` / ``pair_forces`` are appended to
    :attr:`_NeighborForwardOutput.extra_outputs` (positions 4 and 5).  These ride
    along *outside* the ``custom_vjp`` primitive: ``positions`` is detached above, so
    they are autograd-constants (forward-only / zero cotangent), while
    ``distances`` / ``vectors`` are re-attached on the original positions.

    Parameters
    ----------
    positions : jax.Array, shape (N, 3)
        Current atomic positions.
    cell, pbc : jax.Array or None
        Cell geometry and per-axis periodicity. ``cell`` is required when
        ``pbc`` is supplied.
    cutoff : float
        Neighbor cutoff.
    max_neighbors, fill_value : int
        Fixed row capacity and padding value for the neighbor matrix.
    neighbor_matrix, neighbor_matrix_shifts, num_neighbors : jax.Array, optional
        Caller-provided topology operands. Compact rows follow
        ``target_indices`` order.
    neighbor_vectors, neighbor_distances : jax.Array, optional
        Geometry output operands used by differentiable reconstruction.
    shift_range_per_dimension, num_shifts_per_system : jax.Array, optional
        Prepared periodic-image metadata.
    max_shifts_per_system : int, optional
        Fixed periodic-image capacity.
    pair_fn : Warp function or None
        Optional forward-only pair callback.
    pair_params : jax.Array, optional
        Per-atom parameters passed to ``pair_fn``.
    target_indices : jax.Array, optional
        Compact source rows; repeated indices produce repeated output rows.
    half_fill : bool
        Select the canonical half-list convention.
    strategy : {"scalar", "tile"}
        Naive kernel strategy. Tile is used for topology-only compact rows.
    wrap_positions : bool
        Whether the callback wraps live positions before its tile search.
    graph_mode : {"none", "warp"}
        Warp callback mode for supported full-row routes.
    inv_cell, positions_wrapped, per_atom_cell_offsets : jax.Array, optional
        Resolved inverse and wrapping scratch operands for wrapped execution.
    _fixed_cell_geometry : tuple or None
        Prepared fixed-cell cache metadata used to select a readonly inverse.

    Returns
    -------
    _NeighborForwardOutput
        Geometry, active-pair reconstruction data, and topology/callback arrays
        consumed by the public wrapper and autograd primitive.

    Notes
    -----
    Compact topology-only tile execution skips residual pair-index construction.
    Wrapped tile execution receives validated scratch operands; it reuses the
    prepared inverse for fixed-cell geometry and computes from the current cell
    in the default mode. Pair energies and forces remain forward-only outputs.
    """
    positions = jax.lax.stop_gradient(positions)
    if cell is not None:
        cell = jax.lax.stop_gradient(cell)

    total_atoms = positions.shape[0]
    is_partial = target_indices is not None
    if is_partial:
        target_indices = jnp.asarray(target_indices, dtype=jnp.int32)
        num_rows = int(target_indices.shape[0])
    else:
        num_rows = total_atoms
    f64 = positions.dtype == jnp.float64
    wp_dtype = wp.float64 if f64 else wp.float32
    cutoff_sq = float(cutoff * cutoff)
    zero_dt = float(0.0)
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

    ti_arg = target_indices if is_partial else empty_target_indices
    has_pair_fn = pair_fn is not None
    use_tile = strategy == "tile" and is_partial and not has_pair_fn
    if neighbor_matrix is None:
        nm = jnp.full((num_rows, max_neighbors), fill_value, dtype=jnp.int32)
    elif graph_mode == "warp":
        nm = neighbor_matrix
    else:
        nm = neighbor_matrix.at[:].set(jnp.int32(fill_value))
    if num_neighbors is None:
        nn = jnp.zeros(num_rows, dtype=jnp.int32)
    elif graph_mode == "warp":
        nn = num_neighbors
    else:
        nn = num_neighbors.at[:].set(jnp.int32(0))
    if pbc is None and use_tile:
        nms = empty_shifts
    elif neighbor_matrix_shifts is None:
        nms = jnp.zeros((num_rows, max_neighbors, 3), dtype=jnp.int32)
    elif graph_mode == "warp":
        nms = neighbor_matrix_shifts
    else:
        nms = neighbor_matrix_shifts.at[:].set(jnp.int32(0))
    # ``pair_fn`` path: real per-atom params + auto-allocated energy/force buffers.
    # (JAX is functional, so user-supplied energy/force buffers cannot be written
    # in-place; we always allocate fresh and return them — the return contract
    # matches torch, the in-place-buffer aspect does not.)
    if has_pair_fn:
        pp_arg = jnp.asarray(pair_params, dtype=positions.dtype)
        pe = jnp.zeros((num_rows, max_neighbors), dtype=positions.dtype)
        pf = jnp.zeros((num_rows, max_neighbors, 3), dtype=positions.dtype)
    else:
        pp_arg = empty_pair_params
        pe = None
        pf = None

    if use_tile:
        # Tile kernels write topology only and do not consume geometry buffers.
        # Keep the zero-sized sentinels instead of materializing discarded
        # ``(num_rows, max_neighbors, 3)`` and ``(num_rows, max_neighbors)``
        # arrays for topology-only partial calls.
        nv = empty_vectors
        nd = empty_distances
    else:
        if neighbor_vectors is None:
            nv = jnp.zeros((num_rows, max_neighbors, 3), dtype=positions.dtype)
        else:
            nv = neighbor_vectors.at[:].set(jnp.asarray(0.0, dtype=positions.dtype))
        if neighbor_distances is None:
            nd = jnp.zeros((num_rows, max_neighbors), dtype=positions.dtype)
        else:
            nd = neighbor_distances.at[:].set(jnp.asarray(0.0, dtype=positions.dtype))

    if use_tile and pbc is None:
        tile_callable = _GRAPH_NAIVE_TILE_CALLABLES[(False, False, positions.dtype)]
        nm, nn = tile_callable(
            positions,
            ti_arg,
            nm,
            nn,
            float(cutoff),
            half_fill,
            True,
        )
    elif pbc is None:
        if has_pair_fn:
            kernel = _get_jax_naive_pair_fn_kernel(
                pair_fn, wp_dtype, "none", half_fill, is_partial
            )
        elif is_partial:
            kernel = _get_jax_naive_pair_kernel(wp_dtype, "none", half_fill, is_partial)
        elif half_fill:
            kernel = _DIRECT_NAIVE_GEOMETRY_KERNELS[("none", True)][positions.dtype]
        else:
            kernel = _DIRECT_NAIVE_GEOMETRY_KERNELS[("none", False)][positions.dtype]
        outs = kernel(
            positions,
            empty_offsets,
            cutoff_sq,
            zero_dt,
            empty_cell,
            empty_shift_range,
            empty_num_shifts,
            empty_batch_idx,
            empty_batch_ptr,
            ti_arg,
            nm,
            empty_shifts,
            nn,
            empty_matrix,
            empty_shifts,
            empty_num_neighbors,
            nv,
            nd,
            pp_arg,
            pe if has_pair_fn else empty_energies,
            pf if has_pair_fn else empty_forces,
            empty_rebuild_flags,
            launch_dims=(1, 1, num_rows),
        )
        if has_pair_fn:
            nm, nn, nv, nd, pe, pf = outs
        else:
            nm, nn, nv, nd = outs
    elif use_tile:
        if cell.ndim == 2:
            cell = cell[jnp.newaxis, :, :]
        if pbc.ndim == 1:
            pbc = pbc[jnp.newaxis, :]
        if (
            shift_range_per_dimension is None
            or num_shifts_per_system is None
            or max_shifts_per_system is None
        ):
            (
                shift_range_per_dimension,
                num_shifts_per_system,
                max_shifts_per_system,
            ) = compute_naive_num_shifts(cell, cutoff, pbc)
        tile_callable = _GRAPH_NAIVE_TILE_CALLABLES[
            (True, bool(wrap_positions), positions.dtype)
        ]
        pbc_arg: tuple[jax.Array, ...] = ()
        if wrap_positions:
            inverse, positions_wrapped_work, offsets_work = (
                _prepare_wrapped_tile_operands(
                    positions,
                    cell,
                    inv_cell_buffer=inv_cell,
                    positions_wrapped_buffer=positions_wrapped,
                    per_atom_cell_offsets_buffer=per_atom_cell_offsets,
                    fixed_cell_geometry=_fixed_cell_geometry,
                )
            )
            pbc_arg = (
                pbc,
                inverse,
            )
        tile_args = (
            positions,
            ti_arg,
            cell,
            *pbc_arg,
            shift_range_per_dimension,
            nm,
            nms,
            nn,
            *((positions_wrapped_work, offsets_work) if wrap_positions else ()),
            float(cutoff),
            int(max_shifts_per_system),
            half_fill,
            True,
        )
        tile_outs = tile_callable(*tile_args)
        nm, nms, nn = tile_outs[:3]
    else:
        fixed_mode = _fixed_cell_geometry is not None
        fixed_inverse = (
            None if _fixed_cell_geometry is None else _fixed_cell_geometry[0]
        )
        pbc_mode = (
            "prewrapped" if fixed_mode and not wrap_positions else "wrap_on_entry"
        )
        if has_pair_fn:
            kernel = _get_jax_naive_pair_fn_kernel(
                pair_fn, wp_dtype, pbc_mode, half_fill, is_partial
            )
        elif is_partial or fixed_inverse is not None:
            kernel = _get_jax_naive_pair_kernel(
                wp_dtype, pbc_mode, half_fill, is_partial
            )
        elif half_fill:
            kernel = _DIRECT_NAIVE_GEOMETRY_KERNELS[("wrap_on_entry", True)][
                positions.dtype
            ]
        else:
            kernel = _DIRECT_NAIVE_GEOMETRY_KERNELS[("wrap_on_entry", False)][
                positions.dtype
            ]
        if cell.ndim == 2:
            cell = cell[jnp.newaxis, :, :]
        if pbc.ndim == 1:
            pbc = pbc[jnp.newaxis, :]
        # ``max_shifts`` sizes the middle launch axis: the single-system PBC kernel
        # derives each periodic image from ``ishift = wp.tid()`` (no internal shift
        # loop), so the launch must enumerate every shift.  Pinning it to 1 would
        # silently drop all non-zero images (only ``ishift == 0`` runs), matching
        # neighbors only in the R==1 regime.
        if (
            shift_range_per_dimension is None
            or num_shifts_per_system is None
            or max_shifts_per_system is None
        ):
            (
                shift_range_per_dimension,
                num_shifts_per_system,
                max_shifts_per_system,
            ) = compute_naive_num_shifts(cell, cutoff, pbc)
        offs = jnp.zeros((total_atoms, 3), dtype=jnp.int32)
        positions_kernel = positions
        if fixed_mode and wrap_positions and fixed_inverse is not None:
            positions_kernel = jnp.zeros_like(positions)
        if wrap_positions:
            inv_cell = fixed_inverse if fixed_inverse is not None else inv_cell
            if inv_cell is None:
                inv_cell = jnp.linalg.inv(cell)
            if positions_wrapped is not None:
                positions_kernel = positions_wrapped
            elif positions_kernel is positions:
                positions_kernel = jnp.zeros_like(positions)
            if per_atom_cell_offsets is not None:
                offs = per_atom_cell_offsets
            wrap_kernel = (
                _jax_wrap_positions_single_f64
                if f64
                else _jax_wrap_positions_single_f32
            )
            positions_kernel, offs = wrap_kernel(
                positions,
                cell,
                inv_cell,
                pbc,
                empty_batch_idx,
                positions_kernel,
                offs,
                launch_dims=(total_atoms,),
            )
        active_shift_dim = int(max_shifts_per_system)
        if is_partial and not half_fill:
            active_shift_dim = 2 * active_shift_dim - 1
        outs = kernel(
            positions_kernel,
            offs,
            cutoff_sq,
            zero_dt,
            cell,
            shift_range_per_dimension,
            num_shifts_per_system,
            empty_batch_idx,
            empty_batch_ptr,
            ti_arg,
            nm,
            nms,
            nn,
            empty_matrix,
            empty_shifts,
            empty_num_neighbors,
            nv,
            nd,
            pp_arg,
            pe if has_pair_fn else empty_energies,
            pf if has_pair_fn else empty_forces,
            empty_rebuild_flags,
            launch_dims=(1, active_shift_dim, num_rows),
        )
        if has_pair_fn:
            nm, nms, nn, nv, nd, pe, pf = outs
        else:
            nm, nms, nn, nv, nd = outs

    if use_tile:
        # The topology-only caller consumes only ``extra_outputs``. Avoid
        # constructing the residual index matrices needed solely for the
        # differentiable geometry reconstruction.
        i_idx = j_idx = empty_matrix
        shifts_ret = empty_shifts
        mask_ = jnp.empty((0, 0), dtype=jnp.bool_)
        batch_idx_ret = None
    else:
        i_idx, j_idx, shifts_ret, batch_idx_ret, mask_ = _build_index_residuals(
            nm,
            nn,
            nms,
            target_indices=target_indices if is_partial else None,
        )
    K, M = nm.shape
    extra_outputs = (nm, nn, nms, pe, pf) if has_pair_fn else (nm, nn, nms)
    return _NeighborForwardOutput(
        distances=nd,
        vectors=nv,
        extra_outputs=extra_outputs,
        i_idx=i_idx,
        j_idx=j_idx,
        shifts=shifts_ret,
        batch_idx=batch_idx_ret,
        active_mask=mask_,
        matrix_shape=(K, M),
    )


def _validate_pair_output_buffer(
    name: str,
    array: jax.Array | None,
    expected_shape: tuple[int, ...],
    expected_dtype=None,
) -> None:
    """Validate optional matrix-shaped output buffers for pair-output paths."""
    if array is None:
        return
    if tuple(array.shape) != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}; got {tuple(array.shape)}.",
        )
    if expected_dtype is not None and array.dtype != expected_dtype:
        raise ValueError(
            f"{name} dtype must be {expected_dtype}; got {array.dtype}.",
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
    inv_cell_buffer: jax.Array | None = None,
    positions_wrapped_buffer: jax.Array | None = None,
    per_atom_cell_offsets_buffer: jax.Array | None = None,
    strategy: str = "auto",
    *,
    coo_capacity: int | None = None,
    return_distances: bool = False,
    return_vectors: bool = False,
    neighbor_vectors: jax.Array | None = None,
    neighbor_distances: jax.Array | None = None,
    # Pair-output / partial kwargs.
    target_indices: jax.Array | None = None,
    pair_fn=None,
    pair_params: jax.Array | None = None,
    pair_energies: jax.Array | None = None,
    pair_forces: jax.Array | None = None,
    _fixed_cell_geometry: tuple[jax.Array | None, ...] | None = None,
    # Deprecated kwarg aliases (removed in 0.5):
    inv_cell: jax.Array | None = None,
    positions_wrapped: jax.Array | None = None,
    per_atom_cell_offsets: jax.Array | None = None,
    graph_mode: Literal["none", "warp"] = "none",
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
        Eager calls with zero or negative cutoffs raise ``ValueError``.
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
    neighbor_matrix : jax.Array, shape (num_rows, max_neighbors), dtype=int32, optional
        Neighbor matrix to be filled. Pass in a pre-shaped array to hint buffer reuse
        to XLA; note that JAX returns a new array rather than mutating the input.
        ``num_rows`` is ``total_atoms`` normally and ``len(target_indices)`` when
        partial rows are requested.
        Must be provided if max_neighbors is not provided.
    neighbor_matrix_shifts : jax.Array, shape (num_rows, max_neighbors, 3), dtype=int32, optional
        Shift vectors for each neighbor relationship. Pass in a pre-shaped array to hint
        buffer reuse to XLA; note that JAX returns a new array rather than mutating the input.
        Must be provided if max_neighbors is not provided.
    num_neighbors : jax.Array, shape (num_rows,), dtype=int32, optional
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
    rebuild_flags : jax.Array, shape () or (1,), dtype=bool, optional
        Device-side selective-rebuild flag. When false, supplied topology
        buffers are preserved. Not supported with compact partial rows,
        pair outputs, or ``strategy="tile"``.
    return_neighbor_list : bool, optional - default = False
        If True, convert the neighbor matrix to a neighbor list (idx_i, idx_j) format by
        creating a mask over the fill_value, which can incur a performance penalty.
    coo_capacity : int, optional
        Static COO capacity. With ``return_neighbor_list=True``, returns padded
        fixed-size COO arrays plus raw required row counts and a scalar
        metadata-validity flag. If omitted, returns compact data-dependent COO.
    return_distances : bool, default=False
        Append differentiable per-pair distances. Uses the scalar path.
    return_vectors : bool, default=False
        Append differentiable per-pair displacement vectors. Uses the scalar
        path.
    neighbor_distances : jax.Array, shape (num_rows, max_neighbors), optional
        Pre-shaped distance output for ``return_distances=True`` or ``pair_fn``.
    neighbor_vectors : jax.Array, shape (num_rows, max_neighbors, 3), optional
        Pre-shaped vector output for ``return_vectors=True`` or ``pair_fn``.
    target_indices : jax.Array, shape (num_targets,), dtype=int32, optional
        Indices of the central atoms for compact partial rows. Output row ``r``
        maps to atom ``target_indices[r]``. In COO output, the first row holds
        compact row ids. User buffers must be compact-row shaped, not full
        atom-row shaped. Values must be in bounds; repeated rows are allowed.
    pair_fn : wp.Function, optional
        Module-scope Warp pair potential evaluated during the scalar neighbor
        search. Requires ``pair_params`` and returns forward-only energy and
        force outputs.
    pair_params : jax.Array, shape (total_atoms, K), optional
        Per-atom parameters forwarded to ``pair_fn``.
    pair_energies : jax.Array, shape (num_rows, max_neighbors), optional
        Pre-shaped energy output buffer for ``pair_fn``.
    pair_forces : jax.Array, shape (num_rows, max_neighbors, 3), optional
        Pre-shaped force output buffer for ``pair_fn``.
    wrap_positions : bool, default=True
        If True, wrap input positions into the primary cell before
        neighbor search. Set to False when positions are already
        wrapped (e.g. by a preceding integration step) to save two
        GPU kernel launches per call.
    strategy : {"auto", "scalar", "tile"}, default="auto"
        Selects the underlying Warp kernel variant. ``"scalar"`` uses the
        per-atom scalar kernel. ``"tile"`` uses the tile-cooperative
        ``wp.launch_tiled`` kernel and is **CUDA-only**: requesting it on a
        CPU device raises ``ValueError``. Tile supports topology-only compact
        ``target_indices`` rows; geometry and pair outputs use the scalar path.
        For concrete-placement single-system CUDA arrays, topology-only partial
        ``"auto"`` selects tile at float64 ``N >= 256`` and float32
        ``N >= 1024``. Under ``jax.jit``, ``positions`` may be a tracer whose
        device placement is unavailable during Python tracing, so ``"auto"``
        may conservatively remain scalar even above these thresholds; pass
        ``strategy="tile"`` for deterministic tiled execution. Batched partial
        auto remains scalar. Explicit tile rejects geometry and pair outputs.
        ``target_indices`` with
        ``graph_mode="warp"`` is rejected. Partial neighbor lists do not
        support ``rebuild_flags``. For compact partial rows, when neither result
        overflows capacity, scalar and tile stored ``(neighbor, shift)``
        multisets agree, although ordering may differ.
    inv_cell_buffer : jax.Array, shape (1, 3, 3), dtype matching positions, optional
        Inverse-cell buffer for wrapped PBC calls. Default wrapped tile calls
        treat this as inverse scratch and recompute from the current cell;
        prepared fixed-cell tile calls reuse their cached inverse. Full-row
        scalar topology calls and supported topology-only Warp replay calls use
        the supplied precomputed inverse.
    positions_wrapped_buffer : jax.Array, shape (total_atoms, 3), dtype matching positions, optional
        Scratch buffer written by the position-wrapping kernel.
    per_atom_cell_offsets_buffer : jax.Array, shape (total_atoms, 3), dtype=int32, optional
        Scratch buffer for per-atom wrapping offsets.
    inv_cell, positions_wrapped, per_atom_cell_offsets : jax.Array, optional
        Deprecated aliases for the corresponding ``*_buffer`` arguments.
    graph_mode : {"none", "warp"}, default="none"
        Execution mode for the underlying Warp launches. ``"none"``
        preserves the existing per-kernel ``jax_kernel`` dispatch path.
        ``"warp"`` uses fused ``jax_callable(..., graph_mode=JaxCallableGraphMode.WARP)``
        callbacks and is intended for ``jax.jit`` call sites that donate
        reusable output buffers. ``target_indices`` calls are rejected in this
        mode; supported full-row graph behavior is unchanged.

    Returns
    -------
    results : tuple of jax.Array
        Variable-length tuple depending on input parameters. The return pattern follows:

        - No PBC, matrix format: ``(neighbor_matrix, num_neighbors)``
        - No PBC, list format: ``(neighbor_list, neighbor_ptr)``
        - With PBC, matrix format: ``(neighbor_matrix, num_neighbors, neighbor_matrix_shifts)``
        - With PBC, list format: ``(neighbor_list, neighbor_ptr, neighbor_list_shifts)``

        With ``coo_capacity``, raw ``num_neighbors`` and scalar
        ``metadata_valid`` follow the topology tuple. When metadata is valid,
        a row is complete exactly when
        ``neighbor_ptr[r + 1] - neighbor_ptr[r] == num_neighbors[r]``. When
        ``metadata_valid`` is false, every returned count is ``-1``. Requested
        distances, vectors, energies, and forces follow in that order when
        enabled.

        **Components returned:**

        - **neighbor_data** (array): Neighbor indices, format depends on ``return_neighbor_list``:

            * If ``return_neighbor_list=False`` (default): Returns ``neighbor_matrix``
              with shape (num_rows, max_neighbors), dtype int32. Row ``r`` contains
              neighbors for atom ``r`` or ``target_indices[r]`` when partial rows
              are requested.
            * If ``return_neighbor_list=True``: Returns ``neighbor_list`` with shape
              (2, num_pairs), dtype int32, in COO format [central_rows, neighbor_atoms].
              With ``target_indices``, central rows are compact row ids.

        - **num_neighbor_data** (array): Information about the number of neighbors for each atom,
          format depends on ``return_neighbor_list``:

            * If ``return_neighbor_list=False`` (default): Returns ``num_neighbors`` with shape (num_rows,), dtype int32.
              Count of neighbors found for each atom. Always returned.
            * If ``return_neighbor_list=True``: Returns ``neighbor_ptr`` with shape (num_rows + 1,), dtype int32.
              CSR-style pointer arrays where ``neighbor_ptr_data[i]`` to ``neighbor_ptr_data[i+1]`` gives the range of
              neighbors for row i in the flattened neighbor list.

        - **neighbor_shift_data** (array, optional): Periodic shift vectors, only when ``pbc`` is provided:
          format depends on ``return_neighbor_list``:

            * If ``return_neighbor_list=False`` (default): Returns ``neighbor_matrix_shifts`` with
              shape (num_rows, max_neighbors, 3), dtype int32.
            * If ``return_neighbor_list=True``: Returns ``unit_shifts`` with shape
              (num_pairs, 3), dtype int32.

    Examples
    --------
    Basic usage without periodic boundary conditions:

    >>> import jax.numpy as jnp
    >>> from nvalchemiops.jax.neighbors import compute_naive_num_shifts, naive_neighbor_list
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

    Warp graph replay with donated buffers (PBC + wrap_positions=True):

    >>> import functools
    >>> import jax
    >>> # Pre-allocate the wrap kernel's scratch buffers and inverse cell once.
    >>> # Capturing them in the closure (rather than donating) keeps their
    >>> # buffer pointers stable across calls, which is what Warp's graph
    >>> # cache keys on. Only the buffers naive_neighbor_list returns are
    >>> # donated, so the in/out arity of the jit'ed step matches.
    >>> inv_cell_buffer = jnp.linalg.inv(cell)
    >>> positions_wrapped_buffer = jnp.zeros_like(positions)
    >>> per_atom_cell_offsets_buffer = jnp.zeros((positions.shape[0], 3), dtype=jnp.int32)
    >>> shift_range, num_shifts_per_system, max_shifts_per_system = (
    ...     compute_naive_num_shifts(cell, cutoff, pbc)
    ... )
    >>> @functools.partial(jax.jit, donate_argnums=(1, 2, 3))
    ... def md_step(positions, neighbor_matrix, num_neighbors, shifts):
    ...     return naive_neighbor_list(
    ...         positions,
    ...         cutoff,
    ...         cell=cell,
    ...         pbc=pbc,
    ...         neighbor_matrix=neighbor_matrix,
    ...         num_neighbors=num_neighbors,
    ...         neighbor_matrix_shifts=shifts,
    ...         inv_cell_buffer=inv_cell_buffer,
    ...         positions_wrapped_buffer=positions_wrapped_buffer,
    ...         per_atom_cell_offsets_buffer=per_atom_cell_offsets_buffer,
    ...         shift_range_per_dimension=shift_range,
    ...         num_shifts_per_system=num_shifts_per_system,
    ...         max_shifts_per_system=max_shifts_per_system,
    ...         graph_mode="warp",
    ...     )

    See Also
    --------
    nvalchemiops.neighbors.naive.naive_neighbor_matrix : Core warp launcher (no PBC)
    nvalchemiops.neighbors.naive.naive_neighbor_matrix_pbc : Core warp launcher (with PBC)
    cell_list : O(N) cell list method for larger systems

    Notes
    -----
    For lower host-side launch overhead on supported GPUs, setting
    ``XLA_FLAGS=--xla_gpu_enable_command_buffer=CUSTOM_CALL`` before
    importing JAX can improve the steady-state ``graph_mode="none"`` and
    ``graph_mode="warp"`` paths. Advanced users can bound each callable's
    graph cache with the ``graph_cache_max`` argument to
    ``warp.jax_callable(...)``.

    For ``graph_mode="warp"`` to replay rather than recapture every call,
    every input and output buffer pointer seen by the fused callable must stay
    stable. User-provided output buffers (``neighbor_matrix``,
    ``num_neighbors``, and ``neighbor_matrix_shifts`` when applicable) must be
    included in ``donate_argnums`` of the enclosing ``jax.jit`` so they
    round-trip between calls. Wrapped PBC also requires stable
    ``inv_cell_buffer``, ``positions_wrapped_buffer``, and
    ``per_atom_cell_offsets_buffer`` arrays; preallocate them once and capture
    them in the jit closure. For every ``jax.jit`` PBC call, precompute
    ``shift_range_per_dimension``, ``num_shifts_per_system``, and
    ``max_shifts_per_system`` via
    :func:`compute_naive_num_shifts` outside the compiled boundary. Eager PBC
    calls may omit these values. Cutoff, ``half_fill``, and PBC shift metadata
    are statically specialized. Scratch buffers allocated inside this function
    prevent stable graph replay and add repeated capture overhead.
    """
    if cutoff <= 0:
        raise ValueError("cutoff must be positive")

    graph_mode = _validate_graph_mode(graph_mode)
    coo_capacity = _validate_coo_capacity(coo_capacity, return_neighbor_list)

    if strategy not in {"auto", "scalar", "tile"}:
        raise ValueError(
            f"strategy must be 'auto' | 'scalar' | 'tile', got {strategy!r}",
        )

    # ``pair_fn`` requires per-atom ``pair_params``.  Note: under JAX (functional
    # arrays) any user-supplied ``pair_energies`` / ``pair_forces`` cannot be written
    # in-place — they are auto-allocated and returned, so the *return* contract
    # matches torch while the in-place-buffer aspect does not.
    if pair_fn is not None and pair_params is None:
        raise ValueError(
            "pair_fn requires pair_params (a per-atom (n_atoms, K) parameter array).",
        )
    if pair_fn is None:
        if pair_params is not None:
            raise ValueError("pair_params requires pair_fn.")
        if pair_energies is not None:
            raise ValueError("pair_energies requires pair_fn.")
        if pair_forces is not None:
            raise ValueError("pair_forces requires pair_fn.")

    if pbc is None and cell is not None:
        raise ValueError("If cell is provided, pbc must also be provided")
    if pbc is not None and cell is None:
        raise ValueError("If pbc is provided, cell must also be provided")

    if target_indices is not None and (
        target_indices.ndim != 1 or target_indices.dtype != jnp.int32
    ):
        raise ValueError("target_indices must be a rank-one int32 array.")
    if target_indices is not None and not isinstance(target_indices, jax.core.Tracer):
        concrete_target_indices = np.asarray(target_indices)
        if np.any(concrete_target_indices < 0) or np.any(
            concrete_target_indices >= positions.shape[0]
        ):
            raise ValueError("target_indices must contain in-bounds atom indices.")
    if target_indices is not None and graph_mode == "warp":
        raise ValueError("graph_mode='warp' does not support target_indices.")
    if target_indices is not None and rebuild_flags is not None:
        raise NotImplementedError(
            "Partial neighbor lists do not support rebuild_flags",
        )

    real_geometry_or_pair_outputs = (
        bool(return_distances)
        or bool(return_vectors)
        or neighbor_vectors is not None
        or neighbor_distances is not None
        or pair_fn is not None
    )
    compact_topology_partial = (
        target_indices is not None and not real_geometry_or_pair_outputs
    )
    partial_tile_selected = False
    if compact_topology_partial:
        strategy = _resolve_naive_strategy(
            strategy,
            _NaiveWorkload(
                device_kind=_jax_array_device_kind(positions),
                wp_dtype=(wp.float64 if positions.dtype == jnp.float64 else wp.float32),
                num_atoms=int(positions.shape[0]),
                num_systems=1,
                partial=True,
                batched=False,
                pbc=pbc is not None,
                wrap_positions=wrap_positions,
                geometry_outputs=False,
            ),
        )
        partial_tile_selected = strategy == "tile"

    if strategy == "tile":
        # The tile-cooperative kernel is CUDA-only and has no geometry/pair_fn,
        # selective (rebuild_flags), or full-row CUDA-graph variant. Compact
        # topology-only rows launch the tiled kernel directly.
        device_kind = _jax_array_device_kind(positions)
        if device_kind == "cpu":
            _require_cuda_tile_device("cpu")
        if (
            bool(return_distances)
            or bool(return_vectors)
            or neighbor_vectors is not None
            or neighbor_distances is not None
            or pair_fn is not None
        ):
            raise NotImplementedError(
                "strategy='tile' has no pair-output (return_distances / "
                "return_vectors / pair_fn) variant; use strategy='scalar'.",
            )
        if rebuild_flags is not None:
            raise NotImplementedError(
                "strategy='tile' has no selective (rebuild_flags) "
                "variant; use strategy='scalar'.",
            )
        if graph_mode != "none":
            raise NotImplementedError(
                "strategy='tile' does not support graph_mode='warp'.",
            )

    uses_compact_pair_kernel = (
        real_geometry_or_pair_outputs or target_indices is not None
    )
    if uses_compact_pair_kernel:
        if graph_mode != "none" or rebuild_flags is not None:
            raise NotImplementedError(
                "Pair outputs require graph_mode='none' and no rebuild_flags.",
            )
        num_rows = (
            int(target_indices.shape[0])
            if target_indices is not None
            else int(positions.shape[0])
        )
        if max_neighbors is None and neighbor_matrix is not None:
            max_neighbors = int(neighbor_matrix.shape[1])
        if max_neighbors is None:
            max_neighbors = estimate_max_neighbors(cutoff)
        if fill_value is None:
            fill_value = positions.shape[0]
        _validate_pair_output_buffer(
            "neighbor_matrix",
            neighbor_matrix,
            (num_rows, int(max_neighbors)),
            jnp.int32,
        )
        _validate_pair_output_buffer(
            "num_neighbors",
            num_neighbors,
            (num_rows,),
            jnp.int32,
        )
        if pbc is not None:
            _validate_pair_output_buffer(
                "neighbor_matrix_shifts",
                neighbor_matrix_shifts,
                (num_rows, int(max_neighbors), 3),
                jnp.int32,
            )
        _validate_pair_output_buffer(
            "neighbor_distances",
            neighbor_distances,
            (num_rows, int(max_neighbors)),
            positions.dtype,
        )
        _validate_pair_output_buffer(
            "neighbor_vectors",
            neighbor_vectors,
            (num_rows, int(max_neighbors), 3),
            positions.dtype,
        )
        if target_indices is not None and num_rows == 0:
            matrix_out = (
                jnp.full((num_rows, max_neighbors), fill_value, dtype=jnp.int32)
                if neighbor_matrix is None
                else neighbor_matrix.at[:].set(jnp.int32(fill_value))
            )
            counts_out = (
                jnp.zeros(num_rows, dtype=jnp.int32)
                if num_neighbors is None
                else num_neighbors.at[:].set(jnp.int32(0))
            )
            distances_out = (
                jnp.zeros((num_rows, max_neighbors), dtype=positions.dtype)
                if neighbor_distances is None
                else neighbor_distances.at[:].set(
                    jnp.asarray(0.0, dtype=positions.dtype)
                )
            )
            vectors_out = (
                jnp.zeros((num_rows, max_neighbors, 3), dtype=positions.dtype)
                if neighbor_vectors is None
                else neighbor_vectors.at[:].set(jnp.asarray(0.0, dtype=positions.dtype))
            )
            if pair_fn is not None:
                energies_out = jnp.zeros(
                    (num_rows, max_neighbors),
                    dtype=positions.dtype,
                )
                forces_out = jnp.zeros(
                    (num_rows, max_neighbors, 3),
                    dtype=positions.dtype,
                )
            shifts_out = None
            if pbc is not None:
                shifts_out = (
                    jnp.zeros((num_rows, max_neighbors, 3), dtype=jnp.int32)
                    if neighbor_matrix_shifts is None
                    else neighbor_matrix_shifts.at[:].set(jnp.int32(0))
                )

            if return_neighbor_list:
                if coo_capacity is None:
                    base = (
                        jnp.zeros((2, 0), dtype=jnp.int32),
                        jnp.zeros((num_rows + 1,), dtype=jnp.int32),
                    )
                    if shifts_out is not None:
                        base = (*base, jnp.zeros((0, 3), dtype=jnp.int32))
                    output_pairs = 0
                else:
                    base, _ = _pack_fixed_capacity_neighbor_list_from_neighbor_matrix(
                        matrix_out,
                        counts_out,
                        capacity=coo_capacity,
                        neighbor_shift_matrix=shifts_out,
                        fill_value=int(fill_value),
                        metadata_valid=jnp.ones((), dtype=jnp.bool_),
                    )
                    output_pairs = int(coo_capacity)
                tail: list[jax.Array] = []
                if return_distances:
                    tail.append(jnp.zeros((output_pairs,), dtype=positions.dtype))
                if return_vectors:
                    tail.append(jnp.zeros((output_pairs, 3), dtype=positions.dtype))
                if pair_fn is not None:
                    tail.extend(
                        (
                            jnp.zeros((output_pairs,), dtype=positions.dtype),
                            jnp.zeros((output_pairs, 3), dtype=positions.dtype),
                        ),
                    )
                return (*base, *tail)

            tail = []
            if return_distances:
                tail.append(distances_out)
            if return_vectors:
                tail.append(vectors_out)
            if pair_fn is not None:
                tail.extend((energies_out, forces_out))
            if shifts_out is not None:
                return matrix_out, counts_out, shifts_out, *tail
            return matrix_out, counts_out, *tail
        if cell is not None and cell.ndim == 2:
            cell_norm = cell[jnp.newaxis, :, :]
        else:
            cell_norm = cell
        if cell_norm is not None and cell_norm.dtype != positions.dtype:
            cell_norm = cell_norm.astype(positions.dtype)
        pbc_norm = None
        if pbc is not None:
            pbc_norm = pbc if pbc.ndim == 2 else pbc[jnp.newaxis, :]
        if pbc_norm is not None:
            if (
                shift_range_per_dimension is None
                or num_shifts_per_system is None
                or max_shifts_per_system is None
            ):
                (
                    shift_range_per_dimension,
                    num_shifts_per_system,
                    max_shifts_per_system,
                ) = compute_naive_num_shifts(
                    jax.lax.stop_gradient(cell_norm),
                    cutoff,
                    pbc_norm,
                )
            try:
                max_shifts_per_system = int(max_shifts_per_system)
            except (
                jax.errors.ConcretizationTypeError,
                jax.errors.TracerIntegerConversionError,
            ) as exc:
                raise ValueError(
                    "max_shifts_per_system must be passed as a concrete int when "
                    "calling naive_neighbor_list under jax.jit with PBC and "
                    "target_indices / pair outputs.",
                ) from exc
        forward_kwargs = {
            "pbc": pbc_norm,
            "cutoff": float(cutoff),
            "max_neighbors": int(max_neighbors),
            "fill_value": int(fill_value),
            "neighbor_matrix": neighbor_matrix,
            "neighbor_matrix_shifts": neighbor_matrix_shifts,
            "num_neighbors": num_neighbors,
            "neighbor_vectors": neighbor_vectors,
            "neighbor_distances": neighbor_distances,
            "shift_range_per_dimension": shift_range_per_dimension,
            "num_shifts_per_system": num_shifts_per_system,
            "max_shifts_per_system": max_shifts_per_system,
            "pair_fn": pair_fn,
            "pair_params": pair_params,
            "target_indices": target_indices,
            "half_fill": bool(half_fill),
            "strategy": "tile" if partial_tile_selected else "scalar",
            "wrap_positions": bool(wrap_positions),
            "graph_mode": graph_mode,
            "_fixed_cell_geometry": _fixed_cell_geometry,
        }
        if compact_topology_partial:
            # Do not reconstruct the full compact ``(rows, max_neighbors)``
            # geometry only to discard it. This keeps topology-only partial
            # timing focused on the Warp search/callback and avoids unnecessary
            # gathers, norms, and materialized vector/distance arrays.
            compact_forward_kwargs = forward_kwargs
            if partial_tile_selected and pbc_norm is not None and wrap_positions:
                tile_inv_cell = resolve_buffer_alias(
                    "inv_cell_buffer",
                    inv_cell_buffer,
                    "inv_cell",
                    inv_cell,
                )
                if (
                    _fixed_cell_geometry is not None
                    and _fixed_cell_geometry[0] is not None
                ):
                    tile_inv_cell = _fixed_cell_geometry[0]
                tile_positions_wrapped = resolve_buffer_alias(
                    "positions_wrapped_buffer",
                    positions_wrapped_buffer,
                    "positions_wrapped",
                    positions_wrapped,
                )
                tile_per_atom_cell_offsets = resolve_buffer_alias(
                    "per_atom_cell_offsets_buffer",
                    per_atom_cell_offsets_buffer,
                    "per_atom_cell_offsets",
                    per_atom_cell_offsets,
                )
                compact_forward_kwargs = {
                    **forward_kwargs,
                    "inv_cell": tile_inv_cell,
                    "positions_wrapped": tile_positions_wrapped,
                    "per_atom_cell_offsets": tile_per_atom_cell_offsets,
                }
            forward_out = _naive_pair_outputs_forward(
                positions,
                cell_norm,
                **compact_forward_kwargs,
            )
            nm_out, nn_out, shifts_out = forward_out.extra_outputs
            distances_out = vectors_out = pe_out = pf_out = None
        else:
            route_out = _route_pair_outputs(
                positions,
                cell_norm,
                _naive_pair_outputs_forward,
                forward_kwargs,
            )
            # ``extra_outputs`` carries the per-pair energy/force tail only when
            # ``pair_fn`` is set, so the route return is 5 elements (geometry
            # only) or 7.
            if pair_fn is not None:
                (
                    distances_out,
                    vectors_out,
                    nm_out,
                    nn_out,
                    shifts_out,
                    pe_out,
                    pf_out,
                ) = route_out
            else:
                distances_out, vectors_out, nm_out, nn_out, shifts_out = route_out
                pe_out = pf_out = None
        if return_neighbor_list:
            active = nm_out != int(fill_value)
            if coo_capacity is None and pbc is not None:
                nl, nptr, nl_shifts = get_neighbor_list_from_neighbor_matrix(
                    nm_out,
                    num_neighbors=nn_out,
                    neighbor_shift_matrix=shifts_out,
                    fill_value=int(fill_value),
                )
                base = (nl, nptr, nl_shifts)
                plan = None
            elif coo_capacity is None:
                nl, nptr = get_neighbor_list_from_neighbor_matrix(
                    nm_out,
                    num_neighbors=nn_out,
                    fill_value=int(fill_value),
                )
                base = (nl, nptr)
                plan = None
            else:
                base, plan = _pack_fixed_capacity_neighbor_list_from_neighbor_matrix(
                    nm_out,
                    nn_out,
                    capacity=coo_capacity,
                    neighbor_shift_matrix=shifts_out if pbc is not None else None,
                    fill_value=int(fill_value),
                    metadata_valid=jnp.ones((), dtype=jnp.bool_),
                )
            # Repack per-pair geometry (and pair_fn outputs) into COO order aligned
            # with ``nl``.  Eager-only, like the index conversion.
            if distances_out is not None or vectors_out is not None:
                distances_out, vectors_out = coo_pack_pair_geometry(
                    active,
                    distances_out,
                    vectors_out,
                    capacity=coo_capacity,
                    plan=plan,
                )
            if pair_fn is not None:
                pe_out, pf_out = coo_pack_pair_geometry(
                    active, pe_out, pf_out, capacity=coo_capacity, plan=plan
                )
        elif pbc is not None:
            base = (nm_out, nn_out, shifts_out)
        else:
            base = (nm_out, nn_out)
        # Return tail mirrors the torch contract (torch/.../naive.py): optional
        # distances / vectors, then (pe, pf) whenever ``pair_fn`` is set.
        tail: list = []
        if return_distances:
            tail.append(distances_out)
        if return_vectors:
            tail.append(vectors_out)
        if pair_fn is not None:
            tail.extend((pe_out, pf_out))
        return (*base, *tail)

    if cell is not None:
        cell = cell if cell.ndim == 3 else cell[jnp.newaxis, :, :]
        # Ensure cell dtype matches positions dtype so Warp kernel dispatch is consistent
        if cell.dtype != positions.dtype:
            cell = cell.astype(positions.dtype)
    if pbc is not None:
        pbc = pbc if pbc.ndim == 2 else pbc[jnp.newaxis, :]

    # Resolve deprecated unsuffixed kwarg aliases.
    inv_cell = resolve_buffer_alias(
        "inv_cell_buffer",
        inv_cell_buffer,
        "inv_cell",
        inv_cell,
    )
    if _fixed_cell_geometry is not None and _fixed_cell_geometry[0] is not None:
        inv_cell = _fixed_cell_geometry[0]
    positions_wrapped = resolve_buffer_alias(
        "positions_wrapped_buffer",
        positions_wrapped_buffer,
        "positions_wrapped",
        positions_wrapped,
    )
    per_atom_cell_offsets = resolve_buffer_alias(
        "per_atom_cell_offsets_buffer",
        per_atom_cell_offsets_buffer,
        "per_atom_cell_offsets",
        per_atom_cell_offsets,
    )

    # Validate caller-supplied scratch buffers used by the wrap kernel. Shape
    # or dtype mismatches would silently break graph_mode="warp" cache replay
    # by changing input buffer pointers/layouts on every call, so reject them
    # early with a clear error.
    if inv_cell is not None:
        if inv_cell.shape != (1, 3, 3):
            raise ValueError(
                f"inv_cell_buffer must have shape (1, 3, 3) to match the internal "
                f"cell layout; got {inv_cell.shape}. A mismatched shape "
                f"silently breaks graph_mode='warp' cache replay."
            )
        if inv_cell.dtype != positions.dtype:
            raise ValueError(
                f"inv_cell_buffer dtype must match positions dtype "
                f"({positions.dtype}); got {inv_cell.dtype}."
            )
    if positions_wrapped is not None:
        expected_pw_shape = (positions.shape[0], 3)
        if positions_wrapped.shape != expected_pw_shape:
            raise ValueError(
                f"positions_wrapped_buffer must have shape {expected_pw_shape}; "
                f"got {positions_wrapped.shape}."
            )
        if positions_wrapped.dtype != positions.dtype:
            raise ValueError(
                f"positions_wrapped_buffer dtype must match positions dtype "
                f"({positions.dtype}); got {positions_wrapped.dtype}."
            )
    if per_atom_cell_offsets is not None:
        expected_off_shape = (positions.shape[0], 3)
        if per_atom_cell_offsets.shape != expected_off_shape:
            raise ValueError(
                f"per_atom_cell_offsets_buffer must have shape {expected_off_shape}; "
                f"got {per_atom_cell_offsets.shape}."
            )
        if per_atom_cell_offsets.dtype != jnp.int32:
            raise ValueError(
                f"per_atom_cell_offsets_buffer dtype must be int32; "
                f"got {per_atom_cell_offsets.dtype}."
            )

    if max_neighbors is None and (
        neighbor_matrix is None
        or (neighbor_matrix_shifts is None and pbc is not None)
        or num_neighbors is None
    ):
        max_neighbors = estimate_max_neighbors(cutoff)

    if fill_value is None:
        fill_value = positions.shape[0]

    if neighbor_matrix is None:
        neighbor_matrix = jnp.full(
            (positions.shape[0], max_neighbors),
            fill_value,
            dtype=jnp.int32,
        )
    elif rebuild_flags is None and graph_mode == "none":
        neighbor_matrix = neighbor_matrix.at[:].set(fill_value)

    if num_neighbors is None:
        num_neighbors = jnp.zeros(positions.shape[0], dtype=jnp.int32)
    elif rebuild_flags is None and graph_mode == "none":
        num_neighbors = num_neighbors.at[:].set(jnp.int32(0))

    if pbc is not None:
        if neighbor_matrix_shifts is None:
            neighbor_matrix_shifts = jnp.zeros(
                (positions.shape[0], max_neighbors, 3),
                dtype=jnp.int32,
            )
        elif rebuild_flags is None and graph_mode == "none":
            neighbor_matrix_shifts = neighbor_matrix_shifts.at[:].set(jnp.int32(0))
        if (
            max_shifts_per_system is None
            or num_shifts_per_system is None
            or shift_range_per_dimension is None
        ):
            shift_range_per_dimension, num_shifts_per_system, max_shifts_per_system = (
                compute_naive_num_shifts(cell, cutoff, pbc)
            )

    # Select wrap kernel by dtype; direct naive registrations resolve at launch.
    if positions.dtype == jnp.float64:
        _jax_wrap_single = _jax_wrap_positions_single_f64
    else:
        _jax_wrap_single = _jax_wrap_positions_single_f32
        positions = positions.astype(jnp.float32)

    positions = jax.lax.stop_gradient(positions)
    if cell is not None:
        cell = jax.lax.stop_gradient(cell)
    if inv_cell_buffer is not None:
        inv_cell_buffer = jax.lax.stop_gradient(inv_cell_buffer)
    if positions_wrapped_buffer is not None:
        positions_wrapped_buffer = jax.lax.stop_gradient(positions_wrapped_buffer)
    if per_atom_cell_offsets_buffer is not None:
        per_atom_cell_offsets_buffer = jax.lax.stop_gradient(
            per_atom_cell_offsets_buffer
        )

    total_atoms = positions.shape[0]
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

    if strategy == "tile":
        # Eager tile calls use pre-filled topology outputs and static launch scalars.
        cutoff_static = float(cutoff)
        if pbc is None:
            tile_callable = _GRAPH_NAIVE_TILE_CALLABLES[(False, False, positions.dtype)]
            neighbor_matrix, num_neighbors = tile_callable(
                positions,
                empty_target_indices,
                neighbor_matrix,
                num_neighbors,
                cutoff_static,
                half_fill,
                False,
            )
        else:
            if cell.dtype != positions.dtype:
                cell = cell.astype(positions.dtype)
            num_shifts = int(max_shifts_per_system)
            tile_callable = _GRAPH_NAIVE_TILE_CALLABLES[
                (True, bool(wrap_positions), positions.dtype)
            ]
            pbc_arg: tuple[jax.Array, ...] = ()
            if wrap_positions:
                inverse, positions_wrapped_work, offsets_work = (
                    _prepare_wrapped_tile_operands(
                        positions,
                        cell,
                        inv_cell_buffer=inv_cell,
                        positions_wrapped_buffer=positions_wrapped,
                        per_atom_cell_offsets_buffer=per_atom_cell_offsets,
                        fixed_cell_geometry=_fixed_cell_geometry,
                    )
                )
                pbc_arg = (pbc, inverse)
            tile_args = (
                positions,
                empty_target_indices,
                cell,
                *pbc_arg,
                shift_range_per_dimension,
                neighbor_matrix,
                neighbor_matrix_shifts,
                num_neighbors,
                *((positions_wrapped_work, offsets_work) if wrap_positions else ()),
                cutoff_static,
                num_shifts,
                half_fill,
                False,
            )
            tile_outs = tile_callable(*tile_args)
            neighbor_matrix, neighbor_matrix_shifts, num_neighbors = tile_outs[:3]
    elif graph_mode == "warp":
        has_pbc = pbc is not None
        is_selective = rebuild_flags is not None
        graph_callable = _GRAPH_NAIVE_WARP_CALLABLES[
            (has_pbc, wrap_positions, is_selective, positions.dtype)
        ]
        fill_value_i32 = int(fill_value)
        rf = None
        if is_selective:
            rf = rebuild_flags.flatten()[:1].astype(jnp.bool_)

        if not has_pbc:
            if is_selective:
                neighbor_matrix, num_neighbors = graph_callable(
                    positions,
                    neighbor_matrix,
                    num_neighbors,
                    cutoff_sq,
                    fill_value_i32,
                    half_fill,
                    rf,
                )
            else:
                neighbor_matrix, num_neighbors = graph_callable(
                    positions,
                    neighbor_matrix,
                    num_neighbors,
                    cutoff_sq,
                    fill_value_i32,
                    half_fill,
                )
        else:
            if cell.dtype != positions.dtype:
                cell = cell.astype(positions.dtype)

            num_shifts = int(max_shifts_per_system)
            if wrap_positions:
                if inv_cell is None:
                    inv_cell = jnp.linalg.inv(cell)
                if positions_wrapped is None:
                    positions_wrapped = jnp.zeros_like(positions)
                if per_atom_cell_offsets is None:
                    per_atom_cell_offsets = jnp.zeros((total_atoms, 3), dtype=jnp.int32)
                if is_selective:
                    (
                        positions_wrapped,
                        per_atom_cell_offsets,
                        neighbor_matrix,
                        neighbor_matrix_shifts,
                        num_neighbors,
                    ) = graph_callable(
                        positions,
                        cell,
                        inv_cell,
                        pbc,
                        shift_range_per_dimension,
                        positions_wrapped,
                        per_atom_cell_offsets,
                        neighbor_matrix,
                        neighbor_matrix_shifts,
                        num_neighbors,
                        cutoff_sq,
                        num_shifts,
                        fill_value_i32,
                        half_fill,
                        rf,
                    )
                else:
                    (
                        positions_wrapped,
                        per_atom_cell_offsets,
                        neighbor_matrix,
                        neighbor_matrix_shifts,
                        num_neighbors,
                    ) = graph_callable(
                        positions,
                        cell,
                        inv_cell,
                        pbc,
                        shift_range_per_dimension,
                        positions_wrapped,
                        per_atom_cell_offsets,
                        neighbor_matrix,
                        neighbor_matrix_shifts,
                        num_neighbors,
                        cutoff_sq,
                        num_shifts,
                        fill_value_i32,
                        half_fill,
                    )
            else:
                if is_selective:
                    neighbor_matrix, neighbor_matrix_shifts, num_neighbors = (
                        graph_callable(
                            positions,
                            cell,
                            shift_range_per_dimension,
                            neighbor_matrix,
                            neighbor_matrix_shifts,
                            num_neighbors,
                            cutoff_sq,
                            num_shifts,
                            fill_value_i32,
                            half_fill,
                            rf,
                        )
                    )
                else:
                    neighbor_matrix, neighbor_matrix_shifts, num_neighbors = (
                        graph_callable(
                            positions,
                            cell,
                            shift_range_per_dimension,
                            neighbor_matrix,
                            neighbor_matrix_shifts,
                            num_neighbors,
                            cutoff_sq,
                            num_shifts,
                            fill_value_i32,
                            half_fill,
                        )
                    )
    elif pbc is None:
        # No PBC case
        if rebuild_flags is not None:
            rf = rebuild_flags.flatten()[:1].astype(jnp.bool_)
            num_neighbors = jnp.where(
                rf[0], jnp.zeros_like(num_neighbors), num_neighbors
            )
            neighbor_matrix, num_neighbors = _DIRECT_NAIVE_KERNELS[
                ("none", True, bool(half_fill))
            ][positions.dtype](
                positions,
                empty_offsets,
                cutoff_sq,
                0.0,
                empty_cell,
                empty_shift_range,
                empty_num_shifts,
                empty_batch_idx,
                empty_batch_ptr,
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
                rf,
                launch_dims=(1, 1, total_atoms),
            )
        else:
            neighbor_matrix, num_neighbors = _DIRECT_NAIVE_KERNELS[
                ("none", False, bool(half_fill))
            ][positions.dtype](
                positions,
                empty_offsets,
                cutoff_sq,
                0.0,
                empty_cell,
                empty_shift_range,
                empty_num_shifts,
                empty_batch_idx,
                empty_batch_ptr,
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
                empty_rebuild_flags,
                launch_dims=(1, 1, total_atoms),
            )
    else:
        if cell.dtype != positions.dtype:
            cell = cell.astype(positions.dtype)

        if wrap_positions:
            if inv_cell is None:
                inv_cell = jnp.linalg.inv(cell)
            if positions_wrapped is None:
                positions_wrapped = jnp.zeros_like(positions)
            if per_atom_cell_offsets is None:
                per_atom_cell_offsets = jnp.zeros((total_atoms, 3), dtype=jnp.int32)
            positions_wrapped, per_atom_cell_offsets = _jax_wrap_single(
                positions,
                cell,
                inv_cell,
                pbc,
                jnp.empty((0,), dtype=jnp.int32),
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
                    _DIRECT_NAIVE_KERNELS[("wrap_on_entry", True, bool(half_fill))][
                        positions.dtype
                    ](
                        positions_wrapped,
                        per_atom_cell_offsets,
                        cutoff_sq,
                        0.0,
                        cell,
                        shift_range_per_dimension,
                        empty_num_shifts,
                        empty_batch_idx,
                        empty_batch_ptr,
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
                        rf,
                        launch_dims=(1, max_shifts_per_system, total_atoms),
                    )
                )
            else:
                neighbor_matrix, neighbor_matrix_shifts, num_neighbors = (
                    _DIRECT_NAIVE_KERNELS[("wrap_on_entry", False, bool(half_fill))][
                        positions.dtype
                    ](
                        positions_wrapped,
                        per_atom_cell_offsets,
                        cutoff_sq,
                        0.0,
                        cell,
                        shift_range_per_dimension,
                        empty_num_shifts,
                        empty_batch_idx,
                        empty_batch_ptr,
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
                        empty_rebuild_flags,
                        launch_dims=(1, max_shifts_per_system, total_atoms),
                    )
                )
        else:
            if rebuild_flags is not None:
                rf = rebuild_flags.flatten()[:1].astype(jnp.bool_)
                num_neighbors = jnp.where(
                    rf[0], jnp.zeros_like(num_neighbors), num_neighbors
                )
                neighbor_matrix, neighbor_matrix_shifts, num_neighbors = (
                    _DIRECT_NAIVE_KERNELS[("prewrapped", True, bool(half_fill))][
                        positions.dtype
                    ](
                        positions,
                        empty_offsets,
                        cutoff_sq,
                        0.0,
                        cell,
                        shift_range_per_dimension,
                        empty_num_shifts,
                        empty_batch_idx,
                        empty_batch_ptr,
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
                        rf,
                        launch_dims=(1, max_shifts_per_system, total_atoms),
                    )
                )
            else:
                neighbor_matrix, neighbor_matrix_shifts, num_neighbors = (
                    _DIRECT_NAIVE_KERNELS[("prewrapped", False, bool(half_fill))][
                        positions.dtype
                    ](
                        positions,
                        empty_offsets,
                        cutoff_sq,
                        0.0,
                        cell,
                        shift_range_per_dimension,
                        empty_num_shifts,
                        empty_batch_idx,
                        empty_batch_ptr,
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
                        empty_rebuild_flags,
                        launch_dims=(1, max_shifts_per_system, total_atoms),
                    )
                )

    if return_neighbor_list:
        if coo_capacity is not None and pbc is not None:
            return get_fixed_capacity_neighbor_list_from_neighbor_matrix(
                neighbor_matrix,
                num_neighbors=num_neighbors,
                capacity=coo_capacity,
                neighbor_shift_matrix=neighbor_matrix_shifts,
                fill_value=fill_value,
            )
        elif coo_capacity is not None:
            return get_fixed_capacity_neighbor_list_from_neighbor_matrix(
                neighbor_matrix,
                num_neighbors=num_neighbors,
                capacity=coo_capacity,
                fill_value=fill_value,
            )
        elif pbc is not None:
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
