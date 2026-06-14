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

import functools

import jax
import jax.numpy as jnp
import warp as wp
from warp.jax_experimental import GraphMode, jax_callable, jax_kernel

from nvalchemiops.jax.neighbors._autograd import (
    _build_index_residuals,
    _NeighborForwardOutput,
    _route_pair_outputs,
)
from nvalchemiops.jax.neighbors._dispatch import _is_jax_cpu_array
from nvalchemiops.jax.neighbors.neighbor_utils import (
    build_naive_kernel_tables,
    compute_naive_num_shifts,
    coo_pack_pair_geometry,
    get_neighbor_list_from_neighbor_matrix,
    prepare_batch_idx_ptr,
)
from nvalchemiops.neighbors.naive import (
    get_naive_neighbor_matrix_kernel as _get_naive_kernel,
)
from nvalchemiops.neighbors.naive.launchers import (
    _launch_naive_neighbor_matrix_no_pbc,
    _launch_naive_neighbor_matrix_pbc,
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

(
    _fill_batch_naive_neighbor_matrix_half_kernels,
    _fill_batch_naive_neighbor_matrix_selective_half_kernels,
    _fill_batch_naive_neighbor_matrix_pbc_half_kernels,
    _fill_batch_naive_neighbor_matrix_pbc_selective_half_kernels,
    _fill_batch_naive_neighbor_matrix_pbc_prewrapped_half_kernels,
    _fill_batch_naive_neighbor_matrix_pbc_prewrapped_selective_half_kernels,
) = build_naive_kernel_tables(
    "single_cutoff",
    batched=True,
    dtypes=_DTYPE_TO_BATCH_NAIVE_KERNELS,
    half_fill=True,
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

# Half-fill specializations of the pair-output kernels (selected when
# ``half_fill=True``; ``half_fill`` is a compile-time constant in the factory).
_fill_batch_naive_pair_half_kernels = {
    t: _get_naive_kernel(
        t,
        pbc_mode="none",
        batched=True,
        selective=False,
        return_vectors=True,
        return_distances=True,
        half_fill=True,
    )
    for t in _DTYPE_TO_BATCH_NAIVE_KERNELS
}
_fill_batch_naive_pbc_pair_half_kernels = {
    t: _get_naive_kernel(
        t,
        pbc_mode="wrap_on_entry",
        batched=True,
        selective=False,
        return_vectors=True,
        return_distances=True,
        half_fill=True,
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

# Half-fill batch naive neighbor matrix kernel wrappers
_jax_fill_batch_naive_half_f32 = jax_kernel(
    _fill_batch_naive_neighbor_matrix_half_kernels[wp.float32],
    num_outputs=2,
    in_out_argnames=["neighbor_matrix1", "num_neighbors1"],
    enable_backward=False,
)
_jax_fill_batch_naive_half_f64 = jax_kernel(
    _fill_batch_naive_neighbor_matrix_half_kernels[wp.float64],
    num_outputs=2,
    in_out_argnames=["neighbor_matrix1", "num_neighbors1"],
    enable_backward=False,
)
_jax_fill_batch_naive_pbc_half_f32 = jax_kernel(
    _fill_batch_naive_neighbor_matrix_pbc_half_kernels[wp.float32],
    num_outputs=3,
    in_out_argnames=["neighbor_matrix1", "neighbor_matrix_shifts1", "num_neighbors1"],
    enable_backward=False,
)
_jax_fill_batch_naive_pbc_half_f64 = jax_kernel(
    _fill_batch_naive_neighbor_matrix_pbc_half_kernels[wp.float64],
    num_outputs=3,
    in_out_argnames=["neighbor_matrix1", "neighbor_matrix_shifts1", "num_neighbors1"],
    enable_backward=False,
)
_jax_fill_batch_naive_selective_half_f32 = jax_kernel(
    _fill_batch_naive_neighbor_matrix_selective_half_kernels[wp.float32],
    num_outputs=2,
    in_out_argnames=["neighbor_matrix1", "num_neighbors1"],
    enable_backward=False,
)
_jax_fill_batch_naive_selective_half_f64 = jax_kernel(
    _fill_batch_naive_neighbor_matrix_selective_half_kernels[wp.float64],
    num_outputs=2,
    in_out_argnames=["neighbor_matrix1", "num_neighbors1"],
    enable_backward=False,
)
_jax_fill_batch_naive_pbc_selective_half_f32 = jax_kernel(
    _fill_batch_naive_neighbor_matrix_pbc_selective_half_kernels[wp.float32],
    num_outputs=3,
    in_out_argnames=["neighbor_matrix1", "neighbor_matrix_shifts1", "num_neighbors1"],
    enable_backward=False,
)
_jax_fill_batch_naive_pbc_selective_half_f64 = jax_kernel(
    _fill_batch_naive_neighbor_matrix_pbc_selective_half_kernels[wp.float64],
    num_outputs=3,
    in_out_argnames=["neighbor_matrix1", "neighbor_matrix_shifts1", "num_neighbors1"],
    enable_backward=False,
)
_jax_fill_batch_naive_pbc_prewrapped_half_f32 = jax_kernel(
    _fill_batch_naive_neighbor_matrix_pbc_prewrapped_half_kernels[wp.float32],
    num_outputs=3,
    in_out_argnames=["neighbor_matrix1", "neighbor_matrix_shifts1", "num_neighbors1"],
    enable_backward=False,
)
_jax_fill_batch_naive_pbc_prewrapped_half_f64 = jax_kernel(
    _fill_batch_naive_neighbor_matrix_pbc_prewrapped_half_kernels[wp.float64],
    num_outputs=3,
    in_out_argnames=["neighbor_matrix1", "neighbor_matrix_shifts1", "num_neighbors1"],
    enable_backward=False,
)
_jax_fill_batch_naive_pbc_prewrapped_selective_half_f32 = jax_kernel(
    _fill_batch_naive_neighbor_matrix_pbc_prewrapped_selective_half_kernels[wp.float32],
    num_outputs=3,
    in_out_argnames=["neighbor_matrix1", "neighbor_matrix_shifts1", "num_neighbors1"],
    enable_backward=False,
)
_jax_fill_batch_naive_pbc_prewrapped_selective_half_f64 = jax_kernel(
    _fill_batch_naive_neighbor_matrix_pbc_prewrapped_selective_half_kernels[wp.float64],
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

# Half-fill geometry-only pair-output callables (same I/O as the full-fill ones).
_jax_fill_batch_naive_pair_half_f32 = jax_kernel(
    _fill_batch_naive_pair_half_kernels[wp.float32],
    num_outputs=4,
    in_out_argnames=[
        "neighbor_matrix1",
        "num_neighbors1",
        "neighbor_vectors",
        "neighbor_distances",
    ],
    enable_backward=False,
)
_jax_fill_batch_naive_pair_half_f64 = jax_kernel(
    _fill_batch_naive_pair_half_kernels[wp.float64],
    num_outputs=4,
    in_out_argnames=[
        "neighbor_matrix1",
        "num_neighbors1",
        "neighbor_vectors",
        "neighbor_distances",
    ],
    enable_backward=False,
)
_jax_fill_batch_naive_pbc_pair_half_f32 = jax_kernel(
    _fill_batch_naive_pbc_pair_half_kernels[wp.float32],
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
_jax_fill_batch_naive_pbc_pair_half_f64 = jax_kernel(
    _fill_batch_naive_pbc_pair_half_kernels[wp.float64],
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


@functools.cache
def _get_jax_batch_naive_pair_fn_kernel(
    pair_fn, wp_dtype, pbc_mode: str, half_fill: bool = False
):
    """Build (and cache) a ``jax_kernel`` for a ``pair_fn``-specialized batched naive
    kernel.

    Mirrors ``naive._get_jax_naive_pair_fn_kernel`` with ``batched=True``: the kernel
    is specialized with ``pair_fn`` (so the ``HAS_PAIR_FN`` body runs) and
    ``pair_energies`` / ``pair_forces`` are registered as additional outputs.  Cached
    by ``(pair_fn identity, wp_dtype, pbc_mode)``; one recompile per distinct
    ``pair_fn``.
    """
    kernel = _get_naive_kernel(
        wp_dtype,
        pbc_mode=pbc_mode,
        batched=True,
        selective=False,
        return_vectors=True,
        return_distances=True,
        pair_fn=pair_fn,
        half_fill=half_fill,
    )
    if pbc_mode == "none":
        in_out_argnames = [
            "neighbor_matrix1",
            "num_neighbors1",
            "neighbor_vectors",
            "neighbor_distances",
            "pair_energies",
            "pair_forces",
        ]
    else:  # "wrap_on_entry"
        in_out_argnames = [
            "neighbor_matrix1",
            "neighbor_matrix_shifts1",
            "num_neighbors1",
            "neighbor_vectors",
            "neighbor_distances",
            "pair_energies",
            "pair_forces",
        ]
    return jax_kernel(
        kernel,
        num_outputs=len(in_out_argnames),
        in_out_argnames=in_out_argnames,
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


# ==============================================================================
# Tiled-kernel callables (``native_strategy="tile"``, CUDA-only)
# ==============================================================================
#
# These wrap the *inner* warp launchers ``_launch_naive_neighbor_matrix_no_pbc``
# / ``_launch_naive_neighbor_matrix_pbc`` (with ``batched=True``) inside a
# ``jax_callable`` body and pass ``native_strategy="tile"`` explicitly, so the
# tile-cooperative ``wp.launch_tiled`` kernel is honored unconditionally
# (unlike the "auto" heuristic, which only tiles for few-large-systems).
#
# Mirrors the single-system tile callables in
# ``nvalchemiops.jax.neighbors.naive`` but for the batched launchers, with two
# batched-specific differences:
#
#   * The launchers square ``cutoff`` internally, so the bodies pass the RAW
#     cutoff (NOT ``cutoff**2``) — unlike the surrounding scalar batched path,
#     which passes ``cutoff*cutoff``.
#   * The wrapped-PBC body passes RAW (unwrapped) ``positions`` with
#     ``wrap_positions=True``; the launcher wraps internally using
#     ``batch_idx`` to pick per-atom cells.  No JAX-side pre-wrap is done on the
#     tile path (that would double-wrap).
#
# Batched PREWRAPPED PBC has no tiled kernel (``_make_tile_kernel`` raises), so
# only no-PBC and wrapped-PBC are wired here; the dispatch site rejects
# ``native_strategy="tile"`` + ``wrap_positions=False`` for PBC.
#
# These run only on the eager path, where ``batch_naive_neighbor_list`` already
# pre-fills ``neighbor_matrix=fill_value`` and zeroes ``num_neighbors`` /
# shifts before dispatch, so the bodies perform no reset and take no
# ``fill_value`` argument.  Tile supports ``half_fill`` but has no pair-output /
# ``target_indices`` / selective (``rebuild_flags``) variant.  The static
# scalars (``cutoff``, ``half_fill``, ``max_shifts_per_system`` /
# ``max_atoms_per_system`` for PBC) are concrete host ints computed outside jit.


def _batch_naive_tile_no_pbc_f32(
    positions: wp.array(dtype=wp.vec3f),
    batch_idx: wp.array(dtype=wp.int32),
    batch_ptr: wp.array(dtype=wp.int32),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    cutoff: wp.float32,
    half_fill: wp.bool,
) -> None:
    _launch_naive_neighbor_matrix_no_pbc(
        positions,
        float(cutoff),
        neighbor_matrix,
        num_neighbors,
        wp.float32,
        str(positions.device),
        batched=True,
        batch_idx=batch_idx,
        batch_ptr=batch_ptr,
        half_fill=bool(half_fill),
        native_strategy="tile",
    )


def _batch_naive_tile_no_pbc_f64(
    positions: wp.array(dtype=wp.vec3d),
    batch_idx: wp.array(dtype=wp.int32),
    batch_ptr: wp.array(dtype=wp.int32),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    cutoff: wp.float64,
    half_fill: wp.bool,
) -> None:
    _launch_naive_neighbor_matrix_no_pbc(
        positions,
        float(cutoff),
        neighbor_matrix,
        num_neighbors,
        wp.float64,
        str(positions.device),
        batched=True,
        batch_idx=batch_idx,
        batch_ptr=batch_ptr,
        half_fill=bool(half_fill),
        native_strategy="tile",
    )


def _batch_naive_tile_pbc_wrapped_f32(
    positions: wp.array(dtype=wp.vec3f),
    cell: wp.array(dtype=wp.mat33f),
    pbc: wp.array2d(dtype=wp.bool),
    shift_range: wp.array(dtype=wp.vec3i),
    num_shifts_arr: wp.array(dtype=wp.int32),
    batch_idx: wp.array(dtype=wp.int32),
    batch_ptr: wp.array(dtype=wp.int32),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    cutoff: wp.float32,
    max_shifts_per_system: wp.int32,
    max_atoms_per_system: wp.int32,
    half_fill: wp.bool,
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
        batched=True,
        batch_ptr=batch_ptr,
        batch_idx=batch_idx,
        num_shifts_arr=num_shifts_arr,
        max_shifts_per_system=int(max_shifts_per_system),
        max_atoms_per_system=int(max_atoms_per_system),
        half_fill=bool(half_fill),
        wrap_positions=True,
        native_strategy="tile",
    )


def _batch_naive_tile_pbc_wrapped_f64(
    positions: wp.array(dtype=wp.vec3d),
    cell: wp.array(dtype=wp.mat33d),
    pbc: wp.array2d(dtype=wp.bool),
    shift_range: wp.array(dtype=wp.vec3i),
    num_shifts_arr: wp.array(dtype=wp.int32),
    batch_idx: wp.array(dtype=wp.int32),
    batch_ptr: wp.array(dtype=wp.int32),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    cutoff: wp.float64,
    max_shifts_per_system: wp.int32,
    max_atoms_per_system: wp.int32,
    half_fill: wp.bool,
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
        batched=True,
        batch_ptr=batch_ptr,
        batch_idx=batch_idx,
        num_shifts_arr=num_shifts_arr,
        max_shifts_per_system=int(max_shifts_per_system),
        max_atoms_per_system=int(max_atoms_per_system),
        half_fill=bool(half_fill),
        wrap_positions=True,
        native_strategy="tile",
    )


# Keyed by ``(has_pbc, wrap_positions)``.  Only no-PBC and wrapped-PBC are
# present; batched prewrapped PBC has no tiled kernel.  Tile has no selective
# variant, so the selective axis is omitted; ``native_strategy="tile"`` rejects
# ``rebuild_flags`` at the dispatch site.
_BATCH_NAIVE_TILE_NO_PBC_IN_OUT_ARGS = ("neighbor_matrix", "num_neighbors")
_BATCH_NAIVE_TILE_PBC_IN_OUT_ARGS = (
    "neighbor_matrix",
    "neighbor_matrix_shifts",
    "num_neighbors",
)
_BATCH_NAIVE_TILE_SPECS = {
    (False, False): {
        "num_outputs": 2,
        "in_out_argnames": _BATCH_NAIVE_TILE_NO_PBC_IN_OUT_ARGS,
        jnp.dtype(jnp.float32): _batch_naive_tile_no_pbc_f32,
        jnp.dtype(jnp.float64): _batch_naive_tile_no_pbc_f64,
    },
    (True, True): {
        "num_outputs": 3,
        "in_out_argnames": _BATCH_NAIVE_TILE_PBC_IN_OUT_ARGS,
        jnp.dtype(jnp.float32): _batch_naive_tile_pbc_wrapped_f32,
        jnp.dtype(jnp.float64): _batch_naive_tile_pbc_wrapped_f64,
    },
}


def _register_batch_naive_tile_callables() -> dict[
    tuple[bool, bool, jnp.dtype], object
]:
    """Register GraphMode.NONE tile callables for the batched naive eager path.

    ``GraphMode.NONE`` (not WARP): the tile bodies assume the caller has
    already pre-filled the output buffers, which the eager
    ``batch_naive_neighbor_list`` path does before dispatch.
    """
    registered: dict[tuple[bool, bool, jnp.dtype], object] = {}
    for (has_pbc, wrap_positions), spec in _BATCH_NAIVE_TILE_SPECS.items():
        for dtype in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
            registered[(has_pbc, wrap_positions, dtype)] = jax_callable(
                spec[dtype],
                num_outputs=spec["num_outputs"],
                in_out_argnames=spec["in_out_argnames"],
                graph_mode=GraphMode.NONE,
            )
    return registered


_BATCH_NAIVE_TILE_CALLABLES = _register_batch_naive_tile_callables()


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
    pair_fn=None,
    pair_params: jax.Array | None = None,
    half_fill: bool = False,
) -> _NeighborForwardOutput:
    """Forward closure for the batch_naive autograd path.

    When ``pair_fn`` is set, a ``pair_fn``-specialized kernel writes per-pair
    ``pair_energies`` / ``pair_forces`` which ride along in ``extra_outputs``
    (forward-only); see ``naive._naive_pair_outputs_forward``.
    """
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

    # ``pair_fn`` path: real per-atom params + auto-allocated energy/force buffers
    # (returned via ``extra_outputs``, forward-only).
    has_pair_fn = pair_fn is not None
    if has_pair_fn:
        wp_dtype = wp.float64 if f64 else wp.float32
        pp_arg = jnp.asarray(pair_params, dtype=positions.dtype)
        pe = jnp.zeros((total_atoms, max_neighbors), dtype=positions.dtype)
        pf = jnp.zeros((total_atoms, max_neighbors, 3), dtype=positions.dtype)
    else:
        pp_arg = empty_pair_params
        pe = None
        pf = None

    if pbc is None:
        if has_pair_fn:
            kernel = _get_jax_batch_naive_pair_fn_kernel(
                pair_fn, wp_dtype, "none", half_fill
            )
        elif half_fill:
            kernel = (
                _jax_fill_batch_naive_pair_half_f64
                if f64
                else _jax_fill_batch_naive_pair_half_f32
            )
        else:
            kernel = (
                _jax_fill_batch_naive_pair_f64
                if f64
                else _jax_fill_batch_naive_pair_f32
            )
        outs = kernel(
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
            pp_arg,
            pe if has_pair_fn else empty_energies,
            pf if has_pair_fn else empty_forces,
            empty_rebuild_flags,
            launch_dims=(1, 1, total_atoms),
        )
        if has_pair_fn:
            nm, nn, nv, nd, pe, pf = outs
        else:
            nm, nn, nv, nd = outs
        nms = jnp.zeros((total_atoms, max_neighbors, 3), dtype=jnp.int32)
    else:
        if has_pair_fn:
            kernel = _get_jax_batch_naive_pair_fn_kernel(
                pair_fn, wp_dtype, "wrap_on_entry", half_fill
            )
        elif half_fill:
            kernel = (
                _jax_fill_batch_naive_pbc_pair_half_f64
                if f64
                else _jax_fill_batch_naive_pbc_pair_half_f32
            )
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
            pbc,
            batch_idx_i32,
            positions_wrapped,
            per_atom_cell_offsets,
            launch_dims=(total_atoms,),
        )
        outs = kernel(
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
            pp_arg,
            pe if has_pair_fn else empty_energies,
            pf if has_pair_fn else empty_forces,
            empty_rebuild_flags,
            launch_dims=(
                num_systems,
                max_shifts_per_system,
                max_atoms_per_system,
            ),
        )
        if has_pair_fn:
            nm, nms, nn, nv, nd, pe, pf = outs
        else:
            nm, nms, nn, nv, nd = outs

    i_idx, j_idx, shifts_ret, _, mask_ = _build_index_residuals(nm, nn, nms)
    K, M = nm.shape
    extra_outputs = (nm, nn, nms, pe, pf) if has_pair_fn else (nm, nn, nms)
    return _NeighborForwardOutput(
        distances=nd,
        vectors=nv,
        extra_outputs=extra_outputs,
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
    native_strategy: str = "auto",
    *,
    return_distances: bool = False,
    return_vectors: bool = False,
    pair_fn=None,
    pair_params: jax.Array | None = None,
    pair_energies: jax.Array | None = None,
    pair_forces: jax.Array | None = None,
) -> tuple[jax.Array, ...]:
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
    native_strategy : {"auto", "scalar", "tile"}, default="auto"
        Selects the underlying Warp kernel variant. ``"scalar"`` uses the
        per-atom scalar kernel. ``"tile"`` uses the tile-cooperative
        ``wp.launch_tiled`` kernel and is **CUDA-only**: requesting it on a
        CPU device raises ``ValueError``. The tile path supports the no-PBC
        and PBC-wrapped (``wrap_positions=True``) cases and ``half_fill``, but
        has no pair-output (``return_distances`` / ``return_vectors``) or
        selective (``rebuild_flags``) variant, and there is **no batched
        prewrapped-PBC tiled kernel**: requesting ``native_strategy="tile"``
        with PBC and ``wrap_positions=False`` raises ``NotImplementedError``
        (use ``"scalar"`` for that combination). ``"auto"`` and ``"scalar"``
        preserve the current scalar-dispatch behavior; ``"auto"`` never selects
        tile in this binding (tile is opt-in). The tile and scalar paths
        produce identical pair *sets* (per-row ordering may differ; under
        ``half_fill`` the two pick opposite pair owners, yielding the same
        undirected set with sign-flipped shifts).

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
    if native_strategy not in {"auto", "scalar", "tile"}:
        raise ValueError(
            "native_strategy must be 'auto' | 'scalar' | 'tile', "
            f"got {native_strategy!r}",
        )

    if pbc is None and cell is not None:
        raise ValueError("If cell is provided, pbc must also be provided")
    if pbc is not None and cell is None:
        raise ValueError("If pbc is provided, cell must also be provided")

    if native_strategy == "tile":
        # The tile-cooperative kernel is CUDA-only and has no pair-output or
        # selective (rebuild_flags) variant, and no batched prewrapped-PBC
        # tiled kernel.  Gate here, before any launch, mirroring the warp
        # launcher CPU guard and the single-system tile guards.  (batch_naive
        # has no graph_mode / target_indices params, so those guards do not
        # apply here.)
        if _is_jax_cpu_array(positions):
            raise ValueError(
                "native_strategy='tile' requires CUDA; the tile-cooperative "
                "naive kernel cannot run on a CPU device (Warp forces "
                "block_dim=1). Use native_strategy='scalar' or 'auto' on CPU.",
            )
        if bool(return_distances) or bool(return_vectors):
            raise NotImplementedError(
                "native_strategy='tile' has no pair-output (return_distances / "
                "return_vectors) variant; use native_strategy='scalar'.",
            )
        if rebuild_flags is not None:
            raise NotImplementedError(
                "native_strategy='tile' has no selective (rebuild_flags) "
                "variant; use native_strategy='scalar'.",
            )
        if pbc is not None and not wrap_positions:
            raise NotImplementedError(
                "native_strategy='tile' has no batched prewrapped-PBC tiled "
                "kernel (wrap_positions=False with PBC). Use "
                "native_strategy='scalar', or wrap_positions=True for the "
                "tile path.",
            )

    # Prepare batch indices and pointers
    batch_idx, batch_ptr = prepare_batch_idx_ptr(
        batch_idx, batch_ptr, positions.shape[0]
    )
    num_systems = batch_ptr.shape[0] - 1

    if pair_fn is not None and pair_params is None:
        raise ValueError(
            "pair_fn requires pair_params (a per-atom (n_atoms, K) parameter array).",
        )
    has_pair_outputs = (
        bool(return_distances) or bool(return_vectors) or pair_fn is not None
    )
    if has_pair_outputs:
        if rebuild_flags is not None:
            raise NotImplementedError(
                "Pair outputs are not supported with rebuild_flags.",
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
            "pair_fn": pair_fn,
            "pair_params": pair_params,
            "half_fill": bool(half_fill),
        }
        route_out = _route_pair_outputs(
            positions,
            cell_norm,
            _batch_naive_pair_outputs_forward,
            forward_kwargs,
        )
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
            if pbc is not None:
                nl, nptr, nl_shifts = get_neighbor_list_from_neighbor_matrix(
                    nm_out,
                    num_neighbors=nn_out,
                    neighbor_shift_matrix=shifts_out,
                    fill_value=int(fill_value),
                )
                base = (nl, nptr, nl_shifts)
            else:
                nl, nptr = get_neighbor_list_from_neighbor_matrix(
                    nm_out,
                    num_neighbors=nn_out,
                    fill_value=int(fill_value),
                )
                base = (nl, nptr)
            # Repack per-pair geometry (and pair_fn outputs) into the same COO order
            # as ``nl``.  Eager-only, like the index conversion.
            active = nm_out != int(fill_value)
            distances_out, vectors_out = coo_pack_pair_geometry(
                active, distances_out, vectors_out
            )
            if pair_fn is not None:
                pe_out, pf_out = coo_pack_pair_geometry(active, pe_out, pf_out)
        elif pbc is not None:
            base = (nm_out, nn_out, shifts_out)
        else:
            base = (nm_out, nn_out)
        # Return tail mirrors the torch contract: optional distances / vectors, then
        # (pe, pf) whenever ``pair_fn`` is set.
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

    # Select kernel based on dtype and static half-fill specialization.
    if positions.dtype == jnp.float64:
        if half_fill:
            _jax_fill = _jax_fill_batch_naive_half_f64
            _jax_fill_pbc = _jax_fill_batch_naive_pbc_half_f64
            _jax_fill_selective = _jax_fill_batch_naive_selective_half_f64
            _jax_fill_pbc_selective = _jax_fill_batch_naive_pbc_selective_half_f64
            _jax_fill_pbc_prewrapped = _jax_fill_batch_naive_pbc_prewrapped_half_f64
            _jax_fill_pbc_prewrapped_selective = (
                _jax_fill_batch_naive_pbc_prewrapped_selective_half_f64
            )
        else:
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
        if half_fill:
            _jax_fill = _jax_fill_batch_naive_half_f32
            _jax_fill_pbc = _jax_fill_batch_naive_pbc_half_f32
            _jax_fill_selective = _jax_fill_batch_naive_selective_half_f32
            _jax_fill_pbc_selective = _jax_fill_batch_naive_pbc_selective_half_f32
            _jax_fill_pbc_prewrapped = _jax_fill_batch_naive_pbc_prewrapped_half_f32
            _jax_fill_pbc_prewrapped_selective = (
                _jax_fill_batch_naive_pbc_prewrapped_selective_half_f32
            )
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

    if native_strategy == "tile":
        # CUDA-only tile-cooperative path (eager only). Output buffers were
        # already pre-filled above; the callable bodies wrap the batched inner
        # warp launchers with native_strategy="tile". The launchers square the
        # cutoff internally, so the RAW cutoff is threaded as a static scalar
        # (NOT cutoff*cutoff, unlike the scalar arms below).
        cutoff_static = float(cutoff)
        if pbc is None:
            tile_callable = _BATCH_NAIVE_TILE_CALLABLES[(False, False, positions.dtype)]
            neighbor_matrix, num_neighbors = tile_callable(
                positions,
                batch_idx_i32,
                batch_ptr_i32,
                neighbor_matrix,
                num_neighbors,
                cutoff_static,
                half_fill,
            )
        else:
            # Wrapped-PBC tile (prewrapped already rejected at the guard). The
            # launcher wraps RAW positions internally using batch_idx, so no
            # JAX-side pre-wrap is done here (that would double-wrap).
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
                        "Please provide max_atoms_per_system explicitly when "
                        "using jax.jit with native_strategy='tile'."
                    ) from None
            tile_callable = _BATCH_NAIVE_TILE_CALLABLES[(True, True, positions.dtype)]
            neighbor_matrix, neighbor_matrix_shifts, num_neighbors = tile_callable(
                positions,
                cell,
                pbc,
                shift_range_per_dimension,
                num_shifts_per_system,
                batch_idx_i32,
                batch_ptr_i32,
                neighbor_matrix,
                neighbor_matrix_shifts,
                num_neighbors,
                cutoff_static,
                int(max_shifts_per_system),
                int(max_atoms_per_system),
                half_fill,
            )
    elif pbc is None:
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
                pbc,
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
