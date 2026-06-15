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
import warp as wp
from warp.jax_experimental import GraphMode, jax_callable, jax_kernel

from nvalchemiops.jax.neighbors._autograd import (
    _build_index_residuals,
    _NeighborForwardOutput,
    _route_pair_outputs,
)
from nvalchemiops.jax.neighbors._dispatch import _is_jax_cpu_array
from nvalchemiops.jax.neighbors.neighbor_utils import (
    _validate_graph_mode,
    build_naive_kernel_tables,
    compute_naive_num_shifts,
    coo_pack_pair_geometry,
    get_neighbor_list_from_neighbor_matrix,
)
from nvalchemiops.neighbors.naive import (
    get_naive_neighbor_matrix_kernel as _get_naive_kernel,
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

# Pair-output variants — produced by the same factory but with
# ``return_vectors`` / ``return_distances`` flipped on.  Used by the
# autograd path in :mod:`nvalchemiops.jax.neighbors._autograd`.
#
# The PBC variant is hard-wired to ``pbc_mode='wrap_on_entry'``.  The kernel
# is idempotent on already-wrapped positions and produces correct shifts
# for raw (unwrapped) positions as well, so the autograd path silently
# ignores the public ``wrap_positions`` kwarg.  Callers who pre-wrap to save
# the two extra kernel launches lose that optimization on the autograd
# path but retain numerical equivalence.

_fill_naive_pair_kernels = {
    t: _get_naive_kernel(
        t,
        pbc_mode="none",
        batched=False,
        selective=False,
        return_vectors=True,
        return_distances=True,
    )
    for t in _DTYPE_TO_NAIVE_KERNELS
}
_fill_naive_pbc_pair_kernels = {
    t: _get_naive_kernel(
        t,
        pbc_mode="wrap_on_entry",
        batched=False,
        selective=False,
        return_vectors=True,
        return_distances=True,
    )
    for t in _DTYPE_TO_NAIVE_KERNELS
}

# Half-fill specializations of the pair-output kernels (``half_fill`` is a
# compile-time constant in the Warp factory, so honoring it needs a distinct
# kernel).  Selected by the forward when ``half_fill=True``.
_fill_naive_pair_half_kernels = {
    t: _get_naive_kernel(
        t,
        pbc_mode="none",
        batched=False,
        selective=False,
        return_vectors=True,
        return_distances=True,
        half_fill=True,
    )
    for t in _DTYPE_TO_NAIVE_KERNELS
}
_fill_naive_pbc_pair_half_kernels = {
    t: _get_naive_kernel(
        t,
        pbc_mode="wrap_on_entry",
        batched=False,
        selective=False,
        return_vectors=True,
        return_distances=True,
        half_fill=True,
    )
    for t in _DTYPE_TO_NAIVE_KERNELS
}

__all__ = ["naive_neighbor_list"]

# ==============================================================================
# JAX Kernel Wrappers
# ==============================================================================

# No-PBC naive neighbor matrix kernel wrappers
_jax_fill_naive_f32 = jax_kernel(
    _fill_naive_neighbor_matrix_kernels[wp.float32],
    num_outputs=2,
    in_out_argnames=["neighbor_matrix1", "num_neighbors1"],
    enable_backward=False,
)
_jax_fill_naive_f64 = jax_kernel(
    _fill_naive_neighbor_matrix_kernels[wp.float64],
    num_outputs=2,
    in_out_argnames=["neighbor_matrix1", "num_neighbors1"],
    enable_backward=False,
)

# PBC naive neighbor matrix kernel wrappers
_jax_fill_naive_pbc_f32 = jax_kernel(
    _fill_naive_neighbor_matrix_pbc_kernels[wp.float32],
    num_outputs=3,
    in_out_argnames=["neighbor_matrix1", "neighbor_matrix_shifts1", "num_neighbors1"],
    enable_backward=False,
)
_jax_fill_naive_pbc_f64 = jax_kernel(
    _fill_naive_neighbor_matrix_pbc_kernels[wp.float64],
    num_outputs=3,
    in_out_argnames=["neighbor_matrix1", "neighbor_matrix_shifts1", "num_neighbors1"],
    enable_backward=False,
)

# Selective no-PBC naive neighbor matrix kernel wrappers
_jax_fill_naive_selective_f32 = jax_kernel(
    _fill_naive_neighbor_matrix_selective_kernels[wp.float32],
    num_outputs=2,
    in_out_argnames=["neighbor_matrix1", "num_neighbors1"],
    enable_backward=False,
)
_jax_fill_naive_selective_f64 = jax_kernel(
    _fill_naive_neighbor_matrix_selective_kernels[wp.float64],
    num_outputs=2,
    in_out_argnames=["neighbor_matrix1", "num_neighbors1"],
    enable_backward=False,
)

# Selective PBC naive neighbor matrix kernel wrappers
_jax_fill_naive_pbc_selective_f32 = jax_kernel(
    _fill_naive_neighbor_matrix_pbc_selective_kernels[wp.float32],
    num_outputs=3,
    in_out_argnames=["neighbor_matrix1", "neighbor_matrix_shifts1", "num_neighbors1"],
    enable_backward=False,
)
_jax_fill_naive_pbc_selective_f64 = jax_kernel(
    _fill_naive_neighbor_matrix_pbc_selective_kernels[wp.float64],
    num_outputs=3,
    in_out_argnames=["neighbor_matrix1", "neighbor_matrix_shifts1", "num_neighbors1"],
    enable_backward=False,
)

# PBC prewrapped naive neighbor matrix kernel wrappers
_jax_fill_naive_pbc_prewrapped_f32 = jax_kernel(
    _fill_naive_neighbor_matrix_pbc_prewrapped_kernels[wp.float32],
    num_outputs=3,
    in_out_argnames=["neighbor_matrix1", "neighbor_matrix_shifts1", "num_neighbors1"],
    enable_backward=False,
)
_jax_fill_naive_pbc_prewrapped_f64 = jax_kernel(
    _fill_naive_neighbor_matrix_pbc_prewrapped_kernels[wp.float64],
    num_outputs=3,
    in_out_argnames=["neighbor_matrix1", "neighbor_matrix_shifts1", "num_neighbors1"],
    enable_backward=False,
)

# Selective PBC prewrapped naive neighbor matrix kernel wrappers
_jax_fill_naive_pbc_prewrapped_selective_f32 = jax_kernel(
    _fill_naive_neighbor_matrix_pbc_prewrapped_selective_kernels[wp.float32],
    num_outputs=3,
    in_out_argnames=["neighbor_matrix1", "neighbor_matrix_shifts1", "num_neighbors1"],
    enable_backward=False,
)
_jax_fill_naive_pbc_prewrapped_selective_f64 = jax_kernel(
    _fill_naive_neighbor_matrix_pbc_prewrapped_selective_kernels[wp.float64],
    num_outputs=3,
    in_out_argnames=["neighbor_matrix1", "neighbor_matrix_shifts1", "num_neighbors1"],
    enable_backward=False,
)

# Half-fill naive neighbor matrix kernel wrappers
_jax_fill_naive_half_f32 = jax_kernel(
    _fill_naive_neighbor_matrix_half_kernels[wp.float32],
    num_outputs=2,
    in_out_argnames=["neighbor_matrix1", "num_neighbors1"],
    enable_backward=False,
)
_jax_fill_naive_half_f64 = jax_kernel(
    _fill_naive_neighbor_matrix_half_kernels[wp.float64],
    num_outputs=2,
    in_out_argnames=["neighbor_matrix1", "num_neighbors1"],
    enable_backward=False,
)
_jax_fill_naive_pbc_half_f32 = jax_kernel(
    _fill_naive_neighbor_matrix_pbc_half_kernels[wp.float32],
    num_outputs=3,
    in_out_argnames=["neighbor_matrix1", "neighbor_matrix_shifts1", "num_neighbors1"],
    enable_backward=False,
)
_jax_fill_naive_pbc_half_f64 = jax_kernel(
    _fill_naive_neighbor_matrix_pbc_half_kernels[wp.float64],
    num_outputs=3,
    in_out_argnames=["neighbor_matrix1", "neighbor_matrix_shifts1", "num_neighbors1"],
    enable_backward=False,
)
_jax_fill_naive_selective_half_f32 = jax_kernel(
    _fill_naive_neighbor_matrix_selective_half_kernels[wp.float32],
    num_outputs=2,
    in_out_argnames=["neighbor_matrix1", "num_neighbors1"],
    enable_backward=False,
)
_jax_fill_naive_selective_half_f64 = jax_kernel(
    _fill_naive_neighbor_matrix_selective_half_kernels[wp.float64],
    num_outputs=2,
    in_out_argnames=["neighbor_matrix1", "num_neighbors1"],
    enable_backward=False,
)
_jax_fill_naive_pbc_selective_half_f32 = jax_kernel(
    _fill_naive_neighbor_matrix_pbc_selective_half_kernels[wp.float32],
    num_outputs=3,
    in_out_argnames=["neighbor_matrix1", "neighbor_matrix_shifts1", "num_neighbors1"],
    enable_backward=False,
)
_jax_fill_naive_pbc_selective_half_f64 = jax_kernel(
    _fill_naive_neighbor_matrix_pbc_selective_half_kernels[wp.float64],
    num_outputs=3,
    in_out_argnames=["neighbor_matrix1", "neighbor_matrix_shifts1", "num_neighbors1"],
    enable_backward=False,
)
_jax_fill_naive_pbc_prewrapped_half_f32 = jax_kernel(
    _fill_naive_neighbor_matrix_pbc_prewrapped_half_kernels[wp.float32],
    num_outputs=3,
    in_out_argnames=["neighbor_matrix1", "neighbor_matrix_shifts1", "num_neighbors1"],
    enable_backward=False,
)
_jax_fill_naive_pbc_prewrapped_half_f64 = jax_kernel(
    _fill_naive_neighbor_matrix_pbc_prewrapped_half_kernels[wp.float64],
    num_outputs=3,
    in_out_argnames=["neighbor_matrix1", "neighbor_matrix_shifts1", "num_neighbors1"],
    enable_backward=False,
)
_jax_fill_naive_pbc_prewrapped_selective_half_f32 = jax_kernel(
    _fill_naive_neighbor_matrix_pbc_prewrapped_selective_half_kernels[wp.float32],
    num_outputs=3,
    in_out_argnames=["neighbor_matrix1", "neighbor_matrix_shifts1", "num_neighbors1"],
    enable_backward=False,
)
_jax_fill_naive_pbc_prewrapped_selective_half_f64 = jax_kernel(
    _fill_naive_neighbor_matrix_pbc_prewrapped_selective_half_kernels[wp.float64],
    num_outputs=3,
    in_out_argnames=["neighbor_matrix1", "neighbor_matrix_shifts1", "num_neighbors1"],
    enable_backward=False,
)

# Pair-output kernel wrappers (no PBC).  Returns 4 outputs: neighbor_matrix,
# num_neighbors, neighbor_vectors, neighbor_distances.
_jax_fill_naive_pair_f32 = jax_kernel(
    _fill_naive_pair_kernels[wp.float32],
    num_outputs=4,
    in_out_argnames=[
        "neighbor_matrix1",
        "num_neighbors1",
        "neighbor_vectors",
        "neighbor_distances",
    ],
    enable_backward=False,
)
_jax_fill_naive_pair_f64 = jax_kernel(
    _fill_naive_pair_kernels[wp.float64],
    num_outputs=4,
    in_out_argnames=[
        "neighbor_matrix1",
        "num_neighbors1",
        "neighbor_vectors",
        "neighbor_distances",
    ],
    enable_backward=False,
)

# Pair-output kernel wrappers (PBC, wrap-on-entry mode).  Returns 5 outputs:
# adds neighbor_matrix_shifts.
_jax_fill_naive_pbc_pair_f32 = jax_kernel(
    _fill_naive_pbc_pair_kernels[wp.float32],
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
_jax_fill_naive_pbc_pair_f64 = jax_kernel(
    _fill_naive_pbc_pair_kernels[wp.float64],
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
_jax_fill_naive_pair_half_f32 = jax_kernel(
    _fill_naive_pair_half_kernels[wp.float32],
    num_outputs=4,
    in_out_argnames=[
        "neighbor_matrix1",
        "num_neighbors1",
        "neighbor_vectors",
        "neighbor_distances",
    ],
    enable_backward=False,
)
_jax_fill_naive_pair_half_f64 = jax_kernel(
    _fill_naive_pair_half_kernels[wp.float64],
    num_outputs=4,
    in_out_argnames=[
        "neighbor_matrix1",
        "num_neighbors1",
        "neighbor_vectors",
        "neighbor_distances",
    ],
    enable_backward=False,
)
_jax_fill_naive_pbc_pair_half_f32 = jax_kernel(
    _fill_naive_pbc_pair_half_kernels[wp.float32],
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
_jax_fill_naive_pbc_pair_half_f64 = jax_kernel(
    _fill_naive_pbc_pair_half_kernels[wp.float64],
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
def _get_jax_naive_pair_fn_kernel(
    pair_fn, wp_dtype, pbc_mode: str, half_fill: bool = False
):
    """Build (and cache) a ``jax_kernel`` for a ``pair_fn``-specialized naive kernel.

    The naive kernel signature always carries ``pair_params`` / ``pair_energies`` /
    ``pair_forces`` slots; specializing the factory with ``pair_fn`` flips the
    compile-time ``HAS_PAIR_FN`` constant on so the body actually evaluates the user
    function and writes the energy/force buffers.  Here we register those two buffers
    as additional outputs (so JAX returns them).

    Cached by ``(pair_fn identity, wp_dtype, pbc_mode)`` — Warp ``@wp.func`` objects
    are hashable by identity, so a module-scope singleton ``pair_fn`` recompiles only
    once.  ``jax_kernel`` (rather than ``jax_callable``) mirrors the geometry-only
    pair-output path above: the kernel is fully specialized, so no launcher closure is
    needed.
    """
    kernel = _get_naive_kernel(
        wp_dtype,
        pbc_mode=pbc_mode,
        batched=False,
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
    """Register GraphMode.WARP callables for all naive graph-mode paths."""
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
                graph_mode=GraphMode.WARP,
            )
            for wrap_positions in wrap_positions_values:
                registered[(has_pbc, wrap_positions, selective, dtype)] = callable_obj

    return registered


_GRAPH_NAIVE_WARP_CALLABLES = _register_graph_naive_callables()


# ==============================================================================
# Tiled-kernel callables (``native_strategy="tile"``, CUDA-only)
# ==============================================================================
#
# These wrap the *inner* warp launchers ``_launch_naive_neighbor_matrix_no_pbc``
# / ``_launch_naive_neighbor_matrix_pbc`` inside a ``jax_callable`` body and
# pass ``native_strategy="tile"`` explicitly, so the tile-cooperative
# ``wp.launch_tiled`` kernel is honored unconditionally (unlike the high-level
# ``naive_neighbor_matrix`` launchers, which drop ``native_strategy`` on the
# non-pair branch and would only reach tile via the "auto" heuristic).
#
# The inner launchers own the 2D tile ``dim`` math (``[1, N]`` no-PBC,
# ``[num_shifts, N]`` PBC), the BLOCK_DIM, the scalar sentinels, and the
# internal wrap launch for the wrapped-PBC case, so the JAX bodies stay thin.
#
# These run only on the eager (``graph_mode="none"``) path, where
# ``naive_neighbor_list`` already pre-fills ``neighbor_matrix=fill_value`` and
# zeroes ``num_neighbors`` / shifts before dispatch, so the bodies perform no
# reset and take no ``fill_value`` argument.  ``graph_mode="warp"`` + tile is
# rejected up-front (those pre-fills are skipped under warp).
#
# Tile supports no-PBC and PBC (wrapped + prewrapped) and ``half_fill``; it has
# no pair-output / ``target_indices`` / selective specialization.  The static
# scalars (``cutoff``, ``half_fill``, and ``num_shifts`` for PBC) are already
# host-static in the existing naive graph path — no new host sync.


def _graph_naive_tile_no_pbc_f32(
    positions: wp.array(dtype=wp.vec3f),
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
        batched=False,
        half_fill=bool(half_fill),
        native_strategy="tile",
    )


def _graph_naive_tile_no_pbc_f64(
    positions: wp.array(dtype=wp.vec3d),
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
        batched=False,
        half_fill=bool(half_fill),
        native_strategy="tile",
    )


# Prewrapped PBC kernels consume precomputed shift ranges; ``pbc`` only affects
# position wrapping, so these callables intentionally omit it.
def _graph_naive_tile_pbc_prewrapped_f32(
    positions: wp.array(dtype=wp.vec3f),
    cell: wp.array(dtype=wp.mat33f),
    shift_range: wp.array(dtype=wp.vec3i),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    cutoff: wp.float32,
    num_shifts: wp.int32,
    half_fill: wp.bool,
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
        native_strategy="tile",
    )


def _graph_naive_tile_pbc_prewrapped_f64(
    positions: wp.array(dtype=wp.vec3d),
    cell: wp.array(dtype=wp.mat33d),
    shift_range: wp.array(dtype=wp.vec3i),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    cutoff: wp.float64,
    num_shifts: wp.int32,
    half_fill: wp.bool,
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
        native_strategy="tile",
    )


def _graph_naive_tile_pbc_wrapped_f32(
    positions: wp.array(dtype=wp.vec3f),
    cell: wp.array(dtype=wp.mat33f),
    pbc: wp.array2d(dtype=wp.bool),
    shift_range: wp.array(dtype=wp.vec3i),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    cutoff: wp.float32,
    num_shifts: wp.int32,
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
        batched=False,
        num_shifts=int(num_shifts),
        half_fill=bool(half_fill),
        wrap_positions=True,
        native_strategy="tile",
    )


def _graph_naive_tile_pbc_wrapped_f64(
    positions: wp.array(dtype=wp.vec3d),
    cell: wp.array(dtype=wp.mat33d),
    pbc: wp.array2d(dtype=wp.bool),
    shift_range: wp.array(dtype=wp.vec3i),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    cutoff: wp.float64,
    num_shifts: wp.int32,
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
        batched=False,
        num_shifts=int(num_shifts),
        half_fill=bool(half_fill),
        wrap_positions=True,
        native_strategy="tile",
    )


# Keyed by ``(has_pbc, wrap_positions)``.  Tile has no selective variant, so the
# selective axis is omitted here; ``native_strategy="tile"`` rejects
# ``rebuild_flags`` at the dispatch site.
_GRAPH_NAIVE_TILE_NO_PBC_IN_OUT_ARGS = ("neighbor_matrix", "num_neighbors")
_GRAPH_NAIVE_TILE_PBC_IN_OUT_ARGS = (
    "neighbor_matrix",
    "neighbor_matrix_shifts",
    "num_neighbors",
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
        "num_outputs": 3,
        "in_out_argnames": _GRAPH_NAIVE_TILE_PBC_IN_OUT_ARGS,
        jnp.dtype(jnp.float32): _graph_naive_tile_pbc_wrapped_f32,
        jnp.dtype(jnp.float64): _graph_naive_tile_pbc_wrapped_f64,
    },
}


def _register_graph_naive_tile_callables() -> dict[
    tuple[bool, bool, jnp.dtype], object
]:
    """Register GraphMode.NONE tile callables for the naive eager path.

    ``GraphMode.NONE`` (not WARP): the tile bodies assume the caller has
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
                graph_mode=GraphMode.NONE,
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
    pair_fn=None,
    pair_params: jax.Array | None = None,
    half_fill: bool = False,
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
    """
    positions = jax.lax.stop_gradient(positions)
    if cell is not None:
        cell = jax.lax.stop_gradient(cell)

    total_atoms = positions.shape[0]
    f64 = positions.dtype == jnp.float64
    cutoff_sq = jnp.asarray(cutoff * cutoff, dtype=positions.dtype)
    zero_dt = jnp.asarray(0.0, dtype=positions.dtype)
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

    # ``pair_fn`` path: real per-atom params + auto-allocated energy/force buffers.
    # (JAX is functional, so user-supplied energy/force buffers cannot be written
    # in-place; we always allocate fresh and return them — the return contract
    # matches torch, the in-place-buffer aspect does not.)
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
            kernel = _get_jax_naive_pair_fn_kernel(pair_fn, wp_dtype, "none", half_fill)
        elif half_fill:
            kernel = (
                _jax_fill_naive_pair_half_f64 if f64 else _jax_fill_naive_pair_half_f32
            )
        else:
            kernel = _jax_fill_naive_pair_f64 if f64 else _jax_fill_naive_pair_f32
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
            kernel = _get_jax_naive_pair_fn_kernel(
                pair_fn, wp_dtype, "wrap_on_entry", half_fill
            )
        elif half_fill:
            kernel = (
                _jax_fill_naive_pbc_pair_half_f64
                if f64
                else _jax_fill_naive_pbc_pair_half_f32
            )
        else:
            kernel = (
                _jax_fill_naive_pbc_pair_f64 if f64 else _jax_fill_naive_pbc_pair_f32
            )
        if cell.ndim == 2:
            cell = cell[jnp.newaxis, :, :]
        if pbc.ndim == 1:
            pbc = pbc[jnp.newaxis, :]
        # ``max_shifts`` sizes the middle launch axis: the single-system PBC kernel
        # derives each periodic image from ``ishift = wp.tid()`` (no internal shift
        # loop), so the launch must enumerate every shift.  Pinning it to 1 would
        # silently drop all non-zero images (only ``ishift == 0`` runs), matching
        # neighbors only in the R==1 regime.
        shift_range, num_shifts_arr, max_shifts = compute_naive_num_shifts(
            cell, cutoff, pbc
        )
        nms = jnp.zeros((total_atoms, max_neighbors, 3), dtype=jnp.int32)
        offs = jnp.zeros((total_atoms, 3), dtype=jnp.int32)
        outs = kernel(
            positions,
            offs,
            cutoff_sq,
            zero_dt,
            cell,
            shift_range,
            num_shifts_arr,
            empty_batch_idx,
            empty_batch_ptr,
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
            launch_dims=(1, int(max_shifts), total_atoms),
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
        batch_idx=None,
        active_mask=mask_,
        matrix_shape=(K, M),
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
    native_strategy: str = "auto",
    *,
    return_distances: bool = False,
    return_vectors: bool = False,
    # Pair-output / partial kwargs accepted for signature parity with the torch
    # binding so misuse raises a clear NotImplementedError instead of a bare
    # TypeError; none are wired through the JAX naive binding yet.
    target_indices: jax.Array | None = None,
    pair_fn=None,
    pair_params: jax.Array | None = None,
    pair_energies: jax.Array | None = None,
    pair_forces: jax.Array | None = None,
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
    native_strategy : {"auto", "scalar", "tile"}, default="auto"
        Selects the underlying Warp kernel variant. ``"scalar"`` uses the
        per-atom scalar kernel. ``"tile"`` uses the tile-cooperative
        ``wp.launch_tiled`` kernel and is **CUDA-only**: requesting it on a
        CPU device raises ``ValueError``. The tile path has no pair-output /
        ``target_indices`` / selective (``rebuild_flags``) variant and is not
        supported with ``graph_mode="warp"`` in this binding; requesting any
        of those with ``native_strategy="tile"`` raises. ``"auto"`` preserves
        the current JAX behavior (scalar dispatch) and never selects tile —
        tile is opt-in in the JAX binding (unlike the torch single-system
        binding, whose ``"auto"`` tiles by default). The tile and scalar
        paths produce identical pair *sets* (per-row ordering may differ).
    inv_cell : jax.Array, shape (1, 3, 3), dtype matches positions, optional
        Inverse cell matrix consumed by the wrap kernel. Only used when
        ``pbc`` is provided and ``wrap_positions=True``. Pass in a
        precomputed value to avoid a per-call ``jnp.linalg.inv`` and to
        keep the input pointer stable for ``graph_mode="warp"`` graph
        replay (omitting it forces cache-miss-per-call on the wrapped
        path). If None, computed from ``cell`` each call. The shape must
        be exactly ``(1, 3, 3)`` (matching the internally-normalized
        ``cell``); a ``(3, 3)`` array would silently allocate a different
        buffer per call and break ``graph_mode="warp"`` cache replay,
        which is why a mismatched shape now raises ``ValueError``.
    positions_wrapped : jax.Array, shape (total_atoms, 3), dtype matches positions, optional
        Scratch buffer the wrap kernel writes into. Pass in a pre-shaped
        array to keep the buffer pointer stable across ``graph_mode="warp"``
        calls (required for graph-replay cache hits on the wrapped path).
        If None, allocated fresh each call. A mismatched shape or dtype
        raises ``ValueError`` to prevent silent graph-replay cache misses.
    per_atom_cell_offsets : jax.Array, shape (total_atoms, 3), dtype=int32, optional
        Scratch buffer the wrap kernel uses to record per-atom cell offsets.
        Pass in a pre-shaped array to keep the buffer pointer stable for
        ``graph_mode="warp"`` replay. If None, allocated fresh each call.
        A mismatched shape or dtype raises ``ValueError`` to prevent
        silent graph-replay cache misses.
    graph_mode : {"none", "warp"}, default="none"
        Execution mode for the underlying Warp launches. ``"none"``
        preserves the existing per-kernel ``jax_kernel`` dispatch path.
        ``"warp"`` uses fused ``jax_callable(..., graph_mode=GraphMode.WARP)``
        callbacks and is intended for ``jax.jit`` call sites that donate
        reusable output buffers.

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
    >>> # Pre-allocate the wrap kernel's scratch buffers and inv_cell once.
    >>> # Capturing them in the closure (rather than donating) keeps their
    >>> # buffer pointers stable across calls, which is what Warp's graph
    >>> # cache keys on. Only the buffers naive_neighbor_list returns are
    >>> # donated, so the in/out arity of the jit'ed step matches.
    >>> inv_cell = jnp.linalg.inv(cell)
    >>> positions_wrapped = jnp.zeros_like(positions)
    >>> per_atom_cell_offsets = jnp.zeros((positions.shape[0], 3), dtype=jnp.int32)
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
    ...         inv_cell=inv_cell,
    ...         positions_wrapped=positions_wrapped,
    ...         per_atom_cell_offsets=per_atom_cell_offsets,
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
    ``graph_mode="warp"`` paths. Advanced users can bound Warp's graph
    cache via ``warp.jax_experimental.set_jax_callable_default_graph_cache_max(...)``.

    For ``graph_mode="warp"`` to actually replay (rather than re-capture
    every call), every in/out buffer pointer the fused callable sees must
    be stable across calls. The output buffers (``neighbor_matrix``,
    ``num_neighbors``, ``neighbor_matrix_shifts`` when applicable) must be
    user-provided **and** included in ``donate_argnums`` of the enclosing
    ``jax.jit`` so they round-trip across calls. On the wrapped path
    (``pbc`` provided + ``wrap_positions=True``), ``inv_cell``,
    ``positions_wrapped`` and ``per_atom_cell_offsets`` must also be passed
    in with stable buffer pointers; the simplest way is to pre-allocate them
    once and capture them in the jit'ed closure (see the example above).
    Letting any of these allocate fresh inside ``naive_neighbor_list``
    silently degrades the wrapped path to cold-capture-per-call (correct,
    but significantly slower than the proposal's measured replay numbers).
    """
    graph_mode = _validate_graph_mode(graph_mode)

    if native_strategy not in {"auto", "scalar", "tile"}:
        raise ValueError(
            "native_strategy must be 'auto' | 'scalar' | 'tile', "
            f"got {native_strategy!r}",
        )

    # ``target_indices`` (partial / selective query) is not yet wired through the
    # JAX naive binding (task 5); reject it with a clear message. ``pair_fn`` and
    # friends ARE wired below via the autograd path.
    if target_indices is not None:
        raise NotImplementedError(
            "target_indices (partial neighbor lists) is not yet wired through the "
            "JAX naive binding. Use the torch binding or the warp factory "
            "directly.",
        )
    # ``pair_fn`` requires per-atom ``pair_params``.  Note: under JAX (functional
    # arrays) any user-supplied ``pair_energies`` / ``pair_forces`` cannot be written
    # in-place — they are auto-allocated and returned, so the *return* contract
    # matches torch while the in-place-buffer aspect does not.
    if pair_fn is not None and pair_params is None:
        raise ValueError(
            "pair_fn requires pair_params (a per-atom (n_atoms, K) parameter array).",
        )

    if pbc is None and cell is not None:
        raise ValueError("If cell is provided, pbc must also be provided")
    if pbc is not None and cell is None:
        raise ValueError("If pbc is provided, cell must also be provided")

    if native_strategy == "tile":
        # The tile-cooperative kernel is CUDA-only and has no pair-output,
        # selective (rebuild_flags), or CUDA-graph (graph_mode="warp") variant.
        # Gate here, before any launch, mirroring the warp launcher CPU guard.
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
        if graph_mode != "none":
            raise NotImplementedError(
                "native_strategy='tile' is only supported with "
                "graph_mode='none'; CUDA-graph capture of the tile kernel is a "
                "follow-up.",
            )

    has_pair_outputs = (
        bool(return_distances) or bool(return_vectors) or pair_fn is not None
    )
    if has_pair_outputs:
        if graph_mode != "none" or rebuild_flags is not None:
            raise NotImplementedError(
                "Pair outputs require graph_mode='none' and no rebuild_flags.",
            )
        if max_neighbors is None:
            max_neighbors = estimate_max_neighbors(cutoff)
        if fill_value is None:
            fill_value = positions.shape[0]
        if cell is not None and cell.ndim == 2:
            cell_norm = cell[jnp.newaxis, :, :]
        else:
            cell_norm = cell
        if cell_norm is not None and cell_norm.dtype != positions.dtype:
            cell_norm = cell_norm.astype(positions.dtype)
        pbc_norm = None
        if pbc is not None:
            pbc_norm = pbc if pbc.ndim == 2 else pbc[jnp.newaxis, :]
        forward_kwargs = {
            "pbc": pbc_norm,
            "cutoff": float(cutoff),
            "max_neighbors": int(max_neighbors),
            "fill_value": int(fill_value),
            "pair_fn": pair_fn,
            "pair_params": pair_params,
            "half_fill": bool(half_fill),
        }
        route_out = _route_pair_outputs(
            positions,
            cell_norm,
            _naive_pair_outputs_forward,
            forward_kwargs,
        )
        # ``extra_outputs`` carries the per-pair energy/force tail only when
        # ``pair_fn`` is set, so the route return is 5 elements (geometry only) or 7.
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
            # Repack per-pair geometry (and pair_fn outputs) into COO order aligned
            # with ``nl``.  Eager-only, like the index conversion.
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
                f"inv_cell must have shape (1, 3, 3) to match the internal "
                f"cell layout; got {inv_cell.shape}. A mismatched shape "
                f"silently breaks graph_mode='warp' cache replay."
            )
        if inv_cell.dtype != positions.dtype:
            raise ValueError(
                f"inv_cell dtype must match positions dtype "
                f"({positions.dtype}); got {inv_cell.dtype}."
            )
    if positions_wrapped is not None:
        expected_pw_shape = (positions.shape[0], 3)
        if positions_wrapped.shape != expected_pw_shape:
            raise ValueError(
                f"positions_wrapped must have shape {expected_pw_shape}; "
                f"got {positions_wrapped.shape}."
            )
        if positions_wrapped.dtype != positions.dtype:
            raise ValueError(
                f"positions_wrapped dtype must match positions dtype "
                f"({positions.dtype}); got {positions_wrapped.dtype}."
            )
    if per_atom_cell_offsets is not None:
        expected_off_shape = (positions.shape[0], 3)
        if per_atom_cell_offsets.shape != expected_off_shape:
            raise ValueError(
                f"per_atom_cell_offsets must have shape {expected_off_shape}; "
                f"got {per_atom_cell_offsets.shape}."
            )
        if per_atom_cell_offsets.dtype != jnp.int32:
            raise ValueError(
                f"per_atom_cell_offsets dtype must be int32; "
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

    if cutoff <= 0:
        if rebuild_flags is None and graph_mode == "warp":
            neighbor_matrix = neighbor_matrix.at[:].set(fill_value)
            num_neighbors = num_neighbors.at[:].set(jnp.int32(0))
            if pbc is not None:
                neighbor_matrix_shifts = neighbor_matrix_shifts.at[:].set(jnp.int32(0))
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

    # Select kernel based on dtype and static half-fill specialization.
    if positions.dtype == jnp.float64:
        if half_fill:
            _jax_fill = _jax_fill_naive_half_f64
            _jax_fill_pbc = _jax_fill_naive_pbc_half_f64
            _jax_fill_pbc_prewrapped = _jax_fill_naive_pbc_prewrapped_half_f64
            _jax_fill_selective = _jax_fill_naive_selective_half_f64
            _jax_fill_pbc_selective = _jax_fill_naive_pbc_selective_half_f64
            _jax_fill_pbc_prewrapped_selective = (
                _jax_fill_naive_pbc_prewrapped_selective_half_f64
            )
        else:
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
        if half_fill:
            _jax_fill = _jax_fill_naive_half_f32
            _jax_fill_pbc = _jax_fill_naive_pbc_half_f32
            _jax_fill_pbc_prewrapped = _jax_fill_naive_pbc_prewrapped_half_f32
            _jax_fill_selective = _jax_fill_naive_selective_half_f32
            _jax_fill_pbc_selective = _jax_fill_naive_pbc_selective_half_f32
            _jax_fill_pbc_prewrapped_selective = (
                _jax_fill_naive_pbc_prewrapped_selective_half_f32
            )
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

    if native_strategy == "tile":
        # CUDA-only tile-cooperative path (eager / graph_mode="none" only).
        # Output buffers were already pre-filled above; the callable bodies
        # call the inner warp launchers with native_strategy="tile" and rely on
        # the host-static cutoff / half_fill / num_shifts scalars (no new sync).
        cutoff_static = float(cutoff)
        if pbc is None:
            tile_callable = _GRAPH_NAIVE_TILE_CALLABLES[(False, False, positions.dtype)]
            neighbor_matrix, num_neighbors = tile_callable(
                positions,
                neighbor_matrix,
                num_neighbors,
                cutoff_static,
                half_fill,
            )
        else:
            if cell.dtype != positions.dtype:
                cell = cell.astype(positions.dtype)
            num_shifts = int(max_shifts_per_system)
            tile_callable = _GRAPH_NAIVE_TILE_CALLABLES[
                (True, bool(wrap_positions), positions.dtype)
            ]
            pbc_arg = (pbc,) if wrap_positions else ()
            neighbor_matrix, neighbor_matrix_shifts, num_neighbors = tile_callable(
                positions,
                cell,
                *pbc_arg,
                shift_range_per_dimension,
                neighbor_matrix,
                neighbor_matrix_shifts,
                num_neighbors,
                cutoff_static,
                num_shifts,
                half_fill,
            )
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
            neighbor_matrix, num_neighbors = _jax_fill_selective(
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
            neighbor_matrix, num_neighbors = _jax_fill(
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
                    _jax_fill_pbc_selective(
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
                neighbor_matrix, neighbor_matrix_shifts, num_neighbors = _jax_fill_pbc(
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
        else:
            if rebuild_flags is not None:
                rf = rebuild_flags.flatten()[:1].astype(jnp.bool_)
                num_neighbors = jnp.where(
                    rf[0], jnp.zeros_like(num_neighbors), num_neighbors
                )
                neighbor_matrix, neighbor_matrix_shifts, num_neighbors = (
                    _jax_fill_pbc_prewrapped_selective(
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
                    _jax_fill_pbc_prewrapped(
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
