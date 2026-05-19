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

from typing import Literal

import jax
import jax.numpy as jnp
import warp as wp
from warp.jax_experimental import GraphMode, jax_callable, jax_kernel

from nvalchemiops.jax.neighbors.neighbor_utils import (
    _validate_graph_mode,
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
    _selective_zero_num_neighbors_single,
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


def _run_graph_naive_no_pbc(
    positions,
    neighbor_matrix,
    num_neighbors,
    cutoff_sq,
    fill_value,
    half_fill,
    fill_kernel,
    selective_kernel=None,
    rebuild_flags=None,
) -> None:
    """Execute the no-PBC graph-mode body."""
    total_atoms = positions.shape[0]
    if rebuild_flags is None:
        _reset_graph_neighbor_outputs(neighbor_matrix, num_neighbors, fill_value)
        wp.launch(
            kernel=fill_kernel,
            dim=total_atoms,
            inputs=[positions, cutoff_sq, neighbor_matrix, num_neighbors, half_fill],
        )
    else:
        wp.launch(
            kernel=_selective_zero_num_neighbors_single,
            dim=total_atoms,
            inputs=[num_neighbors, rebuild_flags],
        )
        wp.launch(
            kernel=selective_kernel,
            dim=total_atoms,
            inputs=[
                positions,
                cutoff_sq,
                neighbor_matrix,
                num_neighbors,
                half_fill,
                rebuild_flags,
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
    fill_kernel,
    selective_kernel=None,
    rebuild_flags=None,
) -> None:
    """Execute the prewrapped-PBC graph-mode body."""
    launch_dims = (num_shifts, positions.shape[0])
    if rebuild_flags is None:
        _reset_graph_neighbor_outputs(
            neighbor_matrix,
            num_neighbors,
            fill_value,
            neighbor_matrix_shifts,
        )
        wp.launch(
            kernel=fill_kernel,
            dim=launch_dims,
            inputs=[
                positions,
                cutoff_sq,
                cell,
                shift_range,
                neighbor_matrix,
                neighbor_matrix_shifts,
                num_neighbors,
                half_fill,
            ],
        )
    else:
        wp.launch(
            kernel=_selective_zero_num_neighbors_single,
            dim=positions.shape[0],
            inputs=[num_neighbors, rebuild_flags],
        )
        wp.launch(
            kernel=selective_kernel,
            dim=launch_dims,
            inputs=[
                positions,
                cutoff_sq,
                cell,
                shift_range,
                neighbor_matrix,
                neighbor_matrix_shifts,
                num_neighbors,
                half_fill,
                rebuild_flags,
            ],
        )


def _run_graph_naive_pbc_wrapped(
    positions,
    cell,
    inv_cell,
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
    wrap_kernel,
    fill_kernel,
    selective_kernel=None,
    rebuild_flags=None,
) -> None:
    """Execute the wrapped-PBC graph-mode body."""
    total_atoms = positions.shape[0]
    launch_dims = (num_shifts, total_atoms)
    if rebuild_flags is None:
        _reset_graph_neighbor_outputs(
            neighbor_matrix,
            num_neighbors,
            fill_value,
            neighbor_matrix_shifts,
        )
    else:
        wp.launch(
            kernel=_selective_zero_num_neighbors_single,
            dim=total_atoms,
            inputs=[num_neighbors, rebuild_flags],
        )

    wp.launch(
        kernel=wrap_kernel,
        dim=total_atoms,
        inputs=[positions, cell, inv_cell],
        outputs=[positions_wrapped, per_atom_cell_offsets],
    )

    fill_inputs = [
        positions_wrapped,
        per_atom_cell_offsets,
        cutoff_sq,
        cell,
        shift_range,
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
        half_fill,
    ]
    if rebuild_flags is not None:
        fill_inputs.append(rebuild_flags)

    wp.launch(
        kernel=fill_kernel if rebuild_flags is None else selective_kernel,
        dim=launch_dims,
        inputs=fill_inputs,
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
        _fill_naive_neighbor_matrix_overload[wp.float32],
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
        _fill_naive_neighbor_matrix_overload[wp.float64],
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
        _fill_naive_neighbor_matrix_overload[wp.float32],
        selective_kernel=_fill_naive_neighbor_matrix_selective_overload[wp.float32],
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
        _fill_naive_neighbor_matrix_overload[wp.float64],
        selective_kernel=_fill_naive_neighbor_matrix_selective_overload[wp.float64],
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
        _fill_naive_neighbor_matrix_pbc_prewrapped_overload[wp.float32],
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
        _fill_naive_neighbor_matrix_pbc_prewrapped_overload[wp.float64],
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
        _fill_naive_neighbor_matrix_pbc_prewrapped_overload[wp.float32],
        selective_kernel=_fill_naive_neighbor_matrix_pbc_prewrapped_selective_overload[
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
        _fill_naive_neighbor_matrix_pbc_prewrapped_overload[wp.float64],
        selective_kernel=_fill_naive_neighbor_matrix_pbc_prewrapped_selective_overload[
            wp.float64
        ],
        rebuild_flags=rebuild_flags,
    )


def _graph_naive_pbc_wrapped_f32(
    positions: wp.array(dtype=wp.vec3f),
    cell: wp.array(dtype=wp.mat33f),
    inv_cell: wp.array(dtype=wp.mat33f),
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
        _wrap_positions_single_overload[wp.float32],
        _fill_naive_neighbor_matrix_pbc_overload[wp.float32],
    )


def _graph_naive_pbc_wrapped_f64(
    positions: wp.array(dtype=wp.vec3d),
    cell: wp.array(dtype=wp.mat33d),
    inv_cell: wp.array(dtype=wp.mat33d),
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
        _wrap_positions_single_overload[wp.float64],
        _fill_naive_neighbor_matrix_pbc_overload[wp.float64],
    )


def _graph_naive_pbc_wrapped_selective_f32(
    positions: wp.array(dtype=wp.vec3f),
    cell: wp.array(dtype=wp.mat33f),
    inv_cell: wp.array(dtype=wp.mat33f),
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
        _wrap_positions_single_overload[wp.float32],
        _fill_naive_neighbor_matrix_pbc_overload[wp.float32],
        selective_kernel=_fill_naive_neighbor_matrix_pbc_selective_overload[wp.float32],
        rebuild_flags=rebuild_flags,
    )


def _graph_naive_pbc_wrapped_selective_f64(
    positions: wp.array(dtype=wp.vec3d),
    cell: wp.array(dtype=wp.mat33d),
    inv_cell: wp.array(dtype=wp.mat33d),
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
        _wrap_positions_single_overload[wp.float64],
        _fill_naive_neighbor_matrix_pbc_overload[wp.float64],
        selective_kernel=_fill_naive_neighbor_matrix_pbc_selective_overload[wp.float64],
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
    cutoff_sq = float(cutoff * cutoff)

    if graph_mode == "warp":
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
                cutoff_sq,
                neighbor_matrix,
                num_neighbors,
                half_fill,
                rf,
                launch_dims=(total_atoms,),
            )
        else:
            neighbor_matrix, num_neighbors = _jax_fill(
                positions,
                cutoff_sq,
                neighbor_matrix,
                num_neighbors,
                half_fill,
                launch_dims=(total_atoms,),
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
                    cutoff_sq,
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
                        cutoff_sq,
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
                        cutoff_sq,
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
