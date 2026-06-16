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

"""JAX bindings for batched cell list O(N) neighbor list construction."""

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
from nvalchemiops.jax.neighbors.cell_list import (
    _is_cpu_array,
    _resolve_cell_strategy,
    _validate_atom_centric_path,
    _validate_compact_target_buffers,
    _validate_pair_kwargs,
)
from nvalchemiops.jax.neighbors.neighbor_utils import (
    allocate_cell_list,
    coo_pack_pair_geometry,
    get_neighbor_list_from_neighbor_matrix,
    prepare_batch_idx_ptr,
)
from nvalchemiops.neighbors.cell_list import (
    batch_query_cell_list_pair_centric_sorted as _warp_batch_query_pair_centric,
)
from nvalchemiops.neighbors.cell_list import (
    compute_batch_pair_centric_n_outer,
    get_build_cell_list_kernel,
    get_cell_list_cells_per_system_kernel,
    get_cell_list_gather_kernel,
    get_query_cell_list_kernel,
    is_pair_centric_launch_safe,
    is_pair_centric_parallelism_sufficient,
)
from nvalchemiops.neighbors.cell_list.launchers import (
    _raise_unsafe_pair_centric_launch,
)
from nvalchemiops.neighbors.neighbor_utils import estimate_max_neighbors
from nvalchemiops.neighbors.output_args import (
    _has_partial_or_pair_outputs,
)

# ==============================================================================
# JAX Kernel Wrappers
# ==============================================================================

# Build step 1: Construct bin sizes (per system)
_jax_batch_construct_bin_size_f32 = jax_kernel(
    get_build_cell_list_kernel("construct_bin_size", wp.float32, batched=True),
    num_outputs=1,
    in_out_argnames=["cells_per_dimension_batch"],
    enable_backward=False,
)
_jax_batch_construct_bin_size_f64 = jax_kernel(
    get_build_cell_list_kernel("construct_bin_size", wp.float64, batched=True),
    num_outputs=1,
    in_out_argnames=["cells_per_dimension_batch"],
    enable_backward=False,
)

# Helper: Compute cells per system
_jax_compute_cells_per_system = jax_kernel(
    get_cell_list_cells_per_system_kernel(),
    num_outputs=1,
    in_out_argnames=["cells_per_system"],
    enable_backward=False,
)

# Build step 2: Count atoms per bin
_jax_batch_count_atoms_per_bin_f32 = jax_kernel(
    get_build_cell_list_kernel("count_atoms", wp.float32, batched=True),
    num_outputs=2,
    in_out_argnames=["atoms_per_cell_count", "atom_periodic_shifts"],
    enable_backward=False,
)
_jax_batch_count_atoms_per_bin_f64 = jax_kernel(
    get_build_cell_list_kernel("count_atoms", wp.float64, batched=True),
    num_outputs=2,
    in_out_argnames=["atoms_per_cell_count", "atom_periodic_shifts"],
    enable_backward=False,
)

# Build step 3: Bin atoms into cells
_jax_batch_bin_atoms_f32 = jax_kernel(
    get_build_cell_list_kernel("bin_atoms", wp.float32, batched=True),
    num_outputs=3,
    in_out_argnames=["atom_to_cell_mapping", "atoms_per_cell_count", "cell_atom_list"],
    enable_backward=False,
)
_jax_batch_bin_atoms_f64 = jax_kernel(
    get_build_cell_list_kernel("bin_atoms", wp.float64, batched=True),
    num_outputs=3,
    in_out_argnames=["atom_to_cell_mapping", "atoms_per_cell_count", "cell_atom_list"],
    enable_backward=False,
)

# Gather: pack positions + atom_periodic_shifts into per-cell-contiguous layout
# (cell_atom_list permutation) for coalesced reads by the sorted-build kernel.
_jax_batch_gather_positions_by_cell_f32 = jax_kernel(
    get_cell_list_gather_kernel(wp.float32),
    num_outputs=2,
    in_out_argnames=["sorted_positions", "sorted_shifts"],
    enable_backward=False,
)
_jax_batch_gather_positions_by_cell_f64 = jax_kernel(
    get_cell_list_gather_kernel(wp.float64),
    num_outputs=2,
    in_out_argnames=["sorted_positions", "sorted_shifts"],
    enable_backward=False,
)

# Query: sorted-reads atom-centric batch neighbor matrix kernel.  The same
# kernel handles selective and non-selective callers via the ``rebuild_flags``
# array (always-True for non-selective; per-system bool array otherwise).
_jax_batch_build_neighbor_matrix_local_count_sorted_f32 = jax_kernel(
    get_query_cell_list_kernel(
        wp.float32,
        strategy="atom_centric",
        batched=True,
        selective=True,
        partial=False,
        return_vectors=False,
        return_distances=False,
        pair_fn=None,
    ),
    num_outputs=3,
    in_out_argnames=["neighbor_matrix", "neighbor_matrix_shifts", "num_neighbors"],
    enable_backward=False,
)
_jax_batch_build_neighbor_matrix_local_count_sorted_f64 = jax_kernel(
    get_query_cell_list_kernel(
        wp.float64,
        strategy="atom_centric",
        batched=True,
        selective=True,
        partial=False,
        return_vectors=False,
        return_distances=False,
        pair_fn=None,
    ),
    num_outputs=3,
    in_out_argnames=["neighbor_matrix", "neighbor_matrix_shifts", "num_neighbors"],
    enable_backward=False,
)

# Half-fill variants of the sorted-build kernel.  ``half_fill`` is a compile-time
# specialization in the Warp factory (the runtime ``half_fill`` arg is an ignored
# ABI placeholder), so honoring ``half_fill=True`` requires a distinct kernel.
_jax_batch_build_neighbor_matrix_local_count_sorted_half_f32 = jax_kernel(
    get_query_cell_list_kernel(
        wp.float32,
        strategy="atom_centric",
        batched=True,
        selective=True,
        partial=False,
        half_fill=True,
        return_vectors=False,
        return_distances=False,
        pair_fn=None,
    ),
    num_outputs=3,
    in_out_argnames=["neighbor_matrix", "neighbor_matrix_shifts", "num_neighbors"],
    enable_backward=False,
)
_jax_batch_build_neighbor_matrix_local_count_sorted_half_f64 = jax_kernel(
    get_query_cell_list_kernel(
        wp.float64,
        strategy="atom_centric",
        batched=True,
        selective=True,
        partial=False,
        half_fill=True,
        return_vectors=False,
        return_distances=False,
        pair_fn=None,
    ),
    num_outputs=3,
    in_out_argnames=["neighbor_matrix", "neighbor_matrix_shifts", "num_neighbors"],
    enable_backward=False,
)

# Pair-output variants — consumed by the autograd path when
# ``return_distances`` / ``return_vectors`` is set.  The bytes written into
# ``neighbor_vectors`` / ``neighbor_distances`` are differentiable via the
# JAX autograd primitive in :mod:`nvalchemiops.jax.neighbors._autograd`.
_jax_batch_build_neighbor_matrix_local_count_sorted_pair_f32 = jax_kernel(
    get_query_cell_list_kernel(
        wp.float32,
        strategy="atom_centric",
        batched=True,
        selective=True,
        partial=False,
        return_vectors=True,
        return_distances=True,
        pair_fn=None,
    ),
    num_outputs=5,
    in_out_argnames=[
        "neighbor_matrix",
        "neighbor_matrix_shifts",
        "num_neighbors",
        "neighbor_vectors",
        "neighbor_distances",
    ],
    enable_backward=False,
)
_jax_batch_build_neighbor_matrix_local_count_sorted_pair_f64 = jax_kernel(
    get_query_cell_list_kernel(
        wp.float64,
        strategy="atom_centric",
        batched=True,
        selective=True,
        partial=False,
        return_vectors=True,
        return_distances=True,
        pair_fn=None,
    ),
    num_outputs=5,
    in_out_argnames=[
        "neighbor_matrix",
        "neighbor_matrix_shifts",
        "num_neighbors",
        "neighbor_vectors",
        "neighbor_distances",
    ],
    enable_backward=False,
)

# Half-fill specializations of the atom-centric geometry pair-output kernel
# (selected when ``half_fill=True``; ``half_fill`` is a compile-time constant).
_jax_batch_build_neighbor_matrix_local_count_sorted_pair_half_f32 = jax_kernel(
    get_query_cell_list_kernel(
        wp.float32,
        strategy="atom_centric",
        batched=True,
        selective=True,
        partial=False,
        return_vectors=True,
        return_distances=True,
        pair_fn=None,
        half_fill=True,
    ),
    num_outputs=5,
    in_out_argnames=[
        "neighbor_matrix",
        "neighbor_matrix_shifts",
        "num_neighbors",
        "neighbor_vectors",
        "neighbor_distances",
    ],
    enable_backward=False,
)
_jax_batch_build_neighbor_matrix_local_count_sorted_pair_half_f64 = jax_kernel(
    get_query_cell_list_kernel(
        wp.float64,
        strategy="atom_centric",
        batched=True,
        selective=True,
        partial=False,
        return_vectors=True,
        return_distances=True,
        pair_fn=None,
        half_fill=True,
    ),
    num_outputs=5,
    in_out_argnames=[
        "neighbor_matrix",
        "neighbor_matrix_shifts",
        "num_neighbors",
        "neighbor_vectors",
        "neighbor_distances",
    ],
    enable_backward=False,
)


@functools.cache
def _get_jax_batch_cell_list_pair_outputs_kernel(
    pair_fn, wp_dtype, partial, half_fill: bool = False
):
    """Build (and cache) a ``jax_kernel`` for a batched cell-list atom-centric
    ``sorted`` pair-output kernel.

    Mirrors the geometry registration above, optionally with ``pair_fn`` set
    (so the kernel's ``HAS_PAIR_FN`` body runs and ``pair_energies`` /
    ``pair_forces`` are registered as additional outputs) and/or
    ``partial=True`` (the ``target_indices`` path: output row ``r`` maps to atom
    ``target_indices[r]``, launched ``(num_targets,)``).  Cached by
    ``(pair_fn identity, wp_dtype, partial)``; one recompile per distinct
    ``(pair_fn, partial)`` combination.

    With ``pair_fn`` the kernel has 7 outputs (geometry + ``pe`` / ``pf``);
    without it, 5 (geometry only).
    """
    kernel = get_query_cell_list_kernel(
        wp_dtype,
        strategy="atom_centric",
        batched=True,
        selective=True,
        partial=bool(partial),
        return_vectors=True,
        return_distances=True,
        pair_fn=pair_fn,
        half_fill=bool(half_fill),
    )
    in_out_argnames = [
        "neighbor_matrix",
        "neighbor_matrix_shifts",
        "num_neighbors",
        "neighbor_vectors",
        "neighbor_distances",
    ]
    if pair_fn is not None:
        in_out_argnames += ["pair_energies", "pair_forces"]
    return jax_kernel(
        kernel,
        num_outputs=len(in_out_argnames),
        in_out_argnames=in_out_argnames,
        enable_backward=False,
    )


__all__ = [
    "batch_cell_list",
    "batch_build_cell_list",
    "batch_query_cell_list",
    "estimate_batch_cell_list_sizes",
]


def _normalize_batch_cell_pbc(
    cell: jax.Array | None,
    pbc: jax.Array | None,
    *,
    num_systems: int,
    dtype,
) -> tuple[jax.Array, jax.Array]:
    """Return batched cell/PBC arrays for JAX batch cell-list kernels."""
    if cell is None:
        cell_out = jnp.broadcast_to(
            jnp.eye(3, dtype=dtype),
            (num_systems, 3, 3),
        )
    else:
        cell_out = jnp.asarray(cell)
        if cell_out.ndim == 2:
            cell_out = jnp.broadcast_to(cell_out, (num_systems, 3, 3))
        elif cell_out.ndim != 3:
            raise ValueError(
                "cell must have shape (3, 3) or (num_systems, 3, 3) for "
                "batch cell-list operations.",
            )
        if cell_out.dtype != dtype:
            cell_out = cell_out.astype(dtype)

    if pbc is None:
        pbc_out = jnp.ones((num_systems, 3), dtype=jnp.bool_)
    else:
        pbc_out = jnp.asarray(pbc)
        if pbc_out.ndim == 1:
            pbc_out = jnp.broadcast_to(pbc_out, (num_systems, 3))
        elif pbc_out.ndim != 2:
            raise ValueError(
                "pbc must have shape (3,) or (num_systems, 3) for batch "
                "cell-list operations.",
            )
        pbc_out = pbc_out.astype(jnp.bool_)

    return cell_out, pbc_out


# ==============================================================================
# Batched pair-centric query wrappers (GraphMode.NONE jax_callable)
# ==============================================================================
#
# The batched pair-centric launcher
# (:func:`batch_query_cell_list_pair_centric_sorted`) is CUDA-only and sizes its
# launch grid from host-computed scalars (``total_cells``, ``n_outer``,
# ``R_max``).  Those cannot be CUDA-graph-replayed across a changed radius, so we
# wrap it with ``graph_mode=GraphMode.NONE`` (launch each call, no capture).  The
# launcher itself runs the gather into ``sorted_positions`` /
# ``sorted_atom_periodic_shifts`` and the ``cell_to_system`` map fill before the
# main pair-centric kernel, so this wrapper just forwards the donated scratch +
# output buffers and the static sizing scalars.
#
# In/out (donated) arrays, in order: ``sorted_positions``,
# ``sorted_atom_periodic_shifts``, ``cell_to_system``, ``neighbor_matrix``,
# ``neighbor_matrix_shifts``, ``num_neighbors`` (6 outputs).  Note this adds
# ``cell_to_system`` relative to the single-system pair-centric path and drops
# ``atom_to_cell_mapping`` (the batched launcher does not read it).
#
# ``cells_per_dimension`` / ``neighbor_search_radius`` are ``(num_systems, 3)``
# int32 arrays that map to 1-D ``wp.vec3i`` arrays here (the batched kernels read
# them per-system as vec3i), unlike the single-system ``(3,)`` int32 path.
#
# ``half_fill=False`` is passed explicitly: the batched launcher defaults to
# ``half_fill=True``, and the JAX batch cell-list pair-centric path is full-fill.


def _run_batch_query_cell_list_pair_centric(
    positions,
    cell,
    pbc,
    cells_per_dimension,
    neighbor_search_radius,
    cell_offsets,
    cells_per_system,
    atom_periodic_shifts,
    atoms_per_cell_count,
    cell_atom_start_indices,
    cell_atom_list,
    sorted_positions,
    sorted_atom_periodic_shifts,
    cell_to_system,
    neighbor_matrix,
    neighbor_matrix_shifts,
    num_neighbors,
    cutoff,
    wp_dtype,
    total_cells: int,
    n_outer: int,
    R_max: tuple[int, int, int],
    return_vectors: bool = False,
    return_distances: bool = False,
    pair_fn=None,
    pair_params=None,
    neighbor_vectors=None,
    neighbor_distances=None,
    pair_energies=None,
    pair_forces=None,
) -> None:
    """Execute the batched pair-centric cell-list query callback.

    Non-selective full-fill only (``rebuild_flags=None``, ``half_fill=False``).
    ``num_neighbors`` is accumulated via ``atomic_add`` inside the kernel, so it
    is zeroed here before the launch.

    The optional ``return_vectors`` / ``return_distances`` / ``pair_fn`` (+
    ``pair_params`` and the ``neighbor_vectors`` / ``neighbor_distances`` /
    ``pair_energies`` / ``pair_forces`` output buffers) thread the pair-output
    contract straight to the batched pair-centric launcher.
    """
    num_neighbors.zero_()

    # Graph-capture contract: this body runs under ``GraphMode.NONE`` (no CUDA
    # graph capture), so ``str(positions.device)`` is read each call.  The
    # ``total_cells`` / ``n_outer`` / ``R_max`` scalars are baked at launch-build
    # time from the host-read radius in ``batch_query_cell_list``.
    _warp_batch_query_pair_centric(
        positions=positions,
        cell=cell,
        pbc=pbc,
        cutoff=cutoff,
        cells_per_dimension=cells_per_dimension,
        neighbor_search_radius=neighbor_search_radius,
        cell_offsets=cell_offsets,
        cells_per_system=cells_per_system,
        atom_periodic_shifts=atom_periodic_shifts,
        atoms_per_cell_count=atoms_per_cell_count,
        cell_atom_start_indices=cell_atom_start_indices,
        cell_atom_list=cell_atom_list,
        sorted_positions=sorted_positions,
        sorted_atom_periodic_shifts=sorted_atom_periodic_shifts,
        cell_to_system=cell_to_system,
        neighbor_matrix=neighbor_matrix,
        neighbor_matrix_shifts=neighbor_matrix_shifts,
        num_neighbors=num_neighbors,
        wp_dtype=wp_dtype,
        device=str(positions.device),
        total_cells=int(total_cells),
        n_outer=int(n_outer),
        R_max=R_max,
        half_fill=False,
        return_vectors=return_vectors,
        return_distances=return_distances,
        pair_fn=pair_fn,
        pair_params=pair_params,
        neighbor_vectors=neighbor_vectors,
        neighbor_distances=neighbor_distances,
        pair_energies=pair_energies,
        pair_forces=pair_forces,
    )


def _batch_query_cell_list_pair_centric_f32(
    positions: wp.array(dtype=wp.vec3f),
    cell: wp.array(dtype=wp.mat33f),
    pbc: wp.array(dtype=wp.bool, ndim=2),
    cells_per_dimension: wp.array(dtype=wp.vec3i),
    neighbor_search_radius: wp.array(dtype=wp.vec3i),
    cell_offsets: wp.array(dtype=wp.int32),
    cells_per_system: wp.array(dtype=wp.int32),
    atom_periodic_shifts: wp.array(dtype=wp.vec3i),
    atoms_per_cell_count: wp.array(dtype=wp.int32),
    cell_atom_start_indices: wp.array(dtype=wp.int32),
    cell_atom_list: wp.array(dtype=wp.int32),
    sorted_positions: wp.array(dtype=wp.vec3f),
    sorted_atom_periodic_shifts: wp.array(dtype=wp.vec3i),
    cell_to_system: wp.array(dtype=wp.int32),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    cutoff: wp.float32,
    total_cells: wp.int32,
    n_outer: wp.int32,
    R_max_x: wp.int32,
    R_max_y: wp.int32,
    R_max_z: wp.int32,
) -> None:
    _run_batch_query_cell_list_pair_centric(
        positions,
        cell,
        pbc,
        cells_per_dimension,
        neighbor_search_radius,
        cell_offsets,
        cells_per_system,
        atom_periodic_shifts,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
        sorted_positions,
        sorted_atom_periodic_shifts,
        cell_to_system,
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
        float(cutoff),
        wp.float32,
        int(total_cells),
        int(n_outer),
        (int(R_max_x), int(R_max_y), int(R_max_z)),
    )


def _batch_query_cell_list_pair_centric_f64(
    positions: wp.array(dtype=wp.vec3d),
    cell: wp.array(dtype=wp.mat33d),
    pbc: wp.array(dtype=wp.bool, ndim=2),
    cells_per_dimension: wp.array(dtype=wp.vec3i),
    neighbor_search_radius: wp.array(dtype=wp.vec3i),
    cell_offsets: wp.array(dtype=wp.int32),
    cells_per_system: wp.array(dtype=wp.int32),
    atom_periodic_shifts: wp.array(dtype=wp.vec3i),
    atoms_per_cell_count: wp.array(dtype=wp.int32),
    cell_atom_start_indices: wp.array(dtype=wp.int32),
    cell_atom_list: wp.array(dtype=wp.int32),
    sorted_positions: wp.array(dtype=wp.vec3d),
    sorted_atom_periodic_shifts: wp.array(dtype=wp.vec3i),
    cell_to_system: wp.array(dtype=wp.int32),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    cutoff: wp.float64,
    total_cells: wp.int32,
    n_outer: wp.int32,
    R_max_x: wp.int32,
    R_max_y: wp.int32,
    R_max_z: wp.int32,
) -> None:
    _run_batch_query_cell_list_pair_centric(
        positions,
        cell,
        pbc,
        cells_per_dimension,
        neighbor_search_radius,
        cell_offsets,
        cells_per_system,
        atom_periodic_shifts,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
        sorted_positions,
        sorted_atom_periodic_shifts,
        cell_to_system,
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
        float(cutoff),
        wp.float64,
        int(total_cells),
        int(n_outer),
        (int(R_max_x), int(R_max_y), int(R_max_z)),
    )


# Donated in/out buffers for the pair-centric callable.  ``cell_to_system`` is
# scratch the launcher's ``_build_cell_to_system_map`` fills; the gather scratch
# and the three output buffers complete the set (6 outputs).
_BATCH_PAIR_CENTRIC_INOUT = [
    "sorted_positions",
    "sorted_atom_periodic_shifts",
    "cell_to_system",
    "neighbor_matrix",
    "neighbor_matrix_shifts",
    "num_neighbors",
]
# GraphMode.NONE: the launch dim is baked from the host-read sizing scalars, so
# CUDA-graph replay across a changed radius is unsafe.
_jax_batch_query_cell_list_pair_centric_f32 = jax_callable(
    _batch_query_cell_list_pair_centric_f32,
    num_outputs=len(_BATCH_PAIR_CENTRIC_INOUT),
    in_out_argnames=_BATCH_PAIR_CENTRIC_INOUT,
    graph_mode=GraphMode.NONE,
)
_jax_batch_query_cell_list_pair_centric_f64 = jax_callable(
    _batch_query_cell_list_pair_centric_f64,
    num_outputs=len(_BATCH_PAIR_CENTRIC_INOUT),
    in_out_argnames=_BATCH_PAIR_CENTRIC_INOUT,
    graph_mode=GraphMode.NONE,
)

# --- Batched pair-centric PAIR-OUTPUT callables -----------------------------
# Same launch mechanism as the matrix callables above, with per-pair geometry
# (and optionally ``pair_fn`` energies / forces) written by the same kernel.
# The sizing scalars (``total_cells`` / ``n_outer`` / ``R_max``) are host-read
# statics, so ``GraphMode.NONE`` (eager-on-cutoff, like every pair-output path).
_BATCH_PAIR_CENTRIC_GEOM_INOUT = _BATCH_PAIR_CENTRIC_INOUT + [
    "neighbor_vectors",
    "neighbor_distances",
]
_BATCH_PAIR_CENTRIC_PAIR_FN_INOUT = _BATCH_PAIR_CENTRIC_GEOM_INOUT + [
    "pair_energies",
    "pair_forces",
]


def _batch_query_cell_list_pair_centric_geom_f32(
    positions: wp.array(dtype=wp.vec3f),
    cell: wp.array(dtype=wp.mat33f),
    pbc: wp.array(dtype=wp.bool, ndim=2),
    cells_per_dimension: wp.array(dtype=wp.vec3i),
    neighbor_search_radius: wp.array(dtype=wp.vec3i),
    cell_offsets: wp.array(dtype=wp.int32),
    cells_per_system: wp.array(dtype=wp.int32),
    atom_periodic_shifts: wp.array(dtype=wp.vec3i),
    atoms_per_cell_count: wp.array(dtype=wp.int32),
    cell_atom_start_indices: wp.array(dtype=wp.int32),
    cell_atom_list: wp.array(dtype=wp.int32),
    sorted_positions: wp.array(dtype=wp.vec3f),
    sorted_atom_periodic_shifts: wp.array(dtype=wp.vec3i),
    cell_to_system: wp.array(dtype=wp.int32),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    neighbor_vectors: wp.array(dtype=wp.vec3f, ndim=2),
    neighbor_distances: wp.array(dtype=wp.float32, ndim=2),
    cutoff: wp.float32,
    total_cells: wp.int32,
    n_outer: wp.int32,
    R_max_x: wp.int32,
    R_max_y: wp.int32,
    R_max_z: wp.int32,
) -> None:
    _run_batch_query_cell_list_pair_centric(
        positions,
        cell,
        pbc,
        cells_per_dimension,
        neighbor_search_radius,
        cell_offsets,
        cells_per_system,
        atom_periodic_shifts,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
        sorted_positions,
        sorted_atom_periodic_shifts,
        cell_to_system,
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
        float(cutoff),
        wp.float32,
        int(total_cells),
        int(n_outer),
        (int(R_max_x), int(R_max_y), int(R_max_z)),
        return_vectors=True,
        return_distances=True,
        neighbor_vectors=neighbor_vectors,
        neighbor_distances=neighbor_distances,
    )


def _batch_query_cell_list_pair_centric_geom_f64(
    positions: wp.array(dtype=wp.vec3d),
    cell: wp.array(dtype=wp.mat33d),
    pbc: wp.array(dtype=wp.bool, ndim=2),
    cells_per_dimension: wp.array(dtype=wp.vec3i),
    neighbor_search_radius: wp.array(dtype=wp.vec3i),
    cell_offsets: wp.array(dtype=wp.int32),
    cells_per_system: wp.array(dtype=wp.int32),
    atom_periodic_shifts: wp.array(dtype=wp.vec3i),
    atoms_per_cell_count: wp.array(dtype=wp.int32),
    cell_atom_start_indices: wp.array(dtype=wp.int32),
    cell_atom_list: wp.array(dtype=wp.int32),
    sorted_positions: wp.array(dtype=wp.vec3d),
    sorted_atom_periodic_shifts: wp.array(dtype=wp.vec3i),
    cell_to_system: wp.array(dtype=wp.int32),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    neighbor_vectors: wp.array(dtype=wp.vec3d, ndim=2),
    neighbor_distances: wp.array(dtype=wp.float64, ndim=2),
    cutoff: wp.float64,
    total_cells: wp.int32,
    n_outer: wp.int32,
    R_max_x: wp.int32,
    R_max_y: wp.int32,
    R_max_z: wp.int32,
) -> None:
    _run_batch_query_cell_list_pair_centric(
        positions,
        cell,
        pbc,
        cells_per_dimension,
        neighbor_search_radius,
        cell_offsets,
        cells_per_system,
        atom_periodic_shifts,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
        sorted_positions,
        sorted_atom_periodic_shifts,
        cell_to_system,
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
        float(cutoff),
        wp.float64,
        int(total_cells),
        int(n_outer),
        (int(R_max_x), int(R_max_y), int(R_max_z)),
        return_vectors=True,
        return_distances=True,
        neighbor_vectors=neighbor_vectors,
        neighbor_distances=neighbor_distances,
    )


_jax_batch_query_cell_list_pair_centric_geom_f32 = jax_callable(
    _batch_query_cell_list_pair_centric_geom_f32,
    num_outputs=len(_BATCH_PAIR_CENTRIC_GEOM_INOUT),
    in_out_argnames=_BATCH_PAIR_CENTRIC_GEOM_INOUT,
    graph_mode=GraphMode.NONE,
)
_jax_batch_query_cell_list_pair_centric_geom_f64 = jax_callable(
    _batch_query_cell_list_pair_centric_geom_f64,
    num_outputs=len(_BATCH_PAIR_CENTRIC_GEOM_INOUT),
    in_out_argnames=_BATCH_PAIR_CENTRIC_GEOM_INOUT,
    graph_mode=GraphMode.NONE,
)


@functools.cache
def _get_jax_batch_cell_list_pair_centric_pair_fn_callable(pair_fn, wp_dtype):
    """Build (and cache) a batched pair-centric ``jax_callable`` closing over
    ``pair_fn`` (mirrors the single-system
    ``_get_jax_cell_list_pair_centric_pair_fn_callable``).  Two literal-typed
    callbacks keep the Warp annotations resolvable; cached by
    ``(pair_fn identity, wp_dtype)``.  ``GraphMode.NONE`` + host-read static
    sizing scalars.
    """
    if wp_dtype == wp.float64:

        def _callback(
            positions: wp.array(dtype=wp.vec3d),
            cell: wp.array(dtype=wp.mat33d),
            pbc: wp.array(dtype=wp.bool, ndim=2),
            cells_per_dimension: wp.array(dtype=wp.vec3i),
            neighbor_search_radius: wp.array(dtype=wp.vec3i),
            cell_offsets: wp.array(dtype=wp.int32),
            cells_per_system: wp.array(dtype=wp.int32),
            atom_periodic_shifts: wp.array(dtype=wp.vec3i),
            atoms_per_cell_count: wp.array(dtype=wp.int32),
            cell_atom_start_indices: wp.array(dtype=wp.int32),
            cell_atom_list: wp.array(dtype=wp.int32),
            sorted_positions: wp.array(dtype=wp.vec3d),
            sorted_atom_periodic_shifts: wp.array(dtype=wp.vec3i),
            cell_to_system: wp.array(dtype=wp.int32),
            neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
            neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
            num_neighbors: wp.array(dtype=wp.int32),
            neighbor_vectors: wp.array(dtype=wp.vec3d, ndim=2),
            neighbor_distances: wp.array(dtype=wp.float64, ndim=2),
            pair_params: wp.array(dtype=wp.float64, ndim=2),
            pair_energies: wp.array(dtype=wp.float64, ndim=2),
            pair_forces: wp.array(dtype=wp.vec3d, ndim=2),
            cutoff: wp.float64,
            total_cells: wp.int32,
            n_outer: wp.int32,
            R_max_x: wp.int32,
            R_max_y: wp.int32,
            R_max_z: wp.int32,
        ) -> None:
            _run_batch_query_cell_list_pair_centric(
                positions,
                cell,
                pbc,
                cells_per_dimension,
                neighbor_search_radius,
                cell_offsets,
                cells_per_system,
                atom_periodic_shifts,
                atoms_per_cell_count,
                cell_atom_start_indices,
                cell_atom_list,
                sorted_positions,
                sorted_atom_periodic_shifts,
                cell_to_system,
                neighbor_matrix,
                neighbor_matrix_shifts,
                num_neighbors,
                float(cutoff),
                wp.float64,
                int(total_cells),
                int(n_outer),
                (int(R_max_x), int(R_max_y), int(R_max_z)),
                return_vectors=True,
                return_distances=True,
                pair_fn=pair_fn,
                pair_params=pair_params,
                neighbor_vectors=neighbor_vectors,
                neighbor_distances=neighbor_distances,
                pair_energies=pair_energies,
                pair_forces=pair_forces,
            )

    else:

        def _callback(
            positions: wp.array(dtype=wp.vec3f),
            cell: wp.array(dtype=wp.mat33f),
            pbc: wp.array(dtype=wp.bool, ndim=2),
            cells_per_dimension: wp.array(dtype=wp.vec3i),
            neighbor_search_radius: wp.array(dtype=wp.vec3i),
            cell_offsets: wp.array(dtype=wp.int32),
            cells_per_system: wp.array(dtype=wp.int32),
            atom_periodic_shifts: wp.array(dtype=wp.vec3i),
            atoms_per_cell_count: wp.array(dtype=wp.int32),
            cell_atom_start_indices: wp.array(dtype=wp.int32),
            cell_atom_list: wp.array(dtype=wp.int32),
            sorted_positions: wp.array(dtype=wp.vec3f),
            sorted_atom_periodic_shifts: wp.array(dtype=wp.vec3i),
            cell_to_system: wp.array(dtype=wp.int32),
            neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
            neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
            num_neighbors: wp.array(dtype=wp.int32),
            neighbor_vectors: wp.array(dtype=wp.vec3f, ndim=2),
            neighbor_distances: wp.array(dtype=wp.float32, ndim=2),
            pair_params: wp.array(dtype=wp.float32, ndim=2),
            pair_energies: wp.array(dtype=wp.float32, ndim=2),
            pair_forces: wp.array(dtype=wp.vec3f, ndim=2),
            cutoff: wp.float32,
            total_cells: wp.int32,
            n_outer: wp.int32,
            R_max_x: wp.int32,
            R_max_y: wp.int32,
            R_max_z: wp.int32,
        ) -> None:
            _run_batch_query_cell_list_pair_centric(
                positions,
                cell,
                pbc,
                cells_per_dimension,
                neighbor_search_radius,
                cell_offsets,
                cells_per_system,
                atom_periodic_shifts,
                atoms_per_cell_count,
                cell_atom_start_indices,
                cell_atom_list,
                sorted_positions,
                sorted_atom_periodic_shifts,
                cell_to_system,
                neighbor_matrix,
                neighbor_matrix_shifts,
                num_neighbors,
                float(cutoff),
                wp.float32,
                int(total_cells),
                int(n_outer),
                (int(R_max_x), int(R_max_y), int(R_max_z)),
                return_vectors=True,
                return_distances=True,
                pair_fn=pair_fn,
                pair_params=pair_params,
                neighbor_vectors=neighbor_vectors,
                neighbor_distances=neighbor_distances,
                pair_energies=pair_energies,
                pair_forces=pair_forces,
            )

    return jax_callable(
        _callback,
        num_outputs=len(_BATCH_PAIR_CENTRIC_PAIR_FN_INOUT),
        in_out_argnames=_BATCH_PAIR_CENTRIC_PAIR_FN_INOUT,
        graph_mode=GraphMode.NONE,
    )


def estimate_batch_cell_list_sizes(
    positions: jax.Array,
    batch_ptr: jax.Array | None = None,
    batch_idx: jax.Array | None = None,
    cell: jax.Array | None = None,
    cutoff: float = 5.0,
    pbc: jax.Array | None = None,
    buffer_factor: float = 1.5,
) -> tuple[int, jax.Array, jax.Array]:
    """Estimate required batch cell list sizes.

    Parameters
    ----------
    positions : jax.Array, shape (total_atoms, 3), dtype=float32 or float64
        Atomic coordinates.
    batch_ptr : jax.Array, shape (num_systems + 1,), dtype=int32, optional
        Cumulative atom counts.
    batch_idx : jax.Array, shape (total_atoms,), dtype=int32, optional
        Batch indices for each atom.
    cell : jax.Array, shape (num_systems, 3, 3), dtype=float32 or float64, optional
        Cell matrices for each system.
    cutoff : float, optional
        Cutoff distance. Default is 5.0.
    pbc : jax.Array, shape (num_systems, 3), dtype=bool, optional
        PBC flags.
    buffer_factor : float, optional
        Buffer multiplier. Default is 1.5.

    Returns
    -------
    max_total_cells : int
        Maximum total cells to allocate.
    cells_per_dimension : jax.Array, shape (num_systems, 3)
        Cells per dimension for each system.
    neighbor_search_radius : jax.Array, shape (num_systems, 3)
        Search radius for each system.

    .. warning::

        This function is **not compatible with** ``jax.jit``. The returned
        ``max_total_cells`` is used to determine array allocation sizes, which
        must be concrete (statically known) at JAX trace time. When using
        ``batch_cell_list`` or ``batch_build_cell_list`` inside ``jax.jit``,
        provide ``max_total_cells`` explicitly to bypass this function.
    """

    # Prepare batch info
    batch_idx, batch_ptr = prepare_batch_idx_ptr(
        batch_idx, batch_ptr, positions.shape[0]
    )
    num_systems = batch_ptr.shape[0] - 1
    cell_dtype = positions.dtype if positions.dtype == jnp.float64 else jnp.float32
    cell, _pbc_bool = _normalize_batch_cell_pbc(
        cell,
        pbc,
        num_systems=num_systems,
        dtype=cell_dtype,
    )

    # Simple estimation per system
    max_total_cells = 0
    cells_per_dim_list = []
    search_radius_list = []

    for sys_idx in range(num_systems):
        start_idx = batch_ptr[sys_idx]
        end_idx = batch_ptr[sys_idx + 1]
        num_atoms_in_sys = end_idx - start_idx

        if num_atoms_in_sys == 0:
            cells_per_dim_list.append(jnp.ones(3, dtype=jnp.int32))
            search_radius_list.append(jnp.ones(3, dtype=jnp.int32))
            continue

        # Volume estimation
        det = jnp.linalg.det(cell[sys_idx])
        volume = jnp.abs(det)

        cell_volume = cutoff**3
        # TODO: This estimation derives array sizes from traced input data (cell
        # geometry), which is fundamentally incompatible with jax.jit compilation.
        # The JAX bindings need a refactored usage pattern where sizing is always
        # performed outside the JIT boundary, or a fixed upper-bound allocation
        # strategy is adopted.
        num_cells_est = max(int(volume / cell_volume * buffer_factor), 8)
        max_total_cells += num_cells_est

        cells_per_dim = jnp.ceil(num_cells_est ** (1 / 3)).astype(jnp.int32)
        cells_per_dim_list.append(cells_per_dim * jnp.ones(3, dtype=jnp.int32))
        search_radius_list.append(jnp.ones(3, dtype=jnp.int32))

    cells_per_dimension = jnp.stack(cells_per_dim_list, axis=0)
    neighbor_search_radius = jnp.stack(search_radius_list, axis=0)

    return max_total_cells, cells_per_dimension, neighbor_search_radius


def batch_build_cell_list(
    positions: jax.Array,
    batch_idx: jax.Array | None = None,
    batch_ptr: jax.Array | None = None,
    cell: jax.Array | None = None,
    pbc: jax.Array | None = None,
    cutoff: float = 5.0,
    max_total_cells: int | None = None,
    target_indices: jax.Array | None = None,
    return_vectors: bool = False,
    return_distances: bool = False,
    pair_fn: wp.Function | None = None,
    pair_params: jax.Array | None = None,
    neighbor_vectors: jax.Array | None = None,
    neighbor_distances: jax.Array | None = None,
    pair_energies: jax.Array | None = None,
    pair_forces: jax.Array | None = None,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Build spatial cell lists for batch of systems.

    Parameters
    ----------
    positions : jax.Array, shape (total_atoms, 3), dtype=float32 or float64
        Atomic coordinates.
    batch_idx : jax.Array, shape (total_atoms,), dtype=int32, optional
        Batch indices.
    batch_ptr : jax.Array, shape (num_systems + 1,), dtype=int32, optional
        Cumulative atom counts.
    cell : jax.Array, shape (num_systems, 3, 3), dtype=float32 or float64, optional
        Cell matrices.
    pbc : jax.Array, shape (num_systems, 3), dtype=bool, optional
        PBC flags.
    cutoff : float, optional
        Cutoff distance. Default is 5.0.
    max_total_cells : int, optional
        Maximum cells. If None, will be estimated.

    Returns
    -------
    cells_per_dimension : jax.Array, shape (num_systems, 3), dtype=int32
        Number of cells in x, y, z directions for each system.
    atom_periodic_shifts : jax.Array, shape (total_atoms, 3), dtype=int32
        Periodic boundary crossings for each atom.
    atom_to_cell_mapping : jax.Array, shape (total_atoms, 3), dtype=int32
        3D cell coordinates for each atom.
    atoms_per_cell_count : jax.Array, shape (max_total_cells,), dtype=int32
        Number of atoms in each cell.
    cell_atom_start_indices : jax.Array, shape (max_total_cells,), dtype=int32
        Starting index in ``cell_atom_list`` for each cell.
    cell_atom_list : jax.Array, shape (total_atoms,), dtype=int32
        Flattened list of atom indices organized by cell.
    neighbor_search_radius : jax.Array, shape (num_systems, 3), dtype=int32
        Search radius in neighboring cells for each system.
    cell_origin : jax.Array, shape (3,), dtype same as positions
        Cell origin point (currently zeros).

    Notes
    -----
    When calling inside ``jax.jit``, ``max_total_cells`` **must** be provided
    to avoid calling ``estimate_batch_cell_list_sizes``, which is not JIT-compatible.
    """
    if _has_partial_or_pair_outputs(
        target_indices=target_indices,
        return_vectors=return_vectors,
        return_distances=return_distances,
        pair_fn=pair_fn,
        pair_params=pair_params,
        neighbor_vectors=neighbor_vectors,
        neighbor_distances=neighbor_distances,
        pair_energies=pair_energies,
        pair_forces=pair_forces,
    ):
        raise NotImplementedError(
            "batch_build_cell_list does not accept return_distances / "
            "return_vectors / target_indices / pair_fn-related kwargs.  "
            "Use batch_query_cell_list() or the top-level batch_cell_list() "
            "wrapper.",
        )

    # Prepare batch info
    batch_idx, batch_ptr = prepare_batch_idx_ptr(
        batch_idx, batch_ptr, positions.shape[0]
    )
    num_systems = batch_ptr.shape[0] - 1
    cell_dtype = positions.dtype if positions.dtype == jnp.float64 else jnp.float32
    cell, pbc_bool = _normalize_batch_cell_pbc(
        cell,
        pbc,
        num_systems=num_systems,
        dtype=cell_dtype,
    )

    if max_total_cells is None:
        max_total_cells, cells_per_dim_est, neighbor_search_radius = (
            estimate_batch_cell_list_sizes(
                positions, batch_ptr, batch_idx, cell, cutoff, pbc
            )
        )
        # Ensure neighbor_search_radius is on the correct device
        neighbor_search_radius = neighbor_search_radius
    else:
        neighbor_search_radius = jnp.ones((num_systems, 3), dtype=jnp.int32)

    # Allocate cell list tensors
    (
        cells_per_dimension,
        neighbor_search_radius,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
    ) = allocate_cell_list(
        positions.shape[0],
        max_total_cells,
        neighbor_search_radius,
    )

    # Select kernels based on dtype
    if positions.dtype == jnp.float64:
        _construct = _jax_batch_construct_bin_size_f64
        _count = _jax_batch_count_atoms_per_bin_f64
        _bin = _jax_batch_bin_atoms_f64
    else:
        _construct = _jax_batch_construct_bin_size_f32
        _count = _jax_batch_count_atoms_per_bin_f32
        _bin = _jax_batch_bin_atoms_f32
        positions = positions.astype(jnp.float32)

    if cell.dtype != positions.dtype:
        cell = cell.astype(positions.dtype)

    empty_bool1d = jnp.zeros((0,), dtype=jnp.bool_)
    empty_i32 = jnp.zeros((0,), dtype=jnp.int32)

    total_atoms = positions.shape[0]

    # Step 1: Construct bin sizes (one thread per system)
    (cells_per_dimension,) = _construct(
        cell,
        empty_bool1d,
        pbc_bool,
        empty_i32,
        cells_per_dimension,
        float(cutoff),
        int(max_total_cells),
        launch_dims=(num_systems,),
    )

    # Step 2: Compute cells_per_system and cell_offsets
    cells_per_system = jnp.zeros(num_systems, dtype=jnp.int32)
    (cells_per_system,) = _jax_compute_cells_per_system(
        cells_per_dimension,
        cells_per_system,
        launch_dims=(num_systems,),
    )
    cell_offsets = jnp.concatenate(
        [
            jnp.array([0], dtype=jnp.int32),
            jnp.cumsum(cells_per_system[:-1], dtype=jnp.int32),
        ]
    )

    # Step 3: Count atoms per bin
    atoms_per_cell_count, atom_periodic_shifts = _count(
        positions,
        cell,
        empty_bool1d,
        pbc_bool,
        batch_idx,
        empty_i32,
        cells_per_dimension,
        cell_offsets,
        atoms_per_cell_count,
        atom_periodic_shifts,
        launch_dims=(total_atoms,),
    )

    # Step 4: Compute exclusive prefix sum (replaces wp.utils.array_scan)
    cell_atom_start_indices = jnp.concatenate(
        [
            jnp.array([0], dtype=jnp.int32),
            jnp.cumsum(atoms_per_cell_count[:-1], dtype=jnp.int32),
        ]
    )

    # Step 5: Zero counts before second pass
    atoms_per_cell_count = jnp.zeros_like(atoms_per_cell_count)

    # Step 6: Bin atoms
    atom_to_cell_mapping, atoms_per_cell_count, cell_atom_list = _bin(
        positions,
        cell,
        empty_bool1d,
        pbc_bool,
        batch_idx,
        empty_i32,
        cells_per_dimension,
        cell_offsets,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
        launch_dims=(total_atoms,),
    )

    cell_origin = jnp.zeros(3, dtype=positions.dtype)

    return (
        cells_per_dimension,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
        neighbor_search_radius,
        cell_origin,
    )


def batch_query_cell_list(
    positions: jax.Array,
    batch_idx: jax.Array | None = None,
    batch_ptr: jax.Array | None = None,
    cutoff: float = 5.0,
    cell: jax.Array | None = None,
    pbc: jax.Array | None = None,
    cells_per_dimension: jax.Array | None = None,
    atom_periodic_shifts: jax.Array | None = None,
    atom_to_cell_mapping: jax.Array | None = None,
    cell_atom_start_indices: jax.Array | None = None,
    cell_atom_list: jax.Array | None = None,
    atoms_per_cell_count: jax.Array | None = None,
    neighbor_search_radius: jax.Array | None = None,
    max_neighbors: int | None = None,
    neighbor_matrix: jax.Array | None = None,
    num_neighbors: jax.Array | None = None,
    neighbor_matrix_shifts: jax.Array | None = None,
    rebuild_flags: jax.Array | None = None,
    half_fill: bool = False,
    strategy: str = "auto",
    atom_centric_path: str = "auto",
    target_indices: jax.Array | None = None,
    return_vectors: bool = False,
    return_distances: bool = False,
    pair_fn: wp.Function | None = None,
    pair_params: jax.Array | None = None,
    neighbor_vectors: jax.Array | None = None,
    neighbor_distances: jax.Array | None = None,
    pair_energies: jax.Array | None = None,
    pair_forces: jax.Array | None = None,
) -> tuple[jax.Array, ...]:
    """Query batch cell lists to find neighbors.

    Parameters
    ----------
    positions : jax.Array, shape (total_atoms, 3), dtype=float32 or float64
        Atomic coordinates.
    batch_idx : jax.Array, shape (total_atoms,), dtype=int32, optional
        Batch indices.
    batch_ptr : jax.Array, shape (num_systems + 1,), dtype=int32, optional
        Cumulative atom counts.
    cutoff : float, optional
        Cutoff distance.
    cell : jax.Array, shape (num_systems, 3, 3), dtype=float32 or float64, optional
        Cell matrices.
    pbc : jax.Array, shape (num_systems, 3), dtype=bool, optional
        PBC flags.
    cells_per_dimension : jax.Array, shape (num_systems, 3), dtype=int32, optional
        Cells per dimension.
    atom_periodic_shifts : jax.Array, shape (total_atoms, 3), dtype=int32, optional
        Periodic shifts for each atom (output from ``batch_build_cell_list``).
    atom_to_cell_mapping : jax.Array, shape (total_atoms, 3), dtype=int32, optional
        Cell mappings.
    cell_atom_start_indices : jax.Array, shape (max_total_cells,), dtype=int32, optional
        Start indices.
    cell_atom_list : jax.Array, shape (total_atoms,), dtype=int32, optional
        Cell atom list.
    atoms_per_cell_count : jax.Array, shape (max_total_cells,), dtype=int32, optional
        Number of atoms assigned to each cell. Output from ``batch_build_cell_list``.
    neighbor_search_radius : jax.Array, shape (num_systems, 3), dtype=int32, optional
        Search radius.
    max_neighbors : int, optional
        Maximum neighbors per atom.
    neighbor_matrix : jax.Array, shape (num_rows, max_neighbors), dtype=int32, optional
        Pre-shaped neighbor matrix. ``num_rows`` is ``total_atoms`` normally and
        ``len(target_indices)`` for partial rows.
    num_neighbors : jax.Array, shape (num_rows,), dtype=int32, optional
        Pre-shaped neighbors count array.
    neighbor_matrix_shifts : jax.Array, shape (num_rows, max_neighbors, 3), dtype=int32, optional
        Pre-allocated shift vectors array. Pass in a pre-shaped array to hint buffer
        reuse to XLA; note that JAX returns a new array rather than mutating the input.
    half_fill : bool, optional
        If True, build a half neighbor list (each pair stored once) using the
        half-fill kernel specialization. Default is False.
    strategy : {"auto", "atom_centric", "pair_centric"}, default "auto"
        Cell-list query sub-strategy.  Both strategies produce identical pair
        SETS; only the per-row ordering inside ``neighbor_matrix`` differs
        (pair-centric accumulates via ``atomic_add`` so its row order is
        nondeterministic).  ``"auto"`` resolves via
        :func:`select_cell_list_strategy` ``(total_atoms, cutoff)`` on GPU and
        to ``"atom_centric"`` on CPU.  ``"pair_centric"`` is CUDA-only and
        requires a **concrete** ``neighbor_search_radius`` (its launch grid is
        sized by host-read ``total_cells`` / ``n_outer`` / ``R_max`` scalars
        baked at launch-build time): it works eagerly / outside ``jax.jit`` but
        raises a clear error under ``jax.jit`` with a traced radius.  ``"auto"``
        falls back to ``"atom_centric"`` when pair-centric launch sizing is
        traced.  It is full-fill only (``half_fill=True`` + explicit
        ``pair_centric`` raises) and is registered with ``GraphMode.NONE``.
    atom_centric_path : {"auto", "direct", "sorted"}, default "auto"
        Accepted for signature parity with the Torch binding.  JAX registers
        only the *sorted* atom-centric query kernel, so this option never
        branches: every JAX atom-centric query runs the sorted kernel
        regardless of this value (a documented divergence from Torch, whose
        ``"auto"`` maps to a distinct ``"direct"`` kernel).
    target_indices : jax.Array, shape (num_targets,), dtype=int32, optional
        Compact partial-list source rows. Output row ``r`` maps to atom
        ``target_indices[r]``; COO source rows remain compact row ids.

    Returns
    -------
    results : tuple of jax.Array
        Variable-length tuple depending on requested outputs. Matrix outputs use
        ``num_rows`` rows, where ``num_rows`` is ``total_atoms`` normally and
        ``len(target_indices)`` for partial lists. The base return is
        ``(neighbor_matrix, num_neighbors, neighbor_matrix_shifts)``; optional
        distance/vector arrays and ``pair_fn`` energy/force arrays are appended
        in the same order as ``batch_cell_list``.
    """

    has_pair_outputs = _has_partial_or_pair_outputs(
        target_indices=target_indices,
        return_vectors=return_vectors,
        return_distances=return_distances,
        pair_fn=pair_fn,
        pair_params=pair_params,
        neighbor_vectors=neighbor_vectors,
        neighbor_distances=neighbor_distances,
        pair_energies=pair_energies,
        pair_forces=pair_forces,
    )
    _validate_pair_kwargs(
        pair_fn=pair_fn,
        pair_params=pair_params,
        pair_energies=pair_energies,
        pair_forces=pair_forces,
    )
    if has_pair_outputs and rebuild_flags is not None:
        raise NotImplementedError(
            "return_distances / return_vectors / target_indices / pair_fn "
            "are not supported with rebuild_flags in batch_query_cell_list.",
        )
    if strategy == "pair_centric" and target_indices is not None:
        raise NotImplementedError(
            "strategy='pair_centric' with target_indices (partial neighbor "
            "lists) is not wired through the JAX batch_cell_list binding. Use "
            "strategy='atom_centric' (or 'auto') for identical results.",
        )

    # Validate the sub-strategy options.  ``atom_centric_path`` is accepted for
    # parity but never branches (JAX always runs the sorted atom-centric
    # kernel).  ``strategy`` resolution / launch-safety guards happen below once
    # the concrete ``neighbor_search_radius`` is available.
    _validate_atom_centric_path(atom_centric_path)
    # Only an EXPLICIT pair-centric request collides with half_fill; auto falls
    # back to atom_centric (resolved below).  JAX batch cell_list pair-centric
    # is full-fill, mirroring the single-system path.
    if strategy == "pair_centric" and half_fill:
        raise NotImplementedError(
            "strategy='pair_centric' with half_fill=True is not supported in "
            "the JAX batch cell-list binding (JAX cell_list is full-fill).  Use "
            "strategy='atom_centric' for half_fill, or half_fill=False for "
            "pair-centric.",
        )

    if max_neighbors is None and neighbor_matrix is not None:
        max_neighbors = int(neighbor_matrix.shape[1])
    if max_neighbors is None:
        max_neighbors = estimate_max_neighbors(cutoff)

    # Prepare batch info
    batch_idx, batch_ptr = prepare_batch_idx_ptr(
        batch_idx, batch_ptr, positions.shape[0]
    )
    num_systems = batch_ptr.shape[0] - 1
    cell_dtype = positions.dtype if positions.dtype == jnp.float64 else jnp.float32
    cell, pbc_bool = _normalize_batch_cell_pbc(
        cell,
        pbc,
        num_systems=num_systems,
        dtype=cell_dtype,
    )
    if target_indices is not None:
        target_indices = jnp.asarray(target_indices, dtype=jnp.int32)
        num_rows = int(target_indices.shape[0])
    else:
        num_rows = positions.shape[0]
    _validate_compact_target_buffers(
        target_indices=target_indices,
        num_rows=int(num_rows),
        max_neighbors=int(max_neighbors),
        neighbor_matrix=neighbor_matrix,
        neighbor_matrix_shifts=neighbor_matrix_shifts,
        num_neighbors=num_neighbors,
        neighbor_distances=neighbor_distances,
        neighbor_vectors=neighbor_vectors,
        pair_energies=pair_energies,
        pair_forces=pair_forces,
    )

    if neighbor_matrix is None:
        neighbor_matrix = jnp.full(
            (num_rows, max_neighbors),
            positions.shape[0],
            dtype=jnp.int32,
        )
    elif rebuild_flags is None:
        neighbor_matrix = neighbor_matrix.at[:].set(jnp.int32(positions.shape[0]))

    if num_neighbors is None:
        num_neighbors = jnp.zeros(num_rows, dtype=jnp.int32)
    elif rebuild_flags is None:
        num_neighbors = num_neighbors.at[:].set(jnp.int32(0))

    # Select kernels based on dtype; same sorted-reads kernel for selective
    # and non-selective (controlled by ``rebuild_flags``).
    if positions.dtype == jnp.float64:
        _gather_kernel = _jax_batch_gather_positions_by_cell_f64
        _sorted_build_kernel = (
            _jax_batch_build_neighbor_matrix_local_count_sorted_half_f64
            if half_fill
            else _jax_batch_build_neighbor_matrix_local_count_sorted_f64
        )
    else:
        _gather_kernel = _jax_batch_gather_positions_by_cell_f32
        _sorted_build_kernel = (
            _jax_batch_build_neighbor_matrix_local_count_sorted_half_f32
            if half_fill
            else _jax_batch_build_neighbor_matrix_local_count_sorted_f32
        )
        positions = positions.astype(jnp.float32)

    if cell.dtype != positions.dtype:
        cell = cell.astype(positions.dtype)

    empty_bool1d = jnp.zeros((0,), dtype=jnp.bool_)
    empty_i32 = jnp.zeros((0,), dtype=jnp.int32)
    empty_scalar2d = jnp.zeros((0, 0), dtype=positions.dtype)
    empty_vec_matrix = jnp.zeros((0, 0, 3), dtype=positions.dtype)

    total_atoms = positions.shape[0]

    if neighbor_matrix_shifts is None:
        neighbor_matrix_shifts = jnp.zeros(
            (num_rows, max_neighbors, 3),
            dtype=jnp.int32,
        )
    elif rebuild_flags is None:
        neighbor_matrix_shifts = neighbor_matrix_shifts.at[:].set(jnp.int32(0))

    if atoms_per_cell_count is None:
        max_total_cells = cell_atom_start_indices.shape[0]
        atoms_per_cell_count = jnp.zeros(max_total_cells, dtype=jnp.int32)

    # Compute cell_offsets from cells_per_dimension
    cells_per_system = jnp.prod(cells_per_dimension, axis=1)
    cell_offsets = jnp.concatenate(
        [
            jnp.array([0], dtype=jnp.int32),
            jnp.cumsum(cells_per_system[:-1], dtype=jnp.int32),
        ]
    )

    batch_idx_i32 = batch_idx.astype(jnp.int32)

    if rebuild_flags is not None:
        rf = rebuild_flags.astype(jnp.bool_)
        atom_rebuild = rf[batch_idx_i32]
        num_neighbors = jnp.where(
            atom_rebuild, jnp.zeros_like(num_neighbors), num_neighbors
        )
    else:
        rf = jnp.ones((num_systems,), dtype=jnp.bool_)

    # Resolve the cell-list query sub-strategy.  Reuses the single-system
    # ``_resolve_cell_strategy`` (``select_cell_list_strategy(total_atoms,
    # cutoff)`` on GPU, atom_centric on CPU).  NOTE: the Torch batched path uses
    # ``select_batch_cell_list_strategy`` here; that is a perf heuristic only -
    # both strategies produce identical pair SETS, so reusing the single-system
    # resolver keeps the JAX bindings consistent and the CPU gating clear.
    # ``half_fill`` makes ``"auto"`` resolve to atom_centric (pair-centric is
    # full-fill only), so the default path keeps working for every geometry.
    device_is_cpu = _is_cpu_array(positions)
    chosen = _resolve_cell_strategy(
        strategy,
        total_atoms=int(total_atoms),
        cutoff=float(cutoff),
        device_is_cpu=device_is_cpu,
        half_fill=half_fill,
    )
    # The batched pair-centric jax_callable is non-selective + full-fill; a
    # selective (rebuild_flags) request falls back to the atom-centric kernel.
    if chosen == "pair_centric" and rebuild_flags is not None:
        chosen = "atom_centric"

    if chosen == "pair_centric":
        # Host-read the sizing scalars to bake the pair-centric launch grid.
        # ``R_max`` (cross-system max per-axis radius) and ``total_cells`` are
        # device->host syncs: legal eagerly / with a concrete radius, illegal
        # under jax.jit with a traced ``neighbor_search_radius`` / sizing.
        try:
            R_max_arr = jnp.max(neighbor_search_radius, axis=0)
            R_max = (
                int(R_max_arr[0]),
                int(R_max_arr[1]),
                int(R_max_arr[2]),
            )
            total_cells = int(jnp.sum(cells_per_system))
        except (
            jax.errors.ConcretizationTypeError,
            jax.errors.TracerIntegerConversionError,
        ) as exc:
            if strategy == "auto":
                chosen = "atom_centric"
            else:
                raise ValueError(
                    "strategy='pair_centric' needs a concrete "
                    "neighbor_search_radius to size its launch grid (total_cells / "
                    "n_outer / R_max are host-read).  Compute the cell-list sizing "
                    "outside jax.jit and pass a concrete neighbor_search_radius, or "
                    "use strategy='atom_centric'.",
                ) from exc
        else:
            # JAX batch cell_list is full-fill (half_fill+pair_centric raised above).
            n_outer = compute_batch_pair_centric_n_outer(R_max, False)
            if not is_pair_centric_launch_safe(total_cells, n_outer):
                if strategy == "pair_centric":
                    _raise_unsafe_pair_centric_launch(total_cells, n_outer, 64)
                chosen = "atom_centric"
            elif strategy == "auto" and not is_pair_centric_parallelism_sufficient(
                total_atoms, total_cells, n_outer
            ):
                chosen = "atom_centric"

    if has_pair_outputs:
        if neighbor_distances is None:
            neighbor_distances = jnp.zeros(
                (num_rows, max_neighbors),
                dtype=positions.dtype,
            )
        if neighbor_vectors is None:
            neighbor_vectors = jnp.zeros(
                (num_rows, max_neighbors, 3),
                dtype=positions.dtype,
            )
        pc_strategy = "pair_centric" if chosen == "pair_centric" else "atom_centric"
        forward_kwargs = {
            "pbc_bool": pbc_bool,
            "batch_idx_i32": batch_idx_i32,
            "cells_per_dimension": cells_per_dimension,
            "atom_periodic_shifts": atom_periodic_shifts,
            "atom_to_cell_mapping": atom_to_cell_mapping,
            "atoms_per_cell_count": atoms_per_cell_count,
            "cell_atom_start_indices": cell_atom_start_indices,
            "cell_atom_list": cell_atom_list,
            "cell_offsets": cell_offsets,
            "neighbor_search_radius": neighbor_search_radius,
            "neighbor_matrix": neighbor_matrix,
            "neighbor_matrix_shifts": neighbor_matrix_shifts,
            "num_neighbors": num_neighbors,
            "neighbor_vectors": neighbor_vectors,
            "neighbor_distances": neighbor_distances,
            "cutoff": cutoff,
            "pair_fn": pair_fn,
            "pair_params": pair_params,
            "target_indices": target_indices,
            "strategy": pc_strategy,
            "n_outer": n_outer if pc_strategy == "pair_centric" else 0,
            "total_cells": total_cells if pc_strategy == "pair_centric" else 0,
            "r_max": R_max if pc_strategy == "pair_centric" else (0, 0, 0),
            "half_fill": bool(half_fill),
        }
        route_out = _route_pair_outputs(
            positions,
            cell,
            _batch_cell_list_pair_outputs_forward,
            forward_kwargs,
        )
        if pair_fn is not None:
            distances_out, vectors_out, nm_out, nn_out, shifts_out, pe_out, pf_out = (
                route_out
            )
        else:
            distances_out, vectors_out, nm_out, nn_out, shifts_out = route_out
            pe_out = pf_out = None
        base = (nm_out, nn_out, shifts_out)
        tail: list = []
        if return_distances:
            tail.append(distances_out)
        if return_vectors:
            tail.append(vectors_out)
        if pair_fn is not None:
            tail.extend((pe_out, pf_out))
        return (*base, *tail)

    if chosen == "pair_centric":
        pair_query = (
            _jax_batch_query_cell_list_pair_centric_f64
            if positions.dtype == jnp.float64
            else _jax_batch_query_cell_list_pair_centric_f32
        )
        cells_per_system_i32 = cells_per_system.astype(jnp.int32)
        sorted_positions = jnp.zeros((total_atoms, 3), dtype=positions.dtype)
        sorted_atom_periodic_shifts = jnp.zeros((total_atoms, 3), dtype=jnp.int32)
        # ``cell_to_system`` is scratch the launcher's
        # ``_build_cell_to_system_map`` fills (>= total_cells entries).
        cell_to_system = jnp.zeros(max(total_cells, 1), dtype=jnp.int32)
        # The pair-centric callable runs the internal gather + cell_to_system
        # map and the pair-centric launch; the sizing scalars enter as static
        # args (``R_max`` split into three int32 scalars).
        (
            sorted_positions,
            sorted_atom_periodic_shifts,
            cell_to_system,
            neighbor_matrix,
            neighbor_matrix_shifts,
            num_neighbors,
        ) = pair_query(
            positions,
            cell,
            pbc_bool,
            cells_per_dimension,
            neighbor_search_radius,
            cell_offsets,
            cells_per_system_i32,
            atom_periodic_shifts,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
            sorted_positions,
            sorted_atom_periodic_shifts,
            cell_to_system,
            neighbor_matrix,
            neighbor_matrix_shifts,
            num_neighbors,
            float(cutoff),
            int(total_cells),
            int(n_outer),
            int(R_max[0]),
            int(R_max[1]),
            int(R_max[2]),
        )
        return neighbor_matrix, num_neighbors, neighbor_matrix_shifts

    sorted_positions = jnp.zeros((total_atoms, 3), dtype=positions.dtype)
    sorted_atom_periodic_shifts = jnp.zeros((total_atoms, 3), dtype=jnp.int32)
    sorted_positions, sorted_atom_periodic_shifts = _gather_kernel(
        positions,
        atom_periodic_shifts,
        cell_atom_list,
        sorted_positions,
        sorted_atom_periodic_shifts,
        launch_dims=(total_atoms,),
    )

    neighbor_matrix, neighbor_matrix_shifts, num_neighbors = _sorted_build_kernel(
        positions,
        atom_periodic_shifts,
        sorted_positions,
        sorted_atom_periodic_shifts,
        cell,
        empty_bool1d,
        pbc_bool,
        batch_idx_i32,
        float(cutoff),
        empty_i32,
        cells_per_dimension,
        empty_i32,
        neighbor_search_radius,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
        cell_offsets,
        empty_i32,
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
        empty_vec_matrix,
        empty_scalar2d,
        empty_scalar2d,
        empty_scalar2d,
        empty_vec_matrix,
        rf,
        launch_dims=(total_atoms,),
    )

    return neighbor_matrix, num_neighbors, neighbor_matrix_shifts


def _batch_cell_list_pair_outputs_forward(
    positions: jax.Array,
    cell: jax.Array,
    *,
    pbc_bool: jax.Array,
    batch_idx_i32: jax.Array,
    cells_per_dimension: jax.Array,
    atom_periodic_shifts: jax.Array,
    atom_to_cell_mapping: jax.Array,
    atoms_per_cell_count: jax.Array,
    cell_atom_start_indices: jax.Array,
    cell_atom_list: jax.Array,
    cell_offsets: jax.Array,
    neighbor_search_radius: jax.Array,
    neighbor_matrix: jax.Array,
    neighbor_matrix_shifts: jax.Array,
    num_neighbors: jax.Array,
    neighbor_vectors: jax.Array,
    neighbor_distances: jax.Array,
    cutoff: float,
    pair_fn=None,
    pair_params: jax.Array | None = None,
    target_indices: jax.Array | None = None,
    strategy: str = "atom_centric",
    n_outer: int | None = None,
    total_cells: int | None = None,
    r_max: tuple[int, int, int] | None = None,
    half_fill: bool = False,
) -> _NeighborForwardOutput:
    """Forward closure consumed by ``_route_pair_outputs``.

    Runs the gather + batched pair-output kernel.  The Warp launches do not
    propagate gradients across the JAX boundary, so positions/cell are
    detached here; the autograd primitive's reconstruction backward receives
    the live tensors separately.  When ``pair_fn`` is set, a ``pair_fn``-specialized
    kernel writes per-pair ``pair_energies`` / ``pair_forces`` which ride along in
    ``extra_outputs`` (forward-only).

    When ``target_indices`` is set (partial neighbor lists), the kernel runs
    with ``partial=True``: output row ``r`` maps to atom ``target_indices[r]``,
    the output buffers carry ``num_targets`` compact rows, and the kernel is
    launched ``(num_targets,)``.  The per-atom ``batch_idx`` is still indexed by
    the real atom index (``target_indices[r]``) in the backward.

    When ``strategy == "pair_centric"`` the path instead runs the block-scheduled
    batched pair-centric callable (gather + cell_to_system map + kernel), sized by
    the host-read static ``n_outer`` / ``total_cells`` / ``r_max`` scalars; the
    pair set is identical to atom-centric.  ``target_indices`` is rejected
    upstream for this strategy, so the shared tail is unchanged.
    """
    positions = jax.lax.stop_gradient(positions)
    cell = jax.lax.stop_gradient(cell)

    f64 = positions.dtype == jnp.float64
    gather_kernel = (
        _jax_batch_gather_positions_by_cell_f64
        if f64
        else _jax_batch_gather_positions_by_cell_f32
    )

    total_atoms = positions.shape[0]
    num_systems = pbc_bool.shape[0]
    # Output rows: ``num_targets`` for the partial (``target_indices``) path,
    # else ``total_atoms``.
    num_rows = neighbor_matrix.shape[0]
    max_neighbors = neighbor_matrix.shape[1]
    empty_scalar2d = jnp.zeros((0, 0), dtype=positions.dtype)
    empty_vec_matrix = jnp.zeros((0, 0, 3), dtype=positions.dtype)

    has_pair_fn = pair_fn is not None
    is_partial = target_indices is not None
    is_pair_centric = strategy == "pair_centric"
    wp_dtype = wp.float64 if f64 else wp.float32
    if has_pair_fn:
        pp_arg = jnp.asarray(pair_params, dtype=positions.dtype)
        pe = jnp.zeros((num_rows, max_neighbors), dtype=positions.dtype)
        pf = jnp.zeros((num_rows, max_neighbors, 3), dtype=positions.dtype)
    else:
        pp_arg = empty_scalar2d
        pe = None
        pf = None

    if is_pair_centric:
        # Batched pair-centric: the launcher gathers internally, builds the
        # cell->system map, and runs the block-scheduled kernel sized by the
        # host-read static scalars.  ``target_indices`` is rejected upstream.
        cells_per_system_i32 = jnp.prod(cells_per_dimension, axis=1).astype(jnp.int32)
        sorted_positions = jnp.zeros((total_atoms, 3), dtype=positions.dtype)
        sorted_atom_periodic_shifts = jnp.zeros((total_atoms, 3), dtype=jnp.int32)
        cell_to_system = jnp.zeros(max(int(total_cells), 1), dtype=jnp.int32)
        rmx, rmy, rmz = (int(r_max[0]), int(r_max[1]), int(r_max[2]))
        if has_pair_fn:
            pc_callable = _get_jax_batch_cell_list_pair_centric_pair_fn_callable(
                pair_fn, wp_dtype
            )
            (
                _sp,
                _sas,
                _cts,
                nm_out,
                nms_out,
                nn_out,
                nv_out,
                nd_out,
                pe,
                pf,
            ) = pc_callable(
                positions,
                cell,
                pbc_bool,
                cells_per_dimension,
                neighbor_search_radius,
                cell_offsets,
                cells_per_system_i32,
                atom_periodic_shifts,
                atoms_per_cell_count,
                cell_atom_start_indices,
                cell_atom_list,
                sorted_positions,
                sorted_atom_periodic_shifts,
                cell_to_system,
                neighbor_matrix,
                neighbor_matrix_shifts,
                num_neighbors,
                neighbor_vectors,
                neighbor_distances,
                pp_arg,
                pe,
                pf,
                float(cutoff),
                int(total_cells),
                int(n_outer),
                rmx,
                rmy,
                rmz,
            )
        else:
            pc_callable = (
                _jax_batch_query_cell_list_pair_centric_geom_f64
                if f64
                else _jax_batch_query_cell_list_pair_centric_geom_f32
            )
            (
                _sp,
                _sas,
                _cts,
                nm_out,
                nms_out,
                nn_out,
                nv_out,
                nd_out,
            ) = pc_callable(
                positions,
                cell,
                pbc_bool,
                cells_per_dimension,
                neighbor_search_radius,
                cell_offsets,
                cells_per_system_i32,
                atom_periodic_shifts,
                atoms_per_cell_count,
                cell_atom_start_indices,
                cell_atom_list,
                sorted_positions,
                sorted_atom_periodic_shifts,
                cell_to_system,
                neighbor_matrix,
                neighbor_matrix_shifts,
                num_neighbors,
                neighbor_vectors,
                neighbor_distances,
                float(cutoff),
                int(total_cells),
                int(n_outer),
                rmx,
                rmy,
                rmz,
            )
    else:
        if has_pair_fn or is_partial:
            pair_kernel = _get_jax_batch_cell_list_pair_outputs_kernel(
                pair_fn, wp_dtype, is_partial, half_fill
            )
        elif half_fill:
            pair_kernel = (
                _jax_batch_build_neighbor_matrix_local_count_sorted_pair_half_f64
                if f64
                else _jax_batch_build_neighbor_matrix_local_count_sorted_pair_half_f32
            )
        else:
            pair_kernel = (
                _jax_batch_build_neighbor_matrix_local_count_sorted_pair_f64
                if f64
                else _jax_batch_build_neighbor_matrix_local_count_sorted_pair_f32
            )
        ti_arg = (
            jnp.asarray(target_indices, dtype=jnp.int32)
            if is_partial
            else jnp.zeros((0,), dtype=jnp.int32)
        )

        sorted_positions = jnp.zeros((total_atoms, 3), dtype=positions.dtype)
        sorted_atom_periodic_shifts = jnp.zeros((total_atoms, 3), dtype=jnp.int32)
        sorted_positions, sorted_atom_periodic_shifts = gather_kernel(
            positions,
            atom_periodic_shifts,
            cell_atom_list,
            sorted_positions,
            sorted_atom_periodic_shifts,
            launch_dims=(total_atoms,),
        )

        empty_bool1d = jnp.zeros((0,), dtype=jnp.bool_)
        empty_i32 = jnp.zeros((0,), dtype=jnp.int32)
        rf = jnp.ones((num_systems,), dtype=jnp.bool_)

        outs = pair_kernel(
            positions,
            atom_periodic_shifts,
            sorted_positions,
            sorted_atom_periodic_shifts,
            cell,
            empty_bool1d,
            pbc_bool,
            batch_idx_i32,
            float(cutoff),
            empty_i32,
            cells_per_dimension,
            empty_i32,
            neighbor_search_radius,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
            cell_offsets,
            ti_arg,  # target_indices (real rows when partial, else 0-size sentinel)
            neighbor_matrix,
            neighbor_matrix_shifts,
            num_neighbors,
            neighbor_vectors,
            neighbor_distances,
            pp_arg,  # pair_params
            pe if has_pair_fn else empty_scalar2d,  # pair_energies
            pf if has_pair_fn else empty_vec_matrix,  # pair_forces
            rf,
            launch_dims=(num_rows,),
        )
        if has_pair_fn:
            nm_out, nms_out, nn_out, nv_out, nd_out, pe, pf = outs
        else:
            nm_out, nms_out, nn_out, nv_out, nd_out = outs

    i_idx, j_idx, shifts_ret, _, mask_ = _build_index_residuals(
        nm_out,
        nn_out,
        nms_out,
        target_indices=ti_arg if is_partial else None,
    )
    K, M = nm_out.shape
    extra_outputs = (
        (nm_out, nn_out, nms_out, pe, pf) if has_pair_fn else (nm_out, nn_out, nms_out)
    )
    return _NeighborForwardOutput(
        distances=nd_out,
        vectors=nv_out,
        extra_outputs=extra_outputs,
        i_idx=i_idx,
        j_idx=j_idx,
        shifts=shifts_ret,
        batch_idx=batch_idx_i32,
        active_mask=mask_,
        matrix_shape=(K, M),
    )


def batch_cell_list(
    positions: jax.Array,
    cutoff: float,
    cell: jax.Array | None = None,
    pbc: jax.Array | None = None,
    batch_idx: jax.Array | None = None,
    batch_ptr: jax.Array | None = None,
    max_neighbors: int | None = None,
    max_total_cells: int | None = None,
    neighbor_matrix_shifts: jax.Array | None = None,
    return_neighbor_list: bool = False,
    half_fill: bool = False,
    fill_value: int | None = None,
    strategy: str = "auto",
    atom_centric_path: str = "auto",
    target_indices: jax.Array | None = None,
    return_vectors: bool = False,
    return_distances: bool = False,
    pair_fn: wp.Function | None = None,
    pair_params: jax.Array | None = None,
    neighbor_vectors: jax.Array | None = None,
    neighbor_distances: jax.Array | None = None,
    pair_energies: jax.Array | None = None,
    pair_forces: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array] | tuple[jax.Array, jax.Array, tuple]:
    """Build and query spatial cell lists for batch of systems.

    Parameters
    ----------
    positions : jax.Array, shape (total_atoms, 3), dtype=float32 or float64
        Atomic coordinates.
    cutoff : float
        Cutoff distance for neighbor detection.
    cell : jax.Array, shape (num_systems, 3, 3), dtype=float32 or float64, optional
        Cell matrices defining lattice vectors. Default is identity matrix.
    pbc : jax.Array, shape (num_systems, 3), dtype=bool, optional
        Periodic boundary condition flags. Default is all True.
    batch_idx : jax.Array, shape (total_atoms,), dtype=int32, optional
        Batch indices for each atom.
    batch_ptr : jax.Array, shape (num_systems + 1,), dtype=int32, optional
        Cumulative atom counts defining system boundaries.
    max_neighbors : int, optional
        Maximum number of neighbors per atom. If None, will be estimated.
    max_total_cells : int, optional
        Maximum number of cells to allocate. If None, will be estimated.
    neighbor_matrix_shifts : jax.Array, shape (total_atoms, max_neighbors, 3), dtype=int32, optional
        Pre-allocated shift vectors array. If None, will be allocated internally.
        Pass in a pre-shaped array to hint buffer reuse to XLA; note that JAX returns
        a new array rather than mutating the input.
    return_neighbor_list : bool, optional
        If True, convert result to COO neighbor list format. Default is False.
    half_fill : bool, optional
        If True, build a half neighbor list (each pair stored once) using the
        half-fill kernel specialization. Default is False.
    fill_value : int, optional
        Value used to pad unused entries in the returned ``neighbor_matrix``
        (matrix return path only; the COO path is unaffected). If None, the
        matrix retains the kernel's default padding of ``total_atoms``.
    strategy : {"auto", "atom_centric", "pair_centric"}, default "auto"
        Cell-list query sub-strategy, forwarded to :func:`batch_query_cell_list`.
        Both strategies produce identical pair SETS; only per-row ordering in
        ``neighbor_matrix`` differs.  ``"pair_centric"`` is CUDA-only, requires
        a concrete ``neighbor_search_radius`` (host-read ``total_cells`` /
        ``n_outer`` / ``R_max``), runs full-fill only, and raises a clear error
        under ``jax.jit`` with a traced radius when requested explicitly.
        ``"auto"`` falls back to ``"atom_centric"`` when pair-centric launch
        sizing is traced.  Explicit ``"pair_centric"`` on CPU raises; ``"auto"``
        resolves to ``"atom_centric"`` on CPU.  Not yet wired through the
        pair-output (return_distances / return_vectors) path.
    atom_centric_path : {"auto", "direct", "sorted"}, default "auto"
        Accepted for signature parity with Torch; forwarded to
        :func:`batch_query_cell_list`.  JAX always runs the sorted atom-centric
        kernel (this option never branches).

    Returns
    -------
    neighbor_data : jax.Array
        If ``return_neighbor_list=False`` (default): ``neighbor_matrix`` with shape
        (total_atoms, max_neighbors), dtype int32.
        If ``return_neighbor_list=True``: ``neighbor_list`` with shape
        (2, num_pairs), dtype int32, in COO format.
    neighbor_count : jax.Array
        If ``return_neighbor_list=False``: ``num_neighbors`` with shape
        (total_atoms,), dtype int32.
        If ``return_neighbor_list=True``: ``neighbor_ptr`` with shape
        (total_atoms + 1,), dtype int32.
    shift_data : jax.Array
        If ``return_neighbor_list=False`` (default): ``neighbor_matrix_shifts`` with shape
        (total_atoms, max_neighbors, 3), dtype int32.
        If ``return_neighbor_list=True``: ``neighbor_list_shifts`` with shape
        (num_pairs, 3), dtype int32.
        Periodic shift vectors for each neighbor relationship.

    See Also
    --------
    batch_build_cell_list : Build cell list separately
    batch_query_cell_list : Query cell list separately
    batch_naive_neighbor_list : Naive O(N^2) method
    """

    has_pair_outputs = _has_partial_or_pair_outputs(
        target_indices=target_indices,
        return_vectors=return_vectors,
        return_distances=return_distances,
        pair_fn=pair_fn,
        pair_params=pair_params,
        neighbor_vectors=neighbor_vectors,
        neighbor_distances=neighbor_distances,
        pair_energies=pair_energies,
        pair_forces=pair_forces,
    )
    _validate_pair_kwargs(
        pair_fn=pair_fn,
        pair_params=pair_params,
        pair_energies=pair_energies,
        pair_forces=pair_forces,
    )

    # Validate the sub-strategy options up front.  ``atom_centric_path`` is
    # accepted for parity but never branches (JAX always runs the sorted
    # atom-centric kernel).  ``strategy`` is forwarded to
    # ``batch_query_cell_list``, which owns the host-read sizing + launch-safety
    # guards.  Two guards must live HERE because the pair-output branch below
    # bypasses ``batch_query_cell_list`` entirely.
    _validate_atom_centric_path(atom_centric_path)
    if strategy not in {"auto", "atom_centric", "pair_centric"}:
        raise ValueError(
            f"strategy must be 'auto' | 'atom_centric' | 'pair_centric', "
            f"got {strategy!r}",
        )
    if strategy == "pair_centric" and _is_cpu_array(positions):
        # Pair-centric kernels use CUDA block scheduling.  Raise early here
        # (before batch_build_cell_list) for a clean message, mirroring Torch's
        # CPU guard; ``strategy="auto"`` resolves to atom_centric on CPU.
        raise ValueError(
            "strategy='pair_centric' is not supported on CPU "
            "(kernels use CUDA block scheduling).  Pass 'auto' or "
            "'atom_centric' instead.",
        )
    if strategy == "pair_centric" and target_indices is not None:
        # The pair-centric kernel yields an identical pair set to atom-centric,
        # so partial neighbor lists are fully covered by the atom-centric path;
        # the compact-row ``target_indices`` + pair-centric combination is not
        # wired (no capability gap -- use atom_centric).
        raise NotImplementedError(
            "strategy='pair_centric' with target_indices (partial neighbor "
            "lists) is not wired through the JAX batch_cell_list binding.  Use "
            "strategy='atom_centric' (or 'auto') for identical results.",
        )

    # Preserve LIVE positions/cell for the pair-output autograd primitive; the
    # Warp kernels are non-differentiable across the JAX boundary, so detach
    # topology-side inputs for both pair-output and topology-only paths.
    positions_for_grad = positions
    cell_input_for_grad = cell
    positions = jax.lax.stop_gradient(positions)
    if cell is not None:
        cell = jax.lax.stop_gradient(cell)

    # Prepare batch info
    batch_idx, batch_ptr = prepare_batch_idx_ptr(
        batch_idx, batch_ptr, positions.shape[0]
    )
    num_systems = batch_ptr.shape[0] - 1
    grad_cell_dtype = (
        positions_for_grad.dtype
        if positions_for_grad.dtype == jnp.float64
        else jnp.float32
    )
    cell_for_grad, _ = _normalize_batch_cell_pbc(
        cell_input_for_grad,
        pbc,
        num_systems=num_systems,
        dtype=grad_cell_dtype,
    )
    if positions_for_grad.dtype != jnp.float64:
        positions_for_grad = positions_for_grad.astype(jnp.float32)
    if cell_for_grad.dtype != positions_for_grad.dtype:
        cell_for_grad = cell_for_grad.astype(positions_for_grad.dtype)
    topology_cell_dtype = (
        positions.dtype if positions.dtype == jnp.float64 else jnp.float32
    )
    cell, pbc = _normalize_batch_cell_pbc(
        cell,
        pbc,
        num_systems=num_systems,
        dtype=topology_cell_dtype,
    )

    # Build cell list
    (
        cells_per_dimension,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
        neighbor_search_radius,
        cell_origin,
    ) = batch_build_cell_list(
        positions,
        batch_idx=batch_idx,
        batch_ptr=batch_ptr,
        cell=cell,
        pbc=pbc,
        cutoff=cutoff,
        max_total_cells=max_total_cells,
    )

    if has_pair_outputs:
        num_systems = batch_ptr.shape[0] - 1
        pbc_bool = pbc.astype(jnp.bool_)
        if max_neighbors is None and neighbor_matrix_shifts is not None:
            max_neighbors = int(neighbor_matrix_shifts.shape[1])
        if max_neighbors is None:
            max_neighbors = estimate_max_neighbors(cutoff)
        total_atoms = positions.shape[0]
        # Partial (``target_indices``) path: the compact output has
        # ``num_targets`` rows (row ``r`` -> atom ``target_indices[r]``), not
        # ``total_atoms``.  ``num_rows`` drives every per-row output buffer and
        # the kernel launch dim (the fill sentinel stays ``total_atoms``).
        if target_indices is not None:
            target_indices = jnp.asarray(target_indices, dtype=jnp.int32)
            num_rows = int(target_indices.shape[0])
        else:
            num_rows = total_atoms
        _validate_compact_target_buffers(
            target_indices=target_indices,
            num_rows=int(num_rows),
            max_neighbors=int(max_neighbors),
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            neighbor_distances=neighbor_distances,
            neighbor_vectors=neighbor_vectors,
            pair_energies=pair_energies,
            pair_forces=pair_forces,
        )
        if neighbor_matrix_shifts is None:
            neighbor_matrix_shifts = jnp.zeros(
                (num_rows, max_neighbors, 3), dtype=jnp.int32
            )
        nm = jnp.full((num_rows, max_neighbors), total_atoms, dtype=jnp.int32)
        nn = jnp.zeros(num_rows, dtype=jnp.int32)
        if return_distances and neighbor_distances is None:
            neighbor_distances = jnp.zeros(
                (num_rows, max_neighbors), dtype=positions.dtype
            )
        if return_vectors and neighbor_vectors is None:
            neighbor_vectors = jnp.zeros(
                (num_rows, max_neighbors, 3), dtype=positions.dtype
            )
        if neighbor_distances is None:
            neighbor_distances = jnp.zeros(
                (num_rows, max_neighbors), dtype=positions.dtype
            )
        if neighbor_vectors is None:
            neighbor_vectors = jnp.zeros(
                (num_rows, max_neighbors, 3), dtype=positions.dtype
            )
        cells_per_system = jnp.prod(cells_per_dimension, axis=1)
        cell_offsets = jnp.concatenate(
            [
                jnp.array([0], dtype=jnp.int32),
                jnp.cumsum(cells_per_system[:-1], dtype=jnp.int32),
            ]
        )
        batch_idx_i32 = batch_idx.astype(jnp.int32)

        # Pair-centric pair-output strategy (EXPLICIT only; "auto" resolves to
        # atom_centric here so we skip the parallelism-sufficiency host reads).
        # Host-read the cross-system sizing scalars (R_max, total_cells,
        # n_outer) to bake the block-scheduled launch -- a device->host sync,
        # legal eagerly but illegal under jax.jit with a traced radius (the
        # pair-output path is eager-on-cutoff regardless).
        pc_strategy = "atom_centric"
        pc_n_outer = 0
        pc_total_cells = 0
        pc_r_max = (0, 0, 0)
        if strategy == "pair_centric":
            try:
                R_max_arr = jnp.max(neighbor_search_radius, axis=0)
                pc_r_max = (
                    int(R_max_arr[0]),
                    int(R_max_arr[1]),
                    int(R_max_arr[2]),
                )
                pc_total_cells = int(jnp.sum(cells_per_system))
            except (
                jax.errors.ConcretizationTypeError,
                jax.errors.TracerIntegerConversionError,
            ) as exc:
                raise ValueError(
                    "strategy='pair_centric' needs a concrete "
                    "neighbor_search_radius to size its launch grid (total_cells "
                    "/ n_outer / R_max are host-read).  Compute the cell-list "
                    "sizing outside jax.jit and pass a concrete "
                    "neighbor_search_radius, or use strategy='atom_centric'.",
                ) from exc
            # JAX batch cell_list is full-fill (half_fill+pair_centric raised).
            pc_n_outer = compute_batch_pair_centric_n_outer(pc_r_max, False)
            if not is_pair_centric_launch_safe(pc_total_cells, pc_n_outer):
                _raise_unsafe_pair_centric_launch(pc_total_cells, pc_n_outer, 64)
            pc_strategy = "pair_centric"

        forward_kwargs = {
            "pbc_bool": pbc_bool,
            "batch_idx_i32": batch_idx_i32,
            "cells_per_dimension": cells_per_dimension,
            "atom_periodic_shifts": atom_periodic_shifts,
            "atom_to_cell_mapping": atom_to_cell_mapping,
            "atoms_per_cell_count": atoms_per_cell_count,
            "cell_atom_start_indices": cell_atom_start_indices,
            "cell_atom_list": cell_atom_list,
            "cell_offsets": cell_offsets,
            "neighbor_search_radius": neighbor_search_radius,
            "neighbor_matrix": nm,
            "neighbor_matrix_shifts": neighbor_matrix_shifts,
            "num_neighbors": nn,
            "neighbor_vectors": neighbor_vectors,
            "neighbor_distances": neighbor_distances,
            "cutoff": cutoff,
            "pair_fn": pair_fn,
            "pair_params": pair_params,
            "target_indices": target_indices,
            "strategy": pc_strategy,
            "n_outer": pc_n_outer,
            "total_cells": pc_total_cells,
            "r_max": pc_r_max,
            "half_fill": bool(half_fill),
        }
        route_out = _route_pair_outputs(
            positions_for_grad,
            cell_for_grad,
            _batch_cell_list_pair_outputs_forward,
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
            # COO source index ``nl[0]`` is the matrix ROW index.  For the
            # partial (``target_indices``) path that row is the COMPACT row in
            # ``[0, num_targets)`` -- NOT the atom index -- mirroring the torch
            # binding (the matrix contract is "row r -> atom target_indices[r]";
            # COO inherits the same compact-row contract).
            nl, nptr, nl_shifts = get_neighbor_list_from_neighbor_matrix(
                nm_out,
                num_neighbors=nn_out,
                neighbor_shift_matrix=shifts_out,
                fill_value=total_atoms,
            )
            base = (nl, nptr, nl_shifts)
            # Repack per-pair geometry (and pair_fn outputs) into the same COO order
            # as ``nl``.  Eager-only, like the index conversion.
            active = nm_out != total_atoms
            distances_out, vectors_out = coo_pack_pair_geometry(
                active, distances_out, vectors_out
            )
            if pair_fn is not None:
                pe_out, pf_out = coo_pack_pair_geometry(active, pe_out, pf_out)
        else:
            if fill_value is not None and int(fill_value) != total_atoms:
                # Match the matrix-padding contract: real indices are
                # < total_atoms, so remap only the unfilled tail.
                nm_out = jnp.where(nm_out == total_atoms, jnp.int32(fill_value), nm_out)
            base = (nm_out, nn_out, shifts_out)
        # Return tail mirrors the torch contract: optional distances / vectors,
        # then (pe, pf) whenever ``pair_fn`` is set.
        tail: list = []
        if return_distances:
            tail.append(distances_out)
        if return_vectors:
            tail.append(vectors_out)
        if pair_fn is not None:
            tail.extend((pe_out, pf_out))
        return (*base, *tail)

    # Query cell list
    neighbor_matrix, num_neighbors, neighbor_matrix_shifts = batch_query_cell_list(
        positions=positions,
        batch_idx=batch_idx,
        batch_ptr=batch_ptr,
        cutoff=cutoff,
        cell=cell,
        pbc=pbc,
        cells_per_dimension=cells_per_dimension,
        atom_periodic_shifts=atom_periodic_shifts,
        atom_to_cell_mapping=atom_to_cell_mapping,
        atoms_per_cell_count=atoms_per_cell_count,
        cell_atom_start_indices=cell_atom_start_indices,
        cell_atom_list=cell_atom_list,
        neighbor_search_radius=neighbor_search_radius,
        max_neighbors=max_neighbors,
        neighbor_matrix_shifts=neighbor_matrix_shifts,
        half_fill=half_fill,
        strategy=strategy,
        atom_centric_path=atom_centric_path,
    )

    if return_neighbor_list:
        neighbor_list, neighbor_ptr, neighbor_list_shifts = (
            get_neighbor_list_from_neighbor_matrix(
                neighbor_matrix,
                num_neighbors=num_neighbors,
                neighbor_shift_matrix=neighbor_matrix_shifts,
                fill_value=positions.shape[0],
            )
        )
        return neighbor_list, neighbor_ptr, neighbor_list_shifts
    else:
        if fill_value is not None and int(fill_value) != positions.shape[0]:
            # The kernel pads unfilled matrix entries with ``total_atoms``; real
            # neighbor indices are < total_atoms, so remap only the tail.
            neighbor_matrix = jnp.where(
                neighbor_matrix == positions.shape[0],
                jnp.int32(fill_value),
                neighbor_matrix,
            )
        return neighbor_matrix, num_neighbors, neighbor_matrix_shifts
