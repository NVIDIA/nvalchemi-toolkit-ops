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

"""JAX bindings for unbatched cell list O(N) neighbor list construction."""

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
from nvalchemiops.jax.neighbors.neighbor_utils import (
    _validate_graph_mode,
    coo_pack_pair_geometry,
    get_neighbor_list_from_neighbor_matrix,
)
from nvalchemiops.neighbors.cell_list import (
    build_cell_list as _warp_build_cell_list,
)
from nvalchemiops.neighbors.cell_list import (
    compute_batch_pair_centric_n_outer,
    get_build_cell_list_kernel,
    get_query_cell_list_kernel,
    is_pair_centric_launch_safe,
    is_pair_centric_parallelism_sufficient,
    select_cell_list_strategy,
)
from nvalchemiops.neighbors.cell_list import (
    query_cell_list as _warp_query_cell_list,
)
from nvalchemiops.neighbors.cell_list.launchers import (
    _raise_unsafe_pair_centric_launch,
)
from nvalchemiops.neighbors.neighbor_utils import (
    estimate_max_neighbors,
    get_gather_positions_and_shifts_kernel,
    selective_zero_num_neighbors_single,
)
from nvalchemiops.neighbors.output_args import (
    _has_partial_or_pair_outputs,
)

# ==============================================================================
# JAX Kernel Wrappers
# ==============================================================================

# Build step 1: Construct bin sizes
_jax_construct_bin_size_f32 = jax_kernel(
    get_build_cell_list_kernel("construct_bin_size", wp.float32),
    num_outputs=1,
    in_out_argnames=["cells_per_dimension_single"],
    enable_backward=False,
)
_jax_construct_bin_size_f64 = jax_kernel(
    get_build_cell_list_kernel("construct_bin_size", wp.float64),
    num_outputs=1,
    in_out_argnames=["cells_per_dimension_single"],
    enable_backward=False,
)

# Build step 2: Count atoms per bin
_jax_count_atoms_per_bin_f32 = jax_kernel(
    get_build_cell_list_kernel("count_atoms", wp.float32),
    num_outputs=2,
    in_out_argnames=["atoms_per_cell_count", "atom_periodic_shifts"],
    enable_backward=False,
)
_jax_count_atoms_per_bin_f64 = jax_kernel(
    get_build_cell_list_kernel("count_atoms", wp.float64),
    num_outputs=2,
    in_out_argnames=["atoms_per_cell_count", "atom_periodic_shifts"],
    enable_backward=False,
)

# Build step 3: Bin atoms into cells
_jax_bin_atoms_f32 = jax_kernel(
    get_build_cell_list_kernel("bin_atoms", wp.float32),
    num_outputs=3,
    in_out_argnames=["atom_to_cell_mapping", "atoms_per_cell_count", "cell_atom_list"],
    enable_backward=False,
)
_jax_bin_atoms_f64 = jax_kernel(
    get_build_cell_list_kernel("bin_atoms", wp.float64),
    num_outputs=3,
    in_out_argnames=["atom_to_cell_mapping", "atoms_per_cell_count", "cell_atom_list"],
    enable_backward=False,
)

# Gather: pack positions + atom_periodic_shifts into per-cell-contiguous layout
# (cell_atom_list permutation) for coalesced reads by the sorted-build kernel.
_jax_gather_fused_f32 = jax_kernel(
    get_gather_positions_and_shifts_kernel(wp.float32),
    num_outputs=2,
    in_out_argnames=["dst_pos", "dst_shifts"],
    enable_backward=False,
)
_jax_gather_fused_f64 = jax_kernel(
    get_gather_positions_and_shifts_kernel(wp.float64),
    num_outputs=2,
    in_out_argnames=["dst_pos", "dst_shifts"],
    enable_backward=False,
)

# Query: sorted-reads atom-centric neighbor matrix kernel.  The selective
# kernel is the same Warp kernel; selective callers pass a non-trivial
# ``rebuild_flags``, non-selective callers pass a 1-element always-True flag.
_jax_build_neighbor_matrix_local_count_sorted_f32 = jax_kernel(
    get_query_cell_list_kernel(
        wp.float32,
        strategy="atom_centric",
        batched=False,
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
_jax_build_neighbor_matrix_local_count_sorted_f64 = jax_kernel(
    get_query_cell_list_kernel(
        wp.float64,
        strategy="atom_centric",
        batched=False,
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

# Direct-reads atom-centric query kernel (``atom_centric_path="direct"``).  Reads
# ``positions`` in original order and skips the sorted gather; the kernel ignores the
# ``sorted_positions`` / ``sorted_atom_periodic_shifts`` arrays (pass 0-length sentinels).
# Full-fill only -- the symmetric-full-fill optimization does not apply to half_fill or
# pair outputs, which keep the sorted kernel.
_jax_build_neighbor_matrix_local_count_direct_f32 = jax_kernel(
    get_query_cell_list_kernel(
        wp.float32,
        strategy="atom_centric",
        batched=False,
        selective=True,
        partial=False,
        return_vectors=False,
        return_distances=False,
        pair_fn=None,
        atom_centric_path="direct",
    ),
    num_outputs=3,
    in_out_argnames=["neighbor_matrix", "neighbor_matrix_shifts", "num_neighbors"],
    enable_backward=False,
)
_jax_build_neighbor_matrix_local_count_direct_f64 = jax_kernel(
    get_query_cell_list_kernel(
        wp.float64,
        strategy="atom_centric",
        batched=False,
        selective=True,
        partial=False,
        return_vectors=False,
        return_distances=False,
        pair_fn=None,
        atom_centric_path="direct",
    ),
    num_outputs=3,
    in_out_argnames=["neighbor_matrix", "neighbor_matrix_shifts", "num_neighbors"],
    enable_backward=False,
)

# Half-fill variants of the sorted-build kernel.  ``half_fill`` is a compile-time
# specialization in the Warp factory (the runtime ``half_fill`` arg is an ignored
# ABI placeholder), so honoring ``half_fill=True`` requires a distinct kernel.
_jax_build_neighbor_matrix_local_count_sorted_half_f32 = jax_kernel(
    get_query_cell_list_kernel(
        wp.float32,
        strategy="atom_centric",
        batched=False,
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
_jax_build_neighbor_matrix_local_count_sorted_half_f64 = jax_kernel(
    get_query_cell_list_kernel(
        wp.float64,
        strategy="atom_centric",
        batched=False,
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


# Pair-output variants of the sorted-build kernel.  Used by the autograd path
# when ``return_distances`` / ``return_vectors`` is set; the bytes the kernel
# writes into ``neighbor_vectors`` / ``neighbor_distances`` are consumed by
# the JAX autograd primitive in :mod:`nvalchemiops.jax.neighbors._autograd`.
_jax_build_neighbor_matrix_local_count_sorted_pair_f32 = jax_kernel(
    get_query_cell_list_kernel(
        wp.float32,
        strategy="atom_centric",
        batched=False,
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
_jax_build_neighbor_matrix_local_count_sorted_pair_f64 = jax_kernel(
    get_query_cell_list_kernel(
        wp.float64,
        strategy="atom_centric",
        batched=False,
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
# (``half_fill`` is a compile-time constant; selected when ``half_fill=True``).
_jax_build_neighbor_matrix_local_count_sorted_pair_half_f32 = jax_kernel(
    get_query_cell_list_kernel(
        wp.float32,
        strategy="atom_centric",
        batched=False,
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
_jax_build_neighbor_matrix_local_count_sorted_pair_half_f64 = jax_kernel(
    get_query_cell_list_kernel(
        wp.float64,
        strategy="atom_centric",
        batched=False,
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
def _get_jax_cell_list_pair_outputs_kernel(
    pair_fn, wp_dtype, partial, half_fill: bool = False
):
    """Build (and cache) a ``jax_kernel`` for a cell-list atom-centric ``sorted``
    pair-output kernel.

    Mirrors the module-level geometry registration above, optionally with
    ``pair_fn`` set (so the kernel's ``HAS_PAIR_FN`` body runs and
    ``pair_energies`` / ``pair_forces`` are registered as additional outputs)
    and/or ``partial=True`` (the ``target_indices`` path: output row ``r`` maps
    to atom ``target_indices[r]``, launched ``(num_targets,)``).  Cached by
    ``(pair_fn identity, wp_dtype, partial)``; one recompile per distinct
    ``(pair_fn, partial)`` combination.

    With ``pair_fn`` the kernel has 7 outputs (geometry + ``pe`` / ``pf``);
    without it, 5 (geometry only).
    """
    kernel = get_query_cell_list_kernel(
        wp_dtype,
        strategy="atom_centric",
        batched=False,
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
    "cell_list",
    "build_cell_list",
    "query_cell_list",
    "estimate_cell_list_sizes",
]


def _reset_query_outputs(
    neighbor_matrix,
    neighbor_matrix_shifts,
    num_neighbors,
    fill_value,
) -> None:
    """Reset query outputs inside the Warp callback."""
    neighbor_matrix.fill_(fill_value)
    num_neighbors.zero_()
    neighbor_matrix_shifts.zero_()


def _is_cpu_array(array: jax.Array) -> bool:
    """Return whether ``array`` is backed by a CPU device.

    Under ``jax.jit`` ``array`` is an abstract tracer whose ``.devices()`` is
    not concrete; treat that case as non-CPU (the jit-time device is whatever
    the call is compiled for, and the concrete pair-centric ``n_outer`` read in
    ``query_cell_list`` is the real jit boundary that raises the clear error).
    """
    try:
        return all(device.platform == "cpu" for device in array.devices())
    except (AttributeError, jax.errors.ConcretizationTypeError):
        return False


def _validate_atom_centric_path(atom_centric_path: str) -> str:
    """Validate ``atom_centric_path``.

    Explicit ``"direct"`` selects the direct-reads (symmetric full-fill) kernel on
    the plain full-fill, ``graph_mode="none"`` path (skipping the sorted gather);
    ``"sorted"`` selects the sorted kernel.  ``"auto"`` resolves to ``"sorted"`` on
    JAX (a perf-only divergence from Torch, whose ``"auto"`` resolves to ``"direct"``);
    half_fill / ``graph_mode="warp"`` keep the sorted kernel regardless.  Returns the
    validated string unchanged.
    """
    if atom_centric_path not in {"auto", "direct", "sorted"}:
        raise ValueError(
            "atom_centric_path must be 'auto' | 'direct' | 'sorted', "
            f"got {atom_centric_path!r}",
        )
    return atom_centric_path


def _resolve_cell_strategy(
    strategy: str,
    *,
    total_atoms: int,
    cutoff: float,
    device_is_cpu: bool,
    half_fill: bool = False,
) -> str:
    """Resolve the cell-list query sub-strategy to ``atom_centric``/``pair_centric``.

    Mirrors the Torch resolution (``torch/neighbors/cell_list.py:565-585``):

    - ``"auto"`` -> :func:`select_cell_list_strategy` on GPU, or ``"atom_centric"``
      on CPU (pair-centric kernels use CUDA block scheduling).  When
      ``half_fill`` is set, ``"auto"`` also resolves to ``"atom_centric"``
      because the JAX pair-centric path is full-fill only - so the default
      (no explicit strategy) keeps working for every geometry with half_fill.
    - ``"atom_centric"`` -> ``"atom_centric"``.
    - ``"pair_centric"`` -> ``"pair_centric"`` on GPU; raises on CPU.

    This is the strategy *decision* only.  The pair-centric launch-safety /
    parallelism guards and the ``n_outer`` host read live in ``query_cell_list``
    where the concrete ``neighbor_search_radius`` is available.
    """
    if strategy == "auto":
        if device_is_cpu or half_fill:
            return "atom_centric"
        return select_cell_list_strategy(int(total_atoms), float(cutoff))
    if strategy == "atom_centric":
        return "atom_centric"
    if strategy == "pair_centric":
        if device_is_cpu:
            raise ValueError(
                "strategy='pair_centric' is not supported on CPU "
                "(kernels use CUDA block scheduling).  Pass 'auto' or "
                "'atom_centric' instead.",
            )
        return "pair_centric"
    raise ValueError(
        f"strategy must be 'auto' | 'atom_centric' | 'pair_centric', got {strategy!r}",
    )


def _run_graph_build_cell_list(
    positions,
    cell,
    pbc,
    cells_per_dimension,
    atom_periodic_shifts,
    atom_to_cell_mapping,
    atoms_per_cell_count,
    cell_atom_start_indices,
    cell_atom_list,
    cutoff,
    wp_dtype,
) -> None:
    """Execute the fused cell-list build callback."""
    atoms_per_cell_count.zero_()
    # Graph-capture contract: ``_warp_build_cell_list`` is the Python-level
    # Warp launcher and takes a device string. Inside this jax_callable body
    # the device is a Python constant captured once at CUDA-graph capture
    # time, so subsequent replays reuse the same device. If the launcher
    # signature ever grows a stream/context parameter, that argument must
    # also be hoisted out of the per-replay path here.
    _warp_build_cell_list(
        positions,
        cell,
        pbc,
        cutoff,
        cells_per_dimension,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
        wp_dtype,
        str(positions.device),
    )


def _run_graph_query_cell_list(
    positions,
    cell,
    pbc,
    cells_per_dimension,
    neighbor_search_radius,
    atom_periodic_shifts,
    atom_to_cell_mapping,
    atoms_per_cell_count,
    cell_atom_start_indices,
    cell_atom_list,
    sorted_positions,
    sorted_atom_periodic_shifts,
    rebuild_flags,
    neighbor_matrix,
    neighbor_matrix_shifts,
    num_neighbors,
    cutoff,
    fill_value,
    wp_dtype,
    selective: bool,
    strategy: str = "atom_centric",
    n_outer: int | None = None,
    return_vectors: bool = False,
    return_distances: bool = False,
    pair_fn=None,
    pair_params=None,
    neighbor_vectors=None,
    neighbor_distances=None,
    pair_energies=None,
    pair_forces=None,
) -> None:
    """Execute the fused cell-list query callback.

    ``rebuild_flags`` is always a 1-element ``wp.bool`` array.  Non-selective
    callers pass an always-True flag and we reset all outputs.  Selective
    callers pass the live flag and we only zero ``num_neighbors`` on systems
    whose flag is True.

    ``strategy`` / ``n_outer`` thread the cell-list query sub-strategy through
    to ``_warp_query_cell_list``.  For ``strategy="pair_centric"`` the launcher
    runs the internal ``gather_fused`` into ``sorted_positions`` /
    ``sorted_atom_periodic_shifts`` and the pair-centric linear launch sized by
    the host-computed ``n_outer``.  ``n_outer`` is a static scalar baked at
    launch-build time (see :func:`compute_batch_pair_centric_n_outer`).

    The optional ``return_vectors`` / ``return_distances`` / ``pair_fn`` (+
    ``pair_params`` and the ``neighbor_vectors`` / ``neighbor_distances`` /
    ``pair_energies`` / ``pair_forces`` output buffers) thread the pair-output
    contract straight to ``_warp_query_cell_list``; the pair-centric kernel
    writes per-slot geometry / energies / forces just like the atom-centric one.
    """
    if not selective:
        _reset_query_outputs(
            neighbor_matrix,
            neighbor_matrix_shifts,
            num_neighbors,
            fill_value,
        )
    else:
        selective_zero_num_neighbors_single(
            num_neighbors, rebuild_flags, str(num_neighbors.device)
        )

    # Graph-capture contract: ``_warp_query_cell_list`` is the Python-level
    # Warp launcher and takes a device string. Inside this jax_callable body
    # the device is a Python constant captured once at CUDA-graph capture
    # time, so subsequent replays reuse the same device. If the launcher
    # signature ever grows a stream/context parameter, that argument must
    # also be hoisted out of the per-replay path here.
    _warp_query_cell_list(
        positions=positions,
        cell=cell,
        pbc=pbc,
        cutoff=cutoff,
        cells_per_dimension=cells_per_dimension,
        neighbor_search_radius=neighbor_search_radius,
        atom_periodic_shifts=atom_periodic_shifts,
        atom_to_cell_mapping=atom_to_cell_mapping,
        atoms_per_cell_count=atoms_per_cell_count,
        cell_atom_start_indices=cell_atom_start_indices,
        cell_atom_list=cell_atom_list,
        sorted_positions=sorted_positions,
        sorted_atom_periodic_shifts=sorted_atom_periodic_shifts,
        neighbor_matrix=neighbor_matrix,
        neighbor_matrix_shifts=neighbor_matrix_shifts,
        num_neighbors=num_neighbors,
        rebuild_flags=rebuild_flags,
        wp_dtype=wp_dtype,
        device=str(positions.device),
        half_fill=False,
        strategy=strategy,
        n_outer=n_outer,
        return_vectors=return_vectors,
        return_distances=return_distances,
        pair_fn=pair_fn,
        pair_params=pair_params,
        neighbor_vectors=neighbor_vectors,
        neighbor_distances=neighbor_distances,
        pair_energies=pair_energies,
        pair_forces=pair_forces,
    )


def _run_graph_cell_list(
    positions,
    cell,
    pbc,
    neighbor_search_radius,
    cells_per_dimension,
    atom_periodic_shifts,
    atom_to_cell_mapping,
    atoms_per_cell_count,
    cell_atom_start_indices,
    cell_atom_list,
    sorted_positions,
    sorted_atom_periodic_shifts,
    rebuild_flags,
    neighbor_matrix,
    neighbor_matrix_shifts,
    num_neighbors,
    cutoff,
    fill_value,
    wp_dtype,
) -> None:
    """Execute the fused cell-list build+query callback."""
    _run_graph_build_cell_list(
        positions,
        cell,
        pbc,
        cells_per_dimension,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
        cutoff,
        wp_dtype,
    )
    _run_graph_query_cell_list(
        positions,
        cell,
        pbc,
        cells_per_dimension,
        neighbor_search_radius,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
        sorted_positions,
        sorted_atom_periodic_shifts,
        rebuild_flags,
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
        cutoff,
        fill_value,
        wp_dtype,
        selective=False,
    )


def _graph_build_cell_list_f32(
    positions: wp.array(dtype=wp.vec3f),
    cell: wp.array(dtype=wp.mat33f),
    pbc: wp.array(dtype=wp.bool),
    cells_per_dimension: wp.array(dtype=wp.int32),
    atom_periodic_shifts: wp.array(dtype=wp.vec3i),
    atom_to_cell_mapping: wp.array(dtype=wp.vec3i),
    atoms_per_cell_count: wp.array(dtype=wp.int32),
    cell_atom_start_indices: wp.array(dtype=wp.int32),
    cell_atom_list: wp.array(dtype=wp.int32),
    cutoff: wp.float32,
) -> None:
    _run_graph_build_cell_list(
        positions,
        cell,
        pbc,
        cells_per_dimension,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
        cutoff,
        wp.float32,
    )


def _graph_build_cell_list_f64(
    positions: wp.array(dtype=wp.vec3d),
    cell: wp.array(dtype=wp.mat33d),
    pbc: wp.array(dtype=wp.bool),
    cells_per_dimension: wp.array(dtype=wp.int32),
    atom_periodic_shifts: wp.array(dtype=wp.vec3i),
    atom_to_cell_mapping: wp.array(dtype=wp.vec3i),
    atoms_per_cell_count: wp.array(dtype=wp.int32),
    cell_atom_start_indices: wp.array(dtype=wp.int32),
    cell_atom_list: wp.array(dtype=wp.int32),
    cutoff: wp.float64,
) -> None:
    _run_graph_build_cell_list(
        positions,
        cell,
        pbc,
        cells_per_dimension,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
        cutoff,
        wp.float64,
    )


def _graph_query_cell_list_f32(
    positions: wp.array(dtype=wp.vec3f),
    cell: wp.array(dtype=wp.mat33f),
    pbc: wp.array(dtype=wp.bool),
    cells_per_dimension: wp.array(dtype=wp.int32),
    neighbor_search_radius: wp.array(dtype=wp.int32),
    atom_periodic_shifts: wp.array(dtype=wp.vec3i),
    atom_to_cell_mapping: wp.array(dtype=wp.vec3i),
    atoms_per_cell_count: wp.array(dtype=wp.int32),
    cell_atom_start_indices: wp.array(dtype=wp.int32),
    cell_atom_list: wp.array(dtype=wp.int32),
    sorted_positions: wp.array(dtype=wp.vec3f),
    sorted_atom_periodic_shifts: wp.array(dtype=wp.vec3i),
    rebuild_flags: wp.array(dtype=wp.bool),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    cutoff: wp.float32,
    fill_value: wp.int32,
) -> None:
    _run_graph_query_cell_list(
        positions,
        cell,
        pbc,
        cells_per_dimension,
        neighbor_search_radius,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
        sorted_positions,
        sorted_atom_periodic_shifts,
        rebuild_flags,
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
        cutoff,
        fill_value,
        wp.float32,
        selective=False,
    )


def _graph_query_cell_list_f64(
    positions: wp.array(dtype=wp.vec3d),
    cell: wp.array(dtype=wp.mat33d),
    pbc: wp.array(dtype=wp.bool),
    cells_per_dimension: wp.array(dtype=wp.int32),
    neighbor_search_radius: wp.array(dtype=wp.int32),
    atom_periodic_shifts: wp.array(dtype=wp.vec3i),
    atom_to_cell_mapping: wp.array(dtype=wp.vec3i),
    atoms_per_cell_count: wp.array(dtype=wp.int32),
    cell_atom_start_indices: wp.array(dtype=wp.int32),
    cell_atom_list: wp.array(dtype=wp.int32),
    sorted_positions: wp.array(dtype=wp.vec3d),
    sorted_atom_periodic_shifts: wp.array(dtype=wp.vec3i),
    rebuild_flags: wp.array(dtype=wp.bool),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    cutoff: wp.float64,
    fill_value: wp.int32,
) -> None:
    _run_graph_query_cell_list(
        positions,
        cell,
        pbc,
        cells_per_dimension,
        neighbor_search_radius,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
        sorted_positions,
        sorted_atom_periodic_shifts,
        rebuild_flags,
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
        cutoff,
        fill_value,
        wp.float64,
        selective=False,
    )


def _graph_query_cell_list_selective_f32(
    positions: wp.array(dtype=wp.vec3f),
    cell: wp.array(dtype=wp.mat33f),
    pbc: wp.array(dtype=wp.bool),
    cells_per_dimension: wp.array(dtype=wp.int32),
    neighbor_search_radius: wp.array(dtype=wp.int32),
    atom_periodic_shifts: wp.array(dtype=wp.vec3i),
    atom_to_cell_mapping: wp.array(dtype=wp.vec3i),
    atoms_per_cell_count: wp.array(dtype=wp.int32),
    cell_atom_start_indices: wp.array(dtype=wp.int32),
    cell_atom_list: wp.array(dtype=wp.int32),
    sorted_positions: wp.array(dtype=wp.vec3f),
    sorted_atom_periodic_shifts: wp.array(dtype=wp.vec3i),
    rebuild_flags: wp.array(dtype=wp.bool),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    cutoff: wp.float32,
    fill_value: wp.int32,
) -> None:
    _run_graph_query_cell_list(
        positions,
        cell,
        pbc,
        cells_per_dimension,
        neighbor_search_radius,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
        sorted_positions,
        sorted_atom_periodic_shifts,
        rebuild_flags,
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
        cutoff,
        fill_value,
        wp.float32,
        selective=True,
    )


def _graph_query_cell_list_selective_f64(
    positions: wp.array(dtype=wp.vec3d),
    cell: wp.array(dtype=wp.mat33d),
    pbc: wp.array(dtype=wp.bool),
    cells_per_dimension: wp.array(dtype=wp.int32),
    neighbor_search_radius: wp.array(dtype=wp.int32),
    atom_periodic_shifts: wp.array(dtype=wp.vec3i),
    atom_to_cell_mapping: wp.array(dtype=wp.vec3i),
    atoms_per_cell_count: wp.array(dtype=wp.int32),
    cell_atom_start_indices: wp.array(dtype=wp.int32),
    cell_atom_list: wp.array(dtype=wp.int32),
    sorted_positions: wp.array(dtype=wp.vec3d),
    sorted_atom_periodic_shifts: wp.array(dtype=wp.vec3i),
    rebuild_flags: wp.array(dtype=wp.bool),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    cutoff: wp.float64,
    fill_value: wp.int32,
) -> None:
    _run_graph_query_cell_list(
        positions,
        cell,
        pbc,
        cells_per_dimension,
        neighbor_search_radius,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
        sorted_positions,
        sorted_atom_periodic_shifts,
        rebuild_flags,
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
        cutoff,
        fill_value,
        wp.float64,
        selective=True,
    )


def _graph_query_cell_list_pair_centric_f32(
    positions: wp.array(dtype=wp.vec3f),
    cell: wp.array(dtype=wp.mat33f),
    pbc: wp.array(dtype=wp.bool),
    cells_per_dimension: wp.array(dtype=wp.int32),
    neighbor_search_radius: wp.array(dtype=wp.int32),
    atom_periodic_shifts: wp.array(dtype=wp.vec3i),
    atom_to_cell_mapping: wp.array(dtype=wp.vec3i),
    atoms_per_cell_count: wp.array(dtype=wp.int32),
    cell_atom_start_indices: wp.array(dtype=wp.int32),
    cell_atom_list: wp.array(dtype=wp.int32),
    sorted_positions: wp.array(dtype=wp.vec3f),
    sorted_atom_periodic_shifts: wp.array(dtype=wp.vec3i),
    rebuild_flags: wp.array(dtype=wp.bool),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    cutoff: wp.float32,
    fill_value: wp.int32,
    n_outer: wp.int32,
) -> None:
    _run_graph_query_cell_list(
        positions,
        cell,
        pbc,
        cells_per_dimension,
        neighbor_search_radius,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
        sorted_positions,
        sorted_atom_periodic_shifts,
        rebuild_flags,
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
        cutoff,
        fill_value,
        wp.float32,
        selective=False,
        strategy="pair_centric",
        n_outer=int(n_outer),
    )


def _graph_query_cell_list_pair_centric_f64(
    positions: wp.array(dtype=wp.vec3d),
    cell: wp.array(dtype=wp.mat33d),
    pbc: wp.array(dtype=wp.bool),
    cells_per_dimension: wp.array(dtype=wp.int32),
    neighbor_search_radius: wp.array(dtype=wp.int32),
    atom_periodic_shifts: wp.array(dtype=wp.vec3i),
    atom_to_cell_mapping: wp.array(dtype=wp.vec3i),
    atoms_per_cell_count: wp.array(dtype=wp.int32),
    cell_atom_start_indices: wp.array(dtype=wp.int32),
    cell_atom_list: wp.array(dtype=wp.int32),
    sorted_positions: wp.array(dtype=wp.vec3d),
    sorted_atom_periodic_shifts: wp.array(dtype=wp.vec3i),
    rebuild_flags: wp.array(dtype=wp.bool),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    cutoff: wp.float64,
    fill_value: wp.int32,
    n_outer: wp.int32,
) -> None:
    _run_graph_query_cell_list(
        positions,
        cell,
        pbc,
        cells_per_dimension,
        neighbor_search_radius,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
        sorted_positions,
        sorted_atom_periodic_shifts,
        rebuild_flags,
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
        cutoff,
        fill_value,
        wp.float64,
        selective=False,
        strategy="pair_centric",
        n_outer=int(n_outer),
    )


def _graph_query_cell_list_pair_centric_selective_f32(
    positions: wp.array(dtype=wp.vec3f),
    cell: wp.array(dtype=wp.mat33f),
    pbc: wp.array(dtype=wp.bool),
    cells_per_dimension: wp.array(dtype=wp.int32),
    neighbor_search_radius: wp.array(dtype=wp.int32),
    atom_periodic_shifts: wp.array(dtype=wp.vec3i),
    atom_to_cell_mapping: wp.array(dtype=wp.vec3i),
    atoms_per_cell_count: wp.array(dtype=wp.int32),
    cell_atom_start_indices: wp.array(dtype=wp.int32),
    cell_atom_list: wp.array(dtype=wp.int32),
    sorted_positions: wp.array(dtype=wp.vec3f),
    sorted_atom_periodic_shifts: wp.array(dtype=wp.vec3i),
    rebuild_flags: wp.array(dtype=wp.bool),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    cutoff: wp.float32,
    fill_value: wp.int32,
    n_outer: wp.int32,
) -> None:
    _run_graph_query_cell_list(
        positions,
        cell,
        pbc,
        cells_per_dimension,
        neighbor_search_radius,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
        sorted_positions,
        sorted_atom_periodic_shifts,
        rebuild_flags,
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
        cutoff,
        fill_value,
        wp.float32,
        selective=True,
        strategy="pair_centric",
        n_outer=int(n_outer),
    )


def _graph_query_cell_list_pair_centric_selective_f64(
    positions: wp.array(dtype=wp.vec3d),
    cell: wp.array(dtype=wp.mat33d),
    pbc: wp.array(dtype=wp.bool),
    cells_per_dimension: wp.array(dtype=wp.int32),
    neighbor_search_radius: wp.array(dtype=wp.int32),
    atom_periodic_shifts: wp.array(dtype=wp.vec3i),
    atom_to_cell_mapping: wp.array(dtype=wp.vec3i),
    atoms_per_cell_count: wp.array(dtype=wp.int32),
    cell_atom_start_indices: wp.array(dtype=wp.int32),
    cell_atom_list: wp.array(dtype=wp.int32),
    sorted_positions: wp.array(dtype=wp.vec3d),
    sorted_atom_periodic_shifts: wp.array(dtype=wp.vec3i),
    rebuild_flags: wp.array(dtype=wp.bool),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    cutoff: wp.float64,
    fill_value: wp.int32,
    n_outer: wp.int32,
) -> None:
    _run_graph_query_cell_list(
        positions,
        cell,
        pbc,
        cells_per_dimension,
        neighbor_search_radius,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
        sorted_positions,
        sorted_atom_periodic_shifts,
        rebuild_flags,
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
        cutoff,
        fill_value,
        wp.float64,
        selective=True,
        strategy="pair_centric",
        n_outer=int(n_outer),
    )


def _graph_cell_list_f32(
    positions: wp.array(dtype=wp.vec3f),
    cell: wp.array(dtype=wp.mat33f),
    pbc: wp.array(dtype=wp.bool),
    neighbor_search_radius: wp.array(dtype=wp.int32),
    cells_per_dimension: wp.array(dtype=wp.int32),
    atom_periodic_shifts: wp.array(dtype=wp.vec3i),
    atom_to_cell_mapping: wp.array(dtype=wp.vec3i),
    atoms_per_cell_count: wp.array(dtype=wp.int32),
    cell_atom_start_indices: wp.array(dtype=wp.int32),
    cell_atom_list: wp.array(dtype=wp.int32),
    sorted_positions: wp.array(dtype=wp.vec3f),
    sorted_atom_periodic_shifts: wp.array(dtype=wp.vec3i),
    rebuild_flags: wp.array(dtype=wp.bool),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    cutoff: wp.float32,
    fill_value: wp.int32,
) -> None:
    _run_graph_cell_list(
        positions,
        cell,
        pbc,
        neighbor_search_radius,
        cells_per_dimension,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
        sorted_positions,
        sorted_atom_periodic_shifts,
        rebuild_flags,
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
        cutoff,
        fill_value,
        wp.float32,
    )


def _graph_cell_list_f64(
    positions: wp.array(dtype=wp.vec3d),
    cell: wp.array(dtype=wp.mat33d),
    pbc: wp.array(dtype=wp.bool),
    neighbor_search_radius: wp.array(dtype=wp.int32),
    cells_per_dimension: wp.array(dtype=wp.int32),
    atom_periodic_shifts: wp.array(dtype=wp.vec3i),
    atom_to_cell_mapping: wp.array(dtype=wp.vec3i),
    atoms_per_cell_count: wp.array(dtype=wp.int32),
    cell_atom_start_indices: wp.array(dtype=wp.int32),
    cell_atom_list: wp.array(dtype=wp.int32),
    sorted_positions: wp.array(dtype=wp.vec3d),
    sorted_atom_periodic_shifts: wp.array(dtype=wp.vec3i),
    rebuild_flags: wp.array(dtype=wp.bool),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    cutoff: wp.float64,
    fill_value: wp.int32,
) -> None:
    _run_graph_cell_list(
        positions,
        cell,
        pbc,
        neighbor_search_radius,
        cells_per_dimension,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
        sorted_positions,
        sorted_atom_periodic_shifts,
        rebuild_flags,
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
        cutoff,
        fill_value,
        wp.float64,
    )


_jax_graph_build_cell_list_f32 = jax_callable(
    _graph_build_cell_list_f32,
    num_outputs=6,
    in_out_argnames=[
        "cells_per_dimension",
        "atom_periodic_shifts",
        "atom_to_cell_mapping",
        "atoms_per_cell_count",
        "cell_atom_start_indices",
        "cell_atom_list",
    ],
    graph_mode=GraphMode.WARP,
)
_jax_graph_build_cell_list_f64 = jax_callable(
    _graph_build_cell_list_f64,
    num_outputs=6,
    in_out_argnames=[
        "cells_per_dimension",
        "atom_periodic_shifts",
        "atom_to_cell_mapping",
        "atoms_per_cell_count",
        "cell_atom_start_indices",
        "cell_atom_list",
    ],
    graph_mode=GraphMode.WARP,
)
_GRAPH_QUERY_INOUT = [
    "sorted_positions",
    "sorted_atom_periodic_shifts",
    "neighbor_matrix",
    "neighbor_matrix_shifts",
    "num_neighbors",
]
_jax_graph_query_cell_list_f32 = jax_callable(
    _graph_query_cell_list_f32,
    num_outputs=len(_GRAPH_QUERY_INOUT),
    in_out_argnames=_GRAPH_QUERY_INOUT,
    graph_mode=GraphMode.WARP,
)
_jax_graph_query_cell_list_f64 = jax_callable(
    _graph_query_cell_list_f64,
    num_outputs=len(_GRAPH_QUERY_INOUT),
    in_out_argnames=_GRAPH_QUERY_INOUT,
    graph_mode=GraphMode.WARP,
)
_jax_graph_query_cell_list_selective_f32 = jax_callable(
    _graph_query_cell_list_selective_f32,
    num_outputs=len(_GRAPH_QUERY_INOUT),
    in_out_argnames=_GRAPH_QUERY_INOUT,
    graph_mode=GraphMode.WARP,
)
_jax_graph_query_cell_list_selective_f64 = jax_callable(
    _graph_query_cell_list_selective_f64,
    num_outputs=len(_GRAPH_QUERY_INOUT),
    in_out_argnames=_GRAPH_QUERY_INOUT,
    graph_mode=GraphMode.WARP,
)
# Pair-centric query callables.  The launch dim is baked from the static
# ``n_outer`` scalar (host-read from ``neighbor_search_radius``), so CUDA-graph
# replay across a changed radius would be unsafe -> register with
# ``GraphMode.NONE`` (launch each call, no capture/replay).  These are the only
# ``jax_callable`` instances on the non-graph path; they still participate in
# JAX tracing and in/out buffer aliasing like the ``GraphMode.WARP`` callables.
_jax_query_cell_list_pair_centric_f32 = jax_callable(
    _graph_query_cell_list_pair_centric_f32,
    num_outputs=len(_GRAPH_QUERY_INOUT),
    in_out_argnames=_GRAPH_QUERY_INOUT,
    graph_mode=GraphMode.NONE,
)
_jax_query_cell_list_pair_centric_f64 = jax_callable(
    _graph_query_cell_list_pair_centric_f64,
    num_outputs=len(_GRAPH_QUERY_INOUT),
    in_out_argnames=_GRAPH_QUERY_INOUT,
    graph_mode=GraphMode.NONE,
)
_jax_query_cell_list_pair_centric_selective_f32 = jax_callable(
    _graph_query_cell_list_pair_centric_selective_f32,
    num_outputs=len(_GRAPH_QUERY_INOUT),
    in_out_argnames=_GRAPH_QUERY_INOUT,
    graph_mode=GraphMode.NONE,
)
_jax_query_cell_list_pair_centric_selective_f64 = jax_callable(
    _graph_query_cell_list_pair_centric_selective_f64,
    num_outputs=len(_GRAPH_QUERY_INOUT),
    in_out_argnames=_GRAPH_QUERY_INOUT,
    graph_mode=GraphMode.NONE,
)
# --- Pair-centric PAIR-OUTPUT callables -------------------------------------
# Pair-centric analogue of the matrix callables above, with per-pair geometry
# (and optionally ``pair_fn`` energies / forces) written by the same kernel.
# ``n_outer`` is a host-read static scalar (last positional arg); the launcher
# gathers internally, so there is no separate gather kernel.  ``GraphMode.NONE``
# (eager-on-cutoff, like every pair-output path).  ``selective=True`` + an
# always-true ``rebuild_flags`` mirrors the atom-centric pair-output path.
_GRAPH_QUERY_PAIR_INOUT = [
    "sorted_positions",
    "sorted_atom_periodic_shifts",
    "neighbor_matrix",
    "neighbor_matrix_shifts",
    "num_neighbors",
    "neighbor_vectors",
    "neighbor_distances",
]


def _graph_query_cell_list_pair_centric_geom_f32(
    positions: wp.array(dtype=wp.vec3f),
    cell: wp.array(dtype=wp.mat33f),
    pbc: wp.array(dtype=wp.bool),
    cells_per_dimension: wp.array(dtype=wp.int32),
    neighbor_search_radius: wp.array(dtype=wp.int32),
    atom_periodic_shifts: wp.array(dtype=wp.vec3i),
    atom_to_cell_mapping: wp.array(dtype=wp.vec3i),
    atoms_per_cell_count: wp.array(dtype=wp.int32),
    cell_atom_start_indices: wp.array(dtype=wp.int32),
    cell_atom_list: wp.array(dtype=wp.int32),
    sorted_positions: wp.array(dtype=wp.vec3f),
    sorted_atom_periodic_shifts: wp.array(dtype=wp.vec3i),
    rebuild_flags: wp.array(dtype=wp.bool),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    neighbor_vectors: wp.array(dtype=wp.vec3f, ndim=2),
    neighbor_distances: wp.array(dtype=wp.float32, ndim=2),
    cutoff: wp.float32,
    fill_value: wp.int32,
    n_outer: wp.int32,
) -> None:
    _run_graph_query_cell_list(
        positions,
        cell,
        pbc,
        cells_per_dimension,
        neighbor_search_radius,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
        sorted_positions,
        sorted_atom_periodic_shifts,
        rebuild_flags,
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
        cutoff,
        fill_value,
        wp.float32,
        selective=True,
        strategy="pair_centric",
        n_outer=int(n_outer),
        return_vectors=True,
        return_distances=True,
        neighbor_vectors=neighbor_vectors,
        neighbor_distances=neighbor_distances,
    )


def _graph_query_cell_list_pair_centric_geom_f64(
    positions: wp.array(dtype=wp.vec3d),
    cell: wp.array(dtype=wp.mat33d),
    pbc: wp.array(dtype=wp.bool),
    cells_per_dimension: wp.array(dtype=wp.int32),
    neighbor_search_radius: wp.array(dtype=wp.int32),
    atom_periodic_shifts: wp.array(dtype=wp.vec3i),
    atom_to_cell_mapping: wp.array(dtype=wp.vec3i),
    atoms_per_cell_count: wp.array(dtype=wp.int32),
    cell_atom_start_indices: wp.array(dtype=wp.int32),
    cell_atom_list: wp.array(dtype=wp.int32),
    sorted_positions: wp.array(dtype=wp.vec3d),
    sorted_atom_periodic_shifts: wp.array(dtype=wp.vec3i),
    rebuild_flags: wp.array(dtype=wp.bool),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    neighbor_vectors: wp.array(dtype=wp.vec3d, ndim=2),
    neighbor_distances: wp.array(dtype=wp.float64, ndim=2),
    cutoff: wp.float64,
    fill_value: wp.int32,
    n_outer: wp.int32,
) -> None:
    _run_graph_query_cell_list(
        positions,
        cell,
        pbc,
        cells_per_dimension,
        neighbor_search_radius,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
        sorted_positions,
        sorted_atom_periodic_shifts,
        rebuild_flags,
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
        cutoff,
        fill_value,
        wp.float64,
        selective=True,
        strategy="pair_centric",
        n_outer=int(n_outer),
        return_vectors=True,
        return_distances=True,
        neighbor_vectors=neighbor_vectors,
        neighbor_distances=neighbor_distances,
    )


_jax_query_cell_list_pair_centric_geom_f32 = jax_callable(
    _graph_query_cell_list_pair_centric_geom_f32,
    num_outputs=len(_GRAPH_QUERY_PAIR_INOUT),
    in_out_argnames=_GRAPH_QUERY_PAIR_INOUT,
    graph_mode=GraphMode.NONE,
)
_jax_query_cell_list_pair_centric_geom_f64 = jax_callable(
    _graph_query_cell_list_pair_centric_geom_f64,
    num_outputs=len(_GRAPH_QUERY_PAIR_INOUT),
    in_out_argnames=_GRAPH_QUERY_PAIR_INOUT,
    graph_mode=GraphMode.NONE,
)

_GRAPH_QUERY_PAIR_FN_INOUT = _GRAPH_QUERY_PAIR_INOUT + [
    "pair_energies",
    "pair_forces",
]


@functools.cache
def _get_jax_cell_list_pair_centric_pair_fn_callable(pair_fn, wp_dtype):
    """Build (and cache) a ``jax_callable`` closing over ``pair_fn`` for the
    pair-centric pair-output path.

    Mirrors ``_get_jax_cluster_tile_pair_fn_callable``: ``pair_fn`` cannot cross
    the JAX trace boundary as data, so the callback closes over it and adds the
    ``pair_params`` input + ``pair_energies`` / ``pair_forces`` outputs.  Two
    literal-typed callbacks (f32 / f64) keep the Warp annotations resolvable;
    cached by ``(pair_fn identity, wp_dtype)``.  ``GraphMode.NONE`` + host-read
    static ``n_outer``, like the geometry pair-centric callables.
    """
    if wp_dtype == wp.float64:

        def _callback(
            positions: wp.array(dtype=wp.vec3d),
            cell: wp.array(dtype=wp.mat33d),
            pbc: wp.array(dtype=wp.bool),
            cells_per_dimension: wp.array(dtype=wp.int32),
            neighbor_search_radius: wp.array(dtype=wp.int32),
            atom_periodic_shifts: wp.array(dtype=wp.vec3i),
            atom_to_cell_mapping: wp.array(dtype=wp.vec3i),
            atoms_per_cell_count: wp.array(dtype=wp.int32),
            cell_atom_start_indices: wp.array(dtype=wp.int32),
            cell_atom_list: wp.array(dtype=wp.int32),
            sorted_positions: wp.array(dtype=wp.vec3d),
            sorted_atom_periodic_shifts: wp.array(dtype=wp.vec3i),
            rebuild_flags: wp.array(dtype=wp.bool),
            neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
            neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
            num_neighbors: wp.array(dtype=wp.int32),
            neighbor_vectors: wp.array(dtype=wp.vec3d, ndim=2),
            neighbor_distances: wp.array(dtype=wp.float64, ndim=2),
            pair_params: wp.array(dtype=wp.float64, ndim=2),
            pair_energies: wp.array(dtype=wp.float64, ndim=2),
            pair_forces: wp.array(dtype=wp.vec3d, ndim=2),
            cutoff: wp.float64,
            fill_value: wp.int32,
            n_outer: wp.int32,
        ) -> None:
            _run_graph_query_cell_list(
                positions,
                cell,
                pbc,
                cells_per_dimension,
                neighbor_search_radius,
                atom_periodic_shifts,
                atom_to_cell_mapping,
                atoms_per_cell_count,
                cell_atom_start_indices,
                cell_atom_list,
                sorted_positions,
                sorted_atom_periodic_shifts,
                rebuild_flags,
                neighbor_matrix,
                neighbor_matrix_shifts,
                num_neighbors,
                cutoff,
                fill_value,
                wp.float64,
                selective=True,
                strategy="pair_centric",
                n_outer=int(n_outer),
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
            pbc: wp.array(dtype=wp.bool),
            cells_per_dimension: wp.array(dtype=wp.int32),
            neighbor_search_radius: wp.array(dtype=wp.int32),
            atom_periodic_shifts: wp.array(dtype=wp.vec3i),
            atom_to_cell_mapping: wp.array(dtype=wp.vec3i),
            atoms_per_cell_count: wp.array(dtype=wp.int32),
            cell_atom_start_indices: wp.array(dtype=wp.int32),
            cell_atom_list: wp.array(dtype=wp.int32),
            sorted_positions: wp.array(dtype=wp.vec3f),
            sorted_atom_periodic_shifts: wp.array(dtype=wp.vec3i),
            rebuild_flags: wp.array(dtype=wp.bool),
            neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
            neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
            num_neighbors: wp.array(dtype=wp.int32),
            neighbor_vectors: wp.array(dtype=wp.vec3f, ndim=2),
            neighbor_distances: wp.array(dtype=wp.float32, ndim=2),
            pair_params: wp.array(dtype=wp.float32, ndim=2),
            pair_energies: wp.array(dtype=wp.float32, ndim=2),
            pair_forces: wp.array(dtype=wp.vec3f, ndim=2),
            cutoff: wp.float32,
            fill_value: wp.int32,
            n_outer: wp.int32,
        ) -> None:
            _run_graph_query_cell_list(
                positions,
                cell,
                pbc,
                cells_per_dimension,
                neighbor_search_radius,
                atom_periodic_shifts,
                atom_to_cell_mapping,
                atoms_per_cell_count,
                cell_atom_start_indices,
                cell_atom_list,
                sorted_positions,
                sorted_atom_periodic_shifts,
                rebuild_flags,
                neighbor_matrix,
                neighbor_matrix_shifts,
                num_neighbors,
                cutoff,
                fill_value,
                wp.float32,
                selective=True,
                strategy="pair_centric",
                n_outer=int(n_outer),
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
        num_outputs=len(_GRAPH_QUERY_PAIR_FN_INOUT),
        in_out_argnames=_GRAPH_QUERY_PAIR_FN_INOUT,
        graph_mode=GraphMode.NONE,
    )


_GRAPH_CELL_LIST_INOUT = [
    "cells_per_dimension",
    "atom_periodic_shifts",
    "atom_to_cell_mapping",
    "atoms_per_cell_count",
    "cell_atom_start_indices",
    "cell_atom_list",
    "sorted_positions",
    "sorted_atom_periodic_shifts",
    "neighbor_matrix",
    "neighbor_matrix_shifts",
    "num_neighbors",
]
_jax_graph_cell_list_f32 = jax_callable(
    _graph_cell_list_f32,
    num_outputs=len(_GRAPH_CELL_LIST_INOUT),
    in_out_argnames=_GRAPH_CELL_LIST_INOUT,
    graph_mode=GraphMode.WARP,
)
_jax_graph_cell_list_f64 = jax_callable(
    _graph_cell_list_f64,
    num_outputs=len(_GRAPH_CELL_LIST_INOUT),
    in_out_argnames=_GRAPH_CELL_LIST_INOUT,
    graph_mode=GraphMode.WARP,
)


def estimate_cell_list_sizes(
    positions: jax.Array,
    cell: jax.Array,
    cutoff: float,
    pbc: jax.Array | None = None,
    buffer_factor: float = 1.5,
) -> tuple[int, jax.Array, jax.Array]:
    """Estimate required cell list sizes based on atomic density.

    Parameters
    ----------
    positions : jax.Array, shape (total_atoms, 3), dtype=float32 or float64
        Atomic coordinates in Cartesian space.
    cell : jax.Array, shape (1, 3, 3), dtype=float32 or float64
        Cell matrix defining lattice vectors.
    cutoff : float
        Cutoff distance for neighbor searching.
    pbc : jax.Array, shape (3,) or (1, 3), dtype=bool, optional
        Periodic boundary condition flags. Default is all True.
    buffer_factor : float, optional
        Buffer multiplier for cell count estimation. Default is 1.5.

    Returns
    -------
    max_total_cells : int
        Maximum total number of cells to allocate.
    cells_per_dimension : jax.Array, shape (3,) or (1, 3), dtype=int32
        Estimated number of cells in each dimension.
    neighbor_search_radius : jax.Array, shape (3,), dtype=int32
        Estimated search radius in neighboring cells.

    Notes
    -----
    This function estimates cell list parameters based on atomic positions and
    density. The actual number of cells used will be determined during cell
    list construction.

    .. warning::

        This function is **not compatible with** ``jax.jit``. The returned
        ``max_total_cells`` is used to determine array allocation sizes, which
        must be concrete (statically known) at JAX trace time. When using
        ``cell_list`` or ``build_cell_list`` inside ``jax.jit``, provide
        ``max_total_cells`` explicitly to bypass this function.
    """
    if cell.ndim == 2:
        cell = cell[jnp.newaxis, :, :]
    if pbc is None:
        pbc = jnp.ones((1, 3), dtype=jnp.bool_)
    if pbc.ndim == 1:
        pbc = pbc[jnp.newaxis, :]

    # Simple estimation: compute total volume and estimate cell volume
    # Cell volume = det(cell_matrix)
    det = jnp.linalg.det(cell[0])
    volume = jnp.abs(det)
    cell_volume = cutoff**3
    # TODO: This estimation derives array sizes from traced input data (cell
    # geometry), which is fundamentally incompatible with jax.jit compilation.
    # The JAX bindings need a refactored usage pattern where sizing is always
    # performed outside the JIT boundary, or a fixed upper-bound allocation
    # strategy is adopted.
    num_cells_est = jnp.int32(volume / cell_volume * buffer_factor)
    max_total_cells = jnp.max(jnp.array([num_cells_est, 8]))  # Minimum 8 cells

    # Compute cells_per_dimension and neighbor_search_radius from cell geometry,
    # mirroring the Warp _estimate_cell_list_sizes kernel used by the Torch
    # path: natural cell count, ADAPTIVE_MIN_CELLS=4 promotion on PBC axes,
    # halve-to-fit when total cells > max_total_cells, then compute
    # neighbor_search_radius against the FINAL cells_per_dimension.
    inverse_cell_transpose = jnp.linalg.inv(cell[0]).T
    face_distances = 1.0 / jnp.linalg.norm(inverse_cell_transpose, axis=1)
    cells_per_dimension = jnp.maximum(jnp.int32(face_distances / cutoff), 1)

    pbc_squeezed = pbc.squeeze()[:3] if pbc.ndim > 1 else pbc[:3]

    # ADAPTIVE_MIN_CELLS=4: promote each PBC axis (or any axis already > 1)
    # up to at least 4 cells so the atom-centric query has enough cell-level
    # parallelism.  Open axes with a single cell are left alone.
    ADAPTIVE_MIN_CELLS = 4
    promote_mask = pbc_squeezed | (cells_per_dimension > 1)
    # Bit-trick: smallest power-of-2 multiplier that brings cells_per_dim
    # to >= ADAPTIVE_MIN_CELLS.  At cells_per_dim=1 -> multiplier=4; at 2 ->
    # 2; at >= 4 -> 1.
    needed_mult = jnp.where(
        cells_per_dimension >= ADAPTIVE_MIN_CELLS,
        1,
        ADAPTIVE_MIN_CELLS // jnp.maximum(cells_per_dimension, 1),
    )
    cells_per_dimension = jnp.where(
        promote_mask,
        cells_per_dimension * needed_mult,
        cells_per_dimension,
    )

    # Halve-to-fit: if total cells exceeds max_total_cells, halve each axis
    # (floor with min=1) repeatedly until total <= max.  Mirrors the kernel's
    # ``while total_cells > max_cells_allowed`` loop.
    def _halve_to_fit(cpd):
        total = cpd[0] * cpd[1] * cpd[2]
        cpd = jnp.where(total > max_total_cells, jnp.maximum(cpd // 2, 1), cpd)
        return cpd

    # 3 iterations is enough to halve 4*4*4=64 down to 2*2*2=8.  Cap at 16
    # iterations to handle larger grids without unbounded growth.
    for _ in range(16):
        cells_per_dimension = _halve_to_fit(cells_per_dimension)

    neighbor_search_radius = jnp.where(
        (cells_per_dimension == 1) & ~pbc_squeezed,
        jnp.zeros(3, dtype=jnp.int32),
        jnp.int32(jnp.ceil(cutoff * cells_per_dimension / face_distances)),
    )

    return max_total_cells, cells_per_dimension, neighbor_search_radius


def _cell_list_pair_outputs_forward(
    positions: jax.Array,
    cell: jax.Array,
    *,
    pbc_bool: jax.Array,
    cells_per_dimension: jax.Array,
    atom_periodic_shifts: jax.Array,
    atom_to_cell_mapping: jax.Array,
    atoms_per_cell_count: jax.Array,
    cell_atom_start_indices: jax.Array,
    cell_atom_list: jax.Array,
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
    half_fill: bool = False,
) -> _NeighborForwardOutput:
    """Forward closure consumed by ``_route_pair_outputs``.

    Runs the gather + pair-output kernel.  The Warp kernel does not
    propagate gradients; differentiability is added by the autograd
    primitive's reconstruction-based backward.  When ``pair_fn`` is set, a
    ``pair_fn``-specialized kernel writes per-pair ``pair_energies`` /
    ``pair_forces`` which ride along in ``extra_outputs`` (forward-only).

    When ``target_indices`` is set (partial neighbor lists), the kernel runs
    with ``partial=True``: output row ``r`` maps to atom ``target_indices[r]``,
    the output buffers carry ``num_targets`` compact rows, and the kernel is
    launched ``(num_targets,)``.

    When ``strategy == "pair_centric"`` the path instead runs the block-scheduled
    pair-centric callable (which gathers internally and is sized by the host-read
    static ``n_outer``); the pair set is identical to atom-centric.  The shared
    tail (index residuals + reconstruction) is strategy-agnostic.
    """
    # The warp kernels are non-differentiable across the JAX boundary;
    # detach positions and cell for the kernel call.  The autograd primitive
    # in :mod:`nvalchemiops.jax.neighbors._autograd` separately receives the
    # live positions/cell and produces the analytical backward.
    positions = jax.lax.stop_gradient(positions)
    cell = jax.lax.stop_gradient(cell)

    f64 = positions.dtype == jnp.float64
    gather_kernel = _jax_gather_fused_f64 if f64 else _jax_gather_fused_f32

    total_atoms = positions.shape[0]
    # Output rows: ``num_targets`` for the partial (``target_indices``) path,
    # else ``total_atoms``.  The pair kernel is launched over these rows and the
    # output buffers are pre-sized to match (by the public function / autograd).
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

    rf = jnp.ones((1,), dtype=jnp.bool_)
    # ``fill_value`` is the matrix padding sentinel; the public function
    # pre-fills ``neighbor_matrix`` with ``total_atoms`` and the selective
    # callables only zero ``num_neighbors`` (they do not reset the matrix).
    fill_value = int(total_atoms)

    if is_pair_centric:
        # Pair-centric: the launcher gathers internally and runs the
        # block-scheduled kernel sized by the host-read static ``n_outer``.
        # ``target_indices`` is rejected upstream for this strategy.
        sorted_positions = jnp.zeros((total_atoms, 3), dtype=positions.dtype)
        sorted_atom_periodic_shifts = jnp.zeros((total_atoms, 3), dtype=jnp.int32)
        if has_pair_fn:
            pc_callable = _get_jax_cell_list_pair_centric_pair_fn_callable(
                pair_fn, wp_dtype
            )
            (_sp, _sas, nm_out, nms_out, nn_out, nv_out, nd_out, pe, pf) = pc_callable(
                positions,
                cell,
                pbc_bool,
                cells_per_dimension,
                neighbor_search_radius,
                atom_periodic_shifts,
                atom_to_cell_mapping,
                atoms_per_cell_count,
                cell_atom_start_indices,
                cell_atom_list,
                sorted_positions,
                sorted_atom_periodic_shifts,
                rf,
                neighbor_matrix,
                neighbor_matrix_shifts,
                num_neighbors,
                neighbor_vectors,
                neighbor_distances,
                pp_arg,
                pe,
                pf,
                float(cutoff),
                fill_value,
                int(n_outer),
            )
        else:
            pc_callable = (
                _jax_query_cell_list_pair_centric_geom_f64
                if f64
                else _jax_query_cell_list_pair_centric_geom_f32
            )
            (_sp, _sas, nm_out, nms_out, nn_out, nv_out, nd_out) = pc_callable(
                positions,
                cell,
                pbc_bool,
                cells_per_dimension,
                neighbor_search_radius,
                atom_periodic_shifts,
                atom_to_cell_mapping,
                atoms_per_cell_count,
                cell_atom_start_indices,
                cell_atom_list,
                sorted_positions,
                sorted_atom_periodic_shifts,
                rf,
                neighbor_matrix,
                neighbor_matrix_shifts,
                num_neighbors,
                neighbor_vectors,
                neighbor_distances,
                float(cutoff),
                fill_value,
                int(n_outer),
            )
    else:
        if has_pair_fn or is_partial:
            # The ``partial`` and/or ``pair_fn`` specialization is not a
            # module-level registration; build (and cache) it at call time.
            pair_kernel = _get_jax_cell_list_pair_outputs_kernel(
                pair_fn, wp_dtype, is_partial, half_fill
            )
        elif half_fill:
            pair_kernel = (
                _jax_build_neighbor_matrix_local_count_sorted_pair_half_f64
                if f64
                else _jax_build_neighbor_matrix_local_count_sorted_pair_half_f32
            )
        else:
            pair_kernel = (
                _jax_build_neighbor_matrix_local_count_sorted_pair_f64
                if f64
                else _jax_build_neighbor_matrix_local_count_sorted_pair_f32
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

        empty_bool2d = jnp.zeros((0, 3), dtype=jnp.bool_)
        empty_i32 = jnp.zeros((0,), dtype=jnp.int32)
        empty_vec3i = jnp.zeros((0, 3), dtype=jnp.int32)

        outs = pair_kernel(
            positions,
            atom_periodic_shifts,
            sorted_positions,
            sorted_atom_periodic_shifts,
            cell,
            pbc_bool,
            empty_bool2d,
            empty_i32,
            float(cutoff),
            cells_per_dimension,
            empty_vec3i,
            neighbor_search_radius,
            empty_vec3i,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
            empty_i32,  # cell_offsets
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

    i_idx, j_idx, shifts_ret, batch_idx_flat, mask_ = _build_index_residuals(
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
        batch_idx=batch_idx_flat,
        active_mask=mask_,
        matrix_shape=(K, M),
    )


def build_cell_list(
    positions: jax.Array,
    cutoff: float,
    cell: jax.Array,
    pbc: jax.Array,
    cells_per_dimension: jax.Array | None = None,
    neighbor_search_radius: jax.Array | None = None,
    atom_periodic_shifts: jax.Array | None = None,
    atom_to_cell_mapping: jax.Array | None = None,
    atoms_per_cell_count: jax.Array | None = None,
    cell_atom_start_indices: jax.Array | None = None,
    cell_atom_list: jax.Array | None = None,
    max_total_cells: int | None = None,
    graph_mode: Literal["none", "warp"] = "none",
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
]:
    """Build spatial cell list for efficient neighbor searching.

    Parameters
    ----------
    positions : jax.Array, shape (total_atoms, 3), dtype=float32 or float64
        Atomic coordinates in Cartesian space.
    cutoff : float
        Cutoff distance for neighbor searching. Must be positive.
    cell : jax.Array, shape (1, 3, 3), dtype=float32 or float64
        Cell matrix defining lattice vectors.
    pbc : jax.Array, shape (3,) or (1, 3), dtype=bool
        Periodic boundary condition flags.
    cells_per_dimension : jax.Array, shape (3,), dtype=int32, optional
        OUTPUT: Number of cells in x, y, z directions. If None, allocated.
    neighbor_search_radius : jax.Array, shape (3,), dtype=int32, optional
        Search radius in neighboring cells. If None, allocated.
    atom_periodic_shifts : jax.Array, shape (total_atoms, 3), dtype=int32, optional
        OUTPUT: Periodic boundary crossings for each atom. If None, allocated.
    atom_to_cell_mapping : jax.Array, shape (total_atoms, 3), dtype=int32, optional
        OUTPUT: 3D cell coordinates for each atom. If None, allocated.
    atoms_per_cell_count : jax.Array, shape (max_total_cells,), dtype=int32, optional
        OUTPUT: Number of atoms in each cell. If None, allocated.
    cell_atom_start_indices : jax.Array, shape (max_total_cells,), dtype=int32, optional
        OUTPUT: Starting index in cell_atom_list for each cell. If None, allocated.
    cell_atom_list : jax.Array, shape (total_atoms,), dtype=int32, optional
        OUTPUT: Flattened list of atom indices organized by cell. If None, allocated.
    max_total_cells : int, optional
        Maximum number of cells to allocate. If None, will be estimated.

    Returns
    -------
    cells_per_dimension : jax.Array, shape (3,), dtype=int32
        Number of cells in x, y, z directions.
    atom_periodic_shifts : jax.Array, shape (total_atoms, 3), dtype=int32
        Periodic boundary crossings for each atom.
    atom_to_cell_mapping : jax.Array, shape (total_atoms, 3), dtype=int32
        3D cell coordinates for each atom.
    atoms_per_cell_count : jax.Array, shape (max_total_cells,), dtype=int32
        Number of atoms in each cell.
    cell_atom_start_indices : jax.Array, shape (max_total_cells,), dtype=int32
        Starting index in cell_atom_list for each cell.
    cell_atom_list : jax.Array, shape (total_atoms,), dtype=int32
        Flattened list of atom indices organized by cell.
    neighbor_search_radius : jax.Array, shape (3,), dtype=int32
        Search radius in neighboring cells.

    Notes
    -----
    When calling inside ``jax.jit``, ``max_total_cells`` **must** be provided
    to avoid calling ``estimate_cell_list_sizes``, which is not JIT-compatible.

    ``graph_mode="warp"`` uses a fused ``jax_callable`` that captures the full
    Warp-side build sequence. For replay-friendly usage inside ``jax.jit``,
    donate and reuse the optional cell-list buffers.

    See Also
    --------
    query_cell_list : Query the built cell list for neighbors
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
            "build_cell_list does not accept return_distances / "
            "return_vectors / target_indices / pair_fn-related kwargs.  "
            "Use the top-level cell_list() wrapper, which routes pair "
            "outputs through the JAX autograd path, or call the warp "
            "factory directly for low-level access.",
        )

    graph_mode = _validate_graph_mode(graph_mode)

    if cell.ndim == 2:
        cell = cell[jnp.newaxis, :, :]
    if pbc.ndim == 1:
        pbc = pbc[jnp.newaxis, :]

    if max_total_cells is None:
        max_total_cells, _, neighbor_search_radius_est = estimate_cell_list_sizes(
            positions, cell, cutoff, pbc
        )
        if neighbor_search_radius is None:
            neighbor_search_radius = neighbor_search_radius_est
    else:
        if neighbor_search_radius is None:
            neighbor_search_radius = jnp.ones(3, dtype=jnp.int32)

    # Allocate cell list tensors if not provided
    if cells_per_dimension is None:
        cells_per_dimension = jnp.ones(3, dtype=jnp.int32)
    if atom_periodic_shifts is None:
        atom_periodic_shifts = jnp.zeros((positions.shape[0], 3), dtype=jnp.int32)
    if atom_to_cell_mapping is None:
        atom_to_cell_mapping = jnp.zeros((positions.shape[0], 3), dtype=jnp.int32)
    if atoms_per_cell_count is None:
        atoms_per_cell_count = jnp.zeros(max_total_cells, dtype=jnp.int32)
    elif graph_mode == "none":
        atoms_per_cell_count = atoms_per_cell_count.at[:].set(jnp.int32(0))
    if cell_atom_start_indices is None:
        cell_atom_start_indices = jnp.zeros(max_total_cells, dtype=jnp.int32)
    if cell_atom_list is None:
        cell_atom_list = jnp.zeros(positions.shape[0], dtype=jnp.int32)

    # Select kernels based on dtype
    if positions.dtype == jnp.float64:
        _construct_bin_size = _jax_construct_bin_size_f64
        _count_atoms = _jax_count_atoms_per_bin_f64
        _bin_atoms = _jax_bin_atoms_f64
    else:
        _construct_bin_size = _jax_construct_bin_size_f32
        _count_atoms = _jax_count_atoms_per_bin_f32
        _bin_atoms = _jax_bin_atoms_f32
        positions = positions.astype(jnp.float32)

    # Ensure cell dtype matches positions dtype so Warp kernel dispatch is consistent
    if cell.dtype != positions.dtype:
        cell = cell.astype(positions.dtype)

    total_atoms = positions.shape[0]

    # Squeeze pbc to 1D for the single-system static specialization.
    pbc_1d = pbc.squeeze() if pbc.ndim == 2 else pbc
    pbc_bool = pbc_1d.astype(jnp.bool_)
    empty_bool2d = jnp.zeros((0, 3), dtype=jnp.bool_)
    empty_i32 = jnp.zeros((0,), dtype=jnp.int32)
    empty_vec3i = jnp.zeros((0, 3), dtype=jnp.int32)

    if graph_mode == "warp":
        graph_build = (
            _jax_graph_build_cell_list_f64
            if positions.dtype == jnp.float64
            else _jax_graph_build_cell_list_f32
        )
        (
            cells_per_dimension,
            atom_periodic_shifts,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
        ) = graph_build(
            positions,
            cell,
            pbc_bool,
            cells_per_dimension,
            atom_periodic_shifts,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
            float(cutoff),
        )
    else:
        # Step 1: Construct bin sizes
        (cells_per_dimension,) = _construct_bin_size(
            cell,
            pbc_bool,
            empty_bool2d,
            cells_per_dimension,
            empty_vec3i,
            float(cutoff),
            int(max_total_cells),
            launch_dims=(1,),
        )

        # Step 2: Count atoms per bin
        atoms_per_cell_count, atom_periodic_shifts = _count_atoms(
            positions,
            cell,
            pbc_bool,
            empty_bool2d,
            empty_i32,
            cells_per_dimension,
            empty_vec3i,
            empty_i32,
            atoms_per_cell_count,
            atom_periodic_shifts,
            launch_dims=(total_atoms,),
        )

        # Step 3: Compute exclusive prefix sum (replaces wp.utils.array_scan)
        cell_atom_start_indices = jnp.concatenate(
            [
                jnp.array([0], dtype=jnp.int32),
                jnp.cumsum(atoms_per_cell_count[:-1], dtype=jnp.int32),
            ]
        )

        # Step 4: Zero counts before second pass
        atoms_per_cell_count = jnp.zeros_like(atoms_per_cell_count)

        # Step 5: Bin atoms
        atom_to_cell_mapping, atoms_per_cell_count, cell_atom_list = _bin_atoms(
            positions,
            cell,
            pbc_bool,
            empty_bool2d,
            empty_i32,
            cells_per_dimension,
            empty_vec3i,
            empty_i32,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
            launch_dims=(total_atoms,),
        )

    return (
        cells_per_dimension,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
        neighbor_search_radius,
    )


def query_cell_list(
    positions: jax.Array,
    cutoff: float,
    cell: jax.Array,
    pbc: jax.Array,
    cells_per_dimension: jax.Array,
    atom_periodic_shifts: jax.Array,
    atom_to_cell_mapping: jax.Array,
    atoms_per_cell_count: jax.Array,
    cell_atom_start_indices: jax.Array,
    cell_atom_list: jax.Array,
    neighbor_search_radius: jax.Array,
    max_neighbors: int | None = None,
    neighbor_matrix: jax.Array | None = None,
    neighbor_matrix_shifts: jax.Array | None = None,
    num_neighbors: jax.Array | None = None,
    rebuild_flags: jax.Array | None = None,
    graph_mode: Literal["none", "warp"] = "none",
    half_fill: bool = False,
    strategy: str = "auto",
    atom_centric_path: str = "auto",
    sorted_positions: jax.Array | None = None,
    sorted_atom_periodic_shifts: jax.Array | None = None,
    target_indices: jax.Array | None = None,
    return_vectors: bool = False,
    return_distances: bool = False,
    pair_fn: wp.Function | None = None,
    pair_params: jax.Array | None = None,
    neighbor_vectors: jax.Array | None = None,
    neighbor_distances: jax.Array | None = None,
    pair_energies: jax.Array | None = None,
    pair_forces: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Query cell list to find neighbors within cutoff.

    Parameters
    ----------
    positions : jax.Array, shape (total_atoms, 3), dtype=float32 or float64
        Atomic coordinates in Cartesian space.
    cutoff : float
        Cutoff distance for neighbor detection.
    cell : jax.Array, shape (1, 3, 3), dtype=float32 or float64
        Cell matrix defining lattice vectors.
    pbc : jax.Array, shape (3,) or (1, 3), dtype=bool
        Periodic boundary condition flags.
    cells_per_dimension : jax.Array, shape (3,), dtype=int32
        Number of cells in each dimension.
    atom_periodic_shifts : jax.Array, shape (total_atoms, 3), dtype=int32
        Periodic boundary crossings for each atom (output from ``build_cell_list``).
    atom_to_cell_mapping : jax.Array, shape (total_atoms, 3), dtype=int32
        3D cell coordinates for each atom.
    atoms_per_cell_count : jax.Array, shape (max_total_cells,), dtype=int32
        Number of atoms in each cell (output from ``build_cell_list``).
    cell_atom_start_indices : jax.Array, shape (max_total_cells,), dtype=int32
        Starting index in cell_atom_list for each cell.
    cell_atom_list : jax.Array, shape (total_atoms,), dtype=int32
        Flattened list of atom indices organized by cell.
    neighbor_search_radius : jax.Array, shape (3,), dtype=int32
        Search radius in neighboring cells.
    max_neighbors : int, optional
        Maximum number of neighbors per atom.
    neighbor_matrix : jax.Array, optional
        Pre-allocated neighbor matrix.
    num_neighbors : jax.Array, optional
        Pre-allocated neighbors count array.
    strategy : {"auto", "atom_centric", "pair_centric"}, default "auto"
        Cell-list query sub-strategy.  Both strategies produce identical pair
        SETS; only the per-row ordering inside ``neighbor_matrix`` differs
        (pair-centric accumulates via ``atomic_add`` so its row order is
        nondeterministic).  ``"auto"`` uses :func:`select_cell_list_strategy`
        ``(N, cutoff)`` on GPU and ``"atom_centric"`` on CPU.
        ``"pair_centric"`` is CUDA-only and requires a **concrete**
        ``neighbor_search_radius``: its launch grid is sized by a host-read
        ``n_outer`` baked at launch-build time, so it works eagerly / outside
        ``jax.jit`` but raises a clear error under ``jax.jit`` with a traced
        radius.  ``"pair_centric"`` is registered with ``graph_mode="none"``;
        ``graph_mode="warp"`` + explicit pair-centric raises
        ``NotImplementedError``.
    atom_centric_path : {"auto", "direct", "sorted"}, default "auto"
        ``"direct"`` reads positions in original order and skips the sorted
        gather (the symmetric-full-fill kernel), on the plain full-fill,
        ``graph_mode="none"`` path; ``"sorted"`` uses the gather + sorted kernel.
        ``"auto"`` resolves to ``"sorted"`` on JAX (perf-only divergence from
        Torch's ``"auto"`` -> ``"direct"``); half_fill / ``graph_mode="warp"``
        always use the sorted kernel.
    sorted_positions : jax.Array, shape (total_atoms, 3), optional
        Caller-owned per-cell-contiguous gather scratch (dtype must match
        ``positions``).  When omitted it is allocated internally.
    sorted_atom_periodic_shifts : jax.Array, shape (total_atoms, 3), int32, optional
        Caller-owned per-cell-contiguous gather scratch.  Allocated internally
        when omitted.

    Returns
    -------
    neighbor_matrix : jax.Array, shape (total_atoms, max_neighbors), dtype=int32
        Neighbor matrix with neighbor atom indices.
    num_neighbors : jax.Array, shape (total_atoms,), dtype=int32
        Number of neighbors found for each atom.
    neighbor_matrix_shifts : jax.Array, shape (total_atoms, max_neighbors, 3), dtype=int32
        Periodic shift vectors for each neighbor relationship.

    See Also
    --------
    build_cell_list : Build cell list before querying
    cell_list : Combined build and query operation
    """
    _validate_atom_centric_path(atom_centric_path)

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
            "Pair-output kernels (target_indices, return_vectors, "
            "return_distances, pair_fn, pair_params, neighbor_vectors, "
            "neighbor_distances, pair_energies, "
            "pair_forces) are not yet wired through the JAX cell-list "
            "bindings.  Use the torch bindings or call the warp factory "
            "directly.",
        )

    graph_mode = _validate_graph_mode(graph_mode)
    if half_fill and graph_mode != "none":
        raise NotImplementedError(
            "half_fill=True is only supported with graph_mode='none' in the "
            "JAX cell-list binding; CUDA-graph capture of the half-fill kernel "
            "is a follow-up.",
        )

    # Resolve the cell-list query sub-strategy.  Pair-centric needs CUDA, a
    # concrete radius (host-read ``n_outer``), graph_mode="none", and full-fill.
    # ``half_fill`` makes ``"auto"`` resolve to atom_centric (pair-centric is
    # full-fill only), so the default path keeps working for every geometry.
    device_is_cpu = _is_cpu_array(positions)
    chosen = _resolve_cell_strategy(
        strategy,
        total_atoms=int(positions.shape[0]),
        cutoff=float(cutoff),
        device_is_cpu=device_is_cpu,
        half_fill=half_fill,
    )
    # Only an EXPLICIT pair-centric request collides with half_fill; auto has
    # already fallen back to atom_centric above.
    if strategy == "pair_centric" and half_fill:
        raise NotImplementedError(
            "strategy='pair_centric' with half_fill=True is not supported in "
            "the JAX cell-list binding (JAX cell_list is full-fill).  Use "
            "strategy='atom_centric' for half_fill, or half_fill=False for "
            "pair-centric.",
        )
    if graph_mode == "warp":
        # CUDA-graph replay bakes the launch dim; a pair-centric launch sized
        # from a host-read ``n_outer`` is unsafe to replay across a changed
        # radius.  Explicit pair-centric raises; auto/atom-centric run the
        # existing atom-centric fused graph callable.
        if strategy == "pair_centric":
            raise NotImplementedError(
                "strategy='pair_centric' is not supported with "
                "graph_mode='warp' (the launch dim is baked from a host-read "
                "n_outer; CUDA-graph replay across a changed radius is "
                "unsafe).  Use graph_mode='none' for pair-centric, or "
                "strategy='atom_centric'.",
            )
        chosen = "atom_centric"

    if max_neighbors is None:
        max_neighbors = estimate_max_neighbors(cutoff)

    if neighbor_matrix is None:
        neighbor_matrix = jnp.full(
            (positions.shape[0], max_neighbors),
            positions.shape[0],
            dtype=jnp.int32,
        )
    elif rebuild_flags is None and graph_mode == "none":
        neighbor_matrix = neighbor_matrix.at[:].set(jnp.int32(positions.shape[0]))

    if num_neighbors is None:
        num_neighbors = jnp.zeros(positions.shape[0], dtype=jnp.int32)
    elif rebuild_flags is None and graph_mode == "none":
        num_neighbors = num_neighbors.at[:].set(jnp.int32(0))

    if neighbor_matrix_shifts is None:
        neighbor_matrix_shifts = jnp.zeros(
            (positions.shape[0], max_neighbors, 3),
            dtype=jnp.int32,
        )
    elif rebuild_flags is None and graph_mode == "none":
        neighbor_matrix_shifts = neighbor_matrix_shifts.at[:].set(jnp.int32(0))

    # Select kernels based on dtype.  All paths use the same sorted-reads
    # atom-centric kernel; selective callers supply a non-trivial
    # ``rebuild_flags`` array, non-selective callers an always-True 1-element
    # flag.
    # ``atom_centric_path="direct"`` reads positions in original order and skips the
    # sorted gather (the symmetric-full-fill kernel).  It applies only to the plain
    # full-fill, non-graph path; half_fill / graph_mode="warp" keep the sorted kernel.
    # ``"auto"`` stays sorted on JAX (a documented, perf-only divergence from Torch,
    # which resolves auto->direct); explicit ``"direct"`` now branches here.
    use_direct = (
        atom_centric_path == "direct" and not half_fill and graph_mode == "none"
    )
    if positions.dtype == jnp.float64:
        _gather_kernel = _jax_gather_fused_f64
        if use_direct:
            _build_kernel = _jax_build_neighbor_matrix_local_count_direct_f64
        elif half_fill:
            _build_kernel = _jax_build_neighbor_matrix_local_count_sorted_half_f64
        else:
            _build_kernel = _jax_build_neighbor_matrix_local_count_sorted_f64
    else:
        _gather_kernel = _jax_gather_fused_f32
        if use_direct:
            _build_kernel = _jax_build_neighbor_matrix_local_count_direct_f32
        elif half_fill:
            _build_kernel = _jax_build_neighbor_matrix_local_count_sorted_half_f32
        else:
            _build_kernel = _jax_build_neighbor_matrix_local_count_sorted_f32
        positions = positions.astype(jnp.float32)

    # Ensure cell dtype matches positions dtype so Warp kernel dispatch is consistent
    if cell.dtype != positions.dtype:
        cell = cell.astype(positions.dtype)

    total_atoms = positions.shape[0]

    # Squeeze pbc to 1D for the single-system static specialization.
    pbc_1d = pbc.squeeze() if pbc.ndim == 2 else pbc
    pbc_bool = pbc_1d.astype(jnp.bool_)
    empty_bool2d = jnp.zeros((0, 3), dtype=jnp.bool_)
    empty_i32 = jnp.zeros((0,), dtype=jnp.int32)
    empty_vec3i = jnp.zeros((0, 3), dtype=jnp.int32)
    empty_scalar2d = jnp.zeros((0, 0), dtype=positions.dtype)
    empty_vec_matrix = jnp.zeros((0, 0, 3), dtype=positions.dtype)

    if rebuild_flags is not None:
        rf = rebuild_flags.flatten()[:1].astype(jnp.bool_)
        # Pre-zero num_neighbors when the flag is True so the kernel's
        # atomic emits start from zero; leave it untouched otherwise.
        num_neighbors = jnp.where(rf[0], jnp.zeros_like(num_neighbors), num_neighbors)
    else:
        rf = jnp.ones((1,), dtype=jnp.bool_)

    # Sorted scratch lives next to ``positions`` in dtype/device.  Both or
    # neither must be caller-provided; a mixed state is rejected so a partial
    # capture cannot silently fall back to an internal allocation.
    if (sorted_positions is None) != (sorted_atom_periodic_shifts is None):
        raise ValueError(
            "Pass both sorted_positions and sorted_atom_periodic_shifts, or "
            "neither - got a mixed state.",
        )
    if sorted_positions is None:
        sorted_positions = jnp.zeros((total_atoms, 3), dtype=positions.dtype)
    elif sorted_positions.dtype != positions.dtype:
        sorted_positions = sorted_positions.astype(positions.dtype)
    if sorted_atom_periodic_shifts is None:
        sorted_atom_periodic_shifts = jnp.zeros((total_atoms, 3), dtype=jnp.int32)

    if chosen == "pair_centric":
        # Host-read the per-axis radius to size the pair-centric launch grid.
        # This is a device->host sync: legal eagerly / with a concrete radius,
        # but illegal under jax.jit with a traced ``neighbor_search_radius``.
        try:
            Rx = int(neighbor_search_radius[0])
            Ry = int(neighbor_search_radius[1])
            Rz = int(neighbor_search_radius[2])
        except (
            jax.errors.ConcretizationTypeError,
            jax.errors.TracerIntegerConversionError,
        ) as exc:
            raise ValueError(
                "strategy='pair_centric' needs a concrete "
                "neighbor_search_radius to size its launch grid (n_outer is "
                "host-read).  Compute the cell-list sizing outside jax.jit and "
                "pass a concrete neighbor_search_radius, or use "
                "strategy='atom_centric'.",
            ) from exc
        # JAX cell_list is full-fill (half_fill+pair_centric raised above).
        n_outer = compute_batch_pair_centric_n_outer((Rx, Ry, Rz), False)
        total_cells = int(atoms_per_cell_count.shape[0])
        if not is_pair_centric_launch_safe(total_cells, n_outer):
            if strategy == "pair_centric":
                _raise_unsafe_pair_centric_launch(total_cells, n_outer)
            chosen = "atom_centric"
        elif strategy == "auto" and not is_pair_centric_parallelism_sufficient(
            total_atoms, total_cells, n_outer
        ):
            chosen = "atom_centric"

    if chosen == "pair_centric":
        fill_value = int(positions.shape[0])
        pair_query = (
            (
                _jax_query_cell_list_pair_centric_selective_f64
                if positions.dtype == jnp.float64
                else _jax_query_cell_list_pair_centric_selective_f32
            )
            if rebuild_flags is not None
            else (
                _jax_query_cell_list_pair_centric_f64
                if positions.dtype == jnp.float64
                else _jax_query_cell_list_pair_centric_f32
            )
        )
        # The pair-centric callable runs the internal gather_fused into the
        # sorted scratch and the pair-centric linear launch; ``n_outer`` enters
        # as a static scalar.
        (
            sorted_positions,
            sorted_atom_periodic_shifts,
            neighbor_matrix,
            neighbor_matrix_shifts,
            num_neighbors,
        ) = pair_query(
            positions,
            cell,
            pbc_bool,
            cells_per_dimension,
            neighbor_search_radius,
            atom_periodic_shifts,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
            sorted_positions,
            sorted_atom_periodic_shifts,
            rf,
            neighbor_matrix,
            neighbor_matrix_shifts,
            num_neighbors,
            float(cutoff),
            fill_value,
            int(n_outer),
        )
        return neighbor_matrix, num_neighbors, neighbor_matrix_shifts

    if graph_mode == "warp":
        fill_value = int(positions.shape[0])
        graph_query = (
            (
                _jax_graph_query_cell_list_selective_f64
                if positions.dtype == jnp.float64
                else _jax_graph_query_cell_list_selective_f32
            )
            if rebuild_flags is not None
            else (
                _jax_graph_query_cell_list_f64
                if positions.dtype == jnp.float64
                else _jax_graph_query_cell_list_f32
            )
        )
        (
            sorted_positions,
            sorted_atom_periodic_shifts,
            neighbor_matrix,
            neighbor_matrix_shifts,
            num_neighbors,
        ) = graph_query(
            positions,
            cell,
            pbc_bool,
            cells_per_dimension,
            neighbor_search_radius,
            atom_periodic_shifts,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
            sorted_positions,
            sorted_atom_periodic_shifts,
            rf,
            neighbor_matrix,
            neighbor_matrix_shifts,
            num_neighbors,
            float(cutoff),
            fill_value,
        )
    else:
        if not use_direct:
            # Sorted path: gather positions/shifts into cell order for coalesced
            # reads.  The direct kernel reads ``positions`` directly and ignores the
            # sorted arrays, so it skips the gather (matching the Torch binding).
            sorted_positions, sorted_atom_periodic_shifts = _gather_kernel(
                positions,
                atom_periodic_shifts,
                cell_atom_list,
                sorted_positions,
                sorted_atom_periodic_shifts,
                launch_dims=(total_atoms,),
            )
        neighbor_matrix, neighbor_matrix_shifts, num_neighbors = _build_kernel(
            positions,
            atom_periodic_shifts,
            sorted_positions,
            sorted_atom_periodic_shifts,
            cell,
            pbc_bool,
            empty_bool2d,
            empty_i32,
            float(cutoff),
            cells_per_dimension,
            empty_vec3i,
            neighbor_search_radius,
            empty_vec3i,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
            empty_i32,
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


def cell_list(
    positions: jax.Array,
    cutoff: float,
    cell: jax.Array | None = None,
    pbc: jax.Array | None = None,
    max_neighbors: int | None = None,
    max_total_cells: int | None = None,
    return_neighbor_list: bool = False,
    half_fill: bool = False,
    fill_value: int | None = None,
    strategy: str = "auto",
    atom_centric_path: str = "auto",
    cells_per_dimension: jax.Array | None = None,
    neighbor_search_radius: jax.Array | None = None,
    atom_periodic_shifts: jax.Array | None = None,
    atom_to_cell_mapping: jax.Array | None = None,
    atoms_per_cell_count: jax.Array | None = None,
    cell_atom_start_indices: jax.Array | None = None,
    cell_atom_list: jax.Array | None = None,
    sorted_positions: jax.Array | None = None,
    sorted_atom_periodic_shifts: jax.Array | None = None,
    neighbor_matrix: jax.Array | None = None,
    neighbor_matrix_shifts: jax.Array | None = None,
    num_neighbors: jax.Array | None = None,
    graph_mode: Literal["none", "warp"] = "none",
    target_indices: jax.Array | None = None,
    return_vectors: bool = False,
    return_distances: bool = False,
    pair_fn: wp.Function | None = None,
    pair_params: jax.Array | None = None,
    neighbor_vectors: jax.Array | None = None,
    neighbor_distances: jax.Array | None = None,
    pair_energies: jax.Array | None = None,
    pair_forces: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Build and query spatial cell list for efficient neighbor finding.

    This is a convenience function that combines build_cell_list and query_cell_list
    in a single call.

    Parameters
    ----------
    positions : jax.Array, shape (total_atoms, 3), dtype=float32 or float64
        Atomic coordinates in Cartesian space.
    cutoff : float
        Cutoff distance for neighbor detection.
    cell : jax.Array, shape (1, 3, 3), dtype=float32 or float64, optional
        Cell matrix defining lattice vectors. Default is identity matrix.
    pbc : jax.Array, shape (3,) or (1, 3), dtype=bool, optional
        Periodic boundary condition flags. Default is all True.
    max_neighbors : int, optional
        Maximum number of neighbors per atom. If None, will be estimated.
    max_total_cells : int, optional
        Maximum number of cells to allocate. If None, will be estimated.
    return_neighbor_list : bool, optional
        If True, convert result to COO neighbor list format. Default is False.
    strategy : {"auto", "atom_centric", "pair_centric"}, default "auto"
        Cell-list query sub-strategy, forwarded to :func:`query_cell_list`.
        Both strategies produce identical pair SETS; only per-row ordering in
        ``neighbor_matrix`` differs.  ``"pair_centric"`` is CUDA-only, requires
        a concrete ``neighbor_search_radius`` (host-read ``n_outer``), runs only
        with ``graph_mode="none"`` and full-fill, and raises a clear error
        under ``jax.jit`` with a traced radius.  ``graph_mode="warp"`` +
        explicit ``strategy="pair_centric"`` raises ``NotImplementedError``.
    atom_centric_path : {"auto", "direct", "sorted"}, default "auto"
        Forwarded to :func:`query_cell_list`.  Explicit ``"direct"`` uses the
        direct-reads (gather-skipping) kernel on the plain full-fill,
        ``graph_mode="none"`` path; ``"auto"`` resolves to ``"sorted"`` on JAX
        (perf-only divergence from Torch).
    graph_mode : {"none", "warp"}, default="none"
        Execution mode for the underlying Warp launches. ``"warp"`` is
        intended for jitted call sites that donate and reuse the optional
        cell-list caches and output buffers.

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
        If ``return_neighbor_list=False``: ``neighbor_matrix_shifts`` with shape
        (total_atoms, max_neighbors, 3), dtype int32.
        If ``return_neighbor_list=True``: ``neighbor_list_shifts`` with shape
        (num_pairs, 3), dtype int32.

    See Also
    --------
    build_cell_list : Build cell list separately
    query_cell_list : Query cell list separately
    naive_neighbor_list : Naive O(N^2) method
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
    if pair_fn is not None and pair_params is None:
        raise ValueError(
            "pair_fn requires pair_params (a per-atom (n_atoms, K) parameter array).",
        )

    graph_mode = _validate_graph_mode(graph_mode)
    if has_pair_outputs and graph_mode != "none":
        raise NotImplementedError(
            "return_distances / return_vectors are only supported with "
            "graph_mode='none' in the JAX cell_list binding.  CUDA-graph "
            "capture of the pair-output kernel is a follow-up.",
        )
    if half_fill and graph_mode != "none":
        raise NotImplementedError(
            "half_fill=True is only supported with graph_mode='none' in the "
            "JAX cell_list binding; CUDA-graph capture of the half-fill kernel "
            "is a follow-up.",
        )
    # Validate the sub-strategy options up front.  ``atom_centric_path`` is
    # forwarded to ``query_cell_list`` (explicit "direct" branches to the
    # gather-skipping kernel there).  ``strategy`` is forwarded to ``query_cell_list``,
    # which owns the host-read ``n_outer`` + launch-safety guards.  Two guards
    # must live HERE because the ``graph_mode="warp"`` branch and the
    # pair-output branch below both bypass ``query_cell_list`` entirely.
    _validate_atom_centric_path(atom_centric_path)
    if strategy not in {"auto", "atom_centric", "pair_centric"}:
        raise ValueError(
            f"strategy must be 'auto' | 'atom_centric' | 'pair_centric', "
            f"got {strategy!r}",
        )
    if strategy == "pair_centric" and _is_cpu_array(positions):
        # Pair-centric kernels use CUDA block scheduling.  Raise early here
        # (before build_cell_list) for a clean message, mirroring Torch's CPU
        # guard; ``strategy="auto"`` resolves to atom_centric on CPU.
        raise ValueError(
            "strategy='pair_centric' is not supported on CPU "
            "(kernels use CUDA block scheduling).  Pass 'auto' or "
            "'atom_centric' instead.",
        )
    if strategy == "pair_centric" and graph_mode == "warp":
        raise NotImplementedError(
            "strategy='pair_centric' is not supported with graph_mode='warp' "
            "(the launch dim is baked from a host-read n_outer; CUDA-graph "
            "replay across a changed radius is unsafe).  Use graph_mode='none' "
            "for pair-centric, or strategy='atom_centric'.",
        )
    if strategy == "pair_centric" and target_indices is not None:
        # The pair-centric kernel yields an identical pair set to atom-centric,
        # so partial neighbor lists are fully covered by the atom-centric path;
        # the compact-row ``target_indices`` + pair-centric block scheduling
        # combination is not wired (no capability gap -- use atom_centric).
        raise NotImplementedError(
            "strategy='pair_centric' with target_indices (partial neighbor "
            "lists) is not wired through the JAX cell_list binding.  Use "
            "strategy='atom_centric' (or 'auto') for identical results.",
        )

    # When pair outputs are requested, keep the LIVE positions and cell for
    # the autograd primitive's backward (reconstruction needs live tensors),
    # and use stop_gradient'd copies for the warp-kernel side of the
    # forward.  The warp ``jax_kernel`` callables are registered with
    # ``enable_backward=False``; calling them inside ``jax.grad`` requires
    # detached inputs.
    positions_for_grad = positions
    cell_for_grad = cell
    if has_pair_outputs:
        positions = jax.lax.stop_gradient(positions)
        cell = jax.lax.stop_gradient(cell)

    if cell is None:
        cell = jnp.eye(3, dtype=jnp.float32)[jnp.newaxis, :, :]
    if pbc is None:
        pbc = jnp.ones((1, 3), dtype=jnp.bool_)

    if cell.ndim == 2:
        cell = cell[jnp.newaxis, :, :]
    if pbc.ndim == 1:
        pbc = pbc[jnp.newaxis, :]

    # Partial (``target_indices``) path: the output matrix carries ``num_targets``
    # compact rows (row ``r`` -> atom ``target_indices[r]``), not ``total_atoms``.
    # ``num_rows`` drives the auto-allocated output-buffer shapes below and the
    # kernel launch dim in the forward.  ``target_indices`` always routes through
    # ``has_pair_outputs`` (see ``_has_partial_or_pair_outputs``).
    if target_indices is not None:
        target_indices = jnp.asarray(target_indices, dtype=jnp.int32)
        num_rows = int(target_indices.shape[0])
    else:
        num_rows = positions.shape[0]

    if max_neighbors is None and (
        neighbor_matrix is None
        or neighbor_matrix_shifts is None
        or num_neighbors is None
    ):
        max_neighbors = estimate_max_neighbors(cutoff)

    if max_total_cells is None:
        max_total_cells, _, neighbor_search_radius_est = estimate_cell_list_sizes(
            positions, cell, cutoff, pbc
        )
        if neighbor_search_radius is None:
            neighbor_search_radius = neighbor_search_radius_est
    elif neighbor_search_radius is None:
        neighbor_search_radius = jnp.ones(3, dtype=jnp.int32)

    if cells_per_dimension is None:
        cells_per_dimension = jnp.ones(3, dtype=jnp.int32)
    if atom_periodic_shifts is None:
        atom_periodic_shifts = jnp.zeros((positions.shape[0], 3), dtype=jnp.int32)
    if atom_to_cell_mapping is None:
        atom_to_cell_mapping = jnp.zeros((positions.shape[0], 3), dtype=jnp.int32)
    if atoms_per_cell_count is None:
        atoms_per_cell_count = jnp.zeros(max_total_cells, dtype=jnp.int32)
    elif graph_mode == "none":
        atoms_per_cell_count = atoms_per_cell_count.at[:].set(jnp.int32(0))
    if cell_atom_start_indices is None:
        cell_atom_start_indices = jnp.zeros(max_total_cells, dtype=jnp.int32)
    if cell_atom_list is None:
        cell_atom_list = jnp.zeros(positions.shape[0], dtype=jnp.int32)
    if neighbor_matrix is None:
        neighbor_matrix = jnp.full(
            (num_rows, max_neighbors),
            positions.shape[0],
            dtype=jnp.int32,
        )
    elif graph_mode == "none":
        neighbor_matrix = neighbor_matrix.at[:].set(jnp.int32(positions.shape[0]))
    if neighbor_matrix_shifts is None:
        neighbor_matrix_shifts = jnp.zeros(
            (num_rows, max_neighbors, 3),
            dtype=jnp.int32,
        )
    elif graph_mode == "none":
        neighbor_matrix_shifts = neighbor_matrix_shifts.at[:].set(jnp.int32(0))
    if num_neighbors is None:
        num_neighbors = jnp.zeros(num_rows, dtype=jnp.int32)
    elif graph_mode == "none":
        num_neighbors = num_neighbors.at[:].set(jnp.int32(0))

    if positions.dtype != jnp.float64:
        positions = positions.astype(jnp.float32)
    if cell.dtype != positions.dtype:
        cell = cell.astype(positions.dtype)
    pbc_1d = pbc.squeeze() if pbc.ndim == 2 else pbc
    pbc_bool = pbc_1d.astype(jnp.bool_)

    if graph_mode == "warp":
        graph_cell_list = (
            _jax_graph_cell_list_f64
            if positions.dtype == jnp.float64
            else _jax_graph_cell_list_f32
        )
        sorted_positions = jnp.zeros((positions.shape[0], 3), dtype=positions.dtype)
        sorted_atom_periodic_shifts = jnp.zeros(
            (positions.shape[0], 3), dtype=jnp.int32
        )
        always_true_rebuild_flags = jnp.ones((1,), dtype=jnp.bool_)
        (
            cells_per_dimension,
            atom_periodic_shifts,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
            sorted_positions,
            sorted_atom_periodic_shifts,
            neighbor_matrix,
            neighbor_matrix_shifts,
            num_neighbors,
        ) = graph_cell_list(
            positions,
            cell,
            pbc_bool,
            neighbor_search_radius,
            cells_per_dimension,
            atom_periodic_shifts,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
            sorted_positions,
            sorted_atom_periodic_shifts,
            always_true_rebuild_flags,
            neighbor_matrix,
            neighbor_matrix_shifts,
            num_neighbors,
            float(cutoff),
            int(positions.shape[0]),
        )
    else:
        (
            cells_per_dimension,
            atom_periodic_shifts,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
            neighbor_search_radius,
        ) = build_cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            cells_per_dimension=cells_per_dimension,
            neighbor_search_radius=neighbor_search_radius,
            atom_periodic_shifts=atom_periodic_shifts,
            atom_to_cell_mapping=atom_to_cell_mapping,
            atoms_per_cell_count=atoms_per_cell_count,
            cell_atom_start_indices=cell_atom_start_indices,
            cell_atom_list=cell_atom_list,
            max_total_cells=max_total_cells,
            graph_mode="none",
        )

        if has_pair_outputs:
            # Autograd path.  All warp kernels above ran with stop_gradient'd
            # positions/cell.  Here we hand off to the autograd primitive,
            # which gets the LIVE ``positions_for_grad`` / ``cell_for_grad``
            # so the backward can reconstruct ``r`` for the analytical
            # gradient.
            # Per-pair buffers carry ``num_rows`` rows (``num_targets`` for the
            # partial path, else ``total_atoms``), matching the compact output
            # matrix.
            if return_distances and neighbor_distances is None:
                neighbor_distances = jnp.zeros(
                    (num_rows, max_neighbors),
                    dtype=positions.dtype,
                )
            if return_vectors and neighbor_vectors is None:
                neighbor_vectors = jnp.zeros(
                    (num_rows, max_neighbors, 3),
                    dtype=positions.dtype,
                )
            # The autograd primitive needs both buffers populated even when
            # only one flag is set; allocate dummies for the unused one.
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

            # Pair-centric pair-output strategy (EXPLICIT only; "auto" resolves
            # to atom_centric here so we skip the parallelism-sufficiency host
            # reads).  Host-read the per-axis radius to size the block-scheduled
            # launch -- a device->host sync that is legal eagerly but illegal
            # under jax.jit with a traced ``neighbor_search_radius`` (the
            # pair-output path is eager-on-cutoff regardless).
            pc_strategy = "atom_centric"
            pc_n_outer = 0
            if strategy == "pair_centric":
                try:
                    Rx = int(neighbor_search_radius[0])
                    Ry = int(neighbor_search_radius[1])
                    Rz = int(neighbor_search_radius[2])
                except (
                    jax.errors.ConcretizationTypeError,
                    jax.errors.TracerIntegerConversionError,
                ) as exc:
                    raise ValueError(
                        "strategy='pair_centric' needs a concrete "
                        "neighbor_search_radius to size its launch grid (n_outer "
                        "is host-read).  Compute the cell-list sizing outside "
                        "jax.jit and pass a concrete neighbor_search_radius, or "
                        "use strategy='atom_centric'.",
                    ) from exc
                # JAX cell_list is full-fill (half_fill+pair_centric raised above).
                pc_n_outer = compute_batch_pair_centric_n_outer((Rx, Ry, Rz), False)
                total_cells = int(atoms_per_cell_count.shape[0])
                if not is_pair_centric_launch_safe(total_cells, pc_n_outer):
                    _raise_unsafe_pair_centric_launch(total_cells, pc_n_outer)
                pc_strategy = "pair_centric"

            forward_kwargs = {
                "pbc_bool": (pbc.squeeze() if pbc.ndim == 2 else pbc).astype(jnp.bool_),
                "cells_per_dimension": cells_per_dimension,
                "atom_periodic_shifts": atom_periodic_shifts,
                "atom_to_cell_mapping": atom_to_cell_mapping,
                "atoms_per_cell_count": atoms_per_cell_count,
                "cell_atom_start_indices": cell_atom_start_indices,
                "cell_atom_list": cell_atom_list,
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
                "n_outer": pc_n_outer,
                "half_fill": bool(half_fill),
            }
            route_out = _route_pair_outputs(
                positions_for_grad,
                cell_for_grad,
                _cell_list_pair_outputs_forward,
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
                # partial (``target_indices``) path that row is the COMPACT row
                # in ``[0, num_targets)`` -- NOT the atom index -- mirroring the
                # torch binding exactly (the matrix contract is already "row r ->
                # atom target_indices[r]"; COO inherits the same compact-row
                # contract).  Callers map back via ``target_indices``.
                nl, nptr, nl_shifts = get_neighbor_list_from_neighbor_matrix(
                    nm_out,
                    num_neighbors=nn_out,
                    neighbor_shift_matrix=shifts_out,
                    fill_value=positions.shape[0],
                )
                base = (nl, nptr, nl_shifts)
                # Repack per-pair geometry (and pair_fn outputs) into the same COO
                # order as ``nl`` so they index-align with the neighbor list.
                # Eager-only, like the index conversion.
                active = nm_out != positions.shape[0]
                distances_out, vectors_out = coo_pack_pair_geometry(
                    active, distances_out, vectors_out
                )
                if pair_fn is not None:
                    pe_out, pf_out = coo_pack_pair_geometry(active, pe_out, pf_out)
            else:
                if fill_value is not None and int(fill_value) != positions.shape[0]:
                    # Match the matrix-padding contract: real indices are
                    # < total_atoms, so remap only the unfilled tail.
                    nm_out = jnp.where(
                        nm_out == positions.shape[0],
                        jnp.int32(fill_value),
                        nm_out,
                    )
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

        neighbor_matrix, num_neighbors, neighbor_matrix_shifts = query_cell_list(
            positions=positions,
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
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            num_neighbors=num_neighbors,
            graph_mode="none",
            half_fill=half_fill,
            strategy=strategy,
            atom_centric_path=atom_centric_path,
            sorted_positions=sorted_positions,
            sorted_atom_periodic_shifts=sorted_atom_periodic_shifts,
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
        return (
            neighbor_list,
            neighbor_ptr,
            neighbor_list_shifts,
        )
    else:
        if fill_value is not None and int(fill_value) != positions.shape[0]:
            # The kernel pads unfilled matrix entries with ``total_atoms``; real
            # neighbor indices are < total_atoms, so remap only the tail.
            neighbor_matrix = jnp.where(
                neighbor_matrix == positions.shape[0],
                jnp.int32(fill_value),
                neighbor_matrix,
            )
        return (
            neighbor_matrix,
            num_neighbors,
            neighbor_matrix_shifts,
        )
