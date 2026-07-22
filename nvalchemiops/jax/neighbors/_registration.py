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

"""Common registration helpers for JAX neighbor bindings."""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import jax.numpy as jnp
import warp as wp
from warp.jax_experimental import GraphMode, jax_callable, jax_kernel

from nvalchemiops.neighbors.cell_list import (
    get_build_cell_list_kernel,
    get_cell_list_cells_per_system_kernel,
    get_cell_list_gather_kernel,
    get_query_cell_list_kernel,
)
from nvalchemiops.neighbors.naive import (
    get_naive_neighbor_matrix_dual_cutoff_kernel,
    get_naive_neighbor_matrix_kernel,
)
from nvalchemiops.neighbors.neighbor_utils import (
    get_gather_positions_and_shifts_kernel,
)

__all__: list[str] = []

_SINGLE_NO_PBC_OUTPUTS = ("neighbor_matrix1", "num_neighbors1")
_SINGLE_PBC_OUTPUTS = (
    "neighbor_matrix1",
    "neighbor_matrix_shifts1",
    "num_neighbors1",
)
_DUAL_NO_PBC_OUTPUTS = (
    "neighbor_matrix1",
    "num_neighbors1",
    "neighbor_matrix2",
    "num_neighbors2",
)
_DUAL_PBC_OUTPUTS = (
    "neighbor_matrix1",
    "neighbor_matrix_shifts1",
    "num_neighbors1",
    "neighbor_matrix2",
    "neighbor_matrix_shifts2",
    "num_neighbors2",
)


@dataclass(frozen=True)
class _JaxOutputSchema:
    """Describe the ordered in-place outputs of a JAX Warp registration."""

    in_out_argnames: tuple[str, ...]

    @property
    def num_outputs(self) -> int:
        """Return the number of in-place outputs."""
        return len(self.in_out_argnames)


@dataclass(frozen=True)
class _NaiveJaxKernelSpec:
    """Describe one supported direct naive JAX kernel registration."""

    operation: Literal["single_cutoff", "dual_cutoff"]
    wp_dtype: type
    batched: bool
    pbc_mode: Literal["none", "wrap_on_entry", "prewrapped"]
    selective: bool
    partial: bool
    half_fill: bool
    return_vectors: bool
    return_distances: bool
    pair_fn: Any | None


@dataclass(frozen=True)
class _CellListBuildJaxSpec:
    """Describe one supported direct cell-list build JAX registration."""

    stage: Literal[
        "construct_bin_size",
        "count_atoms",
        "bin_atoms",
        "gather",
        "cells_per_system",
    ]
    wp_dtype: type
    batched: bool


@dataclass(frozen=True)
class _CellListQueryJaxSpec:
    """Describe one supported direct cell-list query JAX registration."""

    wp_dtype: type
    batched: bool
    selective: bool
    partial: bool
    half_fill: bool
    return_vectors: bool
    return_distances: bool
    pair_fn: Any | None
    atom_centric_path: Literal["sorted", "direct"]


@dataclass(frozen=True)
class _ClusterTileBuildJaxSpec:
    """Describe one supported cluster-tile build callback registration."""

    batched: bool
    segmented: bool
    selective: bool


@dataclass(frozen=True)
class _ClusterTileQueryJaxSpec:
    """Describe one supported cluster-tile query callback registration."""

    batched: bool
    output_format: Literal["matrix", "coo"]
    tile_segmented: bool
    coo_segmented: bool
    selective: bool
    dual_cutoff: bool
    return_vectors: bool
    return_distances: bool
    pair_fn: Any | None


def _register_jax_kernel(kernel: Any, output_schema: _JaxOutputSchema) -> Any:
    """Register a Warp kernel using a shared output schema."""
    return jax_kernel(
        kernel,
        num_outputs=output_schema.num_outputs,
        in_out_argnames=list(output_schema.in_out_argnames),
        enable_backward=False,
    )


def _register_jax_callable(
    callable_obj: Callable[..., Any],
    output_schema: _JaxOutputSchema,
    *,
    graph_mode: GraphMode,
) -> Any:
    """Register a Warp callable using a shared output schema."""
    return jax_callable(
        callable_obj,
        num_outputs=output_schema.num_outputs,
        in_out_argnames=list(output_schema.in_out_argnames),
        graph_mode=graph_mode,
    )


def _validate_spec(spec: _NaiveJaxKernelSpec) -> None:
    """Reject unsupported direct naive JAX registration combinations."""
    if spec.operation not in {"single_cutoff", "dual_cutoff"}:
        raise ValueError(f"Unsupported naive operation: {spec.operation!r}.")
    if spec.wp_dtype not in {wp.float32, wp.float64}:
        raise ValueError(f"Unsupported naive Warp dtype: {spec.wp_dtype!r}.")
    if spec.pbc_mode not in {"none", "wrap_on_entry", "prewrapped"}:
        raise ValueError(f"Unsupported naive PBC mode: {spec.pbc_mode!r}.")
    if spec.return_vectors != spec.return_distances:
        raise ValueError(
            "Direct naive registrations require return_vectors and "
            "return_distances together.",
        )
    if spec.pair_fn is not None and not spec.return_vectors:
        raise ValueError("pair_fn requires direct pair-geometry outputs.")
    if spec.partial and (
        spec.operation != "single_cutoff" or not spec.return_vectors or spec.selective
    ):
        raise ValueError(
            "partial direct registrations require single-cutoff non-selective "
            "pair geometry.",
        )
    if spec.operation == "dual_cutoff" and (
        spec.return_vectors
        or spec.pair_fn is not None
        or spec.partial
        or spec.half_fill
    ):
        raise ValueError(
            "Dual-cutoff direct registrations do not support geometry, pair_fn, "
            "partial rows, or half_fill.",
        )


def _output_schema(spec: _NaiveJaxKernelSpec) -> _JaxOutputSchema:
    """Return the ordered direct JAX output schema for ``spec``."""
    if spec.operation == "dual_cutoff":
        outputs = _DUAL_NO_PBC_OUTPUTS if spec.pbc_mode == "none" else _DUAL_PBC_OUTPUTS
    else:
        outputs = (
            _SINGLE_NO_PBC_OUTPUTS if spec.pbc_mode == "none" else _SINGLE_PBC_OUTPUTS
        )
        if spec.return_vectors:
            outputs += ("neighbor_vectors", "neighbor_distances")
        if spec.pair_fn is not None:
            outputs += ("pair_energies", "pair_forces")
    return _JaxOutputSchema(outputs)


def _validate_cell_list_build_spec(spec: _CellListBuildJaxSpec) -> None:
    """Reject unsupported direct cell-list build registrations."""
    if spec.wp_dtype not in {wp.float32, wp.float64}:
        raise ValueError(f"Unsupported cell-list Warp dtype: {spec.wp_dtype!r}.")
    if spec.stage not in {
        "construct_bin_size",
        "count_atoms",
        "bin_atoms",
        "gather",
        "cells_per_system",
    }:
        raise ValueError(f"Unsupported cell-list build stage: {spec.stage!r}.")
    if spec.stage == "cells_per_system" and not spec.batched:
        raise ValueError("cells_per_system is only registered for batch cell lists.")


def _cell_list_build_output_schema(spec: _CellListBuildJaxSpec) -> _JaxOutputSchema:
    """Return the ordered in-place output schema for a build stage."""
    if spec.stage == "construct_bin_size":
        return _JaxOutputSchema(
            (
                "cells_per_dimension_batch"
                if spec.batched
                else "cells_per_dimension_single",
            ),
        )
    if spec.stage == "count_atoms":
        return _JaxOutputSchema(("atoms_per_cell_count", "atom_periodic_shifts"))
    if spec.stage == "bin_atoms":
        return _JaxOutputSchema(
            ("atom_to_cell_mapping", "atoms_per_cell_count", "cell_atom_list"),
        )
    if spec.stage == "gather":
        return _JaxOutputSchema(
            ("sorted_positions", "sorted_shifts")
            if spec.batched
            else ("dst_pos", "dst_shifts"),
        )
    return _JaxOutputSchema(("cells_per_system",))


@functools.cache
def _get_cell_list_build_jax_kernel(spec: _CellListBuildJaxSpec) -> Any:
    """Return a cached direct JAX registration for a cell-list build stage."""
    _validate_cell_list_build_spec(spec)
    if spec.stage == "cells_per_system":
        kernel = get_cell_list_cells_per_system_kernel()
    elif spec.stage == "gather":
        kernel = (
            get_cell_list_gather_kernel(spec.wp_dtype)
            if spec.batched
            else get_gather_positions_and_shifts_kernel(spec.wp_dtype)
        )
    else:
        kernel = get_build_cell_list_kernel(
            spec.stage,
            spec.wp_dtype,
            batched=spec.batched,
        )
    return _register_jax_kernel(kernel, _cell_list_build_output_schema(spec))


def _validate_cell_list_query_spec(spec: _CellListQueryJaxSpec) -> None:
    """Reject unsupported direct cell-list query registrations."""
    if spec.wp_dtype not in {wp.float32, wp.float64}:
        raise ValueError(f"Unsupported cell-list Warp dtype: {spec.wp_dtype!r}.")
    if spec.atom_centric_path not in {"sorted", "direct"}:
        raise ValueError(
            f"Unsupported atom-centric cell-list path: {spec.atom_centric_path!r}.",
        )
    if spec.batched and spec.atom_centric_path == "direct":
        raise ValueError("The direct atom-centric path is only available unbatched.")
    if spec.return_vectors != spec.return_distances:
        raise ValueError(
            "Direct cell-list registrations require return_vectors and "
            "return_distances together.",
        )
    if spec.pair_fn is not None and not spec.return_vectors:
        raise ValueError("pair_fn requires direct pair-geometry outputs.")


def _cell_list_query_output_schema(spec: _CellListQueryJaxSpec) -> _JaxOutputSchema:
    """Return the ordered in-place output schema for a query stage."""
    outputs = (
        "neighbor_matrix",
        "neighbor_matrix_shifts",
        "num_neighbors",
    )
    if spec.return_vectors:
        outputs += ("neighbor_vectors", "neighbor_distances")
    if spec.pair_fn is not None:
        outputs += ("pair_energies", "pair_forces")
    return _JaxOutputSchema(outputs)


@functools.cache
def _get_cell_list_query_jax_kernel(spec: _CellListQueryJaxSpec) -> Any:
    """Return a cached direct JAX registration for a cell-list query."""
    _validate_cell_list_query_spec(spec)
    kernel = get_query_cell_list_kernel(
        spec.wp_dtype,
        strategy="atom_centric",
        batched=spec.batched,
        selective=spec.selective,
        partial=spec.partial,
        half_fill=spec.half_fill,
        return_vectors=spec.return_vectors,
        return_distances=spec.return_distances,
        pair_fn=spec.pair_fn,
        atom_centric_path=spec.atom_centric_path,
    )
    return _register_jax_kernel(kernel, _cell_list_query_output_schema(spec))


@functools.cache
def _get_naive_jax_kernel(spec: _NaiveJaxKernelSpec) -> Any:
    """Return a cached direct JAX registration for a supported naive spec."""
    _validate_spec(spec)
    if spec.operation == "dual_cutoff":
        kernel = get_naive_neighbor_matrix_dual_cutoff_kernel(
            spec.wp_dtype,
            pbc_mode=spec.pbc_mode,
            batched=spec.batched,
            selective=spec.selective,
        )
    else:
        kernel = get_naive_neighbor_matrix_kernel(
            spec.wp_dtype,
            pbc_mode=spec.pbc_mode,
            batched=spec.batched,
            selective=spec.selective,
            partial=spec.partial,
            half_fill=spec.half_fill,
            return_vectors=spec.return_vectors,
            return_distances=spec.return_distances,
            pair_fn=spec.pair_fn,
        )
    return _register_jax_kernel(kernel, _output_schema(spec))


def _validate_cluster_tile_build_spec(spec: _ClusterTileBuildJaxSpec) -> None:
    """Reject unsupported cluster-tile build callback combinations."""
    if not spec.batched and spec.segmented:
        raise ValueError("Single-system cluster-tile builds cannot be segmented.")
    if spec.batched and spec.selective and not spec.segmented:
        raise ValueError(
            "Selective batched cluster-tile builds must be segmented.",
        )


def _cluster_tile_build_output_schema(
    spec: _ClusterTileBuildJaxSpec,
) -> _JaxOutputSchema:
    """Return the ordered in-place ABI for a cluster-tile build callback."""
    outputs = (
        "group_ctr_x",
        "group_ctr_y",
        "group_ctr_z",
        "group_ext_x",
        "group_ext_y",
        "group_ext_z",
        "num_tiles",
    )
    if spec.batched and spec.selective:
        outputs += ("tile_counts",)
    outputs += ("tile_row_group", "tile_col_group")
    if spec.batched:
        outputs += ("tile_system",)
    return _JaxOutputSchema(outputs)


@functools.cache
def _get_cluster_tile_build_jax_callable(
    spec: _ClusterTileBuildJaxSpec,
    callback: Callable[..., Any],
) -> Any:
    """Return a cached WARP graph callback for a cluster-tile build."""
    _validate_cluster_tile_build_spec(spec)
    return _register_jax_callable(
        callback,
        _cluster_tile_build_output_schema(spec),
        graph_mode=GraphMode.WARP,
    )


def _validate_cluster_tile_query_spec(spec: _ClusterTileQueryJaxSpec) -> None:
    """Reject unsupported cluster-tile query callback combinations."""
    if spec.output_format not in {"matrix", "coo"}:
        raise ValueError(
            f"Unsupported cluster-tile output format: {spec.output_format!r}."
        )
    if not spec.batched and spec.tile_segmented:
        raise ValueError("Single-system cluster-tile queries cannot tile-segment.")
    if spec.output_format == "matrix":
        if spec.coo_segmented:
            raise ValueError("Matrix cluster-tile queries cannot use COO segments.")
        if spec.return_vectors != spec.return_distances:
            raise ValueError(
                "Cluster-tile geometry requires vectors and distances together.",
            )
        if (spec.return_vectors or spec.pair_fn is not None) and (
            spec.dual_cutoff or spec.selective
        ):
            raise ValueError(
                "Cluster-tile geometry and pair_fn require non-dual, non-selective "
                "matrix queries.",
            )
        return
    if (
        spec.dual_cutoff
        or spec.return_vectors
        or spec.return_distances
        or spec.pair_fn is not None
    ):
        raise ValueError("COO cluster-tile callbacks support topology output only.")
    if spec.coo_segmented and not spec.selective:
        raise ValueError(
            "Segmented COO requires selective cluster-tile queries.",
        )
    if spec.selective and not spec.coo_segmented:
        raise ValueError("Selective COO requires segmented COO output.")


def _cluster_tile_query_output_schema(
    spec: _ClusterTileQueryJaxSpec,
) -> _JaxOutputSchema:
    """Return the ordered in-place ABI for a cluster-tile query callback."""
    if spec.output_format == "coo":
        outputs = ("pair_counter",)
        if spec.coo_segmented:
            outputs += ("pair_counts",)
        return _JaxOutputSchema(outputs + ("coo_list", "coo_shifts"))

    outputs = ("neighbor_matrix", "num_neighbors", "neighbor_matrix_shifts")
    if spec.dual_cutoff:
        outputs += (
            "neighbor_matrix2",
            "num_neighbors2",
            "neighbor_matrix_shifts2",
        )
    if spec.return_vectors:
        outputs += ("neighbor_vectors", "neighbor_distances")
    if spec.pair_fn is not None:
        outputs += ("pair_energies", "pair_forces")
    return _JaxOutputSchema(outputs)


@functools.cache
def _get_cluster_tile_query_jax_callable(
    spec: _ClusterTileQueryJaxSpec,
    callback: Callable[..., Any],
) -> Any:
    """Return a cached WARP graph callback for a cluster-tile query."""
    _validate_cluster_tile_query_spec(spec)
    return _register_jax_callable(
        callback,
        _cluster_tile_query_output_schema(spec),
        graph_mode=GraphMode.WARP,
    )


class _LazyDtypeRegistrations:
    """Lazily construct JAX wrappers for declared JAX-to-Warp dtype mappings."""

    def __init__(
        self,
        factory: Callable[[Any], Any],
        dtype_map: Mapping[object, Any],
        *,
        cache_key: Callable[[Any], object] | None = None,
    ) -> None:
        self._factory = factory
        self._dtype_map = {
            jnp.dtype(jax_dtype): wp_dtype for jax_dtype, wp_dtype in dtype_map.items()
        }
        self._cache_key = cache_key or (lambda wp_dtype: wp_dtype)
        self._cache: dict[object, Any] = {}

    def __contains__(self, jax_dtype: object) -> bool:
        """Return whether a dtype has a supported Warp registration."""
        try:
            return jnp.dtype(jax_dtype) in self._dtype_map
        except (TypeError, ValueError):
            return False

    def __getitem__(self, jax_dtype: object) -> Any:
        """Return the lazily registered wrapper for a supported JAX dtype."""
        try:
            normalized_dtype = jnp.dtype(jax_dtype)
        except (TypeError, ValueError) as error:
            raise KeyError(jax_dtype) from error

        wp_dtype = self._dtype_map[normalized_dtype]
        cache_key = self._cache_key(wp_dtype)
        if cache_key not in self._cache:
            self._cache[cache_key] = self._factory(wp_dtype)
        return self._cache[cache_key]
