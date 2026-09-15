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
from warp import JaxCallableGraphMode, jax_callable, jax_kernel

from nvalchemiops.jax.neighbors._cluster_tile_preload import (
    _preload_cluster_tile_build_kernel,
    _preload_cluster_tile_coo_kernel,
    _preload_cluster_tile_query_kernel,
    _validate_cluster_tile_build_options,
    _validate_cluster_tile_coo_options,
    _validate_cluster_tile_matrix_options,
)
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

_Outputs = tuple[str, ...]

_NAIVE_DTYPE_MAP = {jnp.float32: wp.float32, jnp.float64: wp.float64}
_CELL_LIST_DTYPE_MAP = {jnp.float32: wp.float32, jnp.float64: wp.float64}
_CELLS_PER_SYSTEM_CACHE_KEY = "cells_per_system"

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


def _register_jax_kernel(kernel: Any, outputs: _Outputs) -> Any:
    """Register a Warp kernel using a shared output tuple.

    Parameters
    ----------
    kernel : Any
        Warp kernel to expose to JAX.
    outputs : tuple[str, ...]
        Ordered names of the kernel's in-place output arguments.

    Returns
    -------
    Any
        The registered JAX kernel wrapper.
    """
    return jax_kernel(
        kernel,
        num_outputs=len(outputs),
        in_out_argnames=list(outputs),
        enable_backward=False,
    )


def _register_jax_callable(
    callable_obj: Callable[..., Any],
    outputs: _Outputs,
    *,
    graph_mode: JaxCallableGraphMode,
) -> Any:
    """Register a Warp callable using a shared output tuple.

    Parameters
    ----------
    callable_obj : Callable[..., Any]
        Warp callable to expose to JAX.
    outputs : tuple[str, ...]
        Ordered names of the callable's in-place output arguments.
    graph_mode : JaxCallableGraphMode
        Execution mode used by Warp graph capture.

    Returns
    -------
    Any
        The registered JAX callable wrapper.
    """
    return jax_callable(
        callable_obj,
        num_outputs=len(outputs),
        in_out_argnames=list(outputs),
        graph_mode=graph_mode,
    )


@dataclass(frozen=True)
class _GraphRegistration:
    """Bundle one Warp graph callable with its matching preload operation.

    Parameters
    ----------
    callable : Any
        Registered JAX callable for the graph operation.
    preload : Callable[..., None]
        Callable that loads the operation's Warp modules before graph capture.
    """

    callable: Any
    preload: Callable[..., None]


def _cluster_tile_build_outputs(
    *,
    batched: bool,
    selective: bool,
) -> _Outputs:
    """Return the ordered in-place ABI for a cluster-tile build callback.

    Parameters
    ----------
    batched : bool
        Whether the callback operates on multiple systems.
    selective : bool
        Whether the callback produces per-tile counts.

    Returns
    -------
    tuple[str, ...]
        Ordered names of the callback's in-place output arguments.
    """
    outputs: _Outputs = (
        "group_ctr_x",
        "group_ctr_y",
        "group_ctr_z",
        "group_ext_x",
        "group_ext_y",
        "group_ext_z",
        "num_tiles",
    )
    if batched and selective:
        outputs += ("tile_counts",)
    outputs += ("tile_row_group", "tile_col_group")
    if batched:
        outputs += ("tile_system",)
    return outputs


def _cluster_tile_matrix_outputs(
    *,
    dual_cutoff: bool,
    geometry: bool,
    pair_fn: Any | None,
) -> _Outputs:
    """Return the ordered in-place ABI for a cluster-tile matrix callback.

    Parameters
    ----------
    dual_cutoff : bool
        Whether the callback emits a second neighbor matrix.
    geometry : bool
        Whether the callback emits pair vectors and distances.
    pair_fn : Any or None
        Pair callback whose energy and force outputs are emitted when supplied.

    Returns
    -------
    tuple[str, ...]
        Ordered names of the callback's in-place output arguments.
    """
    outputs: _Outputs = (
        "neighbor_matrix",
        "num_neighbors",
        "neighbor_matrix_shifts",
    )
    if dual_cutoff:
        outputs += (
            "neighbor_matrix2",
            "num_neighbors2",
            "neighbor_matrix_shifts2",
        )
    if geometry:
        outputs += ("neighbor_vectors", "neighbor_distances")
    if pair_fn is not None:
        outputs += ("pair_energies", "pair_forces")
    return outputs


def _cluster_tile_coo_outputs(*, coo_segmented: bool) -> _Outputs:
    """Return the ordered in-place ABI for a cluster-tile COO callback.

    Parameters
    ----------
    coo_segmented : bool
        Whether the callback emits per-segment pair counts.

    Returns
    -------
    tuple[str, ...]
        Ordered names of the callback's in-place output arguments.
    """
    outputs: _Outputs = ("pair_counter",)
    if coo_segmented:
        outputs += ("pair_counts",)
    return outputs + ("coo_list", "coo_shifts")


def _cluster_tile_build_registration(
    callback: Callable[..., Any],
    *,
    batched: bool,
    segmented: bool,
    selective: bool,
) -> _GraphRegistration:
    """Build a Warp graph registration for a cluster-tile build callback.

    Parameters
    ----------
    callback : Callable[..., Any]
        Warp callback to register.
    batched : bool
        Whether the callback operates on multiple systems.
    segmented : bool
        Whether the tile storage is segmented by system.
    selective : bool
        Whether the callback operates on a selected subset of atoms.

    Returns
    -------
    _GraphRegistration
        Registered callback and its matching preload operation.
    """
    _validate_cluster_tile_build_options(
        batched=batched,
        segmented=segmented,
        selective=selective,
    )
    registered = _register_jax_callable(
        callback,
        _cluster_tile_build_outputs(batched=batched, selective=selective),
        graph_mode=JaxCallableGraphMode.WARP,
    )
    return _GraphRegistration(
        callable=registered,
        preload=_preload_cluster_tile_build_kernel,
    )


def _cluster_tile_matrix_registration(
    callback: Callable[..., Any],
    *,
    batched: bool,
    tile_segmented: bool = False,
    selective: bool = False,
    dual_cutoff: bool = False,
    geometry: bool = False,
    pair_fn: Any | None = None,
) -> _GraphRegistration:
    """Build a Warp graph registration for a cluster-tile matrix query.

    Parameters
    ----------
    callback : Callable[..., Any]
        Warp callback to register.
    batched, tile_segmented, selective, dual_cutoff, geometry : bool
        Static kernel specialization options.
    pair_fn : Any or None
        Optional pair callback used by the query specialization.

    Returns
    -------
    _GraphRegistration
        Registered callback and its matching preload operation.
    """
    _validate_cluster_tile_matrix_options(
        batched=batched,
        tile_segmented=tile_segmented,
        selective=selective,
        dual_cutoff=dual_cutoff,
        geometry=geometry,
        pair_fn=pair_fn,
    )
    registered = _register_jax_callable(
        callback,
        _cluster_tile_matrix_outputs(
            dual_cutoff=dual_cutoff,
            geometry=geometry,
            pair_fn=pair_fn,
        ),
        graph_mode=JaxCallableGraphMode.WARP,
    )
    preload = functools.partial(
        _preload_cluster_tile_query_kernel,
        batched=batched,
        tile_segmented=tile_segmented,
        selective=selective,
        dual_cutoff=dual_cutoff,
        geometry=geometry,
        pair_fn=pair_fn,
    )
    return _GraphRegistration(callable=registered, preload=preload)


def _cluster_tile_coo_registration(
    callback: Callable[..., Any],
    *,
    batched: bool,
    tile_segmented: bool = False,
    coo_segmented: bool = False,
    selective: bool = False,
) -> _GraphRegistration:
    """Build a Warp graph registration for a cluster-tile COO query.

    Parameters
    ----------
    callback : Callable[..., Any]
        Warp callback to register.
    batched, tile_segmented, coo_segmented, selective : bool
        Static kernel specialization options.

    Returns
    -------
    _GraphRegistration
        Registered callback and its matching preload operation.
    """
    _validate_cluster_tile_coo_options(
        batched=batched,
        tile_segmented=tile_segmented,
        coo_segmented=coo_segmented,
        selective=selective,
    )
    registered = _register_jax_callable(
        callback,
        _cluster_tile_coo_outputs(coo_segmented=coo_segmented),
        graph_mode=JaxCallableGraphMode.WARP,
    )
    preload = functools.partial(
        _preload_cluster_tile_coo_kernel,
        batched=batched,
        tile_segmented=tile_segmented,
        coo_segmented=coo_segmented,
        selective=selective,
    )
    return _GraphRegistration(callable=registered, preload=preload)


class _LazyJaxKernel:
    """Lazily construct JAX wrappers for declared JAX-to-Warp dtype mappings."""

    def __init__(
        self,
        build: Callable[[type], tuple[Any, _Outputs]],
        dtype_map: Mapping[object, type],
        *,
        cache_key: Callable[[type], object] | None = None,
    ) -> None:
        self._build = build
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

        if normalized_dtype not in self._dtype_map:
            raise KeyError(jax_dtype)

        wp_dtype = self._dtype_map[normalized_dtype]
        key = self._cache_key(wp_dtype)
        if key not in self._cache:
            kernel, outputs = self._build(wp_dtype)
            self._cache[key] = _register_jax_kernel(kernel, outputs)
        return self._cache[key]


def _naive_outputs(
    *,
    operation: Literal["single_cutoff", "dual_cutoff"],
    pbc_mode: Literal["none", "wrap_on_entry", "prewrapped"],
    geometry: bool,
    pair_fn: Any | None,
) -> _Outputs:
    """Return the ordered direct naive JAX output tuple for static options."""
    if operation == "dual_cutoff":
        outputs = _DUAL_NO_PBC_OUTPUTS if pbc_mode == "none" else _DUAL_PBC_OUTPUTS
    else:
        outputs = _SINGLE_NO_PBC_OUTPUTS if pbc_mode == "none" else _SINGLE_PBC_OUTPUTS
        if geometry:
            outputs += ("neighbor_vectors", "neighbor_distances")
        if pair_fn is not None:
            outputs += ("pair_energies", "pair_forces")
    return outputs


def _validate_lazy_naive_options(
    *,
    operation: Literal["single_cutoff", "dual_cutoff"],
    pbc_mode: Literal["none", "wrap_on_entry", "prewrapped"],
    selective: bool,
    partial: bool,
    half_fill: bool,
    geometry: bool,
    pair_fn: Any | None,
) -> None:
    """Reject unsupported direct naive JAX registration combinations."""
    if operation not in {"single_cutoff", "dual_cutoff"}:
        raise ValueError(f"Unsupported naive operation: {operation!r}.")
    if pbc_mode not in {"none", "wrap_on_entry", "prewrapped"}:
        raise ValueError(f"Unsupported naive PBC mode: {pbc_mode!r}.")
    if pair_fn is not None and not geometry:
        raise ValueError("pair_fn requires direct pair-geometry outputs.")
    if partial and (operation != "single_cutoff" or not geometry or selective):
        raise ValueError(
            "partial direct registrations require single-cutoff non-selective "
            "pair geometry.",
        )
    if operation == "dual_cutoff" and (
        geometry or pair_fn is not None or partial or half_fill
    ):
        raise ValueError(
            "Dual-cutoff direct registrations do not support geometry, pair_fn, "
            "partial rows, or half_fill.",
        )


def _lazy_naive_kernel(
    *,
    operation: Literal["single_cutoff", "dual_cutoff"],
    batched: bool,
    pbc_mode: Literal["none", "wrap_on_entry", "prewrapped"],
    selective: bool = False,
    partial: bool = False,
    half_fill: bool = False,
    geometry: bool = False,
    pair_fn: Any | None = None,
) -> _LazyJaxKernel:
    """Return a lazy direct naive JAX registration for static options."""

    def build(wp_dtype: type) -> tuple[Any, _Outputs]:
        _validate_lazy_naive_options(
            operation=operation,
            pbc_mode=pbc_mode,
            selective=selective,
            partial=partial,
            half_fill=half_fill,
            geometry=geometry,
            pair_fn=pair_fn,
        )
        if wp_dtype not in {wp.float32, wp.float64}:
            raise ValueError(f"Unsupported naive Warp dtype: {wp_dtype!r}.")
        if operation == "dual_cutoff":
            kernel = get_naive_neighbor_matrix_dual_cutoff_kernel(
                wp_dtype,
                pbc_mode=pbc_mode,
                batched=batched,
                selective=selective,
            )
        else:
            kernel = get_naive_neighbor_matrix_kernel(
                wp_dtype,
                pbc_mode=pbc_mode,
                batched=batched,
                selective=selective,
                partial=partial,
                half_fill=half_fill,
                return_vectors=geometry,
                return_distances=geometry,
                pair_fn=pair_fn,
            )
        outputs = _naive_outputs(
            operation=operation,
            pbc_mode=pbc_mode,
            geometry=geometry,
            pair_fn=pair_fn,
        )
        return kernel, outputs

    return _LazyJaxKernel(build, _NAIVE_DTYPE_MAP)


def _cell_list_build_outputs(
    stage: Literal[
        "construct_bin_size",
        "count_atoms",
        "bin_atoms",
        "gather",
        "cells_per_system",
    ],
    batched: bool,
) -> _Outputs:
    """Return the ordered in-place output tuple for a build stage."""
    if stage == "construct_bin_size":
        return (
            "cells_per_dimension_batch" if batched else "cells_per_dimension_single",
        )
    if stage == "count_atoms":
        return ("atoms_per_cell_count", "atom_periodic_shifts")
    if stage == "bin_atoms":
        return ("atom_to_cell_mapping", "atoms_per_cell_count", "cell_atom_list")
    if stage == "gather":
        return (
            ("sorted_positions", "sorted_shifts")
            if batched
            else ("dst_pos", "dst_shifts")
        )
    return ("cells_per_system",)


def _validate_lazy_cell_list_build_options(
    *,
    stage: Literal[
        "construct_bin_size",
        "count_atoms",
        "bin_atoms",
        "gather",
        "cells_per_system",
    ],
    batched: bool,
) -> None:
    """Reject unsupported direct cell-list build registrations."""
    if stage not in {
        "construct_bin_size",
        "count_atoms",
        "bin_atoms",
        "gather",
        "cells_per_system",
    }:
        raise ValueError(f"Unsupported cell-list build stage: {stage!r}.")
    if stage == "cells_per_system" and not batched:
        raise ValueError("cells_per_system is only registered for batch cell lists.")


def _lazy_cell_list_build_kernel(
    *,
    stage: Literal[
        "construct_bin_size",
        "count_atoms",
        "bin_atoms",
        "gather",
        "cells_per_system",
    ],
    batched: bool,
) -> _LazyJaxKernel:
    """Return a lazy direct cell-list build JAX registration."""
    cache_key = (
        (lambda wp_dtype: _CELLS_PER_SYSTEM_CACHE_KEY)
        if stage == "cells_per_system"
        else None
    )

    def build(wp_dtype: type) -> tuple[Any, _Outputs]:
        _validate_lazy_cell_list_build_options(stage=stage, batched=batched)
        if wp_dtype not in {wp.float32, wp.float64}:
            raise ValueError(f"Unsupported cell-list Warp dtype: {wp_dtype!r}.")
        if stage == "cells_per_system":
            kernel = get_cell_list_cells_per_system_kernel()
        elif stage == "gather":
            kernel = (
                get_cell_list_gather_kernel(wp_dtype)
                if batched
                else get_gather_positions_and_shifts_kernel(wp_dtype)
            )
        else:
            kernel = get_build_cell_list_kernel(
                stage,
                wp_dtype,
                batched=batched,
            )
        return kernel, _cell_list_build_outputs(stage, batched)

    return _LazyJaxKernel(
        build,
        _CELL_LIST_DTYPE_MAP,
        cache_key=cache_key,
    )


def _cell_list_query_outputs(
    *,
    geometry: bool,
    pair_fn: Any | None,
) -> _Outputs:
    """Return the ordered in-place output tuple for a query registration."""
    outputs: _Outputs = (
        "neighbor_matrix",
        "neighbor_matrix_shifts",
        "num_neighbors",
    )
    if geometry:
        outputs += ("neighbor_vectors", "neighbor_distances")
    if pair_fn is not None:
        outputs += ("pair_energies", "pair_forces")
    return outputs


def _validate_lazy_cell_list_query_options(
    *,
    batched: bool,
    geometry: bool,
    pair_fn: Any | None,
    atom_centric_path: Literal["sorted", "direct"],
) -> None:
    """Reject unsupported direct cell-list query registrations."""
    if atom_centric_path not in {"sorted", "direct"}:
        raise ValueError(
            f"Unsupported atom-centric cell-list path: {atom_centric_path!r}.",
        )
    if batched and atom_centric_path == "direct":
        raise ValueError("The direct atom-centric path is only available unbatched.")
    if pair_fn is not None and not geometry:
        raise ValueError("pair_fn requires direct pair-geometry outputs.")


def _lazy_cell_list_query_kernel(
    *,
    batched: bool,
    selective: bool = True,
    partial: bool = False,
    half_fill: bool = False,
    geometry: bool = False,
    pair_fn: Any | None = None,
    atom_centric_path: Literal["sorted", "direct"] = "sorted",
) -> _LazyJaxKernel:
    """Return a lazy direct cell-list query JAX registration."""

    def build(wp_dtype: type) -> tuple[Any, _Outputs]:
        _validate_lazy_cell_list_query_options(
            batched=batched,
            geometry=geometry,
            pair_fn=pair_fn,
            atom_centric_path=atom_centric_path,
        )
        if wp_dtype not in {wp.float32, wp.float64}:
            raise ValueError(f"Unsupported cell-list Warp dtype: {wp_dtype!r}.")
        kernel = get_query_cell_list_kernel(
            wp_dtype,
            strategy="atom_centric",
            batched=batched,
            selective=selective,
            partial=partial,
            half_fill=half_fill,
            return_vectors=geometry,
            return_distances=geometry,
            pair_fn=pair_fn,
            atom_centric_path=atom_centric_path,
        )
        return kernel, _cell_list_query_outputs(geometry=geometry, pair_fn=pair_fn)

    return _LazyJaxKernel(build, _CELL_LIST_DTYPE_MAP)
