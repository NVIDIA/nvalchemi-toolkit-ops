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

"""Preload dynamic cluster-tile kernels before Warp graph capture."""

from __future__ import annotations

import functools

import jax
import warp as wp

from nvalchemiops.jax.neighbors._registration import (
    _ClusterTileBuildJaxSpec,
    _ClusterTileQueryJaxSpec,
    _validate_cluster_tile_build_spec,
    _validate_cluster_tile_query_spec,
)
from nvalchemiops.neighbors.cluster_tile.kernels import (
    TILE_GROUP_SIZE,
    _get_build_cluster_tiles_kernel,
    _get_reset_cluster_tile_counts_kernel,
    get_batch_query_cluster_tile_coo_kernel,
    get_batch_query_cluster_tile_kernel,
    get_query_cluster_tile_coo_kernel,
    get_query_cluster_tile_kernel,
)
from nvalchemiops.neighbors.neighbor_utils import empty_sentinel

__all__ = []


def _current_warp_device_alias() -> str:
    """Return the Warp alias for the device selected by JAX."""
    jax_device = jax.config.jax_default_device
    if jax_device is None:
        jax_device = jax.local_devices()[0]
    return str(wp.device_from_jax(jax_device))


def _load_kernel_modules(device, *kernels: wp.Kernel) -> None:
    """Load each distinct kernel module on ``device``."""

    loaded_modules: set[int] = set()
    for kernel in kernels:
        module_id = id(kernel.module)
        if module_id in loaded_modules:
            continue
        kernel.module.load(device, block_dim=TILE_GROUP_SIZE)
        loaded_modules.add(module_id)


@functools.cache
def _preload_cluster_tile_build_module(device_alias: str) -> None:
    """Register every build variant, then load their shared module once."""
    kernels = [
        _get_build_cluster_tiles_kernel(
            batched=False,
            segmented=False,
            selective=selective,
        )
        for selective in (False, True)
    ]
    kernels.extend(
        _get_build_cluster_tiles_kernel(
            batched=True,
            segmented=segmented,
            selective=selective,
        )
        for segmented, selective in (
            (False, False),
            (True, False),
            (True, True),
        )
    )
    kernels.extend(
        _get_reset_cluster_tile_counts_kernel(selective=selective)
        for selective in (False, True)
    )
    device = wp.get_device(device_alias)
    empty_sentinel(1, wp.int32, device)
    empty_sentinel(1, wp.bool, device)
    _load_kernel_modules(device, *kernels)


def _preload_cluster_tile_build_kernel(
    spec: _ClusterTileBuildJaxSpec,
) -> None:
    """Construct and load a cluster-tile build specialization."""
    _validate_cluster_tile_build_spec(spec)
    _preload_cluster_tile_build_module(_current_warp_device_alias())


def _validate_cluster_tile_query_preload_spec(
    spec: _ClusterTileQueryJaxSpec,
    *,
    output_format: str,
) -> None:
    """Validate a query preload spec for its concrete output format."""
    _validate_cluster_tile_query_spec(spec)
    if spec.output_format != output_format:
        raise ValueError(
            f"Cluster-tile {output_format} preload requires "
            f"output_format={output_format!r}.",
        )


@functools.cache
def _preload_cluster_tile_query_kernel_cached(
    device_alias: str,
    spec: _ClusterTileQueryJaxSpec,
) -> None:
    """Load one matrix-query specialization once per device."""
    _validate_cluster_tile_query_preload_spec(spec, output_format="matrix")
    getter = (
        get_batch_query_cluster_tile_kernel
        if spec.batched
        else get_query_cluster_tile_kernel
    )
    kernel = getter(
        tile_segmented=spec.tile_segmented,
        selective=spec.selective,
        dual_cutoff=spec.dual_cutoff,
        return_vectors=spec.return_vectors,
        return_distances=spec.return_distances,
        pair_fn=spec.pair_fn,
    )
    device = wp.get_device(device_alias)
    empty_sentinel(1, wp.int32, device)
    empty_sentinel(2, wp.int32, device)
    empty_sentinel(3, wp.int32, device)
    empty_sentinel(1, wp.bool, device)
    empty_sentinel(2, wp.vec3f, device)
    empty_sentinel(2, wp.float32, device)
    _load_kernel_modules(device, kernel)


def _preload_cluster_tile_query_kernel(
    spec: _ClusterTileQueryJaxSpec,
) -> None:
    """Construct and load a cluster-tile matrix-query specialization."""
    _validate_cluster_tile_query_preload_spec(spec, output_format="matrix")
    _preload_cluster_tile_query_kernel_cached(
        _current_warp_device_alias(),
        spec,
    )


@functools.cache
def _preload_cluster_tile_coo_kernel_cached(
    device_alias: str,
    spec: _ClusterTileQueryJaxSpec,
) -> None:
    """Load one COO-query specialization once per device."""
    _validate_cluster_tile_query_preload_spec(spec, output_format="coo")
    if spec.coo_segmented:
        _preload_cluster_tile_build_module(device_alias)
    getter = (
        get_batch_query_cluster_tile_coo_kernel
        if spec.batched
        else get_query_cluster_tile_coo_kernel
    )
    kernel = getter(
        tile_segmented=spec.tile_segmented,
        coo_segmented=spec.coo_segmented,
        selective=spec.selective,
    )
    device = wp.get_device(device_alias)
    empty_sentinel(1, wp.int32, device)
    empty_sentinel(1, wp.bool, device)
    empty_sentinel(1, wp.vec3f, device)
    empty_sentinel(1, wp.float32, device)
    empty_sentinel(2, wp.float32, device)
    _load_kernel_modules(device, kernel)


def _preload_cluster_tile_coo_kernel(
    spec: _ClusterTileQueryJaxSpec,
) -> None:
    """Construct and load a topology-only cluster-tile COO specialization."""
    _validate_cluster_tile_query_preload_spec(spec, output_format="coo")
    _preload_cluster_tile_coo_kernel_cached(
        _current_warp_device_alias(),
        spec,
    )
