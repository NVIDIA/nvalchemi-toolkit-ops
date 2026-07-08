# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import jax
import warp as wp

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


def _current_warp_device():
    """Return the Warp device selected by the current JAX default."""
    jax_device = jax.config.jax_default_device
    if jax_device is None:
        jax_device = jax.local_devices()[0]
    return wp.device_from_jax(jax_device)


def _load_kernel_modules(device, *kernels: wp.Kernel) -> None:
    """Load each distinct kernel module on ``device``."""

    loaded_modules: set[int] = set()
    for kernel in kernels:
        module_id = id(kernel.module)
        if module_id in loaded_modules:
            continue
        kernel.module.load(device, block_dim=TILE_GROUP_SIZE)
        loaded_modules.add(module_id)


def _preload_cluster_tile_build_kernel(
    *, batched: bool, segmented: bool, selective: bool
) -> None:
    """Construct and load a cluster-tile build specialization."""
    kernels = [
        _get_build_cluster_tiles_kernel(
            batched=batched,
            segmented=segmented,
            selective=selective,
        )
    ]
    needs_reset = (batched and segmented) or (not batched and selective)
    if needs_reset:
        kernels.append(_get_reset_cluster_tile_counts_kernel(selective=selective))
    device = _current_warp_device()
    empty_sentinel(1, wp.int32, device)
    empty_sentinel(1, wp.bool, device)
    _load_kernel_modules(device, *kernels)


def _preload_cluster_tile_query_kernel(
    *,
    batched: bool,
    tile_segmented: bool,
    selective: bool,
    dual_cutoff: bool,
    return_vectors: bool,
    return_distances: bool,
    pair_fn: wp.Function | None,
) -> None:
    """Construct and load a cluster-tile matrix-query specialization."""
    getter = (
        get_batch_query_cluster_tile_kernel
        if batched
        else get_query_cluster_tile_kernel
    )
    kernel = getter(
        tile_segmented=tile_segmented,
        selective=selective,
        dual_cutoff=dual_cutoff,
        return_vectors=return_vectors,
        return_distances=return_distances,
        pair_fn=pair_fn,
    )
    device = _current_warp_device()
    empty_sentinel(1, wp.int32, device)
    empty_sentinel(2, wp.int32, device)
    empty_sentinel(3, wp.int32, device)
    empty_sentinel(1, wp.bool, device)
    empty_sentinel(2, wp.vec3f, device)
    empty_sentinel(2, wp.float32, device)
    _load_kernel_modules(device, kernel)


def _preload_cluster_tile_coo_kernel(
    *,
    batched: bool,
    tile_segmented: bool,
    coo_segmented: bool,
    selective: bool,
) -> None:
    """Construct and load a topology-only cluster-tile COO specialization."""
    kernels: list[wp.Kernel] = []
    if coo_segmented:
        kernels.append(_get_reset_cluster_tile_counts_kernel(selective=selective))
    getter = (
        get_batch_query_cluster_tile_coo_kernel
        if batched
        else get_query_cluster_tile_coo_kernel
    )
    kernels.append(
        getter(
            tile_segmented=tile_segmented,
            coo_segmented=coo_segmented,
            selective=selective,
        )
    )
    device = _current_warp_device()
    empty_sentinel(1, wp.int32, device)
    empty_sentinel(1, wp.bool, device)
    empty_sentinel(1, wp.vec3f, device)
    empty_sentinel(1, wp.float32, device)
    empty_sentinel(2, wp.float32, device)
    _load_kernel_modules(device, *kernels)
