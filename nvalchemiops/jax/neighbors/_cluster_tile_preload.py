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
from typing import Any

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


def _current_warp_device_alias() -> str:
    """Return the Warp alias for JAX's configured default device."""
    jax_device = jax.config.jax_default_device
    if jax_device is None:
        jax_device = jax.local_devices()[0]
    return str(wp.device_from_jax(jax_device))


def _warp_device_aliases(device_source: Any | None = None) -> tuple[str, ...]:
    """Return Warp aliases for the device(s) selected by a JAX array.

    Parameters
    ----------
    device_source : Any or None, optional
        Eager or traced JAX value that identifies the intended execution
        device. When unavailable, local accelerator devices are used.

    Returns
    -------
    tuple[str, ...]
        Warp device aliases requiring kernel-module preload.

    During eager execution, the source array identifies the exact device
    whose modules must be loaded before graph capture. A traced value does
    not expose device placement, so preload every local accelerator to cover
    the eventual compiled execution device (or all local devices when no
    accelerator is available).
    """
    if device_source is None:
        return (_current_warp_device_alias(),)
    try:
        jax_devices = tuple(device_source.devices())
    except (AttributeError, jax.errors.ConcretizationTypeError):
        jax_devices = ()
    if not jax_devices:
        local_devices = tuple(jax.local_devices())
        accelerators = tuple(
            device
            for device in local_devices
            if device.platform in {"gpu", "cuda", "rocm"}
        )
        jax_devices = accelerators or local_devices
    return tuple(str(wp.device_from_jax(device)) for device in jax_devices)


def _load_kernel_modules(device, *kernels: wp.Kernel) -> None:
    """Load each distinct kernel module on ``device``."""

    loaded_modules: set[int] = set()
    for kernel in kernels:
        module_id = id(kernel.module)
        if module_id in loaded_modules:
            continue
        kernel.module.load(device, block_dim=TILE_GROUP_SIZE)
        loaded_modules.add(module_id)


def _validate_cluster_tile_build_options(
    *,
    batched: bool,
    segmented: bool,
    selective: bool,
) -> None:
    """Reject unsupported cluster-tile build preload combinations.

    Parameters
    ----------
    batched, segmented, selective : bool
        Static build-kernel specialization options.

    Raises
    ------
    ValueError
        If the requested combination has no supported build kernel.
    """
    if not batched and segmented:
        raise ValueError("Single-system cluster-tile builds cannot be segmented.")
    if batched and selective and not segmented:
        raise ValueError(
            "Selective batched cluster-tile builds must be segmented.",
        )


def _validate_cluster_tile_matrix_options(
    *,
    batched: bool,
    tile_segmented: bool,
    selective: bool,
    dual_cutoff: bool,
    geometry: bool,
    pair_fn: Any | None,
) -> None:
    """Reject unsupported cluster-tile matrix preload combinations.

    Parameters
    ----------
    batched, tile_segmented, selective, dual_cutoff, geometry : bool
        Static matrix-query specialization options.
    pair_fn : Any or None
        Optional pair callback used by the query specialization.

    Raises
    ------
    ValueError
        If the requested combination has no supported matrix-query kernel.
    """
    if not batched and tile_segmented:
        raise ValueError("Single-system cluster-tile queries cannot tile-segment.")
    if (geometry or pair_fn is not None) and (dual_cutoff or selective):
        raise ValueError(
            "Cluster-tile geometry and pair_fn require non-dual, non-selective "
            "matrix queries.",
        )


def _validate_cluster_tile_coo_options(
    *,
    batched: bool,
    tile_segmented: bool,
    coo_segmented: bool,
    selective: bool,
) -> None:
    """Reject unsupported cluster-tile COO preload combinations.

    Parameters
    ----------
    batched, tile_segmented, coo_segmented, selective : bool
        Static COO-query specialization options.

    Raises
    ------
    ValueError
        If the requested combination has no supported COO-query kernel.
    """
    if not batched and tile_segmented:
        raise ValueError("Single-system cluster-tile queries cannot tile-segment.")
    if coo_segmented and not selective:
        raise ValueError("Segmented COO requires selective cluster-tile queries.")
    if selective and not coo_segmented:
        raise ValueError("Selective COO requires segmented COO output.")


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


def _preload_cluster_tile_build_kernel(*, device_source: Any | None = None) -> None:
    """Construct and load all cluster-tile build specializations.

    Parameters
    ----------
    device_source : Any or None, optional
        JAX value used to select devices for the preload.
    """
    for device_alias in _warp_device_aliases(device_source):
        _preload_cluster_tile_build_module(device_alias)


@functools.cache
def _preload_cluster_tile_query_kernel_cached(
    device_alias: str,
    *,
    batched: bool,
    tile_segmented: bool,
    selective: bool,
    dual_cutoff: bool,
    geometry: bool,
    pair_fn: Any | None,
) -> None:
    """Load one matrix-query specialization once per device."""
    getter = (
        get_batch_query_cluster_tile_kernel
        if batched
        else get_query_cluster_tile_kernel
    )
    kernel = getter(
        tile_segmented=tile_segmented,
        selective=selective,
        dual_cutoff=dual_cutoff,
        return_vectors=geometry,
        return_distances=geometry,
        pair_fn=pair_fn,
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
    *,
    batched: bool,
    tile_segmented: bool = False,
    selective: bool = False,
    dual_cutoff: bool = False,
    geometry: bool = False,
    pair_fn: Any | None = None,
    device_source: Any | None = None,
) -> None:
    """Construct and load a cluster-tile matrix-query specialization.

    Parameters
    ----------
    batched, tile_segmented, selective, dual_cutoff, geometry : bool
        Static matrix-query specialization options.
    pair_fn : Any or None, optional
        Pair callback used by the query specialization.
    device_source : Any or None, optional
        JAX value used to select devices for the preload.
    """
    _validate_cluster_tile_matrix_options(
        batched=batched,
        tile_segmented=tile_segmented,
        selective=selective,
        dual_cutoff=dual_cutoff,
        geometry=geometry,
        pair_fn=pair_fn,
    )
    for device_alias in _warp_device_aliases(device_source):
        _preload_cluster_tile_query_kernel_cached(
            device_alias,
            batched=batched,
            tile_segmented=tile_segmented,
            selective=selective,
            dual_cutoff=dual_cutoff,
            geometry=geometry,
            pair_fn=pair_fn,
        )


@functools.cache
def _preload_cluster_tile_coo_kernel_cached(
    device_alias: str,
    *,
    batched: bool,
    tile_segmented: bool,
    coo_segmented: bool,
    selective: bool,
) -> None:
    """Load one COO-query specialization once per device."""
    if coo_segmented:
        _preload_cluster_tile_build_module(device_alias)
    getter = (
        get_batch_query_cluster_tile_coo_kernel
        if batched
        else get_query_cluster_tile_coo_kernel
    )
    kernel = getter(
        tile_segmented=tile_segmented,
        coo_segmented=coo_segmented,
        selective=selective,
    )
    device = wp.get_device(device_alias)
    empty_sentinel(1, wp.int32, device)
    empty_sentinel(1, wp.bool, device)
    empty_sentinel(1, wp.vec3f, device)
    empty_sentinel(1, wp.float32, device)
    empty_sentinel(2, wp.float32, device)
    _load_kernel_modules(device, kernel)


def _preload_cluster_tile_coo_kernel(
    *,
    batched: bool,
    tile_segmented: bool = False,
    coo_segmented: bool = False,
    selective: bool = False,
    device_source: Any | None = None,
) -> None:
    """Construct and load a topology-only cluster-tile COO specialization.

    Parameters
    ----------
    batched, tile_segmented, coo_segmented, selective : bool
        Static COO-query specialization options.
    device_source : Any or None, optional
        JAX value used to select devices for the preload.
    """
    _validate_cluster_tile_coo_options(
        batched=batched,
        tile_segmented=tile_segmented,
        coo_segmented=coo_segmented,
        selective=selective,
    )
    for device_alias in _warp_device_aliases(device_source):
        _preload_cluster_tile_coo_kernel_cached(
            device_alias,
            batched=batched,
            tile_segmented=tile_segmented,
            coo_segmented=coo_segmented,
            selective=selective,
        )
