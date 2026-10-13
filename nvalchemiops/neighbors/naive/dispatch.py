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

"""Naive neighbor-list dispatch policy and mode parsing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

import warp as wp

from nvalchemiops.neighbors.output_args import _has_partial_or_pair_outputs

__all__: list[str] = []

_SUPPORTED_DTYPES = (wp.float16, wp.float32, wp.float64)
_PARTIAL_TILE_MIN_ATOMS_BY_DTYPE = {
    wp.float16: 16 * 64,
    wp.float32: 16 * 64,
    wp.float64: 4 * 64,
}

_DeviceKind = Literal["cpu", "cuda", "unknown"]


@dataclass(frozen=True)
class _NaiveWorkload:
    """Static properties used by naive strategy resolution."""

    device_kind: _DeviceKind
    wp_dtype: type
    num_atoms: int
    num_systems: int
    partial: bool
    batched: bool
    pbc: bool
    wrap_positions: bool
    geometry_outputs: bool


def _partial_tile_min_atoms(wp_dtype: type) -> int:
    """Return the auto-dispatch atom threshold for partial tiled kernels."""
    try:
        return _PARTIAL_TILE_MIN_ATOMS_BY_DTYPE[wp_dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported naive dtype: {wp_dtype!r}") from exc


def _resolve_naive_strategy(
    requested: Literal["auto", "scalar", "tile"],
    workload: _NaiveWorkload,
) -> Literal["scalar", "tile"]:
    """Resolve a naive launcher strategy from static workload properties."""
    if requested not in {"auto", "scalar", "tile"}:
        raise ValueError(
            f"strategy must be 'auto' | 'scalar' | 'tile', got {requested!r}",
        )
    if workload.device_kind not in {"cpu", "cuda", "unknown"}:
        raise ValueError("device_kind must be 'cpu' | 'cuda' | 'unknown'")
    if workload.wp_dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"Unsupported naive dtype: {workload.wp_dtype!r}")
    if workload.num_atoms < 0:
        raise ValueError("num_atoms must be non-negative")
    if workload.num_systems < 1:
        raise ValueError("num_systems must be at least one")

    if requested == "scalar":
        return "scalar"
    if requested == "tile":
        if workload.geometry_outputs:
            raise ValueError(
                "strategy='tile' requires CUDA and no geometry or pair_fn outputs",
            )
        if workload.device_kind == "cpu":
            raise ValueError("strategy='tile' requires CUDA")
        return "tile"

    if workload.geometry_outputs or workload.device_kind == "cpu":
        return "scalar"
    if workload.partial:
        if workload.batched or workload.device_kind != "cuda":
            return "scalar"
        return (
            "tile"
            if workload.num_atoms >= _partial_tile_min_atoms(workload.wp_dtype)
            else "scalar"
        )
    if not workload.batched:
        return "tile"
    if workload.pbc:
        return "tile" if workload.wrap_positions else "scalar"
    if workload.num_atoms < 2048:
        return "scalar"
    use_tiled = workload.num_atoms >= 256 * workload.num_systems
    if use_tiled and workload.num_atoms > 12288:
        use_tiled = workload.num_atoms >= 512 * workload.num_systems
    return "tile" if use_tiled else "scalar"


def _require_cuda_tile_device(device: str) -> None:
    """Reject tile launches on a concrete CPU Warp device."""
    if wp.get_device(device).is_cpu:
        raise ValueError("strategy='tile' requires CUDA")


class _PBCMode(Enum):
    """Periodic-boundary mode used by naive kernel factories."""

    NONE = "none"
    PREWRAPPED = "prewrapped"
    WRAP_ON_ENTRY = "wrap_on_entry"


class _NaiveStrategy(Enum):
    """Naive implementation strategy."""

    SCALAR = "scalar"
    TILE = "tile"


def _parse_pbc_mode(
    pbc_mode: Literal["none", "prewrapped", "wrap_on_entry"] | _PBCMode,
) -> _PBCMode:
    """Normalize a public PBC mode value to the private enum."""
    if isinstance(pbc_mode, _PBCMode):
        return pbc_mode
    try:
        return _PBCMode(pbc_mode)
    except ValueError as exc:
        raise ValueError(
            "pbc_mode must be 'none', 'prewrapped', or 'wrap_on_entry'"
        ) from exc


def _parse_strategy(
    strategy: Literal["scalar", "tile"] | _NaiveStrategy,
) -> _NaiveStrategy:
    """Normalize a public strategy value to the private enum."""
    if isinstance(strategy, _NaiveStrategy):
        return strategy
    try:
        return _NaiveStrategy(strategy)
    except ValueError as exc:
        raise ValueError("strategy must be 'scalar' or 'tile'") from exc


def _has_naive_pair_outputs(
    target_indices: Any | None,
    return_vectors: bool,
    return_distances: bool,
    pair_fn: wp.Function | None,
    pair_params: wp.array | None,
    neighbor_vectors: Any | None,
    neighbor_distances: Any | None,
    pair_energies: Any | None,
    pair_forces: Any | None,
) -> bool:
    """Return True when the single-cutoff pair-output path is required."""
    return _has_partial_or_pair_outputs(
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


def _is_cpu_device(device: str) -> bool:
    """Return whether a Warp device string names a CPU device."""
    return "cpu" in str(device).lower()


def _pbc_mode_from_wrap(wrap_positions: bool) -> _PBCMode:
    """Return the PBC mode represented by ``wrap_positions``."""
    return _PBCMode.WRAP_ON_ENTRY if wrap_positions else _PBCMode.PREWRAPPED
