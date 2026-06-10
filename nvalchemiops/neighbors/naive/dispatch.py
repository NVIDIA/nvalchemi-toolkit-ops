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

from enum import Enum
from typing import Any, Literal

import warp as wp

from nvalchemiops.neighbors.output_args import _has_partial_or_pair_outputs

__all__: list[str] = []


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
