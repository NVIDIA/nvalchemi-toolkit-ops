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

"""Tests for the pure naive neighbor-list dispatch policy."""

import pytest
import warp as wp

from nvalchemiops.neighbors.naive.dispatch import (
    _NaiveWorkload,
    _partial_tile_min_atoms,
    _resolve_naive_strategy,
)


def _workload(**overrides) -> _NaiveWorkload:
    """Build a representative single-system workload."""
    values = {
        "device_kind": "cuda",
        "wp_dtype": wp.float32,
        "num_atoms": 1024,
        "num_systems": 1,
        "partial": True,
        "batched": False,
        "pbc": False,
        "wrap_positions": True,
        "geometry_outputs": False,
    }
    values.update(overrides)
    return _NaiveWorkload(**values)


@pytest.mark.parametrize("requested", ["scalar", "auto"])
def test_invalid_workload_is_validated_before_strategy_return(requested):
    """Invalid workload fields are rejected even for early-return strategies."""
    with pytest.raises(ValueError, match="device_kind"):
        _resolve_naive_strategy(requested, _workload(device_kind="tpu"))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"wp_dtype": wp.int32}, "dtype"),
        ({"num_atoms": -1}, "num_atoms"),
        ({"num_systems": 0}, "num_systems"),
    ],
)
def test_invalid_workload_fields_are_not_bypassed(overrides, message):
    """Every workload field is validated before scalar or auto decisions."""
    with pytest.raises(ValueError, match=message):
        _resolve_naive_strategy("scalar", _workload(**overrides))


def test_invalid_strategy_is_rejected():
    """Unknown strategy names produce the public validation error."""
    with pytest.raises(ValueError, match="strategy must be"):
        _resolve_naive_strategy("invalid", _workload())


@pytest.mark.parametrize("device_kind", ["cpu", "cuda", "unknown"])
def test_explicit_scalar_always_selects_scalar(device_kind):
    """Explicit scalar is independent of placement."""
    assert (
        _resolve_naive_strategy("scalar", _workload(device_kind=device_kind))
        == "scalar"
    )


def test_explicit_tile_rejects_cpu():
    """Explicit tile cannot target a concrete CPU device."""
    with pytest.raises(ValueError, match="requires CUDA"):
        _resolve_naive_strategy("tile", _workload(device_kind="cpu"))


def test_explicit_tile_allows_unknown_placement():
    """Explicit tile defers unknown placement validation to the launch site."""
    assert _resolve_naive_strategy("tile", _workload(device_kind="unknown")) == "tile"


@pytest.mark.parametrize("num_atoms", [0, 1024, 10000])
def test_partial_auto_unknown_placement_is_scalar(num_atoms):
    """Partial auto stays scalar when the placement is not concrete CUDA."""
    assert (
        _resolve_naive_strategy(
            "auto",
            _workload(device_kind="unknown", num_atoms=num_atoms),
        )
        == "scalar"
    )


def test_geometry_tile_is_rejected():
    """Tile does not support geometry or pair-function outputs."""
    with pytest.raises(ValueError, match="no geometry"):
        _resolve_naive_strategy("tile", _workload(geometry_outputs=True))


@pytest.mark.parametrize(
    ("wp_dtype", "small", "large"),
    [
        (wp.float16, 1023, 1024),
        (wp.float32, 1023, 1024),
        (wp.float64, 255, 256),
    ],
)
def test_partial_thresholds_are_dtype_specific(wp_dtype, small, large):
    """Known CUDA partial auto dispatch uses the dtype threshold."""
    workload = _workload(wp_dtype=wp_dtype, num_atoms=small)
    assert _resolve_naive_strategy("auto", workload) == "scalar"
    assert (
        _resolve_naive_strategy(
            "auto",
            workload.__class__(
                **{
                    **workload.__dict__,
                    "num_atoms": large,
                }
            ),
        )
        == "tile"
    )
    assert _partial_tile_min_atoms(wp_dtype) == large


def test_batched_partial_auto_is_scalar():
    """Batched partial auto remains scalar until benchmarked."""
    assert (
        _resolve_naive_strategy(
            "auto",
            _workload(batched=True, num_systems=4, num_atoms=4096),
        )
        == "scalar"
    )


@pytest.mark.parametrize(
    ("num_atoms", "num_systems", "expected"),
    [
        (2047, 1, "scalar"),
        (2048, 1, "tile"),
        (4095, 16, "scalar"),
        (4096, 16, "tile"),
        (12289, 25, "scalar"),
        (12289, 16, "tile"),
    ],
)
def test_batched_no_pbc_density_boundaries(num_atoms, num_systems, expected):
    """Full-row batched no-PBC auto preserves density thresholds."""
    assert (
        _resolve_naive_strategy(
            "auto",
            _workload(
                num_atoms=num_atoms,
                num_systems=num_systems,
                partial=False,
                batched=True,
            ),
        )
        == expected
    )


def test_batched_pbc_auto_requires_wrapping_for_tile():
    """Full-row batched prewrapped PBC remains scalar under auto."""
    workload = _workload(
        num_atoms=4096,
        num_systems=1,
        partial=False,
        batched=True,
        pbc=True,
    )
    assert _resolve_naive_strategy("auto", workload) == "tile"
    assert (
        _resolve_naive_strategy(
            "auto",
            workload.__class__(**{**workload.__dict__, "wrap_positions": False}),
        )
        == "scalar"
    )
