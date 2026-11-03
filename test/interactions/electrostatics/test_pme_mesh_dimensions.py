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

"""Checks for shared PME mesh rounding at actual integer boundaries."""

import math

import numpy as np
import pytest

from nvalchemiops.interactions.electrostatics._pme_mesh import (
    _round_pme_mesh_dimensions,
)


def test_rounding_returns_the_least_supported_dimension() -> None:
    """Every small integer target uses the first FFT-friendly dimension."""

    def is_smooth(value: int) -> bool:
        for radix in (2, 3, 5, 7):
            while value % radix == 0:
                value //= radix
        return value == 1

    for minimum in range(1, 2049):
        result = _round_pme_mesh_dimensions((minimum,) * 3, spline_order=1)
        expected = max(2, minimum)
        while not is_smooth(expected):
            expected += 1
        assert result == (expected,) * 3


@pytest.mark.parametrize("order", (1, 2, 3, 4, 5, 6))
def test_mesh_minimum_follows_assignment_support(order: int) -> None:
    """Loose targets retain spline support and nonzero reciprocal modes."""
    dimensions = _round_pme_mesh_dimensions((0.1, 1, 1.5), spline_order=order)
    assert all(value >= max(2, order) for value in dimensions)
    assert dimensions == (max(2, order),) * 3


def test_rounding_preserves_exact_and_next_float_boundaries() -> None:
    """A representable target above an integer receives an upward mesh."""
    dimensions = _round_pme_mesh_dimensions(
        (128, math.nextafter(128, -math.inf), math.nextafter(128, math.inf)),
        spline_order=4,
    )
    assert dimensions == (128, 128, 135)


def test_anisotropic_dimensions_are_rounded_independently() -> None:
    """Each axis receives its own supporting FFT-friendly integer."""
    dimensions = _round_pme_mesh_dimensions(
        np.array((65.1, 128, 239)), spline_order=np.int32(4)
    )
    assert dimensions == (70, 128, 240)


def test_mesh_growth_at_power_of_two_boundary() -> None:
    """Crossing a power-of-two boundary takes the next smooth dimension."""
    before = _round_pme_mesh_dimensions((128,) * 3, spline_order=4)
    after = _round_pme_mesh_dimensions(
        (math.nextafter(128, math.inf),) * 3, spline_order=4
    )
    assert math.prod(before) == 128**3
    assert math.prod(after) == 135**3


@pytest.mark.parametrize(
    "target, expected",
    [
        ((500, 500, 500), (512, 512, 512)),
        ((474, 474, 474), (512, 512, 512)),
        ((72, 75, 125), (72, 80, 128)),
        ((405, 16, 16), (420, 16, 16)),
        ((5, 5, 5), (5, 5, 5)),
    ],
)
def test_padding_preferences_respect_the_total_point_budget(
    target: tuple[int, int, int], expected: tuple[int, int, int]
) -> None:
    """Prefer powers of two, then multiples of four, within the same budget."""
    dimensions = _round_pme_mesh_dimensions(
        target, spline_order=5, fft_padding_fraction=0.25
    )
    assert dimensions == expected


def test_padding_budget_applies_to_the_whole_mesh() -> None:
    """Three individually small axis increases still share one point budget."""
    dimensions = _round_pme_mesh_dimensions(
        (112, 112, 112), spline_order=5, fft_padding_fraction=0.25
    )
    assert (128 / 112 - 1) < 0.25
    assert (128 / 112) ** 3 > 1.25
    assert dimensions == (112, 112, 112)


def test_snapping_accepts_an_exact_budget_boundary() -> None:
    """A preferred mesh with exactly the permitted point count is accepted."""
    target = (128, 128, 100)
    at_limit = _round_pme_mesh_dimensions(
        target, spline_order=5, fft_padding_fraction=0.28
    )
    below_limit = _round_pme_mesh_dimensions(
        target, spline_order=5, fft_padding_fraction=0.28 - 1e-6
    )
    assert at_limit == (128, 128, 128)
    assert below_limit == target


@pytest.mark.parametrize("padding", (0.0, 0.05, 0.25, 1.0))
def test_snapped_mesh_covers_targets_with_bounded_total_padding(padding: float) -> None:
    """Snapping preserves support, smooth factors, and its total point limit."""
    for target in range(1, 257):
        raw = (target + 0.1, target * 1.3, target * 0.7)
        smallest = _round_pme_mesh_dimensions(raw, spline_order=5)
        selected = _round_pme_mesh_dimensions(
            raw, spline_order=5, fft_padding_fraction=padding
        )
        assert math.prod(selected) <= (1 + padding) * math.prod(smallest)
        for actual, minimum in zip(selected, smallest, strict=True):
            assert actual >= minimum
            remainder = actual
            for radix in (2, 3, 5, 7):
                while remainder % radix == 0:
                    remainder //= radix
            assert remainder == 1


def test_zero_padding_preserves_the_smallest_smooth_mesh() -> None:
    """Callers can retain the minimum mesh even near a preferred FFT size."""
    assert _round_pme_mesh_dimensions(
        (500, 500, 500), spline_order=5, fft_padding_fraction=0
    ) == (500, 500, 500)


@pytest.mark.parametrize("padding", (0.0, 0.25))
def test_snapping_matches_an_independent_integer_search(padding: float) -> None:
    """Fast selection matches a direct search for anisotropic supported meshes."""

    def is_smooth(value: int) -> bool:
        for radix in (2, 3, 5, 7):
            while value % radix == 0:
                value //= radix
        return value == 1

    supported = [value for value in range(5, 2049) if is_smooth(value)]
    cases = [
        (0.1, 1.0, 1.5),
        (127.9, 128.01, 137.0),
        (405.0, 16.0, 16.0),
        (474.0, 474.0, 474.0),
        (375.8, 242.2, 511.1),
        (500.2, 75.0, 125.0),
    ]
    cases.extend(
        (target + 0.1, target * 1.3, target * 0.7) for target in range(1, 514, 7)
    )
    for raw in cases:
        targets = tuple(max(5, math.ceil(value)) for value in raw)
        smallest = tuple(
            next(value for value in supported if value >= target) for target in targets
        )
        powers = tuple(2 ** math.ceil(math.log2(target)) for target in targets)
        aligned = tuple(
            next(value for value in supported if value >= target and value % 4 == 0)
            for target in targets
        )
        point_limit = (1 + padding) * math.prod(smallest)
        expected = next(
            mesh
            for mesh in (powers, aligned, smallest)
            if math.prod(mesh) <= point_limit
        )
        assert (
            _round_pme_mesh_dimensions(
                raw, spline_order=5, fft_padding_fraction=padding
            )
            == expected
        )


def test_snapped_mesh_growth_at_power_of_two_boundary() -> None:
    """The default padding budget keeps a boundary crossing below a cubic jump."""
    before = _round_pme_mesh_dimensions(
        (128,) * 3, spline_order=5, fft_padding_fraction=0.25
    )
    after = _round_pme_mesh_dimensions(
        (math.nextafter(128, math.inf),) * 3,
        spline_order=5,
        fft_padding_fraction=0.25,
    )
    assert before == (128, 128, 128)
    assert after == (140, 140, 140)
    assert math.prod(after) < 2 * math.prod(before)


@pytest.mark.parametrize("order", (4, 6))
def test_accuracy_rounding_retains_upstream_resolution(order: int) -> None:
    """Accuracy-based meshes keep upstream resolution at rounding boundaries."""
    for target in (0.1, 4.1, 64.0, math.nextafter(64, math.inf), 71.895462, 128.1):
        actual = _round_pme_mesh_dimensions(
            (target,) * 3, spline_order=order, power_of_two=True
        )
        expected = 2 ** math.ceil(math.log2(max(2, order, math.ceil(target))))
        assert actual == (expected,) * 3


@pytest.mark.parametrize("target", (1.0, 13.0, 128.0, 256.01, 2048.0))
def test_mesh_is_monotone_in_target(target: float) -> None:
    """Increasing the continuous size keeps or increases every mesh axis."""
    lower = _round_pme_mesh_dimensions((target,) * 3, spline_order=4)
    upper = _round_pme_mesh_dimensions((target + 1,) * 3, spline_order=4)
    assert all(a <= b for a, b in zip(lower, upper, strict=True))


@pytest.mark.parametrize("value", (0, -1, math.nan, math.inf, True))
def test_invalid_targets_are_rejected(value: float) -> None:
    """Nonpositive and nonfinite targets fail before creating a mesh."""
    with pytest.raises(ValueError, match="positive finite"):
        _round_pme_mesh_dimensions((value, 32, 32), spline_order=4)


@pytest.mark.parametrize("order", (0, -1, True, 4.5))
def test_invalid_assignment_order_is_rejected(order: int) -> None:
    """Assignment support must be described by a positive integer."""
    with pytest.raises(ValueError, match="spline_order"):
        _round_pme_mesh_dimensions((32, 32, 32), spline_order=order)


@pytest.mark.parametrize("padding", (-1, math.nan, math.inf, True))
@pytest.mark.parametrize("power_of_two", (False, True))
def test_invalid_padding_budget_is_rejected(padding: float, power_of_two: bool) -> None:
    """Every rounding mode validates its configured extra-point budget."""
    with pytest.raises(ValueError, match="fft_padding_fraction"):
        _round_pme_mesh_dimensions(
            (32, 32, 32),
            spline_order=5,
            power_of_two=power_of_two,
            fft_padding_fraction=padding,
        )
