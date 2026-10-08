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

"""Upward rounding shared by PME accuracy and spacing estimators."""

from __future__ import annotations

import math
from collections.abc import Iterable
from functools import lru_cache
from numbers import Integral

__all__ = []

_DEFAULT_PME_SPLINE_ORDER = 5
_DEFAULT_PME_FFT_PADDING_FRACTION = 0.25


@lru_cache
def _next_fft_dimension(target: int) -> int:
    """Return the least 2/3/5/7-smooth integer at or above the target.

    The next power of two supplies a finite upper bound. Enumerating powers
    of two, three, and five leaves only the smallest required power of seven
    for each product, so the search uses the requested dimension alone.
    The bounded cache retains these pure integer results across setup calls.
    """
    result = 1 << (target - 1).bit_length()
    power_two = 1
    while power_two < result:
        power_three = power_two
        while power_three < result:
            power_five = power_three
            while power_five < result:
                candidate = power_five
                while candidate < target:
                    candidate *= 7
                result = min(result, candidate)
                power_five *= 5
            power_three *= 3
        power_two *= 2
    return result


def _round_pme_mesh_dimensions(
    raw_dimensions: Iterable[float],
    *,
    spline_order: int,
    power_of_two: bool = False,
    fft_padding_fraction: float = 0.0,
) -> tuple[int, int, int]:
    """Round three positive PME targets upward to FFT-friendly dimensions.

    The minimum axis length accommodates the spline assignment support and
    at least one nonzero reciprocal mode. Rounding preserves the supplied
    targets; the sizing formula and its accuracy interpretation belong to
    the calling estimator. Order-five accuracy sizing and explicit spacing
    use the least smooth dimension that covers the target. Other assignment
    orders retain the upstream power-of-two accuracy sizing. With a positive
    padding budget, snap to powers of two first, then smooth multiples of
    four, allowing at most that fraction of extra total mesh points over
    the smallest smooth mesh. A zero budget retains the smallest mesh.
    """
    if (
        isinstance(spline_order, bool)
        or not isinstance(spline_order, Integral)
        or spline_order < 1
    ):
        raise ValueError("spline_order must be a positive integer")
    if isinstance(fft_padding_fraction, bool):
        raise ValueError("fft_padding_fraction must be finite and nonnegative")
    fft_padding_fraction = float(fft_padding_fraction)
    if not math.isfinite(fft_padding_fraction) or fft_padding_fraction < 0:
        raise ValueError("fft_padding_fraction must be finite and nonnegative")
    values = tuple(raw_dimensions)
    if len(values) != 3:
        raise ValueError("raw_dimensions must contain three positive finite values")
    targets = []
    minimum = max(2, int(spline_order))
    for value in values:
        if isinstance(value, bool):
            raise ValueError("raw_dimensions must contain three positive finite values")
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("raw_dimensions must contain three positive finite values")
        targets.append(max(minimum, math.ceil(value)))

    if power_of_two or fft_padding_fraction > 0:
        powers = tuple(1 << (value - 1).bit_length() for value in targets)
        if power_of_two:
            return powers
        # The integer targets bound the smallest smooth mesh from below.
        # Meeting this tighter limit lets us accept powers of two immediately.
        if math.prod(powers) <= (1.0 + fft_padding_fraction) * math.prod(targets):
            return powers

    # Repeated axis lengths need only one smooth-number search per call.
    smooth = {value: _next_fft_dimension(value) for value in set(targets)}
    smallest = (smooth[targets[0]], smooth[targets[1]], smooth[targets[2]])
    if fft_padding_fraction == 0:
        return smallest
    point_limit = (1.0 + fft_padding_fraction) * math.prod(smallest)
    if math.prod(powers) <= point_limit:
        return powers
    aligned = tuple(
        value if value % 4 == 0 else 4 * _next_fft_dimension((value + 3) // 4)
        for value in smallest
    )
    if math.prod(aligned) <= point_limit:
        return aligned
    return smallest
