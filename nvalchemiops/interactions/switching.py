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

"""
Switching functions for smooth cutoffs.

These utilities are designed to be callable from Warp kernels (``@wp.func``).

We provide a C2-continuous quintic switching function commonly used in MD:

Let :math:`x = (r - r_{on}) / (r_{cut} - r_{on})` in [0, 1]. Then:

.. math::

    s(x) = 1 - 10x^3 + 15x^4 - 6x^5

This yields:
- s = 1 at r = r_on
- s = 0 at r = r_cut
- s', s'' continuous at both endpoints (C2)
"""

from __future__ import annotations

import warp as wp

__all__ = [
    "switch_c2",
]


@wp.func
def switch_c2(
    r: wp.float64, r_on: wp.float64, r_cut: wp.float64
) -> tuple[wp.float64, wp.float64]:
    """C2 switching function and derivative.

    Parameters
    ----------
    r : wp.float64
        Distance.
    r_on : wp.float64
        Switch-on radius. For r <= r_on, s=1.
    r_cut : wp.float64
        Cutoff radius. For r >= r_cut, s=0.

    Returns
    -------
    wp.vec2d
        (s, ds_dr)
    """
    if r <= r_on:
        return wp.float64(1.0), wp.float64(0.0)
    if r >= r_cut:
        return wp.float64(0.0), wp.float64(0.0)

    denom = r_cut - r_on
    # Defensive: if denom is ~0, behave like a hard cutoff
    if denom <= wp.float64(1e-12):
        return wp.float64(0.0), wp.float64(0.0)

    x = (r - r_on) / denom  # in (0,1)

    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    x5 = x4 * x

    s = (
        wp.float64(1.0)
        - wp.float64(10.0) * x3
        + wp.float64(15.0) * x4
        - wp.float64(6.0) * x5
    )

    # ds/dx = -30 x^2 + 60 x^3 - 30 x^4
    ds_dx = -wp.float64(30.0) * x2 + wp.float64(60.0) * x3 - wp.float64(30.0) * x4
    ds_dr = ds_dx / denom

    return s, ds_dr
