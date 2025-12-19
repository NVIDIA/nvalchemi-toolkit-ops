# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Any

import warp as wp


@wp.func
def wp_safe_divide(x: wp.float64, y: wp.float64) -> wp.float64:
    """Safe division.

    Divides x by y, with a safe division to avoid division by zero.
    """
    return wp.where(y < wp.float64(1e-8), wp.float64(0.0), x / y)


@wp.func
def wp_exp_kernel(x: wp.float64, factor: wp.float64) -> wp.float64:
    """
    Safe exponential multiplication and division.

    Calculates exp(-x * factor) / x, with a safe division to avoid division by zero.
    """
    return wp_safe_divide(wp.exp(-x * factor), x)


@wp.func
def wpdivmod(a: int, b: int):  # type: ignore
    """Warp implementation of the divmod utility."""
    div = int(a / b)
    mod = a % b
    if mod < 0:
        div -= 1
        mod = b + mod
    return div, mod


@wp.func
def wp_erfc(x: Any) -> Any:
    """Complementary error function approximation for float32.

    Uses the Abramowitz and Stegun approximation with maximum error ~1.5e-7.
    erfc(x) = 1 - erf(x) for x >= 0, and erfc(-x) = 2 - erfc(x) for x < 0.

    Parameters
    ----------
    x : Any
        Input value

    Returns
    -------
    Any
        erfc(x) approximation
    """
    abs_x = wp.abs(x)

    # Abramowitz and Stegun constants for erfc approximation
    p = type(x)(0.3275911)
    a1 = type(x)(0.254829592)
    a2 = type(x)(-0.284496736)
    a3 = type(x)(1.421413741)
    a4 = type(x)(-1.453152027)
    a5 = type(x)(1.061405429)

    # Compute approximation for |x|
    t = type(x)(1.0) / (type(x)(1.0) + p * abs_x)
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    t5 = t4 * t

    # Polynomial approximation
    poly = a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5

    # Apply exponential factor
    exp_neg_x2 = wp.exp(-abs_x * abs_x)
    erfc_abs_x = poly * exp_neg_x2

    # Handle sign: erfc(-x) = 2 - erfc(x)
    return wp.where(x >= type(x)(0.0), erfc_abs_x, type(x)(2.0) - erfc_abs_x)
