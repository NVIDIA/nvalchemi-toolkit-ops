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

"""Tests for nvalchemiops.interactions.switching module.

This module tests the C2-continuous switching function used for smooth
potential cutoffs in molecular dynamics.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from nvalchemiops.interactions.switching import switch_c2

wp.init()


# =============================================================================
# Test Kernel for switch_c2
# =============================================================================


@wp.kernel
def _test_switch_c2_kernel(
    r_values: wp.array(dtype=wp.float64),
    r_on: wp.float64,
    r_cut: wp.float64,
    s_out: wp.array(dtype=wp.float64),
    ds_dr_out: wp.array(dtype=wp.float64),
):
    """Kernel to test switch_c2 function at multiple r values."""
    i = wp.tid()
    r = r_values[i]
    s, ds_dr = switch_c2(r, r_on, r_cut)
    s_out[i] = s
    ds_dr_out[i] = ds_dr


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def device():
    """Return appropriate device for testing."""
    return "cuda:0" if wp.is_cuda_available() else "cpu"


# =============================================================================
# Test Classes
# =============================================================================


class TestSwitchC2:
    """Tests for the C2 continuous switching function."""

    def test_switch_at_r_on(self, device):
        """Test that s=1 and ds_dr=0 when r <= r_on."""
        r_on = 8.0
        r_cut = 10.0

        # Test exactly at r_on and below
        r_values = np.array([6.0, 7.0, 7.5, 8.0], dtype=np.float64)
        r_wp = wp.array(r_values, dtype=wp.float64, device=device)
        s_out = wp.zeros(len(r_values), dtype=wp.float64, device=device)
        ds_dr_out = wp.zeros(len(r_values), dtype=wp.float64, device=device)

        wp.launch(
            _test_switch_c2_kernel,
            dim=len(r_values),
            inputs=[r_wp, wp.float64(r_on), wp.float64(r_cut), s_out, ds_dr_out],
            device=device,
        )

        s_np = s_out.numpy()
        ds_dr_np = ds_dr_out.numpy()

        np.testing.assert_allclose(s_np, 1.0, rtol=1e-12)
        np.testing.assert_allclose(ds_dr_np, 0.0, atol=1e-12)

    def test_switch_at_r_cut(self, device):
        """Test that s=0 and ds_dr=0 when r >= r_cut."""
        r_on = 8.0
        r_cut = 10.0

        # Test exactly at r_cut and above
        r_values = np.array([10.0, 10.5, 11.0, 12.0], dtype=np.float64)
        r_wp = wp.array(r_values, dtype=wp.float64, device=device)
        s_out = wp.zeros(len(r_values), dtype=wp.float64, device=device)
        ds_dr_out = wp.zeros(len(r_values), dtype=wp.float64, device=device)

        wp.launch(
            _test_switch_c2_kernel,
            dim=len(r_values),
            inputs=[r_wp, wp.float64(r_on), wp.float64(r_cut), s_out, ds_dr_out],
            device=device,
        )

        s_np = s_out.numpy()
        ds_dr_np = ds_dr_out.numpy()

        np.testing.assert_allclose(s_np, 0.0, atol=1e-12)
        np.testing.assert_allclose(ds_dr_np, 0.0, atol=1e-12)

    def test_switch_intermediate_values(self, device):
        """Test switching function in the intermediate region."""
        r_on = 8.0
        r_cut = 10.0

        # Test at midpoint and other intermediate values
        r_values = np.array([8.5, 9.0, 9.5], dtype=np.float64)
        r_wp = wp.array(r_values, dtype=wp.float64, device=device)
        s_out = wp.zeros(len(r_values), dtype=wp.float64, device=device)
        ds_dr_out = wp.zeros(len(r_values), dtype=wp.float64, device=device)

        wp.launch(
            _test_switch_c2_kernel,
            dim=len(r_values),
            inputs=[r_wp, wp.float64(r_on), wp.float64(r_cut), s_out, ds_dr_out],
            device=device,
        )

        s_np = s_out.numpy()
        ds_dr_np = ds_dr_out.numpy()

        # Switch should be strictly between 0 and 1
        assert np.all(s_np > 0.0)
        assert np.all(s_np < 1.0)

        # Switch should be monotonically decreasing
        assert s_np[0] > s_np[1] > s_np[2]

        # Derivative should be negative in the switching region
        assert np.all(ds_dr_np < 0.0)

        # At midpoint (x=0.5), s(0.5) = 1 - 10(0.5)^3 + 15(0.5)^4 - 6(0.5)^5
        # = 1 - 10*0.125 + 15*0.0625 - 6*0.03125 = 1 - 1.25 + 0.9375 - 0.1875 = 0.5
        x_mid = (9.0 - r_on) / (r_cut - r_on)
        expected_s_mid = 1.0 - 10 * x_mid**3 + 15 * x_mid**4 - 6 * x_mid**5
        np.testing.assert_allclose(s_np[1], expected_s_mid, rtol=1e-10)

    def test_switch_c2_continuity(self, device):
        """Test C2 continuity at the boundaries."""
        r_on = 8.0
        r_cut = 10.0
        eps = 1e-8

        # Test continuity at r_on: approach from above
        r_values = np.array([r_on - eps, r_on, r_on + eps], dtype=np.float64)
        r_wp = wp.array(r_values, dtype=wp.float64, device=device)
        s_out = wp.zeros(3, dtype=wp.float64, device=device)
        ds_dr_out = wp.zeros(3, dtype=wp.float64, device=device)

        wp.launch(
            _test_switch_c2_kernel,
            dim=3,
            inputs=[r_wp, wp.float64(r_on), wp.float64(r_cut), s_out, ds_dr_out],
            device=device,
        )

        s_np = s_out.numpy()
        ds_dr_np = ds_dr_out.numpy()

        # s should be continuous at r_on
        np.testing.assert_allclose(s_np[0], s_np[1], rtol=1e-6)
        np.testing.assert_allclose(s_np[1], s_np[2], rtol=1e-6)

        # ds_dr should be continuous (all near 0 at r_on)
        np.testing.assert_allclose(ds_dr_np[0], 0.0, atol=1e-6)
        np.testing.assert_allclose(ds_dr_np[2], 0.0, atol=1e-4)

        # Test continuity at r_cut
        r_values_cut = np.array([r_cut - eps, r_cut, r_cut + eps], dtype=np.float64)
        r_wp_cut = wp.array(r_values_cut, dtype=wp.float64, device=device)
        s_out_cut = wp.zeros(3, dtype=wp.float64, device=device)
        ds_dr_out_cut = wp.zeros(3, dtype=wp.float64, device=device)

        wp.launch(
            _test_switch_c2_kernel,
            dim=3,
            inputs=[
                r_wp_cut,
                wp.float64(r_on),
                wp.float64(r_cut),
                s_out_cut,
                ds_dr_out_cut,
            ],
            device=device,
        )

        s_np_cut = s_out_cut.numpy()

        # s should be continuous at r_cut (use atol for comparison with ~0)
        np.testing.assert_allclose(s_np_cut[0], s_np_cut[1], atol=1e-12)
        np.testing.assert_allclose(s_np_cut[1], s_np_cut[2], atol=1e-12)

    def test_switch_derivative_numerical(self, device):
        """Test derivative against numerical differentiation."""
        r_on = 8.0
        r_cut = 10.0
        h = 1e-6

        # Test points in the switching region
        r_test = np.array([8.3, 8.7, 9.0, 9.3, 9.7], dtype=np.float64)

        for r in r_test:
            r_vals = np.array([r - h, r, r + h], dtype=np.float64)
            r_wp = wp.array(r_vals, dtype=wp.float64, device=device)
            s_out = wp.zeros(3, dtype=wp.float64, device=device)
            ds_dr_out = wp.zeros(3, dtype=wp.float64, device=device)

            wp.launch(
                _test_switch_c2_kernel,
                dim=3,
                inputs=[r_wp, wp.float64(r_on), wp.float64(r_cut), s_out, ds_dr_out],
                device=device,
            )

            s_np = s_out.numpy()
            ds_dr_np = ds_dr_out.numpy()

            # Central difference approximation
            numerical_deriv = (s_np[2] - s_np[0]) / (2 * h)

            np.testing.assert_allclose(
                ds_dr_np[1],
                numerical_deriv,
                rtol=1e-5,
                err_msg=f"Derivative mismatch at r={r}",
            )

    def test_switch_zero_width(self, device):
        """Test behavior when r_on ≈ r_cut (near-zero switching width)."""
        r_on = 10.0
        r_cut = 10.0 + 1e-14  # Near-zero width

        r_values = np.array([9.0, 10.0, 11.0], dtype=np.float64)
        r_wp = wp.array(r_values, dtype=wp.float64, device=device)
        s_out = wp.zeros(3, dtype=wp.float64, device=device)
        ds_dr_out = wp.zeros(3, dtype=wp.float64, device=device)

        wp.launch(
            _test_switch_c2_kernel,
            dim=3,
            inputs=[r_wp, wp.float64(r_on), wp.float64(r_cut), s_out, ds_dr_out],
            device=device,
        )

        s_np = s_out.numpy()

        # Should behave like hard cutoff
        assert s_np[0] == 1.0  # Below r_on
        # At or above r_cut, should be 0 (defensive behavior)
        assert s_np[2] == 0.0

    def test_switch_formula_exact(self, device):
        """Test exact formula: s(x) = 1 - 10x³ + 15x⁴ - 6x⁵."""
        r_on = 5.0
        r_cut = 10.0

        # Create uniform samples in switching region
        n_points = 50
        r_values = np.linspace(r_on + 0.01, r_cut - 0.01, n_points)
        r_wp = wp.array(r_values.astype(np.float64), dtype=wp.float64, device=device)
        s_out = wp.zeros(n_points, dtype=wp.float64, device=device)
        ds_dr_out = wp.zeros(n_points, dtype=wp.float64, device=device)

        wp.launch(
            _test_switch_c2_kernel,
            dim=n_points,
            inputs=[r_wp, wp.float64(r_on), wp.float64(r_cut), s_out, ds_dr_out],
            device=device,
        )

        s_np = s_out.numpy()
        ds_dr_np = ds_dr_out.numpy()

        # Compute expected values using the formula
        x = (r_values - r_on) / (r_cut - r_on)
        expected_s = 1.0 - 10 * x**3 + 15 * x**4 - 6 * x**5
        expected_ds_dx = -30 * x**2 + 60 * x**3 - 30 * x**4
        expected_ds_dr = expected_ds_dx / (r_cut - r_on)

        np.testing.assert_allclose(s_np, expected_s, rtol=1e-8, atol=1e-14)
        np.testing.assert_allclose(ds_dr_np, expected_ds_dr, rtol=1e-8, atol=1e-14)
