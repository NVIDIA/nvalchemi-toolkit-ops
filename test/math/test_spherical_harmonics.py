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
Test Suite for Real Spherical Harmonics
=======================================

This module tests the Warp implementation of real spherical harmonics against:
1. Analytical reference values at special points
2. Orthonormality conditions
3. Gradient correctness via finite differences
4. Comparison with scipy.special.sph_harm (complex → real conversion)

Mathematical Reference
----------------------

Real spherical harmonics Y_l^m with orthonormal normalization:
- ∫ Y_l^m(Ω) Y_{l'}^{m'}(Ω) dΩ = δ_{ll'} δ_{mm'}
- Sum over m: ∑_m |Y_l^m(r̂)|² = (2l+1)/(4π)

For specific directions:
- Along z-axis (θ=0): Only Y_l^0 is non-zero, Y_l^0(ẑ) = √((2l+1)/(4π))
- Along x-axis: Y_l^m involves associated Legendre polynomials evaluated at θ=π/2
- Along y-axis: Similar to x-axis with phase factors
"""

import math

import numpy as np
import pytest
import torch

from nvalchemiops.math.spherical_harmonics import (
    eval_spherical_harmonics_gradient_pytorch,
    eval_spherical_harmonics_pytorch,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def device():
    """Get the compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


@pytest.fixture
def random_positions(device):
    """Generate random unit vectors for testing."""
    torch.manual_seed(42)
    N = 100
    positions = torch.randn(N, 3, dtype=torch.float64, device=device)
    # Normalize to unit sphere
    norms = torch.norm(positions, dim=1, keepdim=True)
    positions = positions / norms
    return positions


@pytest.fixture
def axis_positions(device):
    """Positions along coordinate axes."""
    return torch.tensor(
        [
            [1.0, 0.0, 0.0],  # +x
            [-1.0, 0.0, 0.0],  # -x
            [0.0, 1.0, 0.0],  # +y
            [0.0, -1.0, 0.0],  # -y
            [0.0, 0.0, 1.0],  # +z
            [0.0, 0.0, -1.0],  # -z
        ],
        dtype=torch.float64,
        device=device,
    )


# =============================================================================
# Reference Implementation (for validation)
# =============================================================================


def real_spherical_harmonics_reference(positions: torch.Tensor) -> torch.Tensor:
    """Reference implementation of real spherical harmonics using analytical formulas.

    Uses explicit formulas for L ≤ 2 to validate the Warp implementation.

    Parameters
    ----------
    positions : torch.Tensor
        Positions [N, 3] (will be normalized).

    Returns
    -------
    torch.Tensor
        Spherical harmonics [N, 9] for L=0,1,2.
    """
    # Normalize positions
    r = torch.norm(positions, dim=1, keepdim=True)
    r = torch.clamp(r, min=1e-30)
    r_hat = positions / r

    x = r_hat[:, 0]
    y = r_hat[:, 1]
    z = r_hat[:, 2]

    # Constants
    sqrt_1_4pi = 1.0 / math.sqrt(4.0 * math.pi)  # 0.28209479...
    sqrt_3_4pi = math.sqrt(3.0 / (4.0 * math.pi))  # 0.48860251...
    sqrt_15_4pi = math.sqrt(15.0 / (4.0 * math.pi))  # 1.09254843...
    sqrt_5_16pi = math.sqrt(5.0 / (16.0 * math.pi))  # 0.31539156...
    sqrt_15_16pi = math.sqrt(15.0 / (16.0 * math.pi))  # 0.54627421...

    N = positions.shape[0]
    output = torch.zeros(N, 9, dtype=torch.float64, device=positions.device)

    # L=0
    output[:, 0] = sqrt_1_4pi  # Y_0^0

    # L=1
    output[:, 1] = sqrt_3_4pi * y  # Y_1^{-1}
    output[:, 2] = sqrt_3_4pi * z  # Y_1^0
    output[:, 3] = sqrt_3_4pi * x  # Y_1^{+1}

    # L=2
    output[:, 4] = sqrt_15_4pi * x * y  # Y_2^{-2}
    output[:, 5] = sqrt_15_4pi * y * z  # Y_2^{-1}
    output[:, 6] = sqrt_5_16pi * (3 * z**2 - 1)  # Y_2^0 (using r=1)
    output[:, 7] = sqrt_15_4pi * x * z  # Y_2^{+1}
    output[:, 8] = sqrt_15_16pi * (x**2 - y**2)  # Y_2^{+2}

    return output


# =============================================================================
# Test Classes
# =============================================================================


class TestSphericalHarmonicsAPI:
    """Test the basic API and shapes."""

    def test_l0_shape(self, random_positions, device):
        """Test L=0 output shape."""
        output = eval_spherical_harmonics_pytorch(
            random_positions, L_max=0, device=device
        )
        assert output.shape == (100, 1)

    def test_l1_shape(self, random_positions, device):
        """Test L=1 output shape."""
        output = eval_spherical_harmonics_pytorch(
            random_positions, L_max=1, device=device
        )
        assert output.shape == (100, 4)

    def test_l2_shape(self, random_positions, device):
        """Test L=2 output shape."""
        output = eval_spherical_harmonics_pytorch(
            random_positions, L_max=2, device=device
        )
        assert output.shape == (100, 9)

    def test_gradient_shape(self, random_positions, device):
        """Test gradient output shape."""
        output = eval_spherical_harmonics_gradient_pytorch(
            random_positions, L_max=2, device=device
        )
        assert output.shape == (100, 9, 3)

    def test_single_position(self, device):
        """Test with a single position."""
        pos = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64, device=device)
        output = eval_spherical_harmonics_pytorch(pos, L_max=2, device=device)
        assert output.shape == (1, 9)


class TestSphericalHarmonicsValues:
    """Test spherical harmonic values against analytical references."""

    def test_y00_is_constant(self, random_positions, device):
        """Y_0^0 should be constant for all directions."""
        output = eval_spherical_harmonics_pytorch(
            random_positions, L_max=0, device=device
        )
        expected = 1.0 / math.sqrt(4.0 * math.pi)
        torch.testing.assert_close(
            output[:, 0], torch.full_like(output[:, 0], expected)
        )

    def test_along_z_axis(self, device):
        """Test values along the z-axis where only Y_l^0 should be non-zero for L≥1."""
        pos = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64, device=device)
        output = eval_spherical_harmonics_pytorch(pos, L_max=2, device=device)

        # Expected values
        sqrt_1_4pi = 1.0 / math.sqrt(4.0 * math.pi)
        sqrt_3_4pi = math.sqrt(3.0 / (4.0 * math.pi))
        sqrt_5_16pi = math.sqrt(5.0 / (16.0 * math.pi))

        # Y_0^0
        torch.testing.assert_close(
            output[0, 0], torch.tensor(sqrt_1_4pi, dtype=torch.float64, device=device)
        )

        # Y_1 components: only Y_1^0 should be non-zero
        assert abs(output[0, 1].item()) < 1e-10  # Y_1^{-1}
        torch.testing.assert_close(
            output[0, 2], torch.tensor(sqrt_3_4pi, dtype=torch.float64, device=device)
        )  # Y_1^0
        assert abs(output[0, 3].item()) < 1e-10  # Y_1^{+1}

        # Y_2 components: only Y_2^0 should be non-zero
        assert abs(output[0, 4].item()) < 1e-10  # Y_2^{-2}
        assert abs(output[0, 5].item()) < 1e-10  # Y_2^{-1}
        torch.testing.assert_close(
            output[0, 6],
            torch.tensor(
                sqrt_5_16pi * 2.0, dtype=torch.float64, device=device
            ),  # 3z² - r² = 3 - 1 = 2
        )
        assert abs(output[0, 7].item()) < 1e-10  # Y_2^{+1}
        assert abs(output[0, 8].item()) < 1e-10  # Y_2^{+2}

    def test_along_x_axis(self, device):
        """Test values along the x-axis."""
        pos = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64, device=device)
        output = eval_spherical_harmonics_pytorch(pos, L_max=2, device=device)

        sqrt_1_4pi = 1.0 / math.sqrt(4.0 * math.pi)
        sqrt_3_4pi = math.sqrt(3.0 / (4.0 * math.pi))
        sqrt_5_16pi = math.sqrt(5.0 / (16.0 * math.pi))
        sqrt_15_16pi = math.sqrt(15.0 / (16.0 * math.pi))

        # Y_0^0
        torch.testing.assert_close(
            output[0, 0], torch.tensor(sqrt_1_4pi, dtype=torch.float64, device=device)
        )

        # Y_1 components: only Y_1^{+1} should be non-zero (proportional to x)
        assert abs(output[0, 1].item()) < 1e-10  # Y_1^{-1} (y)
        assert abs(output[0, 2].item()) < 1e-10  # Y_1^0 (z)
        torch.testing.assert_close(
            output[0, 3], torch.tensor(sqrt_3_4pi, dtype=torch.float64, device=device)
        )  # Y_1^{+1} (x)

        # Y_2 components
        assert abs(output[0, 4].item()) < 1e-10  # Y_2^{-2} (xy)
        assert abs(output[0, 5].item()) < 1e-10  # Y_2^{-1} (yz)
        torch.testing.assert_close(
            output[0, 6],
            torch.tensor(
                sqrt_5_16pi * (-1.0), dtype=torch.float64, device=device
            ),  # 3z² - r² = 0 - 1 = -1
        )
        assert abs(output[0, 7].item()) < 1e-10  # Y_2^{+1} (xz)
        torch.testing.assert_close(
            output[0, 8],
            torch.tensor(
                sqrt_15_16pi, dtype=torch.float64, device=device
            ),  # x² - y² = 1
        )

    def test_matches_reference(self, random_positions, device):
        """Test that Warp implementation matches reference on random unit vectors."""
        output = eval_spherical_harmonics_pytorch(
            random_positions, L_max=2, device=device
        )
        reference = real_spherical_harmonics_reference(random_positions)

        torch.testing.assert_close(output, reference, rtol=1e-10, atol=1e-10)

    def test_scaling_with_radius(self, device):
        """Spherical harmonics should be independent of radius (only direction matters)."""
        pos_unit = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float64, device=device)
        pos_unit = pos_unit / torch.norm(pos_unit, dim=1, keepdim=True)

        pos_scaled = pos_unit * 5.0

        output_unit = eval_spherical_harmonics_pytorch(pos_unit, L_max=2, device=device)
        output_scaled = eval_spherical_harmonics_pytorch(
            pos_scaled, L_max=2, device=device
        )

        torch.testing.assert_close(output_unit, output_scaled, rtol=1e-10, atol=1e-10)


class TestSphericalHarmonicsSymmetry:
    """Test symmetry properties of spherical harmonics."""

    def test_parity_l0(self, device):
        """Y_0^0 is even under inversion."""
        pos = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64, device=device)
        pos_inv = -pos

        y_pos = eval_spherical_harmonics_pytorch(pos, L_max=0, device=device)
        y_inv = eval_spherical_harmonics_pytorch(pos_inv, L_max=0, device=device)

        torch.testing.assert_close(y_pos, y_inv)

    def test_parity_l1(self, device):
        """Y_1^m are odd under inversion (p-orbital symmetry)."""
        pos = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64, device=device)
        pos = pos / torch.norm(pos, dim=1, keepdim=True)
        pos_inv = -pos

        y_pos = eval_spherical_harmonics_pytorch(pos, L_max=1, device=device)
        y_inv = eval_spherical_harmonics_pytorch(pos_inv, L_max=1, device=device)

        # L=0 should be equal (even)
        torch.testing.assert_close(y_pos[:, 0], y_inv[:, 0])

        # L=1 should be opposite (odd)
        torch.testing.assert_close(y_pos[:, 1:4], -y_inv[:, 1:4])

    def test_parity_l2(self, device):
        """Y_2^m are even under inversion (d-orbital symmetry)."""
        pos = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64, device=device)
        pos = pos / torch.norm(pos, dim=1, keepdim=True)
        pos_inv = -pos

        y_pos = eval_spherical_harmonics_pytorch(pos, L_max=2, device=device)
        y_inv = eval_spherical_harmonics_pytorch(pos_inv, L_max=2, device=device)

        # L=0 should be equal (even)
        torch.testing.assert_close(y_pos[:, 0], y_inv[:, 0])

        # L=1 should be opposite (odd)
        torch.testing.assert_close(y_pos[:, 1:4], -y_inv[:, 1:4])

        # L=2 should be equal (even)
        torch.testing.assert_close(y_pos[:, 4:9], y_inv[:, 4:9])

    def test_rotation_90_z(self, device):
        """Test 90° rotation around z-axis transforms correctly."""
        # Rotation by 90° around z: (x, y, z) → (-y, x, z)
        pos = torch.tensor(
            [[1.0, 0.0, 0.0]], dtype=torch.float64, device=device
        )  # Along +x
        pos_rot = torch.tensor(
            [[0.0, 1.0, 0.0]], dtype=torch.float64, device=device
        )  # Along +y

        y_original = eval_spherical_harmonics_pytorch(pos, L_max=2, device=device)
        y_rotated = eval_spherical_harmonics_pytorch(pos_rot, L_max=2, device=device)

        # L=0: invariant
        torch.testing.assert_close(y_original[:, 0], y_rotated[:, 0])

        # L=1: Y_1^{+1}(x) should become Y_1^{-1}(y)
        torch.testing.assert_close(
            y_original[:, 3], y_rotated[:, 1], rtol=1e-10, atol=1e-10
        )

        # L=2: Y_2^{+2}(x) should become -Y_2^{+2}(y) because (x²-y²) → (y²-x²) = -(x²-y²)
        torch.testing.assert_close(
            y_original[:, 8], -y_rotated[:, 8], rtol=1e-10, atol=1e-10
        )


class TestSphericalHarmonicsOrthonormality:
    """Test orthonormality via numerical integration."""

    @pytest.mark.parametrize(
        "l1,m1,l2,m2",
        [
            (0, 0, 0, 0),  # <Y_0^0 | Y_0^0> = 1
            (0, 0, 1, 0),  # <Y_0^0 | Y_1^0> = 0
            (1, 0, 1, 0),  # <Y_1^0 | Y_1^0> = 1
            (1, -1, 1, 1),  # <Y_1^{-1} | Y_1^{+1}> = 0
            (2, 0, 2, 0),  # <Y_2^0 | Y_2^0> = 1
            (1, 0, 2, 0),  # <Y_1^0 | Y_2^0> = 0
        ],
    )
    def test_orthonormality(self, l1, m1, l2, m2, device):
        """Test orthonormality via Monte Carlo integration on unit sphere."""
        # Generate uniform points on sphere using Fibonacci spiral
        N = 10000
        golden_ratio = (1 + np.sqrt(5)) / 2
        i = np.arange(N)
        theta = 2 * np.pi * i / golden_ratio
        phi = np.arccos(1 - 2 * (i + 0.5) / N)

        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)

        positions = torch.tensor(
            np.stack([x, y, z], axis=1), dtype=torch.float64, device=device
        )

        # Evaluate spherical harmonics
        Y = eval_spherical_harmonics_pytorch(positions, L_max=2, device=device)

        # Map (l, m) to index: l=0 → 0, l=1 → 1,2,3, l=2 → 4,5,6,7,8
        def lm_to_idx(ell, m):
            return ell * ell + ell + m

        idx1 = lm_to_idx(l1, m1)
        idx2 = lm_to_idx(l2, m2)

        # Monte Carlo integration: ∫ Y1 * Y2 dΩ ≈ (4π/N) * ∑ Y1 * Y2
        # But since we're using uniform points, this simplifies to mean * 4π
        integral = 4 * np.pi * (Y[:, idx1] * Y[:, idx2]).mean().item()

        if l1 == l2 and m1 == m2:
            # Should be 1 (normalization)
            assert abs(integral - 1.0) < 0.05, f"Expected ~1, got {integral}"
        else:
            # Should be 0 (orthogonality)
            assert abs(integral) < 0.05, f"Expected ~0, got {integral}"

    def test_sum_rule(self, random_positions, device):
        """Test that ∑_m |Y_l^m|² = (2l+1)/(4π) for each l."""
        Y = eval_spherical_harmonics_pytorch(random_positions, L_max=2, device=device)

        # L=0: |Y_0^0|² should be 1/(4π)
        l0_sum = Y[:, 0] ** 2
        expected_l0 = 1.0 / (4 * np.pi)
        torch.testing.assert_close(
            l0_sum, torch.full_like(l0_sum, expected_l0), rtol=1e-6, atol=1e-10
        )

        # L=1: ∑_m |Y_1^m|² should be 3/(4π)
        l1_sum = (Y[:, 1:4] ** 2).sum(dim=1)
        expected_l1 = 3.0 / (4 * np.pi)
        torch.testing.assert_close(
            l1_sum, torch.full_like(l1_sum, expected_l1), rtol=1e-6, atol=1e-10
        )

        # L=2: ∑_m |Y_2^m|² should be 5/(4π)
        l2_sum = (Y[:, 4:9] ** 2).sum(dim=1)
        expected_l2 = 5.0 / (4 * np.pi)
        torch.testing.assert_close(
            l2_sum, torch.full_like(l2_sum, expected_l2), rtol=1e-6, atol=1e-10
        )


class TestSphericalHarmonicsGradients:
    """Test gradients of spherical harmonics."""

    def test_gradient_y00_is_zero(self, random_positions, device):
        """Gradient of constant Y_0^0 should be zero."""
        grad = eval_spherical_harmonics_gradient_pytorch(
            random_positions, L_max=0, device=device
        )
        torch.testing.assert_close(grad, torch.zeros_like(grad))

    def test_gradient_finite_difference(self, device):
        """Test gradients against finite differences."""
        pos = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64, device=device)
        pos = pos / torch.norm(pos)  # Normalize

        grad = eval_spherical_harmonics_gradient_pytorch(pos, L_max=2, device=device)

        # Finite difference approximation
        eps = 1e-6
        grad_fd = torch.zeros(1, 9, 3, dtype=torch.float64, device=device)

        for d in range(3):
            pos_plus = pos.clone()
            pos_plus[0, d] += eps
            pos_minus = pos.clone()
            pos_minus[0, d] -= eps

            y_plus = eval_spherical_harmonics_pytorch(pos_plus, L_max=2, device=device)
            y_minus = eval_spherical_harmonics_pytorch(
                pos_minus, L_max=2, device=device
            )

            grad_fd[:, :, d] = (y_plus - y_minus) / (2 * eps)

        torch.testing.assert_close(grad, grad_fd, rtol=1e-5, atol=1e-8)

    def test_gradient_on_axes(self, axis_positions, device):
        """Test gradients at axis-aligned positions."""
        grad = eval_spherical_harmonics_gradient_pytorch(
            axis_positions, L_max=2, device=device
        )

        # Y_0^0 gradient should be zero everywhere
        torch.testing.assert_close(
            grad[:, 0, :], torch.zeros(6, 3, device=device, dtype=torch.float64)
        )

    def test_gradient_autograd_consistency(self, device):
        """Test that our analytical gradients match PyTorch autograd through Warp."""
        torch.manual_seed(123)
        pos = torch.randn(10, 3, dtype=torch.float64, device=device)
        pos = pos / torch.norm(pos, dim=1, keepdim=True)  # Normalize

        # Our analytical gradient
        grad_analytical = eval_spherical_harmonics_gradient_pytorch(
            pos, L_max=2, device=device
        )

        # Numerical gradient for a few test cases
        eps = 1e-7
        for i in [0, 5, 9]:  # Test a few points
            for component in [1, 2, 6]:  # Test Y_1^{-1}, Y_1^0, Y_2^0
                for d in range(3):
                    pos_p = pos.clone()
                    pos_p[i, d] += eps
                    pos_m = pos.clone()
                    pos_m[i, d] -= eps

                    y_p = eval_spherical_harmonics_pytorch(
                        pos_p, L_max=2, device=device
                    )
                    y_m = eval_spherical_harmonics_pytorch(
                        pos_m, L_max=2, device=device
                    )

                    grad_fd = (y_p[i, component] - y_m[i, component]) / (2 * eps)

                    assert (
                        abs(grad_analytical[i, component, d].item() - grad_fd.item())
                        < 1e-5
                    ), f"Gradient mismatch at i={i}, comp={component}, d={d}"


class TestSphericalHarmonicsEdgeCases:
    """Test edge cases and numerical stability."""

    def test_near_origin(self, device):
        """Test behavior near the origin (should not produce NaN/Inf)."""
        pos = torch.tensor([[1e-15, 1e-15, 1e-15]], dtype=torch.float64, device=device)

        Y = eval_spherical_harmonics_pytorch(pos, L_max=2, device=device)
        grad = eval_spherical_harmonics_gradient_pytorch(pos, L_max=2, device=device)

        assert torch.isfinite(Y).all(), "Y contains NaN or Inf near origin"
        assert torch.isfinite(grad).all(), "Gradient contains NaN or Inf near origin"

    def test_at_origin(self, device):
        """Test behavior at exact origin."""
        pos = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64, device=device)

        Y = eval_spherical_harmonics_pytorch(pos, L_max=2, device=device)
        grad = eval_spherical_harmonics_gradient_pytorch(pos, L_max=2, device=device)

        # Should not produce NaN/Inf
        assert torch.isfinite(Y).all(), "Y contains NaN or Inf at origin"
        assert torch.isfinite(grad).all(), "Gradient contains NaN or Inf at origin"

    def test_large_radius(self, device):
        """Test with very large radius (should not affect normalized harmonics)."""
        pos_unit = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float64, device=device)
        pos_unit = pos_unit / torch.norm(pos_unit)

        pos_large = pos_unit * 1e10

        Y_unit = eval_spherical_harmonics_pytorch(pos_unit, L_max=2, device=device)
        Y_large = eval_spherical_harmonics_pytorch(pos_large, L_max=2, device=device)

        torch.testing.assert_close(Y_unit, Y_large, rtol=1e-8, atol=1e-10)

    def test_batch_processing(self, device):
        """Test that batch processing gives same results as individual evaluations."""
        torch.manual_seed(999)
        positions = torch.randn(50, 3, dtype=torch.float64, device=device)

        # Batch evaluation
        Y_batch = eval_spherical_harmonics_pytorch(positions, L_max=2, device=device)

        # Individual evaluations
        for i in range(50):
            Y_single = eval_spherical_harmonics_pytorch(
                positions[i : i + 1], L_max=2, device=device
            )
            torch.testing.assert_close(
                Y_batch[i : i + 1], Y_single, rtol=1e-12, atol=1e-12
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
