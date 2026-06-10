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
Test Suite for Gaussian Type Orbital (GTO) PyTorch bindings
===========================================================

Exercises the host-side PyTorch wrappers and normalization scaffolding in
:mod:`nvalchemiops.torch.math.gto`, which drive the Warp ``@wp.kernel``
evaluators defined in :mod:`nvalchemiops.math.gto`.

Contents:

1. Real-space density evaluators (:func:`eval_gto_density_pytorch`)
2. Fourier-transform evaluators (:func:`eval_gto_fourier_pytorch`)
3. GTO normalization conventions (:class:`NormMode`, :func:`inv_cl`,
   :func:`inv_cl_table`)

Mathematical Reference
----------------------

GTO density: φ_{l,m}(r, σ) = N · Y_l^m(r̂) · exp(-r²/(2σ²))

Key properties tested:
- ∫ φ_{0,0}(r) d³r = 1 (normalization for monopole)
- ∫ φ_{l,m}(r) d³r = 0 for l > 0 (odd parity integrands)
- φ̂(k) = (i/2)^l · √(4π) · Y_l^m(k̂) · exp(-k²σ²/2) (Fourier transform)
- Parseval's theorem: ∫|φ|²dr = (1/2π)³ ∫|φ̂|²dk
"""

import math

import pytest
import torch

from nvalchemiops.torch.math.gto import (
    NormMode,
    eval_gto_density_pytorch,
    eval_gto_fourier_pytorch,
    inv_cl,
    inv_cl_table,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture(scope="class")
def device():
    """Get the compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


@pytest.fixture(scope="class")
def random_positions(device):
    """Generate random positions for testing."""
    torch.manual_seed(42)
    N = 100
    # Random positions in a box of size 5 centered at origin
    positions = (torch.rand(N, 3, dtype=torch.float64, device=device) - 0.5) * 5.0
    return positions


@pytest.fixture(scope="class")
def grid_positions(device):
    """Generate a 3D grid of positions for integration."""
    # Create a cubic grid from -5 to 5 with spacing 0.5
    x = torch.linspace(-5, 5, 21, dtype=torch.float64, device=device)
    y = torch.linspace(-5, 5, 21, dtype=torch.float64, device=device)
    z = torch.linspace(-5, 5, 21, dtype=torch.float64, device=device)
    xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")
    positions = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)
    spacing = 0.5
    return positions, spacing


# =============================================================================
# Reference Implementation
# =============================================================================


def gto_density_reference(
    positions: torch.Tensor, sigma: float, L_max: int = 2
) -> torch.Tensor:
    """Reference implementation of GTO densities.

    Parameters
    ----------
    positions : torch.Tensor
        Positions [N, 3].
    sigma : float
        Gaussian width.
    L_max : int
        Maximum angular momentum.

    Returns
    -------
    torch.Tensor
        GTO densities [N, num_components].
    """
    N = positions.shape[0]
    device = positions.device

    r = positions
    r2 = (r**2).sum(dim=1)
    r_norm = torch.sqrt(r2 + 1e-30)

    x, y, z = r[:, 0], r[:, 1], r[:, 2]

    # Normalization
    sqrt_4pi = math.sqrt(4.0 * math.pi)
    twopi_3_2 = (2.0 * math.pi) ** 1.5
    norm = sqrt_4pi / (twopi_3_2 * sigma**3)

    # Gaussian factor
    gauss = torch.exp(-r2 / (2.0 * sigma**2))

    # Spherical harmonic coefficients
    y00 = 1.0 / math.sqrt(4.0 * math.pi)
    y1_coeff = math.sqrt(3.0 / (4.0 * math.pi))

    num_components = {0: 1, 1: 4, 2: 9}[L_max]
    output = torch.zeros((N, num_components), dtype=torch.float64, device=device)

    prefactor = norm * gauss

    # L=0
    output[:, 0] = prefactor * y00

    if L_max >= 1:
        # L=1: Y_1^m = C * coord / r
        output[:, 1] = prefactor * y1_coeff * y / r_norm  # Y_1^{-1}
        output[:, 2] = prefactor * y1_coeff * z / r_norm  # Y_1^0
        output[:, 3] = prefactor * y1_coeff * x / r_norm  # Y_1^{+1}

    if L_max >= 2:
        # L=2 coefficients
        y2_m2 = math.sqrt(15.0 / (4.0 * math.pi))
        y2_m1 = math.sqrt(15.0 / (4.0 * math.pi))
        y2_0 = math.sqrt(5.0 / (16.0 * math.pi))
        y2_p1 = math.sqrt(15.0 / (4.0 * math.pi))
        y2_p2 = math.sqrt(15.0 / (16.0 * math.pi))

        r2_safe = r2 + 1e-30
        output[:, 4] = prefactor * y2_m2 * x * y / r2_safe  # Y_2^{-2}
        output[:, 5] = prefactor * y2_m1 * y * z / r2_safe  # Y_2^{-1}
        output[:, 6] = prefactor * y2_0 * (3 * z**2 - r2) / r2_safe  # Y_2^0
        output[:, 7] = prefactor * y2_p1 * x * z / r2_safe  # Y_2^{+1}
        output[:, 8] = prefactor * y2_p2 * (x**2 - y**2) / r2_safe  # Y_2^{+2}

    return output


# =============================================================================
# Test Classes
# =============================================================================


class TestGTODensityAPI:
    """Test the basic API and shapes."""

    def test_l0_shape(self, random_positions, device):
        """Test L=0 output shape."""
        output = eval_gto_density_pytorch(
            random_positions, sigma=1.0, L_max=0, device=device
        )
        assert output.shape == (100, 1)

    def test_l1_shape(self, random_positions, device):
        """Test L=1 output shape."""
        output = eval_gto_density_pytorch(
            random_positions, sigma=1.0, L_max=1, device=device
        )
        assert output.shape == (100, 4)

    def test_l2_shape(self, random_positions, device):
        """Test L=2 output shape."""
        output = eval_gto_density_pytorch(
            random_positions, sigma=1.0, L_max=2, device=device
        )
        assert output.shape == (100, 9)

    def test_single_position(self, device):
        """Test with a single position."""
        pos = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64, device=device)
        output = eval_gto_density_pytorch(pos, sigma=1.0, L_max=2, device=device)
        assert output.shape == (1, 9)

    def test_different_sigma(self, device):
        """Test with different sigma values."""
        pos = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float64, device=device)

        # Smaller sigma = more localized
        out_small = eval_gto_density_pytorch(pos, sigma=0.5, L_max=0, device=device)
        out_large = eval_gto_density_pytorch(pos, sigma=2.0, L_max=0, device=device)

        # At r=√3 ≈ 1.73, smaller sigma should give smaller density
        assert out_small[0, 0].item() < out_large[0, 0].item()


class TestGTODensityValues:
    """Test GTO density values against reference."""

    def test_matches_reference(self, random_positions, device):
        """Test that Warp implementation matches reference."""
        sigma = 1.0
        output = eval_gto_density_pytorch(
            random_positions, sigma=sigma, L_max=2, device=device
        )
        reference = gto_density_reference(random_positions, sigma=sigma, L_max=2)

        torch.testing.assert_close(output, reference, rtol=1e-10, atol=1e-10)

    def test_at_origin(self, device):
        """Test density at the origin."""
        pos = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64, device=device)
        sigma = 1.0
        output = eval_gto_density_pytorch(pos, sigma=sigma, L_max=2, device=device)

        # L=0 at origin should have maximum density
        sqrt_4pi = math.sqrt(4.0 * math.pi)
        twopi_3_2 = (2.0 * math.pi) ** 1.5
        y00 = 1.0 / math.sqrt(4.0 * math.pi)
        norm = sqrt_4pi / (twopi_3_2 * sigma**3)
        expected_l0 = norm * y00  # gauss = 1 at origin

        assert abs(output[0, 0].item() - expected_l0) < 1e-10

        # L>0 components involve coord/r which is indeterminate at origin
        # but should be finite (due to EPSILON regularization)
        assert torch.isfinite(output).all()

    def test_exponential_decay(self, device):
        """Test that density decays exponentially with distance."""
        sigma = 1.0

        # Points at different distances along x-axis
        distances = torch.tensor(
            [1.0, 2.0, 3.0, 4.0], dtype=torch.float64, device=device
        )
        positions = torch.stack(
            [distances, torch.zeros_like(distances), torch.zeros_like(distances)], dim=1
        )

        output = eval_gto_density_pytorch(
            positions, sigma=sigma, L_max=0, device=device
        )

        # L=0 density should decay as exp(-r²/(2σ²))
        for i in range(len(distances) - 1):
            r1, r2 = distances[i].item(), distances[i + 1].item()
            expected_ratio = math.exp(-(r2**2 - r1**2) / (2 * sigma**2))
            actual_ratio = output[i + 1, 0].item() / output[i, 0].item()
            assert abs(actual_ratio - expected_ratio) < 1e-10


class TestGTONormalization:
    """Test normalization and integral properties."""

    def test_l0_integral(self, grid_positions, device):
        """Test that L=0 GTO integrates to 1.

        This is a numerical integration test using a grid.
        """
        positions, spacing = grid_positions
        sigma = 1.0

        output = eval_gto_density_pytorch(
            positions, sigma=sigma, L_max=0, device=device
        )

        # Integrate using trapezoidal rule (approximately)
        volume_element = spacing**3
        integral = output[:, 0].sum().item() * volume_element

        # Should be close to 1 (with some discretization error)
        # For σ=1, the Gaussian extends beyond our box, so expect ~0.95-1.0
        assert 0.9 < integral < 1.1, f"L=0 integral = {integral}, expected ~1.0"

    def test_l1_integral_is_zero(self, grid_positions, device):
        """Test that L=1 GTO integrates to 0 (by symmetry)."""
        positions, spacing = grid_positions
        sigma = 1.0

        output = eval_gto_density_pytorch(
            positions, sigma=sigma, L_max=1, device=device
        )

        volume_element = spacing**3

        # L=1 components should integrate to ~0 (odd functions)
        for m in range(3):
            integral = output[:, 1 + m].sum().item() * volume_element
            assert abs(integral) < 0.1, (
                f"L=1 m={m - 1} integral = {integral}, expected ~0"
            )

    def test_sigma_scaling(self, device):
        """Test that density scales correctly with sigma.

        At the origin, the L=0 density should scale as σ^{-3}.
        """
        pos = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64, device=device)

        sigma1, sigma2 = 1.0, 2.0
        density1 = eval_gto_density_pytorch(pos, sigma=sigma1, L_max=0, device=device)[
            0, 0
        ].item()
        density2 = eval_gto_density_pytorch(pos, sigma=sigma2, L_max=0, device=device)[
            0, 0
        ].item()

        # Ratio should be (σ2/σ1)^{-3} = 8
        expected_ratio = (sigma2 / sigma1) ** 3
        actual_ratio = density1 / density2

        assert abs(actual_ratio - expected_ratio) < 1e-10


class TestGTOFourier:
    """Test GTO Fourier transforms."""

    def test_fourier_shape(self, device):
        """Test Fourier transform output shapes."""
        k = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64, device=device
        )

        real, imag = eval_gto_fourier_pytorch(k, sigma=1.0, L_max=2, device=device)

        assert real.shape == (2, 9)
        assert imag.shape == (2, 9)

    def test_l0_fourier_is_real(self, device):
        """Test that L=0 Fourier transform is purely real."""
        torch.manual_seed(123)
        k = torch.randn(10, 3, dtype=torch.float64, device=device)

        real, imag = eval_gto_fourier_pytorch(k, sigma=1.0, L_max=0, device=device)

        # Imaginary part should be zero
        torch.testing.assert_close(imag[:, 0], torch.zeros_like(imag[:, 0]))

        # Real part should be exp(-k²σ²/2)
        k2 = (k**2).sum(dim=1)
        expected = torch.exp(-k2 * 1.0 / 2.0)
        torch.testing.assert_close(real[:, 0], expected)

    def test_l1_fourier_is_imaginary(self, device):
        """Test that L=1 Fourier transform is purely imaginary."""
        torch.manual_seed(456)
        k = torch.randn(10, 3, dtype=torch.float64, device=device)

        real, imag = eval_gto_fourier_pytorch(k, sigma=1.0, L_max=1, device=device)

        # Real parts of L=1 should be zero
        torch.testing.assert_close(real[:, 1:4], torch.zeros_like(real[:, 1:4]))

        # Imaginary parts should be non-zero (for generic k)
        assert imag[:, 1:4].abs().sum() > 0

    def test_l2_fourier_is_real(self, device):
        """Test that L=2 Fourier transform is purely real."""
        torch.manual_seed(789)
        k = torch.randn(10, 3, dtype=torch.float64, device=device)

        real, imag = eval_gto_fourier_pytorch(k, sigma=1.0, L_max=2, device=device)

        # Imaginary parts of L=2 should be zero
        torch.testing.assert_close(imag[:, 4:9], torch.zeros_like(imag[:, 4:9]))

        # Real parts should be non-zero (for generic k)
        assert real[:, 4:9].abs().sum() > 0

    def test_fourier_at_k_zero(self, device):
        """Test Fourier transform at k=0."""
        k = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64, device=device)
        sigma = 1.0

        real, imag = eval_gto_fourier_pytorch(k, sigma=sigma, L_max=2, device=device)

        # At k=0, L=0 should be exp(0) = 1
        assert abs(real[0, 0].item() - 1.0) < 1e-10
        assert abs(imag[0, 0].item()) < 1e-10

    def test_fourier_gaussian_decay(self, device):
        """Test that Fourier transform decays as Gaussian in k."""
        sigma = 1.0

        # k-vectors along x-axis at different magnitudes
        k_mags = torch.tensor([0.5, 1.0, 2.0, 3.0], dtype=torch.float64, device=device)
        k_vectors = torch.stack(
            [k_mags, torch.zeros_like(k_mags), torch.zeros_like(k_mags)], dim=1
        )

        real, _ = eval_gto_fourier_pytorch(
            k_vectors, sigma=sigma, L_max=0, device=device
        )

        # Should decay as exp(-k²σ²/2)
        expected = torch.exp(-(k_mags**2) * sigma**2 / 2.0)
        torch.testing.assert_close(real[:, 0], expected)


class TestGTOParseval:
    """Test Parseval's theorem for GTO Fourier transforms.

    ∫|φ(r)|² d³r = (1/(2π)³) ∫|φ̂(k)|² d³k
    """

    @pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0])
    def test_parseval_l0(self, sigma, device):
        """Test Parseval's theorem for L=0 GTO."""
        # Create grids in real and k-space
        # Real space: larger extent for larger sigma
        extent = 5 * sigma
        n_grid = 31
        spacing_r = 2 * extent / (n_grid - 1)

        r = torch.linspace(-extent, extent, n_grid, dtype=torch.float64, device=device)
        rr_x, rr_y, rr_z = torch.meshgrid(r, r, r, indexing="ij")
        positions = torch.stack([rr_x.flatten(), rr_y.flatten(), rr_z.flatten()], dim=1)

        # Real-space integral of |φ|²
        density = eval_gto_density_pytorch(
            positions, sigma=sigma, L_max=0, device=device
        )
        real_integral = (density[:, 0] ** 2).sum().item() * spacing_r**3

        # k-space: use appropriate extent
        k_extent = 5.0 / sigma  # Higher k needed for smaller sigma
        spacing_k = 2 * k_extent / (n_grid - 1)

        kx = torch.linspace(
            -k_extent, k_extent, n_grid, dtype=torch.float64, device=device
        )
        kk_x, kk_y, kk_z = torch.meshgrid(kx, kx, kx, indexing="ij")
        k_vectors = torch.stack([kk_x.flatten(), kk_y.flatten(), kk_z.flatten()], dim=1)

        # k-space integral of |φ̂|²
        real_part, imag_part = eval_gto_fourier_pytorch(
            k_vectors, sigma=sigma, L_max=0, device=device
        )
        fourier_mag_sq = real_part[:, 0] ** 2 + imag_part[:, 0] ** 2
        k_integral = fourier_mag_sq.sum().item() * spacing_k**3 / (2 * math.pi) ** 3

        # Should be approximately equal (within discretization error)
        rel_diff = abs(real_integral - k_integral) / max(real_integral, k_integral)
        assert rel_diff < 0.2, f"Parseval failed: real={real_integral}, k={k_integral}"


class TestGTOSymmetry:
    """Test symmetry properties of GTOs."""

    def test_l0_spherical_symmetry(self, device):
        """Test that L=0 GTO is spherically symmetric."""
        sigma = 1.0
        r_val = 2.0

        # Points at same distance but different directions
        positions = torch.tensor(
            [
                [r_val, 0.0, 0.0],
                [0.0, r_val, 0.0],
                [0.0, 0.0, r_val],
                [r_val / math.sqrt(3), r_val / math.sqrt(3), r_val / math.sqrt(3)],
            ],
            dtype=torch.float64,
            device=device,
        )

        output = eval_gto_density_pytorch(
            positions, sigma=sigma, L_max=0, device=device
        )

        # All L=0 values should be equal
        torch.testing.assert_close(output[:, 0], output[0, 0].expand(4))

    def test_l1_parity(self, device):
        """Test that L=1 GTOs have odd parity."""
        sigma = 1.0
        pos = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64, device=device)
        pos_neg = -pos

        output_pos = eval_gto_density_pytorch(pos, sigma=sigma, L_max=1, device=device)
        output_neg = eval_gto_density_pytorch(
            pos_neg, sigma=sigma, L_max=1, device=device
        )

        # L=0 should be even (same)
        torch.testing.assert_close(output_pos[:, 0], output_neg[:, 0])

        # L=1 should be odd (opposite)
        torch.testing.assert_close(output_pos[:, 1:4], -output_neg[:, 1:4])

    def test_l2_parity(self, device):
        """Test that L=2 GTOs have even parity."""
        sigma = 1.0
        pos = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64, device=device)
        pos_neg = -pos

        output_pos = eval_gto_density_pytorch(pos, sigma=sigma, L_max=2, device=device)
        output_neg = eval_gto_density_pytorch(
            pos_neg, sigma=sigma, L_max=2, device=device
        )

        # L=0 and L=2 should be even (same)
        torch.testing.assert_close(output_pos[:, 0], output_neg[:, 0])
        torch.testing.assert_close(output_pos[:, 4:9], output_neg[:, 4:9])

        # L=1 should be odd (opposite)
        torch.testing.assert_close(output_pos[:, 1:4], -output_neg[:, 1:4])


class TestGTOEdgeCases:
    """Test edge cases and numerical stability."""

    def test_small_sigma(self, device):
        """Test with very small sigma (highly localized)."""
        pos = torch.tensor([[0.1, 0.1, 0.1]], dtype=torch.float64, device=device)
        sigma = 0.01

        output = eval_gto_density_pytorch(pos, sigma=sigma, L_max=2, device=device)

        # Should be finite (though very small due to exponential decay)
        assert torch.isfinite(output).all()

    def test_large_sigma(self, device):
        """Test with very large sigma (delocalized)."""
        pos = torch.tensor([[10.0, 10.0, 10.0]], dtype=torch.float64, device=device)
        sigma = 10.0

        output = eval_gto_density_pytorch(pos, sigma=sigma, L_max=2, device=device)

        assert torch.isfinite(output).all()
        assert output[:, 0].item() > 0  # L=0 should be positive

    def test_large_distance(self, device):
        """Test at large distances where density should be very small."""
        pos = torch.tensor([[100.0, 0.0, 0.0]], dtype=torch.float64, device=device)
        sigma = 1.0

        output = eval_gto_density_pytorch(pos, sigma=sigma, L_max=0, device=device)

        # exp(-100²/(2*1²)) ≈ exp(-5000) ≈ 0
        assert math.isclose(output[0, 0], 0.0)

    def test_batch_consistency(self, device):
        """Test that batch processing gives same results as individual."""
        torch.manual_seed(999)
        positions = torch.randn(20, 3, dtype=torch.float64, device=device)
        sigma = 1.0

        # Batch evaluation
        batch_output = eval_gto_density_pytorch(
            positions, sigma=sigma, L_max=2, device=device
        )

        # Individual evaluations
        for i in range(20):
            single_output = eval_gto_density_pytorch(
                positions[i : i + 1], sigma=sigma, L_max=2, device=device
            )
            torch.testing.assert_close(batch_output[i : i + 1], single_output)


# =============================================================================
# Normalization Mode Tests
# =============================================================================


def _reference_get_cl_sigma(L: int, sigma: float, normalize: str) -> float:
    """Direct transcription of the customer ``get_Cl_sigma`` reference formula.

    Kept inline as a golden reference so the test has no external dependency.
    """
    if normalize == "none":
        return 1.0
    if normalize == "multipoles":
        l_dep = (
            (4.0 * math.pi / (2 * L + 1)) ** 0.5
            * 2 ** ((2 * L + 1) / 2)
            * math.gamma((2 * L + 3) / 2)
        )
        return 1.0 / (l_dep * sigma ** (2 * L + 3))
    if normalize == "receiver":
        l_dep = 2 ** ((L + 1) / 2) * math.gamma((L + 3) / 2)
        return 1.0 / (l_dep * sigma ** (L + 3))
    raise ValueError(normalize)


class TestNormMode:
    """Tests for :class:`NormMode` and the :func:`inv_cl` family."""

    def test_enum_values_stable(self):
        """Underlying int values must stay stable — they will be passed into Warp kernels."""
        assert int(NormMode.MULTIPOLES) == 0
        assert int(NormMode.RECEIVER) == 1
        assert int(NormMode.NONE) == 2

    def test_inv_cl_none_is_unity(self):
        """NONE mode returns 1.0 for any (sigma, L)."""
        for sigma in [0.1, 1.0, 5.0]:
            for L in [0, 1, 2, 3]:
                assert inv_cl(sigma, L, NormMode.NONE) == 1.0

    def test_inv_cl_multipoles_l0_analytic(self):
        """Closed-form check at L=0: inv_cl = 1 / (π·√2·σ³)."""
        for sigma in [0.3, 1.0, 2.7]:
            expected = 1.0 / (math.pi * math.sqrt(2.0) * sigma**3)
            assert inv_cl(sigma, 0, NormMode.MULTIPOLES) == pytest.approx(
                expected, rel=1e-14
            )

    def test_inv_cl_receiver_l0_analytic(self):
        """Closed-form check at L=0: inv_cl = √(2/π) / σ³."""
        for sigma in [0.3, 1.0, 2.7]:
            expected = math.sqrt(2.0 / math.pi) / sigma**3
            assert inv_cl(sigma, 0, NormMode.RECEIVER) == pytest.approx(
                expected, rel=1e-14
            )

    def test_inv_cl_matches_reference_formula(self):
        """Parity with the customer ``get_Cl_sigma`` reference across modes and L values."""
        for mode_name, mode in [
            ("multipoles", NormMode.MULTIPOLES),
            ("receiver", NormMode.RECEIVER),
            ("none", NormMode.NONE),
        ]:
            for sigma in [0.3, 1.0, 2.7]:
                for L in [0, 1, 2, 3]:
                    ref = _reference_get_cl_sigma(L, sigma, mode_name)
                    got = inv_cl(sigma, L, mode)
                    assert got == pytest.approx(ref, rel=1e-14), (
                        f"mode={mode_name}, sigma={sigma}, L={L}: "
                        f"expected {ref}, got {got}"
                    )

    def test_inv_cl_sigma_scaling_multipoles(self):
        """inv_cl(2σ, L, MULTIPOLES) / inv_cl(σ, L, MULTIPOLES) = 2^-(2L+3)."""
        sigma = 0.7
        for L in [0, 1, 2, 3]:
            ratio = inv_cl(2 * sigma, L, NormMode.MULTIPOLES) / inv_cl(
                sigma, L, NormMode.MULTIPOLES
            )
            expected = 2.0 ** (-(2 * L + 3))
            assert ratio == pytest.approx(expected, rel=1e-14)

    def test_inv_cl_sigma_scaling_receiver(self):
        """inv_cl(2σ, L, RECEIVER) / inv_cl(σ, L, RECEIVER) = 2^-(L+3)."""
        sigma = 0.7
        for L in [0, 1, 2, 3]:
            ratio = inv_cl(2 * sigma, L, NormMode.RECEIVER) / inv_cl(
                sigma, L, NormMode.RECEIVER
            )
            expected = 2.0 ** (-(L + 3))
            assert ratio == pytest.approx(expected, rel=1e-14)

    def test_inv_cl_accepts_int_mode(self):
        """Passing a raw int for mode should be equivalent to the enum."""
        sigma = 1.1
        for L in [0, 1, 2]:
            assert inv_cl(sigma, L, 0) == inv_cl(sigma, L, NormMode.MULTIPOLES)
            assert inv_cl(sigma, L, 1) == inv_cl(sigma, L, NormMode.RECEIVER)
            assert inv_cl(sigma, L, 2) == inv_cl(sigma, L, NormMode.NONE)

    @pytest.mark.parametrize("bad_sigma", [0.0, -0.5])
    def test_inv_cl_rejects_nonpositive_sigma(self, bad_sigma):
        with pytest.raises(ValueError, match="sigma must be positive"):
            inv_cl(bad_sigma, 0, NormMode.MULTIPOLES)

    def test_inv_cl_rejects_negative_L(self):
        with pytest.raises(ValueError, match="L must be non-negative"):
            inv_cl(1.0, -1, NormMode.MULTIPOLES)

    def test_inv_cl_rejects_unknown_mode(self):
        with pytest.raises(ValueError):
            inv_cl(1.0, 0, 99)


class TestInvClTable:
    """Tests for :func:`inv_cl_table`."""

    def test_length_is_max_L_plus_one(self):
        for max_L in [0, 1, 2, 3]:
            table = inv_cl_table(1.0, max_L, NormMode.MULTIPOLES)
            assert len(table) == max_L + 1

    def test_entries_match_scalar_inv_cl(self):
        sigma = 1.3
        max_L = 3
        for mode in [NormMode.MULTIPOLES, NormMode.RECEIVER, NormMode.NONE]:
            table = inv_cl_table(sigma, max_L, mode)
            for L in range(max_L + 1):
                assert table[L] == inv_cl(sigma, L, mode)

    def test_rejects_negative_max_L(self):
        with pytest.raises(ValueError, match="max_L must be non-negative"):
            inv_cl_table(1.0, -1, NormMode.MULTIPOLES)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
