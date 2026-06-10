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

r"""
Test Suite for Regular and Irregular Solid Harmonics
====================================================

Covers the L=0 / L=1 implementations in :mod:`nvalchemiops.math.solid_harmonics`.

The solid harmonics here use the bare convention
:math:`R_l^m = r^l \cdot Y_l^m(\hat{r})` and
:math:`I_l^m = Y_l^m(\hat{r}) / r^{l+1}`. Tests fall into three buckets:

* **Value checks** — closed-form Cartesian expressions for each L.
* **Identity checks** — ``R_l^m / I_l^m = r^{2l+1}`` and ``||R_1||^2 = (3/4π) · r²``.
* **Rotation equivariance** — rotating the input rotates the ``m``-block in the
  expected way, i.e. ``R_1`` behaves like a vector under rotation.
"""

from __future__ import annotations

import math

import pytest
import torch

from nvalchemiops.torch.math.solid_harmonics import (
    eval_irregular_solid_harmonics_pytorch,
    eval_regular_solid_harmonics_pytorch,
)

Y00 = 1.0 / math.sqrt(4.0 * math.pi)
Y1C = math.sqrt(3.0 / (4.0 * math.pi))


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


# =============================================================================
# Shape and API sanity
# =============================================================================


class TestShapesAndValidation:
    """Output shapes and input validation for both kernels."""

    def test_regular_l0_shape(self, device):
        pos = torch.randn(7, 3, dtype=torch.float64, device=device)
        out = eval_regular_solid_harmonics_pytorch(pos, max_L=0)
        assert out.shape == (7, 1)
        assert out.dtype == torch.float64

    def test_regular_l1_shape(self, device):
        pos = torch.randn(5, 3, dtype=torch.float64, device=device)
        out = eval_regular_solid_harmonics_pytorch(pos, max_L=1)
        assert out.shape == (5, 4)

    def test_irregular_l0_shape(self, device):
        pos = torch.rand(4, 3, dtype=torch.float64, device=device) + 1.0
        out = eval_irregular_solid_harmonics_pytorch(pos, max_L=0)
        assert out.shape == (4, 1)

    def test_irregular_l1_shape(self, device):
        pos = torch.rand(6, 3, dtype=torch.float64, device=device) + 1.0
        out = eval_irregular_solid_harmonics_pytorch(pos, max_L=1)
        assert out.shape == (6, 4)

    @pytest.mark.parametrize("bad_L", [-1, 2, 3, 4])
    def test_regular_rejects_unsupported_max_L(self, device, bad_L):
        pos = torch.randn(2, 3, dtype=torch.float64, device=device)
        with pytest.raises(ValueError, match="max_L must be 0 or 1"):
            eval_regular_solid_harmonics_pytorch(pos, max_L=bad_L)

    @pytest.mark.parametrize("bad_L", [-1, 2, 3, 4])
    def test_irregular_rejects_unsupported_max_L(self, device, bad_L):
        pos = torch.randn(2, 3, dtype=torch.float64, device=device) + 1.0
        with pytest.raises(ValueError, match="max_L must be 0 or 1"):
            eval_irregular_solid_harmonics_pytorch(pos, max_L=bad_L)


# =============================================================================
# Regular solid harmonics: closed-form values
# =============================================================================


class TestRegularSolidHarmonicsValues:
    r"""Closed-form checks for :math:`R_l^m`."""

    def test_R00_equals_Y00_everywhere(self, device):
        """R_0^0 is a constant equal to Y_0^0, independent of r."""
        pos = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-4.0, 0.5, -7.0]],
            dtype=torch.float64,
            device=device,
        )
        out = eval_regular_solid_harmonics_pytorch(pos, max_L=0)
        expected = torch.full((3, 1), Y00, dtype=torch.float64, device=device)
        torch.testing.assert_close(out, expected, rtol=1e-15, atol=1e-15)

    def test_R1_is_Cartesian_in_yzx_order(self, device):
        r"""R_1^m = \sqrt{3/(4\pi)} \cdot (y, z, x) for m = -1, 0, +1."""
        pos = torch.tensor(
            [[1.0, 2.0, 3.0], [-5.0, 4.0, -0.25]],
            dtype=torch.float64,
            device=device,
        )
        out = eval_regular_solid_harmonics_pytorch(pos, max_L=1)
        expected = torch.tensor(
            [
                [Y00, Y1C * 2.0, Y1C * 3.0, Y1C * 1.0],
                [Y00, Y1C * 4.0, Y1C * (-0.25), Y1C * (-5.0)],
            ],
            dtype=torch.float64,
            device=device,
        )
        torch.testing.assert_close(out, expected, rtol=1e-15, atol=1e-15)

    def test_R1_vanishes_at_origin(self, device):
        """R_1^m(0) = 0 for all m since the radial factor is r^1."""
        pos = torch.zeros((1, 3), dtype=torch.float64, device=device)
        out = eval_regular_solid_harmonics_pytorch(pos, max_L=1)
        # R_0^0 is still Y_0^0 at the origin.
        assert out[0, 0].item() == pytest.approx(Y00, rel=1e-15)
        torch.testing.assert_close(
            out[0, 1:],
            torch.zeros(3, dtype=torch.float64, device=device),
            rtol=0,
            atol=1e-30,
        )

    def test_R1_norm_squared_scales_as_r_squared(self, device):
        r"""||R_1(r)||^2 = (3/(4\pi)) \cdot r^2 — rotation invariant."""
        torch.manual_seed(17)
        pos = torch.randn(32, 3, dtype=torch.float64, device=device)
        out = eval_regular_solid_harmonics_pytorch(pos, max_L=1)
        r_sq = (pos**2).sum(dim=-1)
        expected = (3.0 / (4.0 * math.pi)) * r_sq
        got = (out[:, 1:] ** 2).sum(dim=-1)
        torch.testing.assert_close(got, expected, rtol=1e-14, atol=1e-14)


# =============================================================================
# Irregular solid harmonics: closed-form values
# =============================================================================


class TestIrregularSolidHarmonicsValues:
    r"""Closed-form checks for :math:`I_l^m` away from the origin."""

    def test_I00_is_Y00_over_r(self, device):
        """I_0^0(r) = Y_0^0 / |r|."""
        pos = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 2.5, 0.0], [1.0, 2.0, 2.0]],
            dtype=torch.float64,
            device=device,
        )
        out = eval_irregular_solid_harmonics_pytorch(pos, max_L=0)
        r_norm = (pos**2).sum(dim=-1).sqrt()
        expected = (Y00 / r_norm).unsqueeze(-1)
        torch.testing.assert_close(out, expected, rtol=1e-14, atol=1e-14)

    def test_I1_is_Y1_over_r_cubed(self, device):
        r"""I_1^m(r) = \sqrt{3/(4\pi)} \cdot (y, z, x) / r^3."""
        pos = torch.tensor(
            [[1.0, 2.0, 3.0], [-0.5, 0.7, -1.2]],
            dtype=torch.float64,
            device=device,
        )
        out = eval_irregular_solid_harmonics_pytorch(pos, max_L=1)
        r_norm = (pos**2).sum(dim=-1).sqrt()
        inv_r3 = (1.0 / r_norm**3).unsqueeze(-1)
        permuted = pos[:, [1, 2, 0]]  # (y, z, x)
        expected_l1 = Y1C * permuted * inv_r3
        expected_l0 = (Y00 / r_norm).unsqueeze(-1)
        expected = torch.cat([expected_l0, expected_l1], dim=-1)
        torch.testing.assert_close(out, expected, rtol=1e-14, atol=1e-14)

    def test_I_on_unit_sphere_matches_spherical_harmonic(self, device):
        """On the unit sphere, I_l^m(r̂) = Y_l^m(r̂) (since 1/r^{l+1} = 1)."""
        torch.manual_seed(23)
        pos = torch.randn(10, 3, dtype=torch.float64, device=device)
        pos = pos / pos.norm(dim=-1, keepdim=True)
        out = eval_irregular_solid_harmonics_pytorch(pos, max_L=1)
        # On unit sphere, irregular values equal bare Y_l^m components:
        # (Y00, Y1C * y, Y1C * z, Y1C * x).
        expected = torch.stack(
            [
                torch.full((10,), Y00, dtype=torch.float64, device=device),
                Y1C * pos[:, 1],
                Y1C * pos[:, 2],
                Y1C * pos[:, 0],
            ],
            dim=-1,
        )
        torch.testing.assert_close(out, expected, rtol=1e-14, atol=1e-14)


# =============================================================================
# Regular / irregular connecting identities
# =============================================================================


class TestConnectingIdentities:
    """Identities that tie R and I together for the same r."""

    def test_R_over_I_equals_r_pow_2l_plus_1(self, device):
        r"""``R_l^m(r) / I_l^m(r) = r^{2l+1}`` for each L and any m."""
        torch.manual_seed(51)
        # Keep away from the origin so I_l^m is well-defined.
        pos = torch.randn(16, 3, dtype=torch.float64, device=device) + 3.0
        R = eval_regular_solid_harmonics_pytorch(pos, max_L=1)
        I_vals = eval_irregular_solid_harmonics_pytorch(pos, max_L=1)
        r_norm = (pos**2).sum(dim=-1).sqrt()
        # L=0 ratio -> r^1
        torch.testing.assert_close(
            R[:, 0] / I_vals[:, 0], r_norm, rtol=1e-13, atol=1e-13
        )
        # L=1 ratio (any m) -> r^3
        for m_idx in range(1, 4):
            torch.testing.assert_close(
                R[:, m_idx] / I_vals[:, m_idx], r_norm**3, rtol=1e-12, atol=1e-12
            )


# =============================================================================
# Rotation behaviour
# =============================================================================


def _rotation_matrix_z(theta: float) -> torch.Tensor:
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64
    )


class TestRotation:
    """Rotation equivariance of the L=1 block and invariance of scalars."""

    def test_R00_is_rotation_invariant(self, device):
        """R_0^0 is a scalar → invariant under rotation."""
        torch.manual_seed(71)
        pos = torch.randn(8, 3, dtype=torch.float64, device=device)
        rot = _rotation_matrix_z(0.37).to(device=device)
        rotated = pos @ rot.T
        r_before = eval_regular_solid_harmonics_pytorch(pos, max_L=0)
        r_after = eval_regular_solid_harmonics_pytorch(rotated, max_L=0)
        torch.testing.assert_close(r_before, r_after, rtol=1e-15, atol=1e-15)

    def test_R1_norm_is_rotation_invariant(self, device):
        r"""``||R_1||^2`` is rotation-invariant (equals ``(3/4\pi) \cdot r^2``)."""
        torch.manual_seed(73)
        pos = torch.randn(16, 3, dtype=torch.float64, device=device)
        rot = _rotation_matrix_z(1.1).to(device=device)
        rotated = pos @ rot.T
        r1_before = eval_regular_solid_harmonics_pytorch(pos, max_L=1)[:, 1:]
        r1_after = eval_regular_solid_harmonics_pytorch(rotated, max_L=1)[:, 1:]
        norm_before = (r1_before**2).sum(dim=-1)
        norm_after = (r1_after**2).sum(dim=-1)
        torch.testing.assert_close(norm_before, norm_after, rtol=1e-13, atol=1e-13)

    def test_I1_norm_is_rotation_invariant(self, device):
        r"""``||I_1||^2`` is rotation-invariant (equals ``(3/4\pi) / r^4``)."""
        torch.manual_seed(79)
        pos = torch.randn(16, 3, dtype=torch.float64, device=device) + 2.0
        rot = _rotation_matrix_z(0.81).to(device=device)
        rotated = pos @ rot.T
        i1_before = eval_irregular_solid_harmonics_pytorch(pos, max_L=1)[:, 1:]
        i1_after = eval_irregular_solid_harmonics_pytorch(rotated, max_L=1)[:, 1:]
        norm_before = (i1_before**2).sum(dim=-1)
        norm_after = (i1_after**2).sum(dim=-1)
        torch.testing.assert_close(norm_before, norm_after, rtol=1e-13, atol=1e-13)

    def test_R1_transforms_as_cartesian_vector(self, device):
        r"""``R_1`` is equivalent to a Cartesian vector under the ``(y, z, x)`` permutation.

        Concretely: under a rotation ``R`` acting in Cartesian ``(x, y, z)``
        space, the ``R_1`` components in ``(m = -1, 0, +1)`` order transform by
        ``P R P^T`` where ``P`` is the permutation ``(y, z, x) <- (x, y, z)``.
        """
        torch.manual_seed(83)
        pos = torch.randn(6, 3, dtype=torch.float64, device=device)
        rot = _rotation_matrix_z(0.5).to(device=device)
        rotated = pos @ rot.T
        r1_before = eval_regular_solid_harmonics_pytorch(pos, max_L=1)[:, 1:]
        r1_after_direct = eval_regular_solid_harmonics_pytorch(rotated, max_L=1)[:, 1:]
        # Permutation P: physics (x,y,z) -> m-order (y,z,x).
        P = torch.tensor(
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
            dtype=torch.float64,
            device=device,
        )
        R_in_m_order = P @ rot @ P.T
        r1_after_transformed = r1_before @ R_in_m_order.T
        torch.testing.assert_close(
            r1_after_direct, r1_after_transformed, rtol=1e-13, atol=1e-13
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
