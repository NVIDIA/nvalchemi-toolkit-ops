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

"""Tests for :mod:`nvalchemiops.torch.math.gto_self_overlap`.

Structural tests exercise layout, input validation, and the
``(L, sigma_r)``-invariance of the ``m`` dimension.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from nvalchemiops.torch.math.gto import NormMode
from nvalchemiops.torch.math.gto_self_overlap import (
    FIELD_CONSTANT,
    compute_overlap_constants,
    flatten_to_reference_layout,
)


class TestComputeOverlapConstants:
    """Core behavior of :func:`compute_overlap_constants`."""

    def test_shape(self):
        out = compute_overlap_constants(
            max_L=1,
            sigma_source=1.0,
            sigmas_receive=[0.5, 1.0, 2.0],
        )
        assert out.shape == (3, 2)
        assert out.dtype == torch.float64

    def test_max_L_zero_shape(self):
        out = compute_overlap_constants(
            max_L=0,
            sigma_source=1.0,
            sigmas_receive=[0.5, 1.0],
        )
        assert out.shape == (2, 1)

    def test_single_sigma(self):
        out = compute_overlap_constants(
            max_L=1,
            sigma_source=1.3,
            sigmas_receive=[1.3],
        )
        assert out.shape == (1, 2)
        assert bool((out > 0.0).all())

    def test_field_constant_is_linear_factor(self):
        """Halving ``field_constant`` should halve every entry."""
        kwargs = dict(max_L=1, sigma_source=1.0, sigmas_receive=[0.5, 1.0])
        default = compute_overlap_constants(**kwargs)
        halved = compute_overlap_constants(
            **kwargs, field_constant=FIELD_CONSTANT / 2.0
        )
        np.testing.assert_allclose(halved, default / 2.0, rtol=1e-14)

    def test_none_mode_strips_Cl_factors(self):
        """With both modes NONE, only the raw radial integral and prefactor remain."""
        out_none = compute_overlap_constants(
            max_L=1,
            sigma_source=1.0,
            sigmas_receive=[1.0],
            normalize_source=NormMode.NONE,
            normalize_receive=NormMode.NONE,
        )
        out_mul = compute_overlap_constants(
            max_L=1,
            sigma_source=1.0,
            sigmas_receive=[1.0],
            normalize_source=NormMode.MULTIPOLES,
            normalize_receive=NormMode.NONE,
        )
        # NONE vs MULTIPOLES on source side differs by exactly inv_cl(source).
        # Since receive mode is the same (NONE), the ratio per L equals
        # inv_cl_multipoles(sigma_source, L) / 1.
        from nvalchemiops.torch.math.gto import inv_cl

        ratios = out_mul[0, :] / out_none[0, :]
        expected = np.array([inv_cl(1.0, L, NormMode.MULTIPOLES) for L in range(2)])
        np.testing.assert_allclose(ratios, expected, rtol=1e-14)

    def test_accepts_int_modes(self):
        """Passing plain ints in place of the enum should work identically."""
        kwargs = dict(max_L=1, sigma_source=1.0, sigmas_receive=[0.5, 1.0])
        enum_result = compute_overlap_constants(
            **kwargs,
            normalize_source=NormMode.MULTIPOLES,
            normalize_receive=NormMode.RECEIVER,
        )
        int_result = compute_overlap_constants(
            **kwargs, normalize_source=0, normalize_receive=1
        )
        np.testing.assert_array_equal(enum_result, int_result)

    @pytest.mark.parametrize(
        "bad_kwargs, match",
        [
            (dict(max_L=-1, sigma_source=1.0, sigmas_receive=[1.0]), "max_L"),
            (dict(max_L=0, sigma_source=0.0, sigmas_receive=[1.0]), "sigma_source"),
            (dict(max_L=0, sigma_source=-0.1, sigmas_receive=[1.0]), "sigma_source"),
            (dict(max_L=0, sigma_source=1.0, sigmas_receive=[]), "sigmas_receive"),
            (dict(max_L=0, sigma_source=1.0, sigmas_receive=[0.5, 0.0]), "positive"),
            (dict(max_L=0, sigma_source=1.0, sigmas_receive=[-0.5]), "positive"),
        ],
    )
    def test_rejects_bad_inputs(self, bad_kwargs, match):
        with pytest.raises(ValueError, match=match):
            compute_overlap_constants(**bad_kwargs)


class TestFlattenToReferenceLayout:
    """Layout conversion for parity with the customer flat buffer."""

    def test_shape(self):
        constants = np.array([[1.0, 2.0], [10.0, 20.0], [100.0, 200.0]])
        flat = flatten_to_reference_layout(constants, max_L=1)
        # n_sigma=3, max_L=1 -> total length 3 * (1+1)^2 = 12.
        assert flat.shape == (3 * (1 + 1) ** 2,)

    def test_index_scheme_for_l_zero(self):
        """For L=0 the layout is trivially ``[c_0, c_1, c_2, ...]`` (one m-slot)."""
        constants = np.array([[10.0], [20.0], [30.0]])
        flat = flatten_to_reference_layout(constants, max_L=0)
        np.testing.assert_array_equal(flat, [10.0, 20.0, 30.0])

    def test_index_scheme_for_l_one(self):
        """For L=1 each (i_sigma, L=1) entry is broadcast to 3 contiguous m-slots."""
        # n_sigma=2, max_L=1 -> flat length = 2 * (1+1)^2 = 8
        # Expected layout (indices per customer scheme):
        #   L=0, sigma=0: idx 0           -> a
        #   L=0, sigma=1: idx 1           -> b
        #   L=1, sigma=0, m=0,1,2: idx 2, 3, 4  -> c, c, c
        #   L=1, sigma=1, m=0,1,2: idx 5, 6, 7  -> d, d, d
        constants = np.array([[1.0, 3.0], [2.0, 4.0]])  # [[a, c], [b, d]]
        flat = flatten_to_reference_layout(constants, max_L=1)
        np.testing.assert_array_equal(flat, [1.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0])

    def test_m_slots_identical_within_each_l_block(self):
        """All 2L+1 m-slots for a given (i_sigma, L) must hold the same value."""
        out = compute_overlap_constants(
            max_L=1, sigma_source=1.0, sigmas_receive=[0.5, 1.0, 2.0]
        )
        flat = flatten_to_reference_layout(out, max_L=1)
        n_sigma = 3
        for L in [0, 1]:
            width = 2 * L + 1
            base = n_sigma * L * L
            for i_sigma in range(n_sigma):
                slot = flat[base + i_sigma * width : base + (i_sigma + 1) * width]
                assert bool((slot == slot[0]).all()), (
                    f"m-slots differ at L={L}, i_sigma={i_sigma}: {slot}"
                )

    def test_rejects_shape_mismatch(self):
        with pytest.raises(ValueError, match="shape"):
            # max_L=1 requires 2 columns, but we pass 3.
            flatten_to_reference_layout(np.zeros((2, 3)), max_L=1)

    def test_rejects_negative_max_L(self):
        with pytest.raises(ValueError, match="max_L"):
            flatten_to_reference_layout(np.zeros((1, 1)), max_L=-1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
