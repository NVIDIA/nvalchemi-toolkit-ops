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

"""Tests for nvalchemiops.segment_ops_backward (PR 1 backward kernels)."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from nvalchemiops.segment_ops_backward import (
    segment_div_backward,
    segment_div_double_backward,
    segmented_add_backward,
    segmented_add_double_backward,
    segmented_axpby_backward,
    segmented_axpby_double_backward,
    segmented_axpy_backward,
    segmented_axpy_double_backward,
    segmented_broadcast_backward,
    segmented_broadcast_double_backward,
    segmented_component_sum_backward,
    segmented_component_sum_double_backward,
    segmented_dot_backward,
    segmented_dot_double_backward,
    segmented_inner_products_backward,
    segmented_inner_products_double_backward,
    segmented_matvec_backward,
    segmented_matvec_double_backward,
    segmented_max_norm_backward,
    segmented_max_norm_double_backward,
    segmented_max_norm_forward_precompute,
    segmented_mean_backward,
    segmented_mean_double_backward,
    segmented_mul_backward,
    segmented_mul_double_backward,
    segmented_rms_norm_backward,
    segmented_rms_norm_double_backward,
    segmented_rms_norm_forward_precompute,
    segmented_sum_backward,
    segmented_sum_double_backward,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

N, M = 30, 5


@pytest.fixture(params=["cpu", "cuda:0"], ids=["cpu", "gpu"])
def device(request):
    """Both CPU and GPU; GPU is skipped if CUDA is not available."""
    device_name = request.param
    if device_name == "cuda:0" and not wp.is_cuda_available():
        pytest.skip("CUDA not available")
    return device_name


def _make_idx(N: int, M: int, rng) -> np.ndarray:
    return np.sort(rng.integers(0, M, N).astype(np.int32))


def _wpa(arr: np.ndarray, dtype, device: str) -> wp.array:
    return wp.array(arr, dtype=dtype, device=device)


def _wpv(arr: np.ndarray, dtype, device: str) -> wp.array:
    """(N, 3) numpy → warp vec3 array."""
    return wp.array([tuple(r) for r in arr], dtype=dtype, device=device)


def _wpm(arr: np.ndarray, dtype, device: str) -> wp.array:
    """(M, 3, 3) numpy → warp mat33 array."""
    return wp.array(
        [tuple(tuple(float(v) for v in row) for row in mat) for mat in arr],
        dtype=dtype,
        device=device,
    )


def _np(wp_arr: wp.array) -> np.ndarray:
    wp.synchronize()
    return wp_arr.numpy()


def _seg_sum(x: np.ndarray, idx: np.ndarray, M: int) -> np.ndarray:
    shape = (M,) if x.ndim == 1 else (M, x.shape[1])
    out = np.zeros(shape, dtype=x.dtype)
    np.add.at(out, idx, x)
    return out


# ---------------------------------------------------------------------------
# segmented_sum backward
# ---------------------------------------------------------------------------


class TestSegmentedSumBackward:
    def test_backward_scalar(self, device):
        rng = np.random.default_rng(0)
        idx = _make_idx(N, M, rng)
        g_out = rng.standard_normal(M).astype(np.float32)
        ref = g_out[idx]

        grad_x = wp.zeros(N, dtype=wp.float32, device=device)
        segmented_sum_backward(
            _wpa(g_out, wp.float32, device), _wpa(idx, wp.int32, device), grad_x
        )
        np.testing.assert_allclose(_np(grad_x), ref, rtol=1e-5)

    def test_backward_vec3(self, device):
        rng = np.random.default_rng(1)
        idx = _make_idx(N, M, rng)
        g_out = rng.standard_normal((M, 3)).astype(np.float32)
        ref = g_out[idx]

        grad_x = wp.zeros(N, dtype=wp.vec3f, device=device)
        segmented_sum_backward(
            _wpv(g_out, wp.vec3f, device), _wpa(idx, wp.int32, device), grad_x
        )
        np.testing.assert_allclose(_np(grad_x), ref, rtol=1e-5)

    def test_double_backward_scalar(self, device):
        rng = np.random.default_rng(2)
        idx = _make_idx(N, M, rng)
        gg_x = rng.standard_normal(N).astype(np.float32)
        ref = _seg_sum(gg_x, idx, M)

        grad_g_out = wp.zeros(M, dtype=wp.float32, device=device)
        segmented_sum_double_backward(
            _wpa(gg_x, wp.float32, device), _wpa(idx, wp.int32, device), M, grad_g_out
        )
        np.testing.assert_allclose(_np(grad_g_out), ref, rtol=1e-5)

    def test_double_backward_vec3(self, device):
        rng = np.random.default_rng(3)
        idx = _make_idx(N, M, rng)
        gg_x = rng.standard_normal((N, 3)).astype(np.float32)
        ref = _seg_sum(gg_x, idx, M)

        grad_g_out = wp.zeros(M, dtype=wp.vec3f, device=device)
        segmented_sum_double_backward(
            _wpv(gg_x, wp.vec3f, device), _wpa(idx, wp.int32, device), M, grad_g_out
        )
        np.testing.assert_allclose(_np(grad_g_out), ref, rtol=1e-4)


# ---------------------------------------------------------------------------
# segmented_broadcast backward
# ---------------------------------------------------------------------------


class TestSegmentedBroadcastBackward:
    def test_backward_scalar(self, device):
        rng = np.random.default_rng(4)
        idx = _make_idx(N, M, rng)
        g_out = rng.standard_normal(N).astype(np.float32)
        ref = _seg_sum(g_out, idx, M)

        grad_values = wp.zeros(M, dtype=wp.float32, device=device)
        segmented_broadcast_backward(
            _wpa(g_out, wp.float32, device), _wpa(idx, wp.int32, device), M, grad_values
        )
        np.testing.assert_allclose(_np(grad_values), ref, rtol=1e-5)

    def test_backward_vec3(self, device):
        rng = np.random.default_rng(5)
        idx = _make_idx(N, M, rng)
        g_out = rng.standard_normal((N, 3)).astype(np.float32)
        ref = _seg_sum(g_out, idx, M)

        grad_values = wp.zeros(M, dtype=wp.vec3f, device=device)
        segmented_broadcast_backward(
            _wpv(g_out, wp.vec3f, device), _wpa(idx, wp.int32, device), M, grad_values
        )
        np.testing.assert_allclose(_np(grad_values), ref, rtol=1e-4)

    def test_double_backward_scalar(self, device):
        rng = np.random.default_rng(6)
        idx = _make_idx(N, M, rng)
        gg_values = rng.standard_normal(M).astype(np.float32)
        ref = gg_values[idx]

        grad_g_out = wp.zeros(N, dtype=wp.float32, device=device)
        segmented_broadcast_double_backward(
            _wpa(gg_values, wp.float32, device), _wpa(idx, wp.int32, device), grad_g_out
        )
        np.testing.assert_allclose(_np(grad_g_out), ref, rtol=1e-5)


# ---------------------------------------------------------------------------
# segmented_component_sum backward
# ---------------------------------------------------------------------------


class TestSegmentedComponentSumBackward:
    def test_backward(self, device):
        rng = np.random.default_rng(7)
        idx = _make_idx(N, M, rng)
        g_out = rng.standard_normal(M).astype(np.float32)
        ref = np.stack([g_out[idx]] * 3, axis=1)

        grad_x = wp.zeros(N, dtype=wp.vec3f, device=device)
        segmented_component_sum_backward(
            _wpa(g_out, wp.float32, device), _wpa(idx, wp.int32, device), grad_x
        )
        np.testing.assert_allclose(_np(grad_x), ref, rtol=1e-5)

    def test_double_backward(self, device):
        rng = np.random.default_rng(8)
        idx = _make_idx(N, M, rng)
        gg_x = rng.standard_normal((N, 3)).astype(np.float32)
        ref = _seg_sum(gg_x.sum(axis=1), idx, M)

        grad_g_out = wp.zeros(M, dtype=wp.float32, device=device)
        segmented_component_sum_double_backward(
            _wpv(gg_x, wp.vec3f, device), _wpa(idx, wp.int32, device), M, grad_g_out
        )
        np.testing.assert_allclose(_np(grad_g_out), ref, rtol=1e-4)


# ---------------------------------------------------------------------------
# segmented_add backward
# ---------------------------------------------------------------------------


class TestSegmentedAddBackward:
    def test_backward_scalar(self, device):
        rng = np.random.default_rng(9)
        idx = _make_idx(N, M, rng)
        g_out = rng.standard_normal(N).astype(np.float32)

        grad_x = wp.zeros(N, dtype=wp.float32, device=device)
        grad_y = wp.zeros(M, dtype=wp.float32, device=device)
        segmented_add_backward(
            _wpa(g_out, wp.float32, device),
            _wpa(idx, wp.int32, device),
            M,
            grad_x,
            grad_y,
        )
        np.testing.assert_allclose(_np(grad_x), g_out, rtol=1e-5)
        np.testing.assert_allclose(_np(grad_y), _seg_sum(g_out, idx, M), rtol=1e-5)

    def test_backward_vec3(self, device):
        rng = np.random.default_rng(10)
        idx = _make_idx(N, M, rng)
        g_out = rng.standard_normal((N, 3)).astype(np.float32)

        grad_x = wp.zeros(N, dtype=wp.vec3f, device=device)
        grad_y = wp.zeros(M, dtype=wp.vec3f, device=device)
        segmented_add_backward(
            _wpv(g_out, wp.vec3f, device),
            _wpa(idx, wp.int32, device),
            M,
            grad_x,
            grad_y,
        )
        np.testing.assert_allclose(_np(grad_x), g_out, rtol=1e-5)
        np.testing.assert_allclose(_np(grad_y), _seg_sum(g_out, idx, M), rtol=1e-4)

    def test_double_backward_scalar(self, device):
        rng = np.random.default_rng(11)
        idx = _make_idx(N, M, rng)
        gg_x = rng.standard_normal(N).astype(np.float32)
        gg_y = rng.standard_normal(M).astype(np.float32)
        ref = gg_x + gg_y[idx]

        grad_g_out = wp.zeros(N, dtype=wp.float32, device=device)
        segmented_add_double_backward(
            _wpa(gg_x, wp.float32, device),
            _wpa(gg_y, wp.float32, device),
            _wpa(idx, wp.int32, device),
            grad_g_out,
        )
        np.testing.assert_allclose(_np(grad_g_out), ref, rtol=1e-5)


# ---------------------------------------------------------------------------
# segmented_mul backward
# ---------------------------------------------------------------------------


class TestSegmentedMulBackward:
    @pytest.mark.parametrize(
        "wp_type,np_type,rtol",
        [(wp.float32, np.float32, 1e-4), (wp.float64, np.float64, 1e-10)],
    )
    def test_backward_scalar(self, device, wp_type, np_type, rtol):
        rng = np.random.default_rng(12)
        idx = _make_idx(N, M, rng)
        x = rng.standard_normal(N).astype(np_type)
        y = rng.standard_normal(M).astype(np_type)
        g_out = rng.standard_normal(N).astype(np_type)

        ref_grad_x = g_out * y[idx]
        ref_grad_y = _seg_sum(g_out * x, idx, M)

        grad_x = wp.zeros(N, dtype=wp_type, device=device)
        grad_y = wp.zeros(M, dtype=wp_type, device=device)
        segmented_mul_backward(
            _wpa(g_out, wp_type, device),
            _wpa(x, wp_type, device),
            _wpa(y, wp_type, device),
            _wpa(idx, wp.int32, device),
            M,
            grad_x,
            grad_y,
        )
        np.testing.assert_allclose(_np(grad_x), ref_grad_x, rtol=rtol)
        np.testing.assert_allclose(_np(grad_y), ref_grad_y, rtol=rtol)

    def test_backward_vec3_scalar(self, device):
        rng = np.random.default_rng(13)
        idx = _make_idx(N, M, rng)
        x = rng.standard_normal((N, 3)).astype(np.float32)
        y = rng.standard_normal(M).astype(np.float32)
        g_out = rng.standard_normal((N, 3)).astype(np.float32)

        ref_grad_x = g_out * y[idx, None]
        ref_grad_y = _seg_sum((g_out * x).sum(axis=1), idx, M)

        grad_x = wp.zeros(N, dtype=wp.vec3f, device=device)
        grad_y = wp.zeros(M, dtype=wp.float32, device=device)
        segmented_mul_backward(
            _wpv(g_out, wp.vec3f, device),
            _wpv(x, wp.vec3f, device),
            _wpa(y, wp.float32, device),
            _wpa(idx, wp.int32, device),
            M,
            grad_x,
            grad_y,
        )
        np.testing.assert_allclose(_np(grad_x), ref_grad_x, rtol=1e-4)
        np.testing.assert_allclose(_np(grad_y), ref_grad_y, rtol=1e-4)

    def test_double_backward_scalar(self, device):
        rng = np.random.default_rng(14)
        idx = _make_idx(N, M, rng)
        g_out = rng.standard_normal(N).astype(np.float32)
        x = rng.standard_normal(N).astype(np.float32)
        y = rng.standard_normal(M).astype(np.float32)
        gg_gx = rng.standard_normal(N).astype(np.float32)
        gg_gy = rng.standard_normal(M).astype(np.float32)

        ref_grad_g_out = gg_gx * y[idx] + gg_gy[idx] * x
        ref_grad_x_extra = gg_gy[idx] * g_out
        ref_grad_y_extra = _seg_sum(gg_gx * g_out, idx, M)

        grad_g_out = wp.zeros(N, dtype=wp.float32, device=device)
        grad_x_extra = wp.zeros(N, dtype=wp.float32, device=device)
        grad_y_extra = wp.zeros(M, dtype=wp.float32, device=device)
        segmented_mul_double_backward(
            _wpa(gg_gx, wp.float32, device),
            _wpa(gg_gy, wp.float32, device),
            _wpa(g_out, wp.float32, device),
            _wpa(x, wp.float32, device),
            _wpa(y, wp.float32, device),
            _wpa(idx, wp.int32, device),
            grad_g_out,
            grad_x_extra,
            grad_y_extra,
        )
        np.testing.assert_allclose(_np(grad_g_out), ref_grad_g_out, rtol=1e-4)
        np.testing.assert_allclose(_np(grad_x_extra), ref_grad_x_extra, rtol=1e-4)
        np.testing.assert_allclose(_np(grad_y_extra), ref_grad_y_extra, rtol=1e-4)


# ---------------------------------------------------------------------------
# segmented_dot backward
# ---------------------------------------------------------------------------


class TestSegmentedDotBackward:
    def test_backward_vec3(self, device):
        rng = np.random.default_rng(15)
        idx = _make_idx(N, M, rng)
        x = rng.standard_normal((N, 3)).astype(np.float32)
        y = rng.standard_normal((N, 3)).astype(np.float32)
        g_out = rng.standard_normal(M).astype(np.float32)

        grad_x = wp.zeros(N, dtype=wp.vec3f, device=device)
        grad_y = wp.zeros(N, dtype=wp.vec3f, device=device)
        segmented_dot_backward(
            _wpa(g_out, wp.float32, device),
            _wpv(x, wp.vec3f, device),
            _wpv(y, wp.vec3f, device),
            _wpa(idx, wp.int32, device),
            grad_x,
            grad_y,
        )
        np.testing.assert_allclose(_np(grad_x), y * g_out[idx, None], rtol=1e-4)
        np.testing.assert_allclose(_np(grad_y), x * g_out[idx, None], rtol=1e-4)

    def test_backward_scalar(self, device):
        rng = np.random.default_rng(16)
        idx = _make_idx(N, M, rng)
        x = rng.standard_normal(N).astype(np.float32)
        y = rng.standard_normal(N).astype(np.float32)
        g_out = rng.standard_normal(M).astype(np.float32)

        grad_x = wp.zeros(N, dtype=wp.float32, device=device)
        grad_y = wp.zeros(N, dtype=wp.float32, device=device)
        segmented_dot_backward(
            _wpa(g_out, wp.float32, device),
            _wpa(x, wp.float32, device),
            _wpa(y, wp.float32, device),
            _wpa(idx, wp.int32, device),
            grad_x,
            grad_y,
        )
        np.testing.assert_allclose(_np(grad_x), y * g_out[idx], rtol=1e-4)
        np.testing.assert_allclose(_np(grad_y), x * g_out[idx], rtol=1e-4)

    def test_double_backward_scalar(self, device):
        rng = np.random.default_rng(17)
        idx = _make_idx(N, M, rng)
        x = rng.standard_normal(N).astype(np.float32)
        y = rng.standard_normal(N).astype(np.float32)
        g_out = rng.standard_normal(M).astype(np.float32)
        gg_gx = rng.standard_normal(N).astype(np.float32)
        gg_gy = rng.standard_normal(N).astype(np.float32)

        ref_g_out = _seg_sum(gg_gx * y, idx, M) + _seg_sum(gg_gy * x, idx, M)
        ref_x_extra = gg_gy * g_out[idx]
        ref_y_extra = gg_gx * g_out[idx]

        grad_g_out = wp.zeros(M, dtype=wp.float32, device=device)
        grad_x_extra = wp.zeros(N, dtype=wp.float32, device=device)
        grad_y_extra = wp.zeros(N, dtype=wp.float32, device=device)
        segmented_dot_double_backward(
            _wpa(gg_gx, wp.float32, device),
            _wpa(gg_gy, wp.float32, device),
            _wpa(g_out, wp.float32, device),
            _wpa(x, wp.float32, device),
            _wpa(y, wp.float32, device),
            _wpa(idx, wp.int32, device),
            M,
            grad_g_out,
            grad_x_extra,
            grad_y_extra,
        )
        np.testing.assert_allclose(_np(grad_g_out), ref_g_out, rtol=1e-4)
        np.testing.assert_allclose(_np(grad_x_extra), ref_x_extra, rtol=1e-4)
        np.testing.assert_allclose(_np(grad_y_extra), ref_y_extra, rtol=1e-4)


# ---------------------------------------------------------------------------
# segmented_inner_products backward
# ---------------------------------------------------------------------------


class TestSegmentedInnerProductsBackward:
    def test_backward_scalar(self, device):
        rng = np.random.default_rng(18)
        idx = _make_idx(N, M, rng)
        x = rng.standard_normal(N).astype(np.float32)
        y = rng.standard_normal(N).astype(np.float32)
        g_xy = rng.standard_normal(M).astype(np.float32)
        g_xx = rng.standard_normal(M).astype(np.float32)
        g_yy = rng.standard_normal(M).astype(np.float32)

        ref_grad_x = g_xy[idx] * y + 2.0 * g_xx[idx] * x
        ref_grad_y = g_xy[idx] * x + 2.0 * g_yy[idx] * y

        grad_x = wp.zeros(N, dtype=wp.float32, device=device)
        grad_y = wp.zeros(N, dtype=wp.float32, device=device)
        segmented_inner_products_backward(
            _wpa(x, wp.float32, device),
            _wpa(y, wp.float32, device),
            _wpa(idx, wp.int32, device),
            _wpa(g_xy, wp.float32, device),
            _wpa(g_xx, wp.float32, device),
            _wpa(g_yy, wp.float32, device),
            grad_x,
            grad_y,
        )
        np.testing.assert_allclose(_np(grad_x), ref_grad_x, rtol=1e-4)
        np.testing.assert_allclose(_np(grad_y), ref_grad_y, rtol=1e-4)

    def test_backward_vec3(self, device):
        rng = np.random.default_rng(19)
        idx = _make_idx(N, M, rng)
        x = rng.standard_normal((N, 3)).astype(np.float32)
        y = rng.standard_normal((N, 3)).astype(np.float32)
        g_xy = rng.standard_normal(M).astype(np.float32)
        g_xx = rng.standard_normal(M).astype(np.float32)
        g_yy = rng.standard_normal(M).astype(np.float32)

        ref_grad_x = g_xy[idx, None] * y + 2.0 * g_xx[idx, None] * x
        ref_grad_y = g_xy[idx, None] * x + 2.0 * g_yy[idx, None] * y

        grad_x = wp.zeros(N, dtype=wp.vec3f, device=device)
        grad_y = wp.zeros(N, dtype=wp.vec3f, device=device)
        segmented_inner_products_backward(
            _wpv(x, wp.vec3f, device),
            _wpv(y, wp.vec3f, device),
            _wpa(idx, wp.int32, device),
            _wpa(g_xy, wp.float32, device),
            _wpa(g_xx, wp.float32, device),
            _wpa(g_yy, wp.float32, device),
            grad_x,
            grad_y,
        )
        np.testing.assert_allclose(_np(grad_x), ref_grad_x, rtol=1e-4)
        np.testing.assert_allclose(_np(grad_y), ref_grad_y, rtol=1e-4)

    def test_double_backward_scalar(self, device):
        rng = np.random.default_rng(40)
        idx = _make_idx(N, M, rng)
        x = rng.standard_normal(N).astype(np.float32)
        y = rng.standard_normal(N).astype(np.float32)
        g_xy = rng.standard_normal(M).astype(np.float32)
        g_xx = rng.standard_normal(M).astype(np.float32)
        g_yy = rng.standard_normal(M).astype(np.float32)
        gg_gx = rng.standard_normal(N).astype(np.float32)
        gg_gy = rng.standard_normal(N).astype(np.float32)

        ref_grad_x_extra = 2.0 * g_xx[idx] * gg_gx + g_xy[idx] * gg_gy
        ref_grad_y_extra = g_xy[idx] * gg_gx + 2.0 * g_yy[idx] * gg_gy
        ref_grad_g_xy = _seg_sum(gg_gx * y + gg_gy * x, idx, M)
        ref_grad_g_xx = 2.0 * _seg_sum(gg_gx * x, idx, M)
        ref_grad_g_yy = 2.0 * _seg_sum(gg_gy * y, idx, M)

        grad_x_extra = wp.zeros(N, dtype=wp.float32, device=device)
        grad_y_extra = wp.zeros(N, dtype=wp.float32, device=device)
        grad_g_xy = wp.zeros(M, dtype=wp.float32, device=device)
        grad_g_xx = wp.zeros(M, dtype=wp.float32, device=device)
        grad_g_yy = wp.zeros(M, dtype=wp.float32, device=device)
        segmented_inner_products_double_backward(
            _wpa(gg_gx, wp.float32, device),
            _wpa(gg_gy, wp.float32, device),
            _wpa(x, wp.float32, device),
            _wpa(y, wp.float32, device),
            _wpa(g_xy, wp.float32, device),
            _wpa(g_xx, wp.float32, device),
            _wpa(g_yy, wp.float32, device),
            _wpa(idx, wp.int32, device),
            M,
            grad_x_extra,
            grad_y_extra,
            grad_g_xy,
            grad_g_xx,
            grad_g_yy,
        )
        np.testing.assert_allclose(_np(grad_x_extra), ref_grad_x_extra, rtol=1e-4)
        np.testing.assert_allclose(_np(grad_y_extra), ref_grad_y_extra, rtol=1e-4)
        np.testing.assert_allclose(_np(grad_g_xy), ref_grad_g_xy, rtol=1e-4)
        np.testing.assert_allclose(_np(grad_g_xx), ref_grad_g_xx, rtol=1e-4)
        np.testing.assert_allclose(_np(grad_g_yy), ref_grad_g_yy, rtol=1e-4)

    def test_double_backward_vec3(self, device):
        rng = np.random.default_rng(41)
        idx = _make_idx(N, M, rng)
        x = rng.standard_normal((N, 3)).astype(np.float32)
        y = rng.standard_normal((N, 3)).astype(np.float32)
        g_xy = rng.standard_normal(M).astype(np.float32)
        g_xx = rng.standard_normal(M).astype(np.float32)
        g_yy = rng.standard_normal(M).astype(np.float32)
        gg_gx = rng.standard_normal((N, 3)).astype(np.float32)
        gg_gy = rng.standard_normal((N, 3)).astype(np.float32)

        ref_grad_x_extra = 2.0 * g_xx[idx, None] * gg_gx + g_xy[idx, None] * gg_gy
        ref_grad_y_extra = g_xy[idx, None] * gg_gx + 2.0 * g_yy[idx, None] * gg_gy
        ref_grad_g_xy = _seg_sum(
            (gg_gx * y).sum(axis=1) + (gg_gy * x).sum(axis=1), idx, M
        )
        ref_grad_g_xx = 2.0 * _seg_sum((gg_gx * x).sum(axis=1), idx, M)
        ref_grad_g_yy = 2.0 * _seg_sum((gg_gy * y).sum(axis=1), idx, M)

        grad_x_extra = wp.zeros(N, dtype=wp.vec3f, device=device)
        grad_y_extra = wp.zeros(N, dtype=wp.vec3f, device=device)
        grad_g_xy = wp.zeros(M, dtype=wp.float32, device=device)
        grad_g_xx = wp.zeros(M, dtype=wp.float32, device=device)
        grad_g_yy = wp.zeros(M, dtype=wp.float32, device=device)
        segmented_inner_products_double_backward(
            _wpv(gg_gx, wp.vec3f, device),
            _wpv(gg_gy, wp.vec3f, device),
            _wpv(x, wp.vec3f, device),
            _wpv(y, wp.vec3f, device),
            _wpa(g_xy, wp.float32, device),
            _wpa(g_xx, wp.float32, device),
            _wpa(g_yy, wp.float32, device),
            _wpa(idx, wp.int32, device),
            M,
            grad_x_extra,
            grad_y_extra,
            grad_g_xy,
            grad_g_xx,
            grad_g_yy,
        )
        np.testing.assert_allclose(_np(grad_x_extra), ref_grad_x_extra, rtol=1e-4)
        np.testing.assert_allclose(_np(grad_y_extra), ref_grad_y_extra, rtol=1e-4)
        np.testing.assert_allclose(_np(grad_g_xy), ref_grad_g_xy, rtol=1e-4)
        np.testing.assert_allclose(_np(grad_g_xx), ref_grad_g_xx, rtol=1e-4)
        np.testing.assert_allclose(_np(grad_g_yy), ref_grad_g_yy, rtol=1e-4)


# ---------------------------------------------------------------------------
# segmented_mean backward
# ---------------------------------------------------------------------------


class TestSegmentedMeanBackward:
    def test_backward_scalar(self, device):
        rng = np.random.default_rng(20)
        idx = _make_idx(N, M, rng)
        counts = np.bincount(idx, minlength=M).astype(np.int32)
        g_out = rng.standard_normal(M).astype(np.float32)
        ref = g_out[idx] / counts[idx].astype(np.float32)

        grad_x = wp.zeros(N, dtype=wp.float32, device=device)
        segmented_mean_backward(
            _wpa(g_out, wp.float32, device),
            _wpa(counts, wp.int32, device),
            _wpa(idx, wp.int32, device),
            grad_x,
        )
        np.testing.assert_allclose(_np(grad_x), ref, rtol=1e-5)

    def test_backward_vec3(self, device):
        rng = np.random.default_rng(21)
        idx = _make_idx(N, M, rng)
        counts = np.bincount(idx, minlength=M).astype(np.int32)
        g_out = rng.standard_normal((M, 3)).astype(np.float32)
        ref = g_out[idx] / counts[idx, None].astype(np.float32)

        grad_x = wp.zeros(N, dtype=wp.vec3f, device=device)
        segmented_mean_backward(
            _wpv(g_out, wp.vec3f, device),
            _wpa(counts, wp.int32, device),
            _wpa(idx, wp.int32, device),
            grad_x,
        )
        np.testing.assert_allclose(_np(grad_x), ref, rtol=1e-5)

    def test_double_backward_scalar(self, device):
        rng = np.random.default_rng(22)
        idx = _make_idx(N, M, rng)
        counts = np.bincount(idx, minlength=M).astype(np.int32)
        gg_x = rng.standard_normal(N).astype(np.float32)
        ref = _seg_sum(gg_x, idx, M) / counts.astype(np.float32)

        grad_g_out = wp.zeros(M, dtype=wp.float32, device=device)
        segmented_mean_double_backward(
            _wpa(gg_x, wp.float32, device),
            _wpa(counts, wp.int32, device),
            _wpa(idx, wp.int32, device),
            grad_g_out,
        )
        np.testing.assert_allclose(_np(grad_g_out), ref, rtol=1e-4)


# ---------------------------------------------------------------------------
# segmented_rms_norm backward
# ---------------------------------------------------------------------------


def _rms_norm_ref(x_np, idx, M):
    """Returns (sum_sq, counts, out, inv_norm) as numpy arrays."""
    sum_sq = np.zeros(M, dtype=x_np.dtype if x_np.ndim == 1 else np.float32)
    counts = np.bincount(idx, minlength=M).astype(np.int32)
    for i in range(len(idx)):
        s = idx[i]
        v = x_np[i]
        sum_sq[s] += float(np.dot(v, v)) if x_np.ndim > 1 else float(v * v)
    out = np.sqrt(sum_sq / np.maximum(counts, 1).astype(sum_sq.dtype))
    denom = out * counts.astype(out.dtype)
    inv_norm = np.where(denom > 0, 1.0 / denom, 0.0).astype(out.dtype)
    return sum_sq, counts, out, inv_norm


class TestSegmentedRmsNormBackward:
    def test_forward_precompute(self, device):
        rng = np.random.default_rng(23)
        idx = _make_idx(N, M, rng)
        x = rng.standard_normal((N, 3)).astype(np.float32)
        _, counts_ref, out_ref, inv_norm_ref = _rms_norm_ref(x, idx, M)

        sum_sq = wp.zeros(M, dtype=wp.float32, device=device)
        counts = wp.zeros(M, dtype=wp.int32, device=device)
        out = wp.zeros(M, dtype=wp.float32, device=device)
        inv_norm = wp.zeros(M, dtype=wp.float32, device=device)
        segmented_rms_norm_forward_precompute(
            _wpv(x, wp.vec3f, device),
            _wpa(idx, wp.int32, device),
            sum_sq,
            counts,
            out,
            inv_norm,
        )
        np.testing.assert_allclose(_np(out), out_ref, rtol=1e-4)
        np.testing.assert_allclose(_np(inv_norm), inv_norm_ref, rtol=1e-4)
        np.testing.assert_array_equal(_np(counts), counts_ref)

    def test_backward(self, device):
        rng = np.random.default_rng(24)
        idx = _make_idx(N, M, rng)
        x = rng.standard_normal((N, 3)).astype(np.float32)
        _, _, _, inv_norm_ref = _rms_norm_ref(x, idx, M)
        g_out = rng.standard_normal(M).astype(np.float32)

        ref = g_out[idx, None] * x * inv_norm_ref[idx, None]

        grad_x = wp.zeros(N, dtype=wp.vec3f, device=device)
        segmented_rms_norm_backward(
            _wpa(g_out, wp.float32, device),
            _wpv(x, wp.vec3f, device),
            _wpa(inv_norm_ref, wp.float32, device),
            _wpa(idx, wp.int32, device),
            grad_x,
        )
        np.testing.assert_allclose(_np(grad_x), ref, rtol=1e-4)

    def test_double_backward(self, device):
        rng = np.random.default_rng(25)
        idx = _make_idx(N, M, rng)
        x = rng.standard_normal((N, 3)).astype(np.float32)
        _, counts, _, inv_norm = _rms_norm_ref(x, idx, M)
        g_out = rng.standard_normal(M).astype(np.float32)
        gg_x = rng.standard_normal((N, 3)).astype(np.float32)

        inner = np.zeros(M, dtype=np.float32)
        for i in range(N):
            inner[idx[i]] += float(np.dot(gg_x[i], x[i]))
        ref_grad_g_out = inner * inv_norm
        ref_grad_x_extra = np.zeros((N, 3), dtype=np.float32)
        for i in range(N):
            s = idx[i]
            n = inv_norm[s]
            c = float(counts[s])
            ref_grad_x_extra[i] = (
                g_out[s] * n * gg_x[i] - g_out[s] * n**3 * c * inner[s] * x[i]
            )

        grad_x_extra = wp.zeros(N, dtype=wp.vec3f, device=device)
        grad_g_out_extra = wp.zeros(M, dtype=wp.float32, device=device)
        segmented_rms_norm_double_backward(
            _wpv(gg_x, wp.vec3f, device),
            _wpv(x, wp.vec3f, device),
            _wpa(g_out, wp.float32, device),
            _wpa(inv_norm, wp.float32, device),
            _wpa(counts, wp.int32, device),
            _wpa(idx, wp.int32, device),
            M,
            grad_x_extra,
            grad_g_out_extra,
        )
        np.testing.assert_allclose(_np(grad_g_out_extra), ref_grad_g_out, rtol=1e-4)
        np.testing.assert_allclose(
            _np(grad_x_extra), ref_grad_x_extra, rtol=1e-3, atol=1e-6
        )


# ---------------------------------------------------------------------------
# segmented_max_norm backward
# ---------------------------------------------------------------------------


def _max_norm_ref(x_np, idx, M):
    """Returns (out, argmax_idx) as numpy arrays."""
    norms = np.linalg.norm(x_np, axis=1)
    out = np.zeros(M, dtype=np.float32)
    argmax = np.full(M, -1, dtype=np.int32)
    for i in range(len(idx)):
        s = idx[i]
        if norms[i] > out[s]:
            out[s] = norms[i]
            argmax[s] = i
    return out, argmax


class TestSegmentedMaxNormBackward:
    def test_forward_precompute(self, device):
        rng = np.random.default_rng(26)
        idx = _make_idx(N, M, rng)
        x = rng.standard_normal((N, 3)).astype(np.float32)
        out_ref, argmax_ref = _max_norm_ref(x, idx, M)

        out = wp.zeros(M, dtype=wp.float32, device=device)
        argmax_idx = wp.zeros(M, dtype=wp.int32, device=device)
        segmented_max_norm_forward_precompute(
            _wpv(x, wp.vec3f, device), _wpa(idx, wp.int32, device), out, argmax_idx
        )
        np.testing.assert_allclose(_np(out), out_ref, rtol=1e-5)
        argmax_np = _np(argmax_idx)
        for s in range(M):
            i = argmax_np[s]
            assert i >= 0, f"argmax for segment {s} not set"
            assert idx[i] == s
            assert np.linalg.norm(x[i]) == pytest.approx(float(out_ref[s]), rel=1e-4)

    def test_backward(self, device):
        rng = np.random.default_rng(27)
        idx = _make_idx(N, M, rng)
        x = rng.standard_normal((N, 3)).astype(np.float32)
        out_ref, argmax_ref = _max_norm_ref(x, idx, M)
        g_out = rng.standard_normal(M).astype(np.float32)

        ref = np.zeros((N, 3), dtype=np.float32)
        for s in range(M):
            i = argmax_ref[s]
            if i >= 0 and out_ref[s] > 0:
                ref[i] = g_out[s] * x[i] / out_ref[s]

        grad_x = wp.zeros(N, dtype=wp.vec3f, device=device)
        segmented_max_norm_backward(
            _wpa(g_out, wp.float32, device),
            _wpv(x, wp.vec3f, device),
            _wpa(argmax_ref, wp.int32, device),
            _wpa(idx, wp.int32, device),
            grad_x,
        )
        np.testing.assert_allclose(_np(grad_x), ref, rtol=1e-5)

    def test_double_backward(self, device):
        rng = np.random.default_rng(28)
        idx = _make_idx(N, M, rng)
        x = rng.standard_normal((N, 3)).astype(np.float32)
        out_ref, argmax_ref = _max_norm_ref(x, idx, M)
        g_out = rng.standard_normal(M).astype(np.float32)
        gg_gx = rng.standard_normal((N, 3)).astype(np.float32)

        ref_grad_x_extra = np.zeros((N, 3), dtype=np.float32)
        ref_grad_g_out = np.zeros(M, dtype=np.float32)
        for s in range(M):
            i = argmax_ref[s]
            if i >= 0 and out_ref[s] > 0:
                n = out_ref[s]
                x_hat = x[i] / n
                proj = float(np.dot(x_hat, gg_gx[i]))
                ref_grad_x_extra[i] = (g_out[s] / n) * (gg_gx[i] - x_hat * proj)
                ref_grad_g_out[s] = proj

        grad_x_extra = wp.zeros(N, dtype=wp.vec3f, device=device)
        grad_g_out = wp.zeros(M, dtype=wp.float32, device=device)
        segmented_max_norm_double_backward(
            _wpv(gg_gx, wp.vec3f, device),
            _wpa(g_out, wp.float32, device),
            _wpv(x, wp.vec3f, device),
            _wpa(argmax_ref, wp.int32, device),
            _wpa(idx, wp.int32, device),
            grad_x_extra,
            grad_g_out,
        )
        np.testing.assert_allclose(_np(grad_x_extra), ref_grad_x_extra, rtol=1e-4)
        np.testing.assert_allclose(_np(grad_g_out), ref_grad_g_out, rtol=1e-4)


# ---------------------------------------------------------------------------
# segmented_matvec backward
# ---------------------------------------------------------------------------


class TestSegmentedMatvecBackward:
    def test_backward(self, device):
        rng = np.random.default_rng(29)
        idx = _make_idx(N, M, rng)
        v = rng.standard_normal((N, 3)).astype(np.float32)
        m = rng.standard_normal((M, 3, 3)).astype(np.float32)
        g_out = rng.standard_normal((N, 3)).astype(np.float32)

        # Forward: out[i] = M[s]^T @ v[i]
        # grad_v[i] = M[s] @ g_out[i]   (non-transposed)
        # grad_M[s][j][k] = sum_i v[i][j] * g_out[i][k]
        ref_grad_v = np.array(
            [m[idx[i]] @ g_out[i] for i in range(N)], dtype=np.float32
        )
        ref_grad_M = np.zeros((M, 3, 3), dtype=np.float32)
        for i in range(N):
            ref_grad_M[idx[i]] += np.outer(v[i], g_out[i])

        grad_v = wp.zeros(N, dtype=wp.vec3f, device=device)
        grad_M = wp.zeros(M, dtype=wp.mat33f, device=device)
        segmented_matvec_backward(
            _wpv(g_out, wp.vec3f, device),
            _wpv(v, wp.vec3f, device),
            _wpm(m, wp.mat33f, device),
            _wpa(idx, wp.int32, device),
            grad_v,
            grad_M,
        )
        np.testing.assert_allclose(_np(grad_v), ref_grad_v, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(_np(grad_M), ref_grad_M, rtol=1e-4, atol=1e-6)

    def test_double_backward(self, device):
        rng = np.random.default_rng(42)
        idx = _make_idx(N, M, rng)
        v = rng.standard_normal((N, 3)).astype(np.float32)
        m = rng.standard_normal((M, 3, 3)).astype(np.float32)
        g_out = rng.standard_normal((N, 3)).astype(np.float32)
        gg_gv = rng.standard_normal((N, 3)).astype(np.float32)
        gg_gM = rng.standard_normal((M, 3, 3)).astype(np.float32)

        # grad_g_out[i] = M[s]^T @ gg_gv[i] + gg_gM[s]^T @ v[i]
        ref_grad_g_out = np.zeros((N, 3), dtype=np.float32)
        for i in range(N):
            s = idx[i]
            ref_grad_g_out[i] = m[s].T @ gg_gv[i] + gg_gM[s].T @ v[i]
        # grad_v_extra[i] = gg_gM[s] @ g_out[i]
        ref_grad_v_extra = np.array(
            [gg_gM[idx[i]] @ g_out[i] for i in range(N)], dtype=np.float32
        )
        # grad_M_extra[s] = sum_i outer(gg_gv[i], g_out[i])
        ref_grad_M_extra = np.zeros((M, 3, 3), dtype=np.float32)
        for i in range(N):
            ref_grad_M_extra[idx[i]] += np.outer(gg_gv[i], g_out[i])

        grad_g_out = wp.zeros(N, dtype=wp.vec3f, device=device)
        grad_v_extra = wp.zeros(N, dtype=wp.vec3f, device=device)
        grad_M_extra = wp.zeros(M, dtype=wp.mat33f, device=device)
        segmented_matvec_double_backward(
            _wpv(gg_gv, wp.vec3f, device),
            _wpm(gg_gM, wp.mat33f, device),
            _wpv(g_out, wp.vec3f, device),
            _wpv(v, wp.vec3f, device),
            _wpm(m, wp.mat33f, device),
            _wpa(idx, wp.int32, device),
            grad_g_out,
            grad_v_extra,
            grad_M_extra,
        )
        np.testing.assert_allclose(
            _np(grad_g_out), ref_grad_g_out, rtol=1e-4, atol=1e-6
        )
        np.testing.assert_allclose(
            _np(grad_v_extra), ref_grad_v_extra, rtol=1e-4, atol=1e-6
        )
        np.testing.assert_allclose(
            _np(grad_M_extra), ref_grad_M_extra, rtol=1e-4, atol=1e-6
        )


# ---------------------------------------------------------------------------
# segment_div backward
# ---------------------------------------------------------------------------


class TestSegmentDivBackward:
    def test_backward(self, device):
        rng = np.random.default_rng(30)
        den = rng.integers(1, 5, N).astype(np.int32)
        g_result = rng.standard_normal(N).astype(np.float32)

        grad_num = wp.zeros(N, dtype=wp.float32, device=device)
        segment_div_backward(
            _wpa(g_result, wp.float32, device),
            _wpa(den, wp.int32, device),
            grad_num,
        )
        np.testing.assert_allclose(
            _np(grad_num), g_result / den.astype(np.float32), rtol=1e-5
        )

    def test_double_backward(self, device):
        rng = np.random.default_rng(31)
        gg_num = rng.standard_normal(N).astype(np.float32)
        den = rng.integers(1, 5, N).astype(np.int32)

        grad_g_result = wp.zeros(N, dtype=wp.float32, device=device)
        segment_div_double_backward(
            _wpa(gg_num, wp.float32, device),
            _wpa(den, wp.int32, device),
            grad_g_result,
        )
        np.testing.assert_allclose(
            _np(grad_g_result), gg_num / den.astype(np.float32), rtol=1e-5
        )


# ---------------------------------------------------------------------------
# segmented_axpy backward
# ---------------------------------------------------------------------------


class TestSegmentedAxpyBackward:
    def test_backward_scalar(self, device):
        rng = np.random.default_rng(32)
        idx = _make_idx(N, M, rng)
        x = rng.standard_normal(N).astype(np.float32)
        a = rng.standard_normal(M).astype(np.float32)
        g_out = rng.standard_normal(N).astype(np.float32)

        grad_y_in = wp.zeros(N, dtype=wp.float32, device=device)
        grad_x = wp.zeros(N, dtype=wp.float32, device=device)
        grad_a = wp.zeros(M, dtype=wp.float32, device=device)
        segmented_axpy_backward(
            _wpa(g_out, wp.float32, device),
            _wpa(x, wp.float32, device),
            _wpa(a, wp.float32, device),
            _wpa(idx, wp.int32, device),
            M,
            grad_y_in,
            grad_x,
            grad_a,
        )
        np.testing.assert_allclose(_np(grad_y_in), g_out, rtol=1e-5)
        np.testing.assert_allclose(_np(grad_x), a[idx] * g_out, rtol=1e-4)
        np.testing.assert_allclose(_np(grad_a), _seg_sum(x * g_out, idx, M), rtol=1e-4)

    def test_backward_vec3(self, device):
        rng = np.random.default_rng(33)
        idx = _make_idx(N, M, rng)
        x = rng.standard_normal((N, 3)).astype(np.float32)
        a = rng.standard_normal(M).astype(np.float32)
        g_out = rng.standard_normal((N, 3)).astype(np.float32)

        ref_grad_a = _seg_sum((x * g_out).sum(axis=1), idx, M)

        grad_y_in = wp.zeros(N, dtype=wp.vec3f, device=device)
        grad_x = wp.zeros(N, dtype=wp.vec3f, device=device)
        grad_a = wp.zeros(M, dtype=wp.float32, device=device)
        segmented_axpy_backward(
            _wpv(g_out, wp.vec3f, device),
            _wpv(x, wp.vec3f, device),
            _wpa(a, wp.float32, device),
            _wpa(idx, wp.int32, device),
            M,
            grad_y_in,
            grad_x,
            grad_a,
        )
        np.testing.assert_allclose(_np(grad_y_in), g_out, rtol=1e-5)
        np.testing.assert_allclose(_np(grad_x), g_out * a[idx, None], rtol=1e-4)
        np.testing.assert_allclose(_np(grad_a), ref_grad_a, rtol=1e-4)

    def test_double_backward_scalar(self, device):
        rng = np.random.default_rng(43)
        idx = _make_idx(N, M, rng)
        x = rng.standard_normal(N).astype(np.float32)
        a = rng.standard_normal(M).astype(np.float32)
        g_out = rng.standard_normal(N).astype(np.float32)
        gg_gy_in = rng.standard_normal(N).astype(np.float32)
        gg_gx = rng.standard_normal(N).astype(np.float32)
        gg_ga = rng.standard_normal(M).astype(np.float32)

        ref_grad_g_out = gg_gy_in + gg_gx * a[idx] + gg_ga[idx] * x
        ref_grad_x_extra = gg_ga[idx] * g_out
        ref_grad_a_extra = _seg_sum(gg_gx * g_out, idx, M)

        grad_g_out = wp.zeros(N, dtype=wp.float32, device=device)
        grad_x_extra = wp.zeros(N, dtype=wp.float32, device=device)
        grad_a_extra = wp.zeros(M, dtype=wp.float32, device=device)
        segmented_axpy_double_backward(
            _wpa(gg_gy_in, wp.float32, device),
            _wpa(gg_gx, wp.float32, device),
            _wpa(gg_ga, wp.float32, device),
            _wpa(g_out, wp.float32, device),
            _wpa(x, wp.float32, device),
            _wpa(a, wp.float32, device),
            _wpa(idx, wp.int32, device),
            grad_g_out,
            grad_x_extra,
            grad_a_extra,
        )
        np.testing.assert_allclose(_np(grad_g_out), ref_grad_g_out, rtol=1e-4)
        np.testing.assert_allclose(_np(grad_x_extra), ref_grad_x_extra, rtol=1e-4)
        np.testing.assert_allclose(_np(grad_a_extra), ref_grad_a_extra, rtol=1e-4)

    def test_double_backward_vec3(self, device):
        rng = np.random.default_rng(44)
        idx = _make_idx(N, M, rng)
        x = rng.standard_normal((N, 3)).astype(np.float32)
        a = rng.standard_normal(M).astype(np.float32)
        g_out = rng.standard_normal((N, 3)).astype(np.float32)
        gg_gy_in = rng.standard_normal((N, 3)).astype(np.float32)
        gg_gx = rng.standard_normal((N, 3)).astype(np.float32)
        gg_ga = rng.standard_normal(M).astype(np.float32)

        ref_grad_g_out = gg_gy_in + gg_gx * a[idx, None] + gg_ga[idx, None] * x
        ref_grad_x_extra = gg_ga[idx, None] * g_out
        ref_grad_a_extra = _seg_sum((gg_gx * g_out).sum(axis=1), idx, M)

        grad_g_out = wp.zeros(N, dtype=wp.vec3f, device=device)
        grad_x_extra = wp.zeros(N, dtype=wp.vec3f, device=device)
        grad_a_extra = wp.zeros(M, dtype=wp.float32, device=device)
        segmented_axpy_double_backward(
            _wpv(gg_gy_in, wp.vec3f, device),
            _wpv(gg_gx, wp.vec3f, device),
            _wpa(gg_ga, wp.float32, device),
            _wpv(g_out, wp.vec3f, device),
            _wpv(x, wp.vec3f, device),
            _wpa(a, wp.float32, device),
            _wpa(idx, wp.int32, device),
            grad_g_out,
            grad_x_extra,
            grad_a_extra,
        )
        np.testing.assert_allclose(_np(grad_g_out), ref_grad_g_out, rtol=1e-4)
        np.testing.assert_allclose(_np(grad_x_extra), ref_grad_x_extra, rtol=1e-4)
        np.testing.assert_allclose(_np(grad_a_extra), ref_grad_a_extra, rtol=1e-4)


# ---------------------------------------------------------------------------
# segmented_axpby backward
# ---------------------------------------------------------------------------


class TestSegmentedAxpbyBackward:
    def test_backward_scalar(self, device):
        rng = np.random.default_rng(34)
        idx = _make_idx(N, M, rng)
        x = rng.standard_normal(N).astype(np.float32)
        y = rng.standard_normal(N).astype(np.float32)
        a = rng.standard_normal(M).astype(np.float32)
        b = rng.standard_normal(M).astype(np.float32)
        g_out = rng.standard_normal(N).astype(np.float32)

        grad_x = wp.zeros(N, dtype=wp.float32, device=device)
        grad_y = wp.zeros(N, dtype=wp.float32, device=device)
        grad_a = wp.zeros(M, dtype=wp.float32, device=device)
        grad_b = wp.zeros(M, dtype=wp.float32, device=device)
        segmented_axpby_backward(
            _wpa(g_out, wp.float32, device),
            _wpa(a, wp.float32, device),
            _wpa(x, wp.float32, device),
            _wpa(b, wp.float32, device),
            _wpa(y, wp.float32, device),
            _wpa(idx, wp.int32, device),
            M,
            grad_x,
            grad_y,
            grad_a,
            grad_b,
        )
        np.testing.assert_allclose(_np(grad_x), a[idx] * g_out, rtol=1e-4)
        np.testing.assert_allclose(_np(grad_y), b[idx] * g_out, rtol=1e-4)
        np.testing.assert_allclose(_np(grad_a), _seg_sum(x * g_out, idx, M), rtol=1e-4)
        np.testing.assert_allclose(_np(grad_b), _seg_sum(y * g_out, idx, M), rtol=1e-4)

    def test_backward_vec3(self, device):
        rng = np.random.default_rng(45)
        idx = _make_idx(N, M, rng)
        x = rng.standard_normal((N, 3)).astype(np.float32)
        y = rng.standard_normal((N, 3)).astype(np.float32)
        a = rng.standard_normal(M).astype(np.float32)
        b = rng.standard_normal(M).astype(np.float32)
        g_out = rng.standard_normal((N, 3)).astype(np.float32)

        ref_grad_x = a[idx, None] * g_out
        ref_grad_y = b[idx, None] * g_out
        ref_grad_a = _seg_sum((x * g_out).sum(axis=1), idx, M)
        ref_grad_b = _seg_sum((y * g_out).sum(axis=1), idx, M)

        grad_x = wp.zeros(N, dtype=wp.vec3f, device=device)
        grad_y = wp.zeros(N, dtype=wp.vec3f, device=device)
        grad_a = wp.zeros(M, dtype=wp.float32, device=device)
        grad_b = wp.zeros(M, dtype=wp.float32, device=device)
        segmented_axpby_backward(
            _wpv(g_out, wp.vec3f, device),
            _wpa(a, wp.float32, device),
            _wpv(x, wp.vec3f, device),
            _wpa(b, wp.float32, device),
            _wpv(y, wp.vec3f, device),
            _wpa(idx, wp.int32, device),
            M,
            grad_x,
            grad_y,
            grad_a,
            grad_b,
        )
        np.testing.assert_allclose(_np(grad_x), ref_grad_x, rtol=1e-4)
        np.testing.assert_allclose(_np(grad_y), ref_grad_y, rtol=1e-4)
        np.testing.assert_allclose(_np(grad_a), ref_grad_a, rtol=1e-4)
        np.testing.assert_allclose(_np(grad_b), ref_grad_b, rtol=1e-4)

    def test_double_backward_scalar(self, device):
        rng = np.random.default_rng(46)
        idx = _make_idx(N, M, rng)
        x = rng.standard_normal(N).astype(np.float32)
        y = rng.standard_normal(N).astype(np.float32)
        a = rng.standard_normal(M).astype(np.float32)
        b = rng.standard_normal(M).astype(np.float32)
        g_out = rng.standard_normal(N).astype(np.float32)
        gg_gx = rng.standard_normal(N).astype(np.float32)
        gg_gy = rng.standard_normal(N).astype(np.float32)
        gg_ga = rng.standard_normal(M).astype(np.float32)
        gg_gb = rng.standard_normal(M).astype(np.float32)

        ref_grad_g_out = (
            gg_gx * a[idx] + gg_gy * b[idx] + gg_ga[idx] * x + gg_gb[idx] * y
        )
        ref_grad_x_extra = gg_ga[idx] * g_out
        ref_grad_y_extra = gg_gb[idx] * g_out
        ref_grad_a_extra = _seg_sum(gg_gx * g_out, idx, M)
        ref_grad_b_extra = _seg_sum(gg_gy * g_out, idx, M)

        grad_g_out = wp.zeros(N, dtype=wp.float32, device=device)
        grad_x_extra = wp.zeros(N, dtype=wp.float32, device=device)
        grad_y_extra = wp.zeros(N, dtype=wp.float32, device=device)
        grad_a_extra = wp.zeros(M, dtype=wp.float32, device=device)
        grad_b_extra = wp.zeros(M, dtype=wp.float32, device=device)
        segmented_axpby_double_backward(
            _wpa(gg_gx, wp.float32, device),
            _wpa(gg_gy, wp.float32, device),
            _wpa(gg_ga, wp.float32, device),
            _wpa(gg_gb, wp.float32, device),
            _wpa(g_out, wp.float32, device),
            _wpa(a, wp.float32, device),
            _wpa(x, wp.float32, device),
            _wpa(b, wp.float32, device),
            _wpa(y, wp.float32, device),
            _wpa(idx, wp.int32, device),
            grad_g_out,
            grad_x_extra,
            grad_y_extra,
            grad_a_extra,
            grad_b_extra,
        )
        np.testing.assert_allclose(_np(grad_g_out), ref_grad_g_out, rtol=1e-4)
        np.testing.assert_allclose(_np(grad_x_extra), ref_grad_x_extra, rtol=1e-4)
        np.testing.assert_allclose(_np(grad_y_extra), ref_grad_y_extra, rtol=1e-4)
        np.testing.assert_allclose(_np(grad_a_extra), ref_grad_a_extra, rtol=1e-4)
        np.testing.assert_allclose(_np(grad_b_extra), ref_grad_b_extra, rtol=1e-4)


# ---------------------------------------------------------------------------
# Edge cases: empty inputs should not crash
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_sum_backward(self, device):
        g_out = wp.zeros(M, dtype=wp.float32, device=device)
        idx = wp.zeros(0, dtype=wp.int32, device=device)
        grad_x = wp.zeros(0, dtype=wp.float32, device=device)
        segmented_sum_backward(g_out, idx, grad_x)
        wp.synchronize()

    def test_empty_dot_backward(self, device):
        g_out = wp.zeros(M, dtype=wp.float32, device=device)
        x = wp.zeros(0, dtype=wp.float32, device=device)
        y = wp.zeros(0, dtype=wp.float32, device=device)
        idx = wp.zeros(0, dtype=wp.int32, device=device)
        grad_x = wp.zeros(0, dtype=wp.float32, device=device)
        grad_y = wp.zeros(0, dtype=wp.float32, device=device)
        segmented_dot_backward(g_out, x, y, idx, grad_x, grad_y)
        wp.synchronize()

    def test_empty_rms_norm_backward(self, device):
        g_out = wp.zeros(M, dtype=wp.float32, device=device)
        x = wp.zeros(0, dtype=wp.vec3f, device=device)
        inv_norm = wp.zeros(M, dtype=wp.float32, device=device)
        idx = wp.zeros(0, dtype=wp.int32, device=device)
        grad_x = wp.zeros(0, dtype=wp.vec3f, device=device)
        segmented_rms_norm_backward(g_out, x, inv_norm, idx, grad_x)
        wp.synchronize()

    def test_empty_reduction_zeros_prefilled_output(self, device):
        """An empty reduction must still zero its pre-filled output buffer.

        Encodes the reviewer's minimal repro: the mathematical result of
        ``sum_i gg_x[i]`` over an empty input is zero, but earlier versions
        early-returned without clearing ``grad_g_out``, leaving stale values.
        """
        gg_x = wp.zeros(0, dtype=wp.float32, device=device)
        idx = wp.zeros(0, dtype=wp.int32, device=device)
        out = wp.array([5.0, 6.0], dtype=wp.float32, device=device)
        segmented_sum_double_backward(gg_x, idx, 2, out)
        np.testing.assert_array_equal(_np(out), np.zeros(2, dtype=np.float32))

    def test_empty_fused_double_backward_zeros_grad_g_out(self, device):
        """Fused-kernel double-backwards must zero ``grad_g_out`` on empty input.

        Three launchers write ``grad_g_out`` via assignment in a single fused
        kernel (no separate reduction or accumulator), so the empty-input
        early-return previously skipped output zeroing entirely.  This test
        pins the contract for all three: a re-used buffer must come back as
        zeros.
        """
        from nvalchemiops.segment_ops_backward import (
            segmented_add_double_backward,
            segmented_axpy_double_backward,
            segmented_matvec_double_backward,
        )

        # add: grad_g_out[i] = gg_x[i] + gg_y[idx[i]]
        gg_x = wp.zeros(0, dtype=wp.float32, device=device)
        gg_y = wp.zeros(2, dtype=wp.float32, device=device)
        idx0 = wp.zeros(0, dtype=wp.int32, device=device)
        grad_g_out = wp.array([7.0, 8.0, 9.0], dtype=wp.float32, device=device)
        segmented_add_double_backward(gg_x, gg_y, idx0, grad_g_out)
        np.testing.assert_array_equal(_np(grad_g_out), np.zeros(3, dtype=np.float32))

        # axpy: grad_g_out[i] = gg_gy_in[i] + gg_gx[i]*a[s] + gg_ga[s]*x[i]
        gg_gy_in = wp.zeros(0, dtype=wp.float32, device=device)
        gg_gx = wp.zeros(0, dtype=wp.float32, device=device)
        gg_ga = wp.zeros(2, dtype=wp.float32, device=device)
        g_out = wp.zeros(0, dtype=wp.float32, device=device)
        x = wp.zeros(0, dtype=wp.float32, device=device)
        a = wp.zeros(2, dtype=wp.float32, device=device)
        grad_g_out = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device=device)
        grad_x_extra = wp.zeros(0, dtype=wp.float32, device=device)
        grad_a_extra = wp.zeros(2, dtype=wp.float32, device=device)
        segmented_axpy_double_backward(
            gg_gy_in,
            gg_gx,
            gg_ga,
            g_out,
            x,
            a,
            idx0,
            grad_g_out,
            grad_x_extra,
            grad_a_extra,
        )
        np.testing.assert_array_equal(_np(grad_g_out), np.zeros(3, dtype=np.float32))

        # matvec: grad_g_out[i] = m[s]^T @ gg_gv[i] + gg_gM[s]^T @ v[i]
        gg_gv = wp.zeros(0, dtype=wp.vec3f, device=device)
        gg_gM = wp.zeros(2, dtype=wp.mat33f, device=device)
        g_out_v = wp.zeros(0, dtype=wp.vec3f, device=device)
        v = wp.zeros(0, dtype=wp.vec3f, device=device)
        m = wp.zeros(2, dtype=wp.mat33f, device=device)
        # Seed grad_g_out with non-zero junk so a missed zero is visible.
        grad_g_out = _wpv(
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
            wp.vec3f,
            device,
        )
        grad_v_extra = wp.zeros(0, dtype=wp.vec3f, device=device)
        grad_M_extra = wp.zeros(2, dtype=wp.mat33f, device=device)
        segmented_matvec_double_backward(
            gg_gv, gg_gM, g_out_v, v, m, idx0, grad_g_out, grad_v_extra, grad_M_extra
        )
        np.testing.assert_array_equal(
            _np(grad_g_out), np.zeros((2, 3), dtype=np.float32)
        )


# ---------------------------------------------------------------------------
# Hard-coded regression: pin specific kernel outputs to literal values.
#
# Unlike the parametrised tests above (which compare against numpy/torch
# references and accept any equivalent answer), these tests fail if the
# kernels start producing different *numerical* values than the original
# implementation — catching silent regressions that the equivalence-based
# tests would let through.
# ---------------------------------------------------------------------------


class TestHardcoded:
    """Bytewise-pinned regression tests against hand-computed expectations."""

    def test_sum_backward_pinned(self, device):
        """``segmented_sum_backward`` is a gather: grad_x[i] = g_out[idx[i]]."""
        # Three elements; segment 0 contains element 0; segment 1 contains
        # elements 1 and 2.  Upstream gradient ``g_out`` is [3.5, -1.25].
        # Expected gather → ``grad_x = [3.5, -1.25, -1.25]``.
        idx = wp.array(np.array([0, 1, 1], dtype=np.int32), device=device)
        g_out = wp.array(np.array([3.5, -1.25], dtype=np.float32), device=device)
        grad_x = wp.zeros(3, dtype=wp.float32, device=device)
        segmented_sum_backward(g_out, idx, grad_x)
        np.testing.assert_array_equal(
            _np(grad_x), np.array([3.5, -1.25, -1.25], dtype=np.float32)
        )

    def test_sum_double_backward_pinned(self, device):
        """``segmented_sum_double_backward`` is a scatter-sum: same as fwd."""
        # gg_x = [1, 2, 3, 4]; idx sends elements (0,1) to segment 0 and (2,3) to segment 1.
        # Expected scatter-sum → ``grad_g_out = [1+2, 3+4] = [3, 7]``.
        idx = wp.array(np.array([0, 0, 1, 1], dtype=np.int32), device=device)
        gg_x = wp.array(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32), device=device)
        grad_g_out = wp.zeros(2, dtype=wp.float32, device=device)
        segmented_sum_double_backward(gg_x, idx, 2, grad_g_out)
        np.testing.assert_array_equal(
            _np(grad_g_out), np.array([3.0, 7.0], dtype=np.float32)
        )


# ---------------------------------------------------------------------------
# Tile fast-path: ``_launch_sum`` switches to a tile-based total-sum kernel
# (``wp.launch_tiled`` + ``wp.tile_*``) when ``M == 1`` and ``N >= 8192``.
# The forward suite (``test/test_segment_ops.py``) covers this regime
# explicitly; these tests pin the same path through the backward launchers
# that reuse ``_launch_sum`` internally:
#
# * ``segmented_sum_double_backward`` calls ``_launch_sum`` directly.
# * ``segmented_broadcast_backward`` is a thin alias of ``_launch_sum``.
# * ``segmented_mean_double_backward`` uses ``_launch_sum`` to build
#   per-segment sums then divides by ``counts``.
#
# Each test pairs a multiple-of-``_BLOCK_DIM`` size (pure tile path) with a
# non-multiple (tile + tail) — matching the forward suite's pattern.
# ---------------------------------------------------------------------------


class TestTilePath:
    """Pin the M=1, N>=8192 tile fast path through backward launchers."""

    @pytest.mark.parametrize("N_tile", [10240, 8500])  # pure tile, then tile+tail
    def test_sum_double_backward_tile_scalar(self, device, N_tile):
        rng = np.random.default_rng(100)
        gg_x_np = rng.standard_normal(N_tile).astype(np.float32)
        idx_np = np.zeros(N_tile, dtype=np.int32)  # M = 1
        gg_x = _wpa(gg_x_np, wp.float32, device)
        idx = _wpa(idx_np, wp.int32, device)
        grad_g_out = wp.array([7.5], dtype=wp.float32, device=device)  # seed non-zero
        segmented_sum_double_backward(gg_x, idx, 1, grad_g_out)
        np.testing.assert_allclose(
            _np(grad_g_out), np.array([gg_x_np.sum()], dtype=np.float32), rtol=1e-5
        )

    @pytest.mark.parametrize("N_tile", [10240, 8500])
    def test_sum_double_backward_tile_vec3(self, device, N_tile):
        rng = np.random.default_rng(101)
        gg_x_np = rng.standard_normal((N_tile, 3)).astype(np.float32)
        idx_np = np.zeros(N_tile, dtype=np.int32)
        gg_x = _wpv(gg_x_np, wp.vec3f, device)
        idx = _wpa(idx_np, wp.int32, device)
        grad_g_out = wp.zeros(1, dtype=wp.vec3f, device=device)
        segmented_sum_double_backward(gg_x, idx, 1, grad_g_out)
        np.testing.assert_allclose(
            _np(grad_g_out),
            np.array([gg_x_np.sum(axis=0)], dtype=np.float32),
            rtol=1e-4,
        )

    @pytest.mark.parametrize("N_tile", [10240, 8500])
    def test_broadcast_backward_tile_scalar(self, device, N_tile):
        rng = np.random.default_rng(102)
        g_out_np = rng.standard_normal(N_tile).astype(np.float32)
        idx_np = np.zeros(N_tile, dtype=np.int32)
        g_out = _wpa(g_out_np, wp.float32, device)
        idx = _wpa(idx_np, wp.int32, device)
        grad_values = wp.array([-3.2], dtype=wp.float32, device=device)
        segmented_broadcast_backward(g_out, idx, 1, grad_values)
        np.testing.assert_allclose(
            _np(grad_values), np.array([g_out_np.sum()], dtype=np.float32), rtol=1e-5
        )

    @pytest.mark.parametrize("N_tile", [10240, 8500])
    def test_mean_double_backward_tile_scalar(self, device, N_tile):
        rng = np.random.default_rng(103)
        gg_x_np = rng.standard_normal(N_tile).astype(np.float32)
        idx_np = np.zeros(N_tile, dtype=np.int32)
        counts_np = np.array([N_tile], dtype=np.int32)
        gg_x = _wpa(gg_x_np, wp.float32, device)
        idx = _wpa(idx_np, wp.int32, device)
        counts = _wpa(counts_np, wp.int32, device)
        grad_g_out = wp.zeros(1, dtype=wp.float32, device=device)
        segmented_mean_double_backward(gg_x, counts, idx, grad_g_out)
        # mean → grad_g_out[s] = sum(gg_x) / count[s]
        np.testing.assert_allclose(
            _np(grad_g_out),
            np.array([gg_x_np.sum() / N_tile], dtype=np.float32),
            rtol=1e-5,
        )
