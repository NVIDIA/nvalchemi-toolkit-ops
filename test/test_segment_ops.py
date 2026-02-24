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

"""Tests for nvalchemiops.segment_ops."""

import numpy as np
import pytest
import warp as wp

from nvalchemiops.segment_ops import (
    compute_ept,
    segment_div,
    segmented_add,
    segmented_axpby,
    segmented_axpy,
    segmented_broadcast,
    segmented_component_sum,
    segmented_count,
    segmented_dot,
    segmented_inner_products,
    segmented_matvec,
    segmented_max,
    segmented_max_norm,
    segmented_mean,
    segmented_min,
    segmented_mul,
    segmented_rms_norm,
    segmented_sum,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Map numpy scalar dtype to the corresponding warp scalar dtype.
NP_TO_WP_SCALAR = {
    np.float16: wp.float16,
    np.float32: wp.float32,
    np.float64: wp.float64,
}


def _half_atol(np_dtype):
    """Return a suitable absolute tolerance for the given numpy dtype."""
    return 5e-2 if np_dtype == np.float16 else 0


def _randn(rng, shape, np_dtype):
    """Generate random test data safe for the given dtype.

    Uses ``uniform(-2, 2)`` for float16 to avoid overflow, and
    ``standard_normal`` for wider types.
    """
    if np_dtype == np.float16:
        return rng.uniform(-2, 2, size=shape).astype(np_dtype)
    return rng.standard_normal(shape).astype(np_dtype)


def _numpy_segmented_sum(x_np: np.ndarray, idx_np: np.ndarray, M: int) -> np.ndarray:
    """Reference segmented sum computed on the host."""
    if x_np.ndim == 1:
        out = np.zeros(M, dtype=x_np.dtype)
    else:
        out = np.zeros((M, x_np.shape[1]), dtype=x_np.dtype)
    np.add.at(out, idx_np, x_np)
    return out


def _make_arrays(x_np, idx_np, M, wp_dtype, device):
    """Build warp arrays from numpy data."""
    x = wp.array(x_np, dtype=wp_dtype, device=device)
    idx = wp.array(idx_np.astype(np.int32), device=device)
    out = wp.zeros(M, dtype=wp_dtype, device=device)
    return x, idx, out


def _wp_vec_array(x_np, wp_dtype, device):
    """Convert (N,3) numpy to warp vec3 array."""
    return wp.array([tuple(r) for r in x_np], dtype=wp_dtype, device=device)


def _make_segments(N, M, rng):
    """Return sorted idx_np of length N with values in [0, M)."""
    return np.sort(rng.integers(0, M, size=N).astype(np.int32))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _available_devices():
    devices = ["cpu"]
    if wp.is_cuda_available():
        devices.append("cuda:0")
    return devices


@pytest.fixture(scope="module", params=_available_devices())
def device(request):
    return request.param


# ---------------------------------------------------------------------------
# Scalar tests
# ---------------------------------------------------------------------------


class TestScalarSegmentReduce:
    """Segmented sum for float16, float32 and float64."""

    @pytest.mark.parametrize(
        "wp_dtype,np_dtype,rtol",
        [
            (wp.float32, np.float32, 1e-5),
            (wp.float64, np.float64, 1e-12),
        ],
    )
    def test_uniform_segments(self, device, wp_dtype, np_dtype, rtol):
        N, M = 1200, 12
        seg_len = N // M
        idx_np = np.repeat(np.arange(M, dtype=np.int32), seg_len)
        x_np = np.random.default_rng(42).standard_normal(N).astype(np_dtype)

        ref = _numpy_segmented_sum(x_np, idx_np, M)
        x, idx, out = _make_arrays(x_np, idx_np, M, wp_dtype, device)

        segmented_sum(x, idx, out)
        wp.synchronize()
        np.testing.assert_allclose(out.numpy(), ref, rtol=rtol)

    @pytest.mark.parametrize(
        "wp_dtype,np_dtype,rtol",
        [
            (wp.float32, np.float32, 1e-5),
            (wp.float64, np.float64, 1e-12),
        ],
    )
    def test_skewed_segments(self, device, wp_dtype, np_dtype, rtol):
        """Mix of large and tiny segments."""
        rng = np.random.default_rng(7)
        lengths = [500, 1, 3, 200, 2, 1, 50, 1, 1, 100]
        M = len(lengths)
        idx_np = np.concatenate(
            [np.full(length, s, dtype=np.int32) for s, length in enumerate(lengths)]
        )
        N = len(idx_np)
        x_np = rng.standard_normal(N).astype(np_dtype)

        ref = _numpy_segmented_sum(x_np, idx_np, M)
        x, idx, out = _make_arrays(x_np, idx_np, M, wp_dtype, device)

        segmented_sum(x, idx, out)
        wp.synchronize()
        np.testing.assert_allclose(out.numpy(), ref, rtol=rtol)

    def test_single_segment(self, device):
        N, M = 1000, 1
        rng = np.random.default_rng(0)
        x_np = rng.standard_normal(N).astype(np.float32)
        idx_np = np.zeros(N, dtype=np.int32)

        ref = _numpy_segmented_sum(x_np, idx_np, M)
        x, idx, out = _make_arrays(x_np, idx_np, M, wp.float32, device)

        segmented_sum(x, idx, out)
        wp.synchronize()
        np.testing.assert_allclose(out.numpy(), ref, rtol=1e-5)

    def test_all_singletons(self, device):
        N = 256
        M = N
        x_np = np.arange(N, dtype=np.float32)
        idx_np = np.arange(N, dtype=np.int32)

        ref = _numpy_segmented_sum(x_np, idx_np, M)
        x, idx, out = _make_arrays(x_np, idx_np, M, wp.float32, device)

        segmented_sum(x, idx, out)
        wp.synchronize()
        np.testing.assert_allclose(out.numpy(), ref, rtol=1e-6)

    @pytest.mark.parametrize(
        "wp_dtype,np_dtype,rtol",
        [
            (wp.float32, np.float32, 1e-4),
            (wp.float64, np.float64, 1e-12),
        ],
    )
    def test_single_segment_large(self, device, wp_dtype, np_dtype, rtol):
        """Large M=1 case -- exercises the tile block-reduction path."""
        N, M = 10_000, 1
        rng = np.random.default_rng(55)
        x_np = rng.standard_normal(N).astype(np_dtype)
        idx_np = np.zeros(N, dtype=np.int32)

        ref = _numpy_segmented_sum(x_np, idx_np, M)
        x, idx, out = _make_arrays(x_np, idx_np, M, wp_dtype, device)

        segmented_sum(x, idx, out)
        wp.synchronize()
        np.testing.assert_allclose(out.numpy(), ref, rtol=rtol)

    def test_single_segment_not_aligned(self, device):
        """M=1 with N not a multiple of tile size -- exercises tail path."""
        N, M = 1_000, 1  # 1000 % 256 == 232
        rng = np.random.default_rng(77)
        x_np = rng.standard_normal(N).astype(np.float32)
        idx_np = np.zeros(N, dtype=np.int32)

        ref = _numpy_segmented_sum(x_np, idx_np, M)
        x, idx, out = _make_arrays(x_np, idx_np, M, wp.float32, device)

        segmented_sum(x, idx, out)
        wp.synchronize()
        np.testing.assert_allclose(out.numpy(), ref, rtol=1e-4)

    @pytest.mark.parametrize(
        "wp_dtype,np_dtype,rtol",
        [
            (wp.float32, np.float32, 1e-4),
            (wp.float64, np.float64, 1e-12),
        ],
    )
    def test_single_segment_remainder(self, device, wp_dtype, np_dtype, rtol):
        """M=1 with N=8500 -- exercises tile fast path + remainder tail."""
        N, M = 8500, 1
        rng = np.random.default_rng(78)
        x_np = rng.standard_normal(N).astype(np_dtype)
        idx_np = np.zeros(N, dtype=np.int32)

        ref = _numpy_segmented_sum(x_np, idx_np, M)
        x, idx, out = _make_arrays(x_np, idx_np, M, wp_dtype, device)

        segmented_sum(x, idx, out)
        wp.synchronize()
        np.testing.assert_allclose(out.numpy(), ref, rtol=rtol)

    def test_empty_input(self, device):
        M = 5
        x = wp.zeros(0, dtype=wp.float32, device=device)
        idx = wp.zeros(0, dtype=wp.int32, device=device)
        out = wp.zeros(M, dtype=wp.float32, device=device)

        segmented_sum(x, idx, out)
        wp.synchronize()
        np.testing.assert_array_equal(out.numpy(), np.zeros(M, dtype=np.float32))


# ---------------------------------------------------------------------------
# Vector tests
# ---------------------------------------------------------------------------


class TestVectorSegmentReduce:
    """Segmented sum for vec3h, vec3f and vec3d."""

    @pytest.mark.parametrize(
        "wp_dtype,np_dtype,rtol",
        [
            (wp.vec3f, np.float32, 1e-4),
            (wp.vec3d, np.float64, 1e-12),
        ],
    )
    def test_uniform_segments(self, device, wp_dtype, np_dtype, rtol):
        N, M = 600, 6
        seg_len = N // M
        idx_np = np.repeat(np.arange(M, dtype=np.int32), seg_len)
        x_np = np.random.default_rng(99).standard_normal((N, 3)).astype(np_dtype)

        ref = _numpy_segmented_sum(x_np, idx_np, M)
        x_tuples = [tuple(row) for row in x_np]
        x = wp.array(x_tuples, dtype=wp_dtype, device=device)
        idx = wp.array(idx_np, device=device)
        out = wp.zeros(M, dtype=wp_dtype, device=device)

        segmented_sum(x, idx, out)
        wp.synchronize()
        np.testing.assert_allclose(out.numpy(), ref, rtol=rtol)

    @pytest.mark.parametrize(
        "wp_dtype,np_dtype,rtol",
        [
            (wp.vec3f, np.float32, 1e-4),
            (wp.vec3d, np.float64, 1e-12),
        ],
    )
    def test_single_segment_vec(self, device, wp_dtype, np_dtype, rtol):
        N, M = 200, 1
        rng = np.random.default_rng(11)
        x_np = rng.standard_normal((N, 3)).astype(np_dtype)
        idx_np = np.zeros(N, dtype=np.int32)

        ref = _numpy_segmented_sum(x_np, idx_np, M)
        x = wp.array([tuple(r) for r in x_np], dtype=wp_dtype, device=device)
        idx = wp.array(idx_np, device=device)
        out = wp.zeros(M, dtype=wp_dtype, device=device)

        segmented_sum(x, idx, out)
        wp.synchronize()
        np.testing.assert_allclose(out.numpy(), ref, rtol=rtol)

    @pytest.mark.parametrize(
        "wp_dtype,np_dtype,rtol",
        [
            (wp.vec3f, np.float32, 1e-3),
            (wp.vec3d, np.float64, 1e-12),
        ],
    )
    def test_single_segment_large_vec(self, device, wp_dtype, np_dtype, rtol):
        """Large M=1 vec3 case -- exercises the tile block-reduction path."""
        N, M = 10_000, 1
        rng = np.random.default_rng(88)
        x_np = rng.standard_normal((N, 3)).astype(np_dtype)
        idx_np = np.zeros(N, dtype=np.int32)

        ref = _numpy_segmented_sum(x_np, idx_np, M)
        x = wp.array([tuple(r) for r in x_np], dtype=wp_dtype, device=device)
        idx = wp.array(idx_np, device=device)
        out = wp.zeros(M, dtype=wp_dtype, device=device)

        segmented_sum(x, idx, out)
        wp.synchronize()
        np.testing.assert_allclose(out.numpy(), ref, rtol=rtol)

    @pytest.mark.parametrize(
        "wp_dtype,np_dtype,rtol",
        [
            (wp.vec3f, np.float32, 1e-3),
            (wp.vec3d, np.float64, 1e-12),
        ],
    )
    def test_single_segment_remainder(self, device, wp_dtype, np_dtype, rtol):
        """M=1 with N=8500 -- tile fast path + remainder tail for vec3."""
        N, M = 8500, 1
        rng = np.random.default_rng(89)
        x_np = rng.standard_normal((N, 3)).astype(np_dtype)
        idx_np = np.zeros(N, dtype=np.int32)

        ref = _numpy_segmented_sum(x_np, idx_np, M)
        x = wp.array([tuple(r) for r in x_np], dtype=wp_dtype, device=device)
        idx = wp.array(idx_np, device=device)
        out = wp.zeros(M, dtype=wp_dtype, device=device)

        segmented_sum(x, idx, out)
        wp.synchronize()
        np.testing.assert_allclose(out.numpy(), ref, rtol=rtol)


# ---------------------------------------------------------------------------
# segmented_component_sum tests
# ---------------------------------------------------------------------------


class TestSegmentedComponentSum:
    @pytest.mark.parametrize(
        "wp_vec,np_dtype,rtol",
        [
            (wp.vec3f, np.float32, 1e-4),
            (wp.vec3d, np.float64, 1e-12),
        ],
    )
    def test_uniform(self, device, wp_vec, np_dtype, rtol):
        N, M = 600, 6
        rng = np.random.default_rng(10)
        idx_np = np.repeat(np.arange(M, dtype=np.int32), N // M)
        x_np = rng.standard_normal((N, 3)).astype(np_dtype)

        ref = np.zeros(M, dtype=np_dtype)
        np.add.at(ref, idx_np, x_np.sum(axis=1))

        x = _wp_vec_array(x_np, wp_vec, device)
        idx = wp.array(idx_np, device=device)
        out_dtype = NP_TO_WP_SCALAR[np_dtype]
        out = wp.zeros(M, dtype=out_dtype, device=device)

        segmented_component_sum(x, idx, out)
        wp.synchronize()
        np.testing.assert_allclose(out.numpy(), ref, rtol=rtol)

    @pytest.mark.parametrize(
        "wp_vec,np_dtype,rtol",
        [
            (wp.vec3f, np.float32, 1e-4),
            (wp.vec3d, np.float64, 1e-12),
        ],
    )
    def test_skewed(self, device, wp_vec, np_dtype, rtol):
        rng = np.random.default_rng(20)
        lengths = [300, 1, 50, 2, 100]
        M = len(lengths)
        idx_np = np.concatenate(
            [np.full(length, s, dtype=np.int32) for s, length in enumerate(lengths)]
        )
        N = len(idx_np)
        x_np = rng.standard_normal((N, 3)).astype(np_dtype)

        ref = np.zeros(M, dtype=np_dtype)
        np.add.at(ref, idx_np, x_np.sum(axis=1))

        x = _wp_vec_array(x_np, wp_vec, device)
        idx = wp.array(idx_np, device=device)
        out_dtype = NP_TO_WP_SCALAR[np_dtype]
        out = wp.zeros(M, dtype=out_dtype, device=device)

        segmented_component_sum(x, idx, out)
        wp.synchronize()
        np.testing.assert_allclose(out.numpy(), ref, rtol=rtol)

    @pytest.mark.parametrize(
        "wp_vec,np_dtype,rtol",
        [
            (wp.vec3f, np.float32, 1e-3),
            (wp.vec3d, np.float64, 1e-12),
        ],
    )
    def test_single_segment_large(self, device, wp_vec, np_dtype, rtol):
        """M=1, N=10000 -- exercises tile fast path."""
        N, M = 10_000, 1
        rng = np.random.default_rng(21)
        x_np = rng.standard_normal((N, 3)).astype(np_dtype)
        idx_np = np.zeros(N, dtype=np.int32)

        ref = np.array([x_np.sum()], dtype=np_dtype)  # sum of all components

        x = _wp_vec_array(x_np, wp_vec, device)
        idx = wp.array(idx_np, device=device)
        out_dtype = NP_TO_WP_SCALAR[np_dtype]
        out = wp.zeros(M, dtype=out_dtype, device=device)

        segmented_component_sum(x, idx, out)
        wp.synchronize()
        np.testing.assert_allclose(out.numpy(), ref, rtol=rtol)

    @pytest.mark.parametrize(
        "wp_vec,np_dtype,rtol",
        [
            (wp.vec3f, np.float32, 1e-3),
            (wp.vec3d, np.float64, 1e-12),
        ],
    )
    def test_single_segment_remainder(self, device, wp_vec, np_dtype, rtol):
        """M=1, N=8500 -- tile fast path + remainder."""
        N, M = 8500, 1
        rng = np.random.default_rng(22)
        x_np = rng.standard_normal((N, 3)).astype(np_dtype)
        idx_np = np.zeros(N, dtype=np.int32)

        ref = np.array([x_np.sum()], dtype=np_dtype)

        x = _wp_vec_array(x_np, wp_vec, device)
        idx = wp.array(idx_np, device=device)
        out_dtype = NP_TO_WP_SCALAR[np_dtype]
        out = wp.zeros(M, dtype=out_dtype, device=device)

        segmented_component_sum(x, idx, out)
        wp.synchronize()
        np.testing.assert_allclose(out.numpy(), ref, rtol=rtol)

    def test_empty_input(self, device):
        x = wp.zeros(0, dtype=wp.vec3f, device=device)
        idx = wp.zeros(0, dtype=wp.int32, device=device)
        out = wp.zeros(3, dtype=wp.float32, device=device)

        segmented_component_sum(x, idx, out)
        wp.synchronize()
        np.testing.assert_array_equal(out.numpy(), np.zeros(3, dtype=np.float32))


# ---------------------------------------------------------------------------
# segmented_dot tests
# ---------------------------------------------------------------------------


class TestSegmentedDot:
    @pytest.mark.parametrize(
        "wp_dtype,np_dtype,rtol",
        [
            (wp.float32, np.float32, 1e-4),
            (wp.float64, np.float64, 1e-12),
        ],
    )
    def test_scalar(self, device, wp_dtype, np_dtype, rtol):
        N, M = 1200, 12
        rng = np.random.default_rng(30)
        idx_np = np.repeat(np.arange(M, dtype=np.int32), N // M)
        x_np = rng.standard_normal(N).astype(np_dtype)
        y_np = rng.standard_normal(N).astype(np_dtype)

        ref = np.zeros(M, dtype=np_dtype)
        np.add.at(ref, idx_np, x_np * y_np)

        x = wp.array(x_np, dtype=wp_dtype, device=device)
        y = wp.array(y_np, dtype=wp_dtype, device=device)
        idx = wp.array(idx_np, device=device)
        out = wp.zeros(M, dtype=wp_dtype, device=device)

        segmented_dot(x, y, idx, out)
        wp.synchronize()
        np.testing.assert_allclose(out.numpy(), ref, rtol=rtol)

    @pytest.mark.parametrize(
        "wp_vec,np_dtype,rtol",
        [
            (wp.vec3f, np.float32, 1e-3),
            (wp.vec3d, np.float64, 1e-12),
        ],
    )
    def test_vec3(self, device, wp_vec, np_dtype, rtol):
        N, M = 600, 6
        rng = np.random.default_rng(31)
        idx_np = np.repeat(np.arange(M, dtype=np.int32), N // M)
        x_np = rng.standard_normal((N, 3)).astype(np_dtype)
        y_np = rng.standard_normal((N, 3)).astype(np_dtype)

        ref = np.zeros(M, dtype=np_dtype)
        np.add.at(ref, idx_np, (x_np * y_np).sum(axis=1))

        x = _wp_vec_array(x_np, wp_vec, device)
        y = _wp_vec_array(y_np, wp_vec, device)
        idx = wp.array(idx_np, device=device)
        out_dtype = NP_TO_WP_SCALAR[np_dtype]
        out = wp.zeros(M, dtype=out_dtype, device=device)

        segmented_dot(x, y, idx, out)
        wp.synchronize()
        np.testing.assert_allclose(out.numpy(), ref, rtol=rtol)

    @pytest.mark.parametrize(
        "wp_dtype,np_dtype,rtol",
        [
            (wp.float32, np.float32, 1e-3),
            (wp.float64, np.float64, 1e-12),
        ],
    )
    def test_scalar_single_segment_large(self, device, wp_dtype, np_dtype, rtol):
        """M=1, N=10000 scalar -- exercises tile fast path."""
        N, M = 10_000, 1
        rng = np.random.default_rng(32)
        x_np = rng.standard_normal(N).astype(np_dtype)
        y_np = rng.standard_normal(N).astype(np_dtype)
        idx_np = np.zeros(N, dtype=np.int32)

        ref = np.array([(x_np * y_np).sum()], dtype=np_dtype)

        x = wp.array(x_np, dtype=wp_dtype, device=device)
        y = wp.array(y_np, dtype=wp_dtype, device=device)
        idx = wp.array(idx_np, device=device)
        out = wp.zeros(M, dtype=wp_dtype, device=device)

        segmented_dot(x, y, idx, out)
        wp.synchronize()
        np.testing.assert_allclose(out.numpy(), ref, rtol=rtol)

    @pytest.mark.parametrize(
        "wp_vec,np_dtype,rtol",
        [
            (wp.vec3f, np.float32, 1e-3),
            (wp.vec3d, np.float64, 1e-12),
        ],
    )
    def test_vec3_single_segment_large(self, device, wp_vec, np_dtype, rtol):
        """M=1, N=10000 vec3 -- exercises tile fast path."""
        N, M = 10_000, 1
        rng = np.random.default_rng(33)
        x_np = rng.standard_normal((N, 3)).astype(np_dtype)
        y_np = rng.standard_normal((N, 3)).astype(np_dtype)
        idx_np = np.zeros(N, dtype=np.int32)

        ref = np.array([(x_np * y_np).sum()], dtype=np_dtype)

        x = _wp_vec_array(x_np, wp_vec, device)
        y = _wp_vec_array(y_np, wp_vec, device)
        idx = wp.array(idx_np, device=device)
        out_dtype = NP_TO_WP_SCALAR[np_dtype]
        out = wp.zeros(M, dtype=out_dtype, device=device)

        segmented_dot(x, y, idx, out)
        wp.synchronize()
        np.testing.assert_allclose(out.numpy(), ref, rtol=rtol)

    @pytest.mark.parametrize(
        "wp_dtype,np_dtype,rtol",
        [
            (wp.float32, np.float32, 1e-3),
            (wp.float64, np.float64, 1e-12),
        ],
    )
    def test_single_segment_remainder(self, device, wp_dtype, np_dtype, rtol):
        """M=1, N=8500 -- tile + remainder."""
        N, M = 8500, 1
        rng = np.random.default_rng(34)
        x_np = rng.standard_normal(N).astype(np_dtype)
        y_np = rng.standard_normal(N).astype(np_dtype)
        idx_np = np.zeros(N, dtype=np.int32)

        ref = np.array([(x_np * y_np).sum()], dtype=np_dtype)

        x = wp.array(x_np, dtype=wp_dtype, device=device)
        y = wp.array(y_np, dtype=wp_dtype, device=device)
        idx = wp.array(idx_np, device=device)
        out = wp.zeros(M, dtype=wp_dtype, device=device)

        segmented_dot(x, y, idx, out)
        wp.synchronize()
        np.testing.assert_allclose(out.numpy(), ref, rtol=rtol)

    def test_empty_input(self, device):
        x = wp.zeros(0, dtype=wp.float32, device=device)
        y = wp.zeros(0, dtype=wp.float32, device=device)
        idx = wp.zeros(0, dtype=wp.int32, device=device)
        out = wp.zeros(3, dtype=wp.float32, device=device)

        segmented_dot(x, y, idx, out)
        wp.synchronize()
        np.testing.assert_array_equal(out.numpy(), np.zeros(3, dtype=np.float32))


# ---------------------------------------------------------------------------
# segmented_max_norm tests
# ---------------------------------------------------------------------------


class TestSegmentedMaxNorm:
    @pytest.mark.parametrize(
        "wp_vec,np_dtype,rtol",
        [
            (wp.vec3f, np.float32, 1e-5),
            (wp.vec3d, np.float64, 1e-12),
        ],
    )
    def test_uniform(self, device, wp_vec, np_dtype, rtol):
        N, M = 600, 6
        rng = np.random.default_rng(40)
        idx_np = np.repeat(np.arange(M, dtype=np.int32), N // M)
        x_np = rng.standard_normal((N, 3)).astype(np_dtype)

        norms = np.linalg.norm(x_np, axis=1).astype(np_dtype)
        ref = np.zeros(M, dtype=np_dtype)
        for s in range(M):
            mask = idx_np == s
            if mask.any():
                ref[s] = norms[mask].max()

        x = _wp_vec_array(x_np, wp_vec, device)
        idx = wp.array(idx_np, device=device)
        out_dtype = NP_TO_WP_SCALAR[np_dtype]
        out = wp.zeros(M, dtype=out_dtype, device=device)

        segmented_max_norm(x, idx, out)
        wp.synchronize()
        np.testing.assert_allclose(out.numpy(), ref, rtol=rtol)

    def test_skewed(self, device):
        rng = np.random.default_rng(41)
        lengths = [200, 1, 3, 100, 50]
        M = len(lengths)
        idx_np = np.concatenate(
            [np.full(length, s, dtype=np.int32) for s, length in enumerate(lengths)]
        )
        N = len(idx_np)
        x_np = rng.standard_normal((N, 3)).astype(np.float32)

        norms = np.linalg.norm(x_np, axis=1).astype(np.float32)
        ref = np.zeros(M, dtype=np.float32)
        for s in range(M):
            mask = idx_np == s
            if mask.any():
                ref[s] = norms[mask].max()

        x = _wp_vec_array(x_np, wp.vec3f, device)
        idx = wp.array(idx_np, device=device)
        out = wp.zeros(M, dtype=wp.float32, device=device)

        segmented_max_norm(x, idx, out)
        wp.synchronize()
        np.testing.assert_allclose(out.numpy(), ref, rtol=1e-5)

    @pytest.mark.parametrize(
        "wp_vec,np_dtype,rtol",
        [
            (wp.vec3f, np.float32, 1e-5),
            (wp.vec3d, np.float64, 1e-12),
        ],
    )
    def test_single_segment_large(self, device, wp_vec, np_dtype, rtol):
        """Large M=1 case -- exercises the total max-norm fast path."""
        N, M = 10_000, 1
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((N, 3)).astype(np_dtype)
        idx_np = np.zeros(N, dtype=np.int32)

        ref = np.array([np.linalg.norm(x_np, axis=1).max()], dtype=np_dtype)

        x = _wp_vec_array(x_np, wp_vec, device)
        idx = wp.array(idx_np, device=device)
        out_dtype = NP_TO_WP_SCALAR[np_dtype]
        out = wp.zeros(M, dtype=out_dtype, device=device)

        segmented_max_norm(x, idx, out)
        wp.synchronize()
        np.testing.assert_allclose(out.numpy(), ref, rtol=rtol)


# ---------------------------------------------------------------------------
# segmented_mul tests
# ---------------------------------------------------------------------------


class TestSegmentedMul:
    @pytest.mark.parametrize(
        "wp_dtype,np_dtype,rtol",
        [
            (wp.float16, np.float16, 5e-2),
            (wp.float32, np.float32, 1e-5),
            (wp.float64, np.float64, 1e-12),
        ],
    )
    def test_scalar_same_type(self, device, wp_dtype, np_dtype, rtol):
        N, M = 500, 5
        rng = np.random.default_rng(50)
        idx_np = _make_segments(N, M, rng)
        x_np = rng.standard_normal(N).astype(np_dtype)
        y_np = rng.standard_normal(M).astype(np_dtype)

        ref = x_np * y_np[idx_np]

        x = wp.array(x_np, dtype=wp_dtype, device=device)
        y = wp.array(y_np, dtype=wp_dtype, device=device)
        idx = wp.array(idx_np, device=device)
        out = wp.zeros(N, dtype=wp_dtype, device=device)

        segmented_mul(x, y, idx, out)
        wp.synchronize()
        np.testing.assert_allclose(out.numpy(), ref, rtol=rtol)

    @pytest.mark.parametrize(
        "wp_vec,wp_scalar,np_dtype,rtol",
        [
            (wp.vec3h, wp.float16, np.float16, 5e-2),
            (wp.vec3f, wp.float32, np.float32, 1e-5),
            (wp.vec3d, wp.float64, np.float64, 1e-12),
        ],
    )
    def test_vec_scalar(self, device, wp_vec, wp_scalar, np_dtype, rtol):
        """vec3 * scalar broadcast."""
        N, M = 500, 5
        rng = np.random.default_rng(51)
        idx_np = _make_segments(N, M, rng)
        x_np = rng.standard_normal((N, 3)).astype(np_dtype)
        y_np = rng.standard_normal(M).astype(np_dtype)

        ref = x_np * y_np[idx_np, None]

        x = _wp_vec_array(x_np, wp_vec, device)
        y = wp.array(y_np, dtype=wp_scalar, device=device)
        idx = wp.array(idx_np, device=device)
        out = wp.zeros(N, dtype=wp_vec, device=device)

        segmented_mul(x, y, idx, out)
        wp.synchronize()
        np.testing.assert_allclose(out.numpy(), ref, rtol=rtol)

    def test_empty_input(self, device):
        x = wp.zeros(0, dtype=wp.float32, device=device)
        y = wp.zeros(3, dtype=wp.float32, device=device)
        idx = wp.zeros(0, dtype=wp.int32, device=device)
        out = wp.zeros(0, dtype=wp.float32, device=device)

        segmented_mul(x, y, idx, out)
        wp.synchronize()
        np.testing.assert_array_equal(out.numpy(), np.zeros(0, dtype=np.float32))


# ---------------------------------------------------------------------------
# segmented_add tests
# ---------------------------------------------------------------------------


class TestSegmentedAdd:
    @pytest.mark.parametrize(
        "wp_dtype,np_dtype,rtol",
        [
            (wp.float16, np.float16, 5e-2),
            (wp.float32, np.float32, 1e-5),
            (wp.float64, np.float64, 1e-12),
        ],
    )
    def test_scalar_same_type(self, device, wp_dtype, np_dtype, rtol):
        N, M = 500, 5
        rng = np.random.default_rng(60)
        idx_np = _make_segments(N, M, rng)
        x_np = rng.standard_normal(N).astype(np_dtype)
        y_np = rng.standard_normal(M).astype(np_dtype)

        ref = x_np + y_np[idx_np]

        x = wp.array(x_np, dtype=wp_dtype, device=device)
        y = wp.array(y_np, dtype=wp_dtype, device=device)
        idx = wp.array(idx_np, device=device)
        out = wp.zeros(N, dtype=wp_dtype, device=device)

        segmented_add(x, y, idx, out)
        wp.synchronize()
        np.testing.assert_allclose(out.numpy(), ref, rtol=rtol)

    @pytest.mark.parametrize(
        "wp_vec,np_dtype,rtol",
        [
            (wp.vec3h, np.float16, 5e-2),
            (wp.vec3f, np.float32, 1e-5),
            (wp.vec3d, np.float64, 1e-12),
        ],
    )
    def test_vec_same_type(self, device, wp_vec, np_dtype, rtol):
        """vec3 + vec3."""
        N, M = 500, 5
        rng = np.random.default_rng(63)
        idx_np = _make_segments(N, M, rng)
        x_np = rng.standard_normal((N, 3)).astype(np_dtype)
        y_np = rng.standard_normal((M, 3)).astype(np_dtype)

        ref = x_np + y_np[idx_np]

        x = _wp_vec_array(x_np, wp_vec, device)
        y = _wp_vec_array(y_np, wp_vec, device)
        idx = wp.array(idx_np, device=device)
        out = wp.zeros(N, dtype=wp_vec, device=device)

        segmented_add(x, y, idx, out)
        wp.synchronize()
        np.testing.assert_allclose(out.numpy(), ref, rtol=rtol)

    @pytest.mark.parametrize(
        "wp_vec,np_dtype,rtol",
        [
            (wp.vec3h, np.float16, 5e-2),
            (wp.vec3f, np.float32, 1e-5),
            (wp.vec3d, np.float64, 1e-12),
        ],
    )
    def test_vec_scalar(self, device, wp_vec, np_dtype, rtol):
        """vec3 + scalar broadcast."""
        N, M = 500, 5
        rng = np.random.default_rng(61)
        idx_np = _make_segments(N, M, rng)
        x_np = rng.standard_normal((N, 3)).astype(np_dtype)
        y_np = rng.standard_normal(M).astype(np_dtype)

        ref = x_np + y_np[idx_np, None]

        x = _wp_vec_array(x_np, wp_vec, device)
        scalar_dtype = NP_TO_WP_SCALAR[np_dtype]
        y = wp.array(y_np, dtype=scalar_dtype, device=device)
        idx = wp.array(idx_np, device=device)
        out = wp.zeros(N, dtype=wp_vec, device=device)

        segmented_add(x, y, idx, out)
        wp.synchronize()
        np.testing.assert_allclose(out.numpy(), ref, rtol=rtol)

    @pytest.mark.parametrize(
        "wp_vec,np_dtype,rtol",
        [
            (wp.vec3h, np.float16, 5e-2),
            (wp.vec3f, np.float32, 1e-5),
            (wp.vec3d, np.float64, 1e-12),
        ],
    )
    def test_scalar_vec(self, device, wp_vec, np_dtype, rtol):
        """scalar + vec3 broadcast."""
        N, M = 500, 5
        rng = np.random.default_rng(62)
        idx_np = _make_segments(N, M, rng)
        x_np = rng.standard_normal(N).astype(np_dtype)
        y_np = rng.standard_normal((M, 3)).astype(np_dtype)

        ref = x_np[:, None] + y_np[idx_np]

        scalar_dtype = NP_TO_WP_SCALAR[np_dtype]
        x = wp.array(x_np, dtype=scalar_dtype, device=device)
        y = _wp_vec_array(y_np, wp_vec, device)
        idx = wp.array(idx_np, device=device)
        out = wp.zeros(N, dtype=wp_vec, device=device)

        segmented_add(x, y, idx, out)
        wp.synchronize()
        np.testing.assert_allclose(out.numpy(), ref, rtol=rtol)

    def test_empty_input(self, device):
        x = wp.zeros(0, dtype=wp.float32, device=device)
        y = wp.zeros(3, dtype=wp.float32, device=device)
        idx = wp.zeros(0, dtype=wp.int32, device=device)
        out = wp.zeros(0, dtype=wp.float32, device=device)

        segmented_add(x, y, idx, out)
        wp.synchronize()
        np.testing.assert_array_equal(out.numpy(), np.zeros(0, dtype=np.float32))


# ---------------------------------------------------------------------------
# segmented_matvec tests
# ---------------------------------------------------------------------------


class TestSegmentedMatvec:
    @pytest.mark.parametrize(
        "wp_vec,wp_mat,np_dtype,rtol",
        [
            (wp.vec3h, wp.mat33h, np.float16, 5e-2),
            (wp.vec3f, wp.mat33f, np.float32, 1e-4),
            (wp.vec3d, wp.mat33d, np.float64, 1e-12),
        ],
    )
    def test_basic(self, device, wp_vec, wp_mat, np_dtype, rtol):
        N, M = 500, 5
        rng = np.random.default_rng(70)
        idx_np = _make_segments(N, M, rng)
        v_np = _randn(rng, (N, 3), np_dtype)
        m_np = _randn(rng, (M, 3, 3), np_dtype)

        # Reference: M^T @ v
        ref = np.zeros((N, 3), dtype=np_dtype)
        for i in range(N):
            ref[i] = m_np[idx_np[i]].T @ v_np[i]

        v = _wp_vec_array(v_np, wp_vec, device)
        m_tuples = [tuple(tuple(row) for row in mat) for mat in m_np]
        m = wp.array(m_tuples, dtype=wp_mat, device=device)
        idx = wp.array(idx_np, device=device)
        out = wp.zeros(N, dtype=wp_vec, device=device)

        segmented_matvec(v, m, idx, out)
        wp.synchronize()
        np.testing.assert_allclose(
            out.numpy(), ref, rtol=rtol, atol=_half_atol(np_dtype)
        )

    @pytest.mark.parametrize(
        "wp_vec,wp_mat,np_dtype,rtol",
        [
            (wp.vec3h, wp.mat33h, np.float16, 5e-2),
            (wp.vec3f, wp.mat33f, np.float32, 1e-4),
            (wp.vec3d, wp.mat33d, np.float64, 1e-12),
        ],
    )
    def test_identity_matrices(self, device, wp_vec, wp_mat, np_dtype, rtol):
        """With identity matrices, output should equal input."""
        N, M = 200, 4
        rng = np.random.default_rng(71)
        idx_np = _make_segments(N, M, rng)
        v_np = rng.standard_normal((N, 3)).astype(np_dtype)
        m_np = np.stack([np.eye(3, dtype=np_dtype)] * M)

        v = _wp_vec_array(v_np, wp_vec, device)
        m_tuples = [tuple(tuple(row) for row in mat) for mat in m_np]
        m = wp.array(m_tuples, dtype=wp_mat, device=device)
        idx = wp.array(idx_np, device=device)
        out = wp.zeros(N, dtype=wp_vec, device=device)

        segmented_matvec(v, m, idx, out)
        wp.synchronize()
        np.testing.assert_allclose(out.numpy(), v_np, rtol=rtol)

    def test_empty_input(self, device):
        v = wp.zeros(0, dtype=wp.vec3f, device=device)
        m = wp.zeros(3, dtype=wp.mat33f, device=device)
        idx = wp.zeros(0, dtype=wp.int32, device=device)
        out = wp.zeros(0, dtype=wp.vec3f, device=device)

        segmented_matvec(v, m, idx, out)
        wp.synchronize()
        np.testing.assert_array_equal(out.numpy().shape, (0, 3))


# ---------------------------------------------------------------------------
# segmented_axpy tests
# ---------------------------------------------------------------------------


class TestSegmentedAxpy:
    """Tests for segmented_axpy: y[i] += x[i] * a[idx[i]]."""

    @pytest.mark.parametrize(
        "wp_vec,wp_scalar,np_dtype,rtol",
        [
            (wp.vec3h, wp.float16, np.float16, 5e-2),
            (wp.vec3f, wp.float32, np.float32, 1e-5),
            (wp.vec3d, wp.float64, np.float64, 1e-12),
        ],
    )
    def test_vec(self, device, wp_vec, wp_scalar, np_dtype, rtol):
        rng = np.random.default_rng(0)
        N, M = 100, 4
        x_np = _randn(rng, (N, 3), np_dtype)
        y_np = _randn(rng, (N, 3), np_dtype)
        a_np = _randn(rng, M, np_dtype)
        idx_np = np.sort(rng.integers(0, M, N)).astype(np.int32)

        x = wp.array(x_np, dtype=wp_vec, device=device)
        y = wp.array(y_np.copy(), dtype=wp_vec, device=device)
        a = wp.array(a_np, dtype=wp_scalar, device=device)
        idx = wp.array(idx_np, dtype=wp.int32, device=device)

        segmented_axpy(y, x, a, idx)
        wp.synchronize()

        expected = y_np + x_np * a_np[idx_np, None]
        result = y.numpy()
        np.testing.assert_allclose(
            result, expected, rtol=rtol, atol=_half_atol(np_dtype)
        )

    @pytest.mark.parametrize(
        "wp_dtype,np_dtype,rtol",
        [
            (wp.float32, np.float32, 1e-4),
            (wp.float64, np.float64, 1e-12),
        ],
    )
    def test_scalar(self, device, wp_dtype, np_dtype, rtol):
        rng = np.random.default_rng(4)
        N, M = 100, 4
        x_np = rng.standard_normal(N).astype(np_dtype)
        y_np = rng.standard_normal(N).astype(np_dtype)
        a_np = rng.standard_normal(M).astype(np_dtype)
        idx_np = np.sort(rng.integers(0, M, N)).astype(np.int32)

        x = wp.array(x_np, dtype=wp_dtype, device=device)
        y = wp.array(y_np.copy(), dtype=wp_dtype, device=device)
        a = wp.array(a_np, dtype=wp_dtype, device=device)
        idx = wp.array(idx_np, dtype=wp.int32, device=device)

        segmented_axpy(y, x, a, idx)
        wp.synchronize()

        expected = y_np + x_np * a_np[idx_np]
        result = y.numpy()
        np.testing.assert_allclose(result, expected, rtol=rtol)

    def test_empty_input(self, device):
        x = wp.zeros(0, dtype=wp.vec3f, device=device)
        y = wp.zeros(0, dtype=wp.vec3f, device=device)
        a = wp.zeros(3, dtype=wp.float32, device=device)
        idx = wp.zeros(0, dtype=wp.int32, device=device)

        segmented_axpy(y, x, a, idx)
        wp.synchronize()
        np.testing.assert_array_equal(y.numpy().shape, (0, 3))


# ---------------------------------------------------------------------------
# segmented_inner_products tests
# ---------------------------------------------------------------------------


class TestSegmentedInnerProducts:
    """Tests for segmented_inner_products: triple reduction."""

    @pytest.mark.parametrize(
        "wp_vec,wp_scalar,np_dtype,rtol",
        [
            (wp.vec3f, wp.float32, np.float32, 1e-4),
            (wp.vec3d, wp.float64, np.float64, 1e-10),
        ],
    )
    def test_vec(self, device, wp_vec, wp_scalar, np_dtype, rtol):
        rng = np.random.default_rng(1)
        N, M = 200, 3
        x_np = rng.standard_normal((N, 3)).astype(np_dtype)
        y_np = rng.standard_normal((N, 3)).astype(np_dtype)
        idx_np = np.sort(rng.integers(0, M, N)).astype(np.int32)

        x = wp.array(x_np, dtype=wp_vec, device=device)
        y = wp.array(y_np, dtype=wp_vec, device=device)
        idx = wp.array(idx_np, dtype=wp.int32, device=device)
        out_xy = wp.zeros(M, dtype=wp_scalar, device=device)
        out_xx = wp.zeros(M, dtype=wp_scalar, device=device)
        out_yy = wp.zeros(M, dtype=wp_scalar, device=device)

        segmented_inner_products(x, y, idx, out_xy, out_xx, out_yy)
        wp.synchronize()

        ref_xy = np.zeros(M, dtype=np_dtype)
        ref_xx = np.zeros(M, dtype=np_dtype)
        ref_yy = np.zeros(M, dtype=np_dtype)
        for i in range(N):
            s = idx_np[i]
            ref_xy[s] += np.dot(x_np[i], y_np[i])
            ref_xx[s] += np.dot(x_np[i], x_np[i])
            ref_yy[s] += np.dot(y_np[i], y_np[i])

        np.testing.assert_allclose(out_xy.numpy(), ref_xy, rtol=rtol)
        np.testing.assert_allclose(out_xx.numpy(), ref_xx, rtol=rtol)
        np.testing.assert_allclose(out_yy.numpy(), ref_yy, rtol=rtol)

    @pytest.mark.parametrize(
        "wp_dtype,np_dtype,rtol",
        [
            (wp.float32, np.float32, 1e-4),
            (wp.float64, np.float64, 1e-12),
        ],
    )
    def test_scalar(self, device, wp_dtype, np_dtype, rtol):
        rng = np.random.default_rng(2)
        N, M = 150, 5
        x_np = rng.standard_normal(N).astype(np_dtype)
        y_np = rng.standard_normal(N).astype(np_dtype)
        idx_np = np.sort(rng.integers(0, M, N)).astype(np.int32)

        x = wp.array(x_np, dtype=wp_dtype, device=device)
        y = wp.array(y_np, dtype=wp_dtype, device=device)
        idx = wp.array(idx_np, dtype=wp.int32, device=device)
        out_xy = wp.zeros(M, dtype=wp_dtype, device=device)
        out_xx = wp.zeros(M, dtype=wp_dtype, device=device)
        out_yy = wp.zeros(M, dtype=wp_dtype, device=device)

        segmented_inner_products(x, y, idx, out_xy, out_xx, out_yy)
        wp.synchronize()

        ref_xy = np.zeros(M, dtype=np_dtype)
        ref_xx = np.zeros(M, dtype=np_dtype)
        ref_yy = np.zeros(M, dtype=np_dtype)
        for i in range(N):
            s = idx_np[i]
            ref_xy[s] += x_np[i] * y_np[i]
            ref_xx[s] += x_np[i] * x_np[i]
            ref_yy[s] += y_np[i] * y_np[i]

        np.testing.assert_allclose(out_xy.numpy(), ref_xy, rtol=rtol)
        np.testing.assert_allclose(out_xx.numpy(), ref_xx, rtol=rtol)
        np.testing.assert_allclose(out_yy.numpy(), ref_yy, rtol=rtol)

    @pytest.mark.parametrize(
        "wp_dtype,np_dtype,rtol",
        [
            (wp.float32, np.float32, 1e-3),
            (wp.float64, np.float64, 1e-12),
        ],
    )
    def test_scalar_single_segment_large(self, device, wp_dtype, np_dtype, rtol):
        """M=1, N=10000 scalar -- exercises tile fast path."""
        rng = np.random.default_rng(5)
        N, M = 10_000, 1
        x_np = rng.standard_normal(N).astype(np_dtype)
        y_np = rng.standard_normal(N).astype(np_dtype)
        idx_np = np.zeros(N, dtype=np.int32)

        x = wp.array(x_np, dtype=wp_dtype, device=device)
        y = wp.array(y_np, dtype=wp_dtype, device=device)
        idx = wp.array(idx_np, dtype=wp.int32, device=device)
        out_xy = wp.zeros(M, dtype=wp_dtype, device=device)
        out_xx = wp.zeros(M, dtype=wp_dtype, device=device)
        out_yy = wp.zeros(M, dtype=wp_dtype, device=device)

        segmented_inner_products(x, y, idx, out_xy, out_xx, out_yy)
        wp.synchronize()

        np.testing.assert_allclose(out_xy.numpy(), [(x_np * y_np).sum()], rtol=rtol)
        np.testing.assert_allclose(out_xx.numpy(), [(x_np * x_np).sum()], rtol=rtol)
        np.testing.assert_allclose(out_yy.numpy(), [(y_np * y_np).sum()], rtol=rtol)

    @pytest.mark.parametrize(
        "wp_vec,wp_scalar,np_dtype,rtol",
        [
            (wp.vec3f, wp.float32, np.float32, 1e-3),
            (wp.vec3d, wp.float64, np.float64, 1e-12),
        ],
    )
    def test_vec_single_segment_large(self, device, wp_vec, wp_scalar, np_dtype, rtol):
        """M=1, N=10000 vec3 -- exercises tile fast path."""
        rng = np.random.default_rng(6)
        N, M = 10_000, 1
        x_np = rng.standard_normal((N, 3)).astype(np_dtype)
        y_np = rng.standard_normal((N, 3)).astype(np_dtype)
        idx_np = np.zeros(N, dtype=np.int32)

        x = wp.array(x_np, dtype=wp_vec, device=device)
        y = wp.array(y_np, dtype=wp_vec, device=device)
        idx = wp.array(idx_np, dtype=wp.int32, device=device)
        out_xy = wp.zeros(M, dtype=wp_scalar, device=device)
        out_xx = wp.zeros(M, dtype=wp_scalar, device=device)
        out_yy = wp.zeros(M, dtype=wp_scalar, device=device)

        segmented_inner_products(x, y, idx, out_xy, out_xx, out_yy)
        wp.synchronize()

        ref_xy = (x_np * y_np).sum()
        ref_xx = (x_np * x_np).sum()
        ref_yy = (y_np * y_np).sum()
        np.testing.assert_allclose(out_xy.numpy(), [ref_xy], rtol=rtol)
        np.testing.assert_allclose(out_xx.numpy(), [ref_xx], rtol=rtol)
        np.testing.assert_allclose(out_yy.numpy(), [ref_yy], rtol=rtol)

    @pytest.mark.parametrize(
        "wp_dtype,np_dtype,rtol",
        [
            (wp.float32, np.float32, 1e-3),
            (wp.float64, np.float64, 1e-12),
        ],
    )
    def test_single_segment_remainder(self, device, wp_dtype, np_dtype, rtol):
        """M=1, N=8500 -- tile + remainder."""
        rng = np.random.default_rng(7)
        N, M = 8500, 1
        x_np = rng.standard_normal(N).astype(np_dtype)
        y_np = rng.standard_normal(N).astype(np_dtype)
        idx_np = np.zeros(N, dtype=np.int32)

        x = wp.array(x_np, dtype=wp_dtype, device=device)
        y = wp.array(y_np, dtype=wp_dtype, device=device)
        idx = wp.array(idx_np, dtype=wp.int32, device=device)
        out_xy = wp.zeros(M, dtype=wp_dtype, device=device)
        out_xx = wp.zeros(M, dtype=wp_dtype, device=device)
        out_yy = wp.zeros(M, dtype=wp_dtype, device=device)

        segmented_inner_products(x, y, idx, out_xy, out_xx, out_yy)
        wp.synchronize()

        np.testing.assert_allclose(out_xy.numpy(), [(x_np * y_np).sum()], rtol=rtol)
        np.testing.assert_allclose(out_xx.numpy(), [(x_np * x_np).sum()], rtol=rtol)
        np.testing.assert_allclose(out_yy.numpy(), [(y_np * y_np).sum()], rtol=rtol)

    def test_empty_input(self, device):
        x = wp.zeros(0, dtype=wp.float32, device=device)
        y = wp.zeros(0, dtype=wp.float32, device=device)
        idx = wp.zeros(0, dtype=wp.int32, device=device)
        out_xy = wp.zeros(3, dtype=wp.float32, device=device)
        out_xx = wp.zeros(3, dtype=wp.float32, device=device)
        out_yy = wp.zeros(3, dtype=wp.float32, device=device)

        segmented_inner_products(x, y, idx, out_xy, out_xx, out_yy)
        wp.synchronize()
        np.testing.assert_array_equal(out_xy.numpy(), np.zeros(3, dtype=np.float32))
        np.testing.assert_array_equal(out_xx.numpy(), np.zeros(3, dtype=np.float32))
        np.testing.assert_array_equal(out_yy.numpy(), np.zeros(3, dtype=np.float32))


# ---------------------------------------------------------------------------
# segmented_axpby tests
# ---------------------------------------------------------------------------


class TestSegmentedAxpby:
    """Tests for segmented_axpby: out = a[s]*x + b[s]*y."""

    @pytest.mark.parametrize(
        "wp_vec,wp_scalar,np_dtype,rtol",
        [
            (wp.vec3h, wp.float16, np.float16, 5e-2),
            (wp.vec3f, wp.float32, np.float32, 1e-5),
            (wp.vec3d, wp.float64, np.float64, 1e-12),
        ],
    )
    def test_vec(self, device, wp_vec, wp_scalar, np_dtype, rtol):
        rng = np.random.default_rng(3)
        N, M = 120, 4
        x_np = _randn(rng, (N, 3), np_dtype)
        y_np = _randn(rng, (N, 3), np_dtype)
        a_np = _randn(rng, M, np_dtype)
        b_np = _randn(rng, M, np_dtype)
        idx_np = np.sort(rng.integers(0, M, N)).astype(np.int32)

        x = wp.array(x_np, dtype=wp_vec, device=device)
        y = wp.array(y_np, dtype=wp_vec, device=device)
        a = wp.array(a_np, dtype=wp_scalar, device=device)
        b = wp.array(b_np, dtype=wp_scalar, device=device)
        idx = wp.array(idx_np, dtype=wp.int32, device=device)
        out = wp.zeros(N, dtype=wp_vec, device=device)

        segmented_axpby(out, a, x, b, y, idx)
        wp.synchronize()

        expected = a_np[idx_np, None] * x_np + b_np[idx_np, None] * y_np
        np.testing.assert_allclose(
            out.numpy(),
            expected,
            rtol=rtol,
            atol=_half_atol(np_dtype),
        )

    @pytest.mark.parametrize(
        "wp_dtype,np_dtype,rtol",
        [
            (wp.float32, np.float32, 1e-5),
            (wp.float64, np.float64, 1e-12),
        ],
    )
    def test_scalar(self, device, wp_dtype, np_dtype, rtol):
        rng = np.random.default_rng(8)
        N, M = 120, 4
        x_np = rng.standard_normal(N).astype(np_dtype)
        y_np = rng.standard_normal(N).astype(np_dtype)
        a_np = rng.standard_normal(M).astype(np_dtype)
        b_np = rng.standard_normal(M).astype(np_dtype)
        idx_np = np.sort(rng.integers(0, M, N)).astype(np.int32)

        x = wp.array(x_np, dtype=wp_dtype, device=device)
        y = wp.array(y_np, dtype=wp_dtype, device=device)
        a = wp.array(a_np, dtype=wp_dtype, device=device)
        b = wp.array(b_np, dtype=wp_dtype, device=device)
        idx = wp.array(idx_np, dtype=wp.int32, device=device)
        out = wp.zeros(N, dtype=wp_dtype, device=device)

        segmented_axpby(out, a, x, b, y, idx)
        wp.synchronize()

        expected = a_np[idx_np] * x_np + b_np[idx_np] * y_np
        np.testing.assert_allclose(out.numpy(), expected, rtol=rtol)

    def test_empty_input(self, device):
        x = wp.zeros(0, dtype=wp.vec3f, device=device)
        y = wp.zeros(0, dtype=wp.vec3f, device=device)
        a = wp.zeros(3, dtype=wp.float32, device=device)
        b = wp.zeros(3, dtype=wp.float32, device=device)
        idx = wp.zeros(0, dtype=wp.int32, device=device)
        out = wp.zeros(0, dtype=wp.vec3f, device=device)

        segmented_axpby(out, a, x, b, y, idx)
        wp.synchronize()
        np.testing.assert_array_equal(out.numpy().shape, (0, 3))


# ---------------------------------------------------------------------------
# segmented_max tests
# ---------------------------------------------------------------------------


class TestSegmentedMax:
    """Tests for segmented_max: scalar max per segment."""

    @pytest.mark.parametrize(
        "wp_dtype,np_dtype,rtol",
        [
            (wp.float32, np.float32, 1e-6),
            (wp.float64, np.float64, 1e-14),
        ],
    )
    def test_basic(self, device, wp_dtype, np_dtype, rtol):
        rng = np.random.default_rng(70)
        N, M = 200, 5
        x_np = rng.standard_normal(N).astype(np_dtype)
        idx_np = np.sort(rng.integers(0, M, N)).astype(np.int32)

        x = wp.array(x_np, dtype=wp_dtype, device=device)
        idx = wp.array(idx_np, device=device)

        # Initialize to -inf
        neg_inf = np.finfo(np_dtype).min
        out = wp.full(M, value=wp_dtype(neg_inf), dtype=wp_dtype, device=device)

        segmented_max(x, idx, out)
        wp.synchronize()

        expected = np.full(M, neg_inf, dtype=np_dtype)
        for s in range(M):
            mask = idx_np == s
            if mask.any():
                expected[s] = x_np[mask].max()

        np.testing.assert_allclose(out.numpy(), expected, rtol=rtol)

    def test_single_segment(self, device):
        """M=1 fast path."""
        rng = np.random.default_rng(72)
        N, M = 500, 1
        x_np = rng.standard_normal(N).astype(np.float32)
        idx_np = np.zeros(N, dtype=np.int32)

        x = wp.array(x_np, dtype=wp.float32, device=device)
        idx = wp.array(idx_np, device=device)
        neg_inf = np.finfo(np.float32).min
        out = wp.full(M, value=wp.float32(neg_inf), dtype=wp.float32, device=device)

        segmented_max(x, idx, out)
        wp.synchronize()
        np.testing.assert_allclose(out.numpy(), [x_np.max()], rtol=1e-6)

    def test_skewed_segments(self, device):
        rng = np.random.default_rng(73)
        lengths = [200, 1, 3, 100, 50]
        M = len(lengths)
        idx_np = np.concatenate(
            [np.full(length, s, dtype=np.int32) for s, length in enumerate(lengths)]
        )
        N = len(idx_np)
        x_np = rng.standard_normal(N).astype(np.float32)

        x = wp.array(x_np, dtype=wp.float32, device=device)
        idx = wp.array(idx_np, device=device)
        neg_inf = np.finfo(np.float32).min
        out = wp.full(M, value=wp.float32(neg_inf), dtype=wp.float32, device=device)

        segmented_max(x, idx, out)
        wp.synchronize()

        expected = np.full(M, neg_inf, dtype=np.float32)
        for s in range(M):
            mask = idx_np == s
            if mask.any():
                expected[s] = x_np[mask].max()
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6)

    def test_empty_input(self, device):
        M = 3
        idx = wp.zeros(0, dtype=wp.int32, device=device)
        x = wp.zeros(0, dtype=wp.float32, device=device)
        out = wp.zeros(M, dtype=wp.float32, device=device)

        segmented_max(x, idx, out)
        wp.synchronize()
        np.testing.assert_array_equal(out.numpy(), np.zeros(M, dtype=np.float32))


# ---------------------------------------------------------------------------
# segmented_min tests
# ---------------------------------------------------------------------------


class TestSegmentedMin:
    """Tests for segmented_min: scalar min per segment."""

    @pytest.mark.parametrize(
        "wp_dtype,np_dtype,rtol",
        [
            (wp.float32, np.float32, 1e-6),
            (wp.float64, np.float64, 1e-14),
        ],
    )
    def test_basic(self, device, wp_dtype, np_dtype, rtol):
        rng = np.random.default_rng(71)
        N, M = 200, 5
        x_np = rng.standard_normal(N).astype(np_dtype)
        idx_np = np.sort(rng.integers(0, M, N)).astype(np.int32)

        x = wp.array(x_np, dtype=wp_dtype, device=device)
        idx = wp.array(idx_np, device=device)

        # Initialize to +inf
        pos_inf = np.finfo(np_dtype).max
        out = wp.full(M, value=wp_dtype(pos_inf), dtype=wp_dtype, device=device)

        segmented_min(x, idx, out)
        wp.synchronize()

        expected = np.full(M, pos_inf, dtype=np_dtype)
        for s in range(M):
            mask = idx_np == s
            if mask.any():
                expected[s] = x_np[mask].min()

        np.testing.assert_allclose(out.numpy(), expected, rtol=rtol)

    def test_single_segment(self, device):
        """M=1 fast path."""
        rng = np.random.default_rng(74)
        N, M = 500, 1
        x_np = rng.standard_normal(N).astype(np.float32)
        idx_np = np.zeros(N, dtype=np.int32)

        x = wp.array(x_np, dtype=wp.float32, device=device)
        idx = wp.array(idx_np, device=device)
        pos_inf = np.finfo(np.float32).max
        out = wp.full(M, value=wp.float32(pos_inf), dtype=wp.float32, device=device)

        segmented_min(x, idx, out)
        wp.synchronize()
        np.testing.assert_allclose(out.numpy(), [x_np.min()], rtol=1e-6)

    def test_skewed_segments(self, device):
        rng = np.random.default_rng(75)
        lengths = [200, 1, 3, 100, 50]
        M = len(lengths)
        idx_np = np.concatenate(
            [np.full(length, s, dtype=np.int32) for s, length in enumerate(lengths)]
        )
        N = len(idx_np)
        x_np = rng.standard_normal(N).astype(np.float32)

        x = wp.array(x_np, dtype=wp.float32, device=device)
        idx = wp.array(idx_np, device=device)
        pos_inf = np.finfo(np.float32).max
        out = wp.full(M, value=wp.float32(pos_inf), dtype=wp.float32, device=device)

        segmented_min(x, idx, out)
        wp.synchronize()

        expected = np.full(M, pos_inf, dtype=np.float32)
        for s in range(M):
            mask = idx_np == s
            if mask.any():
                expected[s] = x_np[mask].min()
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6)

    def test_empty_input(self, device):
        M = 3
        idx = wp.zeros(0, dtype=wp.int32, device=device)
        x = wp.zeros(0, dtype=wp.float32, device=device)
        out = wp.zeros(M, dtype=wp.float32, device=device)

        segmented_min(x, idx, out)
        wp.synchronize()
        np.testing.assert_array_equal(out.numpy(), np.zeros(M, dtype=np.float32))


# ---------------------------------------------------------------------------
# segmented_broadcast tests
# ---------------------------------------------------------------------------


class TestSegmentedBroadcast:
    """Tests for segmented_broadcast: out[i] = values[idx[i]]."""

    @pytest.mark.parametrize(
        "wp_dtype,np_dtype",
        [
            (wp.float16, np.float16),
            (wp.float32, np.float32),
            (wp.float64, np.float64),
        ],
    )
    def test_scalar(self, device, wp_dtype, np_dtype):
        rng = np.random.default_rng(72)
        N, M = 100, 4
        values_np = rng.standard_normal(M).astype(np_dtype)
        idx_np = np.sort(rng.integers(0, M, N)).astype(np.int32)

        values = wp.array(values_np, dtype=wp_dtype, device=device)
        idx = wp.array(idx_np, device=device)
        out = wp.zeros(N, dtype=wp_dtype, device=device)

        segmented_broadcast(values, idx, out)
        wp.synchronize()

        expected = values_np[idx_np]
        np.testing.assert_allclose(out.numpy(), expected)

    @pytest.mark.parametrize(
        "wp_vec,np_dtype",
        [
            (wp.vec3h, np.float16),
            (wp.vec3f, np.float32),
            (wp.vec3d, np.float64),
        ],
    )
    def test_vector(self, device, wp_vec, np_dtype):
        rng = np.random.default_rng(73)
        N, M = 100, 4
        values_np = rng.standard_normal((M, 3)).astype(np_dtype)
        idx_np = np.sort(rng.integers(0, M, N)).astype(np.int32)

        values = _wp_vec_array(values_np, wp_vec, device)
        idx = wp.array(idx_np, device=device)
        out = wp.zeros(N, dtype=wp_vec, device=device)

        segmented_broadcast(values, idx, out)
        wp.synchronize()

        expected = values_np[idx_np]
        np.testing.assert_allclose(out.numpy(), expected)

    def test_empty_input(self, device):
        values = wp.zeros(3, dtype=wp.float32, device=device)
        idx = wp.zeros(0, dtype=wp.int32, device=device)
        out = wp.zeros(0, dtype=wp.float32, device=device)

        segmented_broadcast(values, idx, out)
        wp.synchronize()
        np.testing.assert_array_equal(out.numpy(), np.zeros(0, dtype=np.float32))


# ---------------------------------------------------------------------------
# segment_div tests
# ---------------------------------------------------------------------------


class TestSegmentDiv:
    """Tests for segment_div: element-wise divide with zero guard."""

    @pytest.mark.parametrize(
        "wp_dtype,np_dtype,atol",
        [
            (wp.float16, np.float16, 5e-2),
            (wp.float32, np.float32, 0),
            (wp.float64, np.float64, 0),
        ],
    )
    def test_basic(self, device, wp_dtype, np_dtype, atol):
        num_np = np.array([10.0, 20.0, 30.0, 40.0], dtype=np_dtype)
        denom_np = np.array([2, 5, 0, 4], dtype=np.int32)

        num = wp.array(num_np, dtype=wp_dtype, device=device)
        denom = wp.array(denom_np, device=device)
        result = wp.zeros(4, dtype=wp_dtype, device=device)

        segment_div(num, denom, result)
        wp.synchronize()

        expected = np.array([5.0, 4.0, 0.0, 10.0], dtype=np_dtype)
        np.testing.assert_allclose(result.numpy(), expected, atol=atol)

    def test_empty_input(self, device):
        num = wp.zeros(0, dtype=wp.float32, device=device)
        denom = wp.zeros(0, dtype=wp.int32, device=device)
        result = wp.zeros(0, dtype=wp.float32, device=device)

        segment_div(num, denom, result)
        wp.synchronize()
        np.testing.assert_array_equal(result.numpy(), np.zeros(0, dtype=np.float32))


# ---------------------------------------------------------------------------
# segmented_mean tests
# ---------------------------------------------------------------------------


class TestSegmentedMean:
    """Tests for segmented_mean: mean per segment."""

    @pytest.mark.parametrize(
        "wp_dtype,np_dtype,rtol",
        [
            (wp.float32, np.float32, 1e-5),
            (wp.float64, np.float64, 1e-12),
        ],
    )
    def test_scalar(self, device, wp_dtype, np_dtype, rtol):
        rng = np.random.default_rng(74)
        N, M = 200, 5
        x_np = rng.standard_normal(N).astype(np_dtype)
        idx_np = np.sort(rng.integers(0, M, N)).astype(np.int32)

        x = wp.array(x_np, dtype=wp_dtype, device=device)
        idx = wp.array(idx_np, device=device)
        sums = wp.zeros(M, dtype=wp_dtype, device=device)
        counts = wp.zeros(M, dtype=wp.int32, device=device)
        out = wp.zeros(M, dtype=wp_dtype, device=device)

        segmented_mean(x, idx, sums, counts, out)
        wp.synchronize()

        expected = np.zeros(M, dtype=np_dtype)
        for s in range(M):
            mask = idx_np == s
            if mask.any():
                expected[s] = x_np[mask].mean()

        np.testing.assert_allclose(out.numpy(), expected, rtol=rtol)

    @pytest.mark.parametrize(
        "wp_vec,wp_scalar,np_dtype,rtol",
        [
            (wp.vec3f, wp.float32, np.float32, 1e-5),
            (wp.vec3d, wp.float64, np.float64, 1e-12),
        ],
    )
    def test_vector(self, device, wp_vec, wp_scalar, np_dtype, rtol):
        rng = np.random.default_rng(75)
        N, M = 200, 5
        x_np = rng.standard_normal((N, 3)).astype(np_dtype)
        idx_np = np.sort(rng.integers(0, M, N)).astype(np.int32)

        x = _wp_vec_array(x_np, wp_vec, device)
        idx = wp.array(idx_np, device=device)
        sums = wp.zeros(M, dtype=wp_vec, device=device)
        counts = wp.zeros(M, dtype=wp.int32, device=device)
        out = wp.zeros(M, dtype=wp_vec, device=device)

        segmented_mean(x, idx, sums, counts, out)
        wp.synchronize()

        expected = np.zeros((M, 3), dtype=np_dtype)
        for s in range(M):
            mask = idx_np == s
            if mask.any():
                expected[s] = x_np[mask].mean(axis=0)

        np.testing.assert_allclose(out.numpy(), expected, rtol=rtol)

    def test_empty_input(self, device):
        M = 3
        x = wp.zeros(0, dtype=wp.float32, device=device)
        idx = wp.zeros(0, dtype=wp.int32, device=device)
        sums = wp.zeros(M, dtype=wp.float32, device=device)
        counts = wp.zeros(M, dtype=wp.int32, device=device)
        out = wp.zeros(M, dtype=wp.float32, device=device)

        segmented_mean(x, idx, sums, counts, out)
        wp.synchronize()
        np.testing.assert_array_equal(out.numpy(), np.zeros(M, dtype=np.float32))


# ---------------------------------------------------------------------------
# segmented_rms_norm tests
# ---------------------------------------------------------------------------


class TestSegmentedRmsNorm:
    """Tests for segmented_rms_norm: sqrt(mean(|v|^2)) per segment."""

    @pytest.mark.parametrize(
        "wp_vec,wp_scalar,np_dtype,rtol",
        [
            (wp.vec3f, wp.float32, np.float32, 1e-5),
            (wp.vec3d, wp.float64, np.float64, 1e-12),
        ],
    )
    def test_basic(self, device, wp_vec, wp_scalar, np_dtype, rtol):
        rng = np.random.default_rng(76)
        N, M = 200, 5
        x_np = rng.standard_normal((N, 3)).astype(np_dtype)
        idx_np = np.sort(rng.integers(0, M, N)).astype(np.int32)

        x = _wp_vec_array(x_np, wp_vec, device)
        idx = wp.array(idx_np, device=device)
        sum_sq = wp.zeros(M, dtype=wp_scalar, device=device)
        counts = wp.zeros(M, dtype=wp.int32, device=device)
        out = wp.zeros(M, dtype=wp_scalar, device=device)

        segmented_rms_norm(x, idx, sum_sq, counts, out)
        wp.synchronize()

        expected = np.zeros(M, dtype=np_dtype)
        for s in range(M):
            mask = idx_np == s
            if mask.any():
                norms_sq = (x_np[mask] ** 2).sum(axis=1)
                expected[s] = np.sqrt(norms_sq.mean())

        np.testing.assert_allclose(out.numpy(), expected, rtol=rtol)

    def test_empty_input(self, device):
        M = 3
        x = wp.zeros(0, dtype=wp.vec3f, device=device)
        idx = wp.zeros(0, dtype=wp.int32, device=device)
        sum_sq = wp.zeros(M, dtype=wp.float32, device=device)
        counts = wp.zeros(M, dtype=wp.int32, device=device)
        out = wp.zeros(M, dtype=wp.float32, device=device)

        segmented_rms_norm(x, idx, sum_sq, counts, out)
        wp.synchronize()
        np.testing.assert_array_equal(out.numpy(), np.zeros(M, dtype=np.float32))


# ---------------------------------------------------------------------------
# segmented_count tests
# ---------------------------------------------------------------------------


class TestSegmentedCount:
    """Tests for segmented_count: count elements per segment."""

    def test_basic(self, device):
        idx_np = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2], dtype=np.int32)
        M = 3

        idx = wp.array(idx_np, device=device)
        out = wp.zeros(M, dtype=wp.int32, device=device)

        segmented_count(idx, out)
        wp.synchronize()

        expected = np.array([3, 2, 4], dtype=np.int32)
        np.testing.assert_array_equal(out.numpy(), expected)

    def test_many_segments(self, device):
        rng = np.random.default_rng(77)
        N, M = 500, 20
        idx_np = np.sort(rng.integers(0, M, N)).astype(np.int32)

        idx = wp.array(idx_np, device=device)
        out = wp.zeros(M, dtype=wp.int32, device=device)

        segmented_count(idx, out)
        wp.synchronize()

        expected = np.bincount(idx_np, minlength=M).astype(np.int32)
        np.testing.assert_array_equal(out.numpy(), expected)

    def test_single_segment(self, device):
        """M=1 fast path."""
        N, M = 500, 1
        idx_np = np.zeros(N, dtype=np.int32)

        idx = wp.array(idx_np, device=device)
        out = wp.zeros(M, dtype=wp.int32, device=device)

        segmented_count(idx, out)
        wp.synchronize()
        np.testing.assert_array_equal(out.numpy(), [N])

    def test_empty_input(self, device):
        M = 3
        idx = wp.zeros(0, dtype=wp.int32, device=device)
        out = wp.zeros(M, dtype=wp.int32, device=device)

        segmented_count(idx, out)
        wp.synchronize()
        np.testing.assert_array_equal(out.numpy(), np.zeros(M, dtype=np.int32))


# ---------------------------------------------------------------------------
# compute_ept
# ---------------------------------------------------------------------------


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


class TestComputeEpt:
    """Tests for the public compute_ept helper."""

    # -- Small N (below wave-fill) returns ept_min --------------------------

    def test_small_n_vec3_returns_ept_min(self):
        """N much smaller than sm_count * 512 should return ept_min=2 for vec3."""
        assert compute_ept(N=10, sm_count=80, is_vec3=True) == 2

    def test_small_n_scalar_returns_ept_min(self):
        """N much smaller than sm_count * 512 should return ept_min=4 for scalar."""
        assert compute_ept(N=10, sm_count=80, is_vec3=False) == 4

    # -- Large N returns clamped value within [ept_min, ept_max] ------------

    def test_large_n_vec3_capped_at_8(self):
        """Very large N with vec3 should cap at ept_max=8."""
        result = compute_ept(N=10_000_000, sm_count=1, is_vec3=True)
        assert result == 8

    def test_large_n_scalar_capped_at_16(self):
        """Very large N with scalar should cap at ept_max=16."""
        result = compute_ept(N=10_000_000, sm_count=1, is_vec3=False)
        assert result == 16

    # -- Boundary values ----------------------------------------------------

    def test_n_zero(self):
        """N=0 should still return ept_min (no crash)."""
        assert compute_ept(N=0, sm_count=80, is_vec3=True) == 2
        assert compute_ept(N=0, sm_count=80, is_vec3=False) == 4

    def test_n_one(self):
        """N=1 should return ept_min."""
        assert compute_ept(N=1, sm_count=80, is_vec3=True) == 2
        assert compute_ept(N=1, sm_count=80, is_vec3=False) == 4

    def test_sm_count_one(self):
        """sm_count=1 means w_fill=512; moderate N should still work."""
        result = compute_ept(N=1024, sm_count=1, is_vec3=False)
        assert _is_power_of_two(result)
        assert 4 <= result <= 16

    def test_large_sm_count(self):
        """Very large sm_count keeps result within bounds."""
        result = compute_ept(N=1000, sm_count=10000, is_vec3=True)
        assert result == 2  # ept_min for vec3

    # -- Result is always a power of two within bounds ----------------------

    @pytest.mark.parametrize("sm_count", [1, 40, 80, 132])
    @pytest.mark.parametrize("is_vec3", [True, False])
    def test_result_is_power_of_two(self, sm_count, is_vec3):
        ept_min = 2 if is_vec3 else 4
        ept_max = 8 if is_vec3 else 16
        for N in [0, 1, 100, 512, 1024, 10_000, 100_000, 1_000_000, 50_000_000]:
            result = compute_ept(N, sm_count, is_vec3)
            assert _is_power_of_two(result), (
                f"compute_ept({N}, {sm_count}, {is_vec3}) = {result} is not a power of 2"
            )
            assert ept_min <= result <= ept_max, (
                f"compute_ept({N}, {sm_count}, {is_vec3}) = {result} "
                f"outside [{ept_min}, {ept_max}]"
            )

    # -- Power-of-2 rounding (round down on tie) ----------------------------

    def test_rounding_exact_power_of_two(self):
        """When raw ept is exactly a power of two, result equals that value."""
        # sm_count=1 => w_fill=512; N=2048 => ept_raw=4 (exact power of 2)
        assert compute_ept(N=2048, sm_count=1, is_vec3=False) == 4

    def test_rounding_up_on_tie(self):
        """When ept is equidistant between two powers, round up (keep higher)."""
        # sm_count=1 => w_fill=512; N=512*3=1536 => ept_raw=3
        # Nearest powers: 2 and 4. (4-3)==1 is NOT > (3-2)==1, so p stays 4.
        result = compute_ept(N=1536, sm_count=1, is_vec3=True)
        assert result == 4

    def test_rounding_up_when_closer(self):
        """When ept is closer to the higher power of two, round up."""
        # sm_count=1 => w_fill=512; N=512*7=3584 => ept_raw=7
        # Nearest powers: 4 and 8. Distance to 8 = 1, distance to 4 = 3 => round up to 8
        result = compute_ept(N=3584, sm_count=1, is_vec3=True)
        assert result == 8
