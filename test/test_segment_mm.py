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

"""Correctness and input-contract tests for Warp segmented matrix multiplication."""

import numpy as np
import pytest
import warp as wp

import nvalchemiops.segment_mm as segmm


@pytest.mark.gpu
class TestSegmentMMBackwardB:
    """Check the weight-only primitive used by higher-order framework rules."""

    @pytest.mark.parametrize("dtype", [wp.float32, wp.bfloat16])
    @pytest.mark.parametrize("strided", [False, True])
    def test_weight_gradient(self, device, dtype, strided):
        """Reduce native operands over segments, excluding padded NaNs and empty groups."""
        rng = np.random.default_rng(61)
        values = np.array([0, 1, 1025, 33], dtype=np.int32)
        a_np = rng.standard_normal((2053, 35)).astype(np.float32)
        g_np = rng.standard_normal((2053, 67)).astype(np.float32)
        a_np[values.sum() :] = g_np[values.sum() :] = np.nan
        a, g = (_device_array(x, dtype, device) for x in (a_np, g_np))
        a_ref, g_ref = _numpy(a), _numpy(g)
        lengths = wp.array(values, device=device)
        if strided:
            a = _device_array(a_ref.T.copy(), dtype, device).transpose()
            storage = np.full((4106, 134), np.nan, dtype=np.float32)
            storage[::2, ::2] = g_ref
            g = _device_array(storage, dtype, device)[::2, ::2]
            lengths = wp.array(np.repeat(values, 2), device=device)[::2]
        grad_b = wp.empty((4, 35, 67), dtype=dtype, device=device)
        segmm.segment_mm_backward_b(a, lengths, g, grad_b)
        expected = np.zeros(grad_b.shape, dtype=np.float64)
        start = 0
        for group, length in enumerate(values):
            stop = start + int(length)
            expected[group] = a_ref[start:stop].astype(np.float64).T @ g_ref[
                start:stop
            ].astype(np.float64)
            start = stop
        tolerance = 2e-2 if dtype == wp.bfloat16 else 1e-4
        np.testing.assert_allclose(
            _numpy(grad_b),
            _round_output(expected, dtype),
            atol=tolerance,
            rtol=tolerance,
        )
        np.testing.assert_array_equal(_numpy(grad_b)[0], 0)
        lengths.zero_()
        segmm.segment_mm_backward_b(a, lengths, g, grad_b)
        np.testing.assert_array_equal(_numpy(grad_b), 0)

    @pytest.mark.parametrize("dtype", [wp.float32, wp.bfloat16])
    def test_single_split_rounds_once(self, device, dtype):
        """Keep FP32 cancellation terms and overwrite empty groups with NaN padding."""
        a_np = np.ones((519, 1), dtype=np.float32)
        g_np = np.zeros_like(a_np)
        g_np[0] = 1
        g_np[256] = -1
        g_np[1:32] = g_np[257:288] = g_np[512] = 2**-13
        a_np[513:] = g_np[513:] = np.nan
        a, g = (_device_array(x, dtype, device) for x in (a_np, g_np))
        lengths = wp.array([0, 513, 0], dtype=wp.int32, device=device)
        grad_b = _device_array(np.full((3, 1, 1), np.nan), dtype, device)

        segmm.segment_mm_backward_b(a, lengths, g, grad_b)

        np.testing.assert_array_equal(_numpy(grad_b), [[[0]], [[63 * 2**-13]], [[0]]])
        lengths.zero_()
        segmm.segment_mm_backward_b(a, lengths, g, grad_b)
        np.testing.assert_array_equal(_numpy(grad_b), 0)

    @pytest.mark.parametrize("dtype", [wp.float32, wp.bfloat16])
    @pytest.mark.parametrize("capacity,groups", [(0, 3), (11, 0), (0, 0)])
    def test_empty(self, device, dtype, capacity, groups):
        """Empty buffers overwrite every weight gradient with zero."""
        a = wp.empty((capacity, 4), dtype=dtype, device=device)
        g = wp.empty((capacity, 3), dtype=dtype, device=device)
        lengths = wp.zeros(groups, dtype=wp.int32, device=device)
        grad_b = _device_array(np.full((groups, 4, 3), np.nan), dtype, device)
        segmm.segment_mm_backward_b(a, lengths, g, grad_b)
        np.testing.assert_array_equal(_numpy(grad_b), 0)

    @pytest.mark.parametrize("invalid", ["rows", "dtype", "lengths", "alias"])
    def test_invalid_buffers(self, device, invalid):
        """Reject bad metadata and overlapping output before writing any result."""
        a = wp.ones((8, 4), device=device)
        g = wp.ones((8, 3), device=device)
        lengths = wp.array([3, 2], dtype=wp.int32, device=device)
        grad_b = wp.empty((2, 4, 3), device=device)
        message = "shape|dtype|overlap"
        if invalid == "rows":
            g = wp.ones((7, 3), device=device)
        elif invalid == "dtype":
            g = wp.ones((8, 3), dtype=wp.bfloat16, device=device)
        elif invalid == "lengths":
            lengths = wp.array([3, 2], dtype=wp.int64, device=device)
        else:
            grad_b = wp.array(
                ptr=a.ptr, shape=(2, 4, 3), dtype=wp.float32, device=device, copy=False
            )
        with pytest.raises(ValueError, match=message):
            segmm.segment_mm_backward_b(a, lengths, g, grad_b)


@pytest.fixture(scope="module")
def device():
    """Use a CUDA device for the tile-based implementation."""
    if not wp.is_cuda_available():
        pytest.skip("CUDA is required for segment_mm")
    return wp.get_device("cuda:0")


def _device_array(values, dtype, device):
    """Transfer values, converting BF16 on the device without a framework dependency."""
    array = wp.array(np.asarray(values, dtype=np.float32), device=device)
    if dtype == wp.float32:
        return array
    converted = wp.empty(array.shape, dtype=dtype, device=device)
    wp.utils.array_cast(array.flatten(), converted.flatten())
    return converted


def _numpy(array):
    """Read a Warp floating-point array as NumPy FP32."""
    if array.dtype == wp.float32:
        return array.numpy()
    converted = wp.empty(array.shape, dtype=wp.float32, device=array.device)
    wp.utils.array_cast(array.flatten(), converted.flatten())
    return converted.numpy()


def _reference(a, b, lengths, dtype):
    """Compute independent FP64 matrix products and round once to the output dtype."""
    result = np.zeros((a.shape[0], b.shape[2]), dtype=np.float32)
    start = 0
    for group, length in enumerate(lengths):
        stop = start + int(length)
        result[start:stop] = a[start:stop].astype(np.float64) @ b[group].astype(
            np.float64
        )
        start = stop
    return _round_output(result, dtype)


def _round_output(result, dtype):
    """Round an independent reference to the public gradient/output dtype."""
    result = np.asarray(result, dtype=np.float32)
    if dtype == wp.bfloat16:
        bits = result.view(np.uint32)
        rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(
            0xFFFF0000
        )
        return rounded.view(np.float32)
    return result


def _backward_reference(a, b, lengths, grad_out, dtype):
    """Evaluate both analytical gradients in NumPy FP64, excluding padded rows."""
    grad_a = np.zeros(a.shape, dtype=np.float64)
    grad_b = np.zeros(b.shape, dtype=np.float64)
    start = 0
    for group, length in enumerate(lengths):
        stop = start + int(length)
        grad_a[start:stop] = (
            grad_out[start:stop].astype(np.float64) @ b[group].astype(np.float64).T
        )
        grad_b[group] = a[start:stop].astype(np.float64).T @ grad_out[
            start:stop
        ].astype(np.float64)
        start = stop
    return _round_output(grad_a, dtype), _round_output(grad_b, dtype)


@wp.kernel
def _write_lengths(lengths: wp.array(dtype=wp.int32), state: int):
    """Produce changing segment metadata directly on the current CUDA stream."""
    group = wp.tid()
    lengths[group] = 0
    if group == 0:
        lengths[group] = state
    elif group == 2:
        lengths[group] = 5 - state


@pytest.mark.gpu
class TestSegmentMM:
    """Exercise segmented products, boundary handling, and reusable output buffers."""

    @pytest.mark.parametrize("dtype", [wp.float32, wp.bfloat16])
    @pytest.mark.parametrize("k,n", [(1, 1), (35, 67), (64, 128), (129, 65)])
    def test_mixed_lengths_and_padding(self, device, dtype, k, n):
        """Prevent cross-segment stores, wrong empty-group lookup, and padding leaks."""
        lengths = np.array([0, 1, 31, 0, 32, 33, 0], dtype=np.int32)
        capacity = int(lengths.sum()) + 19
        rng = np.random.default_rng(42)
        a_np = rng.standard_normal((capacity, k)).astype(np.float32)
        b_np = rng.standard_normal((len(lengths), k, n)).astype(np.float32)
        a_np[lengths.sum() :] = np.nan
        b_np[lengths == 0] = np.nan
        a = _device_array(a_np, dtype, device)
        b = _device_array(b_np, dtype, device)
        out = _device_array(np.full((capacity, n), np.nan), dtype, device)
        seglen = wp.array(lengths, device=device)

        assert segmm.segment_mm(a, b, seglen, out) is None

        expected = _reference(_numpy(a), _numpy(b), lengths, dtype)
        actual = _numpy(out)
        tolerance = 2e-2 if dtype == wp.bfloat16 else 3e-5
        np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
        np.testing.assert_array_equal(actual[lengths.sum() :], 0)

    @pytest.mark.parametrize("dtype", [wp.float32, wp.bfloat16])
    @pytest.mark.parametrize(
        "layout",
        [
            "a_transpose",
            "b_transpose",
            "a_slice",
            "b_slice",
            "b_broadcast",
            "seglen_slice",
            "combined",
        ],
    )
    def test_strided_inputs(self, device, dtype, layout):
        """Read input strides without crossing segments or including skipped values."""
        lengths = np.array([0, 1, 33, 0, 5], dtype=np.int32)
        capacity, k, n = 45, 35, 67
        rng = np.random.default_rng(123)
        a_np = rng.integers(-8, 9, (capacity, k)).astype(np.float32) / 8
        b_np = rng.integers(-8, 9, (len(lengths), k, n)).astype(np.float32) / 8
        a_np[lengths.sum() :] = np.nan
        b_np[lengths == 0] = np.nan
        a = _device_array(a_np, dtype, device)
        b = _device_array(b_np, dtype, device)
        seglen = wp.array(lengths, device=device)
        if layout == "a_transpose":
            a = _device_array(a_np.T.copy(), dtype, device).transpose()
        if layout == "b_transpose":
            b = _device_array(b_np.transpose(0, 2, 1).copy(), dtype, device).transpose(
                (0, 2, 1)
            )
        if layout in ("a_slice", "combined"):
            storage = np.full((2 * capacity, 2 * k + 1), np.nan, dtype=np.float32)
            storage[1::2, 1::2] = a_np
            a = _device_array(storage, dtype, device)[1::2, 1::2]
        if layout in ("b_slice", "combined"):
            storage = np.full(
                (2 * len(lengths), 2 * k, 2 * n + 1), np.nan, dtype=np.float32
            )
            storage[1::2, ::2, 1::2] = b_np
            b = _device_array(storage, dtype, device)[1::2, ::2, 1::2]
        if layout == "b_broadcast":
            b_np = np.broadcast_to(b_np[1:2], b_np.shape)
            b_storage = _device_array(b_np[0:1], dtype, device)
            b = wp.array(
                ptr=b_storage.ptr,
                shape=b_np.shape,
                strides=(0, *b_storage.strides[1:]),
                dtype=dtype,
                device=device,
                copy=False,
            )
        if layout in ("seglen_slice", "combined"):
            storage = np.full(2 * len(lengths) + 1, -100, dtype=np.int32)
            storage[1::2] = lengths
            seglen = wp.array(storage, device=device)[1::2]
        assert not all(array.is_contiguous for array in (a, b, seglen))
        out = _device_array(np.full((capacity, n), np.nan), dtype, device)

        segmm.segment_mm(a, b, seglen, out)

        np.testing.assert_array_equal(
            _numpy(out), _reference(a_np, b_np, lengths, dtype)
        )

    @pytest.mark.parametrize("dtype", [wp.float32, wp.bfloat16])
    @pytest.mark.parametrize("placement", ["before", "after"])
    def test_disjoint_views_of_same_storage(self, device, dtype, placement):
        """Allow adjacent output storage without falsely rejecting a shared allocation."""
        storage = _device_array(np.arange(320) / 64, dtype, device)
        a = storage[32:288].reshape((64, 4))[::8]
        a_np = _numpy(storage)[32:288].reshape(64, 4)[::8].copy()
        b_np = np.ones((2, 4, 4), dtype=np.float32)
        b = _device_array(b_np, dtype, device)
        lengths = wp.array(np.array([3, 5], dtype=np.int32), device=device)
        out = (storage[:32] if placement == "before" else storage[260:292]).reshape(
            (8, 4)
        )

        segmm.segment_mm(a, b, lengths, out)

        np.testing.assert_array_equal(
            _numpy(out), _reference(a_np, b_np, [3, 5], dtype)
        )

    @pytest.mark.parametrize("dtype", [wp.float32, wp.bfloat16])
    def test_lengths_change_and_output_is_reused(self, device, dtype):
        """Prevent stale output when active rows shrink or the active group changes."""
        a = _device_array(np.arange(16 * 7).reshape(16, 7) / 64, dtype, device)
        b = _device_array(np.arange(4 * 7 * 9).reshape(4, 7, 9) / 128, dtype, device)
        out = wp.empty((16, 9), dtype=dtype, device=device)
        lengths = wp.empty(4, dtype=wp.int32, device=device)
        for values in ([16, 0, 0, 0], [0, 3, 0, 5], [0, 0, 0, 0], [0, 0, 16, 0]):
            wp.copy(lengths, wp.array(np.array(values, dtype=np.int32), device=device))
            segmm.segment_mm(a, b, lengths, out)
            expected = _reference(_numpy(a), _numpy(b), values, dtype)
            np.testing.assert_array_equal(_numpy(out), expected)

    @pytest.mark.parametrize("dtype", [wp.float32, wp.bfloat16])
    @pytest.mark.parametrize("groups", [128, 129])
    def test_group_boundary_replays_changing_lengths(self, device, dtype, groups):
        """Keep prefix paths correct across graph replays, strides, and NaN padding."""
        base = np.resize(np.array([0, 1, 31, 32, 33], dtype=np.int32), groups)
        capacity = int(base.sum()) + 13
        rng = np.random.default_rng(19)
        a_np = rng.integers(-8, 9, (capacity, 3)).astype(np.float32) / 8
        b_np = rng.integers(-8, 9, (groups, 3, 5)).astype(np.float32) / 8
        dy_np = rng.integers(-8, 9, (capacity, 5)).astype(np.float32) / 8
        length_storage = np.full(2 * groups, -100, dtype=np.int32)
        length_storage[::2] = 0
        with wp.ScopedStream(wp.Stream(device)):
            a, b, dy = (_device_array(x, dtype, device) for x in (a_np, b_np, dy_np))
            lengths = wp.array(length_storage, device=device)[::2]
            out, da, db = (
                _device_array(np.full(shape, np.nan), dtype, device)
                for shape in (dy.shape, a.shape, b.shape)
            )
            segmm.segment_mm(a, b, lengths, out)
            segmm.segment_mm_backward(a, b, lengths, dy, da, db)
            with wp.ScopedCapture(device=device) as capture:
                segmm.segment_mm(a, b, lengths, out)
                segmm.segment_mm_backward(a, b, lengths, dy, da, db)
            for values in (
                base,
                np.roll(base, 7) // 2,
                np.zeros_like(base),
                base[::-1].copy(),
            ):
                current_a, current_b, current_dy = (
                    x.copy() for x in (a_np, b_np, dy_np)
                )
                current_a[values.sum() :] = current_dy[values.sum() :] = np.nan
                current_b[values == 0] = np.nan
                for target, value in (
                    (a, current_a),
                    (b, current_b),
                    (dy, current_dy),
                ):
                    wp.copy(target, _device_array(value, dtype, device))
                wp.copy(lengths, wp.array(values, device=device))
                wp.capture_launch(capture.graph)

                expected = _reference(current_a, current_b, values, dtype)
                expected_a, expected_b = _backward_reference(
                    current_a, current_b, values, current_dy, dtype
                )
                np.testing.assert_array_equal(_numpy(out), expected)
                np.testing.assert_array_equal(_numpy(da), expected_a)
                np.testing.assert_array_equal(_numpy(db), expected_b)

    @pytest.mark.parametrize("dtype", [wp.float32, wp.bfloat16])
    @pytest.mark.parametrize("capacity,groups", [(0, 3), (11, 0), (0, 0)])
    def test_empty_buffers(self, device, dtype, capacity, groups):
        """Empty inputs must not launch invalid grids or retain stale output."""
        a = wp.empty((capacity, 8), dtype=dtype, device=device)
        b = wp.empty((groups, 8, 5), dtype=dtype, device=device)
        lengths = wp.zeros(groups, dtype=wp.int32, device=device)
        out = _device_array(np.full((capacity, 5), np.nan), dtype, device)
        segmm.segment_mm(a, b, lengths, out)
        np.testing.assert_array_equal(_numpy(out), np.zeros((capacity, 5)))

    def test_bfloat16_accumulates_in_float32(self, device):
        """Small contributions must survive accumulation across many K tiles."""
        a = _device_array(np.ones((1, 1025)), wp.bfloat16, device)
        weights = np.full((1, 1025, 1), 2**-13, dtype=np.float32)
        weights[0, 0, 0] = 1.0
        b = _device_array(weights, wp.bfloat16, device)
        lengths = wp.array(np.array([1], dtype=np.int32), device=device)
        out = wp.empty((1, 1), dtype=wp.bfloat16, device=device)
        segmm.segment_mm(a, b, lengths, out)
        np.testing.assert_array_equal(_numpy(out), [[1.125]])

    def test_float32_preserves_input_precision(self, device):
        """The FP32 path must not silently truncate its operands to TF32."""
        value = np.float32(1 + 2**-12)
        a = _device_array([[value]], wp.float32, device)
        b = _device_array([[[value]]], wp.float32, device)
        lengths = wp.array(np.array([1], dtype=np.int32), device=device)
        out = wp.empty((1, 1), dtype=wp.float32, device=device)
        segmm.segment_mm(a, b, lengths, out)
        np.testing.assert_array_equal(_numpy(out), [[np.float32(value * value)]])

    @pytest.mark.parametrize("strided", [False, True])
    def test_cuda_graph_replays_device_lengths(self, device, strided):
        """Captured launches must read new device lengths on a non-default stream."""
        with wp.ScopedStream(wp.Stream(device)):
            a = _device_array(np.arange(7 * 4).reshape(7, 4) / 16, wp.float32, device)
            b = _device_array(
                np.arange(3 * 4 * 5).reshape(3, 4, 5) / 32, wp.float32, device
            )
            lengths = wp.zeros(3, dtype=wp.int32, device=device)
            if strided:
                a = _device_array(_numpy(a).T.copy(), wp.float32, device).transpose()
                b = _device_array(
                    _numpy(b).transpose(0, 2, 1).copy(), wp.float32, device
                ).transpose((0, 2, 1))
                lengths = wp.zeros(6, dtype=wp.int32, device=device)[::2]
            out = wp.empty((7, 5), dtype=wp.float32, device=device)
            segmm.segment_mm(a, b, lengths, out)
            wp.launch(_write_lengths, dim=3, inputs=[lengths, 0], device=device)
            with wp.ScopedCapture(device=device) as capture:
                segmm.segment_mm(a, b, lengths, out)
            for state in (0, 2, 5):
                wp.launch(_write_lengths, dim=3, inputs=[lengths, state], device=device)
                wp.capture_launch(capture.graph)
                expected = _reference(
                    _numpy(a), _numpy(b), [state, 0, 5 - state], wp.float32
                )
                np.testing.assert_array_equal(_numpy(out), expected)


@pytest.mark.gpu
class TestSegmentMMBackward:
    """Verify explicit operand gradients without depending on a framework's AD."""

    @pytest.mark.parametrize("dtype", [wp.float32, wp.bfloat16])
    @pytest.mark.parametrize("k,n", [(1, 1), (35, 67), (64, 128), (129, 65)])
    def test_gradients(self, device, dtype, k, n):
        """Both gradients must respect empty segments, tile boundaries, and NaN padding."""
        values = np.array([0, 1, 31, 0, 32, 33, 0], dtype=np.int32)
        capacity = int(values.sum()) + 19
        rng = np.random.default_rng(55)
        a_np = rng.standard_normal((capacity, k)).astype(np.float32)
        b_np = rng.standard_normal((len(values), k, n)).astype(np.float32)
        dy_np = rng.standard_normal((capacity, n)).astype(np.float32)
        a_np[values.sum() :] = np.nan
        dy_np[values.sum() :] = np.nan
        b_np[values == 0] = np.nan
        a, b, dy = (_device_array(x, dtype, device) for x in (a_np, b_np, dy_np))
        lengths = wp.array(values, device=device)
        da = _device_array(np.full(a.shape, np.nan), dtype, device)
        db = _device_array(np.full(b.shape, np.nan), dtype, device)

        assert segmm.segment_mm_backward(a, b, lengths, dy, da, db) is None

        expected_a, expected_b = _backward_reference(
            _numpy(a), _numpy(b), values, _numpy(dy), dtype
        )
        tolerance = 2e-2 if dtype == wp.bfloat16 else 3e-5
        np.testing.assert_allclose(
            _numpy(da), expected_a, rtol=tolerance, atol=tolerance
        )
        np.testing.assert_allclose(
            _numpy(db), expected_b, rtol=tolerance, atol=tolerance
        )
        np.testing.assert_array_equal(_numpy(da)[values.sum() :], 0)
        np.testing.assert_array_equal(_numpy(db)[values == 0], 0)

    @pytest.mark.parametrize("dtype", [wp.float32, wp.bfloat16])
    @pytest.mark.parametrize(
        "a_layout", ["transposed", "sliced", "row_broadcast", "column_broadcast"]
    )
    def test_strided_inputs(self, device, dtype, a_layout):
        """Accept strided primal arrays, cotangents, and device lengths together."""
        values = np.array([1, 0, 33], dtype=np.int32)
        rng = np.random.default_rng(7)
        a_np = rng.integers(-8, 9, (39, 35)).astype(np.float32) / 8
        b_np = rng.integers(-8, 9, (3, 35, 67)).astype(np.float32) / 8
        dy_np = rng.integers(-8, 9, (39, 67)).astype(np.float32) / 8
        a_np[34:] = dy_np[34:] = np.nan
        if a_layout == "transposed":
            a = _device_array(a_np.T.copy(), dtype, device).transpose()
        elif a_layout == "sliced":
            a_storage = np.full((78, 71), np.nan, dtype=np.float32)
            a_storage[1::2, 1::2] = a_np
            a = _device_array(a_storage, dtype, device)[1::2, 1::2]
        elif a_layout == "row_broadcast":
            a_np[:34] = a_np[:1]
            a_storage = _device_array(a_np[:1].copy(), dtype, device)
            a = wp.array(
                ptr=a_storage.ptr,
                shape=a_np.shape,
                strides=(0, a_storage.strides[1]),
                dtype=dtype,
                device=device,
                copy=False,
            )
        else:
            a_np[:] = a_np[:, :1]
            a_storage = _device_array(a_np[:, :1].copy(), dtype, device)
            a = wp.array(
                ptr=a_storage.ptr,
                shape=a_np.shape,
                strides=(a_storage.strides[0], 0),
                dtype=dtype,
                device=device,
                copy=False,
            )
        b = _device_array(b_np.transpose(0, 2, 1).copy(), dtype, device).transpose(
            (0, 2, 1)
        )
        storage = np.full((78, 135), np.nan, dtype=np.float32)
        storage[1::2, 1::2] = dy_np
        dy = _device_array(storage, dtype, device)[1::2, 1::2]
        lengths = wp.array(
            np.array([1, -1, 0, -1, 33, -1], dtype=np.int32), device=device
        )[::2]
        da = wp.empty(a.shape, dtype=dtype, device=device)
        db = wp.empty(b.shape, dtype=dtype, device=device)

        segmm.segment_mm_backward(a, b, lengths, dy, da, db)

        expected_a, expected_b = _backward_reference(a_np, b_np, values, dy_np, dtype)
        np.testing.assert_array_equal(_numpy(da), expected_a)
        np.testing.assert_array_equal(_numpy(db), expected_b)

    @pytest.mark.parametrize("dtype", [wp.float32, wp.bfloat16])
    @pytest.mark.parametrize("capacity,groups", [(0, 3), (11, 0), (0, 0)])
    def test_empty_buffers(self, device, dtype, capacity, groups):
        """Zero every gradient even when no data kernel can be launched."""
        a = wp.empty((capacity, 8), dtype=dtype, device=device)
        b = wp.empty((groups, 8, 5), dtype=dtype, device=device)
        lengths = wp.zeros(groups, dtype=wp.int32, device=device)
        dy = wp.empty((capacity, 5), dtype=dtype, device=device)
        da = _device_array(np.full(a.shape, np.nan), dtype, device)
        db = _device_array(np.full(b.shape, np.nan), dtype, device)

        segmm.segment_mm_backward(a, b, lengths, dy, da, db)

        np.testing.assert_array_equal(_numpy(da), np.zeros(a.shape))
        np.testing.assert_array_equal(_numpy(db), np.zeros(b.shape))

    @pytest.mark.parametrize("dtype", [wp.float32, wp.bfloat16])
    def test_reused_gradients(self, device, dtype):
        """Changing lengths must overwrite both gradients, including newly empty groups."""
        a = _device_array(np.arange(16 * 7).reshape(16, 7) / 64, dtype, device)
        b = _device_array(np.arange(4 * 7 * 9).reshape(4, 7, 9) / 128, dtype, device)
        dy = _device_array(np.ones((16, 9)), dtype, device)
        lengths = wp.empty(4, dtype=wp.int32, device=device)
        da = wp.empty(a.shape, dtype=dtype, device=device)
        db = wp.empty(b.shape, dtype=dtype, device=device)
        for values in ([16, 0, 0, 0], [0, 3, 0, 5], [0, 0, 0, 0], [0, 0, 16, 0]):
            wp.copy(lengths, wp.array(np.array(values, dtype=np.int32), device=device))
            segmm.segment_mm_backward(a, b, lengths, dy, da, db)
            expected_a, expected_b = _backward_reference(
                _numpy(a), _numpy(b), values, _numpy(dy), dtype
            )
            np.testing.assert_array_equal(_numpy(da), expected_a)
            np.testing.assert_array_equal(_numpy(db), expected_b)

    @pytest.mark.parametrize("dtype", [wp.float32, wp.bfloat16])
    def test_split_reduction_mixed_lengths(self, device, dtype):
        """Reduce long and short systems together, including empty split ranges."""
        values = np.array([0, 1, 2305, 33], dtype=np.int32)
        rng = np.random.default_rng(19)
        a_np = rng.integers(-8, 9, (4096, 35)).astype(np.float32) / 8
        b_np = rng.integers(-8, 9, (4, 35, 67)).astype(np.float32) / 8
        dy_np = rng.integers(-8, 9, (4096, 67)).astype(np.float32) / 8
        a_np[values.sum() :] = dy_np[values.sum() :] = np.nan
        b_np[0] = np.nan
        a, b, dy = (_device_array(x, dtype, device) for x in (a_np, b_np, dy_np))
        lengths = wp.array(values, device=device)
        da = _device_array(np.full(a.shape, np.nan), dtype, device)
        db = _device_array(np.full(b.shape, np.nan), dtype, device)

        segmm.segment_mm_backward(a, b, lengths, dy, da, db)

        expected_a, expected_b = _backward_reference(a_np, b_np, values, dy_np, dtype)
        np.testing.assert_array_equal(_numpy(da), expected_a)
        np.testing.assert_array_equal(_numpy(db), expected_b)

    def test_bfloat16_reduction_rounds_once(self, device):
        """Keep small contributions through FP32 split reduction before BF16 rounding."""
        a = _device_array(np.ones((512, 1)), wp.bfloat16, device)
        b = _device_array(np.ones((1, 1, 1)), wp.bfloat16, device)
        dy_np = np.zeros((512, 1), dtype=np.float32)
        dy_np[0] = 1
        dy_np[256] = -1
        dy_np[1:32] = dy_np[257:288] = 2**-13
        dy = _device_array(dy_np, wp.bfloat16, device)
        lengths = wp.array(np.array([512], dtype=np.int32), device=device)
        da = wp.empty(a.shape, dtype=wp.bfloat16, device=device)
        db = wp.empty(b.shape, dtype=wp.bfloat16, device=device)

        segmm.segment_mm_backward(a, b, lengths, dy, da, db)

        np.testing.assert_array_equal(_numpy(da), dy_np)
        np.testing.assert_array_equal(_numpy(db), [[[62 * 2**-13]]])

    def test_bfloat16_grad_a_accumulates_in_float32(self, device):
        """The channel reduction must not round small products away in BF16."""
        a = _device_array([[1]], wp.bfloat16, device)
        b_np = np.full((1, 1, 1025), 2**-13, dtype=np.float32)
        b_np[0, 0, 0] = 1
        b = _device_array(b_np, wp.bfloat16, device)
        dy = _device_array(np.ones((1, 1025)), wp.bfloat16, device)
        lengths = wp.array(np.array([1], dtype=np.int32), device=device)
        da = wp.empty(a.shape, dtype=wp.bfloat16, device=device)
        db = wp.empty(b.shape, dtype=wp.bfloat16, device=device)

        segmm.segment_mm_backward(a, b, lengths, dy, da, db)

        np.testing.assert_array_equal(_numpy(da), [[1.125]])
        np.testing.assert_array_equal(_numpy(db), np.ones(b.shape))

    def test_float32_preserves_input_precision(self, device):
        """Neither gradient may silently truncate FP32 multiplication inputs to TF32."""
        value = np.float32(1 + 2**-12)
        a = _device_array([[value]], wp.float32, device)
        b = _device_array([[[value]]], wp.float32, device)
        dy = _device_array([[value]], wp.float32, device)
        lengths = wp.array(np.array([1], dtype=np.int32), device=device)
        da = wp.empty(a.shape, device=device)
        db = wp.empty(b.shape, device=device)
        segmm.segment_mm_backward(a, b, lengths, dy, da, db)
        np.testing.assert_array_equal(_numpy(da), [[np.float32(value * value)]])
        np.testing.assert_array_equal(_numpy(db), [[[np.float32(value * value)]]])

    @pytest.mark.parametrize("operand", ["a", "b"])
    def test_directional_finite_difference(self, device, operand):
        """Compare each gradient to changes in the actual forward operation."""
        rng = np.random.default_rng(11)
        a_np = rng.standard_normal((4, 3)).astype(np.float32)
        b_np = rng.standard_normal((3, 3, 2)).astype(np.float32)
        dy_np = rng.standard_normal((4, 2)).astype(np.float32)
        a, b, dy = (_device_array(x, wp.float32, device) for x in (a_np, b_np, dy_np))
        lengths = wp.array(np.array([1, 0, 2], dtype=np.int32), device=device)
        da, db = wp.empty(a.shape, device=device), wp.empty(b.shape, device=device)
        segmm.segment_mm_backward(a, b, lengths, dy, da, db)
        primal = a_np if operand == "a" else b_np
        direction = rng.standard_normal(primal.shape).astype(np.float32)
        analytical = np.sum((_numpy(da) if operand == "a" else _numpy(db)) * direction)
        losses = []
        epsilon = 1e-3
        out = wp.empty(dy.shape, device=device)
        for sign in (-1, 1):
            perturbed = _device_array(
                primal + sign * epsilon * direction, wp.float32, device
            )
            segmm.segment_mm(
                perturbed if operand == "a" else a,
                perturbed if operand == "b" else b,
                lengths,
                out,
            )
            losses.append(np.sum(_numpy(out).astype(np.float64) * dy_np))
        np.testing.assert_allclose(
            analytical, (losses[1] - losses[0]) / (2 * epsilon), rtol=2e-3, atol=2e-3
        )

    @pytest.mark.parametrize("dtype", [wp.float32, wp.bfloat16])
    @pytest.mark.parametrize("strided", [False, True])
    def test_cuda_graph_replay(self, device, dtype, strided):
        """Replay transpose workspaces with new input data and GPU lengths."""
        a_np = np.arange(7 * 4, dtype=np.float32).reshape(7, 4) / 16
        b_np = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5) / 32
        dy_np = np.ones((7, 5), dtype=np.float32)
        with wp.ScopedStream(wp.Stream(device)):
            if strided:
                a = _device_array(a_np.T.copy(), dtype, device).transpose()
                b = _device_array(
                    b_np.transpose(0, 2, 1).copy(), dtype, device
                ).transpose((0, 2, 1))
                dy = _device_array(dy_np.T.copy(), dtype, device).transpose()
            else:
                a = _device_array(a_np, dtype, device)
                b = _device_array(b_np, dtype, device)
                dy = _device_array(dy_np, dtype, device)
            lengths = wp.zeros(6, dtype=wp.int32, device=device)[::2]
            da, db = (wp.empty(x.shape, dtype=dtype, device=device) for x in (a, b))
            segmm.segment_mm_backward(a, b, lengths, dy, da, db)
            wp.launch(_write_lengths, dim=3, inputs=[lengths, 0], device=device)
            with wp.ScopedCapture(device=device) as capture:
                segmm.segment_mm_backward(a, b, lengths, dy, da, db)
            for state in (0, 2, 5):
                current_a, current_b, current_dy = (
                    a_np + state / 16,
                    b_np + state / 16,
                    dy_np + state / 8,
                )
                wp.copy(a, _device_array(current_a, dtype, device))
                wp.copy(b, _device_array(current_b, dtype, device))
                wp.copy(dy, _device_array(current_dy, dtype, device))
                wp.launch(_write_lengths, dim=3, inputs=[lengths, state], device=device)
                wp.capture_launch(capture.graph)
                expected_a, expected_b = _backward_reference(
                    current_a, current_b, [state, 0, 5 - state], current_dy, dtype
                )
                np.testing.assert_array_equal(_numpy(da), expected_a)
                np.testing.assert_array_equal(_numpy(db), expected_b)


@pytest.mark.gpu
class TestSegmentMMBackwardValidation:
    """Reject invalid backward buffers before either gradient is written."""

    @pytest.mark.parametrize(
        "case,match",
        [
            ("a_rank", "a.*2D"),
            ("b_shape", "channel"),
            ("length_dtype", "int32"),
            ("dy_rank", "grad_out.*2D"),
            ("dy_shape", "grad_out.*shape"),
            ("dy_dtype", "dtype"),
            ("dy_device", "device"),
            ("da_shape", "grad_a.*shape"),
            ("db_shape", "grad_b.*shape"),
            ("da_dtype", "dtype"),
            ("db_dtype", "dtype"),
            ("da_strided", "grad_a.*contiguous"),
            ("db_strided", "grad_b.*contiguous"),
            ("da_device", "device"),
            ("db_device", "device"),
        ],
    )
    def test_invalid_arrays(self, device, case, match):
        """Validate cotangent and both output contracts before launching any work."""
        a = wp.zeros((8, 4), device=device)
        b = wp.zeros((2, 4, 4), device=device)
        lengths = wp.zeros(2, dtype=wp.int32, device=device)
        dy = wp.zeros((8, 4), device=device)
        da = _device_array(np.full(a.shape, np.nan), wp.float32, device)
        db = _device_array(np.full(b.shape, np.nan), wp.float32, device)
        if case == "a_rank":
            a = a.flatten()
        elif case == "b_shape":
            b = wp.empty((2, 5, 4), device=device)
        elif case == "length_dtype":
            lengths = wp.zeros(2, dtype=wp.int64, device=device)
        elif case == "dy_rank":
            dy = dy.flatten()
        elif case == "dy_shape":
            dy = wp.empty((7, 4), device=device)
        elif case == "dy_dtype":
            dy = wp.empty(dy.shape, dtype=wp.bfloat16, device=device)
        elif case == "dy_device":
            dy = wp.empty(dy.shape, device="cpu")
        elif case == "da_shape":
            da = wp.empty((7, 4), device=device)
        elif case == "db_shape":
            db = wp.empty((1, 4, 4), device=device)
        elif case == "da_dtype":
            da = wp.empty(da.shape, dtype=wp.bfloat16, device=device)
        elif case == "db_dtype":
            db = wp.empty(db.shape, dtype=wp.bfloat16, device=device)
        elif case == "da_strided":
            da = wp.empty((4, 8), device=device).transpose()
        elif case == "db_strided":
            db = db.transpose((0, 2, 1))
        elif case == "da_device":
            da = wp.empty(da.shape, device="cpu")
        elif case == "db_device":
            db = wp.empty(db.shape, device="cpu")

        with pytest.raises(ValueError, match=match):
            segmm.segment_mm_backward(a, b, lengths, dy, da, db)
        if case.startswith("db_"):
            assert np.isnan(_numpy(da)).all()

    @pytest.mark.parametrize("output", ["grad_a", "grad_b"])
    @pytest.mark.parametrize(
        "source", ["a", "b", "grad_out", "seglen", "other_gradient"]
    )
    def test_overlapping_gradients(self, device, output, source):
        """Outputs may not corrupt inputs or each other's gradient storage."""
        length_storage = wp.zeros(32, dtype=wp.int32, device=device)
        arrays = {
            "a": wp.zeros((8, 4), device=device),
            "b": wp.zeros((2, 4, 4), device=device),
            "seglen": length_storage[::16],
            "grad_out": wp.zeros((8, 4), device=device),
            "grad_a": wp.empty((8, 4), device=device),
            "grad_b": wp.empty((2, 4, 4), device=device),
        }
        shape = arrays[output].shape
        if source == "seglen":
            arrays[output] = wp.array(
                ptr=length_storage.ptr,
                shape=shape,
                dtype=wp.float32,
                device=device,
                copy=False,
            )
        else:
            key = (
                ("grad_b" if output == "grad_a" else "grad_a")
                if source == "other_gradient"
                else source
            )
            arrays[output] = arrays[key].reshape(shape)
        with pytest.raises(ValueError, match="overlap"):
            segmm.segment_mm_backward(**arrays)


@pytest.mark.gpu
class TestSegmentMMLengthBounds:
    """Keep invalid device lengths from escaping input capacity or wrapping offsets."""

    @pytest.mark.parametrize("dtype", [wp.float32, wp.bfloat16])
    @pytest.mark.parametrize("groups", [4, 129])
    @pytest.mark.parametrize(
        "values,effective",
        [
            pytest.param([-1, 33, -(2**31), 3], [0, 33, 0, 3], id="negative"),
            pytest.param([1, 40, 2, 0], [1, 36, 0, 0], id="over_capacity"),
            pytest.param(
                [1, 2**31 - 1, 2**31 - 1, 3],
                [1, 36, 0, 0],
                id="int32_overflow",
            ),
            pytest.param([-(2**31), -1, -33, -9], [0, 0, 0, 0], id="all_negative"),
        ],
    )
    def test_bounded_lengths(
        self, device, monkeypatch, dtype, groups, values, effective
    ):
        """Check safe prefixes before executing products and both gradient entry points."""
        capacity, k, n = 37, 3, 5
        values = np.pad(np.asarray(values, dtype=np.int32), (0, groups - 4))
        effective = np.pad(np.asarray(effective, dtype=np.int32), (0, groups - 4))
        expected_rows = np.concatenate(([0], np.cumsum(effective, dtype=np.int64)))
        expected_tiles = np.concatenate(
            ([0], np.cumsum((effective.astype(np.int64) + 31) // 32))
        )
        original_offsets = segmm._segment_offsets

        def checked_offsets(*args, **kwargs):
            """Stop before unsafe matrix access if the real GPU prefix builder regresses."""
            rows, tiles = original_offsets(*args, **kwargs)
            np.testing.assert_array_equal(rows.numpy(), expected_rows)
            np.testing.assert_array_equal(tiles.numpy(), expected_tiles)
            return rows, tiles

        monkeypatch.setattr(segmm, "_segment_offsets", checked_offsets)
        a_np = (np.arange(capacity * k).reshape(capacity, k) % 7 - 3).astype(np.float32)
        b_np = (np.arange(groups * k * n).reshape(groups, k, n) % 5 - 2).astype(
            np.float32
        )
        g_np = (np.arange(capacity * n).reshape(capacity, n) % 3 - 1).astype(np.float32)
        a_np[effective.sum() :] = g_np[effective.sum() :] = np.nan
        b_np[effective == 0] = np.nan
        a, b, g = (_device_array(x, dtype, device) for x in (a_np, b_np, g_np))
        lengths = wp.array(values, dtype=wp.int32, device=device)
        out_storage_np = np.full((capacity + 2, n), np.nan, dtype=np.float32)
        da_storage_np = np.full((capacity + 2, k), np.nan, dtype=np.float32)
        out_storage_np[[0, -1]] = da_storage_np[[0, -1]] = 127
        out_storage = _device_array(out_storage_np, dtype, device)
        da_storage = _device_array(da_storage_np, dtype, device)
        out, da = out_storage[1:-1], da_storage[1:-1]
        db = _device_array(np.full(b.shape, np.nan), dtype, device)
        db_only = _device_array(np.full(b.shape, np.nan), dtype, device)

        segmm.segment_mm(a, b, lengths, out)
        segmm.segment_mm_backward(a, b, lengths, g, da, db)
        segmm.segment_mm_backward_b(a, lengths, g, db_only)

        expected = _reference(a_np, b_np, effective, dtype)
        expected_a, expected_b = _backward_reference(a_np, b_np, effective, g_np, dtype)
        np.testing.assert_array_equal(_numpy(out), expected)
        np.testing.assert_array_equal(_numpy(da), expected_a)
        np.testing.assert_array_equal(_numpy(db), expected_b)
        np.testing.assert_array_equal(_numpy(db_only), expected_b)
        np.testing.assert_array_equal(_numpy(out_storage)[[0, -1]], 127)
        np.testing.assert_array_equal(_numpy(da_storage)[[0, -1]], 127)


@pytest.mark.gpu
class TestSegmentMMValidation:
    """Reject invalid host-visible metadata before any computation is launched."""

    @pytest.mark.parametrize(
        "case,match",
        [
            ("a_rank", "a.*2D"),
            ("b_rank", "b.*3D"),
            ("length_rank", "seglen.*1D"),
            ("out_rank", "out.*2D"),
            ("k_mismatch", "channel"),
            ("group_mismatch", "segment"),
            ("out_shape", "out.*shape"),
            ("zero_channels", "positive"),
            ("a_dtype", "dtype"),
            ("b_dtype", "dtype"),
            ("out_dtype", "dtype"),
            ("length_dtype", "int32"),
            ("strided_out", "out.*contiguous"),
            ("device", "device"),
            ("alias_a", "overlap"),
            ("alias_b", "overlap"),
            ("alias_partial", "overlap"),
        ],
    )
    def test_invalid_arrays(self, device, case, match):
        """Bad layouts, types, and overlapping outputs must not reach a GPU kernel."""
        a = wp.empty((8, 4), dtype=wp.float32, device=device)
        b = wp.empty((2, 4, 4), dtype=wp.float32, device=device)
        lengths = wp.zeros(2, dtype=wp.int32, device=device)
        out = wp.empty((8, 4), dtype=wp.float32, device=device)
        if case == "a_rank":
            a = a.flatten()
        elif case == "b_rank":
            b = b.flatten()
        elif case == "length_rank":
            lengths = lengths.reshape((1, 2))
        elif case == "out_rank":
            out = out.flatten()
        elif case == "k_mismatch":
            b = wp.empty((2, 5, 4), device=device)
        elif case == "group_mismatch":
            lengths = wp.zeros(3, dtype=wp.int32, device=device)
        elif case == "out_shape":
            out = wp.empty((7, 4), device=device)
        elif case == "zero_channels":
            a = wp.empty((8, 0), device=device)
            b = wp.empty((2, 0, 4), device=device)
        elif case == "a_dtype":
            a = wp.empty(a.shape, dtype=wp.float16, device=device)
        elif case == "b_dtype":
            b = wp.empty(b.shape, dtype=wp.bfloat16, device=device)
        elif case == "out_dtype":
            out = wp.empty(out.shape, dtype=wp.bfloat16, device=device)
        elif case == "length_dtype":
            lengths = wp.empty(2, dtype=wp.int64, device=device)
        elif case == "strided_out":
            out = wp.empty((4, 8), device=device).transpose()
        elif case == "device":
            lengths = wp.zeros(2, dtype=wp.int32, device="cpu")
        elif case == "alias_a":
            out = a
        elif case == "alias_b":
            out = b.reshape((8, 4))
        elif case == "alias_partial":
            storage = wp.empty((9, 4), device=device)
            a, out = storage[:8], storage[1:]
        with pytest.raises(ValueError, match=match):
            segmm.segment_mm(a, b, lengths, out)

    @pytest.mark.parametrize("dtype", [wp.float32, wp.bfloat16])
    @pytest.mark.parametrize("operand", ["a", "b", "seglen"])
    def test_strided_input_overlap_is_rejected(self, device, dtype, operand):
        """Reject overlap near the far end of storage, beyond a dense-size estimate."""
        storage = wp.empty(512, dtype=dtype, device=device)
        a = wp.empty((8, 4), dtype=dtype, device=device)
        b = wp.empty((2, 4, 4), dtype=dtype, device=device)
        lengths = wp.zeros(2, dtype=wp.int32, device=device)
        out = storage[128:160].reshape((8, 4))
        if operand == "a":
            a = storage[:256].reshape((64, 4))[::8]
        elif operand == "b":
            b = storage[:256].reshape((16, 4, 4))[::8]
        else:
            length_storage = wp.zeros(128, dtype=wp.int32, device=device)
            lengths = length_storage[::64]
            out = wp.array(
                ptr=length_storage.ptr + 256,
                shape=(8, 4),
                dtype=dtype,
                device=device,
                copy=False,
            )

        with pytest.raises(ValueError, match="overlap"):
            segmm.segment_mm(a, b, lengths, out)

    @pytest.mark.parametrize("operand", ["a", "b", "seglen"])
    @pytest.mark.parametrize("stride", [-4, 2**31])
    def test_unsupported_strides_are_rejected(self, device, operand, stride):
        """Reject reverse views and byte strides that overflow Warp's descriptor."""
        arrays = {
            "a": wp.zeros((1, 1), dtype=wp.float32, device=device),
            "b": wp.zeros((1, 1, 1), dtype=wp.float32, device=device),
            "seglen": wp.zeros(1, dtype=wp.int32, device=device),
        }
        storage = arrays[operand]
        arrays[operand] = wp.array(
            ptr=storage.ptr,
            shape=storage.shape,
            strides=(stride, *storage.strides[1:]),
            dtype=storage.dtype,
            device=device,
            copy=False,
        )
        out = wp.empty((1, 1), device=device)

        with pytest.raises(ValueError, match="strides"):
            segmm.segment_mm(**arrays, out=out)


class TestSegmentMMCPU:
    """Exercise complete tiles and explicit gradients with one CPU lane."""

    @pytest.mark.parametrize("dtype", [wp.float32, wp.bfloat16])
    @pytest.mark.parametrize("groups", [4, 129])
    @pytest.mark.parametrize("strided", [False, True])
    def test_forward_and_gradients(self, dtype, groups, strided):
        """Write every tile element, skip empty groups, and exclude padded NaNs."""
        rng = np.random.default_rng(73)
        values = np.pad(np.array([0, 33, 0, 37], dtype=np.int32), (0, groups - 4))
        a_np = rng.integers(-4, 5, (75, 35)).astype(np.float32) / 8
        b_np = rng.integers(-4, 5, (groups, 35, 67)).astype(np.float32) / 8
        g_np = rng.integers(-4, 5, (75, 67)).astype(np.float32) / 8
        a_np[70:] = g_np[70:] = np.nan
        b_np[values == 0] = np.nan
        a, b, g = (_device_array(x, dtype, "cpu") for x in (a_np, b_np, g_np))
        lengths = wp.array(values, dtype=wp.int32, device="cpu")
        if strided:
            a = _device_array(a_np.T.copy(), dtype, "cpu").transpose()
            b = _device_array(b_np.transpose(0, 2, 1).copy(), dtype, "cpu").transpose(
                (0, 2, 1)
            )
            g = _device_array(g_np.T.copy(), dtype, "cpu").transpose()
            lengths = wp.array(np.repeat(values, 2), device="cpu")[::2]
        out, da, db, db_only = (
            _device_array(np.full(shape, np.nan), dtype, "cpu")
            for shape in (g.shape, a.shape, b.shape, b.shape)
        )

        segmm.segment_mm(a, b, lengths, out)
        segmm.segment_mm_backward(a, b, lengths, g, da, db)
        segmm.segment_mm_backward_b(a, lengths, g, db_only)

        expected_a, expected_b = _backward_reference(a_np, b_np, values, g_np, dtype)
        np.testing.assert_array_equal(
            _numpy(out), _reference(a_np, b_np, values, dtype)
        )
        np.testing.assert_array_equal(_numpy(da), expected_a)
        np.testing.assert_array_equal(_numpy(db), expected_b)
        np.testing.assert_array_equal(_numpy(db_only), expected_b)

    @pytest.mark.parametrize("dtype", [wp.float32, wp.bfloat16])
    @pytest.mark.parametrize("capacity,groups", [(0, 3), (11, 0), (0, 0)])
    def test_empty(self, dtype, capacity, groups):
        """Empty CPU inputs overwrite preallocated outputs without tile launches."""
        a = wp.empty((capacity, 4), dtype=dtype, device="cpu")
        b = wp.empty((groups, 4, 3), dtype=dtype, device="cpu")
        g = wp.empty((capacity, 3), dtype=dtype, device="cpu")
        lengths = wp.zeros(groups, dtype=wp.int32, device="cpu")
        out, da, db = (
            _device_array(np.full(shape, np.nan), dtype, "cpu")
            for shape in (g.shape, a.shape, b.shape)
        )
        segmm.segment_mm(a, b, lengths, out)
        segmm.segment_mm_backward(a, b, lengths, g, da, db)
        np.testing.assert_array_equal(_numpy(out), 0)
        np.testing.assert_array_equal(_numpy(da), 0)
        np.testing.assert_array_equal(_numpy(db), 0)

    @pytest.mark.parametrize("dtype", [wp.float32, wp.bfloat16])
    def test_launch_leaves_use_caller_owned_scratch(self, dtype, monkeypatch):
        """Framework callbacks can compute split gradients without Warp allocation."""
        a_np = np.ones((1031, 3), dtype=np.float32)
        g_np = np.ones((1031, 5), dtype=np.float32) / 8
        values = np.array([0, 1025, 0], dtype=np.int32)
        a_np[1025:] = g_np[1025:] = np.nan
        b_np = np.ones((3, 3, 5), dtype=np.float32) / 4
        a, b, g = (_device_array(x, dtype, "cpu") for x in (a_np, b_np, g_np))
        lengths = wp.array(values, device="cpu")
        rows, tiles = segmm._segment_offsets(lengths, a.shape[0])
        out = wp.empty(g.shape, dtype=dtype, device="cpu")
        db = wp.empty(b.shape, dtype=dtype, device="cpu")
        splits, _ = segmm._grad_b_launch_config(
            1031, 3, 3, 5, 2 if dtype == wp.bfloat16 else 4
        )
        partials = wp.empty((3 * splits, 3, 5), dtype=wp.float32, device="cpu")
        # Warm compilation before enforcing the callback's storage contract.
        segmm._launch_forward(a, b, rows, tiles, out)
        segmm._launch_grad_b(a, rows, g, db, partials, splits)

        def reject_allocation(*args, **kwargs):
            """Detect storage created inside either otherwise-real launch leaf."""
            pytest.fail("launch leaves must use caller-owned storage")

        with monkeypatch.context() as patch:
            patch.setattr(wp, "empty", reject_allocation)
            patch.setattr(wp, "zeros", reject_allocation)
            segmm._launch_forward(a, b, rows, tiles, out)
            segmm._launch_grad_b(a, rows, g, db, partials, splits)

        np.testing.assert_array_equal(
            _numpy(out), _reference(a_np, b_np, values, dtype)
        )
        expected_b = _backward_reference(a_np, b_np, values, g_np, dtype)[1]
        np.testing.assert_array_equal(_numpy(db), expected_b)
