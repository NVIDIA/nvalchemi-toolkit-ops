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
    def test_gradients(self, device, dtype, k, n, monkeypatch):
        """Respect empty segments and padding without copying full BF16 weights."""
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

        copy = wp.copy

        def copy_without_weights(dest, src, *args, **kwargs):
            """Reject materializing B while allowing unrelated runtime copies."""
            if (
                dtype == wp.bfloat16
                and src.ptr == b.ptr
                and src.size == b.size
                and dest.ptr != b.ptr
            ):
                pytest.fail("BF16 input gradients must not copy the full weight array")
            return copy(dest, src, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(wp, "copy", copy_without_weights)
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
    def test_strided_inputs(self, device, dtype, a_layout, monkeypatch):
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

        empty = wp.empty

        def without_weight_scratch(shape, *args, **kwargs):
            """A contiguous B transpose must not allocate another full weight buffer."""
            assert shape != (b.shape[0], b.shape[2], b.shape[1])
            return empty(shape, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(wp, "empty", without_weight_scratch)
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
        """Replay backward with new original weights, input data, and GPU lengths."""
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
        original_offsets = segmm._segment_offsets

        def checked_offsets(
            seglen, capacity, tile_rows=32, skip_first_tile=False, *, validation=None
        ):
            """Stop before unsafe matrix access if the real GPU prefix builder regresses."""
            rows, tiles = original_offsets(
                seglen, capacity, tile_rows, skip_first_tile, validation=validation
            )
            counts = (effective.astype(np.int64) + tile_rows - 1) // tile_rows
            if skip_first_tile:
                counts = np.maximum(counts - 1, 0)
            expected_tiles = np.concatenate(([0], np.cumsum(counts)))
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


class TestSegmentMMCompactScratch:
    """Keep splitting available when segment count exceeds the scratch budget."""

    @pytest.mark.parametrize("groups", [1023, 1024, 1025])
    def test_many_segments_retain_split_budget(self, groups):
        """Short segments must not consume scratch needed by a long neighbor."""
        capacity, k, n = 118784, 128, 128
        chunk_rows, slots = segmm._grad_b_launch_config(capacity, groups, k, n, 4)
        assert chunk_rows < 16000
        extra_capacity = (capacity - 1) // chunk_rows
        assert slots == extra_capacity + min(groups, extra_capacity)
        assert slots * k * n * 4 <= 64 << 20
        lengths = np.full(groups, 100, np.int64)
        lengths[0] = 16000
        extras = np.maximum((lengths + chunk_rows - 1) // chunk_rows - 1, 0)
        assert extras[0] > 0
        assert extras.sum() <= extra_capacity

    @pytest.mark.parametrize(
        "capacity,groups,chunk_rows,slots",
        [(131072, 1, 1024, 128), (131072, 4, 1056, 128), (262144, 128, 512, 639)],
    )
    def test_existing_large_extra_budget_is_preserved(
        self, capacity, groups, chunk_rows, slots
    ):
        """Compaction must preserve MD parallelism and avoid expanding padded scratch."""
        assert segmm._grad_b_launch_config(capacity, groups, 128, 128, 4) == (
            chunk_rows,
            slots,
        )

    @pytest.mark.parametrize("dtype", [wp.float32, wp.bfloat16])
    @pytest.mark.parametrize("groups", [4, 32])
    def test_compact_partials_change_split_owners(self, dtype, groups):
        """Both scratch layouts ignore stale slots between changing split owners."""
        capacity, k, n = 4096, 3, 5
        chunk_rows, slots = segmm._grad_b_launch_config(
            capacity, groups, k, n, 2 if dtype == wp.bfloat16 else 4
        )
        extra_capacity = (capacity - 1) // chunk_rows
        assert slots == extra_capacity + min(groups, extra_capacity)
        partials = wp.array(np.full((slots, k, n), np.nan, np.float32), device="cpu")
        output = _device_array(np.full((groups, k, n), np.nan), dtype, "cpu")
        states = (
            ([513, 0, 1, 1025], [0, 257, 769, 0], [0, 0, 0, 0], [1, 0, 31, 0])
            if groups == 4
            else (
                [0, 1, 1025, 0, 513, 0, 0, 257],
                [513, 0, 1, 0, 0, 2049, 0, 3],
                [0] * 8,
                [1, 0, 31, 0, 0, 0, 0, 0],
            )
        )
        for state in states:
            values = np.pad(np.asarray(state, np.int32), (0, groups - len(state)))
            a_np = np.full((capacity, k), np.nan, np.float32)
            g_np = np.full((capacity, n), np.nan, np.float32)
            expected = np.zeros((groups, k, n), np.float32)
            start = 0
            for group, length in enumerate(values):
                stop = start + int(length)
                a_np[start:stop] = (group + 1) / 8
                g_np[start:stop] = (group % 3 + 1) / 16
                expected[group] = int(length) * (group + 1) * (group % 3 + 1) / 128
                start = stop
            a, g = (_device_array(x, dtype, "cpu") for x in (a_np, g_np))
            lengths = wp.array(values, device="cpu")
            rows, chunks = segmm._segment_offsets(
                lengths, capacity, chunk_rows, skip_first_tile=True
            )
            segmm._launch_grad_b(a, rows, chunks, g, output, partials, chunk_rows)
            np.testing.assert_array_equal(
                _numpy(output), _round_output(expected, dtype)
            )


class TestSegmentMMPersistentReduction:
    """Check sparse split reduction, including fixed graph replay and CPU lanes."""

    @pytest.mark.parametrize(
        "device_name", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)]
    )
    @pytest.mark.parametrize("dtype", [wp.float32, wp.bfloat16])
    @pytest.mark.parametrize("k,n", [(3, 5), (35, 67), (129, 129)])
    def test_poisoned_partials_and_prefix_changes(self, device_name, dtype, k, n):
        """Only owned FP32 slots are read, and each split output rounds once."""
        if device_name == "cuda" and not wp.is_cuda_available():
            pytest.skip("CUDA is unavailable")
        device = wp.get_device(device_name)
        groups, extra_capacity = 128, 127
        tiles = (k * n + 127) // 128
        work_blocks = min(512, extra_capacity * tiles)
        rng = np.random.default_rng(731)
        partials = wp.empty((2 * extra_capacity, k, n), dtype=wp.float32, device=device)
        offsets = wp.empty(groups + 1, dtype=wp.int32, device=device)
        for active in ({}, {37: 127}, {1: 1, 7: 3, 96: 2}, {}):
            extras = np.zeros(groups, np.int32)
            for group, count in active.items():
                extras[group] = count
            prefix = np.concatenate(([0], np.cumsum(extras))).astype(np.int32)
            scratch = np.full(partials.shape, np.nan, np.float32)
            initial = np.full((groups, k, n), 0.25, np.float32)
            expected = initial.copy()
            for group in active:
                first = extra_capacity + int(prefix[group])
                scratch[first] = rng.normal(size=(k, n)).astype(np.float32)
                scratch[first, 0, 0] = 1.0
                value = scratch[first].copy()
                for chunk in range(int(prefix[group]), int(prefix[group + 1])):
                    scratch[chunk] = rng.normal(size=(k, n)).astype(np.float32)
                    scratch[chunk, 0, 0] = 2**-8
                    value = np.add(value, scratch[chunk], dtype=np.float32)
                expected[group] = value
            partials.assign(scratch)
            offsets.assign(prefix)
            output = _device_array(initial, dtype, device)
            wp.launch_tiled(
                segmm._reduce_grad_b_persistent_overloads[dtype],
                dim=(work_blocks,),
                inputs=[partials, offsets, work_blocks, output],
                block_dim=128,
                device=device,
            )
            np.testing.assert_array_equal(
                _numpy(output), _round_output(expected, dtype)
            )

    @pytest.mark.gpu
    @pytest.mark.parametrize("dtype", [wp.float32, wp.bfloat16])
    @pytest.mark.parametrize("joint", [False, True])
    @pytest.mark.parametrize("k,n", [(3, 5), (129, 129)])
    def test_public_capture_with_changing_split_groups(
        self, device, dtype, joint, k, n
    ):
        """Captured public gradients survive all-short, skewed, mixed, all-short replay."""
        capacity, groups = 8192, 64
        chunk_rows, slots = segmm._grad_b_launch_config(
            capacity, groups, k, n, 2 if dtype == wp.bfloat16 else 4
        )
        extra_capacity = (capacity - 1) // chunk_rows
        assert chunk_rows < capacity and 0 < extra_capacity < groups
        assert slots == 2 * extra_capacity
        states = [
            np.full(groups, 96, np.int32),
            np.zeros(groups, np.int32),
            np.zeros(groups, np.int32),
            np.full(groups, 32, np.int32),
        ]
        states[1][37] = 6145
        states[2][[1, 17, 40]] = [513, 2049, 1]
        with wp.ScopedStream(wp.Stream(device)):
            a = _device_array(np.full((capacity, k), 0.5), dtype, device)
            g = _device_array(np.full((capacity, n), 0.25), dtype, device)
            b = _device_array(np.full((groups, k, n), 0.125), dtype, device)
            lengths = wp.array(states[0], device=device)
            da = wp.empty(a.shape, dtype=dtype, device=device)
            db = wp.empty(b.shape, dtype=dtype, device=device)

            def launch():
                """Exercise either public route with the same live device buffers."""
                if joint:
                    segmm.segment_mm_backward(a, b, lengths, g, da, db)
                else:
                    segmm.segment_mm_backward_b(a, lengths, g, db)

            launch()
            with wp.ScopedCapture(device=device) as capture:
                launch()
            for values in states:
                current_a = np.full(a.shape, 0.5, np.float32)
                current_g = np.full(g.shape, 0.25, np.float32)
                current_a[values.sum() :] = current_g[values.sum() :] = np.nan
                wp.copy(a, _device_array(current_a, dtype, device))
                wp.copy(g, _device_array(current_g, dtype, device))
                lengths.assign(values)
                wp.capture_launch(capture.graph)
                expected_b = np.broadcast_to(values[:, None, None] / 8, db.shape)
                np.testing.assert_array_equal(
                    _numpy(db), _round_output(expected_b, dtype)
                )
                if joint:
                    expected_a = np.zeros(da.shape, np.float32)
                    expected_a[: values.sum()] = n / 32
                    np.testing.assert_array_equal(
                        _numpy(da), _round_output(expected_a, dtype)
                    )


class TestSegmentMMCPU:
    """Exercise complete tiles and explicit gradients with one CPU lane."""

    @pytest.mark.parametrize("groups,k,n", [(4, 3, 5), (129, 256, 256)])
    def test_adaptive_chunks_fill_fixed_workspace(self, groups, k, n):
        """Padded capacity must not reduce active work below the fixed slot budget."""
        capacity = 131072
        chunk_rows, slots = segmm._grad_b_launch_config(capacity, groups, k, n, 4)
        cases = (
            ([25000] * 4, [31] * 4),
            ([25001, 25000, 25000, 25000], [30] * 4),
            ([100000, 15000, 2000, 0], [104, 15, 2, 0]),
            ([0, 0, 0, 0], [0, 0, 0, 0]),
            ([2147483647, -1, 0, 0], [124, 0, 0, 0]),
        )
        if groups == 129:
            # The fixed extra-slot bound rounds 100k active rows to 800-row chunks.
            cases = (
                ([25000] * 4, [31] * 4),
                ([0, 0, 0, 0], [0, 0, 0, 0]),
                ([513, 512, 1, 0], [1, 0, 0, 0]),
            )
        for values, extra_counts in cases:
            lengths = wp.array(
                np.pad(values, (0, groups - 4)).astype(np.int32), device="cpu"
            )
            rows, extras = segmm._segment_offsets(
                lengths, capacity, chunk_rows, skip_first_tile=True
            )
            expected = np.concatenate(([0], np.cumsum(extra_counts)))
            expected = np.pad(expected, (0, groups - 4), mode="edge")
            np.testing.assert_array_equal(extras.numpy(), expected)
            active_extras = int(extras.numpy()[-1])
            assert active_extras <= (capacity - 1) // chunk_rows
            assert active_extras + min(groups, active_extras) <= slots
            fused = wp.empty(groups + 1, dtype=wp.int32, device="cpu")
            fused_rows, _ = segmm._segment_offsets(
                lengths, capacity, chunk_rows=chunk_rows, chunk_offsets=fused
            )
            np.testing.assert_array_equal(fused_rows.numpy(), rows.numpy())
            np.testing.assert_array_equal(fused.numpy(), expected)

    @pytest.mark.parametrize("dtype", [wp.float32, wp.bfloat16])
    def test_adaptive_chunk_replay_uses_live_lengths(self, dtype):
        """Changing chunk sizes must overwrite active partials and ignore stale slots."""
        capacity, groups, k, n = 131072, 4, 3, 5
        chunk_rows, slots = segmm._grad_b_launch_config(capacity, groups, k, n, 4)
        partials = wp.array(np.full((slots, k, n), np.nan, np.float32), device="cpu")
        db = _device_array(np.full((groups, k, n), np.nan), dtype, "cpu")
        for values in (
            [25000] * 4,
            [0] * 4,
            [257, 511, 0, 1],
            [100000, 15000, 2000, 0],
        ):
            a_np = np.full((capacity, k), 0.5, np.float32)
            g_np = np.full((capacity, n), 0.25, np.float32)
            a_np[sum(values) :] = g_np[sum(values) :] = np.nan
            a, g = (_device_array(x, dtype, "cpu") for x in (a_np, g_np))
            lengths = wp.array(values, dtype=wp.int32, device="cpu")
            rows, extras = segmm._segment_offsets(lengths, capacity, chunk_rows, True)
            segmm._launch_grad_b(a, rows, extras, g, db, partials, chunk_rows)
            expected = np.broadcast_to(np.asarray(values)[:, None, None] / 8, db.shape)
            np.testing.assert_array_equal(_numpy(db), _round_output(expected, dtype))

    @pytest.mark.parametrize("groups", [4, 129])
    def test_joint_backward_builds_row_prefix_once(self, groups, monkeypatch):
        """Both gradients must share one scheduling pass without changing results."""
        capacity, k, n = 2048, 3, 5
        values = np.pad(np.array([0, 1025, 0, 33], np.int32), (0, groups - 4))
        a_np = np.full((capacity, k), 0.5, np.float32)
        g_np = np.full((capacity, n), 0.25, np.float32)
        a_np[values.sum() :] = g_np[values.sum() :] = np.nan
        b_np = np.full((groups, k, n), 0.125, np.float32)
        a, b, g = (_device_array(x, wp.float32, "cpu") for x in (a_np, b_np, g_np))
        lengths = wp.array(values, device="cpu")
        da = wp.empty(a.shape, dtype=wp.float32, device="cpu")
        db = wp.empty(b.shape, dtype=wp.float32, device="cpu")
        calls = []
        original = segmm._segment_offsets

        def observe(*args, **kwargs):
            """Count real schedule construction while retaining all device work."""
            calls.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(segmm, "_segment_offsets", observe)
        segmm.segment_mm_backward(a, b, lengths, g, da, db)
        expected_a, expected_b = _backward_reference(
            a_np, b_np, values, g_np, wp.float32
        )
        np.testing.assert_array_equal(_numpy(da), expected_a)
        np.testing.assert_array_equal(_numpy(db), expected_b)
        assert len(calls) == 1

    @pytest.mark.parametrize(
        "capacity,groups,k,n",
        [
            (7, 129, 3, 5),
            (4096, 4, 35, 67),
            (40000, 129, 3, 5),
            (65536, 64, 512, 512),
            (1000000, 1, 1, 1),
        ],
    )
    def test_chunk_workspace_covers_skewed_lengths(self, capacity, groups, k, n):
        """Every possible tail-heavy schedule must fit the bounded FP32 workspace."""
        chunk_rows, chunk_capacity = segmm._grad_b_launch_config(
            capacity, groups, k, n, 2
        )
        nonempty = min(capacity, groups)
        lengths = np.zeros(groups, dtype=np.int64)
        lengths[:nonempty] = 1
        lengths[nonempty - 1] += capacity - nonempty
        counts = lengths // chunk_rows + (lengths % chunk_rows != 0)
        extras = np.maximum(counts - 1, 0)
        assert extras.sum() + np.count_nonzero(extras) <= chunk_capacity
        one_segment_chunks = capacity // chunk_rows + (capacity % chunk_rows != 0)
        assert one_segment_chunks - 1 + int(one_segment_chunks > 1) <= chunk_capacity
        assert chunk_capacity * k * n * 4 <= 64 << 20
        extra_capacity = (capacity - 1) // chunk_rows
        assert groups + extra_capacity <= max(128, 32 * groups)
        if chunk_rows >= capacity:
            assert chunk_capacity == 0
        assert chunk_rows % 32 == 0

    @pytest.mark.parametrize("dtype", [wp.float32, wp.bfloat16])
    @pytest.mark.parametrize("groups", [4, 129])
    @pytest.mark.parametrize("k,n", [(3, 5), (35, 67)])
    def test_chunk_replay_with_singletons_and_empty_segments(self, dtype, groups, k, n):
        """Changing chunk ownership must ignore stale singleton and unused partials."""
        capacity = 4096
        chunk_rows, chunk_capacity = segmm._grad_b_launch_config(
            capacity, groups, k, n, 2 if dtype == wp.bfloat16 else 4
        )
        partials = wp.array(
            np.full((chunk_capacity, k, n), np.nan, np.float32), device="cpu"
        )
        db = _device_array(np.full((groups, k, n), np.nan), dtype, "cpu")
        for values in (
            [0, 1, 2305, 33],
            [0, 0, 0, 0],
            [0, 0, 31, 0],
            [1025, 0, 1, 513],
        ):
            values = np.pad(np.asarray(values, dtype=np.int32), (0, groups - 4))
            used = sum(values)
            a_np = np.full((capacity, k), 0.5, np.float32)
            g_np = np.full((capacity, n), 0.25, np.float32)
            a_np[used:] = g_np[used:] = np.nan
            a, g = (_device_array(x, dtype, "cpu") for x in (a_np, g_np))
            lengths = wp.array(values, dtype=wp.int32, device="cpu")
            rows, chunks = segmm._segment_offsets(
                lengths, capacity, chunk_rows, skip_first_tile=True
            )
            segmm._launch_grad_b(a, rows, chunks, g, db, partials, chunk_rows)
            expected = np.broadcast_to(np.asarray(values)[:, None, None] / 8, db.shape)
            np.testing.assert_array_equal(_numpy(db), _round_output(expected, dtype))

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
        da = wp.empty(a.shape, dtype=dtype, device="cpu")
        db = wp.empty(b.shape, dtype=dtype, device="cpu")
        chunk_rows, chunk_capacity = segmm._grad_b_launch_config(
            1031, 3, 3, 5, 2 if dtype == wp.bfloat16 else 4
        )
        _, chunks = segmm._segment_offsets(
            lengths, a.shape[0], chunk_rows, skip_first_tile=True
        )
        partials = wp.empty((chunk_capacity, 3, 5), dtype=wp.float32, device="cpu")
        transposed = wp.empty((3, 5, 3), dtype=dtype, device="cpu")
        # Warm compilation before enforcing the callback's storage contract.
        segmm._launch_forward(a, b, rows, tiles, out)
        segmm._launch_grad_a(g, b, rows, tiles, da, transposed)
        segmm._launch_grad_b(a, rows, chunks, g, db, partials, chunk_rows)

        def reject_allocation(*args, **kwargs):
            """Detect storage created inside the otherwise-real launch leaves."""
            pytest.fail("launch leaves must use caller-owned storage")

        with monkeypatch.context() as patch:
            patch.setattr(wp, "empty", reject_allocation)
            patch.setattr(wp, "zeros", reject_allocation)
            segmm._launch_forward(a, b, rows, tiles, out)
            segmm._launch_grad_a(g, b, rows, tiles, da, transposed)
            segmm._launch_grad_b(a, rows, chunks, g, db, partials, chunk_rows)

        np.testing.assert_array_equal(
            _numpy(out), _reference(a_np, b_np, values, dtype)
        )
        expected_a, expected_b = _backward_reference(a_np, b_np, values, g_np, dtype)
        np.testing.assert_array_equal(_numpy(da), expected_a)
        np.testing.assert_array_equal(_numpy(db), expected_b)


class TestSegmentMMDeviceValidation:
    """Expose device length validity without reading lengths or status on the host."""

    @pytest.fixture(params=["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
    def validation_device(self, request):
        """Run the same public status contract on CPU and CUDA."""
        if request.param == "cuda" and not wp.is_cuda_available():
            pytest.skip("CUDA is unavailable")
        return wp.get_device(request.param)

    @staticmethod
    def _buffers(device, dtype=wp.float32, capacity=7, groups=4):
        """Build small exact-valued operands and poisoned caller-owned outputs."""
        arrays = {}
        for name, shape, divisor in (
            ("a", (capacity, 3), 8),
            ("b", (groups, 3, 5), 4),
            ("grad_out", (capacity, 5), 8),
        ):
            values = (np.arange(np.prod(shape)).reshape(shape) % 7 - 3) / divisor
            arrays[name] = _device_array(values, dtype, device)
        arrays["seglen"] = wp.zeros(groups, dtype=wp.int32, device=device)
        for name, shape in (
            ("out", (capacity, 5)),
            ("grad_a", (capacity, 3)),
            ("grad_b", (groups, 3, 5)),
            ("grad_b_only", (groups, 3, 5)),
        ):
            arrays[name] = _device_array(np.full(shape, np.nan), dtype, device)
        return arrays

    @staticmethod
    def _calls(arrays):
        """List the real public calls and their positional operands."""
        return (
            (
                segmm.segment_mm,
                tuple(arrays[name] for name in ("a", "b", "seglen", "out")),
            ),
            (
                segmm.segment_mm_backward,
                tuple(
                    arrays[name]
                    for name in ("a", "b", "seglen", "grad_out", "grad_a", "grad_b")
                ),
            ),
            (
                segmm.segment_mm_backward_b,
                tuple(
                    arrays[name] for name in ("a", "seglen", "grad_out", "grad_b_only")
                ),
            ),
        )

    @staticmethod
    def _assert_results(arrays, values, dtype):
        """Check public results against independent NumPy products and gradients."""
        a, b, grad_out = (_numpy(arrays[name]) for name in ("a", "b", "grad_out"))
        expected_a, expected_b = _backward_reference(a, b, values, grad_out, dtype)
        np.testing.assert_array_equal(
            _numpy(arrays["out"]), _reference(a, b, values, dtype)
        )
        np.testing.assert_array_equal(_numpy(arrays["grad_a"]), expected_a)
        np.testing.assert_array_equal(_numpy(arrays["grad_b"]), expected_b)
        np.testing.assert_array_equal(_numpy(arrays["grad_b_only"]), expected_b)

    @pytest.mark.parametrize("dtype", [wp.float32, wp.bfloat16])
    @pytest.mark.parametrize("groups", [4, 129, 1025])
    def test_status_tracks_live_lengths(self, validation_device, dtype, groups):
        """Checked scalar, cooperative, and scanned paths replace stale validity."""
        arrays = self._buffers(validation_device, dtype, capacity=513, groups=groups)
        # Preserve support for strided raw int32 lengths in the checked path.
        storage = wp.zeros(2 * groups, dtype=wp.int32, device=validation_device)
        arrays["seglen"] = storage[::2]
        validation = wp.array([-7, -7, -7], dtype=wp.int32, device=validation_device)
        for leading, expected in (
            ([0, 257, 1, 0], 1),
            ([-1, 2, 0, 1], 0),
            ([513, 0, 0, 0], 1),
            ([1, 513, 0, 0], 0),
            ([1, 0, 512, 0], 1),
            ([-(2**31), 2, 0, 0], 0),
            ([2**31 - 1, 2**31 - 1, 1, 0], 0),
            ([0, 0, 0, 0], 1),
        ):
            values = np.pad(np.asarray(leading, dtype=np.int32), (0, groups - 4))
            arrays["seglen"].assign(values)
            for index, (call, arguments) in enumerate(self._calls(arrays)):
                assert (
                    call(*arguments, validation=validation[index : index + 1]) is None
                )
            np.testing.assert_array_equal(validation.numpy(), [expected] * 3)
            np.testing.assert_array_equal(arrays["seglen"].numpy(), values)
            if expected:
                self._assert_results(arrays, values, dtype)

    @pytest.mark.parametrize("dtype", [wp.float32, wp.bfloat16])
    @pytest.mark.parametrize(
        "capacity,values,expected",
        [
            (0, [0, 0], 1),
            (0, [1, 0], 0),
            (0, [-1, 0], 0),
            (0, [-(2**31), 2**31 - 1], 0),
            (0, [], 1),
            (7, [], 1),
        ],
    )
    def test_empty_calls_overwrite_status(
        self, validation_device, dtype, capacity, values, expected
    ):
        """Capacity and group early returns must still write the current validity."""
        arrays = self._buffers(validation_device, dtype, capacity, len(values))
        arrays["seglen"].assign(np.asarray(values, dtype=np.int32))
        validation = wp.array([-7], dtype=wp.int32, device=validation_device)
        for call, arguments in self._calls(arrays):
            validation.fill_(-7)
            call(*arguments, validation=validation)
            np.testing.assert_array_equal(validation.numpy(), [expected])
        if expected:
            self._assert_results(arrays, values, dtype)

    @pytest.mark.parametrize(
        "invalid,match",
        [
            ("not_warp", "Warp array"),
            ("rank", "1D|shape"),
            ("shape", "shape"),
            ("float", "dtype"),
            ("int64", "dtype"),
            ("strided", "contiguous"),
        ],
    )
    def test_status_metadata_is_rejected(self, validation_device, invalid, match):
        """All entry points reject status buffers that cannot hold one contiguous int32."""
        arrays = self._buffers(validation_device)
        storage = wp.zeros(2, dtype=wp.int32, device=validation_device)
        if invalid == "not_warp":
            validation = np.zeros(1, dtype=np.int32)
        elif invalid == "rank":
            validation = wp.zeros((1, 1), dtype=wp.int32, device=validation_device)
        elif invalid == "shape":
            validation = storage
        elif invalid in ("float", "int64"):
            dtype = wp.float32 if invalid == "float" else wp.int64
            validation = wp.zeros(1, dtype=dtype, device=validation_device)
        else:
            validation = wp.array(
                ptr=storage.ptr,
                shape=(1,),
                strides=(8,),
                dtype=wp.int32,
                device=validation_device,
                copy=False,
            )
        for call, arguments in self._calls(arrays):
            with pytest.raises(ValueError, match=match):
                call(*arguments, validation=validation)

    @pytest.mark.gpu
    def test_status_must_share_the_operand_device(self, device):
        """Reject a host status buffer for CUDA operands before launching work."""
        arrays = self._buffers(device)
        validation = wp.zeros(1, dtype=wp.int32, device="cpu")
        for call, arguments in self._calls(arrays):
            with pytest.raises(ValueError, match="device"):
                call(*arguments, validation=validation)

    @pytest.mark.parametrize(
        "operation,source",
        [
            (0, "a"),
            (0, "b"),
            (0, "seglen"),
            (0, "out"),
            (1, "a"),
            (1, "b"),
            (1, "seglen"),
            (1, "grad_out"),
            (1, "grad_a"),
            (1, "grad_b"),
            (2, "a"),
            (2, "seglen"),
            (2, "grad_out"),
            (2, "grad_b_only"),
        ],
    )
    def test_status_cannot_alias_any_operand(
        self, validation_device, operation, source
    ):
        """Reject interior int32 views of inputs or outputs, including float storage."""
        arrays = self._buffers(validation_device)
        validation = wp.array(
            ptr=arrays[source].ptr + 4,
            shape=(1,),
            dtype=wp.int32,
            device=validation_device,
            copy=False,
        )
        call, arguments = self._calls(arrays)[operation]
        with pytest.raises(ValueError, match="overlap"):
            call(*arguments, validation=validation)

    def test_status_cannot_occupy_a_strided_input_gap(self, validation_device):
        """The entire strided input span is excluded from status storage."""
        arrays = self._buffers(validation_device)
        storage = wp.ones((14, 3), device=validation_device)
        arrays["a"] = storage[::2]
        validation = wp.array(
            ptr=storage.ptr + 3 * 4,
            shape=(1,),
            dtype=wp.int32,
            device=validation_device,
            copy=False,
        )
        for call, arguments in self._calls(arrays):
            with pytest.raises(ValueError, match="overlap"):
                call(*arguments, validation=validation)

    def test_checked_calls_still_require_raw_int32_lengths(self, validation_device):
        """Opting into status reporting must not widen the raw length dtype contract."""
        arrays = self._buffers(validation_device)
        arrays["seglen"] = wp.zeros(4, dtype=wp.int64, device=validation_device)
        validation = wp.zeros(1, dtype=wp.int32, device=validation_device)
        for call, arguments in self._calls(arrays):
            with pytest.raises(ValueError, match="int32"):
                call(*arguments, validation=validation)

    def test_status_does_not_require_host_readback(
        self, validation_device, monkeypatch
    ):
        """Public checked calls enqueue work without NumPy reads or host synchronization."""
        arrays = self._buffers(validation_device)
        values = np.array([0, 2, 1, 0], dtype=np.int32)
        arrays["seglen"].assign(values)
        validation = wp.array([-7, -7, -7], dtype=wp.int32, device=validation_device)
        calls = self._calls(arrays)
        for index, (call, arguments) in enumerate(calls):
            call(*arguments, validation=validation[index : index + 1])
        validation.fill_(-7)

        def reject_readback(*args, **kwargs):
            """Fail only when the real call attempts an explicitly forbidden operation."""
            pytest.fail("checked calls must not read back or synchronize device data")

        with monkeypatch.context() as patch:
            patch.setattr(wp.array, "numpy", reject_readback)
            for name in ("synchronize", "synchronize_device", "synchronize_stream"):
                patch.setattr(wp, name, reject_readback)
            for index, (call, arguments) in enumerate(calls):
                call(*arguments, validation=validation[index : index + 1])
        np.testing.assert_array_equal(validation.numpy(), [1, 1, 1])
        self._assert_results(arrays, values, wp.float32)

    @pytest.mark.gpu
    @pytest.mark.parametrize("dtype", [wp.float32, wp.bfloat16])
    @pytest.mark.parametrize("groups", [4, 129, 1025])
    def test_graph_replay_and_device_status_consumer(self, device, dtype, groups):
        """A later kernel on a non-default stream observes each replay's live status."""
        with wp.ScopedStream(wp.Stream(device)):
            arrays = self._buffers(device, dtype, capacity=513, groups=groups)
            validation = wp.array([-7, -7, -7], dtype=wp.int32, device=device)
            observed = wp.empty(3, dtype=wp.int32, device=device)
            calls = self._calls(arrays)
            statuses = [validation[index : index + 1] for index in range(3)]
            for (call, arguments), status in zip(calls, statuses, strict=True):
                call(*arguments, validation=status)
            wp.launch(
                _consume_segmm_validation,
                dim=3,
                inputs=[validation, observed],
                device=device,
            )
            with wp.ScopedCapture(device=device) as capture:
                for (call, arguments), status in zip(calls, statuses, strict=True):
                    call(*arguments, validation=status)
                wp.launch(
                    _consume_segmm_validation,
                    dim=3,
                    inputs=[validation, observed],
                    device=device,
                )
            for state, expected in ((0, 1), (-1, 0), (2, 1), (8, 0), (5, 1)):
                wp.launch(
                    _write_lengths,
                    dim=groups,
                    inputs=[arrays["seglen"], state],
                    device=device,
                )
                wp.capture_launch(capture.graph)
                np.testing.assert_array_equal(observed.numpy(), [expected] * 3)
                if expected:
                    values = np.pad(
                        np.array([state, 0, 5 - state], dtype=np.int32),
                        (0, groups - 3),
                    )
                    self._assert_results(arrays, values, dtype)


@wp.kernel
def _consume_segmm_validation(
    validation: wp.array(dtype=wp.int32), observed: wp.array(dtype=wp.int32)
):
    """Read validation from a later launch on the current stream."""
    index = wp.tid()
    observed[index] = validation[index]
