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

"""CPU and CUDA integration tests for JAX segment_mm and higher-order autodiff."""

import importlib
import re
from functools import partial

import numpy as np
import pytest
import warp as wp

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from nvalchemiops.jax.segment_mm import _row_offsets, segment_mm  # noqa: E402


@pytest.fixture(
    params=["cpu", pytest.param("gpu", marks=pytest.mark.gpu)], autouse=True
)
def execution_device(request):
    """Exercise small integration cases on CPU and each available CUDA backend."""
    devices = (
        jax.devices("cpu")
        if request.param == "cpu"
        else [device for device in jax.devices() if device.platform == "gpu"]
    )
    if not devices:
        pytest.skip("Requires CUDA JAX")
    with jax.default_device(devices[0]):
        yield devices[0]


def _inputs(dtype=jnp.float32, capacity=19, groups=4):
    """Generate native-dtype, independent operands and a cotangent."""
    rng = np.random.default_rng(724)
    a = jnp.asarray(rng.normal(size=(capacity, 7)), dtype=dtype)
    b = jnp.asarray(rng.normal(size=(groups, 7, 5)), dtype=dtype)
    g = jnp.asarray(rng.normal(size=(capacity, 5)), dtype=dtype)
    return a, b, g


def _reference(a, b, seglen):
    """Use masked dense products, independent of Warp's segmented algorithm."""
    rows = jnp.arange(a.shape[0])
    offset = jnp.int32(0)
    out = jnp.zeros((a.shape[0], b.shape[2]), dtype=a.dtype)
    for group in range(b.shape[0]):
        mask = (rows >= offset) & (rows < offset + seglen[group])
        features = jnp.where(mask[:, None], a, 0)
        product = jnp.matmul(features, b[group], precision=jax.lax.Precision.HIGHEST)
        out = out + jnp.where(mask[:, None], product, 0)
        offset = offset + seglen[group]
    return out


def _assert_close(actual, expected, *, tolerance=None):
    """Check dtype and numerical agreement at the computation's precision."""
    assert actual.dtype == expected.dtype
    if tolerance is None:
        tolerance = 3e-2 if actual.dtype == jnp.bfloat16 else 2e-5
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float32),
        np.asarray(expected, dtype=np.float32),
        atol=tolerance,
        rtol=tolerance,
    )


class TestSegmentMM:
    @pytest.mark.parametrize(
        "groups", [1, 31, 32, 33, 64, 65, 128, 129, 256, 257, 512, 513, 1023, 1024]
    )
    def test_callback_chunk_prefix_matches_jax_schedule(self, execution_device, groups):
        """The callback scan matches bounded JAX prefixes across live lengths and buckets."""
        from nvalchemiops.batch_utils import _segment_chunk_offsets

        module = importlib.import_module("nvalchemiops.jax.segment_mm")
        capacity, chunk_rows = 131072, 1024
        extra_slots = (capacity - 1) // chunk_rows
        device = "cuda:0" if execution_device.platform == "gpu" else "cpu"
        rows = wp.empty(groups + 1, dtype=wp.int32, device=device)
        chunks = wp.array(np.full(groups + 1, -99, np.int32), device=device)
        bucket = max(32, 1 << (groups - 1).bit_length())
        schedule = jax.jit(
            lambda ll: module._row_schedule(ll, capacity, chunk_rows, True)
        )
        cases = (
            ([100000], [0], [2147483647], [1], [0])
            if groups == 1
            else (
                [25000] * 4,
                [0] * 4,
                [-2147483648, 513, 2147483647, 2147483647],
                [0, 1, 256, 512],
                [0] * 4,
            )
        )
        for values in cases:
            lengths = jnp.pad(jnp.asarray(values, jnp.int32), (0, groups - len(values)))
            expected_rows, expected_chunks = schedule(lengths)
            rows.assign(np.asarray(expected_rows))
            wp.launch_tiled(
                _segment_chunk_offsets[bucket],
                dim=(1,),
                inputs=[rows, chunk_rows, extra_slots, chunks],
                block_dim=min(bucket, 128),
                device=device,
            )
            np.testing.assert_array_equal(chunks.numpy(), np.asarray(expected_chunks))
            assert int(chunks.numpy()[-1]) <= extra_slots

    @pytest.mark.parametrize("groups,static_rows", [(4, 1056), (129, 1024)])
    def test_adaptive_chunk_schedule(self, groups, static_rows):
        """One compiled schedule recomputes work from active device row counts."""
        module = importlib.import_module("nvalchemiops.jax.segment_mm")
        lengths = jnp.pad(jnp.array([25000] * 4, jnp.int32), (0, groups - 4))
        executable = (
            jax.jit(lambda ll: module._row_schedule(ll, 131072, static_rows, True))
            .lower(lengths)
            .compile()
        )
        for values, extras in (([25000] * 4, [31] * 4), ([0] * 4, [0] * 4)):
            current = jnp.pad(jnp.array(values, jnp.int32), (0, groups - 4))
            _, actual = executable(current)
            expected = np.pad(
                np.concatenate(([0], np.cumsum(extras))), (0, groups - 4), mode="edge"
            )
            np.testing.assert_array_equal(np.asarray(actual), expected)

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
    def test_adaptive_chunks_through_second_derivatives(self, dtype):
        """Device-selected chunk sizes retain the complete differentiated product."""
        a = jnp.full((40000, 2), 0.125, dtype)
        b = jnp.full((4, 2, 3), 0.25, dtype)
        tangent = (jnp.full_like(a, 0.03125), jnp.full_like(b, 0.03125))

        def evaluate(fn, a, b, lengths):
            def energy(a, b):
                out = fn(a, b, lengths)
                return jnp.sum(out.astype(jnp.float32) ** 2)

            return jax.jvp(jax.grad(energy, (0, 1)), (a, b), tangent)

        initial = jnp.array([1000, 2500, 0, 127], jnp.int32)
        executable = (
            jax.jit(lambda a, b, ll: evaluate(segment_mm, a, b, ll))
            .lower(a, b, initial)
            .compile()
        )
        for values in (
            [1000, 2500, 0, 127],
            [0, 0, 0, 0],
            [10000, 10000, 10000, 10000],
        ):
            lengths = jnp.array(values, jnp.int32)
            actual = executable(a, b, lengths)
            expected = evaluate(_reference, a, b, lengths)
            for value, ref in zip(
                jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
            ):
                _assert_close(value, ref)

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
    @pytest.mark.parametrize("compiled", [False, True])
    def test_forward_and_vjp(self, dtype, compiled):
        """Forward and both operand VJPs use the correct native dtype."""
        a, b, g = _inputs(dtype)
        lengths = jnp.array([3, 0, 9, 2], dtype=jnp.int32)
        before = [np.array(x) for x in (a, b, lengths)]

        def evaluate(a, b, lengths, g):
            out, pullback = jax.vjp(segment_mm, a, b, lengths)
            ga, gb, _ = pullback(g)
            return out, ga, gb

        evaluate = jax.jit(evaluate) if compiled else evaluate
        actual = evaluate(a, b, lengths, g)
        expected, pullback = jax.vjp(_reference, a, b, lengths)
        ga, gb, _ = pullback(g)
        for value, ref in zip(actual, (expected, ga, gb), strict=True):
            _assert_close(value, ref)
        for value, original in zip((a, b, lengths), before, strict=True):
            np.testing.assert_array_equal(np.asarray(value), original)

    @pytest.mark.parametrize("operand", [0, 1])
    def test_only_one_differentiable_operand(self, operand):
        """Reverse mode works when only one floating operand is active."""
        a, b, g = _inputs()
        lengths = jnp.array([0, 2, 8, 9], dtype=jnp.int32)

        def loss(fn, a, b):
            return jnp.sum(fn(a, b, lengths) * g)

        actual = jax.jit(jax.grad(lambda a, b: loss(segment_mm, a, b), operand))(a, b)
        expected = jax.grad(lambda a, b: loss(_reference, a, b), operand)(a, b)
        _assert_close(actual, expected)

    def test_bf16_input_vjp_does_not_copy_transposed_weights(self):
        """Compiled BF16 dA consumes original B without a full-B conversion."""
        a, b, g = _inputs(jnp.bfloat16)
        lengths = jnp.array([3, 0, 9, 2], jnp.int32)

        def input_vjp(a, b, lengths, g):
            return jax.vjp(lambda a: segment_mm(a, b, lengths), a)[1](g)[0]

        executable = jax.jit(input_vjp).lower(a, b, lengths, g).compile()
        expected = jax.vjp(lambda a: _reference(a, b, lengths), a)[1](g)[0]
        _assert_close(executable(a, b, lengths, g), expected)

        # Inspect optimized HLO, including fusion bodies: a logical transpose
        # or physical layout copy of either full-B shape violates the contract.
        weight_shapes = "|".join(
            ",".join(map(str, shape))
            for shape in (b.shape, (b.shape[0], b.shape[2], b.shape[1]))
        )
        conversions = re.findall(
            rf"(?:f32|bf16)\[(?:{weight_shapes})\](?:\{{[^}}]*\}})? "
            r"(?:transpose|copy)\([^\n]*",
            executable.as_text(),
        )
        assert not conversions, "Full-B transpose/layout conversion: " + str(
            conversions
        )

    def test_runtime_lengths_and_fresh_arrays(self):
        """One executable replays with new addresses, values, and device lengths."""
        a, b, g = _inputs()
        lengths = jnp.array([3, 0, 9, 2], dtype=jnp.int32)
        executable = (
            jax.jit(
                lambda a, b, lengths, g: (
                    segment_mm(a, b, lengths),
                    jax.vjp(lambda a, b: segment_mm(a, b, lengths), a, b)[1](g),
                )
            )
            .lower(a, b, lengths, g)
            .compile()
        )
        for index, values in enumerate(([3, 0, 9, 2], [0, 5, 1, 13], [0, 0, 0, 0])):
            aa, bb, gg = a + index, b - index, g * (index + 1)
            ll = jnp.array(values, dtype=jnp.int32)
            actual, (ga, gb) = executable(aa, bb, ll, gg)
            expected, pullback = jax.vjp(lambda a, b: _reference(a, b, ll), aa, bb)
            ref_ga, ref_gb = pullback(gg)
            for value, ref in zip(
                (actual, ga, gb), (expected, ref_ga, ref_gb), strict=True
            ):
                _assert_close(value, ref)

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
    @pytest.mark.parametrize("compiled", [False, True])
    def test_int64_lengths_forward_and_derivatives(self, dtype, compiled):
        """Wide runtime lengths preserve forward, both VJPs, and joint JVPs."""
        with jax.enable_x64(True):
            a, b, g = _inputs(dtype)
            lengths = jnp.array([3, 0, 9, 2], jnp.int64)
            tangents = (jnp.full_like(a, 0.125), jnp.full_like(b, -0.125))

            def evaluate(fn, a, b, lengths):
                def operation(aa, bb):
                    return fn(aa, bb, lengths)

                out, pullback = jax.vjp(operation, a, b)
                return out, *pullback(g), jax.jvp(operation, (a, b), tangents)[1]

            def operation(a, b, lengths):
                return evaluate(segment_mm, a, b, lengths)

            execute = (
                jax.jit(operation).lower(a, b, lengths).compile()
                if compiled
                else operation
            )
            for values in ([3, 0, 9, 2], [0, 5, 1, 13], [0, 0, 0, 0]):
                current = jnp.array(values, jnp.int64)
                actual = execute(a, b, current)
                expected = evaluate(_reference, a, b, current)
                for value, reference in zip(actual, expected, strict=True):
                    _assert_close(value, reference)

    def test_int64_large_capacity_prefix(self):
        """The saturating prefix keeps int32 offsets with extreme int64 inputs."""
        capacity = 2**30 + 17
        with jax.enable_x64(True):
            lengths = jnp.array([13, capacity - 5, 2**63 - 1, -(2**63)], jnp.int64)
            offsets = jax.jit(lambda values: _row_offsets(values, capacity))(lengths)
            assert offsets.dtype == jnp.int32
            np.testing.assert_array_equal(
                np.asarray(offsets), [0, 13, capacity, capacity, capacity]
            )

    def test_jit_scan_and_gradient(self):
        """Scan residuals retain runtime segment lengths for reverse mode."""
        a, b, g = _inputs()
        lengths = jnp.array([[3, 0, 9, 2], [0, 5, 1, 13], [1, 1, 1, 1]], jnp.int32)

        def run(fn, a, b, lengths):
            def step(carry, ll):
                out = fn(a + carry, b, ll)
                return carry + jnp.float32(0.25), out

            return jax.lax.scan(step, jnp.float32(0), lengths)[1]

        actual = jax.jit(lambda a, b, ll: run(segment_mm, a, b, ll))(a, b, lengths)
        expected = jax.jit(lambda a, b, ll: run(_reference, a, b, ll))(a, b, lengths)
        _assert_close(actual, expected)
        actual_grad = jax.jit(
            jax.grad(lambda a, b: jnp.sum(run(segment_mm, a, b, lengths) * g), (0, 1))
        )(a, b)
        expected_grad = jax.grad(
            lambda a, b: jnp.sum(run(_reference, a, b, lengths) * g), (0, 1)
        )(a, b)
        for value, ref in zip(actual_grad, expected_grad, strict=True):
            _assert_close(value, ref)

    @pytest.mark.parametrize("capacity,groups", [(0, 4), (0, 0), (19, 0), (19, 4)])
    def test_empty(self, capacity, groups):
        """Empty capacity, missing groups, and zero lengths have zero gradients."""
        a, b, g = _inputs(capacity=capacity, groups=groups)
        lengths = jnp.zeros(groups, dtype=jnp.int32)
        out, pullback = jax.vjp(jax.jit(segment_mm), a, b, lengths)
        ga, gb, _ = pullback(g)
        for value in (out, ga, gb):
            np.testing.assert_array_equal(np.asarray(value), 0)

    @pytest.mark.parametrize("compiled", [False, True])
    def test_empty_large_channels_need_no_gradient_workspace(self, compiled):
        """Empty BF16 gradients do not require a hypothetical FP32 workspace."""
        channels = 23171
        a = jnp.empty((0, channels), jnp.bfloat16)
        b = jnp.empty((0, channels, channels), jnp.bfloat16)
        lengths = jnp.empty((0,), jnp.int32)

        def evaluate(a, b, lengths):
            out, pullback = jax.vjp(lambda a, b: segment_mm(a, b, lengths), a, b)
            return out, *pullback(jnp.zeros_like(out))

        out, grad_a, grad_b = (jax.jit(evaluate) if compiled else evaluate)(
            a, b, lengths
        )
        assert out.shape == grad_a.shape == a.shape
        assert grad_b.shape == b.shape
        assert out.dtype == grad_a.dtype == grad_b.dtype == jnp.bfloat16

    def test_nan_padding_and_transposed_inputs(self):
        """Padding is zero even for NaNs; JAX transpose layouts are accepted."""
        a, b, g = _inputs()
        a = a.at[14:].set(jnp.nan).T.copy().T
        b = jnp.swapaxes(jnp.swapaxes(b, 1, 2).copy(), 1, 2)
        g = g.at[14:].set(jnp.nan)
        lengths = jnp.array([3, 0, 9, 2], jnp.int32)
        out, pullback = jax.vjp(segment_mm, a, b, lengths)
        ga, gb, _ = pullback(g)
        expected, ref_pullback = jax.vjp(_reference, a, b, lengths)
        ref_ga, ref_gb, _ = ref_pullback(g)
        for value, ref in zip((out, ga, gb), (expected, ref_ga, ref_gb), strict=True):
            _assert_close(value, ref)
        assert np.isfinite(np.asarray(gb)).all()

    @pytest.mark.parametrize(
        "bad_input,match",
        [
            ("rank", "2D"),
            ("dtype", "float32 or bfloat16"),
            ("mixed_dtype", "same float32 or bfloat16"),
            ("length_dtype", "int32"),
            ("channels", "channel"),
            ("groups", "one length"),
            ("zero_channels", "positive"),
        ],
    )
    def test_validation(self, bad_input, match):
        """Metadata errors are rejected before issuing a Warp operation."""
        a, b, _ = _inputs()
        lengths = jnp.array([3, 0, 9, 2], jnp.int32)
        if bad_input == "rank":
            a = a[0]
        elif bad_input == "dtype":
            b = b.astype(jnp.float16)
        elif bad_input == "mixed_dtype":
            b = b.astype(jnp.bfloat16)
        elif bad_input == "length_dtype":
            lengths = lengths.astype(jnp.float32)
        elif bad_input == "channels":
            b = b[:, :3]
        elif bad_input == "groups":
            lengths = lengths[:2]
        elif bad_input == "zero_channels":
            a, b = a[:, :0], b[:, :0]
        with pytest.raises(ValueError, match=match):
            jax.jit(segment_mm)(a, b, lengths)

    def test_cpu_placement_overrides_default_device(self):
        """Committed CPU inputs stay on CPU even when the default device is CUDA."""
        cpu = jax.devices("cpu")[0]
        a, b, _ = _inputs()
        lengths = jnp.array([3, 0, 9, 2], jnp.int32)
        a, b, lengths = (jax.device_put(x, cpu) for x in (a, b, lengths))
        actual = segment_mm(a, b, lengths)
        assert actual.devices() == {cpu}
        _assert_close(actual, _reference(a, b, lengths))

    @pytest.mark.parametrize("groups", [4, 128, 129])
    def test_bounded_lengths_without_x64(self, groups):
        """Negative and overflowing lengths remain bounded with 32-bit JAX enabled."""
        with jax.enable_x64(False):
            a, b, g = _inputs(groups=groups)
            lengths = jnp.array(
                [-2147483648, 3, 2147483647, 2147483647] + [0] * (groups - 4),
                jnp.int32,
            )

            def evaluate(a, b, lengths, g):
                out, pullback = jax.vjp(lambda a, b: segment_mm(a, b, lengths), a, b)
                return out, *pullback(g)

            actual = jax.jit(evaluate)(a, b, lengths, g)
            expected, pullback = jax.vjp(
                lambda a, b: _reference(a, b[:4], jnp.array([0, 3, 16, 0], jnp.int32)),
                a,
                b,
            )
            for value, ref in zip(actual, (expected, *pullback(g)), strict=True):
                _assert_close(value, ref)

    def test_large_capacity_prefix_without_x64(self):
        """Wide hypothetical row counts saturate without allocating feature rows."""
        capacity = 2**30 + 17
        with jax.enable_x64(False):
            lengths = jnp.array([13, capacity - 5, 2147483647, -7], jnp.int32)
            offsets = jax.jit(lambda values: _row_offsets(values, capacity))(lengths)
            assert offsets.dtype == jnp.int32
            np.testing.assert_array_equal(
                np.asarray(offsets), [0, 13, capacity, capacity, capacity]
            )

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
    @pytest.mark.parametrize("groups", [129, 1024, 1025])
    def test_callback_scratch_is_framework_owned(self, dtype, groups, monkeypatch):
        """Fresh forward/backward calls need no Warp allocation or native scan."""
        capacity = groups * 256 + 1
        a = jnp.full((capacity, 1), 0.5, dtype)
        b = jnp.ones((groups, 1, 1), dtype)
        g = jnp.ones((capacity, 1), dtype)

        @jax.jit
        def evaluate(a, b, lengths, g):
            out, pullback = jax.vjp(lambda a, b: segment_mm(a, b, lengths), a, b)
            return out, *pullback(g)

        lengths = jnp.array([4] + [0] * (groups - 1), jnp.int32)
        jax.block_until_ready(evaluate(a, b, lengths, g))

        def reject_warp_scratch(*args, **kwargs):
            raise AssertionError("JAX callbacks must use framework-owned scratch")

        monkeypatch.setattr(wp, "empty", reject_warp_scratch)
        monkeypatch.setattr(wp.utils, "array_scan", reject_warp_scratch)
        current_a, current_b = a + jnp.asarray(0.25, dtype), b * jnp.asarray(2, dtype)
        for active_rows in (0, 3, 1025, 0):
            current_lengths = jnp.array([active_rows] + [0] * (groups - 1), jnp.int32)
            out, grad_a, grad_b = evaluate(current_a, current_b, current_lengths, g)
            np.testing.assert_array_equal(np.asarray(out[:active_rows]), 1.5)
            np.testing.assert_array_equal(np.asarray(out[active_rows:]), 0)
            np.testing.assert_array_equal(np.asarray(grad_a[:active_rows]), 2)
            np.testing.assert_array_equal(np.asarray(grad_a[active_rows:]), 0)
            np.testing.assert_array_equal(
                np.asarray(grad_b[0]),
                np.asarray(jnp.asarray(active_rows * 0.75, dtype)),
            )
            np.testing.assert_array_equal(np.asarray(grad_b[1:]), 0)

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
    def test_large_group_chunk_replay_with_mixed_owners(self, dtype):
        """The JAX prefix fallback refreshes mixed chunk owners and excludes NaN padding."""
        capacity, groups, k, n = 4096, 1025, 3, 5
        rng = np.random.default_rng(735)
        a_host = rng.integers(-4, 5, (capacity, k)).astype(np.float32) / 8
        b_host = rng.integers(-4, 5, (groups, k, n)).astype(np.float32) / 8
        g_host = rng.integers(-4, 5, (capacity, n)).astype(np.float32) / 8
        a, b, g = (jnp.asarray(value, dtype) for value in (a_host, b_host, g_host))

        def evaluate(a, b, lengths, g):
            out, pullback = jax.vjp(lambda a, b: segment_mm(a, b, lengths), a, b)
            return out, *pullback(g)

        empty = jnp.zeros(groups, jnp.int32)
        executable = jax.jit(evaluate).lower(a, b, empty, g).compile()
        for active in ({}, {1024: 2305}, {1: 513, 511: 1025, 1024: 1}, {}):
            lengths = np.zeros(groups, np.int32)
            for group, length in active.items():
                lengths[group] = length
            used = int(lengths.sum())
            current_a, current_g = a_host.copy(), g_host.copy()
            current_a[used:] = current_g[used:] = np.nan
            actual = executable(
                jnp.asarray(current_a, dtype),
                b,
                jnp.asarray(lengths),
                jnp.asarray(current_g, dtype),
            )

            # Slice only active rows; this oracle creates no groups-by-capacity mask.
            expected = (
                np.zeros((capacity, n), np.float64),
                np.zeros((capacity, k), np.float64),
                np.zeros((groups, k, n), np.float64),
            )
            start = 0
            for group in np.flatnonzero(lengths):
                stop = start + int(lengths[group])
                aa = current_a[start:stop].astype(np.float64)
                gg = current_g[start:stop].astype(np.float64)
                bb = b_host[group].astype(np.float64)
                expected[0][start:stop] = aa @ bb
                expected[1][start:stop] = gg @ bb.T
                expected[2][group] = aa.T @ gg
                start = stop
            for value, reference in zip(actual, expected, strict=True):
                _assert_close(value, jnp.asarray(reference, dtype))

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
    @pytest.mark.parametrize("compiled", [False, True])
    def test_joint_jvp(self, dtype, compiled):
        """Both operand tangents contribute with native BF16 or FP32 semantics."""
        a, b, _ = _inputs(dtype)
        lengths = jnp.array([3, 0, 9, 2], jnp.int32)
        ta, tb = a * jnp.asarray(0.2, dtype), b * jnp.asarray(-0.3, dtype)

        def evaluate(fn, a, b, ta, tb):
            return jax.jvp(lambda x, w: fn(x, w, lengths), (a, b), (ta, tb))

        def operation(a, b, ta, tb):
            return evaluate(segment_mm, a, b, ta, tb)

        actual = (jax.jit(operation) if compiled else operation)(a, b, ta, tb)
        expected = evaluate(_reference, a, b, ta, tb)
        for value, ref in zip(actual, expected, strict=True):
            _assert_close(value, ref)

    @pytest.mark.parametrize("operand", [0, 1])
    def test_single_operand_jvp(self, operand):
        """Symbolic zero tangents for the other operand avoid spurious contributions."""
        a, b, _ = _inputs()
        lengths = jnp.array([0, 2, 8, 9], jnp.int32)

        def evaluate(fn):
            operation = (
                (lambda x: fn(x, b, lengths))
                if operand == 0
                else (lambda w: fn(a, w, lengths))
            )
            value = (a, b)[operand]
            return jax.jvp(operation, (value,), (jnp.ones_like(value),))[1]

        _assert_close(jax.jit(lambda: evaluate(segment_mm))(), evaluate(_reference))

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
    def test_jvp_over_vjp_all_inputs(self, dtype):
        """Differentiate both backward outputs through operands and the cotangent."""
        a, b, g = _inputs(dtype)
        lengths = jnp.array([3, 0, 9, 2], jnp.int32)
        tangents = tuple(
            jnp.ones_like(x) * jnp.asarray(0.125, dtype) for x in (a, b, g)
        )

        def evaluate(fn, a, b, g):
            def pullback(a, b, g):
                return jax.vjp(lambda a, b: fn(a, b, lengths), a, b)[1](g)

            return jax.jvp(pullback, (a, b, g), tangents)

        actual = jax.jit(lambda a, b, g: evaluate(segment_mm, a, b, g))(a, b, g)
        expected = evaluate(_reference, a, b, g)
        for value, ref in zip(
            jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
        ):
            _assert_close(value, ref)

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
    def test_force_loss_gradients(self, dtype):
        """A nonlinear energy yields correct force-loss gradients for both parameter sets."""
        rng = np.random.default_rng(12)
        positions = jnp.asarray(rng.normal(size=(11, 3)), jnp.float32)
        embedding = jnp.asarray(rng.normal(size=(3, 7)) * 0.2, jnp.float32)
        weights = jnp.asarray(rng.normal(size=(4, 7, 5)) * 0.2, dtype)
        target = jnp.asarray(rng.normal(size=(11, 3)) * 0.1, jnp.float32)

        def loss(fn, embedding, weights, lengths):
            def energy(positions):
                features = jnp.sin(positions @ embedding).astype(weights.dtype)
                y = fn(features, weights, lengths).astype(jnp.float32)
                return jnp.sum(jnp.sin(y) + jnp.float32(0.25) * y**2)

            forces = -jax.grad(energy)(positions)
            return jnp.sum((forces - target) ** 2)

        operation = jax.jit(
            jax.value_and_grad(lambda e, w, ll: loss(segment_mm, e, w, ll), (0, 1))
        )
        for values in ([3, 0, 5, 1], [0, 2, 0, 9], [0, 0, 0, 0]):
            ll = jnp.array(values, jnp.int32)
            actual = operation(embedding, weights, ll)
            expected = jax.value_and_grad(
                lambda e, w: loss(_reference, e, w, ll), (0, 1)
            )(embedding, weights)
            for value, ref in zip(
                jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
            ):
                _assert_close(value, ref)
            if dtype == jnp.float32 and values == [3, 0, 5, 1]:
                direction_e = jnp.asarray(rng.normal(size=embedding.shape), jnp.float32)
                direction_w = jnp.asarray(rng.normal(size=weights.shape), jnp.float32)
                direction_e = direction_e / jnp.linalg.norm(direction_e)
                direction_w = direction_w / jnp.linalg.norm(direction_w)
                epsilon = jnp.float32(1e-3)
                loss_value = jax.jit(lambda e, w: loss(segment_mm, e, w, ll))
                plus = loss_value(
                    embedding + epsilon * direction_e, weights + epsilon * direction_w
                )
                minus = loss_value(
                    embedding - epsilon * direction_e, weights - epsilon * direction_w
                )
                finite_difference = (plus - minus) / (2 * epsilon)
                directional = jnp.sum(actual[1][0] * direction_e) + jnp.sum(
                    actual[1][1] * direction_w
                )
                np.testing.assert_allclose(
                    np.asarray(directional),
                    np.asarray(finite_difference),
                    rtol=2e-3,
                    atol=3e-4,
                )

    def test_second_order_through_scan(self):
        """Second-order rules compose with scan residuals and changing segment lengths."""
        a, b, _ = _inputs()
        a, b = a * 0.1, b * 0.1
        lengths = jnp.array([[3, 0, 9, 2], [0, 5, 1, 13], [0, 0, 0, 0]], jnp.int32)

        def energy(fn, a, b):
            def step(carry, ll):
                y = fn(jnp.sin(a + carry), b, ll)
                return carry + 0.1, jnp.sum(jnp.sin(y))

            return jnp.sum(jax.lax.scan(step, jnp.float32(0), lengths)[1])

        def force_loss(fn, a, b):
            force = jax.grad(lambda a: energy(fn, a, b))(a)
            return jnp.sum(force**2)

        actual = jax.jit(jax.grad(lambda a, b: force_loss(segment_mm, a, b), (0, 1)))(
            a, b
        )
        expected = jax.grad(lambda a, b: force_loss(_reference, a, b), (0, 1))(a, b)
        for value, ref in zip(actual, expected, strict=True):
            _assert_close(value, ref)

    def test_jacobian_and_third_directional_derivative(self):
        """Batched Jacobian transforms and a third derivative retain the full algebra."""
        a = jnp.array([[0.2, -0.1], [0.3, 0.4], [0.0, 0.0]], jnp.float32)
        b = jnp.array([[[0.5], [-0.3]], [[0.2], [0.1]]], jnp.float32)
        lengths = jnp.array([2, 0], jnp.int32)
        for transform in (jax.jacfwd, jax.jacrev):
            actual = jax.jit(transform(lambda a: segment_mm(a, b, lengths)))(a)
            expected = transform(lambda a: _reference(a, b, lengths))(a)
            _assert_close(actual, expected)

        def third(fn, b):
            def energy(b):
                return jnp.sum(jnp.sin(fn(a, b, lengths)))

            def second(b):
                return jax.jvp(jax.grad(energy), (b,), (jnp.ones_like(b),))[1]

            return jax.jvp(second, (b,), (jnp.ones_like(b),))[1]

        _assert_close(jax.jit(lambda b: third(segment_mm, b))(b), third(_reference, b))


class TestGraphCachePolicy:
    """Static cache choices survive transforms without mutating other callables."""

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
    @pytest.mark.parametrize("product", ["forward", "input", "weight"])
    def test_eager_reuses_call_descriptors(self, execution_device, dtype, product):
        """Repeated eager products reuse their compiled FFI call descriptors."""
        module = importlib.import_module("nvalchemiops.jax.segment_mm")
        index = ("forward", "input", "weight").index(product)
        # A separate policy per route keeps unrelated tests out of these caches.
        limit = 101 + index
        mode = (
            wp.JaxCallableGraphMode.NONE
            if execution_device.platform == "cpu"
            else wp.JaxCallableGraphMode.WARP
        )
        callback = module._callables(str(jnp.dtype(dtype)), mode, limit)[index]
        a, b, g = _inputs(dtype)
        lengths = jnp.array([3, 0, 9, 2], jnp.int32)
        out, pullback = jax.vjp(lambda a, b: _reference(a, b, lengths), a, b)
        expected = (out, *pullback(g))[index]
        if product == "forward":
            evaluate = partial(segment_mm, a, b, lengths, graph_cache_max=limit)
        else:
            # Bind the AD leaves directly so an independently allocated
            # cotangent cannot change the graph's input-buffer addresses.
            primitive, operands = (
                (module._input_p, (g, b, lengths))
                if product == "input"
                else (module._weight_p, (a, g, lengths))
            )
            evaluate = partial(
                module._bind_product, primitive, *operands, graph_cache_max=limit
            )

        try:
            for _ in range(5):
                actual = jax.block_until_ready(evaluate())
                _assert_close(actual, expected)
                del actual
            descriptors = len(callback.call_descriptors)
            assert descriptors > 0
            assert (callback.graph_cache_size > 0) == (
                execution_device.platform == "gpu"
            )

            for _ in range(5):
                actual = jax.block_until_ready(evaluate())
                _assert_close(actual, expected)
                del actual
                assert len(callback.call_descriptors) == descriptors
        finally:
            wp.clear_jax_callable_graph_cache(callback)

    @pytest.mark.parametrize("limit", [0, -1, True, False, 1.5, "256"])
    def test_validation(self, limit):
        """Reject invalid cache bounds before tracing or entering Warp."""
        a, b, _ = _inputs()
        lengths = jnp.array([3, 0, 9, 2], jnp.int32)
        with pytest.raises(ValueError, match="graph_cache_max"):
            segment_mm(a, b, lengths, graph_cache_max=limit)

    @pytest.mark.parametrize("limit", [17, None])
    def test_transforms_preserve_policy(self, limit, monkeypatch):
        """JIT, batching, and second derivatives all reach the requested cache."""
        module = importlib.import_module("nvalchemiops.jax.segment_mm")
        original = module._callables
        used = []

        def observe(*args, **kwargs):
            callbacks = original(*args, **kwargs)
            used.extend(callbacks)
            return callbacks

        monkeypatch.setattr(module, "_callables", observe)
        a, b, _ = _inputs()
        a, b = a * 0.1, b * 0.1
        lengths = jnp.array([[3, 0, 9, 2], [0, 5, 1, 13]], jnp.int32)
        operation = partial(segment_mm, graph_cache_max=limit)

        def evaluate(fn, a, b):
            def energy(a, b):
                outputs = jax.vmap(lambda ll: fn(a, b, ll))(lengths)
                return jnp.sum(jnp.sin(outputs))

            return jax.jvp(
                jax.value_and_grad(energy, argnums=(0, 1)),
                (a, b),
                (jnp.ones_like(a), jnp.ones_like(b)),
            )

        actual = jax.jit(lambda a, b: evaluate(operation, a, b))(a, b)
        expected = evaluate(_reference, a, b)
        for value, ref in zip(
            jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
        ):
            _assert_close(value, ref)
        assert used
        assert all(callback.graph_cache_max == limit for callback in used)

    def test_callable_policies_remain_independent(self, execution_device):
        """Visiting other limits never changes or recreates an existing policy."""
        module = importlib.import_module("nvalchemiops.jax.segment_mm")
        mode = (
            wp.JaxCallableGraphMode.NONE
            if execution_device.platform == "cpu"
            else wp.JaxCallableGraphMode.WARP
        )
        first = module._callables("float32", mode, 11)
        for limit in (13, 19, 23, 29, None):
            other = module._callables("float32", mode, limit)
            assert all(callback.graph_cache_max == limit for callback in other)
        assert module._callables("float32", mode, 11) is first
        assert all(callback.graph_cache_max == 11 for callback in first)


class TestExplicitMixedPrecision:
    """Check explicit compute casts while retaining original parameter dtypes."""

    @pytest.mark.parametrize("a_dtype", [jnp.float32, jnp.bfloat16])
    def test_jit_forward_and_vjp_to_original_parameters(self, a_dtype):
        """BF16 casts preserve VJPs to the original, possibly mixed-dtype inputs."""
        a, b, g = _inputs()
        a = a.astype(a_dtype)
        g = g.astype(jnp.bfloat16)

        def evaluate(fn, a, b, lengths, g):
            def operation(a, b):
                return fn(a.astype(jnp.bfloat16), b.astype(jnp.bfloat16), lengths)

            out, pullback = jax.vjp(operation, a, b)
            return out, *pullback(g)

        operation = jax.jit(lambda a, b, ll, g: evaluate(segment_mm, a, b, ll, g))
        for values in ([3, 0, 9, 2], [0, 0, 0, 0]):
            lengths = jnp.array(values, jnp.int32)
            actual = operation(a, b, lengths, g)
            expected = evaluate(_reference, a, b, lengths, g)
            assert tuple(value.dtype for value in actual) == (
                jnp.bfloat16,
                a_dtype,
                jnp.float32,
            )
            for value, ref in zip(actual, expected, strict=True):
                # Casting a BF16 cotangent back to FP32 restores storage dtype,
                # not the precision lost in the derivative products.
                _assert_close(value, ref, tolerance=3e-2)
            active_rows = sum(values)
            np.testing.assert_array_equal(np.asarray(actual[0][active_rows:]), 0)
            np.testing.assert_array_equal(np.asarray(actual[1][active_rows:]), 0)
            for group, length in enumerate(values):
                if length == 0:
                    np.testing.assert_array_equal(np.asarray(actual[2][group]), 0)

    @pytest.mark.parametrize("a_dtype", [jnp.float32, jnp.bfloat16])
    def test_jit_joint_jvp_through_casts(self, a_dtype):
        """Both original-input tangents pass through casts into BF16 products."""
        a, b, _ = _inputs()
        a = a.astype(a_dtype)
        lengths = jnp.array([3, 0, 9, 2], jnp.int32)
        ta, tb = jnp.full_like(a, 0.125), jnp.full_like(b, -0.25)

        def evaluate(fn, a, b, ta, tb):
            return jax.jvp(
                lambda a, b: fn(
                    a.astype(jnp.bfloat16), b.astype(jnp.bfloat16), lengths
                ),
                (a, b),
                (ta, tb),
            )

        actual = jax.jit(lambda a, b, ta, tb: evaluate(segment_mm, a, b, ta, tb))(
            a, b, ta, tb
        )
        expected = evaluate(_reference, a, b, ta, tb)
        for value, ref in zip(actual, expected, strict=True):
            assert value.dtype == jnp.bfloat16
            _assert_close(value, ref)

    def test_jit_hessian_vector_product_to_fp32_parameters(self):
        """A nonlinear loss has FP32 gradients and Hessian products through BF16."""
        a, b, _ = _inputs()
        a, b = a * 0.1, b * 0.1
        lengths = jnp.array([3, 0, 9, 2], jnp.int32)
        ta, tb = jnp.full_like(a, 0.125), jnp.full_like(b, -0.25)

        def evaluate(fn, a, b, ta, tb):
            def loss(a, b):
                out = fn(a.astype(jnp.bfloat16), b.astype(jnp.bfloat16), lengths)
                out = out.astype(jnp.float32)
                return jnp.sum(jnp.sin(out) + jnp.float32(0.25) * out**2)

            return jax.jvp(jax.grad(loss, argnums=(0, 1)), (a, b), (ta, tb))

        actual = jax.jit(lambda a, b, ta, tb: evaluate(segment_mm, a, b, ta, tb))(
            a, b, ta, tb
        )
        expected = evaluate(_reference, a, b, ta, tb)
        for value, ref in zip(
            jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
        ):
            assert value.dtype == jnp.float32
            _assert_close(value, ref, tolerance=3e-2)
