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

"""CUDA integration tests for JAX segment_mm and higher-order autodiff."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

if not any(device.platform == "gpu" for device in jax.devices()):
    pytest.skip("segment_mm requires CUDA JAX", allow_module_level=True)

from nvalchemiops.jax.segment_mm import segment_mm  # noqa: E402

pytestmark = pytest.mark.gpu


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

    def test_cpu_rejected(self):
        """The public CUDA-only API rejects concrete CPU inputs."""
        cpu = jax.devices("cpu")[0]
        a, b, _ = _inputs()
        lengths = jnp.array([3, 0, 9, 2], jnp.int32)
        with pytest.raises(RuntimeError, match="CUDA"):
            segment_mm(*(jax.device_put(x, cpu) for x in (a, b, lengths)))

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
