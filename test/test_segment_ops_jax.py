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

"""Tests for nvalchemiops.jax.segment_ops (PR 3 JAX bindings).

Mirrors the PyTorch suite: forward parity, first- and second-order autograd
checked against finite differences via ``jax.test_util.check_grads``, plus
a cross-check against the PyTorch binding on shared inputs.

All gradient checks run in float64.  The bindings target CUDA (``jax_callable``
is CUDA-only), so the whole file is skipped on hosts without a CUDA-capable
JAX device.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

# jax_callable requires a CUDA backend.
if not any(d.platform == "gpu" for d in jax.devices()):
    pytest.skip(
        "JAX segment-op bindings require a CUDA-capable JAX device",
        allow_module_level=True,
    )

jax.config.update("jax_enable_x64", True)

from jax.test_util import check_grads  # noqa: E402

from nvalchemiops.jax import segment_ops as so  # noqa: E402

# Optional dependency: ``TestTorchParity`` cross-checks the JAX bindings
# against the torch bindings on shared inputs.  Skip that class cleanly when
# torch is not installed.
try:
    import torch  # noqa: E402

    HAS_TORCH = True
except ModuleNotFoundError:
    HAS_TORCH = False
    torch = None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N, M = 12, 4

# Representative ``(N, M)`` shapes for ``TestShapeVariety`` — kept in sync
# with the torch suite so both bindings exercise the same regimes: a small
# balanced baseline, a medium case with longer segments, a large case with
# few long segments, and the singleton edge case where N == M.
_SHAPES = [
    (12, 4),
    (64, 8),
    (100, 5),
    (24, 24),
]


def _make_idx(seed: int = 0, n: int = N, m: int = M) -> jnp.ndarray:
    """Sorted int32 idx with every segment non-empty (required for mean/rms_norm)."""
    rng = np.random.default_rng(seed)
    base = np.arange(m, dtype=np.int32)
    extra = rng.integers(0, m, n - m).astype(np.int32)
    return jnp.array(np.sort(np.concatenate([base, extra])))


def _randn(shape, *, seed: int, dtype=jnp.float64, offset: float = 0.0):
    rng = np.random.default_rng(seed)
    return jnp.array(rng.standard_normal(shape) + offset, dtype=dtype)


# ---------------------------------------------------------------------------
# segmented_sum
# ---------------------------------------------------------------------------


class TestSegmentedSum:
    def test_forward_scalar(self):
        idx = _make_idx(seed=1)
        x = _randn((N,), seed=2, dtype=jnp.float32)
        out = so.segmented_sum(x, idx, M)
        ref = np.zeros(M, dtype=np.float32)
        np.add.at(ref, np.asarray(idx), np.asarray(x))
        np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-5)

    def test_forward_vec3(self):
        idx = _make_idx(seed=3)
        x = _randn((N, 3), seed=4, dtype=jnp.float32)
        out = so.segmented_sum(x, idx, M)
        ref = np.zeros((M, 3), dtype=np.float32)
        np.add.at(ref, np.asarray(idx), np.asarray(x))
        np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-5)

    def test_grads_scalar(self):
        idx = _make_idx(seed=5)
        x = _randn((N,), seed=6)
        check_grads(lambda v: so.segmented_sum(v, idx, M), (x,), order=2, modes=["rev"])

    def test_grads_vec3(self):
        idx = _make_idx(seed=7)
        x = _randn((N, 3), seed=8)
        check_grads(lambda v: so.segmented_sum(v, idx, M), (x,), order=2, modes=["rev"])


# ---------------------------------------------------------------------------
# segmented_dot
# ---------------------------------------------------------------------------


class TestSegmentedDot:
    def test_forward(self):
        idx = _make_idx(seed=10)
        x = _randn((N, 3), seed=11, dtype=jnp.float32)
        y = _randn((N, 3), seed=12, dtype=jnp.float32)
        out = so.segmented_dot(x, y, idx, M)
        ref = np.zeros(M, dtype=np.float32)
        np.add.at(ref, np.asarray(idx), (np.asarray(x) * np.asarray(y)).sum(axis=1))
        np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-4)

    def test_grads(self):
        idx = _make_idx(seed=13)
        x = _randn((N, 3), seed=14)
        y = _randn((N, 3), seed=15)
        check_grads(
            lambda a, b: so.segmented_dot(a, b, idx, M), (x, y), order=2, modes=["rev"]
        )


# ---------------------------------------------------------------------------
# segmented_mul
# ---------------------------------------------------------------------------


class TestSegmentedMul:
    def test_forward(self):
        idx = _make_idx(seed=20)
        x = _randn((N, 3), seed=21, dtype=jnp.float32)
        y = _randn((M,), seed=22, dtype=jnp.float32)
        out = so.segmented_mul(x, y, idx, M)
        ref = np.asarray(x) * np.asarray(y)[np.asarray(idx), None]
        np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-5)

    def test_grads(self):
        idx = _make_idx(seed=23)
        x = _randn((N, 3), seed=24)
        y = _randn((M,), seed=25)
        check_grads(
            lambda a, b: so.segmented_mul(a, b, idx, M), (x, y), order=2, modes=["rev"]
        )


# ---------------------------------------------------------------------------
# segmented_mean
# ---------------------------------------------------------------------------


class TestSegmentedMean:
    def test_forward(self):
        idx = _make_idx(seed=30)
        x = _randn((N, 3), seed=31, dtype=jnp.float32)
        out = so.segmented_mean(x, idx, M)
        idx_np = np.asarray(idx)
        counts = np.bincount(idx_np, minlength=M).astype(np.float32)
        sums = np.zeros((M, 3), dtype=np.float32)
        np.add.at(sums, idx_np, np.asarray(x))
        ref = sums / counts[:, None]
        np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-5)

    def test_grads_scalar(self):
        idx = _make_idx(seed=32)
        x = _randn((N,), seed=33)
        check_grads(
            lambda v: so.segmented_mean(v, idx, M), (x,), order=2, modes=["rev"]
        )

    def test_grads_vec3(self):
        idx = _make_idx(seed=34)
        x = _randn((N, 3), seed=35)
        check_grads(
            lambda v: so.segmented_mean(v, idx, M), (x,), order=2, modes=["rev"]
        )


# ---------------------------------------------------------------------------
# segmented_rms_norm
# ---------------------------------------------------------------------------


class TestSegmentedRmsNorm:
    def test_forward(self):
        idx = _make_idx(seed=40)
        x = _randn((N, 3), seed=41, dtype=jnp.float64)
        out = so.segmented_rms_norm(x, idx, M)
        idx_np = np.asarray(idx)
        x_np = np.asarray(x)
        counts = np.bincount(idx_np, minlength=M).astype(np.float64)
        sum_sq = np.zeros(M, dtype=np.float64)
        np.add.at(sum_sq, idx_np, (x_np * x_np).sum(axis=1))
        ref = np.sqrt(sum_sq / np.maximum(counts, 1))
        np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-10)

    def test_grads(self):
        idx = _make_idx(seed=42)
        # Bias x away from 0 to keep inv_norm well-conditioned.
        x = _randn((N, 3), seed=43, offset=2.0)
        check_grads(
            lambda v: so.segmented_rms_norm(v, idx, M), (x,), order=2, modes=["rev"]
        )


# ---------------------------------------------------------------------------
# segmented_matvec
# ---------------------------------------------------------------------------


class TestSegmentedMatvec:
    def test_forward(self):
        idx = _make_idx(seed=50)
        v = _randn((N, 3), seed=51, dtype=jnp.float32)
        m = _randn((M, 3, 3), seed=52, dtype=jnp.float32)
        out = so.segmented_matvec(v, m, idx, M)
        idx_np = np.asarray(idx)
        v_np = np.asarray(v)
        m_np = np.asarray(m)
        ref = np.stack([m_np[idx_np[i]].T @ v_np[i] for i in range(N)], axis=0)
        np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-5, atol=1e-6)

    def test_grads(self):
        idx = _make_idx(seed=53)
        v = _randn((N, 3), seed=54)
        m = _randn((M, 3, 3), seed=55)
        check_grads(
            lambda a, b: so.segmented_matvec(a, b, idx, M),
            (v, m),
            order=2,
            modes=["rev"],
        )


# ---------------------------------------------------------------------------
# Cross-check vs the PyTorch binding on shared inputs
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is not installed")
class TestTorchParity:
    """Identical inputs through the JAX and PyTorch bindings should produce
    the same values and the same gradients."""

    def test_sum_vec3(self):
        from nvalchemiops.torch.segment_ops import segmented_sum as t_sum

        idx_np = np.asarray(_make_idx(seed=60))
        x_np = np.asarray(_randn((N, 3), seed=61))

        # JAX
        x_j = jnp.array(x_np)
        idx_j = jnp.array(idx_np.astype(np.int32))
        out_j = so.segmented_sum(x_j, idx_j, M)
        grad_j = jax.grad(lambda v: so.segmented_sum(v, idx_j, M).sum())(x_j)

        # Torch
        x_t = torch.from_numpy(x_np).requires_grad_(True)
        idx_t = torch.from_numpy(idx_np.astype(np.int32))
        out_t = t_sum(x_t, idx_t, M)
        out_t.sum().backward()

        np.testing.assert_allclose(
            np.asarray(out_j), out_t.detach().numpy(), rtol=1e-10
        )
        np.testing.assert_allclose(np.asarray(grad_j), x_t.grad.numpy(), rtol=1e-10)

    def test_matvec(self):
        from nvalchemiops.torch.segment_ops import segmented_matvec as t_matvec

        idx_np = np.asarray(_make_idx(seed=62))
        v_np = np.asarray(_randn((N, 3), seed=63))
        m_np = np.asarray(_randn((M, 3, 3), seed=64))

        # JAX
        v_j = jnp.array(v_np)
        m_j = jnp.array(m_np)
        idx_j = jnp.array(idx_np.astype(np.int32))
        out_j = so.segmented_matvec(v_j, m_j, idx_j, M)
        grad_v_j, grad_m_j = jax.grad(
            lambda a, b: so.segmented_matvec(a, b, idx_j, M).sum(), argnums=(0, 1)
        )(v_j, m_j)

        # Torch
        v_t = torch.from_numpy(v_np).requires_grad_(True)
        m_t = torch.from_numpy(m_np).requires_grad_(True)
        idx_t = torch.from_numpy(idx_np.astype(np.int32))
        out_t = t_matvec(v_t, m_t, idx_t, M)
        out_t.sum().backward()

        np.testing.assert_allclose(np.asarray(out_j), out_t.detach().numpy(), rtol=1e-8)
        np.testing.assert_allclose(np.asarray(grad_v_j), v_t.grad.numpy(), rtol=1e-8)
        np.testing.assert_allclose(np.asarray(grad_m_j), m_t.grad.numpy(), rtol=1e-8)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_segment(self):
        idx = jnp.zeros(N, dtype=jnp.int32)
        x = _randn((N, 3), seed=70)
        check_grads(lambda v: so.segmented_sum(v, idx, 1), (x,), order=2, modes=["rev"])

    def test_singletons(self):
        # One element per segment (N == M).
        idx = jnp.arange(M, dtype=jnp.int32)
        x = _randn((M, 3), seed=71)
        check_grads(
            lambda v: so.segmented_mean(v, idx, M), (x,), order=2, modes=["rev"]
        )

    def test_idx_non_differentiable(self):
        """idx must be excluded from the differentiated leaves."""
        idx = _make_idx(seed=72)
        x = _randn((N, 3), seed=73)
        # No error when differentiating only x:
        jax.grad(lambda v: so.segmented_sum(v, idx, M).sum())(x)
        # If we tried to differentiate idx (an int array), JAX would refuse
        # via its own dtype rules; we don't simulate that here.

    def test_jit(self):
        """Bindings work inside jax.jit."""
        idx = _make_idx(seed=74)
        x = _randn((N, 3), seed=75)
        f = jax.jit(lambda v: so.segmented_sum(v, idx, M).sum())
        val = f(x)
        ref = float(np.asarray(so.segmented_sum(x, idx, M)).sum())
        np.testing.assert_allclose(float(val), ref, rtol=1e-10)


# ---------------------------------------------------------------------------
# Shape variety: every op exercised against four ``(N, M)`` tuples covering
# short, long, and unit-length segments.  Mirrors ``TestShapeVariety`` in the
# torch suite so both bindings have parity coverage on segment-length regimes.
#
# Each test does a forward parity check against a NumPy reference *plus* a
# first-order ``check_grads`` to confirm the autograd path holds up across
# shapes.  (Second-order ``check_grads`` is already exhaustively covered at
# the baseline shape in the per-op classes above.)
# ---------------------------------------------------------------------------


class TestShapeVariety:
    """Run each op across a small variety of ``(N, M)`` shapes."""

    @pytest.mark.parametrize("n,m", _SHAPES)
    def test_sum(self, n, m):
        idx = _make_idx(seed=80, n=n, m=m)
        x = _randn((n, 3), seed=81)
        out = so.segmented_sum(x, idx, m)
        ref = np.zeros((m, 3), dtype=np.float64)
        np.add.at(ref, np.asarray(idx), np.asarray(x))
        np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-10)
        check_grads(lambda v: so.segmented_sum(v, idx, m), (x,), order=1, modes=["rev"])

    @pytest.mark.parametrize("n,m", _SHAPES)
    def test_dot(self, n, m):
        idx = _make_idx(seed=82, n=n, m=m)
        x = _randn((n, 3), seed=83)
        y = _randn((n, 3), seed=84)
        out = so.segmented_dot(x, y, idx, m)
        ref = np.zeros(m, dtype=np.float64)
        np.add.at(ref, np.asarray(idx), (np.asarray(x) * np.asarray(y)).sum(axis=1))
        np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-10)
        check_grads(
            lambda a, b: so.segmented_dot(a, b, idx, m),
            (x, y),
            order=1,
            modes=["rev"],
        )

    @pytest.mark.parametrize("n,m", _SHAPES)
    def test_mul(self, n, m):
        idx = _make_idx(seed=85, n=n, m=m)
        x = _randn((n, 3), seed=86)
        y = _randn((m,), seed=87)
        out = so.segmented_mul(x, y, idx, m)
        ref = np.asarray(x) * np.asarray(y)[np.asarray(idx), None]
        np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-10)
        check_grads(
            lambda a, b: so.segmented_mul(a, b, idx, m),
            (x, y),
            order=1,
            modes=["rev"],
        )

    @pytest.mark.parametrize("n,m", _SHAPES)
    def test_mean(self, n, m):
        idx = _make_idx(seed=88, n=n, m=m)
        x = _randn((n, 3), seed=89)
        out = so.segmented_mean(x, idx, m)
        idx_np = np.asarray(idx)
        counts = np.bincount(idx_np, minlength=m).astype(np.float64)
        sums = np.zeros((m, 3), dtype=np.float64)
        np.add.at(sums, idx_np, np.asarray(x))
        ref = sums / counts[:, None]
        np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-10)
        check_grads(
            lambda v: so.segmented_mean(v, idx, m), (x,), order=1, modes=["rev"]
        )

    @pytest.mark.parametrize("n,m", _SHAPES)
    def test_rms_norm(self, n, m):
        idx = _make_idx(seed=90, n=n, m=m)
        # Bias away from zero so the inverse-norm divisor stays well-conditioned.
        x = _randn((n, 3), seed=91, offset=2.0)
        out = so.segmented_rms_norm(x, idx, m)
        idx_np = np.asarray(idx)
        x_np = np.asarray(x)
        counts = np.bincount(idx_np, minlength=m).astype(np.float64)
        sum_sq = np.zeros(m, dtype=np.float64)
        np.add.at(sum_sq, idx_np, (x_np * x_np).sum(axis=1))
        ref = np.sqrt(sum_sq / np.maximum(counts, 1))
        np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-10)
        check_grads(
            lambda v: so.segmented_rms_norm(v, idx, m), (x,), order=1, modes=["rev"]
        )

    @pytest.mark.parametrize("n,m", _SHAPES)
    def test_matvec(self, n, m):
        idx = _make_idx(seed=92, n=n, m=m)
        v = _randn((n, 3), seed=93)
        mat = _randn((m, 3, 3), seed=94)
        out = so.segmented_matvec(v, mat, idx, m)
        idx_np = np.asarray(idx)
        v_np = np.asarray(v)
        m_np = np.asarray(mat)
        ref = np.stack([m_np[idx_np[i]].T @ v_np[i] for i in range(n)], axis=0)
        np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-10, atol=1e-12)
        check_grads(
            lambda a, b: so.segmented_matvec(a, b, idx, m),
            (v, mat),
            order=1,
            modes=["rev"],
        )


# ---------------------------------------------------------------------------
# Hard-coded regression: pin specific binding outputs (and gradients) to
# literal values, matching the torch suite's ``TestHardcoded``.  Catches
# silent numerical regressions that equivalence-based checks would let pass.
# ---------------------------------------------------------------------------


class TestHardcoded:
    """Bytewise-pinned regression tests against hand-computed expectations."""

    def test_sum_vec3_forward_pinned(self):
        """``segmented_sum`` forward against a hand-computed result.

        Same construction as the torch suite's hard-coded test::

            x = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
            idx = [0, 0, 1]
            => out = [[5, 7, 9], [7, 8, 9]]
        """
        x = jnp.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=jnp.float64
        )
        idx = jnp.array([0, 0, 1], dtype=jnp.int32)
        out = so.segmented_sum(x, idx, 2)
        expected = np.array([[5.0, 7.0, 9.0], [7.0, 8.0, 9.0]], dtype=np.float64)
        np.testing.assert_array_equal(np.asarray(out), expected)

    def test_dot_grad_pinned(self):
        """``segmented_dot`` forward and ``jax.grad`` against hand-computed values.

        Matches the torch suite's hard-coded test setup::

            x = [[1, 0, 0], [0, 2, 0], [0, 0, 3], [4, 0, 0]]
            y = [[2, 3, 4], [5, 6, 7], [8, 9, 10], [11, 12, 13]]
            idx = [0, 0, 1, 1]
            => out = [14, 74]

        With loss = out.sum() (``g_out = [1, 1]``)::

            d/dx_i = g_out[s] * y_i = y      (so grad_x == y)
            d/dy_i = g_out[s] * x_i = x      (so grad_y == x)
        """
        x_np = np.array(
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0], [4.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        y_np = np.array(
            [
                [2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0],
                [11.0, 12.0, 13.0],
            ],
            dtype=np.float64,
        )
        x = jnp.array(x_np)
        y = jnp.array(y_np)
        idx = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
        out = so.segmented_dot(x, y, idx, 2)
        np.testing.assert_array_equal(
            np.asarray(out), np.array([14.0, 74.0], dtype=np.float64)
        )
        grad_x, grad_y = jax.grad(
            lambda a, b: so.segmented_dot(a, b, idx, 2).sum(), argnums=(0, 1)
        )(x, y)
        np.testing.assert_array_equal(np.asarray(grad_x), y_np)
        np.testing.assert_array_equal(np.asarray(grad_y), x_np)
