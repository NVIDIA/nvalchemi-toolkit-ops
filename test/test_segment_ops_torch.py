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

"""Tests for nvalchemiops.torch.segment_ops (PR 2 Torch bindings).

Coverage per public op
----------------------
- forward parity with the underlying Warp implementation
- ``torch.autograd.gradcheck`` (first-order)  on float64 inputs
- ``torch.autograd.gradgradcheck`` (second-order) on float64 inputs
- scalar and vec3 variants where the binding supports both
- edge cases: empty segments, single segment, singletons

The precompute path of ``segmented_rms_norm`` is exercised by checking that
the public function returns the same value as a NumPy reference (the binding
always takes the precompute path because backward requires the saved state).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import warp as wp

from nvalchemiops.torch.segment_ops import (
    segmented_dot,
    segmented_matvec,
    segmented_mean,
    segmented_mul,
    segmented_rms_norm,
    segmented_sum,
)

wp.init()


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _available_devices():
    devices = ["cpu"]
    if torch.cuda.is_available() and wp.is_cuda_available():
        devices.append("cuda:0")
    return devices


@pytest.fixture(scope="module", params=_available_devices())
def device(request):
    return request.param


N, M = 12, 4


def _make_idx(device: str, *, n: int = N, m: int = M, seed: int = 0) -> torch.Tensor:
    """Sorted int32 segment index of length ``n`` with at least one entry per segment."""
    rng = np.random.default_rng(seed)
    # Guarantee every segment is non-empty so segmented_mean / segmented_rms_norm
    # are well-defined for gradcheck.
    base = np.arange(m, dtype=np.int32)
    extra = rng.integers(0, m, n - m).astype(np.int32)
    idx = np.sort(np.concatenate([base, extra]))
    return torch.from_numpy(idx).to(device)


def _leaf(shape, device: str, *, dtype=torch.float64, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(shape, generator=g, dtype=dtype).to(device).requires_grad_(True)


# ---------------------------------------------------------------------------
# segmented_sum
# ---------------------------------------------------------------------------


class TestSegmentedSum:
    def test_forward_scalar(self, device):
        idx = _make_idx(device)
        x = _leaf((N,), device, dtype=torch.float32, seed=1)
        out = segmented_sum(x, idx, M)
        ref = np.zeros(M, dtype=np.float32)
        np.add.at(ref, idx.cpu().numpy(), x.detach().cpu().numpy())
        np.testing.assert_allclose(out.detach().cpu().numpy(), ref, rtol=1e-5)

    def test_forward_vec3(self, device):
        idx = _make_idx(device)
        x = _leaf((N, 3), device, dtype=torch.float32, seed=2)
        out = segmented_sum(x, idx, M)
        ref = np.zeros((M, 3), dtype=np.float32)
        np.add.at(ref, idx.cpu().numpy(), x.detach().cpu().numpy())
        np.testing.assert_allclose(out.detach().cpu().numpy(), ref, rtol=1e-5)

    def test_gradcheck_scalar(self, device):
        idx = _make_idx(device)
        x = _leaf((N,), device, seed=3)
        assert torch.autograd.gradcheck(
            lambda v: segmented_sum(v, idx, M), (x,), eps=1e-6, atol=1e-5
        )

    def test_gradcheck_vec3(self, device):
        idx = _make_idx(device)
        x = _leaf((N, 3), device, seed=4)
        assert torch.autograd.gradcheck(
            lambda v: segmented_sum(v, idx, M), (x,), eps=1e-6, atol=1e-5
        )

    def test_gradgradcheck_scalar(self, device):
        idx = _make_idx(device)
        x = _leaf((N,), device, seed=5)
        assert torch.autograd.gradgradcheck(
            lambda v: segmented_sum(v, idx, M), (x,), eps=1e-6, atol=1e-5
        )

    def test_gradgradcheck_vec3(self, device):
        idx = _make_idx(device)
        x = _leaf((N, 3), device, seed=6)
        assert torch.autograd.gradgradcheck(
            lambda v: segmented_sum(v, idx, M), (x,), eps=1e-6, atol=1e-5
        )


# ---------------------------------------------------------------------------
# segmented_dot
# ---------------------------------------------------------------------------


class TestSegmentedDot:
    def test_forward_vec3(self, device):
        idx = _make_idx(device)
        x = _leaf((N, 3), device, dtype=torch.float32, seed=10)
        y = _leaf((N, 3), device, dtype=torch.float32, seed=11)
        out = segmented_dot(x, y, idx, M)
        ref = np.zeros(M, dtype=np.float32)
        np.add.at(
            ref,
            idx.cpu().numpy(),
            (x.detach().cpu().numpy() * y.detach().cpu().numpy()).sum(axis=1),
        )
        np.testing.assert_allclose(out.detach().cpu().numpy(), ref, rtol=1e-4)

    def test_gradcheck_vec3(self, device):
        idx = _make_idx(device)
        x = _leaf((N, 3), device, seed=12)
        y = _leaf((N, 3), device, seed=13)
        assert torch.autograd.gradcheck(
            lambda a, b: segmented_dot(a, b, idx, M), (x, y), eps=1e-6, atol=1e-5
        )

    def test_gradgradcheck_vec3(self, device):
        idx = _make_idx(device)
        x = _leaf((N, 3), device, seed=14)
        y = _leaf((N, 3), device, seed=15)
        assert torch.autograd.gradgradcheck(
            lambda a, b: segmented_dot(a, b, idx, M),
            (x, y),
            eps=1e-6,
            atol=1e-4,
        )


# ---------------------------------------------------------------------------
# segmented_mul
# ---------------------------------------------------------------------------


class TestSegmentedMul:
    def test_forward_vec3(self, device):
        idx = _make_idx(device)
        x = _leaf((N, 3), device, dtype=torch.float32, seed=20)
        y = _leaf((M,), device, dtype=torch.float32, seed=21)
        out = segmented_mul(x, y, idx, M)
        ref = (
            x.detach().cpu().numpy() * y.detach().cpu().numpy()[idx.cpu().numpy(), None]
        )
        np.testing.assert_allclose(out.detach().cpu().numpy(), ref, rtol=1e-5)

    def test_gradcheck_vec3(self, device):
        idx = _make_idx(device)
        x = _leaf((N, 3), device, seed=22)
        y = _leaf((M,), device, seed=23)
        assert torch.autograd.gradcheck(
            lambda a, b: segmented_mul(a, b, idx, M), (x, y), eps=1e-6, atol=1e-5
        )

    def test_gradgradcheck_vec3(self, device):
        idx = _make_idx(device)
        x = _leaf((N, 3), device, seed=24)
        y = _leaf((M,), device, seed=25)
        assert torch.autograd.gradgradcheck(
            lambda a, b: segmented_mul(a, b, idx, M),
            (x, y),
            eps=1e-6,
            atol=1e-4,
        )


# ---------------------------------------------------------------------------
# segmented_mean
# ---------------------------------------------------------------------------


class TestSegmentedMean:
    def test_forward_vec3(self, device):
        idx = _make_idx(device)
        x = _leaf((N, 3), device, dtype=torch.float32, seed=30)
        out = segmented_mean(x, idx, M)
        idx_np = idx.cpu().numpy()
        counts = np.bincount(idx_np, minlength=M).astype(np.float32)
        sums = np.zeros((M, 3), dtype=np.float32)
        np.add.at(sums, idx_np, x.detach().cpu().numpy())
        ref = sums / counts[:, None]
        np.testing.assert_allclose(out.detach().cpu().numpy(), ref, rtol=1e-5)

    def test_gradcheck_scalar(self, device):
        idx = _make_idx(device)
        x = _leaf((N,), device, seed=31)
        assert torch.autograd.gradcheck(
            lambda v: segmented_mean(v, idx, M), (x,), eps=1e-6, atol=1e-5
        )

    def test_gradcheck_vec3(self, device):
        idx = _make_idx(device)
        x = _leaf((N, 3), device, seed=32)
        assert torch.autograd.gradcheck(
            lambda v: segmented_mean(v, idx, M), (x,), eps=1e-6, atol=1e-5
        )

    def test_gradgradcheck_vec3(self, device):
        idx = _make_idx(device)
        x = _leaf((N, 3), device, seed=33)
        assert torch.autograd.gradgradcheck(
            lambda v: segmented_mean(v, idx, M), (x,), eps=1e-6, atol=1e-5
        )


# ---------------------------------------------------------------------------
# segmented_rms_norm  (vec3 only; precompute path is the default)
# ---------------------------------------------------------------------------


class TestSegmentedRmsNorm:
    def test_forward_matches_reference(self, device):
        """Forward result equals NumPy reference — confirms precompute path."""
        idx = _make_idx(device)
        x = _leaf((N, 3), device, dtype=torch.float64, seed=40)
        out = segmented_rms_norm(x, idx, M)
        idx_np = idx.cpu().numpy()
        x_np = x.detach().cpu().numpy()
        counts = np.bincount(idx_np, minlength=M).astype(np.float64)
        sum_sq = np.zeros(M, dtype=np.float64)
        np.add.at(sum_sq, idx_np, (x_np * x_np).sum(axis=1))
        ref = np.sqrt(sum_sq / np.maximum(counts, 1))
        np.testing.assert_allclose(out.detach().cpu().numpy(), ref, rtol=1e-10)

    def test_gradcheck_vec3(self, device):
        idx = _make_idx(device)
        # Bias away from zero so the inverse-norm divisor stays well-conditioned.
        x = _leaf((N, 3), device, seed=41) + 2.0
        x = x.detach().clone().requires_grad_(True)
        assert torch.autograd.gradcheck(
            lambda v: segmented_rms_norm(v, idx, M), (x,), eps=1e-6, atol=1e-5
        )

    def test_gradgradcheck_vec3(self, device):
        idx = _make_idx(device)
        x = _leaf((N, 3), device, seed=42) + 2.0
        x = x.detach().clone().requires_grad_(True)
        assert torch.autograd.gradgradcheck(
            lambda v: segmented_rms_norm(v, idx, M), (x,), eps=1e-6, atol=1e-4
        )


# ---------------------------------------------------------------------------
# segmented_matvec
# ---------------------------------------------------------------------------


class TestSegmentedMatvec:
    def test_forward(self, device):
        idx = _make_idx(device)
        v = _leaf((N, 3), device, dtype=torch.float32, seed=50)
        m = _leaf((M, 3, 3), device, dtype=torch.float32, seed=51)
        out = segmented_matvec(v, m, idx, M)
        idx_np = idx.cpu().numpy()
        v_np = v.detach().cpu().numpy()
        m_np = m.detach().cpu().numpy()
        ref = np.stack([m_np[idx_np[i]].T @ v_np[i] for i in range(N)], axis=0)
        np.testing.assert_allclose(
            out.detach().cpu().numpy(), ref, rtol=1e-5, atol=1e-6
        )

    def test_gradcheck(self, device):
        idx = _make_idx(device)
        v = _leaf((N, 3), device, seed=52)
        m = _leaf((M, 3, 3), device, seed=53)
        assert torch.autograd.gradcheck(
            lambda a, b: segmented_matvec(a, b, idx, M),
            (v, m),
            eps=1e-6,
            atol=1e-5,
        )

    def test_gradgradcheck(self, device):
        idx = _make_idx(device)
        v = _leaf((N, 3), device, seed=54)
        m = _leaf((M, 3, 3), device, seed=55)
        assert torch.autograd.gradgradcheck(
            lambda a, b: segmented_matvec(a, b, idx, M),
            (v, m),
            eps=1e-6,
            atol=1e-4,
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_segment(self, device):
        idx = torch.zeros(N, dtype=torch.int32, device=device)
        x = _leaf((N, 3), device, seed=60)
        assert torch.autograd.gradcheck(
            lambda v: segmented_sum(v, idx, 1), (x,), eps=1e-6, atol=1e-5
        )

    def test_singletons(self, device):
        # Every segment has exactly one element (N == M).
        idx = torch.arange(M, dtype=torch.int32, device=device)
        x = _leaf((M, 3), device, seed=61)
        assert torch.autograd.gradcheck(
            lambda v: segmented_mean(v, idx, M), (x,), eps=1e-6, atol=1e-5
        )

    def test_idx_not_used_for_grad(self, device):
        """idx is integer metadata; its gradient slot must be None."""
        idx = _make_idx(device)
        x = _leaf((N, 3), device, seed=62)
        out = segmented_sum(x, idx, M)
        loss = out.pow(2).sum()
        loss.backward()
        assert x.grad is not None
        assert idx.grad is None
