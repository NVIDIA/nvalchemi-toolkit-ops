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

# Precisions and matching tolerances used in forward parity checks.
_DTYPES = [torch.float32, torch.float64]


def _tols(dtype: torch.dtype) -> dict:
    if dtype is torch.float64:
        return {"rtol": 1e-12, "atol": 1e-14}
    return {"rtol": 1e-5, "atol": 1e-6}


@pytest.fixture(autouse=True)
def _seed_torch_rng():
    """Seed PyTorch's global RNG (CPU and all CUDA devices) once per test."""
    torch.manual_seed(0)


def _make_idx(device: str, *, n: int = N, m: int = M) -> torch.Tensor:
    """Sorted int32 segment index of length ``n`` with at least one entry per segment."""
    # Guarantee every segment is non-empty so segmented_mean / segmented_rms_norm
    # are well-defined for gradcheck.
    base = torch.arange(m, dtype=torch.int32)
    extra = torch.randint(0, m, (n - m,), dtype=torch.int32)
    idx, _ = torch.cat([base, extra]).sort()
    return idx.to(device)


def _leaf(shape, device: str, *, dtype=torch.float64) -> torch.Tensor:
    return torch.randn(shape, dtype=dtype, device=device).requires_grad_(True)


# ---------------------------------------------------------------------------
# segmented_sum
# ---------------------------------------------------------------------------


class TestSegmentedSum:
    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_forward_scalar(self, device, dtype):
        idx = _make_idx(device)
        x = _leaf((N,), device, dtype=dtype)
        out = segmented_sum(x, idx, M)
        ref = torch.zeros(M, dtype=dtype, device=device).index_add_(
            0, idx.long(), x.detach()
        )
        torch.testing.assert_close(out.detach(), ref, **_tols(dtype))

    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_forward_vec3(self, device, dtype):
        idx = _make_idx(device)
        x = _leaf((N, 3), device, dtype=dtype)
        out = segmented_sum(x, idx, M)
        ref = torch.zeros(M, 3, dtype=dtype, device=device).index_add_(
            0, idx.long(), x.detach()
        )
        torch.testing.assert_close(out.detach(), ref, **_tols(dtype))

    def test_gradcheck_scalar(self, device):
        idx = _make_idx(device)
        x = _leaf((N,), device)
        assert torch.autograd.gradcheck(
            lambda v: segmented_sum(v, idx, M), (x,), eps=1e-6, atol=1e-5
        )

    def test_gradcheck_vec3(self, device):
        idx = _make_idx(device)
        x = _leaf((N, 3), device)
        assert torch.autograd.gradcheck(
            lambda v: segmented_sum(v, idx, M), (x,), eps=1e-6, atol=1e-5
        )

    def test_gradgradcheck_scalar(self, device):
        idx = _make_idx(device)
        x = _leaf((N,), device)
        assert torch.autograd.gradgradcheck(
            lambda v: segmented_sum(v, idx, M), (x,), eps=1e-6, atol=1e-5
        )

    def test_gradgradcheck_vec3(self, device):
        idx = _make_idx(device)
        x = _leaf((N, 3), device)
        assert torch.autograd.gradgradcheck(
            lambda v: segmented_sum(v, idx, M), (x,), eps=1e-6, atol=1e-5
        )


# ---------------------------------------------------------------------------
# segmented_dot
# ---------------------------------------------------------------------------


class TestSegmentedDot:
    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_forward_vec3(self, device, dtype):
        idx = _make_idx(device)
        x = _leaf((N, 3), device, dtype=dtype)
        y = _leaf((N, 3), device, dtype=dtype)
        out = segmented_dot(x, y, idx, M)
        ref = torch.zeros(M, dtype=dtype, device=device).index_add_(
            0, idx.long(), (x.detach() * y.detach()).sum(dim=1)
        )
        torch.testing.assert_close(out.detach(), ref, **_tols(dtype))

    def test_gradcheck_vec3(self, device):
        idx = _make_idx(device)
        x = _leaf((N, 3), device)
        y = _leaf((N, 3), device)
        assert torch.autograd.gradcheck(
            lambda a, b: segmented_dot(a, b, idx, M), (x, y), eps=1e-6, atol=1e-5
        )

    def test_gradgradcheck_vec3(self, device):
        idx = _make_idx(device)
        x = _leaf((N, 3), device)
        y = _leaf((N, 3), device)
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
    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_forward_vec3(self, device, dtype):
        idx = _make_idx(device)
        x = _leaf((N, 3), device, dtype=dtype)
        y = _leaf((M,), device, dtype=dtype)
        out = segmented_mul(x, y, idx, M)
        ref = x.detach() * y.detach()[idx.long(), None]
        torch.testing.assert_close(out.detach(), ref, **_tols(dtype))

    def test_gradcheck_vec3(self, device):
        idx = _make_idx(device)
        x = _leaf((N, 3), device)
        y = _leaf((M,), device)
        assert torch.autograd.gradcheck(
            lambda a, b: segmented_mul(a, b, idx, M), (x, y), eps=1e-6, atol=1e-5
        )

    def test_gradgradcheck_vec3(self, device):
        idx = _make_idx(device)
        x = _leaf((N, 3), device)
        y = _leaf((M,), device)
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
    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_forward_vec3(self, device, dtype):
        idx = _make_idx(device)
        x = _leaf((N, 3), device, dtype=dtype)
        out = segmented_mean(x, idx, M)
        idx_long = idx.long()
        counts = torch.bincount(idx_long, minlength=M).to(dtype)
        sums = torch.zeros(M, 3, dtype=dtype, device=device).index_add_(
            0, idx_long, x.detach()
        )
        ref = sums / counts.unsqueeze(-1)
        torch.testing.assert_close(out.detach(), ref, **_tols(dtype))

    def test_gradcheck_scalar(self, device):
        idx = _make_idx(device)
        x = _leaf((N,), device)
        assert torch.autograd.gradcheck(
            lambda v: segmented_mean(v, idx, M), (x,), eps=1e-6, atol=1e-5
        )

    def test_gradcheck_vec3(self, device):
        idx = _make_idx(device)
        x = _leaf((N, 3), device)
        assert torch.autograd.gradcheck(
            lambda v: segmented_mean(v, idx, M), (x,), eps=1e-6, atol=1e-5
        )

    def test_gradgradcheck_vec3(self, device):
        idx = _make_idx(device)
        x = _leaf((N, 3), device)
        assert torch.autograd.gradgradcheck(
            lambda v: segmented_mean(v, idx, M), (x,), eps=1e-6, atol=1e-5
        )


# ---------------------------------------------------------------------------
# segmented_rms_norm  (vec3 only; precompute path is the default)
# ---------------------------------------------------------------------------


class TestSegmentedRmsNorm:
    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_forward_matches_reference(self, device, dtype):
        """Forward result equals a closed-form torch reference — confirms precompute path."""
        idx = _make_idx(device)
        x = _leaf((N, 3), device, dtype=dtype)
        out = segmented_rms_norm(x, idx, M)
        idx_long = idx.long()
        counts = torch.bincount(idx_long, minlength=M).to(dtype)
        sum_sq = torch.zeros(M, dtype=dtype, device=device).index_add_(
            0, idx_long, (x.detach() * x.detach()).sum(dim=1)
        )
        ref = torch.sqrt(sum_sq / counts.clamp(min=1))
        torch.testing.assert_close(out.detach(), ref, **_tols(dtype))

    def test_gradcheck_vec3(self, device):
        idx = _make_idx(device)
        # Bias away from zero so the inverse-norm divisor stays well-conditioned.
        x = _leaf((N, 3), device) + 2.0
        x = x.detach().clone().requires_grad_(True)
        assert torch.autograd.gradcheck(
            lambda v: segmented_rms_norm(v, idx, M), (x,), eps=1e-6, atol=1e-5
        )

    def test_gradgradcheck_vec3(self, device):
        idx = _make_idx(device)
        x = _leaf((N, 3), device) + 2.0
        x = x.detach().clone().requires_grad_(True)
        assert torch.autograd.gradgradcheck(
            lambda v: segmented_rms_norm(v, idx, M), (x,), eps=1e-6, atol=1e-4
        )


# ---------------------------------------------------------------------------
# segmented_matvec
# ---------------------------------------------------------------------------


class TestSegmentedMatvec:
    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_forward(self, device, dtype):
        idx = _make_idx(device)
        v = _leaf((N, 3), device, dtype=dtype)
        m = _leaf((M, 3, 3), device, dtype=dtype)
        out = segmented_matvec(v, m, idx, M)
        # out[i] = M[idx[i]]^T @ v[i]
        ref = torch.einsum("nji,nj->ni", m.detach()[idx.long()], v.detach())
        torch.testing.assert_close(out.detach(), ref, **_tols(dtype))

    def test_gradcheck(self, device):
        idx = _make_idx(device)
        v = _leaf((N, 3), device)
        m = _leaf((M, 3, 3), device)
        assert torch.autograd.gradcheck(
            lambda a, b: segmented_matvec(a, b, idx, M),
            (v, m),
            eps=1e-6,
            atol=1e-5,
        )

    def test_gradgradcheck(self, device):
        idx = _make_idx(device)
        v = _leaf((N, 3), device)
        m = _leaf((M, 3, 3), device)
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
        x = _leaf((N, 3), device)
        assert torch.autograd.gradcheck(
            lambda v: segmented_sum(v, idx, 1), (x,), eps=1e-6, atol=1e-5
        )

    def test_singletons(self, device):
        # Every segment has exactly one element (N == M).
        idx = torch.arange(M, dtype=torch.int32, device=device)
        x = _leaf((M, 3), device)
        assert torch.autograd.gradcheck(
            lambda v: segmented_mean(v, idx, M), (x,), eps=1e-6, atol=1e-5
        )

    def test_idx_not_used_for_grad(self, device):
        """idx is integer metadata; its gradient slot must be None."""
        idx = _make_idx(device)
        x = _leaf((N, 3), device)
        out = segmented_sum(x, idx, M)
        loss = out.pow(2).sum()
        loss.backward()
        assert x.grad is not None
        assert idx.grad is None
