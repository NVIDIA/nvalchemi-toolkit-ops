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
    SegmentedDot,
    SegmentedDotBwd,
    SegmentedMatvec,
    SegmentedMatvecBwd,
    SegmentedMean,
    SegmentedMeanBwd,
    SegmentedMul,
    SegmentedMulBwd,
    SegmentedRmsNorm,
    SegmentedRmsNormBwd,
    SegmentedSum,
    SegmentedSumBwd,
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


@pytest.fixture(params=["cpu", "cuda:0"], ids=["cpu", "gpu"])
def device(request):
    """Both CPU and GPU; GPU is skipped if CUDA is not available."""
    device_name = request.param
    if device_name == "cuda:0" and not (
        torch.cuda.is_available() and wp.is_cuda_available()
    ):
        pytest.skip("CUDA not available")
    return device_name


N, M = 12, 4

# Precisions and matching tolerances used in forward parity checks.
_DTYPES = [torch.float32, torch.float64]

# Representative ``(N, M)`` shapes for ``TestShapeVariety``: covers a small
# balanced baseline, a medium case with longer segments (large L = N/M), a
# moderate-sized case, and the singleton edge case where every segment has
# exactly one element (N == M).  Together they exercise short, long, and
# unit-length segments without blowing up the test count.
_SHAPES = [
    (12, 4),  # baseline (also the value of N, M above)
    (64, 8),  # medium, L = 8
    (100, 5),  # large with few long segments, L = 20
    (24, 24),  # singletons, every segment has exactly one element
]


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

    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_forward_scalar(self, device, dtype):
        idx = _make_idx(device)
        x = _leaf((N,), device, dtype=dtype)
        y = _leaf((N,), device, dtype=dtype)
        out = segmented_dot(x, y, idx, M)
        ref = torch.zeros(M, dtype=dtype, device=device).index_add_(
            0, idx.long(), x.detach() * y.detach()
        )
        torch.testing.assert_close(out.detach(), ref, **_tols(dtype))

    def test_gradcheck_vec3(self, device):
        idx = _make_idx(device)
        x = _leaf((N, 3), device)
        y = _leaf((N, 3), device)
        assert torch.autograd.gradcheck(
            lambda a, b: segmented_dot(a, b, idx, M), (x, y), eps=1e-6, atol=1e-5
        )

    def test_gradcheck_scalar(self, device):
        idx = _make_idx(device)
        x = _leaf((N,), device)
        y = _leaf((N,), device)
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

    def test_gradgradcheck_scalar(self, device):
        idx = _make_idx(device)
        x = _leaf((N,), device)
        y = _leaf((N,), device)
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

    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_forward_scalar(self, device, dtype):
        idx = _make_idx(device)
        x = _leaf((N,), device, dtype=dtype)
        y = _leaf((M,), device, dtype=dtype)
        out = segmented_mul(x, y, idx, M)
        ref = x.detach() * y.detach()[idx.long()]
        torch.testing.assert_close(out.detach(), ref, **_tols(dtype))

    def test_gradcheck_vec3(self, device):
        idx = _make_idx(device)
        x = _leaf((N, 3), device)
        y = _leaf((M,), device)
        assert torch.autograd.gradcheck(
            lambda a, b: segmented_mul(a, b, idx, M), (x, y), eps=1e-6, atol=1e-5
        )

    def test_gradcheck_scalar(self, device):
        idx = _make_idx(device)
        x = _leaf((N,), device)
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

    def test_gradgradcheck_scalar(self, device):
        idx = _make_idx(device)
        x = _leaf((N,), device)
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

    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_forward_scalar(self, device, dtype):
        idx = _make_idx(device)
        x = _leaf((N,), device, dtype=dtype)
        out = segmented_mean(x, idx, M)
        idx_long = idx.long()
        counts = torch.bincount(idx_long, minlength=M).to(dtype)
        sums = torch.zeros(M, dtype=dtype, device=device).index_add_(
            0, idx_long, x.detach()
        )
        ref = sums / counts
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

    def test_gradgradcheck_scalar(self, device):
        idx = _make_idx(device)
        x = _leaf((N,), device)
        assert torch.autograd.gradgradcheck(
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

    def test_mul_num_segments_mismatch_raises(self, device):
        """``segmented_mul`` must validate ``num_segments == y.shape[0]``.

        Without the guard, forward succeeds (``y`` is only indexed via ``idx``)
        but backward allocates ``grad_y`` with the wrong leading dimension —
        a tensor whose shape disagrees with the leaf ``y``.
        """
        x = torch.randn(3, device=device, requires_grad=True)
        y = torch.randn(2, device=device, requires_grad=True)
        idx = torch.tensor([0, 1, 0], dtype=torch.int32, device=device)
        with pytest.raises(ValueError, match="num_segments"):
            segmented_mul(x, y, idx, num_segments=3)

    def test_matvec_num_segments_mismatch_raises(self, device):
        """``segmented_matvec`` must validate ``num_segments == m.shape[0]``."""
        v = torch.randn(3, 3, device=device, requires_grad=True)
        m = torch.randn(2, 3, 3, device=device, requires_grad=True)
        idx = torch.tensor([0, 1, 0], dtype=torch.int32, device=device)
        with pytest.raises(ValueError, match="num_segments"):
            segmented_matvec(v, m, idx, num_segments=3)

    # --- idx validation -----------------------------------------------------
    #
    # ``idx`` is used as a raw memory index inside the Warp kernels.  The
    # public wrappers must validate dtype, rank, and range before dispatch
    # so a stray value produces a clear ``ValueError`` rather than an
    # out-of-bounds device read/write.

    def test_sum_idx_out_of_range_raises(self, device):
        """``segmented_sum`` must reject ``idx.max() >= num_segments``.

        Reproduces the reviewer's minimal contract test.
        """
        x = torch.ones(2, device=device)
        idx = torch.tensor([0, 2], dtype=torch.int32, device=device)
        with pytest.raises(ValueError, match="idx.*range"):
            segmented_sum(x, idx, num_segments=2)

    def test_sum_idx_negative_raises(self, device):
        x = torch.ones(2, device=device)
        idx = torch.tensor([0, -1], dtype=torch.int32, device=device)
        with pytest.raises(ValueError, match="idx.*negative"):
            segmented_sum(x, idx, num_segments=2)

    def test_sum_idx_wrong_dtype_raises(self, device):
        x = torch.ones(2, device=device)
        idx = torch.tensor([0, 1], dtype=torch.int64, device=device)
        with pytest.raises(ValueError, match="idx.*int32"):
            segmented_sum(x, idx, num_segments=2)

    def test_sum_idx_wrong_rank_raises(self, device):
        x = torch.ones(2, device=device)
        idx = torch.tensor([[0], [1]], dtype=torch.int32, device=device)
        with pytest.raises(ValueError, match="idx.*1-D"):
            segmented_sum(x, idx, num_segments=2)

    @pytest.mark.parametrize(
        "op_name,op,arg_factory",
        [
            (
                "segmented_dot",
                segmented_dot,
                lambda dev: (
                    torch.ones(2, 3, device=dev),
                    torch.ones(2, 3, device=dev),
                ),
            ),
            (
                "segmented_mul",
                segmented_mul,
                lambda dev: (
                    torch.ones(2, 3, device=dev),
                    torch.ones(2, device=dev),
                ),
            ),
            (
                "segmented_mean",
                segmented_mean,
                lambda dev: (torch.ones(2, 3, device=dev),),
            ),
            (
                "segmented_rms_norm",
                segmented_rms_norm,
                lambda dev: (torch.ones(2, 3, device=dev) + 1.0,),
            ),
            (
                "segmented_matvec",
                segmented_matvec,
                lambda dev: (
                    torch.ones(2, 3, device=dev),
                    torch.ones(2, 3, 3, device=dev),
                ),
            ),
        ],
    )
    def test_other_ops_idx_out_of_range_raises(self, device, op_name, op, arg_factory):
        """The same idx-range validation must apply to every other public wrapper."""
        args = arg_factory(device)
        idx = torch.tensor([0, 2], dtype=torch.int32, device=device)
        with pytest.raises(ValueError, match="idx.*range"):
            op(*args, idx, 2)


# ---------------------------------------------------------------------------
# loss.backward() — the typical user-facing flow
# ---------------------------------------------------------------------------


class TestBackwardFlow:
    """Realistic training-loop pattern: build a loss, call loss.backward(),
    then read ``.grad`` off each leaf.  Distinct from gradcheck which validates
    via finite differences — these tests catch gradient *accumulation* bugs."""

    def test_backward_grad_accumulates(self, device):
        idx = _make_idx(device)
        x = _leaf((N, 3), device, dtype=torch.float64)
        # First pass populates .grad
        segmented_sum(x, idx, M).pow(2).sum().backward()
        first = x.grad.clone()
        # Second pass without zero_grad should double the accumulated grad
        segmented_sum(x, idx, M).pow(2).sum().backward()
        torch.testing.assert_close(x.grad, 2.0 * first, rtol=1e-12, atol=1e-14)

    def test_backward_two_outputs_two_leaves(self, device):
        idx = _make_idx(device)
        v = _leaf((N, 3), device, dtype=torch.float64)
        m = _leaf((M, 3, 3), device, dtype=torch.float64)
        # Mix two ops with shared idx to exercise grad path for both leaves.
        out_v = segmented_matvec(v, m, idx, M).pow(2).sum()
        out_dot = segmented_dot(v, v, idx, M).sum()
        (out_v + out_dot).backward()
        assert v.grad is not None and v.grad.shape == v.shape
        assert m.grad is not None and m.grad.shape == m.shape

    def test_retain_graph_second_backward(self, device):
        idx = _make_idx(device)
        x = _leaf((N,), device, dtype=torch.float64)
        loss = segmented_sum(x, idx, M).pow(2).sum()
        loss.backward(retain_graph=True)
        first = x.grad.clone()
        loss.backward()  # second backward through the same graph
        torch.testing.assert_close(x.grad, 2.0 * first, rtol=1e-12, atol=1e-14)


# ---------------------------------------------------------------------------
# Direct ``.apply()`` invocation on the public autograd.Function classes
# ---------------------------------------------------------------------------


class TestFunctionApplyDirect:
    """The public Function classes (SegmentedSum, …) should be callable via
    ``.apply`` without going through the convenience wrappers."""

    def test_sum_apply_matches_wrapper(self, device):
        from nvalchemiops.torch.segment_ops import SegmentedSum

        idx = _make_idx(device)
        x = _leaf((N, 3), device, dtype=torch.float64)
        torch.testing.assert_close(
            SegmentedSum.apply(x, idx, M),
            segmented_sum(x, idx, M),
            rtol=1e-12,
            atol=1e-14,
        )

    def test_mean_apply_returns_out_and_counts(self, device):
        from nvalchemiops.torch.segment_ops import SegmentedMean

        idx = _make_idx(device)
        x = _leaf((N, 3), device, dtype=torch.float64)
        out, counts = SegmentedMean.apply(x, idx, M)
        torch.testing.assert_close(
            out, segmented_mean(x, idx, M), rtol=1e-12, atol=1e-14
        )
        # counts is integer-typed; wrapper drops it.  Verify it's consistent.
        ref_counts = torch.bincount(idx.long(), minlength=M).to(torch.int32)
        torch.testing.assert_close(counts, ref_counts, rtol=0, atol=0)

    def test_rms_norm_apply_returns_saved_state(self, device):
        from nvalchemiops.torch.segment_ops import SegmentedRmsNorm

        idx = _make_idx(device)
        x = _leaf((N, 3), device, dtype=torch.float64)
        out, inv_norm, counts = SegmentedRmsNorm.apply(x, idx, M)
        torch.testing.assert_close(
            out, segmented_rms_norm(x, idx, M), rtol=1e-12, atol=1e-14
        )
        # inv_norm[s] = 1 / (out[s] * counts[s]) when denom > 0
        denom = out * counts.to(out.dtype)
        ref_inv = torch.where(denom > 0, denom.reciprocal(), torch.zeros_like(denom))
        torch.testing.assert_close(inv_norm, ref_inv, rtol=1e-10, atol=1e-12)

    def test_rms_norm_apply_saved_state_is_non_differentiable(self, device):
        """``inv_norm`` and ``counts`` are saved state, not differentiable outputs.

        Without ``ctx.mark_non_differentiable`` the backward silently drops the
        cotangent on ``inv_norm`` — a loss depending on it would *compile* but
        leave ``x.grad`` empty.  Asserts both saved-state tensors have
        ``requires_grad=False`` and that backpropping through them raises.
        """
        from nvalchemiops.torch.segment_ops import SegmentedRmsNorm

        idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)
        x = torch.randn(4, 3, dtype=torch.float64, device=device, requires_grad=True)
        out, inv_norm, counts = SegmentedRmsNorm.apply(x, idx, 2)
        # ``out`` is the only differentiable output.
        assert out.requires_grad
        assert not inv_norm.requires_grad
        assert not counts.requires_grad
        # Attempting to backprop through inv_norm must raise rather than
        # silently zero out the gradient on ``x``.
        with pytest.raises(RuntimeError):
            inv_norm.sum().backward()
        # The real path still works.
        out.pow(2).sum().backward()
        assert x.grad is not None and x.grad.abs().sum() > 0

    def test_mean_apply_counts_is_non_differentiable(self, device):
        """``counts`` from SegmentedMean.apply is saved state, not differentiable."""
        from nvalchemiops.torch.segment_ops import SegmentedMean

        idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)
        x = torch.randn(4, 3, dtype=torch.float64, device=device, requires_grad=True)
        out, counts = SegmentedMean.apply(x, idx, 2)
        assert out.requires_grad
        assert not counts.requires_grad


# ---------------------------------------------------------------------------
# Empty / degenerate inputs
# ---------------------------------------------------------------------------


class TestEmptyInputs:
    """``N == 0`` and segments with no entries should not crash and should
    produce zero-valued outputs of the right shape."""

    def test_sum_empty(self, device):
        idx = torch.empty(0, dtype=torch.int32, device=device)
        x = torch.empty(0, dtype=torch.float32, device=device)
        out = segmented_sum(x, idx, M)
        assert out.shape == (M,)
        torch.testing.assert_close(out, torch.zeros_like(out), rtol=0, atol=0)

    def test_mean_with_empty_segment_at_end(self, device):
        """One segment has no atoms.  Mean should be zero there (no NaN)."""
        # Three segments declared, only segments 0 and 1 receive atoms.
        idx = torch.tensor([0, 0, 1], dtype=torch.int32, device=device)
        x = torch.ones(3, 3, dtype=torch.float32, device=device)
        out = segmented_mean(x, idx, 3)
        # Segments 0 and 1 → mean is 1; segment 2 → 0 (empty).
        expected = torch.tensor(
            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
            dtype=torch.float32,
            device=device,
        )
        torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-7)


# ---------------------------------------------------------------------------
# Non-contiguous inputs
# ---------------------------------------------------------------------------


class TestNonContiguous:
    """The bindings call ``.contiguous()`` internally on inputs.  This test
    feeds a non-contiguous tensor (via slicing) and confirms the result still
    matches a fresh contiguous copy."""

    def test_sum_non_contiguous_x(self, device):
        idx = _make_idx(device)
        # Allocate (N, 6) then slice columns [:, ::2] → stride-2 view, non-contiguous.
        big = _leaf((N, 6), device, dtype=torch.float64)
        x_view = big[:, ::2]
        assert not x_view.is_contiguous()
        out_view = segmented_sum(x_view, idx, M)
        out_copy = segmented_sum(x_view.contiguous(), idx, M)
        torch.testing.assert_close(out_view, out_copy, rtol=1e-12, atol=1e-14)

    def test_dot_non_contiguous_y(self, device):
        idx = _make_idx(device)
        x = _leaf((N, 3), device, dtype=torch.float64)
        y_big = _leaf((N, 6), device, dtype=torch.float64)
        y_view = y_big[:, ::2]
        out_view = segmented_dot(x, y_view, idx, M)
        out_copy = segmented_dot(x, y_view.contiguous(), idx, M)
        torch.testing.assert_close(out_view, out_copy, rtol=1e-12, atol=1e-14)


# ---------------------------------------------------------------------------
# idx reuse across calls (different op, same idx)
# ---------------------------------------------------------------------------


class TestIdxReuse:
    """``idx`` is a non-differentiable metadata tensor; reusing it across
    consecutive ops with autograd enabled must not corrupt state."""

    def test_same_idx_two_ops_same_backward(self, device):
        idx = _make_idx(device)
        x = _leaf((N, 3), device, dtype=torch.float64)
        loss = segmented_sum(x, idx, M).sum() + segmented_mean(x, idx, M).sum()
        loss.backward()
        assert x.grad is not None
        # Closed-form reference: d/dx [sum_s sum_j x_j + sum_s mean_s] = 1 + 1/counts[s]
        counts = torch.bincount(idx.long(), minlength=M).to(x.dtype)
        expected = 1.0 + 1.0 / counts[idx.long(), None].expand(-1, 3)
        torch.testing.assert_close(x.grad, expected, rtol=1e-12, atol=1e-14)


# ---------------------------------------------------------------------------
# Direct ``.forward / .setup_context / .backward`` invocation on every
# ``torch.autograd.Function`` class.
#
# ``Function.apply(...)`` runs these static methods through PyTorch's C++
# autograd dispatch, which bypasses ``sys.settrace`` — coverage tools never
# see the Python method bodies execute.  Calling them directly with a fake
# ctx restores tracer visibility.
# ---------------------------------------------------------------------------


class _FakeCtx:
    """Minimal stand-in for ``torch.autograd.function.FunctionCtx``.

    ``setup_context`` needs ``save_for_backward`` and
    ``mark_non_differentiable`` plus arbitrary attribute assignment;
    ``backward`` only reads ``ctx.saved_tensors`` and the stashed attributes.
    No need for the full C++ object.
    """

    def save_for_backward(self, *tensors: torch.Tensor) -> None:
        self.saved_tensors = tuple(tensors)

    def mark_non_differentiable(self, *tensors: torch.Tensor) -> None:
        # Real ctx flips ``.requires_grad`` on each tensor; for coverage tests
        # we only need to record that the method was called.
        self.non_differentiable = tuple(tensors)


class TestDirectFunctionInvocation:
    """Cover every Function class's static method bodies."""

    # ----- segmented_sum -----------------------------------------------------

    def test_segmented_sum_static_methods(self, device):
        idx = _make_idx(device)
        x = _leaf((N, 3), device, dtype=torch.float64).detach()
        out = SegmentedSum.forward(x, idx, M)
        torch.testing.assert_close(
            out, segmented_sum(x, idx, M).detach(), rtol=1e-12, atol=1e-14
        )
        ctx = _FakeCtx()
        SegmentedSum.setup_context(ctx, (x, idx, M), out)
        assert ctx.saved_tensors == (idx,)
        assert ctx.num_segments == M
        g_out = torch.randn_like(out)
        grad_x, grad_idx, grad_num_segments = SegmentedSum.backward(ctx, g_out)
        assert grad_x.shape == x.shape
        assert grad_idx is None and grad_num_segments is None

    def test_segmented_sum_bwd_static_methods(self, device):
        idx = _make_idx(device)
        g_out = torch.randn(M, 3, dtype=torch.float64, device=device)
        grad_x = SegmentedSumBwd.forward(g_out, idx, M)
        assert grad_x.shape == (N, 3)
        ctx = _FakeCtx()
        SegmentedSumBwd.setup_context(ctx, (g_out, idx, M), grad_x)
        assert ctx.saved_tensors == (idx,)
        assert ctx.num_segments == M
        gg_x = torch.randn_like(grad_x)
        grad_g_out, grad_idx, grad_num_segments = SegmentedSumBwd.backward(ctx, gg_x)
        assert grad_g_out.shape == g_out.shape
        assert grad_idx is None and grad_num_segments is None

    # ----- segmented_dot -----------------------------------------------------

    def test_segmented_dot_static_methods(self, device):
        idx = _make_idx(device)
        x = _leaf((N, 3), device, dtype=torch.float64).detach()
        y = _leaf((N, 3), device, dtype=torch.float64).detach()
        out = SegmentedDot.forward(x, y, idx, M)
        torch.testing.assert_close(
            out, segmented_dot(x, y, idx, M).detach(), rtol=1e-12, atol=1e-14
        )
        ctx = _FakeCtx()
        SegmentedDot.setup_context(ctx, (x, y, idx, M), out)
        assert ctx.saved_tensors == (x, y, idx)
        assert ctx.num_segments == M
        g_out = torch.randn_like(out)
        grad_x, grad_y, *rest = SegmentedDot.backward(ctx, g_out)
        assert grad_x.shape == x.shape and grad_y.shape == y.shape
        assert all(r is None for r in rest)

    def test_segmented_dot_bwd_static_methods(self, device):
        idx = _make_idx(device)
        x = _leaf((N, 3), device, dtype=torch.float64).detach()
        y = _leaf((N, 3), device, dtype=torch.float64).detach()
        g_out = torch.randn(M, dtype=torch.float64, device=device)
        grad_x, grad_y = SegmentedDotBwd.forward(g_out, x, y, idx)
        assert grad_x.shape == x.shape and grad_y.shape == y.shape
        ctx = _FakeCtx()
        SegmentedDotBwd.setup_context(ctx, (g_out, x, y, idx), (grad_x, grad_y))
        assert ctx.saved_tensors == (g_out, x, y, idx)
        assert ctx.num_segments == g_out.shape[0]
        gg_gx = torch.randn_like(x)
        gg_gy = torch.randn_like(y)
        grad_g_out, grad_x_ex, grad_y_ex, grad_idx = SegmentedDotBwd.backward(
            ctx, gg_gx, gg_gy
        )
        assert grad_g_out.shape == g_out.shape
        assert grad_x_ex.shape == x.shape and grad_y_ex.shape == y.shape
        assert grad_idx is None

    # ----- segmented_mul -----------------------------------------------------

    def test_segmented_mul_static_methods(self, device):
        idx = _make_idx(device)
        x = _leaf((N, 3), device, dtype=torch.float64).detach()
        y = _leaf((M,), device, dtype=torch.float64).detach()
        out = SegmentedMul.forward(x, y, idx, M)
        torch.testing.assert_close(
            out, segmented_mul(x, y, idx, M).detach(), rtol=1e-12, atol=1e-14
        )
        ctx = _FakeCtx()
        SegmentedMul.setup_context(ctx, (x, y, idx, M), out)
        assert ctx.saved_tensors == (x, y, idx)
        assert ctx.num_segments == M
        g_out = torch.randn_like(out)
        grad_x, grad_y, *rest = SegmentedMul.backward(ctx, g_out)
        assert grad_x.shape == x.shape and grad_y.shape == y.shape
        assert all(r is None for r in rest)

    def test_segmented_mul_bwd_static_methods(self, device):
        idx = _make_idx(device)
        x = _leaf((N, 3), device, dtype=torch.float64).detach()
        y = _leaf((M,), device, dtype=torch.float64).detach()
        g_out = torch.randn_like(x)
        grad_x, grad_y = SegmentedMulBwd.forward(g_out, x, y, idx, M)
        assert grad_x.shape == x.shape and grad_y.shape == y.shape
        ctx = _FakeCtx()
        SegmentedMulBwd.setup_context(ctx, (g_out, x, y, idx, M), (grad_x, grad_y))
        assert ctx.saved_tensors == (g_out, x, y, idx)
        assert ctx.num_segments == M
        gg_gx = torch.randn_like(x)
        gg_gy = torch.randn_like(y)
        grad_g_out, grad_x_ex, grad_y_ex, *rest = SegmentedMulBwd.backward(
            ctx, gg_gx, gg_gy
        )
        assert grad_g_out.shape == g_out.shape
        assert grad_x_ex.shape == x.shape and grad_y_ex.shape == y.shape
        assert all(r is None for r in rest)

    # ----- segmented_mean ----------------------------------------------------

    def test_segmented_mean_static_methods(self, device):
        idx = _make_idx(device)
        x = _leaf((N, 3), device, dtype=torch.float64).detach()
        out, counts = SegmentedMean.forward(x, idx, M)
        torch.testing.assert_close(
            out, segmented_mean(x, idx, M).detach(), rtol=1e-12, atol=1e-14
        )
        ctx = _FakeCtx()
        SegmentedMean.setup_context(ctx, (x, idx, M), (out, counts))
        assert ctx.saved_tensors == (counts, idx)
        assert ctx.num_segments == M
        g_out = torch.randn_like(out)
        grad_x, grad_idx, grad_num_segments = SegmentedMean.backward(ctx, g_out, None)
        assert grad_x.shape == x.shape
        assert grad_idx is None and grad_num_segments is None

    def test_segmented_mean_bwd_static_methods(self, device):
        idx = _make_idx(device)
        counts = torch.bincount(idx.long(), minlength=M).to(torch.int32)
        g_out = torch.randn(M, 3, dtype=torch.float64, device=device)
        grad_x = SegmentedMeanBwd.forward(g_out, counts, idx)
        assert grad_x.shape == (N, 3)
        ctx = _FakeCtx()
        SegmentedMeanBwd.setup_context(ctx, (g_out, counts, idx), grad_x)
        assert ctx.saved_tensors == (counts, idx)
        assert ctx.num_segments == M
        gg_x = torch.randn_like(grad_x)
        grad_g_out, grad_counts, grad_idx = SegmentedMeanBwd.backward(ctx, gg_x)
        assert grad_g_out.shape == g_out.shape
        assert grad_counts is None and grad_idx is None

    # ----- segmented_rms_norm ------------------------------------------------

    def test_segmented_rms_norm_static_methods(self, device):
        idx = _make_idx(device)
        x = (_leaf((N, 3), device, dtype=torch.float64) + 2.0).detach()
        out, inv_norm, counts = SegmentedRmsNorm.forward(x, idx, M)
        torch.testing.assert_close(
            out, segmented_rms_norm(x, idx, M).detach(), rtol=1e-12, atol=1e-14
        )
        ctx = _FakeCtx()
        SegmentedRmsNorm.setup_context(ctx, (x, idx, M), (out, inv_norm, counts))
        assert ctx.saved_tensors == (x, inv_norm, counts, idx)
        assert ctx.num_segments == M
        g_out = torch.randn_like(out)
        grad_x, grad_idx, grad_num_segments = SegmentedRmsNorm.backward(
            ctx, g_out, None, None
        )
        assert grad_x.shape == x.shape
        assert grad_idx is None and grad_num_segments is None

    def test_segmented_rms_norm_bwd_static_methods(self, device):
        idx = _make_idx(device)
        x = (_leaf((N, 3), device, dtype=torch.float64) + 2.0).detach()
        counts = torch.bincount(idx.long(), minlength=M).to(torch.int32)
        # Build inv_norm from the precompute-equivalent formula.
        sum_sq = torch.zeros(M, dtype=x.dtype, device=device).index_add_(
            0, idx.long(), (x * x).sum(dim=1)
        )
        out_norm = torch.sqrt(sum_sq / counts.to(x.dtype).clamp(min=1))
        denom = out_norm * counts.to(x.dtype)
        inv_norm = torch.where(denom > 0, denom.reciprocal(), torch.zeros_like(denom))
        g_out = torch.randn_like(out_norm)
        grad_x = SegmentedRmsNormBwd.forward(g_out, x, inv_norm, counts, idx)
        assert grad_x.shape == x.shape
        ctx = _FakeCtx()
        SegmentedRmsNormBwd.setup_context(
            ctx, (g_out, x, inv_norm, counts, idx), grad_x
        )
        assert ctx.saved_tensors == (g_out, x, inv_norm, counts, idx)
        assert ctx.num_segments == g_out.shape[0]
        gg_x = torch.randn_like(grad_x)
        (
            grad_g_out,
            grad_x_extra,
            grad_inv_norm,
            grad_counts,
            grad_idx,
        ) = SegmentedRmsNormBwd.backward(ctx, gg_x)
        assert grad_g_out.shape == g_out.shape
        assert grad_x_extra.shape == x.shape
        assert grad_inv_norm is None and grad_counts is None and grad_idx is None

    # ----- segmented_matvec --------------------------------------------------

    def test_segmented_matvec_static_methods(self, device):
        idx = _make_idx(device)
        v = _leaf((N, 3), device, dtype=torch.float64).detach()
        m = _leaf((M, 3, 3), device, dtype=torch.float64).detach()
        out = SegmentedMatvec.forward(v, m, idx, M)
        torch.testing.assert_close(
            out, segmented_matvec(v, m, idx, M).detach(), rtol=1e-12, atol=1e-14
        )
        ctx = _FakeCtx()
        SegmentedMatvec.setup_context(ctx, (v, m, idx, M), out)
        assert ctx.saved_tensors == (v, m, idx)
        assert ctx.num_segments == M
        g_out = torch.randn_like(out)
        grad_v, grad_m, *rest = SegmentedMatvec.backward(ctx, g_out)
        assert grad_v.shape == v.shape and grad_m.shape == m.shape
        assert all(r is None for r in rest)

    def test_segmented_matvec_bwd_static_methods(self, device):
        idx = _make_idx(device)
        v = _leaf((N, 3), device, dtype=torch.float64).detach()
        m = _leaf((M, 3, 3), device, dtype=torch.float64).detach()
        g_out = torch.randn_like(v)
        grad_v, grad_m = SegmentedMatvecBwd.forward(g_out, v, m, idx, M)
        assert grad_v.shape == v.shape and grad_m.shape == m.shape
        ctx = _FakeCtx()
        SegmentedMatvecBwd.setup_context(ctx, (g_out, v, m, idx, M), (grad_v, grad_m))
        assert ctx.saved_tensors == (g_out, v, m, idx)
        assert ctx.num_segments == M
        gg_gv = torch.randn_like(v)
        gg_gm = torch.randn_like(m)
        (
            grad_g_out,
            grad_v_extra,
            grad_m_extra,
            *rest,
        ) = SegmentedMatvecBwd.backward(ctx, gg_gv, gg_gm)
        assert grad_g_out.shape == g_out.shape
        assert grad_v_extra.shape == v.shape and grad_m_extra.shape == m.shape
        assert all(r is None for r in rest)


# ---------------------------------------------------------------------------
# Regression tests: binding vs pure-torch scatter implementation.
#
# For each op, run forward + backward through both the binding and the
# canonical torch (``index_add_`` / ``einsum`` / broadcast) implementation
# on the same inputs, then assert both the forward outputs and the per-leaf
# ``.grad`` tensors match to dtype-appropriate tolerance.
#
# ``float32`` / ``float64`` should pass.  ``float16`` / ``bfloat16`` are
# marked xfail: the binding's dtype dispatch only knows about
# ``float32``/``float64``, so a low-precision input raises ``KeyError`` at
# dispatch time.  ``strict=False`` is used because some low-precision
# variants may happen to succeed on certain devices.
# ---------------------------------------------------------------------------


_REGRESSION_DTYPES = [
    torch.float32,
    torch.float64,
    pytest.param(
        torch.float16,
        marks=pytest.mark.xfail(
            reason="binding dispatch supports float32/float64 only",
            strict=False,
        ),
    ),
    pytest.param(
        torch.bfloat16,
        marks=pytest.mark.xfail(
            reason="binding dispatch supports float32/float64 only",
            strict=False,
        ),
    ),
]


class TestRegressionVsTorchScatter:
    """End-to-end (forward + ``.grad``) parity with pure-torch references."""

    @pytest.mark.parametrize("dtype", _REGRESSION_DTYPES)
    def test_sum_regression(self, device, dtype):
        idx = _make_idx(device)
        x_data = torch.randn(N, 3, dtype=dtype, device=device)
        # Binding path
        x_b = x_data.clone().detach().requires_grad_(True)
        out_b = segmented_sum(x_b, idx, M)
        out_b.pow(2).sum().backward()
        # Pure-torch reference: scatter-add via index_add_
        x_r = x_data.clone().detach().requires_grad_(True)
        out_r = torch.zeros(M, 3, dtype=dtype, device=device).index_add_(
            0, idx.long(), x_r
        )
        out_r.pow(2).sum().backward()
        torch.testing.assert_close(out_b.detach(), out_r.detach(), **_tols(dtype))
        torch.testing.assert_close(x_b.grad, x_r.grad, **_tols(dtype))

    @pytest.mark.parametrize("dtype", _REGRESSION_DTYPES)
    def test_dot_regression(self, device, dtype):
        idx = _make_idx(device)
        x_data = torch.randn(N, 3, dtype=dtype, device=device)
        y_data = torch.randn(N, 3, dtype=dtype, device=device)
        x_b = x_data.clone().detach().requires_grad_(True)
        y_b = y_data.clone().detach().requires_grad_(True)
        out_b = segmented_dot(x_b, y_b, idx, M)
        out_b.pow(2).sum().backward()
        x_r = x_data.clone().detach().requires_grad_(True)
        y_r = y_data.clone().detach().requires_grad_(True)
        out_r = torch.zeros(M, dtype=dtype, device=device).index_add_(
            0, idx.long(), (x_r * y_r).sum(dim=1)
        )
        out_r.pow(2).sum().backward()
        torch.testing.assert_close(out_b.detach(), out_r.detach(), **_tols(dtype))
        torch.testing.assert_close(x_b.grad, x_r.grad, **_tols(dtype))
        torch.testing.assert_close(y_b.grad, y_r.grad, **_tols(dtype))

    @pytest.mark.parametrize("dtype", _REGRESSION_DTYPES)
    def test_mul_regression(self, device, dtype):
        idx = _make_idx(device)
        x_data = torch.randn(N, 3, dtype=dtype, device=device)
        y_data = torch.randn(M, dtype=dtype, device=device)
        x_b = x_data.clone().detach().requires_grad_(True)
        y_b = y_data.clone().detach().requires_grad_(True)
        out_b = segmented_mul(x_b, y_b, idx, M)
        out_b.pow(2).sum().backward()
        x_r = x_data.clone().detach().requires_grad_(True)
        y_r = y_data.clone().detach().requires_grad_(True)
        # Pure-torch reference: gather + broadcast multiply
        out_r = x_r * y_r[idx.long(), None]
        out_r.pow(2).sum().backward()
        torch.testing.assert_close(out_b.detach(), out_r.detach(), **_tols(dtype))
        torch.testing.assert_close(x_b.grad, x_r.grad, **_tols(dtype))
        torch.testing.assert_close(y_b.grad, y_r.grad, **_tols(dtype))

    @pytest.mark.parametrize("dtype", _REGRESSION_DTYPES)
    def test_mean_regression(self, device, dtype):
        idx = _make_idx(device)
        x_data = torch.randn(N, 3, dtype=dtype, device=device)
        x_b = x_data.clone().detach().requires_grad_(True)
        out_b = segmented_mean(x_b, idx, M)
        out_b.pow(2).sum().backward()
        # Pure-torch reference: sum / count, with count from bincount
        x_r = x_data.clone().detach().requires_grad_(True)
        sums = torch.zeros(M, 3, dtype=dtype, device=device).index_add_(
            0, idx.long(), x_r
        )
        counts = torch.bincount(idx.long(), minlength=M).to(dtype)
        out_r = sums / counts.unsqueeze(-1)
        out_r.pow(2).sum().backward()
        torch.testing.assert_close(out_b.detach(), out_r.detach(), **_tols(dtype))
        torch.testing.assert_close(x_b.grad, x_r.grad, **_tols(dtype))

    @pytest.mark.parametrize("dtype", _REGRESSION_DTYPES)
    def test_rms_norm_regression(self, device, dtype):
        idx = _make_idx(device)
        # Bias away from zero so the inverse-norm singularity stays well-conditioned.
        x_data = torch.randn(N, 3, dtype=dtype, device=device) + 2.0
        x_b = x_data.clone().detach().requires_grad_(True)
        out_b = segmented_rms_norm(x_b, idx, M)
        out_b.pow(2).sum().backward()
        # Pure-torch reference: sqrt(mean of squared norms per segment)
        x_r = x_data.clone().detach().requires_grad_(True)
        sum_sq = torch.zeros(M, dtype=dtype, device=device).index_add_(
            0, idx.long(), (x_r * x_r).sum(dim=1)
        )
        counts = torch.bincount(idx.long(), minlength=M).to(dtype).clamp(min=1)
        out_r = torch.sqrt(sum_sq / counts)
        out_r.pow(2).sum().backward()
        torch.testing.assert_close(out_b.detach(), out_r.detach(), **_tols(dtype))
        torch.testing.assert_close(x_b.grad, x_r.grad, **_tols(dtype))

    @pytest.mark.parametrize("dtype", _REGRESSION_DTYPES)
    def test_matvec_regression(self, device, dtype):
        idx = _make_idx(device)
        v_data = torch.randn(N, 3, dtype=dtype, device=device)
        m_data = torch.randn(M, 3, 3, dtype=dtype, device=device)
        v_b = v_data.clone().detach().requires_grad_(True)
        m_b = m_data.clone().detach().requires_grad_(True)
        out_b = segmented_matvec(v_b, m_b, idx, M)
        out_b.pow(2).sum().backward()
        # Pure-torch reference: gather + einsum.  Forward computes
        # out[i] = M[idx[i]]^T @ v[i] — matches the binding's transpose convention.
        v_r = v_data.clone().detach().requires_grad_(True)
        m_r = m_data.clone().detach().requires_grad_(True)
        out_r = torch.einsum("nji,nj->ni", m_r[idx.long()], v_r)
        out_r.pow(2).sum().backward()
        torch.testing.assert_close(out_b.detach(), out_r.detach(), **_tols(dtype))
        torch.testing.assert_close(v_b.grad, v_r.grad, **_tols(dtype))
        torch.testing.assert_close(m_b.grad, m_r.grad, **_tols(dtype))


# ---------------------------------------------------------------------------
# Shape variety: every op exercised against four ``(N, M)`` tuples covering
# short, long, and unit-length segments so we don't rely on a single size.
#
# Each test does a forward parity check against the torch-scatter reference
# *plus* a first-order ``gradcheck`` to confirm the autograd path holds up
# across shapes.  Tolerances follow the float64 defaults of ``_tols``.
# ---------------------------------------------------------------------------


class TestShapeVariety:
    """Run each op across a small variety of ``(N, M)`` shapes."""

    @pytest.mark.parametrize("n,m", _SHAPES)
    def test_sum_vec3(self, device, n, m):
        idx = _make_idx(device, n=n, m=m)
        x = _leaf((n, 3), device)
        out = segmented_sum(x, idx, m)
        ref = torch.zeros(m, 3, dtype=x.dtype, device=device).index_add_(
            0, idx.long(), x.detach()
        )
        torch.testing.assert_close(out.detach(), ref, **_tols(x.dtype))
        assert torch.autograd.gradcheck(
            lambda v: segmented_sum(v, idx, m), (x,), eps=1e-6, atol=1e-5
        )

    @pytest.mark.parametrize("n,m", _SHAPES)
    def test_dot_vec3(self, device, n, m):
        idx = _make_idx(device, n=n, m=m)
        x = _leaf((n, 3), device)
        y = _leaf((n, 3), device)
        out = segmented_dot(x, y, idx, m)
        ref = torch.zeros(m, dtype=x.dtype, device=device).index_add_(
            0, idx.long(), (x.detach() * y.detach()).sum(dim=1)
        )
        torch.testing.assert_close(out.detach(), ref, **_tols(x.dtype))
        assert torch.autograd.gradcheck(
            lambda a, b: segmented_dot(a, b, idx, m), (x, y), eps=1e-6, atol=1e-5
        )

    @pytest.mark.parametrize("n,m", _SHAPES)
    def test_mul_vec3(self, device, n, m):
        idx = _make_idx(device, n=n, m=m)
        x = _leaf((n, 3), device)
        y = _leaf((m,), device)
        out = segmented_mul(x, y, idx, m)
        ref = x.detach() * y.detach()[idx.long(), None]
        torch.testing.assert_close(out.detach(), ref, **_tols(x.dtype))
        assert torch.autograd.gradcheck(
            lambda a, b: segmented_mul(a, b, idx, m), (x, y), eps=1e-6, atol=1e-5
        )

    @pytest.mark.parametrize("n,m", _SHAPES)
    def test_mean_vec3(self, device, n, m):
        idx = _make_idx(device, n=n, m=m)
        x = _leaf((n, 3), device)
        out = segmented_mean(x, idx, m)
        idx_long = idx.long()
        counts = torch.bincount(idx_long, minlength=m).to(x.dtype)
        sums = torch.zeros(m, 3, dtype=x.dtype, device=device).index_add_(
            0, idx_long, x.detach()
        )
        ref = sums / counts.unsqueeze(-1)
        torch.testing.assert_close(out.detach(), ref, **_tols(x.dtype))
        assert torch.autograd.gradcheck(
            lambda v: segmented_mean(v, idx, m), (x,), eps=1e-6, atol=1e-5
        )

    @pytest.mark.parametrize("n,m", _SHAPES)
    def test_rms_norm_vec3(self, device, n, m):
        idx = _make_idx(device, n=n, m=m)
        # Bias x away from zero so the inverse-norm divisor stays well-conditioned.
        x = _leaf((n, 3), device) + 2.0
        x = x.detach().clone().requires_grad_(True)
        out = segmented_rms_norm(x, idx, m)
        idx_long = idx.long()
        counts = torch.bincount(idx_long, minlength=m).to(x.dtype)
        sum_sq = torch.zeros(m, dtype=x.dtype, device=device).index_add_(
            0, idx_long, (x.detach() * x.detach()).sum(dim=1)
        )
        ref = torch.sqrt(sum_sq / counts.clamp(min=1))
        torch.testing.assert_close(out.detach(), ref, **_tols(x.dtype))
        assert torch.autograd.gradcheck(
            lambda v: segmented_rms_norm(v, idx, m), (x,), eps=1e-6, atol=1e-5
        )

    @pytest.mark.parametrize("n,m", _SHAPES)
    def test_matvec(self, device, n, m):
        idx = _make_idx(device, n=n, m=m)
        v = _leaf((n, 3), device)
        mat = _leaf((m, 3, 3), device)
        out = segmented_matvec(v, mat, idx, m)
        ref = torch.einsum("nji,nj->ni", mat.detach()[idx.long()], v.detach())
        torch.testing.assert_close(out.detach(), ref, **_tols(v.dtype))
        assert torch.autograd.gradcheck(
            lambda a, b: segmented_matvec(a, b, idx, m),
            (v, mat),
            eps=1e-6,
            atol=1e-5,
        )
