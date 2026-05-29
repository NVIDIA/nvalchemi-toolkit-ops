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

"""PyTorch autograd bindings for segment operations (PR 2).

Each public function accepts PyTorch tensors and returns a PyTorch tensor with
full first-order and second-order backward support.  Integer metadata (``idx``,
``num_segments``) always receives ``None`` gradient.

Tensor layout conventions
-------------------------
- Scalar arrays  : shape ``(N,)`` or ``(num_segments,)``
- Vec3 arrays    : shape ``(N, 3)`` or ``(num_segments, 3)``
- Mat33 arrays   : shape ``(num_segments, 3, 3)``

The dtype (float32 / float64) is inferred from the input tensor.

Public API
----------
segmented_sum      : sum per segment; differentiable w.r.t. x.
segmented_dot      : per-segment dot product; differentiable w.r.t. x, y.
segmented_mul      : per-element scale by per-segment scalar; d.w.r.t. x, y.
segmented_mean     : per-segment mean; differentiable w.r.t. x.
segmented_rms_norm : RMS vector norm per segment; differentiable w.r.t. x.
segmented_matvec   : per-segment matrix-vector multiply; d.w.r.t. v, m.
"""

from __future__ import annotations

import torch
import warp as wp

from nvalchemiops.segment_ops import (
    segmented_dot as _wp_segmented_dot,
)
from nvalchemiops.segment_ops import (
    segmented_matvec as _wp_segmented_matvec,
)
from nvalchemiops.segment_ops import (
    segmented_mean as _wp_segmented_mean,
)
from nvalchemiops.segment_ops import (
    segmented_mul as _wp_segmented_mul,
)
from nvalchemiops.segment_ops import (
    segmented_sum as _wp_segmented_sum,
)
from nvalchemiops.segment_ops_backward import (
    _launch_segmented_dot_backward,
    _launch_segmented_dot_double_backward,
    _launch_segmented_matvec_backward,
    _launch_segmented_matvec_double_backward,
    _launch_segmented_mean_backward,
    _launch_segmented_mean_double_backward,
    _launch_segmented_mul_backward,
    _launch_segmented_mul_double_backward,
    _launch_segmented_rms_norm_backward,
    _launch_segmented_rms_norm_double_backward,
    _launch_segmented_rms_norm_forward_precompute,
    _launch_segmented_sum_backward,
    _launch_segmented_sum_double_backward,
)
from nvalchemiops.torch.autograd import warp_stream_from_torch

# from nvalchemiops.torch.types import get_wp_dtype, get_wp_mat_dtype  #, get_wp_vec_dtype

__all__ = [
    # torch.autograd.Function classes — callers may use ``.apply`` directly.
    "SegmentedDot",
    "SegmentedMatvec",
    "SegmentedMean",
    "SegmentedMul",
    "SegmentedRmsNorm",
    "SegmentedSum",
    # Convenience wrappers around ``.apply``.
    "segmented_dot",
    "segmented_matvec",
    "segmented_mean",
    "segmented_mul",
    "segmented_rms_norm",
    "segmented_sum",
]

# =============================================================================
# Internal helpers
# =============================================================================

_VEC_DTYPE = {torch.float32: wp.vec3f, torch.float64: wp.vec3d}
_MAT_DTYPE = {torch.float32: wp.mat33f, torch.float64: wp.mat33d}
_SCALAR_DTYPE = {torch.float32: wp.float32, torch.float64: wp.float64}


def _infer_wp_dtype(t: torch.Tensor):
    if t.ndim == 3 and t.shape[-2:] == (3, 3):
        return _MAT_DTYPE[t.dtype]
    if t.ndim == 2 and t.shape[-1] == 3:
        return _VEC_DTYPE[t.dtype]
    return _SCALAR_DTYPE[t.dtype]


def _inp(t: torch.Tensor) -> wp.array:
    """Read-only contiguous Warp view (no grad tracking)."""
    return wp.from_torch(t.contiguous().detach(), dtype=_infer_wp_dtype(t))


def _inp_int(t: torch.Tensor) -> wp.array:
    return wp.from_torch(t.contiguous().detach(), dtype=wp.int32)


def _out(t: torch.Tensor) -> wp.array:
    """Writable Warp view of a freshly-allocated tensor (shared memory)."""
    return wp.from_torch(t, dtype=_infer_wp_dtype(t))


def _out_int(t: torch.Tensor) -> wp.array:
    return wp.from_torch(t, dtype=wp.int32)


# =============================================================================
# segmented_sum
# =============================================================================


class SegmentedSumBwd(torch.autograd.Function):
    """Differentiable backward of segmented_sum.

    Forward : ``grad_x[i] = g_out[idx[i]]``    (broadcast — linear in g_out)
    Backward: ``grad_g_out[s] = sum_i gg_x[i]``  (scatter-sum — same as fwd)
    """

    @staticmethod
    def forward(
        g_out: torch.Tensor, idx: torch.Tensor, num_segments: int
    ) -> torch.Tensor:
        """Launch the segmented_sum first-order backward kernel."""
        N = idx.shape[0]
        out_shape = (N, 3) if g_out.ndim == 2 else (N,)
        grad_x = g_out.new_zeros(out_shape)
        with warp_stream_from_torch(g_out):
            _launch_segmented_sum_backward(_inp(g_out), _inp_int(idx), _out(grad_x))
        return grad_x

    @staticmethod
    def setup_context(ctx, inputs, output):
        """Save the cotangents and saved state needed by the segmented_sum double-backward."""
        g_out, idx, num_segments = inputs
        ctx.save_for_backward(idx)
        ctx.num_segments = num_segments

    @staticmethod
    def backward(ctx, gg_x: torch.Tensor):
        """Launch the segmented_sum double-backward kernel."""
        (idx,) = ctx.saved_tensors
        num_segments = ctx.num_segments
        out_shape = (num_segments, 3) if gg_x.ndim == 2 else (num_segments,)
        grad_g_out = gg_x.new_zeros(out_shape)
        with warp_stream_from_torch(gg_x):
            _launch_segmented_sum_double_backward(
                _inp(gg_x), _inp_int(idx), num_segments, _out(grad_g_out)
            )
        return grad_g_out, None, None


class SegmentedSum(torch.autograd.Function):
    """``torch.autograd.Function`` for the segmented sum.

    Forward signature is ``apply(x, idx, num_segments) -> out`` where ``x`` is the per-element
    tensor and ``out[s]`` is the sum of entries with ``idx[i] == s``.  First-order
    backward (``out → x``) is a gather; the gather is itself wrapped in
    :class:`SegmentedSumBwd`, so :func:`torch.autograd.grad` over the gather's
    output produces the second-order adjoint (a scatter-sum).  Index inputs are
    non-differentiable and receive ``None`` gradient slots.

    Prefer :func:`segmented_sum` in user code; this class is exposed for callers
    that want to invoke :py:meth:`apply` directly to avoid the wrapper.
    """

    @staticmethod
    def forward(x: torch.Tensor, idx: torch.Tensor, num_segments: int) -> torch.Tensor:
        """Launch the segmented_sum forward Warp kernel."""
        out_shape = (num_segments, 3) if x.ndim == 2 else (num_segments,)
        out = x.new_zeros(out_shape)
        with warp_stream_from_torch(x):
            _wp_segmented_sum(_inp(x), _inp_int(idx), _out(out))
        return out

    @staticmethod
    def setup_context(ctx, inputs, output):
        """Save tensors and ``num_segments`` needed by the segmented_sum backward."""
        _, idx, num_segments = inputs
        ctx.save_for_backward(idx)
        ctx.num_segments = num_segments

    @staticmethod
    def backward(ctx, g_out: torch.Tensor):
        """Dispatch to the paired ``SegmentedSumBwd`` to produce gradients (differentiable)."""
        (idx,) = ctx.saved_tensors
        num_segments = ctx.num_segments
        grad_x = SegmentedSumBwd.apply(g_out.contiguous(), idx, num_segments)
        return grad_x, None, None


def segmented_sum(
    x: torch.Tensor, idx: torch.Tensor, num_segments: int
) -> torch.Tensor:
    """Differentiable segmented sum.

    Parameters
    ----------
    x : torch.Tensor
        Shape ``(N,)`` or ``(N, 3)``.  dtype float32 or float64.
    idx : torch.Tensor
        Shape ``(N,)``, dtype int32.  Sorted segment indices in ``[0, num_segments)``.
    num_segments : int
        Number of segments.

    Returns
    -------
    torch.Tensor
        Shape ``(num_segments,)`` or ``(num_segments, 3)``.
    """
    return SegmentedSum.apply(x, idx, num_segments)


# =============================================================================
# segmented_dot
# =============================================================================


class SegmentedDotBwd(torch.autograd.Function):
    """Differentiable backward of segmented_dot.

    Forward:
        grad_x[i] = g_out[s] * y[i]
        grad_y[i] = g_out[s] * x[i]
    Backward (double-backward):
        grad_g_out[s]  = sum_i dot(gg_gx[i], y[i]) + sum_i dot(gg_gy[i], x[i])
        grad_x_extra[i] = gg_gy[i] * g_out[s]
        grad_y_extra[i] = gg_gx[i] * g_out[s]
    """

    @staticmethod
    def forward(
        g_out: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        idx: torch.Tensor,
    ):
        """Launch the segmented_dot first-order backward kernel."""
        N = x.shape[0]
        out_shape = (N, 3) if x.ndim == 2 else (N,)
        grad_x = x.new_zeros(out_shape)
        grad_y = y.new_zeros(out_shape)
        with warp_stream_from_torch(g_out):
            _launch_segmented_dot_backward(
                _inp(g_out), _inp(x), _inp(y), _inp_int(idx), _out(grad_x), _out(grad_y)
            )
        return grad_x, grad_y

    @staticmethod
    def setup_context(ctx, inputs, output):
        """Save the cotangents and saved state needed by the segmented_dot double-backward."""
        g_out, x, y, idx = inputs
        ctx.save_for_backward(g_out, x, y, idx)
        ctx.num_segments = g_out.shape[0]

    @staticmethod
    def backward(ctx, gg_gx: torch.Tensor, gg_gy: torch.Tensor):
        """Launch the segmented_dot double-backward kernel."""
        g_out, x, y, idx = ctx.saved_tensors
        num_segments = ctx.num_segments
        grad_g_out = g_out.new_zeros((num_segments,))
        grad_x_extra = x.new_zeros(x.shape)
        grad_y_extra = y.new_zeros(y.shape)
        with warp_stream_from_torch(gg_gx):
            _launch_segmented_dot_double_backward(
                _inp(gg_gx),
                _inp(gg_gy),
                _inp(g_out),
                _inp(x),
                _inp(y),
                _inp_int(idx),
                num_segments,
                _out(grad_g_out),
                _out(grad_x_extra),
                _out(grad_y_extra),
            )
        return grad_g_out, grad_x_extra, grad_y_extra, None


class SegmentedDot(torch.autograd.Function):
    """``torch.autograd.Function`` for the per-segment dot product.

    Forward signature is ``apply(x, y, idx, num_segments) -> out`` where
    ``out[s] = sum_{i: idx[i]==s} dot(x[i], y[i])``.  Backward returns
    ``(grad_x, grad_y, None, None)``; the gradient path is wrapped in
    :class:`SegmentedDotBwd`, enabling clean double-backward.

    Prefer :func:`segmented_dot` in user code; this class is exposed for callers
    that want to invoke :py:meth:`apply` directly to avoid the wrapper.
    """

    @staticmethod
    def forward(
        x: torch.Tensor, y: torch.Tensor, idx: torch.Tensor, num_segments: int
    ) -> torch.Tensor:
        """Launch the segmented_dot forward Warp kernel."""
        out = x.new_zeros((num_segments,))
        with warp_stream_from_torch(x):
            _wp_segmented_dot(_inp(x), _inp(y), _inp_int(idx), _out(out))
        return out

    @staticmethod
    def setup_context(ctx, inputs, output):
        """Save tensors and ``num_segments`` needed by the segmented_dot backward."""
        x, y, idx, num_segments = inputs
        ctx.save_for_backward(x, y, idx)
        ctx.num_segments = num_segments

    @staticmethod
    def backward(ctx, g_out: torch.Tensor):
        """Dispatch to the paired ``SegmentedDotBwd`` to produce gradients (differentiable)."""
        x, y, idx = ctx.saved_tensors
        grad_x, grad_y = SegmentedDotBwd.apply(g_out.contiguous(), x, y, idx)
        return grad_x, grad_y, None, None


def segmented_dot(
    x: torch.Tensor, y: torch.Tensor, idx: torch.Tensor, num_segments: int
) -> torch.Tensor:
    """Differentiable per-segment dot product.

    ``out[s] = sum_i dot(x[i], y[i])``

    Parameters
    ----------
    x, y : torch.Tensor
        Shape ``(N,)`` or ``(N, 3)``.  Same dtype and device.
    idx : torch.Tensor
        Shape ``(N,)``, dtype int32.
    num_segments : int
        Number of segments.

    Returns
    -------
    torch.Tensor
        Shape ``(num_segments,)`` — scalar per segment.
    """
    return SegmentedDot.apply(x, y, idx, num_segments)


# =============================================================================
# segmented_mul   (x: vec3 or scalar, y: per-segment scalar)
# =============================================================================


class SegmentedMulBwd(torch.autograd.Function):
    """Differentiable backward of segmented_mul.

    Forward:
        grad_x[i] = g_out[i] * y[s]
        grad_y[s] = sum_i dot(g_out[i], x[i])
    Backward (double-backward):
        grad_g_out[i]   = gg_gx[i]*y[s] + gg_gy[s]*x[i]
        grad_x_extra[i] = gg_gy[s] * g_out[i]
        grad_y_extra[s] = sum_i dot(gg_gx[i], g_out[i])
    """

    @staticmethod
    def forward(
        g_out: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        idx: torch.Tensor,
        num_segments: int,
    ):
        # N = x.shape[0]
        """Launch the segmented_mul first-order backward kernel."""
        grad_x = x.new_zeros(x.shape)
        grad_y = y.new_zeros((num_segments,))
        with warp_stream_from_torch(g_out):
            _launch_segmented_mul_backward(
                _inp(g_out),
                _inp(x),
                _inp(y),
                _inp_int(idx),
                num_segments,
                _out(grad_x),
                _out(grad_y),
            )
        return grad_x, grad_y

    @staticmethod
    def setup_context(ctx, inputs, output):
        """Save the cotangents and saved state needed by the segmented_mul double-backward."""
        g_out, x, y, idx, num_segments = inputs
        ctx.save_for_backward(g_out, x, y, idx)
        ctx.num_segments = num_segments

    @staticmethod
    def backward(ctx, gg_gx: torch.Tensor, gg_gy: torch.Tensor):
        """Launch the segmented_mul double-backward kernel."""
        g_out, x, y, idx = ctx.saved_tensors
        num_segments = ctx.num_segments
        grad_g_out = g_out.new_zeros(g_out.shape)
        grad_x_extra = x.new_zeros(x.shape)
        grad_y_extra = y.new_zeros((num_segments,))
        with warp_stream_from_torch(gg_gx):
            _launch_segmented_mul_double_backward(
                _inp(gg_gx),
                _inp(gg_gy),
                _inp(g_out),
                _inp(x),
                _inp(y),
                _inp_int(idx),
                _out(grad_g_out),
                _out(grad_x_extra),
                _out(grad_y_extra),
            )
        return grad_g_out, grad_x_extra, grad_y_extra, None, None


class SegmentedMul(torch.autograd.Function):
    """``torch.autograd.Function`` for ``out[i] = x[i] * y[idx[i]]``.

    Forward signature is ``apply(x, y, idx, num_segments) -> out``: each element of ``x`` is
    scaled by the per-segment scalar in ``y``.  Backward returns
    ``(grad_x, grad_y, None, None)`` via :class:`SegmentedMulBwd`, which is
    itself differentiable so double-backward works through both leaves.

    Prefer :func:`segmented_mul` in user code; this class is exposed for callers
    that want to invoke :py:meth:`apply` directly to avoid the wrapper.
    """

    @staticmethod
    def forward(
        x: torch.Tensor, y: torch.Tensor, idx: torch.Tensor, num_segments: int
    ) -> torch.Tensor:
        """Launch the segmented_mul forward Warp kernel."""
        out = x.new_zeros(x.shape)
        with warp_stream_from_torch(x):
            _wp_segmented_mul(_inp(x), _inp(y), _inp_int(idx), _out(out))
        return out

    @staticmethod
    def setup_context(ctx, inputs, output):
        """Save tensors and ``num_segments`` needed by the segmented_mul backward."""
        x, y, idx, num_segments = inputs
        ctx.save_for_backward(x, y, idx)
        ctx.num_segments = num_segments

    @staticmethod
    def backward(ctx, g_out: torch.Tensor):
        """Dispatch to the paired ``SegmentedMulBwd`` to produce gradients (differentiable)."""
        x, y, idx = ctx.saved_tensors
        num_segments = ctx.num_segments
        grad_x, grad_y = SegmentedMulBwd.apply(
            g_out.contiguous(), x, y, idx, num_segments
        )
        return grad_x, grad_y, None, None


def segmented_mul(
    x: torch.Tensor, y: torch.Tensor, idx: torch.Tensor, num_segments: int
) -> torch.Tensor:
    """Differentiable per-element scale by a per-segment scalar.

    ``out[i] = x[i] * y[idx[i]]``

    Parameters
    ----------
    x : torch.Tensor
        Shape ``(N,)`` or ``(N, 3)``.
    y : torch.Tensor
        Shape ``(num_segments,)`` — one scalar per segment.
    idx : torch.Tensor
        Shape ``(N,)``, dtype int32.
    num_segments : int
        Number of segments.

    Returns
    -------
    torch.Tensor
        Same shape as ``x``.
    """
    return SegmentedMul.apply(x, y, idx, num_segments)


# =============================================================================
# segmented_mean
# =============================================================================


class SegmentedMeanBwd(torch.autograd.Function):
    """Differentiable backward of segmented_mean.

    Forward : ``grad_x[i] = g_out[s] / count[s]``  (linear in g_out)
    Backward: ``grad_g_out[s] = sum_i gg_x[i] / count[s]``  (mean of gg_x)
    """

    @staticmethod
    def forward(
        g_out: torch.Tensor, counts: torch.Tensor, idx: torch.Tensor
    ) -> torch.Tensor:
        """Launch the segmented_mean first-order backward kernel."""
        N = idx.shape[0]
        out_shape = (N, 3) if g_out.ndim == 2 else (N,)
        grad_x = g_out.new_zeros(out_shape)
        with warp_stream_from_torch(g_out):
            _launch_segmented_mean_backward(
                _inp(g_out), _inp_int(counts), _inp_int(idx), _out(grad_x)
            )
        return grad_x

    @staticmethod
    def setup_context(ctx, inputs, output):
        """Save the cotangents and saved state needed by the segmented_mean double-backward."""
        g_out, counts, idx = inputs
        ctx.save_for_backward(counts, idx)
        ctx.num_segments = g_out.shape[0]

    @staticmethod
    def backward(ctx, gg_x: torch.Tensor):
        """Launch the segmented_mean double-backward kernel."""
        counts, idx = ctx.saved_tensors
        num_segments = ctx.num_segments
        out_shape = (num_segments, 3) if gg_x.ndim == 2 else (num_segments,)
        grad_g_out = gg_x.new_zeros(out_shape)
        with warp_stream_from_torch(gg_x):
            _launch_segmented_mean_double_backward(
                _inp(gg_x), _inp_int(counts), _inp_int(idx), _out(grad_g_out)
            )
        return grad_g_out, None, None


class SegmentedMean(torch.autograd.Function):
    """``torch.autograd.Function`` for the per-segment mean.

    Forward signature is ``apply(x, idx, num_segments) -> (out, counts)``: ``out[s]`` is the
    mean of entries with ``idx[i] == s`` and ``counts[s]`` is the per-segment
    population (returned so callers and the saved-state path don't recompute).
    Only ``out`` carries a gradient; ``counts`` is integer and non-differentiable.

    Prefer :func:`segmented_mean` in user code; this class is exposed for callers
    that want to invoke :py:meth:`apply` directly to avoid the wrapper.
    """

    @staticmethod
    def forward(x: torch.Tensor, idx: torch.Tensor, num_segments: int):
        """Launch the segmented_mean forward Warp kernel."""
        out_shape = (num_segments, 3) if x.ndim == 2 else (num_segments,)
        out = x.new_zeros(out_shape)
        sums = x.new_zeros(out_shape)
        counts = x.new_zeros((num_segments,), dtype=torch.int32)
        with warp_stream_from_torch(x):
            _wp_segmented_mean(
                _inp(x), _inp_int(idx), _out(sums), _out_int(counts), _out(out)
            )
        return out, counts

    @staticmethod
    def setup_context(ctx, inputs, output):
        """Save tensors and ``num_segments`` needed by the segmented_mean backward."""
        _, idx, num_segments = inputs
        out, counts = output
        ctx.save_for_backward(counts, idx)
        ctx.num_segments = num_segments
        # ``counts`` is saved state, not a differentiable output.  Marking it
        # non-differentiable makes ``counts.requires_grad`` False so a caller
        # that tries to backprop through it gets a clear ``RuntimeError`` rather
        # than a silently dropped gradient.
        ctx.mark_non_differentiable(counts)

    @staticmethod
    def backward(ctx, g_out: torch.Tensor, _g_counts):
        """Dispatch to the paired ``SegmentedMeanBwd`` to produce gradients (differentiable)."""
        counts, idx = ctx.saved_tensors
        grad_x = SegmentedMeanBwd.apply(g_out.contiguous(), counts, idx)
        return grad_x, None, None


def segmented_mean(
    x: torch.Tensor, idx: torch.Tensor, num_segments: int
) -> torch.Tensor:
    """Differentiable per-segment mean.

    ``out[s] = mean(x[i] for i in segment s)``

    Parameters
    ----------
    x : torch.Tensor
        Shape ``(N,)`` or ``(N, 3)``.
    idx : torch.Tensor
        Shape ``(N,)``, dtype int32.  Sorted.
    num_segments : int
        Number of segments.

    Returns
    -------
    torch.Tensor
        Shape ``(num_segments,)`` or ``(num_segments, 3)``.
    """
    out, _ = SegmentedMean.apply(x, idx, num_segments)
    return out


# =============================================================================
# segmented_rms_norm
# =============================================================================


class SegmentedRmsNormBwd(torch.autograd.Function):
    """Differentiable backward of segmented_rms_norm.

    Forward : ``grad_x[i] = g_out[s] * x[i] * inv_norm[s]``
    Backward (double-backward):
        ``inner[s] = sum_i dot(gg_x[i], x[i])``
        ``grad_g_out[s]  = inner[s] * inv_norm[s]``
        ``grad_x_extra[i] = g_out[s]*inv_norm[s]*gg_x[i]
                           - g_out[s]*inv_norm[s]^3*count[s]*inner[s]*x[i]``
    """

    @staticmethod
    def forward(
        g_out: torch.Tensor,
        x: torch.Tensor,
        inv_norm: torch.Tensor,
        counts: torch.Tensor,
        idx: torch.Tensor,
    ) -> torch.Tensor:
        """Launch the segmented_rms_norm first-order backward kernel."""
        grad_x = x.new_zeros(x.shape)
        with warp_stream_from_torch(g_out):
            _launch_segmented_rms_norm_backward(
                _inp(g_out), _inp(x), _inp(inv_norm), _inp_int(idx), _out(grad_x)
            )
        return grad_x

    @staticmethod
    def setup_context(ctx, inputs, output):
        """Save the cotangents and saved state needed by the segmented_rms_norm double-backward."""
        g_out, x, inv_norm, counts, idx = inputs
        ctx.save_for_backward(g_out, x, inv_norm, counts, idx)
        ctx.num_segments = g_out.shape[0]

    @staticmethod
    def backward(ctx, gg_x: torch.Tensor):
        """Launch the segmented_rms_norm double-backward kernel."""
        g_out, x, inv_norm, counts, idx = ctx.saved_tensors
        num_segments = ctx.num_segments
        grad_x_extra = x.new_zeros(x.shape)
        grad_g_out = g_out.new_zeros((num_segments,))
        with warp_stream_from_torch(gg_x):
            _launch_segmented_rms_norm_double_backward(
                _inp(gg_x),
                _inp(x),
                _inp(g_out),
                _inp(inv_norm),
                _inp_int(counts),
                _inp_int(idx),
                num_segments,
                _out(grad_x_extra),
                _out(grad_g_out),
            )
        return grad_g_out, grad_x_extra, None, None, None


class SegmentedRmsNorm(torch.autograd.Function):
    """``torch.autograd.Function`` for the per-segment RMS vector norm.

    Forward signature is ``apply(x, idx, num_segments) -> (out, inv_norm, counts)``: the
    forward kernel takes the precompute path so backward can consume the saved
    ``inv_norm`` and ``counts`` without recomputing them.  Only ``out`` carries
    a gradient; ``inv_norm`` and ``counts`` are returned for inspection / reuse.

    Prefer :func:`segmented_rms_norm` in user code; this class is exposed for
    callers that want to invoke :py:meth:`apply` directly to avoid the wrapper.
    """

    @staticmethod
    def forward(x: torch.Tensor, idx: torch.Tensor, num_segments: int):
        """Launch the segmented_rms_norm forward Warp kernel."""
        out = x.new_zeros((num_segments,))
        sum_sq = x.new_zeros((num_segments,))
        counts = x.new_zeros((num_segments,), dtype=torch.int32)
        inv_norm = x.new_zeros((num_segments,))
        with warp_stream_from_torch(x):
            _launch_segmented_rms_norm_forward_precompute(
                _inp(x),
                _inp_int(idx),
                _out(sum_sq),
                _out_int(counts),
                _out(out),
                _out(inv_norm),
            )
        return out, inv_norm, counts

    @staticmethod
    def setup_context(ctx, inputs, output):
        """Save tensors and ``num_segments`` needed by the segmented_rms_norm backward."""
        x, idx, num_segments = inputs
        out, inv_norm, counts = output
        ctx.save_for_backward(x, inv_norm, counts, idx)
        ctx.num_segments = num_segments
        # ``inv_norm`` and ``counts`` are saved state, not first-class
        # differentiable outputs — backward only consumes the cotangent of
        # ``out`` and discards the other slots.  Marking them
        # non-differentiable sets ``requires_grad=False`` on both, so a caller
        # that tries ``inv_norm.sum().backward()`` gets a clear ``RuntimeError``
        # rather than a silently dropped VJP into ``x``.
        ctx.mark_non_differentiable(inv_norm, counts)

    @staticmethod
    def backward(ctx, g_out: torch.Tensor, _g_inv_norm, _g_counts):
        """Dispatch to the paired ``SegmentedRmsNormBwd`` to produce gradients (differentiable)."""
        x, inv_norm, counts, idx = ctx.saved_tensors
        grad_x = SegmentedRmsNormBwd.apply(g_out.contiguous(), x, inv_norm, counts, idx)
        return grad_x, None, None


def segmented_rms_norm(
    x: torch.Tensor, idx: torch.Tensor, num_segments: int
) -> torch.Tensor:
    """Differentiable per-segment RMS vector norm.

    ``out[s] = sqrt(mean(||x[i]||^2 for i in segment s))``

    Parameters
    ----------
    x : torch.Tensor
        Shape ``(N, 3)``.  dtype float32 or float64.
    idx : torch.Tensor
        Shape ``(N,)``, dtype int32.  Sorted.
    num_segments : int
        Number of segments.

    Returns
    -------
    torch.Tensor
        Shape ``(num_segments,)`` — scalar RMS norm per segment.
    """
    out, _, _ = SegmentedRmsNorm.apply(x, idx, num_segments)
    return out


# =============================================================================
# segmented_matvec
# =============================================================================


class SegmentedMatvecBwd(torch.autograd.Function):
    """Differentiable backward of segmented_matvec.

    Forward convention: ``out[i] = num_segments[s]^T @ v[i]``

    Backward:
        grad_v[i]    = num_segments[s] @ g_out[i]
        grad_m[s]    = sum_i outer(v[i], g_out[i])
    Double-backward (w.r.t. gg_gv, gg_gm):
        grad_g_out[i]   = num_segments[s]^T @ gg_gv[i] + gg_gm[s]^T @ v[i]
        grad_v_extra[i] = gg_gm[s] @ g_out[i]
        grad_m_extra[s] = sum_i outer(gg_gv[i], g_out[i])
    """

    @staticmethod
    def forward(
        g_out: torch.Tensor,
        v: torch.Tensor,
        m: torch.Tensor,
        idx: torch.Tensor,
        num_segments: int,
    ):
        """Launch the segmented_matvec first-order backward kernel."""
        grad_v = v.new_zeros(v.shape)
        grad_m = m.new_zeros(m.shape)
        with warp_stream_from_torch(g_out):
            _launch_segmented_matvec_backward(
                _inp(g_out),
                _inp(v),
                _inp(m),
                _inp_int(idx),
                _out(grad_v),
                _out(grad_m),
            )
        return grad_v, grad_m

    @staticmethod
    def setup_context(ctx, inputs, output):
        """Save the cotangents and saved state needed by the segmented_matvec double-backward."""
        g_out, v, m, idx, num_segments = inputs
        ctx.save_for_backward(g_out, v, m, idx)
        ctx.num_segments = num_segments

    @staticmethod
    def backward(ctx, gg_gv: torch.Tensor, gg_gm: torch.Tensor):
        """Launch the segmented_matvec double-backward kernel."""
        g_out, v, m, idx = ctx.saved_tensors
        # num_segments = ctx.num_segments
        grad_g_out = g_out.new_zeros(g_out.shape)
        grad_v_extra = v.new_zeros(v.shape)
        grad_m_extra = m.new_zeros(m.shape)
        with warp_stream_from_torch(gg_gv):
            _launch_segmented_matvec_double_backward(
                _inp(gg_gv),
                _inp(gg_gm),
                _inp(g_out),
                _inp(v),
                _inp(m),
                _inp_int(idx),
                _out(grad_g_out),
                _out(grad_v_extra),
                _out(grad_m_extra),
            )
        return grad_g_out, grad_v_extra, grad_m_extra, None, None


class SegmentedMatvec(torch.autograd.Function):
    """``torch.autograd.Function`` for ``out[i] = m[idx[i]]^T @ v[i]``.

    Forward signature is ``apply(v, m, idx, num_segments) -> out``: each per-atom vector is
    transformed by the matrix assigned to its segment.  Backward returns
    ``(grad_v, grad_m, None, None)`` via :class:`SegmentedMatvecBwd`, which is
    itself differentiable so double-backward works through both leaves.

    Prefer :func:`segmented_matvec` in user code; this class is exposed for
    callers that want to invoke :py:meth:`apply` directly to avoid the wrapper.
    """

    @staticmethod
    def forward(v: torch.Tensor, m: torch.Tensor, idx: torch.Tensor, num_segments: int):
        """Launch the segmented_matvec forward Warp kernel."""
        out = v.new_zeros(v.shape)
        with warp_stream_from_torch(v):
            _wp_segmented_matvec(_inp(v), _inp(m), _inp_int(idx), _out(out))
        return out

    @staticmethod
    def setup_context(ctx, inputs, output):
        """Save tensors and ``num_segments`` needed by the segmented_matvec backward."""
        v, m, idx, num_segments = inputs
        ctx.save_for_backward(v, m, idx)
        ctx.num_segments = num_segments

    @staticmethod
    def backward(ctx, g_out: torch.Tensor):
        """Dispatch to the paired ``SegmentedMatvecBwd`` to produce gradients (differentiable)."""
        v, m, idx = ctx.saved_tensors
        num_segments = ctx.num_segments
        grad_v, grad_m = SegmentedMatvecBwd.apply(
            g_out.contiguous(), v, m, idx, num_segments
        )
        return grad_v, grad_m, None, None


def segmented_matvec(
    v: torch.Tensor, m: torch.Tensor, idx: torch.Tensor, num_segments: int
) -> torch.Tensor:
    """Differentiable per-segment matrix-vector multiply.

    ``out[i] = m[idx[i]]^T @ v[i]``

    Parameters
    ----------
    v : torch.Tensor
        Shape ``(N, 3)``.
    m : torch.Tensor
        Shape ``(num_segments, 3, 3)`` — one matrix per segment.
    idx : torch.Tensor
        Shape ``(N,)``, dtype int32.
    num_segments : int
        Number of segments.

    Returns
    -------
    torch.Tensor
        Shape ``(N, 3)``.
    """
    return SegmentedMatvec.apply(v, m, idx, num_segments)
