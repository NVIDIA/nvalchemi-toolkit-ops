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
``M``) always receives ``None`` gradient.

Tensor layout conventions
-------------------------
- Scalar arrays  : shape ``(N,)`` or ``(M,)``
- Vec3 arrays    : shape ``(N, 3)`` or ``(M, 3)``
- Mat33 arrays   : shape ``(M, 3, 3)``

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


def _zeros(shape, ref: torch.Tensor) -> torch.Tensor:
    return torch.zeros(shape, dtype=ref.dtype, device=ref.device)


def _zeros_int(shape, ref: torch.Tensor) -> torch.Tensor:
    return torch.zeros(shape, dtype=torch.int32, device=ref.device)


# =============================================================================
# segmented_sum
# =============================================================================


class _SegmentedSumBwd(torch.autograd.Function):
    """Differentiable backward of segmented_sum.

    Forward : ``grad_x[i] = g_out[idx[i]]``    (broadcast — linear in g_out)
    Backward: ``grad_g_out[s] = sum_i gg_x[i]``  (scatter-sum — same as fwd)
    """

    @staticmethod
    def forward(g_out: torch.Tensor, idx: torch.Tensor, M: int) -> torch.Tensor:
        N = idx.shape[0]
        out_shape = (N, 3) if g_out.ndim == 2 else (N,)
        grad_x = _zeros(out_shape, g_out)
        with warp_stream_from_torch(g_out):
            _launch_segmented_sum_backward(_inp(g_out), _inp_int(idx), _out(grad_x))
        return grad_x

    @staticmethod
    def setup_context(ctx, inputs, output):
        g_out, idx, M = inputs
        ctx.save_for_backward(idx)
        ctx.M = M

    @staticmethod
    def backward(ctx, gg_x: torch.Tensor):
        (idx,) = ctx.saved_tensors
        M = ctx.M
        out_shape = (M, 3) if gg_x.ndim == 2 else (M,)
        grad_g_out = _zeros(out_shape, gg_x)
        with warp_stream_from_torch(gg_x):
            _launch_segmented_sum_double_backward(
                _inp(gg_x), _inp_int(idx), M, _out(grad_g_out)
            )
        return grad_g_out, None, None


class _SegmentedSum(torch.autograd.Function):
    @staticmethod
    def forward(x: torch.Tensor, idx: torch.Tensor, M: int) -> torch.Tensor:
        out_shape = (M, 3) if x.ndim == 2 else (M,)
        out = _zeros(out_shape, x)
        with warp_stream_from_torch(x):
            _wp_segmented_sum(_inp(x), _inp_int(idx), _out(out))
        return out

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, idx, M = inputs
        ctx.save_for_backward(idx)
        ctx.M = M

    @staticmethod
    def backward(ctx, g_out: torch.Tensor):
        (idx,) = ctx.saved_tensors
        M = ctx.M
        grad_x = _SegmentedSumBwd.apply(g_out.contiguous(), idx, M)
        return grad_x, None, None


def segmented_sum(x: torch.Tensor, idx: torch.Tensor, M: int) -> torch.Tensor:
    """Differentiable segmented sum.

    Parameters
    ----------
    x : torch.Tensor
        Shape ``(N,)`` or ``(N, 3)``.  dtype float32 or float64.
    idx : torch.Tensor
        Shape ``(N,)``, dtype int32.  Sorted segment indices in ``[0, M)``.
    M : int
        Number of segments.

    Returns
    -------
    torch.Tensor
        Shape ``(M,)`` or ``(M, 3)``.
    """
    return _SegmentedSum.apply(x, idx, M)


# =============================================================================
# segmented_dot
# =============================================================================


class _SegmentedDotBwd(torch.autograd.Function):
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
        N = x.shape[0]
        out_shape = (N, 3) if x.ndim == 2 else (N,)
        grad_x = _zeros(out_shape, x)
        grad_y = _zeros(out_shape, y)
        with warp_stream_from_torch(g_out):
            _launch_segmented_dot_backward(
                _inp(g_out), _inp(x), _inp(y), _inp_int(idx), _out(grad_x), _out(grad_y)
            )
        return grad_x, grad_y

    @staticmethod
    def setup_context(ctx, inputs, output):
        g_out, x, y, idx = inputs
        ctx.save_for_backward(g_out, x, y, idx)
        ctx.M = g_out.shape[0]

    @staticmethod
    def backward(ctx, gg_gx: torch.Tensor, gg_gy: torch.Tensor):
        g_out, x, y, idx = ctx.saved_tensors
        M = ctx.M
        grad_g_out = _zeros((M,), g_out)
        grad_x_extra = _zeros(x.shape, x)
        grad_y_extra = _zeros(y.shape, y)
        with warp_stream_from_torch(gg_gx):
            _launch_segmented_dot_double_backward(
                _inp(gg_gx),
                _inp(gg_gy),
                _inp(g_out),
                _inp(x),
                _inp(y),
                _inp_int(idx),
                M,
                _out(grad_g_out),
                _out(grad_x_extra),
                _out(grad_y_extra),
            )
        return grad_g_out, grad_x_extra, grad_y_extra, None


class _SegmentedDot(torch.autograd.Function):
    @staticmethod
    def forward(
        x: torch.Tensor, y: torch.Tensor, idx: torch.Tensor, M: int
    ) -> torch.Tensor:
        out = _zeros((M,), x)
        with warp_stream_from_torch(x):
            # scalar_dtype = _SCALAR_DTYPE[x.dtype]
            _wp_segmented_dot(_inp(x), _inp(y), _inp_int(idx), _out(out))
        return out

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, y, idx, M = inputs
        ctx.save_for_backward(x, y, idx)
        ctx.M = M

    @staticmethod
    def backward(ctx, g_out: torch.Tensor):
        x, y, idx = ctx.saved_tensors
        # M = ctx.M
        grad_x, grad_y = _SegmentedDotBwd.apply(g_out.contiguous(), x, y, idx)
        return grad_x, grad_y, None, None


def segmented_dot(
    x: torch.Tensor, y: torch.Tensor, idx: torch.Tensor, M: int
) -> torch.Tensor:
    """Differentiable per-segment dot product.

    ``out[s] = sum_i dot(x[i], y[i])``

    Parameters
    ----------
    x, y : torch.Tensor
        Shape ``(N,)`` or ``(N, 3)``.  Same dtype and device.
    idx : torch.Tensor
        Shape ``(N,)``, dtype int32.
    M : int
        Number of segments.

    Returns
    -------
    torch.Tensor
        Shape ``(M,)`` — scalar per segment.
    """
    return _SegmentedDot.apply(x, y, idx, M)


# =============================================================================
# segmented_mul   (x: vec3 or scalar, y: per-segment scalar)
# =============================================================================


class _SegmentedMulBwd(torch.autograd.Function):
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
        M: int,
    ):
        # N = x.shape[0]
        grad_x = _zeros(x.shape, x)
        grad_y = _zeros((M,), y)
        with warp_stream_from_torch(g_out):
            _launch_segmented_mul_backward(
                _inp(g_out),
                _inp(x),
                _inp(y),
                _inp_int(idx),
                M,
                _out(grad_x),
                _out(grad_y),
            )
        return grad_x, grad_y

    @staticmethod
    def setup_context(ctx, inputs, output):
        g_out, x, y, idx, M = inputs
        ctx.save_for_backward(g_out, x, y, idx)
        ctx.M = M

    @staticmethod
    def backward(ctx, gg_gx: torch.Tensor, gg_gy: torch.Tensor):
        g_out, x, y, idx = ctx.saved_tensors
        M = ctx.M
        grad_g_out = _zeros(g_out.shape, g_out)
        grad_x_extra = _zeros(x.shape, x)
        grad_y_extra = _zeros((M,), y)
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


class _SegmentedMul(torch.autograd.Function):
    @staticmethod
    def forward(
        x: torch.Tensor, y: torch.Tensor, idx: torch.Tensor, M: int
    ) -> torch.Tensor:
        out = _zeros(x.shape, x)
        with warp_stream_from_torch(x):
            _wp_segmented_mul(_inp(x), _inp(y), _inp_int(idx), _out(out))
        return out

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, y, idx, M = inputs
        ctx.save_for_backward(x, y, idx)
        ctx.M = M

    @staticmethod
    def backward(ctx, g_out: torch.Tensor):
        x, y, idx = ctx.saved_tensors
        M = ctx.M
        grad_x, grad_y = _SegmentedMulBwd.apply(g_out.contiguous(), x, y, idx, M)
        return grad_x, grad_y, None, None


def segmented_mul(
    x: torch.Tensor, y: torch.Tensor, idx: torch.Tensor, M: int
) -> torch.Tensor:
    """Differentiable per-element scale by a per-segment scalar.

    ``out[i] = x[i] * y[idx[i]]``

    Parameters
    ----------
    x : torch.Tensor
        Shape ``(N,)`` or ``(N, 3)``.
    y : torch.Tensor
        Shape ``(M,)`` — one scalar per segment.
    idx : torch.Tensor
        Shape ``(N,)``, dtype int32.
    M : int
        Number of segments.

    Returns
    -------
    torch.Tensor
        Same shape as ``x``.
    """
    return _SegmentedMul.apply(x, y, idx, M)


# =============================================================================
# segmented_mean
# =============================================================================


class _SegmentedMeanBwd(torch.autograd.Function):
    """Differentiable backward of segmented_mean.

    Forward : ``grad_x[i] = g_out[s] / count[s]``  (linear in g_out)
    Backward: ``grad_g_out[s] = sum_i gg_x[i] / count[s]``  (mean of gg_x)
    """

    @staticmethod
    def forward(
        g_out: torch.Tensor, counts: torch.Tensor, idx: torch.Tensor
    ) -> torch.Tensor:
        N = idx.shape[0]
        out_shape = (N, 3) if g_out.ndim == 2 else (N,)
        grad_x = _zeros(out_shape, g_out)
        with warp_stream_from_torch(g_out):
            _launch_segmented_mean_backward(
                _inp(g_out), _inp_int(counts), _inp_int(idx), _out(grad_x)
            )
        return grad_x

    @staticmethod
    def setup_context(ctx, inputs, output):
        g_out, counts, idx = inputs
        ctx.save_for_backward(counts, idx)
        ctx.M = g_out.shape[0]

    @staticmethod
    def backward(ctx, gg_x: torch.Tensor):
        counts, idx = ctx.saved_tensors
        M = ctx.M
        out_shape = (M, 3) if gg_x.ndim == 2 else (M,)
        grad_g_out = _zeros(out_shape, gg_x)
        with warp_stream_from_torch(gg_x):
            _launch_segmented_mean_double_backward(
                _inp(gg_x), _inp_int(counts), _inp_int(idx), _out(grad_g_out)
            )
        return grad_g_out, None, None


class _SegmentedMean(torch.autograd.Function):
    @staticmethod
    def forward(x: torch.Tensor, idx: torch.Tensor, M: int):
        out_shape = (M, 3) if x.ndim == 2 else (M,)
        out = _zeros(out_shape, x)
        sums = _zeros(out_shape, x)
        counts = _zeros_int((M,), x)
        with warp_stream_from_torch(x):
            _wp_segmented_mean(
                _inp(x), _inp_int(idx), _out(sums), _out_int(counts), _out(out)
            )
        return out, counts

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, idx, M = inputs
        out, counts = output
        ctx.save_for_backward(counts, idx)
        ctx.M = M

    @staticmethod
    def backward(ctx, g_out: torch.Tensor, _g_counts):
        counts, idx = ctx.saved_tensors
        # M = ctx.M
        grad_x = _SegmentedMeanBwd.apply(g_out.contiguous(), counts, idx)
        return grad_x, None, None


def segmented_mean(x: torch.Tensor, idx: torch.Tensor, M: int) -> torch.Tensor:
    """Differentiable per-segment mean.

    ``out[s] = mean(x[i] for i in segment s)``

    Parameters
    ----------
    x : torch.Tensor
        Shape ``(N,)`` or ``(N, 3)``.
    idx : torch.Tensor
        Shape ``(N,)``, dtype int32.  Sorted.
    M : int
        Number of segments.

    Returns
    -------
    torch.Tensor
        Shape ``(M,)`` or ``(M, 3)``.
    """
    out, _ = _SegmentedMean.apply(x, idx, M)
    return out


# =============================================================================
# segmented_rms_norm
# =============================================================================


class _SegmentedRmsNormBwd(torch.autograd.Function):
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
        grad_x = _zeros(x.shape, x)
        with warp_stream_from_torch(g_out):
            _launch_segmented_rms_norm_backward(
                _inp(g_out), _inp(x), _inp(inv_norm), _inp_int(idx), _out(grad_x)
            )
        return grad_x

    @staticmethod
    def setup_context(ctx, inputs, output):
        g_out, x, inv_norm, counts, idx = inputs
        ctx.save_for_backward(g_out, x, inv_norm, counts, idx)
        ctx.M = g_out.shape[0]

    @staticmethod
    def backward(ctx, gg_x: torch.Tensor):
        g_out, x, inv_norm, counts, idx = ctx.saved_tensors
        M = ctx.M
        grad_x_extra = _zeros(x.shape, x)
        grad_g_out = _zeros((M,), g_out)
        with warp_stream_from_torch(gg_x):
            _launch_segmented_rms_norm_double_backward(
                _inp(gg_x),
                _inp(x),
                _inp(g_out),
                _inp(inv_norm),
                _inp_int(counts),
                _inp_int(idx),
                M,
                _out(grad_x_extra),
                _out(grad_g_out),
            )
        return grad_g_out, grad_x_extra, None, None, None


class _SegmentedRmsNorm(torch.autograd.Function):
    @staticmethod
    def forward(x: torch.Tensor, idx: torch.Tensor, M: int):
        # scalar_dtype = x.dtype
        out = _zeros((M,), x)
        sum_sq = _zeros((M,), x)
        counts = _zeros_int((M,), x)
        inv_norm = _zeros((M,), x)
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
        x, idx, M = inputs
        out, inv_norm, counts = output
        ctx.save_for_backward(x, inv_norm, counts, idx)
        ctx.M = M

    @staticmethod
    def backward(ctx, g_out: torch.Tensor, _g_inv_norm, _g_counts):
        x, inv_norm, counts, idx = ctx.saved_tensors
        # M = ctx.M
        grad_x = _SegmentedRmsNormBwd.apply(
            g_out.contiguous(), x, inv_norm, counts, idx
        )
        return grad_x, None, None


def segmented_rms_norm(x: torch.Tensor, idx: torch.Tensor, M: int) -> torch.Tensor:
    """Differentiable per-segment RMS vector norm.

    ``out[s] = sqrt(mean(||x[i]||^2 for i in segment s))``

    Parameters
    ----------
    x : torch.Tensor
        Shape ``(N, 3)``.  dtype float32 or float64.
    idx : torch.Tensor
        Shape ``(N,)``, dtype int32.  Sorted.
    M : int
        Number of segments.

    Returns
    -------
    torch.Tensor
        Shape ``(M,)`` — scalar RMS norm per segment.
    """
    out, _, _ = _SegmentedRmsNorm.apply(x, idx, M)
    return out


# =============================================================================
# segmented_matvec
# =============================================================================


class _SegmentedMatvecBwd(torch.autograd.Function):
    """Differentiable backward of segmented_matvec.

    Forward convention: ``out[i] = M[s]^T @ v[i]``

    Backward:
        grad_v[i]    = M[s] @ g_out[i]
        grad_m[s]    = sum_i outer(v[i], g_out[i])
    Double-backward (w.r.t. gg_gv, gg_gm):
        grad_g_out[i]   = M[s]^T @ gg_gv[i] + gg_gm[s]^T @ v[i]
        grad_v_extra[i] = gg_gm[s] @ g_out[i]
        grad_m_extra[s] = sum_i outer(gg_gv[i], g_out[i])
    """

    @staticmethod
    def forward(
        g_out: torch.Tensor,
        v: torch.Tensor,
        m: torch.Tensor,
        idx: torch.Tensor,
        M: int,
    ):
        grad_v = _zeros(v.shape, v)
        grad_m = _zeros(m.shape, m)
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
        g_out, v, m, idx, M = inputs
        ctx.save_for_backward(g_out, v, m, idx)
        ctx.M = M

    @staticmethod
    def backward(ctx, gg_gv: torch.Tensor, gg_gm: torch.Tensor):
        g_out, v, m, idx = ctx.saved_tensors
        # M = ctx.M
        grad_g_out = _zeros(g_out.shape, g_out)
        grad_v_extra = _zeros(v.shape, v)
        grad_m_extra = _zeros(m.shape, m)
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


class _SegmentedMatvec(torch.autograd.Function):
    @staticmethod
    def forward(v: torch.Tensor, m: torch.Tensor, idx: torch.Tensor, M: int):
        out = _zeros(v.shape, v)
        with warp_stream_from_torch(v):
            _wp_segmented_matvec(_inp(v), _inp(m), _inp_int(idx), _out(out))
        return out

    @staticmethod
    def setup_context(ctx, inputs, output):
        v, m, idx, M = inputs
        ctx.save_for_backward(v, m, idx)
        ctx.M = M

    @staticmethod
    def backward(ctx, g_out: torch.Tensor):
        v, m, idx = ctx.saved_tensors
        M = ctx.M
        grad_v, grad_m = _SegmentedMatvecBwd.apply(g_out.contiguous(), v, m, idx, M)
        return grad_v, grad_m, None, None


def segmented_matvec(
    v: torch.Tensor, m: torch.Tensor, idx: torch.Tensor, M: int
) -> torch.Tensor:
    """Differentiable per-segment matrix-vector multiply.

    ``out[i] = M[idx[i]]^T @ v[i]``

    Parameters
    ----------
    v : torch.Tensor
        Shape ``(N, 3)``.
    m : torch.Tensor
        Shape ``(M, 3, 3)`` — one matrix per segment.
    idx : torch.Tensor
        Shape ``(N,)``, dtype int32.
    M : int
        Number of segments.

    Returns
    -------
    torch.Tensor
        Shape ``(N, 3)``.
    """
    return _SegmentedMatvec.apply(v, m, idx, M)
