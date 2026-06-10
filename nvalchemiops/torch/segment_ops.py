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

"""PyTorch autograd bindings for segment operations.

Each public function accepts PyTorch tensors and returns a PyTorch tensor with
full first-order and second-order backward support.  Integer metadata (``idx``,
``num_segments``) always receives ``None`` gradient.

Every op is wired through :func:`register_warp_op_chain`, so the Warp launches
are opaque ``torch.library`` custom ops: the bindings are ``torch.compile``-clean
(single-graph capturable, no graph breaks) and differentiable to second order.

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
    segmented_dot_backward,
    segmented_dot_double_backward,
    segmented_matvec_backward,
    segmented_matvec_double_backward,
    segmented_mean_backward,
    segmented_mean_double_backward,
    segmented_mul_backward,
    segmented_mul_double_backward,
    segmented_rms_norm_backward,
    segmented_rms_norm_double_backward,
    segmented_rms_norm_forward_precompute,
    segmented_sum_backward,
    segmented_sum_double_backward,
)
from nvalchemiops.torch._warp_op_helpers import (
    register_warp_op_chain,
    scoped_warp_stream,
)

# from nvalchemiops.torch.types import get_wp_dtype, get_wp_mat_dtype  #, get_wp_vec_dtype

__all__ = [
    # All six ops are ``register_warp_op_chain`` custom-op chains: opaque to
    # TorchDynamo (single-graph capturable) with full first- and second-order
    # autograd. Callers use these public functions; there are no longer any
    # ``torch.autograd.Function`` classes to invoke via ``.apply``.
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


def _validate_idx(idx: torch.Tensor, num_segments: int, op: str) -> None:
    """Validate segment-index metadata before dispatching to a Warp kernel.

    ``idx`` is used as a raw memory index inside the Warp kernels, so a stray
    value (wrong dtype, wrong rank, negative, or ``>= num_segments``) becomes
    an out-of-bounds memory access rather than a clear Python error.  This
    runs at the public-wrapper boundary so users get a typed exception before
    any device launch.

    Parameters
    ----------
    idx : torch.Tensor
        The segment-index tensor to validate.
    num_segments : int
        Declared number of segments.  Every entry of ``idx`` must satisfy
        ``0 <= idx[i] < num_segments``.
    op : str
        Name of the calling op (used in the error message).

    Raises
    ------
    ValueError
        On dtype mismatch (not ``int32``), wrong rank (not 1-D), or — in eager
        mode only — any value outside ``[0, num_segments)``.

    Notes
    -----
    The range check reads ``idx.min()`` / ``idx.max()`` as scalars, which forces
    a CUDA → host synchronization *and* a ``torch.compile`` graph break. We skip
    it under ``torch.compiler.is_compiling()`` so the public wrappers stay
    fullgraph-clean when a caller (e.g. an MLIP model) compiles straight through
    them; compiled callers are trusted to pass ``idx`` already validated at
    construction. The cheap dtype/rank guards add no sync and run on every path.
    """
    if idx.dtype != torch.int32:
        raise ValueError(f"{op}: idx must be int32; got dtype={idx.dtype}.")
    if idx.ndim != 1:
        raise ValueError(f"{op}: idx must be 1-D; got shape={tuple(idx.shape)}.")
    if torch.compiler.is_compiling():
        return
    if idx.numel() == 0:
        return
    idx_min = int(idx.min().item())
    idx_max = int(idx.max().item())
    if idx_min < 0:
        raise ValueError(
            f"{op}: idx contains negative values (min={idx_min}); all values "
            f"must be in the range [0, num_segments={num_segments})."
        )
    if idx_max >= num_segments:
        raise ValueError(
            f"{op}: idx contains out-of-range values (max={idx_max}, "
            f"num_segments={num_segments}); all values must be in the range "
            f"[0, num_segments)."
        )


# =============================================================================
# segmented_sum
# =============================================================================


def _segmented_sum_forward(
    x: torch.Tensor, idx: torch.Tensor, num_segments: int
) -> torch.Tensor:
    """Forward launcher: ``out[s] = sum_i x[i] where idx[i] == s``."""
    out_shape = (num_segments, 3) if x.ndim == 2 else (num_segments,)
    out = x.new_zeros(out_shape)
    with scoped_warp_stream(x.device):
        _wp_segmented_sum(_inp(x), _inp_int(idx), _out(out))
    return out


def _segmented_sum_forward_fake(
    x: torch.Tensor, idx: torch.Tensor, num_segments: int
) -> torch.Tensor:
    # Output is per-segment, so it does NOT share x's leading dim — the default
    # ``empty_like(x)`` fake would report the wrong shape under torch.compile.
    out_shape = (num_segments, 3) if x.ndim == 2 else (num_segments,)
    return x.new_empty(out_shape)


def _segmented_sum_backward(
    g_out: torch.Tensor, x: torch.Tensor, idx: torch.Tensor, num_segments: int
) -> torch.Tensor:
    """First-order backward: ``grad_x[i] = g_out[idx[i]]`` (a gather).

    ``x`` is unused in the computation; it is part of the signature because the
    chain convention passes ``(*cotangents, *forward_inputs)`` and the helper
    reshapes the returned grad against the matching forward input.
    """
    N = idx.shape[0]
    out_shape = (N, 3) if g_out.ndim == 2 else (N,)
    grad_x = g_out.new_zeros(out_shape)
    with scoped_warp_stream(g_out.device):
        segmented_sum_backward(_inp(g_out), _inp_int(idx), _out(grad_x))
    return grad_x


def _segmented_sum_double_backward(
    gg_x: torch.Tensor,
    g_out: torch.Tensor,
    x: torch.Tensor,
    idx: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    """Double-backward: ``grad_g_out[s] = sum_i gg_x[i]`` (scatter-sum).

    The first backward is linear in ``g_out``, so the second-order adjoint of
    that backward w.r.t. ``g_out`` is the segmented sum of ``gg_x`` — the same
    scatter-sum as the original forward.
    """
    out_shape = (num_segments, 3) if gg_x.ndim == 2 else (num_segments,)
    grad_g_out = gg_x.new_zeros(out_shape)
    with scoped_warp_stream(gg_x.device):
        segmented_sum_double_backward(
            _inp(gg_x), _inp_int(idx), num_segments, _out(grad_g_out)
        )
    return grad_g_out


# Forward op + first-order backward + double-backward, wired for autograd and
# torch.compile (the Warp launches are opaque to the inductor tracer). The
# first backward is a gather (grad w.r.t. ``x`` at position 0); its own
# backward is the scatter-sum above (grad w.r.t. ``g_out`` at position 0 of the
# backward op's inputs).
_SEGMENTED_SUM_OPS = register_warp_op_chain(
    name="nvalchemiops::segmented_sum",
    forward=_segmented_sum_forward,
    backward=_segmented_sum_backward,
    double_backward=_segmented_sum_double_backward,
    diff_input_positions=(0,),
    n_forward_inputs=3,
    second_order_diff_positions=(0,),
    n_backward_inputs=4,
    forward_fake=_segmented_sum_forward_fake,
)


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
    _validate_idx(idx, num_segments, op="segmented_sum")
    return _SEGMENTED_SUM_OPS["forward"](x, idx, num_segments)


# =============================================================================
# segmented_dot
# =============================================================================


def _segmented_dot_forward(
    x: torch.Tensor, y: torch.Tensor, idx: torch.Tensor, num_segments: int
) -> torch.Tensor:
    """Forward launcher: ``out[s] = sum_i dot(x[i], y[i]) where idx[i] == s``."""
    out = x.new_zeros((num_segments,))
    with scoped_warp_stream(x.device):
        _wp_segmented_dot(_inp(x), _inp(y), _inp_int(idx), _out(out))
    return out


def _segmented_dot_forward_fake(
    x: torch.Tensor, y: torch.Tensor, idx: torch.Tensor, num_segments: int
) -> torch.Tensor:
    # Output is one scalar per segment, not per element.
    return x.new_empty((num_segments,))


def _segmented_dot_backward(
    g_out: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    idx: torch.Tensor,
    num_segments: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """First-order backward: ``grad_x[i] = g_out[s]*y[i]``, ``grad_y[i] = g_out[s]*x[i]``."""
    grad_x = x.new_zeros(x.shape)
    grad_y = y.new_zeros(y.shape)
    with scoped_warp_stream(g_out.device):
        segmented_dot_backward(
            _inp(g_out), _inp(x), _inp(y), _inp_int(idx), _out(grad_x), _out(grad_y)
        )
    return grad_x, grad_y


def _segmented_dot_double_backward(
    gg_gx: torch.Tensor,
    gg_gy: torch.Tensor,
    g_out: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    idx: torch.Tensor,
    num_segments: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Double-backward w.r.t. the first backward's diff inputs ``(g_out, x, y)``."""
    grad_g_out = g_out.new_zeros((num_segments,))
    grad_x_extra = x.new_zeros(x.shape)
    grad_y_extra = y.new_zeros(y.shape)
    with scoped_warp_stream(gg_gx.device):
        segmented_dot_double_backward(
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
    return grad_g_out, grad_x_extra, grad_y_extra


# Diff forward inputs x(0), y(1). Backward returns (grad_x, grad_y); its own
# backward differentiates (g_out, x, y) at backward-input positions (0, 1, 2).
_SEGMENTED_DOT_OPS = register_warp_op_chain(
    name="nvalchemiops::segmented_dot",
    forward=_segmented_dot_forward,
    backward=_segmented_dot_backward,
    double_backward=_segmented_dot_double_backward,
    forward_fake=_segmented_dot_forward_fake,
    diff_input_positions=(0, 1),
    n_forward_inputs=4,
    second_order_diff_positions=(0, 1, 2),
    n_backward_inputs=5,
)


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
    _validate_idx(idx, num_segments, op="segmented_dot")
    return _SEGMENTED_DOT_OPS["forward"](x, y, idx, num_segments)


# =============================================================================
# segmented_mul   (x: vec3 or scalar, y: per-segment scalar)
# =============================================================================


def _segmented_mul_forward(
    x: torch.Tensor, y: torch.Tensor, idx: torch.Tensor, num_segments: int
) -> torch.Tensor:
    """Forward launcher: ``out[i] = x[i] * y[idx[i]]`` (per-element, same shape as ``x``)."""
    out = x.new_zeros(x.shape)
    with scoped_warp_stream(x.device):
        _wp_segmented_mul(_inp(x), _inp(y), _inp_int(idx), _out(out))
    return out


def _segmented_mul_backward(
    g_out: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    idx: torch.Tensor,
    num_segments: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """First-order backward: ``grad_x[i] = g_out[i]*y[s]``, ``grad_y[s] = sum_i dot(g_out[i], x[i])``."""
    grad_x = x.new_zeros(x.shape)
    grad_y = y.new_zeros((num_segments,))
    with scoped_warp_stream(g_out.device):
        segmented_mul_backward(
            _inp(g_out),
            _inp(x),
            _inp(y),
            _inp_int(idx),
            num_segments,
            _out(grad_x),
            _out(grad_y),
        )
    return grad_x, grad_y


def _segmented_mul_double_backward(
    gg_gx: torch.Tensor,
    gg_gy: torch.Tensor,
    g_out: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    idx: torch.Tensor,
    num_segments: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Double-backward w.r.t. the first backward's diff inputs ``(g_out, x, y)``."""
    grad_g_out = g_out.new_zeros(g_out.shape)
    grad_x_extra = x.new_zeros(x.shape)
    grad_y_extra = y.new_zeros((num_segments,))
    with scoped_warp_stream(gg_gx.device):
        segmented_mul_double_backward(
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
    return grad_g_out, grad_x_extra, grad_y_extra


# Diff forward inputs x(0), y(1). Output matches x's shape, so the default
# forward fake (``empty_like(x)``) is correct. Backward differentiates
# (g_out, x, y) at backward-input positions (0, 1, 2).
_SEGMENTED_MUL_OPS = register_warp_op_chain(
    name="nvalchemiops::segmented_mul",
    forward=_segmented_mul_forward,
    backward=_segmented_mul_backward,
    double_backward=_segmented_mul_double_backward,
    diff_input_positions=(0, 1),
    n_forward_inputs=4,
    second_order_diff_positions=(0, 1, 2),
    n_backward_inputs=5,
)


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
        Number of segments.  Must equal ``y.shape[0]``.

    Returns
    -------
    torch.Tensor
        Same shape as ``x``.

    Raises
    ------
    ValueError
        If ``num_segments != y.shape[0]``.  Without this guard, ``forward``
        would still succeed (``y`` is only indexed by ``idx``) but ``backward``
        would allocate ``grad_y`` with shape ``(num_segments,)`` — a tensor
        whose shape disagrees with the leaf ``y``, breaking the autograd
        contract.
    """
    if y.shape[0] != num_segments:
        raise ValueError(
            f"segmented_mul: num_segments ({num_segments}) must equal "
            f"y.shape[0] ({y.shape[0]}); y is the per-segment broadcast operand."
        )
    _validate_idx(idx, num_segments, op="segmented_mul")
    return _SEGMENTED_MUL_OPS["forward"](x, y, idx, num_segments)


# =============================================================================
# segmented_mean
# =============================================================================


def _segmented_mean_forward(x: torch.Tensor, idx: torch.Tensor, num_segments: int):
    """Forward launcher: ``(out, counts)`` where ``out[s] = mean(x[i] : idx[i]==s)``.

    ``counts`` (per-segment population, int32) is returned as a second output so
    the backward op can consume it via ``save_forward_outputs`` without
    recomputing — it is non-differentiable (integer dtype).
    """
    out_shape = (num_segments, 3) if x.ndim == 2 else (num_segments,)
    out = x.new_zeros(out_shape)
    sums = x.new_zeros(out_shape)
    counts = x.new_zeros((num_segments,), dtype=torch.int32)
    with scoped_warp_stream(x.device):
        _wp_segmented_mean(
            _inp(x), _inp_int(idx), _out(sums), _out_int(counts), _out(out)
        )
    return out, counts


def _segmented_mean_forward_fake(x: torch.Tensor, idx: torch.Tensor, num_segments: int):
    out_shape = (num_segments, 3) if x.ndim == 2 else (num_segments,)
    return (
        x.new_empty(out_shape),
        x.new_empty((num_segments,), dtype=torch.int32),
    )


def _segmented_mean_backward(
    counts: torch.Tensor,
    g_out: torch.Tensor,
    x: torch.Tensor,
    idx: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    """First-order backward: ``grad_x[i] = g_out[s] / count[s]`` (linear in g_out).

    ``counts`` is prepended by ``save_forward_outputs``; ``x`` is unused (the
    gradient does not depend on the forward input values).
    """
    N = idx.shape[0]
    out_shape = (N, 3) if g_out.ndim == 2 else (N,)
    grad_x = g_out.new_zeros(out_shape)
    with scoped_warp_stream(g_out.device):
        segmented_mean_backward(
            _inp(g_out), _inp_int(counts), _inp_int(idx), _out(grad_x)
        )
    return grad_x


def _segmented_mean_backward_fake(counts, g_out, x, idx, num_segments) -> torch.Tensor:
    return torch.empty_like(x)


def _segmented_mean_double_backward(
    gg_x: torch.Tensor,
    counts: torch.Tensor,
    g_out: torch.Tensor,
    x: torch.Tensor,
    idx: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    """Double-backward: ``grad_g_out[s] = sum_i gg_x[i] / count[s]`` (mean of gg_x).

    The first backward is linear in ``g_out`` (and independent of ``x``), so the
    only second-order term is w.r.t. ``g_out``.
    """
    out_shape = (num_segments, 3) if gg_x.ndim == 2 else (num_segments,)
    grad_g_out = gg_x.new_zeros(out_shape)
    with scoped_warp_stream(gg_x.device):
        segmented_mean_double_backward(
            _inp(gg_x), _inp_int(counts), _inp_int(idx), _out(grad_g_out)
        )
    return grad_g_out


def _segmented_mean_double_backward_fake(
    gg_x, counts, g_out, x, idx, num_segments
) -> torch.Tensor:
    return torch.empty_like(g_out)


# Forward returns (out, counts); only ``out``'s cotangent drives the backward
# (propagate_outputs=(0,)) and ``counts`` is threaded to the backward op as a
# detached cache (save_forward_outputs=(1,)). The backward op's inputs are
# therefore (counts, g_out, x, idx, num_segments) — the first backward is linear
# in g_out, so its own backward differentiates only g_out at position 1.
_SEGMENTED_MEAN_OPS = register_warp_op_chain(
    name="nvalchemiops::segmented_mean",
    forward=_segmented_mean_forward,
    backward=_segmented_mean_backward,
    double_backward=_segmented_mean_double_backward,
    forward_fake=_segmented_mean_forward_fake,
    backward_fake=_segmented_mean_backward_fake,
    double_backward_fake=_segmented_mean_double_backward_fake,
    forward_return_arity=2,
    propagate_outputs=(0,),
    save_forward_outputs=(1,),
    diff_input_positions=(0,),
    n_forward_inputs=3,
    second_order_diff_positions=(1,),
    n_backward_inputs=5,
)


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
    _validate_idx(idx, num_segments, op="segmented_mean")
    out, _ = _SEGMENTED_MEAN_OPS["forward"](x, idx, num_segments)
    return out


# =============================================================================
# segmented_rms_norm
# =============================================================================


def _segmented_rms_norm_forward(x: torch.Tensor, idx: torch.Tensor, num_segments: int):
    """Forward launcher: ``(out, inv_norm, counts)``.

    ``out[s] = sqrt(mean(||x[i]||^2 : idx[i]==s))``. The precompute path also
    emits ``inv_norm`` and ``counts`` so the backward op can consume them via
    ``save_forward_outputs`` without recomputing. Neither auxiliary output is
    surfaced by the public wrapper (it returns ``out`` only).
    """
    out = x.new_zeros((num_segments,))
    sum_sq = x.new_zeros((num_segments,))
    counts = x.new_zeros((num_segments,), dtype=torch.int32)
    inv_norm = x.new_zeros((num_segments,))
    with scoped_warp_stream(x.device):
        segmented_rms_norm_forward_precompute(
            _inp(x),
            _inp_int(idx),
            _out(sum_sq),
            _out_int(counts),
            _out(out),
            _out(inv_norm),
        )
    return out, inv_norm, counts


def _segmented_rms_norm_forward_fake(
    x: torch.Tensor, idx: torch.Tensor, num_segments: int
):
    return (
        x.new_empty((num_segments,)),
        x.new_empty((num_segments,)),
        x.new_empty((num_segments,), dtype=torch.int32),
    )


def _segmented_rms_norm_backward(
    inv_norm: torch.Tensor,
    counts: torch.Tensor,
    g_out: torch.Tensor,
    x: torch.Tensor,
    idx: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    """First-order backward: ``grad_x[i] = g_out[s] * x[i] * inv_norm[s]``.

    ``inv_norm`` and ``counts`` are prepended by ``save_forward_outputs``;
    ``counts`` is unused at first order (only the double-backward needs it).
    """
    grad_x = x.new_zeros(x.shape)
    with scoped_warp_stream(g_out.device):
        segmented_rms_norm_backward(
            _inp(g_out), _inp(x), _inp(inv_norm), _inp_int(idx), _out(grad_x)
        )
    return grad_x


def _segmented_rms_norm_backward_fake(
    inv_norm, counts, g_out, x, idx, num_segments
) -> torch.Tensor:
    return torch.empty_like(x)


def _segmented_rms_norm_double_backward(
    gg_x: torch.Tensor,
    inv_norm: torch.Tensor,
    counts: torch.Tensor,
    g_out: torch.Tensor,
    x: torch.Tensor,
    idx: torch.Tensor,
    num_segments: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Double-backward w.r.t. the first backward's diff inputs ``(g_out, x)``.

    Returns ``(grad_g_out, grad_x_extra)`` — order matches the backward op's
    input positions ``(2, 3)`` = ``(g_out, x)``.
    """
    grad_x_extra = x.new_zeros(x.shape)
    grad_g_out = g_out.new_zeros((num_segments,))
    with scoped_warp_stream(gg_x.device):
        segmented_rms_norm_double_backward(
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
    return grad_g_out, grad_x_extra


def _segmented_rms_norm_double_backward_fake(
    gg_x, inv_norm, counts, g_out, x, idx, num_segments
):
    return (torch.empty_like(g_out), torch.empty_like(x))


# Forward returns (out, inv_norm, counts); only ``out``'s cotangent drives the
# backward (propagate_outputs=(0,)), with ``inv_norm``/``counts`` threaded to the
# backward op as detached caches (save_forward_outputs=(1, 2)). The backward op's
# inputs are (inv_norm, counts, g_out, x, idx, num_segments); its own backward
# differentiates (g_out, x) at positions (2, 3).
_SEGMENTED_RMS_NORM_OPS = register_warp_op_chain(
    name="nvalchemiops::segmented_rms_norm",
    forward=_segmented_rms_norm_forward,
    backward=_segmented_rms_norm_backward,
    double_backward=_segmented_rms_norm_double_backward,
    forward_fake=_segmented_rms_norm_forward_fake,
    backward_fake=_segmented_rms_norm_backward_fake,
    double_backward_fake=_segmented_rms_norm_double_backward_fake,
    forward_return_arity=3,
    propagate_outputs=(0,),
    save_forward_outputs=(1, 2),
    diff_input_positions=(0,),
    n_forward_inputs=3,
    second_order_diff_positions=(2, 3),
    n_backward_inputs=6,
    double_backward_return_arity=2,
)


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
    _validate_idx(idx, num_segments, op="segmented_rms_norm")
    out, _, _ = _SEGMENTED_RMS_NORM_OPS["forward"](x, idx, num_segments)
    return out


# =============================================================================
# segmented_matvec
# =============================================================================


def _segmented_matvec_forward(
    v: torch.Tensor, m: torch.Tensor, idx: torch.Tensor, num_segments: int
) -> torch.Tensor:
    """Forward launcher: ``out[i] = m[idx[i]]^T @ v[i]`` (per-element, same shape as ``v``)."""
    out = v.new_zeros(v.shape)
    with scoped_warp_stream(v.device):
        _wp_segmented_matvec(_inp(v), _inp(m), _inp_int(idx), _out(out))
    return out


def _segmented_matvec_backward(
    g_out: torch.Tensor,
    v: torch.Tensor,
    m: torch.Tensor,
    idx: torch.Tensor,
    num_segments: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """First-order backward: ``grad_v[i] = m[s] @ g_out[i]``, ``grad_m[s] = sum_i outer(v[i], g_out[i])``."""
    grad_v = v.new_zeros(v.shape)
    grad_m = m.new_zeros(m.shape)
    with scoped_warp_stream(g_out.device):
        segmented_matvec_backward(
            _inp(g_out),
            _inp(v),
            _inp(m),
            _inp_int(idx),
            _out(grad_v),
            _out(grad_m),
        )
    return grad_v, grad_m


def _segmented_matvec_double_backward(
    gg_gv: torch.Tensor,
    gg_gm: torch.Tensor,
    g_out: torch.Tensor,
    v: torch.Tensor,
    m: torch.Tensor,
    idx: torch.Tensor,
    num_segments: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Double-backward w.r.t. the first backward's diff inputs ``(g_out, v, m)``."""
    grad_g_out = g_out.new_zeros(g_out.shape)
    grad_v_extra = v.new_zeros(v.shape)
    grad_m_extra = m.new_zeros(m.shape)
    with scoped_warp_stream(gg_gv.device):
        segmented_matvec_double_backward(
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
    return grad_g_out, grad_v_extra, grad_m_extra


# Diff forward inputs v(0), m(1). Output matches v's shape, so the default
# forward fake (``empty_like(v)``) is correct. Backward differentiates
# (g_out, v, m) at backward-input positions (0, 1, 2).
_SEGMENTED_MATVEC_OPS = register_warp_op_chain(
    name="nvalchemiops::segmented_matvec",
    forward=_segmented_matvec_forward,
    backward=_segmented_matvec_backward,
    double_backward=_segmented_matvec_double_backward,
    diff_input_positions=(0, 1),
    n_forward_inputs=4,
    second_order_diff_positions=(0, 1, 2),
    n_backward_inputs=5,
)


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
        Number of segments.  Must equal ``m.shape[0]``.

    Returns
    -------
    torch.Tensor
        Shape ``(N, 3)``.

    Raises
    ------
    ValueError
        If ``num_segments != m.shape[0]``.  Without this guard, ``forward``
        would succeed (``m`` is only indexed by ``idx``) but ``backward``
        would allocate ``grad_m`` with the wrong leading dimension, breaking
        the autograd contract.
    """
    if m.shape[0] != num_segments:
        raise ValueError(
            f"segmented_matvec: num_segments ({num_segments}) must equal "
            f"m.shape[0] ({m.shape[0]}); m is the per-segment matrix operand."
        )
    _validate_idx(idx, num_segments, op="segmented_matvec")
    return _SEGMENTED_MATVEC_OPS["forward"](v, m, idx, num_segments)
