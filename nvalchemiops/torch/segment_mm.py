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

"""PyTorch segmented matrix multiplication with forward and reverse derivatives."""

from __future__ import annotations

from contextlib import nullcontext

import torch
import warp as wp

from nvalchemiops.segment_mm import segment_mm as _warp_segment_mm
from nvalchemiops.segment_mm import segment_mm_backward as _warp_segment_mm_backward
from nvalchemiops.segment_mm import segment_mm_backward_b as _warp_segment_mm_backward_b
from nvalchemiops.torch._warp_op_helpers import (
    register_warp_op_chain,
    scoped_warp_stream,
)

__all__ = ["segment_mm"]


def _validate_inputs(a: torch.Tensor, b: torch.Tensor, seglen: torch.Tensor) -> None:
    """Check tensor metadata without reading device segment lengths."""
    for name, value, ndim in (("a", a, 2), ("b", b, 3), ("seglen", seglen, 1)):
        if not isinstance(value, torch.Tensor) or value.ndim != ndim:
            raise ValueError(f"{name} must be a {ndim}D tensor")
    if b.device != a.device or seglen.device != a.device:
        raise ValueError("All tensors must be on the same device")
    if a.device.type != "cuda":
        raise RuntimeError("segment_mm requires a CUDA device")
    if a.dtype not in (torch.float32, torch.bfloat16) or b.dtype not in (
        torch.float32,
        torch.bfloat16,
    ):
        raise ValueError("a and b must have float32 or bfloat16 dtype")
    if seglen.dtype != torch.int32:
        raise ValueError("seglen must have dtype int32")
    if a.shape[1] <= 0 or b.shape[2] <= 0:
        raise ValueError("Channel dimensions K and N must be positive")
    if b.shape[1] != a.shape[1]:
        raise ValueError("Input channel dimensions of a and b must match")
    if seglen.shape[0] != b.shape[0]:
        raise ValueError("seglen must contain one length per segment in b")


def _capture_scope(device: torch.device):
    """Register externally captured launches so Warp retains captured scratch safely."""
    stream = wp.get_stream(str(device))
    if stream.is_capturing and stream not in stream.device.captures:
        # Raw Torch capture does not notify Warp's allocator. External capture
        # registration attaches scratch lifetime to the owning CUDA graph.
        # SegMM does not pause/resume capture; Warp leaves the owner's mode intact.
        return wp.ScopedCapture(stream=stream, external=True)
    return nullcontext()


def _forward(a: torch.Tensor, b: torch.Tensor, seglen: torch.Tensor) -> torch.Tensor:
    """Wrap strided tensor storage and launch on the current Torch CUDA stream."""
    with scoped_warp_stream(a.device), _capture_scope(a.device):
        out = a.new_empty((a.shape[0], b.shape[2]))
        _warp_segment_mm(
            wp.from_torch(a, requires_grad=False),
            wp.from_torch(b, requires_grad=False),
            wp.from_torch(seglen, requires_grad=False),
            wp.from_torch(out, requires_grad=False),
        )
    return out


def _forward_fake(
    a: torch.Tensor, b: torch.Tensor, seglen: torch.Tensor
) -> torch.Tensor:
    """Describe a fresh contiguous output without running a Warp kernel."""
    return a.new_empty((a.shape[0], b.shape[2]))


def _backward(
    grad_out: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    seglen: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate contiguous gradients and invoke the explicit first-order backward."""
    with scoped_warp_stream(a.device), _capture_scope(a.device):
        grad_a = a.new_empty(a.shape)
        grad_b = b.new_empty(b.shape)
        _warp_segment_mm_backward(
            wp.from_torch(a, requires_grad=False),
            wp.from_torch(b, requires_grad=False),
            wp.from_torch(seglen, requires_grad=False),
            wp.from_torch(grad_out, requires_grad=False),
            wp.from_torch(grad_a, requires_grad=False),
            wp.from_torch(grad_b, requires_grad=False),
        )
    return grad_a, grad_b


def _backward_fake(
    grad_out: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    seglen: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Describe contiguous gradients even when the original inputs are strided."""
    return a.new_empty(a.shape), b.new_empty(b.shape)


def _weight_product(
    a: torch.Tensor, grad_out: torch.Tensor, seglen: torch.Tensor
) -> torch.Tensor:
    """Compute only the segmented weight product, retaining native operand dtype."""
    with scoped_warp_stream(a.device), _capture_scope(a.device):
        grad_b = a.new_empty((seglen.shape[0], a.shape[1], grad_out.shape[1]))
        _warp_segment_mm_backward_b(
            wp.from_torch(a, requires_grad=False),
            wp.from_torch(seglen, requires_grad=False),
            wp.from_torch(grad_out, requires_grad=False),
            wp.from_torch(grad_b, requires_grad=False),
        )
    return grad_b


def _weight_product_fake(
    a: torch.Tensor, grad_out: torch.Tensor, seglen: torch.Tensor
) -> torch.Tensor:
    """Describe a fresh contiguous weight-product output."""
    return a.new_empty((seglen.shape[0], a.shape[1], grad_out.shape[1]))


# Autograd belongs to the transformable Functions below. These internal ops
# provide execution and fake metadata without a parallel derivative chain.
_FORWARD_OP = register_warp_op_chain(
    name="nvalchemiops::segment_mm",
    forward=_forward,
    forward_fake=_forward_fake,
)["forward"]
_BACKWARD_OP = register_warp_op_chain(
    name="nvalchemiops::segment_mm_backward",
    forward=_backward,
    forward_fake=_backward_fake,
    forward_return_arity=2,
)["forward"]
_WEIGHT_PRODUCT_OP = register_warp_op_chain(
    name="nvalchemiops::segment_mm_weight_product",
    forward=_weight_product,
    forward_fake=_weight_product_fake,
)["forward"]


@torch.compiler.allow_in_graph
def _apply_forward(a, b, seglen):
    """Expose transformable autograd rules to AOTAutograd without Dynamo inlining."""
    return _SegmentMMFunction.apply(a, b, seglen)


@torch.compiler.allow_in_graph
def _apply_backward(grad_out, a, b, seglen):
    """Apply the shared backward with a derivative rule at every order."""
    return _SegmentMMBackwardFunction.apply(grad_out, a, b, seglen)


@torch.compiler.allow_in_graph
def _apply_weight_product(a, grad_out, seglen):
    """Expose a weight-only product with composable derivative rules."""
    return _SegmentMMWeightProductFunction.apply(a, grad_out, seglen)


def _backward_vjp(grad_out, a, b, seglen, grad_grad_a, grad_grad_b):
    """Transpose the explicit backward using the same segmented products."""
    if grad_grad_a is None and grad_grad_b is None:
        return None, None, None, None
    if grad_grad_a is None:
        grad_a = _apply_forward(grad_out, grad_grad_b.transpose(1, 2), seglen)
        grad_b = None
        grad_grad_out = _apply_forward(a, grad_grad_b, seglen)
    elif grad_grad_b is None:
        grad_a = None
        grad_b = _apply_weight_product(grad_grad_a, grad_out, seglen)
        grad_grad_out = _apply_forward(grad_grad_a, b, seglen)
    else:
        grad_a, grad_b = _apply_backward(grad_out, grad_grad_a, grad_grad_b, seglen)
        grad_grad_out = _apply_forward(grad_grad_a, b, seglen) + _apply_forward(
            a, grad_grad_b, seglen
        )
    return grad_grad_out, grad_a, grad_b, None


class _SegmentMMFunction(torch.autograd.Function):
    """Expose the opaque forward to PyTorch function transforms and forward AD."""

    @staticmethod
    def forward(a, b, seglen):
        return _FORWARD_OP(a, b, seglen)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(*inputs)
        ctx.save_for_forward(*inputs)
        ctx.set_materialize_grads(False)

    @staticmethod
    def backward(ctx, grad_out):
        if grad_out is None:
            return None, None, None
        a, b, seglen = ctx.saved_tensors
        grad_a, grad_b = _apply_backward(grad_out, a, b, seglen)
        return grad_a, grad_b, None

    @staticmethod
    def jvp(ctx, tangent_a, tangent_b, tangent_seglen):
        a, b, seglen = ctx.saved_tensors
        tangent = None
        if tangent_a is not None:
            tangent = _apply_forward(tangent_a, b, seglen)
        if tangent_b is not None:
            term = _apply_forward(a, tangent_b, seglen)
            tangent = term if tangent is None else tangent + term
        if tangent is None:
            tangent = a.new_zeros((a.shape[0], b.shape[2]))
        return tangent


class _SegmentMMBackwardFunction(torch.autograd.Function):
    """Differentiate both operand gradients through the shared Warp backward."""

    @staticmethod
    def forward(grad_out, a, b, seglen):
        return _BACKWARD_OP(grad_out, a, b, seglen)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(*inputs)
        ctx.save_for_forward(*inputs)
        ctx.set_materialize_grads(False)

    @staticmethod
    def backward(ctx, grad_grad_a, grad_grad_b):
        return _backward_vjp(*ctx.saved_tensors, grad_grad_a, grad_grad_b)

    @staticmethod
    def jvp(ctx, tangent_grad_out, tangent_a, tangent_b, tangent_seglen):
        grad_out, a, b, seglen = ctx.saved_tensors
        tangent_grad_a, tangent_grad_b = None, None
        if tangent_a is not None and tangent_b is not None:
            tangent_grad_a, tangent_grad_b = _apply_backward(
                grad_out, tangent_a, tangent_b, seglen
            )
        elif tangent_a is not None:
            tangent_grad_b = _apply_weight_product(tangent_a, grad_out, seglen)
        elif tangent_b is not None:
            tangent_grad_a = _apply_forward(grad_out, tangent_b.transpose(1, 2), seglen)
        if tangent_grad_out is not None:
            term_a, term_b = _apply_backward(tangent_grad_out, a, b, seglen)
            tangent_grad_a = (
                term_a if tangent_grad_a is None else tangent_grad_a + term_a
            )
            tangent_grad_b = (
                term_b if tangent_grad_b is None else tangent_grad_b + term_b
            )
        if tangent_grad_a is None:
            tangent_grad_a = torch.zeros_like(a)
        if tangent_grad_b is None:
            tangent_grad_b = torch.zeros_like(b)
        return tangent_grad_a, tangent_grad_b


class _SegmentMMWeightProductFunction(torch.autograd.Function):
    """Differentiate a weight product without allocating an input gradient."""

    @staticmethod
    def forward(a, grad_out, seglen):
        return _WEIGHT_PRODUCT_OP(a, grad_out, seglen)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(*inputs)
        ctx.save_for_forward(*inputs)
        ctx.set_materialize_grads(False)

    @staticmethod
    def backward(ctx, grad_b):
        if grad_b is None:
            return None, None, None
        a, grad_out, seglen = ctx.saved_tensors
        return (
            _apply_forward(grad_out, grad_b.transpose(1, 2), seglen),
            _apply_forward(a, grad_b, seglen),
            None,
        )

    @staticmethod
    def jvp(ctx, tangent_a, tangent_grad_out, tangent_seglen):
        a, grad_out, seglen = ctx.saved_tensors
        tangent = None
        if tangent_a is not None:
            tangent = _apply_weight_product(tangent_a, grad_out, seglen)
        if tangent_grad_out is not None:
            term = _apply_weight_product(a, tangent_grad_out, seglen)
            tangent = term if tangent is None else tangent + term
        if tangent is None:
            tangent = a.new_zeros((seglen.shape[0], a.shape[1], grad_out.shape[1]))
        return tangent


def segment_mm(a: torch.Tensor, b: torch.Tensor, seglen: torch.Tensor) -> torch.Tensor:
    """Multiply each segment of feature rows by its corresponding weight matrix.

    For ``offsets = [0, cumsum(seglen)]``, segment ``g`` computes
    ``out[offsets[g]:offsets[g+1]] = a[offsets[g]:offsets[g+1]] @ b[g]``.
    Rows after ``sum(seglen)`` are zero, even when input padding contains NaNs.

    Parameters
    ----------
    a : torch.Tensor, shape (capacity, K)
        CUDA input features with dtype float32 or bfloat16.
    b : torch.Tensor, shape (R, K, N)
        Per-segment weights on the same CUDA device as ``a``. The dtype must
        match ``a`` unless CUDA bfloat16 autocast is enabled.
    seglen : torch.Tensor, shape (R,), dtype torch.int32
        Runtime lengths on the same CUDA device. Values must be nonnegative
        and sum to at most ``capacity``. Zero-length segments are supported.
        These value constraints are caller preconditions, not host checks.

    Returns
    -------
    torch.Tensor, shape (capacity, N)
        Newly allocated contiguous output on the same device as ``a``. The
        dtype matches ``a``, or is bfloat16 under CUDA bfloat16 autocast.
        Products accumulate in FP32 and round once to the output dtype.

    Raises
    ------
    ValueError
        If ranks, shapes, dtypes, devices, or core stride/index limits are invalid.
    RuntimeError
        If tensors are not on CUDA.

    Notes
    -----
    K and N must be positive. Inputs are wrapped without copying unless autocast
    requires dtype conversion; transposed, sliced, and broadcast views with
    nonnegative int32 byte strides are supported.
    Segment lengths and prefix sums remain on the GPU. Work, including temporary
    allocation, uses the current PyTorch CUDA stream.

    CUDA bfloat16 autocast converts float32 or bfloat16 operands to bfloat16
    before the custom autograd function. Gradients propagate through these
    casts to the original operands in their original dtypes. FP32 parameter
    gradients therefore retain BF16 numerical precision. CUDA float16 autocast
    is unsupported and raises ValueError. Outside autocast, or inside a nested
    disabled context, both operands must have the same supported dtype.

    Reverse-mode gradients are supported for either or both operands in eager
    mode and ``torch.compile(fullgraph=True)``. Gradients have the input
    dtype; BF16 multiplication operands remain BF16 and accumulation uses FP32.
    Lengths are not differentiated. Forward AD, ``torch.func.jvp``, and
    differentiation through the backward support force-loss training.
    For compiled higher derivatives, compile the complete ``torch.func.grad``
    or ``torch.func.jvp`` computation. PyTorch 2.10 AOTAutograd does not support
    a second backward through a separately compiled forward.
    CUDA graph capture requires kernel warmup and ``wp.config.verify_cuda=False``.
    Keep the SegMM Warp module loaded and unchanged until all replays finish and
    captured Torch graphs are discarded. Warm up and recapture after unloading,
    reloading, or changing module options.

    See Also
    --------
    nvalchemiops.segment_mm.segment_mm : Shared Warp forward implementation.
    nvalchemiops.segment_mm.segment_mm_backward : Explicit first-order backward.
    """
    _validate_inputs(a, b, seglen)
    if torch.is_autocast_enabled("cuda"):
        if torch.get_autocast_dtype("cuda") != torch.bfloat16:
            raise ValueError("segment_mm supports CUDA autocast only with bfloat16")
        a, b = a.to(torch.bfloat16), b.to(torch.bfloat16)
    elif a.dtype != b.dtype:
        raise ValueError("a and b must have the same dtype outside CUDA autocast")
    return _apply_forward(a, b, seglen)
