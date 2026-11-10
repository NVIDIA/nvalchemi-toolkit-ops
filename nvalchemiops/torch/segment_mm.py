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

import torch
import warp as wp

from nvalchemiops.batch_utils import _segment_counts, _segment_counts_checked
from nvalchemiops.segment_mm import (
    _BLOCK_DIM,
    _INDEX_LIMIT,
    _TILE_M,
    _TILE_N,
    _grad_b_launch_config,
    _launch_forward,
    _launch_grad_a,
    _launch_grad_b,
    _launch_segment_offsets,
    _segment_offsets_requires_scan,
    _validate_grad_out,
    _validate_output,
)
from nvalchemiops.segment_mm import _validate_inputs as _validate_warp_inputs
from nvalchemiops.torch._warp_op_helpers import (
    register_warp_op_chain,
    scoped_warp_capture,
    scoped_warp_stream,
)

__all__ = ["segment_mm"]


def _validate_inputs(a: torch.Tensor, b: torch.Tensor, seglen: torch.Tensor) -> None:
    """Check tensor metadata without inspecting runtime segment values."""
    for name, value, ndim in (("a", a, 2), ("b", b, 3), ("seglen", seglen, 1)):
        if not isinstance(value, torch.Tensor) or value.ndim != ndim:
            raise ValueError(f"{name} must be a {ndim}D tensor")
    if b.device != a.device or seglen.device != a.device:
        raise ValueError("All tensors must be on the same device")
    if a.device.type not in ("cpu", "cuda"):
        raise RuntimeError("segment_mm requires a CPU or CUDA device")
    if a.dtype not in (torch.float32, torch.bfloat16) or b.dtype not in (
        torch.float32,
        torch.bfloat16,
    ):
        raise ValueError("a and b must have float32 or bfloat16 dtype")
    if seglen.dtype not in (torch.int32, torch.int64):
        raise ValueError("seglen must have dtype int32 or int64")
    if a.shape[1] <= 0 or b.shape[2] <= 0:
        raise ValueError("Channel dimensions K and N must be positive")
    if b.shape[1] != a.shape[1]:
        raise ValueError("Input channel dimensions of a and b must match")
    if seglen.shape[0] != b.shape[0]:
        raise ValueError("seglen must contain one length per segment in b")


def _prepare_lengths(
    seglen: torch.Tensor, capacity: int, validate: bool
) -> torch.Tensor:
    """Validate concrete CPU values and preserve wide CUDA lengths for checking."""
    if validate and seglen.is_cuda:
        return seglen
    if validate and seglen.numel():
        if seglen.min().item() < 0:
            raise ValueError("seglen values must be nonnegative")
        if (
            seglen.dtype == torch.int64
            and seglen.max().item() > torch.iinfo(torch.int32).max
        ):
            raise ValueError("seglen values must be representable as int32")
        if seglen.sum(dtype=torch.int64).item() > capacity:
            raise ValueError("seglen sum must be at most row_capacity")
    return seglen.to(torch.int32)


def _segment_offsets(
    seglen: wp.array,
    capacity: int,
    device: torch.device,
    tile_rows: int = _TILE_M,
    skip_first_tile: bool = False,
    *,
    chunk_rows: int = 0,
    validate: bool = False,
) -> tuple[wp.array, wp.array, wp.array]:
    """Build device schedules with Torch-owned offsets and scan workspace.

    Called inside the binding's stream scope after metadata validation. The
    returned Warp views retain their Torch tensors. When no separate chunk
    schedule is requested, the third result aliases the tile offsets.
    """
    groups = seglen.shape[0]
    validation = (
        torch.empty(1, dtype=torch.int32, device=device)
        if validate and seglen.device.is_cuda
        else None
    )
    validation_wp = (
        wp.from_torch(validation, requires_grad=False)
        if validation is not None
        else None
    )
    rows = torch.empty(groups + 1, dtype=torch.int32, device=device)
    tiles = torch.empty_like(rows)
    chunks = torch.empty_like(rows) if chunk_rows else tiles
    row_offsets = wp.from_torch(rows, requires_grad=False)
    tile_offsets = wp.from_torch(tiles, requires_grad=False)
    chunk_offsets = (
        wp.from_torch(chunks, requires_grad=False) if chunk_rows else tile_offsets
    )
    needs_scan = _segment_offsets_requires_scan(
        groups, seglen.device.is_cuda, bool(chunk_rows)
    )
    row_prefix = None
    if needs_scan:
        # Widen before summing so even invalid int32 lengths cannot overflow.
        prefix = torch.empty(groups + 1, dtype=torch.int64, device=device)
        row_prefix = wp.from_torch(prefix, requires_grad=False)
        wp.launch(
            _segment_counts_checked if validation is not None else _segment_counts,
            dim=groups + 1,
            inputs=[seglen, row_prefix, capacity]
            if validation is not None
            else [seglen, row_prefix],
            device=seglen.device,
        )
        torch.cumsum(prefix, dim=0, dtype=prefix.dtype, out=prefix)
    _launch_segment_offsets(
        seglen,
        capacity,
        row_offsets,
        tile_offsets,
        tile_rows,
        skip_first_tile,
        chunk_rows=chunk_rows,
        chunk_offsets=chunk_offsets if chunk_rows else None,
        row_prefix=row_prefix,
        validation=validation_wp,
    )
    if validation is not None:
        torch._assert_async(
            validation, "seglen must be nonnegative and sum to at most row_capacity"
        )
    if needs_scan:
        torch.cumsum(tiles, dim=0, dtype=tiles.dtype, out=tiles)
        if chunk_rows:
            torch.cumsum(chunks, dim=0, dtype=chunks.dtype, out=chunks)
    return row_offsets, tile_offsets, chunk_offsets


def _forward(
    a: torch.Tensor, b: torch.Tensor, seglen: torch.Tensor, validate: bool = False
) -> torch.Tensor:
    """Wrap strided storage and use the current Torch stream on CUDA."""
    with scoped_warp_stream(a.device), scoped_warp_capture(a.device):
        seglen = _prepare_lengths(seglen, a.shape[0], validate)
        out = a.new_empty((a.shape[0], b.shape[2]))
        a_wp = wp.from_torch(a, requires_grad=False)
        b_wp = wp.from_torch(b, requires_grad=False)
        seglen_wp = wp.from_torch(seglen, requires_grad=False)
        out_wp = wp.from_torch(out, requires_grad=False)
        _validate_warp_inputs(a_wp, b_wp, seglen_wp, allow_int64_lengths=validate)
        _validate_output(out_wp, "out", out.shape, a_wp, (a_wp, b_wp, seglen_wp))
        if a.shape[0] == 0 or b.shape[0] == 0:
            if validate and b.shape[0] > 0:
                _segment_offsets(seglen_wp, 0, a.device, validate=True)
            out.zero_()
            return out
        row_offsets, tile_offsets, _ = _segment_offsets(
            seglen_wp, a.shape[0], a.device, validate=validate
        )
        _launch_forward(a_wp, b_wp, row_offsets, tile_offsets, out_wp)
    return out


def _forward_fake(
    a: torch.Tensor, b: torch.Tensor, seglen: torch.Tensor, validate: bool = False
) -> torch.Tensor:
    """Describe a fresh contiguous output without running a Warp kernel."""
    return a.new_empty((a.shape[0], b.shape[2]))


def _backward(
    grad_out: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    seglen: torch.Tensor,
    validate: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate contiguous gradients and invoke the explicit first-order backward."""
    with scoped_warp_stream(a.device), scoped_warp_capture(a.device):
        seglen = _prepare_lengths(seglen, a.shape[0], validate)
        grad_a = a.new_empty(a.shape)
        grad_b = b.new_empty(b.shape)
        a_wp = wp.from_torch(a, requires_grad=False)
        b_wp = wp.from_torch(b, requires_grad=False)
        seglen_wp = wp.from_torch(seglen, requires_grad=False)
        grad_out_wp = wp.from_torch(grad_out, requires_grad=False)
        grad_a_wp = wp.from_torch(grad_a, requires_grad=False)
        grad_b_wp = wp.from_torch(grad_b, requires_grad=False)
        _validate_warp_inputs(a_wp, b_wp, seglen_wp, allow_int64_lengths=validate)
        capacity, k = a.shape
        groups, _, n = b.shape
        _validate_grad_out(a_wp, grad_out_wp, n)
        inputs = (a_wp, b_wp, seglen_wp, grad_out_wp)
        _validate_output(grad_a_wp, "grad_a", a.shape, a_wp, inputs)
        _validate_output(grad_b_wp, "grad_b", b.shape, a_wp, (*inputs, grad_a_wp))
        if capacity == 0 or groups == 0:
            if validate and groups > 0:
                _segment_offsets(seglen_wp, 0, a.device, validate=True)
            grad_a.zero_()
            grad_b.zero_()
            return grad_a, grad_b

        chunk_rows, scratch_slots = _grad_b_launch_config(
            capacity, groups, k, n, a.element_size()
        )
        row_tasks = (capacity + _TILE_M - 1) // _TILE_M + groups
        da_column_tiles = (k + _TILE_N - 1) // _TILE_N
        if row_tasks * da_column_tiles * _BLOCK_DIM > _INDEX_LIMIT:
            raise ValueError(
                "Backward workspace or launch dimensions exceed the int32 index range"
            )
        row_offsets, tile_offsets, chunk_offsets = _segment_offsets(
            seglen_wp,
            capacity,
            a.device,
            chunk_rows=chunk_rows if chunk_rows < capacity else 0,
            validate=validate,
        )
        weights_wp = b_wp
        if b.dtype == torch.float32:
            weights = b.transpose(1, 2).contiguous()
            weights_wp = wp.from_torch(weights, requires_grad=False)
            b_wp = weights_wp.transpose((0, 2, 1))
        _launch_grad_a(
            grad_out_wp, b_wp, row_offsets, tile_offsets, grad_a_wp, weights_wp
        )
        partials_wp = grad_b_wp
        if chunk_rows < capacity:
            partials = torch.empty(
                (scratch_slots, k, n), dtype=torch.float32, device=a.device
            )
            partials_wp = wp.from_torch(partials, requires_grad=False)
        _launch_grad_b(
            a_wp,
            row_offsets,
            chunk_offsets,
            grad_out_wp,
            grad_b_wp,
            partials_wp,
            chunk_rows,
        )
    return grad_a, grad_b


def _backward_fake(
    grad_out: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    seglen: torch.Tensor,
    validate: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Describe contiguous gradients even when the original inputs are strided."""
    return a.new_empty(a.shape), b.new_empty(b.shape)


def _weight_product(
    a: torch.Tensor,
    grad_out: torch.Tensor,
    seglen: torch.Tensor,
    validate: bool = False,
) -> torch.Tensor:
    """Compute only the segmented weight product, retaining native operand dtype."""
    with scoped_warp_stream(a.device), scoped_warp_capture(a.device):
        seglen = _prepare_lengths(seglen, a.shape[0], validate)
        grad_b = a.new_empty((seglen.shape[0], a.shape[1], grad_out.shape[1]))
        a_wp = wp.from_torch(a, requires_grad=False)
        seglen_wp = wp.from_torch(seglen, requires_grad=False)
        grad_out_wp = wp.from_torch(grad_out, requires_grad=False)
        grad_b_wp = wp.from_torch(grad_b, requires_grad=False)
        _validate_warp_inputs(a_wp, grad_b_wp, seglen_wp, allow_int64_lengths=validate)
        _validate_grad_out(a_wp, grad_out_wp, grad_out.shape[1])
        _validate_output(
            grad_b_wp, "grad_b", grad_b.shape, a_wp, (a_wp, seglen_wp, grad_out_wp)
        )
        if a.shape[0] == 0 or seglen.shape[0] == 0:
            if validate and seglen.shape[0] > 0:
                _segment_offsets(seglen_wp, 0, a.device, validate=True)
            grad_b.zero_()
            return grad_b
        groups, k, n = grad_b.shape
        chunk_rows, scratch_slots = _grad_b_launch_config(
            a.shape[0], groups, k, n, a.element_size()
        )
        row_offsets, chunk_offsets, _ = _segment_offsets(
            seglen_wp,
            a.shape[0],
            a.device,
            chunk_rows,
            skip_first_tile=True,
            validate=validate,
        )
        partials_wp = grad_b_wp
        if chunk_rows < a.shape[0]:
            partials = torch.empty(
                (scratch_slots, k, n), dtype=torch.float32, device=a.device
            )
            partials_wp = wp.from_torch(partials, requires_grad=False)
        _launch_grad_b(
            a_wp,
            row_offsets,
            chunk_offsets,
            grad_out_wp,
            grad_b_wp,
            partials_wp,
            chunk_rows,
        )
    return grad_b


def _weight_product_fake(
    a: torch.Tensor,
    grad_out: torch.Tensor,
    seglen: torch.Tensor,
    validate: bool = False,
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
def _apply_forward(a, b, seglen, validate=False):
    """Expose transformable autograd rules to AOTAutograd without Dynamo inlining."""
    return _SegmentMMFunction.apply(a, b, seglen, validate)


@torch.compiler.allow_in_graph
def _apply_backward(grad_out, a, b, seglen, validate=False):
    """Apply the shared backward with a derivative rule at every order."""
    return _SegmentMMBackwardFunction.apply(grad_out, a, b, seglen, validate)


@torch.compiler.allow_in_graph
def _apply_weight_product(a, grad_out, seglen, validate=False):
    """Expose a weight-only product with composable derivative rules."""
    return _SegmentMMWeightProductFunction.apply(a, grad_out, seglen, validate)


def _backward_vjp(grad_out, a, b, seglen, grad_grad_a, grad_grad_b, validate=False):
    """Transpose the explicit backward using the same segmented products."""
    if grad_grad_a is None and grad_grad_b is None:
        return None, None, None, None, None
    if grad_grad_a is None:
        grad_a = _apply_forward(grad_out, grad_grad_b.transpose(1, 2), seglen, validate)
        grad_b = None
        grad_grad_out = _apply_forward(a, grad_grad_b, seglen, validate)
    elif grad_grad_b is None:
        grad_a = None
        grad_b = _apply_weight_product(grad_grad_a, grad_out, seglen, validate)
        grad_grad_out = _apply_forward(grad_grad_a, b, seglen, validate)
    else:
        grad_a, grad_b = _apply_backward(
            grad_out, grad_grad_a, grad_grad_b, seglen, validate
        )
        grad_grad_out = _apply_forward(
            grad_grad_a, b, seglen, validate
        ) + _apply_forward(a, grad_grad_b, seglen, validate)
    return grad_grad_out, grad_a, grad_b, None, None


# The shared registration helper owns opaque execution and fake metadata.
# These analytical Functions add JVPs and compose explicit derivative products
# at higher orders; the tape-backed warp_custom_op utility supports neither.
class _SegmentMMFunction(torch.autograd.Function):
    """Expose the opaque forward to PyTorch function transforms and forward AD."""

    @staticmethod
    def forward(a, b, seglen, validate):
        return _FORWARD_OP(a, b, seglen, validate)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(*inputs[:-1])
        ctx.save_for_forward(*inputs[:-1])
        ctx.validate = inputs[-1]
        ctx.set_materialize_grads(False)

    @staticmethod
    def backward(ctx, grad_out):
        if grad_out is None:
            return None, None, None, None
        a, b, seglen = ctx.saved_tensors
        need_a, need_b = ctx.needs_input_grad[:2]
        if need_a and need_b:
            grad_a, grad_b = _apply_backward(grad_out, a, b, seglen, ctx.validate)
        else:
            grad_a = (
                _apply_forward(grad_out, b.transpose(1, 2), seglen, ctx.validate)
                if need_a
                else None
            )
            grad_b = (
                _apply_weight_product(a, grad_out, seglen, ctx.validate)
                if need_b
                else None
            )
        return grad_a, grad_b, None, None

    @staticmethod
    def jvp(ctx, tangent_a, tangent_b, tangent_seglen, tangent_validate):
        a, b, seglen = ctx.saved_tensors
        tangent = None
        if tangent_a is not None:
            tangent = _apply_forward(tangent_a, b, seglen, ctx.validate)
        if tangent_b is not None:
            term = _apply_forward(a, tangent_b, seglen, ctx.validate)
            tangent = term if tangent is None else tangent + term
        if tangent is None:
            tangent = a.new_zeros((a.shape[0], b.shape[2]))
        return tangent


class _SegmentMMBackwardFunction(torch.autograd.Function):
    """Differentiate both operand gradients through the shared Warp backward."""

    @staticmethod
    def forward(grad_out, a, b, seglen, validate):
        return _BACKWARD_OP(grad_out, a, b, seglen, validate)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(*inputs[:-1])
        ctx.save_for_forward(*inputs[:-1])
        ctx.validate = inputs[-1]
        ctx.set_materialize_grads(False)

    @staticmethod
    def backward(ctx, grad_grad_a, grad_grad_b):
        return _backward_vjp(*ctx.saved_tensors, grad_grad_a, grad_grad_b, ctx.validate)

    @staticmethod
    def jvp(
        ctx, tangent_grad_out, tangent_a, tangent_b, tangent_seglen, tangent_validate
    ):
        grad_out, a, b, seglen = ctx.saved_tensors
        tangent_grad_a, tangent_grad_b = None, None
        if tangent_a is not None and tangent_b is not None:
            tangent_grad_a, tangent_grad_b = _apply_backward(
                grad_out, tangent_a, tangent_b, seglen, ctx.validate
            )
        elif tangent_a is not None:
            tangent_grad_b = _apply_weight_product(
                tangent_a, grad_out, seglen, ctx.validate
            )
        elif tangent_b is not None:
            tangent_grad_a = _apply_forward(
                grad_out, tangent_b.transpose(1, 2), seglen, ctx.validate
            )
        if tangent_grad_out is not None:
            term_a, term_b = _apply_backward(
                tangent_grad_out, a, b, seglen, ctx.validate
            )
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
    def forward(a, grad_out, seglen, validate):
        return _WEIGHT_PRODUCT_OP(a, grad_out, seglen, validate)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(*inputs[:-1])
        ctx.save_for_forward(*inputs[:-1])
        ctx.validate = inputs[-1]
        ctx.set_materialize_grads(False)

    @staticmethod
    def backward(ctx, grad_b):
        if grad_b is None:
            return None, None, None, None
        a, grad_out, seglen = ctx.saved_tensors
        need_a, need_grad_out = ctx.needs_input_grad[:2]
        return (
            _apply_forward(grad_out, grad_b.transpose(1, 2), seglen, ctx.validate)
            if need_a
            else None,
            _apply_forward(a, grad_b, seglen, ctx.validate) if need_grad_out else None,
            None,
            None,
        )

    @staticmethod
    def jvp(ctx, tangent_a, tangent_grad_out, tangent_seglen, tangent_validate):
        a, grad_out, seglen = ctx.saved_tensors
        tangent = None
        if tangent_a is not None:
            tangent = _apply_weight_product(tangent_a, grad_out, seglen, ctx.validate)
        if tangent_grad_out is not None:
            term = _apply_weight_product(a, tangent_grad_out, seglen, ctx.validate)
            tangent = term if tangent is None else tangent + term
        if tangent is None:
            tangent = a.new_zeros((seglen.shape[0], a.shape[1], grad_out.shape[1]))
        return tangent


def segment_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    seglen: torch.Tensor,
    *,
    validate: bool = False,
) -> torch.Tensor:
    """Multiply each segment of feature rows by its corresponding weight matrix.

    For ``offsets = [0, cumsum(seglen)]``, segment ``g`` computes
    ``out[offsets[g]:offsets[g+1]] = a[offsets[g]:offsets[g+1]] @ b[g]``.
    Rows after ``sum(seglen)`` are zero, even when input padding contains NaNs.

    ``row_capacity`` is the allocated row count, including padding, and
    ``num_segments`` is the number of segments. ``in_channels`` and
    ``out_channels`` are the input and output feature counts.

    Parameters
    ----------
    a : torch.Tensor, shape (row_capacity, in_channels), dtype float32 or bfloat16
        Packed input features on a CPU or CUDA device.
    b : torch.Tensor, shape (num_segments, in_channels, out_channels), dtype float32 or bfloat16
        One weight matrix per segment, on the same device as ``a``.
        The dtype must match ``a`` unless CUDA bfloat16 autocast is enabled.
    seglen : torch.Tensor, shape (num_segments,), dtype torch.int32 or torch.int64
        Runtime lengths on the same device. Values must be nonnegative
        and sum to at most ``row_capacity``. Zero-length segments are supported.
        Int64 lengths must fit int32.
    validate : bool, optional
        Check segment values during execution. Defaults to False; shape and
        dtype checks always run. True raises ValueError for invalid CPU lengths
        and uses an asynchronous device assertion on CUDA.
        The option also applies to gradients and JVPs.

    Returns
    -------
    torch.Tensor, shape (row_capacity, out_channels)
        Newly allocated contiguous output on the same device as ``a``. The
        dtype matches ``a``, or is bfloat16 under CUDA bfloat16 autocast.
        Products accumulate in FP32 and round once to the output dtype.

    Raises
    ------
    ValueError
        If ranks, shapes, dtypes, devices, or core stride/index limits are invalid,
        or if checked CPU segment lengths violate the value constraints.
    RuntimeError
        If tensors are not on CPU or CUDA. Invalid checked CUDA lengths trigger
        a device assertion, reported asynchronously as a RuntimeError.

    Notes
    -----
    ``in_channels`` and ``out_channels`` must be positive. Feature and weight
    inputs are wrapped without copying unless autocast requires dtype conversion;
    transposed, sliced, and broadcast views with nonnegative int32 byte strides
    are supported. Prefix sums and any int64-to-int32 conversion run on the input
    device. CUDA work, including temporary allocation, uses the current PyTorch
    CUDA stream.

    With ``validate=False``, valid segment values are a caller precondition.
    ``validate=True`` checks them in eager execution, ``torch.compile``, and
    CUDA graph replay. Restart the process after a failed CUDA device assertion.

    CUDA bfloat16 autocast converts float32 or bfloat16 operands to bfloat16
    before the custom autograd function. Gradients propagate through these
    casts to the original operands in their original dtypes. FP32 parameter
    gradients therefore retain BF16 numerical precision. CUDA float16 autocast
    is unsupported and raises ValueError. Outside autocast, or inside a nested
    disabled context, both operands must have the same supported dtype.
    CPU operands use their explicit dtype; CPU autocast does not convert them.

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
    nvalchemiops.jax.segment_mm.segment_mm : Differentiable JAX binding.

    Examples
    --------
    Multiply a batch with one empty segment and one padding row, then
    differentiate with respect to both operands on a CUDA device:

    >>> import torch
    >>> from nvalchemiops.torch.segment_mm import segment_mm
    >>> a = torch.ones((6, 4), device="cuda", requires_grad=True)
    >>> b = torch.ones((3, 4, 2), device="cuda", requires_grad=True)
    >>> lengths = torch.tensor([2, 0, 3], device="cuda", dtype=torch.int32)
    >>> out = segment_mm(a, b, lengths)
    >>> tuple(out.shape)
    (6, 2)
    >>> grad_a, grad_b = torch.autograd.grad(out.square().sum(), (a, b))
    >>> tuple(grad_a.shape), tuple(grad_b.shape)
    ((6, 4), (3, 4, 2))
    """
    _validate_inputs(a, b, seglen)
    if not isinstance(validate, bool):
        raise ValueError("validate must be a bool")
    if not validate:
        seglen = seglen.to(torch.int32)
    if a.device.type == "cuda" and torch.is_autocast_enabled("cuda"):
        if torch.get_autocast_dtype("cuda") != torch.bfloat16:
            raise ValueError("segment_mm supports CUDA autocast only with bfloat16")
        a, b = a.to(torch.bfloat16), b.to(torch.bfloat16)
    elif a.dtype != b.dtype:
        raise ValueError("a and b must have the same dtype outside CUDA autocast")
    return _apply_forward(a, b, seglen, validate)
