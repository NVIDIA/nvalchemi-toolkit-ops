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

"""Warp segmented matrix multiplication and explicit operand gradients."""

from typing import Any

import warp as wp
from warp._src.types import type_size_in_bytes

from nvalchemiops.warp_dispatch import register_overloads

__all__ = ["segment_mm", "segment_mm_backward", "segment_mm_backward_b"]

wp.set_module_options({"enable_backward": False, "fast_math": False})

_TILE_M = wp.constant(32)
_TILE_N = wp.constant(64)
_TILE_K = wp.constant(32)
_BLOCK_DIM = wp.constant(128)
_DTYPES = (wp.float32, wp.bfloat16)
_INDEX_LIMIT = 2**31 - 1 - max(_TILE_M, _TILE_N, _TILE_K)


@wp.kernel(enable_backward=False)
def _segment_counts(
    seglen: wp.array(dtype=wp.int32),
    row_counts: wp.array(dtype=wp.int64),
):
    """Write nonnegative row counts for an overflow-safe inclusive device scan.

    Notes
    -----
    Thread launch: one thread per segment plus one for the leading zero.
    Modifies: every element of ``row_counts``.
    """
    index = wp.tid()
    if index == 0:
        row_counts[0] = wp.int64(0)
    else:
        row_counts[index] = wp.int64(wp.max(seglen[index - 1], 0))


@wp.kernel(enable_backward=False)
def _segment_offsets_small(
    seglen: wp.array(dtype=wp.int32),
    capacity: int,
    row_offsets: wp.array(dtype=wp.int32),
    tile_offsets: wp.array(dtype=wp.int32),
):
    """Build capacity-bounded prefix arrays directly for at most 128 segments.

    Notes
    -----
    Thread launch: one thread per segment plus one for the leading zero.
    Each thread sums at most 128 preceding segments.
    Modifies: every element of ``row_offsets`` and ``tile_offsets``.
    """
    index = wp.tid()
    rows = int(0)
    tiles = int(0)
    for group in range(index):
        length = wp.min(wp.max(seglen[group], 0), capacity - rows)
        rows += length
        tiles += length / _TILE_M + int(length % _TILE_M != 0)
    row_offsets[index] = rows
    tile_offsets[index] = tiles


@wp.kernel(enable_backward=False)
def _segment_offsets_clamped(
    row_prefix: wp.array(dtype=wp.int64),
    capacity: int,
    row_offsets: wp.array(dtype=wp.int32),
    tile_offsets: wp.array(dtype=wp.int32),
):
    """Clamp wide row prefixes to capacity and count tiles over the clipped ranges.

    Notes
    -----
    Thread launch: one thread per segment plus one for the leading zero.
    Modifies: every element of ``row_offsets`` and ``tile_offsets``.
    """
    index = wp.tid()
    row = int(wp.min(row_prefix[index], wp.int64(capacity)))
    row_offsets[index] = row
    if index == 0:
        tile_offsets[0] = 0
    else:
        previous = int(wp.min(row_prefix[index - 1], wp.int64(capacity)))
        length = row - previous
        tile_offsets[index] = length / _TILE_M + int(length % _TILE_M != 0)


@wp.func
def _find_segment(tile_offsets: wp.array(dtype=wp.int32), task: int) -> int:
    """Find the nonempty segment owning a row-tile task."""
    lower = int(0)
    upper = tile_offsets.shape[0] - 1
    while lower < upper:
        middle = lower + (upper - lower) / 2
        if tile_offsets[middle + 1] <= task:
            lower = middle + 1
        else:
            upper = middle
    return lower


@wp.kernel(enable_backward=False)
def _segment_mm_kernel(
    a: wp.array2d(dtype=Any),
    b: wp.array3d(dtype=Any),
    row_offsets: wp.array(dtype=wp.int32),
    tile_offsets: wp.array(dtype=wp.int32),
    out: wp.array2d(dtype=Any),
):
    """Multiply each segment's feature rows by its weight matrix.

    Notes
    -----
    Thread launch: ``(row_tasks, column_tiles, _BLOCK_DIM)`` with one
    cooperative block per output tile. Excess row tasks perform no work.
    Modifies: active rows of ``out``; padding is handled separately.
    """
    task, column_tile, lane = wp.tid()
    groups = b.shape[0]
    if task < tile_offsets[groups]:
        group = _find_segment(tile_offsets, task)
        row = row_offsets[group] + (task - tile_offsets[group]) * _TILE_M
        column = column_tile * _TILE_N
        accumulator = wp.tile_zeros(shape=(_TILE_M, _TILE_N), dtype=wp.float32)
        reduction_tiles = a.shape[1] / _TILE_K + int(a.shape[1] % _TILE_K != 0)
        for reduction_tile in range(reduction_tiles):
            reduction = reduction_tile * _TILE_K
            a_tile = wp.tile_load(a, shape=(_TILE_M, _TILE_K), offset=(row, reduction))
            b_tile = wp.tile_load(
                b[group], shape=(_TILE_K, _TILE_N), offset=(reduction, column)
            )
            wp.tile_matmul(a_tile, b_tile, accumulator)

        for step in range((_TILE_M * _TILE_N) / _BLOCK_DIM):
            element = lane + step * _BLOCK_DIM
            local_row = element / _TILE_N
            local_column = element % _TILE_N
            if (
                local_row < row_offsets[group + 1] - row
                and column + local_column < out.shape[1]
            ):
                out[row + local_row, column + local_column] = out.dtype(
                    wp.tile_extract(accumulator, local_row, local_column)
                )


@wp.kernel(enable_backward=False)
def _zero_padding(
    row_offsets: wp.array(dtype=wp.int32),
    out: wp.array2d(dtype=Any),
):
    """Zero output rows outside the active segments without reading input padding.

    Notes
    -----
    Thread launch: one thread per output element.
    Modifies: rows at or beyond the total active row count in ``out``.
    """
    row, column = wp.tid()
    if row >= row_offsets[row_offsets.shape[0] - 1]:
        out[row, column] = out.dtype(0)


@wp.func
def _keep_reduction_row(value: wp.float32, row: int, valid_rows: int):
    """Exclude reduction rows without propagating NaNs from neighboring segments."""
    if row < valid_rows:
        return value
    return wp.float32(0)


@wp.func
def _keep_reduction_row(value: wp.bfloat16, row: int, valid_rows: int):
    """Mask BF16 reduction rows without promoting the multiplication operands."""
    if row < valid_rows:
        return value
    return wp.bfloat16(0)


@wp.kernel(enable_backward=False)
def _segment_mm_grad_b_kernel(
    a: wp.array2d(dtype=Any),
    grad_out: wp.array2d(dtype=Any),
    row_offsets: wp.array(dtype=wp.int32),
    splits: int,
    partials: wp.array3d(dtype=Any),
):
    """Write weight gradients or FP32 partials over disjoint segment row ranges.

    Notes
    -----
    Thread launch: ``(groups * splits, channel_tiles, _BLOCK_DIM)``.
    Activation tiles are transposed in shared memory without copying all of A.
    Products accumulate in FP32; a single split writes the final output dtype.
    Modifies: every element of ``partials``, including empty split ranges.
    """
    group_split, channel_tile, lane = wp.tid()
    group = group_split / splits
    split = group_split % splits
    column_tiles = partials.shape[2] / _TILE_N + int(partials.shape[2] % _TILE_N != 0)
    input_channel = (channel_tile / column_tiles) * _TILE_K
    output_channel = (channel_tile % column_tiles) * _TILE_N
    length = row_offsets[group + 1] - row_offsets[group]
    row_tiles = length / _TILE_M + int(length % _TILE_M != 0)
    tiles_per_split = (row_tiles + splits - 1) / splits
    first = split * tiles_per_split
    last = wp.min(first + tiles_per_split, row_tiles)
    accumulator = wp.tile_zeros(shape=(_TILE_K, _TILE_N), dtype=wp.float32)
    rows = wp.tile_arange(0, _TILE_M, dtype=wp.int32)
    a_rows = wp.tile_broadcast(
        wp.tile_reshape(rows, shape=(1, _TILE_M)), shape=(_TILE_K, _TILE_M)
    )
    grad_rows = wp.tile_broadcast(
        wp.tile_reshape(rows, shape=(_TILE_M, 1)), shape=(_TILE_M, _TILE_N)
    )
    for row_tile in range(first, last):
        row = row_offsets[group] + row_tile * _TILE_M
        valid_rows = row_offsets[group + 1] - row
        a_rows_tile = wp.tile_load(
            a,
            shape=(_TILE_M, _TILE_K),
            offset=(row, input_channel),
            storage="shared",
        )
        a_transposed_tile = wp.tile_transpose(a_rows_tile)
        grad_tile = wp.tile_load(
            grad_out,
            shape=(_TILE_M, _TILE_N),
            offset=(row, output_channel),
            storage="shared",
        )
        a_tile = wp.tile_map(_keep_reduction_row, a_transposed_tile, a_rows, valid_rows)
        grad_tile = wp.tile_map(_keep_reduction_row, grad_tile, grad_rows, valid_rows)
        wp.tile_matmul(a_tile, grad_tile, accumulator)

    for step in range((_TILE_K * _TILE_N) / _BLOCK_DIM):
        element = lane + step * _BLOCK_DIM
        local_row = element / _TILE_N
        local_column = element % _TILE_N
        if (
            input_channel + local_row < partials.shape[1]
            and output_channel + local_column < partials.shape[2]
        ):
            partials[
                group_split, input_channel + local_row, output_channel + local_column
            ] = partials.dtype(wp.tile_extract(accumulator, local_row, local_column))


@wp.kernel(enable_backward=False)
def _reduce_grad_b(
    partials: wp.array3d(dtype=wp.float32),
    splits: int,
    grad_b: wp.array3d(dtype=Any),
):
    """Reduce weight-gradient partials in FP32 and round once to the output dtype.

    Notes
    -----
    Thread launch: one thread per element of ``grad_b``.
    Modifies: every element of ``grad_b`` in a fixed split-reduction order.
    """
    group, input_channel, output_channel = wp.tid()
    value = wp.float32(0)
    for split in range(splits):
        value += partials[group * splits + split, input_channel, output_channel]
    grad_b[group, input_channel, output_channel] = grad_b.dtype(value)


_segment_mm_overloads = register_overloads(
    _segment_mm_kernel,
    lambda dtype: [
        wp.array2d(dtype=dtype),
        wp.array3d(dtype=dtype),
        wp.array(dtype=wp.int32),
        wp.array(dtype=wp.int32),
        wp.array2d(dtype=dtype),
    ],
    dtypes=_DTYPES,
)
_zero_padding_overloads = register_overloads(
    _zero_padding,
    lambda dtype: [wp.array(dtype=wp.int32), wp.array2d(dtype=dtype)],
    dtypes=_DTYPES,
)
_segment_mm_grad_b_overloads = register_overloads(
    _segment_mm_grad_b_kernel,
    lambda input_dtype, output_dtype: [
        wp.array2d(dtype=input_dtype),
        wp.array2d(dtype=input_dtype),
        wp.array(dtype=wp.int32),
        wp.int32,
        wp.array3d(dtype=output_dtype),
    ],
    dtype_pairs=(
        (wp.float32, wp.float32),
        (wp.bfloat16, wp.float32),
        (wp.bfloat16, wp.bfloat16),
    ),
    key_fn=lambda input_dtype, output_dtype: (input_dtype, output_dtype),
)
_reduce_grad_b_overloads = register_overloads(
    _reduce_grad_b,
    lambda dtype: [wp.array3d(dtype=wp.float32), wp.int32, wp.array3d(dtype=dtype)],
    dtypes=_DTYPES,
)


def _validate_array(array: wp.array, name: str, ndim: int) -> None:
    """Validate array metadata used by forward and backward launches."""
    if not isinstance(array, wp.array) or array.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}D Warp array")
    if any(stride < 0 or stride > 2**31 - 1 for stride in array.strides):
        raise ValueError(f"{name} byte strides must be nonnegative and fit int32")
    if array.size > _INDEX_LIMIT:
        raise ValueError(f"{name} array size exceeds the int32 index range")


def _validate_inputs(a: wp.array, b: wp.array, seglen: wp.array) -> None:
    """Validate the shared segmented matrix multiplication input contract."""
    for name, array, ndim in (("a", a, 2), ("b", b, 3), ("seglen", seglen, 1)):
        _validate_array(array, name, ndim)
    if b.device != a.device or seglen.device != a.device:
        raise ValueError("All arrays must be on the same device")
    if not a.device.is_cuda:
        raise RuntimeError("segment_mm requires a CUDA device")
    if a.dtype not in _DTYPES or b.dtype != a.dtype:
        raise ValueError("a and b must have the same float32 or bfloat16 dtype")
    if seglen.dtype != wp.int32:
        raise ValueError("seglen must have dtype int32")
    if a.shape[1] <= 0 or b.shape[2] <= 0:
        raise ValueError("Channel dimensions K and N must be positive")
    if b.shape[1] != a.shape[1]:
        raise ValueError("Input channel dimensions of a and b must match")
    if seglen.shape[0] != b.shape[0]:
        raise ValueError("seglen must contain one length per segment in b")


def _validate_output(
    out: wp.array,
    name: str,
    shape: tuple[int, ...],
    like: wp.array,
    inputs: tuple[wp.array, ...],
) -> None:
    """Validate a contiguous output that cannot overlap any input memory span."""
    _validate_array(out, name, len(shape))
    if not out.is_contiguous:
        raise ValueError(f"{name} must be contiguous")
    if out.device != like.device:
        raise ValueError("All arrays must be on the same device")
    if out.dtype != like.dtype:
        raise ValueError(f"{name} must have the same dtype as a")
    if out.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {out.shape}")
    if out.size:
        out_end = out.ptr + out.size * type_size_in_bytes(out.dtype)
        for array in inputs:
            if not array.size:
                continue
            input_end = (
                array.ptr
                + type_size_in_bytes(array.dtype)
                + sum(
                    (size - 1) * stride
                    for size, stride in zip(array.shape, array.strides, strict=True)
                )
            )
            if out.ptr < input_end and array.ptr < out_end:
                raise ValueError(
                    f"{name} must not overlap an input buffer or another output"
                )


def _segment_offsets(seglen: wp.array, capacity: int) -> tuple[wp.array, wp.array]:
    """Build capacity-bounded offsets without reading device lengths on the host."""
    groups = seglen.shape[0]
    row_offsets = wp.empty(groups + 1, dtype=wp.int32, device=seglen.device)
    tile_offsets = wp.empty(groups + 1, dtype=wp.int32, device=seglen.device)
    if groups <= 128:
        wp.launch(
            _segment_offsets_small,
            dim=groups + 1,
            inputs=[seglen, capacity, row_offsets, tile_offsets],
            device=seglen.device,
        )
    else:
        # Both group count and each nonnegative length fit int32, so their
        # sum fits int64 even when the caller's length preconditions are broken.
        row_prefix = wp.empty(groups + 1, dtype=wp.int64, device=seglen.device)
        wp.launch(
            _segment_counts,
            dim=groups + 1,
            inputs=[seglen, row_prefix],
            device=seglen.device,
        )
        wp.utils.array_scan(row_prefix, row_prefix)
        wp.launch(
            _segment_offsets_clamped,
            dim=groups + 1,
            inputs=[row_prefix, capacity, row_offsets, tile_offsets],
            device=seglen.device,
        )
        # Each nonempty range needs at most one tile per row, so the tile
        # prefix is bounded by capacity and remains safe in int32.
        wp.utils.array_scan(tile_offsets, tile_offsets)
    return row_offsets, tile_offsets


def segment_mm(a: wp.array, b: wp.array, seglen: wp.array, out: wp.array) -> None:
    """Compute segmented matrix products with shared input and output channels.

    For ``offsets = [0, cumsum(seglen)]``, each segment ``g`` computes
    ``out[offsets[g]:offsets[g+1]] = a[offsets[g]:offsets[g+1]] @ b[g]``.
    Output rows after ``sum(seglen)`` are set to zero.

    Parameters
    ----------
    a : wp.array, shape (capacity, K)
        Input features, with dtype float32 or bfloat16.
    b : wp.array, shape (R, K, N)
        Per-segment weights, with the same dtype as ``a``.
    seglen : wp.array, shape (R,), dtype int32
        Runtime device segment lengths. Values must be nonnegative and sum
        to at most ``capacity``. Zero-length segments are supported.
    out : wp.array, shape (capacity, N)
        Contiguous output buffer with the same dtype as ``a``. Must not
        overlap any input's memory span, including gaps in strided views.
        Every output element is overwritten.

    Raises
    ------
    ValueError
        If array ranks, shapes, dtypes, devices, strides, output contiguity, or
        non-overlap requirements are violated.
    RuntimeError
        If the arrays are not on a CUDA device.

    Notes
    -----
    All arrays must reside on the same CUDA device. K and N must be positive.
    Inputs may be transposed, sliced, or broadcast views with nonnegative
    byte strides that fit signed int32. They are read without making copies.
    Products accumulate in float32 and are cast once to the output dtype.
    The float32 path does not request reduced-precision multiplication.

    Segment lengths and prefix sums stay on the device. Their value
    constraints are caller preconditions and are not checked on the host.
    For bounds safety, device offsets treat negative lengths as zero and
    truncate segment ranges at capacity without raising an error.
    Array sizes and launch dimensions must fit the signed 32-bit index range.
    Work is enqueued on the current Warp stream for the input device.

    Automatic Warp differentiation is disabled. Use ``segment_mm_backward``
    for explicit operand gradients. CUDA graph capture requires warming the
    kernels beforehand.
    """
    _validate_inputs(a, b, seglen)
    capacity, groups, n = a.shape[0], b.shape[0], b.shape[2]
    _validate_output(out, "out", (capacity, n), a, (a, b, seglen))
    row_tasks = (capacity + _TILE_M - 1) // _TILE_M + groups
    column_tiles = (n + _TILE_N - 1) // _TILE_N
    if row_tasks * column_tiles * _BLOCK_DIM > _INDEX_LIMIT:
        raise ValueError("Launch dimensions exceed the int32 index range")

    with wp.ScopedDevice(a.device):
        if capacity == 0:
            return
        if groups == 0:
            out.zero_()
            return

        row_offsets, tile_offsets = _segment_offsets(seglen, capacity)
        wp.launch(
            _zero_padding_overloads[a.dtype],
            dim=out.shape,
            inputs=[row_offsets, out],
            device=a.device,
        )
        wp.launch(
            _segment_mm_overloads[a.dtype],
            dim=(row_tasks, column_tiles, _BLOCK_DIM),
            inputs=[a, b, row_offsets, tile_offsets, out],
            block_dim=_BLOCK_DIM,
            device=a.device,
        )


def _validate_grad_out(a: wp.array, grad_out: wp.array, n: int) -> None:
    """Validate a row cotangent without reading its values."""
    _validate_array(grad_out, "grad_out", 2)
    if grad_out.device != a.device:
        raise ValueError("All arrays must be on the same device")
    if grad_out.dtype != a.dtype:
        raise ValueError("grad_out must have the same dtype as a")
    shape = (a.shape[0], n)
    if grad_out.shape != shape:
        raise ValueError(f"grad_out must have shape {shape}, got {grad_out.shape}")


def _grad_b_launch_config(a: wp.array, grad_b: wp.array) -> tuple[int, int]:
    """Choose shape-dependent splits and validate weight-gradient workspace limits."""
    capacity, k = a.shape
    groups, _, n = grad_b.shape
    splits = max(1, min(32, (capacity + groups * 256 - 1) // (groups * 256)))
    splits = min(splits, max(1, (64 << 20) // (groups * k * n * 4)))
    channel_tiles = ((k + _TILE_K - 1) // _TILE_K) * ((n + _TILE_N - 1) // _TILE_N)
    if (
        max(
            groups * splits * channel_tiles * _BLOCK_DIM,
            groups * splits * k * n,
            capacity * type_size_in_bytes(a.dtype),
            k * n * 4,
        )
        > _INDEX_LIMIT
    ):
        raise ValueError(
            "Backward workspace or launch dimensions exceed the int32 index range"
        )
    return splits, channel_tiles


def _launch_grad_b(
    a: wp.array,
    grad_out: wp.array,
    row_offsets: wp.array,
    grad_b: wp.array,
    splits: int,
    channel_tiles: int,
) -> None:
    """Write a single split directly, or reduce FP32 partials without copying A."""
    groups, k, n = grad_b.shape
    partials = (
        grad_b
        if splits == 1
        else wp.empty((groups * splits, k, n), dtype=wp.float32, device=a.device)
    )
    wp.launch(
        _segment_mm_grad_b_overloads[a.dtype, partials.dtype],
        dim=(groups * splits, channel_tiles, _BLOCK_DIM),
        inputs=[a, grad_out, row_offsets, splits, partials],
        block_dim=_BLOCK_DIM,
        device=a.device,
    )
    if splits > 1:
        wp.launch(
            _reduce_grad_b_overloads[a.dtype],
            dim=grad_b.shape,
            inputs=[partials, splits, grad_b],
            device=a.device,
        )


def segment_mm_backward_b(
    a: wp.array, seglen: wp.array, grad_out: wp.array, grad_b: wp.array
) -> None:
    """Compute only the per-segment weight gradient ``a[g].T @ grad_out[g]``.

    Parameters
    ----------
    a : wp.array, shape (capacity, K)
        CUDA features with dtype float32 or bfloat16.
    seglen : wp.array, shape (R,), dtype int32
        Runtime CUDA lengths, nonnegative and summing to at most capacity.
    grad_out : wp.array, shape (capacity, N)
        CUDA row cotangent with the same dtype and device as ``a``.
    grad_b : wp.array, shape (R, K, N)
        Contiguous output with the same dtype and device as ``a``. Must not
        overlap input memory spans. Every element is overwritten.

    Raises
    ------
    ValueError
        If ranks, shapes, dtypes, devices, strides, index limits, or output
        contiguity/non-overlap requirements are violated.
    RuntimeError
        If the arrays are not on CUDA.

    Notes
    -----
    K and N must be positive. Inputs may have nonnegative int32 byte strides.
    Padding rows are ignored, including NaNs. Empty segments produce zeros.
    Native input products and split sums accumulate in FP32 and are cast once
    to the output dtype. FP32 multiplication uses full input precision.

    This uses the same deterministic split reduction as ``segment_mm_backward``,
    without computing an input gradient. A is read directly and transposed one
    tile at a time in shared memory; no full activation transpose is allocated.
    A single split writes ``grad_b`` directly without a partial-gradient buffer.
    Length values stay on the GPU and their constraints are caller preconditions.
    For bounds safety, device offsets treat negative lengths as zero and
    truncate segment ranges at capacity without raising an error.
    Work uses the current Warp stream. CUDA graph capture requires kernel warmup.

    See Also
    --------
    segment_mm_backward : Compute both operand gradients.
    """
    _validate_inputs(a, grad_b, seglen)
    _validate_grad_out(a, grad_out, grad_b.shape[2])
    _validate_output(
        grad_b,
        "grad_b",
        (seglen.shape[0], a.shape[1], grad_out.shape[1]),
        a,
        (a, seglen, grad_out),
    )
    with wp.ScopedDevice(a.device):
        if a.shape[0] == 0 or seglen.shape[0] == 0:
            grad_b.zero_()
            return
        splits, channel_tiles = _grad_b_launch_config(a, grad_b)
        row_offsets, _ = _segment_offsets(seglen, a.shape[0])
        _launch_grad_b(a, grad_out, row_offsets, grad_b, splits, channel_tiles)


def segment_mm_backward(
    a: wp.array,
    b: wp.array,
    seglen: wp.array,
    grad_out: wp.array,
    grad_a: wp.array,
    grad_b: wp.array,
) -> None:
    """Compute gradients of segmented matrix multiplication with respect to A and B.

    Each segment computes ``grad_a[g] = grad_out[g] @ b[g].T`` and
    ``grad_b[g] = a[g].T @ grad_out[g]``, where ``a[g]`` and ``grad_out[g]``
    denote the row ranges selected by ``seglen``. Padding in ``grad_out`` is
    ignored. Both outputs are overwritten, not accumulated into.

    Parameters
    ----------
    a : wp.array, shape (capacity, K)
        Input features, with dtype float32 or bfloat16.
    b : wp.array, shape (R, K, N)
        Per-segment weights, with the same dtype as ``a``.
    seglen : wp.array, shape (R,), dtype int32
        Runtime device lengths, nonnegative and summing to at most capacity.
        Empty segments are supported. Lengths are not differentiated.
    grad_out : wp.array, shape (capacity, N)
        Output cotangent, with the same dtype as ``a``.
    grad_a : wp.array, shape (capacity, K)
        Contiguous output gradient, with the same dtype as ``a``. Padding
        rows are set to zero.
    grad_b : wp.array, shape (R, K, N)
        Contiguous output gradient, with the same dtype as ``a``. Empty
        segments are set to zero.

    Raises
    ------
    ValueError
        If shapes, dtypes, devices, strides, index limits, output contiguity,
        or output non-overlap requirements are violated.
    RuntimeError
        If the arrays are not on a CUDA device.

    Notes
    -----
    All arrays must reside on the same CUDA device. Inputs support nonnegative
    int32 byte strides. Outputs must not overlap each other or any input's
    memory span, including gaps in strided views.

    BF16 operands remain BF16 during multiplication. Products and split sums
    accumulate in FP32; gradients are cast once to their output dtype. FP32
    multiplication does not request reduced precision. Weight-gradient splits
    use a fixed reduction order without floating-point atomics.

    Backward uses a GPU transpose workspace for B. Weight gradients write
    directly to ``grad_b`` for one split, otherwise using FP32 partial gradients.
    A is read directly and transposed one tile at a time in shared memory.
    Partial storage is limited to 64 MiB or one FP32 B-sized buffer,
    whichever is larger. Splits depend only on static buffer shapes. Segment
    lengths stay on the device; their value constraints remain caller
    preconditions. Work uses the current Warp stream for the input device.
    For bounds safety, device offsets treat negative lengths as zero and
    truncate segment ranges at capacity without raising an error.

    This is an explicit first-order backward operation, not an automatic Warp
    adjoint or a framework autodiff registration. CUDA graph capture requires
    warming the kernels beforehand.
    """
    _validate_inputs(a, b, seglen)
    capacity, k = a.shape
    groups, _, n = b.shape
    _validate_grad_out(a, grad_out, n)
    inputs = (a, b, seglen, grad_out)
    _validate_output(grad_a, "grad_a", a.shape, a, inputs)
    _validate_output(grad_b, "grad_b", b.shape, a, (*inputs, grad_a))

    with wp.ScopedDevice(a.device):
        if capacity == 0 or groups == 0:
            grad_a.zero_()
            grad_b.zero_()
            return

        splits, db_channel_tiles = _grad_b_launch_config(a, grad_b)
        row_tasks = (capacity + _TILE_M - 1) // _TILE_M + groups
        da_column_tiles = (k + _TILE_N - 1) // _TILE_N
        if row_tasks * da_column_tiles * _BLOCK_DIM > _INDEX_LIMIT:
            raise ValueError(
                "Backward workspace or launch dimensions exceed the int32 index range"
            )

        row_offsets, tile_offsets = _segment_offsets(seglen, capacity)
        b_transposed = b.transpose((0, 2, 1)).contiguous()
        wp.launch(
            _zero_padding_overloads[a.dtype],
            dim=grad_a.shape,
            inputs=[row_offsets, grad_a],
            device=a.device,
        )
        wp.launch(
            _segment_mm_overloads[a.dtype],
            dim=(row_tasks, da_column_tiles, _BLOCK_DIM),
            inputs=[grad_out, b_transposed, row_offsets, tile_offsets, grad_a],
            block_dim=_BLOCK_DIM,
            device=a.device,
        )
        _launch_grad_b(a, grad_out, row_offsets, grad_b, splits, db_channel_tiles)
