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

from collections.abc import Callable
from typing import Any

import warp as wp
from warp._src.types import type_size_in_bytes

from nvalchemiops.batch_utils import (
    _segment_chunk_rows,
    _segment_counts,
    _segment_offsets_clamped,
    _segment_offsets_clamped_fused,
    _segment_offsets_cooperative,
    _segment_offsets_small,
    _segment_offsets_small_fused,
    _zero_padding_overloads,
)
from nvalchemiops.warp_dispatch import register_overloads

__all__ = ["segment_mm", "segment_mm_backward", "segment_mm_backward_b"]

wp.set_module_options({"enable_backward": False, "fast_math": False})

_TILE_M = wp.constant(32)
_TILE_N = wp.constant(64)
_TILE_K = wp.constant(32)
_BLOCK_DIM = wp.constant(128)
_DTYPES = (wp.float32, wp.bfloat16)
_INDEX_LIMIT = 2**31 - 1 - max(_TILE_M, _TILE_N, _TILE_K)


def _make_segment_mm_kernel(transpose_b: bool) -> wp.Kernel:
    """Specialize the shared product kernel for forward or input gradients.

    The transposed specialization reads BF16 weights in their original layout
    and transposes each tile in shared memory without a full-weight copy.
    """

    @wp.kernel(enable_backward=False)
    def _segment_mm_kernel(
        a: wp.array2d(dtype=Any),
        b: wp.array3d(dtype=Any),
        row_offsets: wp.array(dtype=wp.int32),
        tile_offsets: wp.array(dtype=wp.int32),
        out: wp.array2d(dtype=Any),
    ):
        """Multiply each segment's feature rows by its weight matrix.

        Parameters
        ----------
        a : wp.array, shape (capacity, K), dtype float32 or bfloat16
            Packed feature rows.
        b : wp.array, shape (R, K, N), dtype matches a
            Per-segment weights, shaped ``(R, N, K)`` when transposed.
        row_offsets : wp.array, shape (R + 1,), dtype int32
            Cumulative active row counts with a leading zero.
        tile_offsets : wp.array, shape (R + 1,), dtype int32
            Cumulative counts of ``_TILE_M``-row tiles with a leading zero.
        out : wp.array, shape (capacity, N), dtype matches a
            OUTPUT: Pre-allocated product buffer. Only active rows are written.

        Notes
        -----
        Thread launch: tiled ``(row_tasks, column_tiles)`` with one
        cooperative block per output tile. Excess row tasks perform no work.
        ``row_tasks`` must cover ``tile_offsets[R]``, and ``column_tiles`` is
        ``ceil(N / _TILE_N)``. Launch with ``block_dim=_BLOCK_DIM``; CPU execution
        uses one lane.
        Modifies: active rows of ``out``; padding is handled separately.

        See Also
        --------
        nvalchemiops.batch_utils._zero_padding : Zero rows outside the active segments.
        """
        task, column_tile, lane = wp.tid()
        groups = b.shape[0]
        if task < tile_offsets[groups]:
            # First boundary greater than task skips empty segments at that boundary.
            group = wp.lower_bound(tile_offsets, task + 1) - 1
            row = row_offsets[group] + (task - tile_offsets[group]) * _TILE_M
            column = column_tile * _TILE_N
            # BF16 tile_matmul requires an FP32 accumulator; round only at the store.
            accumulator = wp.tile_zeros(shape=(_TILE_M, _TILE_N), dtype=wp.float32)
            reduction_tiles = a.shape[1] / _TILE_K + int(a.shape[1] % _TILE_K != 0)
            for reduction_tile in range(reduction_tiles):
                reduction = reduction_tile * _TILE_K
                a_tile = wp.tile_load(
                    a, shape=(_TILE_M, _TILE_K), offset=(row, reduction)
                )
                if wp.static(transpose_b):
                    b_rows = wp.tile_load(
                        b[group],
                        shape=(_TILE_N, _TILE_K),
                        offset=(column, reduction),
                        storage="shared",
                    )
                    b_tile = wp.tile_transpose(b_rows)
                else:
                    b_tile = wp.tile_load(
                        b[group], shape=(_TILE_K, _TILE_N), offset=(reduction, column)
                    )
                wp.tile_matmul(a_tile, b_tile, accumulator)

            for step in range((_TILE_M * _TILE_N) / wp.block_dim()):
                element = lane + step * wp.block_dim()
                local_row = element / _TILE_N
                local_column = element % _TILE_N
                if (
                    local_row < row_offsets[group + 1] - row
                    and column + local_column < out.shape[1]
                ):
                    out[row + local_row, column + local_column] = out.dtype(
                        wp.tile_extract(accumulator, local_row, local_column)
                    )

    return _segment_mm_kernel


@wp.func
def _keep_reduction_row(value: wp.float32, row: int, valid_rows: int):
    """Exclude reduction rows without propagating NaNs from neighboring segments.

    Parameters
    ----------
    value : wp.float32
        Scalar element of an input or output-cotangent tile.
    row : int
        Nonnegative int32 row index within the tile.
    valid_rows : int
        Int32 count of segment rows remaining from the tile's first row.

    Returns
    -------
    wp.float32
        ``value`` when ``row < valid_rows``; otherwise zero.

    Notes
    -----
    Thread launch: applied elementwise by ``wp.tile_map`` within a tile block.
    Modifies: no input arrays.

    See Also
    --------
    _make_segment_mm_grad_b_kernel : Mask rows before the weight-gradient reduction.
    """
    if row < valid_rows:
        return value
    return wp.float32(0)


@wp.func
def _keep_reduction_row(value: wp.bfloat16, row: int, valid_rows: int):
    """Mask BF16 reduction rows without promoting the multiplication operands.

    Parameters
    ----------
    value : wp.bfloat16
        Scalar element of an input or output-cotangent tile.
    row : int
        Nonnegative int32 row index within the tile.
    valid_rows : int
        Int32 count of segment rows remaining from the tile's first row.

    Returns
    -------
    wp.bfloat16
        ``value`` when ``row < valid_rows``; otherwise zero.

    Notes
    -----
    Thread launch: applied elementwise by ``wp.tile_map`` within a tile block.
    Modifies: no input arrays.

    See Also
    --------
    _make_segment_mm_grad_b_kernel : Mask rows before the weight-gradient reduction.
    """
    if row < valid_rows:
        return value
    return wp.bfloat16(0)


def _make_segment_mm_grad_b_kernel(direct: bool) -> wp.Kernel:
    """Specialize one shared dB body for direct or bounded-chunk scheduling."""

    @wp.kernel(enable_backward=False)
    def _segment_mm_grad_b_kernel(
        a: wp.array2d(dtype=Any),
        grad_out: wp.array2d(dtype=Any),
        row_offsets: wp.array(dtype=wp.int32),
        chunk_offsets: wp.array(dtype=wp.int32),
        chunk_rows: int,
        partials: wp.array3d(dtype=Any),
        grad_b: wp.array3d(dtype=Any),
    ):
        """Compute equally sized row chunks, writing singleton segments directly.

        Parameters
        ----------
        a : wp.array, shape (capacity, K), dtype float32 or bfloat16
            Packed feature rows.
        grad_out : wp.array, shape (capacity, N), dtype matches a
            Output cotangent. Rows outside each segment are masked before reduction.
        row_offsets : wp.array, shape (R + 1,), dtype int32
            Cumulative active row counts with a leading zero.
        chunk_offsets : wp.array, shape (R + 1,), dtype int32
            Cumulative counts of additional chunks after each segment's first.
            Unused when chunk_rows >= capacity.
        chunk_rows : int
            Static maximum chunk size. Split execution can shorten it from the
            active row total while retaining the same bounded scratch shape.
        partials : wp.array, shape (chunk_capacity, K, N)
            OUTPUT: FP32 scratch for segments spanning multiple chunks. Unused
            for singleton segments and when chunk_rows >= capacity.
        grad_b : wp.array, shape (R, K, N), dtype matches a
            OUTPUT: Final gradients for segments that require only one chunk.

        Notes
        -----
        Thread launch: tiled ``(chunk_capacity, channel_tiles)``, or
        ``(R, channel_tiles)`` when chunk_rows >= capacity, with
        ``block_dim=_BLOCK_DIM`` (one lane on CPU). Each block computes one channel
        tile for one chunk; ``channel_tiles = ceil(K / _TILE_K) * ceil(N / _TILE_N)``.
        Activation tiles are transposed in shared memory without copying all of A.
        Products accumulate in FP32; singleton segments write the final dtype.
        Modifies: active multi-chunk partials, singleton gradients, and empty-segment
        zeros. The first R tasks map directly to segments, avoiding a prefix search.

        See Also
        --------
        _keep_reduction_row : Mask rows beyond a segment boundary.
        _reduce_grad_b : Sum multiple chunks into the final weight gradient.
        """
        task, channel_tile, lane = wp.tid()
        group = task
        chunk_count = int(1)
        first = int(0)
        if wp.static(not direct):
            minimum_rows = int(256)
            if grad_b.shape[0] >= 32:
                minimum_rows = 512
            if chunk_rows > minimum_rows:
                chunk_rows = _segment_chunk_rows(
                    row_offsets[grad_b.shape[0]],
                    partials.shape[0] - grad_b.shape[0],
                    minimum_rows,
                )
            if task >= grad_b.shape[0]:
                extra = task - grad_b.shape[0]
                if extra >= chunk_offsets[grad_b.shape[0]]:
                    return
                group = wp.lower_bound(chunk_offsets, extra + 1) - 1
                first = (extra - chunk_offsets[group] + 1) * (chunk_rows / _TILE_M)
            chunk_count = chunk_offsets[group + 1] - chunk_offsets[group] + 1
        column_tiles = grad_b.shape[2] / _TILE_N + int(grad_b.shape[2] % _TILE_N != 0)
        input_channel = (channel_tile / column_tiles) * _TILE_K
        output_channel = (channel_tile % column_tiles) * _TILE_N
        length = row_offsets[group + 1] - row_offsets[group]
        row_tiles = length / _TILE_M + int(length % _TILE_M != 0)
        last = row_tiles
        if wp.static(not direct):
            tiles_per_chunk = chunk_rows / _TILE_M
            last = wp.min(first + tiles_per_chunk, row_tiles)
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
            a_tile = wp.tile_map(
                _keep_reduction_row, a_transposed_tile, a_rows, valid_rows
            )
            grad_tile = wp.tile_map(
                _keep_reduction_row, grad_tile, grad_rows, valid_rows
            )
            wp.tile_matmul(a_tile, grad_tile, accumulator)

        for step in range((_TILE_K * _TILE_N) / wp.block_dim()):
            element = lane + step * wp.block_dim()
            local_row = element / _TILE_N
            local_column = element % _TILE_N
            if (
                input_channel + local_row < grad_b.shape[1]
                and output_channel + local_column < grad_b.shape[2]
            ):
                value = wp.tile_extract(accumulator, local_row, local_column)
                if chunk_count == 1:
                    grad_b[
                        group, input_channel + local_row, output_channel + local_column
                    ] = grad_b.dtype(value)
                else:
                    partials[
                        task, input_channel + local_row, output_channel + local_column
                    ] = partials.dtype(value)

    return _segment_mm_grad_b_kernel


@wp.kernel(enable_backward=False)
def _reduce_grad_b(
    partials: wp.array3d(dtype=wp.float32),
    chunk_offsets: wp.array(dtype=wp.int32),
    elements_per_block: int,
    grad_b: wp.array3d(dtype=Any),
):
    """Reduce weight-gradient partials in FP32 and round once to the output dtype.

    Parameters
    ----------
    partials : wp.array, shape (chunk_capacity, K, N), dtype float32
        Weight-gradient partials for segments requiring multiple chunks.
    chunk_offsets : wp.array, shape (R + 1,), dtype int32
        Cumulative counts of additional chunks after each segment's first.
    elements_per_block : int
        Shape-dependent multiple of _BLOCK_DIM, at most _TILE_K * _TILE_N.
    grad_b : wp.array, shape (R, K, N), dtype float32 or bfloat16
        OUTPUT: Multi-chunk gradients. Singleton and empty segments have
        already been written and are left unchanged.

    Notes
    -----
    Thread launch: tiled ``(R, ceil(K * N / elements_per_block))`` with
    ``block_dim=_BLOCK_DIM``. Each block skips singleton segments together,
    or cooperatively reduces a contiguous tile of gradient elements.
    Modifies: multi-chunk segments in ``grad_b`` in fixed chunk order.

    See Also
    --------
    _make_segment_mm_grad_b_kernel : Compute partial weight gradients.
    """
    group, element_tile, lane = wp.tid()
    first = chunk_offsets[group]
    last = chunk_offsets[group + 1]
    if last > first:
        for step in range(elements_per_block / wp.block_dim()):
            element = element_tile * elements_per_block + lane + step * wp.block_dim()
            if element < grad_b.shape[1] * grad_b.shape[2]:
                input_channel = element / grad_b.shape[2]
                output_channel = element % grad_b.shape[2]
                value = partials[group, input_channel, output_channel]
                for task in range(first, last):
                    value += partials[
                        grad_b.shape[0] + task, input_channel, output_channel
                    ]
                grad_b[group, input_channel, output_channel] = grad_b.dtype(value)


@wp.kernel(enable_backward=False)
def _reduce_grad_b_persistent(
    partials: wp.array3d(dtype=wp.float32),
    chunk_offsets: wp.array(dtype=wp.int32),
    work_blocks: int,
    grad_b: wp.array3d(dtype=Any),
):
    """Reduce split segments with a bounded grid of 128-element output tiles.

    Parameters
    ----------
    partials : wp.array, shape (chunk_capacity, K, N), dtype float32
        Weight-gradient partials for segments spanning multiple chunks. The
        first R slots belong to the first chunk of each segment; later slots
        hold additional chunks in prefix order.
    chunk_offsets : wp.array, shape (R + 1,), dtype int32
        Cumulative additional-chunk counts with a leading zero.
    work_blocks : int
        Positive number of persistent blocks, at most 512.
    grad_b : wp.array, shape (R, K, N), dtype float32 or bfloat16
        OUTPUT: Multi-chunk gradients. Singleton and empty segments retain
        the result already written by the main kernel.

    Notes
    -----
    Thread launch: tiled ``(work_blocks,)`` with ``block_dim=_BLOCK_DIM``.
    Each block owns one residue class of flat extra-chunk/output-tile tasks.
    Only a segment's first extra chunk reduces its output; later chunks are
    skipped together without reading partials. Empty and singleton segments
    retain the result written by the main kernel.

    The main launch bound gives ``extra_capacity * tiles <= _INDEX_LIMIT / 8``,
    so flat task arithmetic, including at most 511 tasks of jump-rounding
    overshoot, fits int32.
    Modifies: multi-chunk gradients, accumulating FP32 partials in their original
    order and rounding once when storing the final output dtype.
    """
    block, lane = wp.tid()
    groups = grad_b.shape[0]
    elements = grad_b.shape[1] * grad_b.shape[2]
    tiles = elements / _BLOCK_DIM + int(elements % _BLOCK_DIM != 0)
    stop = chunk_offsets[groups] * tiles
    task = block
    while task < stop:
        extra = task / tiles
        group = wp.lower_bound(chunk_offsets, extra + 1) - 1
        first = chunk_offsets[group]
        last = chunk_offsets[group + 1]
        if extra == first:
            element_tile = task % tiles
            for step in range(_BLOCK_DIM / wp.block_dim()):
                element = element_tile * _BLOCK_DIM + lane + step * wp.block_dim()
                if element < elements:
                    input_channel = element / grad_b.shape[2]
                    output_channel = element % grad_b.shape[2]
                    value = partials[group, input_channel, output_channel]
                    for chunk in range(first, last):
                        value += partials[groups + chunk, input_channel, output_channel]
                    grad_b[group, input_channel, output_channel] = grad_b.dtype(value)
            task += work_blocks
        else:
            distance = last * tiles - task
            task += ((distance - 1) / work_blocks + 1) * work_blocks


_segment_mm_overloads = {
    transpose_b: register_overloads(
        _make_segment_mm_kernel(transpose_b),
        lambda dtype: [
            wp.array2d(dtype=dtype),
            wp.array3d(dtype=dtype),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32),
            wp.array2d(dtype=dtype),
        ],
        dtypes=(wp.bfloat16,) if transpose_b else _DTYPES,
    )
    for transpose_b in (False, True)
}
_segment_mm_grad_b_overloads = {
    direct: register_overloads(
        _make_segment_mm_grad_b_kernel(direct),
        lambda input_dtype, output_dtype: [
            wp.array2d(dtype=input_dtype),
            wp.array2d(dtype=input_dtype),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32),
            wp.int32,
            wp.array3d(dtype=output_dtype),
            wp.array3d(dtype=input_dtype),
        ],
        dtype_pairs=(
            (wp.float32, wp.float32),
            (wp.bfloat16, wp.float32),
            (wp.bfloat16, wp.bfloat16),
        ),
        key_fn=lambda input_dtype, output_dtype: (input_dtype, output_dtype),
    )
    for direct in (False, True)
}
_reduce_grad_b_overloads = register_overloads(
    _reduce_grad_b,
    lambda dtype: [
        wp.array3d(dtype=wp.float32),
        wp.array(dtype=wp.int32),
        wp.int32,
        wp.array3d(dtype=dtype),
    ],
    dtypes=_DTYPES,
)
_reduce_grad_b_persistent_overloads = register_overloads(
    _reduce_grad_b_persistent,
    lambda dtype: [
        wp.array3d(dtype=wp.float32),
        wp.array(dtype=wp.int32),
        wp.int32,
        wp.array3d(dtype=dtype),
    ],
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


def _segment_offsets(
    seglen: wp.array,
    capacity: int,
    tile_rows: int = _TILE_M,
    skip_first_tile: bool = False,
    *,
    chunk_rows: int = 0,
    chunk_offsets: wp.array | None = None,
) -> tuple[wp.array, wp.array]:
    """Allocate row offsets and cumulative counts of SegMM row-tile tasks.

    Each segment contributes ``ceil(clipped_length / tile_rows)`` tasks,
    optionally excluding its first tile for the weight-gradient schedule.
    Lengths and prefixes remain on the device. Fused CUDA schedules with
    32 to 1024 segments use one cooperative block. Other small schedules use
    one scalar launch; larger inputs use an overflow-safe int64 row scan.
    When supplied, caller-owned ``chunk_offsets`` receives the prefix of
    additional ``chunk_rows``-row chunks after each segment's first chunk.
    Both schedules share the row-prefix construction.
    Weight schedules shorten chunks from the clipped active total when the
    static size exceeds the 256/512-row floor, preserving their slot budget.
    """
    groups = seglen.shape[0]
    row_offsets = wp.empty(groups + 1, dtype=wp.int32, device=seglen.device)
    tile_offsets = wp.empty(groups + 1, dtype=wp.int32, device=seglen.device)
    fused = chunk_offsets is not None
    chunk_output = chunk_offsets if fused else tile_offsets
    minimum_rows = 512 if groups >= 32 else 256
    scheduled_rows = chunk_rows if fused else tile_rows
    minimum_chunk_rows = (
        minimum_rows
        if (fused or skip_first_tile) and minimum_rows < scheduled_rows < capacity
        else 0
    )
    if fused and seglen.device.is_cuda and 32 <= groups <= 1024:
        bucket = 1 << (groups - 1).bit_length()
        wp.launch_tiled(
            _segment_offsets_cooperative[bucket],
            dim=(1,),
            inputs=[
                seglen,
                capacity,
                tile_rows,
                skip_first_tile,
                row_offsets,
                tile_offsets,
                chunk_rows,
                chunk_offsets,
                minimum_chunk_rows,
            ],
            block_dim=min(bucket, _BLOCK_DIM),
            device=seglen.device,
        )
    elif groups <= 128:
        wp.launch(
            _segment_offsets_small_fused if fused else _segment_offsets_small,
            dim=groups + 1,
            inputs=[
                seglen,
                capacity,
                tile_rows,
                skip_first_tile,
                row_offsets,
                tile_offsets,
                chunk_rows,
                chunk_output,
                minimum_chunk_rows,
            ],
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
            _segment_offsets_clamped_fused if fused else _segment_offsets_clamped,
            dim=groups + 1,
            inputs=[
                row_prefix,
                capacity,
                tile_rows,
                skip_first_tile,
                row_offsets,
                tile_offsets,
                chunk_rows,
                chunk_output,
                minimum_chunk_rows,
            ],
            device=seglen.device,
        )
        # Each nonempty range needs at most one tile per row, so the tile
        # prefix is bounded by capacity and remains safe in int32.
        wp.utils.array_scan(tile_offsets, tile_offsets)
        if fused:
            wp.utils.array_scan(chunk_offsets, chunk_offsets)
    return row_offsets, tile_offsets


def _launch_forward(
    a: wp.array,
    b: wp.array,
    row_offsets: wp.array,
    tile_offsets: wp.array,
    out: wp.array,
    *,
    transpose_b: bool = False,
) -> None:
    """Launch products and padding writes using caller-owned offsets and output.

    Arrays must already satisfy the public shape, dtype, device, and stride
    contract. This leaf allocates no storage and uses the current Warp stream.
    """
    capacity, groups, n = a.shape[0], b.shape[0], out.shape[1]
    row_tasks = (capacity + _TILE_M - 1) // _TILE_M + groups
    column_tiles = (n + _TILE_N - 1) // _TILE_N
    if row_tasks * column_tiles * _BLOCK_DIM > _INDEX_LIMIT:
        raise ValueError("Launch dimensions exceed the int32 index range")
    if capacity == 0:
        return
    if groups == 0:
        out.zero_()
        return
    wp.launch(
        _zero_padding_overloads[a.dtype],
        dim=out.shape,
        inputs=[row_offsets, out],
        device=a.device,
    )
    wp.launch_tiled(
        _segment_mm_overloads[transpose_b][a.dtype],
        dim=(row_tasks, column_tiles),
        inputs=[a, b, row_offsets, tile_offsets, out],
        block_dim=_BLOCK_DIM,
        device=a.device,
    )


def _launch_grad_a(
    grad_out: wp.array,
    b: wp.array,
    row_offsets: wp.array,
    tile_offsets: wp.array,
    grad_a: wp.array,
    b_transposed: wp.array,
) -> None:
    """Compute input gradients from original weights using caller-owned storage.

    ``b`` has shape ``(R, K, N)`` and ``grad_out`` has shape ``(capacity, N)``.
    FP32 uses contiguous transposed weights to avoid repeated strided tile loads;
    BF16 transposes each weight tile in shared memory. FP32 ``b_transposed``
    is contiguous with shape ``(R, N, K)`` and the same dtype as B. The workspace
    is unused for BF16 or an already-contiguous transpose view.
    Offsets and contiguous ``grad_a`` satisfy ``_launch_forward`` contracts.
    No storage is allocated inside this leaf.
    """
    if grad_out.shape[0] == 0:
        return
    if b.dtype == wp.float32:
        weights = b.transpose((0, 2, 1))
        if not weights.is_contiguous:
            wp.copy(b_transposed, weights)
            weights = b_transposed
        _launch_forward(grad_out, weights, row_offsets, tile_offsets, grad_a)
    else:
        _launch_forward(
            grad_out, b, row_offsets, tile_offsets, grad_a, transpose_b=True
        )


def segment_mm(a: wp.array, b: wp.array, seglen: wp.array, out: wp.array) -> None:
    """Compute segmented matrix products with shared input and output channels.

    For ``offsets = [0, cumsum(seglen)]``, each segment ``g`` computes
    ``out[offsets[g]:offsets[g+1]] = a[offsets[g]:offsets[g+1]] @ b[g]``.
    Output rows after ``sum(seglen)`` are set to zero.

    ``row_capacity`` is the allocated row count, including padding, and
    ``num_segments`` is the number of segments. ``in_channels`` and
    ``out_channels`` are the input and output feature counts.

    Parameters
    ----------
    a : wp.array, shape (row_capacity, in_channels), dtype float32 or bfloat16
        Packed input features.
    b : wp.array, shape (num_segments, in_channels, out_channels), dtype matches a
        One weight matrix per segment.
    seglen : wp.array, shape (num_segments,), dtype int32
        Runtime device segment lengths. Values must be nonnegative and sum
        to at most ``row_capacity``. Zero-length segments are supported.
    out : wp.array, shape (row_capacity, out_channels), dtype matches a
        OUTPUT: Pre-allocated contiguous product buffer. Must not overlap
        any input's memory span, including gaps in strided views. Every
        output element is overwritten.

    Raises
    ------
    ValueError
        If array ranks, shapes, dtypes, devices, strides, output contiguity, or
        non-overlap requirements are violated.

    Notes
    -----
    Thread launch: cooperative blocks compute output tiles, and a separate
    elementwise launch zeroes padding rows.
    Modifies: every element of ``out``. Inputs are read-only.

    All arrays must reside on the same CPU or CUDA device.
    ``in_channels`` and ``out_channels`` must be positive.
    Inputs may be transposed, sliced, or broadcast views with nonnegative
    byte strides that fit signed int32. They are read without making copies.
    Products accumulate in float32 and are cast once to the output dtype.
    The float32 path does not request reduced-precision multiplication.

    Segment lengths and prefix sums stay on the device. Their value
    constraints are caller preconditions and are not checked on the host.
    For bounds safety, device offsets treat negative lengths as zero and
    truncate segment ranges at ``row_capacity`` without raising an error.
    Array sizes and launch dimensions must fit the signed 32-bit index range.
    Work is enqueued on the current Warp stream for the input device.

    Automatic Warp differentiation is disabled. Use ``segment_mm_backward``
    for explicit operand gradients. CUDA graph capture requires warming the
    kernels beforehand.

    See Also
    --------
    segment_mm_backward : Compute both operand gradients.
    segment_mm_backward_b : Compute only the weight gradient.
    nvalchemiops.torch.segment_mm.segment_mm : Differentiable PyTorch binding.
    nvalchemiops.jax.segment_mm.segment_mm : Differentiable JAX binding.
    """
    _validate_inputs(a, b, seglen)
    capacity, groups, n = a.shape[0], b.shape[0], b.shape[2]
    _validate_output(out, "out", (capacity, n), a, (a, b, seglen))
    with wp.ScopedDevice(a.device):
        if capacity == 0:
            return
        if groups == 0:
            out.zero_()
            return

        row_offsets, tile_offsets = _segment_offsets(seglen, capacity)
        _launch_forward(a, b, row_offsets, tile_offsets, out)


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


def _grad_b_reduction_elements(groups: int, k: int, n: int) -> int:
    """Size reduction blocks to expose at least 128 blocks when outputs permit."""
    if groups < 32:
        return _BLOCK_DIM
    elements = (groups * k * n + _BLOCK_DIM - 1) // _BLOCK_DIM
    elements = (elements + _BLOCK_DIM - 1) // _BLOCK_DIM * _BLOCK_DIM
    return min(_TILE_K * _TILE_N, max(_BLOCK_DIM, elements))


def _grad_b_launch_config(
    capacity: int, groups: int, k: int, n: int, itemsize: int
) -> tuple[int, int]:
    """Choose a static chunk ceiling and fixed scratch/launch capacity."""
    chunk_rows = 512 if groups >= 32 else 256
    chunk_capacity = 0
    if capacity and groups:
        slot_budget = min(max(groups, (64 << 20) // (k * n * 4)), max(128, 32 * groups))
        minimum_rows = (capacity - 1) // (slot_budget - groups + 1) + 1
        chunk_rows = (max(chunk_rows, minimum_rows) + _TILE_M - 1) // _TILE_M * _TILE_M
        # Reserve the first groups slots for direct segment mapping. At least
        # one active row is consumed before any segment can add extra chunks.
        chunk_capacity = groups + (capacity - 1) // chunk_rows
    channel_tiles = ((k + _TILE_K - 1) // _TILE_K) * ((n + _TILE_N - 1) // _TILE_N)
    direct = chunk_rows >= capacity
    tasks = groups if direct else chunk_capacity
    elements_per_block = _grad_b_reduction_elements(groups, k, n)
    reduction_tiles = (k * n + elements_per_block - 1) // elements_per_block
    if (
        max(
            tasks * channel_tiles * _BLOCK_DIM,
            (0 if direct else groups) * reduction_tiles * _BLOCK_DIM,
            (0 if direct else chunk_capacity) * k * n,
            capacity * itemsize,
            k * n * (itemsize if direct else 4),
        )
        > _INDEX_LIMIT
    ):
        raise ValueError(
            "Backward workspace or launch dimensions exceed the int32 index range"
        )
    return chunk_rows, chunk_capacity


def _launch_grad_b(
    a: wp.array,
    row_offsets: wp.array,
    chunk_offsets: wp.array,
    grad_out: wp.array,
    grad_b: wp.array,
    partials: wp.array,
    chunk_rows: int,
) -> None:
    """Write weight gradients using caller-owned scratch on the current stream.

    Pass ``grad_b`` as ``partials`` when chunk_rows >= capacity; otherwise
    pass a bounded FP32 ``(chunk_capacity, K, N)`` array. Chunk prefixes and
    shapes must have been validated before this allocation-free leaf is called.
    """
    groups, k, n = grad_b.shape
    if a.shape[0] == 0 or groups == 0:
        grad_b.zero_()
        return
    channel_tiles = ((k + _TILE_K - 1) // _TILE_K) * ((n + _TILE_N - 1) // _TILE_N)
    direct = chunk_rows >= a.shape[0]
    wp.launch_tiled(
        _segment_mm_grad_b_overloads[direct][a.dtype, partials.dtype],
        dim=(groups if direct else partials.shape[0], channel_tiles),
        inputs=[a, grad_out, row_offsets, chunk_offsets, chunk_rows, partials, grad_b],
        block_dim=_BLOCK_DIM,
        device=a.device,
    )
    if not direct:
        extra_capacity = partials.shape[0] - groups
        if 0 < extra_capacity < groups:
            tiles = (k * n + _BLOCK_DIM - 1) // _BLOCK_DIM
            work_blocks = min(512, extra_capacity * tiles)
            wp.launch_tiled(
                _reduce_grad_b_persistent_overloads[a.dtype],
                dim=(work_blocks,),
                inputs=[partials, chunk_offsets, work_blocks, grad_b],
                block_dim=_BLOCK_DIM,
                device=a.device,
            )
            return
        elements_per_block = _grad_b_reduction_elements(groups, k, n)
        wp.launch_tiled(
            _reduce_grad_b_overloads[a.dtype],
            dim=(groups, (k * n + elements_per_block - 1) // elements_per_block),
            inputs=[partials, chunk_offsets, elements_per_block, grad_b],
            block_dim=_BLOCK_DIM,
            device=a.device,
        )


def segment_mm_backward_b(
    a: wp.array, seglen: wp.array, grad_out: wp.array, grad_b: wp.array
) -> None:
    """Compute only the per-segment weight gradient ``a[g].T @ grad_out[g]``.

    ``row_capacity`` is the allocated row count, including padding, and
    ``num_segments`` is the number of segments. ``in_channels`` and
    ``out_channels`` are the input and output feature counts.

    Parameters
    ----------
    a : wp.array, shape (row_capacity, in_channels), dtype float32 or bfloat16
        Packed input features on a CPU or CUDA device.
    seglen : wp.array, shape (num_segments,), dtype int32
        Runtime device lengths, nonnegative and summing to at most ``row_capacity``.
    grad_out : wp.array, shape (row_capacity, out_channels), dtype matches a
        Output cotangent on the same device as ``a``.
    grad_b : wp.array, shape (num_segments, in_channels, out_channels), dtype matches a
        OUTPUT: Pre-allocated contiguous weight gradient on the same device
        as ``a``. Must not overlap input memory spans, including gaps in
        strided views. Every element is overwritten.

    Raises
    ------
    ValueError
        If ranks, shapes, dtypes, devices, strides, index limits, or output
        contiguity/non-overlap requirements are violated.

    Notes
    -----
    Thread launch: cooperative blocks compute channel tiles over equal-sized
    row chunks; an elementwise reduction combines multiple chunks when needed.
    Modifies: every element of ``grad_b``. Inputs are read-only.

    All arrays must reside on the same CPU or CUDA device.
    ``in_channels`` and ``out_channels`` must be positive.
    Inputs may have nonnegative int32 byte strides.
    Padding rows are ignored, including NaNs. Empty segments produce zeros.
    Native input products and chunk sums accumulate in FP32 and are cast once
    to the output dtype. FP32 multiplication uses full input precision.

    This uses the same deterministic chunk reduction as ``segment_mm_backward``,
    without computing an input gradient. A is read directly and transposed one
    tile at a time in shared memory; no full activation transpose is allocated.
    Singleton segments write ``grad_b`` directly. When ``row_capacity`` fits one chunk,
    no partial-gradient buffer or final reduction is needed.
    Length values stay on the device and their constraints are caller preconditions.
    For bounds safety, device offsets treat negative lengths as zero and
    truncate segment ranges at ``row_capacity`` without raising an error.
    Work uses the current Warp stream. CUDA graph capture requires kernel warmup.

    See Also
    --------
    segment_mm : Compute the forward segmented products.
    segment_mm_backward : Compute both operand gradients.
    """
    _segment_mm_backward_b_impl(a, seglen, grad_out, grad_b)


def _segment_mm_backward_b_impl(
    a: wp.array,
    seglen: wp.array,
    grad_out: wp.array,
    grad_b: wp.array,
    partials_factory: Callable[[tuple[int, int, int]], wp.array] | None = None,
) -> None:
    """Compute dB with optional framework-owned FP32 chunk scratch.

    The optional factory allocates fresh contiguous FP32 storage of the requested
    shape on the input device. It is called only when chunks require scratch.
    Public Warp calls allocate their own scratch at the same point.
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
        groups, k, n = grad_b.shape
        chunk_rows, chunk_capacity = _grad_b_launch_config(
            a.shape[0], groups, k, n, type_size_in_bytes(a.dtype)
        )
        row_offsets, chunk_offsets = _segment_offsets(
            seglen, a.shape[0], chunk_rows, skip_first_tile=True
        )
        if chunk_rows >= a.shape[0]:
            partials = grad_b
        elif partials_factory is None:
            partials = wp.empty(
                (chunk_capacity, k, n), dtype=wp.float32, device=a.device
            )
        else:
            partials = partials_factory((chunk_capacity, k, n))
        _launch_grad_b(
            a, row_offsets, chunk_offsets, grad_out, grad_b, partials, chunk_rows
        )


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

    ``row_capacity`` is the allocated row count, including padding, and
    ``num_segments`` is the number of segments. ``in_channels`` and
    ``out_channels`` are the input and output feature counts.

    Parameters
    ----------
    a : wp.array, shape (row_capacity, in_channels), dtype float32 or bfloat16
        Packed input features.
    b : wp.array, shape (num_segments, in_channels, out_channels), dtype matches a
        One weight matrix per segment.
    seglen : wp.array, shape (num_segments,), dtype int32
        Runtime device lengths, nonnegative and summing to at most ``row_capacity``.
        Empty segments are supported. Lengths are not differentiated.
    grad_out : wp.array, shape (row_capacity, out_channels), dtype matches a
        Output cotangent.
    grad_a : wp.array, shape (row_capacity, in_channels), dtype matches a
        OUTPUT: Pre-allocated contiguous feature gradient. Every element is
        overwritten, and padding rows are set to zero.
    grad_b : wp.array, shape (num_segments, in_channels, out_channels), dtype matches a
        OUTPUT: Pre-allocated contiguous weight gradient. Every element is
        overwritten, and empty segments are set to zero.

    Raises
    ------
    ValueError
        If shapes, dtypes, devices, strides, index limits, output contiguity,
        or output non-overlap requirements are violated.

    Notes
    -----
    Thread launch: cooperative blocks compute feature-gradient and
    weight-gradient tiles. Elementwise launches zero feature padding and
    combine weight-gradient chunks when needed.
    Modifies: every element of ``grad_a`` and ``grad_b``. Inputs are read-only.

    All arrays must reside on the same CPU or CUDA device. Inputs support nonnegative
    int32 byte strides. Outputs must not overlap each other or any input's
    memory span, including gaps in strided views.

    BF16 operands remain BF16 during multiplication. Products and chunk sums
    accumulate in FP32; gradients are cast once to their output dtype. FP32
    multiplication does not request reduced precision. Weight-gradient chunks
    use a fixed reduction order without floating-point atomics.

    FP32 input gradients use a contiguous transposed B workspace. BF16 input
    gradients transpose weight tiles in shared memory without a full-weight copy.
    Weight gradients write directly to ``grad_b`` for singleton segments,
    otherwise using FP32 partial gradients.
    A is read directly and transposed one tile at a time in shared memory.
    Partial storage is limited to 64 MiB or one FP32 B-sized buffer,
    whichever is larger. The chunk ceiling and maximum task count depend only
    on static buffer shapes. Runtime active rows can shorten chunks within the
    same scratch capacity; prefixes assign more chunks to long segments.
    Segment lengths stay on the device; their value constraints remain caller
    preconditions. Work uses the current Warp stream for the input device.
    For bounds safety, device offsets treat negative lengths as zero and
    truncate segment ranges at ``row_capacity`` without raising an error.

    This is an explicit first-order backward operation, not an automatic Warp
    adjoint or a framework autodiff registration. CUDA graph capture requires
    warming the kernels beforehand.

    See Also
    --------
    segment_mm : Compute the forward segmented products.
    segment_mm_backward_b : Compute only the weight gradient.
    """
    _segment_mm_backward_impl(a, b, seglen, grad_out, grad_a, grad_b)


def _segment_mm_backward_impl(
    a: wp.array,
    b: wp.array,
    seglen: wp.array,
    grad_out: wp.array,
    grad_a: wp.array,
    grad_b: wp.array,
    partials_factory: Callable[[tuple[int, int, int]], wp.array] | None = None,
) -> None:
    """Compute both gradients with optional framework-owned FP32 chunk scratch.

    The optional factory satisfies ``_segment_mm_backward_b_impl``'s contract.
    Validation, scheduling, and numerical kernels are shared with the public
    Warp operation. Scratch is allocated after enqueuing the input gradient.
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

        chunk_rows, chunk_capacity = _grad_b_launch_config(
            capacity, groups, k, n, type_size_in_bytes(a.dtype)
        )
        row_tasks = (capacity + _TILE_M - 1) // _TILE_M + groups
        da_column_tiles = (k + _TILE_N - 1) // _TILE_N
        if row_tasks * da_column_tiles * _BLOCK_DIM > _INDEX_LIMIT:
            raise ValueError(
                "Backward workspace or launch dimensions exceed the int32 index range"
            )

        if chunk_rows < capacity:
            chunk_offsets = wp.empty(groups + 1, dtype=wp.int32, device=seglen.device)
            row_offsets, tile_offsets = _segment_offsets(
                seglen, capacity, chunk_rows=chunk_rows, chunk_offsets=chunk_offsets
            )
        else:
            row_offsets, tile_offsets = _segment_offsets(seglen, capacity)
            chunk_offsets = tile_offsets
        b_transposed = b.transpose((0, 2, 1))
        if b.dtype == wp.float32 and not b_transposed.is_contiguous:
            b_transposed = wp.empty(b_transposed.shape, dtype=b.dtype, device=b.device)
        _launch_grad_a(grad_out, b, row_offsets, tile_offsets, grad_a, b_transposed)
        if chunk_rows >= capacity:
            partials = grad_b
        elif partials_factory is None:
            partials = wp.empty(
                (chunk_capacity, k, n), dtype=wp.float32, device=a.device
            )
        else:
            partials = partials_factory((chunk_capacity, k, n))
        _launch_grad_b(
            a, row_offsets, chunk_offsets, grad_out, grad_b, partials, chunk_rows
        )
