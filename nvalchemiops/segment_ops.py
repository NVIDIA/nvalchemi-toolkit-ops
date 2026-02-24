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

"""Segmented operations for sorted segment indices.

Provides segmented reductions (sum, component_sum, dot, max_norm) and
broadcasts (mul, add, matvec) over arrays grouped by sorted segment
indices *idx*.

Reduction kernels use a run-length approach: each thread processes a
contiguous chunk of elements, accumulates within runs of identical
segment ids, and emits one atomic per segment boundary.  The chunk size
is auto-tuned based on N and the GPU's SM count.
"""

from __future__ import annotations

from typing import Any

import warp as wp

from nvalchemiops.warp_dispatch import register_overloads

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TILE_SIZE = wp.constant(256)
_BLOCK_DIM = 256

# Types that support atomic_add / atomic_max / atomic_min (used in reductions)
_SCALAR_TYPES = [wp.float32, wp.float64]
_VEC_TYPES = [wp.vec3f, wp.vec3d]

# All scalar/vec/mat types including float16 (element-wise / broadcast only)
_ALL_SCALAR_TYPES = [wp.float16, wp.float32, wp.float64]
_ALL_VEC_TYPES = [wp.vec3h, wp.vec3f, wp.vec3d]
_MAT_TYPES = [wp.mat33h, wp.mat33f, wp.mat33d]

_VEC_TO_SCALAR = {wp.vec3h: wp.float16, wp.vec3f: wp.float32, wp.vec3d: wp.float64}
_SCALAR_TO_VEC = {wp.float16: wp.vec3h, wp.float32: wp.vec3f, wp.float64: wp.vec3d}

# Dtype pairs for register_overloads (pair mode)
_VEC_SCALAR_PAIRS = tuple(zip(_VEC_TYPES, _SCALAR_TYPES))
_ALL_VEC_SCALAR_PAIRS = tuple(zip(_ALL_VEC_TYPES, _ALL_SCALAR_TYPES))
_VEC_MAT_PAIRS = tuple(zip(_ALL_VEC_TYPES, _MAT_TYPES))

# Reduction-safe types (atomic ops work)
_SUPPORTED_TYPES = [wp.float32, wp.float64, wp.vec3f, wp.vec3d]
# All types including float16/vec3h (for element-wise/broadcast ops)
_ALL_SUPPORTED_TYPES = [
    wp.float16,
    wp.float32,
    wp.float64,
    wp.vec3h,
    wp.vec3f,
    wp.vec3d,
]


def compute_ept(N: int, sm_count: int, is_vec3: bool) -> int:
    """Return the elements-per-thread (EPT) for segmented reduction kernels.

    The value is derived from *N* (total element count) and the GPU's
    streaming-multiprocessor count so that the grid neither under- nor
    over-subscribes the device.  The raw ratio ``N / (sm_count * 512)``
    is rounded to the nearest power of two (rounding down on a tie) and
    then clamped to ``[ept_min, ept_max]``.

    Parameters
    ----------
    N : int
        Total number of elements to reduce.
    sm_count : int
        Number of streaming multiprocessors on the target device.
    is_vec3 : bool
        If ``True`` the reduction operates on vec3 data; limits are
        tighter (``ept_min=2, ept_max=8``) than for scalars
        (``ept_min=4, ept_max=16``).

    Returns
    -------
    int
        Optimal elements-per-thread, guaranteed to be a power of two
        within ``[ept_min, ept_max]``.
    """
    w_fill = sm_count * 512
    ept_max = 8 if is_vec3 else 16
    ept_min = 2 if is_vec3 else 4
    ept = max(1, N // w_fill)
    p = 1
    while p < ept:
        p <<= 1
    if p > 1 and (p - ept) > (ept - (p >> 1)):
        p >>= 1
    return max(ept_min, min(p, ept_max))


def _launch_rle(
    overloads: dict, key, N: int, device, *args, is_vec: bool = False
) -> None:
    """Launch an RLE-style reduction kernel with auto-tuned elements-per-thread.

    All RLE kernels share the convention that their last two arguments are
    ``(N, elems_per_thread)``.  This helper computes the optimal EPT,
    derives the grid dimension, and calls ``wp.launch``.
    """
    ept = compute_ept(N, max(device.sm_count, 1), is_vec)
    dim = (N + ept - 1) // ept
    wp.launch(overloads[key], dim=dim, inputs=[*args, N, ept], device=device)


# ---------------------------------------------------------------------------
# Kernels -- segmented_sum
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _total_sum_tile_kernel(
    x: wp.array(dtype=Any),
    out: wp.array(dtype=Any),
):
    """Block-cooperative total sum for single-segment case (M=1 specialization).

    Each block loads _TILE_SIZE elements, reduces via shared memory using
    tile operations, and emits one atomic add to out[0]. This provides
    optimal performance for the common case of reducing all elements to
    a single output value.

    Launch Grid
    -----------
    dim = [num_blocks], where num_blocks = N // TILE_SIZE
    block_dim = _BLOCK_DIM

    Parameters
    ----------
    x : wp.array, shape (N,), dtype float32/float64/vec3f/vec3d
        Input values to sum.
    out : wp.array, shape (1,), dtype matches x
        OUTPUT: Accumulated sum. Zeroed internally before each use.

    Notes
    -----
    - Uses warp tile operations for efficient block-level reduction
    - Only processes full blocks; remainder handled separately by caller
    - Requires N >= 8192 for efficiency (checked by caller)
    """
    i = wp.tid()
    t = wp.tile_load(x, shape=_TILE_SIZE, offset=i * _TILE_SIZE)
    s = wp.tile_sum(t)
    wp.tile_atomic_add(out, s, 0)


@wp.kernel(enable_backward=False)
def _segmented_sum_kernel(
    x: wp.array(dtype=Any),
    idx: wp.array(dtype=wp.int32),
    out: wp.array(dtype=Any),
    N: wp.int32,
    elems_per_thread: wp.int32,
):
    """Run-length segmented sum exploiting sorted segment indices.

    Computes ``out[s] = sum(x[i] for i where idx[i] == s)`` using a
    run-length encoding approach. Each thread processes a contiguous
    chunk of elements, accumulates within runs of identical segment IDs,
    and emits one atomic add per segment boundary.

    This algorithm exploits spatial locality when idx is sorted, minimizing
    atomic contention by accumulating locally before writing to global memory.

    Launch Grid
    -----------
    dim = ceil(N / elems_per_thread)

    Parameters
    ----------
    x : wp.array, shape (N,), dtype float32/float64/vec3f/vec3d
        Input values to sum per segment.
    idx : wp.array, shape (N,), dtype int32
        Sorted segment indices in [0, M). Must be sorted in non-decreasing order.
    out : wp.array, shape (M,), dtype matches x
        OUTPUT: Per-segment sums. Zeroed internally before each use.
    N : int32
        Total number of elements.
    elems_per_thread : int32
        Number of elements processed per thread (auto-tuned based on array size).

    Notes
    -----
    - Requires idx to be sorted for correctness
    - Each thread may span multiple segments; atomic adds occur at boundaries
    - Elements-per-thread is auto-tuned based on total count and SM count
    """
    t = wp.tid()
    start = t * elems_per_thread
    if start >= N:
        return
    end = wp.min(start + elems_per_thread, N)

    s_cur = idx[start]
    acc = x[start]
    for i in range(start + 1, end):
        s = idx[i]
        if s == s_cur:
            acc = acc + x[i]
        else:
            wp.atomic_add(out, s_cur, acc)
            s_cur = s
            acc = x[i]
    wp.atomic_add(out, s_cur, acc)


# ---------------------------------------------------------------------------
# Kernels -- segmented_component_sum
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _segmented_component_sum_kernel(
    x: wp.array(dtype=Any),
    idx: wp.array(dtype=wp.int32),
    out: wp.array(dtype=Any),
    N: wp.int32,
    elems_per_thread: wp.int32,
):
    """Sum vec3 components to scalar per segment using run-length encoding.

    Computes ``out[s] = sum(x[i][0] + x[i][1] + x[i][2] for i where idx[i] == s)``.
    This is equivalent to summing all vector components within each segment.

    Launch Grid
    -----------
    dim = ceil(N / elems_per_thread)

    Parameters
    ----------
    x : wp.array, shape (N,), dtype vec3f/vec3d
        Input vec3 values.
    idx : wp.array, shape (N,), dtype int32
        Sorted segment indices in [0, M).
    out : wp.array, shape (M,), dtype float32/float64
        OUTPUT: Scalar sums of all components per segment. Zeroed internally before each use.
    N : int32
        Total number of elements.
    elems_per_thread : int32
        Number of elements processed per thread.

    Notes
    -----
    - Output dtype is scalar (float32 for vec3f, float64 for vec3d)
    - Useful for computing total kinetic energy or similar scalar reductions
    """
    t = wp.tid()
    start = t * elems_per_thread
    if start >= N:
        return
    end = wp.min(start + elems_per_thread, N)

    s_cur = idx[start]
    _x = x[start]
    acc = _x[0] + _x[1] + _x[2]
    for i in range(start + 1, end):
        s = idx[i]
        _x = x[i]
        val = _x[0] + _x[1] + _x[2]
        if s == s_cur:
            acc = acc + val
        else:
            wp.atomic_add(out, s_cur, acc)
            s_cur = s
            acc = val
    wp.atomic_add(out, s_cur, acc)


@wp.func
def _component_sum(v: wp.vec3f) -> wp.float32:
    return v[0] + v[1] + v[2]


@wp.func
def _component_sum(v: wp.vec3d) -> wp.float64:
    return v[0] + v[1] + v[2]


@wp.kernel(enable_backward=False)
def _total_component_sum_tile_kernel(
    x: wp.array(dtype=Any),
    out: wp.array(dtype=Any),
):
    """Block-cooperative total component sum for M=1 specialization.

    Computes ``out[0] += sum(x[i][0] + x[i][1] + x[i][2])`` for one block
    of _TILE_SIZE elements using tile_map and tile operations.

    Launch Grid
    -----------
    dim = [num_blocks], block_dim = _BLOCK_DIM

    Parameters
    ----------
    x : wp.array, shape (N,), dtype vec3f/vec3d
        Input vector array.
    out : wp.array, shape (1,), dtype float32/float64
        Accumulated component sum. Zeroed internally before each use.
    """
    i = wp.tid()
    t = wp.tile_load(x, shape=_TILE_SIZE, offset=i * _TILE_SIZE)
    comps = wp.tile_map(_component_sum, t)
    s = wp.tile_sum(comps)
    wp.tile_atomic_add(out, s, 0)


# ---------------------------------------------------------------------------
# Kernels -- segmented_dot
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _segmented_dot_scalar_kernel(
    x: wp.array(dtype=Any),
    y: wp.array(dtype=Any),
    idx: wp.array(dtype=wp.int32),
    out: wp.array(dtype=Any),
    N: wp.int32,
    elems_per_thread: wp.int32,
):
    """Scalar element-wise product reduction per segment.

    Computes ``out[s] = sum(x[i] * y[i] for i where idx[i] == s)``.

    Launch Grid
    -----------
    dim = ceil(N / elems_per_thread)

    Parameters
    ----------
    x : wp.array, shape (N,), dtype float32/float64
        First input array.
    y : wp.array, shape (N,), dtype matches x
        Second input array.
    idx : wp.array, shape (N,), dtype int32
        Sorted segment indices in [0, M).
    out : wp.array, shape (M,), dtype matches x
        OUTPUT: Dot product per segment. Zeroed internally before each use.
    N : int32
        Total number of elements.
    elems_per_thread : int32
        Number of elements processed per thread.

    Notes
    -----
    - For scalar arrays, computes element-wise product sum
    - See _segmented_dot_vec_kernel for vector dot products
    """
    t = wp.tid()
    start = t * elems_per_thread
    if start >= N:
        return
    end = wp.min(start + elems_per_thread, N)

    s_cur = idx[start]
    acc = x[start] * y[start]
    for i in range(start + 1, end):
        s = idx[i]
        val = x[i] * y[i]
        if s == s_cur:
            acc = acc + val
        else:
            wp.atomic_add(out, s_cur, acc)
            s_cur = s
            acc = val
    wp.atomic_add(out, s_cur, acc)


@wp.kernel(enable_backward=False)
def _segmented_dot_vec_kernel(
    x: wp.array(dtype=Any),
    y: wp.array(dtype=Any),
    idx: wp.array(dtype=wp.int32),
    out: wp.array(dtype=Any),
    N: wp.int32,
    elems_per_thread: wp.int32,
):
    """Vec3 dot-product reduction per segment.

    Computes ``out[s] = sum(dot(x[i], y[i]) for i where idx[i] == s)``
    using vector dot products.

    Launch Grid
    -----------
    dim = ceil(N / elems_per_thread)

    Parameters
    ----------
    x : wp.array, shape (N,), dtype vec3f/vec3d
        First input vector array.
    y : wp.array, shape (N,), dtype matches x
        Second input vector array.
    idx : wp.array, shape (N,), dtype int32
        Sorted segment indices in [0, M).
    out : wp.array, shape (M,), dtype float32/float64
        OUTPUT: Scalar dot product sum per segment. Zeroed internally before each use.
    N : int32
        Total number of elements.
    elems_per_thread : int32
        Number of elements processed per thread.

    Notes
    -----
    - Output is scalar (float32 for vec3f input, float64 for vec3d input)
    - Useful for computing v·f power in FIRE optimizer
    """
    t = wp.tid()
    start = t * elems_per_thread
    if start >= N:
        return
    end = wp.min(start + elems_per_thread, N)

    s_cur = idx[start]
    acc = wp.dot(x[start], y[start])
    for i in range(start + 1, end):
        s = idx[i]
        val = wp.dot(x[i], y[i])
        if s == s_cur:
            acc = acc + val
        else:
            wp.atomic_add(out, s_cur, acc)
            s_cur = s
            acc = val
    wp.atomic_add(out, s_cur, acc)


@wp.kernel(enable_backward=False)
def _total_dot_scalar_tile_kernel(
    x: wp.array(dtype=Any),
    y: wp.array(dtype=Any),
    out: wp.array(dtype=Any),
):
    """Block-cooperative total scalar dot product for M=1 specialization.

    Computes ``out[0] += sum(x[i] * y[i])`` for one block of _TILE_SIZE
    elements using tile operations.

    Launch Grid
    -----------
    dim = [num_blocks], block_dim = _BLOCK_DIM

    Parameters
    ----------
    x, y : wp.array, shape (N,), dtype float32/float64
        Input scalar arrays.
    out : wp.array, shape (1,), dtype matches x
        Accumulated dot product. Zeroed internally before each use.
    """
    i = wp.tid()
    tx = wp.tile_load(x, shape=_TILE_SIZE, offset=i * _TILE_SIZE)
    ty = wp.tile_load(y, shape=_TILE_SIZE, offset=i * _TILE_SIZE)
    s = wp.tile_sum(wp.tile_map(wp.mul, tx, ty))
    wp.tile_atomic_add(out, s, 0)


@wp.kernel(enable_backward=False)
def _total_dot_vec_tile_kernel(
    x: wp.array(dtype=Any),
    y: wp.array(dtype=Any),
    out: wp.array(dtype=Any),
):
    """Block-cooperative total vector dot product for M=1 specialization.

    Computes ``out[0] += sum(dot(x[i], y[i]))`` for one block of _TILE_SIZE
    elements using tile_map(wp.dot, ...) and tile operations.

    Launch Grid
    -----------
    dim = [num_blocks], block_dim = _BLOCK_DIM

    Parameters
    ----------
    x, y : wp.array, shape (N,), dtype vec3f/vec3d
        Input vector arrays.
    out : wp.array, shape (1,), dtype float32/float64
        Accumulated dot product sum. Zeroed internally before each use.
    """
    i = wp.tid()
    tx = wp.tile_load(x, shape=_TILE_SIZE, offset=i * _TILE_SIZE)
    ty = wp.tile_load(y, shape=_TILE_SIZE, offset=i * _TILE_SIZE)
    dots = wp.tile_map(wp.dot, tx, ty)
    s = wp.tile_sum(dots)
    wp.tile_atomic_add(out, s, 0)


# ---------------------------------------------------------------------------
# Kernels -- segmented_max_norm
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _segmented_max_norm_kernel(
    x: wp.array(dtype=Any),
    idx: wp.array(dtype=wp.int32),
    out: wp.array(dtype=Any),
    N: wp.int32,
    elems_per_thread: wp.int32,
):
    """Maximum vector norm per segment using run-length encoding.

    Computes ``out[s] = max(length(x[i]) for i where idx[i] == s)``.

    Launch Grid
    -----------
    dim = ceil(N / elems_per_thread)

    Parameters
    ----------
    x : wp.array, shape (N,), dtype vec3f/vec3d
        Input vector array.
    idx : wp.array, shape (N,), dtype int32
        Sorted segment indices in [0, M).
    out : wp.array, shape (M,), dtype float32/float64
        OUTPUT: Maximum vector norm per segment. Zeroed internally before each use.
    N : int32
        Total number of elements.
    elems_per_thread : int32
        Number of elements processed per thread.

    Notes
    -----
    - Useful for convergence checks in geometry optimization (e.g., max force magnitude)
    - Output is scalar (float32 for vec3f input, float64 for vec3d input)
    """
    t = wp.tid()
    start = t * elems_per_thread
    if start >= N:
        return
    end = wp.min(start + elems_per_thread, N)

    s_cur = idx[start]
    max_val = wp.length(x[start])
    for i in range(start + 1, end):
        s = idx[i]
        val = wp.length(x[i])
        if s == s_cur:
            max_val = wp.max(max_val, val)
        else:
            wp.atomic_max(out, s_cur, max_val)
            s_cur = s
            max_val = val
    wp.atomic_max(out, s_cur, max_val)


@wp.kernel(enable_backward=False)
def _total_max_norm_kernel(
    x: wp.array(dtype=Any),
    out: wp.array(dtype=Any),
    N: wp.int32,
    elems_per_thread: wp.int32,
):
    """Total maximum norm reduction for single-segment case (M=1 specialization).

    Computes ``out[0] = max(length(x[i]) for all i)``. Each thread processes
    a chunk of elements and emits one atomic_max to the global output.
    No segment-boundary logic is needed since there is only one segment.

    Launch Grid
    -----------
    dim = ceil(N / elems_per_thread)

    Parameters
    ----------
    x : wp.array, shape (N,), dtype vec3f/vec3d
        Input vector array.
    out : wp.array, shape (1,), dtype float32/float64
        OUTPUT: Maximum norm across all elements. Zeroed internally before each use.
    N : int32
        Total number of elements.
    elems_per_thread : int32
        Number of elements processed per thread (large EPT reduces atomic contention).

    Notes
    -----
    - Uses large elements-per-thread to minimize atomic_max contention
    - Requires N >= 8192 for efficiency (checked by caller)
    """
    t = wp.tid()
    start = t * elems_per_thread
    if start >= N:
        return
    end = wp.min(start + elems_per_thread, N)
    max_val = wp.length(x[start])
    for i in range(start + 1, end):
        max_val = wp.max(max_val, wp.length(x[i]))
    wp.atomic_max(out, 0, max_val)


# ---------------------------------------------------------------------------
# Kernels -- segmented_axpy (in-place broadcast FMA)
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _segmented_axpy_kernel(
    y: wp.array(dtype=Any),
    x: wp.array(dtype=Any),
    a: wp.array(dtype=Any),
    idx: wp.array(dtype=wp.int32),
):
    """In-place broadcast fused multiply-add (AXPY): y[i] += x[i] * a[idx[i]].

    Performs a segment-indexed AXPY operation where each element is scaled
    by a per-segment coefficient before adding to the accumulator.

    Launch Grid
    -----------
    dim = N (total elements)

    Parameters
    ----------
    y : wp.array, shape (N,), dtype vec3f/vec3d/float32/float64
        Accumulator array, modified in-place.
    x : wp.array, shape (N,), dtype matches y
        Input array to scale and add.
    a : wp.array, shape (M,), dtype float32/float64
        Per-segment scalar coefficients.
    idx : wp.array, shape (N,), dtype int32
        Segment indices in [0, M) (need not be sorted for this kernel).

    Notes
    -----
    - Modifies y in-place
    - Supports both scalar and vector types
    - For vectors, scalar coefficient a[s] is broadcast to all components
    """
    tid = wp.tid()
    y[tid] = y[tid] + x[tid] * a[idx[tid]]


# ---------------------------------------------------------------------------
# Kernels -- segmented_inner_products (triple reduction)
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _segmented_inner_products_scalar_kernel(
    x: wp.array(dtype=Any),
    y: wp.array(dtype=Any),
    idx: wp.array(dtype=wp.int32),
    out_xy: wp.array(dtype=Any),
    out_xx: wp.array(dtype=Any),
    out_yy: wp.array(dtype=Any),
    N: wp.int32,
    elems_per_thread: wp.int32,
):
    """Triple inner-product reduction per segment for scalar arrays.

    Computes three dot products in a single pass:
    - ``out_xy[s] = sum(x[i] * y[i] for i where idx[i] == s)``
    - ``out_xx[s] = sum(x[i] * x[i] for i where idx[i] == s)``
    - ``out_yy[s] = sum(y[i] * y[i] for i where idx[i] == s)``

    This fused operation is more efficient than three separate reductions.

    Launch Grid
    -----------
    dim = ceil(N / elems_per_thread)

    Parameters
    ----------
    x : wp.array, shape (N,), dtype float32/float64
        First input array.
    y : wp.array, shape (N,), dtype matches x
        Second input array.
    idx : wp.array, shape (N,), dtype int32
        Sorted segment indices in [0, M).
    out_xy : wp.array, shape (M,), dtype matches x
        OUTPUT: x·y per segment. Zeroed internally before each use.
    out_xx : wp.array, shape (M,), dtype matches x
        OUTPUT: x·x per segment. Zeroed internally before each use.
    out_yy : wp.array, shape (M,), dtype matches x
        OUTPUT: y·y per segment. Zeroed internally before each use.
    N : int32
        Total number of elements.
    elems_per_thread : int32
        Number of elements processed per thread.

    Notes
    -----
    - Useful for FIRE2 optimizer which needs v·f, v·v, and f·f simultaneously
    - Reduces memory traffic compared to three separate kernel launches
    """
    t = wp.tid()
    start = t * elems_per_thread
    if start >= N:
        return
    end = wp.min(start + elems_per_thread, N)

    s_cur = idx[start]
    xi = x[start]
    yi = y[start]
    acc_xy = xi * yi
    acc_xx = xi * xi
    acc_yy = yi * yi
    for i in range(start + 1, end):
        s = idx[i]
        xi = x[i]
        yi = y[i]
        if s == s_cur:
            acc_xy = acc_xy + xi * yi
            acc_xx = acc_xx + xi * xi
            acc_yy = acc_yy + yi * yi
        else:
            wp.atomic_add(out_xy, s_cur, acc_xy)
            wp.atomic_add(out_xx, s_cur, acc_xx)
            wp.atomic_add(out_yy, s_cur, acc_yy)
            s_cur = s
            acc_xy = xi * yi
            acc_xx = xi * xi
            acc_yy = yi * yi
    wp.atomic_add(out_xy, s_cur, acc_xy)
    wp.atomic_add(out_xx, s_cur, acc_xx)
    wp.atomic_add(out_yy, s_cur, acc_yy)


@wp.kernel(enable_backward=False)
def _segmented_inner_products_vec_kernel(
    x: wp.array(dtype=Any),
    y: wp.array(dtype=Any),
    idx: wp.array(dtype=wp.int32),
    out_xy: wp.array(dtype=Any),
    out_xx: wp.array(dtype=Any),
    out_yy: wp.array(dtype=Any),
    N: wp.int32,
    elems_per_thread: wp.int32,
):
    """Triple inner-product reduction per segment for vector arrays.

    Computes three dot products in a single pass:
    - ``out_xy[s] = sum(dot(x[i], y[i]) for i where idx[i] == s)``
    - ``out_xx[s] = sum(dot(x[i], x[i]) for i where idx[i] == s)``
    - ``out_yy[s] = sum(dot(y[i], y[i]) for i where idx[i] == s)``

    Launch Grid
    -----------
    dim = ceil(N / elems_per_thread)

    Parameters
    ----------
    x : wp.array, shape (N,), dtype vec3f/vec3d
        First input vector array.
    y : wp.array, shape (N,), dtype matches x
        Second input vector array.
    idx : wp.array, shape (N,), dtype int32
        Sorted segment indices in [0, M).
    out_xy : wp.array, shape (M,), dtype float32/float64
        OUTPUT: x·y per segment. Zeroed internally before each use.
    out_xx : wp.array, shape (M,), dtype float32/float64
        OUTPUT: x·x per segment. Zeroed internally before each use.
    out_yy : wp.array, shape (M,), dtype float32/float64
        OUTPUT: y·y per segment. Zeroed internally before each use.
    N : int32
        Total number of elements.
    elems_per_thread : int32
        Number of elements processed per thread.

    Notes
    -----
    - Output is scalar (float32 for vec3f input, float64 for vec3d input)
    - Useful for FIRE2 optimizer which needs velocity·force, velocity·velocity,
      and force·force simultaneously for adaptive parameter updates
    """
    t = wp.tid()
    start = t * elems_per_thread
    if start >= N:
        return
    end = wp.min(start + elems_per_thread, N)

    s_cur = idx[start]
    acc_xy = wp.dot(x[start], y[start])
    acc_xx = wp.dot(x[start], x[start])
    acc_yy = wp.dot(y[start], y[start])
    for i in range(start + 1, end):
        s = idx[i]
        if s == s_cur:
            acc_xy = acc_xy + wp.dot(x[i], y[i])
            acc_xx = acc_xx + wp.dot(x[i], x[i])
            acc_yy = acc_yy + wp.dot(y[i], y[i])
        else:
            wp.atomic_add(out_xy, s_cur, acc_xy)
            wp.atomic_add(out_xx, s_cur, acc_xx)
            wp.atomic_add(out_yy, s_cur, acc_yy)
            s_cur = s
            acc_xy = wp.dot(x[i], y[i])
            acc_xx = wp.dot(x[i], x[i])
            acc_yy = wp.dot(y[i], y[i])
    wp.atomic_add(out_xy, s_cur, acc_xy)
    wp.atomic_add(out_xx, s_cur, acc_xx)
    wp.atomic_add(out_yy, s_cur, acc_yy)


@wp.kernel(enable_backward=False)
def _total_inner_products_scalar_tile_kernel(
    x: wp.array(dtype=Any),
    y: wp.array(dtype=Any),
    out_xy: wp.array(dtype=Any),
    out_xx: wp.array(dtype=Any),
    out_yy: wp.array(dtype=Any),
):
    """Block-cooperative total scalar triple dot for M=1 specialization.

    Computes x*y, x*x, y*y sums for one block of _TILE_SIZE elements
    using tile operations.

    Launch Grid
    -----------
    dim = [num_blocks], block_dim = _BLOCK_DIM

    Parameters
    ----------
    x, y : wp.array, shape (N,), dtype float32/float64
        Input scalar arrays.
    out_xy, out_xx, out_yy : wp.array, shape (1,), dtype matches x
        Accumulated triple dot products. Zeroed internally before each use.
    """
    i = wp.tid()
    tx = wp.tile_load(x, shape=_TILE_SIZE, offset=i * _TILE_SIZE)
    ty = wp.tile_load(y, shape=_TILE_SIZE, offset=i * _TILE_SIZE)
    wp.tile_atomic_add(out_xy, wp.tile_sum(wp.tile_map(wp.mul, tx, ty)), 0)
    wp.tile_atomic_add(out_xx, wp.tile_sum(wp.tile_map(wp.mul, tx, tx)), 0)
    wp.tile_atomic_add(out_yy, wp.tile_sum(wp.tile_map(wp.mul, ty, ty)), 0)


@wp.kernel(enable_backward=False)
def _total_inner_products_vec_tile_kernel(
    x: wp.array(dtype=Any),
    y: wp.array(dtype=Any),
    out_xy: wp.array(dtype=Any),
    out_xx: wp.array(dtype=Any),
    out_yy: wp.array(dtype=Any),
):
    """Block-cooperative total vector triple dot for M=1 specialization.

    Computes x*y, x*x, y*y dot-product sums for one block of _TILE_SIZE
    elements using tile_map(wp.dot, ...) and tile operations.

    Launch Grid
    -----------
    dim = [num_blocks], block_dim = _BLOCK_DIM

    Parameters
    ----------
    x, y : wp.array, shape (N,), dtype vec3f/vec3d
        Input vector arrays.
    out_xy, out_xx, out_yy : wp.array, shape (1,), dtype float32/float64
        Accumulated triple dot products. Zeroed internally before each use.
    """
    i = wp.tid()
    tx = wp.tile_load(x, shape=_TILE_SIZE, offset=i * _TILE_SIZE)
    ty = wp.tile_load(y, shape=_TILE_SIZE, offset=i * _TILE_SIZE)
    wp.tile_atomic_add(out_xy, wp.tile_sum(wp.tile_map(wp.dot, tx, ty)), 0)
    wp.tile_atomic_add(out_xx, wp.tile_sum(wp.tile_map(wp.dot, tx, tx)), 0)
    wp.tile_atomic_add(out_yy, wp.tile_sum(wp.tile_map(wp.dot, ty, ty)), 0)


# ---------------------------------------------------------------------------
# Kernels -- segmented_axpby (broadcast linear combination)
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _segmented_axpby_kernel(
    out: wp.array(dtype=Any),
    a: wp.array(dtype=Any),
    x: wp.array(dtype=Any),
    b: wp.array(dtype=Any),
    y: wp.array(dtype=Any),
    idx: wp.array(dtype=wp.int32),
):
    """Broadcast linear combination: out[i] = a[idx[i]] * x[i] + b[idx[i]] * y[i].

    Computes a segment-indexed linear combination where each element is
    scaled by per-segment coefficients before summing.

    Launch Grid
    -----------
    dim = N (total elements)

    Parameters
    ----------
    out : wp.array, shape (N,), dtype vec3f/vec3d/float32/float64
        OUTPUT: Result of linear combination.
    a : wp.array, shape (M,), dtype float32/float64
        Per-segment scalar coefficients for x.
    x : wp.array, shape (N,), dtype matches out
        First input array.
    b : wp.array, shape (M,), dtype float32/float64
        Per-segment scalar coefficients for y.
    y : wp.array, shape (N,), dtype matches out
        Second input array.
    idx : wp.array, shape (N,), dtype int32
        Segment indices in [0, M) (need not be sorted for this kernel).

    Notes
    -----
    - Supports both scalar and vector types
    - For vectors, scalar coefficients are broadcast to all components
    - Common in iterative solvers and optimization algorithms
    """
    tid = wp.tid()
    s = idx[tid]
    out[tid] = a[s] * x[tid] + b[s] * y[tid]


# ---------------------------------------------------------------------------
# Kernels -- segmented_mul (broadcast)
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _segmented_mul_kernel(
    x: wp.array(dtype=Any),
    y: wp.array(dtype=Any),
    idx: wp.array(dtype=wp.int32),
    out: wp.array(dtype=Any),
):
    """Broadcast multiply: out[i] = x[i] * y[idx[i]].

    Multiplies each element by a per-segment value.

    Launch Grid
    -----------
    dim = N (total elements)

    Parameters
    ----------
    x : wp.array, shape (N,), dtype vec3f/vec3d/float32/float64
        Per-element input array.
    y : wp.array, shape (M,), dtype float32/float64 or matches x
        Per-segment broadcast values.
    idx : wp.array, shape (N,), dtype int32
        Segment indices in [0, M) (need not be sorted for this kernel).
    out : wp.array, shape (N,), dtype matches x
        OUTPUT: Element-wise product.

    Notes
    -----
    - Supports scalar-scalar and vector-scalar multiplication
    - For vector-scalar, scalar is broadcast to all components
    """
    tid = wp.tid()
    out[tid] = x[tid] * y[idx[tid]]


# ---------------------------------------------------------------------------
# Kernels -- segmented_add (broadcast)
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _segmented_add_kernel(
    x: wp.array(dtype=Any),
    y: wp.array(dtype=Any),
    idx: wp.array(dtype=wp.int32),
    out: wp.array(dtype=Any),
):
    """Broadcast add: out[i] = x[i] + y[idx[i]] (same-type variant).

    Adds a per-segment value to each element.

    Launch Grid
    -----------
    dim = N (total elements)

    Parameters
    ----------
    x : wp.array, shape (N,), dtype vec3f/vec3d/float32/float64
        Per-element input array.
    y : wp.array, shape (M,), dtype matches x
        Per-segment broadcast values.
    idx : wp.array, shape (N,), dtype int32
        Segment indices in [0, M) (need not be sorted for this kernel).
    out : wp.array, shape (N,), dtype matches x
        OUTPUT: Element-wise sum.

    Notes
    -----
    - This variant requires x and y to have the same dtype.
    - See _segmented_add_vec_scalar_kernel / _segmented_add_scalar_vec_kernel
      for mixed-type variants (vec + scalar or scalar + vec).
    """
    tid = wp.tid()
    out[tid] = x[tid] + y[idx[tid]]


@wp.kernel(enable_backward=False)
def _segmented_add_vec_scalar_kernel(
    x: wp.array(dtype=Any),
    y: wp.array(dtype=Any),
    idx: wp.array(dtype=wp.int32),
    out: wp.array(dtype=Any),
):
    """Broadcast add: out[i] = x[i] + y[idx[i]] (vec + scalar variant).

    Adds a per-segment scalar to each component of a vector.

    Launch Grid
    -----------
    dim = N (total elements)

    Parameters
    ----------
    x : wp.array, shape (N,), dtype vec3h/vec3f/vec3d
        Per-element input vectors.
    y : wp.array, shape (M,), dtype float16/float32/float64
        Per-segment scalar values to broadcast.
    idx : wp.array, shape (N,), dtype int32
        Segment indices in [0, M).
    out : wp.array, shape (N,), dtype matches x
        Output: out[i] = [x[i][0] + y[s], x[i][1] + y[s], x[i][2] + y[s]].
    """
    tid = wp.tid()
    _x = x[tid]
    _y = y[idx[tid]]
    out[tid] = type(_x)(_x[0] + _y, _x[1] + _y, _x[2] + _y)


@wp.kernel(enable_backward=False)
def _segmented_add_scalar_vec_kernel(
    x: wp.array(dtype=Any),
    y: wp.array(dtype=Any),
    idx: wp.array(dtype=wp.int32),
    out: wp.array(dtype=Any),
):
    """Broadcast add: out[i] = x[i] + y[idx[i]] (scalar + vec variant).

    Adds per-element scalars to per-segment vectors component-wise.

    Launch Grid
    -----------
    dim = N (total elements)

    Parameters
    ----------
    x : wp.array, shape (N,), dtype float16/float32/float64
        Per-element scalar values.
    y : wp.array, shape (M,), dtype vec3h/vec3f/vec3d
        Per-segment vectors to broadcast.
    idx : wp.array, shape (N,), dtype int32
        Segment indices in [0, M).
    out : wp.array, shape (N,), dtype matches y
        Output: out[i] = [x[i] + y[s][0], x[i] + y[s][1], x[i] + y[s][2]].
    """
    tid = wp.tid()
    _x = x[tid]
    _y = y[idx[tid]]
    out[tid] = type(_y)(_x + _y[0], _x + _y[1], _x + _y[2])


# ---------------------------------------------------------------------------
# Kernels -- segmented_max (scalar)
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _segmented_max_kernel(
    x: wp.array(dtype=Any),
    idx: wp.array(dtype=wp.int32),
    out: wp.array(dtype=Any),
    N: wp.int32,
    elems_per_thread: wp.int32,
):
    """Maximum scalar value per segment using run-length encoding.

    Computes ``out[s] = max(x[i] for i where idx[i] == s)``.

    Launch Grid
    -----------
    dim = ceil(N / elems_per_thread)

    Parameters
    ----------
    x : wp.array, shape (N,), dtype float32/float64
        Input scalar values.
    idx : wp.array, shape (N,), dtype int32
        Sorted segment indices in [0, M).
    out : wp.array, shape (M,), dtype matches x
        OUTPUT: Maximum value per segment. Must be initialized to -inf by caller.
    N : int32
        Total number of elements.
    elems_per_thread : int32
        Number of elements processed per thread.
    """
    t = wp.tid()
    start = t * elems_per_thread
    if start >= N:
        return
    end = wp.min(start + elems_per_thread, N)

    s_cur = idx[start]
    max_val = x[start]
    for i in range(start + 1, end):
        s = idx[i]
        if s == s_cur:
            max_val = wp.max(max_val, x[i])
        else:
            wp.atomic_max(out, s_cur, max_val)
            s_cur = s
            max_val = x[i]
    wp.atomic_max(out, s_cur, max_val)


# ---------------------------------------------------------------------------
# Kernels -- segmented_min (scalar)
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _segmented_min_kernel(
    x: wp.array(dtype=Any),
    idx: wp.array(dtype=wp.int32),
    out: wp.array(dtype=Any),
    N: wp.int32,
    elems_per_thread: wp.int32,
):
    """Minimum scalar value per segment using run-length encoding.

    Computes ``out[s] = min(x[i] for i where idx[i] == s)``.

    Launch Grid
    -----------
    dim = ceil(N / elems_per_thread)

    Parameters
    ----------
    x : wp.array, shape (N,), dtype float32/float64
        Input scalar values.
    idx : wp.array, shape (N,), dtype int32
        Sorted segment indices in [0, M).
    out : wp.array, shape (M,), dtype matches x
        OUTPUT: Minimum value per segment. Must be initialized to +inf by caller.
    N : int32
        Total number of elements.
    elems_per_thread : int32
        Number of elements processed per thread.
    """
    t = wp.tid()
    start = t * elems_per_thread
    if start >= N:
        return
    end = wp.min(start + elems_per_thread, N)

    s_cur = idx[start]
    min_val = x[start]
    for i in range(start + 1, end):
        s = idx[i]
        if s == s_cur:
            min_val = wp.min(min_val, x[i])
        else:
            wp.atomic_min(out, s_cur, min_val)
            s_cur = s
            min_val = x[i]
    wp.atomic_min(out, s_cur, min_val)


# ---------------------------------------------------------------------------
# Kernels -- segmented_broadcast (pure gather)
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _segmented_broadcast_kernel(
    values: wp.array(dtype=Any),
    idx: wp.array(dtype=wp.int32),
    out: wp.array(dtype=Any),
):
    """Pure gather broadcast: out[i] = values[idx[i]].

    Launch Grid
    -----------
    dim = N (total elements)

    Parameters
    ----------
    values : wp.array, shape (M,), dtype any supported type
        Per-segment values to broadcast.
    idx : wp.array, shape (N,), dtype int32
        Segment indices in [0, M). Need not be sorted.
    out : wp.array, shape (N,), dtype matches values
        OUTPUT: Broadcast values.
    """
    tid = wp.tid()
    out[tid] = values[idx[tid]]


# ---------------------------------------------------------------------------
# Kernels -- segment_div (element-wise divide with zero guard)
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _segment_div_kernel(
    numerator: wp.array(dtype=Any),
    denominator: wp.array(dtype=wp.int32),
    result: wp.array(dtype=Any),
):
    """Element-wise division with zero-denominator guard.

    Computes ``result[i] = numerator[i] / denominator[i]`` where zero
    denominators produce zero results.

    Launch Grid
    -----------
    dim = N

    Parameters
    ----------
    numerator : wp.array, shape (N,), dtype float16/float32/float64
        Numerator values.
    denominator : wp.array, shape (N,), dtype int32
        Denominator values (e.g., segment counts).
    result : wp.array, shape (N,), dtype matches numerator
        OUTPUT: Division results. Zero denominators yield 0.0.
    """
    i = wp.tid()
    if denominator[i] > 0:
        result[i] = numerator[i] / type(numerator[0])(denominator[i])
    else:
        result[i] = type(numerator[0])(0.0)


# ---------------------------------------------------------------------------
# Kernels -- _segmented_rms_norm_finalize (sqrt(sum_sq / count))
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _segmented_rms_norm_finalize_kernel(
    sum_sq: wp.array(dtype=Any),
    counts: wp.array(dtype=wp.int32),
    out: wp.array(dtype=Any),
):
    """Finalize RMS norm: out[s] = sqrt(sum_sq[s] / counts[s]).

    Launch Grid
    -----------
    dim = M (number of segments)

    Parameters
    ----------
    sum_sq : wp.array, shape (M,), dtype float32/float64
        Sum of squared norms per segment.
    counts : wp.array, shape (M,), dtype int32
        Number of elements per segment.
    out : wp.array, shape (M,), dtype matches sum_sq
        OUTPUT: RMS norm per segment. Zero counts yield 0.0.
    """
    i = wp.tid()
    if counts[i] > 0:
        out[i] = wp.sqrt(sum_sq[i] / type(sum_sq[0])(counts[i]))
    else:
        out[i] = type(sum_sq[0])(0.0)


# ---------------------------------------------------------------------------
# Kernels -- vector divide-by-count (for segmented_mean vec path)
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _segmented_vec_div_by_count_kernel(
    sums: wp.array(dtype=Any),
    counts: wp.array(dtype=wp.int32),
    out: wp.array(dtype=Any),
):
    """Divide vector sums by integer counts: out[i] = sums[i] / counts[i].

    Launch Grid
    -----------
    dim = M (number of segments)

    Parameters
    ----------
    sums : wp.array, shape (M,), dtype vec3f/vec3d
        Per-segment vector sums.
    counts : wp.array, shape (M,), dtype int32
        Per-segment element counts.
    out : wp.array, shape (M,), dtype matches sums
        Output mean vectors. Zero counts yield zero vector.
    """
    i = wp.tid()
    c = counts[i]
    if c > 0:
        out[i] = sums[i] / type(sums[0][0])(c)
    else:
        out[i] = type(sums[0])()


# ---------------------------------------------------------------------------
# Kernels -- segmented_count (count elements per segment)
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _segmented_count_kernel(
    idx: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.int32),
    N: wp.int32,
    elems_per_thread: wp.int32,
):
    """Count elements per segment using run-length encoding.

    Launch Grid
    -----------
    dim = ceil(N / elems_per_thread)

    Parameters
    ----------
    idx : wp.array, shape (N,), dtype int32
        Sorted segment indices in [0, M).
    out : wp.array, shape (M,), dtype int32
        OUTPUT: Element counts per segment. Zeroed internally before each use.
    N : int32
        Total number of elements.
    elems_per_thread : int32
        Number of elements processed per thread.
    """
    t = wp.tid()
    start = t * elems_per_thread
    if start >= N:
        return
    end = wp.min(start + elems_per_thread, N)

    s_cur = idx[start]
    count = wp.int32(1)
    for i in range(start + 1, end):
        s = idx[i]
        if s == s_cur:
            count = count + wp.int32(1)
        else:
            wp.atomic_add(out, s_cur, count)
            s_cur = s
            count = wp.int32(1)
    wp.atomic_add(out, s_cur, count)


# ---------------------------------------------------------------------------
# Overloads (module-level, keyed by dtype)
# ---------------------------------------------------------------------------

# -- Reduction overloads (atomic-safe types: float32/float64, vec3f/vec3d) ---

_total_sum_tile_overloads = register_overloads(
    _total_sum_tile_kernel,
    lambda t: [wp.array(dtype=t), wp.array(dtype=t)],
    dtypes=_SUPPORTED_TYPES,
)

_segmented_sum_overloads = register_overloads(
    _segmented_sum_kernel,
    lambda t: [
        wp.array(dtype=t),
        wp.array(dtype=wp.int32),
        wp.array(dtype=t),
        wp.int32,
        wp.int32,
    ],
    dtypes=_SUPPORTED_TYPES,
)

_segmented_component_sum_overloads = register_overloads(
    _segmented_component_sum_kernel,
    lambda v, s: [
        wp.array(dtype=v),
        wp.array(dtype=wp.int32),
        wp.array(dtype=s),
        wp.int32,
        wp.int32,
    ],
    dtype_pairs=_VEC_SCALAR_PAIRS,
)

_total_component_sum_tile_overloads = register_overloads(
    _total_component_sum_tile_kernel,
    lambda v, s: [wp.array(dtype=v), wp.array(dtype=s)],
    dtype_pairs=_VEC_SCALAR_PAIRS,
)

_total_dot_scalar_tile_overloads = register_overloads(
    _total_dot_scalar_tile_kernel,
    lambda t: [wp.array(dtype=t), wp.array(dtype=t), wp.array(dtype=t)],
    dtypes=_SCALAR_TYPES,
)

_total_dot_vec_tile_overloads = register_overloads(
    _total_dot_vec_tile_kernel,
    lambda v, s: [wp.array(dtype=v), wp.array(dtype=v), wp.array(dtype=s)],
    dtype_pairs=_VEC_SCALAR_PAIRS,
)

_segmented_dot_overloads = register_overloads(
    _segmented_dot_scalar_kernel,
    lambda t: [
        wp.array(dtype=t),
        wp.array(dtype=t),
        wp.array(dtype=wp.int32),
        wp.array(dtype=t),
        wp.int32,
        wp.int32,
    ],
    dtypes=_SCALAR_TYPES,
)
_segmented_dot_overloads.update(
    register_overloads(
        _segmented_dot_vec_kernel,
        lambda v, s: [
            wp.array(dtype=v),
            wp.array(dtype=v),
            wp.array(dtype=wp.int32),
            wp.array(dtype=s),
            wp.int32,
            wp.int32,
        ],
        dtype_pairs=_VEC_SCALAR_PAIRS,
    )
)

_segmented_max_norm_overloads = register_overloads(
    _segmented_max_norm_kernel,
    lambda v, s: [
        wp.array(dtype=v),
        wp.array(dtype=wp.int32),
        wp.array(dtype=s),
        wp.int32,
        wp.int32,
    ],
    dtype_pairs=_VEC_SCALAR_PAIRS,
)

_total_max_norm_overloads = register_overloads(
    _total_max_norm_kernel,
    lambda v, s: [wp.array(dtype=v), wp.array(dtype=s), wp.int32, wp.int32],
    dtype_pairs=_VEC_SCALAR_PAIRS,
)

_total_inner_products_scalar_tile_overloads = register_overloads(
    _total_inner_products_scalar_tile_kernel,
    lambda t: [wp.array(dtype=t)] * 2 + [wp.array(dtype=t)] * 3,
    dtypes=_SCALAR_TYPES,
)

_total_inner_products_vec_tile_overloads = register_overloads(
    _total_inner_products_vec_tile_kernel,
    lambda v, s: [wp.array(dtype=v)] * 2 + [wp.array(dtype=s)] * 3,
    dtype_pairs=_VEC_SCALAR_PAIRS,
)

_segmented_inner_products_overloads = register_overloads(
    _segmented_inner_products_scalar_kernel,
    lambda t: [wp.array(dtype=t)] * 2
    + [wp.array(dtype=wp.int32)]
    + [wp.array(dtype=t)] * 3
    + [wp.int32, wp.int32],
    dtypes=_SCALAR_TYPES,
)
_segmented_inner_products_overloads.update(
    register_overloads(
        _segmented_inner_products_vec_kernel,
        lambda v, s: [wp.array(dtype=v)] * 2
        + [wp.array(dtype=wp.int32)]
        + [wp.array(dtype=s)] * 3
        + [wp.int32, wp.int32],
        dtype_pairs=_VEC_SCALAR_PAIRS,
    )
)

_segmented_max_overloads = register_overloads(
    _segmented_max_kernel,
    lambda t: [
        wp.array(dtype=t),
        wp.array(dtype=wp.int32),
        wp.array(dtype=t),
        wp.int32,
        wp.int32,
    ],
    dtypes=_SCALAR_TYPES,
)

_segmented_min_overloads = register_overloads(
    _segmented_min_kernel,
    lambda t: [
        wp.array(dtype=t),
        wp.array(dtype=wp.int32),
        wp.array(dtype=t),
        wp.int32,
        wp.int32,
    ],
    dtypes=_SCALAR_TYPES,
)

_segmented_rms_norm_finalize_overloads = register_overloads(
    _segmented_rms_norm_finalize_kernel,
    lambda t: [wp.array(dtype=t), wp.array(dtype=wp.int32), wp.array(dtype=t)],
    dtypes=_SCALAR_TYPES,
)

_segmented_vec_div_by_count_overloads = register_overloads(
    _segmented_vec_div_by_count_kernel,
    lambda v, s: [wp.array(dtype=v), wp.array(dtype=wp.int32), wp.array(dtype=v)],
    dtype_pairs=_VEC_SCALAR_PAIRS,
)

# -- Element-wise / broadcast overloads (include float16/vec3h) -------------

_segmented_axpy_overloads = register_overloads(
    _segmented_axpy_kernel,
    lambda v, s: [
        wp.array(dtype=v),
        wp.array(dtype=v),
        wp.array(dtype=s),
        wp.array(dtype=wp.int32),
    ],
    dtype_pairs=_ALL_VEC_SCALAR_PAIRS,
)
_segmented_axpy_overloads.update(
    register_overloads(
        _segmented_axpy_kernel,
        lambda t: [wp.array(dtype=t)] * 3 + [wp.array(dtype=wp.int32)],
        dtypes=_ALL_SCALAR_TYPES,
    )
)

_segmented_axpby_overloads = register_overloads(
    _segmented_axpby_kernel,
    lambda v, s: [
        wp.array(dtype=v),
        wp.array(dtype=s),
        wp.array(dtype=v),
        wp.array(dtype=s),
        wp.array(dtype=v),
        wp.array(dtype=wp.int32),
    ],
    dtype_pairs=_ALL_VEC_SCALAR_PAIRS,
)
_segmented_axpby_overloads.update(
    register_overloads(
        _segmented_axpby_kernel,
        lambda t: [wp.array(dtype=t)] * 5 + [wp.array(dtype=wp.int32)],
        dtypes=_ALL_SCALAR_TYPES,
    )
)

_segmented_mul_overloads = register_overloads(
    _segmented_mul_kernel,
    lambda t: [wp.array(dtype=t)] * 2 + [wp.array(dtype=wp.int32), wp.array(dtype=t)],
    dtypes=_ALL_SCALAR_TYPES,
    key_fn=lambda t: (t, t),
)
_segmented_mul_overloads.update(
    register_overloads(
        _segmented_mul_kernel,
        lambda v, s: [
            wp.array(dtype=v),
            wp.array(dtype=s),
            wp.array(dtype=wp.int32),
            wp.array(dtype=v),
        ],
        dtype_pairs=_ALL_VEC_SCALAR_PAIRS,
        key_fn=lambda v, s: (v, s),
    )
)
_segmented_mul_overloads.update(
    register_overloads(
        _segmented_mul_kernel,
        lambda v, m: [
            wp.array(dtype=v),
            wp.array(dtype=m),
            wp.array(dtype=wp.int32),
            wp.array(dtype=v),
        ],
        dtype_pairs=_VEC_MAT_PAIRS,
        key_fn=lambda v, m: (v, m),
    )
)

_segmented_add_overloads = register_overloads(
    _segmented_add_kernel,
    lambda t: [wp.array(dtype=t)] * 2 + [wp.array(dtype=wp.int32), wp.array(dtype=t)],
    dtypes=_ALL_SUPPORTED_TYPES,
    key_fn=lambda t: (t, t),
)
_segmented_add_overloads.update(
    register_overloads(
        _segmented_add_vec_scalar_kernel,
        lambda v, s: [
            wp.array(dtype=v),
            wp.array(dtype=s),
            wp.array(dtype=wp.int32),
            wp.array(dtype=v),
        ],
        dtype_pairs=_ALL_VEC_SCALAR_PAIRS,
        key_fn=lambda v, s: (v, s),
    )
)
_segmented_add_overloads.update(
    register_overloads(
        _segmented_add_scalar_vec_kernel,
        lambda v, s: [
            wp.array(dtype=s),
            wp.array(dtype=v),
            wp.array(dtype=wp.int32),
            wp.array(dtype=v),
        ],
        dtype_pairs=_ALL_VEC_SCALAR_PAIRS,
        key_fn=lambda v, s: (s, v),
    )
)

_segmented_broadcast_overloads = register_overloads(
    _segmented_broadcast_kernel,
    lambda t: [wp.array(dtype=t), wp.array(dtype=wp.int32), wp.array(dtype=t)],
    dtypes=_ALL_SUPPORTED_TYPES,
)

_segment_div_overloads = register_overloads(
    _segment_div_kernel,
    lambda t: [wp.array(dtype=t), wp.array(dtype=wp.int32), wp.array(dtype=t)],
    dtypes=_ALL_SCALAR_TYPES,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def segmented_sum(
    x: wp.array,
    idx: wp.array,
    out: wp.array,
) -> None:
    """Compute per-segment sum using run-length encoded reduction.

    ``out[s] = sum(x[i] for i where idx[i] == s)``

    Requires sorted ``idx`` in non-decreasing order.

    Parameters
    ----------
    x : wp.array, shape (N,)
        Input values. Supported dtypes: float32, float64, vec3f, vec3d.
    idx : wp.array(dtype=int32), shape (N,)
        Sorted segment indices in ``[0, M)``.
    out : wp.array, shape (M,), dtype matches x
        Per-segment sums. Zeroed internally before each use.
    """
    N = x.shape[0]
    if N == 0:
        return

    out.zero_()
    device = x.device
    M = out.shape[0]

    # -- M=1 fast path: tile-based block reduction --------------------------
    if M == 1 and N >= 8192:
        full_blocks = N // _BLOCK_DIM
        wp.launch_tiled(
            _total_sum_tile_overloads[x.dtype],
            dim=full_blocks,
            inputs=[x, out],
            block_dim=_BLOCK_DIM,
            device=device,
        )
        remainder = N - full_blocks * _BLOCK_DIM
        if remainder > 0:
            x_tail = x[full_blocks * _BLOCK_DIM :]
            idx_tail = idx[full_blocks * _BLOCK_DIM :]
            wp.launch(
                _segmented_sum_overloads[x.dtype],
                dim=remainder,
                inputs=[x_tail, idx_tail, out, remainder, 1],
                device=device,
            )
        return

    # -- General path: run-length segmented sum -----------------------------
    ept = compute_ept(N, max(device.sm_count, 1), x.dtype in _VEC_TYPES)
    dim = (N + ept - 1) // ept
    wp.launch(
        _segmented_sum_overloads[x.dtype],
        dim=dim,
        inputs=[x, idx, out, N, ept],
        device=device,
    )


def segmented_component_sum(
    x: wp.array,
    idx: wp.array,
    out: wp.array,
) -> None:
    """Sum all vector components to scalar per segment using RLE reduction.

    ``out[s] = sum(x[i][0] + x[i][1] + x[i][2] for i where idx[i] == s)``

    Requires sorted ``idx`` in non-decreasing order.

    Parameters
    ----------
    x : wp.array, shape (N,), dtype vec3f / vec3d
        Input 3D vectors.
    idx : wp.array(dtype=int32), shape (N,)
        Sorted segment indices in ``[0, M)``.
    out : wp.array, shape (M,), dtype float32 / float64
        Per-segment scalar sums. Zeroed internally before each use.

    See Also
    --------
    segmented_sum : Element-wise sum (preserves vector dtype)
    segmented_dot : Dot product per segment
    """
    N = x.shape[0]
    if N == 0:
        return

    out.zero_()
    device = x.device
    M = out.shape[0]

    # -- M=1 fast path: tile-based block reduction --------------------------
    if M == 1 and N >= 8192:
        full_blocks = N // _BLOCK_DIM
        wp.launch_tiled(
            _total_component_sum_tile_overloads[x.dtype],
            dim=full_blocks,
            inputs=[x, out],
            block_dim=_BLOCK_DIM,
            device=device,
        )
        remainder = N - full_blocks * _BLOCK_DIM
        if remainder > 0:
            x_tail = x[full_blocks * _BLOCK_DIM :]
            idx_tail = idx[full_blocks * _BLOCK_DIM :]
            wp.launch(
                _segmented_component_sum_overloads[x.dtype],
                dim=remainder,
                inputs=[x_tail, idx_tail, out, remainder, 1],
                device=device,
            )
        return

    # -- General path: run-length segmented component sum -------------------
    ept = compute_ept(N, max(device.sm_count, 1), True)
    dim = (N + ept - 1) // ept
    wp.launch(
        _segmented_component_sum_overloads[x.dtype],
        dim=dim,
        inputs=[x, idx, out, N, ept],
        device=device,
    )


def segmented_dot(
    x: wp.array,
    y: wp.array,
    idx: wp.array,
    out: wp.array,
) -> None:
    """Compute per-segment dot product reduction using RLE.

    - Scalar types: ``out[s] = sum(x[i] * y[i] for i where idx[i] == s)``
    - Vector types: ``out[s] = sum(dot(x[i], y[i]) for i where idx[i] == s)``

    Requires sorted ``idx`` in non-decreasing order.

    Parameters
    ----------
    x, y : wp.array, shape (N,)
        Input arrays. Must have same dtype.
        Supported: float32, float64, vec3f, vec3d.
    idx : wp.array(dtype=int32), shape (N,)
        Sorted segment indices in ``[0, M)``.
    out : wp.array, shape (M,), dtype float32 / float64
        Per-segment dot products. Zeroed internally before each use.

    See Also
    --------
    segmented_inner_products : Compute x·y, x·x, and y·y in one pass
    segmented_sum : Element-wise sum per segment
    """
    N = x.shape[0]
    if N == 0:
        return

    out.zero_()
    device = x.device
    M = out.shape[0]

    # -- M=1 fast path: tile-based block reduction --------------------------
    if M == 1 and N >= 8192:
        tile_overloads = (
            _total_dot_scalar_tile_overloads
            if x.dtype in _SCALAR_TYPES
            else _total_dot_vec_tile_overloads
        )
        full_blocks = N // _BLOCK_DIM
        wp.launch_tiled(
            tile_overloads[x.dtype],
            dim=full_blocks,
            inputs=[x, y, out],
            block_dim=_BLOCK_DIM,
            device=device,
        )
        remainder = N - full_blocks * _BLOCK_DIM
        if remainder > 0:
            x_tail = x[full_blocks * _BLOCK_DIM :]
            y_tail = y[full_blocks * _BLOCK_DIM :]
            idx_tail = idx[full_blocks * _BLOCK_DIM :]
            wp.launch(
                _segmented_dot_overloads[x.dtype],
                dim=remainder,
                inputs=[x_tail, y_tail, idx_tail, out, remainder, 1],
                device=device,
            )
        return

    # -- General path: run-length segmented dot -----------------------------
    ept = compute_ept(N, max(device.sm_count, 1), x.dtype in _VEC_TYPES)
    dim = (N + ept - 1) // ept
    wp.launch(
        _segmented_dot_overloads[x.dtype],
        dim=dim,
        inputs=[x, y, idx, out, N, ept],
        device=device,
    )


def segmented_max_norm(
    x: wp.array,
    idx: wp.array,
    out: wp.array,
) -> None:
    """Compute maximum vector norm per segment using RLE reduction.

    ``out[s] = max(length(x[i]) for i where idx[i] == s)``

    Requires sorted ``idx`` in non-decreasing order.  For single-segment
    cases (M=1) with N >= 8192, an optimized fast path reduces atomic
    contention.

    Parameters
    ----------
    x : wp.array, shape (N,), dtype vec3f / vec3d
        Input 3D vectors.
    idx : wp.array(dtype=int32), shape (N,)
        Sorted segment indices in ``[0, M)``.
    out : wp.array, shape (M,), dtype float32 / float64
        Maximum norms per segment. Zeroed internally before each use.

    See Also
    --------
    segmented_dot : Squared norms (v·v) per segment
    """
    N = x.shape[0]
    if N == 0:
        return

    out.zero_()
    device = x.device
    M = out.shape[0]

    # -- M=1 fast path: large EPT to minimize atomic_max contention ----------
    if M == 1 and N >= 8192:
        sm = max(device.sm_count, 1)
        ept = max(64, N // (sm * 4))
        dim = (N + ept - 1) // ept
        wp.launch(
            _total_max_norm_overloads[x.dtype],
            dim=dim,
            inputs=[x, out, N, ept],
            device=device,
        )
        return

    # -- General path: run-length segmented max norm -------------------------
    ept = compute_ept(N, max(device.sm_count, 1), True)
    dim = (N + ept - 1) // ept
    wp.launch(
        _segmented_max_norm_overloads[x.dtype],
        dim=dim,
        inputs=[x, idx, out, N, ept],
        device=device,
    )


def segmented_axpy(
    y: wp.array,
    x: wp.array,
    a: wp.array,
    idx: wp.array,
) -> None:
    """In-place segmented broadcast FMA: ``y[i] += x[i] * a[idx[i]]``.

    Modifies ``y`` in-place.  ``idx`` need not be sorted.

    Parameters
    ----------
    y : wp.array, shape (N,)
        Accumulator, modified in-place.
        Supported dtypes: vec3f, vec3d, float32, float64
        (also vec3h, float16 for half-precision).
    x : wp.array, shape (N,), dtype matches y
        Input array to scale and add.
    a : wp.array, shape (M,)
        Per-segment scalar multipliers (precision matches y).
    idx : wp.array(dtype=int32), shape (N,)
        Segment indices in ``[0, M)``.

    See Also
    --------
    segmented_axpby : Generalized linear combination (a*x + b*y)
    segmented_mul : Broadcast multiply (out = x * y[idx])
    """
    N = y.shape[0]
    if N == 0:
        return
    wp.launch(
        _segmented_axpy_overloads[y.dtype],
        dim=N,
        inputs=[y, x, a, idx],
        device=y.device,
    )


def segmented_inner_products(
    x: wp.array,
    y: wp.array,
    idx: wp.array,
    out_xy: wp.array,
    out_xx: wp.array,
    out_yy: wp.array,
) -> None:
    """Compute three inner products per segment in one fused pass using RLE.

    Computes ``out_xy[s]``, ``out_xx[s]``, ``out_yy[s]`` (the x·y, x·x, y·y
    inner products) in a single kernel launch, roughly 3x faster than three
    separate ``segmented_dot`` calls.

    Requires sorted ``idx`` in non-decreasing order.

    Parameters
    ----------
    x, y : wp.array, shape (N,)
        Input arrays. Must have same dtype.
        Supported: float32, float64, vec3f, vec3d.
    idx : wp.array(dtype=int32), shape (N,)
        Sorted segment indices in ``[0, M)``.
    out_xy, out_xx, out_yy : wp.array, shape (M,), dtype float32 / float64
        Per-segment inner products. Zeroed internally before each use.

    See Also
    --------
    segmented_dot : Single dot product per segment
    """
    N = x.shape[0]
    if N == 0:
        return

    out_xy.zero_()
    out_xx.zero_()
    out_yy.zero_()
    device = x.device
    M = out_xy.shape[0]

    # -- M=1 fast path: tile-based block reduction --------------------------
    if M == 1 and N >= 8192:
        tile_overloads = (
            _total_inner_products_scalar_tile_overloads
            if x.dtype in _SCALAR_TYPES
            else _total_inner_products_vec_tile_overloads
        )
        full_blocks = N // _BLOCK_DIM
        wp.launch_tiled(
            tile_overloads[x.dtype],
            dim=full_blocks,
            inputs=[x, y, out_xy, out_xx, out_yy],
            block_dim=_BLOCK_DIM,
            device=device,
        )
        remainder = N - full_blocks * _BLOCK_DIM
        if remainder > 0:
            x_tail = x[full_blocks * _BLOCK_DIM :]
            y_tail = y[full_blocks * _BLOCK_DIM :]
            idx_tail = idx[full_blocks * _BLOCK_DIM :]
            wp.launch(
                _segmented_inner_products_overloads[x.dtype],
                dim=remainder,
                inputs=[x_tail, y_tail, idx_tail, out_xy, out_xx, out_yy, remainder, 1],
                device=device,
            )
        return

    # -- General path: run-length segmented inner products ------------------
    ept = compute_ept(N, max(device.sm_count, 1), x.dtype in _VEC_TYPES)
    dim = (N + ept - 1) // ept
    wp.launch(
        _segmented_inner_products_overloads[x.dtype],
        dim=dim,
        inputs=[x, y, idx, out_xy, out_xx, out_yy, N, ept],
        device=device,
    )


def segmented_axpby(
    out: wp.array,
    a: wp.array,
    x: wp.array,
    b: wp.array,
    y: wp.array,
    idx: wp.array,
) -> None:
    """Broadcast linear combination: ``out[i] = a[idx[i]] * x[i] + b[idx[i]] * y[i]``.

    ``idx`` need not be sorted.

    Parameters
    ----------
    out : wp.array, shape (N,)
        Output array.  Supported dtypes: vec3f, vec3d, float32, float64
        (also vec3h, float16).
    a : wp.array, shape (M,)
        Per-segment scalar multipliers for ``x``.
    x : wp.array, shape (N,), dtype matches out
        First input array.
    b : wp.array, shape (M,)
        Per-segment scalar multipliers for ``y``.
    y : wp.array, shape (N,), dtype matches out
        Second input array.
    idx : wp.array(dtype=int32), shape (N,)
        Segment indices in ``[0, M)``.

    See Also
    --------
    segmented_axpy : Simpler one-term version (y += a*x)
    segmented_mul : Broadcast multiply only
    segmented_add : Broadcast add only
    """
    N = out.shape[0]
    if N == 0:
        return
    wp.launch(
        _segmented_axpby_overloads[x.dtype],
        dim=N,
        inputs=[out, a, x, b, y, idx],
        device=out.device,
    )


def segmented_mul(
    x: wp.array,
    y: wp.array,
    idx: wp.array,
    out: wp.array,
) -> None:
    """Broadcast multiply: ``out[i] = x[i] * y[idx[i]]``.

    Supports mixed types (vec3 * scalar) as well as same-type (scalar * scalar).
    ``idx`` need not be sorted.

    Parameters
    ----------
    x : wp.array, shape (N,)
        Per-atom input values.
        Supported: vec3h, vec3f, vec3d, float16, float32, float64.
    y : wp.array, shape (M,)
        Per-segment scalar multipliers (precision matches x).
    idx : wp.array(dtype=int32), shape (N,)
        Segment indices in ``[0, M)``.
    out : wp.array, shape (N,), dtype matches x
        Scaled output.

    See Also
    --------
    segmented_add : Broadcast addition
    segmented_axpy : Fused multiply-add (y += a*x)
    """
    N = x.shape[0]
    if N == 0:
        return
    wp.launch(
        _segmented_mul_overloads[(x.dtype, y.dtype)],
        dim=N,
        inputs=[x, y, idx, out],
        device=x.device,
    )


def segmented_add(
    x: wp.array,
    y: wp.array,
    idx: wp.array,
    out: wp.array,
) -> None:
    """Broadcast add: ``out[i] = x[i] + y[idx[i]]``.

    Supports mixed types (vec3 + scalar, scalar + vec3) as well as same-type.
    ``idx`` need not be sorted.

    Parameters
    ----------
    x : wp.array, shape (N,)
        Per-atom input values.
        Supported: vec3h, vec3f, vec3d, float16, float32, float64.
    y : wp.array, shape (M,)
        Per-segment values to broadcast and add.
    idx : wp.array(dtype=int32), shape (N,)
        Segment indices in ``[0, M)``.
    out : wp.array, shape (N,)
        Output array.

    See Also
    --------
    segmented_mul : Broadcast multiplication
    segmented_axpy : Fused multiply-add
    """
    N = x.shape[0]
    if N == 0:
        return
    wp.launch(
        _segmented_add_overloads[(x.dtype, y.dtype)],
        dim=N,
        inputs=[x, y, idx, out],
        device=x.device,
    )


def segmented_matvec(
    v: wp.array,
    m: wp.array,
    idx: wp.array,
    out: wp.array,
) -> None:
    """Per-segment matrix-vector multiply: ``out[i] = M[idx[i]]^T @ v[i]``.

    Computes the transpose matrix-vector product for each element using
    its segment's matrix.  ``idx`` need not be sorted.

    Dispatches through :data:`_segmented_mul_overloads` since Warp's native
    ``mul(vec, mat)`` computes ``v^T @ M = M^T @ v``.

    Parameters
    ----------
    v : wp.array, shape (N,), dtype vec3h / vec3f / vec3d
        Per-atom input vectors.
    m : wp.array, shape (M,), dtype mat33h / mat33f / mat33d
        Per-segment 3x3 matrices (precision must match v).
    idx : wp.array(dtype=int32), shape (N,)
        Segment indices in ``[0, M)``.
    out : wp.array, shape (N,), dtype matches v
        Output transformed vectors.

    Notes
    -----
    Uses transpose convention: ``M^T @ v``, not ``M @ v``.

    See Also
    --------
    segmented_mul : Broadcast scalar multiplication
    segmented_add : Broadcast vector addition
    """
    N = v.shape[0]
    if N == 0:
        return
    wp.launch(
        _segmented_mul_overloads[(v.dtype, m.dtype)],
        dim=N,
        inputs=[v, m, idx, out],
        device=v.device,
    )


def segmented_max(
    x: wp.array,
    idx: wp.array,
    out: wp.array,
) -> None:
    """Compute maximum scalar value per segment using RLE reduction.

    ``out[s] = max(x[i] for i where idx[i] == s)``

    Requires sorted ``idx`` in non-decreasing order.  Caller must
    initialize ``out`` to ``-inf`` before calling.

    Parameters
    ----------
    x : wp.array, shape (N,), dtype float32 / float64
        Input scalar values.
    idx : wp.array(dtype=int32), shape (N,)
        Sorted segment indices in ``[0, M)``.
    out : wp.array, shape (M,), dtype matches x
        Maximum values per segment. Must be initialized to -inf by caller.

    Notes
    -----
    float16 is not supported (Warp atomic_max requires float32/float64).
    """
    N = x.shape[0]
    if N == 0:
        return
    device = x.device
    ept = compute_ept(N, max(device.sm_count, 1), False)
    dim = (N + ept - 1) // ept
    wp.launch(
        _segmented_max_overloads[x.dtype],
        dim=dim,
        inputs=[x, idx, out, N, ept],
        device=device,
    )


def segmented_min(
    x: wp.array,
    idx: wp.array,
    out: wp.array,
) -> None:
    """Compute minimum scalar value per segment using RLE reduction.

    ``out[s] = min(x[i] for i where idx[i] == s)``

    Requires sorted ``idx`` in non-decreasing order.  Caller must
    initialize ``out`` to ``+inf`` before calling.

    Parameters
    ----------
    x : wp.array, shape (N,), dtype float32 / float64
        Input scalar values.
    idx : wp.array(dtype=int32), shape (N,)
        Sorted segment indices in ``[0, M)``.
    out : wp.array, shape (M,), dtype matches x
        Minimum values per segment. Must be initialized to +inf by caller.

    Notes
    -----
    float16 is not supported (Warp atomic_min requires float32/float64).
    """
    N = x.shape[0]
    if N == 0:
        return
    device = x.device
    ept = compute_ept(N, max(device.sm_count, 1), False)
    dim = (N + ept - 1) // ept
    wp.launch(
        _segmented_min_overloads[x.dtype],
        dim=dim,
        inputs=[x, idx, out, N, ept],
        device=device,
    )


def segmented_broadcast(
    values: wp.array,
    idx: wp.array,
    out: wp.array,
) -> None:
    """Broadcast per-segment values to per-element array: ``out[i] = values[idx[i]]``.

    Pure gather operation that copies a per-segment value to every element
    belonging to that segment.

    Parameters
    ----------
    values : wp.array, shape (M,)
        Per-segment values to broadcast.
        Supported dtypes: ``float16``, ``float32``, ``float64``,
        ``vec3h``, ``vec3f``, ``vec3d``.
    idx : wp.array(dtype=int32), shape (N,)
        Segment indices in ``[0, M)``. Need not be sorted.
    out : wp.array, shape (N,), dtype matches values
        Output array with broadcast values.
    """
    N = out.shape[0]
    if N == 0:
        return
    wp.launch(
        _segmented_broadcast_overloads[values.dtype],
        dim=N,
        inputs=[values, idx, out],
        device=values.device,
    )


def segment_div(
    numerator: wp.array,
    denominator: wp.array,
    result: wp.array,
) -> None:
    """Element-wise division with zero-denominator guard.

    Computes ``result[i] = numerator[i] / denominator[i]`` where zero
    denominators produce zero results instead of NaN/inf.

    Parameters
    ----------
    numerator : wp.array, shape (N,), dtype float16 / float32 / float64
        Numerator values.
    denominator : wp.array, shape (N,), dtype int32
        Denominator values (e.g., segment element counts).
    result : wp.array, shape (N,), dtype matches numerator
        Output division results.
    """
    N = numerator.shape[0]
    if N == 0:
        return
    wp.launch(
        _segment_div_overloads[numerator.dtype],
        dim=N,
        inputs=[numerator, denominator, result],
        device=numerator.device,
    )


def segmented_mean(
    x: wp.array,
    idx: wp.array,
    sums: wp.array,
    counts: wp.array,
    out: wp.array,
) -> None:
    """Compute per-segment mean by composing sum + count + divide.

    ``out[s] = sum(x[i] for i where idx[i] == s) / count(s)``

    For vector types (vec3f/vec3d), computes the mean vector per segment.
    Requires sorted ``idx`` in non-decreasing order.

    All scratch arrays must be pre-allocated. This avoids hidden allocations
    and allows reuse across calls.

    Parameters
    ----------
    x : wp.array, shape (N,), dtype float32 / float64 / vec3f / vec3d
        Input values to average per segment.
    idx : wp.array(dtype=int32), shape (N,)
        Sorted segment indices in ``[0, M)``.
    sums : wp.array, shape (M,), dtype matches x
        Scratch array for per-segment sums. Zeroed internally before each use.
    counts : wp.array(dtype=int32), shape (M,)
        Scratch array for per-segment element counts. Zeroed internally before each use.
    out : wp.array, shape (M,), dtype matches x
        Output mean values per segment.

    Notes
    -----
    - float16/vec3h not supported (reduction requires atomics).
    - Composes ``segmented_sum`` + ``segmented_count`` + divide internally.
    - ``sums`` and ``counts`` are written to as scratch space.
    """
    N = x.shape[0]
    M = out.shape[0]
    if N == 0:
        return
    device = x.device
    dtype = x.dtype

    segmented_sum(x, idx, sums)
    segmented_count(idx, counts)

    if dtype in _VEC_TO_SCALAR:
        wp.launch(
            _segmented_vec_div_by_count_overloads[dtype],
            dim=M,
            inputs=[sums, counts, out],
            device=device,
        )
    else:
        segment_div(sums, counts, out)


def segmented_rms_norm(
    x: wp.array,
    idx: wp.array,
    sum_sq: wp.array,
    counts: wp.array,
    out: wp.array,
) -> None:
    """Compute RMS (root mean square) vector norm per segment.

    ``out[s] = sqrt(mean(dot(x[i], x[i]) for i where idx[i] == s))``

    This is ``sqrt(sum_of_squared_norms / count)`` for each segment.
    Requires sorted ``idx`` in non-decreasing order.

    All scratch arrays must be pre-allocated.

    Parameters
    ----------
    x : wp.array, shape (N,), dtype vec3f / vec3d
        Input 3D vectors.
    idx : wp.array(dtype=int32), shape (N,)
        Sorted segment indices in ``[0, M)``.
    sum_sq : wp.array, shape (M,), dtype float32 / float64
        Scratch array for per-segment sum of squared norms.
        Zeroed internally before each use.
    counts : wp.array(dtype=int32), shape (M,)
        Scratch array for per-segment element counts.
        Zeroed internally before each use.
    out : wp.array, shape (M,), dtype float32 / float64
        Output RMS norms per segment. Precision matches ``x``.

    Notes
    -----
    - float16/vec3h not supported (reduction requires atomics).
    - Composes ``segmented_dot`` + ``segmented_count`` + finalize internally.
    - ``sum_sq`` and ``counts`` are written to as scratch space.
    """
    N = x.shape[0]
    M = out.shape[0]
    if N == 0:
        return
    device = x.device
    scalar_dtype = _VEC_TO_SCALAR[x.dtype]

    segmented_dot(x, x, idx, sum_sq)
    segmented_count(idx, counts)

    wp.launch(
        _segmented_rms_norm_finalize_overloads[scalar_dtype],
        dim=M,
        inputs=[sum_sq, counts, out],
        device=device,
    )


def segmented_count(
    idx: wp.array,
    out: wp.array,
) -> None:
    """Count elements per segment using run-length encoding.

    ``out[s] = count(i where idx[i] == s)``

    Requires sorted ``idx`` in non-decreasing order.

    Parameters
    ----------
    idx : wp.array(dtype=int32), shape (N,)
        Sorted segment indices in ``[0, M)``.
    out : wp.array(dtype=int32), shape (M,)
        Per-segment element counts. Zeroed internally before each use.
    """
    N = idx.shape[0]
    if N == 0:
        return

    out.zero_()
    device = idx.device
    M = out.shape[0]

    # -- M=1 fast path: single thread counts everything ---------------------
    if M == 1:
        wp.launch(
            _segmented_count_kernel,
            dim=1,
            inputs=[idx, out, N, N],
            device=device,
        )
        return

    # -- General path: run-length segmented count ---------------------------
    ept = compute_ept(N, max(device.sm_count, 1), False)
    dim = (N + ept - 1) // ept
    wp.launch(
        _segmented_count_kernel,
        dim=dim,
        inputs=[idx, out, N, ept],
        device=device,
    )
