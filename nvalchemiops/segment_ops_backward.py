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

"""Explicit first- and second-order backward kernels for segment ops (PR 1).

No public API change.  All ``_launch_*`` functions are internal contracts
consumed by the Torch bindings in PR 2 and the JAX bindings in PR 3.

Design
------
- Every differentiable op gets an explicit first-order backward kernel and an
  explicit double-backward kernel, registered via ``register_overloads``.
- Linear ops (sum, broadcast, add, mean, segment_div) reuse existing forward
  kernels; only their launch functions are new.
- Bilinear ops (dot, mul, axpy, axpby, inner_products, matvec) require a small
  number of new element-wise kernels for the mixed-operand terms.
- Nonlinear ops (rms_norm, max_norm) have precompute-backward variants that
  save cheap intermediates during the forward pass.
- ``idx``, ``segment_ptr``, and ``num_segments`` are integer meta; their
  gradient slots always return ``None`` at the Torch/JAX layer.
"""

from __future__ import annotations

from typing import Any

import warp as wp

from nvalchemiops.segment_ops import (
    _ALL_SCALAR_TYPES,
    _ALL_VEC_SCALAR_PAIRS,
    _BLOCK_DIM,
    _SCALAR_TYPES,
    _VEC_MAT_PAIRS,
    _VEC_SCALAR_PAIRS,
    _VEC_TO_SCALAR,
    _VEC_TYPES,
    _segment_div_overloads,
    _segmented_broadcast_overloads,
    _segmented_component_sum_overloads,
    _segmented_dot_overloads,
    _segmented_mul_overloads,
    _segmented_sum_overloads,
    _segmented_vec_div_by_count_overloads,
    _total_sum_tile_overloads,
    compute_ept,
    segmented_count,
    segmented_dot,
)
from nvalchemiops.warp_dispatch import register_overloads

# ---------------------------------------------------------------------------
# Helpers shared with forward layer
# ---------------------------------------------------------------------------

_SCALAR_TO_VEC = {wp.float32: wp.vec3f, wp.float64: wp.vec3d}


def _launch_sum(x: wp.array, idx: wp.array, out: wp.array) -> None:
    """Zero out and run segmented_sum via the forward overloads."""
    N = x.shape[0]
    if N == 0:
        return
    out.zero_()
    device = x.device
    M = out.shape[0]
    if M == 1 and N >= 8192:
        full_blocks = N // _BLOCK_DIM
        wp.launch_tiled(
            _total_sum_tile_overloads[x.dtype],
            dim=full_blocks,
            inputs=[x, out],
            block_dim=_BLOCK_DIM,
            device=device,
        )
        rem = N - full_blocks * _BLOCK_DIM
        if rem > 0:
            wp.launch(
                _segmented_sum_overloads[x.dtype],
                dim=rem,
                inputs=[
                    x[full_blocks * _BLOCK_DIM :],
                    idx[full_blocks * _BLOCK_DIM :],
                    out,
                    rem,
                    1,
                ],
                device=device,
            )
        return
    ept = compute_ept(N, max(device.sm_count, 1), x.dtype in _VEC_TYPES)
    dim = (N + ept - 1) // ept
    wp.launch(
        _segmented_sum_overloads[x.dtype],
        dim=dim,
        inputs=[x, idx, out, N, ept],
        device=device,
    )


def _launch_broadcast(values: wp.array, idx: wp.array, out: wp.array) -> None:
    """Zero out and run segmented_broadcast via the forward overloads."""
    N = out.shape[0]
    if N == 0:
        return
    out.zero_()
    wp.launch(
        _segmented_broadcast_overloads[values.dtype],
        dim=N,
        inputs=[values, idx, out],
        device=values.device,
    )


# ===========================================================================
# Section 1 – component_sum backward
# ===========================================================================
# Forward:   out[s]  = sum_i (x[i][0]+x[i][1]+x[i][2])   x:vec3, out:scalar
# Backward:  grad_x[i]  = vec3(g_out[s], g_out[s], g_out[s])
# Dbl-bwd:   grad_g_out[s] = sum_i (gg_x[i][0]+gg_x[i][1]+gg_x[i][2])
#             → reuses _segmented_component_sum_overloads


@wp.kernel(enable_backward=False)
def _segmented_component_sum_backward_kernel(
    g_out: wp.array(dtype=Any),
    idx: wp.array(dtype=wp.int32),
    grad_x: wp.array(dtype=Any),
):
    i = wp.tid()
    s = idx[i]
    v = g_out[s]
    grad_x[i] = type(grad_x[0])(v, v, v)


_segmented_component_sum_backward_overloads = register_overloads(
    _segmented_component_sum_backward_kernel,
    lambda v, s: [wp.array(dtype=s), wp.array(dtype=wp.int32), wp.array(dtype=v)],
    dtype_pairs=_VEC_SCALAR_PAIRS,
)


# ===========================================================================
# Section 2 – inner_products backward
# ===========================================================================
# Forward:  out_xy[s]=sum x[i]*y[i], out_xx=sum x*x, out_yy=sum y*y
# Backward: grad_x[i] = g_xy[s]*y[i] + 2*g_xx[s]*x[i]
#           grad_y[i] = g_xy[s]*x[i] + 2*g_yy[s]*y[i]


@wp.kernel(enable_backward=False)
def _segmented_inner_products_backward_scalar_kernel(
    x: wp.array(dtype=Any),
    y: wp.array(dtype=Any),
    idx: wp.array(dtype=wp.int32),
    g_xy: wp.array(dtype=Any),
    g_xx: wp.array(dtype=Any),
    g_yy: wp.array(dtype=Any),
    grad_x: wp.array(dtype=Any),
    grad_y: wp.array(dtype=Any),
):
    i = wp.tid()
    s = idx[i]
    two = type(x[0])(2.0)
    grad_x[i] = g_xy[s] * y[i] + two * g_xx[s] * x[i]
    grad_y[i] = g_xy[s] * x[i] + two * g_yy[s] * y[i]


@wp.kernel(enable_backward=False)
def _segmented_inner_products_backward_vec_kernel(
    x: wp.array(dtype=Any),
    y: wp.array(dtype=Any),
    idx: wp.array(dtype=wp.int32),
    g_xy: wp.array(dtype=Any),
    g_xx: wp.array(dtype=Any),
    g_yy: wp.array(dtype=Any),
    grad_x: wp.array(dtype=Any),
    grad_y: wp.array(dtype=Any),
):
    i = wp.tid()
    s = idx[i]
    two = type(g_xy[0])(2.0)
    grad_x[i] = g_xy[s] * y[i] + two * g_xx[s] * x[i]
    grad_y[i] = g_xy[s] * x[i] + two * g_yy[s] * y[i]


_segmented_inner_products_backward_overloads = register_overloads(
    _segmented_inner_products_backward_scalar_kernel,
    lambda t: [wp.array(dtype=t)] * 2
    + [wp.array(dtype=wp.int32)]
    + [wp.array(dtype=t)] * 5,
    dtypes=_SCALAR_TYPES,
)
_segmented_inner_products_backward_overloads.update(
    register_overloads(
        _segmented_inner_products_backward_vec_kernel,
        lambda v, s: [wp.array(dtype=v)] * 2
        + [wp.array(dtype=wp.int32)]
        + [wp.array(dtype=s)] * 3
        + [wp.array(dtype=v)] * 2,
        dtype_pairs=_VEC_SCALAR_PAIRS,
    )
)


# ===========================================================================
# Section 3 – mean backward
# ===========================================================================
# Forward (composed): out[s] = sum(x)/count[s]
# Backward: grad_x[i] = g_out[idx[i]] / float(counts[idx[i]])


@wp.kernel(enable_backward=False)
def _segmented_mean_backward_scalar_kernel(
    g_out: wp.array(dtype=Any),
    counts: wp.array(dtype=wp.int32),
    idx: wp.array(dtype=wp.int32),
    grad_x: wp.array(dtype=Any),
):
    i = wp.tid()
    s = idx[i]
    c = counts[s]
    if c > 0:
        grad_x[i] = g_out[s] / type(g_out[0])(c)
    else:
        grad_x[i] = type(g_out[0])(0.0)


@wp.kernel(enable_backward=False)
def _segmented_mean_backward_vec_kernel(
    g_out: wp.array(dtype=Any),
    counts: wp.array(dtype=wp.int32),
    idx: wp.array(dtype=wp.int32),
    grad_x: wp.array(dtype=Any),
):
    i = wp.tid()
    s = idx[i]
    c = counts[s]
    if c > 0:
        grad_x[i] = g_out[s] / type(g_out[0][0])(c)
    else:
        grad_x[i] = type(g_out[0])()


_segmented_mean_backward_scalar_overloads = register_overloads(
    _segmented_mean_backward_scalar_kernel,
    lambda t: [
        wp.array(dtype=t),
        wp.array(dtype=wp.int32),
        wp.array(dtype=wp.int32),
        wp.array(dtype=t),
    ],
    dtypes=_SCALAR_TYPES,
)

_segmented_mean_backward_vec_overloads = register_overloads(
    _segmented_mean_backward_vec_kernel,
    lambda v, s: [
        wp.array(dtype=v),
        wp.array(dtype=wp.int32),
        wp.array(dtype=wp.int32),
        wp.array(dtype=v),
    ],
    dtype_pairs=_VEC_SCALAR_PAIRS,
)


# ===========================================================================
# Section 4 – rms_norm: precompute-forward, backward, double-backward
# ===========================================================================
# Forward (composed): sum_sq[s]=sum dot(x,x), count[s], out[s]=sqrt(sum_sq/count)
# Precompute saves: inv_norm[s] = 1/(out[s]*count[s])
# Backward: grad_x[i] = g_out[s] * x[i] * inv_norm[s]
# Double-backward:
#   inner[s] = sum_i dot(gg_x[i], x[i])            → reuse segmented_dot
#   grad_g_out[s] = inner[s] * inv_norm[s]
#   grad_x_extra[i] = g_out[s]*inv_norm[s]*gg_x[i]
#                   - g_out[s]*inv_norm[s]^3*count[s]*inner[s]*x[i]


@wp.kernel(enable_backward=False)
def _segmented_rms_norm_finalize_and_save_kernel(
    sum_sq: wp.array(dtype=Any),
    counts: wp.array(dtype=wp.int32),
    out: wp.array(dtype=Any),
    inv_norm: wp.array(dtype=Any),
):
    """out[s] = sqrt(sum_sq/count); inv_norm[s] = 1/(out[s]*count[s])."""
    s = wp.tid()
    c = counts[s]
    if c > 0:
        r = wp.sqrt(sum_sq[s] / type(sum_sq[0])(c))
        out[s] = r
        denom = r * type(r)(c)
        if denom > type(r)(0.0):
            inv_norm[s] = type(r)(1.0) / denom
        else:
            inv_norm[s] = type(r)(0.0)
    else:
        out[s] = type(sum_sq[0])(0.0)
        inv_norm[s] = type(sum_sq[0])(0.0)


_segmented_rms_norm_finalize_and_save_overloads = register_overloads(
    _segmented_rms_norm_finalize_and_save_kernel,
    lambda t: [
        wp.array(dtype=t),
        wp.array(dtype=wp.int32),
        wp.array(dtype=t),
        wp.array(dtype=t),
    ],
    dtypes=_SCALAR_TYPES,
)


@wp.kernel(enable_backward=False)
def _segmented_rms_norm_backward_kernel(
    g_out: wp.array(dtype=Any),
    x: wp.array(dtype=Any),
    inv_norm: wp.array(dtype=Any),
    idx: wp.array(dtype=wp.int32),
    grad_x: wp.array(dtype=Any),
):
    """grad_x[i] = g_out[idx[i]] * x[i] * inv_norm[idx[i]]."""
    i = wp.tid()
    s = idx[i]
    grad_x[i] = g_out[s] * inv_norm[s] * x[i]


_segmented_rms_norm_backward_overloads = register_overloads(
    _segmented_rms_norm_backward_kernel,
    lambda v, s: [
        wp.array(dtype=s),
        wp.array(dtype=v),
        wp.array(dtype=s),
        wp.array(dtype=wp.int32),
        wp.array(dtype=v),
    ],
    dtype_pairs=_VEC_SCALAR_PAIRS,
)


@wp.kernel(enable_backward=False)
def _segmented_rms_norm_dbl_bwd_grad_g_out_kernel(
    inner: wp.array(dtype=Any),
    inv_norm: wp.array(dtype=Any),
    grad_g_out: wp.array(dtype=Any),
):
    """grad_g_out[s] = inner[s] * inv_norm[s]  (per-segment, dim=M)."""
    s = wp.tid()
    grad_g_out[s] = inner[s] * inv_norm[s]


_segmented_rms_norm_dbl_bwd_grad_g_out_overloads = register_overloads(
    _segmented_rms_norm_dbl_bwd_grad_g_out_kernel,
    lambda t: [wp.array(dtype=t), wp.array(dtype=t), wp.array(dtype=t)],
    dtypes=_SCALAR_TYPES,
)


@wp.kernel(enable_backward=False)
def _segmented_rms_norm_dbl_bwd_grad_x_kernel(
    gg_x: wp.array(dtype=Any),
    x: wp.array(dtype=Any),
    g_out: wp.array(dtype=Any),
    inv_norm: wp.array(dtype=Any),
    counts: wp.array(dtype=wp.int32),
    inner: wp.array(dtype=Any),
    idx: wp.array(dtype=wp.int32),
    grad_x_extra: wp.array(dtype=Any),
):
    """Per-element element of the double-backward for rms_norm."""
    i = wp.tid()
    s = idx[i]
    c = type(g_out[0])(counts[s])
    n = inv_norm[s]
    # grad_x[i] = g_out[s]*n*gg_x[i] - g_out[s]*n^3*c*inner[s]*x[i]
    coeff_direct = g_out[s] * n
    coeff_cross = g_out[s] * n * n * n * c * inner[s]
    grad_x_extra[i] = coeff_direct * gg_x[i] - coeff_cross * x[i]


_segmented_rms_norm_dbl_bwd_grad_x_overloads = register_overloads(
    _segmented_rms_norm_dbl_bwd_grad_x_kernel,
    lambda v, s: [
        wp.array(dtype=v),
        wp.array(dtype=v),
        wp.array(dtype=s),
        wp.array(dtype=s),
        wp.array(dtype=wp.int32),
        wp.array(dtype=s),
        wp.array(dtype=wp.int32),
        wp.array(dtype=v),
    ],
    dtype_pairs=_VEC_SCALAR_PAIRS,
)


# ===========================================================================
# Section 5 – max_norm: precompute-forward, backward, double-backward
# ===========================================================================
# Precompute (second pass): argmax_idx[s] = max index i achieving max_norm[s]
# Backward (subgradient): only the argmax element receives gradient
#   grad_x[i] = g_out[s] * x[i]/||x[i]||   if i==argmax_idx[s] and ||x[i]||>0
# Double-backward (tangent plane projection at argmax):
#   grad_x_extra[i*] = g_out[s]/||x[i*]|| * (gg_gx[i*] - x_hat*dot(x_hat,gg_gx[i*]))
#   grad_g_out[s]    = dot(x_hat[i*], gg_gx[i*])


@wp.kernel(enable_backward=False)
def _segmented_max_norm_argmax_kernel(
    x: wp.array(dtype=Any),
    idx: wp.array(dtype=wp.int32),
    max_norms: wp.array(dtype=Any),
    argmax_idx: wp.array(dtype=wp.int32),
    N: wp.int32,
    elems_per_thread: wp.int32,
):
    """For each i where length(x[i]) == max_norms[idx[i]], record i via atomic_max."""
    t = wp.tid()
    start = t * elems_per_thread
    if start >= N:
        return
    end = wp.min(start + elems_per_thread, N)
    for i in range(start, end):
        s = idx[i]
        if wp.length(x[i]) == max_norms[s]:
            wp.atomic_max(argmax_idx, s, i)


_segmented_max_norm_argmax_overloads = register_overloads(
    _segmented_max_norm_argmax_kernel,
    lambda v, s: [
        wp.array(dtype=v),
        wp.array(dtype=wp.int32),
        wp.array(dtype=s),
        wp.array(dtype=wp.int32),
        wp.int32,
        wp.int32,
    ],
    dtype_pairs=_VEC_SCALAR_PAIRS,
)


@wp.kernel(enable_backward=False)
def _segmented_max_norm_backward_kernel(
    g_out: wp.array(dtype=Any),
    x: wp.array(dtype=Any),
    argmax_idx: wp.array(dtype=wp.int32),
    idx: wp.array(dtype=wp.int32),
    grad_x: wp.array(dtype=Any),
):
    """Subgradient: grad_x[i] = g_out[s]*x[i]/||x[i]|| only at argmax element."""
    i = wp.tid()
    s = idx[i]
    if i == argmax_idx[s]:
        n = wp.length(x[i])
        if n > type(n)(0.0):
            grad_x[i] = g_out[s] * x[i] / n
        # else grad_x[i] stays zero (already zeroed)


_segmented_max_norm_backward_overloads = register_overloads(
    _segmented_max_norm_backward_kernel,
    lambda v, s: [
        wp.array(dtype=s),
        wp.array(dtype=v),
        wp.array(dtype=wp.int32),
        wp.array(dtype=wp.int32),
        wp.array(dtype=v),
    ],
    dtype_pairs=_VEC_SCALAR_PAIRS,
)


@wp.kernel(enable_backward=False)
def _segmented_max_norm_double_backward_kernel(
    gg_gx: wp.array(dtype=Any),
    g_out: wp.array(dtype=Any),
    x: wp.array(dtype=Any),
    argmax_idx: wp.array(dtype=wp.int32),
    idx: wp.array(dtype=wp.int32),
    grad_x_extra: wp.array(dtype=Any),
    grad_g_out: wp.array(dtype=Any),
):
    """Tangent-plane projection at argmax element."""
    i = wp.tid()
    s = idx[i]
    if i == argmax_idx[s]:
        n = wp.length(x[i])
        if n > type(n)(0.0):
            x_hat = x[i] / n
            proj = wp.dot(x_hat, gg_gx[i])
            grad_x_extra[i] = (g_out[s] / n) * (gg_gx[i] - x_hat * proj)
            wp.atomic_add(grad_g_out, s, proj)


_segmented_max_norm_double_backward_overloads = register_overloads(
    _segmented_max_norm_double_backward_kernel,
    lambda v, s: [
        wp.array(dtype=v),
        wp.array(dtype=s),
        wp.array(dtype=v),
        wp.array(dtype=wp.int32),
        wp.array(dtype=wp.int32),
        wp.array(dtype=v),
        wp.array(dtype=s),
    ],
    dtype_pairs=_VEC_SCALAR_PAIRS,
)


# ===========================================================================
# Section 6 – matvec backward and double-backward
# ===========================================================================
# Forward:   out[i] = M[s]^T @ v[i]   (wp.mul(v[i], m[s]) = v^T M = M^T v)
# Backward:  grad_v[i] = M[s] @ g_out[i]   (wp.mul(m[s], g_out[i]))
#            grad_M[s] = sum_i outer(v[i], g_out[i])
# Double-bwd from {gg_gv, gg_gM}:
#   grad_g_out[i] = M[s]^T @ gg_gv[i]          [reuse fwd mul overload]
#                 + gg_gM[s]^T @ v[i]           [reuse fwd mul overload]
#   grad_v_extra[i] = gg_gM[s] @ g_out[i]       [reuse backward_v kernel]
#   grad_M_extra[s] = sum_i outer(gg_gv[i], g_out[i])   [reuse backward_M kernel]


@wp.kernel(enable_backward=False)
def _segmented_matvec_backward_v_kernel(
    g_out: wp.array(dtype=Any),
    m: wp.array(dtype=Any),
    idx: wp.array(dtype=wp.int32),
    grad_v: wp.array(dtype=Any),
):
    """grad_v[i] = M[idx[i]] @ g_out[i]  (standard mat-vec, no transpose)."""
    i = wp.tid()
    grad_v[i] = wp.mul(m[idx[i]], g_out[i])


_segmented_matvec_backward_v_overloads = register_overloads(
    _segmented_matvec_backward_v_kernel,
    lambda v, m: [
        wp.array(dtype=v),
        wp.array(dtype=m),
        wp.array(dtype=wp.int32),
        wp.array(dtype=v),
    ],
    dtype_pairs=_VEC_MAT_PAIRS,
    key_fn=lambda v, m: (v, m),
)


@wp.kernel(enable_backward=False)
def _segmented_matvec_backward_M_kernel(
    g_out: wp.array(dtype=Any),
    v: wp.array(dtype=Any),
    idx: wp.array(dtype=wp.int32),
    grad_M: wp.array(dtype=Any),
    N: wp.int32,
    elems_per_thread: wp.int32,
):
    """grad_M[s] = sum_i outer(v[i], g_out[i])  using RLE for low atomics."""
    t = wp.tid()
    start = t * elems_per_thread
    if start >= N:
        return
    end = wp.min(start + elems_per_thread, N)
    s_cur = idx[start]
    acc = wp.outer(v[start], g_out[start])
    for i in range(start + 1, end):
        s = idx[i]
        if s == s_cur:
            acc = acc + wp.outer(v[i], g_out[i])
        else:
            wp.atomic_add(grad_M, s_cur, acc)
            s_cur = s
            acc = wp.outer(v[i], g_out[i])
    wp.atomic_add(grad_M, s_cur, acc)


_segmented_matvec_backward_M_overloads = register_overloads(
    _segmented_matvec_backward_M_kernel,
    lambda v, m: [
        wp.array(dtype=v),
        wp.array(dtype=v),
        wp.array(dtype=wp.int32),
        wp.array(dtype=m),
        wp.int32,
        wp.int32,
    ],
    dtype_pairs=_VEC_MAT_PAIRS,
    key_fn=lambda v, m: (v, m),
)


# ===========================================================================
# Section 7 – mul double-backward
# ===========================================================================
# Forward:   out[i] = x[i] * y[idx[i]]
# Backward:  grad_x[i] = g_out[i]*y[s]   (reuse _segmented_mul)
#            grad_y[s] = sum dot(g_out,x)  (reuse _segmented_dot)
# Double-bwd: grad_g_out[i] = gg_gx[i]*y[s] + gg_gy[s]*x[i]
#             grad_x_extra[i] = gg_gy[s]*g_out[i]   (reuse _segmented_mul)
#             grad_y_extra[s] = sum dot(gg_gx, g_out) (reuse _segmented_dot)


@wp.kernel(enable_backward=False)
def _segmented_mul_dbl_bwd_grad_out_scalar_kernel(
    gg_gx: wp.array(dtype=Any),
    y: wp.array(dtype=Any),
    gg_gy: wp.array(dtype=Any),
    x: wp.array(dtype=Any),
    idx: wp.array(dtype=wp.int32),
    grad_g_out: wp.array(dtype=Any),
):
    """Scalar: grad_g_out[i] = gg_gx[i]*y[s] + gg_gy[s]*x[i]."""
    i = wp.tid()
    s = idx[i]
    grad_g_out[i] = gg_gx[i] * y[s] + gg_gy[s] * x[i]


@wp.kernel(enable_backward=False)
def _segmented_mul_dbl_bwd_grad_out_vec_scalar_kernel(
    gg_gx: wp.array(dtype=Any),
    y: wp.array(dtype=Any),
    gg_gy: wp.array(dtype=Any),
    x: wp.array(dtype=Any),
    idx: wp.array(dtype=wp.int32),
    grad_g_out: wp.array(dtype=Any),
):
    """Vec-scalar: grad_g_out[i] = gg_gx_vec[i]*y_scalar[s] + gg_gy_scalar[s]*x_vec[i]."""
    i = wp.tid()
    s = idx[i]
    grad_g_out[i] = gg_gx[i] * y[s] + gg_gy[s] * x[i]


_segmented_mul_dbl_bwd_grad_out_overloads = register_overloads(
    _segmented_mul_dbl_bwd_grad_out_scalar_kernel,
    lambda t: [wp.array(dtype=t)] * 4 + [wp.array(dtype=wp.int32), wp.array(dtype=t)],
    dtypes=_ALL_SCALAR_TYPES,
    key_fn=lambda t: (t, t),
)
_segmented_mul_dbl_bwd_grad_out_overloads.update(
    register_overloads(
        _segmented_mul_dbl_bwd_grad_out_vec_scalar_kernel,
        lambda v, s: [
            wp.array(dtype=v),
            wp.array(dtype=s),
            wp.array(dtype=s),
            wp.array(dtype=v),
            wp.array(dtype=wp.int32),
            wp.array(dtype=v),
        ],
        dtype_pairs=_ALL_VEC_SCALAR_PAIRS,
        key_fn=lambda v, s: (v, s),
    )
)


# ===========================================================================
# Section 8 – axpby double-backward
# ===========================================================================
# Forward:   out[i] = a[s]*x[i] + b[s]*y[i]
# Backward:  grad_x[i]=a[s]*g_out[i], grad_y[i]=b[s]*g_out[i]
#            grad_a[s]=sum dot(x,g_out), grad_b[s]=sum dot(y,g_out)
# Double-bwd: grad_g_out[i]=gg_gx[i]*a[s]+gg_gy[i]*b[s]+gg_ga[s]*x[i]+gg_gb[s]*y[i]
#             grad_x_extra[i]=gg_ga[s]*g_out[i]  (reuse mul)
#             grad_y_extra[i]=gg_gb[s]*g_out[i]  (reuse mul)
#             grad_a_extra[s]=sum dot(gg_gx,g_out)  (reuse dot)
#             grad_b_extra[s]=sum dot(gg_gy,g_out)  (reuse dot)


@wp.kernel(enable_backward=False)
def _segmented_axpby_dbl_bwd_grad_out_scalar_kernel(
    gg_gx: wp.array(dtype=Any),
    a: wp.array(dtype=Any),
    gg_gy: wp.array(dtype=Any),
    b: wp.array(dtype=Any),
    gg_ga: wp.array(dtype=Any),
    x: wp.array(dtype=Any),
    gg_gb: wp.array(dtype=Any),
    y: wp.array(dtype=Any),
    idx: wp.array(dtype=wp.int32),
    grad_g_out: wp.array(dtype=Any),
):
    i = wp.tid()
    s = idx[i]
    grad_g_out[i] = (
        gg_gx[i] * a[s] + gg_gy[i] * b[s] + gg_ga[s] * x[i] + gg_gb[s] * y[i]
    )


@wp.kernel(enable_backward=False)
def _segmented_axpby_dbl_bwd_grad_out_vec_scalar_kernel(
    gg_gx: wp.array(dtype=Any),
    a: wp.array(dtype=Any),
    gg_gy: wp.array(dtype=Any),
    b: wp.array(dtype=Any),
    gg_ga: wp.array(dtype=Any),
    x: wp.array(dtype=Any),
    gg_gb: wp.array(dtype=Any),
    y: wp.array(dtype=Any),
    idx: wp.array(dtype=wp.int32),
    grad_g_out: wp.array(dtype=Any),
):
    i = wp.tid()
    s = idx[i]
    # gg_gx/gg_gy/x/y are vec3; a/b/gg_ga/gg_gb are scalar
    grad_g_out[i] = (
        a[s] * gg_gx[i] + b[s] * gg_gy[i] + gg_ga[s] * x[i] + gg_gb[s] * y[i]
    )


_segmented_axpby_dbl_bwd_grad_out_overloads = register_overloads(
    _segmented_axpby_dbl_bwd_grad_out_scalar_kernel,
    lambda t: [wp.array(dtype=t)] * 8 + [wp.array(dtype=wp.int32), wp.array(dtype=t)],
    dtypes=_ALL_SCALAR_TYPES,
)
_segmented_axpby_dbl_bwd_grad_out_overloads.update(
    register_overloads(
        _segmented_axpby_dbl_bwd_grad_out_vec_scalar_kernel,
        lambda v, s: [
            wp.array(dtype=v),
            wp.array(dtype=s),
            wp.array(dtype=v),
            wp.array(dtype=s),
            wp.array(dtype=s),
            wp.array(dtype=v),
            wp.array(dtype=s),
            wp.array(dtype=v),
            wp.array(dtype=wp.int32),
            wp.array(dtype=v),
        ],
        dtype_pairs=_ALL_VEC_SCALAR_PAIRS,
    )
)


# ===========================================================================
# Section 9 – inner_products double-backward
# ===========================================================================
# Double-bwd: from gg_gx, gg_gy (second-order adjoints of grad_x, grad_y)
#   grad_x_extra[i] = 2*gg_gx[i]*g_xx[s] + gg_gy[i]*g_xy[s]
#   grad_y_extra[i] = gg_gx[i]*g_xy[s]   + 2*gg_gy[i]*g_yy[s]
# Reductions (via existing segmented_dot):
#   grad_g_xy_extra[s] = sum dot(gg_gx,y) + sum dot(gg_gy,x)
#   grad_g_xx_extra[s] = 2*sum dot(gg_gx,x)
#   grad_g_yy_extra[s] = 2*sum dot(gg_gy,y)


@wp.kernel(enable_backward=False)
def _segmented_inner_products_dbl_bwd_scalar_kernel(
    gg_gx: wp.array(dtype=Any),
    gg_gy: wp.array(dtype=Any),
    g_xy: wp.array(dtype=Any),
    g_xx: wp.array(dtype=Any),
    g_yy: wp.array(dtype=Any),
    idx: wp.array(dtype=wp.int32),
    grad_x_extra: wp.array(dtype=Any),
    grad_y_extra: wp.array(dtype=Any),
):
    i = wp.tid()
    s = idx[i]
    two = type(gg_gx[0])(2.0)
    grad_x_extra[i] = two * gg_gx[i] * g_xx[s] + gg_gy[i] * g_xy[s]
    grad_y_extra[i] = gg_gx[i] * g_xy[s] + two * gg_gy[i] * g_yy[s]


@wp.kernel(enable_backward=False)
def _segmented_inner_products_dbl_bwd_vec_kernel(
    gg_gx: wp.array(dtype=Any),
    gg_gy: wp.array(dtype=Any),
    g_xy: wp.array(dtype=Any),
    g_xx: wp.array(dtype=Any),
    g_yy: wp.array(dtype=Any),
    idx: wp.array(dtype=wp.int32),
    grad_x_extra: wp.array(dtype=Any),
    grad_y_extra: wp.array(dtype=Any),
):
    i = wp.tid()
    s = idx[i]
    two = type(g_xy[0])(2.0)
    grad_x_extra[i] = two * g_xx[s] * gg_gx[i] + g_xy[s] * gg_gy[i]
    grad_y_extra[i] = g_xy[s] * gg_gx[i] + two * g_yy[s] * gg_gy[i]


_segmented_inner_products_dbl_bwd_overloads = register_overloads(
    _segmented_inner_products_dbl_bwd_scalar_kernel,
    lambda t: [wp.array(dtype=t)] * 5
    + [wp.array(dtype=wp.int32)]
    + [wp.array(dtype=t)] * 2,
    dtypes=_SCALAR_TYPES,
)
_segmented_inner_products_dbl_bwd_overloads.update(
    register_overloads(
        _segmented_inner_products_dbl_bwd_vec_kernel,
        lambda v, s: [wp.array(dtype=v)] * 2
        + [wp.array(dtype=s)] * 3
        + [wp.array(dtype=wp.int32)]
        + [wp.array(dtype=v)] * 2,
        dtype_pairs=_VEC_SCALAR_PAIRS,
    )
)


# ===========================================================================
# Internal launch functions
# All functions below are the contracts consumed by PR 2 (Torch) and PR 3 (JAX).
# Convention: output arrays are zeroed by the callee before writing.
# ===========================================================================

# ---------------------------------------------------------------------------
# segmented_sum
# ---------------------------------------------------------------------------


def _launch_segmented_sum_backward(
    g_out: wp.array,
    idx: wp.array,
    grad_x: wp.array,
) -> None:
    """grad_x[i] = g_out[idx[i]]  (gather)."""
    _launch_broadcast(g_out, idx, grad_x)


def _launch_segmented_sum_double_backward(
    gg_x: wp.array,
    idx: wp.array,
    M: int,
    grad_g_out: wp.array,
) -> None:
    """grad_g_out[s] = sum_i gg_x[i]  (scatter-sum; same as forward)."""
    _launch_sum(gg_x, idx, grad_g_out)


# ---------------------------------------------------------------------------
# segmented_broadcast
# ---------------------------------------------------------------------------


def _launch_segmented_broadcast_backward(
    g_out: wp.array,
    idx: wp.array,
    M: int,
    grad_values: wp.array,
) -> None:
    """grad_values[s] = sum_i g_out[i]."""
    _launch_sum(g_out, idx, grad_values)


def _launch_segmented_broadcast_double_backward(
    gg_values: wp.array,
    idx: wp.array,
    grad_g_out: wp.array,
) -> None:
    """grad_g_out[i] = gg_values[idx[i]]."""
    _launch_broadcast(gg_values, idx, grad_g_out)


# ---------------------------------------------------------------------------
# segmented_component_sum
# ---------------------------------------------------------------------------


def _launch_segmented_component_sum_backward(
    g_out: wp.array,
    idx: wp.array,
    grad_x: wp.array,
) -> None:
    """grad_x[i] = vec3(g_out[s], g_out[s], g_out[s])."""
    N = grad_x.shape[0]
    if N == 0:
        return
    grad_x.zero_()
    wp.launch(
        _segmented_component_sum_backward_overloads[grad_x.dtype],
        dim=N,
        inputs=[g_out, idx, grad_x],
        device=grad_x.device,
    )


def _launch_segmented_component_sum_double_backward(
    gg_x: wp.array,
    idx: wp.array,
    M: int,
    grad_g_out: wp.array,
) -> None:
    """grad_g_out[s] = sum_i (gg_x[i][0]+gg_x[i][1]+gg_x[i][2])  (component_sum fwd)."""
    N = gg_x.shape[0]
    if N == 0:
        return
    grad_g_out.zero_()
    device = gg_x.device
    ept = compute_ept(N, max(device.sm_count, 1), True)
    dim = (N + ept - 1) // ept
    wp.launch(
        _segmented_component_sum_overloads[gg_x.dtype],
        dim=dim,
        inputs=[gg_x, idx, grad_g_out, N, ept],
        device=device,
    )


# ---------------------------------------------------------------------------
# segmented_add
# ---------------------------------------------------------------------------


def _launch_segmented_add_backward(
    g_out: wp.array,
    idx: wp.array,
    M: int,
    grad_x: wp.array,
    grad_y: wp.array,
) -> None:
    """grad_x[i] = g_out[i]; grad_y[s] = sum_i g_out[i].

    For mixed-type variants (vec+scalar or scalar+vec), the caller should use
    the type-specific helpers below instead.
    """
    if g_out.shape[0] == 0:
        return
    grad_x.zero_()
    grad_y.zero_()
    wp.copy(grad_x, g_out)
    _launch_sum(g_out, idx, grad_y)


def _launch_segmented_add_double_backward(
    gg_x: wp.array,
    gg_y: wp.array,
    idx: wp.array,
    grad_g_out: wp.array,
) -> None:
    """grad_g_out[i] = gg_x[i] + gg_y[idx[i]]."""
    N = gg_x.shape[0]
    if N == 0:
        return
    grad_g_out.zero_()
    # grad_g_out = gg_x + broadcast(gg_y)
    wp.copy(grad_g_out, gg_x)
    # Add gg_y[idx[i]]: use segmented_broadcast into a tmp then add inplace
    tmp = wp.zeros(N, dtype=gg_x.dtype, device=gg_x.device)
    wp.launch(
        _segmented_broadcast_overloads[gg_y.dtype],
        dim=N,
        inputs=[gg_y, idx, tmp],
        device=gg_x.device,
    )
    _add_arrays_inplace(grad_g_out, tmp)


# ---------------------------------------------------------------------------
# segmented_mul
# ---------------------------------------------------------------------------


def _launch_segmented_mul_backward(
    g_out: wp.array,
    x: wp.array,
    y: wp.array,
    idx: wp.array,
    M: int,
    grad_x: wp.array,
    grad_y: wp.array,
) -> None:
    """grad_x[i] = g_out[i]*y[s]; grad_y[s] = sum dot(g_out[i], x[i])."""
    N = g_out.shape[0]
    if N == 0:
        return
    grad_x.zero_()
    grad_y.zero_()
    # grad_x[i] = g_out[i] * y[s]
    wp.launch(
        _segmented_mul_overloads[(g_out.dtype, y.dtype)],
        dim=N,
        inputs=[g_out, y, idx, grad_x],
        device=g_out.device,
    )
    # grad_y[s] = sum dot(g_out, x)  →  segmented_dot
    device = g_out.device
    ept = compute_ept(N, max(device.sm_count, 1), g_out.dtype in _VEC_TYPES)
    dim = (N + ept - 1) // ept
    wp.launch(
        _segmented_dot_overloads[g_out.dtype],
        dim=dim,
        inputs=[g_out, x, idx, grad_y, N, ept],
        device=device,
    )


def _launch_segmented_mul_double_backward(
    gg_gx: wp.array,
    gg_gy: wp.array,
    g_out: wp.array,
    x: wp.array,
    y: wp.array,
    idx: wp.array,
    grad_g_out: wp.array,
    grad_x_extra: wp.array,
    grad_y_extra: wp.array,
) -> None:
    """Double-backward for segmented_mul."""
    N = g_out.shape[0]
    if N == 0:
        return
    grad_g_out.zero_()
    grad_x_extra.zero_()
    grad_y_extra.zero_()
    device = g_out.device
    # grad_g_out[i] = gg_gx[i]*y[s] + gg_gy[s]*x[i]
    wp.launch(
        _segmented_mul_dbl_bwd_grad_out_overloads[(g_out.dtype, y.dtype)],
        dim=N,
        inputs=[gg_gx, y, gg_gy, x, idx, grad_g_out],
        device=device,
    )
    # grad_x_extra[i] = gg_gy[s]*g_out[i]
    wp.launch(
        _segmented_mul_overloads[(g_out.dtype, gg_gy.dtype)],
        dim=N,
        inputs=[g_out, gg_gy, idx, grad_x_extra],
        device=device,
    )
    # grad_y_extra[s] = sum dot(gg_gx, g_out)
    ept = compute_ept(N, max(device.sm_count, 1), g_out.dtype in _VEC_TYPES)
    dim = (N + ept - 1) // ept
    wp.launch(
        _segmented_dot_overloads[g_out.dtype],
        dim=dim,
        inputs=[gg_gx, g_out, idx, grad_y_extra, N, ept],
        device=device,
    )


# ---------------------------------------------------------------------------
# segmented_dot
# ---------------------------------------------------------------------------


def _launch_segmented_dot_backward(
    g_out: wp.array,
    x: wp.array,
    y: wp.array,
    idx: wp.array,
    grad_x: wp.array,
    grad_y: wp.array,
) -> None:
    """grad_x[i] = g_out[s]*y[i]; grad_y[i] = g_out[s]*x[i]."""
    N = x.shape[0]
    if N == 0:
        return
    grad_x.zero_()
    grad_y.zero_()
    # grad_x[i] = y[i] * g_out[s]  → _segmented_mul(y, g_out, idx, grad_x)
    wp.launch(
        _segmented_mul_overloads[(y.dtype, g_out.dtype)],
        dim=N,
        inputs=[y, g_out, idx, grad_x],
        device=x.device,
    )
    # grad_y[i] = x[i] * g_out[s]
    wp.launch(
        _segmented_mul_overloads[(x.dtype, g_out.dtype)],
        dim=N,
        inputs=[x, g_out, idx, grad_y],
        device=x.device,
    )


def _launch_segmented_dot_double_backward(
    gg_gx: wp.array,
    gg_gy: wp.array,
    g_out: wp.array,
    x: wp.array,
    y: wp.array,
    idx: wp.array,
    M: int,
    grad_g_out: wp.array,
    grad_x_extra: wp.array,
    grad_y_extra: wp.array,
) -> None:
    """Double-backward for segmented_dot."""
    N = x.shape[0]
    if N == 0:
        return
    grad_g_out.zero_()
    grad_x_extra.zero_()
    grad_y_extra.zero_()
    device = x.device
    ept = compute_ept(N, max(device.sm_count, 1), x.dtype in _VEC_TYPES)
    dim_rle = (N + ept - 1) // ept
    # grad_g_out[s] = sum dot(gg_gx, y) + sum dot(gg_gy, x)
    tmp = wp.zeros(M, dtype=g_out.dtype, device=device)
    wp.launch(
        _segmented_dot_overloads[x.dtype],
        dim=dim_rle,
        inputs=[gg_gx, y, idx, grad_g_out, N, ept],
        device=device,
    )
    wp.launch(
        _segmented_dot_overloads[x.dtype],
        dim=dim_rle,
        inputs=[gg_gy, x, idx, tmp, N, ept],
        device=device,
    )
    _add_arrays_inplace(grad_g_out, tmp)
    # grad_x_extra[i] = gg_gy[i]*g_out[s]
    wp.launch(
        _segmented_mul_overloads[(gg_gy.dtype, g_out.dtype)],
        dim=N,
        inputs=[gg_gy, g_out, idx, grad_x_extra],
        device=device,
    )
    # grad_y_extra[i] = gg_gx[i]*g_out[s]
    wp.launch(
        _segmented_mul_overloads[(gg_gx.dtype, g_out.dtype)],
        dim=N,
        inputs=[gg_gx, g_out, idx, grad_y_extra],
        device=device,
    )


# ---------------------------------------------------------------------------
# segmented_inner_products
# ---------------------------------------------------------------------------


def _launch_segmented_inner_products_backward(
    x: wp.array,
    y: wp.array,
    idx: wp.array,
    g_xy: wp.array,
    g_xx: wp.array,
    g_yy: wp.array,
    grad_x: wp.array,
    grad_y: wp.array,
) -> None:
    """grad_x[i]=g_xy[s]*y[i]+2*g_xx[s]*x[i]; grad_y[i]=g_xy[s]*x[i]+2*g_yy[s]*y[i]."""
    N = x.shape[0]
    if N == 0:
        return
    grad_x.zero_()
    grad_y.zero_()
    wp.launch(
        _segmented_inner_products_backward_overloads[x.dtype],
        dim=N,
        inputs=[x, y, idx, g_xy, g_xx, g_yy, grad_x, grad_y],
        device=x.device,
    )


def _launch_segmented_inner_products_double_backward(
    gg_gx: wp.array,
    gg_gy: wp.array,
    x: wp.array,
    y: wp.array,
    g_xy: wp.array,
    g_xx: wp.array,
    g_yy: wp.array,
    idx: wp.array,
    M: int,
    grad_x_extra: wp.array,
    grad_y_extra: wp.array,
    grad_g_xy_extra: wp.array,
    grad_g_xx_extra: wp.array,
    grad_g_yy_extra: wp.array,
) -> None:
    N = x.shape[0]
    if N == 0:
        return
    for arr in (
        grad_x_extra,
        grad_y_extra,
        grad_g_xy_extra,
        grad_g_xx_extra,
        grad_g_yy_extra,
    ):
        arr.zero_()
    device = x.device
    ept = compute_ept(N, max(device.sm_count, 1), x.dtype in _VEC_TYPES)
    dim_rle = (N + ept - 1) // ept
    # element-wise grad_x_extra, grad_y_extra
    wp.launch(
        _segmented_inner_products_dbl_bwd_overloads[x.dtype],
        dim=N,
        inputs=[gg_gx, gg_gy, g_xy, g_xx, g_yy, idx, grad_x_extra, grad_y_extra],
        device=device,
    )
    # scalar outputs use segmented_dot
    scalar_dt = _VEC_TO_SCALAR.get(x.dtype, x.dtype)
    tmp = wp.zeros(M, dtype=scalar_dt, device=device)
    # grad_g_xy_extra[s] = sum dot(gg_gx,y) + sum dot(gg_gy,x)
    wp.launch(
        _segmented_dot_overloads[x.dtype],
        dim=dim_rle,
        inputs=[gg_gx, y, idx, grad_g_xy_extra, N, ept],
        device=device,
    )
    wp.launch(
        _segmented_dot_overloads[x.dtype],
        dim=dim_rle,
        inputs=[gg_gy, x, idx, tmp, N, ept],
        device=device,
    )
    _add_arrays_inplace(grad_g_xy_extra, tmp)
    # grad_g_xx_extra[s] = 2*sum dot(gg_gx, x)
    tmp.zero_()
    wp.launch(
        _segmented_dot_overloads[x.dtype],
        dim=dim_rle,
        inputs=[gg_gx, x, idx, tmp, N, ept],
        device=device,
    )
    _scale_and_add_inplace(grad_g_xx_extra, tmp, type_=scalar_dt, factor=2.0)
    # grad_g_yy_extra[s] = 2*sum dot(gg_gy, y)
    tmp.zero_()
    wp.launch(
        _segmented_dot_overloads[x.dtype],
        dim=dim_rle,
        inputs=[gg_gy, y, idx, tmp, N, ept],
        device=device,
    )
    _scale_and_add_inplace(grad_g_yy_extra, tmp, type_=scalar_dt, factor=2.0)


# ---------------------------------------------------------------------------
# segmented_axpy
# ---------------------------------------------------------------------------


def _launch_segmented_axpy_backward(
    g_out: wp.array,
    x: wp.array,
    a: wp.array,
    idx: wp.array,
    M: int,
    grad_y_in: wp.array,
    grad_x: wp.array,
    grad_a: wp.array,
) -> None:
    """Treat axpy as out[i]=y_in[i]+x[i]*a[s]; backward into (y_in,x,a)."""
    N = g_out.shape[0]
    if N == 0:
        return
    grad_y_in.zero_()
    grad_x.zero_()
    grad_a.zero_()
    wp.copy(grad_y_in, g_out)
    # grad_x[i] = a[s]*g_out[i]
    wp.launch(
        _segmented_mul_overloads[(g_out.dtype, a.dtype)],
        dim=N,
        inputs=[g_out, a, idx, grad_x],
        device=g_out.device,
    )
    # grad_a[s] = sum dot(x, g_out)
    device = g_out.device
    ept = compute_ept(N, max(device.sm_count, 1), x.dtype in _VEC_TYPES)
    dim = (N + ept - 1) // ept
    wp.launch(
        _segmented_dot_overloads[x.dtype],
        dim=dim,
        inputs=[x, g_out, idx, grad_a, N, ept],
        device=device,
    )


def _launch_segmented_axpy_double_backward(
    gg_gy_in: wp.array,
    gg_gx: wp.array,
    gg_ga: wp.array,
    g_out: wp.array,
    x: wp.array,
    a: wp.array,
    idx: wp.array,
    grad_g_out: wp.array,
    grad_x_extra: wp.array,
    grad_a_extra: wp.array,
) -> None:
    """Double-backward for segmented_axpy (treating it as out = y_in + x*a[s])."""
    N = g_out.shape[0]
    if N == 0:
        return
    grad_g_out.zero_()
    grad_x_extra.zero_()
    grad_a_extra.zero_()
    device = g_out.device
    # M = a.shape[0]
    ept = compute_ept(N, max(device.sm_count, 1), x.dtype in _VEC_TYPES)
    dim_rle = (N + ept - 1) // ept
    # From grad_y_in = g_out: grad_g_out += gg_gy_in (identity)
    wp.copy(grad_g_out, gg_gy_in)
    # From grad_x[i]=a[s]*g_out[i]:
    #   grad_g_out[i] += gg_gx[i]*a[s]   → mul(gg_gx, a)
    tmp = wp.zeros(N, dtype=g_out.dtype, device=device)
    wp.launch(
        _segmented_mul_overloads[(g_out.dtype, a.dtype)],
        dim=N,
        inputs=[gg_gx, a, idx, tmp],
        device=device,
    )
    _add_arrays_inplace(grad_g_out, tmp)
    #   grad_a_extra[s] += sum dot(gg_gx, g_out)
    wp.launch(
        _segmented_dot_overloads[x.dtype],
        dim=dim_rle,
        inputs=[gg_gx, g_out, idx, grad_a_extra, N, ept],
        device=device,
    )
    # From grad_a[s]=sum dot(x, g_out):
    #   grad_g_out[i] += gg_ga[s]*x[i]
    tmp.zero_()
    wp.launch(
        _segmented_mul_overloads[(x.dtype, gg_ga.dtype)],
        dim=N,
        inputs=[x, gg_ga, idx, tmp],
        device=device,
    )
    _add_arrays_inplace(grad_g_out, tmp)
    #   grad_x_extra[i] += gg_ga[s]*g_out[i]
    wp.launch(
        _segmented_mul_overloads[(g_out.dtype, gg_ga.dtype)],
        dim=N,
        inputs=[g_out, gg_ga, idx, grad_x_extra],
        device=device,
    )


# ---------------------------------------------------------------------------
# segmented_axpby
# ---------------------------------------------------------------------------


def _launch_segmented_axpby_backward(
    g_out: wp.array,
    a: wp.array,
    x: wp.array,
    b: wp.array,
    y: wp.array,
    idx: wp.array,
    M: int,
    grad_x: wp.array,
    grad_y: wp.array,
    grad_a: wp.array,
    grad_b: wp.array,
) -> None:
    """grad_x[i]=a[s]*g_out; grad_y[i]=b[s]*g_out; grad_a=sum dot(x,g_out); grad_b=sum dot(y,g_out)."""
    N = g_out.shape[0]
    if N == 0:
        return
    for arr in (grad_x, grad_y, grad_a, grad_b):
        arr.zero_()
    device = g_out.device
    wp.launch(
        _segmented_mul_overloads[(g_out.dtype, a.dtype)],
        dim=N,
        inputs=[g_out, a, idx, grad_x],
        device=device,
    )
    wp.launch(
        _segmented_mul_overloads[(g_out.dtype, b.dtype)],
        dim=N,
        inputs=[g_out, b, idx, grad_y],
        device=device,
    )
    ept = compute_ept(N, max(device.sm_count, 1), x.dtype in _VEC_TYPES)
    dim = (N + ept - 1) // ept
    wp.launch(
        _segmented_dot_overloads[x.dtype],
        dim=dim,
        inputs=[x, g_out, idx, grad_a, N, ept],
        device=device,
    )
    wp.launch(
        _segmented_dot_overloads[y.dtype],
        dim=dim,
        inputs=[y, g_out, idx, grad_b, N, ept],
        device=device,
    )


def _launch_segmented_axpby_double_backward(
    gg_gx: wp.array,
    gg_gy: wp.array,
    gg_ga: wp.array,
    gg_gb: wp.array,
    g_out: wp.array,
    a: wp.array,
    x: wp.array,
    b: wp.array,
    y: wp.array,
    idx: wp.array,
    grad_g_out: wp.array,
    grad_x_extra: wp.array,
    grad_y_extra: wp.array,
    grad_a_extra: wp.array,
    grad_b_extra: wp.array,
) -> None:
    N = g_out.shape[0]
    if N == 0:
        return
    for arr in (grad_g_out, grad_x_extra, grad_y_extra, grad_a_extra, grad_b_extra):
        arr.zero_()
    device = g_out.device
    ept = compute_ept(N, max(device.sm_count, 1), x.dtype in _VEC_TYPES)
    dim_rle = (N + ept - 1) // ept
    # combined grad_g_out
    wp.launch(
        _segmented_axpby_dbl_bwd_grad_out_overloads[g_out.dtype],
        dim=N,
        inputs=[gg_gx, a, gg_gy, b, gg_ga, x, gg_gb, y, idx, grad_g_out],
        device=device,
    )
    # grad_x_extra[i] = gg_ga[s]*g_out[i]
    wp.launch(
        _segmented_mul_overloads[(g_out.dtype, gg_ga.dtype)],
        dim=N,
        inputs=[g_out, gg_ga, idx, grad_x_extra],
        device=device,
    )
    # grad_y_extra[i] = gg_gb[s]*g_out[i]
    wp.launch(
        _segmented_mul_overloads[(g_out.dtype, gg_gb.dtype)],
        dim=N,
        inputs=[g_out, gg_gb, idx, grad_y_extra],
        device=device,
    )
    # grad_a_extra[s] = sum dot(gg_gx, g_out)
    wp.launch(
        _segmented_dot_overloads[g_out.dtype],
        dim=dim_rle,
        inputs=[gg_gx, g_out, idx, grad_a_extra, N, ept],
        device=device,
    )
    # grad_b_extra[s] = sum dot(gg_gy, g_out)
    wp.launch(
        _segmented_dot_overloads[g_out.dtype],
        dim=dim_rle,
        inputs=[gg_gy, g_out, idx, grad_b_extra, N, ept],
        device=device,
    )


# ---------------------------------------------------------------------------
# segmented_mean
# ---------------------------------------------------------------------------


def _launch_segmented_mean_backward(
    g_out: wp.array,
    counts: wp.array,
    idx: wp.array,
    grad_x: wp.array,
) -> None:
    """grad_x[i] = g_out[idx[i]] / count[idx[i]]."""
    N = grad_x.shape[0]
    if N == 0:
        return
    grad_x.zero_()
    key = g_out.dtype
    if key in _segmented_mean_backward_scalar_overloads:
        wp.launch(
            _segmented_mean_backward_scalar_overloads[key],
            dim=N,
            inputs=[g_out, counts, idx, grad_x],
            device=grad_x.device,
        )
    else:
        wp.launch(
            _segmented_mean_backward_vec_overloads[key],
            dim=N,
            inputs=[g_out, counts, idx, grad_x],
            device=grad_x.device,
        )


def _launch_segmented_mean_double_backward(
    gg_x: wp.array,
    counts: wp.array,
    idx: wp.array,
    grad_g_out: wp.array,
) -> None:
    """grad_g_out[s] = sum_i gg_x[i] / count[s]  (mean of gg_x per segment)."""
    M = grad_g_out.shape[0]
    if gg_x.shape[0] == 0:
        return
    grad_g_out.zero_()
    device = gg_x.device
    sums = wp.zeros(M, dtype=gg_x.dtype, device=device)
    _launch_sum(gg_x, idx, sums)
    if gg_x.dtype in _VEC_TYPES:
        wp.launch(
            _segmented_vec_div_by_count_overloads[gg_x.dtype],
            dim=M,
            inputs=[sums, counts, grad_g_out],
            device=device,
        )
    else:
        # _segment_div_overloads: result[i] = numerator[i] / int(denominator[i])
        wp.launch(
            _segment_div_overloads[gg_x.dtype],
            dim=M,
            inputs=[sums, counts, grad_g_out],
            device=device,
        )


# ---------------------------------------------------------------------------
# segmented_rms_norm
# ---------------------------------------------------------------------------


def _launch_segmented_rms_norm_forward_precompute(
    x: wp.array,
    idx: wp.array,
    sum_sq: wp.array,
    counts: wp.array,
    out: wp.array,
    inv_norm: wp.array,
) -> None:
    """Extended forward that also writes inv_norm = 1/(out*count) for the backward."""
    N = x.shape[0]
    M = out.shape[0]
    if N == 0:
        return
    device = x.device
    scalar_dtype = _VEC_TO_SCALAR[x.dtype]
    segmented_dot(x, x, idx, sum_sq)
    segmented_count(idx, counts)
    wp.launch(
        _segmented_rms_norm_finalize_and_save_overloads[scalar_dtype],
        dim=M,
        inputs=[sum_sq, counts, out, inv_norm],
        device=device,
    )


def _launch_segmented_rms_norm_backward(
    g_out: wp.array,
    x: wp.array,
    inv_norm: wp.array,
    idx: wp.array,
    grad_x: wp.array,
) -> None:
    """grad_x[i] = g_out[idx[i]] * x[i] * inv_norm[idx[i]]."""
    N = grad_x.shape[0]
    if N == 0:
        return
    grad_x.zero_()
    wp.launch(
        _segmented_rms_norm_backward_overloads[x.dtype],
        dim=N,
        inputs=[g_out, x, inv_norm, idx, grad_x],
        device=grad_x.device,
    )


def _launch_segmented_rms_norm_double_backward(
    gg_x: wp.array,
    x: wp.array,
    g_out: wp.array,
    inv_norm: wp.array,
    counts: wp.array,
    idx: wp.array,
    M: int,
    grad_x_extra: wp.array,
    grad_g_out_extra: wp.array,
) -> None:
    """Double-backward for rms_norm: grad_x_extra and grad_g_out_extra."""
    N = x.shape[0]
    if N == 0:
        return
    grad_x_extra.zero_()
    grad_g_out_extra.zero_()
    device = x.device
    scalar_dtype = _VEC_TO_SCALAR[x.dtype]
    # Step 1: inner[s] = sum dot(gg_x[i], x[i])
    inner = wp.zeros(M, dtype=scalar_dtype, device=device)
    segmented_dot(gg_x, x, idx, inner)
    # Step 2: grad_g_out[s] = inner[s] * inv_norm[s]
    wp.launch(
        _segmented_rms_norm_dbl_bwd_grad_g_out_overloads[scalar_dtype],
        dim=M,
        inputs=[inner, inv_norm, grad_g_out_extra],
        device=device,
    )
    # Step 3: element-wise grad_x_extra
    wp.launch(
        _segmented_rms_norm_dbl_bwd_grad_x_overloads[x.dtype],
        dim=N,
        inputs=[gg_x, x, g_out, inv_norm, counts, inner, idx, grad_x_extra],
        device=device,
    )


# ---------------------------------------------------------------------------
# segmented_max_norm
# ---------------------------------------------------------------------------


def _launch_segmented_max_norm_forward_precompute(
    x: wp.array,
    idx: wp.array,
    out: wp.array,
    argmax_idx: wp.array,
) -> None:
    """Run forward then find argmax element per segment."""
    from nvalchemiops.segment_ops import segmented_max_norm as _fwd_max_norm

    N = x.shape[0]
    M = out.shape[0]
    if N == 0:
        return
    device = x.device
    # Initialize argmax_idx to -1 so that the first valid write wins via atomic_max
    wp.launch(_fill_int32_kernel, dim=M, inputs=[argmax_idx, -1], device=device)
    _fwd_max_norm(x, idx, out)
    ept = compute_ept(N, max(device.sm_count, 1), True)
    dim = (N + ept - 1) // ept
    wp.launch(
        _segmented_max_norm_argmax_overloads[x.dtype],
        dim=dim,
        inputs=[x, idx, out, argmax_idx, N, ept],
        device=device,
    )


def _launch_segmented_max_norm_backward(
    g_out: wp.array,
    x: wp.array,
    argmax_idx: wp.array,
    idx: wp.array,
    grad_x: wp.array,
) -> None:
    """Subgradient: only argmax element gets nonzero gradient."""
    N = grad_x.shape[0]
    if N == 0:
        return
    grad_x.zero_()
    wp.launch(
        _segmented_max_norm_backward_overloads[x.dtype],
        dim=N,
        inputs=[g_out, x, argmax_idx, idx, grad_x],
        device=grad_x.device,
    )


def _launch_segmented_max_norm_double_backward(
    gg_gx: wp.array,
    g_out: wp.array,
    x: wp.array,
    argmax_idx: wp.array,
    idx: wp.array,
    grad_x_extra: wp.array,
    grad_g_out: wp.array,
) -> None:
    """Tangent-plane double-backward for max_norm (only at argmax element)."""
    N = x.shape[0]
    if N == 0:
        return
    grad_x_extra.zero_()
    grad_g_out.zero_()
    wp.launch(
        _segmented_max_norm_double_backward_overloads[x.dtype],
        dim=N,
        inputs=[gg_gx, g_out, x, argmax_idx, idx, grad_x_extra, grad_g_out],
        device=grad_x_extra.device,
    )


# ---------------------------------------------------------------------------
# segmented_matvec
# ---------------------------------------------------------------------------


def _launch_segmented_matvec_backward(
    g_out: wp.array,
    v: wp.array,
    m: wp.array,
    idx: wp.array,
    grad_v: wp.array,
    grad_M: wp.array,
) -> None:
    """grad_v[i]=M[s]@g_out[i]; grad_M[s]=sum outer(v[i], g_out[i])."""
    N = v.shape[0]
    if N == 0:
        return
    grad_v.zero_()
    grad_M.zero_()
    device = v.device
    wp.launch(
        _segmented_matvec_backward_v_overloads[(v.dtype, m.dtype)],
        dim=N,
        inputs=[g_out, m, idx, grad_v],
        device=device,
    )
    ept = compute_ept(N, max(device.sm_count, 1), True)
    dim = (N + ept - 1) // ept
    wp.launch(
        _segmented_matvec_backward_M_overloads[(v.dtype, m.dtype)],
        dim=dim,
        inputs=[g_out, v, idx, grad_M, N, ept],
        device=device,
    )


def _launch_segmented_matvec_double_backward(
    gg_gv: wp.array,
    gg_gM: wp.array,
    g_out: wp.array,
    v: wp.array,
    m: wp.array,
    idx: wp.array,
    grad_g_out: wp.array,
    grad_v_extra: wp.array,
    grad_M_extra: wp.array,
) -> None:
    """Double-backward for segmented_matvec."""
    N = v.shape[0]
    if N == 0:
        return
    grad_g_out.zero_()
    grad_v_extra.zero_()
    grad_M_extra.zero_()
    device = v.device
    ept = compute_ept(N, max(device.sm_count, 1), True)
    dim_rle = (N + ept - 1) // ept
    # grad_g_out[i] = M[s]^T @ gg_gv[i]  +  gg_gM[s]^T @ v[i]
    # First term: matvec fwd (v=gg_gv, M=m_fwd): out[i] = m[s]^T @ gg_gv[i]
    tmp = wp.zeros(N, dtype=v.dtype, device=device)
    wp.launch(
        _segmented_mul_overloads[(v.dtype, m.dtype)],
        dim=N,
        inputs=[gg_gv, m, idx, grad_g_out],
        device=device,
    )
    # Second term: matvec fwd (v=v_fwd, M=gg_gM): out[i] = gg_gM[s]^T @ v[i]
    wp.launch(
        _segmented_mul_overloads[(v.dtype, m.dtype)],
        dim=N,
        inputs=[v, gg_gM, idx, tmp],
        device=device,
    )
    _add_arrays_inplace(grad_g_out, tmp)
    # grad_v_extra[i] = gg_gM[s] @ g_out[i]   (non-transposed matvec)
    wp.launch(
        _segmented_matvec_backward_v_overloads[(v.dtype, m.dtype)],
        dim=N,
        inputs=[g_out, gg_gM, idx, grad_v_extra],
        device=device,
    )
    # grad_M_extra[s] = sum outer(gg_gv[i], g_out[i])
    wp.launch(
        _segmented_matvec_backward_M_overloads[(v.dtype, m.dtype)],
        dim=dim_rle,
        inputs=[gg_gv, g_out, idx, grad_M_extra, N, ept],
        device=device,
    )


# ---------------------------------------------------------------------------
# segment_div
# ---------------------------------------------------------------------------


def _launch_segment_div_backward(
    g_result: wp.array,
    denominator: wp.array,
    grad_numerator: wp.array,
) -> None:
    """grad_numerator[i] = g_result[i] / denominator[i]."""
    N = g_result.shape[0]
    if N == 0:
        return
    grad_numerator.zero_()
    wp.launch(
        _segment_div_overloads[g_result.dtype],
        dim=N,
        inputs=[g_result, denominator, grad_numerator],
        device=g_result.device,
    )


def _launch_segment_div_double_backward(
    gg_numerator: wp.array,
    denominator: wp.array,
    grad_g_result: wp.array,
) -> None:
    """grad_g_result[i] = gg_numerator[i] / denominator[i]."""
    _launch_segment_div_backward(gg_numerator, denominator, grad_g_result)


# ===========================================================================
# Utility kernels and helpers used internally above
# ===========================================================================


@wp.kernel(enable_backward=False)
def _fill_int32_kernel(
    out: wp.array(dtype=wp.int32),
    value: wp.int32,
):
    out[wp.tid()] = value


@wp.kernel(enable_backward=False)
def _add_arrays_inplace_float32_kernel(
    a: wp.array(dtype=wp.float32),
    b: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    a[i] = a[i] + b[i]


@wp.kernel(enable_backward=False)
def _add_arrays_inplace_float64_kernel(
    a: wp.array(dtype=wp.float64),
    b: wp.array(dtype=wp.float64),
):
    i = wp.tid()
    a[i] = a[i] + b[i]


@wp.kernel(enable_backward=False)
def _add_arrays_inplace_vec3f_kernel(
    a: wp.array(dtype=wp.vec3f),
    b: wp.array(dtype=wp.vec3f),
):
    i = wp.tid()
    a[i] = a[i] + b[i]


@wp.kernel(enable_backward=False)
def _add_arrays_inplace_vec3d_kernel(
    a: wp.array(dtype=wp.vec3d),
    b: wp.array(dtype=wp.vec3d),
):
    i = wp.tid()
    a[i] = a[i] + b[i]


_ADD_INPLACE = {
    wp.float32: _add_arrays_inplace_float32_kernel,
    wp.float64: _add_arrays_inplace_float64_kernel,
    wp.vec3f: _add_arrays_inplace_vec3f_kernel,
    wp.vec3d: _add_arrays_inplace_vec3d_kernel,
}


def _add_arrays_inplace(a: wp.array, b: wp.array) -> None:
    """a[i] += b[i]  for matching Warp arrays."""
    if a.shape[0] == 0:
        return
    wp.launch(_ADD_INPLACE[a.dtype], dim=a.shape[0], inputs=[a, b], device=a.device)


@wp.kernel(enable_backward=False)
def _scale_add_float32_kernel(
    dst: wp.array(dtype=wp.float32),
    src: wp.array(dtype=wp.float32),
    scale: wp.float32,
):
    i = wp.tid()
    dst[i] = dst[i] + src[i] * scale


@wp.kernel(enable_backward=False)
def _scale_add_float64_kernel(
    dst: wp.array(dtype=wp.float64),
    src: wp.array(dtype=wp.float64),
    scale: wp.float64,
):
    i = wp.tid()
    dst[i] = dst[i] + src[i] * scale


_SCALE_ADD = {
    wp.float32: _scale_add_float32_kernel,
    wp.float64: _scale_add_float64_kernel,
}


def _scale_and_add_inplace(dst: wp.array, src: wp.array, type_, factor: float) -> None:
    """dst[i] += src[i] * factor.  type_ is wp.float32 or wp.float64."""
    if dst.shape[0] == 0:
        return
    # Warp converts plain Python float to the kernel's declared scalar type automatically.
    wp.launch(
        _SCALE_ADD[type_],
        dim=dst.shape[0],
        inputs=[dst, src, float(factor)],
        device=dst.device,
    )
