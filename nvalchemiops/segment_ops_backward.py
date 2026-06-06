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

Output-buffer zeroing policy
----------------------------
Backward launchers used to zero every output buffer unconditionally before
launching the kernel.  That was correct but expensive: for the simplest
gather-shaped backwards (e.g. ``segmented_sum``'s first-order ``grad_x[i] =
g_out[idx[i]]``), the pre-zero was a full-buffer memset on the same bytes the
kernel was about to overwrite — a measurable ~50% slowdown vs torch on
memory-bound paths.

The current policy is: **a launcher zeros an output only when the kernel
won't otherwise leave it fully defined.**  Concretely:

- **Required** — keep the pre-zero when the kernel uses ``wp.atomic_add`` to
  scatter into the buffer (the RLE-EPT segmented_sum / segmented_dot paths),
  when the kernel writes the buffer only at a subset of indices
  (``segmented_max_norm`` writes only at argmax positions), or when the
  kernel may run with ``dim < buffer.shape[0]`` and downstream code reads the
  whole buffer.
- **Redundant — do NOT add** — when the kernel writes every element of the
  output exactly once via pure assignment (``arr[i] = ...`` over ``dim=N``,
  ``wp.copy``, or a fused element-wise overload over the full output).

- **Empty-input handling**: when ``N == 0`` the kernel is skipped, but
  callers may pass re-used output buffers whose contents would otherwise be
  stale.  Launchers must still zero on the empty-input early-return path.
  Several launchers do this by zeroing inside the ``if N == 0: return``
  branch; the three fused-double-backward ``grad_g_out`` outputs
  (``segmented_add_double_backward``,
  ``segmented_axpy_double_backward``,
  ``segmented_matvec_double_backward``) keep an unconditional
  pre-guard zero, pinned by
  ``test_empty_fused_double_backward_zeros_grad_g_out``.

When in doubt, follow the kernel: if every element of ``out`` is written by a
``dim == out.shape[0]`` pure-assignment kernel, the pre-zero is wasted memory
bandwidth.  Tests in ``test/test_segment_ops_backward.py`` cover the
``N == 0`` and atomic-accumulation paths, so removing or adding a zero in
the wrong place will fail visibly rather than silently.
"""

from __future__ import annotations

from typing import Any

import warp as wp

from nvalchemiops.segment_ops import (
    _ALL_SCALAR_TYPES,
    _ALL_SUPPORTED_TYPES,
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
    """Run the forward ``segmented_sum`` overloads (scatter-reduce).

    Shared dispatcher for any backward path that needs a segmented sum
    reduction (e.g. ``segmented_sum`` double-backward, ``segmented_broadcast``
    backward, ``axpy``/``axpby`` grad-a reductions).

    Parameters
    ----------
    x : wp.array, shape ``(N,)``, dtype float32 / float64 / vec3f / vec3d
        Per-element inputs to reduce.
    idx : wp.array, shape ``(N,)``, dtype int32
        Sorted segment indices in ``[0, M)``.
    out : wp.array, shape ``(M,)``, dtype matches ``x``
        OUTPUT: per-segment reduction.

    Notes
    -----
    Two-mode dispatch: a tile-based total-sum kernel when ``M == 1`` and
    ``N >= 8192``, otherwise RLE-EPT segmented_sum with ``dim=ceil(N/EPT)``
    and ``atomic_add`` accumulation per segment.  ``out`` is zeroed here
    because the ``atomic_add`` reduction needs a zero baseline (per the
    output-buffer policy in the module docstring).

    See Also
    --------
    _launch_broadcast : Inverse op (per-segment scalar → per-element gather).
    """
    out.zero_()
    N = x.shape[0]
    if N == 0:
        return
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
    """Run the forward ``segmented_broadcast`` overloads (pure gather).

    Gathers per-segment ``values`` into per-element ``out`` via ``idx``.  Shared
    dispatcher for any backward path that needs a scatter→gather (e.g.
    ``segmented_sum`` backward, ``segmented_broadcast`` double-backward).

    Parameters
    ----------
    values : wp.array, shape ``(M,)``, dtype float32 / float64 / vec3f / vec3d
        Per-segment values to broadcast.
    idx : wp.array, shape ``(N,)``, dtype int32
        Segment indices; need not be sorted.
    out : wp.array, shape ``(N,)``, dtype matches ``values``
        OUTPUT: ``out[i] = values[idx[i]]``.  Every element is written by the
        gather kernel, so no pre-zero is performed (saves a full-buffer memset
        on what is otherwise a bandwidth-bound op).  When ``N == 0`` the
        output is empty and no write is needed.

    Notes
    -----
    One element-wise launch with ``dim=N``; no reduction or atomics.

    See Also
    --------
    _launch_sum : Inverse op (per-element scatter-sum → per-segment).
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
    """For each i where length(x[i]) == max_norms[idx[i]], record i via atomic_max.

    Requires ``argmax_idx`` to be pre-filled with ``-1`` (or any value smaller
    than every valid index) — the ``atomic_max`` only keeps the *largest* index
    it sees, so a buffer left at zero or stuffed with stale values from a
    previous call will silently retain the wrong index when the true argmax has
    a smaller ``i``.  ``segmented_max_norm_forward_precompute`` handles
    the initialization; callers must not invoke this kernel directly without
    that pre-fill.
    """
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
# Section 9b – axpy double-backward grad_g_out (fused element-wise)
# ===========================================================================
# grad_g_out[i] = gg_gy_in[i] + gg_gx[i]*a[s] + gg_ga[s]*x[i]


@wp.kernel(enable_backward=False)
def _segmented_axpy_dbl_bwd_grad_out_scalar_kernel(
    gg_gy_in: wp.array(dtype=Any),
    gg_gx: wp.array(dtype=Any),
    a: wp.array(dtype=Any),
    gg_ga: wp.array(dtype=Any),
    x: wp.array(dtype=Any),
    idx: wp.array(dtype=wp.int32),
    grad_g_out: wp.array(dtype=Any),
):
    i = wp.tid()
    s = idx[i]
    grad_g_out[i] = gg_gy_in[i] + gg_gx[i] * a[s] + gg_ga[s] * x[i]


@wp.kernel(enable_backward=False)
def _segmented_axpy_dbl_bwd_grad_out_vec_scalar_kernel(
    gg_gy_in: wp.array(dtype=Any),
    gg_gx: wp.array(dtype=Any),
    a: wp.array(dtype=Any),
    gg_ga: wp.array(dtype=Any),
    x: wp.array(dtype=Any),
    idx: wp.array(dtype=wp.int32),
    grad_g_out: wp.array(dtype=Any),
):
    i = wp.tid()
    s = idx[i]
    # gg_gy_in/gg_gx/x are vec3; a/gg_ga are scalar
    grad_g_out[i] = gg_gy_in[i] + a[s] * gg_gx[i] + gg_ga[s] * x[i]


_segmented_axpy_dbl_bwd_grad_out_overloads = register_overloads(
    _segmented_axpy_dbl_bwd_grad_out_scalar_kernel,
    lambda t: [wp.array(dtype=t)] * 5 + [wp.array(dtype=wp.int32), wp.array(dtype=t)],
    dtypes=_ALL_SCALAR_TYPES,
)
_segmented_axpy_dbl_bwd_grad_out_overloads.update(
    register_overloads(
        _segmented_axpy_dbl_bwd_grad_out_vec_scalar_kernel,
        lambda v, s: [
            wp.array(dtype=v),
            wp.array(dtype=v),
            wp.array(dtype=s),
            wp.array(dtype=s),
            wp.array(dtype=v),
            wp.array(dtype=wp.int32),
            wp.array(dtype=v),
        ],
        dtype_pairs=_ALL_VEC_SCALAR_PAIRS,
    )
)


# ===========================================================================
# Section 9c – matvec double-backward grad_g_out (fused two-term matvec)
# ===========================================================================
# grad_g_out[i] = m[s]^T @ gg_gv[i] + gg_gM[s]^T @ v[i]
# (warp's wp.mul(vec, mat) computes v^T @ M = M^T @ v)


@wp.kernel(enable_backward=False)
def _segmented_matvec_dbl_bwd_grad_out_kernel(
    gg_gv: wp.array(dtype=Any),
    m: wp.array(dtype=Any),
    v: wp.array(dtype=Any),
    gg_gM: wp.array(dtype=Any),
    idx: wp.array(dtype=wp.int32),
    grad_g_out: wp.array(dtype=Any),
):
    i = wp.tid()
    s = idx[i]
    grad_g_out[i] = wp.mul(gg_gv[i], m[s]) + wp.mul(v[i], gg_gM[s])


_segmented_matvec_dbl_bwd_grad_out_overloads = register_overloads(
    _segmented_matvec_dbl_bwd_grad_out_kernel,
    lambda v, m: [
        wp.array(dtype=v),
        wp.array(dtype=m),
        wp.array(dtype=v),
        wp.array(dtype=m),
        wp.array(dtype=wp.int32),
        wp.array(dtype=v),
    ],
    dtype_pairs=_VEC_MAT_PAIRS,
    key_fn=lambda v, m: (v, m),
)


# ===========================================================================
# Section 10 – add double-backward (fused broadcast + add)
# ===========================================================================
# Forward:        out[i] = x[i] + y[idx[i]]
# Backward:       grad_x[i] = g_out[i];  grad_y[s] = sum_i g_out[i]
# Double-backward: grad_g_out[i] = gg_x[i] + gg_y[idx[i]]
#
# One element-wise kernel writes the full result directly into grad_g_out,
# avoiding a tmp buffer + separate "broadcast then add-inplace" pair.


@wp.kernel(enable_backward=False)
def _segmented_add_dbl_bwd_grad_out_kernel(
    gg_x: wp.array(dtype=Any),
    gg_y: wp.array(dtype=Any),
    idx: wp.array(dtype=wp.int32),
    grad_g_out: wp.array(dtype=Any),
):
    """grad_g_out[i] = gg_x[i] + gg_y[idx[i]]."""
    i = wp.tid()
    grad_g_out[i] = gg_x[i] + gg_y[idx[i]]


_segmented_add_dbl_bwd_grad_out_overloads = register_overloads(
    _segmented_add_dbl_bwd_grad_out_kernel,
    lambda t: [wp.array(dtype=t)] * 2 + [wp.array(dtype=wp.int32), wp.array(dtype=t)],
    dtypes=_ALL_SUPPORTED_TYPES,
)


# ===========================================================================
# Internal launch functions
# All functions below are the contracts consumed by PR 2 (Torch) and PR 3 (JAX).
#
# Output buffer convention (see module docstring for the full rationale):
#   Each launcher zeros only those outputs whose kernel doesn't fully define
#   them — outputs accumulated via wp.atomic_add, written at a subset of
#   indices, or returned on the N == 0 fast path.  Outputs fully written via
#   dim=N pure-assignment kernels (or wp.copy) are NOT pre-zeroed: that
#   pre-zero is a full-buffer memset over bytes the kernel is about to
#   overwrite.  Adding a zero where it isn't needed is a real perf cost; do
#   not add one without verifying the kernel doesn't already write every
#   element.
# ===========================================================================

# ---------------------------------------------------------------------------
# segmented_sum
# ---------------------------------------------------------------------------


def segmented_sum_backward(
    g_out: wp.array,
    idx: wp.array,
    grad_x: wp.array,
) -> None:
    """First-order backward of segmented_sum: ``grad_x[i] = g_out[idx[i]]``.

    Parameters
    ----------
    g_out : wp.array, shape ``(M,)``, dtype float32 / float64 / vec3f / vec3d
        Upstream gradient of the per-segment forward output.
    idx : wp.array, shape ``(N,)``, dtype int32
        Sorted segment indices in ``[0, M)``.
    grad_x : wp.array, shape ``(N,)``, dtype matches ``g_out``
        OUTPUT: per-element gradient.

    Notes
    -----
    A pure gather (no atomics, no reduction).  Delegates to
    :func:`_launch_broadcast` which dispatches one element-wise kernel
    with ``dim=N``.

    See Also
    --------
    segmented_sum_double_backward : Symmetric scatter-sum.
    """
    _launch_broadcast(g_out, idx, grad_x)


def segmented_sum_double_backward(
    gg_x: wp.array,
    idx: wp.array,
    M: int,
    grad_g_out: wp.array,
) -> None:
    """Double-backward of segmented_sum: ``grad_g_out[s] = sum_i gg_x[i]``.

    Parameters
    ----------
    gg_x : wp.array, shape ``(N,)``, dtype float32 / float64 / vec3f / vec3d
        Cotangent of the first-order grad_x.
    idx : wp.array, shape ``(N,)``, dtype int32
        Sorted segment indices in ``[0, M)``.
    M : int
        Number of segments.  Unused; retained for API symmetry with
        ``grad_g_out.shape[0]``.
    grad_g_out : wp.array, shape ``(M,)``, dtype matches ``gg_x``
        OUTPUT: per-segment second-order adjoint.

    Notes
    -----
    A scatter-sum; same kernel as the forward.  Delegates to
    :func:`_launch_sum` (RLE-EPT segmented reduction with ``atomic_add``).

    See Also
    --------
    segmented_sum_backward : First-order backward (gather).
    """
    _launch_sum(gg_x, idx, grad_g_out)


# ---------------------------------------------------------------------------
# segmented_broadcast
# ---------------------------------------------------------------------------


def segmented_broadcast_backward(
    g_out: wp.array,
    idx: wp.array,
    M: int,
    grad_values: wp.array,
) -> None:
    """First-order backward of segmented_broadcast: ``grad_values[s] = sum_i g_out[i]``.

    Parameters
    ----------
    g_out : wp.array, shape ``(N,)``, dtype float32 / float64 / vec3f / vec3d
        Upstream gradient of the per-element forward output.
    idx : wp.array, shape ``(N,)``, dtype int32
        Sorted segment indices in ``[0, M)``.
    M : int
        Number of segments.  Unused; retained for API symmetry.
    grad_values : wp.array, shape ``(M,)``, dtype matches ``g_out``
        OUTPUT: per-segment gradient.

    Notes
    -----
    Delegates to :func:`_launch_sum`: RLE-EPT segmented reduction with
    ``dim=ceil(N/EPT)`` and ``atomic_add`` per segment.

    See Also
    --------
    segmented_broadcast_double_backward : Symmetric gather.
    """
    _launch_sum(g_out, idx, grad_values)


def segmented_broadcast_double_backward(
    gg_values: wp.array,
    idx: wp.array,
    grad_g_out: wp.array,
) -> None:
    """Double-backward of segmented_broadcast: ``grad_g_out[i] = gg_values[idx[i]]``.

    Parameters
    ----------
    gg_values : wp.array, shape ``(M,)``, dtype float32 / float64 / vec3f / vec3d
        Cotangent of the first-order grad_values.
    idx : wp.array, shape ``(N,)``, dtype int32
        Sorted segment indices in ``[0, M)``.
    grad_g_out : wp.array, shape ``(N,)``, dtype matches ``gg_values``
        OUTPUT: per-element second-order adjoint.

    Notes
    -----
    A pure gather; delegates to :func:`_launch_broadcast`.

    See Also
    --------
    segmented_broadcast_backward : First-order backward (scatter-sum).
    """
    _launch_broadcast(gg_values, idx, grad_g_out)


# ---------------------------------------------------------------------------
# segmented_component_sum
# ---------------------------------------------------------------------------


def segmented_component_sum_backward(
    g_out: wp.array,
    idx: wp.array,
    grad_x: wp.array,
) -> None:
    """First-order backward of segmented_component_sum.

    Broadcasts each per-segment scalar ``g_out[s]`` into the three components
    of ``grad_x[i]``: ``grad_x[i] = vec3(g_out[s], g_out[s], g_out[s])``.

    Parameters
    ----------
    g_out : wp.array, shape ``(M,)``, dtype float32 / float64
        Upstream scalar gradient per segment.
    idx : wp.array, shape ``(N,)``, dtype int32
        Sorted segment indices in ``[0, M)``.
    grad_x : wp.array, shape ``(N,)``, dtype vec3f / vec3d (matches ``g_out`` precision)
        OUTPUT: per-element vec3 gradient.

    Notes
    -----
    One element-wise launch with ``dim=N``; no atomics.

    See Also
    --------
    segmented_component_sum_double_backward : Symmetric reduction.
    """
    N = grad_x.shape[0]
    if N == 0:
        return
    wp.launch(
        _segmented_component_sum_backward_overloads[grad_x.dtype],
        dim=N,
        inputs=[g_out, idx, grad_x],
        device=grad_x.device,
    )


def segmented_component_sum_double_backward(
    gg_x: wp.array,
    idx: wp.array,
    M: int,
    grad_g_out: wp.array,
) -> None:
    """Double-backward of segmented_component_sum (mirrors the forward).

    ``grad_g_out[s] = sum_{i:idx[i]=s} (gg_x[i][0] + gg_x[i][1] + gg_x[i][2])``.

    Parameters
    ----------
    gg_x : wp.array, shape ``(N,)``, dtype vec3f / vec3d
        Cotangent of the first-order grad_x.
    idx : wp.array, shape ``(N,)``, dtype int32
        Sorted segment indices in ``[0, M)``.
    M : int
        Number of segments.  Unused; retained for API symmetry.
    grad_g_out : wp.array, shape ``(M,)``, dtype float32 / float64
        OUTPUT: per-segment second-order adjoint.

    Notes
    -----
    Reuses the forward component-sum overloads with ``dim=ceil(N/EPT)`` RLE
    and ``atomic_add`` per segment.

    See Also
    --------
    segmented_component_sum_backward : First-order backward (broadcast).
    """
    grad_g_out.zero_()
    N = gg_x.shape[0]
    if N == 0:
        return
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


def segmented_add_backward(
    g_out: wp.array,
    idx: wp.array,
    M: int,
    grad_x: wp.array,
    grad_y: wp.array,
) -> None:
    """First-order backward of segmented_add (forward: ``out[i] = x[i] + y[idx[i]]``).

    ``grad_x[i] = g_out[i]`` (identity copy); ``grad_y[s] = sum_i g_out[i]``
    (scatter-sum per segment).

    Parameters
    ----------
    g_out : wp.array, shape ``(N,)``, dtype float32 / float64 / vec3f / vec3d
        Upstream gradient of the per-element forward output.
    idx : wp.array, shape ``(N,)``, dtype int32
        Sorted segment indices in ``[0, M)``.
    M : int
        Number of segments.  Unused; retained for API symmetry.
    grad_x : wp.array, shape ``(N,)``, dtype matches ``g_out``
        OUTPUT: per-element gradient (copy of ``g_out``).
    grad_y : wp.array, shape ``(M,)``, dtype matches ``g_out``
        OUTPUT: per-segment scatter-sum of ``g_out``.

    Notes
    -----
    Two launches: a ``wp.copy`` (memcpy) for ``grad_x`` plus a segmented
    reduction via :func:`_launch_sum` (RLE-EPT, ``atomic_add``) for ``grad_y``.
    Mixed-type variants (vec+scalar or scalar+vec) should call type-specific
    helpers; this launcher handles the matching-dtype case.

    See Also
    --------
    segmented_add_double_backward : Fused gather-and-add second order.
    """
    if g_out.shape[0] == 0:
        grad_y.zero_()
        return
    wp.copy(grad_x, g_out)
    _launch_sum(g_out, idx, grad_y)


def segmented_add_double_backward(
    gg_x: wp.array,
    gg_y: wp.array,
    idx: wp.array,
    grad_g_out: wp.array,
) -> None:
    """Double-backward of segmented_add (fused single-kernel implementation).

    ``grad_g_out[i] = gg_x[i] + gg_y[idx[i]]`` — combines a gather of
    ``gg_y`` and the per-element ``gg_x`` into one element-wise kernel,
    avoiding a temp buffer + ``_add_arrays_inplace`` pair.

    Parameters
    ----------
    gg_x : wp.array, shape ``(N,)``, dtype float32 / float64 / vec3f / vec3d
        Cotangent of the first-order grad_x.
    gg_y : wp.array, shape ``(M,)``, dtype matches ``gg_x``
        Cotangent of the first-order grad_y.
    idx : wp.array, shape ``(N,)``, dtype int32
        Sorted segment indices in ``[0, M)``.
    grad_g_out : wp.array, shape ``(N,)``, dtype matches ``gg_x``
        OUTPUT: per-element second-order adjoint.

    Notes
    -----
    One fused element-wise launch with ``dim=N``; no atomics, no scratch.
    ``grad_g_out`` is zeroed unconditionally before the empty-input guard so
    callers that pass a re-used buffer never observe stale values on
    ``N == 0`` segments.

    See Also
    --------
    segmented_add_backward : First-order backward.
    """
    grad_g_out.zero_()
    N = gg_x.shape[0]
    if N == 0:
        return
    wp.launch(
        _segmented_add_dbl_bwd_grad_out_overloads[gg_x.dtype],
        dim=N,
        inputs=[gg_x, gg_y, idx, grad_g_out],
        device=gg_x.device,
    )


# ---------------------------------------------------------------------------
# segmented_mul
# ---------------------------------------------------------------------------


def segmented_mul_backward(
    g_out: wp.array,
    x: wp.array,
    y: wp.array,
    idx: wp.array,
    M: int,
    grad_x: wp.array,
    grad_y: wp.array,
) -> None:
    """First-order backward of segmented_mul (``out[i] = x[i] * y[idx[i]]``).

    ``grad_x[i] = g_out[i] * y[s]`` (per-element scaled broadcast);
    ``grad_y[s] = sum_i dot(g_out[i], x[i])`` (per-segment reduction).

    Parameters
    ----------
    g_out : wp.array, shape ``(N,)``, dtype float32 / float64 / vec3f / vec3d
        Upstream gradient of the forward output.
    x : wp.array, shape ``(N,)``, dtype matches ``g_out``
        Per-element operand from the forward.
    y : wp.array, shape ``(M,)``, dtype float32 / float64
        Per-segment scalar operand from the forward.
    idx : wp.array, shape ``(N,)``, dtype int32
        Sorted segment indices in ``[0, M)``.
    M : int
        Number of segments.  Unused; retained for API symmetry.
    grad_x : wp.array, shape ``(N,)``, dtype matches ``g_out``
        OUTPUT: per-element gradient.
    grad_y : wp.array, shape ``(M,)``, dtype matches ``y``
        OUTPUT: per-segment gradient.

    Notes
    -----
    Two launches: one element-wise mul (``dim=N``) for ``grad_x`` and one
    RLE-EPT segmented_dot reduction (``dim=ceil(N/EPT)``) with ``atomic_add``
    for ``grad_y``.

    See Also
    --------
    segmented_mul_double_backward : Second-order backward.
    """
    # grad_x[i] = g_out[i] * y[s]
    N = g_out.shape[0]
    if N == 0:
        grad_x.zero_()
        grad_y.zero_()
        return
    # grad_y accumulates via atomic_add in segmented_dot, so it must start zeroed.
    grad_y.zero_()
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


def segmented_mul_double_backward(
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
    """Double-backward of segmented_mul.

    Given cotangents ``(gg_gx, gg_gy)`` of the first-order grads, produces::

        grad_g_out[i]    = gg_gx[i]*y[s] + gg_gy[s]*x[i]
        grad_x_extra[i]  = gg_gy[s] * g_out[i]
        grad_y_extra[s]  = sum_i dot(gg_gx[i], g_out[i])

    Parameters
    ----------
    gg_gx : wp.array, shape ``(N,)``, dtype matches ``g_out``
        Cotangent of the first-order grad_x.
    gg_gy : wp.array, shape ``(M,)``, dtype matches ``y``
        Cotangent of the first-order grad_y.
    g_out : wp.array, shape ``(N,)``, dtype float32 / float64 / vec3f / vec3d
        Upstream gradient of the forward output.
    x : wp.array, shape ``(N,)``, dtype matches ``g_out``
        Per-element operand from the forward.
    y : wp.array, shape ``(M,)``, dtype float32 / float64
        Per-segment scalar operand from the forward.
    idx : wp.array, shape ``(N,)``, dtype int32
        Sorted segment indices in ``[0, M)``.
    grad_g_out : wp.array, shape ``(N,)``, dtype matches ``g_out``
        OUTPUT: second-order adjoint w.r.t. ``g_out``.
    grad_x_extra : wp.array, shape ``(N,)``, dtype matches ``g_out``
        OUTPUT: extra contribution to the second-order adjoint of ``x``.
    grad_y_extra : wp.array, shape ``(M,)``, dtype matches ``y``
        OUTPUT: extra contribution to the second-order adjoint of ``y``.

    Notes
    -----
    Three launches: one fused element-wise kernel for ``grad_g_out``
    (``dim=N``), one element-wise mul for ``grad_x_extra`` (``dim=N``), one
    RLE-EPT segmented_dot reduction for ``grad_y_extra``
    (``dim=ceil(N/EPT)`` with ``atomic_add``).

    See Also
    --------
    segmented_mul_backward : First-order backward.
    """
    N = g_out.shape[0]
    if N == 0:
        grad_g_out.zero_()
        grad_x_extra.zero_()
        grad_y_extra.zero_()
        return
    # grad_y_extra accumulates via atomic_add in segmented_dot, so it must start zeroed.
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


def segmented_dot_backward(
    g_out: wp.array,
    x: wp.array,
    y: wp.array,
    idx: wp.array,
    grad_x: wp.array,
    grad_y: wp.array,
) -> None:
    """First-order backward of segmented_dot (forward: ``out[s] = sum_i dot(x[i], y[i])``).

    ``grad_x[i] = g_out[s] * y[i]``; ``grad_y[i] = g_out[s] * x[i]``.

    Parameters
    ----------
    g_out : wp.array, shape ``(M,)``, dtype float32 / float64
        Upstream gradient of the per-segment dot output.
    x, y : wp.array, shape ``(N,)``, dtype float32 / float64 / vec3f / vec3d
        Operands from the forward.
    idx : wp.array, shape ``(N,)``, dtype int32
        Sorted segment indices in ``[0, M)``.
    grad_x, grad_y : wp.array, shape ``(N,)``, dtype matches ``x``
        OUTPUT: per-element gradients.

    Notes
    -----
    Two element-wise scaled-broadcast launches (``dim=N`` each); no atomics.

    See Also
    --------
    segmented_dot_double_backward : Second-order backward.
    """
    # grad_x[i] = y[i] * g_out[s]  → _segmented_mul(y, g_out, idx, grad_x)
    N = x.shape[0]
    if N == 0:
        grad_x.zero_()
        grad_y.zero_()
        return
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


def segmented_dot_double_backward(
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
    """Double-backward of segmented_dot.

    Given cotangents ``(gg_gx, gg_gy)``::

        grad_g_out[s]   = sum_i dot(gg_gx[i], y[i]) + sum_i dot(gg_gy[i], x[i])
        grad_x_extra[i] = gg_gy[i] * g_out[s]
        grad_y_extra[i] = gg_gx[i] * g_out[s]

    Parameters
    ----------
    gg_gx, gg_gy : wp.array, shape ``(N,)``, dtype matches ``x``
        Cotangents of grad_x, grad_y.
    g_out : wp.array, shape ``(M,)``, dtype float32 / float64
        Upstream gradient of the forward output.
    x, y : wp.array, shape ``(N,)``, dtype float32 / float64 / vec3f / vec3d
        Operands from the forward.
    idx : wp.array, shape ``(N,)``, dtype int32
        Sorted segment indices in ``[0, M)``.
    M : int
        Number of segments.  Unused; retained for API symmetry.
    grad_g_out : wp.array, shape ``(M,)``, dtype matches ``g_out``
        OUTPUT: per-segment second-order adjoint.
    grad_x_extra, grad_y_extra : wp.array, shape ``(N,)``, dtype matches ``x``
        OUTPUT: extra per-element second-order adjoints.

    Notes
    -----
    Four launches: two RLE-EPT segmented_dot reductions ``atomic_add`` into
    the same pre-zeroed ``grad_g_out`` buffer, plus two element-wise
    scaled-broadcast launches (``dim=N``) for the ``_extra`` outputs.

    See Also
    --------
    segmented_dot_backward : First-order backward.
    """
    N = x.shape[0]
    if N == 0:
        grad_g_out.zero_()
        grad_x_extra.zero_()
        grad_y_extra.zero_()
        return
    # grad_g_out accumulates via two atomic_add reductions, so it must start zeroed.
    grad_g_out.zero_()
    device = x.device
    ept = compute_ept(N, max(device.sm_count, 1), x.dtype in _VEC_TYPES)
    dim_rle = (N + ept - 1) // ept
    # grad_g_out[s] = sum dot(gg_gx, y) + sum dot(gg_gy, x)
    # Both reductions atomic_add into the pre-zeroed grad_g_out — no tmp needed.
    wp.launch(
        _segmented_dot_overloads[x.dtype],
        dim=dim_rle,
        inputs=[gg_gx, y, idx, grad_g_out, N, ept],
        device=device,
    )
    wp.launch(
        _segmented_dot_overloads[x.dtype],
        dim=dim_rle,
        inputs=[gg_gy, x, idx, grad_g_out, N, ept],
        device=device,
    )
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


def segmented_inner_products_backward(
    x: wp.array,
    y: wp.array,
    idx: wp.array,
    g_xy: wp.array,
    g_xx: wp.array,
    g_yy: wp.array,
    grad_x: wp.array,
    grad_y: wp.array,
) -> None:
    """First-order backward of segmented_inner_products.

    Forward produces three per-segment outputs ``(xy_s, xx_s, yy_s)``; with
    matching cotangents ``(g_xy, g_xx, g_yy)``::

        grad_x[i] = g_xy[s]*y[i] + 2*g_xx[s]*x[i]
        grad_y[i] = g_xy[s]*x[i] + 2*g_yy[s]*y[i]

    Parameters
    ----------
    x, y : wp.array, shape ``(N,)``, dtype float32 / float64 / vec3f / vec3d
        Forward operands.
    idx : wp.array, shape ``(N,)``, dtype int32
        Sorted segment indices in ``[0, M)``.
    g_xy, g_xx, g_yy : wp.array, shape ``(M,)``, dtype float32 / float64
        Cotangents of the three forward outputs.
    grad_x, grad_y : wp.array, shape ``(N,)``, dtype matches ``x``
        OUTPUT: per-element gradients.

    Notes
    -----
    One fused element-wise launch with ``dim=N``; no atomics.

    See Also
    --------
    segmented_inner_products_double_backward : Second-order backward.
    """
    N = x.shape[0]
    if N == 0:
        grad_x.zero_()
        grad_y.zero_()
        return
    wp.launch(
        _segmented_inner_products_backward_overloads[x.dtype],
        dim=N,
        inputs=[x, y, idx, g_xy, g_xx, g_yy, grad_x, grad_y],
        device=x.device,
    )


def segmented_inner_products_double_backward(
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
    """Double-backward of segmented_inner_products.

    Given cotangents ``gg_gx``, ``gg_gy`` of the first-order grads of ``x``, ``y``,
    produces the five second-order adjoints w.r.t. ``(x, y, g_xy, g_xx, g_yy)``::

        grad_x_extra[i]    = 2*g_xx[s]*gg_gx[i] + g_xy[s]*gg_gy[i]
        grad_y_extra[i]    = g_xy[s]*gg_gx[i]   + 2*g_yy[s]*gg_gy[i]
        grad_g_xy_extra[s] = sum_i (dot(gg_gx[i], y[i]) + dot(gg_gy[i], x[i]))
        grad_g_xx_extra[s] = 2 * sum_i dot(gg_gx[i], x[i])
        grad_g_yy_extra[s] = 2 * sum_i dot(gg_gy[i], y[i])

    Parameters
    ----------
    gg_gx, gg_gy : wp.array, shape ``(N,)``, dtype float32 / float64 / vec3f / vec3d
        Cotangents of the first-order grads.  Must match ``x``/``y`` dtype.
    x, y : wp.array, shape ``(N,)``, dtype matches gg_gx
        Original forward operands saved from the forward pass.
    g_xy, g_xx, g_yy : wp.array, shape ``(M,)``, dtype float32 / float64
        Cotangents of the three forward outputs.
    idx : wp.array, shape ``(N,)``, dtype int32
        Sorted segment indices in ``[0, M)``.
    M : int
        Number of segments.
    grad_x_extra, grad_y_extra : wp.array, shape ``(N,)``, dtype matches ``x``
        OUTPUT: element-wise second-order adjoints w.r.t. ``x``, ``y``.
    grad_g_xy_extra, grad_g_xx_extra, grad_g_yy_extra : wp.array, shape ``(M,)``,
        dtype matches ``g_xy``
        OUTPUT: per-segment second-order adjoints w.r.t. the three forward outputs.

    Notes
    -----
    Six kernel launches, all writing into pre-zeroed buffers:

    * one element-wise kernel (``dim=N``, no reduction) for the two ``*_extra``
      element outputs,
    * two RLE segmented_dot reductions (``dim=ceil(N/EPT)``) that ``atomic_add``
      into the same ``grad_g_xy_extra`` buffer,
    * one segmented_dot reduction into ``grad_g_xx_extra`` followed by
      ``_scale_inplace`` by 2.0,
    * one segmented_dot reduction into ``grad_g_yy_extra`` followed by
      ``_scale_inplace`` by 2.0.

    See Also
    --------
    segmented_inner_products_backward : First-order backward.
    segmented_dot_double_backward : Same RLE-reduction pattern.
    """
    for arr in (
        grad_x_extra,
        grad_y_extra,
        grad_g_xy_extra,
        grad_g_xx_extra,
        grad_g_yy_extra,
    ):
        arr.zero_()
    N = x.shape[0]
    if N == 0:
        return
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
    # scalar outputs use segmented_dot (which atomic_adds into a pre-zeroed buffer)
    scalar_dt = _VEC_TO_SCALAR.get(x.dtype, x.dtype)
    # grad_g_xy_extra[s] = sum dot(gg_gx,y) + sum dot(gg_gy,x) — accumulate in place.
    wp.launch(
        _segmented_dot_overloads[x.dtype],
        dim=dim_rle,
        inputs=[gg_gx, y, idx, grad_g_xy_extra, N, ept],
        device=device,
    )
    wp.launch(
        _segmented_dot_overloads[x.dtype],
        dim=dim_rle,
        inputs=[gg_gy, x, idx, grad_g_xy_extra, N, ept],
        device=device,
    )
    # grad_g_xx_extra[s] = 2*sum dot(gg_gx, x) — reduce then scale in place.
    wp.launch(
        _segmented_dot_overloads[x.dtype],
        dim=dim_rle,
        inputs=[gg_gx, x, idx, grad_g_xx_extra, N, ept],
        device=device,
    )
    _scale_inplace(grad_g_xx_extra, type_=scalar_dt, factor=2.0)
    # grad_g_yy_extra[s] = 2*sum dot(gg_gy, y) — reduce then scale in place.
    wp.launch(
        _segmented_dot_overloads[x.dtype],
        dim=dim_rle,
        inputs=[gg_gy, y, idx, grad_g_yy_extra, N, ept],
        device=device,
    )
    _scale_inplace(grad_g_yy_extra, type_=scalar_dt, factor=2.0)


# ---------------------------------------------------------------------------
# segmented_axpy
# ---------------------------------------------------------------------------


def segmented_axpy_backward(
    g_out: wp.array,
    x: wp.array,
    a: wp.array,
    idx: wp.array,
    M: int,
    grad_y_in: wp.array,
    grad_x: wp.array,
    grad_a: wp.array,
) -> None:
    """First-order backward of segmented_axpy (forward: ``out[i] = y_in[i] + x[i]*a[s]``).

    Three outputs::

        grad_y_in[i] = g_out[i]                       (identity copy)
        grad_x[i]    = a[s] * g_out[i]                (per-element scaled broadcast)
        grad_a[s]    = sum_i dot(x[i], g_out[i])      (per-segment reduction)

    Parameters
    ----------
    g_out : wp.array, shape ``(N,)``, dtype float32 / float64 / vec3f / vec3d
        Upstream gradient of the forward output.
    x : wp.array, shape ``(N,)``, dtype matches ``g_out``
        Per-element operand.
    a : wp.array, shape ``(M,)``, dtype float32 / float64
        Per-segment scalar.
    idx : wp.array, shape ``(N,)``, dtype int32
        Sorted segment indices in ``[0, M)``.
    M : int
        Number of segments.  Unused; retained for API symmetry.
    grad_y_in : wp.array, shape ``(N,)``, dtype matches ``g_out``
        OUTPUT: identity copy of ``g_out``.
    grad_x : wp.array, shape ``(N,)``, dtype matches ``g_out``
        OUTPUT: per-element gradient.
    grad_a : wp.array, shape ``(M,)``, dtype matches ``a``
        OUTPUT: per-segment scalar gradient.

    Notes
    -----
    Three launches: ``wp.copy`` for the identity ``grad_y_in``, an element-wise
    mul (``dim=N``) for ``grad_x``, and an RLE-EPT segmented_dot reduction
    (``dim=ceil(N/EPT)``) with ``atomic_add`` for ``grad_a``.

    See Also
    --------
    segmented_axpy_double_backward : Second-order backward.
    segmented_axpby_backward : Two-coefficient variant.
    """
    N = g_out.shape[0]
    if N == 0:
        grad_y_in.zero_()
        grad_x.zero_()
        grad_a.zero_()
        return
    # grad_a accumulates via atomic_add in segmented_dot, so it must start zeroed.
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


def segmented_axpy_double_backward(
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
    """Double-backward of segmented_axpy (``out = y_in + a[s]*x[i]``).

    Given cotangents ``(gg_gy_in, gg_gx, gg_ga)``::

        grad_g_out[i]   = gg_gy_in[i] + gg_gx[i]*a[s] + gg_ga[s]*x[i]
        grad_x_extra[i] = gg_ga[s] * g_out[i]
        grad_a_extra[s] = sum_i dot(gg_gx[i], g_out[i])

    Parameters
    ----------
    gg_gy_in : wp.array, shape ``(N,)``, dtype matches ``g_out``
        Cotangent of grad_y_in (identity slot in the forward backward).
    gg_gx : wp.array, shape ``(N,)``, dtype matches ``g_out``
        Cotangent of grad_x.
    gg_ga : wp.array, shape ``(M,)``, dtype matches ``a``
        Cotangent of grad_a.
    g_out : wp.array, shape ``(N,)``, dtype float32 / float64 / vec3f / vec3d
        Upstream gradient of the forward output.
    x : wp.array, shape ``(N,)``, dtype matches ``g_out``
        Per-element operand.
    a : wp.array, shape ``(M,)``, dtype float32 / float64
        Per-segment scalar.
    idx : wp.array, shape ``(N,)``, dtype int32
        Sorted segment indices in ``[0, M)``.
    grad_g_out : wp.array, shape ``(N,)``, dtype matches ``g_out``
        OUTPUT: second-order adjoint w.r.t. ``g_out``.
    grad_x_extra : wp.array, shape ``(N,)``, dtype matches ``g_out``
        OUTPUT: extra second-order adjoint of ``x``.
    grad_a_extra : wp.array, shape ``(M,)``, dtype matches ``a``
        OUTPUT: extra second-order adjoint of ``a``.

    Notes
    -----
    Three launches: one fused element-wise kernel writes ``grad_g_out``
    directly (``dim=N``); one element-wise mul writes ``grad_x_extra``
    (``dim=N``); one RLE-EPT segmented_dot reduction (``atomic_add``,
    ``dim=ceil(N/EPT)``) writes ``grad_a_extra``.  ``grad_g_out`` is zeroed
    unconditionally before the empty-input guard so callers that pass a
    re-used buffer never observe stale values on ``N == 0`` segments.

    See Also
    --------
    segmented_axpy_backward : First-order backward.
    segmented_axpby_double_backward : Two-coefficient variant.
    """
    grad_g_out.zero_()
    N = g_out.shape[0]
    if N == 0:
        grad_x_extra.zero_()
        grad_a_extra.zero_()
        return
    # grad_a_extra accumulates via atomic_add in segmented_dot, so it must start zeroed.
    grad_a_extra.zero_()
    device = g_out.device
    ept = compute_ept(N, max(device.sm_count, 1), x.dtype in _VEC_TYPES)
    dim_rle = (N + ept - 1) // ept
    # grad_g_out[i] = gg_gy_in[i] + gg_gx[i]*a[s] + gg_ga[s]*x[i]  (single fused kernel)
    wp.launch(
        _segmented_axpy_dbl_bwd_grad_out_overloads[g_out.dtype],
        dim=N,
        inputs=[gg_gy_in, gg_gx, a, gg_ga, x, idx, grad_g_out],
        device=device,
    )
    # grad_a_extra[s] = sum dot(gg_gx, g_out)
    wp.launch(
        _segmented_dot_overloads[x.dtype],
        dim=dim_rle,
        inputs=[gg_gx, g_out, idx, grad_a_extra, N, ept],
        device=device,
    )
    # grad_x_extra[i] = gg_ga[s]*g_out[i]
    wp.launch(
        _segmented_mul_overloads[(g_out.dtype, gg_ga.dtype)],
        dim=N,
        inputs=[g_out, gg_ga, idx, grad_x_extra],
        device=device,
    )


# ---------------------------------------------------------------------------
# segmented_axpby
# ---------------------------------------------------------------------------


def segmented_axpby_backward(
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
    """First-order backward of segmented_axpby (forward: ``out[i] = a[s]*x[i] + b[s]*y[i]``).

    Four outputs::

        grad_x[i] = a[s] * g_out[i]
        grad_y[i] = b[s] * g_out[i]
        grad_a[s] = sum_i dot(x[i], g_out[i])
        grad_b[s] = sum_i dot(y[i], g_out[i])

    Parameters
    ----------
    g_out : wp.array, shape ``(N,)``, dtype float32 / float64 / vec3f / vec3d
        Upstream gradient of the forward output.
    a, b : wp.array, shape ``(M,)``, dtype float32 / float64
        Per-segment scalar coefficients.
    x, y : wp.array, shape ``(N,)``, dtype matches ``g_out``
        Per-element operands.
    idx : wp.array, shape ``(N,)``, dtype int32
        Sorted segment indices in ``[0, M)``.
    M : int
        Number of segments.  Unused; retained for API symmetry.
    grad_x, grad_y : wp.array, shape ``(N,)``, dtype matches ``g_out``
        OUTPUT: per-element gradients.
    grad_a, grad_b : wp.array, shape ``(M,)``, dtype matches ``a``
        OUTPUT: per-segment gradients.

    Notes
    -----
    Four launches: two element-wise muls (``dim=N``) for ``grad_x``/``grad_y``
    and two RLE-EPT segmented_dot reductions (``dim=ceil(N/EPT)``,
    ``atomic_add``) for ``grad_a``/``grad_b``.

    See Also
    --------
    segmented_axpby_double_backward : Second-order backward.
    segmented_axpy_backward : Single-coefficient variant.
    """
    for arr in (grad_x, grad_y, grad_a, grad_b):
        arr.zero_()
    N = g_out.shape[0]
    if N == 0:
        return
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


def segmented_axpby_double_backward(
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
    """Double-backward of segmented_axpby (``out[i] = a[s]*x[i] + b[s]*y[i]``).

    Given cotangents ``(gg_gx, gg_gy, gg_ga, gg_gb)`` of the four first-order grads,
    produces the five second-order adjoints w.r.t. ``(g_out, x, y, a, b)``::

        grad_g_out[i]    = a[s]*gg_gx[i] + b[s]*gg_gy[i] + gg_ga[s]*x[i] + gg_gb[s]*y[i]
        grad_x_extra[i]  = gg_ga[s] * g_out[i]
        grad_y_extra[i]  = gg_gb[s] * g_out[i]
        grad_a_extra[s]  = sum_i dot(gg_gx[i], g_out[i])
        grad_b_extra[s]  = sum_i dot(gg_gy[i], g_out[i])

    Parameters
    ----------
    gg_gx, gg_gy : wp.array, shape ``(N,)``, dtype matches ``x``
        Cotangents of grad_x, grad_y.
    gg_ga, gg_gb : wp.array, shape ``(M,)``, dtype float32 / float64
        Cotangents of grad_a, grad_b.
    g_out : wp.array, shape ``(N,)``, dtype matches ``x``
        Upstream gradient of the forward output.
    a, b : wp.array, shape ``(M,)``, dtype float32 / float64
        Per-segment scalar coefficients from the forward.
    x, y : wp.array, shape ``(N,)``, dtype float32 / float64 / vec3f / vec3d
        Per-element operands from the forward.
    idx : wp.array, shape ``(N,)``, dtype int32
        Sorted segment indices in ``[0, M)``.
    grad_g_out : wp.array, shape ``(N,)``, dtype matches ``x``
        OUTPUT: second-order adjoint w.r.t. ``g_out``.
    grad_x_extra, grad_y_extra : wp.array, shape ``(N,)``, dtype matches ``x``
        OUTPUT: extra contributions to the second-order adjoints of ``x``, ``y``.
    grad_a_extra, grad_b_extra : wp.array, shape ``(M,)``, dtype float32 / float64
        OUTPUT: extra contributions to the second-order adjoints of ``a``, ``b``.

    Notes
    -----
    Five kernel launches into pre-zeroed buffers:

    * one fused element-wise kernel (``dim=N``) writes ``grad_g_out`` directly,
    * two element-wise ``segmented_mul`` launches (``dim=N``) write
      ``grad_x_extra`` and ``grad_y_extra``,
    * two RLE segmented_dot reductions (``dim=ceil(N/EPT)``) ``atomic_add`` into
      ``grad_a_extra`` and ``grad_b_extra``.

    See Also
    --------
    segmented_axpby_backward : First-order backward.
    segmented_axpy_double_backward : Single-coefficient variant.
    """
    for arr in (grad_g_out, grad_x_extra, grad_y_extra, grad_a_extra, grad_b_extra):
        arr.zero_()
    N = g_out.shape[0]
    if N == 0:
        return
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


def segmented_mean_backward(
    g_out: wp.array,
    counts: wp.array,
    idx: wp.array,
    grad_x: wp.array,
) -> None:
    """First-order backward of segmented_mean.

    ``grad_x[i] = g_out[idx[i]] / count[idx[i]]`` (gather divided by per-segment
    population).

    Parameters
    ----------
    g_out : wp.array, shape ``(M,)``, dtype float32 / float64 / vec3f / vec3d
        Upstream gradient of the per-segment mean output.
    counts : wp.array, shape ``(M,)``, dtype int32
        Per-segment element counts saved from the forward (via
        ``segmented_count``).
    idx : wp.array, shape ``(N,)``, dtype int32
        Sorted segment indices in ``[0, M)``.
    grad_x : wp.array, shape ``(N,)``, dtype matches ``g_out``
        OUTPUT: per-element gradient.

    Notes
    -----
    One element-wise launch with ``dim=N``; no atomics.  Scalar and vec3
    variants dispatch to type-specific overloads.

    See Also
    --------
    segmented_mean_double_backward : Symmetric mean of ``gg_x``.
    """
    N = grad_x.shape[0]
    if N == 0:
        return
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


def segmented_mean_double_backward(
    gg_x: wp.array,
    counts: wp.array,
    idx: wp.array,
    grad_g_out: wp.array,
) -> None:
    """Double-backward of segmented_mean.

    ``grad_g_out[s] = sum_i gg_x[i] / count[s]`` — the per-segment mean of
    ``gg_x``.  Symmetric with the forward (mean → mean).

    Parameters
    ----------
    gg_x : wp.array, shape ``(N,)``, dtype float32 / float64 / vec3f / vec3d
        Cotangent of the first-order grad_x.
    counts : wp.array, shape ``(M,)``, dtype int32
        Per-segment population.
    idx : wp.array, shape ``(N,)``, dtype int32
        Sorted segment indices in ``[0, M)``.
    grad_g_out : wp.array, shape ``(M,)``, dtype matches ``gg_x``
        OUTPUT: per-segment second-order adjoint.

    Notes
    -----
    Two-step: :func:`_launch_sum` (RLE-EPT segmented reduction with
    ``atomic_add``) accumulates per-segment sums into a scratch buffer, then
    one ``dim=M`` element-wise divide-by-count produces ``grad_g_out``.  Scalar
    and vec3 paths use different divide overloads.

    See Also
    --------
    segmented_mean_backward : First-order backward.
    """
    M = grad_g_out.shape[0]
    grad_g_out.zero_()
    if gg_x.shape[0] == 0:
        return
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


def segmented_rms_norm_forward_precompute(
    x: wp.array,
    idx: wp.array,
    sum_sq: wp.array,
    counts: wp.array,
    out: wp.array,
    inv_norm: wp.array,
) -> None:
    """Forward of segmented_rms_norm with precomputed state for backward.

    Beyond the plain RMS norm ``out[s] = sqrt(sum_sq[s] / counts[s])``, this
    variant also writes ``inv_norm[s] = 1 / (out[s] * counts[s])`` (or zero
    where ``denom <= 0``) — the value the backward kernel needs to avoid
    recomputing a divide inside the gradient launch.

    Parameters
    ----------
    x : wp.array, shape ``(N,)``, dtype vec3f / vec3d
        Per-element input vectors.
    idx : wp.array, shape ``(N,)``, dtype int32
        Sorted segment indices in ``[0, M)``.
    sum_sq : wp.array, shape ``(M,)``, dtype float32 / float64
        OUTPUT scratch: per-segment ``sum_i ||x[i]||^2``.  Overwritten.
    counts : wp.array, shape ``(M,)``, dtype int32
        OUTPUT scratch: per-segment element counts.  Overwritten.
    out : wp.array, shape ``(M,)``, dtype float32 / float64
        OUTPUT: per-segment RMS norm.  Overwritten.
    inv_norm : wp.array, shape ``(M,)``, dtype matches ``out``
        OUTPUT: saved state for the backward kernel.  Overwritten.

    Notes
    -----
    Three sub-ops: a ``segmented_dot(x, x, …)`` reduction into ``sum_sq``, a
    ``segmented_count`` write into ``counts``, and a ``dim=M`` finalize
    kernel that writes ``out`` and ``inv_norm``.

    See Also
    --------
    segmented_rms_norm_backward : Consumes ``inv_norm`` to skip a divide.
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
        _segmented_rms_norm_finalize_and_save_overloads[scalar_dtype],
        dim=M,
        inputs=[sum_sq, counts, out, inv_norm],
        device=device,
    )


def segmented_rms_norm_backward(
    g_out: wp.array,
    x: wp.array,
    inv_norm: wp.array,
    idx: wp.array,
    grad_x: wp.array,
) -> None:
    """First-order backward of segmented_rms_norm.

    ``grad_x[i] = g_out[idx[i]] * x[i] * inv_norm[idx[i]]``.  Reuses the
    ``inv_norm`` term saved by :func:`segmented_rms_norm_forward_precompute`
    to skip a per-element divide.

    Parameters
    ----------
    g_out : wp.array, shape ``(M,)``, dtype float32 / float64
        Upstream gradient of the per-segment RMS norm.
    x : wp.array, shape ``(N,)``, dtype vec3f / vec3d
        Per-element input vectors from the forward.
    inv_norm : wp.array, shape ``(M,)``, dtype matches ``g_out``
        Saved state from the precompute forward.  Required to be the output of
        :func:`segmented_rms_norm_forward_precompute` against the same
        ``(x, idx)``.
    idx : wp.array, shape ``(N,)``, dtype int32
        Sorted segment indices in ``[0, M)``.
    grad_x : wp.array, shape ``(N,)``, dtype matches ``x``
        OUTPUT: per-element gradient.

    Notes
    -----
    One element-wise launch with ``dim=N``; no atomics, no reductions.

    See Also
    --------
    segmented_rms_norm_forward_precompute : Produces ``inv_norm``.
    segmented_rms_norm_double_backward : Second-order backward.
    """
    N = grad_x.shape[0]
    if N == 0:
        return
    wp.launch(
        _segmented_rms_norm_backward_overloads[x.dtype],
        dim=N,
        inputs=[g_out, x, inv_norm, idx, grad_x],
        device=grad_x.device,
    )


def segmented_rms_norm_double_backward(
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
    """Double-backward of segmented_rms_norm.

    Let ``inner[s] = sum_i dot(gg_x[i], x[i])``.  Then::

        grad_g_out_extra[s] = inner[s] * inv_norm[s]
        grad_x_extra[i]     = g_out[s]*inv_norm[s] * gg_x[i]
                              - g_out[s]*inv_norm[s]^3 * counts[s] * inner[s] * x[i]

    Parameters
    ----------
    gg_x : wp.array, shape ``(N,)``, dtype vec3f / vec3d
        Cotangent of the first-order grad_x.
    x : wp.array, shape ``(N,)``, dtype matches ``gg_x``
        Forward operand.
    g_out : wp.array, shape ``(M,)``, dtype float32 / float64
        Upstream gradient of the forward output.
    inv_norm : wp.array, shape ``(M,)``, dtype matches ``g_out``
        Saved state from the precompute forward.
    counts : wp.array, shape ``(M,)``, dtype int32
        Per-segment population from the precompute forward.
    idx : wp.array, shape ``(N,)``, dtype int32
        Sorted segment indices in ``[0, M)``.
    M : int
        Number of segments.
    grad_x_extra : wp.array, shape ``(N,)``, dtype matches ``x``
        OUTPUT: extra second-order adjoint w.r.t. ``x``.
    grad_g_out_extra : wp.array, shape ``(M,)``, dtype matches ``g_out``
        OUTPUT: second-order adjoint w.r.t. ``g_out``.

    Notes
    -----
    Three sub-ops: a ``segmented_dot(gg_x, x, …)`` reduction into a scratch
    ``inner`` buffer; a ``dim=M`` element-wise multiply for ``grad_g_out_extra``;
    a fused ``dim=N`` element-wise kernel for ``grad_x_extra``.

    See Also
    --------
    segmented_rms_norm_backward : First-order backward.
    """
    N = x.shape[0]
    if N == 0:
        grad_x_extra.zero_()
        grad_g_out_extra.zero_()
        return
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


def segmented_max_norm_forward_precompute(
    x: wp.array,
    idx: wp.array,
    out: wp.array,
    argmax_idx: wp.array,
) -> None:
    """Forward of segmented_max_norm with precomputed state for backward.

    Writes both ``out[s] = max_i ||x[i]||`` and
    ``argmax_idx[s] = arg max_i ||x[i]||``.  ``argmax_idx`` is initialized to
    ``-1`` here before the argmax scan runs, so the buffer the caller passes
    in does not need to be pre-filled — but it MUST be passed to the backward
    launchers below as-is, without any intermediate reuse that would clobber
    the recorded indices.

    Parameters
    ----------
    x : wp.array, shape ``(N,)``, dtype vec3f / vec3d
        Per-element input vectors.
    idx : wp.array, shape ``(N,)``, dtype int32
        Sorted segment indices in ``[0, M)``.
    out : wp.array, shape ``(M,)``, dtype float32 / float64
        OUTPUT: per-segment max norm.  Overwritten.
    argmax_idx : wp.array, shape ``(M,)``, dtype int32
        OUTPUT: per-segment argmax index (``-1`` for empty segments).
        Filled to ``-1`` internally, then populated.

    Notes
    -----
    Two-step: a forward ``segmented_max_norm`` call (one ``dim=N`` element-wise
    plus one RLE reduction internally) produces ``out``; an RLE-EPT
    ``argmax_kernel`` (``dim=ceil(N/EPT)``) then locates each per-segment
    argmax via ``atomic_max`` on ``argmax_idx`` — relies on the ``-1`` fill so
    any valid index wins.

    See Also
    --------
    segmented_max_norm_backward : Consumes ``argmax_idx``.
    """
    from nvalchemiops.segment_ops import segmented_max_norm as _fwd_max_norm

    N = x.shape[0]
    if N == 0:
        return
    device = x.device
    # Initialize argmax_idx to -1 so that the first valid write wins via atomic_max.
    # An empty segment retains -1 (skipped by the backward kernel's ``i == argmax_idx[s]``
    # gate since tid() is always >= 0).
    argmax_idx.fill_(-1)
    _fwd_max_norm(x, idx, out)
    ept = compute_ept(N, max(device.sm_count, 1), True)
    dim = (N + ept - 1) // ept
    wp.launch(
        _segmented_max_norm_argmax_overloads[x.dtype],
        dim=dim,
        inputs=[x, idx, out, argmax_idx, N, ept],
        device=device,
    )


def segmented_max_norm_backward(
    g_out: wp.array,
    x: wp.array,
    argmax_idx: wp.array,
    idx: wp.array,
    grad_x: wp.array,
) -> None:
    """First-order backward of segmented_max_norm (subgradient at argmax).

    Only the argmax element of each segment receives nonzero gradient::

        grad_x[argmax_idx[s]] = g_out[s] * x[argmax_idx[s]] / ||x[argmax_idx[s]]||

    Parameters
    ----------
    g_out : wp.array, shape ``(M,)``, dtype float32 / float64
        Upstream gradient of the per-segment max norm.
    x : wp.array, shape ``(N,)``, dtype vec3f / vec3d
        Per-element input vectors from the forward.
    argmax_idx : wp.array, shape ``(M,)``, dtype int32
        Per-segment argmax indices.  MUST be the output of
        :func:`segmented_max_norm_forward_precompute` against the same
        ``(x, idx)``; passing a zero-initialized or stale buffer produces wrong
        gradients silently (the kernel writes at ``i == argmax_idx[s]`` — with
        zeros that's always ``i = 0``).
    idx : wp.array, shape ``(N,)``, dtype int32
        Sorted segment indices in ``[0, M)``.
    grad_x : wp.array, shape ``(N,)``, dtype matches ``x``
        OUTPUT: per-element gradient.

    Notes
    -----
    One element-wise launch with ``dim=N``; each thread tests its own
    ``i == argmax_idx[idx[i]]`` and writes only if it matches.  Empty
    segments (``argmax_idx[s] == -1``) are skipped naturally because
    ``tid()`` is always non-negative.

    See Also
    --------
    segmented_max_norm_forward_precompute : Produces ``argmax_idx``.
    segmented_max_norm_double_backward : Second-order backward.
    """
    grad_x.zero_()
    N = grad_x.shape[0]
    if N == 0:
        return
    wp.launch(
        _segmented_max_norm_backward_overloads[x.dtype],
        dim=N,
        inputs=[g_out, x, argmax_idx, idx, grad_x],
        device=grad_x.device,
    )


def segmented_max_norm_double_backward(
    gg_gx: wp.array,
    g_out: wp.array,
    x: wp.array,
    argmax_idx: wp.array,
    idx: wp.array,
    grad_x_extra: wp.array,
    grad_g_out: wp.array,
) -> None:
    """Double-backward of segmented_max_norm (tangent-plane projection at argmax).

    Let ``i* = argmax_idx[s]`` and ``x_hat = x[i*] / ||x[i*]||``::

        grad_x_extra[i*] = (g_out[s] / ||x[i*]||) * (gg_gx[i*] - x_hat * dot(x_hat, gg_gx[i*]))
        grad_g_out[s]    = dot(x_hat, gg_gx[i*])

    All non-argmax elements receive zero contribution.

    Parameters
    ----------
    gg_gx : wp.array, shape ``(N,)``, dtype vec3f / vec3d
        Cotangent of grad_x.
    g_out : wp.array, shape ``(M,)``, dtype float32 / float64
        Upstream gradient of the forward output.
    x : wp.array, shape ``(N,)``, dtype matches ``gg_gx``
        Per-element forward operand.
    argmax_idx : wp.array, shape ``(M,)``, dtype int32
        Per-segment argmax indices.  MUST come from
        :func:`segmented_max_norm_forward_precompute` — see the
        contract on :func:`segmented_max_norm_backward` for the failure
        mode if the buffer is constructed any other way.
    idx : wp.array, shape ``(N,)``, dtype int32
        Sorted segment indices in ``[0, M)``.
    grad_x_extra : wp.array, shape ``(N,)``, dtype matches ``x``
        OUTPUT: extra second-order adjoint w.r.t. ``x``.
    grad_g_out : wp.array, shape ``(M,)``, dtype matches ``g_out``
        OUTPUT: second-order adjoint w.r.t. ``g_out``.

    Notes
    -----
    One element-wise launch with ``dim=N``; each thread tests
    ``i == argmax_idx[idx[i]]`` and computes the tangent-plane projection at
    matching elements.  ``grad_g_out`` accumulates via ``atomic_add`` on the
    single argmax index per segment.

    See Also
    --------
    segmented_max_norm_backward : First-order backward.
    """
    grad_x_extra.zero_()
    grad_g_out.zero_()
    N = x.shape[0]
    if N == 0:
        return
    wp.launch(
        _segmented_max_norm_double_backward_overloads[x.dtype],
        dim=N,
        inputs=[gg_gx, g_out, x, argmax_idx, idx, grad_x_extra, grad_g_out],
        device=grad_x_extra.device,
    )


# ---------------------------------------------------------------------------
# segmented_matvec
# ---------------------------------------------------------------------------


def segmented_matvec_backward(
    g_out: wp.array,
    v: wp.array,
    m: wp.array,
    idx: wp.array,
    grad_v: wp.array,
    grad_M: wp.array,
) -> None:
    """First-order backward of segmented_matvec (forward: ``out[i] = M[s]^T @ v[i]``).

    Two outputs::

        grad_v[i] = M[s] @ g_out[i]                          (non-transposed matvec)
        grad_M[s] = sum_{i:idx[i]=s} outer(v[i], g_out[i])   (per-segment outer-sum)

    Parameters
    ----------
    g_out : wp.array, shape ``(N,)``, dtype vec3f / vec3d
        Upstream gradient of the per-element forward output.
    v : wp.array, shape ``(N,)``, dtype matches ``g_out``
        Per-element forward operand.
    m : wp.array, shape ``(M,)``, dtype mat33f / mat33d (matching ``v``'s precision)
        Per-segment matrix from the forward.
    idx : wp.array, shape ``(N,)``, dtype int32
        Sorted segment indices in ``[0, M)``.
    grad_v : wp.array, shape ``(N,)``, dtype matches ``v``
        OUTPUT: per-element gradient.
    grad_M : wp.array, shape ``(M,)``, dtype matches ``m``
        OUTPUT: per-segment matrix gradient.

    Notes
    -----
    Two launches: one element-wise matvec (``dim=N``) for ``grad_v``; one
    RLE-EPT segmented outer-product reduction (``dim=ceil(N/EPT)``) with
    ``atomic_add`` for ``grad_M``.

    See Also
    --------
    segmented_matvec_double_backward : Second-order backward.
    """
    N = v.shape[0]
    if N == 0:
        grad_v.zero_()
        grad_M.zero_()
        return
    # grad_M accumulates via atomic_add in the outer-product reduction, so it must start zeroed.
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


def segmented_matvec_double_backward(
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
    """Double-backward of segmented_matvec.

    Given cotangents ``(gg_gv, gg_gM)`` of the first-order grads::

        grad_g_out[i]   = m[s]^T @ gg_gv[i] + gg_gM[s]^T @ v[i]
        grad_v_extra[i] = gg_gM[s] @ g_out[i]
        grad_M_extra[s] = sum_{i:idx[i]=s} outer(gg_gv[i], g_out[i])

    Parameters
    ----------
    gg_gv : wp.array, shape ``(N,)``, dtype vec3f / vec3d
        Cotangent of grad_v.
    gg_gM : wp.array, shape ``(M,)``, dtype mat33f / mat33d
        Cotangent of grad_M.
    g_out : wp.array, shape ``(N,)``, dtype matches ``gg_gv``
        Upstream gradient of the forward output.
    v : wp.array, shape ``(N,)``, dtype matches ``gg_gv``
        Per-element forward operand.
    m : wp.array, shape ``(M,)``, dtype matches ``gg_gM``
        Per-segment matrix from the forward.
    idx : wp.array, shape ``(N,)``, dtype int32
        Sorted segment indices in ``[0, M)``.
    grad_g_out : wp.array, shape ``(N,)``, dtype matches ``v``
        OUTPUT: second-order adjoint w.r.t. ``g_out``.
    grad_v_extra : wp.array, shape ``(N,)``, dtype matches ``v``
        OUTPUT: extra second-order adjoint of ``v``.
    grad_M_extra : wp.array, shape ``(M,)``, dtype matches ``m``
        OUTPUT: extra second-order adjoint of ``M``.

    Notes
    -----
    Three launches: one fused element-wise kernel for ``grad_g_out``
    (``dim=N``); one element-wise non-transposed matvec for ``grad_v_extra``
    (``dim=N``); one RLE-EPT outer-product reduction for ``grad_M_extra``
    (``dim=ceil(N/EPT)`` with ``atomic_add``).  ``grad_g_out`` is zeroed
    unconditionally before the empty-input guard so callers that pass a
    re-used buffer never observe stale values on ``N == 0`` segments.

    See Also
    --------
    segmented_matvec_backward : First-order backward.
    """
    grad_g_out.zero_()
    N = v.shape[0]
    if N == 0:
        grad_v_extra.zero_()
        grad_M_extra.zero_()
        return
    # grad_M_extra accumulates via atomic_add in the outer-product reduction, so it must start zeroed.
    grad_M_extra.zero_()
    device = v.device
    ept = compute_ept(N, max(device.sm_count, 1), True)
    dim_rle = (N + ept - 1) // ept
    # grad_g_out[i] = m[s]^T @ gg_gv[i] + gg_gM[s]^T @ v[i]  (single fused kernel)
    wp.launch(
        _segmented_matvec_dbl_bwd_grad_out_overloads[(v.dtype, m.dtype)],
        dim=N,
        inputs=[gg_gv, m, v, gg_gM, idx, grad_g_out],
        device=device,
    )
    # grad_v_extra[i] = gg_gM[s] @ g_out[i]   (non-transposed matvec)
    wp.launch(
        _segmented_matvec_backward_v_overloads[(v.dtype, m.dtype)],
        dim=N,
        inputs=[g_out, gg_gM, idx, grad_v_extra],
        device=device,
    )
    # grad_M_extra[s] = sum outer(gg_gv[i], g_out[i])
    # Kernel signature is (g_out, v, ...) and computes outer(v, g_out), so we
    # pass g_out in the first slot and gg_gv in the second to get the documented
    # outer(gg_gv, g_out).
    wp.launch(
        _segmented_matvec_backward_M_overloads[(v.dtype, m.dtype)],
        dim=dim_rle,
        inputs=[g_out, gg_gv, idx, grad_M_extra, N, ept],
        device=device,
    )


# ---------------------------------------------------------------------------
# segment_div
# ---------------------------------------------------------------------------


def segment_div_backward(
    g_result: wp.array,
    denominator: wp.array,
    grad_numerator: wp.array,
) -> None:
    """First-order backward of segment_div (forward: ``result[i] = numerator[i] / denominator[i]``).

    Only the numerator is differentiable; ``grad_numerator[i] = g_result[i] / denominator[i]``.

    Parameters
    ----------
    g_result : wp.array, shape ``(N,)``, dtype float32 / float64
        Upstream gradient of the per-element forward output.
    denominator : wp.array, shape ``(N,)``, dtype int32
        Per-element integer denominator from the forward (e.g. counts).
    grad_numerator : wp.array, shape ``(N,)``, dtype matches ``g_result``
        OUTPUT: per-element gradient.

    Notes
    -----
    One element-wise launch with ``dim=N``; no atomics.

    See Also
    --------
    segment_div_double_backward : Symmetric divide.
    """
    N = g_result.shape[0]
    if N == 0:
        return
    wp.launch(
        _segment_div_overloads[g_result.dtype],
        dim=N,
        inputs=[g_result, denominator, grad_numerator],
        device=g_result.device,
    )


def segment_div_double_backward(
    gg_numerator: wp.array,
    denominator: wp.array,
    grad_g_result: wp.array,
) -> None:
    """Double-backward of segment_div (mirror of the first-order backward).

    ``grad_g_result[i] = gg_numerator[i] / denominator[i]`` — segment_div is
    linear in the numerator, so its second-order backward is the same elementwise
    divide as the first-order.

    Parameters
    ----------
    gg_numerator : wp.array, shape ``(N,)``, dtype float32 / float64
        Cotangent of grad_numerator.
    denominator : wp.array, shape ``(N,)``, dtype int32
        Per-element integer denominator from the forward.
    grad_g_result : wp.array, shape ``(N,)``, dtype matches ``gg_numerator``
        OUTPUT: per-element second-order adjoint.

    Notes
    -----
    Delegates to :func:`segment_div_backward`; one element-wise launch
    with ``dim=N``.

    See Also
    --------
    segment_div_backward : First-order backward (identical math).
    """
    segment_div_backward(gg_numerator, denominator, grad_g_result)


# ===========================================================================
# Utility kernels and helpers used internally above
# ===========================================================================


@wp.kernel(enable_backward=False)
def _scale_inplace_float32_kernel(
    dst: wp.array(dtype=wp.float32),
    scale: wp.float32,
):
    i = wp.tid()
    dst[i] = dst[i] * scale


@wp.kernel(enable_backward=False)
def _scale_inplace_float64_kernel(
    dst: wp.array(dtype=wp.float64),
    scale: wp.float64,
):
    i = wp.tid()
    dst[i] = dst[i] * scale


_SCALE_INPLACE = {
    wp.float32: _scale_inplace_float32_kernel,
    wp.float64: _scale_inplace_float64_kernel,
}


def _scale_inplace(dst: wp.array, type_, factor: float) -> None:
    """dst[i] *= factor.  type_ is wp.float32 or wp.float64."""
    if dst.shape[0] == 0:
        return
    wp.launch(
        _SCALE_INPLACE[type_],
        dim=dst.shape[0],
        inputs=[dst, float(factor)],
        device=dst.device,
    )
