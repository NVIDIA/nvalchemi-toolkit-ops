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

"""JAX autograd bindings for segment operations (PR 3).

Mirrors the public Torch API in ``nvalchemiops.torch.segment_ops``.  Each op is
registered with ``jax.custom_vjp`` and its backward is itself wrapped in a
``jax.custom_vjp`` so the second-order adjoint can be triggered by
``jax.grad(jax.grad(...))`` or ``jax.jacfwd(jax.jacrev(...))``.

Kernel orchestration uses :func:`warp.jax_experimental.jax_callable`, which
runs the existing ``_launch_*`` Python wrappers directly on the JAX device
arrays — no host roundtrip.

Public functions
----------------
segmented_sum      : reduce per segment (scalar or vec3, f32 or f64)
segmented_dot      : per-segment dot product (vec3, f32 or f64)
segmented_mul      : per-element scale by per-segment scalar (vec3 × scalar)
segmented_mean     : per-segment mean (scalar or vec3, f32 or f64)
segmented_rms_norm : per-segment RMS norm of vec3 inputs
segmented_matvec   : per-segment matvec ``out[i] = m[idx[i]]^T @ v[i]``

Index inputs (``idx``) are non-differentiable — JAX returns ``None`` for them.
``num_segments`` (number of segments) is a static argument (compile-time constant).
"""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import warp as wp
from warp.jax_experimental import GraphMode
from warp.jax_experimental import jax_callable as _raw_jax_callable

from nvalchemiops.segment_ops import (
    segment_div as _wp_segment_div,
)
from nvalchemiops.segment_ops import (
    segmented_count as _wp_segmented_count,
)
from nvalchemiops.segment_ops import (
    segmented_dot as _wp_segmented_dot,
)
from nvalchemiops.segment_ops import (
    segmented_matvec as _wp_segmented_matvec,
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


def jax_callable(*args, **kwargs):
    """Wrap warp.jax_experimental.jax_callable with ``GraphMode.WARP``.

    The default ``GraphMode.JAX`` fails when the orchestrator issues more than
    one Warp kernel launch (e.g. mean, rms_norm forward, dot/matvec double-
    backward), because JAX can't embed multi-command FFI calls as subgraphs
    in nested ``jax.jit`` + ``jax.grad`` contexts.  ``GraphMode.WARP`` lets
    Warp own the capture and presents the call as opaque to JAX.
    """
    kwargs.setdefault("graph_mode", GraphMode.WARP)
    return _raw_jax_callable(*args, **kwargs)


__all__ = [
    "segmented_dot",
    "segmented_matvec",
    "segmented_mean",
    "segmented_mul",
    "segmented_rms_norm",
    "segmented_sum",
]


# =============================================================================
# Dtype dispatch helpers
# =============================================================================

_F = jnp.float32
_D = jnp.float64


def _norm_dtype(dtype) -> type:
    """Normalize a numpy/jax dtype to ``jnp.float32`` / ``jnp.float64`` for dict lookup."""
    if dtype == jnp.float32 or str(dtype) == "float32":
        return _F
    if dtype == jnp.float64 or str(dtype) == "float64":
        return _D
    raise ValueError(f"Unsupported dtype for segment ops: {dtype}")


def _make_callable_dict(
    builder: Callable[[type, type, str], Callable],
    variants: list[tuple],
    *,
    num_outputs: int,
    in_out_argnames: list[str] | None = None,
) -> dict:
    """Materialize a ``{key: jax_callable(...)}`` dispatch table."""
    out = {}
    for key in variants:
        out[key] = jax_callable(
            builder(*key),
            num_outputs=num_outputs,
            in_out_argnames=in_out_argnames,
        )
    return out


# =============================================================================
# segmented_sum
# =============================================================================
# Forward: out[s] = sum_{i : idx[i]=s} x[i]
# First-order bwd:   grad_x[i] = g_out[idx[i]]  (gather)
# Second-order bwd:  grad_g_out[s] = sum_{i : idx[i]=s} gg_x[i]  (scatter-sum)


def _make_sum_fwd(wp_dtype):
    def fn(
        x: wp.array(dtype=wp_dtype),
        idx: wp.array(dtype=wp.int32),
        out: wp.array(dtype=wp_dtype),
    ):
        _wp_segmented_sum(x, idx, out)

    return fn


def _make_sum_bwd(wp_dtype):
    def fn(
        g_out: wp.array(dtype=wp_dtype),
        idx: wp.array(dtype=wp.int32),
        grad_x: wp.array(dtype=wp_dtype),
    ):
        segmented_sum_backward(g_out, idx, grad_x)

    return fn


def _make_sum_dbl_bwd(wp_dtype):
    def fn(
        gg_x: wp.array(dtype=wp_dtype),
        idx: wp.array(dtype=wp.int32),
        grad_g_out: wp.array(dtype=wp_dtype),
    ):
        segmented_sum_double_backward(gg_x, idx, grad_g_out.shape[0], grad_g_out)

    return fn


# (jax_dtype, "scalar"|"vec3") -> wp dtype
_SUM_WP = {
    (_F, "scalar"): wp.float32,
    (_D, "scalar"): wp.float64,
    (_F, "vec3"): wp.vec3f,
    (_D, "vec3"): wp.vec3d,
}

_FWD_SUM = {
    k: jax_callable(_make_sum_fwd(v), num_outputs=1) for k, v in _SUM_WP.items()
}
_BWD_SUM = {
    k: jax_callable(_make_sum_bwd(v), num_outputs=1) for k, v in _SUM_WP.items()
}
_DBL_SUM = {
    k: jax_callable(_make_sum_dbl_bwd(v), num_outputs=1) for k, v in _SUM_WP.items()
}


def _sum_kind(x: jax.Array) -> tuple:
    return (_norm_dtype(x.dtype), "vec3" if x.ndim == 2 else "scalar")


# Double-backward of segmented_sum.  ``kind`` and ``num_segments`` are static.
@partial(jax.custom_vjp, nondiff_argnums=(1, 2))
def _sum_bwd_op(idx, kind, num_segments, g_out):
    n = idx.shape[0]
    out_shape = (n, 3) if kind[1] == "vec3" else (n,)
    (grad_x,) = _BWD_SUM[kind](g_out, idx, output_dims={"grad_x": out_shape})
    return grad_x


def _sum_bwd_op_fwd(idx, kind, num_segments, g_out):
    return _sum_bwd_op(idx, kind, num_segments, g_out), (idx,)


def _sum_bwd_op_bwd(kind, num_segments, residuals, gg_x):
    (idx,) = residuals
    out_shape = (num_segments, 3) if kind[1] == "vec3" else (num_segments,)
    (grad_g_out,) = _DBL_SUM[kind](gg_x, idx, output_dims={"grad_g_out": out_shape})
    # Return grads matching (idx, g_out).  idx is integer / non-differentiable.
    return (jnp.zeros_like(idx), grad_g_out)


_sum_bwd_op.defvjp(_sum_bwd_op_fwd, _sum_bwd_op_bwd)


# First-order forward.  ``kind`` and ``num_segments`` are static.
@partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def _sum_op(x, idx, num_segments, kind):
    out_shape = (num_segments, 3) if kind[1] == "vec3" else (num_segments,)
    (out,) = _FWD_SUM[kind](x, idx, output_dims={"out": out_shape})
    return out


def _sum_op_fwd(x, idx, num_segments, kind):
    return _sum_op(x, idx, num_segments, kind), (idx,)


def _sum_op_bwd(num_segments, kind, residuals, g_out):
    (idx,) = residuals
    grad_x = _sum_bwd_op(idx, kind, num_segments, g_out)
    return grad_x, None


_sum_op.defvjp(_sum_op_fwd, _sum_op_bwd)


def segmented_sum(x: jax.Array, idx: jax.Array, num_segments: int) -> jax.Array:
    """Differentiable segmented sum.

    Parameters
    ----------
    x   : jax.Array, shape ``(N,)`` or ``(N, 3)``, float32 / float64.
    idx : jax.Array, shape ``(N,)``, dtype int32, sorted in ``[0, num_segments)``.
    num_segments   : int — number of segments.  Static argument.

    Returns
    -------
    jax.Array of shape ``(num_segments,)`` or ``(num_segments, 3)``.
    """
    return _sum_op(x, idx, num_segments, _sum_kind(x))


# =============================================================================
# segmented_dot   (vec3 only in the public API)
# =============================================================================
# Forward: out[s] = sum_{i: idx[i]=s} dot(x[i], y[i])
# Bwd:    grad_x[i] = g_out[s] * y[i],  grad_y[i] = g_out[s] * x[i]
# Dbl-bwd: grad_g_out[s] = sum dot(gg_gx, y) + sum dot(gg_gy, x)
#          grad_x_extra[i] = gg_gy[i] * g_out[s]
#          grad_y_extra[i] = gg_gx[i] * g_out[s]


def _make_dot_fwd(vec_t):
    def fn(
        x: wp.array(dtype=vec_t),
        y: wp.array(dtype=vec_t),
        idx: wp.array(dtype=wp.int32),
        out: wp.array(dtype=wp.float32 if vec_t is wp.vec3f else wp.float64),
    ):
        _wp_segmented_dot(x, y, idx, out)

    return fn


def _make_dot_bwd(vec_t):
    scalar_t = wp.float32 if vec_t is wp.vec3f else wp.float64

    def fn(
        g_out: wp.array(dtype=scalar_t),
        x: wp.array(dtype=vec_t),
        y: wp.array(dtype=vec_t),
        idx: wp.array(dtype=wp.int32),
        grad_x: wp.array(dtype=vec_t),
        grad_y: wp.array(dtype=vec_t),
    ):
        segmented_dot_backward(g_out, x, y, idx, grad_x, grad_y)

    return fn


def _make_dot_dbl_bwd(vec_t):
    scalar_t = wp.float32 if vec_t is wp.vec3f else wp.float64

    def fn(
        gg_gx: wp.array(dtype=vec_t),
        gg_gy: wp.array(dtype=vec_t),
        g_out: wp.array(dtype=scalar_t),
        x: wp.array(dtype=vec_t),
        y: wp.array(dtype=vec_t),
        idx: wp.array(dtype=wp.int32),
        grad_g_out: wp.array(dtype=scalar_t),
        grad_x_extra: wp.array(dtype=vec_t),
        grad_y_extra: wp.array(dtype=vec_t),
    ):
        segmented_dot_double_backward(
            gg_gx,
            gg_gy,
            g_out,
            x,
            y,
            idx,
            grad_g_out.shape[0],
            grad_g_out,
            grad_x_extra,
            grad_y_extra,
        )

    return fn


_DOT_WP = {_F: wp.vec3f, _D: wp.vec3d}
_FWD_DOT = {
    k: jax_callable(_make_dot_fwd(v), num_outputs=1) for k, v in _DOT_WP.items()
}
_BWD_DOT = {
    k: jax_callable(_make_dot_bwd(v), num_outputs=2) for k, v in _DOT_WP.items()
}
_DBL_DOT = {
    k: jax_callable(_make_dot_dbl_bwd(v), num_outputs=3) for k, v in _DOT_WP.items()
}


# Double-backward of segmented_dot.  ``dtype`` and ``num_segments`` static.
@partial(jax.custom_vjp, nondiff_argnums=(1, 2))
def _dot_bwd_op(idx, dtype, num_segments, g_out, x, y):
    n = x.shape[0]
    (grad_x, grad_y) = _BWD_DOT[dtype](
        g_out,
        x,
        y,
        idx,
        output_dims={"grad_x": (n, 3), "grad_y": (n, 3)},
    )
    return grad_x, grad_y


def _dot_bwd_op_fwd(idx, dtype, num_segments, g_out, x, y):
    return _dot_bwd_op(idx, dtype, num_segments, g_out, x, y), (idx, g_out, x, y)


def _dot_bwd_op_bwd(dtype, num_segments, residuals, cotangents):
    idx, g_out, x, y = residuals
    gg_gx, gg_gy = cotangents
    n = x.shape[0]
    (grad_g_out, grad_x_extra, grad_y_extra) = _DBL_DOT[dtype](
        gg_gx,
        gg_gy,
        g_out,
        x,
        y,
        idx,
        output_dims={
            "grad_g_out": (num_segments,),
            "grad_x_extra": (n, 3),
            "grad_y_extra": (n, 3),
        },
    )
    return (jnp.zeros_like(idx), grad_g_out, grad_x_extra, grad_y_extra)


_dot_bwd_op.defvjp(_dot_bwd_op_fwd, _dot_bwd_op_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def _dot_op(x, y, idx, num_segments, dtype):
    (out,) = _FWD_DOT[dtype](x, y, idx, output_dims={"out": (num_segments,)})
    return out


def _dot_op_fwd(x, y, idx, num_segments, dtype):
    return _dot_op(x, y, idx, num_segments, dtype), (x, y, idx)


def _dot_op_bwd(num_segments, dtype, residuals, g_out):
    x, y, idx = residuals
    grad_x, grad_y = _dot_bwd_op(idx, dtype, num_segments, g_out, x, y)
    return grad_x, grad_y, None


_dot_op.defvjp(_dot_op_fwd, _dot_op_bwd)


def segmented_dot(
    x: jax.Array, y: jax.Array, idx: jax.Array, num_segments: int
) -> jax.Array:
    """Differentiable per-segment dot product (vec3 inputs)."""
    return _dot_op(x, y, idx, num_segments, _norm_dtype(x.dtype))


# =============================================================================
# segmented_mul   (vec3 × per-segment scalar)
# =============================================================================
# Forward: out[i] = x[i] * y[idx[i]]                          (x: vec3, y: scalar)
# Bwd:     grad_x[i] = g_out[i] * y[s]
#          grad_y[s] = sum_{i: idx[i]=s} dot(g_out[i], x[i])
# Dbl-bwd: grad_g_out[i]   = gg_gx[i]*y[s] + gg_gy[s]*x[i]
#          grad_x_extra[i] = gg_gy[s] * g_out[i]
#          grad_y_extra[s] = sum dot(gg_gx[i], g_out[i])


def _make_mul_fwd(vec_t, scalar_t):
    def fn(
        x: wp.array(dtype=vec_t),
        y: wp.array(dtype=scalar_t),
        idx: wp.array(dtype=wp.int32),
        out: wp.array(dtype=vec_t),
    ):
        _wp_segmented_mul(x, y, idx, out)

    return fn


def _make_mul_bwd(vec_t, scalar_t):
    def fn(
        g_out: wp.array(dtype=vec_t),
        x: wp.array(dtype=vec_t),
        y: wp.array(dtype=scalar_t),
        idx: wp.array(dtype=wp.int32),
        grad_x: wp.array(dtype=vec_t),
        grad_y: wp.array(dtype=scalar_t),
    ):
        segmented_mul_backward(g_out, x, y, idx, grad_y.shape[0], grad_x, grad_y)

    return fn


def _make_mul_dbl_bwd(vec_t, scalar_t):
    def fn(
        gg_gx: wp.array(dtype=vec_t),
        gg_gy: wp.array(dtype=scalar_t),
        g_out: wp.array(dtype=vec_t),
        x: wp.array(dtype=vec_t),
        y: wp.array(dtype=scalar_t),
        idx: wp.array(dtype=wp.int32),
        grad_g_out: wp.array(dtype=vec_t),
        grad_x_extra: wp.array(dtype=vec_t),
        grad_y_extra: wp.array(dtype=scalar_t),
    ):
        segmented_mul_double_backward(
            gg_gx,
            gg_gy,
            g_out,
            x,
            y,
            idx,
            grad_g_out,
            grad_x_extra,
            grad_y_extra,
        )

    return fn


_MUL_WP = {_F: (wp.vec3f, wp.float32), _D: (wp.vec3d, wp.float64)}
_FWD_MUL = {
    k: jax_callable(_make_mul_fwd(*v), num_outputs=1) for k, v in _MUL_WP.items()
}
_BWD_MUL = {
    k: jax_callable(_make_mul_bwd(*v), num_outputs=2) for k, v in _MUL_WP.items()
}
_DBL_MUL = {
    k: jax_callable(_make_mul_dbl_bwd(*v), num_outputs=3) for k, v in _MUL_WP.items()
}


@partial(jax.custom_vjp, nondiff_argnums=(1, 2))
def _mul_bwd_op(idx, dtype, num_segments, g_out, x, y):
    (grad_x, grad_y) = _BWD_MUL[dtype](
        g_out,
        x,
        y,
        idx,
        output_dims={"grad_x": x.shape, "grad_y": (num_segments,)},
    )
    return grad_x, grad_y


def _mul_bwd_op_fwd(idx, dtype, num_segments, g_out, x, y):
    return _mul_bwd_op(idx, dtype, num_segments, g_out, x, y), (idx, g_out, x, y)


def _mul_bwd_op_bwd(dtype, num_segments, residuals, cotangents):
    idx, g_out, x, y = residuals
    gg_gx, gg_gy = cotangents
    (grad_g_out, grad_x_extra, grad_y_extra) = _DBL_MUL[dtype](
        gg_gx,
        gg_gy,
        g_out,
        x,
        y,
        idx,
        output_dims={
            "grad_g_out": x.shape,
            "grad_x_extra": x.shape,
            "grad_y_extra": (num_segments,),
        },
    )
    return (jnp.zeros_like(idx), grad_g_out, grad_x_extra, grad_y_extra)


_mul_bwd_op.defvjp(_mul_bwd_op_fwd, _mul_bwd_op_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def _mul_op(x, y, idx, num_segments, dtype):
    (out,) = _FWD_MUL[dtype](x, y, idx, output_dims={"out": x.shape})
    return out


def _mul_op_fwd(x, y, idx, num_segments, dtype):
    return _mul_op(x, y, idx, num_segments, dtype), (x, y, idx)


def _mul_op_bwd(num_segments, dtype, residuals, g_out):
    x, y, idx = residuals
    grad_x, grad_y = _mul_bwd_op(idx, dtype, num_segments, g_out, x, y)
    return grad_x, grad_y, None


_mul_op.defvjp(_mul_op_fwd, _mul_op_bwd)


def segmented_mul(
    x: jax.Array, y: jax.Array, idx: jax.Array, num_segments: int
) -> jax.Array:
    """Differentiable per-element scale ``out[i] = x[i] * y[idx[i]]``."""
    return _mul_op(x, y, idx, num_segments, _norm_dtype(x.dtype))


# =============================================================================
# segmented_mean
# =============================================================================
# Forward: out[s] = mean(x[i] for i in segment s)
# Forward kernel also produces counts[s] which the backward needs.
# Bwd:     grad_x[i] = g_out[s] / counts[s]
# Dbl-bwd: grad_g_out[s] = sum_{i: idx[i]=s} gg_x[i] / counts[s]


def _make_mean_fwd(wp_dtype):
    is_vec = wp_dtype in (wp.vec3f, wp.vec3d)

    def fn(
        x: wp.array(dtype=wp_dtype),
        idx: wp.array(dtype=wp.int32),
        out: wp.array(dtype=wp_dtype),
        counts: wp.array(dtype=wp.int32),
    ):
        # ``counts`` is exposed as a kernel output so the JAX VJP can save it
        # as residual state without recomputing ``jnp.bincount`` on the host.
        num_segments = out.shape[0]
        sums = wp.zeros(num_segments, dtype=wp_dtype, device=x.device)
        _wp_segmented_sum(x, idx, sums)
        _wp_segmented_count(idx, counts)
        from nvalchemiops.segment_ops import (
            _segmented_vec_div_by_count_overloads,
        )

        if is_vec:
            wp.launch(
                _segmented_vec_div_by_count_overloads[wp_dtype],
                dim=num_segments,
                inputs=[sums, counts, out],
                device=x.device,
            )
        else:
            _wp_segment_div(sums, counts, out)

    return fn


def _make_mean_bwd(wp_dtype):
    def fn(
        g_out: wp.array(dtype=wp_dtype),
        counts: wp.array(dtype=wp.int32),
        idx: wp.array(dtype=wp.int32),
        grad_x: wp.array(dtype=wp_dtype),
    ):
        segmented_mean_backward(g_out, counts, idx, grad_x)

    return fn


def _make_mean_dbl_bwd(wp_dtype):
    def fn(
        gg_x: wp.array(dtype=wp_dtype),
        counts: wp.array(dtype=wp.int32),
        idx: wp.array(dtype=wp.int32),
        grad_g_out: wp.array(dtype=wp_dtype),
    ):
        segmented_mean_double_backward(gg_x, counts, idx, grad_g_out)

    return fn


_MEAN_WP = _SUM_WP
_FWD_MEAN = {
    k: jax_callable(_make_mean_fwd(v), num_outputs=2) for k, v in _MEAN_WP.items()
}
_BWD_MEAN = {
    k: jax_callable(_make_mean_bwd(v), num_outputs=1) for k, v in _MEAN_WP.items()
}
_DBL_MEAN = {
    k: jax_callable(_make_mean_dbl_bwd(v), num_outputs=1) for k, v in _MEAN_WP.items()
}


@partial(jax.custom_vjp, nondiff_argnums=(1, 2))
def _mean_bwd_op(idx, kind, num_segments, g_out, counts):
    n = idx.shape[0]
    out_shape = (n, 3) if kind[1] == "vec3" else (n,)
    (grad_x,) = _BWD_MEAN[kind](g_out, counts, idx, output_dims={"grad_x": out_shape})
    return grad_x


def _mean_bwd_op_fwd(idx, kind, num_segments, g_out, counts):
    return _mean_bwd_op(idx, kind, num_segments, g_out, counts), (idx, counts)


def _mean_bwd_op_bwd(kind, num_segments, residuals, gg_x):
    idx, counts = residuals
    out_shape = (num_segments, 3) if kind[1] == "vec3" else (num_segments,)
    (grad_g_out,) = _DBL_MEAN[kind](
        gg_x, counts, idx, output_dims={"grad_g_out": out_shape}
    )
    return (jnp.zeros_like(idx), grad_g_out, jnp.zeros_like(counts))


_mean_bwd_op.defvjp(_mean_bwd_op_fwd, _mean_bwd_op_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def _mean_op(x, idx, num_segments, kind):
    # Returns (out, counts) so the VJP can save ``counts`` as residual state
    # without recomputing it.  ``counts`` is int32 → JAX treats it as
    # non-differentiable automatically.
    out_shape = (num_segments, 3) if kind[1] == "vec3" else (num_segments,)
    out, counts = _FWD_MEAN[kind](
        x, idx, output_dims={"out": out_shape, "counts": (num_segments,)}
    )
    return out, counts


def _mean_op_fwd(x, idx, num_segments, kind):
    out, counts = _mean_op(x, idx, num_segments, kind)
    return (out, counts), (idx, counts)


def _mean_op_bwd(num_segments, kind, residuals, cotangents):
    # ``cotangents`` is the pair (g_out, g_counts); g_counts is the zero
    # JVP-tangent of an int32 output and is unused.
    g_out, _g_counts = cotangents
    idx, counts = residuals
    grad_x = _mean_bwd_op(idx, kind, num_segments, g_out, counts)
    return grad_x, None


_mean_op.defvjp(_mean_op_fwd, _mean_op_bwd)


def segmented_mean(x: jax.Array, idx: jax.Array, num_segments: int) -> jax.Array:
    """Differentiable per-segment mean."""
    out, _counts = _mean_op(x, idx, num_segments, _sum_kind(x))
    return out


# =============================================================================
# segmented_rms_norm   (vec3 only)
# =============================================================================
# Forward (precompute): out[s] = sqrt(mean(||x[i]||² for i in s))
#                       and saves inv_norm[s], counts[s] for the backward.
# Bwd:    grad_x[i] = g_out[s] * x[i] * inv_norm[s]
# Dbl-bwd: (see segmented_rms_norm_double_backward)


def _make_rms_fwd(vec_t):
    scalar_t = wp.float32 if vec_t is wp.vec3f else wp.float64

    def fn(
        x: wp.array(dtype=vec_t),
        idx: wp.array(dtype=wp.int32),
        out: wp.array(dtype=scalar_t),
        inv_norm: wp.array(dtype=scalar_t),
        counts: wp.array(dtype=wp.int32),
    ):
        # ``inv_norm`` and ``counts`` are exposed as kernel outputs so the JAX
        # VJP can save them as residual state without recomputing the divide
        # and the bincount on the host.
        num_segments = out.shape[0]
        sum_sq = wp.zeros(num_segments, dtype=scalar_t, device=x.device)
        segmented_rms_norm_forward_precompute(x, idx, sum_sq, counts, out, inv_norm)

    return fn


def _make_rms_bwd(vec_t):
    scalar_t = wp.float32 if vec_t is wp.vec3f else wp.float64

    def fn(
        g_out: wp.array(dtype=scalar_t),
        x: wp.array(dtype=vec_t),
        inv_norm: wp.array(dtype=scalar_t),
        idx: wp.array(dtype=wp.int32),
        grad_x: wp.array(dtype=vec_t),
    ):
        segmented_rms_norm_backward(g_out, x, inv_norm, idx, grad_x)

    return fn


def _make_rms_dbl_bwd(vec_t):
    scalar_t = wp.float32 if vec_t is wp.vec3f else wp.float64

    def fn(
        gg_x: wp.array(dtype=vec_t),
        x: wp.array(dtype=vec_t),
        g_out: wp.array(dtype=scalar_t),
        inv_norm: wp.array(dtype=scalar_t),
        counts: wp.array(dtype=wp.int32),
        idx: wp.array(dtype=wp.int32),
        grad_x_extra: wp.array(dtype=vec_t),
        grad_g_out_extra: wp.array(dtype=scalar_t),
    ):
        segmented_rms_norm_double_backward(
            gg_x,
            x,
            g_out,
            inv_norm,
            counts,
            idx,
            grad_g_out_extra.shape[0],
            grad_x_extra,
            grad_g_out_extra,
        )

    return fn


_RMS_WP = {_F: wp.vec3f, _D: wp.vec3d}
_FWD_RMS = {
    k: jax_callable(_make_rms_fwd(v), num_outputs=3) for k, v in _RMS_WP.items()
}
_BWD_RMS = {
    k: jax_callable(_make_rms_bwd(v), num_outputs=1) for k, v in _RMS_WP.items()
}
_DBL_RMS = {
    k: jax_callable(_make_rms_dbl_bwd(v), num_outputs=2) for k, v in _RMS_WP.items()
}


@partial(jax.custom_vjp, nondiff_argnums=(1, 2))
def _rms_bwd_op(idx, dtype, num_segments, g_out, x, inv_norm, counts):
    (grad_x,) = _BWD_RMS[dtype](
        g_out,
        x,
        inv_norm,
        idx,
        output_dims={"grad_x": x.shape},
    )
    return grad_x


def _rms_bwd_op_fwd(idx, dtype, num_segments, g_out, x, inv_norm, counts):
    return _rms_bwd_op(idx, dtype, num_segments, g_out, x, inv_norm, counts), (
        idx,
        g_out,
        x,
        inv_norm,
        counts,
    )


def _rms_bwd_op_bwd(dtype, num_segments, residuals, gg_x):
    idx, g_out, x, inv_norm, counts = residuals
    (grad_x_extra, grad_g_out_extra) = _DBL_RMS[dtype](
        gg_x,
        x,
        g_out,
        inv_norm,
        counts,
        idx,
        output_dims={"grad_x_extra": x.shape, "grad_g_out_extra": (num_segments,)},
    )
    return (
        jnp.zeros_like(idx),
        grad_g_out_extra,
        grad_x_extra,
        jnp.zeros_like(inv_norm),
        jnp.zeros_like(counts),
    )


_rms_bwd_op.defvjp(_rms_bwd_op_fwd, _rms_bwd_op_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def _rms_op(x, idx, num_segments, dtype):
    # Returns (out, inv_norm, counts) so the VJP can save the precompute
    # state as residuals.  ``counts`` is int32 and ``inv_norm`` is the
    # already-saved-state slot from the underlying Warp precompute kernel
    # — JAX treats counts as non-differentiable automatically, and we
    # don't differentiate inv_norm because it's bookkeeping for the bwd.
    out, inv_norm, counts = _FWD_RMS[dtype](
        x,
        idx,
        output_dims={
            "out": (num_segments,),
            "inv_norm": (num_segments,),
            "counts": (num_segments,),
        },
    )
    return out, inv_norm, counts


def _rms_op_fwd(x, idx, num_segments, dtype):
    out, inv_norm, counts = _rms_op(x, idx, num_segments, dtype)
    return (out, inv_norm, counts), (idx, x, inv_norm, counts)


def _rms_op_bwd(num_segments, dtype, residuals, cotangents):
    # ``cotangents`` is the triple (g_out, g_inv_norm, g_counts); only
    # g_out is meaningful (the other two are saved state).
    g_out, _g_inv_norm, _g_counts = cotangents
    idx, x, inv_norm, counts = residuals
    grad_x = _rms_bwd_op(idx, dtype, num_segments, g_out, x, inv_norm, counts)
    return grad_x, None


_rms_op.defvjp(_rms_op_fwd, _rms_op_bwd)


def segmented_rms_norm(x: jax.Array, idx: jax.Array, num_segments: int) -> jax.Array:
    """Differentiable per-segment RMS norm."""
    out, _inv_norm, _counts = _rms_op(x, idx, num_segments, _norm_dtype(x.dtype))
    return out


# =============================================================================
# segmented_matvec
# =============================================================================
# Forward:  out[i] = m[idx[i]]^T @ v[i]
# Bwd:      grad_v[i] = num_segments[s] @ g_out[i]
#           grad_M[s] = sum_{i: idx[i]=s} outer(v[i], g_out[i])
# Dbl-bwd:  grad_g_out[i]   = num_segments[s]^T @ gg_gv[i] + gg_gM[s]^T @ v[i]
#           grad_v_extra[i] = gg_gM[s] @ g_out[i]
#           grad_M_extra[s] = sum outer(gg_gv[i], g_out[i])


def _make_matvec_fwd(vec_t, mat_t):
    def fn(
        v: wp.array(dtype=vec_t),
        m: wp.array(dtype=mat_t),
        idx: wp.array(dtype=wp.int32),
        out: wp.array(dtype=vec_t),
    ):
        _wp_segmented_matvec(v, m, idx, out)

    return fn


def _make_matvec_bwd(vec_t, mat_t):
    def fn(
        g_out: wp.array(dtype=vec_t),
        v: wp.array(dtype=vec_t),
        m: wp.array(dtype=mat_t),
        idx: wp.array(dtype=wp.int32),
        grad_v: wp.array(dtype=vec_t),
        grad_M: wp.array(dtype=mat_t),
    ):
        segmented_matvec_backward(g_out, v, m, idx, grad_v, grad_M)

    return fn


def _make_matvec_dbl_bwd(vec_t, mat_t):
    def fn(
        gg_gv: wp.array(dtype=vec_t),
        gg_gM: wp.array(dtype=mat_t),
        g_out: wp.array(dtype=vec_t),
        v: wp.array(dtype=vec_t),
        m: wp.array(dtype=mat_t),
        idx: wp.array(dtype=wp.int32),
        grad_g_out: wp.array(dtype=vec_t),
        grad_v_extra: wp.array(dtype=vec_t),
        grad_M_extra: wp.array(dtype=mat_t),
    ):
        segmented_matvec_double_backward(
            gg_gv,
            gg_gM,
            g_out,
            v,
            m,
            idx,
            grad_g_out,
            grad_v_extra,
            grad_M_extra,
        )

    return fn


_MAT_WP = {_F: (wp.vec3f, wp.mat33f), _D: (wp.vec3d, wp.mat33d)}
_FWD_MAT = {
    k: jax_callable(_make_matvec_fwd(*v), num_outputs=1) for k, v in _MAT_WP.items()
}
_BWD_MAT = {
    k: jax_callable(_make_matvec_bwd(*v), num_outputs=2) for k, v in _MAT_WP.items()
}
_DBL_MAT = {
    k: jax_callable(_make_matvec_dbl_bwd(*v), num_outputs=3) for k, v in _MAT_WP.items()
}


@partial(jax.custom_vjp, nondiff_argnums=(1, 2))
def _matvec_bwd_op(idx, dtype, num_segments, g_out, v, m):
    (grad_v, grad_M) = _BWD_MAT[dtype](
        g_out,
        v,
        m,
        idx,
        output_dims={"grad_v": v.shape, "grad_M": m.shape},
    )
    return grad_v, grad_M


def _matvec_bwd_op_fwd(idx, dtype, num_segments, g_out, v, m):
    return _matvec_bwd_op(idx, dtype, num_segments, g_out, v, m), (idx, g_out, v, m)


def _matvec_bwd_op_bwd(dtype, num_segments, residuals, cotangents):
    idx, g_out, v, m = residuals
    gg_gv, gg_gM = cotangents
    (grad_g_out, grad_v_extra, grad_M_extra) = _DBL_MAT[dtype](
        gg_gv,
        gg_gM,
        g_out,
        v,
        m,
        idx,
        output_dims={
            "grad_g_out": v.shape,
            "grad_v_extra": v.shape,
            "grad_M_extra": m.shape,
        },
    )
    return (jnp.zeros_like(idx), grad_g_out, grad_v_extra, grad_M_extra)


_matvec_bwd_op.defvjp(_matvec_bwd_op_fwd, _matvec_bwd_op_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def _matvec_op(v, m, idx, num_segments, dtype):
    (out,) = _FWD_MAT[dtype](v, m, idx, output_dims={"out": v.shape})
    return out


def _matvec_op_fwd(v, m, idx, num_segments, dtype):
    return _matvec_op(v, m, idx, num_segments, dtype), (v, m, idx)


def _matvec_op_bwd(num_segments, dtype, residuals, g_out):
    v, m, idx = residuals
    grad_v, grad_M = _matvec_bwd_op(idx, dtype, num_segments, g_out, v, m)
    return grad_v, grad_M, None


_matvec_op.defvjp(_matvec_op_fwd, _matvec_op_bwd)


def segmented_matvec(
    v: jax.Array, m: jax.Array, idx: jax.Array, num_segments: int
) -> jax.Array:
    """Differentiable per-segment matvec ``out[i] = m[idx[i]]^T @ v[i]``."""
    return _matvec_op(v, m, idx, num_segments, _norm_dtype(v.dtype))
