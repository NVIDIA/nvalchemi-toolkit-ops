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

"""
Segment Op Autograd: First- and Second-Order Backward
=====================================================

This example exercises the explicit backward and double-backward kernels in
``nvalchemiops.segment_ops_backward`` through the public Torch wrappers in
``nvalchemiops.torch.segment_ops``.

For each of the six differentiable segment ops we

1. run the forward pass,
2. call ``torch.autograd.grad(..., create_graph=True)`` to invoke the
   first-order backward kernel,
3. differentiate the result once more to invoke the double-backward kernel.

Inputs are tiny (``N = 12`` elements over ``M = 3`` segments) so the printed
gradients can be eyeballed.  The script runs on CUDA if available, otherwise
CPU.

Ops covered
-----------
- ``segmented_sum``       scatter-add: ``out[s] = sum_{i: idx[i]==s} x[i]``
- ``segmented_dot``       per-segment dot product
- ``segmented_mul``       per-element scale by a per-segment scalar
- ``segmented_mean``      per-segment mean
- ``segmented_rms_norm``  per-segment RMS norm of vec3 inputs
- ``segmented_matvec``    per-segment matvec ``out[i] = M[idx[i]]^T @ v[i]``
"""

from __future__ import annotations

import torch

from nvalchemiops.torch.segment_ops import (
    segmented_dot,
    segmented_matvec,
    segmented_mean,
    segmented_mul,
    segmented_rms_norm,
    segmented_sum,
)

# %%
# Problem setup
# -------------
# A sorted segment index of length ``N`` mapping into ``M`` segments. The
# library expects ``idx`` to be int32 and sorted in non-decreasing order.

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

N, M = 12, 3
idx = torch.tensor(
    [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2], dtype=torch.int32, device=device
)


def _leaf(shape, *, dtype=torch.float32):
    """Fresh leaf tensor with ``requires_grad=True``."""
    return torch.randn(shape, dtype=dtype, device=device, requires_grad=True)


def demo(name, forward, leaves):
    """Run forward, first-order backward, and double-backward for one op.

    ``leaves`` is the tuple of floating-point inputs whose gradients we want.
    ``forward`` is a zero-arg closure that calls the op on those leaves (it
    may also capture non-differentiable arguments like ``idx`` and ``M``).
    """
    print(f"\n=== {name} ===")
    out = forward()
    print(f"  forward out shape  : {tuple(out.shape)}")

    # --- 1st-order backward -------------------------------------------------
    # An external ``grad_outputs`` tensor lets the double-backward depend on
    # it, which is what triggers ``_launch_*_double_backward``.
    g_out = torch.randn_like(out, requires_grad=True)
    grads = torch.autograd.grad(out, leaves, grad_outputs=g_out, create_graph=True)
    for leaf, g in zip(leaves, grads):
        print(f"  d(out·g_out)/d{tuple(leaf.shape)} : norm={g.norm().item():.4f}")

    # --- 2nd-order backward -------------------------------------------------
    # Reduce ``grads`` to a scalar and differentiate w.r.t. ``g_out`` and the
    # original leaves. For linear ops (sum, mean, broadcast) the leaf-leaf
    # cross term is zero and only ``d/d g_out`` is non-trivial. For non-linear
    # ops (rms_norm, dot, mul, matvec) the leaf-leaf term is also non-zero.
    scalar = sum(g.sum() for g in grads)
    second = torch.autograd.grad(
        scalar, (g_out, *leaves), allow_unused=True, retain_graph=False
    )
    print(
        f"  d²/d g_out shape   : {tuple(second[0].shape)}  "
        f"norm={second[0].norm().item():.4f}"
    )
    for leaf, s in zip(leaves, second[1:]):
        tag = "zero" if s is None else f"norm={s.norm().item():.4f}"
        print(f"  d²/d{tuple(leaf.shape)} cross : {tag}")


# %%
# segmented_sum
# -------------
# Linear in ``x``, so the double-backward only feeds ``g_out``.

x = _leaf((N,))
demo("segmented_sum (scalar)", lambda: segmented_sum(x, idx, M), (x,))

xv = _leaf((N, 3))
demo("segmented_sum (vec3)", lambda: segmented_sum(xv, idx, M), (xv,))

# %%
# segmented_dot
# -------------
# Bilinear in ``(x, y)``: backward mixes the two, and the double-backward has
# non-zero cross terms with respect to both inputs.

x = _leaf((N, 3))
y = _leaf((N, 3))
demo("segmented_dot", lambda: segmented_dot(x, y, idx, M), (x, y))

# %%
# segmented_mul
# -------------
# ``out[i] = x[i] * y[idx[i]]``.  ``x`` is per-element, ``y`` is per-segment.

x = _leaf((N, 3))
y = _leaf((M,))
demo("segmented_mul", lambda: segmented_mul(x, y, idx, M), (x, y))

# %%
# segmented_mean
# --------------
# Linear in ``x``; division by the per-segment count is folded into the
# backward kernel (counts are cached in the autograd context).

x = _leaf((N, 3))
demo("segmented_mean", lambda: segmented_mean(x, idx, M), (x,))

# %%
# segmented_rms_norm
# ------------------
# Non-linear: backward uses a cached ``inv_norm`` precomputed during forward,
# and the double-backward returns the projection-onto-tangent term.

x = _leaf((N, 3))
demo("segmented_rms_norm", lambda: segmented_rms_norm(x, idx, M), (x,))

# %%
# segmented_matvec
# ----------------
# Bilinear: ``out[i] = m[idx[i]]^T @ v[i]``. The double-backward has both a
# ``grad_v_extra`` term and a ``grad_m_extra`` term (see the kernel docstring
# in ``segment_ops_backward.py``).

v = _leaf((N, 3))
m = _leaf((M, 3, 3))
demo("segmented_matvec", lambda: segmented_matvec(v, m, idx, M), (v, m))

print("\nAll forward, backward, and double-backward calls completed.")
