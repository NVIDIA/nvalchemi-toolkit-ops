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
FIRE2 Optimizer Kernels
=======================

GPU-accelerated Warp kernels for the FIRE2 (Fast Inertial Relaxation
Engine v2) geometry optimizer.

This module provides three highly-fused kernels that implement a complete
FIRE2 step in only **3 kernel launches**, minimizing Python-side and
launch overhead.

FIRE2 ALGORITHM (Guenole et al., 2020)
=======================================

Given positions *r*, velocities *v*, and forces *f*:

1. Half-step velocity update:  v += f * dt
2. Compute power:  P = sum(v . f)  per system
3. Adaptive parameter update:
   - If P > 0: increment counter, optionally grow dt, shrink alpha
   - If P <= 0: reset counter, shrink dt, reset alpha
4. Velocity mixing:
   v = (1 - alpha) * v + alpha * sqrt(v.v / f.f) * f
5. Compute step:  step = v * dt
6. Uphill correction:
   if P <= 0:  step = -0.5 * dt * v_mixed;  v = 0
7. Step clamping + position update + coupled dt scaling

KERNEL STRUCTURE
================

Kernel 1 (_fire2_reduce_only):
    Runs-based triple inner-product reduction (vf, v.v, f.f) with
    deferred half-step computed in registers only (no velocity write).

Kernel 2 (_fire2_fused_mix_maxnorm):
    Fuses per-system parameter update, deferred half-step, velocity
    mixing, and runs-based max-norm reduction into a single launch.
    Each thread redundantly computes the parameter update for its
    segment from shared read-only inputs, avoiding inter-thread
    synchronization.

Kernel 3 (_fire2_clamp_apply_recompute):
    Recomputes step from mixed velocities, applies step clamping,
    position update, deferred velocity zeroing for uphill systems,
    and coupled dt scaling.

REFERENCES
==========

- Guenole et al. (2020). Comp. Mat. Sci. 175, 109584 (FIRE2)
- Bitzek et al. (2006). Phys. Rev. Lett. 97, 170201 (FIRE)
"""

from __future__ import annotations

import os
from typing import Any

import warp as wp

from nvalchemiops.dynamics.utils.kernel_functions import compute_vf_vv_ff
from nvalchemiops.segment_ops import compute_ept

# =============================================================================
# Kernel 1: Triple inner-product reduction (deferred half-step)
# =============================================================================

# Tile block size for cooperative reductions
TILE_DIM = int(os.getenv("NVALCHEMIOPS_DYNAMICS_TILE_DIM", 256))


@wp.kernel(enable_backward=False)
def _fire2_reduce_only_kernel(
    velocities: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    dt: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    vf: wp.array(dtype=Any),
    v_sumsq: wp.array(dtype=Any),
    f_sumsq: wp.array(dtype=Any),
    N: wp.int32,
    elems_per_thread: wp.int32,
):
    """Triple inner-product reduction with deferred velocity half-step.

    Computes three inner products per segment without modifying velocities:
    - ``vf[s] = sum(dot(v_upd[i], f[i]) for i where batch_idx[i] == s)``
    - ``v_sumsq[s] = sum(dot(v_upd[i], v_upd[i]) for i where batch_idx[i] == s)``
    - ``f_sumsq[s] = sum(dot(f[i], f[i]) for i where batch_idx[i] == s)``

    where ``v_upd[i] = velocities[i] + forces[i] * dt[batch_idx[i]]``.

    The half-step velocity update is computed in registers only and NOT written
    back to the velocities array. This deferred write is performed by the
    subsequent fused mixing kernel, which algebraically combines the half-step
    with the velocity mixing operation.

    Launch Grid
    -----------
    dim = ceil(N / elems_per_thread)

    Parameters
    ----------
    velocities : wp.array, shape (N,), dtype vec3f/vec3d
        Atomic velocities, read-only (not modified by this kernel).
    forces : wp.array, shape (N,), dtype vec3f/vec3d
        Forces on atoms.
    dt : wp.array, shape (M,), dtype float32/float64
        Per-system timestep (scalar dtype matching vector precision).
    batch_idx : wp.array, shape (N,), dtype int32
        Sorted system index per atom in [0, M).
    vf : wp.array, shape (M,), dtype float32/float64
        OUTPUT: v_upd·f per segment. Zeroed internally before each use.
    v_sumsq : wp.array, shape (M,), dtype float32/float64
        OUTPUT: v_upd·v_upd per segment. Zeroed internally before each use.
    f_sumsq : wp.array, shape (M,), dtype float32/float64
        OUTPUT: f·f per segment. Zeroed internally before each use.
    N : int32
        Total number of atoms.
    elems_per_thread : int32
        Elements processed per thread (auto-tuned based on array size and SM count).

    Notes
    -----
    - batch_idx must be sorted in non-decreasing order for correctness
    - Uses run-length encoding to minimize atomic operations
    - Part of the FIRE2 3-kernel optimization strategy
    - The deferred half-step approach avoids an intermediate velocity write
    """
    t = wp.tid()
    start = t * elems_per_thread
    if start >= N:
        return
    end = wp.min(start + elems_per_thread, N)

    # First element -- compute v_upd in register, do NOT write back
    s_cur = batch_idx[start]
    v_upd = velocities[start] + forces[start] * dt[s_cur]
    acc_vf, acc_vv, acc_ff = compute_vf_vv_ff(v_upd, forces[start])

    for i in range(start + 1, end):
        s = batch_idx[i]
        v_upd = velocities[i] + forces[i] * dt[s]
        val_vf, val_vv, val_ff = compute_vf_vv_ff(v_upd, forces[i])
        if s == s_cur:
            acc_vf = acc_vf + val_vf
            acc_vv = acc_vv + val_vv
            acc_ff = acc_ff + val_ff
        else:
            wp.atomic_add(vf, s_cur, acc_vf)
            wp.atomic_add(v_sumsq, s_cur, acc_vv)
            wp.atomic_add(f_sumsq, s_cur, acc_ff)
            s_cur = s
            acc_vf = val_vf
            acc_vv = val_vv
            acc_ff = val_ff

    wp.atomic_add(vf, s_cur, acc_vf)
    wp.atomic_add(v_sumsq, s_cur, acc_vv)
    wp.atomic_add(f_sumsq, s_cur, acc_ff)


@wp.kernel(enable_backward=False)
def _fire2_reduce_only_tiled_kernel(
    velocities: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    dt: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    vf: wp.array(dtype=Any),
    v_sumsq: wp.array(dtype=Any),
    f_sumsq: wp.array(dtype=Any),
):
    """Triple inner-product reduction with tile reductions (per-atom processing).

    Computes three inner products per system using block-level tile reductions:
    - vf[s] = sum(dot(v_upd[i], f[i]) for i where batch_idx[i] == s)
    - v_sumsq[s] = sum(dot(v_upd[i], v_upd[i]) for i where batch_idx[i] == s)
    - f_sumsq[s] = sum(dot(f[i], f[i]) for i where batch_idx[i] == s)

    where v_upd[i] = velocities[i] + forces[i] * dt[batch_idx[i]].

    Launch Grid: dim = [N_atoms], block_dim = TILE_DIM

    Notes
    -----
    - Simpler per-atom processing (no RLE complexity)
    - Uses wp.tile() and wp.tile_sum() for cooperative reduction
    - Reduces atomics from N to N/TILE_DIM per system
    """
    atom_idx = wp.tid()
    system_id = batch_idx[atom_idx]

    # Compute deferred half-step in register only
    v_upd = velocities[atom_idx] + forces[atom_idx] * dt[system_id]

    # Compute local contributions
    local_vf, local_vv, local_ff = compute_vf_vv_ff(v_upd, forces[atom_idx])

    # Convert to tiles for block-level reduction
    t_vf = wp.tile(local_vf)
    t_vv = wp.tile(local_vv)
    t_ff = wp.tile(local_ff)

    # Cooperative sum within block
    s_vf = wp.tile_sum(t_vf)
    s_vv = wp.tile_sum(t_vv)
    s_ff = wp.tile_sum(t_ff)

    # Extract scalar values from tile sums
    sum_vf = s_vf[0]
    sum_vv = s_vv[0]
    sum_ff = s_ff[0]

    # Only first thread in block writes (3 atomics per block)
    if atom_idx % TILE_DIM == 0:
        wp.atomic_add(vf, system_id, sum_vf)
        wp.atomic_add(v_sumsq, system_id, sum_vv)
        wp.atomic_add(f_sumsq, system_id, sum_ff)


# =============================================================================
# Kernel 2: Fused param update + deferred halfstep + mix + max-norm
# =============================================================================


@wp.kernel(enable_backward=False)
def _fire2_fused_mix_maxnorm_kernel(
    velocities: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    dt: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    vf: wp.array(dtype=Any),
    v_sumsq: wp.array(dtype=Any),
    f_sumsq: wp.array(dtype=Any),
    alpha: wp.array(dtype=Any),
    nsteps_inc: wp.array(dtype=wp.int32),
    max_norm: wp.array(dtype=Any),
    N: wp.int32,
    elems_per_thread: wp.int32,
    delaystep: wp.int32,
    dtgrow: Any,
    dtshrink: Any,
    alphashrink: Any,
    alpha0: Any,
    tmax: Any,
    tmin: Any,
):
    """Fused adaptive parameter update, deferred half-step, velocity mixing, and max-norm reduction.

    This kernel performs four operations in a single launch:

    1. **Adaptive parameter update** (per-segment, redundantly computed):
       - If ``vf[s] > 0`` (downhill): increment counter, optionally grow dt, shrink alpha
       - If ``vf[s] <= 0`` (uphill): reset counter, shrink dt, reset alpha

    2. **Deferred half-step + velocity mixing** (algebraically combined):
       ``v[i] = mix_a * v[i] + (mix_a * dt_old + mix_b) * f[i]``
       where ``mix_a = 1 - alpha``, ``mix_b = alpha * sqrt(v·v / f·f)``

    3. **State updates** (first atom per segment writes):
       Updates ``alpha[s]``, ``dt[s]``, and ``nsteps_inc[s]``

    4. **Max-norm reduction** (run-length encoded):
       Computes ``max_norm[s] = max(||step[i]|| for i where batch_idx[i] == s)``
       where step depends on uphill/downhill status

    Each thread redundantly computes the per-system parameter update from shared
    read-only inputs (vf, v_sumsq, f_sumsq), avoiding inter-thread synchronization.

    Launch Grid
    -----------
    dim = ceil(N / elems_per_thread)

    Parameters
    ----------
    velocities : wp.array, shape (N,), dtype vec3f/vec3d
        Atomic velocities, modified in-place. Must hold pre-halfstep values
        (kernel 1 did not modify them).
    forces : wp.array, shape (N,), dtype vec3f/vec3d
        Forces on atoms (read-only).
    dt : wp.array, shape (M,), dtype float32/float64
        Per-system timestep. Modified in-place by first atom per segment.
    batch_idx : wp.array, shape (N,), dtype int32
        Sorted system index per atom in [0, M).
    vf : wp.array, shape (M,), dtype float32/float64
        v·f inner product per segment from kernel 1 (read-only).
    v_sumsq : wp.array, shape (M,), dtype float32/float64
        v·v inner product per segment from kernel 1 (read-only).
    f_sumsq : wp.array, shape (M,), dtype float32/float64
        f·f inner product per segment from kernel 1 (read-only).
    alpha : wp.array, shape (M,), dtype float32/float64
        FIRE2 mixing parameter. Modified in-place by first atom per segment.
    nsteps_inc : wp.array, shape (M,), dtype int32
        Consecutive positive-power step counter. Modified by first atom per segment.
    max_norm : wp.array, shape (M,), dtype float32/float64
        OUTPUT: Maximum step norm per segment. Zeroed internally before each use.
    N : int32
        Total number of atoms.
    elems_per_thread : int32
        Elements processed per thread (auto-tuned based on array size).
    delaystep : int32
        Minimum consecutive positive steps before dt growth.
    dtgrow : float32/float64
        Timestep growth factor (typically 1.05).
    dtshrink : float32/float64
        Timestep shrink factor (typically 0.75).
    alphashrink : float32/float64
        Alpha decay factor (typically 0.985).
    alpha0 : float32/float64
        Alpha reset value (typically 0.09).
    tmax : float32/float64
        Maximum allowed timestep.
    tmin : float32/float64
        Minimum allowed timestep.

    Notes
    -----
    - batch_idx must be sorted in non-decreasing order
    - Only the first atom in each segment writes to alpha, dt, nsteps_inc
    - Parameter updates are computed redundantly by each thread to avoid synchronization
    - The algebraic combination of half-step and mixing eliminates one velocity write
    - For uphill systems (vf <= 0), step norm uses factor -0.5 for the correction
    """
    t = wp.tid()
    start = t * elems_per_thread
    if start >= N:
        return
    end = wp.min(start + elems_per_thread, N)

    s_cur = batch_idx[start]

    # --- Redundant param-update computation for the first segment ---
    _vf = vf[s_cur]
    _vv = v_sumsq[s_cur]
    _ff = f_sumsq[s_cur]
    _a = alpha[s_cur]
    _dt = dt[s_cur]
    dt_old = _dt  # pre-update dt for the deferred half-step

    zero = type(_dt)(0.0)
    one = type(_dt)(1.0)
    w_inc = _vf > zero

    if w_inc:
        _nsi = nsteps_inc[s_cur] + 1
        if _nsi > delaystep:
            _dt = wp.min(dtgrow * _dt, tmax)
            _a = alphashrink * _a
    else:
        _nsi = 0
        _a = alpha0
        _dt = wp.max(dtshrink * _dt, tmin)

    # First atom per segment writes updated params
    if start == 0 or batch_idx[start - 1] != s_cur:
        alpha[s_cur] = _a
        dt[s_cur] = _dt
        nsteps_inc[s_cur] = _nsi

    if _ff > zero:
        ratio = wp.sqrt(_vv / _ff)
    else:
        ratio = zero
    mix_a = one - _a
    mix_b = _a * ratio
    w_dec = not w_inc

    # --- Process first element: deferred halfstep + mix (algebraic combo) ---
    f_coeff = mix_a * dt_old + mix_b
    velocities[start] = mix_a * velocities[start] + f_coeff * forces[start]
    if w_dec:
        max_val = wp.length(type(_dt)(-0.5) * _dt * velocities[start])
    else:
        max_val = wp.length(_dt * velocities[start])

    for i in range(start + 1, end):
        s = batch_idx[i]
        if s != s_cur:
            # Flush max_norm for previous segment
            wp.atomic_max(max_norm, s_cur, max_val)
            s_cur = s

            # --- Redundant param-update computation for new segment ---
            _vf = vf[s]
            _vv = v_sumsq[s]
            _ff = f_sumsq[s]
            _a = alpha[s]
            _dt = dt[s]
            dt_old = _dt

            w_inc = _vf > zero
            if w_inc:
                _nsi = nsteps_inc[s] + 1
                if _nsi > delaystep:
                    _dt = wp.min(dtgrow * _dt, tmax)
                    _a = alphashrink * _a
            else:
                _nsi = 0
                _a = alpha0
                _dt = wp.max(dtshrink * _dt, tmin)

            if batch_idx[i - 1] != s:
                alpha[s] = _a
                dt[s] = _dt
                nsteps_inc[s] = _nsi

            if _ff > zero:
                ratio = wp.sqrt(_vv / _ff)
            else:
                ratio = zero
            mix_a = one - _a
            mix_b = _a * ratio
            w_dec = not w_inc
            f_coeff = mix_a * dt_old + mix_b
            max_val = type(_dt)(0.0)

        # Deferred halfstep + mix (algebraic combo)
        velocities[i] = mix_a * velocities[i] + f_coeff * forces[i]
        if w_dec:
            norm = wp.length(type(_dt)(-0.5) * _dt * velocities[i])
        else:
            norm = wp.length(_dt * velocities[i])
        max_val = wp.max(max_val, norm)

    wp.atomic_max(max_norm, s_cur, max_val)


# =============================================================================
# Kernel 3: Step recompute + clamping + position update + velocity zeroing
# =============================================================================


@wp.kernel(enable_backward=False)
def _fire2_clamp_apply_recompute_kernel(
    positions: wp.array(dtype=Any),
    velocities: wp.array(dtype=Any),
    dt: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    max_norm: wp.array(dtype=Any),
    vf: wp.array(dtype=Any),
    maxstep: Any,
):
    """Step recomputation, clamping, position update, velocity zeroing, and coupled dt scaling.

    This kernel performs the final operations of the FIRE2 step:

    1. **Step recomputation** from mixed velocities (avoids storing step buffer)
    2. **Uphill correction**: For ``vf[s] <= 0``, applies ``step = -0.5 * dt * v``
    3. **Step clamping**: Scales step by ``min(1.0, maxstep / max_norm[s])``
    4. **Position update**: ``positions[i] += clamped_step``
    5. **Velocity zeroing**: Sets ``velocities[i] = 0`` for uphill systems
    6. **Coupled dt scaling**: Scales ``dt[s]`` by the same clamping factor

    The algorithm:
    ```
    local_dt = dt[s]  (snapshot before any thread modifies it)
    inv = min(1.0, maxstep / max_norm[s])
    if vf[s] <= 0:  # uphill
        step = -0.5 * local_dt * v[i]
        v[i] = 0
    else:  # downhill
        step = local_dt * v[i]
    positions[i] += step * inv
    if first_atom_in_segment:
        dt[s] = local_dt * inv
    ```

    Launch Grid
    -----------
    dim = N (total atoms)

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype vec3f/vec3d
        Atomic positions, modified in-place.
    velocities : wp.array, shape (N,), dtype vec3f/vec3d
        Atomic velocities, modified in-place (zeroed for uphill systems).
    dt : wp.array, shape (M,), dtype float32/float64
        Per-system timestep, modified in-place by first atom per segment
        (clamped proportionally to step scaling).
    batch_idx : wp.array, shape (N,), dtype int32
        Sorted system index per atom in [0, M).
    max_norm : wp.array, shape (M,), dtype float32/float64
        Maximum step norm per segment from kernel 2.
    vf : wp.array, shape (M,), dtype float32/float64
        v·f inner product per segment from kernel 1. System is uphill if vf[s] <= 0.
    maxstep : float32/float64
        Maximum allowed step size (FIRE2 hyperparameter).

    Notes
    -----
    - Each thread reads dt[s] before any thread writes to avoid race conditions
    - Only the first atom in each segment (batch_idx[i-1] != batch_idx[i]) writes dt[s]
    - Velocity zeroing for uphill systems is deferred to this kernel for efficiency
    - Coupled dt scaling ensures consistency between step size and timestep
    - The -0.5 factor for uphill correction is part of the FIRE2 algorithm
    """
    tid = wp.tid()
    s = batch_idx[tid]
    # Snapshot dt before any thread writes to it (race-condition guard)
    local_dt = dt[s]
    mn = max_norm[s]
    inv = wp.min(type(mn)(1.0), maxstep / mn)

    if vf[s] <= type(mn)(0.0):
        local_step = type(mn)(-0.5) * local_dt * velocities[tid]
        velocities[tid] = type(velocities[tid])()
    else:
        local_step = local_dt * velocities[tid]

    positions[tid] = positions[tid] + local_step * inv
    # Only first atom in segment updates dt (idx is sorted)
    if tid == 0 or batch_idx[tid - 1] != s:
        dt[s] = local_dt * inv


# =============================================================================
# Overloads
# =============================================================================

_T = [wp.float32, wp.float64]
_V = [wp.vec3f, wp.vec3d]

_fire2_reduce_only_overloads = {}
_fire2_reduce_only_tiled_overloads = {}
_fire2_fused_mix_maxnorm_overloads = {}
_fire2_clamp_apply_recompute_overloads = {}

for _t, _v in zip(_T, _V):
    _fire2_reduce_only_overloads[_v] = wp.overload(
        _fire2_reduce_only_kernel,
        [
            wp.array(dtype=_v),  # velocities
            wp.array(dtype=_v),  # forces
            wp.array(dtype=_t),  # dt
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array(dtype=_t),  # vf
            wp.array(dtype=_t),  # v_sumsq
            wp.array(dtype=_t),  # f_sumsq
            wp.int32,  # N
            wp.int32,  # elems_per_thread
        ],
    )

    _fire2_reduce_only_tiled_overloads[_v] = wp.overload(
        _fire2_reduce_only_tiled_kernel,
        [
            wp.array(dtype=_v),  # velocities
            wp.array(dtype=_v),  # forces
            wp.array(dtype=_t),  # dt
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array(dtype=_t),  # vf
            wp.array(dtype=_t),  # v_sumsq
            wp.array(dtype=_t),  # f_sumsq
        ],
    )

    _fire2_fused_mix_maxnorm_overloads[_v] = wp.overload(
        _fire2_fused_mix_maxnorm_kernel,
        [
            wp.array(dtype=_v),  # velocities
            wp.array(dtype=_v),  # forces
            wp.array(dtype=_t),  # dt
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array(dtype=_t),  # vf
            wp.array(dtype=_t),  # v_sumsq
            wp.array(dtype=_t),  # f_sumsq
            wp.array(dtype=_t),  # alpha
            wp.array(dtype=wp.int32),  # nsteps_inc
            wp.array(dtype=_t),  # max_norm
            wp.int32,  # N
            wp.int32,  # elems_per_thread
            wp.int32,  # delaystep
            _t,  # dtgrow
            _t,  # dtshrink
            _t,  # alphashrink
            _t,  # alpha0
            _t,  # tmax
            _t,  # tmin
        ],
    )

    _fire2_clamp_apply_recompute_overloads[_v] = wp.overload(
        _fire2_clamp_apply_recompute_kernel,
        [
            wp.array(dtype=_v),  # positions
            wp.array(dtype=_v),  # velocities
            wp.array(dtype=_t),  # dt
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array(dtype=_t),  # max_norm
            wp.array(dtype=_t),  # vf (v.f inner product)
            _t,  # maxstep
        ],
    )


# =============================================================================
# Public API
# =============================================================================


def fire2_step(
    # Per-atom arrays (N,)
    positions: wp.array,
    velocities: wp.array,
    forces: wp.array,
    batch_idx: wp.array,
    # Per-system state (M,)
    alpha: wp.array,
    dt: wp.array,
    nsteps_inc: wp.array,
    # Scratch buffers (M,)
    vf: wp.array,
    v_sumsq: wp.array,
    f_sumsq: wp.array,
    max_norm: wp.array,
    # Hyperparameters (Python scalars)
    delaystep: int = 60,
    dtgrow: float = 1.05,
    dtshrink: float = 0.75,
    alphashrink: float = 0.985,
    alpha0: float = 0.09,
    tmax: float = 0.08,
    tmin: float = 0.005,
    maxstep: float = 0.1,
) -> None:
    """Complete FIRE2 optimization step.

    Modifies *positions*, *velocities*, *alpha*, *dt*, and *nsteps_inc* in-place.

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype vec3f/vec3d
        Atomic positions.
    velocities : wp.array, shape (N,), dtype vec3f/vec3d
        Atomic velocities.
    forces : wp.array, shape (N,), dtype vec3f/vec3d
        Forces on atoms (read-only).
    batch_idx : wp.array, shape (N,), dtype int32
        Sorted system index per atom (required).
    alpha : wp.array, shape (M,), dtype float*
        FIRE2 mixing parameter.
    dt : wp.array, shape (M,), dtype float*
        Per-system timestep.
    nsteps_inc : wp.array, shape (M,), dtype int32
        Consecutive positive-power step counter.
    vf, v_sumsq, f_sumsq, max_norm : wp.array, shape (M,), dtype float*
        Scratch buffers for reductions. Zeroed internally before each use.
    delaystep : int
        Minimum positive steps before dt growth.
    dtgrow, dtshrink : float
        Timestep growth/shrink factors.
    alphashrink : float
        Alpha decay factor.
    alpha0 : float
        Alpha reset value.
    tmax, tmin : float
        Timestep bounds.
    maxstep : float
        Maximum step magnitude per system.

    Notes
    -----
    - ``batch_idx`` must be sorted; segment reductions assume contiguous
      atom ranges per system.
    Examples
    --------
    >>> fire2_step(positions, velocities, forces, batch_idx,
    ...            alpha, dt, nsteps_inc,
    ...            vf, v_sumsq, f_sumsq, max_norm)
    """
    # --- Input validation ---
    N = positions.shape[0]

    if batch_idx is None:
        raise ValueError("batch_idx is required for fire2_step")

    if velocities.shape[0] != N:
        raise ValueError(
            f"velocities length {velocities.shape[0]} != positions length {N}"
        )
    if forces.shape[0] != N:
        raise ValueError(f"forces length {forces.shape[0]} != positions length {N}")
    if batch_idx.shape[0] != N:
        raise ValueError(
            f"batch_idx length {batch_idx.shape[0]} != positions length {N}"
        )

    M = alpha.shape[0]
    if dt.shape[0] != M:
        raise ValueError(f"dt length {dt.shape[0]} != alpha length {M}")
    if nsteps_inc.shape[0] != M:
        raise ValueError(f"nsteps_inc length {nsteps_inc.shape[0]} != alpha length {M}")

    vec_dtype = positions.dtype
    device = positions.device

    vf.zero_()
    v_sumsq.zero_()
    f_sumsq.zero_()
    max_norm.zero_()
    sm = max(device.sm_count, 1)

    # Kernel 1: reduce only (no velocity write, deferred to fused kernel)
    ept1 = compute_ept(N, sm, True)
    dim1 = (N + ept1 - 1) // ept1
    wp.launch(
        _fire2_reduce_only_overloads[vec_dtype],
        dim=dim1,
        inputs=[velocities, forces, dt, batch_idx, vf, v_sumsq, f_sumsq, N, ept1],
        device=device,
    )

    # Kernel 2: param update + deferred halfstep + mix + maxnorm
    ept2 = compute_ept(N, sm, True)
    dim2 = (N + ept2 - 1) // ept2
    wp.launch(
        _fire2_fused_mix_maxnorm_overloads[vec_dtype],
        dim=dim2,
        inputs=[
            velocities,
            forces,
            dt,
            batch_idx,
            vf,
            v_sumsq,
            f_sumsq,
            alpha,
            nsteps_inc,
            max_norm,
            N,
            ept2,
            delaystep,
            dtgrow,
            dtshrink,
            alphashrink,
            alpha0,
            tmax,
            tmin,
        ],
        device=device,
    )

    # Kernel 3: recompute step + clamp + position update + velocity zeroing
    wp.launch(
        _fire2_clamp_apply_recompute_overloads[vec_dtype],
        dim=N,
        inputs=[
            positions,
            velocities,
            dt,
            batch_idx,
            max_norm,
            vf,  # vf holds v.f; uphill if <= 0
            maxstep,
        ],
        device=device,
    )
