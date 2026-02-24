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
FIRE and FIRE2 Optimizer Kernels
================================

GPU-accelerated Warp kernels for FIRE (Fast Inertial Relaxation Engine)
geometry optimization and its improved FIRE2 variant.

This module provides both mutating (in-place) and non-mutating versions
of each kernel for gradient tracking compatibility.

MATHEMATICAL FORMULATION
========================

FIRE uses MD-like dynamics with velocity modification:

Velocity mixing:

.. math::

    \\mathbf{v}(t) \\leftarrow (1-\\alpha) \\mathbf{v}(t)
                              + \\alpha \\hat{\\mathbf{F}}(t) |\\mathbf{v}(t)|

Adaptive parameter update based on power :math:`P = \\mathbf{F} \\cdot \\mathbf{v}`:

If :math:`P > 0` for :math:`N_{\\min}` consecutive steps:
    - :math:`\\Delta t \\leftarrow \\min(\\Delta t \\cdot f_{\\text{inc}}, \\Delta t_{\\max})`
    - :math:`\\alpha \\leftarrow \\alpha \\cdot f_\\alpha`

If :math:`P \\leq 0`:
    - :math:`\\mathbf{v} \\leftarrow 0`
    - :math:`\\Delta t \\leftarrow \\max(\\Delta t \\cdot f_{\\text{dec}}, \\Delta t_{\\min})`
    - :math:`\\alpha \\leftarrow \\alpha_{\\text{start}}`

TYPICAL FIRE PARAMETERS
=======================

- dt_start: 0.1 (initial timestep)
- dt_max: 1.0 (maximum timestep)
- dt_min: 0.01 (minimum timestep)
- n_min: 5 (minimum steps before dt increase)
- f_inc: 1.1 (timestep increase factor)
- f_dec: 0.5 (timestep decrease factor)
- alpha_start: 0.1 (initial mixing parameter)
- f_alpha: 0.99 (alpha decrease factor)

REFERENCES
==========

- Bitzek et al. (2006). Phys. Rev. Lett. 97, 170201 (FIRE)
- Guénolé et al. (2020). Comp. Mat. Sci. 175, 109584 (FIRE2)
"""

from __future__ import annotations

from typing import Any

import warp as wp

from nvalchemiops.dynamics.utils.kernel_functions import (
    clamp_displacement,
    compute_vf_vv_ff,
    fire_velocity_mixing,
    is_first_atom_of_system,
)
from nvalchemiops.dynamics.utils.launch_helpers import (
    ExecutionMode,
    resolve_execution_mode,
)
from nvalchemiops.segment_ops import compute_ept


@wp.kernel(enable_backward=False)
def _fire_uphill_check_kernel(
    energy: wp.array(dtype=Any),
    energy_last: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    uphill_flag: wp.array(dtype=wp.int32),
):
    """Per-system uphill check for FIRE downhill variant.

    Compares current energy to last accepted energy and sets uphill flag.
    Only the first atom per system writes the energy updates.

    Launch Grid
    -----------
    dim = N (total atoms)

    Parameters
    ----------
    energy : wp.array, shape (M,), dtype float*
        Current per-system energies.
    energy_last : wp.array, shape (M,), dtype float*
        Last accepted per-system energies.
    batch_idx : wp.array, shape (N,), dtype int32
        Sorted system index per atom. **MUST BE SORTED**.
    uphill_flag : wp.array, shape (M,), dtype int32
        OUTPUT: 1 if system is uphill, 0 otherwise.

    Notes
    -----
    - batch_idx MUST be sorted for correct first-atom detection
    - Only first atom per system modifies energy arrays
    - All atoms read uphill_flag for their system
    """
    atom_idx = wp.tid()
    sys = batch_idx[atom_idx]

    # All atoms check uphill condition
    is_uphill = energy[sys] > energy_last[sys]

    # Only first atom per system writes state
    if is_first_atom_of_system(atom_idx, batch_idx):
        if is_uphill:
            uphill_flag[sys] = 1
            energy[sys] = energy_last[sys]  # Revert energy
        else:
            uphill_flag[sys] = 0
            energy_last[sys] = energy[sys]  # Accept energy


@wp.kernel(enable_backward=False)
def _fire_revert_and_reduce_kernel(
    positions: wp.array(dtype=Any),
    velocities: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    positions_last: wp.array(dtype=Any),
    velocities_last: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    uphill_flag: wp.array(dtype=wp.int32),
    vf: wp.array(dtype=Any),
    vv: wp.array(dtype=Any),
    ff: wp.array(dtype=Any),
    N: wp.int32,
    elems_per_thread: wp.int32,
):
    """Revert uphill systems and perform RLE-based reduction.

    For uphill systems, reverts positions/velocities to last accepted state.
    Then performs RLE reduction for vf, vv, ff diagnostics.

    Launch Grid
    -----------
    dim = ceil(N / elems_per_thread)

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype vec3*
        Atomic positions, modified in-place for uphill systems.
    velocities : wp.array, shape (N,), dtype vec3*
        Atomic velocities, modified in-place for uphill systems.
    forces : wp.array, shape (N,), dtype vec3*
        Forces (read-only).
    positions_last : wp.array, shape (N,), dtype vec3*
        Last accepted positions. Modified in-place for downhill systems.
    velocities_last : wp.array, shape (N,), dtype vec3*
        Last accepted velocities. Modified in-place for downhill systems.
    batch_idx : wp.array, shape (N,), dtype int32
        Sorted system index per atom. **MUST BE SORTED**.
    uphill_flag : wp.array, shape (M,), dtype int32
        Per-system uphill flags from uphill check kernel.
    vf, vv, ff : wp.array, shape (M,), dtype float*
        OUTPUT: Diagnostic accumulators. Zeroed internally before each use.
    N : int32
        Total number of atoms.
    elems_per_thread : int32
        Elements per thread (auto-tuned).

    Notes
    -----
    - Uphill systems: revert from positions_last/velocities_last
    - Downhill systems: update positions_last/velocities_last
    - RLE reduction minimizes atomic operations
    """
    t = wp.tid()
    start = t * elems_per_thread
    if start >= N:
        return
    end = wp.min(start + elems_per_thread, N)

    # First element
    s_cur = batch_idx[start]
    is_uphill = uphill_flag[s_cur] != 0

    if is_uphill:
        positions[start] = positions_last[start]
        velocities[start] = velocities_last[start]
    else:
        positions_last[start] = positions[start]
        velocities_last[start] = velocities[start]

    acc_vf, acc_vv, acc_ff = compute_vf_vv_ff(velocities[start], forces[start])

    # Process remaining elements
    for i in range(start + 1, end):
        s = batch_idx[i]

        # Handle revert/accept on segment boundary
        if s != s_cur:
            # Flush accumulation for previous segment
            wp.atomic_add(vf, s_cur, acc_vf)
            wp.atomic_add(vv, s_cur, acc_vv)
            wp.atomic_add(ff, s_cur, acc_ff)

            # Start new segment
            s_cur = s
            is_uphill = uphill_flag[s] != 0
            acc_vf = type(acc_vf)(0.0)
            acc_vv = type(acc_vv)(0.0)
            acc_ff = type(acc_ff)(0.0)

        # Revert or accept state
        if is_uphill:
            positions[i] = positions_last[i]
            velocities[i] = velocities_last[i]
        else:
            positions_last[i] = positions[i]
            velocities_last[i] = velocities[i]

        # Accumulate diagnostics
        val_vf, val_vv, val_ff = compute_vf_vv_ff(velocities[i], forces[i])
        acc_vf = acc_vf + val_vf
        acc_vv = acc_vv + val_vv
        acc_ff = acc_ff + val_ff

    # Flush final segment
    wp.atomic_add(vf, s_cur, acc_vf)
    wp.atomic_add(vv, s_cur, acc_vv)
    wp.atomic_add(ff, s_cur, acc_ff)


@wp.kernel(enable_backward=False)
def _fire_update_downhill_batch_idx_kernel(
    positions: wp.array(dtype=Any),
    velocities: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    alpha: wp.array(dtype=Any),
    dt: wp.array(dtype=Any),
    alpha_start: wp.array(dtype=Any),
    f_alpha: wp.array(dtype=Any),
    dt_min: wp.array(dtype=Any),
    dt_max: wp.array(dtype=Any),
    maxstep: wp.array(dtype=Any),
    n_steps_positive: wp.array(dtype=wp.int32),
    n_min: wp.array(dtype=wp.int32),
    f_dec: wp.array(dtype=Any),
    f_inc: wp.array(dtype=Any),
    vf: wp.array(dtype=Any),
    vv: wp.array(dtype=Any),
    ff: wp.array(dtype=Any),
    uphill_flag: wp.array(dtype=wp.int32),
):
    """Parameter update for FIRE downhill variant with uphill masking.

    Same as no_downhill update kernel but vf_mask includes uphill check.

    Launch Grid
    -----------
    dim = N (total atoms)

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype vec3*
        Atomic positions, modified in-place.
    velocities : wp.array, shape (N,), dtype vec3*
        Atomic velocities, modified in-place.
    forces : wp.array, shape (N,), dtype vec3*
        Forces (read-only).
    masses : wp.array, shape (N,), dtype float*
        Per-atom masses (read-only).
    batch_idx : wp.array, shape (N,), dtype int32
        Sorted system index per atom. **MUST BE SORTED**.
    alpha, dt, etc. : wp.array, shape (M,), dtype float*
        Per-system FIRE parameters.
    vf, vv, ff : wp.array, shape (M,), dtype float*
        Diagnostic values from reduction kernel (read-only).
    uphill_flag : wp.array, shape (M,), dtype int32
        Per-system uphill flags (read-only).

    Notes
    -----
    - Redundant computation of parameter updates (no synchronization)
    - Only first atom per segment writes dt, alpha, n_steps_positive
    - Uphill systems are masked out from velocity mixing
    """
    atom_idx = wp.tid()
    sys = batch_idx[atom_idx]

    # Snapshot dt before any thread modifies it
    local_dt = dt[sys]
    zero = type(local_dt)(0.0)

    # Redundantly compute parameter updates
    _vf = vf[sys]
    _vv = vv[sys]
    _ff = ff[sys]
    is_uphill = uphill_flag[sys] != 0

    vf_mask = (_vf > zero) and (not is_uphill)
    if vf_mask:
        _nsi = n_steps_positive[sys] + 1
        n_steps_positive_mask = _nsi >= n_min[sys]
        if n_steps_positive_mask:
            new_dt = wp.min(local_dt * f_inc[sys], dt_max[sys])
            new_alpha = alpha[sys] * f_alpha[sys]
        else:
            new_dt = local_dt
            new_alpha = alpha[sys]
    else:
        _nsi = 0
        new_dt = wp.max(local_dt * f_dec[sys], dt_min[sys])
        new_alpha = alpha_start[sys]

    # First atom per segment writes
    if is_first_atom_of_system(atom_idx, batch_idx):
        dt[sys] = new_dt
        alpha[sys] = new_alpha
        n_steps_positive[sys] = _nsi

    # Velocity mixing with uphill masking
    if vf_mask:
        velocities[atom_idx] = fire_velocity_mixing(
            velocities[atom_idx], forces[atom_idx], new_alpha, _vv, _ff
        )
    else:
        velocities[atom_idx] = zero * velocities[atom_idx]

    # Position update
    mass = masses[atom_idx]
    inv_mass = wp.where(mass > type(mass)(0.0), type(mass)(1.0) / mass, type(mass)(0.0))
    velocities[atom_idx] = velocities[atom_idx] + local_dt * forces[atom_idx] * inv_mass
    dr = local_dt * velocities[atom_idx]
    dr_clamped = clamp_displacement(dr, maxstep[sys])
    positions[atom_idx] = positions[atom_idx] + dr_clamped


@wp.kernel(enable_backward=False)
def _fire_update_only_batch_idx_kernel(
    velocities: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    alpha: wp.array(dtype=Any),
    dt: wp.array(dtype=Any),
    alpha_start: wp.array(dtype=Any),
    f_alpha: wp.array(dtype=Any),
    dt_min: wp.array(dtype=Any),
    dt_max: wp.array(dtype=Any),
    n_steps_positive: wp.array(dtype=wp.int32),
    n_min: wp.array(dtype=wp.int32),
    f_dec: wp.array(dtype=Any),
    f_inc: wp.array(dtype=Any),
    vf: wp.array(dtype=Any),
    vv: wp.array(dtype=Any),
    ff: wp.array(dtype=Any),
):
    """FIRE velocity mixing and parameter update (no MD step).

    Like _fire_update_batch_idx_kernel but WITHOUT the MD integration step
    (no velocity kick, no position update). For use by fire_update().

    Each thread redundantly computes per-system parameter updates from
    pre-computed read-only vf/vv/ff. Only the first atom per segment
    writes shared state (dt, alpha, n_steps_positive).

    Launch Grid
    -----------
    dim = N (total atoms)
    """
    atom_idx = wp.tid()
    sys = batch_idx[atom_idx]

    local_dt = dt[sys]
    zero = type(local_dt)(0.0)

    _vf = vf[sys]
    _vv = vv[sys]
    _ff = ff[sys]

    vf_mask = _vf > zero
    if vf_mask:
        _nsi = n_steps_positive[sys] + 1
        n_steps_positive_mask = _nsi >= n_min[sys]
        if n_steps_positive_mask:
            new_dt = wp.min(local_dt * f_inc[sys], dt_max[sys])
            new_alpha = alpha[sys] * f_alpha[sys]
        else:
            new_dt = local_dt
            new_alpha = alpha[sys]
    else:
        _nsi = 0
        new_dt = wp.max(local_dt * f_dec[sys], dt_min[sys])
        new_alpha = alpha_start[sys]

    if is_first_atom_of_system(atom_idx, batch_idx):
        dt[sys] = new_dt
        alpha[sys] = new_alpha
        n_steps_positive[sys] = _nsi

    if vf_mask:
        velocities[atom_idx] = fire_velocity_mixing(
            velocities[atom_idx], forces[atom_idx], new_alpha, _vv, _ff
        )
    else:
        velocities[atom_idx] = zero * velocities[atom_idx]


@wp.kernel(enable_backward=False)
def _fire_update_only_downhill_batch_idx_kernel(
    velocities: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    alpha: wp.array(dtype=Any),
    dt: wp.array(dtype=Any),
    alpha_start: wp.array(dtype=Any),
    f_alpha: wp.array(dtype=Any),
    dt_min: wp.array(dtype=Any),
    dt_max: wp.array(dtype=Any),
    n_steps_positive: wp.array(dtype=wp.int32),
    n_min: wp.array(dtype=wp.int32),
    f_dec: wp.array(dtype=Any),
    f_inc: wp.array(dtype=Any),
    vf: wp.array(dtype=Any),
    vv: wp.array(dtype=Any),
    ff: wp.array(dtype=Any),
    uphill_flag: wp.array(dtype=wp.int32),
):
    """FIRE velocity mixing and parameter update with uphill masking (no MD step).

    Like _fire_update_downhill_batch_idx_kernel but WITHOUT the MD integration
    step. For use by fire_update() in downhill mode.

    Launch Grid
    -----------
    dim = N (total atoms)
    """
    atom_idx = wp.tid()
    sys = batch_idx[atom_idx]

    local_dt = dt[sys]
    zero = type(local_dt)(0.0)

    _vf = vf[sys]
    _vv = vv[sys]
    _ff = ff[sys]
    is_uphill = uphill_flag[sys] != 0

    vf_mask = (_vf > zero) and (not is_uphill)
    if vf_mask:
        _nsi = n_steps_positive[sys] + 1
        n_steps_positive_mask = _nsi >= n_min[sys]
        if n_steps_positive_mask:
            new_dt = wp.min(local_dt * f_inc[sys], dt_max[sys])
            new_alpha = alpha[sys] * f_alpha[sys]
        else:
            new_dt = local_dt
            new_alpha = alpha[sys]
    else:
        _nsi = 0
        new_dt = wp.max(local_dt * f_dec[sys], dt_min[sys])
        new_alpha = alpha_start[sys]

    if is_first_atom_of_system(atom_idx, batch_idx):
        dt[sys] = new_dt
        alpha[sys] = new_alpha
        n_steps_positive[sys] = _nsi

    if vf_mask:
        velocities[atom_idx] = fire_velocity_mixing(
            velocities[atom_idx], forces[atom_idx], new_alpha, _vv, _ff
        )
    else:
        velocities[atom_idx] = zero * velocities[atom_idx]


@wp.kernel(enable_backward=False)
def _fire_reduce_batch_idx_rle_kernel(
    velocities: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    vf: wp.array(dtype=Any),
    vv: wp.array(dtype=Any),
    ff: wp.array(dtype=Any),
    N: wp.int32,
    elems_per_thread: wp.int32,
):
    """RLE-based reduction for FIRE diagnostics (vf, vv, ff).

    Uses run-length encoding to minimize atomic operations: accumulates
    locally while batch_idx stays constant, emits atomic_add only on
    segment boundaries.

    This kernel implements the reduction phase for FIRE optimization,
    computing three per-system inner products:
    - vf[s] = sum(v·f for atoms in system s)
    - vv[s] = sum(v·v for atoms in system s)
    - ff[s] = sum(f·f for atoms in system s)

    Launch Grid
    -----------
    dim = ceil(N / elems_per_thread)

    Parameters
    ----------
    velocities : wp.array, shape (N,), dtype vec3f/vec3d
        Atomic velocities (read-only).
    forces : wp.array, shape (N,), dtype vec3f/vec3d
        Forces on atoms (read-only).
    batch_idx : wp.array, shape (N,), dtype int32
        Sorted system index per atom in [0, M). **MUST BE SORTED**.
    vf : wp.array, shape (M,), dtype float32/float64
        OUTPUT: v·f per system. Zeroed internally before each use.
    vv : wp.array, shape (M,), dtype float32/float64
        OUTPUT: v·v per system. Zeroed internally before each use.
    ff : wp.array, shape (M,), dtype float32/float64
        OUTPUT: f·f per system. Zeroed internally before each use.
    N : int32
        Total number of atoms.
    elems_per_thread : int32
        Elements processed per thread (auto-tuned based on array size).

    Notes
    -----
    - batch_idx MUST be sorted in non-decreasing order for correctness
    - Uses run-length encoding: O(segments) atomic operations instead of O(N)
    - Typically reduces atomics by 100-1000x compared to naive approach
    """
    t = wp.tid()
    start = t * elems_per_thread
    if start >= N:
        return
    end = wp.min(start + elems_per_thread, N)

    # First element
    s_cur = batch_idx[start]
    acc_vf, acc_vv, acc_ff = compute_vf_vv_ff(velocities[start], forces[start])

    # Process remaining elements in chunk
    for i in range(start + 1, end):
        s = batch_idx[i]
        val_vf, val_vv, val_ff = compute_vf_vv_ff(velocities[i], forces[i])
        if s == s_cur:
            # Same segment: accumulate locally
            acc_vf = acc_vf + val_vf
            acc_vv = acc_vv + val_vv
            acc_ff = acc_ff + val_ff
        else:
            # Segment boundary: emit atomic and start new run
            wp.atomic_add(vf, s_cur, acc_vf)
            wp.atomic_add(vv, s_cur, acc_vv)
            wp.atomic_add(ff, s_cur, acc_ff)
            s_cur = s
            acc_vf = val_vf
            acc_vv = val_vv
            acc_ff = val_ff

    # Flush final run
    wp.atomic_add(vf, s_cur, acc_vf)
    wp.atomic_add(vv, s_cur, acc_vv)
    wp.atomic_add(ff, s_cur, acc_ff)


@wp.kernel(enable_backward=False)
def _fire_update_batch_idx_kernel(
    positions: wp.array(dtype=Any),
    velocities: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    alpha: wp.array(dtype=Any),
    dt: wp.array(dtype=Any),
    alpha_start: wp.array(dtype=Any),
    f_alpha: wp.array(dtype=Any),
    dt_min: wp.array(dtype=Any),
    dt_max: wp.array(dtype=Any),
    maxstep: wp.array(dtype=Any),
    n_steps_positive: wp.array(dtype=wp.int32),
    n_min: wp.array(dtype=wp.int32),
    f_dec: wp.array(dtype=Any),
    f_inc: wp.array(dtype=Any),
    vf: wp.array(dtype=Any),
    vv: wp.array(dtype=Any),
    ff: wp.array(dtype=Any),
):
    """Parameter update, velocity mixing, and position update for FIRE.

    This kernel performs the second phase of FIRE optimization after
    reduction is complete. Each thread redundantly computes per-system
    parameter updates from shared read-only inputs (vf, vv, ff), avoiding
    inter-thread synchronization. Only the first atom per segment writes
    shared state.

    Launch Grid
    -----------
    dim = N (total atoms)

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype vec3f/vec3d
        Atomic positions, modified in-place.
    velocities : wp.array, shape (N,), dtype vec3f/vec3d
        Atomic velocities, modified in-place.
    forces : wp.array, shape (N,), dtype vec3f/vec3d
        Forces on atoms (read-only).
    masses : wp.array, shape (N,), dtype float32/float64
        Per-atom masses (read-only).
    batch_idx : wp.array, shape (N,), dtype int32
        Sorted system index per atom. **MUST BE SORTED**.
    alpha, dt, alpha_start, f_alpha, dt_min, dt_max, maxstep : wp.array, shape (M,), dtype float*
        Per-system FIRE parameters. dt and alpha modified in-place.
    n_steps_positive, n_min : wp.array, shape (M,), dtype int32
        Per-system counters. n_steps_positive modified in-place.
    f_dec, f_inc : wp.array, shape (M,), dtype float*
        Per-system timestep factors (read-only).
    vf, vv, ff : wp.array, shape (M,), dtype float*
        Per-system diagnostic values from reduction kernel (read-only).

    Notes
    -----
    - batch_idx MUST be sorted for correct first-atom-per-segment detection
    - Each thread redundantly computes parameter updates (no synchronization)
    - Only first atom in each segment writes dt, alpha, n_steps_positive
    - Position updates use snapshot of dt before any thread modifies it
    """
    atom_idx = wp.tid()
    sys = batch_idx[atom_idx]

    # Snapshot dt before any thread modifies it
    local_dt = dt[sys]
    zero = type(local_dt)(0.0)

    # Redundantly compute per-system parameter updates from read-only inputs
    _vf = vf[sys]
    _vv = vv[sys]
    _ff = ff[sys]

    vf_mask = _vf > zero
    if vf_mask:
        _nsi = n_steps_positive[sys] + 1
        n_steps_positive_mask = _nsi >= n_min[sys]
        if n_steps_positive_mask:
            new_dt = wp.min(local_dt * f_inc[sys], dt_max[sys])
            new_alpha = alpha[sys] * f_alpha[sys]
        else:
            new_dt = local_dt
            new_alpha = alpha[sys]
    else:
        _nsi = 0
        new_dt = wp.max(local_dt * f_dec[sys], dt_min[sys])
        new_alpha = alpha_start[sys]

    # First atom per segment writes updated params
    if is_first_atom_of_system(atom_idx, batch_idx):
        dt[sys] = new_dt
        alpha[sys] = new_alpha
        n_steps_positive[sys] = _nsi

    # Velocity mixing (all atoms)
    if vf_mask:
        velocities[atom_idx] = fire_velocity_mixing(
            velocities[atom_idx], forces[atom_idx], new_alpha, _vv, _ff
        )
    else:
        velocities[atom_idx] = zero * velocities[atom_idx]

    # Update velocities with forces (mass-aware) and positions
    mass = masses[atom_idx]
    inv_mass = wp.where(mass > type(mass)(0.0), type(mass)(1.0) / mass, type(mass)(0.0))
    velocities[atom_idx] = velocities[atom_idx] + local_dt * forces[atom_idx] * inv_mass
    dr = local_dt * velocities[atom_idx]
    dr_clamped = clamp_displacement(dr, maxstep[sys])
    positions[atom_idx] = positions[atom_idx] + dr_clamped


@wp.kernel
def _fire_step_no_downhill_ptr_kernel(
    positions: wp.array(dtype=Any),
    velocities: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    alpha: wp.array(dtype=Any),
    dt: wp.array(dtype=Any),
    alpha_start: wp.array(dtype=Any),
    f_alpha: wp.array(dtype=Any),
    dt_min: wp.array(dtype=Any),
    dt_max: wp.array(dtype=Any),
    maxstep: wp.array(dtype=Any),
    n_steps_positive: wp.array(dtype=wp.int32),
    n_min: wp.array(dtype=wp.int32),
    f_dec: wp.array(dtype=Any),
    f_inc: wp.array(dtype=Any),
    atom_ptr: wp.array(dtype=wp.int32),
):
    """FIRE no-downhill step (ptr/CSR batched; launched over systems).

    This is the ptr-based ("CSR") batching formulation analogous to the reference
    implementation you shared:

    - Launch grid is over systems: `dim = [num_systems]`
    - Each thread processes the contiguous atom range:
      `i in [atom_ptr[sys], atom_ptr[sys+1])`
    - All per-system reductions (`vf/vv/ff`) and parameter updates happen within
      the same thread, so no cross-thread synchronization is required.

    Parameters
    ----------
    positions : wp.array, shape (N_total,), dtype=wp.vec3*
        Concatenated positions for all systems (in-place).
    velocities : wp.array, shape (N_total,), dtype=wp.vec3*
        Concatenated velocities for all systems (in-place).
    forces : wp.array, shape (N_total,), dtype=wp.vec3*
        Concatenated forces for all systems.
    masses : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        Concatenated masses.
    alpha, dt, alpha_start, f_alpha, dt_min, dt_max, maxstep : wp.array, shape (B,), dtype=wp.float*
        Per-system FIRE parameters.
    n_steps_positive, n_min : wp.array, shape (B,), dtype=wp.int32
        Per-system counters/thresholds.
    f_dec, f_inc : wp.array, shape (B,), dtype=wp.float*
        Per-system timestep factors.
    atom_ptr : wp.array, shape (B+1,), dtype=wp.int32
        CSR pointer giving the start/end atom indices for each system.

    Launch Grid
    -----------
    dim = [num_systems]

    Notes
    -----
    - This formulation is typically the best choice for a fully fused FIRE step
      because the entire system update is carried out within a single thread.
    """
    sys = wp.tid()
    a0 = atom_ptr[sys]
    a1 = atom_ptr[sys + 1]

    # Compute diagnostics within system
    vf = type(dt[sys])(0.0)
    vv = type(dt[sys])(0.0)
    ff = type(dt[sys])(0.0)
    for i in range(a0, a1):
        vf_val, vv_val, ff_val = compute_vf_vv_ff(velocities[i], forces[i])
        vf += vf_val
        vv += vv_val
        ff += ff_val

    vf_mask = vf > type(dt[sys])(0.0)
    n_steps_positive[sys] = wp.where(vf_mask, n_steps_positive[sys] + 1, 0)
    n_steps_positive_mask = n_steps_positive[sys] >= n_min[sys]

    # Guard against division by zero when forces are zero
    zero = type(dt[sys])(0.0)
    if ff > zero:
        ratio = wp.sqrt(vv / ff)
    else:
        ratio = zero

    for i in range(a0, a1):
        velocities[i] = wp.where(
            vf_mask,
            (type(dt[sys])(1.0) - alpha[sys]) * velocities[i]
            + (alpha[sys] * forces[i] * ratio),
            zero * velocities[i],
        )
    dt[sys] = wp.where(
        vf_mask,
        wp.where(
            n_steps_positive_mask,
            wp.min(dt[sys] * f_inc[sys], dt_max[sys]),
            dt[sys],
        ),
        wp.max(dt[sys] * f_dec[sys], dt_min[sys]),
    )
    alpha[sys] = wp.where(
        vf_mask,
        wp.where(
            n_steps_positive_mask,
            alpha[sys] * f_alpha[sys],
            alpha[sys],
        ),
        alpha_start[sys],
    )
    for i in range(a0, a1):
        mass = masses[i]
        inv_mass = wp.where(
            mass > type(mass)(0.0), type(mass)(1.0) / mass, type(mass)(0.0)
        )
        velocities[i] += dt[sys] * forces[i] * inv_mass
        dr = dt[sys] * velocities[i]
        dr_clamped = clamp_displacement(dr, maxstep[sys])
        positions[i] += dr_clamped


@wp.kernel
def _fire_step_downhill_ptr_kernel(
    energy: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    positions: wp.array(dtype=Any),
    velocities: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    alpha: wp.array(dtype=Any),
    dt: wp.array(dtype=Any),
    alpha_start: wp.array(dtype=Any),
    f_alpha: wp.array(dtype=Any),
    dt_min: wp.array(dtype=Any),
    dt_max: wp.array(dtype=Any),
    maxstep: wp.array(dtype=Any),
    n_steps_positive: wp.array(dtype=wp.int32),
    n_min: wp.array(dtype=wp.int32),
    f_dec: wp.array(dtype=Any),
    f_inc: wp.array(dtype=Any),
    energy_last: wp.array(dtype=Any),
    positions_last: wp.array(dtype=Any),
    velocities_last: wp.array(dtype=Any),
    atom_ptr: wp.array(dtype=wp.int32),
):
    """FIRE downhill-check step (ptr/CSR batched; launched over systems).

    This is the ptr-based ("CSR") batched formulation: each thread owns a full
    system range `[atom_ptr[sys], atom_ptr[sys+1])` and performs the downhill
    check, FIRE updates, and MD-like step for that system without cross-thread
    synchronization.

    Parameters
    ----------
    energy : wp.array, shape (B,), dtype=wp.float*
        Per-system energies.
    forces : wp.array, shape (N_total,), dtype=wp.vec3*
        Concatenated forces.
    positions : wp.array, shape (N_total,), dtype=wp.vec3*
        Concatenated positions (in-place).
    velocities : wp.array, shape (N_total,), dtype=wp.vec3*
        Concatenated velocities (in-place).
    masses : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        Concatenated masses.
    alpha, dt, alpha_start, f_alpha, dt_min, dt_max, maxstep : wp.array, shape (B,), dtype=wp.float*
        Per-system FIRE parameters.
    n_steps_positive, n_min : wp.array, shape (B,), dtype=wp.int32
        Per-system counters/thresholds.
    f_dec, f_inc : wp.array, shape (B,), dtype=wp.float*
        Per-system timestep factors.
    energy_last : wp.array, shape (B,), dtype=wp.float*
        Per-system last accepted energies.
    positions_last : wp.array, shape (N_total,), dtype=wp.vec3*
        Per-atom last accepted positions.
    velocities_last : wp.array, shape (N_total,), dtype=wp.vec3*
        Per-atom last accepted velocities.
    atom_ptr : wp.array, shape (B+1,), dtype=wp.int32
        CSR pointer giving the start/end atom indices for each system.

    Launch Grid
    -----------
    dim = [num_systems]

    Notes
    -----
    - This formulation is the most natural way to keep the downhill logic fully
      fused because each system is processed by a single thread.
    """
    sys = wp.tid()
    a0 = atom_ptr[sys]
    a1 = atom_ptr[sys + 1]

    # Uphill check
    is_uphill = False
    if energy[sys] > energy_last[sys]:
        is_uphill = True
        energy[sys] = energy_last[sys]
        for i in range(a0, a1):
            positions[i] = positions_last[i]
            velocities[i] = velocities_last[i]
    else:
        energy_last[sys] = energy[sys]
        for i in range(a0, a1):
            positions_last[i] = positions[i]
            velocities_last[i] = velocities[i]

    vf = type(dt[sys])(0.0)
    vv = type(dt[sys])(0.0)
    ff = type(dt[sys])(0.0)
    for i in range(a0, a1):
        vf_val, vv_val, ff_val = compute_vf_vv_ff(velocities[i], forces[i])
        vf += vf_val
        vv += vv_val
        ff += ff_val

    vf_mask = (vf > type(dt[sys])(0.0)) and (not is_uphill)
    n_steps_positive[sys] = wp.where(vf_mask, n_steps_positive[sys] + 1, 0)
    n_steps_positive_mask = n_steps_positive[sys] >= n_min[sys]

    # Guard against division by zero when forces are zero
    zero = type(dt[sys])(0.0)
    if ff > zero:
        ratio = wp.sqrt(vv / ff)
    else:
        ratio = zero

    for i in range(a0, a1):
        velocities[i] = wp.where(
            vf_mask,
            (type(dt[sys])(1.0) - alpha[sys]) * velocities[i]
            + (alpha[sys] * forces[i] * ratio),
            zero * velocities[i],
        )

    dt[sys] = wp.where(
        vf_mask,
        wp.where(
            n_steps_positive_mask,
            wp.min(dt[sys] * f_inc[sys], dt_max[sys]),
            dt[sys],
        ),
        wp.max(dt[sys] * f_dec[sys], dt_min[sys]),
    )
    alpha[sys] = wp.where(
        vf_mask,
        wp.where(
            n_steps_positive_mask,
            alpha[sys] * f_alpha[sys],
            alpha[sys],
        ),
        alpha_start[sys],
    )

    for i in range(a0, a1):
        mass = masses[i]
        inv_mass = wp.where(
            mass > type(mass)(0.0), type(mass)(1.0) / mass, type(mass)(0.0)
        )
        velocities[i] += dt[sys] * forces[i] * inv_mass
        dr = dt[sys] * velocities[i]
        dr_clamped = clamp_displacement(dr, maxstep[sys])
        positions[i] += dr_clamped


@wp.kernel
def _fire_update_params_no_downhill_ptr_kernel(
    velocities: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    alpha: wp.array(dtype=Any),
    dt: wp.array(dtype=Any),
    alpha_start: wp.array(dtype=Any),
    f_alpha: wp.array(dtype=Any),
    dt_min: wp.array(dtype=Any),
    dt_max: wp.array(dtype=Any),
    n_steps_positive: wp.array(dtype=wp.int32),
    n_min: wp.array(dtype=wp.int32),
    f_dec: wp.array(dtype=Any),
    f_inc: wp.array(dtype=Any),
    atom_ptr: wp.array(dtype=wp.int32),
):
    r"""FIRE parameter update (no downhill; ptr/CSR).

    Each thread owns a full system range `[atom_ptr[sys], atom_ptr[sys+1])` and
    computes the diagnostic scalars (\\(v\\cdot f\\), \\(v\\cdot v\\), \\(f\\cdot f\\)),
    performs velocity mixing, and updates per-system `dt`, `alpha`, and
    `n_steps_positive` **without** performing any MD step.

    Parameters
    ----------
    velocities : wp.array, shape (N_total,), dtype=wp.vec3*
        Concatenated atomic velocities (in-place; mixed according to FIRE rule).
    forces : wp.array, shape (N_total,), dtype=wp.vec3*
        Concatenated forces on atoms.
    alpha : wp.array, shape (B,), dtype=wp.float*
        Per-system FIRE mixing parameter \\(\\alpha\\).
    dt : wp.array, shape (B,), dtype=wp.float*
        Per-system FIRE timestep \\(\\Delta t\\).
    alpha_start : wp.array, shape (B,), dtype=wp.float*
        Per-system reset value for \\(\\alpha\\).
    f_alpha : wp.array, shape (B,), dtype=wp.float*
        Per-system multiplicative decay factor for \\(\\alpha\\).
    dt_min : wp.array, shape (B,), dtype=wp.float*
        Per-system minimum allowed timestep.
    dt_max : wp.array, shape (B,), dtype=wp.float*
        Per-system maximum allowed timestep.
    n_steps_positive : wp.array, shape (B,), dtype=wp.int32
        Per-system counter for consecutive steps with \\(v\\cdot f > 0\\).
    n_min : wp.array, shape (B,), dtype=wp.int32
        Per-system threshold for when to start increasing `dt`.
    f_dec : wp.array, shape (B,), dtype=wp.float*
        Per-system decay factor for `dt` when \\(v\\cdot f \\le 0\\).
    f_inc : wp.array, shape (B,), dtype=wp.float*
        Per-system growth factor for `dt` after `n_min` positive steps.
    atom_ptr : wp.array, shape (B+1,), dtype=wp.int32
        CSR pointer giving the start/end atom indices for each system.

    Launch Grid
    -----------
    dim = [num_systems]

    Notes
    -----
    - This kernel does NOT perform the MD step (velocity integration + position update).
    - Each system is processed by a single thread, so no cross-thread synchronization
      is required for the diagnostic reductions.
    """
    sys = wp.tid()

    a0 = atom_ptr[sys]
    a1 = atom_ptr[sys + 1]
    vv = type(dt[sys])(0.0)
    ff = type(dt[sys])(0.0)
    vf = type(dt[sys])(0.0)
    for i in range(a0, a1):
        vf_val, vv_val, ff_val = compute_vf_vv_ff(velocities[i], forces[i])
        vf += vf_val
        vv += vv_val
        ff += ff_val

    vf_mask = vf > type(dt[sys])(0.0)
    n_steps_positive[sys] = wp.where(vf_mask, n_steps_positive[sys] + 1, 0)
    n_steps_positive_mask = n_steps_positive[sys] >= n_min[sys]

    # Guard against division by zero when forces are zero
    zero = type(dt[sys])(0.0)
    if ff > zero:
        ratio = wp.sqrt(vv / ff)
    else:
        ratio = zero

    for i in range(a0, a1):
        velocities[i] = wp.where(
            vf_mask,
            (type(dt[sys])(1.0) - alpha[sys]) * velocities[i]
            + (alpha[sys] * forces[i] * ratio),
            zero * velocities[i],
        )
    dt[sys] = wp.where(
        vf_mask,
        wp.where(
            n_steps_positive_mask,
            wp.min(dt[sys] * f_inc[sys], dt_max[sys]),
            dt[sys],
        ),
        wp.max(dt[sys] * f_dec[sys], dt_min[sys]),
    )
    alpha[sys] = wp.where(
        vf_mask,
        wp.where(
            n_steps_positive_mask,
            alpha[sys] * f_alpha[sys],
            alpha[sys],
        ),
        alpha_start[sys],
    )


@wp.kernel
def _fire_update_params_downhill_ptr_kernel(
    energy: wp.array(dtype=Any),
    energy_last: wp.array(dtype=Any),
    positions: wp.array(dtype=Any),
    positions_last: wp.array(dtype=Any),
    velocities: wp.array(dtype=Any),
    velocities_last: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    alpha: wp.array(dtype=Any),
    dt: wp.array(dtype=Any),
    alpha_start: wp.array(dtype=Any),
    f_alpha: wp.array(dtype=Any),
    dt_min: wp.array(dtype=Any),
    dt_max: wp.array(dtype=Any),
    n_steps_positive: wp.array(dtype=wp.int32),
    n_min: wp.array(dtype=wp.int32),
    f_dec: wp.array(dtype=Any),
    f_inc: wp.array(dtype=Any),
    atom_ptr: wp.array(dtype=wp.int32),
):
    r"""FIRE parameter update (downhill; ptr/CSR).

    Each thread owns a full system range `[atom_ptr[sys], atom_ptr[sys+1])` and
    performs the downhill check, computes diagnostic scalars, applies velocity
    mixing, and updates per-system `dt`, `alpha`, and `n_steps_positive`
    **without** performing any MD step.

    Parameters
    ----------
    energy : wp.array, shape (B,), dtype=wp.float*
        Per-system current energies. Rolled back if uphill.
    energy_last : wp.array, shape (B,), dtype=wp.float*
        Per-system last accepted energies.
    positions : wp.array, shape (N_total,), dtype=wp.vec3*
        Concatenated atomic positions (in-place; rolled back if uphill).
    positions_last : wp.array, shape (N_total,), dtype=wp.vec3*
        Concatenated last accepted positions.
    velocities : wp.array, shape (N_total,), dtype=wp.vec3*
        Concatenated atomic velocities (in-place; rolled back if uphill, then mixed).
    velocities_last : wp.array, shape (N_total,), dtype=wp.vec3*
        Concatenated last accepted velocities.
    forces : wp.array, shape (N_total,), dtype=wp.vec3*
        Concatenated forces on atoms.
    alpha : wp.array, shape (B,), dtype=wp.float*
        Per-system FIRE mixing parameter \\(\\alpha\\).
    dt : wp.array, shape (B,), dtype=wp.float*
        Per-system FIRE timestep \\(\\Delta t\\).
    alpha_start : wp.array, shape (B,), dtype=wp.float*
        Per-system reset value for \\(\\alpha\\).
    f_alpha : wp.array, shape (B,), dtype=wp.float*
        Per-system multiplicative decay factor for \\(\\alpha\\).
    dt_min : wp.array, shape (B,), dtype=wp.float*
        Per-system minimum allowed timestep.
    dt_max : wp.array, shape (B,), dtype=wp.float*
        Per-system maximum allowed timestep.
    n_steps_positive : wp.array, shape (B,), dtype=wp.int32
        Per-system counter for consecutive steps with \\(v\\cdot f > 0\\).
    n_min : wp.array, shape (B,), dtype=wp.int32
        Per-system threshold for when to start increasing `dt`.
    f_dec : wp.array, shape (B,), dtype=wp.float*
        Per-system decay factor for `dt` when uphill or \\(v\\cdot f \\le 0\\).
    f_inc : wp.array, shape (B,), dtype=wp.float*
        Per-system growth factor for `dt` after `n_min` positive steps.
    atom_ptr : wp.array, shape (B+1,), dtype=wp.int32
        CSR pointer giving the start/end atom indices for each system.

    Launch Grid
    -----------
    dim = [num_systems]

    Notes
    -----
    - This kernel does NOT perform the MD step (velocity integration + position update).
    - Each system is processed by a single thread, so no cross-thread synchronization
      is required for the diagnostic reductions or rollback.
    """
    sys = wp.tid()
    a0 = atom_ptr[sys]
    a1 = atom_ptr[sys + 1]

    # Downhill check
    is_uphill = False
    if energy[sys] > energy_last[sys]:
        is_uphill = True
        energy[sys] = energy_last[sys]
        for i in range(a0, a1):
            positions[i] = positions_last[i]
            velocities[i] = velocities_last[i]
    else:
        energy_last[sys] = energy[sys]
        for i in range(a0, a1):
            positions_last[i] = positions[i]
            velocities_last[i] = velocities[i]

    # Compute diagnostics
    vf = type(dt[sys])(0.0)
    vv = type(dt[sys])(0.0)
    ff = type(dt[sys])(0.0)
    for i in range(a0, a1):
        vf += wp.dot(velocities[i], forces[i])
        vv += wp.dot(velocities[i], velocities[i])
        ff += wp.dot(forces[i], forces[i])

    vf_mask = (vf > type(dt[sys])(0.0)) and (not is_uphill)
    n_steps_positive[sys] = wp.where(vf_mask, n_steps_positive[sys] + 1, 0)
    n_steps_positive_mask = n_steps_positive[sys] >= n_min[sys]

    # Guard against division by zero when forces are zero
    zero = type(dt[sys])(0.0)
    if ff > zero:
        ratio = wp.sqrt(vv / ff)
    else:
        ratio = zero

    # Velocity mixing
    for i in range(a0, a1):
        velocities[i] = wp.where(
            vf_mask,
            (type(dt[sys])(1.0) - alpha[sys]) * velocities[i]
            + (alpha[sys] * forces[i] * ratio),
            zero * velocities[i],
        )

    dt[sys] = wp.where(
        vf_mask,
        wp.where(
            n_steps_positive_mask,
            wp.min(dt[sys] * f_inc[sys], dt_max[sys]),
            dt[sys],
        ),
        wp.max(dt[sys] * f_dec[sys], dt_min[sys]),
    )
    alpha[sys] = wp.where(
        vf_mask,
        wp.where(
            n_steps_positive_mask,
            alpha[sys] * f_alpha[sys],
            alpha[sys],
        ),
        alpha_start[sys],
    )


# =============================================================================
# Kernel Overloads for Explicit Typing
# =============================================================================

_T = [wp.float32, wp.float64]  # Scalar types
_V = [wp.vec3f, wp.vec3d]  # Vector types

# Step kernels (with MD integration)
_fire_step_no_downhill_ptr_kernel_overload = {}
_fire_step_downhill_ptr_kernel_overload = {}

# RLE-based kernels
_fire_reduce_batch_idx_rle_kernel_overload = {}
_fire_update_batch_idx_kernel_overload = {}
_fire_uphill_check_kernel_overload = {}
_fire_revert_and_reduce_kernel_overload = {}
_fire_update_downhill_batch_idx_kernel_overload = {}

# RLE-based update-only kernels (no MD step)
_fire_update_only_batch_idx_kernel_overload = {}
_fire_update_only_downhill_batch_idx_kernel_overload = {}

# Update-only kernels (no MD integration) - ptr variants only
_fire_update_params_no_downhill_ptr_kernel_overload = {}
_fire_update_params_downhill_ptr_kernel_overload = {}

for t, v in zip(_T, _V):
    _fire_step_no_downhill_ptr_kernel_overload[v] = wp.overload(
        _fire_step_no_downhill_ptr_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=v),  # velocities
            wp.array(dtype=v),  # forces
            wp.array(dtype=t),  # masses
            wp.array(dtype=t),  # alpha
            wp.array(dtype=t),  # dt
            wp.array(dtype=t),  # alpha_start
            wp.array(dtype=t),  # f_alpha
            wp.array(dtype=t),  # dt_min
            wp.array(dtype=t),  # dt_max
            wp.array(dtype=t),  # maxstep
            wp.array(dtype=wp.int32),  # n_steps_positive
            wp.array(dtype=wp.int32),  # n_min
            wp.array(dtype=t),  # f_dec
            wp.array(dtype=t),  # f_inc
            wp.array(dtype=wp.int32),  # atom_ptr
        ],
    )

    # RLE-based reduction kernel
    _fire_reduce_batch_idx_rle_kernel_overload[v] = wp.overload(
        _fire_reduce_batch_idx_rle_kernel,
        [
            wp.array(dtype=v),  # velocities
            wp.array(dtype=v),  # forces
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array(dtype=t),  # vf
            wp.array(dtype=t),  # vv
            wp.array(dtype=t),  # ff
            wp.int32,  # N
            wp.int32,  # elems_per_thread
        ],
    )

    # RLE-based update kernel
    _fire_update_batch_idx_kernel_overload[v] = wp.overload(
        _fire_update_batch_idx_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=v),  # velocities
            wp.array(dtype=v),  # forces
            wp.array(dtype=t),  # masses
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array(dtype=t),  # alpha
            wp.array(dtype=t),  # dt
            wp.array(dtype=t),  # alpha_start
            wp.array(dtype=t),  # f_alpha
            wp.array(dtype=t),  # dt_min
            wp.array(dtype=t),  # dt_max
            wp.array(dtype=t),  # maxstep
            wp.array(dtype=wp.int32),  # n_steps_positive
            wp.array(dtype=wp.int32),  # n_min
            wp.array(dtype=t),  # f_dec
            wp.array(dtype=t),  # f_inc
            wp.array(dtype=t),  # vf
            wp.array(dtype=t),  # vv
            wp.array(dtype=t),  # ff
        ],
    )

    # RLE-based downhill kernels
    _fire_uphill_check_kernel_overload[v] = wp.overload(
        _fire_uphill_check_kernel,
        [
            wp.array(dtype=t),  # energy
            wp.array(dtype=t),  # energy_last
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array(dtype=wp.int32),  # uphill_flag
        ],
    )

    _fire_revert_and_reduce_kernel_overload[v] = wp.overload(
        _fire_revert_and_reduce_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=v),  # velocities
            wp.array(dtype=v),  # forces
            wp.array(dtype=v),  # positions_last
            wp.array(dtype=v),  # velocities_last
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array(dtype=wp.int32),  # uphill_flag
            wp.array(dtype=t),  # vf
            wp.array(dtype=t),  # vv
            wp.array(dtype=t),  # ff
            wp.int32,  # N
            wp.int32,  # elems_per_thread
        ],
    )

    _fire_update_downhill_batch_idx_kernel_overload[v] = wp.overload(
        _fire_update_downhill_batch_idx_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=v),  # velocities
            wp.array(dtype=v),  # forces
            wp.array(dtype=t),  # masses
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array(dtype=t),  # alpha
            wp.array(dtype=t),  # dt
            wp.array(dtype=t),  # alpha_start
            wp.array(dtype=t),  # f_alpha
            wp.array(dtype=t),  # dt_min
            wp.array(dtype=t),  # dt_max
            wp.array(dtype=t),  # maxstep
            wp.array(dtype=wp.int32),  # n_steps_positive
            wp.array(dtype=wp.int32),  # n_min
            wp.array(dtype=t),  # f_dec
            wp.array(dtype=t),  # f_inc
            wp.array(dtype=t),  # vf
            wp.array(dtype=t),  # vv
            wp.array(dtype=t),  # ff
            wp.array(dtype=wp.int32),  # uphill_flag
        ],
    )

    # RLE-based update-only kernels (no MD step)
    _fire_update_only_batch_idx_kernel_overload[v] = wp.overload(
        _fire_update_only_batch_idx_kernel,
        [
            wp.array(dtype=v),  # velocities
            wp.array(dtype=v),  # forces
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array(dtype=t),  # alpha
            wp.array(dtype=t),  # dt
            wp.array(dtype=t),  # alpha_start
            wp.array(dtype=t),  # f_alpha
            wp.array(dtype=t),  # dt_min
            wp.array(dtype=t),  # dt_max
            wp.array(dtype=wp.int32),  # n_steps_positive
            wp.array(dtype=wp.int32),  # n_min
            wp.array(dtype=t),  # f_dec
            wp.array(dtype=t),  # f_inc
            wp.array(dtype=t),  # vf
            wp.array(dtype=t),  # vv
            wp.array(dtype=t),  # ff
        ],
    )

    _fire_update_only_downhill_batch_idx_kernel_overload[v] = wp.overload(
        _fire_update_only_downhill_batch_idx_kernel,
        [
            wp.array(dtype=v),  # velocities
            wp.array(dtype=v),  # forces
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array(dtype=t),  # alpha
            wp.array(dtype=t),  # dt
            wp.array(dtype=t),  # alpha_start
            wp.array(dtype=t),  # f_alpha
            wp.array(dtype=t),  # dt_min
            wp.array(dtype=t),  # dt_max
            wp.array(dtype=wp.int32),  # n_steps_positive
            wp.array(dtype=wp.int32),  # n_min
            wp.array(dtype=t),  # f_dec
            wp.array(dtype=t),  # f_inc
            wp.array(dtype=t),  # vf
            wp.array(dtype=t),  # vv
            wp.array(dtype=t),  # ff
            wp.array(dtype=wp.int32),  # uphill_flag
        ],
    )

    _fire_step_downhill_ptr_kernel_overload[v] = wp.overload(
        _fire_step_downhill_ptr_kernel,
        [
            wp.array(dtype=t),  # energy
            wp.array(dtype=v),  # forces
            wp.array(dtype=v),  # positions
            wp.array(dtype=v),  # velocities
            wp.array(dtype=t),  # masses
            wp.array(dtype=t),  # alpha
            wp.array(dtype=t),  # dt
            wp.array(dtype=t),  # alpha_start
            wp.array(dtype=t),  # f_alpha
            wp.array(dtype=t),  # dt_min
            wp.array(dtype=t),  # dt_max
            wp.array(dtype=t),  # maxstep
            wp.array(dtype=wp.int32),  # n_steps_positive
            wp.array(dtype=wp.int32),  # n_min
            wp.array(dtype=t),  # f_dec
            wp.array(dtype=t),  # f_inc
            wp.array(dtype=t),  # energy_last
            wp.array(dtype=v),  # positions_last
            wp.array(dtype=v),  # velocities_last
            wp.array(dtype=wp.int32),  # atom_ptr
        ],
    )

    _fire_update_params_no_downhill_ptr_kernel_overload[v] = wp.overload(
        _fire_update_params_no_downhill_ptr_kernel,
        [
            wp.array(dtype=v),  # velocities
            wp.array(dtype=v),  # forces
            wp.array(dtype=t),  # alpha
            wp.array(dtype=t),  # dt
            wp.array(dtype=t),  # alpha_start
            wp.array(dtype=t),  # f_alpha
            wp.array(dtype=t),  # dt_min
            wp.array(dtype=t),  # dt_max
            wp.array(dtype=wp.int32),  # n_steps_positive
            wp.array(dtype=wp.int32),  # n_min
            wp.array(dtype=t),  # f_dec
            wp.array(dtype=t),  # f_inc
            wp.array(dtype=wp.int32),  # atom_ptr
        ],
    )

    _fire_update_params_downhill_ptr_kernel_overload[v] = wp.overload(
        _fire_update_params_downhill_ptr_kernel,
        [
            wp.array(dtype=t),  # energy
            wp.array(dtype=t),  # energy_last
            wp.array(dtype=v),  # positions
            wp.array(dtype=v),  # positions_last
            wp.array(dtype=v),  # velocities
            wp.array(dtype=v),  # velocities_last
            wp.array(dtype=v),  # forces
            wp.array(dtype=t),  # alpha
            wp.array(dtype=t),  # dt
            wp.array(dtype=t),  # alpha_start
            wp.array(dtype=t),  # f_alpha
            wp.array(dtype=t),  # dt_min
            wp.array(dtype=t),  # dt_max
            wp.array(dtype=wp.int32),  # n_steps_positive
            wp.array(dtype=wp.int32),  # n_min
            wp.array(dtype=t),  # f_dec
            wp.array(dtype=t),  # f_inc
            wp.array(dtype=wp.int32),  # atom_ptr
        ],
    )


# =============================================================================
# Dispatch tables – keyed by ``downhill_enabled`` (bool)
# =============================================================================

# fire_step: PTR-mode fused kernels
_FIRE_STEP_PTR_OVERLOADS = {
    True: _fire_step_downhill_ptr_kernel_overload,
    False: _fire_step_no_downhill_ptr_kernel_overload,
}

# fire_update: PTR-mode fused kernels
_FIRE_UPDATE_PTR_OVERLOADS = {
    True: _fire_update_params_downhill_ptr_kernel_overload,
    False: _fire_update_params_no_downhill_ptr_kernel_overload,
}

# fire_step: batch_idx final-update kernel (with/without MD integration)
_FIRE_STEP_BATCH_UPDATE_OVERLOADS = {
    True: _fire_update_downhill_batch_idx_kernel_overload,
    False: _fire_update_batch_idx_kernel_overload,
}

# fire_update: batch_idx final-update kernel (velocity mixing only, no MD)
_FIRE_UPDATE_BATCH_UPDATE_OVERLOADS = {
    True: _fire_update_only_downhill_batch_idx_kernel_overload,
    False: _fire_update_only_batch_idx_kernel_overload,
}

# =============================================================================
# Public API: Unified FIRE Step Functions
# =============================================================================


def fire_step(
    # Core DOFs (required)
    positions: wp.array,
    velocities: wp.array,
    forces: wp.array,
    masses: wp.array,
    # FIRE control parameters (required)
    alpha: wp.array,
    dt: wp.array,
    alpha_start: wp.array,
    f_alpha: wp.array,
    dt_min: wp.array,
    dt_max: wp.array,
    maxstep: wp.array,
    n_steps_positive: wp.array,
    n_min: wp.array,
    f_dec: wp.array,
    f_inc: wp.array,
    # Scratch arrays
    uphill_flag: wp.array,
    # Accumulators (required for single/batch_idx; ignored for ptr)
    vf: wp.array = None,
    vv: wp.array = None,
    ff: wp.array = None,
    # Batching (mutually exclusive - if neither, assumes single system)
    batch_idx: wp.array = None,
    atom_ptr: wp.array = None,
    # Downhill check (optional - provide ALL or NONE)
    energy: wp.array = None,
    energy_last: wp.array = None,
    positions_last: wp.array = None,
    velocities_last: wp.array = None,
) -> None:
    """
    Unified FIRE optimization step with MD integration.

    This function dispatches to the appropriate kernel based on:
    - Batching mode: single system, batch_idx, or atom_ptr
    - Downhill check: enabled if all downhill arrays are provided

    Parameters
    ----------
    positions : wp.array, shape (N,) or (N_total,), dtype=wp.vec3*
        Atomic positions (modified in-place).
    velocities : wp.array, shape (N,) or (N_total,), dtype=wp.vec3*
        Atomic velocities (modified in-place).
    forces : wp.array, shape (N,) or (N_total,), dtype=wp.vec3*
        Forces on atoms.
    masses : wp.array, shape (N,) or (N_total,), dtype=wp.float*
        Per-atom masses.
    alpha : wp.array, shape (1,) or (B,), dtype=wp.float*
        FIRE mixing parameter.
    dt : wp.array, shape (1,) or (B,), dtype=wp.float*
        FIRE timestep.
    alpha_start : wp.array, shape (1,) or (B,), dtype=wp.float*
        Reset value for alpha.
    f_alpha : wp.array, shape (1,) or (B,), dtype=wp.float*
        Alpha decay factor.
    dt_min : wp.array, shape (1,) or (B,), dtype=wp.float*
        Minimum timestep.
    dt_max : wp.array, shape (1,) or (B,), dtype=wp.float*
        Maximum timestep.
    maxstep : wp.array, shape (1,) or (B,), dtype=wp.float*
        Maximum displacement per step.
    n_steps_positive : wp.array, shape (1,) or (B,), dtype=wp.int32
        Counter for consecutive positive power steps.
    n_min : wp.array, shape (1,) or (B,), dtype=wp.int32
        Steps before dt increase / alpha decrease.
    f_dec : wp.array, shape (1,) or (B,), dtype=wp.float*
        Timestep decrease factor.
    f_inc : wp.array, shape (1,) or (B,), dtype=wp.float*
        Timestep increase factor.
    vf, vv, ff : wp.array, shape (1,) or (B,), dtype=wp.float*
        Accumulators for diagnostics. Zeroed internally before each use.
        Required for single/batch_idx modes. Ignored for atom_ptr mode.
    uphill_flag : wp.array, shape (B,), dtype=wp.int32, optional
        Scratch array for uphill detection. Shape (B,) where B = num_systems.
        Only used when downhill_enabled=True and batch_idx is provided.
    batch_idx : wp.array, shape (N_total,), dtype=wp.int32, optional
        System index per atom. If provided, uses batch_idx kernel.
    atom_ptr : wp.array, shape (B+1,), dtype=wp.int32, optional
        CSR pointers for atom ranges. If provided, uses ptr kernel.
    energy : wp.array, shape (1,) or (B,), dtype=wp.float*, optional
        Current energies (for downhill check).
    energy_last : wp.array, shape (1,) or (B,), dtype=wp.float*, optional
        Last accepted energies (for downhill check).
    positions_last : wp.array, shape (N,) or (N_total,), dtype=wp.vec3*, optional
        Last accepted positions (for downhill rollback).
    velocities_last : wp.array, shape (N,) or (N_total,), dtype=wp.vec3*, optional
        Last accepted velocities (for downhill rollback).

    Examples
    --------
    Single system (no downhill):

    >>> fire_step(positions, velocities, forces, masses,
    ...           alpha, dt, alpha_start, f_alpha, dt_min, dt_max,
    ...           maxstep, n_steps_positive, n_min, f_dec, f_inc,
    ...           vf, vv, ff)

    Batched with batch_idx:

    >>> fire_step(positions, velocities, forces, masses,
    ...           alpha, dt, alpha_start, f_alpha, dt_min, dt_max,
    ...           maxstep, n_steps_positive, n_min, f_dec, f_inc,
    ...           vf, vv, ff, batch_idx=batch_idx)

    Batched with atom_ptr:

    >>> fire_step(positions, velocities, forces, masses,
    ...           alpha, dt, alpha_start, f_alpha, dt_min, dt_max,
    ...           maxstep, n_steps_positive, n_min, f_dec, f_inc,
    ...           atom_ptr=atom_ptr)

    With downhill check:

    >>> fire_step(positions, velocities, forces, masses,
    ...           alpha, dt, alpha_start, f_alpha, dt_min, dt_max,
    ...           maxstep, n_steps_positive, n_min, f_dec, f_inc,
    ...           vf, vv, ff,
    ...           energy=energy, energy_last=energy_last,
    ...           positions_last=positions_last, velocities_last=velocities_last)
    """
    device = positions.device

    if vf is not None:
        vf.zero_()
        vv.zero_()
        ff.zero_()

    num_atoms = positions.shape[0]
    vec_dtype = positions.dtype

    # Determine batching mode
    exec_mode = resolve_execution_mode(batch_idx, atom_ptr)

    # Determine if downhill check is enabled
    downhill_arrays = [energy, energy_last, positions_last, velocities_last]
    downhill_enabled = all(arr is not None for arr in downhill_arrays)
    if any(arr is not None for arr in downhill_arrays) and not downhill_enabled:
        raise ValueError(
            "For downhill check, must provide ALL of: "
            "energy, energy_last, positions_last, velocities_last"
        )

    # Dispatch to appropriate kernel
    if exec_mode is ExecutionMode.ATOM_PTR:
        # PTR mode – one fused kernel per system
        num_systems = atom_ptr.shape[0] - 1
        kernel = _FIRE_STEP_PTR_OVERLOADS[downhill_enabled][vec_dtype]
        if downhill_enabled:
            inputs = [
                energy,
                forces,
                positions,
                velocities,
                masses,
                alpha,
                dt,
                alpha_start,
                f_alpha,
                dt_min,
                dt_max,
                maxstep,
                n_steps_positive,
                n_min,
                f_dec,
                f_inc,
                energy_last,
                positions_last,
                velocities_last,
                atom_ptr,
            ]
        else:
            inputs = [
                positions,
                velocities,
                forces,
                masses,
                alpha,
                dt,
                alpha_start,
                f_alpha,
                dt_min,
                dt_max,
                maxstep,
                n_steps_positive,
                n_min,
                f_dec,
                f_inc,
                atom_ptr,
            ]
        wp.launch(kernel, dim=num_systems, inputs=inputs, device=device)

    else:
        # BATCH_IDX / SINGLE mode – RLE-based multi-kernel pipeline
        if vf is None or vv is None or ff is None:
            raise ValueError(
                "vf, vv, ff accumulators required for batch_idx/single mode"
            )

        if exec_mode is ExecutionMode.SINGLE:
            batch_idx = wp.zeros(num_atoms, dtype=wp.int32, device=device)

        sm = max(device.sm_count, 1) if hasattr(device, "sm_count") else 1
        ept = compute_ept(num_atoms, sm, is_vec3=True)
        dim_reduce = (num_atoms + ept - 1) // ept

        if downhill_enabled:
            # Kernel 1: Uphill check
            wp.launch(
                _fire_uphill_check_kernel_overload[vec_dtype],
                dim=num_atoms,
                inputs=[energy, energy_last, batch_idx, uphill_flag],
                device=device,
            )

            # Kernel 2: Revert if uphill + RLE reduction
            wp.launch(
                _fire_revert_and_reduce_kernel_overload[vec_dtype],
                dim=dim_reduce,
                inputs=[
                    positions,
                    velocities,
                    forces,
                    positions_last,
                    velocities_last,
                    batch_idx,
                    uphill_flag,
                    vf,
                    vv,
                    ff,
                    num_atoms,
                    ept,
                ],
                device=device,
            )

            # Kernel 3: Parameter update + velocity mixing
            wp.launch(
                _FIRE_STEP_BATCH_UPDATE_OVERLOADS[True][vec_dtype],
                dim=num_atoms,
                inputs=[
                    positions,
                    velocities,
                    forces,
                    masses,
                    batch_idx,
                    alpha,
                    dt,
                    alpha_start,
                    f_alpha,
                    dt_min,
                    dt_max,
                    maxstep,
                    n_steps_positive,
                    n_min,
                    f_dec,
                    f_inc,
                    vf,
                    vv,
                    ff,
                    uphill_flag,
                ],
                device=device,
            )
        else:
            # Kernel 1: RLE-based reduction
            wp.launch(
                _fire_reduce_batch_idx_rle_kernel_overload[vec_dtype],
                dim=dim_reduce,
                inputs=[
                    velocities,
                    forces,
                    batch_idx,
                    vf,
                    vv,
                    ff,
                    num_atoms,
                    ept,
                ],
                device=device,
            )

            # Kernel 2: Parameter update + velocity mixing + position update
            wp.launch(
                _FIRE_STEP_BATCH_UPDATE_OVERLOADS[False][vec_dtype],
                dim=num_atoms,
                inputs=[
                    positions,
                    velocities,
                    forces,
                    masses,
                    batch_idx,
                    alpha,
                    dt,
                    alpha_start,
                    f_alpha,
                    dt_min,
                    dt_max,
                    maxstep,
                    n_steps_positive,
                    n_min,
                    f_dec,
                    f_inc,
                    vf,
                    vv,
                    ff,
                ],
                device=device,
            )


def fire_update(
    # Core arrays (required)
    velocities: wp.array,
    forces: wp.array,
    # FIRE control parameters (required)
    alpha: wp.array,
    dt: wp.array,
    alpha_start: wp.array,
    f_alpha: wp.array,
    dt_min: wp.array,
    dt_max: wp.array,
    n_steps_positive: wp.array,
    n_min: wp.array,
    f_dec: wp.array,
    f_inc: wp.array,
    # Accumulators (required for single/batch_idx; ignored for ptr)
    vf: wp.array = None,
    vv: wp.array = None,
    ff: wp.array = None,
    # Batching (mutually exclusive)
    batch_idx: wp.array = None,
    atom_ptr: wp.array = None,
    # Downhill check (optional - provide ALL or NONE)
    energy: wp.array = None,
    energy_last: wp.array = None,
    positions: wp.array = None,
    positions_last: wp.array = None,
    velocities_last: wp.array = None,
) -> None:
    """
    FIRE parameter update and velocity mixing WITHOUT MD integration.

    Use this for variable-cell optimization where you want to:
    1. Pack atomic + cell DOFs into extended arrays
    2. Apply FIRE velocity mixing to extended velocities
    3. Perform your own MD step (e.g., with cell-aware position scaling)

    This function dispatches to the appropriate "update params" kernel based on:
    - Batching mode: single system, batch_idx, or atom_ptr
    - Downhill check: enabled if all downhill arrays are provided

    Parameters
    ----------
    velocities : wp.array, shape (N,) or (N_total,), dtype=wp.vec3*
        Velocities (modified in-place with FIRE mixing).
    forces : wp.array, shape (N,) or (N_total,), dtype=wp.vec3*
        Atomic forces.
    alpha : wp.array, shape (1,) or (B,), dtype=wp.float*
        FIRE mixing parameter.
    dt : wp.array, shape (1,) or (B,), dtype=wp.float*
        FIRE timestep.
    alpha_start : wp.array, shape (1,) or (B,), dtype=wp.float*
        Reset value for alpha.
    f_alpha : wp.array, shape (1,) or (B,), dtype=wp.float*
        Alpha decay factor.
    dt_min : wp.array, shape (1,) or (B,), dtype=wp.float*
        Minimum timestep.
    dt_max : wp.array, shape (1,) or (B,), dtype=wp.float*
        Maximum timestep.
    n_steps_positive : wp.array, shape (1,) or (B,), dtype=wp.int32
        Counter for consecutive positive power steps.
    n_min : wp.array, shape (1,) or (B,), dtype=wp.int32
        Steps before dt increase / alpha decrease.
    f_dec : wp.array, shape (1,) or (B,), dtype=wp.float*
        Timestep decrease factor.
    f_inc : wp.array, shape (1,) or (B,), dtype=wp.float*
        Timestep increase factor.
    vf, vv, ff : wp.array, shape (1,) or (B,), dtype=wp.float*
        Accumulators for diagnostics. Zeroed internally before each use.
        Required for single/batch_idx modes. Ignored for atom_ptr mode.
    batch_idx : wp.array, shape (N_total,), dtype=wp.int32, optional
        System index per atom. If provided, uses batch_idx kernel.
    atom_ptr : wp.array, shape (B+1,), dtype=wp.int32, optional
        CSR pointers for atom ranges. If provided, uses ptr kernel.
    energy : wp.array, shape (1,) or (B,), dtype=wp.float*, optional
        Current energies (for downhill check).
    energy_last : wp.array, shape (1,) or (B,), dtype=wp.float*, optional
        Last accepted energies (for downhill check).
    positions : wp.array, shape (N,) or (N_total,), dtype=wp.vec3*, optional
        Positions (for downhill rollback). Required if downhill enabled.
    positions_last : wp.array, shape (N,) or (N_total,), dtype=wp.vec3*, optional
        Last accepted positions (for downhill rollback).
    velocities_last : wp.array, shape (N,) or (N_total,), dtype=wp.vec3*, optional
        Last accepted velocities (for downhill rollback).

    Examples
    --------
    Variable-cell optimization workflow:

    >>> # Pack extended arrays (atomic + cell DOFs)
    >>> ext_pos = pack_positions_with_cell(positions, cell)
    >>> ext_vel = pack_velocities_with_cell(velocities, cell_velocity)
    >>> ext_forces = pack_forces_with_cell(forces, cell_force)
    >>>
    >>> # FIRE velocity mixing only (no position update)
    >>> fire_update(ext_vel, ext_forces,
    ...             alpha, dt, alpha_start, f_alpha, dt_min, dt_max,
    ...             n_steps_positive, n_min, f_dec, f_inc,
    ...             vf, vv, ff)
    >>>
    >>> # Perform your own MD step with cell-aware scaling
    >>> ext_vel += dt * ext_forces / ext_masses
    >>> ext_pos += dt * ext_vel  # (with maxstep capping)
    >>>
    >>> # Unpack results
    >>> positions, cell = unpack_positions_with_cell(ext_pos, num_atoms)
    """
    device = velocities.device

    if vf is not None:
        vf.zero_()
        vv.zero_()
        ff.zero_()

    num_atoms = velocities.shape[0]

    # Determine batching mode
    exec_mode = resolve_execution_mode(batch_idx, atom_ptr)

    # Determine if downhill check is enabled
    downhill_arrays = [energy, energy_last, positions, positions_last, velocities_last]
    downhill_enabled = all(arr is not None for arr in downhill_arrays)
    if any(arr is not None for arr in downhill_arrays) and not downhill_enabled:
        raise ValueError(
            "For downhill check, must provide ALL of: "
            "energy, energy_last, positions, positions_last, velocities_last"
        )

    vec_dtype = velocities.dtype

    # Dispatch to appropriate kernel
    if exec_mode is ExecutionMode.ATOM_PTR:
        # PTR mode – one fused kernel per system
        num_systems = atom_ptr.shape[0] - 1
        kernel = _FIRE_UPDATE_PTR_OVERLOADS[downhill_enabled][vec_dtype]
        if downhill_enabled:
            inputs = [
                energy,
                energy_last,
                positions,
                positions_last,
                velocities,
                velocities_last,
                forces,
                alpha,
                dt,
                alpha_start,
                f_alpha,
                dt_min,
                dt_max,
                n_steps_positive,
                n_min,
                f_dec,
                f_inc,
                atom_ptr,
            ]
        else:
            inputs = [
                velocities,
                forces,
                alpha,
                dt,
                alpha_start,
                f_alpha,
                dt_min,
                dt_max,
                n_steps_positive,
                n_min,
                f_dec,
                f_inc,
                atom_ptr,
            ]
        wp.launch(kernel, dim=num_systems, inputs=inputs, device=device)

    else:
        # BATCH_IDX / SINGLE mode – RLE-based multi-kernel pipeline
        if vf is None or vv is None or ff is None:
            raise ValueError(
                "vf, vv, ff accumulators required for batch_idx/single mode"
            )

        if exec_mode is ExecutionMode.SINGLE:
            batch_idx = wp.zeros(num_atoms, dtype=wp.int32, device=device)

        num_systems = dt.shape[0]
        sm = max(device.sm_count, 1) if hasattr(device, "sm_count") else 1
        ept = compute_ept(num_atoms, sm, is_vec3=True)
        dim_reduce = (num_atoms + ept - 1) // ept

        if downhill_enabled:
            uphill_flag = wp.zeros(num_systems, dtype=wp.int32, device=device)

            # Kernel 1: Uphill check
            wp.launch(
                _fire_uphill_check_kernel_overload[vec_dtype],
                dim=num_atoms,
                inputs=[energy, energy_last, batch_idx, uphill_flag],
                device=device,
            )

            # Kernel 2: Revert if uphill + RLE reduction
            wp.launch(
                _fire_revert_and_reduce_kernel_overload[vec_dtype],
                dim=dim_reduce,
                inputs=[
                    positions,
                    velocities,
                    forces,
                    positions_last,
                    velocities_last,
                    batch_idx,
                    uphill_flag,
                    vf,
                    vv,
                    ff,
                    num_atoms,
                    ept,
                ],
                device=device,
            )

            # Kernel 3: Parameter update + velocity mixing (no MD)
            wp.launch(
                _FIRE_UPDATE_BATCH_UPDATE_OVERLOADS[True][vec_dtype],
                dim=num_atoms,
                inputs=[
                    velocities,
                    forces,
                    batch_idx,
                    alpha,
                    dt,
                    alpha_start,
                    f_alpha,
                    dt_min,
                    dt_max,
                    n_steps_positive,
                    n_min,
                    f_dec,
                    f_inc,
                    vf,
                    vv,
                    ff,
                    uphill_flag,
                ],
                device=device,
            )
        else:
            # Kernel 1: RLE-based reduction
            wp.launch(
                _fire_reduce_batch_idx_rle_kernel_overload[vec_dtype],
                dim=dim_reduce,
                inputs=[
                    velocities,
                    forces,
                    batch_idx,
                    vf,
                    vv,
                    ff,
                    num_atoms,
                    ept,
                ],
                device=device,
            )

            # Kernel 2: Parameter update + velocity mixing (no MD)
            wp.launch(
                _FIRE_UPDATE_BATCH_UPDATE_OVERLOADS[False][vec_dtype],
                dim=num_atoms,
                inputs=[
                    velocities,
                    forces,
                    batch_idx,
                    alpha,
                    dt,
                    alpha_start,
                    f_alpha,
                    dt_min,
                    dt_max,
                    n_steps_positive,
                    n_min,
                    f_dec,
                    f_inc,
                    vf,
                    vv,
                    ff,
                ],
                device=device,
            )
