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
Unit tests for FIRE2 optimizer kernels.

Tests cover:
- fire2_step: Complete FIRE2 step with position update
- Correctness against a pure-Python reference implementation
- Single-system and multi-system (batch_idx) batching
- Float32 and float64 support
- Convergence on harmonic and anharmonic potentials
- Batched independent convergence (variable displacement magnitudes)
- Variable atom counts per system
- Algorithmic boundary behaviors (uphill reset, acceleration, dt/maxstep clamping)
- Edge cases (single atom, near-equilibrium, cold start, many systems)
- PyTorch adapter: fire2_step_coord (with/without scratch buffers, correctness, errors)
"""

import math

import numpy as np
import pytest
import torch
import warp as wp

from nvalchemiops.dynamics.optimizers import (
    fire2_apply_step,
    fire2_step,
    fire2_update,
)
from nvalchemiops.torch.fire2 import (
    fire2_compute_extended_reductions,
    fire2_step_coord,
    fire2_step_coord_cell,
    fire2_step_coord_cell_apply,
    fire2_step_coord_cell_couple,
    fire2_step_coord_cell_mix,
    fire2_step_extended,
)

# ==============================================================================
# Configuration
# ==============================================================================

DEVICES = ["cuda:0"]

DTYPE_CONFIGS = [
    pytest.param(wp.vec3f, wp.float32, np.float32, id="float32"),
    pytest.param(wp.vec3d, wp.float64, np.float64, id="float64"),
]

# Default FIRE2 hyperparameters (matching reference)
FIRE2_DEFAULTS = dict(
    delaystep=5,
    dtgrow=1.05,
    dtshrink=0.75,
    alphashrink=0.985,
    alpha0=0.09,
    tmax=0.08,
    tmin=0.005,
    maxstep=0.1,
)

FIRE2_CELL_DEFAULTS = {**FIRE2_DEFAULTS, "cell_force_scale": 1.0}


# ==============================================================================
# Reference implementation (pure Python/NumPy)
# ==============================================================================


def _fire2_reference_step(
    positions,
    velocities,
    forces,
    batch_idx,
    alpha,
    dt,
    nsteps_inc,
    delaystep,
    dtgrow,
    dtshrink,
    alphashrink,
    alpha0,
    tmax,
    tmin,
    maxstep,
):
    """Pure NumPy FIRE2 reference for correctness testing.

    Modifies arrays in-place and returns updated state.
    """
    N = positions.shape[0]
    M = alpha.shape[0]

    # 1. Half-step: v += f * dt[s]
    for i in range(N):
        s = batch_idx[i]
        velocities[i] += forces[i] * dt[s]

    # 2. Inner products per system
    vf = np.zeros(M, dtype=alpha.dtype)
    v_sumsq = np.zeros(M, dtype=alpha.dtype)
    f_sumsq = np.zeros(M, dtype=alpha.dtype)
    for i in range(N):
        s = batch_idx[i]
        vf[s] += np.dot(velocities[i], forces[i])
        v_sumsq[s] += np.dot(velocities[i], velocities[i])
        f_sumsq[s] += np.dot(forces[i], forces[i])

    # 3. Parameter update
    w_dec = np.zeros(M, dtype=bool)
    for s in range(M):
        if vf[s] > 0:
            nsteps_inc[s] += 1
            if nsteps_inc[s] > delaystep:
                dt[s] = min(dtgrow * dt[s], tmax)
                alpha[s] = alphashrink * alpha[s]
        else:
            w_dec[s] = True
            nsteps_inc[s] = 0
            alpha[s] = alpha0
            dt[s] = max(dtshrink * dt[s], tmin)

    # 4. Velocity mixing
    for s in range(M):
        ratio = math.sqrt(v_sumsq[s] / f_sumsq[s]) if f_sumsq[s] > 0 else 0.0
        mix_a = 1.0 - alpha[s]
        mix_b = alpha[s] * ratio
        for i in range(N):
            if batch_idx[i] == s:
                velocities[i] = mix_a * velocities[i] + mix_b * forces[i]

    # 5. Step + uphill correction
    step = np.zeros_like(positions)
    for i in range(N):
        s = batch_idx[i]
        if w_dec[s]:
            step[i] = -0.5 * dt[s] * velocities[i]
            velocities[i] = 0.0
        else:
            step[i] = dt[s] * velocities[i]

    # 6. Max norm per system
    max_norm = np.zeros(M, dtype=alpha.dtype)
    for i in range(N):
        s = batch_idx[i]
        norm = np.linalg.norm(step[i])
        max_norm[s] = max(max_norm[s], norm)

    # 7. Clamping + position update + dt scaling
    for i in range(N):
        s = batch_idx[i]
        inv = min(1.0, maxstep / max_norm[s]) if max_norm[s] > 0 else 1.0
        positions[i] += step[i] * inv
    for s in range(M):
        inv = min(1.0, maxstep / max_norm[s]) if max_norm[s] > 0 else 1.0
        dt[s] *= inv


def _fire2_reference_update(
    velocities: np.ndarray,
    forces: np.ndarray,
    batch_idx: np.ndarray,
    alpha: np.ndarray,
    dt: np.ndarray,
    nsteps_inc: np.ndarray,
    *,
    delaystep: int,
    dtgrow: float,
    dtshrink: float,
    alphashrink: float,
    alpha0: float,
    tmax: float,
    tmin: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pure NumPy reference for fire2_update."""
    N = velocities.shape[0]
    M = alpha.shape[0]
    dt_old = dt.copy()

    velocities_half = velocities.copy()
    for i in range(N):
        s = batch_idx[i]
        velocities_half[i] += dt_old[s] * forces[i]

    vf = np.zeros(M, dtype=alpha.dtype)
    v_sumsq = np.zeros(M, dtype=alpha.dtype)
    f_sumsq = np.zeros(M, dtype=alpha.dtype)
    for i in range(N):
        s = batch_idx[i]
        vf[s] += np.dot(velocities_half[i], forces[i])
        v_sumsq[s] += np.dot(velocities_half[i], velocities_half[i])
        f_sumsq[s] += np.dot(forces[i], forces[i])

    for s in range(M):
        if vf[s] > 0.0:
            nsteps_inc[s] += 1
            if nsteps_inc[s] > delaystep:
                dt[s] = min(dtgrow * dt[s], tmax)
                alpha[s] = alphashrink * alpha[s]
        else:
            nsteps_inc[s] = 0
            alpha[s] = alpha0
            dt[s] = max(dtshrink * dt[s], tmin)

    for s in range(M):
        ratio = math.sqrt(v_sumsq[s] / f_sumsq[s]) if f_sumsq[s] > 0.0 else 0.0
        mix_a = 1.0 - alpha[s]
        mix_b = alpha[s] * ratio
        for i in range(N):
            if batch_idx[i] == s:
                velocities[i] = mix_a * velocities_half[i] + mix_b * forces[i]

    max_norm = np.zeros(M, dtype=alpha.dtype)
    for i in range(N):
        s = batch_idx[i]
        factor = -0.5 if vf[s] <= 0.0 else 1.0
        step = factor * dt[s] * velocities[i]
        max_norm[s] = max(max_norm[s], np.linalg.norm(step))

    return vf, v_sumsq, f_sumsq, max_norm


# ==============================================================================
# Helpers
# ==============================================================================


def make_fire2_scratch(M, dtype_scalar, device):
    """Create zeroed scratch buffers for fire2_step."""
    vf = wp.zeros(M, dtype=dtype_scalar, device=device)
    v_sumsq = wp.zeros(M, dtype=dtype_scalar, device=device)
    f_sumsq = wp.zeros(M, dtype=dtype_scalar, device=device)
    max_norm = wp.zeros(M, dtype=dtype_scalar, device=device)
    return vf, v_sumsq, f_sumsq, max_norm


def _pack_upper_triangular_cell_np(cell: np.ndarray) -> np.ndarray:
    """Pack one upper-triangular cell matrix into the 2x3 FIRE2 layout."""
    return np.array(
        [
            [cell[0, 0], cell[1, 0], cell[2, 0]],
            [cell[1, 1], cell[2, 1], cell[2, 2]],
        ],
        dtype=cell.dtype,
    )


def _unpack_upper_triangular_cell_np(packed: np.ndarray) -> np.ndarray:
    """Unpack the 2x3 FIRE2 cell layout into a 3x3 upper-triangular matrix."""
    return np.array(
        [
            [packed[0, 0], 0.0, 0.0],
            [packed[0, 1], packed[1, 0], 0.0],
            [packed[0, 2], packed[1, 1], packed[1, 2]],
        ],
        dtype=packed.dtype,
    )


def _fractional_coordinates_np(
    positions: np.ndarray, cells: np.ndarray, batch_idx: np.ndarray
) -> np.ndarray:
    """Compute fractional coordinates for row-major NumPy position arrays."""
    cells_inv_t = np.linalg.inv(cells)[batch_idx].transpose(0, 2, 1)
    return np.einsum("ni,nij->nj", positions, cells_inv_t)


def _fire2_coord_cell_reference_step(
    positions: np.ndarray,
    velocities: np.ndarray,
    forces: np.ndarray,
    cell: np.ndarray,
    cell_velocities: np.ndarray,
    cell_force: np.ndarray,
    alpha: np.ndarray,
    dt: np.ndarray,
    nsteps_inc: np.ndarray,
    *,
    delaystep: int,
    dtgrow: float,
    dtshrink: float,
    alphashrink: float,
    alpha0: float,
    tmax: float,
    tmin: float,
    maxstep: float,
    cell_force_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Single-system NumPy reference for the coupled FIRE2 variable-cell step."""
    ext_vel = np.vstack([velocities, _pack_upper_triangular_cell_np(cell_velocities)])
    cell_force_divisor = positions.shape[0] * cell_force_scale
    ext_forces = np.vstack(
        [forces, _pack_upper_triangular_cell_np(cell_force / cell_force_divisor)]
    )

    dt_old = dt[0]
    alpha_old = alpha[0]

    ext_vel_half = ext_vel + dt_old * ext_forces
    vf = np.sum(ext_vel_half * ext_forces)
    v_sumsq = np.sum(ext_vel_half * ext_vel_half)
    f_sumsq = np.sum(ext_forces * ext_forces)

    alpha_new = alpha_old
    dt_new = dt_old
    nsteps_new = nsteps_inc[0]
    if vf > 0.0:
        nsteps_new += 1
        if nsteps_new > delaystep:
            dt_new = min(dtgrow * dt_new, tmax)
            alpha_new = alphashrink * alpha_new
    else:
        nsteps_new = 0
        alpha_new = alpha0
        dt_new = max(dtshrink * dt_new, tmin)

    ratio = math.sqrt(v_sumsq / f_sumsq) if f_sumsq > 0.0 else 0.0
    ext_vel_mixed = (1.0 - alpha_new) * ext_vel_half + alpha_new * ratio * ext_forces

    velocities_mixed = ext_vel_mixed[: positions.shape[0]]
    cell_velocities_mixed = _unpack_upper_triangular_cell_np(
        ext_vel_mixed[positions.shape[0] :]
    )

    factor = -0.5 if vf <= 0.0 else 1.0
    raw_cell_step = factor * dt_new * cell_velocities_mixed
    transform = (cell + raw_cell_step) @ np.linalg.inv(cell)
    raw_displacement = (
        positions @ transform.T + factor * dt_new * velocities_mixed - positions
    )
    max_norm = np.linalg.norm(raw_displacement, axis=1).max()
    inv = min(1.0, maxstep / max_norm) if max_norm > 0.0 else 1.0

    positions_new = positions + inv * raw_displacement
    cell_new = cell + inv * raw_cell_step
    dt_out = np.array([dt_new * inv], dtype=dt.dtype)
    alpha_out = np.array([alpha_new], dtype=alpha.dtype)
    nsteps_out = np.array([nsteps_new], dtype=nsteps_inc.dtype)

    if vf <= 0.0:
        velocities_out = np.zeros_like(velocities_mixed)
        cell_velocities_out = np.zeros_like(cell_velocities_mixed)
    else:
        velocities_out = velocities_mixed
        cell_velocities_out = cell_velocities_mixed

    return (
        positions_new,
        velocities_out,
        cell_new,
        cell_velocities_out,
        alpha_out,
        dt_out,
        nsteps_out,
    )


def make_fire2_state(N, M, dtype_vec, dtype_scalar, np_dtype, device, *, rng=None):
    """Create random FIRE2 state arrays."""
    if rng is None:
        rng = np.random.default_rng(42)

    # Per-atom
    pos_np = rng.standard_normal((N, 3)).astype(np_dtype)
    vel_np = rng.standard_normal((N, 3)).astype(np_dtype) * 0.01
    forces_np = rng.standard_normal((N, 3)).astype(np_dtype)

    # Batch idx: sorted, each system gets N//M atoms
    atoms_per_sys = N // M
    batch_idx_np = np.repeat(np.arange(M, dtype=np.int32), atoms_per_sys)

    # Per-system
    alpha_np = np.full(M, 0.09, dtype=np_dtype)
    dt_np = np.full(M, 0.05, dtype=np_dtype)
    nsteps_inc_np = np.zeros(M, dtype=np.int32)

    positions = wp.array(pos_np, dtype=dtype_vec, device=device)
    velocities = wp.array(vel_np, dtype=dtype_vec, device=device)
    forces = wp.array(forces_np, dtype=dtype_vec, device=device)
    batch_idx = wp.array(batch_idx_np, dtype=wp.int32, device=device)
    alpha = wp.array(alpha_np, dtype=dtype_scalar, device=device)
    dt = wp.array(dt_np, dtype=dtype_scalar, device=device)
    nsteps_inc = wp.array(nsteps_inc_np, dtype=wp.int32, device=device)

    return (
        positions,
        velocities,
        forces,
        batch_idx,
        alpha,
        dt,
        nsteps_inc,
        pos_np.copy(),
        vel_np.copy(),
        forces_np.copy(),
        batch_idx_np.copy(),
        alpha_np.copy(),
        dt_np.copy(),
        nsteps_inc_np.copy(),
    )


def make_fire2_torch_state(N, M, torch_dtype, device, *, rng=None):
    """Create random FIRE2 state as PyTorch tensors."""
    if rng is None:
        rng = np.random.default_rng(42)
    np_dtype = np.float32 if torch_dtype == torch.float32 else np.float64

    pos_np = rng.standard_normal((N, 3)).astype(np_dtype)
    vel_np = rng.standard_normal((N, 3)).astype(np_dtype) * 0.01
    forces_np = rng.standard_normal((N, 3)).astype(np_dtype)
    bidx_np = np.repeat(np.arange(M, dtype=np.int32), N // M)
    alpha_np = np.full(M, 0.09, dtype=np_dtype)
    dt_np = np.full(M, 0.05, dtype=np_dtype)
    nsteps_np = np.zeros(M, dtype=np.int32)

    pos = torch.tensor(pos_np, dtype=torch_dtype, device=device)
    vel = torch.tensor(vel_np, dtype=torch_dtype, device=device)
    forces = torch.tensor(forces_np, dtype=torch_dtype, device=device)
    batch_idx = torch.tensor(bidx_np, dtype=torch.int32, device=device)
    alpha = torch.tensor(alpha_np, dtype=torch_dtype, device=device)
    dt = torch.tensor(dt_np, dtype=torch_dtype, device=device)
    nsteps_inc = torch.tensor(nsteps_np, dtype=torch.int32, device=device)

    return (
        pos,
        vel,
        forces,
        batch_idx,
        alpha,
        dt,
        nsteps_inc,
        pos_np.copy(),
        vel_np.copy(),
        forces_np.copy(),
        bidx_np.copy(),
        alpha_np.copy(),
        dt_np.copy(),
        nsteps_np.copy(),
    )


def make_fire2_variable_state(
    atom_counts, dtype_vec, dtype_scalar, np_dtype, device, *, rng=None
):
    """Create FIRE2 state with variable atoms per system.

    Parameters
    ----------
    atom_counts : list[int]
        Number of atoms per system (M systems total, N = sum).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    M = len(atom_counts)
    N = sum(atom_counts)

    pos_np = rng.standard_normal((N, 3)).astype(np_dtype)
    vel_np = rng.standard_normal((N, 3)).astype(np_dtype) * 0.01
    forces_np = rng.standard_normal((N, 3)).astype(np_dtype)
    batch_idx_np = np.concatenate(
        [np.full(n, i, dtype=np.int32) for i, n in enumerate(atom_counts)]
    )

    alpha_np = np.full(M, 0.09, dtype=np_dtype)
    dt_np = np.full(M, 0.05, dtype=np_dtype)
    nsteps_inc_np = np.zeros(M, dtype=np.int32)

    positions = wp.array(pos_np, dtype=dtype_vec, device=device)
    velocities = wp.array(vel_np, dtype=dtype_vec, device=device)
    forces = wp.array(forces_np, dtype=dtype_vec, device=device)
    batch_idx = wp.array(batch_idx_np, dtype=wp.int32, device=device)
    alpha = wp.array(alpha_np, dtype=dtype_scalar, device=device)
    dt = wp.array(dt_np, dtype=dtype_scalar, device=device)
    nsteps_inc = wp.array(nsteps_inc_np, dtype=wp.int32, device=device)

    return (
        positions,
        velocities,
        forces,
        batch_idx,
        alpha,
        dt,
        nsteps_inc,
        pos_np.copy(),
        vel_np.copy(),
        forces_np.copy(),
        batch_idx_np.copy(),
        alpha_np.copy(),
        dt_np.copy(),
        nsteps_inc_np.copy(),
    )


# ==============================================================================
# Tests: FIRE2 Warp Kernels
# ==============================================================================


class TestFire2Step:
    """Correctness tests for fire2_step against NumPy reference."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_fire2_update_matches_reference(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """fire2_update matches reference and leaves final apply to callers."""
        rng = np.random.default_rng(401)
        N, M = 9, 3
        vel_np = rng.standard_normal((N, 3)).astype(np_dtype) * 0.2
        forces_np = rng.standard_normal((N, 3)).astype(np_dtype) * 0.1
        bidx_np = np.repeat(np.arange(M, dtype=np.int32), N // M)

        sys0 = bidx_np == 0
        sys1 = bidx_np == 1
        sys2 = bidx_np == 2
        forces_np[sys0] = vel_np[sys0]
        forces_np[sys1] = -vel_np[sys1]
        forces_np[sys2] = 0.5 * vel_np[sys2]

        alpha_np = np.array([0.07, 0.04, 0.11], dtype=np_dtype)
        dt_np = np.array([0.05, 0.04, 0.03], dtype=np_dtype)
        nsteps_np = np.array([5, 8, 4], dtype=np.int32)

        vel_ref = vel_np.copy()
        alpha_ref = alpha_np.copy()
        dt_ref = dt_np.copy()
        nsteps_ref = nsteps_np.copy()
        vf_ref, v_sumsq_ref, f_sumsq_ref, max_norm_ref = _fire2_reference_update(
            vel_ref,
            forces_np,
            bidx_np,
            alpha_ref,
            dt_ref,
            nsteps_ref,
            delaystep=FIRE2_DEFAULTS["delaystep"],
            dtgrow=FIRE2_DEFAULTS["dtgrow"],
            dtshrink=FIRE2_DEFAULTS["dtshrink"],
            alphashrink=FIRE2_DEFAULTS["alphashrink"],
            alpha0=FIRE2_DEFAULTS["alpha0"],
            tmax=FIRE2_DEFAULTS["tmax"],
            tmin=FIRE2_DEFAULTS["tmin"],
        )

        vel = wp.array(vel_np.copy(), dtype=dtype_vec, device=device)
        forces = wp.array(forces_np.copy(), dtype=dtype_vec, device=device)
        bidx = wp.array(bidx_np.copy(), dtype=wp.int32, device=device)
        alpha = wp.array(alpha_np.copy(), dtype=dtype_scalar, device=device)
        dt = wp.array(dt_np.copy(), dtype=dtype_scalar, device=device)
        nsteps_inc = wp.array(nsteps_np.copy(), dtype=wp.int32, device=device)
        vf = wp.array(
            np.full(M, 999.0, dtype=np_dtype), dtype=dtype_scalar, device=device
        )
        v_sumsq = wp.array(
            np.full(M, 999.0, dtype=np_dtype), dtype=dtype_scalar, device=device
        )
        f_sumsq = wp.array(
            np.full(M, 999.0, dtype=np_dtype), dtype=dtype_scalar, device=device
        )
        max_norm = wp.array(
            np.full(M, 999.0, dtype=np_dtype), dtype=dtype_scalar, device=device
        )

        fire2_update(
            vel,
            forces,
            bidx,
            alpha,
            dt,
            nsteps_inc,
            vf,
            v_sumsq,
            f_sumsq,
            max_norm,
            delaystep=FIRE2_DEFAULTS["delaystep"],
            dtgrow=FIRE2_DEFAULTS["dtgrow"],
            dtshrink=FIRE2_DEFAULTS["dtshrink"],
            alphashrink=FIRE2_DEFAULTS["alphashrink"],
            alpha0=FIRE2_DEFAULTS["alpha0"],
            tmax=FIRE2_DEFAULTS["tmax"],
            tmin=FIRE2_DEFAULTS["tmin"],
        )
        wp.synchronize()

        rtol = 1e-4 if np_dtype == np.float32 else 1e-10
        np.testing.assert_allclose(vel.numpy(), vel_ref, rtol=rtol, atol=1e-7)
        np.testing.assert_allclose(alpha.numpy(), alpha_ref, rtol=rtol)
        np.testing.assert_allclose(dt.numpy(), dt_ref, rtol=rtol)
        np.testing.assert_array_equal(nsteps_inc.numpy(), nsteps_ref)
        np.testing.assert_allclose(vf.numpy(), vf_ref, rtol=rtol, atol=1e-7)
        np.testing.assert_allclose(v_sumsq.numpy(), v_sumsq_ref, rtol=rtol, atol=1e-7)
        np.testing.assert_allclose(f_sumsq.numpy(), f_sumsq_ref, rtol=rtol, atol=1e-7)
        np.testing.assert_allclose(max_norm.numpy(), max_norm_ref, rtol=rtol, atol=1e-7)

        uphill = vf_ref <= 0.0
        assert uphill.any()
        assert np.linalg.norm(vel.numpy()[bidx_np == np.nonzero(uphill)[0][0]]) > 0.0

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_fire2_update_can_skip_max_norm(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """fire2_update can leave max_norm untouched for custom final apply phases."""
        rng = np.random.default_rng(402)
        N, M = 8, 2
        vel_np = rng.standard_normal((N, 3)).astype(np_dtype) * 0.2
        forces_np = rng.standard_normal((N, 3)).astype(np_dtype) * 0.1
        bidx_np = np.repeat(np.arange(M, dtype=np.int32), N // M)
        alpha_np = np.array([0.07, 0.04], dtype=np_dtype)
        dt_np = np.array([0.05, 0.04], dtype=np_dtype)
        nsteps_np = np.array([5, 8], dtype=np.int32)

        vel_ref = vel_np.copy()
        alpha_ref = alpha_np.copy()
        dt_ref = dt_np.copy()
        nsteps_ref = nsteps_np.copy()
        vf_ref, v_sumsq_ref, f_sumsq_ref, _ = _fire2_reference_update(
            vel_ref,
            forces_np,
            bidx_np,
            alpha_ref,
            dt_ref,
            nsteps_ref,
            delaystep=FIRE2_DEFAULTS["delaystep"],
            dtgrow=FIRE2_DEFAULTS["dtgrow"],
            dtshrink=FIRE2_DEFAULTS["dtshrink"],
            alphashrink=FIRE2_DEFAULTS["alphashrink"],
            alpha0=FIRE2_DEFAULTS["alpha0"],
            tmax=FIRE2_DEFAULTS["tmax"],
            tmin=FIRE2_DEFAULTS["tmin"],
        )

        vel = wp.array(vel_np.copy(), dtype=dtype_vec, device=device)
        forces = wp.array(forces_np.copy(), dtype=dtype_vec, device=device)
        bidx = wp.array(bidx_np.copy(), dtype=wp.int32, device=device)
        alpha = wp.array(alpha_np.copy(), dtype=dtype_scalar, device=device)
        dt = wp.array(dt_np.copy(), dtype=dtype_scalar, device=device)
        nsteps_inc = wp.array(nsteps_np.copy(), dtype=wp.int32, device=device)
        vf = wp.zeros(M, dtype=dtype_scalar, device=device)
        v_sumsq = wp.zeros(M, dtype=dtype_scalar, device=device)
        f_sumsq = wp.zeros(M, dtype=dtype_scalar, device=device)
        max_norm = wp.array(
            np.full(M, 123.0, dtype=np_dtype), dtype=dtype_scalar, device=device
        )

        fire2_update(
            vel,
            forces,
            bidx,
            alpha,
            dt,
            nsteps_inc,
            vf,
            v_sumsq,
            f_sumsq,
            max_norm,
            compute_max_norm=False,
            delaystep=FIRE2_DEFAULTS["delaystep"],
            dtgrow=FIRE2_DEFAULTS["dtgrow"],
            dtshrink=FIRE2_DEFAULTS["dtshrink"],
            alphashrink=FIRE2_DEFAULTS["alphashrink"],
            alpha0=FIRE2_DEFAULTS["alpha0"],
            tmax=FIRE2_DEFAULTS["tmax"],
            tmin=FIRE2_DEFAULTS["tmin"],
        )
        wp.synchronize()

        rtol = 1e-4 if np_dtype == np.float32 else 1e-10
        np.testing.assert_allclose(vel.numpy(), vel_ref, rtol=rtol, atol=1e-7)
        np.testing.assert_allclose(alpha.numpy(), alpha_ref, rtol=rtol)
        np.testing.assert_allclose(dt.numpy(), dt_ref, rtol=rtol)
        np.testing.assert_array_equal(nsteps_inc.numpy(), nsteps_ref)
        np.testing.assert_allclose(vf.numpy(), vf_ref, rtol=rtol, atol=1e-7)
        np.testing.assert_allclose(v_sumsq.numpy(), v_sumsq_ref, rtol=rtol, atol=1e-7)
        np.testing.assert_allclose(f_sumsq.numpy(), f_sumsq_ref, rtol=rtol, atol=1e-7)
        np.testing.assert_allclose(max_norm.numpy(), np.full(M, 123.0, dtype=np_dtype))

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_fire2_step_zero_atoms_returns(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """fire2_step exits early for zero atoms and zeroes scratch buffers."""
        M = 2
        positions = wp.empty(0, dtype=dtype_vec, device=device)
        velocities = wp.empty(0, dtype=dtype_vec, device=device)
        forces = wp.empty(0, dtype=dtype_vec, device=device)
        alpha = wp.array(
            np.array([0.07, 0.04], dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        dt = wp.array(
            np.array([0.05, 0.04], dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        nsteps_inc = wp.array(
            np.array([5, 8], dtype=np.int32), dtype=wp.int32, device=device
        )
        vf = wp.array(
            np.full(M, 9.0, dtype=np_dtype), dtype=dtype_scalar, device=device
        )
        v_sumsq = wp.array(
            np.full(M, 8.0, dtype=np_dtype), dtype=dtype_scalar, device=device
        )
        f_sumsq = wp.array(
            np.full(M, 7.0, dtype=np_dtype), dtype=dtype_scalar, device=device
        )
        max_norm = wp.array(
            np.full(M, 6.0, dtype=np_dtype), dtype=dtype_scalar, device=device
        )

        alpha_before = alpha.numpy().copy()
        dt_before = dt.numpy().copy()
        nsteps_before = nsteps_inc.numpy().copy()

        fire2_step(
            positions,
            velocities,
            forces,
            None,
            alpha,
            dt,
            nsteps_inc,
            vf,
            v_sumsq,
            f_sumsq,
            max_norm,
            **FIRE2_DEFAULTS,
        )
        wp.synchronize()

        np.testing.assert_allclose(alpha.numpy(), alpha_before)
        np.testing.assert_allclose(dt.numpy(), dt_before)
        np.testing.assert_array_equal(nsteps_inc.numpy(), nsteps_before)
        np.testing.assert_allclose(vf.numpy(), np.zeros(M))
        np.testing.assert_allclose(v_sumsq.numpy(), np.zeros(M))
        np.testing.assert_allclose(f_sumsq.numpy(), np.zeros(M))
        np.testing.assert_allclose(max_norm.numpy(), np.zeros(M))

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_fire2_update_zero_atoms_returns(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """fire2_update exits early for zero atoms and zeroes scratch buffers."""
        M = 2
        velocities = wp.empty(0, dtype=dtype_vec, device=device)
        forces = wp.empty(0, dtype=dtype_vec, device=device)
        alpha = wp.array(
            np.array([0.07, 0.04], dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        dt = wp.array(
            np.array([0.05, 0.04], dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        nsteps_inc = wp.array(
            np.array([5, 8], dtype=np.int32), dtype=wp.int32, device=device
        )
        vf = wp.array(
            np.full(M, 9.0, dtype=np_dtype), dtype=dtype_scalar, device=device
        )
        v_sumsq = wp.array(
            np.full(M, 8.0, dtype=np_dtype), dtype=dtype_scalar, device=device
        )
        f_sumsq = wp.array(
            np.full(M, 7.0, dtype=np_dtype), dtype=dtype_scalar, device=device
        )
        max_norm = wp.array(
            np.full(M, 6.0, dtype=np_dtype), dtype=dtype_scalar, device=device
        )

        alpha_before = alpha.numpy().copy()
        dt_before = dt.numpy().copy()
        nsteps_before = nsteps_inc.numpy().copy()

        fire2_update(
            velocities,
            forces,
            None,
            alpha,
            dt,
            nsteps_inc,
            vf,
            v_sumsq,
            f_sumsq,
            max_norm,
            compute_max_norm=False,
        )
        wp.synchronize()

        np.testing.assert_allclose(alpha.numpy(), alpha_before)
        np.testing.assert_allclose(dt.numpy(), dt_before)
        np.testing.assert_array_equal(nsteps_inc.numpy(), nsteps_before)
        np.testing.assert_allclose(vf.numpy(), np.zeros(M))
        np.testing.assert_allclose(v_sumsq.numpy(), np.zeros(M))
        np.testing.assert_allclose(f_sumsq.numpy(), np.zeros(M))
        np.testing.assert_allclose(max_norm.numpy(), np.zeros(M))

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_single_system(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Single system: fire2_step matches reference."""
        N, M = 50, 1
        (
            pos,
            vel,
            forces,
            bidx,
            alpha,
            dt,
            nsteps_inc,
            pos_np,
            vel_np,
            forces_np,
            bidx_np,
            alpha_np,
            dt_np,
            nsteps_np,
        ) = make_fire2_state(N, M, dtype_vec, dtype_scalar, np_dtype, device)

        vf, v_sumsq, f_sumsq, max_norm = make_fire2_scratch(M, dtype_scalar, device)

        fire2_step(
            pos,
            vel,
            forces,
            bidx,
            alpha,
            dt,
            nsteps_inc,
            vf,
            v_sumsq,
            f_sumsq,
            max_norm,
            **FIRE2_DEFAULTS,
        )
        wp.synchronize()

        _fire2_reference_step(
            pos_np,
            vel_np,
            forces_np,
            bidx_np,
            alpha_np,
            dt_np,
            nsteps_np,
            **FIRE2_DEFAULTS,
        )

        rtol = 1e-4 if np_dtype == np.float32 else 1e-10
        np.testing.assert_allclose(pos.numpy(), pos_np, rtol=rtol, atol=1e-7)
        np.testing.assert_allclose(vel.numpy(), vel_np, rtol=rtol, atol=1e-7)
        np.testing.assert_allclose(alpha.numpy(), alpha_np, rtol=rtol)
        np.testing.assert_allclose(dt.numpy(), dt_np, rtol=rtol)
        np.testing.assert_array_equal(nsteps_inc.numpy(), nsteps_np)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_multi_system(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Multi-system: fire2_step matches reference."""
        N, M = 120, 4  # 30 atoms per system
        (
            pos,
            vel,
            forces,
            bidx,
            alpha,
            dt,
            nsteps_inc,
            pos_np,
            vel_np,
            forces_np,
            bidx_np,
            alpha_np,
            dt_np,
            nsteps_np,
        ) = make_fire2_state(N, M, dtype_vec, dtype_scalar, np_dtype, device)

        vf, v_sumsq, f_sumsq, max_norm = make_fire2_scratch(M, dtype_scalar, device)

        fire2_step(
            pos,
            vel,
            forces,
            bidx,
            alpha,
            dt,
            nsteps_inc,
            vf,
            v_sumsq,
            f_sumsq,
            max_norm,
            **FIRE2_DEFAULTS,
        )
        wp.synchronize()

        _fire2_reference_step(
            pos_np,
            vel_np,
            forces_np,
            bidx_np,
            alpha_np,
            dt_np,
            nsteps_np,
            **FIRE2_DEFAULTS,
        )

        rtol = 1e-4 if np_dtype == np.float32 else 1e-10
        np.testing.assert_allclose(pos.numpy(), pos_np, rtol=rtol, atol=1e-7)
        np.testing.assert_allclose(vel.numpy(), vel_np, rtol=rtol, atol=1e-7)
        np.testing.assert_allclose(alpha.numpy(), alpha_np, rtol=rtol)
        np.testing.assert_allclose(dt.numpy(), dt_np, rtol=rtol)

    @pytest.mark.parametrize("device", DEVICES)
    def test_multiple_steps(self, device):
        """Run several FIRE2 steps and verify state remains consistent."""
        N, M = 60, 2
        dtype_vec, dtype_scalar, np_dtype = wp.vec3f, wp.float32, np.float32
        (
            pos,
            vel,
            forces,
            bidx,
            alpha,
            dt,
            nsteps_inc,
            pos_np,
            vel_np,
            forces_np,
            bidx_np,
            alpha_np,
            dt_np,
            nsteps_np,
        ) = make_fire2_state(N, M, dtype_vec, dtype_scalar, np_dtype, device)

        for _ in range(5):
            vf, v_sumsq, f_sumsq, max_norm = make_fire2_scratch(M, dtype_scalar, device)
            fire2_step(
                pos,
                vel,
                forces,
                bidx,
                alpha,
                dt,
                nsteps_inc,
                vf,
                v_sumsq,
                f_sumsq,
                max_norm,
                **FIRE2_DEFAULTS,
            )
            _fire2_reference_step(
                pos_np,
                vel_np,
                forces_np,
                bidx_np,
                alpha_np,
                dt_np,
                nsteps_np,
                **FIRE2_DEFAULTS,
            )

        wp.synchronize()
        np.testing.assert_allclose(pos.numpy(), pos_np, rtol=1e-3, atol=1e-5)
        np.testing.assert_allclose(vel.numpy(), vel_np, rtol=1e-3, atol=1e-5)


# ==============================================================================
# Tests: FIRE2 Convergence
# ==============================================================================


class TestFire2Convergence:
    """Test FIRE2 convergence on a simple harmonic potential."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_harmonic_convergence(self, device):
        """Minimize E = 0.5 * sum(x^2) from random initial positions."""
        N, M = 20, 1
        rng = np.random.default_rng(99)
        np_dtype = np.float64
        dtype_vec, dtype_scalar = wp.vec3d, wp.float64

        pos_np = rng.standard_normal((N, 3)).astype(np_dtype)
        vel_np = np.zeros((N, 3), dtype=np_dtype)
        bidx_np = np.zeros(N, dtype=np.int32)
        alpha_np = np.full(1, 0.09, dtype=np_dtype)
        dt_np = np.full(1, 0.05, dtype=np_dtype)
        nsteps_np = np.zeros(1, dtype=np.int32)

        pos = wp.array(pos_np, dtype=dtype_vec, device=device)
        vel = wp.array(vel_np, dtype=dtype_vec, device=device)
        bidx = wp.array(bidx_np, dtype=wp.int32, device=device)
        alpha = wp.array(alpha_np, dtype=dtype_scalar, device=device)
        dt = wp.array(dt_np, dtype=dtype_scalar, device=device)
        nsteps_inc = wp.array(nsteps_np, dtype=wp.int32, device=device)

        for _ in range(200):
            # Forces = -grad(E) = -x
            pos_current = pos.numpy()
            forces_np = -pos_current
            forces = wp.array(forces_np, dtype=dtype_vec, device=device)

            vf, v_sumsq, f_sumsq, max_norm = make_fire2_scratch(M, dtype_scalar, device)
            fire2_step(
                pos,
                vel,
                forces,
                bidx,
                alpha,
                dt,
                nsteps_inc,
                vf,
                v_sumsq,
                f_sumsq,
                max_norm,
                **FIRE2_DEFAULTS,
            )
            wp.synchronize()

        final_pos = pos.numpy()
        max_displacement = np.max(np.abs(final_pos))
        assert max_displacement < 0.1, (
            f"FIRE2 should converge to origin; max |x| = {max_displacement}"
        )

    @pytest.mark.parametrize("device", DEVICES)
    def test_batched_independent_convergence(self, device):
        """Two systems with different displacements converge independently."""
        N1, N2 = 15, 25
        N = N1 + N2
        M = 2
        rng = np.random.default_rng(123)
        np_dtype = np.float64
        dtype_vec, dtype_scalar = wp.vec3d, wp.float64

        pos_np = np.zeros((N, 3), dtype=np_dtype)
        pos_np[:N1] = rng.standard_normal((N1, 3)) * 0.5  # small
        pos_np[N1:] = rng.standard_normal((N2, 3)) * 2.0  # large
        vel_np = np.zeros((N, 3), dtype=np_dtype)
        bidx_np = np.concatenate(
            [
                np.zeros(N1, dtype=np.int32),
                np.ones(N2, dtype=np.int32),
            ]
        )
        alpha_np = np.full(M, 0.09, dtype=np_dtype)
        dt_np = np.full(M, 0.05, dtype=np_dtype)
        nsteps_np = np.zeros(M, dtype=np.int32)

        pos = wp.array(pos_np, dtype=dtype_vec, device=device)
        vel = wp.array(vel_np, dtype=dtype_vec, device=device)
        bidx = wp.array(bidx_np, dtype=wp.int32, device=device)
        alpha = wp.array(alpha_np, dtype=dtype_scalar, device=device)
        dt_arr = wp.array(dt_np, dtype=dtype_scalar, device=device)
        nsteps_inc = wp.array(nsteps_np, dtype=wp.int32, device=device)

        for _ in range(300):
            # Harmonic: forces = -x
            pos_current = pos.numpy()
            forces_np = -pos_current
            forces = wp.array(forces_np, dtype=dtype_vec, device=device)

            vf, v_sumsq, f_sumsq, max_norm = make_fire2_scratch(M, dtype_scalar, device)
            fire2_step(
                pos,
                vel,
                forces,
                bidx,
                alpha,
                dt_arr,
                nsteps_inc,
                vf,
                v_sumsq,
                f_sumsq,
                max_norm,
                **FIRE2_DEFAULTS,
            )
            wp.synchronize()

        final_pos = pos.numpy()
        max_disp_0 = np.max(np.abs(final_pos[:N1]))
        max_disp_1 = np.max(np.abs(final_pos[N1:]))
        assert max_disp_0 < 0.1, (
            f"System 0 (small init) should converge; max |x| = {max_disp_0}"
        )
        assert max_disp_1 < 0.5, (
            f"System 1 (large init) should converge; max |x| = {max_disp_1}"
        )

    @pytest.mark.parametrize("device", DEVICES)
    def test_anharmonic_convergence(self, device):
        """Minimize Morse dimer pair potential (anharmonic, equilibrium at r0).

        Uses a pair of atoms with Morse potential V = D*(1-exp(-a*(r-r0)))^2
        where r is the inter-atomic distance. The equilibrium distance is r0.
        Also exercises the f_sumsq=0 guard in the mixing kernel as forces
        approach zero near convergence.
        """
        N, M = 2, 1
        np_dtype = np.float64
        dtype_vec, dtype_scalar = wp.vec3d, wp.float64
        D, a, r0 = 1.0, 2.0, 1.5

        # Start the dimer stretched beyond equilibrium
        pos_np = np.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]], dtype=np_dtype)
        vel_np = np.zeros((N, 3), dtype=np_dtype)
        bidx_np = np.zeros(N, dtype=np.int32)
        alpha_np = np.full(1, 0.09, dtype=np_dtype)
        dt_np = np.full(1, 0.02, dtype=np_dtype)
        nsteps_np = np.zeros(1, dtype=np.int32)

        pos = wp.array(pos_np, dtype=dtype_vec, device=device)
        vel = wp.array(vel_np, dtype=dtype_vec, device=device)
        bidx = wp.array(bidx_np, dtype=wp.int32, device=device)
        alpha = wp.array(alpha_np, dtype=dtype_scalar, device=device)
        dt_arr = wp.array(dt_np, dtype=dtype_scalar, device=device)
        nsteps_inc = wp.array(nsteps_np, dtype=wp.int32, device=device)

        for _ in range(300):
            pos_current = pos.numpy()
            r_vec = pos_current[1] - pos_current[0]
            r = np.linalg.norm(r_vec)
            r_hat = r_vec / r
            exp_term = np.exp(-a * (r - r0))
            # dV/dr = 2*D*a*(1-exp)*exp; positive when r > r0 (attractive)
            dVdr = 2.0 * D * a * (1.0 - exp_term) * exp_term
            forces_np = np.zeros_like(pos_current)
            forces_np[0] = dVdr * r_hat  # toward atom 1
            forces_np[1] = -dVdr * r_hat  # toward atom 0

            forces = wp.array(forces_np, dtype=dtype_vec, device=device)
            vf, v_sumsq, f_sumsq, max_norm = make_fire2_scratch(M, dtype_scalar, device)
            fire2_step(
                pos,
                vel,
                forces,
                bidx,
                alpha,
                dt_arr,
                nsteps_inc,
                vf,
                v_sumsq,
                f_sumsq,
                max_norm,
                **FIRE2_DEFAULTS,
            )
            wp.synchronize()

        final_pos = pos.numpy()
        assert np.isfinite(final_pos).all(), "Outputs should be finite"
        final_r = np.linalg.norm(final_pos[1] - final_pos[0])
        assert abs(final_r - r0) < 0.1, (
            f"Morse dimer should converge to r0={r0}; got r={final_r}"
        )


# ==============================================================================
# Tests: PyTorch Adapter
# ==============================================================================


class TestFire2TorchCoord:
    """Tests for the PyTorch adapter fire2_step_coord."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("torch_dtype", [torch.float32, torch.float64])
    def test_coord_step_basic(self, device, torch_dtype):
        """Positions should change and all outputs should be finite."""
        rng = np.random.default_rng(7)
        N, M = 60, 2
        (
            pos,
            vel,
            forces,
            batch_idx,
            alpha,
            dt,
            nsteps_inc,
            *_,
        ) = make_fire2_torch_state(N, M, torch_dtype, device, rng=rng)

        pos_before = pos.clone()
        fire2_step_coord(
            pos,
            vel,
            forces,
            batch_idx,
            alpha,
            dt,
            nsteps_inc,
            **FIRE2_DEFAULTS,
        )
        torch.cuda.synchronize()

        assert not torch.allclose(pos, pos_before), "Positions should be updated"
        assert torch.isfinite(pos).all(), "Positions should be finite"
        assert torch.isfinite(vel).all(), "Velocities should be finite"
        assert torch.isfinite(alpha).all(), "Alpha should be finite"
        assert torch.isfinite(dt).all(), "dt should be finite"

    @pytest.mark.parametrize("device", DEVICES)
    def test_coord_step_requires_contiguous(self, device):
        """fire2_step_coord raises RuntimeError for non-contiguous compound-type tensors."""
        N, M = 20, 1
        dtype = torch.float32
        rng = np.random.default_rng(42)
        (
            pos,
            vel,
            forces,
            batch_idx,
            alpha,
            dt,
            nsteps_inc,
            *_,
        ) = make_fire2_torch_state(N, M, dtype, device, rng=rng)
        # Non-contiguous view: (N, 3, 2) -> [:, :, 0] has shape (N, 3) but is not contiguous
        base = torch.randn(N, 3, 2, device=device, dtype=dtype)
        pos_view = base[:, :, 0]
        assert not pos_view.is_contiguous()
        with pytest.raises(RuntimeError):
            fire2_step_coord(
                pos_view,
                vel,
                forces,
                batch_idx,
                alpha,
                dt,
                nsteps_inc,
                **FIRE2_DEFAULTS,
            )

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("torch_dtype", [torch.float32, torch.float64])
    def test_coord_step_correctness(self, device, torch_dtype):
        """Verify fire2_step_coord matches NumPy reference."""
        rng = np.random.default_rng(10)
        N, M = 60, 2
        np_dtype = np.float32 if torch_dtype == torch.float32 else np.float64

        (
            pos,
            vel,
            forces,
            batch_idx,
            alpha,
            dt,
            nsteps_inc,
            pos_np,
            vel_np,
            forces_np,
            bidx_np,
            alpha_np,
            dt_np,
            nsteps_np,
        ) = make_fire2_torch_state(N, M, torch_dtype, device, rng=rng)

        fire2_step_coord(
            pos,
            vel,
            forces,
            batch_idx,
            alpha,
            dt,
            nsteps_inc,
            **FIRE2_DEFAULTS,
        )
        torch.cuda.synchronize()

        _fire2_reference_step(
            pos_np,
            vel_np,
            forces_np,
            bidx_np,
            alpha_np,
            dt_np,
            nsteps_np,
            **FIRE2_DEFAULTS,
        )

        rtol = 1e-4 if np_dtype == np.float32 else 1e-10
        np.testing.assert_allclose(
            pos.cpu().numpy(),
            pos_np,
            rtol=rtol,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            vel.cpu().numpy(),
            vel_np,
            rtol=rtol,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            alpha.cpu().numpy(),
            alpha_np,
            rtol=rtol,
        )
        np.testing.assert_allclose(
            dt.cpu().numpy(),
            dt_np,
            rtol=rtol,
        )
        np.testing.assert_array_equal(
            nsteps_inc.cpu().numpy(),
            nsteps_np,
        )

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("torch_dtype", [torch.float32, torch.float64])
    def test_coord_step_with_scratch(self, device, torch_dtype):
        """Verify fire2_step_coord works with pre-allocated scratch buffers."""
        rng = np.random.default_rng(11)
        N, M = 60, 2
        np_dtype = np.float32 if torch_dtype == torch.float32 else np.float64

        (
            pos,
            vel,
            forces,
            batch_idx,
            alpha,
            dt,
            nsteps_inc,
            pos_np,
            vel_np,
            forces_np,
            bidx_np,
            alpha_np,
            dt_np,
            nsteps_np,
        ) = make_fire2_torch_state(N, M, torch_dtype, device, rng=rng)

        # Pre-allocate scratch buffers (with junk data to verify zeroing)
        vf = torch.ones(M, dtype=torch_dtype, device=device) * 999.0
        v_sumsq = torch.ones(M, dtype=torch_dtype, device=device) * 999.0
        f_sumsq = torch.ones(M, dtype=torch_dtype, device=device) * 999.0
        max_norm_buf = torch.ones(M, dtype=torch_dtype, device=device) * 999.0

        fire2_step_coord(
            pos,
            vel,
            forces,
            batch_idx,
            alpha,
            dt,
            nsteps_inc,
            vf=vf,
            v_sumsq=v_sumsq,
            f_sumsq=f_sumsq,
            max_norm=max_norm_buf,
            **FIRE2_DEFAULTS,
        )
        torch.cuda.synchronize()

        _fire2_reference_step(
            pos_np,
            vel_np,
            forces_np,
            bidx_np,
            alpha_np,
            dt_np,
            nsteps_np,
            **FIRE2_DEFAULTS,
        )

        rtol = 1e-4 if np_dtype == np.float32 else 1e-10
        np.testing.assert_allclose(
            pos.cpu().numpy(),
            pos_np,
            rtol=rtol,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            vel.cpu().numpy(),
            vel_np,
            rtol=rtol,
            atol=1e-7,
        )

    @pytest.mark.parametrize("device", DEVICES)
    def test_coord_step_scratch_reuse(self, device):
        """Verify scratch buffers can be reused across multiple steps."""
        rng = np.random.default_rng(12)
        N, M = 40, 2
        torch_dtype = torch.float64

        (
            pos,
            vel,
            forces,
            batch_idx,
            alpha,
            dt,
            nsteps_inc,
            pos_np,
            vel_np,
            forces_np,
            bidx_np,
            alpha_np,
            dt_np,
            nsteps_np,
        ) = make_fire2_torch_state(N, M, torch_dtype, device, rng=rng)

        # Pre-allocate scratch buffers once
        vf = torch.empty(M, dtype=torch_dtype, device=device)
        v_sumsq = torch.empty(M, dtype=torch_dtype, device=device)
        f_sumsq = torch.empty(M, dtype=torch_dtype, device=device)
        max_norm_buf = torch.empty(M, dtype=torch_dtype, device=device)

        for _ in range(5):
            fire2_step_coord(
                pos,
                vel,
                forces,
                batch_idx,
                alpha,
                dt,
                nsteps_inc,
                vf=vf,
                v_sumsq=v_sumsq,
                f_sumsq=f_sumsq,
                max_norm=max_norm_buf,
                **FIRE2_DEFAULTS,
            )
            _fire2_reference_step(
                pos_np,
                vel_np,
                forces_np,
                bidx_np,
                alpha_np,
                dt_np,
                nsteps_np,
                **FIRE2_DEFAULTS,
            )

        torch.cuda.synchronize()
        np.testing.assert_allclose(
            pos.cpu().numpy(),
            pos_np,
            rtol=1e-8,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            vel.cpu().numpy(),
            vel_np,
            rtol=1e-8,
            atol=1e-7,
        )

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("torch_dtype", [torch.float32, torch.float64])
    def test_torch_fire2_zero_atoms_return(self, device, torch_dtype):
        """Torch FIRE2 adapters return early for empty coordinate arrays."""
        M = 2
        positions = torch.empty(0, 3, dtype=torch_dtype, device=device)
        velocities = torch.empty(0, 3, dtype=torch_dtype, device=device)
        forces = torch.empty(0, 3, dtype=torch_dtype, device=device)
        batch_idx = torch.empty(0, dtype=torch.int32, device=device)
        alpha = torch.tensor([0.07, 0.04], dtype=torch_dtype, device=device)
        dt = torch.tensor([0.05, 0.04], dtype=torch_dtype, device=device)
        nsteps_inc = torch.tensor([5, 8], dtype=torch.int32, device=device)
        vf = torch.full((M,), 9.0, dtype=torch_dtype, device=device)
        v_sumsq = torch.full((M,), 8.0, dtype=torch_dtype, device=device)
        f_sumsq = torch.full((M,), 7.0, dtype=torch_dtype, device=device)
        max_norm = torch.full((M,), 6.0, dtype=torch_dtype, device=device)

        alpha_before = alpha.clone()
        dt_before = dt.clone()
        nsteps_before = nsteps_inc.clone()

        fire2_step_coord(
            positions,
            velocities,
            forces,
            batch_idx,
            alpha,
            dt,
            nsteps_inc,
            vf=vf,
            v_sumsq=v_sumsq,
            f_sumsq=f_sumsq,
            max_norm=max_norm,
            **FIRE2_DEFAULTS,
        )

        torch.testing.assert_close(alpha, alpha_before, atol=0, rtol=0)
        torch.testing.assert_close(dt, dt_before, atol=0, rtol=0)
        torch.testing.assert_close(nsteps_inc, nsteps_before, atol=0, rtol=0)
        torch.testing.assert_close(vf, torch.zeros_like(vf), atol=0, rtol=0)
        torch.testing.assert_close(v_sumsq, torch.zeros_like(v_sumsq), atol=0, rtol=0)
        torch.testing.assert_close(f_sumsq, torch.zeros_like(f_sumsq), atol=0, rtol=0)
        torch.testing.assert_close(max_norm, torch.zeros_like(max_norm), atol=0, rtol=0)

        vf.fill_(9.0)
        v_sumsq.fill_(8.0)
        f_sumsq.fill_(7.0)
        max_norm.fill_(6.0)

        fire2_step_extended(
            positions,
            velocities,
            forces,
            batch_idx,
            alpha,
            dt,
            nsteps_inc,
            vf=vf,
            v_sumsq=v_sumsq,
            f_sumsq=f_sumsq,
            max_norm=max_norm,
            **FIRE2_DEFAULTS,
        )

        torch.testing.assert_close(alpha, alpha_before, atol=0, rtol=0)
        torch.testing.assert_close(dt, dt_before, atol=0, rtol=0)
        torch.testing.assert_close(nsteps_inc, nsteps_before, atol=0, rtol=0)
        torch.testing.assert_close(vf, torch.zeros_like(vf), atol=0, rtol=0)
        torch.testing.assert_close(v_sumsq, torch.zeros_like(v_sumsq), atol=0, rtol=0)
        torch.testing.assert_close(f_sumsq, torch.zeros_like(f_sumsq), atol=0, rtol=0)
        torch.testing.assert_close(max_norm, torch.zeros_like(max_norm), atol=0, rtol=0)


# ==============================================================================
# Tests: Error Handling
# ==============================================================================


class TestFire2StepErrors:
    """Error handling tests for fire2_step input validation."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_missing_batch_idx_error(self, device):
        """fire2_step raises ValueError when batch_idx is None."""
        N, M = 20, 1
        dtype_vec, dtype_scalar, np_dtype = wp.vec3f, wp.float32, np.float32
        (
            pos,
            vel,
            forces,
            _bidx,
            alpha,
            dt,
            nsteps_inc,
            *_,
        ) = make_fire2_state(N, M, dtype_vec, dtype_scalar, np_dtype, device)

        vf, v_sumsq, f_sumsq, max_norm = make_fire2_scratch(M, dtype_scalar, device)

        with pytest.raises(ValueError, match="batch_idx is required"):
            fire2_step(
                pos,
                vel,
                forces,
                None,
                alpha,
                dt,
                nsteps_inc,
                vf,
                v_sumsq,
                f_sumsq,
                max_norm,
                **FIRE2_DEFAULTS,
            )

    @pytest.mark.parametrize("device", DEVICES)
    def test_positions_velocities_shape_mismatch_error(self, device):
        """fire2_step raises ValueError when positions/velocities shapes differ."""
        N, M = 20, 1
        dtype_vec, dtype_scalar, np_dtype = wp.vec3f, wp.float32, np.float32
        (
            pos,
            vel,
            forces,
            bidx,
            alpha,
            dt,
            nsteps_inc,
            *_,
        ) = make_fire2_state(N, M, dtype_vec, dtype_scalar, np_dtype, device)

        # Create mismatched velocities
        vel_wrong = wp.zeros(N + 5, dtype=dtype_vec, device=device)

        vf, v_sumsq, f_sumsq, max_norm = make_fire2_scratch(M, dtype_scalar, device)

        with pytest.raises(ValueError, match="velocities length"):
            fire2_step(
                pos,
                vel_wrong,
                forces,
                bidx,
                alpha,
                dt,
                nsteps_inc,
                vf,
                v_sumsq,
                f_sumsq,
                max_norm,
                **FIRE2_DEFAULTS,
            )

    @pytest.mark.parametrize("device", DEVICES)
    def test_forces_shape_mismatch_error(self, device):
        """fire2_step raises ValueError when forces shape differs from positions."""
        N, M = 20, 1
        dtype_vec, dtype_scalar, np_dtype = wp.vec3f, wp.float32, np.float32
        (
            pos,
            vel,
            forces,
            bidx,
            alpha,
            dt,
            nsteps_inc,
            *_,
        ) = make_fire2_state(N, M, dtype_vec, dtype_scalar, np_dtype, device)

        forces_wrong = wp.zeros(N + 3, dtype=dtype_vec, device=device)

        vf, v_sumsq, f_sumsq, max_norm = make_fire2_scratch(M, dtype_scalar, device)

        with pytest.raises(ValueError, match="forces length"):
            fire2_step(
                pos,
                vel,
                forces_wrong,
                bidx,
                alpha,
                dt,
                nsteps_inc,
                vf,
                v_sumsq,
                f_sumsq,
                max_norm,
                **FIRE2_DEFAULTS,
            )

    @pytest.mark.parametrize("device", DEVICES)
    def test_per_system_shape_mismatch_error(self, device):
        """fire2_step raises ValueError when per-system arrays have inconsistent shapes."""
        N, M = 20, 1
        dtype_vec, dtype_scalar, np_dtype = wp.vec3f, wp.float32, np.float32
        (
            pos,
            vel,
            forces,
            bidx,
            alpha,
            dt,
            nsteps_inc,
            *_,
        ) = make_fire2_state(N, M, dtype_vec, dtype_scalar, np_dtype, device)

        # dt has wrong shape
        dt_wrong = wp.zeros(M + 2, dtype=dtype_scalar, device=device)

        vf, v_sumsq, f_sumsq, max_norm = make_fire2_scratch(M, dtype_scalar, device)

        with pytest.raises(ValueError, match="dt length"):
            fire2_step(
                pos,
                vel,
                forces,
                bidx,
                alpha,
                dt_wrong,
                nsteps_inc,
                vf,
                v_sumsq,
                f_sumsq,
                max_norm,
                **FIRE2_DEFAULTS,
            )

    @pytest.mark.parametrize("device", DEVICES)
    def test_missing_scratch_buffers_error(self, device):
        """fire2_step raises TypeError when scratch buffers are omitted."""
        N, M = 20, 1
        dtype_vec, dtype_scalar, np_dtype = wp.vec3f, wp.float32, np.float32
        (
            pos,
            vel,
            forces,
            bidx,
            alpha,
            dt,
            nsteps_inc,
            *_,
        ) = make_fire2_state(N, M, dtype_vec, dtype_scalar, np_dtype, device)

        with pytest.raises(TypeError):
            fire2_step(
                pos,
                vel,
                forces,
                bidx,
                alpha,
                dt,
                nsteps_inc,
                **FIRE2_DEFAULTS,
            )


# ==============================================================================
# Tests: Device Inference
# ==============================================================================


class TestFire2DeviceInference:
    """Verify fire2_step correctly infers device from positions."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_device_inference(self, device, dtype_vec, dtype_scalar, np_dtype):
        """fire2_step should infer device from positions when not specified."""
        N, M = 30, 1
        (
            pos,
            vel,
            forces,
            bidx,
            alpha,
            dt,
            nsteps_inc,
            *_,
        ) = make_fire2_state(N, M, dtype_vec, dtype_scalar, np_dtype, device)

        vf, v_sumsq, f_sumsq, max_norm = make_fire2_scratch(M, dtype_scalar, device)

        pos_before = pos.numpy().copy()

        # No explicit device -- should infer from positions
        fire2_step(
            pos,
            vel,
            forces,
            bidx,
            alpha,
            dt,
            nsteps_inc,
            vf,
            v_sumsq,
            f_sumsq,
            max_norm,
            **FIRE2_DEFAULTS,
        )
        wp.synchronize()

        # Positions should be modified
        assert not np.allclose(pos.numpy(), pos_before), (
            "Positions should change after fire2_step"
        )

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_device_string_accepted(self, device, dtype_vec, dtype_scalar, np_dtype):
        """fire2_step accepts device as a string (e.g. 'cuda:0') without AttributeError."""
        N, M = 20, 1
        (
            pos,
            vel,
            forces,
            bidx,
            alpha,
            dt,
            nsteps_inc,
            *_,
        ) = make_fire2_state(N, M, dtype_vec, dtype_scalar, np_dtype, device)

        vf, v_sumsq, f_sumsq, max_norm = make_fire2_scratch(M, dtype_scalar, device)
        pos_before = pos.numpy().copy()

        # Explicit device as string (docstring says str is supported)
        fire2_step(
            pos,
            vel,
            forces,
            bidx,
            alpha,
            dt,
            nsteps_inc,
            vf,
            v_sumsq,
            f_sumsq,
            max_norm,
            **FIRE2_DEFAULTS,
        )
        wp.synchronize()

        assert not np.allclose(pos.numpy(), pos_before), (
            "Positions should change after fire2_step with device=str"
        )


# ==============================================================================
# Tests: State Modification
# ==============================================================================


class TestFire2StateModification:
    """Verify which arrays are modified in-place by fire2_step."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_positions_modified(self, device, dtype_vec, dtype_scalar, np_dtype):
        """fire2_step modifies positions in-place."""
        N, M = 40, 1
        (
            pos,
            vel,
            forces,
            bidx,
            alpha,
            dt,
            nsteps_inc,
            *_,
        ) = make_fire2_state(N, M, dtype_vec, dtype_scalar, np_dtype, device)

        pos_before = pos.numpy().copy()

        vf, v_sumsq, f_sumsq, max_norm = make_fire2_scratch(M, dtype_scalar, device)
        fire2_step(
            pos,
            vel,
            forces,
            bidx,
            alpha,
            dt,
            nsteps_inc,
            vf,
            v_sumsq,
            f_sumsq,
            max_norm,
            **FIRE2_DEFAULTS,
        )
        wp.synchronize()

        assert not np.allclose(pos.numpy(), pos_before), (
            "Positions should be modified in-place"
        )

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_velocities_modified(self, device, dtype_vec, dtype_scalar, np_dtype):
        """fire2_step modifies velocities in-place."""
        N, M = 40, 1
        (
            pos,
            vel,
            forces,
            bidx,
            alpha,
            dt,
            nsteps_inc,
            *_,
        ) = make_fire2_state(N, M, dtype_vec, dtype_scalar, np_dtype, device)

        vel_before = vel.numpy().copy()

        vf, v_sumsq, f_sumsq, max_norm = make_fire2_scratch(M, dtype_scalar, device)
        fire2_step(
            pos,
            vel,
            forces,
            bidx,
            alpha,
            dt,
            nsteps_inc,
            vf,
            v_sumsq,
            f_sumsq,
            max_norm,
            **FIRE2_DEFAULTS,
        )
        wp.synchronize()

        assert not np.allclose(vel.numpy(), vel_before), (
            "Velocities should be modified in-place"
        )

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_forces_not_modified(self, device, dtype_vec, dtype_scalar, np_dtype):
        """fire2_step does not modify forces."""
        N, M = 40, 1
        (
            pos,
            vel,
            forces,
            bidx,
            alpha,
            dt,
            nsteps_inc,
            *_,
        ) = make_fire2_state(N, M, dtype_vec, dtype_scalar, np_dtype, device)

        forces_before = forces.numpy().copy()

        vf, v_sumsq, f_sumsq, max_norm = make_fire2_scratch(M, dtype_scalar, device)
        fire2_step(
            pos,
            vel,
            forces,
            bidx,
            alpha,
            dt,
            nsteps_inc,
            vf,
            v_sumsq,
            f_sumsq,
            max_norm,
            **FIRE2_DEFAULTS,
        )
        wp.synchronize()

        np.testing.assert_array_equal(
            forces.numpy(),
            forces_before,
            err_msg="Forces should not be modified by fire2_step",
        )


# ==============================================================================
# Tests: Variable System Sizes
# ==============================================================================


class TestFire2VariableSystemSizes:
    """Tests for fire2_step with variable atom counts per system."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_variable_atom_counts(self, device, dtype_vec, dtype_scalar, np_dtype):
        """fire2_step correctness with different atom counts per system."""
        atom_counts = [10, 30, 5, 15]  # 4 systems, different sizes
        (
            pos,
            vel,
            forces,
            bidx,
            alpha,
            dt,
            nsteps_inc,
            pos_np,
            vel_np,
            forces_np,
            bidx_np,
            alpha_np,
            dt_np,
            nsteps_np,
        ) = make_fire2_variable_state(
            atom_counts, dtype_vec, dtype_scalar, np_dtype, device
        )

        M = len(atom_counts)
        vf, v_sumsq, f_sumsq, max_norm = make_fire2_scratch(M, dtype_scalar, device)

        fire2_step(
            pos,
            vel,
            forces,
            bidx,
            alpha,
            dt,
            nsteps_inc,
            vf,
            v_sumsq,
            f_sumsq,
            max_norm,
            **FIRE2_DEFAULTS,
        )
        wp.synchronize()

        _fire2_reference_step(
            pos_np,
            vel_np,
            forces_np,
            bidx_np,
            alpha_np,
            dt_np,
            nsteps_np,
            **FIRE2_DEFAULTS,
        )

        rtol = 1e-4 if np_dtype == np.float32 else 1e-10
        np.testing.assert_allclose(pos.numpy(), pos_np, rtol=rtol, atol=1e-7)
        np.testing.assert_allclose(vel.numpy(), vel_np, rtol=rtol, atol=1e-7)
        np.testing.assert_allclose(alpha.numpy(), alpha_np, rtol=rtol)
        np.testing.assert_allclose(dt.numpy(), dt_np, rtol=rtol)

    @pytest.mark.parametrize("device", DEVICES)
    def test_variable_sizes_multiple_steps(self, device):
        """Multiple FIRE2 steps with variable atom counts remain correct."""
        atom_counts = [8, 20, 12]
        dtype_vec, dtype_scalar, np_dtype = wp.vec3d, wp.float64, np.float64
        (
            pos,
            vel,
            forces,
            bidx,
            alpha,
            dt,
            nsteps_inc,
            pos_np,
            vel_np,
            forces_np,
            bidx_np,
            alpha_np,
            dt_np,
            nsteps_np,
        ) = make_fire2_variable_state(
            atom_counts, dtype_vec, dtype_scalar, np_dtype, device
        )

        M = len(atom_counts)
        for _ in range(5):
            vf, v_sumsq, f_sumsq, max_norm = make_fire2_scratch(M, dtype_scalar, device)
            fire2_step(
                pos,
                vel,
                forces,
                bidx,
                alpha,
                dt,
                nsteps_inc,
                vf,
                v_sumsq,
                f_sumsq,
                max_norm,
                **FIRE2_DEFAULTS,
            )
            _fire2_reference_step(
                pos_np,
                vel_np,
                forces_np,
                bidx_np,
                alpha_np,
                dt_np,
                nsteps_np,
                **FIRE2_DEFAULTS,
            )

        wp.synchronize()
        np.testing.assert_allclose(pos.numpy(), pos_np, rtol=1e-8, atol=1e-7)
        np.testing.assert_allclose(vel.numpy(), vel_np, rtol=1e-8, atol=1e-7)


# ==============================================================================
# Tests: Algorithmic Behavior
# ==============================================================================


class TestFire2AlgorithmicBehavior:
    """Test FIRE2-specific algorithmic behaviors."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_uphill_resets_state(self, device):
        """When P <= 0 (uphill), dt shrinks, alpha resets, velocities zeroed."""
        N, M = 10, 1
        np_dtype = np.float64
        dtype_vec, dtype_scalar = wp.vec3d, wp.float64

        # Velocities opposite to forces -> P < 0 after half-step
        pos_np = np.zeros((N, 3), dtype=np_dtype)
        vel_np = np.tile([-10.0, 0.0, 0.0], (N, 1)).astype(np_dtype)
        forces_np = np.tile([1.0, 0.0, 0.0], (N, 1)).astype(np_dtype)
        bidx_np = np.zeros(N, dtype=np.int32)
        alpha_np = np.full(1, 0.05, dtype=np_dtype)  # non-default alpha
        dt_np = np.full(1, 0.04, dtype=np_dtype)
        nsteps_np = np.array([10], dtype=np.int32)  # some positive count

        pos_ref = pos_np.copy()
        vel_ref = vel_np.copy()
        forces_ref = forces_np.copy()
        bidx_ref = bidx_np.copy()
        alpha_ref = alpha_np.copy()
        dt_ref = dt_np.copy()
        nsteps_ref = nsteps_np.copy()

        pos = wp.array(pos_np, dtype=dtype_vec, device=device)
        vel = wp.array(vel_np, dtype=dtype_vec, device=device)
        forces = wp.array(forces_np, dtype=dtype_vec, device=device)
        bidx = wp.array(bidx_np, dtype=wp.int32, device=device)
        alpha = wp.array(alpha_np, dtype=dtype_scalar, device=device)
        dt_arr = wp.array(dt_np, dtype=dtype_scalar, device=device)
        nsteps_inc = wp.array(nsteps_np, dtype=wp.int32, device=device)

        vf, v_sumsq, f_sumsq, max_norm = make_fire2_scratch(M, dtype_scalar, device)
        fire2_step(
            pos,
            vel,
            forces,
            bidx,
            alpha,
            dt_arr,
            nsteps_inc,
            vf,
            v_sumsq,
            f_sumsq,
            max_norm,
            **FIRE2_DEFAULTS,
        )
        wp.synchronize()

        _fire2_reference_step(
            pos_ref,
            vel_ref,
            forces_ref,
            bidx_ref,
            alpha_ref,
            dt_ref,
            nsteps_ref,
            **FIRE2_DEFAULTS,
        )

        # Behavioral assertions
        assert nsteps_inc.numpy()[0] == 0, "nsteps_inc should reset to 0"
        assert alpha.numpy()[0] == pytest.approx(FIRE2_DEFAULTS["alpha0"]), (
            "alpha should reset to alpha0"
        )
        assert dt_arr.numpy()[0] < 0.04, "dt should be shrunk"
        # Velocities should be zeroed (uphill correction)
        np.testing.assert_allclose(
            vel.numpy(),
            0.0,
            atol=1e-12,
            err_msg="Velocities should be zeroed for uphill step",
        )
        # Exact match with reference
        np.testing.assert_allclose(pos.numpy(), pos_ref, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(dt_arr.numpy(), dt_ref, rtol=1e-10)

    @pytest.mark.parametrize("device", DEVICES)
    def test_acceleration_after_delaystep(self, device):
        """After delaystep downhill steps, dt grows and alpha shrinks."""
        N, M = 10, 1
        np_dtype = np.float64
        dtype_vec, dtype_scalar = wp.vec3d, wp.float64
        delaystep = FIRE2_DEFAULTS["delaystep"]

        # Velocities aligned with forces -> P > 0 after half-step
        pos_np = np.zeros((N, 3), dtype=np_dtype)
        vel_np = np.tile([0.5, 0.0, 0.0], (N, 1)).astype(np_dtype)
        forces_np = np.tile([1.0, 0.0, 0.0], (N, 1)).astype(np_dtype)
        bidx_np = np.zeros(N, dtype=np.int32)
        alpha_np = np.full(1, 0.09, dtype=np_dtype)
        dt_np = np.full(1, 0.04, dtype=np_dtype)
        # nsteps_inc at delaystep: next step will be delaystep+1 > delaystep
        nsteps_np = np.array([delaystep], dtype=np.int32)

        alpha_before = alpha_np[0]

        pos = wp.array(pos_np, dtype=dtype_vec, device=device)
        vel = wp.array(vel_np, dtype=dtype_vec, device=device)
        forces = wp.array(forces_np, dtype=dtype_vec, device=device)
        bidx = wp.array(bidx_np, dtype=wp.int32, device=device)
        alpha = wp.array(alpha_np, dtype=dtype_scalar, device=device)
        dt_arr = wp.array(dt_np, dtype=dtype_scalar, device=device)
        nsteps_inc = wp.array(nsteps_np, dtype=wp.int32, device=device)

        pos_ref = pos_np.copy()
        vel_ref = vel_np.copy()
        forces_ref = forces_np.copy()
        bidx_ref = bidx_np.copy()
        alpha_ref = alpha_np.copy()
        dt_ref = dt_np.copy()
        nsteps_ref = nsteps_np.copy()

        vf, v_sumsq, f_sumsq, max_norm = make_fire2_scratch(M, dtype_scalar, device)
        fire2_step(
            pos,
            vel,
            forces,
            bidx,
            alpha,
            dt_arr,
            nsteps_inc,
            vf,
            v_sumsq,
            f_sumsq,
            max_norm,
            **FIRE2_DEFAULTS,
        )
        wp.synchronize()

        _fire2_reference_step(
            pos_ref,
            vel_ref,
            forces_ref,
            bidx_ref,
            alpha_ref,
            dt_ref,
            nsteps_ref,
            **FIRE2_DEFAULTS,
        )

        # nsteps_inc should now be delaystep + 1
        assert nsteps_inc.numpy()[0] == delaystep + 1, (
            f"nsteps_inc should be {delaystep + 1}"
        )
        # alpha should shrink
        assert alpha.numpy()[0] < alpha_before, (
            f"alpha should shrink: got {alpha.numpy()[0]} >= {alpha_before}"
        )
        # Exact match with reference
        np.testing.assert_allclose(pos.numpy(), pos_ref, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(
            alpha.numpy(),
            alpha_ref,
            rtol=1e-10,
            atol=1e-12,
        )
        np.testing.assert_allclose(dt_arr.numpy(), dt_ref, rtol=1e-10)

    @pytest.mark.parametrize("device", DEVICES)
    def test_dt_clamped_to_tmax(self, device):
        """dt does not exceed tmax even when growth is triggered."""
        N, M = 10, 1
        np_dtype = np.float64
        dtype_vec, dtype_scalar = wp.vec3d, wp.float64
        tmax = FIRE2_DEFAULTS["tmax"]
        delaystep = FIRE2_DEFAULTS["delaystep"]

        # Small velocities aligned with small forces (P > 0), dt close to tmax
        pos_np = np.zeros((N, 3), dtype=np_dtype)
        vel_np = np.tile([0.01, 0.0, 0.0], (N, 1)).astype(np_dtype)
        forces_np = np.tile([0.01, 0.0, 0.0], (N, 1)).astype(np_dtype)
        bidx_np = np.zeros(N, dtype=np.int32)
        alpha_np = np.full(1, 0.09, dtype=np_dtype)
        dt_np = np.full(1, tmax - 0.001, dtype=np_dtype)  # just below tmax
        nsteps_np = np.array([delaystep + 5], dtype=np.int32)  # well past delay

        pos = wp.array(pos_np, dtype=dtype_vec, device=device)
        vel = wp.array(vel_np, dtype=dtype_vec, device=device)
        forces = wp.array(forces_np, dtype=dtype_vec, device=device)
        bidx = wp.array(bidx_np, dtype=wp.int32, device=device)
        alpha = wp.array(alpha_np, dtype=dtype_scalar, device=device)
        dt_arr = wp.array(dt_np, dtype=dtype_scalar, device=device)
        nsteps_inc = wp.array(nsteps_np, dtype=wp.int32, device=device)

        vf, v_sumsq, f_sumsq, max_norm = make_fire2_scratch(M, dtype_scalar, device)
        fire2_step(
            pos,
            vel,
            forces,
            bidx,
            alpha,
            dt_arr,
            nsteps_inc,
            vf,
            v_sumsq,
            f_sumsq,
            max_norm,
            **FIRE2_DEFAULTS,
        )
        wp.synchronize()

        # dt should not exceed tmax (maxstep clamping may reduce it further)
        assert dt_arr.numpy()[0] <= tmax + 1e-12, (
            f"dt should not exceed tmax={tmax}; got {dt_arr.numpy()[0]}"
        )

    @pytest.mark.parametrize("device", DEVICES)
    def test_dt_clamped_to_tmin(self, device):
        """dt param-update shrink is floored at tmin."""
        N, M = 10, 1
        np_dtype = np.float64
        dtype_vec, dtype_scalar = wp.vec3d, wp.float64
        tmin = FIRE2_DEFAULTS["tmin"]
        dtshrink = FIRE2_DEFAULTS["dtshrink"]

        # Small velocities opposite to small forces (P < 0), tiny steps so
        # maxstep clamping won't further reduce dt
        pos_np = np.zeros((N, 3), dtype=np_dtype)
        vel_np = np.tile([-0.01, 0.0, 0.0], (N, 1)).astype(np_dtype)
        forces_np = np.tile([0.001, 0.0, 0.0], (N, 1)).astype(np_dtype)
        bidx_np = np.zeros(N, dtype=np.int32)
        alpha_np = np.full(1, 0.09, dtype=np_dtype)
        # dt such that dtshrink * dt < tmin
        dt_val = tmin / dtshrink * 0.9  # 0.006 for defaults
        dt_np = np.full(1, dt_val, dtype=np_dtype)
        nsteps_np = np.zeros(1, dtype=np.int32)

        pos = wp.array(pos_np, dtype=dtype_vec, device=device)
        vel = wp.array(vel_np, dtype=dtype_vec, device=device)
        forces = wp.array(forces_np, dtype=dtype_vec, device=device)
        bidx = wp.array(bidx_np, dtype=wp.int32, device=device)
        alpha = wp.array(alpha_np, dtype=dtype_scalar, device=device)
        dt_arr = wp.array(dt_np, dtype=dtype_scalar, device=device)
        nsteps_inc = wp.array(nsteps_np, dtype=wp.int32, device=device)

        vf, v_sumsq, f_sumsq, max_norm = make_fire2_scratch(M, dtype_scalar, device)
        fire2_step(
            pos,
            vel,
            forces,
            bidx,
            alpha,
            dt_arr,
            nsteps_inc,
            vf,
            v_sumsq,
            f_sumsq,
            max_norm,
            **FIRE2_DEFAULTS,
        )
        wp.synchronize()

        # With small velocities, maxstep clamping shouldn't kick in,
        # so dt should be floored at tmin
        assert dt_arr.numpy()[0] >= tmin - 1e-12, (
            f"dt should not go below tmin={tmin}; got {dt_arr.numpy()[0]}"
        )

    @pytest.mark.parametrize("device", DEVICES)
    def test_maxstep_clamping(self, device):
        """Max displacement per system does not exceed maxstep."""
        N, M = 20, 1
        np_dtype = np.float64
        dtype_vec, dtype_scalar = wp.vec3d, wp.float64
        maxstep = FIRE2_DEFAULTS["maxstep"]

        rng = np.random.default_rng(55)
        # Large velocities aligned with forces -> big steps that need clamping
        pos_np = np.zeros((N, 3), dtype=np_dtype)
        vel_np = rng.standard_normal((N, 3)).astype(np_dtype) * 5.0
        forces_np = vel_np.copy()  # aligned with velocities for P > 0
        bidx_np = np.zeros(N, dtype=np.int32)
        alpha_np = np.full(1, 0.09, dtype=np_dtype)
        dt_np = np.full(1, 0.05, dtype=np_dtype)
        nsteps_np = np.zeros(1, dtype=np.int32)

        pos_before = pos_np.copy()

        pos = wp.array(pos_np, dtype=dtype_vec, device=device)
        vel = wp.array(vel_np, dtype=dtype_vec, device=device)
        forces = wp.array(forces_np, dtype=dtype_vec, device=device)
        bidx = wp.array(bidx_np, dtype=wp.int32, device=device)
        alpha = wp.array(alpha_np, dtype=dtype_scalar, device=device)
        dt_arr = wp.array(dt_np, dtype=dtype_scalar, device=device)
        nsteps_inc = wp.array(nsteps_np, dtype=wp.int32, device=device)

        vf, v_sumsq, f_sumsq, max_norm = make_fire2_scratch(M, dtype_scalar, device)
        fire2_step(
            pos,
            vel,
            forces,
            bidx,
            alpha,
            dt_arr,
            nsteps_inc,
            vf,
            v_sumsq,
            f_sumsq,
            max_norm,
            **FIRE2_DEFAULTS,
        )
        wp.synchronize()

        final_pos = pos.numpy()
        displacements = np.linalg.norm(final_pos - pos_before, axis=1)
        max_disp = np.max(displacements)
        assert max_disp <= maxstep + 1e-7, (
            f"Max displacement {max_disp} exceeds maxstep={maxstep}"
        )


# ==============================================================================
# Tests: Edge Cases
# ==============================================================================


class TestFire2EdgeCases:
    """Edge case tests for fire2_step."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_single_atom(self, device):
        """fire2_step works correctly with a single atom."""
        N, M = 1, 1
        np_dtype = np.float64
        dtype_vec, dtype_scalar = wp.vec3d, wp.float64
        rng = np.random.default_rng(200)
        (
            pos,
            vel,
            forces,
            bidx,
            alpha,
            dt,
            nsteps_inc,
            pos_np,
            vel_np,
            forces_np,
            bidx_np,
            alpha_np,
            dt_np,
            nsteps_np,
        ) = make_fire2_state(N, M, dtype_vec, dtype_scalar, np_dtype, device, rng=rng)

        vf, v_sumsq, f_sumsq, max_norm = make_fire2_scratch(M, dtype_scalar, device)
        fire2_step(
            pos,
            vel,
            forces,
            bidx,
            alpha,
            dt,
            nsteps_inc,
            vf,
            v_sumsq,
            f_sumsq,
            max_norm,
            **FIRE2_DEFAULTS,
        )
        wp.synchronize()

        _fire2_reference_step(
            pos_np,
            vel_np,
            forces_np,
            bidx_np,
            alpha_np,
            dt_np,
            nsteps_np,
            **FIRE2_DEFAULTS,
        )

        np.testing.assert_allclose(pos.numpy(), pos_np, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(vel.numpy(), vel_np, rtol=1e-10, atol=1e-12)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_zero_forces(self, device, dtype_vec, dtype_scalar, np_dtype):
        """fire2_step with exactly zero forces produces finite outputs.

        Exercises the f_sumsq=0 guard in the velocity mixing kernel.
        With zero forces and zero velocities, positions must not change and
        all outputs must remain finite (no NaN from 0/0).
        """
        N, M = 20, 1

        pos_np = np.ones((N, 3), dtype=np_dtype)
        vel_np = np.zeros((N, 3), dtype=np_dtype)
        forces_np = np.zeros((N, 3), dtype=np_dtype)
        bidx_np = np.zeros(N, dtype=np.int32)
        alpha_np = np.full(1, 0.09, dtype=np_dtype)
        dt_np = np.full(1, 0.05, dtype=np_dtype)
        nsteps_np = np.zeros(1, dtype=np.int32)

        pos_before = pos_np.copy()

        pos = wp.array(pos_np, dtype=dtype_vec, device=device)
        vel = wp.array(vel_np, dtype=dtype_vec, device=device)
        forces = wp.array(forces_np, dtype=dtype_vec, device=device)
        bidx = wp.array(bidx_np, dtype=wp.int32, device=device)
        alpha = wp.array(alpha_np, dtype=dtype_scalar, device=device)
        dt_arr = wp.array(dt_np, dtype=dtype_scalar, device=device)
        nsteps_inc = wp.array(nsteps_np, dtype=wp.int32, device=device)

        vf, v_sumsq, f_sumsq, max_norm = make_fire2_scratch(M, dtype_scalar, device)
        fire2_step(
            pos,
            vel,
            forces,
            bidx,
            alpha,
            dt_arr,
            nsteps_inc,
            vf,
            v_sumsq,
            f_sumsq,
            max_norm,
            **FIRE2_DEFAULTS,
        )
        wp.synchronize()

        final_pos = pos.numpy()
        assert np.isfinite(final_pos).all(), "Outputs must be finite with zero forces"
        assert np.isfinite(vel.numpy()).all(), (
            "Velocities must be finite with zero forces"
        )
        np.testing.assert_allclose(
            final_pos,
            pos_before,
            atol=1e-12,
            err_msg="Positions should not change with zero forces and velocities",
        )

    @pytest.mark.parametrize("device", DEVICES)
    def test_zero_forces_nonzero_velocities(self, device):
        """fire2_step with zero forces but non-zero velocities stays finite.

        When all forces are zero but velocities are non-zero: P=v.f=0 (uphill),
        the f_sumsq=0 guard ensures ratio=0, and kernel 3 zeros velocities
        for the uphill system. All outputs must remain finite.
        """
        N, M = 10, 1
        np_dtype = np.float64
        dtype_vec, dtype_scalar = wp.vec3d, wp.float64

        pos_np = np.ones((N, 3), dtype=np_dtype)
        vel_np = np.tile([1.0, 0.5, -0.3], (N, 1)).astype(np_dtype)
        forces_np = np.zeros((N, 3), dtype=np_dtype)
        bidx_np = np.zeros(N, dtype=np.int32)
        alpha_np = np.full(1, 0.09, dtype=np_dtype)
        dt_np = np.full(1, 0.05, dtype=np_dtype)
        nsteps_np = np.zeros(1, dtype=np.int32)

        pos = wp.array(pos_np, dtype=dtype_vec, device=device)
        vel = wp.array(vel_np, dtype=dtype_vec, device=device)
        forces = wp.array(forces_np, dtype=dtype_vec, device=device)
        bidx = wp.array(bidx_np, dtype=wp.int32, device=device)
        alpha = wp.array(alpha_np, dtype=dtype_scalar, device=device)
        dt_arr = wp.array(dt_np, dtype=dtype_scalar, device=device)
        nsteps_inc = wp.array(nsteps_np, dtype=wp.int32, device=device)

        pos_ref = pos_np.copy()
        vel_ref = vel_np.copy()
        forces_ref = forces_np.copy()
        bidx_ref = bidx_np.copy()
        alpha_ref = alpha_np.copy()
        dt_ref = dt_np.copy()
        nsteps_ref = nsteps_np.copy()

        vf, v_sumsq, f_sumsq, max_norm = make_fire2_scratch(M, dtype_scalar, device)
        fire2_step(
            pos,
            vel,
            forces,
            bidx,
            alpha,
            dt_arr,
            nsteps_inc,
            vf,
            v_sumsq,
            f_sumsq,
            max_norm,
            **FIRE2_DEFAULTS,
        )
        wp.synchronize()

        _fire2_reference_step(
            pos_ref,
            vel_ref,
            forces_ref,
            bidx_ref,
            alpha_ref,
            dt_ref,
            nsteps_ref,
            **FIRE2_DEFAULTS,
        )

        assert np.isfinite(pos.numpy()).all(), (
            "Positions must be finite with zero forces"
        )
        assert np.isfinite(vel.numpy()).all(), (
            "Velocities must be finite with zero forces"
        )
        # Velocities should be zeroed (uphill correction: P = v.f = 0 <= 0)
        np.testing.assert_allclose(
            vel.numpy(),
            0.0,
            atol=1e-12,
            err_msg="Velocities should be zeroed (uphill: P=0)",
        )
        np.testing.assert_allclose(
            pos.numpy(),
            pos_ref,
            rtol=1e-10,
            atol=1e-12,
        )

    @pytest.mark.parametrize("device", DEVICES)
    def test_zero_initial_velocities(self, device):
        """Cold start: fire2_step moves atoms along force direction from v=0."""
        N, M = 10, 1
        np_dtype = np.float64
        dtype_vec, dtype_scalar = wp.vec3d, wp.float64

        pos_np = np.ones((N, 3), dtype=np_dtype)
        vel_np = np.zeros((N, 3), dtype=np_dtype)
        forces_np = np.tile([1.0, 0.0, 0.0], (N, 1)).astype(np_dtype)
        bidx_np = np.zeros(N, dtype=np.int32)
        alpha_np = np.full(1, 0.09, dtype=np_dtype)
        dt_np = np.full(1, 0.05, dtype=np_dtype)
        nsteps_np = np.zeros(1, dtype=np.int32)

        pos_before = pos_np.copy()

        pos = wp.array(pos_np, dtype=dtype_vec, device=device)
        vel = wp.array(vel_np, dtype=dtype_vec, device=device)
        forces = wp.array(forces_np, dtype=dtype_vec, device=device)
        bidx = wp.array(bidx_np, dtype=wp.int32, device=device)
        alpha = wp.array(alpha_np, dtype=dtype_scalar, device=device)
        dt_arr = wp.array(dt_np, dtype=dtype_scalar, device=device)
        nsteps_inc = wp.array(nsteps_np, dtype=wp.int32, device=device)

        vf, v_sumsq, f_sumsq, max_norm = make_fire2_scratch(M, dtype_scalar, device)
        fire2_step(
            pos,
            vel,
            forces,
            bidx,
            alpha,
            dt_arr,
            nsteps_inc,
            vf,
            v_sumsq,
            f_sumsq,
            max_norm,
            **FIRE2_DEFAULTS,
        )
        wp.synchronize()

        final_pos = pos.numpy()
        # Positions should move in the positive x direction (force direction)
        assert np.all(final_pos[:, 0] > pos_before[:, 0]), (
            "Atoms should move along force direction from cold start"
        )
        # Y and Z should not change
        np.testing.assert_allclose(
            final_pos[:, 1:],
            pos_before[:, 1:],
            atol=1e-12,
            err_msg="Off-axis positions should not change for axis-aligned forces",
        )

    @pytest.mark.parametrize("device", DEVICES)
    def test_large_number_of_systems(self, device):
        """fire2_step correctness with many batched systems (M=32)."""
        M = 32
        atoms_per_sys = 10
        N = M * atoms_per_sys
        np_dtype = np.float64
        dtype_vec, dtype_scalar = wp.vec3d, wp.float64

        rng = np.random.default_rng(300)
        (
            pos,
            vel,
            forces,
            bidx,
            alpha,
            dt,
            nsteps_inc,
            pos_np,
            vel_np,
            forces_np,
            bidx_np,
            alpha_np,
            dt_np,
            nsteps_np,
        ) = make_fire2_state(N, M, dtype_vec, dtype_scalar, np_dtype, device, rng=rng)

        vf, v_sumsq, f_sumsq, max_norm = make_fire2_scratch(M, dtype_scalar, device)
        fire2_step(
            pos,
            vel,
            forces,
            bidx,
            alpha,
            dt,
            nsteps_inc,
            vf,
            v_sumsq,
            f_sumsq,
            max_norm,
            **FIRE2_DEFAULTS,
        )
        wp.synchronize()

        _fire2_reference_step(
            pos_np,
            vel_np,
            forces_np,
            bidx_np,
            alpha_np,
            dt_np,
            nsteps_np,
            **FIRE2_DEFAULTS,
        )

        np.testing.assert_allclose(pos.numpy(), pos_np, rtol=1e-10, atol=1e-7)
        np.testing.assert_allclose(vel.numpy(), vel_np, rtol=1e-10, atol=1e-7)
        assert np.isfinite(pos.numpy()).all()
        assert np.isfinite(vel.numpy()).all()


# ==============================================================================
# Tests: PyTorch Adapter Errors
# ==============================================================================


class TestFire2TorchCoordErrors:
    """Error handling tests for the PyTorch adapter fire2_step_coord."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_none_batch_idx(self, device):
        """fire2_step_coord raises when batch_idx is None."""
        N, M = 20, 1
        torch_dtype = torch.float32
        (
            pos,
            vel,
            forces,
            _bidx,
            alpha,
            dt,
            nsteps_inc,
            *_,
        ) = make_fire2_torch_state(N, M, torch_dtype, device)

        with pytest.raises((AttributeError, TypeError)):
            fire2_step_coord(
                pos,
                vel,
                forces,
                None,
                alpha,
                dt,
                nsteps_inc,
                **FIRE2_DEFAULTS,
            )

    @pytest.mark.parametrize("device", DEVICES)
    def test_unsupported_dtype(self, device):
        """fire2_step_coord raises on unsupported tensor dtype."""
        N, M = 20, 1
        torch_dtype = torch.float32
        (
            pos,
            vel,
            forces,
            batch_idx,
            alpha,
            dt,
            nsteps_inc,
            *_,
        ) = make_fire2_torch_state(N, M, torch_dtype, device)

        # float16 is not supported by the Warp kernel
        pos_f16 = pos.half()
        with pytest.raises(KeyError):
            fire2_step_coord(
                pos_f16,
                vel,
                forces,
                batch_idx,
                alpha,
                dt,
                nsteps_inc,
                **FIRE2_DEFAULTS,
            )


# ==============================================================================
# Tests: PyTorch Adapter – Variable-Cell (fire2_step_coord_cell)
# ==============================================================================


def _make_upper_triangular_cell(M, np_dtype, *, rng=None, scale=5.0):
    """Create random upper-triangular cell matrices (M, 3, 3)."""
    if rng is None:
        rng = np.random.default_rng(99)
    cells = np.zeros((M, 3, 3), dtype=np_dtype)
    for i in range(M):
        # a, b*cos(gamma), b*sin(gamma), c1, c2, c3
        a = scale + rng.random() * 0.5
        cells[i, 0, 0] = a
        cells[i, 1, 0] = rng.random() * 0.1
        cells[i, 1, 1] = scale + rng.random() * 0.5
        cells[i, 2, 0] = rng.random() * 0.1
        cells[i, 2, 1] = rng.random() * 0.1
        cells[i, 2, 2] = scale + rng.random() * 0.5
    return cells


class TestFire2TorchCoordCell:
    """Tests for the PyTorch adapter fire2_step_coord_cell."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("torch_dtype", [torch.float32, torch.float64])
    def test_cell_step_basic(self, device, torch_dtype):
        """Positions, cell, and velocities should change; all outputs finite."""
        rng = np.random.default_rng(50)
        N, M = 60, 2
        np_dtype = np.float32 if torch_dtype == torch.float32 else np.float64

        (pos, vel, forces, batch_idx, alpha, dt, nsteps_inc, *_) = (
            make_fire2_torch_state(N, M, torch_dtype, device, rng=rng)
        )

        cell_np = _make_upper_triangular_cell(M, np_dtype, rng=rng)
        cell = torch.tensor(cell_np, dtype=torch_dtype, device=device)
        cell_vel = torch.zeros(M, 3, 3, dtype=torch_dtype, device=device)
        cell_force_np = rng.standard_normal((M, 3, 3)).astype(np_dtype) * 0.01
        cell_force = torch.tensor(cell_force_np, dtype=torch_dtype, device=device)

        pos_before = pos.clone()
        cell_before = cell.clone()

        fire2_step_coord_cell(
            pos,
            vel,
            forces,
            cell,
            cell_vel,
            cell_force,
            batch_idx,
            alpha,
            dt,
            nsteps_inc,
            **FIRE2_CELL_DEFAULTS,
        )
        torch.cuda.synchronize()

        assert not torch.allclose(pos, pos_before), "Positions should be updated"
        assert not torch.allclose(cell, cell_before), "Cell should be updated"
        assert torch.isfinite(pos).all(), "Positions should be finite"
        assert torch.isfinite(vel).all(), "Velocities should be finite"
        assert torch.isfinite(cell).all(), "Cell should be finite"
        assert torch.isfinite(cell_vel).all(), "Cell velocities should be finite"
        assert torch.isfinite(alpha).all(), "Alpha should be finite"
        assert torch.isfinite(dt).all(), "dt should be finite"

    @pytest.mark.parametrize("device", DEVICES)
    def test_cell_force_scale_is_extra_atom_count_multiplier(self, device):
        """cell_force_scale is an extra multiplier after atom normalization."""
        rng = np.random.default_rng(522)
        N, M = 24, 2
        torch_dtype = torch.float64
        np_dtype = np.float64

        (pos_a, vel_a, forces, batch_idx, alpha_a, dt_a, nsteps_a, *_) = (
            make_fire2_torch_state(N, M, torch_dtype, device, rng=rng)
        )
        cell_np = _make_upper_triangular_cell(M, np_dtype, rng=rng)
        cell_a = torch.tensor(cell_np, dtype=torch_dtype, device=device)
        cell_vel_a = torch.zeros(M, 3, 3, dtype=torch_dtype, device=device)
        cell_force_np = rng.standard_normal((M, 3, 3)).astype(np_dtype) * 0.2
        cell_force = torch.tensor(cell_force_np, dtype=torch_dtype, device=device)

        pos_b = pos_a.clone()
        vel_b = vel_a.clone()
        cell_b = cell_a.clone()
        cell_vel_b = cell_vel_a.clone()
        alpha_b = alpha_a.clone()
        dt_b = dt_a.clone()
        nsteps_b = nsteps_a.clone()

        fire2_step_coord_cell(
            pos_a,
            vel_a,
            forces,
            cell_a,
            cell_vel_a,
            cell_force,
            batch_idx,
            alpha_a,
            dt_a,
            nsteps_a,
            **FIRE2_DEFAULTS,
        )
        fire2_step_coord_cell(
            pos_b,
            vel_b,
            forces,
            cell_b,
            cell_vel_b,
            cell_force * 2.0,
            batch_idx,
            alpha_b,
            dt_b,
            nsteps_b,
            cell_force_scale=2.0,
            **FIRE2_DEFAULTS,
        )
        torch.cuda.synchronize()

        torch.testing.assert_close(pos_a, pos_b, atol=0, rtol=0)
        torch.testing.assert_close(vel_a, vel_b, atol=0, rtol=0)
        torch.testing.assert_close(cell_a, cell_b, atol=0, rtol=0)
        torch.testing.assert_close(cell_vel_a, cell_vel_b, atol=0, rtol=0)
        torch.testing.assert_close(alpha_a, alpha_b, atol=0, rtol=0)
        torch.testing.assert_close(dt_a, dt_b, atol=0, rtol=0)
        torch.testing.assert_close(nsteps_a, nsteps_b, atol=0, rtol=0)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("cell_force_scale", [0.0, -1.0])
    def test_cell_force_scale_rejects_non_positive(self, device, cell_force_scale):
        """cell_force_scale must stay positive."""
        rng = np.random.default_rng(523)
        N, M = 6, 1
        torch_dtype = torch.float64
        np_dtype = np.float64

        (pos, vel, forces, batch_idx, alpha, dt, nsteps_inc, *_) = (
            make_fire2_torch_state(N, M, torch_dtype, device, rng=rng)
        )
        cell_np = _make_upper_triangular_cell(M, np_dtype, rng=rng)
        cell = torch.tensor(cell_np, dtype=torch_dtype, device=device)
        cell_vel = torch.zeros(M, 3, 3, dtype=torch_dtype, device=device)
        cell_force = torch.zeros(M, 3, 3, dtype=torch_dtype, device=device)

        with pytest.raises(ValueError, match="cell_force_scale must be positive"):
            fire2_step_coord_cell(
                pos,
                vel,
                forces,
                cell,
                cell_vel,
                cell_force,
                batch_idx,
                alpha,
                dt,
                nsteps_inc,
                cell_force_scale=cell_force_scale,
                **FIRE2_DEFAULTS,
            )

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("torch_dtype", [torch.float32, torch.float64])
    def test_cell_step_zero_atoms_returns(self, device, torch_dtype):
        """fire2_step_coord_cell returns early and zeroes provided scratch buffers."""
        M = 2
        np_dtype = np.float32 if torch_dtype == torch.float32 else np.float64
        positions = torch.empty(0, 3, dtype=torch_dtype, device=device)
        velocities = torch.empty(0, 3, dtype=torch_dtype, device=device)
        forces = torch.empty(0, 3, dtype=torch_dtype, device=device)
        batch_idx = torch.empty(0, dtype=torch.int32, device=device)
        cell = torch.tensor(
            _make_upper_triangular_cell(M, np_dtype),
            dtype=torch_dtype,
            device=device,
        )
        cell_vel = torch.zeros(M, 3, 3, dtype=torch_dtype, device=device)
        cell_force = torch.ones(M, 3, 3, dtype=torch_dtype, device=device)
        alpha = torch.tensor([0.07, 0.04], dtype=torch_dtype, device=device)
        dt = torch.tensor([0.05, 0.04], dtype=torch_dtype, device=device)
        nsteps_inc = torch.tensor([5, 8], dtype=torch.int32, device=device)
        vf = torch.full((M,), 9.0, dtype=torch_dtype, device=device)
        v_sumsq = torch.full((M,), 8.0, dtype=torch_dtype, device=device)
        f_sumsq = torch.full((M,), 7.0, dtype=torch_dtype, device=device)
        max_norm = torch.full((M,), 6.0, dtype=torch_dtype, device=device)

        cell_before = cell.clone()
        cell_vel_before = cell_vel.clone()
        alpha_before = alpha.clone()
        dt_before = dt.clone()
        nsteps_before = nsteps_inc.clone()

        fire2_step_coord_cell(
            positions,
            velocities,
            forces,
            cell,
            cell_vel,
            cell_force,
            batch_idx,
            alpha,
            dt,
            nsteps_inc,
            vf=vf,
            v_sumsq=v_sumsq,
            f_sumsq=f_sumsq,
            max_norm=max_norm,
            **FIRE2_CELL_DEFAULTS,
        )

        torch.testing.assert_close(cell, cell_before, atol=0, rtol=0)
        torch.testing.assert_close(cell_vel, cell_vel_before, atol=0, rtol=0)
        torch.testing.assert_close(alpha, alpha_before, atol=0, rtol=0)
        torch.testing.assert_close(dt, dt_before, atol=0, rtol=0)
        torch.testing.assert_close(nsteps_inc, nsteps_before, atol=0, rtol=0)
        torch.testing.assert_close(vf, torch.zeros_like(vf), atol=0, rtol=0)
        torch.testing.assert_close(v_sumsq, torch.zeros_like(v_sumsq), atol=0, rtol=0)
        torch.testing.assert_close(f_sumsq, torch.zeros_like(f_sumsq), atol=0, rtol=0)
        torch.testing.assert_close(max_norm, torch.zeros_like(max_norm), atol=0, rtol=0)

    @pytest.mark.parametrize("device", DEVICES)
    def test_cell_step_rejects_empty_system_in_batch(self, device):
        """Each system must have at least one atom before cell-force normalization."""
        rng = np.random.default_rng(524)
        N, M = 4, 2
        torch_dtype = torch.float64
        np_dtype = np.float64
        positions = torch.tensor(
            rng.standard_normal((N, 3)).astype(np_dtype),
            dtype=torch_dtype,
            device=device,
        )
        velocities = torch.zeros_like(positions)
        forces = torch.zeros_like(positions)
        batch_idx = torch.zeros(N, dtype=torch.int32, device=device)
        cell = torch.tensor(
            _make_upper_triangular_cell(M, np_dtype, rng=rng),
            dtype=torch_dtype,
            device=device,
        )
        cell_vel = torch.zeros(M, 3, 3, dtype=torch_dtype, device=device)
        cell_force = torch.zeros(M, 3, 3, dtype=torch_dtype, device=device)
        alpha = torch.full((M,), 0.09, dtype=torch_dtype, device=device)
        dt = torch.full((M,), 0.05, dtype=torch_dtype, device=device)
        nsteps_inc = torch.zeros(M, dtype=torch.int32, device=device)

        with pytest.raises(ValueError, match="at least one atom per system"):
            fire2_step_coord_cell(
                positions,
                velocities,
                forces,
                cell,
                cell_vel,
                cell_force,
                batch_idx,
                alpha,
                dt,
                nsteps_inc,
                **FIRE2_CELL_DEFAULTS,
            )

    @pytest.mark.parametrize("device", DEVICES)
    def test_cell_only_motion_preserves_fractional_coordinates(self, device):
        """Cell-only coupled motion preserves atom fractional coordinates."""
        rng = np.random.default_rng(525)
        N, M = 12, 2
        torch_dtype = torch.float64
        np_dtype = np.float64
        pos_np = rng.standard_normal((N, 3)).astype(np_dtype)
        batch_idx_np = np.repeat(np.arange(M, dtype=np.int32), N // M)
        cell_np = _make_upper_triangular_cell(M, np_dtype, rng=rng)
        cell_force_np = np.zeros((M, 3, 3), dtype=np_dtype)
        cell_force_np[:, 0, 0] = np.array([0.5, -0.3], dtype=np_dtype)
        cell_force_np[:, 1, 1] = np.array([0.2, 0.4], dtype=np_dtype)
        cell_force_np[:, 2, 2] = np.array([-0.1, 0.3], dtype=np_dtype)

        positions = torch.tensor(pos_np.copy(), dtype=torch_dtype, device=device)
        velocities = torch.zeros(N, 3, dtype=torch_dtype, device=device)
        forces = torch.zeros(N, 3, dtype=torch_dtype, device=device)
        batch_idx = torch.tensor(batch_idx_np, dtype=torch.int32, device=device)
        cell = torch.tensor(cell_np.copy(), dtype=torch_dtype, device=device)
        cell_before = cell.clone()
        cell_vel = torch.zeros(M, 3, 3, dtype=torch_dtype, device=device)
        cell_force = torch.tensor(cell_force_np, dtype=torch_dtype, device=device)
        alpha = torch.full((M,), 0.09, dtype=torch_dtype, device=device)
        dt = torch.full((M,), 0.05, dtype=torch_dtype, device=device)
        nsteps_inc = torch.zeros(M, dtype=torch.int32, device=device)
        defaults = {**FIRE2_CELL_DEFAULTS, "maxstep": 10.0}

        frac_before = _fractional_coordinates_np(pos_np, cell_np, batch_idx_np)

        fire2_step_coord_cell(
            positions,
            velocities,
            forces,
            cell,
            cell_vel,
            cell_force,
            batch_idx,
            alpha,
            dt,
            nsteps_inc,
            **defaults,
        )
        torch.cuda.synchronize()

        frac_after = _fractional_coordinates_np(
            positions.cpu().numpy(), cell.cpu().numpy(), batch_idx_np
        )
        np.testing.assert_allclose(frac_after, frac_before, atol=1e-10, rtol=1e-10)
        assert not torch.allclose(cell, cell_before)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("torch_dtype", [torch.float32, torch.float64])
    def test_cell_step_matches_coupled_reference(self, device, torch_dtype):
        """fire2_step_coord_cell matches the single-system coupled reference."""
        rng = np.random.default_rng(51)
        N, M = 6, 1
        np_dtype = np.float32 if torch_dtype == torch.float32 else np.float64
        pos_np = rng.standard_normal((N, 3)).astype(np_dtype)
        vel_np = rng.standard_normal((N, 3)).astype(np_dtype) * 0.2
        forces_np = rng.standard_normal((N, 3)).astype(np_dtype)
        bidx_np = np.zeros(N, dtype=np.int32)
        alpha_np = np.array([0.07], dtype=np_dtype)
        dt_np = np.array([0.05], dtype=np_dtype)
        nsteps_np = np.array([5], dtype=np.int32)
        cell_np = _make_upper_triangular_cell(M, np_dtype, rng=rng)[0]
        cell_vel_np = rng.standard_normal((3, 3)).astype(np_dtype) * 0.05
        cell_vel_np[np.triu_indices(3, k=1)] = 0.0
        cell_force_np = rng.standard_normal((3, 3)).astype(np_dtype) * 0.3
        cell_force_np[np.triu_indices(3, k=1)] = 0.0
        defaults = {**FIRE2_CELL_DEFAULTS, "maxstep": 0.035}

        pos = torch.tensor(pos_np.copy(), dtype=torch_dtype, device=device)
        vel = torch.tensor(vel_np.copy(), dtype=torch_dtype, device=device)
        forces = torch.tensor(forces_np.copy(), dtype=torch_dtype, device=device)
        batch_idx = torch.tensor(bidx_np.copy(), dtype=torch.int32, device=device)
        alpha = torch.tensor(alpha_np.copy(), dtype=torch_dtype, device=device)
        dt = torch.tensor(dt_np.copy(), dtype=torch_dtype, device=device)
        nsteps_inc = torch.tensor(nsteps_np.copy(), dtype=torch.int32, device=device)
        cell = torch.tensor(cell_np[None, ...], dtype=torch_dtype, device=device)
        cell_vel = torch.tensor(
            cell_vel_np[None, ...], dtype=torch_dtype, device=device
        )
        cell_force = torch.tensor(
            cell_force_np[None, ...], dtype=torch_dtype, device=device
        )

        fire2_step_coord_cell(
            pos,
            vel,
            forces,
            cell,
            cell_vel,
            cell_force,
            batch_idx,
            alpha,
            dt,
            nsteps_inc,
            **defaults,
        )
        torch.cuda.synchronize()

        (
            pos_ref,
            vel_ref,
            cell_ref,
            cell_vel_ref,
            alpha_ref,
            dt_ref,
            nsteps_ref,
        ) = _fire2_coord_cell_reference_step(
            pos_np,
            vel_np,
            forces_np,
            cell_np,
            cell_vel_np,
            cell_force_np,
            alpha_np,
            dt_np,
            nsteps_np,
            **defaults,
        )

        rtol = 1e-4 if np_dtype == np.float32 else 1e-10
        np.testing.assert_allclose(pos.cpu().numpy(), pos_ref, rtol=rtol, atol=1e-7)
        np.testing.assert_allclose(vel.cpu().numpy(), vel_ref, rtol=rtol, atol=1e-7)
        np.testing.assert_allclose(
            cell.cpu().numpy()[0], cell_ref, rtol=rtol, atol=1e-7
        )
        np.testing.assert_allclose(
            cell_vel.cpu().numpy()[0], cell_vel_ref, rtol=rtol, atol=1e-7
        )
        np.testing.assert_allclose(alpha.cpu().numpy(), alpha_ref, rtol=rtol, atol=1e-7)
        np.testing.assert_allclose(dt.cpu().numpy(), dt_ref, rtol=rtol, atol=1e-7)
        np.testing.assert_array_equal(nsteps_inc.cpu().numpy(), nsteps_ref)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("torch_dtype", [torch.float32, torch.float64])
    def test_cell_step_matches_batched_coupled_reference(self, device, torch_dtype):
        """Batched variable-cell FIRE2 matches per-system coupled references."""
        rng = np.random.default_rng(512)
        atom_counts = [4, 7, 3]
        M = len(atom_counts)
        N = sum(atom_counts)
        np_dtype = np.float32 if torch_dtype == torch.float32 else np.float64

        pos_np = rng.standard_normal((N, 3)).astype(np_dtype)
        vel_np = rng.standard_normal((N, 3)).astype(np_dtype) * 0.15
        forces_np = rng.standard_normal((N, 3)).astype(np_dtype) * 0.2
        bidx_np = np.concatenate(
            [np.full(n, i, dtype=np.int32) for i, n in enumerate(atom_counts)]
        )
        cell_np = _make_upper_triangular_cell(M, np_dtype, rng=rng)
        cell_vel_np = rng.standard_normal((M, 3, 3)).astype(np_dtype) * 0.04
        cell_vel_np[:, np.triu_indices(3, k=1)[0], np.triu_indices(3, k=1)[1]] = 0.0
        cell_force_np = rng.standard_normal((M, 3, 3)).astype(np_dtype) * 0.25
        cell_force_np[:, np.triu_indices(3, k=1)[0], np.triu_indices(3, k=1)[1]] = 0.0

        # Force one system downhill and one uphill so both branches are checked.
        sys0 = bidx_np == 0
        sys1 = bidx_np == 1
        forces_np[sys0] = vel_np[sys0]
        forces_np[sys1] = -vel_np[sys1]
        cell_force_np[0] = cell_vel_np[0]
        cell_force_np[1] = -cell_vel_np[1]

        alpha_np = np.array([0.07, 0.04, 0.11], dtype=np_dtype)
        dt_np = np.array([0.05, 0.04, 0.03], dtype=np_dtype)
        nsteps_np = np.array([61, 8, 59], dtype=np.int32)
        defaults = {**FIRE2_CELL_DEFAULTS, "maxstep": 0.045}

        pos = torch.tensor(pos_np.copy(), dtype=torch_dtype, device=device)
        vel = torch.tensor(vel_np.copy(), dtype=torch_dtype, device=device)
        forces = torch.tensor(forces_np.copy(), dtype=torch_dtype, device=device)
        batch_idx = torch.tensor(bidx_np.copy(), dtype=torch.int32, device=device)
        cell = torch.tensor(cell_np.copy(), dtype=torch_dtype, device=device)
        cell_vel = torch.tensor(cell_vel_np.copy(), dtype=torch_dtype, device=device)
        cell_force = torch.tensor(
            cell_force_np.copy(), dtype=torch_dtype, device=device
        )
        alpha = torch.tensor(alpha_np.copy(), dtype=torch_dtype, device=device)
        dt = torch.tensor(dt_np.copy(), dtype=torch_dtype, device=device)
        nsteps_inc = torch.tensor(nsteps_np.copy(), dtype=torch.int32, device=device)

        fire2_step_coord_cell(
            pos,
            vel,
            forces,
            cell,
            cell_vel,
            cell_force,
            batch_idx,
            alpha,
            dt,
            nsteps_inc,
            **defaults,
        )
        torch.cuda.synchronize()

        pos_ref = pos_np.copy()
        vel_ref = vel_np.copy()
        cell_ref = cell_np.copy()
        cell_vel_ref = cell_vel_np.copy()
        alpha_ref = alpha_np.copy()
        dt_ref = dt_np.copy()
        nsteps_ref = nsteps_np.copy()

        for sys, count in enumerate(atom_counts):
            start = sum(atom_counts[:sys])
            end = start + count
            (
                pos_ref[start:end],
                vel_ref[start:end],
                cell_ref[sys],
                cell_vel_ref[sys],
                alpha_ref[sys : sys + 1],
                dt_ref[sys : sys + 1],
                nsteps_ref[sys : sys + 1],
            ) = _fire2_coord_cell_reference_step(
                pos_np[start:end],
                vel_np[start:end],
                forces_np[start:end],
                cell_np[sys],
                cell_vel_np[sys],
                cell_force_np[sys],
                alpha_np[sys : sys + 1],
                dt_np[sys : sys + 1],
                nsteps_np[sys : sys + 1],
                **defaults,
            )

        rtol = 1e-4 if np_dtype == np.float32 else 1e-10
        np.testing.assert_allclose(pos.cpu().numpy(), pos_ref, rtol=rtol, atol=1e-7)
        np.testing.assert_allclose(vel.cpu().numpy(), vel_ref, rtol=rtol, atol=1e-7)
        np.testing.assert_allclose(cell.cpu().numpy(), cell_ref, rtol=rtol, atol=1e-7)
        np.testing.assert_allclose(
            cell_vel.cpu().numpy(), cell_vel_ref, rtol=rtol, atol=1e-7
        )
        np.testing.assert_allclose(alpha.cpu().numpy(), alpha_ref, rtol=rtol, atol=1e-7)
        np.testing.assert_allclose(dt.cpu().numpy(), dt_ref, rtol=rtol, atol=1e-7)
        np.testing.assert_array_equal(nsteps_inc.cpu().numpy(), nsteps_ref)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("torch_dtype", [torch.float32, torch.float64])
    def test_cell_step_uphill_zeroes_atomic_and_cell_velocities(
        self, device, torch_dtype
    ):
        """Uphill variable-cell FIRE2 zeroes both atomic and cell velocities."""
        rng = np.random.default_rng(511)
        N, M = 10, 1
        np_dtype = np.float32 if torch_dtype == torch.float32 else np.float64

        pos_np = rng.standard_normal((N, 3)).astype(np_dtype)
        vel_np = rng.standard_normal((N, 3)).astype(np_dtype) * 0.2
        forces_np = -vel_np.copy()
        batch_idx_np = np.zeros(N, dtype=np.int32)
        cell_np = _make_upper_triangular_cell(M, np_dtype, rng=rng)
        cell_vel_np = rng.standard_normal((M, 3, 3)).astype(np_dtype) * 0.05
        cell_vel_np[:, np.triu_indices(3, k=1)[0], np.triu_indices(3, k=1)[1]] = 0.0
        cell_force_np = -cell_vel_np.copy()
        alpha_np = np.array([0.04], dtype=np_dtype)
        dt_np = np.array([0.05], dtype=np_dtype)
        nsteps_np = np.array([7], dtype=np.int32)

        pos = torch.tensor(pos_np, dtype=torch_dtype, device=device)
        vel = torch.tensor(vel_np, dtype=torch_dtype, device=device)
        forces = torch.tensor(forces_np, dtype=torch_dtype, device=device)
        batch_idx = torch.tensor(batch_idx_np, dtype=torch.int32, device=device)
        cell = torch.tensor(cell_np, dtype=torch_dtype, device=device)
        cell_vel = torch.tensor(cell_vel_np, dtype=torch_dtype, device=device)
        cell_force = torch.tensor(cell_force_np, dtype=torch_dtype, device=device)
        alpha = torch.tensor(alpha_np, dtype=torch_dtype, device=device)
        dt = torch.tensor(dt_np, dtype=torch_dtype, device=device)
        nsteps_inc = torch.tensor(nsteps_np, dtype=torch.int32, device=device)

        fire2_step_coord_cell(
            pos,
            vel,
            forces,
            cell,
            cell_vel,
            cell_force,
            batch_idx,
            alpha,
            dt,
            nsteps_inc,
            **FIRE2_CELL_DEFAULTS,
        )
        torch.cuda.synchronize()

        torch.testing.assert_close(vel, torch.zeros_like(vel), atol=0, rtol=0)
        torch.testing.assert_close(cell_vel, torch.zeros_like(cell_vel), atol=0, rtol=0)
        assert alpha.item() == pytest.approx(FIRE2_DEFAULTS["alpha0"])
        assert dt.item() <= dt_np[0] * FIRE2_DEFAULTS["dtshrink"] + 1e-12
        assert nsteps_inc.item() == 0

    @pytest.mark.parametrize("device", DEVICES)
    def test_cell_step_with_scratch_buffers(self, device):
        """fire2_step_coord_cell works with pre-allocated scratch buffers."""
        rng = np.random.default_rng(52)
        N, M = 60, 2
        torch_dtype = torch.float64
        np_dtype = np.float64

        (pos, vel, forces, batch_idx, alpha, dt, nsteps_inc, *_) = (
            make_fire2_torch_state(N, M, torch_dtype, device, rng=rng)
        )
        cell_np = _make_upper_triangular_cell(M, np_dtype, rng=rng)
        cell = torch.tensor(cell_np, dtype=torch_dtype, device=device)
        cell_vel = torch.zeros(M, 3, 3, dtype=torch_dtype, device=device)
        cell_force_np = rng.standard_normal((M, 3, 3)).astype(np_dtype) * 0.01
        cell_force = torch.tensor(cell_force_np, dtype=torch_dtype, device=device)

        N_ext = N + 2 * M
        ext_vel = torch.empty(N_ext, 3, dtype=torch_dtype, device=device)
        ext_forces = torch.empty(N_ext, 3, dtype=torch_dtype, device=device)
        vf = torch.ones(M, dtype=torch_dtype, device=device) * 999.0
        v_sumsq = torch.ones(M, dtype=torch_dtype, device=device) * 999.0
        f_sumsq = torch.ones(M, dtype=torch_dtype, device=device) * 999.0
        max_norm_buf = torch.ones(M, dtype=torch_dtype, device=device) * 999.0

        pos_before = pos.clone()
        fire2_step_coord_cell(
            pos,
            vel,
            forces,
            cell,
            cell_vel,
            cell_force,
            batch_idx,
            alpha,
            dt,
            nsteps_inc,
            ext_velocities=ext_vel,
            ext_forces=ext_forces,
            vf=vf,
            v_sumsq=v_sumsq,
            f_sumsq=f_sumsq,
            max_norm=max_norm_buf,
            **FIRE2_CELL_DEFAULTS,
        )
        torch.cuda.synchronize()

        assert not torch.allclose(pos, pos_before), "Positions should be updated"
        assert torch.isfinite(pos).all()
        assert torch.isfinite(cell).all()

    @pytest.mark.parametrize("device", DEVICES)
    def test_cell_step_deprecated_ext_positions_warns(self, device):
        """The deprecated ext_positions keyword is accepted with a warning."""
        rng = np.random.default_rng(521)
        N, M = 20, 2
        torch_dtype = torch.float64
        np_dtype = np.float64

        (pos, vel, forces, batch_idx, alpha, dt, nsteps_inc, *_) = (
            make_fire2_torch_state(N, M, torch_dtype, device, rng=rng)
        )
        cell_np = _make_upper_triangular_cell(M, np_dtype, rng=rng)
        cell = torch.tensor(cell_np, dtype=torch_dtype, device=device)
        cell_vel = torch.zeros(M, 3, 3, dtype=torch_dtype, device=device)
        cell_force_np = rng.standard_normal((M, 3, 3)).astype(np_dtype) * 0.01
        cell_force = torch.tensor(cell_force_np, dtype=torch_dtype, device=device)
        ext_positions = torch.empty(N + 2 * M, 3, dtype=torch_dtype, device=device)

        with pytest.warns(DeprecationWarning, match="ext_positions"):
            fire2_step_coord_cell(
                pos,
                vel,
                forces,
                cell,
                cell_vel,
                cell_force,
                batch_idx,
                alpha,
                dt,
                nsteps_inc,
                ext_positions=ext_positions,
                **FIRE2_CELL_DEFAULTS,
            )

        torch.cuda.synchronize()
        assert torch.isfinite(pos).all()
        assert torch.isfinite(cell).all()

    @pytest.mark.parametrize("device", DEVICES)
    def test_cell_step_scratch_reuse(self, device):
        """Scratch buffers and static metadata can be reused across steps."""
        rng = np.random.default_rng(53)
        N, M = 40, 2
        torch_dtype = torch.float64
        np_dtype = np.float64

        (pos, vel, forces, batch_idx, alpha, dt, nsteps_inc, *_) = (
            make_fire2_torch_state(N, M, torch_dtype, device, rng=rng)
        )
        cell_np = _make_upper_triangular_cell(M, np_dtype, rng=rng)
        cell = torch.tensor(cell_np, dtype=torch_dtype, device=device)
        cell_vel = torch.zeros(M, 3, 3, dtype=torch_dtype, device=device)
        cell_force_np = rng.standard_normal((M, 3, 3)).astype(np_dtype) * 0.01
        cell_force = torch.tensor(cell_force_np, dtype=torch_dtype, device=device)

        N_ext = N + 2 * M
        ext_vel = torch.empty(N_ext, 3, dtype=torch_dtype, device=device)
        ext_forces = torch.empty(N_ext, 3, dtype=torch_dtype, device=device)
        vf = torch.empty(M, dtype=torch_dtype, device=device)
        v_sumsq = torch.empty(M, dtype=torch_dtype, device=device)
        f_sumsq = torch.empty(M, dtype=torch_dtype, device=device)
        max_norm_buf = torch.empty(M, dtype=torch_dtype, device=device)

        # First call: let the function compute atom_ptr, ext_atom_ptr,
        # ext_batch_idx internally (pass None).
        fire2_step_coord_cell(
            pos,
            vel,
            forces,
            cell,
            cell_vel,
            cell_force,
            batch_idx,
            alpha,
            dt,
            nsteps_inc,
            ext_velocities=ext_vel,
            ext_forces=ext_forces,
            vf=vf,
            v_sumsq=v_sumsq,
            f_sumsq=f_sumsq,
            max_norm=max_norm_buf,
            **FIRE2_CELL_DEFAULTS,
        )

        # Pre-compute static metadata for subsequent reuse.
        from nvalchemiops.batch_utils import (
            atom_ptr_to_batch_idx as _a2b,
        )
        from nvalchemiops.batch_utils import (
            batch_idx_to_atom_ptr as _b2a,
        )
        from nvalchemiops.dynamics.utils.cell_filter import (
            extend_atom_ptr as _eap,
        )

        atom_ptr_t = torch.zeros(M + 1, dtype=torch.int32, device=device)
        atom_counts_t = torch.zeros(M, dtype=torch.int32, device=device)
        _b2a(
            wp.from_torch(batch_idx, dtype=wp.int32),
            wp.from_torch(atom_counts_t, dtype=wp.int32),
            wp.from_torch(atom_ptr_t, dtype=wp.int32),
        )
        ext_atom_ptr_t = torch.zeros(M + 1, dtype=torch.int32, device=device)
        _eap(
            wp.from_torch(atom_ptr_t, dtype=wp.int32),
            wp.from_torch(ext_atom_ptr_t, dtype=wp.int32),
        )
        ext_bidx = torch.empty(N_ext, dtype=torch.int32, device=device)
        _a2b(
            wp.from_torch(ext_atom_ptr_t, dtype=wp.int32),
            wp.from_torch(ext_bidx, dtype=wp.int32),
        )

        # Remaining 4 calls reuse all pre-computed metadata.
        for _ in range(4):
            fire2_step_coord_cell(
                pos,
                vel,
                forces,
                cell,
                cell_vel,
                cell_force,
                batch_idx,
                alpha,
                dt,
                nsteps_inc,
                atom_ptr=atom_ptr_t,
                ext_atom_ptr=ext_atom_ptr_t,
                ext_velocities=ext_vel,
                ext_forces=ext_forces,
                ext_batch_idx=ext_bidx,
                vf=vf,
                v_sumsq=v_sumsq,
                f_sumsq=f_sumsq,
                max_norm=max_norm_buf,
                **FIRE2_CELL_DEFAULTS,
            )

        torch.cuda.synchronize()
        assert torch.isfinite(pos).all(), "Positions should stay finite over 5 steps"
        assert torch.isfinite(cell).all(), "Cell should stay finite over 5 steps"
        assert torch.isfinite(vel).all(), "Velocities should stay finite"
        assert torch.isfinite(cell_vel).all(), "Cell velocities should stay finite"

    @pytest.mark.parametrize("device", DEVICES)
    def test_cell_step_single_system(self, device):
        """fire2_step_coord_cell works for a single-system (M=1) case."""
        rng = np.random.default_rng(54)
        N, M = 30, 1
        torch_dtype = torch.float64
        np_dtype = np.float64

        (pos, vel, forces, batch_idx, alpha, dt, nsteps_inc, *_) = (
            make_fire2_torch_state(N, M, torch_dtype, device, rng=rng)
        )
        cell_np = _make_upper_triangular_cell(M, np_dtype, rng=rng)
        cell = torch.tensor(cell_np, dtype=torch_dtype, device=device)
        cell_vel = torch.zeros(M, 3, 3, dtype=torch_dtype, device=device)
        cell_force_np = rng.standard_normal((M, 3, 3)).astype(np_dtype) * 0.01
        cell_force = torch.tensor(cell_force_np, dtype=torch_dtype, device=device)

        pos_before = pos.clone()
        cell_before = cell.clone()

        fire2_step_coord_cell(
            pos,
            vel,
            forces,
            cell,
            cell_vel,
            cell_force,
            batch_idx,
            alpha,
            dt,
            nsteps_inc,
            **FIRE2_CELL_DEFAULTS,
        )
        torch.cuda.synchronize()

        assert not torch.allclose(pos, pos_before)
        assert not torch.allclose(cell, cell_before)
        assert torch.isfinite(pos).all()
        assert torch.isfinite(cell).all()

    @pytest.mark.parametrize("device", DEVICES)
    def test_cell_step_single_system_multi_step(self, device):
        """M==1 cell step stays stable over multiple steps."""
        rng = np.random.default_rng(55)
        N, M = 50, 1
        torch_dtype = torch.float64
        np_dtype = np.float64

        (pos, vel, forces, batch_idx, alpha, dt, nsteps_inc, *_) = (
            make_fire2_torch_state(N, M, torch_dtype, device, rng=rng)
        )
        cell_np = _make_upper_triangular_cell(M, np_dtype, rng=rng)
        cell = torch.tensor(cell_np, dtype=torch_dtype, device=device)
        cell_vel = torch.zeros(M, 3, 3, dtype=torch_dtype, device=device)
        cell_force_np = rng.standard_normal((M, 3, 3)).astype(np_dtype) * 0.01
        cell_force = torch.tensor(cell_force_np, dtype=torch_dtype, device=device)

        for _ in range(10):
            fire2_step_coord_cell(
                pos,
                vel,
                forces,
                cell,
                cell_vel,
                cell_force,
                batch_idx,
                alpha,
                dt,
                nsteps_inc,
                **FIRE2_CELL_DEFAULTS,
            )

        torch.cuda.synchronize()
        assert torch.isfinite(pos).all(), "Positions should stay finite over 10 steps"
        assert torch.isfinite(cell).all(), "Cell should stay finite over 10 steps"
        assert torch.isfinite(vel).all(), "Velocities should stay finite"
        assert torch.isfinite(cell_vel).all(), "Cell velocities should stay finite"

    @pytest.mark.parametrize("device", DEVICES)
    def test_cell_step_single_system_metadata_parity(self, device):
        """M==1 matches when metadata is supplied explicitly."""
        rng = np.random.default_rng(56)
        N, M = 40, 1
        torch_dtype = torch.float64
        np_dtype = np.float64

        (pos_a, vel_a, forces, batch_idx, alpha_a, dt_a, nsteps_a, *_) = (
            make_fire2_torch_state(N, M, torch_dtype, device, rng=rng)
        )
        cell_np = _make_upper_triangular_cell(M, np_dtype, rng=rng)
        cell_a = torch.tensor(cell_np, dtype=torch_dtype, device=device)
        cell_vel_a = torch.zeros(M, 3, 3, dtype=torch_dtype, device=device)
        cell_force_np = rng.standard_normal((M, 3, 3)).astype(np_dtype) * 0.01
        cell_force = torch.tensor(cell_force_np, dtype=torch_dtype, device=device)

        pos_b = pos_a.clone()
        vel_b = vel_a.clone()
        cell_b = cell_a.clone()
        cell_vel_b = cell_vel_a.clone()
        alpha_b = alpha_a.clone()
        dt_b = dt_a.clone()
        nsteps_b = nsteps_a.clone()

        fire2_step_coord_cell(
            pos_a,
            vel_a,
            forces,
            cell_a,
            cell_vel_a,
            cell_force,
            batch_idx,
            alpha_a,
            dt_a,
            nsteps_a,
            **FIRE2_CELL_DEFAULTS,
        )

        from nvalchemiops.batch_utils import (
            atom_ptr_to_batch_idx as _a2b,
        )
        from nvalchemiops.batch_utils import (
            batch_idx_to_atom_ptr as _b2a,
        )
        from nvalchemiops.dynamics.utils.cell_filter import (
            extend_atom_ptr as _eap,
        )

        atom_ptr = torch.zeros(M + 1, dtype=torch.int32, device=device)
        atom_counts = torch.zeros(M, dtype=torch.int32, device=device)
        _b2a(
            wp.from_torch(batch_idx, dtype=wp.int32),
            wp.from_torch(atom_counts, dtype=wp.int32),
            wp.from_torch(atom_ptr, dtype=wp.int32),
        )
        ext_atom_ptr = torch.zeros(M + 1, dtype=torch.int32, device=device)
        _eap(
            wp.from_torch(atom_ptr, dtype=wp.int32),
            wp.from_torch(ext_atom_ptr, dtype=wp.int32),
        )
        N_ext = N + 2 * M
        ext_bidx = torch.empty(N_ext, dtype=torch.int32, device=device)
        _a2b(
            wp.from_torch(ext_atom_ptr, dtype=wp.int32),
            wp.from_torch(ext_bidx, dtype=wp.int32),
        )
        ext_vel = torch.empty(N_ext, 3, dtype=torch_dtype, device=device)
        ext_forces = torch.empty(N_ext, 3, dtype=torch_dtype, device=device)

        fire2_step_coord_cell(
            pos_b,
            vel_b,
            forces,
            cell_b,
            cell_vel_b,
            cell_force,
            batch_idx,
            alpha_b,
            dt_b,
            nsteps_b,
            atom_ptr=atom_ptr,
            ext_atom_ptr=ext_atom_ptr,
            ext_velocities=ext_vel,
            ext_forces=ext_forces,
            ext_batch_idx=ext_bidx,
            **FIRE2_CELL_DEFAULTS,
        )

        torch.cuda.synchronize()

        torch.testing.assert_close(pos_a, pos_b, atol=0, rtol=0)
        torch.testing.assert_close(vel_a, vel_b, atol=0, rtol=0)
        torch.testing.assert_close(cell_a, cell_b, atol=0, rtol=0)
        torch.testing.assert_close(cell_vel_a, cell_vel_b, atol=0, rtol=0)
        torch.testing.assert_close(alpha_a, alpha_b, atol=0, rtol=0)
        torch.testing.assert_close(dt_a, dt_b, atol=0, rtol=0)

    @pytest.mark.parametrize("device", DEVICES)
    def test_cell_step_uneven_system_sizes(self, device):
        """fire2_step_coord_cell works with highly uneven system sizes."""
        rng = np.random.default_rng(57)
        # 3 systems with very different sizes: 10, 100, 500 atoms
        atoms_per_system = [10, 100, 500]
        M = len(atoms_per_system)
        N = sum(atoms_per_system)
        torch_dtype = torch.float64
        np_dtype = np.float64

        # Build batch_idx for uneven systems
        bidx_np = np.concatenate(
            [np.full(n, i, dtype=np.int32) for i, n in enumerate(atoms_per_system)]
        )
        batch_idx = torch.tensor(bidx_np, dtype=torch.int32, device=device)

        # Random state
        pos = torch.tensor(
            rng.standard_normal((N, 3)).astype(np_dtype),
            dtype=torch_dtype,
            device=device,
        )
        vel = torch.tensor(
            rng.standard_normal((N, 3)).astype(np_dtype) * 0.01,
            dtype=torch_dtype,
            device=device,
        )
        forces = torch.tensor(
            rng.standard_normal((N, 3)).astype(np_dtype),
            dtype=torch_dtype,
            device=device,
        )
        alpha = torch.full((M,), 0.09, dtype=torch_dtype, device=device)
        dt = torch.full((M,), 0.05, dtype=torch_dtype, device=device)
        nsteps_inc = torch.zeros(M, dtype=torch.int32, device=device)

        cell_np = _make_upper_triangular_cell(M, np_dtype, rng=rng)
        cell = torch.tensor(cell_np, dtype=torch_dtype, device=device)
        cell_vel = torch.zeros(M, 3, 3, dtype=torch_dtype, device=device)
        cell_force_np = rng.standard_normal((M, 3, 3)).astype(np_dtype) * 0.01
        cell_force = torch.tensor(cell_force_np, dtype=torch_dtype, device=device)

        pos_before = pos.clone()
        cell_before = cell.clone()

        # Run 5 steps
        for _ in range(5):
            fire2_step_coord_cell(
                pos,
                vel,
                forces,
                cell,
                cell_vel,
                cell_force,
                batch_idx,
                alpha,
                dt,
                nsteps_inc,
                **FIRE2_CELL_DEFAULTS,
            )

        torch.cuda.synchronize()

        assert not torch.allclose(pos, pos_before), "Positions should be updated"
        assert not torch.allclose(cell, cell_before), "Cell should be updated"
        assert torch.isfinite(pos).all(), "Positions should stay finite"
        assert torch.isfinite(vel).all(), "Velocities should stay finite"
        assert torch.isfinite(cell).all(), "Cell should stay finite"
        assert torch.isfinite(cell_vel).all(), "Cell velocities should stay finite"

    @pytest.mark.parametrize("device", DEVICES)
    def test_fire2_step_extended_differs_from_coupled_cell_update(self, device):
        """The generic extended path does not reproduce coupled cell kinematics."""
        rng = np.random.default_rng(58)
        N, M = 12, 2
        torch_dtype = torch.float64
        np_dtype = np.float64

        pos_np = rng.standard_normal((N, 3)).astype(np_dtype)
        vel_np = np.zeros((N, 3), dtype=np_dtype)
        forces_np = np.zeros((N, 3), dtype=np_dtype)
        batch_idx_np = np.repeat(np.arange(M, dtype=np.int32), N // M)
        cell_np = _make_upper_triangular_cell(M, np_dtype, rng=rng)
        cell_force_np = np.zeros((M, 3, 3), dtype=np_dtype)
        cell_force_np[:, 0, 0] = np.array([0.8, -0.6], dtype=np_dtype)
        cell_force_np[:, 1, 1] = np.array([0.4, 0.5], dtype=np_dtype)
        cell_force_np[:, 2, 2] = np.array([-0.3, 0.2], dtype=np_dtype)
        alpha_np = np.full(M, 0.09, dtype=np_dtype)
        dt_np = np.full(M, 0.05, dtype=np_dtype)
        nsteps_np = np.zeros(M, dtype=np.int32)

        pos_a = torch.tensor(pos_np.copy(), dtype=torch_dtype, device=device)
        vel_a = torch.tensor(vel_np.copy(), dtype=torch_dtype, device=device)
        forces_a = torch.tensor(forces_np.copy(), dtype=torch_dtype, device=device)
        batch_idx = torch.tensor(batch_idx_np.copy(), dtype=torch.int32, device=device)
        cell_a = torch.tensor(cell_np.copy(), dtype=torch_dtype, device=device)
        cell_vel_a = torch.zeros(M, 3, 3, dtype=torch_dtype, device=device)
        cell_force_a = torch.tensor(
            cell_force_np.copy(), dtype=torch_dtype, device=device
        )
        alpha_a = torch.tensor(alpha_np.copy(), dtype=torch_dtype, device=device)
        dt_a = torch.tensor(dt_np.copy(), dtype=torch_dtype, device=device)
        nsteps_a = torch.tensor(nsteps_np.copy(), dtype=torch.int32, device=device)

        fire2_step_coord_cell(
            pos_a,
            vel_a,
            forces_a,
            cell_a,
            cell_vel_a,
            cell_force_a,
            batch_idx,
            alpha_a,
            dt_a,
            nsteps_a,
            **FIRE2_CELL_DEFAULTS,
        )

        from nvalchemiops.batch_utils import (
            atom_ptr_to_batch_idx as _a2b,
        )
        from nvalchemiops.batch_utils import (
            batch_idx_to_atom_ptr as _b2a,
        )
        from nvalchemiops.dynamics.utils.cell_filter import (
            extend_atom_ptr as _eap,
        )
        from nvalchemiops.dynamics.utils.cell_filter import (
            pack_forces_with_cell,
            pack_positions_with_cell,
            pack_velocities_with_cell,
            unpack_positions_with_cell,
            unpack_velocities_with_cell,
        )

        vec_type = wp.vec3d
        mat_type = wp.mat33d
        wp_device = wp.device_from_torch(device)

        pos_b = torch.tensor(pos_np.copy(), dtype=torch_dtype, device=device)
        vel_b = torch.tensor(vel_np.copy(), dtype=torch_dtype, device=device)
        forces_b = torch.tensor(forces_np.copy(), dtype=torch_dtype, device=device)
        batch_idx_b = torch.tensor(
            batch_idx_np.copy(), dtype=torch.int32, device=device
        )
        cell_b = torch.tensor(cell_np.copy(), dtype=torch_dtype, device=device)
        cell_vel_b = torch.zeros(M, 3, 3, dtype=torch_dtype, device=device)
        cell_force_b = torch.tensor(
            cell_force_np.copy(), dtype=torch_dtype, device=device
        )
        alpha_b = torch.tensor(alpha_np.copy(), dtype=torch_dtype, device=device)
        dt_b = torch.tensor(dt_np.copy(), dtype=torch_dtype, device=device)
        nsteps_b = torch.tensor(nsteps_np.copy(), dtype=torch.int32, device=device)

        atom_ptr = torch.zeros(M + 1, dtype=torch.int32, device=device)
        atom_counts = torch.zeros(M, dtype=torch.int32, device=device)
        _b2a(
            wp.from_torch(batch_idx_b, dtype=wp.int32),
            wp.from_torch(atom_counts, dtype=wp.int32),
            wp.from_torch(atom_ptr, dtype=wp.int32),
        )
        ext_atom_ptr = torch.zeros(M + 1, dtype=torch.int32, device=device)
        _eap(
            wp.from_torch(atom_ptr, dtype=wp.int32),
            wp.from_torch(ext_atom_ptr, dtype=wp.int32),
        )
        N_ext = N + 2 * M
        ext_bidx = torch.empty(N_ext, dtype=torch.int32, device=device)
        _a2b(
            wp.from_torch(ext_atom_ptr, dtype=wp.int32),
            wp.from_torch(ext_bidx, dtype=wp.int32),
        )

        wp_atom_ptr = wp.from_torch(atom_ptr, dtype=wp.int32)
        wp_ext_atom_ptr = wp.from_torch(ext_atom_ptr, dtype=wp.int32)
        wp_bidx = wp.from_torch(batch_idx_b, dtype=wp.int32)

        ext_pos = torch.empty(N_ext, 3, dtype=torch_dtype, device=device)
        ext_vel = torch.empty(N_ext, 3, dtype=torch_dtype, device=device)
        ext_forces = torch.empty(N_ext, 3, dtype=torch_dtype, device=device)

        pack_positions_with_cell(
            wp.from_torch(pos_b, dtype=vec_type),
            wp.from_torch(cell_b, dtype=mat_type),
            wp.from_torch(ext_pos, dtype=vec_type),
            wp_atom_ptr,
            wp_ext_atom_ptr,
            device=wp_device,
            batch_idx=wp_bidx,
        )
        pack_velocities_with_cell(
            wp.from_torch(vel_b, dtype=vec_type),
            wp.from_torch(cell_vel_b, dtype=mat_type),
            wp.from_torch(ext_vel, dtype=vec_type),
            wp_atom_ptr,
            wp_ext_atom_ptr,
            device=wp_device,
            batch_idx=wp_bidx,
        )
        atom_counts_for_scale = torch.tensor(
            np.bincount(batch_idx_np, minlength=M),
            dtype=torch_dtype,
            device=device,
        ).reshape(M, 1, 1)
        normalized_cell_force_b = (cell_force_b / atom_counts_for_scale).contiguous()
        pack_forces_with_cell(
            wp.from_torch(forces_b, dtype=vec_type),
            wp.from_torch(normalized_cell_force_b, dtype=mat_type),
            wp.from_torch(ext_forces, dtype=vec_type),
            wp_atom_ptr,
            wp_ext_atom_ptr,
            device=wp_device,
            batch_idx=wp_bidx,
        )

        fire2_step_extended(
            ext_pos,
            ext_vel,
            ext_forces,
            ext_bidx,
            alpha_b,
            dt_b,
            nsteps_b,
            **FIRE2_DEFAULTS,
        )

        unpack_positions_with_cell(
            wp.from_torch(ext_pos, dtype=vec_type),
            wp.from_torch(pos_b, dtype=vec_type),
            wp.from_torch(cell_b, dtype=mat_type),
            atom_ptr=wp_atom_ptr,
            ext_atom_ptr=wp_ext_atom_ptr,
            device=wp_device,
            batch_idx=wp_bidx,
        )
        unpack_velocities_with_cell(
            wp.from_torch(ext_vel, dtype=vec_type),
            wp.from_torch(vel_b, dtype=vec_type),
            wp.from_torch(cell_vel_b, dtype=mat_type),
            atom_ptr=wp_atom_ptr,
            ext_atom_ptr=wp_ext_atom_ptr,
            device=wp_device,
            batch_idx=wp_bidx,
        )

        torch.cuda.synchronize()

        frac_before = _fractional_coordinates_np(pos_np, cell_np, batch_idx_np)
        frac_after_coupled = _fractional_coordinates_np(
            pos_a.cpu().numpy(), cell_a.cpu().numpy(), batch_idx_np
        )
        frac_after_extended = _fractional_coordinates_np(
            pos_b.cpu().numpy(), cell_b.cpu().numpy(), batch_idx_np
        )

        np.testing.assert_allclose(
            frac_after_coupled, frac_before, atol=1e-10, rtol=1e-10
        )
        assert not np.allclose(pos_a.cpu().numpy(), pos_b.cpu().numpy())
        assert not np.allclose(frac_after_extended, frac_before)


# ==============================================================================
# compute_reductions flag (caller-supplied precomputed reductions)
# ==============================================================================


class TestFire2ComputeReductions:
    """compute_reductions=True must match =False fed the same reductions."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("torch_dtype", [torch.float32, torch.float64])
    def test_coord_compute_reductions_parity(self, device, torch_dtype):
        N, M = 40, 2

        def run(compute_reductions, seed_bufs=None):
            (pos, vel, forces, batch_idx, alpha, dt, nsteps_inc, *_) = (
                make_fire2_torch_state(
                    N, M, torch_dtype, device, rng=np.random.default_rng(7)
                )
            )
            vf = torch.zeros(M, dtype=torch_dtype, device=device)
            vsq = torch.zeros(M, dtype=torch_dtype, device=device)
            fsq = torch.zeros(M, dtype=torch_dtype, device=device)
            mn = torch.zeros(M, dtype=torch_dtype, device=device)
            if seed_bufs is not None:
                vf.copy_(seed_bufs[0])
                vsq.copy_(seed_bufs[1])
                fsq.copy_(seed_bufs[2])
            fire2_step_coord(
                pos,
                vel,
                forces,
                batch_idx,
                alpha,
                dt,
                nsteps_inc,
                vf=vf,
                v_sumsq=vsq,
                f_sumsq=fsq,
                max_norm=mn,
                compute_reductions=compute_reductions,
                **FIRE2_DEFAULTS,
            )
            return pos, vel, alpha, dt, nsteps_inc, (vf, vsq, fsq)

        rp, rv, ra, rd, rn, bufs = run(True)
        dp, dv, da, dd, dn, _ = run(False, seed_bufs=[b.clone() for b in bufs])
        torch.cuda.synchronize()

        assert torch.equal(dp, rp)
        assert torch.equal(dv, rv)
        assert torch.equal(da, ra)
        assert torch.equal(dd, rd)
        assert torch.equal(dn, rn)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("torch_dtype", [torch.float32, torch.float64])
    def test_coord_cell_compute_reductions_parity(self, device, torch_dtype):
        N, M = 40, 2
        np_dtype = np.float32 if torch_dtype == torch.float32 else np.float64

        def run(compute_reductions, seed_bufs=None):
            rng = np.random.default_rng(11)
            (pos, vel, forces, batch_idx, alpha, dt, nsteps_inc, *_) = (
                make_fire2_torch_state(N, M, torch_dtype, device, rng=rng)
            )
            cell = torch.tensor(
                _make_upper_triangular_cell(M, np_dtype, rng=rng),
                dtype=torch_dtype,
                device=device,
            )
            cell_vel = torch.zeros(M, 3, 3, dtype=torch_dtype, device=device)
            cell_force = torch.tensor(
                rng.standard_normal((M, 3, 3)).astype(np_dtype) * 0.01,
                dtype=torch_dtype,
                device=device,
            )
            vf = torch.zeros(M, dtype=torch_dtype, device=device)
            vsq = torch.zeros(M, dtype=torch_dtype, device=device)
            fsq = torch.zeros(M, dtype=torch_dtype, device=device)
            mn = torch.zeros(M, dtype=torch_dtype, device=device)
            if seed_bufs is not None:
                vf.copy_(seed_bufs[0])
                vsq.copy_(seed_bufs[1])
                fsq.copy_(seed_bufs[2])
            fire2_step_coord_cell(
                pos,
                vel,
                forces,
                cell,
                cell_vel,
                cell_force,
                batch_idx,
                alpha,
                dt,
                nsteps_inc,
                vf=vf,
                v_sumsq=vsq,
                f_sumsq=fsq,
                max_norm=mn,
                compute_reductions=compute_reductions,
                **FIRE2_CELL_DEFAULTS,
            )
            return pos, vel, cell, cell_vel, alpha, dt, nsteps_inc, (vf, vsq, fsq)

        rp, rv, rc, rcv, ra, rd, rn, bufs = run(True)
        dp, dv, dc, dcv, da, dd, dn, _ = run(False, seed_bufs=[b.clone() for b in bufs])
        torch.cuda.synchronize()

        assert torch.equal(dp, rp)
        assert torch.equal(dv, rv)
        assert torch.equal(dc, rc)
        assert torch.equal(dcv, rcv)
        assert torch.equal(da, ra)
        assert torch.equal(dd, rd)
        assert torch.equal(dn, rn)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_apply_step_two_phase_matches_fire2_step(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """fire2_update + fire2_apply_step must reproduce fire2_step exactly.

        This is the seam that lets a caller post-process max_norm (e.g. combine
        per-partition maxima) between the velocity mix and the clamp.
        """
        N, M = 12, 2
        batch_np = np.repeat(np.arange(M, dtype=np.int32), N // M)
        rng = np.random.default_rng(9)
        pos_np = rng.standard_normal((N, 3)).astype(np_dtype)
        vel_np = (rng.standard_normal((N, 3)) * 0.01).astype(np_dtype)
        frc_np = rng.standard_normal((N, 3)).astype(np_dtype)

        def make():
            return (
                wp.array(pos_np.copy(), dtype=dtype_vec, device=device),
                wp.array(vel_np.copy(), dtype=dtype_vec, device=device),
                wp.array(frc_np, dtype=dtype_vec, device=device),
                wp.array(batch_np, dtype=wp.int32, device=device),
                wp.array(np.full(M, 0.09, np_dtype), dtype=dtype_scalar, device=device),
                wp.array(np.full(M, 0.05, np_dtype), dtype=dtype_scalar, device=device),
                wp.zeros(M, dtype=wp.int32, device=device),
            )

        def bufs():
            return [wp.zeros(M, dtype=dtype_scalar, device=device) for _ in range(4)]

        p1, v1, f1, b1, a1, d1, n1 = make()
        vf1, vs1, fs1, mn1 = bufs()
        fire2_step(p1, v1, f1, b1, a1, d1, n1, vf1, vs1, fs1, mn1, maxstep=0.1)

        p2, v2, f2, b2, a2, d2, n2 = make()
        vf2, vs2, fs2, mn2 = bufs()
        fire2_update(v2, f2, b2, a2, d2, n2, vf2, vs2, fs2, mn2)
        fire2_apply_step(p2, v2, d2, b2, mn2, vf2, maxstep=0.1)
        wp.synchronize_device(device)

        np.testing.assert_array_equal(p2.numpy(), p1.numpy())
        np.testing.assert_array_equal(v2.numpy(), v1.numpy())
        np.testing.assert_array_equal(d2.numpy(), d1.numpy())

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("torch_dtype", [torch.float32, torch.float64])
    def test_coord_cell_three_call_matches_monolith(self, device, torch_dtype):
        """mix + couple + apply must reproduce fire2_step_coord_cell exactly."""
        N, M = 40, 2
        np_dtype = np.float32 if torch_dtype == torch.float32 else np.float64

        def state():
            rng = np.random.default_rng(3)
            pos = torch.tensor(
                rng.standard_normal((N, 3)).astype(np_dtype), device=device
            )
            vel = torch.tensor(
                (rng.standard_normal((N, 3)) * 0.01).astype(np_dtype), device=device
            )
            frc = torch.tensor(
                rng.standard_normal((N, 3)).astype(np_dtype), device=device
            )
            cell = torch.tensor(
                _make_upper_triangular_cell(M, np_dtype, rng=rng),
                dtype=torch_dtype,
                device=device,
            )
            cell_vel = torch.zeros(M, 3, 3, dtype=torch_dtype, device=device)
            cell_force = torch.tensor(
                rng.standard_normal((M, 3, 3)).astype(np_dtype) * 0.01,
                dtype=torch_dtype,
                device=device,
            )
            batch_idx = torch.tensor(
                np.repeat(np.arange(M, dtype=np.int32), N // M),
                dtype=torch.int32,
                device=device,
            )
            alpha = torch.full((M,), 0.09, dtype=torch_dtype, device=device)
            dt = torch.full((M,), 0.05, dtype=torch_dtype, device=device)
            nsteps = torch.zeros(M, dtype=torch.int32, device=device)
            return (
                pos,
                vel,
                frc,
                cell,
                cell_vel,
                cell_force,
                batch_idx,
                alpha,
                dt,
                nsteps,
            )

        p1, v1, f1, c1, cv1, cf1, b1, a1, d1, n1 = state()
        fire2_step_coord_cell(
            p1, v1, f1, c1, cv1, cf1, b1, a1, d1, n1, **FIRE2_CELL_DEFAULTS
        )

        p2, v2, f2, c2, cv2, cf2, b2, a2, d2, n2 = state()
        vf = torch.zeros(M, dtype=torch_dtype, device=device)
        vs = torch.zeros(M, dtype=torch_dtype, device=device)
        fs = torch.zeros(M, dtype=torch_dtype, device=device)
        mn = torch.zeros(M, dtype=torch_dtype, device=device)
        fire2_step_coord_cell_mix(
            p2,
            v2,
            f2,
            c2,
            cv2,
            cf2,
            b2,
            a2,
            d2,
            n2,
            vf=vf,
            v_sumsq=vs,
            f_sumsq=fs,
            max_norm=mn,
            **{k: v for k, v in FIRE2_CELL_DEFAULTS.items() if k != "maxstep"},
        )
        fire2_step_coord_cell_couple(p2, v2, c2, cv2, d2, vf, b2, mn)
        fire2_step_coord_cell_apply(
            p2, v2, c2, cv2, d2, vf, b2, mn, maxstep=FIRE2_CELL_DEFAULTS["maxstep"]
        )
        torch.cuda.synchronize()

        assert torch.equal(p2, p1)
        assert torch.equal(c2, c1)
        assert torch.equal(v2, v1)
        assert torch.equal(cv2, cv1)
        assert torch.equal(d2, d1)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("torch_dtype", [torch.float32, torch.float64])
    def test_extended_reductions_split_matches_internal(self, device, torch_dtype):
        """atom_partial + cell_term fed with compute_reductions=False must match
        the internally-computed reduction (bit-identical mixed state)."""
        N, M = 40, 2
        np_dtype = np.float32 if torch_dtype == torch.float32 else np.float64

        def state():
            rng = np.random.default_rng(3)
            return (
                torch.tensor(
                    rng.standard_normal((N, 3)).astype(np_dtype), device=device
                ),
                torch.tensor(
                    (rng.standard_normal((N, 3)) * 0.01).astype(np_dtype), device=device
                ),
                torch.tensor(
                    rng.standard_normal((N, 3)).astype(np_dtype), device=device
                ),
                torch.tensor(
                    _make_upper_triangular_cell(M, np_dtype, rng=rng),
                    dtype=torch_dtype,
                    device=device,
                ),
                torch.zeros(M, 3, 3, dtype=torch_dtype, device=device),
                torch.tensor(
                    rng.standard_normal((M, 3, 3)).astype(np_dtype) * 0.01,
                    dtype=torch_dtype,
                    device=device,
                ),
                torch.tensor(
                    np.repeat(np.arange(M, dtype=np.int32), N // M),
                    dtype=torch.int32,
                    device=device,
                ),
                torch.full((M,), 0.09, dtype=torch_dtype, device=device),
                torch.full((M,), 0.05, dtype=torch_dtype, device=device),
                torch.zeros(M, dtype=torch.int32, device=device),
            )

        hp = {k: v for k, v in FIRE2_CELL_DEFAULTS.items() if k != "maxstep"}

        # Reference: internal reduction.
        pr, vr, fr, cr, cvr, cfr, br, ar, dr, nr = state()
        vfr = torch.zeros(M, dtype=torch_dtype, device=device)
        vsr = torch.zeros(M, dtype=torch_dtype, device=device)
        fsr = torch.zeros(M, dtype=torch_dtype, device=device)
        mnr = torch.zeros(M, dtype=torch_dtype, device=device)
        fire2_step_coord_cell_mix(
            pr,
            vr,
            fr,
            cr,
            cvr,
            cfr,
            br,
            ar,
            dr,
            nr,
            vf=vfr,
            v_sumsq=vsr,
            f_sumsq=fsr,
            max_norm=mnr,
            **hp,
        )

        # Combined split reductions fed with compute_reductions=False.
        pd, vd, fd, cd, cvd, cfd, bd, ad, dd, nd_ = state()
        atom, cell_t = fire2_compute_extended_reductions(
            pd, vd, fd, cd, cvd, cfd, bd, dd, cell_force_scale=1.0
        )
        vfd = (atom[0] + cell_t[0]).clone()
        vsd = (atom[1] + cell_t[1]).clone()
        fsd = (atom[2] + cell_t[2]).clone()
        mnd = torch.zeros(M, dtype=torch_dtype, device=device)
        fire2_step_coord_cell_mix(
            pd,
            vd,
            fd,
            cd,
            cvd,
            cfd,
            bd,
            ad,
            dd,
            nd_,
            vf=vfd,
            v_sumsq=vsd,
            f_sumsq=fsd,
            max_norm=mnd,
            compute_reductions=False,
            **hp,
        )
        torch.cuda.synchronize()

        rtol = 1e-6 if np_dtype == np.float32 else 1e-12
        torch.testing.assert_close(vd, vr, rtol=rtol, atol=rtol)
        torch.testing.assert_close(dd, dr, rtol=rtol, atol=rtol)
        torch.testing.assert_close(ad, ar, rtol=rtol, atol=rtol)
