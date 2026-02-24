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

from nvalchemiops.dynamics.optimizers import fire2_step
from nvalchemiops.torch.fire2 import (
    fire2_step_coord,
    fire2_step_coord_cell,
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
            **FIRE2_DEFAULTS,
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
    @pytest.mark.parametrize("torch_dtype", [torch.float32, torch.float64])
    def test_cell_step_matches_manual_pack(self, device, torch_dtype):
        """fire2_step_coord_cell must match manually packed fire2_step_coord."""
        rng = np.random.default_rng(51)
        N, M = 60, 2
        np_dtype = np.float32 if torch_dtype == torch.float32 else np.float64
        vec_type = wp.vec3f if torch_dtype == torch.float32 else wp.vec3d
        mat_type = wp.mat33f if torch_dtype == torch.float32 else wp.mat33d

        # Create identical state for both methods
        pos_np = rng.standard_normal((N, 3)).astype(np_dtype)
        vel_np = rng.standard_normal((N, 3)).astype(np_dtype) * 0.01
        forces_np = rng.standard_normal((N, 3)).astype(np_dtype)
        bidx_np = np.repeat(np.arange(M, dtype=np.int32), N // M)
        alpha_np = np.full(M, 0.09, dtype=np_dtype)
        dt_np = np.full(M, 0.05, dtype=np_dtype)
        nsteps_np = np.zeros(M, dtype=np.int32)
        cell_np = _make_upper_triangular_cell(M, np_dtype, rng=rng)
        cell_vel_np = np.zeros((M, 3, 3), dtype=np_dtype)
        cell_force_np = rng.standard_normal((M, 3, 3)).astype(np_dtype) * 0.01

        # --- Method A: fire2_step_coord_cell ---
        pos_a = torch.tensor(pos_np.copy(), dtype=torch_dtype, device=device)
        vel_a = torch.tensor(vel_np.copy(), dtype=torch_dtype, device=device)
        forces_a = torch.tensor(forces_np.copy(), dtype=torch_dtype, device=device)
        bidx_a = torch.tensor(bidx_np.copy(), dtype=torch.int32, device=device)
        alpha_a = torch.tensor(alpha_np.copy(), dtype=torch_dtype, device=device)
        dt_a = torch.tensor(dt_np.copy(), dtype=torch_dtype, device=device)
        nsteps_a = torch.tensor(nsteps_np.copy(), dtype=torch.int32, device=device)
        cell_a = torch.tensor(cell_np.copy(), dtype=torch_dtype, device=device)
        cell_vel_a = torch.tensor(cell_vel_np.copy(), dtype=torch_dtype, device=device)
        cell_force_a = torch.tensor(
            cell_force_np.copy(), dtype=torch_dtype, device=device
        )

        fire2_step_coord_cell(
            pos_a,
            vel_a,
            forces_a,
            cell_a,
            cell_vel_a,
            cell_force_a,
            bidx_a,
            alpha_a,
            dt_a,
            nsteps_a,
            **FIRE2_DEFAULTS,
        )
        torch.cuda.synchronize()

        # --- Method B: manual pack + fire2_step (Warp) + unpack ---
        from nvalchemiops.batch_utils import (
            atom_ptr_to_batch_idx,
            batch_idx_to_atom_ptr,
        )
        from nvalchemiops.dynamics.utils.cell_filter import (
            extend_atom_ptr,
            pack_forces_with_cell,
            pack_positions_with_cell,
            pack_velocities_with_cell,
            unpack_positions_with_cell,
            unpack_velocities_with_cell,
        )

        wp_device = wp.device_from_torch(torch.device(device))

        wp_pos = wp.array(pos_np.copy(), dtype=vec_type, device=wp_device)
        wp_vel = wp.array(vel_np.copy(), dtype=vec_type, device=wp_device)
        wp_forces = wp.array(forces_np.copy(), dtype=vec_type, device=wp_device)
        wp_bidx = wp.array(bidx_np.copy(), dtype=wp.int32, device=wp_device)
        wp_cell = wp.array(cell_np.copy(), dtype=mat_type, device=wp_device)
        wp_cell_vel = wp.array(cell_vel_np.copy(), dtype=mat_type, device=wp_device)
        wp_cell_force = wp.array(cell_force_np.copy(), dtype=mat_type, device=wp_device)
        scalar_dtype = wp.float32 if torch_dtype == torch.float32 else wp.float64
        wp_alpha = wp.array(alpha_np.copy(), dtype=scalar_dtype, device=wp_device)
        wp_dt = wp.array(dt_np.copy(), dtype=scalar_dtype, device=wp_device)
        wp_nsteps = wp.array(nsteps_np.copy(), dtype=wp.int32, device=wp_device)

        N_ext = N + 2 * M

        # Compute atom_ptr / ext_atom_ptr for batched pack/unpack
        wp_atom_counts = wp.zeros(M, dtype=wp.int32, device=wp_device)
        wp_atom_ptr = wp.zeros(M + 1, dtype=wp.int32, device=wp_device)
        batch_idx_to_atom_ptr(wp_bidx, wp_atom_counts, wp_atom_ptr)
        wp_ext_atom_ptr = wp.zeros(M + 1, dtype=wp.int32, device=wp_device)
        extend_atom_ptr(wp_atom_ptr, wp_ext_atom_ptr, device=wp_device)

        # Pre-allocate extended arrays
        ext_pos = wp.zeros(N_ext, dtype=vec_type, device=wp_device)
        ext_vel = wp.zeros(N_ext, dtype=vec_type, device=wp_device)
        ext_forces_b = wp.zeros(N_ext, dtype=vec_type, device=wp_device)
        ext_bidx = wp.zeros(N_ext, dtype=wp.int32, device=wp_device)

        pack_positions_with_cell(
            wp_pos,
            wp_cell,
            ext_pos,
            wp_atom_ptr,
            wp_ext_atom_ptr,
            device=wp_device,
        )
        pack_velocities_with_cell(
            wp_vel,
            wp_cell_vel,
            ext_vel,
            wp_atom_ptr,
            wp_ext_atom_ptr,
            device=wp_device,
        )
        pack_forces_with_cell(
            wp_forces,
            wp_cell_force,
            ext_forces_b,
            wp_atom_ptr,
            wp_ext_atom_ptr,
            device=wp_device,
        )
        atom_ptr_to_batch_idx(wp_ext_atom_ptr, ext_bidx)

        vf, v_sumsq, f_sumsq, max_norm_buf = make_fire2_scratch(
            M, scalar_dtype, wp_device
        )

        fire2_step(
            ext_pos,
            ext_vel,
            ext_forces_b,
            ext_bidx,
            wp_alpha,
            wp_dt,
            wp_nsteps,
            vf,
            v_sumsq,
            f_sumsq,
            max_norm_buf,
            **FIRE2_DEFAULTS,
        )

        # Pre-allocate unpack outputs
        wp_pos_b = wp.zeros(N, dtype=vec_type, device=wp_device)
        wp_cell_b = wp.zeros(M, dtype=mat_type, device=wp_device)
        wp_vel_b = wp.zeros(N, dtype=vec_type, device=wp_device)
        wp_cell_vel_b = wp.zeros(M, dtype=mat_type, device=wp_device)

        unpack_positions_with_cell(
            ext_pos,
            wp_pos_b,
            wp_cell_b,
            atom_ptr=wp_atom_ptr,
            ext_atom_ptr=wp_ext_atom_ptr,
            device=wp_device,
        )
        unpack_velocities_with_cell(
            ext_vel,
            wp_vel_b,
            wp_cell_vel_b,
            atom_ptr=wp_atom_ptr,
            ext_atom_ptr=wp_ext_atom_ptr,
            device=wp_device,
        )
        wp.synchronize()

        pos_b = wp.to_torch(wp_pos_b).reshape(-1, 3)
        vel_b = wp.to_torch(wp_vel_b).reshape(-1, 3)
        cell_b = wp.to_torch(wp_cell_b).reshape(M, 3, 3)
        cell_vel_b = wp.to_torch(wp_cell_vel_b).reshape(M, 3, 3)
        alpha_b = wp.to_torch(wp_alpha)
        dt_b = wp.to_torch(wp_dt)
        nsteps_b = wp.to_torch(wp_nsteps)

        # --- Compare ---
        rtol = 1e-4 if np_dtype == np.float32 else 1e-10
        np.testing.assert_allclose(
            pos_a.cpu().numpy(),
            pos_b.cpu().numpy(),
            rtol=rtol,
            atol=1e-7,
            err_msg="Positions mismatch",
        )
        np.testing.assert_allclose(
            vel_a.cpu().numpy(),
            vel_b.cpu().numpy(),
            rtol=rtol,
            atol=1e-7,
            err_msg="Velocities mismatch",
        )
        np.testing.assert_allclose(
            cell_a.cpu().numpy(),
            cell_b.cpu().numpy(),
            rtol=rtol,
            atol=1e-7,
            err_msg="Cell mismatch",
        )
        np.testing.assert_allclose(
            cell_vel_a.cpu().numpy(),
            cell_vel_b.cpu().numpy(),
            rtol=rtol,
            atol=1e-7,
            err_msg="Cell velocities mismatch",
        )
        np.testing.assert_allclose(
            alpha_a.cpu().numpy(),
            alpha_b.cpu().numpy(),
            rtol=rtol,
            err_msg="Alpha mismatch",
        )
        np.testing.assert_allclose(
            dt_a.cpu().numpy(),
            dt_b.cpu().numpy(),
            rtol=rtol,
            err_msg="dt mismatch",
        )
        np.testing.assert_array_equal(
            nsteps_a.cpu().numpy(),
            nsteps_b.cpu().numpy(),
        )

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
        ext_pos = torch.empty(N_ext, 3, dtype=torch_dtype, device=device)
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
            ext_positions=ext_pos,
            ext_velocities=ext_vel,
            ext_forces=ext_forces,
            vf=vf,
            v_sumsq=v_sumsq,
            f_sumsq=f_sumsq,
            max_norm=max_norm_buf,
            **FIRE2_DEFAULTS,
        )
        torch.cuda.synchronize()

        assert not torch.allclose(pos, pos_before), "Positions should be updated"
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
        ext_pos = torch.empty(N_ext, 3, dtype=torch_dtype, device=device)
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
            ext_positions=ext_pos,
            ext_velocities=ext_vel,
            ext_forces=ext_forces,
            vf=vf,
            v_sumsq=v_sumsq,
            f_sumsq=f_sumsq,
            max_norm=max_norm_buf,
            **FIRE2_DEFAULTS,
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
                ext_positions=ext_pos,
                ext_velocities=ext_vel,
                ext_forces=ext_forces,
                ext_batch_idx=ext_bidx,
                vf=vf,
                v_sumsq=v_sumsq,
                f_sumsq=f_sumsq,
                max_norm=max_norm_buf,
                **FIRE2_DEFAULTS,
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
            **FIRE2_DEFAULTS,
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
                **FIRE2_DEFAULTS,
            )

        torch.cuda.synchronize()
        assert torch.isfinite(pos).all(), "Positions should stay finite over 10 steps"
        assert torch.isfinite(cell).all(), "Cell should stay finite over 10 steps"
        assert torch.isfinite(vel).all(), "Velocities should stay finite"
        assert torch.isfinite(cell_vel).all(), "Cell velocities should stay finite"

    @pytest.mark.parametrize("device", DEVICES)
    def test_cell_step_single_system_parity(self, device):
        """M==1 produces identical results to batched path."""
        rng = np.random.default_rng(56)
        N, M = 40, 1
        torch_dtype = torch.float64
        np_dtype = np.float64

        # Create identical state for both runs
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

        # Run A: M==1 (convenience API)
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

        # Run B: Force batched path by providing atom_ptr
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

        # Use M=1 but provide atom_ptr to force batched pack/unpack path.
        # Run B: Manual pack/fire2_step/unpack using batched kernels.
        from nvalchemiops.dynamics.optimizers.fire2 import (
            fire2_step as _fire2_step,
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

        wp_pos_b = wp.from_torch(pos_b, dtype=vec_type)
        wp_vel_b = wp.from_torch(vel_b, dtype=vec_type)
        wp_forces = wp.from_torch(forces, dtype=vec_type)
        wp_cell_b = wp.from_torch(cell_b, dtype=mat_type)
        wp_cell_vel_b = wp.from_torch(cell_vel_b, dtype=mat_type)
        wp_cell_force = wp.from_torch(cell_force, dtype=mat_type)
        wp_atom_ptr = wp.from_torch(atom_ptr, dtype=wp.int32)
        wp_ext_atom_ptr = wp.from_torch(ext_atom_ptr, dtype=wp.int32)

        ext_pos = torch.empty(N_ext, 3, dtype=torch_dtype, device=device)
        ext_vel = torch.empty(N_ext, 3, dtype=torch_dtype, device=device)
        ext_forces = torch.empty(N_ext, 3, dtype=torch_dtype, device=device)
        wp_ext_pos = wp.from_torch(ext_pos, dtype=vec_type)
        wp_ext_vel = wp.from_torch(ext_vel, dtype=vec_type)
        wp_ext_forces = wp.from_torch(ext_forces, dtype=vec_type)

        # Pack using batched kernels
        pack_positions_with_cell(
            wp_pos_b,
            wp_cell_b,
            wp_ext_pos,
            wp_atom_ptr,
            wp_ext_atom_ptr,
            device=wp_device,
        )
        pack_velocities_with_cell(
            wp_vel_b,
            wp_cell_vel_b,
            wp_ext_vel,
            wp_atom_ptr,
            wp_ext_atom_ptr,
            device=wp_device,
        )
        pack_forces_with_cell(
            wp_forces,
            wp_cell_force,
            wp_ext_forces,
            wp_atom_ptr,
            wp_ext_atom_ptr,
            device=wp_device,
        )

        vf = torch.zeros(M, dtype=torch_dtype, device=device)
        v_sumsq = torch.zeros(M, dtype=torch_dtype, device=device)
        f_sumsq = torch.zeros(M, dtype=torch_dtype, device=device)
        max_norm_buf = torch.zeros(M, dtype=torch_dtype, device=device)

        _fire2_step(
            wp_ext_pos,
            wp_ext_vel,
            wp_ext_forces,
            wp.from_torch(ext_bidx, dtype=wp.int32),
            wp.from_torch(alpha_b),
            wp.from_torch(dt_b),
            wp.from_torch(nsteps_b, dtype=wp.int32),
            wp.from_torch(vf),
            wp.from_torch(v_sumsq),
            wp.from_torch(f_sumsq),
            wp.from_torch(max_norm_buf),
            **FIRE2_DEFAULTS,
        )

        # Unpack using batched kernels
        unpack_positions_with_cell(
            wp_ext_pos,
            wp_pos_b,
            wp_cell_b,
            atom_ptr=wp_atom_ptr,
            ext_atom_ptr=wp_ext_atom_ptr,
            device=wp_device,
        )
        unpack_velocities_with_cell(
            wp_ext_vel,
            wp_vel_b,
            wp_cell_vel_b,
            atom_ptr=wp_atom_ptr,
            ext_atom_ptr=wp_ext_atom_ptr,
            device=wp_device,
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
                **FIRE2_DEFAULTS,
            )

        torch.cuda.synchronize()

        assert not torch.allclose(pos, pos_before), "Positions should be updated"
        assert not torch.allclose(cell, cell_before), "Cell should be updated"
        assert torch.isfinite(pos).all(), "Positions should stay finite"
        assert torch.isfinite(vel).all(), "Velocities should stay finite"
        assert torch.isfinite(cell).all(), "Cell should stay finite"
        assert torch.isfinite(cell_vel).all(), "Cell velocities should stay finite"

    @pytest.mark.parametrize("device", DEVICES)
    def test_fire2_step_extended_parity(self, device):
        """fire2_step_extended matches fire2_step_coord_cell over multiple steps."""
        rng = np.random.default_rng(58)
        N, M = 40, 2
        torch_dtype = torch.float64
        np_dtype = np.float64
        num_steps = 5

        # --- Path A: fire2_step_coord_cell (convenience API) ---
        (pos_a, vel_a, forces_a, batch_idx, alpha_a, dt_a, nsteps_a, *_) = (
            make_fire2_torch_state(N, M, torch_dtype, device, rng=rng)
        )
        cell_np = _make_upper_triangular_cell(M, np_dtype, rng=rng)
        cell_a = torch.tensor(cell_np, dtype=torch_dtype, device=device)
        cell_vel_a = torch.zeros(M, 3, 3, dtype=torch_dtype, device=device)
        cell_force_np = rng.standard_normal((M, 3, 3)).astype(np_dtype) * 0.01
        cell_force_a = torch.tensor(cell_force_np, dtype=torch_dtype, device=device)

        for _ in range(num_steps):
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
                **FIRE2_DEFAULTS,
            )

        # --- Path B: manual pack + fire2_step_extended + manual unpack ---
        rng2 = np.random.default_rng(58)  # same seed for identical initial data
        (pos_b, vel_b, forces_b, batch_idx_b, alpha_b, dt_b, nsteps_b, *_) = (
            make_fire2_torch_state(N, M, torch_dtype, device, rng=rng2)
        )
        cell_np_b = _make_upper_triangular_cell(M, np_dtype, rng=rng2)
        cell_b = torch.tensor(cell_np_b, dtype=torch_dtype, device=device)
        cell_vel_b = torch.zeros(M, 3, 3, dtype=torch_dtype, device=device)
        cell_force_np_b = rng2.standard_normal((M, 3, 3)).astype(np_dtype) * 0.01
        cell_force_b = torch.tensor(cell_force_np_b, dtype=torch_dtype, device=device)

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

        for _ in range(num_steps):
            # Pack
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
            pack_forces_with_cell(
                wp.from_torch(forces_b, dtype=vec_type),
                wp.from_torch(cell_force_b, dtype=mat_type),
                wp.from_torch(ext_forces, dtype=vec_type),
                wp_atom_ptr,
                wp_ext_atom_ptr,
                device=wp_device,
                batch_idx=wp_bidx,
            )

            # FIRE2 on extended arrays
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

            # Unpack
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

        torch.testing.assert_close(pos_a, pos_b, atol=0, rtol=0)
        torch.testing.assert_close(vel_a, vel_b, atol=0, rtol=0)
        torch.testing.assert_close(cell_a, cell_b, atol=0, rtol=0)
        torch.testing.assert_close(cell_vel_a, cell_vel_b, atol=0, rtol=0)
        torch.testing.assert_close(alpha_a, alpha_b, atol=0, rtol=0)
        torch.testing.assert_close(dt_a, dt_b, atol=0, rtol=0)
