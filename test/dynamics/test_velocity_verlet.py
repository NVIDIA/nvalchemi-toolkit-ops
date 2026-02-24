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
Unit tests for Velocity Verlet integrator.

Tests cover:
- Basic API functionality (single and batched)
- Position and velocity update correctness
- Energy conservation for harmonic oscillator
- Symplectic property verification
- Float32 and float64 support
"""

import numpy as np
import pytest
import warp as wp

from nvalchemiops.dynamics.integrators import (
    velocity_verlet_position_update,
    velocity_verlet_position_update_out,
    velocity_verlet_velocity_finalize,
    velocity_verlet_velocity_finalize_out,
)

# ==============================================================================
# Test Configuration
# ==============================================================================

DEVICES = ["cuda:0"]

DTYPE_CONFIGS = [
    pytest.param(wp.vec3f, wp.float32, np.float32, id="float32"),
    pytest.param(wp.vec3d, wp.float64, np.float64, id="float64"),
]


# ==============================================================================
# Helper Functions for Tests
# ==============================================================================


@wp.kernel
def _compute_harmonic_forces_kernel_f32(
    positions: wp.array(dtype=wp.vec3f),
    forces: wp.array(dtype=wp.vec3f),
    spring_k: wp.float32,
):
    """Compute harmonic forces F = -k * r for testing (float32)."""
    i = wp.tid()
    pos = positions[i]
    forces[i] = wp.vec3f(-spring_k * pos[0], -spring_k * pos[1], -spring_k * pos[2])


@wp.kernel
def _compute_harmonic_forces_kernel_f64(
    positions: wp.array(dtype=wp.vec3d),
    forces: wp.array(dtype=wp.vec3d),
    spring_k: wp.float64,
):
    """Compute harmonic forces F = -k * r for testing (float64)."""
    i = wp.tid()
    pos = positions[i]
    forces[i] = wp.vec3d(-spring_k * pos[0], -spring_k * pos[1], -spring_k * pos[2])


def compute_harmonic_forces(positions: wp.array, forces: wp.array, k: float):
    """Compute harmonic forces F = -k * r."""
    if positions.dtype == wp.vec3f:
        wp.launch(
            _compute_harmonic_forces_kernel_f32,
            dim=positions.shape[0],
            inputs=[positions, forces, wp.float32(k)],
            device=positions.device,
        )
    else:
        wp.launch(
            _compute_harmonic_forces_kernel_f64,
            dim=positions.shape[0],
            inputs=[positions, forces, wp.float64(k)],
            device=positions.device,
        )


def compute_harmonic_energy(positions: np.ndarray, k: float) -> float:
    """Compute harmonic potential energy E = 0.5 * k * |r|^2."""
    r_sq = np.sum(positions**2)
    return 0.5 * k * r_sq


def compute_kinetic_energy_np(velocities: np.ndarray, masses: np.ndarray) -> float:
    """Compute kinetic energy KE = 0.5 * sum(m * v^2)."""
    v_sq = np.sum(velocities**2, axis=1)
    return 0.5 * np.sum(masses * v_sq)


def compute_morse_forces_and_energy(
    positions: np.ndarray, D_e: float, a: float, r_e: float
) -> tuple[np.ndarray, float]:
    """Compute Morse potential forces and energy for a dimer."""
    forces = np.zeros_like(positions)
    r_vec = positions[1] - positions[0]
    r = np.linalg.norm(r_vec)

    if r > 1e-10:
        r_hat = r_vec / r
        exp_term = np.exp(-a * (r - r_e))
        dVdr = 2 * D_e * a * (1 - exp_term) * exp_term
        forces[1] = -dVdr * r_hat
        forces[0] = dVdr * r_hat
        energy = D_e * (1 - exp_term) ** 2
    else:
        energy = 0.0

    return forces, float(energy)


# ==============================================================================
# Single System API Tests
# ==============================================================================


class TestVelocityVerletAPI:
    """Test single-system API functionality including non-mutating variants."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_position_update_runs(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test that velocity_verlet_position_update executes without error."""
        num_atoms = 10
        np.random.seed(42)

        positions = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        forces = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        dt = wp.array([0.001], dtype=dtype_scalar, device=device)

        velocity_verlet_position_update(positions, velocities, forces, masses, dt)
        wp.synchronize_device(device)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_position_update_device_inference(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test that device is inferred from positions."""
        num_atoms = 20

        positions = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype) * 0.1,
            dtype=dtype_vec,
            device=device,
        )
        forces = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        dt = wp.array([0.001], dtype=dtype_scalar, device=device)

        # Call without explicit device
        velocity_verlet_position_update(positions, velocities, forces, masses, dt)

        wp.synchronize_device(device)
        assert positions.shape[0] == num_atoms

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_position_update_out_preserves_input(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test that non-mutating velocity_verlet_position_update_out preserves input."""
        num_atoms = 10
        np.random.seed(42)

        positions = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        forces = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        dt = wp.array([0.001], dtype=dtype_scalar, device=device)

        pos_orig = positions.numpy().copy()
        vel_orig = velocities.numpy().copy()

        positions_out = wp.empty_like(positions)
        velocities_out = wp.empty_like(velocities)

        pos_out, vel_out = velocity_verlet_position_update_out(
            positions, velocities, forces, masses, dt, positions_out, velocities_out
        )
        wp.synchronize_device(device)

        np.testing.assert_array_equal(positions.numpy(), pos_orig)
        np.testing.assert_array_equal(velocities.numpy(), vel_orig)
        assert not np.allclose(pos_out.numpy(), pos_orig)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_position_update_out_with_preallocated(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test position_update_out with pre-allocated outputs."""
        num_atoms = 20

        positions = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype) * 0.1,
            dtype=dtype_vec,
            device=device,
        )
        forces = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        dt = wp.array([0.001], dtype=dtype_scalar, device=device)

        positions_out = wp.empty_like(positions)
        velocities_out = wp.empty_like(velocities)

        pos_out, vel_out = velocity_verlet_position_update_out(
            positions,
            velocities,
            forces,
            masses,
            dt,
            positions_out,
            velocities_out,
            device=device,
        )

        wp.synchronize_device(device)
        assert pos_out is positions_out
        assert vel_out is velocities_out

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_velocity_finalize_runs(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test that velocity_verlet_velocity_finalize executes without error."""
        num_atoms = 10
        np.random.seed(42)

        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        new_forces = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        dt = wp.array([0.001], dtype=dtype_scalar, device=device)

        velocity_verlet_velocity_finalize(velocities, new_forces, masses, dt)
        wp.synchronize_device(device)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_velocity_finalize_device_inference(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test that device is inferred from velocities."""
        num_atoms = 20

        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        forces_new = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        dt = wp.array([0.001], dtype=dtype_scalar, device=device)

        velocity_verlet_velocity_finalize(velocities, forces_new, masses, dt)

        wp.synchronize_device(device)
        assert velocities.shape[0] == num_atoms

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_velocity_finalize_out_preserves_input(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test non-mutating velocity finalize preserves input."""
        num_atoms = 20

        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        forces_new = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        dt = wp.array([0.001], dtype=dtype_scalar, device=device)

        vel_orig = velocities.numpy().copy()

        velocities_out = wp.empty_like(velocities)

        vel_out = velocity_verlet_velocity_finalize_out(
            velocities, forces_new, masses, dt, velocities_out, device=device
        )

        np.testing.assert_array_equal(velocities.numpy(), vel_orig)
        assert vel_out.shape[0] == num_atoms

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_velocity_finalize_out_with_preallocated(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test velocity_finalize_out with pre-allocated output."""
        num_atoms = 20

        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        forces_new = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        dt = wp.array([0.001], dtype=dtype_scalar, device=device)

        velocities_out = wp.empty_like(velocities)

        vel_out = velocity_verlet_velocity_finalize_out(
            velocities,
            forces_new,
            masses,
            dt,
            velocities_out,
            device=device,
        )

        wp.synchronize_device(device)
        assert vel_out is velocities_out


# ==============================================================================
# Batched API Tests
# ==============================================================================


class TestVelocityVerletBatched:
    """Test batched API functionality including non-mutating variants."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_batched_position_update_runs(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test that batched velocity_verlet_position_update executes correctly."""
        num_systems = 3
        atoms_per_system = 5
        total_atoms = num_systems * atoms_per_system
        np.random.seed(42)

        positions = wp.array(
            np.random.randn(total_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        velocities = wp.array(
            np.random.randn(total_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        forces = wp.array(
            np.random.randn(total_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(total_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        dt = wp.array([0.001, 0.002, 0.003], dtype=dtype_scalar, device=device)

        batch_idx = wp.array(
            np.repeat(np.arange(num_systems), atoms_per_system).astype(np.int32),
            dtype=wp.int32,
            device=device,
        )

        velocity_verlet_position_update(
            positions, velocities, forces, masses, dt, batch_idx=batch_idx
        )
        wp.synchronize_device(device)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_batched_position_update_out(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test non-mutating position update for batched systems."""
        num_atoms = 40

        positions = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype) * 0.1,
            dtype=dtype_vec,
            device=device,
        )
        forces = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        batch_idx = wp.array(
            np.array([0] * 20 + [1] * 20, dtype=np.int32),
            dtype=wp.int32,
            device=device,
        )
        dt = wp.array([0.001, 0.001], dtype=dtype_scalar, device=device)

        positions_out = wp.empty_like(positions)
        velocities_out = wp.empty_like(velocities)

        pos_out, vel_out = velocity_verlet_position_update_out(
            positions,
            velocities,
            forces,
            masses,
            dt,
            positions_out,
            velocities_out,
            batch_idx=batch_idx,
            device=device,
        )

        assert pos_out.shape[0] == num_atoms
        assert vel_out.shape[0] == num_atoms

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_batched_velocity_finalize_runs(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test that batched velocity_verlet_velocity_finalize executes correctly."""
        num_systems = 3
        atoms_per_system = 5
        total_atoms = num_systems * atoms_per_system
        np.random.seed(42)

        velocities = wp.array(
            np.random.randn(total_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        new_forces = wp.array(
            np.random.randn(total_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(total_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        dt = wp.array([0.001, 0.002, 0.003], dtype=dtype_scalar, device=device)
        batch_idx = wp.array(
            np.repeat(np.arange(num_systems), atoms_per_system).astype(np.int32),
            dtype=wp.int32,
            device=device,
        )

        velocity_verlet_velocity_finalize(
            velocities, new_forces, masses, dt, batch_idx=batch_idx
        )
        wp.synchronize_device(device)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_batched_velocity_finalize_out(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test non-mutating velocity finalize for batched systems."""
        num_atoms = 40

        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        forces_new = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        batch_idx = wp.array(
            np.array([0] * 20 + [1] * 20, dtype=np.int32),
            dtype=wp.int32,
            device=device,
        )
        dt = wp.array([0.001, 0.001], dtype=dtype_scalar, device=device)

        velocities_out = wp.empty_like(velocities)

        vel_out = velocity_verlet_velocity_finalize_out(
            velocities,
            forces_new,
            masses,
            dt,
            velocities_out,
            batch_idx=batch_idx,
            device=device,
        )

        assert vel_out.shape[0] == num_atoms

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_batched_uses_per_system_dt(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test that batched version uses per-system timesteps correctly."""
        pos = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np_dtype)
        vel = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np_dtype)
        force = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np_dtype)
        mass = np.array([1.0, 1.0], dtype=np_dtype)
        dt_vals = [0.01, 0.02]

        positions = wp.array(pos, dtype=dtype_vec, device=device)
        velocities = wp.array(vel, dtype=dtype_vec, device=device)
        forces = wp.array(force, dtype=dtype_vec, device=device)
        masses = wp.array(mass, dtype=dtype_scalar, device=device)
        dt = wp.array(dt_vals, dtype=dtype_scalar, device=device)
        batch_idx = wp.array([0, 1], dtype=wp.int32, device=device)

        velocity_verlet_position_update(
            positions, velocities, forces, masses, dt, batch_idx=batch_idx
        )
        wp.synchronize_device(device)

        result_pos = positions.numpy()

        displacement_0 = result_pos[0, 0] - 1.0
        displacement_1 = result_pos[1, 0] - 1.0

        ratio = displacement_1 / displacement_0
        np.testing.assert_allclose(ratio, 4.0, rtol=0.01)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_batched_energy_conservation(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test energy conservation for multiple independent systems."""
        num_systems = 3
        atoms_per_system = 20
        total_atoms = num_systems * atoms_per_system
        spring_k = 1.0
        dt_val = 0.01
        num_steps = 200

        np.random.seed(42)

        initial_pos = np.random.randn(total_atoms, 3).astype(np_dtype) * 0.5
        initial_vel = np.random.randn(total_atoms, 3).astype(np_dtype) * 0.1
        masses_np = np.ones(total_atoms, dtype=np_dtype)

        positions = wp.array(initial_pos.copy(), dtype=dtype_vec, device=device)
        velocities = wp.array(initial_vel.copy(), dtype=dtype_vec, device=device)
        forces = wp.zeros(total_atoms, dtype=dtype_vec, device=device)
        masses = wp.array(masses_np, dtype=dtype_scalar, device=device)
        dt = wp.array([dt_val] * num_systems, dtype=dtype_scalar, device=device)
        batch_idx = wp.array(
            np.repeat(np.arange(num_systems), atoms_per_system).astype(np.int32),
            dtype=wp.int32,
            device=device,
        )

        compute_harmonic_forces(positions, forces, spring_k)

        wp.synchronize_device(device)
        pos_np = positions.numpy()
        vel_np = velocities.numpy()

        initial_energies = []
        for sys_id in range(num_systems):
            start = sys_id * atoms_per_system
            end = (sys_id + 1) * atoms_per_system
            ke = compute_kinetic_energy_np(vel_np[start:end], masses_np[start:end])
            pe = compute_harmonic_energy(pos_np[start:end], spring_k)
            initial_energies.append(ke + pe)

        for step in range(num_steps):
            velocity_verlet_position_update(
                positions, velocities, forces, masses, dt, batch_idx=batch_idx
            )
            compute_harmonic_forces(positions, forces, spring_k)
            velocity_verlet_velocity_finalize(
                velocities, forces, masses, dt, batch_idx=batch_idx
            )

        wp.synchronize_device(device)
        pos_np = positions.numpy()
        vel_np = velocities.numpy()

        for sys_id in range(num_systems):
            start = sys_id * atoms_per_system
            end = (sys_id + 1) * atoms_per_system
            ke = compute_kinetic_energy_np(vel_np[start:end], masses_np[start:end])
            pe = compute_harmonic_energy(pos_np[start:end], spring_k)
            final_energy = ke + pe

            drift = (
                abs(final_energy - initial_energies[sys_id]) / initial_energies[sys_id]
            )
            assert drift < 0.02, f"System {sys_id} energy drift too large: {drift}"


# ==============================================================================
# Physics and Correctness Tests
# ==============================================================================


class TestVelocityVerletPhysics:
    """Test mathematical correctness and physics behavior."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_position_update_formula(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test that position update follows: r' = r + v*dt + 0.5*a*dt^2."""
        dt_val = 0.01

        pos = np.array([[1.0, 2.0, 3.0]], dtype=np_dtype)
        vel = np.array([[0.1, 0.2, 0.3]], dtype=np_dtype)
        force = np.array([[1.0, 2.0, 3.0]], dtype=np_dtype)
        mass = np.array([2.0], dtype=np_dtype)

        positions = wp.array(pos, dtype=dtype_vec, device=device)
        velocities = wp.array(vel, dtype=dtype_vec, device=device)
        forces = wp.array(force, dtype=dtype_vec, device=device)
        masses = wp.array(mass, dtype=dtype_scalar, device=device)
        dt = wp.array([dt_val], dtype=dtype_scalar, device=device)

        velocity_verlet_position_update(positions, velocities, forces, masses, dt)
        wp.synchronize_device(device)

        acc = force / mass[:, np.newaxis]
        expected_pos = pos + vel * dt_val + 0.5 * acc * dt_val * dt_val

        result_pos = positions.numpy()
        np.testing.assert_allclose(result_pos, expected_pos, rtol=1e-5)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_velocity_half_step(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test that velocity is updated to half-step after position_update."""
        dt_val = 0.01

        pos = np.array([[0.0, 0.0, 0.0]], dtype=np_dtype)
        vel = np.array([[1.0, 0.0, 0.0]], dtype=np_dtype)
        force = np.array([[2.0, 0.0, 0.0]], dtype=np_dtype)
        mass = np.array([1.0], dtype=np_dtype)

        positions = wp.array(pos, dtype=dtype_vec, device=device)
        velocities = wp.array(vel, dtype=dtype_vec, device=device)
        forces = wp.array(force, dtype=dtype_vec, device=device)
        masses = wp.array(mass, dtype=dtype_scalar, device=device)
        dt = wp.array([dt_val], dtype=dtype_scalar, device=device)

        velocity_verlet_position_update(positions, velocities, forces, masses, dt)
        wp.synchronize_device(device)

        acc = force / mass[:, np.newaxis]
        expected_vel_half = vel + 0.5 * acc * dt_val

        result_vel = velocities.numpy()
        np.testing.assert_allclose(result_vel, expected_vel_half, rtol=1e-5)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_harmonic_oscillator_energy_conservation(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test energy conservation for a harmonic oscillator system."""
        num_atoms = 100
        spring_k = 1.0
        dt_val = 0.01
        num_steps = 1000

        np.random.seed(42)

        initial_pos = np.random.randn(num_atoms, 3).astype(np_dtype) * 0.5
        initial_vel = np.random.randn(num_atoms, 3).astype(np_dtype) * 0.1
        masses_np = np.ones(num_atoms, dtype=np_dtype)

        positions = wp.array(initial_pos.copy(), dtype=dtype_vec, device=device)
        velocities = wp.array(initial_vel.copy(), dtype=dtype_vec, device=device)
        forces = wp.zeros(num_atoms, dtype=dtype_vec, device=device)
        masses = wp.array(masses_np, dtype=dtype_scalar, device=device)
        dt = wp.array([dt_val], dtype=dtype_scalar, device=device)

        compute_harmonic_forces(positions, forces, spring_k)

        wp.synchronize_device(device)
        pos_np = positions.numpy()
        vel_np = velocities.numpy()
        initial_ke = compute_kinetic_energy_np(vel_np, masses_np)
        initial_pe = compute_harmonic_energy(pos_np, spring_k)
        initial_total = initial_ke + initial_pe

        energies = []
        for step in range(num_steps):
            velocity_verlet_position_update(positions, velocities, forces, masses, dt)
            compute_harmonic_forces(positions, forces, spring_k)
            velocity_verlet_velocity_finalize(velocities, forces, masses, dt)

            if step % 100 == 0:
                wp.synchronize_device(device)
                pos_np = positions.numpy()
                vel_np = velocities.numpy()
                ke = compute_kinetic_energy_np(vel_np, masses_np)
                pe = compute_harmonic_energy(pos_np, spring_k)
                energies.append(ke + pe)

        energies = np.array(energies)
        energy_drift = (energies[-1] - initial_total) / initial_total
        energy_fluctuation = np.std(energies) / np.mean(energies)

        assert abs(energy_drift) < 0.01, f"Energy drift too large: {energy_drift}"
        assert energy_fluctuation < 0.01, (
            f"Energy fluctuation too large: {energy_fluctuation}"
        )

    @pytest.mark.parametrize("device", DEVICES)
    def test_free_particle_linear_motion(self, device):
        """Test that free particle moves in straight line with constant velocity."""
        dt_val = 0.01
        num_steps = 100

        pos = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        vel = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        force = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        mass = np.array([1.0], dtype=np.float32)

        positions = wp.array(pos, dtype=wp.vec3f, device=device)
        velocities = wp.array(vel, dtype=wp.vec3f, device=device)
        forces = wp.array(force, dtype=wp.vec3f, device=device)
        masses = wp.array(mass, dtype=wp.float32, device=device)
        dt = wp.array([dt_val], dtype=wp.float32, device=device)

        for _ in range(num_steps):
            velocity_verlet_position_update(positions, velocities, forces, masses, dt)
            velocity_verlet_velocity_finalize(velocities, forces, masses, dt)

        wp.synchronize_device(device)

        expected_pos = np.array([[1.0, 0.0, 0.0]])
        result_pos = positions.numpy()

        np.testing.assert_allclose(result_pos, expected_pos, rtol=1e-4)

        result_vel = velocities.numpy()
        np.testing.assert_allclose(result_vel, vel, rtol=1e-5)

    @pytest.mark.parametrize("device", DEVICES)
    def test_single_harmonic_oscillator_period(self, device):
        """Test that a single harmonic oscillator has correct period."""
        dt_val = 0.001
        spring_k = 1.0

        pos = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        vel = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        mass = np.array([1.0], dtype=np.float32)

        positions = wp.array(pos, dtype=wp.vec3f, device=device)
        velocities = wp.array(vel, dtype=wp.vec3f, device=device)
        forces = wp.zeros(1, dtype=wp.vec3f, device=device)
        masses = wp.array(mass, dtype=wp.float32, device=device)
        dt = wp.array([dt_val], dtype=wp.float32, device=device)

        compute_harmonic_forces(positions, forces, spring_k)

        expected_period = 2 * np.pi
        num_steps = int(expected_period / dt_val)

        for _ in range(num_steps):
            velocity_verlet_position_update(positions, velocities, forces, masses, dt)
            compute_harmonic_forces(positions, forces, spring_k)
            velocity_verlet_velocity_finalize(velocities, forces, masses, dt)

        wp.synchronize_device(device)

        result_pos = positions.numpy()
        np.testing.assert_allclose(result_pos[0, 0], 1.0, atol=0.05)

    @pytest.mark.parametrize("device", DEVICES)
    def test_morse_dimer_energy_conservation(self, device):
        """Test energy conservation for a Morse potential dimer."""
        D_e = 1.0
        a = 2.0
        r_e = 1.5

        dt_val = 0.001
        num_steps = 5000

        initial_r = 1.8
        initial_pos = np.array(
            [
                [0.0, 0.0, 0.0],
                [initial_r, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        initial_vel = np.array(
            [
                [-0.3, 0.0, 0.0],
                [0.3, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        masses_np = np.array([1.0, 1.0], dtype=np.float32)
        num_atoms = 2

        positions = wp.array(initial_pos.copy(), dtype=wp.vec3f, device=device)
        velocities = wp.array(initial_vel.copy(), dtype=wp.vec3f, device=device)
        forces = wp.zeros(num_atoms, dtype=wp.vec3f, device=device)
        masses = wp.array(masses_np, dtype=wp.float32, device=device)
        dt = wp.array([dt_val], dtype=wp.float32, device=device)

        wp.synchronize_device(device)
        pos_np = positions.numpy()
        forces_np, pe = compute_morse_forces_and_energy(pos_np, D_e, a, r_e)
        forces = wp.array(forces_np.astype(np.float32), dtype=wp.vec3f, device=device)

        vel_np = velocities.numpy()
        initial_ke = compute_kinetic_energy_np(vel_np, masses_np)
        initial_total = initial_ke + pe

        assert initial_total > 0.01, (
            f"Initial energy should be significant: {initial_total}"
        )

        energies = []
        bond_lengths = []

        for step in range(num_steps):
            velocity_verlet_position_update(positions, velocities, forces, masses, dt)

            wp.synchronize_device(device)
            pos_np = positions.numpy()
            forces_np, pe = compute_morse_forces_and_energy(pos_np, D_e, a, r_e)
            forces = wp.array(
                forces_np.astype(np.float32), dtype=wp.vec3f, device=device
            )

            velocity_verlet_velocity_finalize(velocities, forces, masses, dt)

            if step % 100 == 0:
                wp.synchronize_device(device)
                pos_np = positions.numpy()
                vel_np = velocities.numpy()
                ke = compute_kinetic_energy_np(vel_np, masses_np)
                _, pe = compute_morse_forces_and_energy(pos_np, D_e, a, r_e)
                energies.append(ke + pe)

                r = np.linalg.norm(pos_np[1] - pos_np[0])
                bond_lengths.append(r)

        energies = np.array(energies)
        bond_lengths = np.array(bond_lengths)

        energy_drift = (energies[-1] - initial_total) / initial_total
        energy_fluctuation = np.std(energies) / np.mean(energies)

        assert abs(energy_drift) < 0.01, f"Energy drift too large: {energy_drift}"
        assert energy_fluctuation < 0.02, (
            f"Energy fluctuation too large: {energy_fluctuation}"
        )
        assert bond_lengths.min() < r_e
        assert bond_lengths.max() > r_e


# ==============================================================================
# Atom Pointer (CSR) Batch Mode Tests
# ==============================================================================


class TestVelocityVerletAtomPtr:
    """Test atom_ptr batch mode functionality for velocity Verlet integrator."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_atom_ptr_position_update_runs(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test that velocity_verlet_position_update executes with atom_ptr."""
        # 3 systems with different sizes: 10, 25, 15 atoms
        atom_counts = [10, 25, 15]
        total_atoms = sum(atom_counts)
        np.random.seed(42)

        # Create CSR-style atom_ptr: [0, 10, 35, 50]
        atom_ptr_np = np.concatenate([[0], np.cumsum(atom_counts)]).astype(np.int32)
        atom_ptr = wp.array(atom_ptr_np, dtype=wp.int32, device=device)

        positions = wp.array(
            np.random.randn(total_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        velocities = wp.array(
            np.random.randn(total_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        forces = wp.array(
            np.random.randn(total_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(total_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        # Different timesteps per system
        dt = wp.array([0.001, 0.002, 0.0015], dtype=dtype_scalar, device=device)

        velocity_verlet_position_update(
            positions, velocities, forces, masses, dt, atom_ptr=atom_ptr
        )
        wp.synchronize_device(device)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_atom_ptr_velocity_finalize_runs(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test that velocity_verlet_velocity_finalize executes with atom_ptr."""
        atom_counts = [10, 25, 15]
        total_atoms = sum(atom_counts)
        np.random.seed(42)

        atom_ptr_np = np.concatenate([[0], np.cumsum(atom_counts)]).astype(np.int32)
        atom_ptr = wp.array(atom_ptr_np, dtype=wp.int32, device=device)

        velocities = wp.array(
            np.random.randn(total_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        forces = wp.array(
            np.random.randn(total_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(total_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        dt = wp.array([0.001, 0.002, 0.0015], dtype=dtype_scalar, device=device)

        velocity_verlet_velocity_finalize(
            velocities, forces, masses, dt, atom_ptr=atom_ptr
        )
        wp.synchronize_device(device)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_atom_ptr_position_update_out(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test non-mutating position_update with atom_ptr."""
        atom_counts = [10, 20, 10]
        total_atoms = sum(atom_counts)

        atom_ptr_np = np.concatenate([[0], np.cumsum(atom_counts)]).astype(np.int32)
        atom_ptr = wp.array(atom_ptr_np, dtype=wp.int32, device=device)

        positions = wp.array(
            np.random.randn(total_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        velocities = wp.array(
            np.random.randn(total_atoms, 3).astype(np_dtype) * 0.1,
            dtype=dtype_vec,
            device=device,
        )
        forces = wp.array(
            np.random.randn(total_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(total_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        dt = wp.array([0.001, 0.001, 0.001], dtype=dtype_scalar, device=device)

        pos_orig = positions.numpy().copy()
        vel_orig = velocities.numpy().copy()

        positions_out = wp.empty_like(positions)
        velocities_out = wp.empty_like(velocities)

        pos_out, vel_out = velocity_verlet_position_update_out(
            positions,
            velocities,
            forces,
            masses,
            dt,
            positions_out,
            velocities_out,
            atom_ptr=atom_ptr,
            device=device,
        )

        wp.synchronize_device(device)

        # Check input preserved
        np.testing.assert_array_equal(positions.numpy(), pos_orig)
        np.testing.assert_array_equal(velocities.numpy(), vel_orig)

        # Check output modified
        assert pos_out.shape[0] == total_atoms
        assert vel_out.shape[0] == total_atoms
        assert not np.allclose(pos_out.numpy(), pos_orig)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_atom_ptr_velocity_finalize_out(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test non-mutating velocity_finalize with atom_ptr."""
        atom_counts = [15, 15, 10]
        total_atoms = sum(atom_counts)

        atom_ptr_np = np.concatenate([[0], np.cumsum(atom_counts)]).astype(np.int32)
        atom_ptr = wp.array(atom_ptr_np, dtype=wp.int32, device=device)

        velocities = wp.array(
            np.random.randn(total_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        forces_new = wp.array(
            np.random.randn(total_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(total_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        dt = wp.array([0.001, 0.001, 0.001], dtype=dtype_scalar, device=device)

        vel_orig = velocities.numpy().copy()

        velocities_out = wp.empty_like(velocities)

        vel_out = velocity_verlet_velocity_finalize_out(
            velocities,
            forces_new,
            masses,
            dt,
            velocities_out,
            atom_ptr=atom_ptr,
            device=device,
        )

        wp.synchronize_device(device)

        np.testing.assert_array_equal(velocities.numpy(), vel_orig)
        assert vel_out.shape[0] == total_atoms
        assert not np.allclose(vel_out.numpy(), vel_orig)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_atom_ptr_uses_per_system_dt(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test that atom_ptr mode uses per-system timesteps correctly."""
        # 3 systems with 1 atom each for easy verification

        atom_ptr_np = np.array([0, 1, 2, 3], dtype=np.int32)
        atom_ptr = wp.array(atom_ptr_np, dtype=wp.int32, device=device)

        # Same initial conditions, different timesteps
        pos = np.array(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np_dtype
        )
        vel = np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np_dtype
        )
        force = np.array(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np_dtype
        )
        mass = np.array([1.0, 1.0, 1.0], dtype=np_dtype)

        # Different timesteps: dt[1] = 2*dt[0], dt[2] = 3*dt[0]
        dt_vals = [0.01, 0.02, 0.03]

        positions = wp.array(pos, dtype=dtype_vec, device=device)
        velocities = wp.array(vel, dtype=dtype_vec, device=device)
        forces = wp.array(force, dtype=dtype_vec, device=device)
        masses = wp.array(mass, dtype=dtype_scalar, device=device)
        dt = wp.array(dt_vals, dtype=dtype_scalar, device=device)

        velocity_verlet_position_update(
            positions, velocities, forces, masses, dt, atom_ptr=atom_ptr
        )
        wp.synchronize_device(device)

        result_pos = positions.numpy()

        # Displacement should be: 0.5 * a * dt^2 = 0.5 * 1.0 * dt^2
        displacement_0 = result_pos[0, 0] - 1.0
        displacement_1 = result_pos[1, 0] - 1.0
        displacement_2 = result_pos[2, 0] - 1.0

        # Check ratios: displacement ~ dt^2
        ratio_1_0 = displacement_1 / displacement_0
        ratio_2_0 = displacement_2 / displacement_0

        np.testing.assert_allclose(ratio_1_0, 4.0, rtol=0.01)  # (0.02/0.01)^2 = 4
        np.testing.assert_allclose(ratio_2_0, 9.0, rtol=0.01)  # (0.03/0.01)^2 = 9

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_atom_ptr_vs_batch_idx_equivalence(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test that atom_ptr and batch_idx produce identical results for same-sized systems."""
        # 3 systems, all with 20 atoms (same size)
        num_systems = 3
        atoms_per_system = 20
        total_atoms = num_systems * atoms_per_system
        dt_val = 0.01

        np.random.seed(42)
        initial_pos = np.random.randn(total_atoms, 3).astype(np_dtype) * 0.5
        initial_vel = np.random.randn(total_atoms, 3).astype(np_dtype) * 0.1
        initial_force = np.random.randn(total_atoms, 3).astype(np_dtype)
        masses_np = np.ones(total_atoms, dtype=np_dtype)

        # Setup for batch_idx mode
        positions_batch = wp.array(initial_pos.copy(), dtype=dtype_vec, device=device)
        velocities_batch = wp.array(initial_vel.copy(), dtype=dtype_vec, device=device)
        forces_batch = wp.array(initial_force.copy(), dtype=dtype_vec, device=device)
        masses_batch = wp.array(masses_np, dtype=dtype_scalar, device=device)
        dt_batch = wp.array([dt_val] * num_systems, dtype=dtype_scalar, device=device)
        batch_idx = wp.array(
            np.repeat(np.arange(num_systems), atoms_per_system).astype(np.int32),
            dtype=wp.int32,
            device=device,
        )

        # Setup for atom_ptr mode
        positions_ptr = wp.array(initial_pos.copy(), dtype=dtype_vec, device=device)
        velocities_ptr = wp.array(initial_vel.copy(), dtype=dtype_vec, device=device)
        forces_ptr = wp.array(initial_force.copy(), dtype=dtype_vec, device=device)
        masses_ptr = wp.array(masses_np, dtype=dtype_scalar, device=device)
        dt_ptr = wp.array([dt_val] * num_systems, dtype=dtype_scalar, device=device)
        atom_ptr_np = np.array([0, 20, 40, 60], dtype=np.int32)
        atom_ptr = wp.array(atom_ptr_np, dtype=wp.int32, device=device)

        # Execute with batch_idx
        velocity_verlet_position_update(
            positions_batch,
            velocities_batch,
            forces_batch,
            masses_batch,
            dt_batch,
            batch_idx=batch_idx,
        )

        # Execute with atom_ptr
        velocity_verlet_position_update(
            positions_ptr,
            velocities_ptr,
            forces_ptr,
            masses_ptr,
            dt_ptr,
            atom_ptr=atom_ptr,
        )

        wp.synchronize_device(device)

        # Results should be identical
        np.testing.assert_allclose(
            positions_batch.numpy(), positions_ptr.numpy(), rtol=1e-6
        )
        np.testing.assert_allclose(
            velocities_batch.numpy(), velocities_ptr.numpy(), rtol=1e-6
        )

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_atom_ptr_energy_conservation(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test energy conservation for multiple systems with atom_ptr."""
        # 3 systems with different sizes
        atom_counts = [20, 30, 25]
        total_atoms = sum(atom_counts)
        num_systems = len(atom_counts)
        spring_k = 1.0
        dt_val = 0.01
        num_steps = 200

        np.random.seed(42)

        atom_ptr_np = np.concatenate([[0], np.cumsum(atom_counts)]).astype(np.int32)
        atom_ptr = wp.array(atom_ptr_np, dtype=wp.int32, device=device)

        initial_pos = np.random.randn(total_atoms, 3).astype(np_dtype) * 0.5
        initial_vel = np.random.randn(total_atoms, 3).astype(np_dtype) * 0.1
        masses_np = np.ones(total_atoms, dtype=np_dtype)

        positions = wp.array(initial_pos.copy(), dtype=dtype_vec, device=device)
        velocities = wp.array(initial_vel.copy(), dtype=dtype_vec, device=device)
        forces = wp.zeros(total_atoms, dtype=dtype_vec, device=device)
        masses = wp.array(masses_np, dtype=dtype_scalar, device=device)
        dt = wp.array([dt_val] * num_systems, dtype=dtype_scalar, device=device)

        compute_harmonic_forces(positions, forces, spring_k)

        wp.synchronize_device(device)
        pos_np = positions.numpy()
        vel_np = velocities.numpy()

        # Compute initial energy for each system
        initial_energies = []
        offset = 0
        for sys_id in range(num_systems):
            n = atom_counts[sys_id]
            ke = compute_kinetic_energy_np(
                vel_np[offset : offset + n], masses_np[offset : offset + n]
            )
            pe = compute_harmonic_energy(pos_np[offset : offset + n], spring_k)
            initial_energies.append(ke + pe)
            offset += n

        # Run MD
        for step in range(num_steps):
            velocity_verlet_position_update(
                positions, velocities, forces, masses, dt, atom_ptr=atom_ptr
            )
            compute_harmonic_forces(positions, forces, spring_k)
            velocity_verlet_velocity_finalize(
                velocities, forces, masses, dt, atom_ptr=atom_ptr
            )

        wp.synchronize_device(device)
        pos_np = positions.numpy()
        vel_np = velocities.numpy()

        # Check energy conservation for each system
        offset = 0
        for sys_id in range(num_systems):
            n = atom_counts[sys_id]
            ke = compute_kinetic_energy_np(
                vel_np[offset : offset + n], masses_np[offset : offset + n]
            )
            pe = compute_harmonic_energy(pos_np[offset : offset + n], spring_k)
            final_energy = ke + pe

            drift = (
                abs(final_energy - initial_energies[sys_id]) / initial_energies[sys_id]
            )
            assert drift < 0.02, (
                f"System {sys_id} (size={n}) energy drift too large: {drift}"
            )
            offset += n

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_atom_ptr_mutual_exclusivity_position_update(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test that providing both batch_idx and atom_ptr raises ValueError for position_update."""
        total_atoms = 20

        positions = wp.array(
            np.random.randn(total_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        velocities = wp.array(
            np.random.randn(total_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        forces = wp.zeros(total_atoms, dtype=dtype_vec, device=device)
        masses = wp.ones(total_atoms, dtype=dtype_scalar, device=device)
        dt = wp.array([0.001, 0.001], dtype=dtype_scalar, device=device)

        batch_idx = wp.array([0] * 10 + [1] * 10, dtype=wp.int32, device=device)
        atom_ptr = wp.array([0, 10, 20], dtype=wp.int32, device=device)

        # Should raise ValueError
        with pytest.raises(ValueError, match="Provide batch_idx OR atom_ptr, not both"):
            velocity_verlet_position_update(
                positions,
                velocities,
                forces,
                masses,
                dt,
                batch_idx=batch_idx,
                atom_ptr=atom_ptr,
            )

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_atom_ptr_mutual_exclusivity_velocity_finalize(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test that providing both batch_idx and atom_ptr raises ValueError for velocity_finalize."""
        total_atoms = 20

        velocities = wp.array(
            np.random.randn(total_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        forces = wp.zeros(total_atoms, dtype=dtype_vec, device=device)
        masses = wp.ones(total_atoms, dtype=dtype_scalar, device=device)
        dt = wp.array([0.001, 0.001], dtype=dtype_scalar, device=device)

        batch_idx = wp.array([0] * 10 + [1] * 10, dtype=wp.int32, device=device)
        atom_ptr = wp.array([0, 10, 20], dtype=wp.int32, device=device)

        # Should raise ValueError
        with pytest.raises(ValueError, match="Provide batch_idx OR atom_ptr, not both"):
            velocity_verlet_velocity_finalize(
                velocities, forces, masses, dt, batch_idx=batch_idx, atom_ptr=atom_ptr
            )

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_atom_ptr_variable_system_sizes(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test atom_ptr with highly variable system sizes."""
        # Systems with very different sizes: 5, 50, 10, 35 atoms
        atom_counts = [5, 50, 10, 35]
        total_atoms = sum(atom_counts)
        num_systems = len(atom_counts)
        dt_val = 0.01

        np.random.seed(43)

        atom_ptr_np = np.concatenate([[0], np.cumsum(atom_counts)]).astype(np.int32)
        atom_ptr = wp.array(atom_ptr_np, dtype=wp.int32, device=device)

        positions = wp.array(
            np.random.randn(total_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        velocities = wp.array(
            np.random.randn(total_atoms, 3).astype(np_dtype) * 0.1,
            dtype=dtype_vec,
            device=device,
        )
        forces = wp.array(
            np.random.randn(total_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(total_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        dt = wp.array([dt_val] * num_systems, dtype=dtype_scalar, device=device)

        pos_orig = positions.numpy().copy()

        # Execute integration step
        velocity_verlet_position_update(
            positions, velocities, forces, masses, dt, atom_ptr=atom_ptr
        )
        wp.synchronize_device(device)

        result_pos = positions.numpy()

        # Verify all systems were updated
        offset = 0
        for sys_id in range(num_systems):
            n = atom_counts[sys_id]
            sys_pos_orig = pos_orig[offset : offset + n]
            sys_pos_result = result_pos[offset : offset + n]
            # Each system should have moved
            assert not np.allclose(sys_pos_orig, sys_pos_result), (
                f"System {sys_id} (size={n}) was not updated"
            )
            offset += n


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
