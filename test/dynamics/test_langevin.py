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
Unit tests for Langevin integrator (BAOAB scheme).

Tests cover:
- Basic API functionality (single and batched)
- Temperature equilibration
- Thermostat behavior
- Float32 and float64 support
"""

import numpy as np
import pytest
import warp as wp

from nvalchemiops.dynamics.integrators import (
    langevin_baoab_finalize,
    langevin_baoab_finalize_out,
    langevin_baoab_half_step,
    langevin_baoab_half_step_out,
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


def compute_kinetic_energy_np(velocities: np.ndarray, masses: np.ndarray) -> float:
    """Compute kinetic energy KE = 0.5 * sum(m * v^2)."""
    v_sq = np.sum(velocities**2, axis=1)
    return 0.5 * np.sum(masses * v_sq)


def compute_temperature_np(
    velocities: np.ndarray, masses: np.ndarray, n_atoms: int
) -> float:
    """Compute instantaneous temperature from kinetic energy."""
    ke = compute_kinetic_energy_np(velocities, masses)
    n_dof = 3 * n_atoms
    kB = 1.0  # In simulation units
    return 2.0 * ke / (n_dof * kB)


def compute_morse_forces(
    positions: np.ndarray, D_e: float, a: float, r_e: float
) -> np.ndarray:
    """Compute Morse potential forces for a dimer."""
    forces = np.zeros_like(positions)
    r_vec = positions[1] - positions[0]
    r = np.linalg.norm(r_vec)

    if r > 1e-10:
        r_hat = r_vec / r
        exp_term = np.exp(-a * (r - r_e))
        dVdr = 2 * D_e * a * (1 - exp_term) * exp_term
        forces[1] = -dVdr * r_hat
        forces[0] = dVdr * r_hat

    return forces


# ==============================================================================
# Single System API Tests
# ==============================================================================


class TestLangevinAPI:
    """Test single-system API functionality including non-mutating variants."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_half_step_runs(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test that langevin_baoab_half_step executes without error."""
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
        temperature = wp.array([1.0], dtype=dtype_scalar, device=device)
        friction = wp.array([1.0], dtype=dtype_scalar, device=device)
        random_seed = 12345

        langevin_baoab_half_step(
            positions,
            velocities,
            forces,
            masses,
            dt,
            temperature,
            friction,
            random_seed,
        )
        wp.synchronize_device(device)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_half_step_device_inference(
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
        forces = wp.zeros(num_atoms, dtype=dtype_vec, device=device)
        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        dt = wp.array([0.001], dtype=dtype_scalar, device=device)
        temperature = wp.array([1.0], dtype=dtype_scalar, device=device)
        friction = wp.array([0.01], dtype=dtype_scalar, device=device)

        # Call without explicit device
        langevin_baoab_half_step(
            positions,
            velocities,
            forces,
            masses,
            dt,
            temperature,
            friction,
            random_seed=42,
        )

        wp.synchronize_device(device)
        assert positions.shape[0] == num_atoms

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_half_step_out_preserves_input(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test that non-mutating langevin_baoab_half_step_out preserves input."""
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
        temperature = wp.array([1.0], dtype=dtype_scalar, device=device)
        friction = wp.array([1.0], dtype=dtype_scalar, device=device)
        random_seed = 12345

        pos_orig = positions.numpy().copy()
        vel_orig = velocities.numpy().copy()

        positions_out = wp.empty_like(positions)
        velocities_out = wp.empty_like(velocities)

        pos_out, vel_out = langevin_baoab_half_step_out(
            positions,
            velocities,
            forces,
            masses,
            dt,
            temperature,
            friction,
            random_seed,
            positions_out,
            velocities_out,
        )
        wp.synchronize_device(device)

        np.testing.assert_array_equal(positions.numpy(), pos_orig)
        np.testing.assert_array_equal(velocities.numpy(), vel_orig)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_half_step_out_with_preallocated(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test half_step_out with pre-allocated output arrays."""
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
        forces = wp.zeros(num_atoms, dtype=dtype_vec, device=device)
        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        dt = wp.array([0.001], dtype=dtype_scalar, device=device)
        temperature = wp.array([1.0], dtype=dtype_scalar, device=device)
        friction = wp.array([0.01], dtype=dtype_scalar, device=device)

        positions_out = wp.empty_like(positions)
        velocities_out = wp.empty_like(velocities)

        pos_out, vel_out = langevin_baoab_half_step_out(
            positions,
            velocities,
            forces,
            masses,
            dt,
            temperature,
            friction,
            42,
            positions_out,
            velocities_out,
            device=device,
        )

        wp.synchronize_device(device)
        assert pos_out is positions_out
        assert vel_out is velocities_out

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_finalize_runs(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test that langevin_baoab_finalize executes without error."""
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

        langevin_baoab_finalize(velocities, new_forces, masses, dt)
        wp.synchronize_device(device)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_finalize_out_preserves_input(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test non-mutating finalize for single system preserves input."""
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
        velocities_out = langevin_baoab_finalize_out(
            velocities, forces_new, masses, dt, velocities_out, device=device
        )

        np.testing.assert_array_equal(velocities.numpy(), vel_orig)
        assert velocities_out.shape[0] == num_atoms

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_finalize_out_device_inference(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test langevin_baoab_finalize_out with device inference."""
        num_atoms = 20

        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype) * 0.1,
            dtype=dtype_vec,
            device=device,
        )
        forces_new = wp.zeros(num_atoms, dtype=dtype_vec, device=device)
        masses = wp.ones(num_atoms, dtype=dtype_scalar, device=device)
        dt = wp.array([0.001], dtype=dtype_scalar, device=device)

        # Call without explicit device
        velocities_out = wp.empty_like(velocities)
        vel_out = langevin_baoab_finalize_out(
            velocities, forces_new, masses, dt, velocities_out
        )

        wp.synchronize_device(device)
        assert vel_out.shape[0] == num_atoms


# ==============================================================================
# Batched API Tests
# ==============================================================================


class TestLangevinBatched:
    """Test batched API functionality including non-mutating variants."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_batched_half_step_runs(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test that batched langevin_baoab_half_step executes correctly."""
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
        temperature = wp.array([1.0, 2.0, 3.0], dtype=dtype_scalar, device=device)
        friction = wp.array([1.0, 1.0, 1.0], dtype=dtype_scalar, device=device)
        random_seed = 12345

        batch_idx = wp.array(
            np.repeat(np.arange(num_systems), atoms_per_system).astype(np.int32),
            dtype=wp.int32,
            device=device,
        )

        langevin_baoab_half_step(
            positions,
            velocities,
            forces,
            masses,
            dt,
            temperature,
            friction,
            random_seed,
            batch_idx=batch_idx,
        )
        wp.synchronize_device(device)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_batched_half_step_out(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test langevin_baoab_half_step_out with batched mode."""
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
        forces = wp.zeros(num_atoms, dtype=dtype_vec, device=device)
        masses = wp.ones(num_atoms, dtype=dtype_scalar, device=device)
        batch_idx = wp.array([0] * 10 + [1] * 10, dtype=wp.int32, device=device)
        dt = wp.array([0.001, 0.001], dtype=dtype_scalar, device=device)
        temperature = wp.array([1.0, 1.5], dtype=dtype_scalar, device=device)
        friction = wp.array([0.01, 0.01], dtype=dtype_scalar, device=device)

        positions_out = wp.empty_like(positions)
        velocities_out = wp.empty_like(velocities)

        pos_out, vel_out = langevin_baoab_half_step_out(
            positions,
            velocities,
            forces,
            masses,
            dt,
            temperature,
            friction,
            42,
            positions_out,
            velocities_out,
            batch_idx=batch_idx,
            device=device,
        )

        wp.synchronize_device(device)
        assert pos_out.shape[0] == num_atoms
        assert vel_out.shape[0] == num_atoms

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_batched_finalize_runs(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test that batched langevin_baoab_finalize executes correctly."""
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

        langevin_baoab_finalize(velocities, new_forces, masses, dt, batch_idx=batch_idx)
        wp.synchronize_device(device)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_batched_finalize_out(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test non-mutating finalize for batched systems."""
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
        velocities_out = langevin_baoab_finalize_out(
            velocities,
            forces_new,
            masses,
            dt,
            velocities_out,
            batch_idx=batch_idx,
            device=device,
        )

        assert velocities_out.shape[0] == num_atoms

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_batched_uses_per_system_temperature(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test that batched version uses per-system temperatures correctly."""
        num_systems = 2
        atoms_per_system = 50
        total_atoms = num_systems * atoms_per_system
        dt_val = 0.01
        friction_val = 10.0
        num_steps = 2000

        target_temps = [0.5, 2.0]

        np.random.seed(42)

        initial_pos = np.zeros((total_atoms, 3), dtype=np_dtype)
        initial_vel = np.random.randn(total_atoms, 3).astype(np_dtype) * 0.1
        masses_np = np.ones(total_atoms, dtype=np_dtype)

        positions = wp.array(initial_pos, dtype=dtype_vec, device=device)
        velocities = wp.array(initial_vel.copy(), dtype=dtype_vec, device=device)
        forces = wp.zeros(total_atoms, dtype=dtype_vec, device=device)
        masses = wp.array(masses_np, dtype=dtype_scalar, device=device)
        dt = wp.array([dt_val] * num_systems, dtype=dtype_scalar, device=device)
        temperature = wp.array(target_temps, dtype=dtype_scalar, device=device)
        friction = wp.array(
            [friction_val] * num_systems, dtype=dtype_scalar, device=device
        )

        batch_idx_np = np.repeat(np.arange(num_systems), atoms_per_system).astype(
            np.int32
        )
        batch_idx = wp.array(batch_idx_np, dtype=wp.int32, device=device)

        for step in range(num_steps):
            langevin_baoab_half_step(
                positions,
                velocities,
                forces,
                masses,
                dt,
                temperature,
                friction,
                step,
                batch_idx=batch_idx,
            )
            langevin_baoab_finalize(velocities, forces, masses, dt, batch_idx=batch_idx)

        wp.synchronize_device(device)
        vel_np = velocities.numpy()

        measured_temps = []
        for sys_id in range(num_systems):
            start = sys_id * atoms_per_system
            end = (sys_id + 1) * atoms_per_system
            temp = compute_temperature_np(
                vel_np[start:end], masses_np[start:end], atoms_per_system
            )
            measured_temps.append(temp)

        assert measured_temps[1] > measured_temps[0], (
            f"Higher temp system should be hotter: {measured_temps}"
        )

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_batched_temperature_equilibration(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test that multiple systems equilibrate to their target temperatures."""
        num_systems = 3
        atoms_per_system = 50
        total_atoms = num_systems * atoms_per_system
        dt_val = 0.01
        friction_val = 5.0
        num_steps = 3000
        equilibration_steps = 1500

        target_temps = [0.5, 1.0, 2.0]

        np.random.seed(42)

        initial_vel = np.random.randn(total_atoms, 3).astype(np_dtype) * 0.1
        masses_np = np.ones(total_atoms, dtype=np_dtype)

        positions = wp.zeros(total_atoms, dtype=dtype_vec, device=device)
        velocities = wp.array(initial_vel.copy(), dtype=dtype_vec, device=device)
        forces = wp.zeros(total_atoms, dtype=dtype_vec, device=device)
        masses = wp.array(masses_np, dtype=dtype_scalar, device=device)
        dt = wp.array([dt_val] * num_systems, dtype=dtype_scalar, device=device)
        temperature = wp.array(target_temps, dtype=dtype_scalar, device=device)
        friction = wp.array(
            [friction_val] * num_systems, dtype=dtype_scalar, device=device
        )

        batch_idx_np = np.repeat(np.arange(num_systems), atoms_per_system).astype(
            np.int32
        )
        batch_idx = wp.array(batch_idx_np, dtype=wp.int32, device=device)

        temps_history = [[] for _ in range(num_systems)]

        for step in range(num_steps):
            langevin_baoab_half_step(
                positions,
                velocities,
                forces,
                masses,
                dt,
                temperature,
                friction,
                step,
                batch_idx=batch_idx,
            )
            langevin_baoab_finalize(velocities, forces, masses, dt, batch_idx=batch_idx)

            if step >= equilibration_steps and step % 50 == 0:
                wp.synchronize_device(device)
                vel_np = velocities.numpy()

                for sys_id in range(num_systems):
                    start = sys_id * atoms_per_system
                    end = (sys_id + 1) * atoms_per_system
                    temp = compute_temperature_np(
                        vel_np[start:end], masses_np[start:end], atoms_per_system
                    )
                    temps_history[sys_id].append(temp)

        for sys_id in range(num_systems):
            mean_temp = np.mean(temps_history[sys_id])
            target = target_temps[sys_id]
            assert abs(mean_temp - target) < 0.4, (
                f"System {sys_id}: mean temp {mean_temp:.3f} differs from target {target:.3f}"
            )


# ==============================================================================
# Physics and Thermostating Tests
# ==============================================================================


class TestLangevinPhysics:
    """Test physics correctness: temperature equilibration, friction effects, Morse dimer."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_temperature_equilibration_harmonic(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test that system equilibrates to target temperature in harmonic potential."""
        num_atoms = 100
        spring_k = 1.0
        target_temp = 1.0
        dt_val = 0.01
        friction_val = 1.0
        num_steps = 5000
        equilibration_steps = 2000

        np.random.seed(42)

        initial_pos = np.random.randn(num_atoms, 3).astype(np_dtype) * 0.5
        initial_vel = np.random.randn(num_atoms, 3).astype(np_dtype) * 0.1
        masses_np = np.ones(num_atoms, dtype=np_dtype)

        positions = wp.array(initial_pos, dtype=dtype_vec, device=device)
        velocities = wp.array(initial_vel.copy(), dtype=dtype_vec, device=device)
        forces = wp.zeros(num_atoms, dtype=dtype_vec, device=device)
        masses = wp.array(masses_np, dtype=dtype_scalar, device=device)
        dt = wp.array([dt_val], dtype=dtype_scalar, device=device)
        temperature = wp.array([target_temp], dtype=dtype_scalar, device=device)
        friction = wp.array([friction_val], dtype=dtype_scalar, device=device)

        compute_harmonic_forces(positions, forces, spring_k)

        temps = []
        for step in range(num_steps):
            langevin_baoab_half_step(
                positions,
                velocities,
                forces,
                masses,
                dt,
                temperature,
                friction,
                step,
            )
            compute_harmonic_forces(positions, forces, spring_k)
            langevin_baoab_finalize(velocities, forces, masses, dt)

            if step >= equilibration_steps and step % 50 == 0:
                wp.synchronize_device(device)
                vel_np = velocities.numpy()
                temp = compute_temperature_np(vel_np, masses_np, num_atoms)
                temps.append(temp)

        temps = np.array(temps)
        mean_temp = np.mean(temps)
        std_temp = np.std(temps)

        assert abs(mean_temp - target_temp) < 0.2, (
            f"Mean temperature {mean_temp:.3f} differs from target {target_temp}"
        )

        expected_std = target_temp * np.sqrt(2.0 / (3 * num_atoms))
        assert std_temp < 3 * expected_std, (
            f"Temperature fluctuation too large: {std_temp:.3f}"
        )

    @pytest.mark.parametrize("device", DEVICES)
    def test_high_friction_quick_equilibration(self, device):
        """Test that high friction leads to faster temperature equilibration."""
        num_atoms = 50
        target_temp = 1.0
        dt_val = 0.01

        np.random.seed(42)

        initial_vel = np.random.randn(num_atoms, 3).astype(np.float32) * 5.0
        masses_np = np.ones(num_atoms, dtype=np.float32)

        def run_with_friction(friction_val, n_steps=500):
            positions = wp.zeros(num_atoms, dtype=wp.vec3f, device=device)
            velocities = wp.array(initial_vel.copy(), dtype=wp.vec3f, device=device)
            forces = wp.zeros(num_atoms, dtype=wp.vec3f, device=device)
            masses = wp.array(masses_np, dtype=wp.float32, device=device)
            dt = wp.array([dt_val], dtype=wp.float32, device=device)
            temperature = wp.array([target_temp], dtype=wp.float32, device=device)
            friction = wp.array([friction_val], dtype=wp.float32, device=device)

            temps = []
            for step in range(n_steps):
                langevin_baoab_half_step(
                    positions,
                    velocities,
                    forces,
                    masses,
                    dt,
                    temperature,
                    friction,
                    step,
                )
                langevin_baoab_finalize(velocities, forces, masses, dt)

                if step % 10 == 0:
                    wp.synchronize_device(device)
                    vel_np = velocities.numpy()
                    temp = compute_temperature_np(vel_np, masses_np, num_atoms)
                    temps.append(temp)

            return np.array(temps)

        temps_low = run_with_friction(0.1)
        temps_high = run_with_friction(10.0)

        deviation_low = np.abs(temps_low - target_temp)
        deviation_high = np.abs(temps_high - target_temp)

        assert np.mean(deviation_high[-10:]) < np.mean(deviation_low[-10:]) or (
            np.mean(deviation_high[-10:]) < 0.3
        ), "High friction should equilibrate faster or be well equilibrated"

    @pytest.mark.parametrize("device", DEVICES)
    def test_different_target_temperatures(self, device):
        """Test that system equilibrates to different target temperatures."""
        num_atoms = 100
        dt_val = 0.01
        friction_val = 1.0
        num_steps = 4000
        equilibration_steps = 2000

        np.random.seed(42)

        masses_np = np.ones(num_atoms, dtype=np.float32)

        def measure_temperature(target_temp):
            positions = wp.zeros(num_atoms, dtype=wp.vec3f, device=device)
            velocities = wp.zeros(num_atoms, dtype=wp.vec3f, device=device)
            forces = wp.zeros(num_atoms, dtype=wp.vec3f, device=device)
            masses = wp.array(masses_np, dtype=wp.float32, device=device)
            dt = wp.array([dt_val], dtype=wp.float32, device=device)
            temperature = wp.array([target_temp], dtype=wp.float32, device=device)
            friction = wp.array([friction_val], dtype=wp.float32, device=device)

            temps = []
            for step in range(num_steps):
                langevin_baoab_half_step(
                    positions,
                    velocities,
                    forces,
                    masses,
                    dt,
                    temperature,
                    friction,
                    step,
                )
                langevin_baoab_finalize(velocities, forces, masses, dt)

                if step >= equilibration_steps and step % 50 == 0:
                    wp.synchronize_device(device)
                    vel_np = velocities.numpy()
                    temp = compute_temperature_np(vel_np, masses_np, num_atoms)
                    temps.append(temp)

            return np.mean(temps)

        temp_05 = measure_temperature(0.5)
        temp_10 = measure_temperature(1.0)
        temp_20 = measure_temperature(2.0)

        assert temp_05 < temp_10 < temp_20, (
            f"Temperatures should scale with target: {temp_05:.3f}, {temp_10:.3f}, {temp_20:.3f}"
        )

        np.testing.assert_allclose(temp_05, 0.5, rtol=0.3)
        np.testing.assert_allclose(temp_10, 1.0, rtol=0.3)
        np.testing.assert_allclose(temp_20, 2.0, rtol=0.3)

    @pytest.mark.parametrize("device", DEVICES)
    def test_morse_dimer_temperature_equilibration(self, device):
        """Test that Morse dimer equilibrates to target temperature without dissociating."""
        D_e = 10.0
        a = 2.0
        r_e = 1.5
        target_temp = 0.1

        dt_val = 0.001
        friction_val = 5.0
        num_steps = 10000
        equilibration_steps = 5000

        num_atoms = 2
        masses_np = np.array([1.0, 1.0], dtype=np.float32)

        initial_pos = np.array(
            [
                [0.0, 0.0, 0.0],
                [r_e, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        initial_vel = np.zeros((num_atoms, 3), dtype=np.float32)

        positions = wp.array(initial_pos.copy(), dtype=wp.vec3f, device=device)
        velocities = wp.array(initial_vel.copy(), dtype=wp.vec3f, device=device)
        forces = wp.zeros(num_atoms, dtype=wp.vec3f, device=device)
        masses = wp.array(masses_np, dtype=wp.float32, device=device)
        dt = wp.array([dt_val], dtype=wp.float32, device=device)
        temperature = wp.array([target_temp], dtype=wp.float32, device=device)
        friction = wp.array([friction_val], dtype=wp.float32, device=device)

        wp.synchronize_device(device)
        pos_np = positions.numpy()
        forces_np = compute_morse_forces(pos_np, D_e, a, r_e)
        forces = wp.array(forces_np.astype(np.float32), dtype=wp.vec3f, device=device)

        bond_lengths = []
        temps = []

        for step in range(num_steps):
            langevin_baoab_half_step(
                positions,
                velocities,
                forces,
                masses,
                dt,
                temperature,
                friction,
                step,
            )

            wp.synchronize_device(device)
            pos_np = positions.numpy()
            forces_np = compute_morse_forces(pos_np, D_e, a, r_e)
            forces = wp.array(
                forces_np.astype(np.float32), dtype=wp.vec3f, device=device
            )

            langevin_baoab_finalize(velocities, forces, masses, dt)

            if step >= equilibration_steps and step % 100 == 0:
                wp.synchronize_device(device)
                pos_np = positions.numpy()
                vel_np = velocities.numpy()

                r = np.linalg.norm(pos_np[1] - pos_np[0])
                bond_lengths.append(r)

                temp = compute_temperature_np(vel_np, masses_np, num_atoms)
                temps.append(temp)

        bond_lengths = np.array(bond_lengths)
        temps = np.array(temps)

        mean_bond_length = np.mean(bond_lengths)
        std_bond_length = np.std(bond_lengths)

        assert mean_bond_length < r_e + 0.5, (
            f"Mean bond length {mean_bond_length:.3f} too far from equilibrium {r_e}"
        )
        assert std_bond_length < 0.3, (
            f"Bond length fluctuation too large: {std_bond_length:.3f}"
        )

        max_bond_length = np.max(bond_lengths)
        assert max_bond_length < 3.0 * r_e, (
            f"Dimer dissociated: max bond length = {max_bond_length:.3f}"
        )


# ==============================================================================
# Atom Pointer (CSR) Batch Mode Tests
# ==============================================================================


class TestLangevinAtomPtr:
    """Test atom_ptr batch mode functionality for Langevin integrator."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_atom_ptr_half_step_runs(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test that langevin_baoab_half_step executes with atom_ptr."""
        # 3 systems with different sizes: 10, 25, 15 atoms
        atom_counts = [10, 25, 15]
        total_atoms = sum(atom_counts)
        np.random.seed(42)

        # Create CSR-style atom_ptr
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
        dt = wp.array([0.001, 0.002, 0.0015], dtype=dtype_scalar, device=device)
        temperature = wp.array([1.0, 1.5, 2.0], dtype=dtype_scalar, device=device)
        friction = wp.array([1.0, 1.5, 0.5], dtype=dtype_scalar, device=device)

        langevin_baoab_half_step(
            positions,
            velocities,
            forces,
            masses,
            dt,
            temperature,
            friction,
            random_seed=42,
            atom_ptr=atom_ptr,
        )
        wp.synchronize_device(device)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_atom_ptr_finalize_runs(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test that langevin_baoab_finalize executes with atom_ptr."""
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

        langevin_baoab_finalize(velocities, forces, masses, dt, atom_ptr=atom_ptr)
        wp.synchronize_device(device)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_atom_ptr_half_step_out(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test non-mutating half_step with atom_ptr."""
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
        dt = wp.array([0.001, 0.001, 0.001], dtype=dtype_scalar, device=device)
        temperature = wp.array([1.0, 1.0, 1.0], dtype=dtype_scalar, device=device)
        friction = wp.array([1.0, 1.0, 1.0], dtype=dtype_scalar, device=device)

        pos_orig = positions.numpy().copy()
        vel_orig = velocities.numpy().copy()

        positions_out = wp.empty_like(positions)
        velocities_out = wp.empty_like(velocities)

        pos_out, vel_out = langevin_baoab_half_step_out(
            positions,
            velocities,
            forces,
            masses,
            dt,
            temperature,
            friction,
            42,
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

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_atom_ptr_finalize_out(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test non-mutating finalize with atom_ptr."""
        atom_counts = [15, 15, 10]
        total_atoms = sum(atom_counts)

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
        dt = wp.array([0.001, 0.001, 0.001], dtype=dtype_scalar, device=device)

        vel_orig = velocities.numpy().copy()

        velocities_out = wp.empty_like(velocities)
        vel_out = langevin_baoab_finalize_out(
            velocities,
            forces,
            masses,
            dt,
            velocities_out,
            atom_ptr=atom_ptr,
            device=device,
        )

        wp.synchronize_device(device)

        np.testing.assert_array_equal(velocities.numpy(), vel_orig)
        assert vel_out.shape[0] == total_atoms

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_atom_ptr_uses_per_system_temperature(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test that atom_ptr mode uses per-system temperatures correctly."""
        num_systems = 2
        atoms_per_system = 50
        total_atoms = num_systems * atoms_per_system
        dt_val = 0.01
        friction_val = 10.0
        num_steps = 2000

        target_temps = [0.5, 2.0]

        np.random.seed(42)

        atom_ptr = wp.array([0, 50, 100], dtype=wp.int32, device=device)

        initial_pos = np.zeros((total_atoms, 3), dtype=np_dtype)
        initial_vel = np.random.randn(total_atoms, 3).astype(np_dtype) * 0.1
        masses_np = np.ones(total_atoms, dtype=np_dtype)

        positions = wp.array(initial_pos, dtype=dtype_vec, device=device)
        velocities = wp.array(initial_vel.copy(), dtype=dtype_vec, device=device)
        forces = wp.zeros(total_atoms, dtype=dtype_vec, device=device)
        masses = wp.array(masses_np, dtype=dtype_scalar, device=device)
        dt = wp.array([dt_val] * num_systems, dtype=dtype_scalar, device=device)
        temperature = wp.array(target_temps, dtype=dtype_scalar, device=device)
        friction = wp.array(
            [friction_val] * num_systems, dtype=dtype_scalar, device=device
        )

        for step in range(num_steps):
            langevin_baoab_half_step(
                positions,
                velocities,
                forces,
                masses,
                dt,
                temperature,
                friction,
                step,
                atom_ptr=atom_ptr,
            )
            langevin_baoab_finalize(velocities, forces, masses, dt, atom_ptr=atom_ptr)

        wp.synchronize_device(device)
        vel_np = velocities.numpy()

        measured_temps = []
        for sys_id in range(num_systems):
            start = sys_id * atoms_per_system
            end = (sys_id + 1) * atoms_per_system
            temp = compute_temperature_np(
                vel_np[start:end], masses_np[start:end], atoms_per_system
            )
            measured_temps.append(temp)

        assert measured_temps[1] > measured_temps[0], (
            f"Higher temp system should be hotter: {measured_temps}"
        )

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_atom_ptr_vs_batch_idx_equivalence(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test that atom_ptr and batch_idx produce identical results for same-sized systems."""
        num_systems = 3
        atoms_per_system = 20
        total_atoms = num_systems * atoms_per_system
        dt_val = 0.01
        temp_val = 1.0
        friction_val = 1.0

        np.random.seed(42)
        initial_pos = np.random.randn(total_atoms, 3).astype(np_dtype)
        initial_vel = np.random.randn(total_atoms, 3).astype(np_dtype) * 0.1
        initial_force = np.random.randn(total_atoms, 3).astype(np_dtype)
        masses_np = np.ones(total_atoms, dtype=np_dtype)

        # Setup for batch_idx mode
        positions_batch = wp.array(initial_pos.copy(), dtype=dtype_vec, device=device)
        velocities_batch = wp.array(initial_vel.copy(), dtype=dtype_vec, device=device)
        forces_batch = wp.array(initial_force.copy(), dtype=dtype_vec, device=device)
        masses_batch = wp.array(masses_np, dtype=dtype_scalar, device=device)
        dt_batch = wp.array([dt_val] * num_systems, dtype=dtype_scalar, device=device)
        temperature_batch = wp.array(
            [temp_val] * num_systems, dtype=dtype_scalar, device=device
        )
        friction_batch = wp.array(
            [friction_val] * num_systems, dtype=dtype_scalar, device=device
        )
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
        temperature_ptr = wp.array(
            [temp_val] * num_systems, dtype=dtype_scalar, device=device
        )
        friction_ptr = wp.array(
            [friction_val] * num_systems, dtype=dtype_scalar, device=device
        )
        atom_ptr = wp.array([0, 20, 40, 60], dtype=wp.int32, device=device)

        # Execute with batch_idx
        langevin_baoab_half_step(
            positions_batch,
            velocities_batch,
            forces_batch,
            masses_batch,
            dt_batch,
            temperature_batch,
            friction_batch,
            random_seed=42,
            batch_idx=batch_idx,
        )

        # Execute with atom_ptr
        langevin_baoab_half_step(
            positions_ptr,
            velocities_ptr,
            forces_ptr,
            masses_ptr,
            dt_ptr,
            temperature_ptr,
            friction_ptr,
            random_seed=42,
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
    def test_atom_ptr_temperature_equilibration(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test that multiple systems equilibrate to target temperatures with atom_ptr."""
        atom_counts = [50, 50, 50]
        total_atoms = sum(atom_counts)
        num_systems = len(atom_counts)
        dt_val = 0.01
        friction_val = 5.0
        num_steps = 3000
        equilibration_steps = 1500

        target_temps = [0.5, 1.0, 2.0]

        np.random.seed(42)

        atom_ptr_np = np.concatenate([[0], np.cumsum(atom_counts)]).astype(np.int32)
        atom_ptr = wp.array(atom_ptr_np, dtype=wp.int32, device=device)

        initial_vel = np.random.randn(total_atoms, 3).astype(np_dtype) * 0.1
        masses_np = np.ones(total_atoms, dtype=np_dtype)

        positions = wp.zeros(total_atoms, dtype=dtype_vec, device=device)
        velocities = wp.array(initial_vel.copy(), dtype=dtype_vec, device=device)
        forces = wp.zeros(total_atoms, dtype=dtype_vec, device=device)
        masses = wp.array(masses_np, dtype=dtype_scalar, device=device)
        dt = wp.array([dt_val] * num_systems, dtype=dtype_scalar, device=device)
        temperature = wp.array(target_temps, dtype=dtype_scalar, device=device)
        friction = wp.array(
            [friction_val] * num_systems, dtype=dtype_scalar, device=device
        )

        temps_history = [[] for _ in range(num_systems)]

        for step in range(num_steps):
            langevin_baoab_half_step(
                positions,
                velocities,
                forces,
                masses,
                dt,
                temperature,
                friction,
                step,
                atom_ptr=atom_ptr,
            )
            langevin_baoab_finalize(velocities, forces, masses, dt, atom_ptr=atom_ptr)

            if step >= equilibration_steps and step % 50 == 0:
                wp.synchronize_device(device)
                vel_np = velocities.numpy()

                offset = 0
                for sys_id in range(num_systems):
                    n = atom_counts[sys_id]
                    temp = compute_temperature_np(
                        vel_np[offset : offset + n], masses_np[offset : offset + n], n
                    )
                    temps_history[sys_id].append(temp)
                    offset += n

        for sys_id in range(num_systems):
            mean_temp = np.mean(temps_history[sys_id])
            target = target_temps[sys_id]
            assert abs(mean_temp - target) < 0.4, (
                f"System {sys_id}: mean temp {mean_temp:.3f} differs from target {target:.3f}"
            )

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_atom_ptr_mutual_exclusivity_half_step(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test that providing both batch_idx and atom_ptr raises ValueError for half_step."""
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
        temperature = wp.array([1.0, 1.0], dtype=dtype_scalar, device=device)
        friction = wp.array([1.0, 1.0], dtype=dtype_scalar, device=device)

        batch_idx = wp.array([0] * 10 + [1] * 10, dtype=wp.int32, device=device)
        atom_ptr = wp.array([0, 10, 20], dtype=wp.int32, device=device)

        # Should raise ValueError
        with pytest.raises(ValueError, match="Provide batch_idx OR atom_ptr, not both"):
            langevin_baoab_half_step(
                positions,
                velocities,
                forces,
                masses,
                dt,
                temperature,
                friction,
                random_seed=42,
                batch_idx=batch_idx,
                atom_ptr=atom_ptr,
            )

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_atom_ptr_mutual_exclusivity_finalize(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test that providing both batch_idx and atom_ptr raises ValueError for finalize."""
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
            langevin_baoab_finalize(
                velocities,
                forces,
                masses,
                dt,
                batch_idx=batch_idx,
                atom_ptr=atom_ptr,
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
        temperature = wp.array([1.0] * num_systems, dtype=dtype_scalar, device=device)
        friction = wp.array([1.0] * num_systems, dtype=dtype_scalar, device=device)

        pos_orig = positions.numpy().copy()

        # Execute integration step
        langevin_baoab_half_step(
            positions,
            velocities,
            forces,
            masses,
            dt,
            temperature,
            friction,
            random_seed=42,
            atom_ptr=atom_ptr,
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
