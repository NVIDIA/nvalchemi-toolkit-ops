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
Unit tests for dynamics utility functions.

Tests cover:
- Kinetic energy computation
- Temperature calculation
- Maxwell-Boltzmann velocity initialization
- Center of mass motion removal
- Float32 and float64 support
"""

import numpy as np
import pytest
import warp as wp

from nvalchemiops.dynamics.utils import (
    compute_kinetic_energy,
    compute_temperature,
    initialize_velocities,
    initialize_velocities_out,
    remove_com_motion,
    remove_com_motion_out,
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
# Kinetic Energy Tests
# ==============================================================================


class TestKineticEnergy:
    """Test kinetic energy computation functions."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_compute_kinetic_energy_runs(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test that compute_kinetic_energy executes without error."""
        num_atoms = 100
        np.random.seed(42)

        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )

        kinetic_energy = wp.zeros(1, dtype=dtype_scalar, device=device)
        ke = compute_kinetic_energy(velocities, masses, kinetic_energy, device=device)
        wp.synchronize_device(device)

        assert ke.shape[0] == 1
        ke_val = ke.numpy()[0]
        assert ke_val > 0

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_compute_kinetic_energy_device_inference(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test kinetic energy computation with device inferred from arrays."""
        num_atoms = 50

        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )

        # Call without explicit device
        kinetic_energy = wp.zeros(1, dtype=dtype_scalar, device=device)
        ke = compute_kinetic_energy(velocities, masses, kinetic_energy)
        wp.synchronize_device(device)

        assert ke.shape[0] == 1

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_compute_kinetic_energy_preallocated(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test kinetic energy computation with pre-allocated output array."""
        num_atoms = 50

        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        kinetic_energy_out = wp.zeros(1, dtype=dtype_scalar, device=device)

        ke = compute_kinetic_energy(
            velocities, masses, kinetic_energy_out, device=device
        )

        wp.synchronize_device(device)
        assert ke is kinetic_energy_out

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_compute_kinetic_energy_value(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test that kinetic energy is computed correctly: KE = 0.5 * sum(m * v^2)."""
        vel = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np_dtype)
        mass = np.array([1.0, 1.0], dtype=np_dtype)

        velocities = wp.array(vel, dtype=dtype_vec, device=device)
        masses = wp.array(mass, dtype=dtype_scalar, device=device)

        kinetic_energy = wp.zeros(1, dtype=dtype_scalar, device=device)
        ke = compute_kinetic_energy(velocities, masses, kinetic_energy, device=device)
        wp.synchronize_device(device)

        expected_ke = 0.5 * (1.0 * 1.0 + 1.0 * 4.0)
        np.testing.assert_allclose(ke.numpy()[0], expected_ke, rtol=1e-5)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_compute_kinetic_energy_batched(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test batched kinetic energy computation."""
        num_systems = 3
        atoms_per_system = 10
        total_atoms = num_systems * atoms_per_system
        np.random.seed(42)

        velocities = wp.array(
            np.random.randn(total_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(total_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        batch_idx = wp.array(
            np.repeat(np.arange(num_systems), atoms_per_system).astype(np.int32),
            dtype=wp.int32,
            device=device,
        )

        kinetic_energy = wp.zeros(num_systems, dtype=dtype_scalar, device=device)
        ke = compute_kinetic_energy(
            velocities,
            masses,
            kinetic_energy,
            batch_idx=batch_idx,
            num_systems=num_systems,
            device=device,
        )
        wp.synchronize_device(device)

        assert ke.shape[0] == num_systems
        ke_vals = ke.numpy()
        for i in range(num_systems):
            assert ke_vals[i] > 0

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_compute_kinetic_energy_batched_preallocated(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test batched kinetic energy computation with pre-allocated output."""
        num_systems = 2
        num_atoms = 40

        velocities = wp.array(
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
        kinetic_energy_out = wp.zeros(num_systems, dtype=dtype_scalar, device=device)

        ke = compute_kinetic_energy(
            velocities,
            masses,
            kinetic_energy_out,
            batch_idx=batch_idx,
            num_systems=num_systems,
            device=device,
        )

        wp.synchronize_device(device)
        assert ke is kinetic_energy_out


# ==============================================================================
# Temperature Computation Tests
# ==============================================================================


class TestTemperatureComputation:
    """Test temperature computation functions."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_compute_temperature_runs(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test that compute_temperature executes without error."""
        num_atoms = 100

        np.random.seed(42)

        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )

        kinetic_energy = wp.zeros(1, dtype=dtype_scalar, device=device)
        compute_kinetic_energy(velocities, masses, kinetic_energy, device=device)
        temperature = wp.empty(1, dtype=dtype_scalar, device=device)
        num_atoms_per_system = wp.array([num_atoms], dtype=wp.int32, device=device)
        temp = compute_temperature(kinetic_energy, temperature, num_atoms_per_system)
        wp.synchronize_device(device)

        assert temp.shape[0] == 1
        temp_val = temp.numpy()[0]
        assert temp_val > 0

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_compute_temperature_device_inference(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test temperature computation with device inferred from arrays."""
        num_atoms = 50

        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )

        kinetic_energy = wp.zeros(1, dtype=dtype_scalar, device=device)
        compute_kinetic_energy(velocities, masses, kinetic_energy, device=device)
        temperature = wp.empty(1, dtype=dtype_scalar, device=device)
        num_atoms_per_system = wp.array([num_atoms], dtype=wp.int32, device=device)
        # Call without explicit device
        temp = compute_temperature(kinetic_energy, temperature, num_atoms_per_system)
        wp.synchronize_device(device)

        assert temp.shape[0] == 1

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_compute_temperature_preallocated_ke(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test temperature computation with pre-computed kinetic energy."""
        num_atoms = 50

        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        # Pre-compute kinetic energy
        ke = wp.zeros(1, dtype=dtype_scalar, device=device)
        compute_kinetic_energy(velocities, masses, ke, device=device)

        temperature = wp.empty(1, dtype=dtype_scalar, device=device)
        num_atoms_per_system = wp.array([num_atoms], dtype=wp.int32, device=device)
        temp = compute_temperature(ke, temperature, num_atoms_per_system)

        wp.synchronize_device(device)
        assert temp.shape[0] == 1

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_compute_temperature_value(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test that temperature follows kT = 2*KE / dof."""
        num_atoms = 100
        dof = 3 * num_atoms - 3
        np.random.seed(42)

        vel = np.random.randn(num_atoms, 3).astype(np_dtype)
        mass = np.ones(num_atoms, dtype=np_dtype)

        velocities = wp.array(vel, dtype=dtype_vec, device=device)
        masses = wp.array(mass, dtype=dtype_scalar, device=device)

        kinetic_energy = wp.zeros(1, dtype=dtype_scalar, device=device)
        compute_kinetic_energy(velocities, masses, kinetic_energy, device=device)
        temperature = wp.empty(1, dtype=dtype_scalar, device=device)
        num_atoms_per_system = wp.array([num_atoms], dtype=wp.int32, device=device)
        temp = compute_temperature(kinetic_energy, temperature, num_atoms_per_system)
        wp.synchronize_device(device)

        ke = 0.5 * np.sum(mass[:, np.newaxis] * vel**2)
        expected_temp = 2.0 * ke / dof

        np.testing.assert_allclose(temp.numpy()[0], expected_temp, rtol=1e-4)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_compute_temperature_batched(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test batched temperature computation."""
        num_systems = 3
        atoms_per_system = 50
        total_atoms = num_systems * atoms_per_system
        np.random.seed(42)

        velocities = wp.array(
            np.random.randn(total_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(total_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        batch_idx = wp.array(
            np.repeat(np.arange(num_systems), atoms_per_system).astype(np.int32),
            dtype=wp.int32,
            device=device,
        )

        kinetic_energy = wp.zeros(num_systems, dtype=dtype_scalar, device=device)
        compute_kinetic_energy(
            velocities,
            masses,
            kinetic_energy,
            batch_idx=batch_idx,
            num_systems=num_systems,
            device=device,
        )
        temperature = wp.empty(num_systems, dtype=dtype_scalar, device=device)
        num_atoms_per_system = wp.array(
            [atoms_per_system] * num_systems, dtype=wp.int32, device=device
        )
        temp = compute_temperature(
            kinetic_energy,
            temperature,
            num_atoms_per_system,
        )
        wp.synchronize_device(device)

        assert temp.shape[0] == num_systems

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_compute_temperature_batched_with_ke(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test batched temperature computation with pre-computed kinetic energy."""
        num_systems = 2
        atoms_per_system = 20
        total_atoms = num_systems * atoms_per_system

        velocities = wp.array(
            np.random.randn(total_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(total_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        batch_idx = wp.array(
            np.array([0] * atoms_per_system + [1] * atoms_per_system, dtype=np.int32),
            dtype=wp.int32,
            device=device,
        )
        # Pre-compute kinetic energy
        ke = wp.zeros(num_systems, dtype=dtype_scalar, device=device)
        compute_kinetic_energy(
            velocities,
            masses,
            ke,
            batch_idx=batch_idx,
            num_systems=num_systems,
            device=device,
        )

        temperature = wp.empty(num_systems, dtype=dtype_scalar, device=device)
        num_atoms_per_system = wp.array(
            [atoms_per_system] * num_systems, dtype=wp.int32, device=device
        )
        temp = compute_temperature(
            ke,
            temperature,
            num_atoms_per_system,
        )

        wp.synchronize_device(device)
        assert temp.shape[0] == num_systems


# ==============================================================================
# Velocity Initialization Tests
# ==============================================================================


class TestVelocityInitialization:
    """Test Maxwell-Boltzmann velocity initialization."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_initialize_velocities_runs(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test that initialize_velocities executes without error."""
        num_atoms = 100

        velocities = wp.empty(num_atoms, dtype=dtype_vec, device=device)
        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        temperature = wp.array([1.0], dtype=dtype_scalar, device=device)
        total_momentum = wp.zeros(1, dtype=dtype_vec, device=device)
        total_mass = wp.zeros(1, dtype=dtype_scalar, device=device)
        com_velocities = wp.empty(1, dtype=dtype_vec, device=device)

        initialize_velocities(
            velocities,
            masses,
            temperature,
            total_momentum,
            total_mass,
            com_velocities,
            random_seed=42,
            device=device,
        )
        wp.synchronize_device(device)

        vel_np = velocities.numpy()
        assert not np.allclose(vel_np, 0.0)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_initialize_velocities_device_inference(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test velocity initialization with device inferred from arrays."""
        num_atoms = 50

        velocities = wp.empty(num_atoms, dtype=dtype_vec, device=device)
        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        temperature = wp.array([1.0], dtype=dtype_scalar, device=device)
        total_momentum = wp.zeros(1, dtype=dtype_vec, device=device)
        total_mass = wp.zeros(1, dtype=dtype_scalar, device=device)
        com_velocities = wp.empty(1, dtype=dtype_vec, device=device)

        # Call without explicit device
        initialize_velocities(
            velocities,
            masses,
            temperature,
            total_momentum,
            total_mass,
            com_velocities,
            random_seed=42,
        )
        wp.synchronize_device(device)

        vel_np = velocities.numpy()
        assert not np.allclose(vel_np, 0.0)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_initialize_velocities_out_runs(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test non-mutating velocity initialization."""
        num_atoms = 100

        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        temperature = wp.array([1.0], dtype=dtype_scalar, device=device)
        velocities_out = wp.empty(num_atoms, dtype=dtype_vec, device=device)
        total_momentum = wp.zeros(1, dtype=dtype_vec, device=device)
        total_mass = wp.zeros(1, dtype=dtype_scalar, device=device)
        com_velocities = wp.empty(1, dtype=dtype_vec, device=device)

        velocities = initialize_velocities_out(
            masses,
            temperature,
            velocities_out,
            total_momentum,
            total_mass,
            com_velocities,
            random_seed=42,
            device=device,
        )
        wp.synchronize_device(device)

        assert velocities.shape[0] == num_atoms
        vel_np = velocities.numpy()
        assert not np.allclose(vel_np, 0.0)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_initialize_velocities_out_preallocated(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test velocity initialization with pre-allocated output (COM removal disabled)."""
        num_atoms = 50

        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        temperature = wp.array([1.0], dtype=dtype_scalar, device=device)
        velocities_out = wp.empty(num_atoms, dtype=dtype_vec, device=device)
        total_momentum = wp.zeros(1, dtype=dtype_vec, device=device)
        total_mass = wp.zeros(1, dtype=dtype_scalar, device=device)
        com_velocities = wp.empty(1, dtype=dtype_vec, device=device)

        # Disable COM removal to get the same array back
        vel = initialize_velocities_out(
            masses,
            temperature,
            velocities_out,
            total_momentum,
            total_mass,
            com_velocities,
            random_seed=42,
            remove_com=False,
            device=device,
        )

        wp.synchronize_device(device)
        assert vel is velocities_out
        assert not np.allclose(vel.numpy(), 0.0)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_initialize_velocities_temperature(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test that initialized velocities give correct temperature."""
        num_atoms = 10000
        target_temp = 1.0

        velocities = wp.empty(num_atoms, dtype=dtype_vec, device=device)
        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        temperature = wp.array([target_temp], dtype=dtype_scalar, device=device)
        total_momentum = wp.zeros(1, dtype=dtype_vec, device=device)
        total_mass = wp.zeros(1, dtype=dtype_scalar, device=device)
        com_velocities = wp.empty(1, dtype=dtype_vec, device=device)

        initialize_velocities(
            velocities,
            masses,
            temperature,
            total_momentum,
            total_mass,
            com_velocities,
            random_seed=42,
            device=device,
        )

        kinetic_energy = wp.zeros(1, dtype=dtype_scalar, device=device)
        compute_kinetic_energy(velocities, masses, kinetic_energy, device=device)
        measured_temp_arr = wp.empty(1, dtype=dtype_scalar, device=device)
        num_atoms_per_system = wp.array([num_atoms], dtype=wp.int32, device=device)
        measured_temp = compute_temperature(
            kinetic_energy, measured_temp_arr, num_atoms_per_system
        )
        wp.synchronize_device(device)

        np.testing.assert_allclose(measured_temp.numpy()[0], target_temp, rtol=0.1)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_initialize_velocities_batched(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test batched velocity initialization."""
        num_systems = 3
        atoms_per_system = 100
        total_atoms = num_systems * atoms_per_system

        velocities = wp.empty(total_atoms, dtype=dtype_vec, device=device)
        masses = wp.array(
            np.ones(total_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        temperatures = wp.array([0.5, 1.0, 2.0], dtype=dtype_scalar, device=device)
        batch_idx = wp.array(
            np.repeat(np.arange(num_systems), atoms_per_system).astype(np.int32),
            dtype=wp.int32,
            device=device,
        )
        total_momentum = wp.zeros(num_systems, dtype=dtype_vec, device=device)
        total_mass = wp.zeros(num_systems, dtype=dtype_scalar, device=device)
        com_velocities = wp.empty(num_systems, dtype=dtype_vec, device=device)

        initialize_velocities(
            velocities,
            masses,
            temperatures,
            total_momentum,
            total_mass,
            com_velocities,
            random_seed=42,
            batch_idx=batch_idx,
            device=device,
        )
        wp.synchronize_device(device)

        vel_np = velocities.numpy()
        for sys_id in range(num_systems):
            start = sys_id * atoms_per_system
            end = (sys_id + 1) * atoms_per_system
            assert not np.allclose(vel_np[start:end], 0.0)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_initialize_velocities_out_batched(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test non-mutating batched velocity initialization."""
        num_atoms = 40
        num_systems = 2

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
        temperature = wp.array([1.0, 1.0], dtype=dtype_scalar, device=device)
        velocities_out = wp.empty(num_atoms, dtype=dtype_vec, device=device)
        total_momentum = wp.zeros(num_systems, dtype=dtype_vec, device=device)
        total_mass = wp.zeros(num_systems, dtype=dtype_scalar, device=device)
        com_velocities = wp.empty(num_systems, dtype=dtype_vec, device=device)

        vel_out = initialize_velocities_out(
            masses,
            temperature,
            velocities_out,
            total_momentum,
            total_mass,
            com_velocities,
            random_seed=42,
            batch_idx=batch_idx,
            device=device,
        )

        wp.synchronize_device(device)
        assert vel_out.shape[0] == num_atoms

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize(
        "dtype_vec,dtype_scalar,np_dtype", [(wp.vec3d, wp.float64, np.float64)]
    )
    def test_initialize_velocities_different_temps(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test batched initialization with different temperatures per system."""
        num_systems = 2
        atoms_per_system = 5000
        total_atoms = num_systems * atoms_per_system
        temps = [0.5, 2.0]
        # DOF is 3*N - 3 because COM motion is removed by default
        dof = 3 * atoms_per_system - 3

        velocities = wp.zeros(total_atoms, dtype=dtype_vec, device=device)
        masses = wp.array(
            np.ones(total_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        temperatures = wp.array(temps, dtype=dtype_scalar, device=device)
        batch_idx = wp.array(
            np.repeat(np.arange(num_systems), atoms_per_system).astype(np.int32),
            dtype=wp.int32,
            device=device,
        )

        total_momentum = wp.zeros(num_systems, dtype=dtype_vec, device=device)
        total_mass = wp.zeros(num_systems, dtype=dtype_scalar, device=device)
        com_velocities = wp.empty(num_systems, dtype=dtype_vec, device=device)

        initialize_velocities(
            velocities,
            masses,
            temperatures,
            total_momentum,
            total_mass,
            com_velocities,
            random_seed=42,
            batch_idx=batch_idx,
            device=device,
        )
        wp.synchronize()

        vel_np = velocities.numpy()
        mass_np = masses.numpy()
        for sys_id in range(num_systems):
            start = sys_id * atoms_per_system
            end = (sys_id + 1) * atoms_per_system
            print(f"System {sys_id}: start={start}, end={end}")
            sys_vel = vel_np[start:end]
            sys_mass = mass_np[start:end]
            ke = 0.5 * np.sum(sys_mass[:, np.newaxis] * sys_vel**2)
            measured_temp = 2.0 * ke / dof
            print(measured_temp, temps[sys_id])
            np.testing.assert_allclose(measured_temp, temps[sys_id], rtol=0.15)


# ==============================================================================
# COM Motion Removal Tests
# ==============================================================================


class TestCOMMotionRemoval:
    """Test center of mass motion removal functions."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_remove_com_motion_runs(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test that remove_com_motion executes without error."""
        num_atoms = 100
        np.random.seed(42)

        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        total_momentum = wp.zeros(1, dtype=dtype_vec, device=device)
        total_mass = wp.zeros(1, dtype=dtype_scalar, device=device)
        com_velocities = wp.empty(1, dtype=dtype_vec, device=device)

        remove_com_motion(
            velocities,
            masses,
            total_momentum,
            total_mass,
            com_velocities,
            device=device,
        )
        wp.synchronize_device(device)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_remove_com_motion_device_inference(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test COM removal with device inferred from arrays."""
        num_atoms = 50

        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        total_momentum = wp.zeros(1, dtype=dtype_vec, device=device)
        total_mass = wp.zeros(1, dtype=dtype_scalar, device=device)
        com_velocities = wp.empty(1, dtype=dtype_vec, device=device)

        # Call without explicit device
        remove_com_motion(
            velocities,
            masses,
            total_momentum,
            total_mass,
            com_velocities,
        )
        wp.synchronize_device(device)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_remove_com_motion_out_runs(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test non-mutating COM motion removal."""
        num_atoms = 100
        np.random.seed(42)

        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        total_momentum = wp.zeros(1, dtype=dtype_vec, device=device)
        total_mass = wp.zeros(1, dtype=dtype_scalar, device=device)
        com_velocities_scratch = wp.empty(1, dtype=dtype_vec, device=device)
        velocities_out = wp.empty(num_atoms, dtype=dtype_vec, device=device)

        vel_out = remove_com_motion_out(
            velocities,
            masses,
            total_momentum,
            total_mass,
            com_velocities_scratch,
            velocities_out,
            device=device,
        )
        wp.synchronize_device(device)

        assert vel_out.shape[0] == num_atoms

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_remove_com_motion_out_preserves_input(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test that non-mutating COM removal preserves input."""
        num_atoms = 50

        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )

        vel_orig = velocities.numpy().copy()

        total_momentum = wp.zeros(1, dtype=dtype_vec, device=device)
        total_mass = wp.zeros(1, dtype=dtype_scalar, device=device)
        com_velocities_scratch = wp.empty(1, dtype=dtype_vec, device=device)
        velocities_out = wp.empty(num_atoms, dtype=dtype_vec, device=device)

        vel_out = remove_com_motion_out(
            velocities,
            masses,
            total_momentum,
            total_mass,
            com_velocities_scratch,
            velocities_out,
            device=device,
        )
        wp.synchronize_device(device)

        np.testing.assert_array_equal(velocities.numpy(), vel_orig)
        assert not np.allclose(vel_out.numpy(), vel_orig)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_remove_com_motion_out_preallocated(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test COM removal with pre-allocated output."""
        num_atoms = 50

        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        total_momentum = wp.zeros(1, dtype=dtype_vec, device=device)
        total_mass = wp.zeros(1, dtype=dtype_scalar, device=device)
        com_velocities_scratch = wp.empty(1, dtype=dtype_vec, device=device)
        velocities_out = wp.empty_like(velocities)

        vel_out = remove_com_motion_out(
            velocities,
            masses,
            total_momentum,
            total_mass,
            com_velocities_scratch,
            velocities_out,
            device=device,
        )

        wp.synchronize_device(device)
        assert vel_out is velocities_out

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_remove_com_motion_value(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test that COM velocity becomes zero after removal."""
        num_atoms = 100
        np.random.seed(42)

        vel = np.random.randn(num_atoms, 3).astype(np_dtype)
        mass = np.ones(num_atoms, dtype=np_dtype)

        velocities = wp.array(vel, dtype=dtype_vec, device=device)
        masses = wp.array(mass, dtype=dtype_scalar, device=device)

        initial_com = np.sum(mass[:, np.newaxis] * vel, axis=0) / np.sum(mass)
        assert np.linalg.norm(initial_com) > 0.01

        total_momentum = wp.zeros(1, dtype=dtype_vec, device=device)
        total_mass_arr = wp.zeros(1, dtype=dtype_scalar, device=device)
        com_velocities = wp.empty(1, dtype=dtype_vec, device=device)

        remove_com_motion(
            velocities,
            masses,
            total_momentum,
            total_mass_arr,
            com_velocities,
            device=device,
        )
        wp.synchronize_device(device)

        vel_result = velocities.numpy()
        final_com = np.sum(mass[:, np.newaxis] * vel_result, axis=0) / np.sum(mass)
        np.testing.assert_allclose(final_com, 0.0, atol=1e-5)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_remove_com_motion_batched(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test batched COM motion removal."""
        num_systems = 3
        atoms_per_system = 50
        total_atoms = num_systems * atoms_per_system
        np.random.seed(42)

        velocities = wp.array(
            np.random.randn(total_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(total_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        batch_idx = wp.array(
            np.repeat(np.arange(num_systems), atoms_per_system).astype(np.int32),
            dtype=wp.int32,
            device=device,
        )
        total_momentum = wp.zeros(num_systems, dtype=dtype_vec, device=device)
        total_mass = wp.zeros(num_systems, dtype=dtype_scalar, device=device)
        com_velocities = wp.empty(num_systems, dtype=dtype_vec, device=device)

        remove_com_motion(
            velocities,
            masses,
            total_momentum,
            total_mass,
            com_velocities,
            batch_idx=batch_idx,
            num_systems=num_systems,
            device=device,
        )
        wp.synchronize_device(device)

        vel_result = velocities.numpy()
        mass_result = masses.numpy()
        for sys_id in range(num_systems):
            start = sys_id * atoms_per_system
            end = (sys_id + 1) * atoms_per_system
            sys_vel = vel_result[start:end]
            sys_mass = mass_result[start:end]
            sys_com = np.sum(sys_mass[:, np.newaxis] * sys_vel, axis=0) / np.sum(
                sys_mass
            )
            np.testing.assert_allclose(sys_com, 0.0, atol=1e-5)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_remove_com_motion_out_batched(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test non-mutating batched COM removal."""
        num_atoms = 40
        num_systems = 2

        velocities = wp.array(
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
        total_momentum = wp.zeros(num_systems, dtype=dtype_vec, device=device)
        total_mass = wp.zeros(num_systems, dtype=dtype_scalar, device=device)
        com_velocities_scratch = wp.empty(num_systems, dtype=dtype_vec, device=device)
        velocities_out = wp.empty(num_atoms, dtype=dtype_vec, device=device)

        vel_out = remove_com_motion_out(
            velocities,
            masses,
            total_momentum,
            total_mass,
            com_velocities_scratch,
            velocities_out,
            batch_idx=batch_idx,
            num_systems=num_systems,
            device=device,
        )

        wp.synchronize_device(device)
        assert vel_out.shape[0] == num_atoms

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_remove_com_motion_different_masses(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test COM removal with non-uniform masses."""
        vel = np.array(
            [
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np_dtype,
        )
        mass = np.array([1.0, 2.0, 3.0, 4.0], dtype=np_dtype)

        velocities = wp.array(vel, dtype=dtype_vec, device=device)
        masses = wp.array(mass, dtype=dtype_scalar, device=device)

        total_mass = np.sum(mass)
        initial_momentum = np.sum(mass[:, np.newaxis] * vel, axis=0)
        initial_com = initial_momentum / total_mass
        assert np.linalg.norm(initial_com) > 0.01

        total_momentum = wp.zeros(1, dtype=dtype_vec, device=device)
        total_mass_arr = wp.zeros(1, dtype=dtype_scalar, device=device)
        com_velocities = wp.empty(1, dtype=dtype_vec, device=device)

        remove_com_motion(
            velocities,
            masses,
            total_momentum,
            total_mass_arr,
            com_velocities,
            device=device,
        )
        wp.synchronize_device(device)

        vel_result = velocities.numpy()
        final_momentum = np.sum(mass[:, np.newaxis] * vel_result, axis=0)
        np.testing.assert_allclose(final_momentum, 0.0, atol=1e-5)


# ==============================================================================
# Atom Pointer (CSR) Batch Mode Tests
# ==============================================================================


class TestKineticEnergyAtomPtr:
    """Test atom_ptr batch mode for kinetic energy computation."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_atom_ptr_compute_kinetic_energy_runs(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test that compute_kinetic_energy executes with atom_ptr."""
        atom_counts = [10, 25, 15]
        total_atoms = sum(atom_counts)
        num_systems = len(atom_counts)
        np.random.seed(42)

        atom_ptr_np = np.concatenate([[0], np.cumsum(atom_counts)]).astype(np.int32)
        atom_ptr = wp.array(atom_ptr_np, dtype=wp.int32, device=device)

        velocities = wp.array(
            np.random.randn(total_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(total_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )

        kinetic_energy = wp.zeros(num_systems, dtype=dtype_scalar, device=device)
        ke = compute_kinetic_energy(
            velocities,
            masses,
            kinetic_energy,
            atom_ptr=atom_ptr,
            num_systems=num_systems,
            device=device,
        )
        wp.synchronize_device(device)

        assert ke.shape[0] == num_systems
        ke_vals = ke.numpy()
        for i in range(num_systems):
            assert ke_vals[i] > 0

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_atom_ptr_vs_batch_idx_equivalence_ke(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test that atom_ptr and batch_idx produce identical KE for same-sized systems."""
        num_systems = 3
        atoms_per_system = 20
        total_atoms = num_systems * atoms_per_system

        np.random.seed(42)
        initial_vel = np.random.randn(total_atoms, 3).astype(np_dtype)
        masses_np = np.ones(total_atoms, dtype=np_dtype)

        # Setup for batch_idx mode
        velocities_batch = wp.array(initial_vel.copy(), dtype=dtype_vec, device=device)
        masses_batch = wp.array(masses_np, dtype=dtype_scalar, device=device)
        batch_idx = wp.array(
            np.repeat(np.arange(num_systems), atoms_per_system).astype(np.int32),
            dtype=wp.int32,
            device=device,
        )

        # Setup for atom_ptr mode
        velocities_ptr = wp.array(initial_vel.copy(), dtype=dtype_vec, device=device)
        masses_ptr = wp.array(masses_np, dtype=dtype_scalar, device=device)
        atom_ptr = wp.array([0, 20, 40, 60], dtype=wp.int32, device=device)

        # Execute with batch_idx
        ke_batch_arr = wp.zeros(num_systems, dtype=dtype_scalar, device=device)
        ke_batch = compute_kinetic_energy(
            velocities_batch,
            masses_batch,
            ke_batch_arr,
            batch_idx=batch_idx,
            num_systems=num_systems,
            device=device,
        )

        # Execute with atom_ptr
        ke_ptr_arr = wp.zeros(num_systems, dtype=dtype_scalar, device=device)
        ke_ptr = compute_kinetic_energy(
            velocities_ptr,
            masses_ptr,
            ke_ptr_arr,
            atom_ptr=atom_ptr,
            num_systems=num_systems,
            device=device,
        )

        wp.synchronize_device(device)

        # Results should be identical
        np.testing.assert_allclose(ke_batch.numpy(), ke_ptr.numpy(), rtol=1e-6)


class TestTemperatureAtomPtr:
    """Test atom_ptr batch mode for temperature computation."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_atom_ptr_compute_temperature_runs(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test that compute_temperature executes with atom_ptr."""
        atom_counts = [50, 50, 50]
        total_atoms = sum(atom_counts)
        num_systems = len(atom_counts)
        num_atoms_per_system = wp.array(atom_counts, dtype=wp.int32, device=device)
        np.random.seed(42)

        atom_ptr_np = np.concatenate([[0], np.cumsum(atom_counts)]).astype(np.int32)
        atom_ptr = wp.array(atom_ptr_np, dtype=wp.int32, device=device)

        velocities = wp.array(
            np.random.randn(total_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(total_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )

        kinetic_energy = wp.zeros(num_systems, dtype=dtype_scalar, device=device)
        compute_kinetic_energy(
            velocities,
            masses,
            kinetic_energy,
            atom_ptr=atom_ptr,
            num_systems=num_systems,
            device=device,
        )
        temperature = wp.empty(num_systems, dtype=dtype_scalar, device=device)
        temp = compute_temperature(
            kinetic_energy,
            temperature,
            num_atoms_per_system,
        )
        wp.synchronize_device(device)

        assert temp.shape[0] == num_systems
        temp_vals = temp.numpy()
        for i in range(num_systems):
            assert temp_vals[i] > 0

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_atom_ptr_vs_batch_idx_equivalence_temp(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test that atom_ptr and batch_idx produce identical temperature for same-sized systems."""
        num_systems = 3
        atoms_per_system = 20
        total_atoms = num_systems * atoms_per_system
        num_atoms_per_system = wp.array(
            [atoms_per_system] * num_systems, dtype=wp.int32, device=device
        )

        np.random.seed(42)
        initial_vel = np.random.randn(total_atoms, 3).astype(np_dtype)
        masses_np = np.ones(total_atoms, dtype=np_dtype)

        # Setup for batch_idx mode
        velocities_batch = wp.array(initial_vel.copy(), dtype=dtype_vec, device=device)
        masses_batch = wp.array(masses_np, dtype=dtype_scalar, device=device)
        batch_idx = wp.array(
            np.repeat(np.arange(num_systems), atoms_per_system).astype(np.int32),
            dtype=wp.int32,
            device=device,
        )

        # Setup for atom_ptr mode
        velocities_ptr = wp.array(initial_vel.copy(), dtype=dtype_vec, device=device)
        masses_ptr = wp.array(masses_np, dtype=dtype_scalar, device=device)
        atom_ptr = wp.array([0, 20, 40, 60], dtype=wp.int32, device=device)

        # Execute with batch_idx
        ke_batch = wp.zeros(num_systems, dtype=dtype_scalar, device=device)
        compute_kinetic_energy(
            velocities_batch,
            masses_batch,
            ke_batch,
            batch_idx=batch_idx,
            num_systems=num_systems,
            device=device,
        )
        temp_batch_arr = wp.empty(num_systems, dtype=dtype_scalar, device=device)
        temp_batch = compute_temperature(
            ke_batch,
            temp_batch_arr,
            num_atoms_per_system,
        )

        # Execute with atom_ptr
        ke_ptr = wp.zeros(num_systems, dtype=dtype_scalar, device=device)
        compute_kinetic_energy(
            velocities_ptr,
            masses_ptr,
            ke_ptr,
            atom_ptr=atom_ptr,
            num_systems=num_systems,
            device=device,
        )
        temp_ptr_arr = wp.empty(num_systems, dtype=dtype_scalar, device=device)
        num_atoms_per_system = wp.array(
            [atoms_per_system] * num_systems, dtype=wp.int32, device=device
        )
        temp_ptr = compute_temperature(
            ke_ptr,
            temp_ptr_arr,
            num_atoms_per_system,
        )

        wp.synchronize_device(device)

        # Results should be identical
        np.testing.assert_allclose(temp_batch.numpy(), temp_ptr.numpy(), rtol=1e-6)


class TestInitializeVelocitiesAtomPtr:
    """Test atom_ptr batch mode for velocity initialization."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_atom_ptr_initialize_velocities_runs(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test that initialize_velocities executes with atom_ptr."""
        atom_counts = [50, 50, 50]
        total_atoms = sum(atom_counts)
        num_systems = len(atom_counts)

        atom_ptr_np = np.concatenate([[0], np.cumsum(atom_counts)]).astype(np.int32)
        atom_ptr = wp.array(atom_ptr_np, dtype=wp.int32, device=device)

        velocities = wp.zeros(total_atoms, dtype=dtype_vec, device=device)
        masses = wp.array(
            np.ones(total_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        temperatures = wp.array([0.5, 1.0, 2.0], dtype=dtype_scalar, device=device)

        total_momentum = wp.zeros(num_systems, dtype=dtype_vec, device=device)
        total_mass = wp.zeros(num_systems, dtype=dtype_scalar, device=device)
        com_velocities = wp.empty(num_systems, dtype=dtype_vec, device=device)

        initialize_velocities(
            velocities,
            masses,
            temperatures,
            total_momentum,
            total_mass,
            com_velocities,
            random_seed=42,
            atom_ptr=atom_ptr,
            device=device,
        )
        wp.synchronize_device(device)

        vel_np = velocities.numpy()
        offset = 0
        for sys_id in range(num_systems):
            n = atom_counts[sys_id]
            assert not np.allclose(vel_np[offset : offset + n], 0.0)
            offset += n

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_atom_ptr_initialize_velocities_out(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test non-mutating initialize_velocities_out with atom_ptr."""
        atom_counts = [30, 30, 40]
        total_atoms = sum(atom_counts)

        atom_ptr_np = np.concatenate([[0], np.cumsum(atom_counts)]).astype(np.int32)
        atom_ptr = wp.array(atom_ptr_np, dtype=wp.int32, device=device)

        masses = wp.array(
            np.ones(total_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        temperatures = wp.array([1.0, 1.5, 0.5], dtype=dtype_scalar, device=device)

        num_systems = len(atom_counts)
        velocities_out = wp.empty(total_atoms, dtype=dtype_vec, device=device)
        total_momentum = wp.zeros(num_systems, dtype=dtype_vec, device=device)
        total_mass = wp.zeros(num_systems, dtype=dtype_scalar, device=device)
        com_velocities = wp.empty(num_systems, dtype=dtype_vec, device=device)

        velocities = initialize_velocities_out(
            masses,
            temperatures,
            velocities_out,
            total_momentum,
            total_mass,
            com_velocities,
            random_seed=42,
            atom_ptr=atom_ptr,
            device=device,
        )

        wp.synchronize_device(device)
        assert velocities.shape[0] == total_atoms
        vel_np = velocities.numpy()
        assert not np.allclose(vel_np, 0.0)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_atom_ptr_variable_system_sizes_init(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test velocity initialization with highly variable system sizes using atom_ptr."""
        atom_counts = [5, 50, 10, 35]
        total_atoms = sum(atom_counts)
        num_systems = len(atom_counts)

        np.random.seed(43)

        atom_ptr_np = np.concatenate([[0], np.cumsum(atom_counts)]).astype(np.int32)
        atom_ptr = wp.array(atom_ptr_np, dtype=wp.int32, device=device)

        velocities = wp.zeros(total_atoms, dtype=dtype_vec, device=device)
        masses = wp.array(
            np.ones(total_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        temperatures = wp.array([1.0, 2.0, 1.5, 0.8], dtype=dtype_scalar, device=device)

        total_momentum = wp.zeros(num_systems, dtype=dtype_vec, device=device)
        total_mass = wp.zeros(num_systems, dtype=dtype_scalar, device=device)
        com_velocities = wp.empty(num_systems, dtype=dtype_vec, device=device)

        initialize_velocities(
            velocities,
            masses,
            temperatures,
            total_momentum,
            total_mass,
            com_velocities,
            random_seed=42,
            atom_ptr=atom_ptr,
            device=device,
        )
        wp.synchronize_device(device)

        vel_np = velocities.numpy()

        # Verify all systems have non-zero velocities
        offset = 0
        for sys_id in range(num_systems):
            n = atom_counts[sys_id]
            sys_vel = vel_np[offset : offset + n]
            assert not np.allclose(sys_vel, 0.0), (
                f"System {sys_id} (size={n}) velocities not initialized"
            )
            offset += n


class TestRemoveCOMMotionAtomPtr:
    """Test atom_ptr batch mode for COM motion removal."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_atom_ptr_remove_com_motion_runs(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test that remove_com_motion executes with atom_ptr."""
        atom_counts = [50, 50, 50]
        total_atoms = sum(atom_counts)
        num_systems = len(atom_counts)
        np.random.seed(42)

        atom_ptr_np = np.concatenate([[0], np.cumsum(atom_counts)]).astype(np.int32)
        atom_ptr = wp.array(atom_ptr_np, dtype=wp.int32, device=device)

        velocities = wp.array(
            np.random.randn(total_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(total_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )

        total_momentum = wp.zeros(num_systems, dtype=dtype_vec, device=device)
        total_mass = wp.zeros(num_systems, dtype=dtype_scalar, device=device)
        com_velocities = wp.empty(num_systems, dtype=dtype_vec, device=device)

        remove_com_motion(
            velocities,
            masses,
            total_momentum,
            total_mass,
            com_velocities,
            atom_ptr=atom_ptr,
            num_systems=num_systems,
            device=device,
        )
        wp.synchronize_device(device)

        vel_result = velocities.numpy()
        mass_result = masses.numpy()

        # Check COM is zero for each system
        offset = 0
        for sys_id in range(num_systems):
            n = atom_counts[sys_id]
            sys_vel = vel_result[offset : offset + n]
            sys_mass = mass_result[offset : offset + n]
            sys_com = np.sum(sys_mass[:, np.newaxis] * sys_vel, axis=0) / np.sum(
                sys_mass
            )
            np.testing.assert_allclose(sys_com, 0.0, atol=1e-5)
            offset += n

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_atom_ptr_remove_com_motion_out(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test non-mutating remove_com_motion_out with atom_ptr."""
        atom_counts = [30, 30, 40]
        total_atoms = sum(atom_counts)
        num_systems = len(atom_counts)

        atom_ptr_np = np.concatenate([[0], np.cumsum(atom_counts)]).astype(np.int32)
        atom_ptr = wp.array(atom_ptr_np, dtype=wp.int32, device=device)

        velocities = wp.array(
            np.random.randn(total_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(total_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )

        vel_orig = velocities.numpy().copy()

        total_momentum = wp.zeros(num_systems, dtype=dtype_vec, device=device)
        total_mass = wp.zeros(num_systems, dtype=dtype_scalar, device=device)
        com_velocities_scratch = wp.empty(num_systems, dtype=dtype_vec, device=device)
        velocities_out = wp.empty(total_atoms, dtype=dtype_vec, device=device)

        vel_out = remove_com_motion_out(
            velocities,
            masses,
            total_momentum,
            total_mass,
            com_velocities_scratch,
            velocities_out,
            atom_ptr=atom_ptr,
            num_systems=num_systems,
            device=device,
        )

        wp.synchronize_device(device)

        # Check input preserved
        np.testing.assert_array_equal(velocities.numpy(), vel_orig)

        # Check output has zero COM for each system
        vel_result = vel_out.numpy()
        mass_result = masses.numpy()
        offset = 0
        for sys_id in range(num_systems):
            n = atom_counts[sys_id]
            sys_vel = vel_result[offset : offset + n]
            sys_mass = mass_result[offset : offset + n]
            sys_com = np.sum(sys_mass[:, np.newaxis] * sys_vel, axis=0) / np.sum(
                sys_mass
            )
            np.testing.assert_allclose(sys_com, 0.0, atol=1e-5)
            offset += n

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_atom_ptr_vs_batch_idx_equivalence_com(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test that atom_ptr and batch_idx produce identical COM removal for same-sized systems."""
        num_systems = 3
        atoms_per_system = 20
        total_atoms = num_systems * atoms_per_system

        np.random.seed(42)
        initial_vel = np.random.randn(total_atoms, 3).astype(np_dtype)
        masses_np = np.ones(total_atoms, dtype=np_dtype)

        # Setup for batch_idx mode
        velocities_batch = wp.array(initial_vel.copy(), dtype=dtype_vec, device=device)
        masses_batch = wp.array(masses_np, dtype=dtype_scalar, device=device)
        batch_idx = wp.array(
            np.repeat(np.arange(num_systems), atoms_per_system).astype(np.int32),
            dtype=wp.int32,
            device=device,
        )

        # Setup for atom_ptr mode
        velocities_ptr = wp.array(initial_vel.copy(), dtype=dtype_vec, device=device)
        masses_ptr = wp.array(masses_np, dtype=dtype_scalar, device=device)
        atom_ptr = wp.array([0, 20, 40, 60], dtype=wp.int32, device=device)

        # Execute with batch_idx
        total_momentum_batch = wp.zeros(num_systems, dtype=dtype_vec, device=device)
        total_mass_batch = wp.zeros(num_systems, dtype=dtype_scalar, device=device)
        com_velocities_batch = wp.empty(num_systems, dtype=dtype_vec, device=device)

        remove_com_motion(
            velocities_batch,
            masses_batch,
            total_momentum_batch,
            total_mass_batch,
            com_velocities_batch,
            batch_idx=batch_idx,
            num_systems=num_systems,
            device=device,
        )

        # Execute with atom_ptr
        total_momentum_ptr = wp.zeros(num_systems, dtype=dtype_vec, device=device)
        total_mass_ptr = wp.zeros(num_systems, dtype=dtype_scalar, device=device)
        com_velocities_ptr = wp.empty(num_systems, dtype=dtype_vec, device=device)

        remove_com_motion(
            velocities_ptr,
            masses_ptr,
            total_momentum_ptr,
            total_mass_ptr,
            com_velocities_ptr,
            atom_ptr=atom_ptr,
            num_systems=num_systems,
            device=device,
        )

        wp.synchronize_device(device)

        # Results should be identical
        np.testing.assert_allclose(
            velocities_batch.numpy(), velocities_ptr.numpy(), rtol=1e-6
        )

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_atom_ptr_variable_system_sizes_com(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test COM removal with highly variable system sizes using atom_ptr."""
        atom_counts = [5, 50, 10, 35]
        total_atoms = sum(atom_counts)
        num_systems = len(atom_counts)

        np.random.seed(43)

        atom_ptr_np = np.concatenate([[0], np.cumsum(atom_counts)]).astype(np.int32)
        atom_ptr = wp.array(atom_ptr_np, dtype=wp.int32, device=device)

        velocities = wp.array(
            np.random.randn(total_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(total_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )

        total_momentum = wp.zeros(num_systems, dtype=dtype_vec, device=device)
        total_mass = wp.zeros(num_systems, dtype=dtype_scalar, device=device)
        com_velocities = wp.empty(num_systems, dtype=dtype_vec, device=device)

        remove_com_motion(
            velocities,
            masses,
            total_momentum,
            total_mass,
            com_velocities,
            atom_ptr=atom_ptr,
            num_systems=num_systems,
            device=device,
        )
        wp.synchronize_device(device)

        vel_result = velocities.numpy()
        mass_result = masses.numpy()

        # Verify COM is zero for each system
        offset = 0
        for sys_id in range(num_systems):
            n = atom_counts[sys_id]
            sys_vel = vel_result[offset : offset + n]
            sys_mass = mass_result[offset : offset + n]
            sys_com = np.sum(sys_mass[:, np.newaxis] * sys_vel, axis=0) / np.sum(
                sys_mass
            )
            np.testing.assert_allclose(
                sys_com,
                0.0,
                atol=1e-5,
                err_msg=(f"System {sys_id} (size={n}) COM not zero"),
            )
            offset += n


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
