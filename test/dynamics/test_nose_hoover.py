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
Unit tests for Nosé-Hoover Chain (NHC) thermostat.

Tests cover:
- Basic API functionality (single and batched)
- Chain mass computation
- Temperature equilibration
- Extended Hamiltonian conservation
- Float32 and float64 support
"""

import numpy as np
import pytest
import warp as wp

from nvalchemiops.dynamics.integrators import (
    nhc_compute_chain_energy,
    nhc_compute_masses,
    nhc_position_update,
    nhc_position_update_out,
    nhc_thermostat_chain_update,
    nhc_thermostat_chain_update_out,
    nhc_velocity_half_step,
    nhc_velocity_half_step_out,
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
# Helper Functions
# ==============================================================================


@wp.kernel
def _compute_harmonic_forces_kernel_f32(
    positions: wp.array(dtype=wp.vec3f),
    forces: wp.array(dtype=wp.vec3f),
    spring_k: wp.float32,
):
    """Compute harmonic forces F = -k * r (float32)."""
    i = wp.tid()
    pos = positions[i]
    forces[i] = wp.vec3f(-spring_k * pos[0], -spring_k * pos[1], -spring_k * pos[2])


@wp.kernel
def _compute_harmonic_forces_kernel_f64(
    positions: wp.array(dtype=wp.vec3d),
    forces: wp.array(dtype=wp.vec3d),
    spring_k: wp.float64,
):
    """Compute harmonic forces F = -k * r (float64)."""
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


def compute_temperature_np(
    velocities: np.ndarray, masses: np.ndarray, ndof: int
) -> float:
    """Compute instantaneous temperature from kinetic energy."""
    ke = compute_kinetic_energy_np(velocities, masses)
    return 2.0 * ke / ndof


def run_nhc_nvt_step(
    positions: wp.array,
    velocities: wp.array,
    forces: wp.array,
    masses: wp.array,
    eta: wp.array,
    eta_dot: wp.array,
    eta_mass: wp.array,
    target_temp: wp.array,
    dt: wp.array,
    ndof: wp.array,
    compute_forces_fn,
    nloops: int = 1,
    device: str = None,
):
    """Run one complete NVT integration step with Nosé-Hoover chain."""
    dev = device if device is not None else velocities.device
    chain_dtype = eta.dtype

    # First thermostat half-step
    nhc_thermostat_chain_update(
        velocities,
        masses,
        eta,
        eta_dot,
        eta_mass,
        target_temp,
        dt,
        ndof,
        ke2=wp.empty(1, dtype=chain_dtype, device=dev),
        total_scale=wp.ones(1, dtype=chain_dtype, device=dev),
        step_scale=wp.empty(1, dtype=chain_dtype, device=dev),
        dt_chain=wp.empty(1, dtype=chain_dtype, device=dev),
        nloops=nloops,
        device=device,
    )
    nhc_velocity_half_step(velocities, forces, masses, dt, device=device)
    nhc_position_update(positions, velocities, dt, device=device)
    compute_forces_fn()
    nhc_velocity_half_step(velocities, forces, masses, dt, device=device)

    # Second thermostat half-step
    nhc_thermostat_chain_update(
        velocities,
        masses,
        eta,
        eta_dot,
        eta_mass,
        target_temp,
        dt,
        ndof,
        ke2=wp.empty(1, dtype=chain_dtype, device=dev),
        total_scale=wp.ones(1, dtype=chain_dtype, device=dev),
        step_scale=wp.empty(1, dtype=chain_dtype, device=dev),
        dt_chain=wp.empty(1, dtype=chain_dtype, device=dev),
        nloops=nloops,
        device=device,
    )


# ==============================================================================
# Chain Mass Computation Tests
# ==============================================================================


class TestNHCMassComputation:
    """Test Nosé-Hoover chain mass computation."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_nhc_compute_masses_basic(self, device):
        """Test basic chain mass computation."""
        ndof = 297
        target_temp = 1.0
        tau = 0.1
        chain_length = 3

        masses = wp.empty(chain_length, dtype=wp.float64, device=device)
        nhc_compute_masses(ndof, target_temp, tau, chain_length, masses, device=device)

        assert len(masses) == chain_length
        expected_q0 = ndof * target_temp * tau * tau
        np.testing.assert_allclose(masses.numpy()[0], expected_q0, rtol=1e-10)

        expected_qk = target_temp * tau * tau
        for k in range(1, chain_length):
            np.testing.assert_allclose(masses.numpy()[k], expected_qk, rtol=1e-10)

    @pytest.mark.parametrize("device", DEVICES)
    def test_nhc_compute_masses_different_chain_lengths(self, device):
        """Test chain mass computation with different chain lengths."""
        ndof = 100
        target_temp = 2.0
        tau = 0.05

        for chain_length in [1, 3, 5]:
            masses = wp.empty(chain_length, dtype=wp.float64, device=device)
            nhc_compute_masses(
                ndof, target_temp, tau, chain_length, masses, device=device
            )
            assert len(masses) == chain_length


# ==============================================================================
# Single System API Tests
# ==============================================================================


class TestNHCAPI:
    """Test single-system API functionality including non-mutating variants."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_velocity_half_step_runs(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test that nhc_velocity_half_step executes without error."""
        num_atoms = 10
        np.random.seed(42)

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

        nhc_velocity_half_step(velocities, forces, masses, dt)
        wp.synchronize_device(device)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_velocity_half_step_out_preserves_input(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test that nhc_velocity_half_step_out preserves input arrays."""
        num_atoms = 10
        np.random.seed(42)

        vel_np = np.random.randn(num_atoms, 3).astype(np_dtype)
        velocities = wp.array(vel_np, dtype=dtype_vec, device=device)
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

        velocities_out = wp.empty_like(velocities)
        vel_out = nhc_velocity_half_step_out(
            velocities, forces, masses, dt, velocities_out
        )
        wp.synchronize_device(device)

        np.testing.assert_array_equal(velocities.numpy(), vel_np)
        assert not np.allclose(vel_out.numpy(), vel_np)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_position_update_runs(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test that nhc_position_update executes without error."""
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
        dt = wp.array([0.001], dtype=dtype_scalar, device=device)

        nhc_position_update(positions, velocities, dt)
        wp.synchronize_device(device)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_position_update_out_preserves_input(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test that nhc_position_update_out preserves input arrays."""
        num_atoms = 10
        np.random.seed(42)

        pos_np = np.random.randn(num_atoms, 3).astype(np_dtype)
        positions = wp.array(pos_np, dtype=dtype_vec, device=device)
        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        dt = wp.array([0.001], dtype=dtype_scalar, device=device)

        positions_out = wp.empty_like(positions)
        pos_out = nhc_position_update_out(positions, velocities, dt, positions_out)
        wp.synchronize_device(device)

        np.testing.assert_array_equal(positions.numpy(), pos_np)
        assert not np.allclose(pos_out.numpy(), pos_np)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_thermostat_chain_update_runs(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test that nhc_thermostat_chain_update executes without error."""
        num_atoms = 10
        chain_length = 3
        target_temp = 1.0
        tau = 0.1
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

        ndof = 3 * num_atoms - 3
        chain_masses = wp.empty(chain_length, dtype=wp.float64, device=device)
        nhc_compute_masses(
            ndof, target_temp, tau, chain_length, chain_masses, device=device
        )

        eta = wp.zeros(chain_length, dtype=dtype_scalar, device=device)
        eta_dot = wp.zeros(chain_length, dtype=dtype_scalar, device=device)
        eta_mass = wp.array(chain_masses, dtype=dtype_scalar, device=device)
        target_temp_arr = wp.array([target_temp], dtype=dtype_scalar, device=device)
        dt = wp.array([0.001], dtype=dtype_scalar, device=device)
        ndof_arr = wp.array([float(ndof)], dtype=dtype_scalar, device=device)

        nhc_thermostat_chain_update(
            velocities,
            masses,
            eta,
            eta_dot,
            eta_mass,
            target_temp_arr,
            dt,
            ndof_arr,
            ke2=wp.empty(1, dtype=dtype_scalar, device=device),
            total_scale=wp.ones(1, dtype=dtype_scalar, device=device),
            step_scale=wp.empty(1, dtype=dtype_scalar, device=device),
            dt_chain=wp.empty(1, dtype=dtype_scalar, device=device),
            device=device,
        )
        wp.synchronize_device(device)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_thermostat_chain_update_out_preserves_input(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test that nhc_thermostat_chain_update_out preserves input arrays."""
        num_atoms = 10
        chain_length = 3
        target_temp = 1.0
        tau = 0.1
        np.random.seed(42)

        vel_np = np.random.randn(num_atoms, 3).astype(np_dtype)
        velocities = wp.array(vel_np, dtype=dtype_vec, device=device)
        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )

        ndof = 3 * num_atoms - 3
        chain_masses = wp.empty(chain_length, dtype=wp.float64, device=device)
        nhc_compute_masses(
            ndof, target_temp, tau, chain_length, chain_masses, device=device
        )

        eta_np = np.zeros(chain_length)
        eta_dot_np = np.zeros(chain_length)
        eta = wp.array(eta_np, dtype=dtype_scalar, device=device)
        eta_dot = wp.array(eta_dot_np, dtype=dtype_scalar, device=device)
        eta_mass = wp.array(chain_masses, dtype=dtype_scalar, device=device)
        target_temp_arr = wp.array([target_temp], dtype=dtype_scalar, device=device)
        dt = wp.array([0.001], dtype=dtype_scalar, device=device)
        ndof_arr = wp.array([float(ndof)], dtype=dtype_scalar, device=device)

        velocities_out = wp.empty_like(velocities)
        eta_out_arr = wp.empty_like(eta)
        eta_dot_out_arr = wp.empty_like(eta_dot)
        vel_out, eta_out, eta_dot_out = nhc_thermostat_chain_update_out(
            velocities,
            masses,
            eta,
            eta_dot,
            eta_mass,
            target_temp_arr,
            dt,
            ndof_arr,
            ke2=wp.empty(1, dtype=dtype_scalar, device=device),
            total_scale=wp.ones(1, dtype=dtype_scalar, device=device),
            step_scale=wp.empty(1, dtype=dtype_scalar, device=device),
            dt_chain=wp.empty(1, dtype=dtype_scalar, device=device),
            velocities_out=velocities_out,
            eta_out=eta_out_arr,
            eta_dot_out=eta_dot_out_arr,
            device=device,
        )
        wp.synchronize_device(device)

        np.testing.assert_array_equal(velocities.numpy(), vel_np)
        np.testing.assert_array_equal(eta.numpy(), eta_np)
        np.testing.assert_array_equal(eta_dot.numpy(), eta_dot_np)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_compute_chain_energy_runs(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test that nhc_compute_chain_energy executes without error."""
        chain_length = 3
        target_temp = 1.0
        ndof = 297

        eta = wp.array(np.zeros(chain_length), dtype=dtype_scalar, device=device)
        eta_dot = wp.array(
            np.random.randn(chain_length) * 0.1, dtype=dtype_scalar, device=device
        )
        eta_mass = wp.array(
            [ndof * target_temp * 0.01, target_temp * 0.01, target_temp * 0.01],
            dtype=dtype_scalar,
            device=device,
        )
        target_temp_arr = wp.array([target_temp], dtype=dtype_scalar, device=device)
        ndof_arr = wp.array([float(ndof)], dtype=dtype_scalar, device=device)

        ke_chain = wp.empty(1, dtype=dtype_scalar, device=device)
        pe_chain = wp.empty(1, dtype=dtype_scalar, device=device)
        nhc_compute_chain_energy(
            eta,
            eta_dot,
            eta_mass,
            target_temp_arr,
            ndof_arr,
            ke_chain,
            pe_chain,
            device=device,
        )
        wp.synchronize_device(device)

        assert ke_chain.numpy()[0] >= 0

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_device_inference(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test that device is correctly inferred from input arrays."""
        num_atoms = 20
        chain_length = 3
        target_temp = 1.0
        tau = 0.1
        ndof = 3 * num_atoms - 3
        dt_val = 0.01

        masses_np = np.ones(num_atoms, dtype=np_dtype)
        chain_masses = wp.empty(chain_length, dtype=wp.float64, device=device)
        nhc_compute_masses(
            ndof, target_temp, tau, chain_length, chain_masses, device=device
        )

        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype) * 0.1,
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(masses_np, dtype=dtype_scalar, device=device)

        eta = wp.zeros(chain_length, dtype=dtype_scalar, device=device)
        eta_dot = wp.zeros(chain_length, dtype=dtype_scalar, device=device)
        eta_mass = wp.array(chain_masses, dtype=dtype_scalar, device=device)
        target_temp_arr = wp.array([target_temp], dtype=dtype_scalar, device=device)
        dt = wp.array([dt_val], dtype=dtype_scalar, device=device)
        ndof_arr = wp.array([float(ndof)], dtype=dtype_scalar, device=device)

        # Call without explicit device (should infer from velocities)
        nhc_thermostat_chain_update(
            velocities,
            masses,
            eta,
            eta_dot,
            eta_mass,
            target_temp_arr,
            dt,
            ndof_arr,
            ke2=wp.empty(1, dtype=dtype_scalar, device=device),
            total_scale=wp.ones(1, dtype=dtype_scalar, device=device),
            step_scale=wp.empty(1, dtype=dtype_scalar, device=device),
            dt_chain=wp.empty(1, dtype=dtype_scalar, device=device),
            nloops=1,
        )

        wp.synchronize_device(device)
        assert velocities.shape[0] == num_atoms

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_out_with_preallocated_arrays(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test _out functions with pre-allocated output arrays."""
        num_atoms = 20

        np.random.seed(42)
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
        dt = wp.array([0.01], dtype=dtype_scalar, device=device)

        positions_out = wp.empty_like(positions)
        velocities_out = wp.empty_like(velocities)

        nhc_position_update_out(positions, velocities, dt, positions_out, device=device)

        forces = wp.zeros(num_atoms, dtype=dtype_vec, device=device)
        masses = wp.ones(num_atoms, dtype=dtype_scalar, device=device)
        nhc_velocity_half_step_out(
            velocities, forces, masses, dt, velocities_out, device=device
        )

        wp.synchronize_device(device)
        assert positions_out.shape[0] == num_atoms
        assert velocities_out.shape[0] == num_atoms


# ==============================================================================
# Batched API Tests
# ==============================================================================


class TestNHCBatched:
    """Test batched API functionality including non-mutating variants."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_batched_velocity_half_step_runs(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test that batched nhc_velocity_half_step executes correctly."""
        num_systems = 3
        atoms_per_system = 5
        total_atoms = num_systems * atoms_per_system
        np.random.seed(42)

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

        nhc_velocity_half_step(velocities, forces, masses, dt, batch_idx=batch_idx)
        wp.synchronize_device(device)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_batched_velocity_half_step_out(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test nhc_velocity_half_step_out with batched mode."""
        num_atoms = 20

        np.random.seed(42)
        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype) * 0.1,
            dtype=dtype_vec,
            device=device,
        )
        forces = wp.zeros(num_atoms, dtype=dtype_vec, device=device)
        masses = wp.ones(num_atoms, dtype=dtype_scalar, device=device)
        batch_idx = wp.array([0] * 10 + [1] * 10, dtype=wp.int32, device=device)
        dt = wp.array([0.001, 0.001], dtype=dtype_scalar, device=device)

        velocities_out = wp.empty_like(velocities)
        vel_out = nhc_velocity_half_step_out(
            velocities,
            forces,
            masses,
            dt,
            velocities_out,
            batch_idx=batch_idx,
            device=device,
        )

        wp.synchronize_device(device)
        assert vel_out.shape[0] == num_atoms

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_batched_position_update_runs(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test that batched nhc_position_update executes correctly."""
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
        dt = wp.array([0.001, 0.002, 0.003], dtype=dtype_scalar, device=device)
        batch_idx = wp.array(
            np.repeat(np.arange(num_systems), atoms_per_system).astype(np.int32),
            dtype=wp.int32,
            device=device,
        )

        nhc_position_update(positions, velocities, dt, batch_idx=batch_idx)
        wp.synchronize_device(device)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_batched_position_update_out(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test nhc_position_update_out with batched mode."""
        num_atoms = 20

        np.random.seed(42)
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
        batch_idx = wp.array([0] * 10 + [1] * 10, dtype=wp.int32, device=device)
        dt = wp.array([0.001, 0.001], dtype=dtype_scalar, device=device)

        positions_out = wp.empty_like(positions)
        pos_out = nhc_position_update_out(
            positions, velocities, dt, positions_out, batch_idx=batch_idx, device=device
        )

        wp.synchronize_device(device)
        assert pos_out.shape[0] == num_atoms

    @pytest.mark.parametrize("device", DEVICES)
    def test_compute_chain_energy_batched_2d(self, device):
        """Test nhc_compute_chain_energy with batched 2D arrays."""
        num_systems = 2

        eta = wp.array(
            [[0.1, 0.05, 0.02], [0.2, 0.1, 0.05]],
            dtype=wp.float64,
            device=device,
        )
        eta_dot = wp.array(
            [[0.01, 0.005, 0.002], [0.02, 0.01, 0.005]],
            dtype=wp.float64,
            device=device,
        )
        eta_mass = wp.array(
            [[100.0, 100.0, 100.0], [100.0, 100.0, 100.0]],
            dtype=wp.float64,
            device=device,
        )
        target_temp = wp.array([1.0, 1.5], dtype=wp.float64, device=device)
        ndof = wp.array([27.0, 27.0], dtype=wp.float64, device=device)

        ke_chain = wp.empty(num_systems, dtype=wp.float64, device=device)
        pe_chain = wp.empty(num_systems, dtype=wp.float64, device=device)
        nhc_compute_chain_energy(
            eta,
            eta_dot,
            eta_mass,
            target_temp,
            ndof,
            ke_chain,
            pe_chain,
            num_systems=num_systems,
        )

        wp.synchronize_device(device)
        assert ke_chain.shape[0] == num_systems
        assert pe_chain.shape[0] == num_systems


# ==============================================================================
# Correctness and Physics Tests
# ==============================================================================


class TestNHCPhysics:
    """Test mathematical correctness and thermodynamic behavior."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_velocity_half_step_formula(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test velocity half-step follows: v += 0.5 * F/m * dt."""
        dt_val = 0.01

        vel = np.array([[1.0, 2.0, 3.0]], dtype=np_dtype)
        force = np.array([[1.0, 2.0, 3.0]], dtype=np_dtype)
        mass = np.array([2.0], dtype=np_dtype)

        velocities = wp.array(vel.copy(), dtype=dtype_vec, device=device)
        forces = wp.array(force, dtype=dtype_vec, device=device)
        masses = wp.array(mass, dtype=dtype_scalar, device=device)
        dt = wp.array([dt_val], dtype=dtype_scalar, device=device)

        nhc_velocity_half_step(velocities, forces, masses, dt)
        wp.synchronize_device(device)

        acc = force / mass[:, np.newaxis]
        expected_vel = vel + 0.5 * acc * dt_val

        np.testing.assert_allclose(velocities.numpy(), expected_vel, rtol=1e-5)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_position_update_formula(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test position update follows: r += v * dt."""
        dt_val = 0.01

        pos = np.array([[1.0, 2.0, 3.0]], dtype=np_dtype)
        vel = np.array([[0.1, 0.2, 0.3]], dtype=np_dtype)

        positions = wp.array(pos.copy(), dtype=dtype_vec, device=device)
        velocities = wp.array(vel, dtype=dtype_vec, device=device)
        dt = wp.array([dt_val], dtype=dtype_scalar, device=device)

        nhc_position_update(positions, velocities, dt)
        wp.synchronize_device(device)

        expected_pos = pos + vel * dt_val

        np.testing.assert_allclose(positions.numpy(), expected_pos, rtol=1e-5)

    @pytest.mark.parametrize("device", DEVICES)
    def test_chain_ke_formula(self, device):
        """Test chain kinetic energy follows: KE = 0.5 * sum(Q * eta_dot^2)."""
        chain_length = 3
        target_temp = 1.0
        ndof = 297

        eta_dot_vals = np.array([0.1, 0.2, 0.3])
        eta_mass_vals = np.array(
            [ndof * target_temp * 0.01, target_temp * 0.01, target_temp * 0.01]
        )

        eta = wp.zeros(chain_length, dtype=wp.float64, device=device)
        eta_dot = wp.array(eta_dot_vals, dtype=wp.float64, device=device)
        eta_mass = wp.array(eta_mass_vals, dtype=wp.float64, device=device)
        target_temp_arr = wp.array([target_temp], dtype=wp.float64, device=device)
        ndof_arr = wp.array([float(ndof)], dtype=wp.float64, device=device)

        ke_chain = wp.empty(1, dtype=wp.float64, device=device)
        pe_chain = wp.empty(1, dtype=wp.float64, device=device)
        nhc_compute_chain_energy(
            eta,
            eta_dot,
            eta_mass,
            target_temp_arr,
            ndof_arr,
            ke_chain,
            pe_chain,
            device=device,
        )
        wp.synchronize_device(device)

        expected_ke = 0.5 * np.sum(eta_mass_vals * eta_dot_vals**2)
        np.testing.assert_allclose(ke_chain.numpy()[0], expected_ke, rtol=1e-10)

    @pytest.mark.parametrize("device", DEVICES)
    def test_chain_pe_formula(self, device):
        """Test chain potential energy follows: PE = ndof*kT*eta[0] + kT*sum(eta[1:])."""
        chain_length = 3
        target_temp = 2.0
        ndof = 100

        eta_vals = np.array([0.5, 1.0, -0.5])

        eta = wp.array(eta_vals, dtype=wp.float64, device=device)
        eta_dot = wp.zeros(chain_length, dtype=wp.float64, device=device)
        eta_mass = wp.ones(chain_length, dtype=wp.float64, device=device)
        target_temp_arr = wp.array([target_temp], dtype=wp.float64, device=device)
        ndof_arr = wp.array([float(ndof)], dtype=wp.float64, device=device)

        ke_chain = wp.empty(1, dtype=wp.float64, device=device)
        pe_chain = wp.empty(1, dtype=wp.float64, device=device)
        nhc_compute_chain_energy(
            eta,
            eta_dot,
            eta_mass,
            target_temp_arr,
            ndof_arr,
            ke_chain,
            pe_chain,
            device=device,
        )
        wp.synchronize_device(device)

        expected_pe = ndof * target_temp * eta_vals[0] + target_temp * np.sum(
            eta_vals[1:]
        )
        np.testing.assert_allclose(pe_chain.numpy()[0], expected_pe, rtol=1e-10)

    @pytest.mark.parametrize("device", DEVICES)
    def test_temperature_equilibration_harmonic(self, device):
        """Test that NHC equilibrates to target temperature in harmonic potential."""
        num_atoms = 100
        spring_k = 1.0
        target_temp = 1.0
        tau = 0.1
        dt_val = 0.01
        chain_length = 3
        num_steps = 5000
        equilibration_steps = 2000

        np.random.seed(42)

        ndof = 3 * num_atoms - 3
        chain_masses = wp.empty(chain_length, dtype=wp.float64, device=device)
        nhc_compute_masses(
            ndof, target_temp, tau, chain_length, chain_masses, device=device
        )
        masses_np = np.ones(num_atoms, dtype=np.float32)

        initial_pos = np.random.randn(num_atoms, 3).astype(np.float32) * 0.5
        initial_vel = np.random.randn(num_atoms, 3).astype(np.float32) * 0.5

        positions = wp.array(initial_pos.copy(), dtype=wp.vec3f, device=device)
        velocities = wp.array(initial_vel.copy(), dtype=wp.vec3f, device=device)
        forces = wp.zeros(num_atoms, dtype=wp.vec3f, device=device)
        masses = wp.array(masses_np, dtype=wp.float32, device=device)

        eta = wp.zeros(chain_length, dtype=wp.float64, device=device)
        eta_dot = wp.zeros(chain_length, dtype=wp.float64, device=device)
        eta_mass = chain_masses
        target_temp_arr = wp.array([target_temp], dtype=wp.float64, device=device)
        dt = wp.array([dt_val], dtype=wp.float32, device=device)
        ndof_arr = wp.array([float(ndof)], dtype=wp.float64, device=device)

        def compute_forces_fn():
            compute_harmonic_forces(positions, forces, spring_k)

        compute_forces_fn()

        temps = []
        for step in range(num_steps):
            run_nhc_nvt_step(
                positions,
                velocities,
                forces,
                masses,
                eta,
                eta_dot,
                eta_mass,
                target_temp_arr,
                dt,
                ndof_arr,
                compute_forces_fn,
                nloops=3,
                device=device,
            )

            if step >= equilibration_steps and step % 50 == 0:
                wp.synchronize_device(device)
                vel_np = velocities.numpy()
                temp = compute_temperature_np(vel_np, masses_np, ndof)
                temps.append(temp)

        temps = np.array(temps)
        mean_temp = np.mean(temps)
        std_temp = np.std(temps)

        assert abs(mean_temp - target_temp) < 0.2, (
            f"Mean temperature {mean_temp:.3f} differs from target {target_temp}"
        )

        expected_std = target_temp * np.sqrt(2.0 / ndof)
        assert std_temp < 5 * expected_std, (
            f"Temperature fluctuation too large: {std_temp:.3f} vs expected ~{expected_std:.3f}"
        )

    @pytest.mark.parametrize("device", DEVICES)
    def test_extended_hamiltonian_bounded(self, device):
        """Test that extended Hamiltonian remains bounded during NHC-NVT dynamics."""
        num_atoms = 50
        spring_k = 1.0
        target_temp = 1.0
        tau = 0.1
        dt_val = 0.002
        chain_length = 3
        num_steps = 500

        np.random.seed(42)

        ndof = 3 * num_atoms - 3
        chain_masses = wp.empty(chain_length, dtype=wp.float64, device=device)
        nhc_compute_masses(
            ndof, target_temp, tau, chain_length, chain_masses, device=device
        )
        masses_np = np.ones(num_atoms, dtype=np.float32)

        initial_pos = np.random.randn(num_atoms, 3).astype(np.float32) * 0.3
        initial_vel = np.random.randn(num_atoms, 3).astype(np.float32) * np.sqrt(
            target_temp
        )

        positions = wp.array(initial_pos.copy(), dtype=wp.vec3f, device=device)
        velocities = wp.array(initial_vel.copy(), dtype=wp.vec3f, device=device)
        forces = wp.zeros(num_atoms, dtype=wp.vec3f, device=device)
        masses = wp.array(masses_np, dtype=wp.float32, device=device)

        eta = wp.zeros(chain_length, dtype=wp.float64, device=device)
        eta_dot = wp.zeros(chain_length, dtype=wp.float64, device=device)
        eta_mass = chain_masses
        target_temp_arr = wp.array([target_temp], dtype=wp.float64, device=device)
        dt = wp.array([dt_val], dtype=wp.float32, device=device)
        ndof_arr = wp.array([float(ndof)], dtype=wp.float64, device=device)

        def compute_forces_fn():
            compute_harmonic_forces(positions, forces, spring_k)

        compute_forces_fn()

        def compute_extended_hamiltonian():
            wp.synchronize_device(device)
            pos_np = positions.numpy()
            vel_np = velocities.numpy()

            ke_particles = compute_kinetic_energy_np(vel_np, masses_np)
            pe_particles = compute_harmonic_energy(pos_np, spring_k)

            ke_chain = wp.empty(1, dtype=wp.float64, device=device)
            pe_chain = wp.empty(1, dtype=wp.float64, device=device)
            nhc_compute_chain_energy(
                eta,
                eta_dot,
                eta_mass,
                target_temp_arr,
                ndof_arr,
                ke_chain,
                pe_chain,
                device=device,
            )
            wp.synchronize_device(device)

            return (
                ke_particles + pe_particles + ke_chain.numpy()[0] + pe_chain.numpy()[0]
            )

        initial_H = compute_extended_hamiltonian()
        energies = [initial_H]

        for step in range(num_steps):
            run_nhc_nvt_step(
                positions,
                velocities,
                forces,
                masses,
                eta,
                eta_dot,
                eta_mass,
                target_temp_arr,
                dt,
                ndof_arr,
                compute_forces_fn,
                nloops=3,
                device=device,
            )

            if step % 50 == 0:
                H = compute_extended_hamiltonian()
                energies.append(H)

        energies = np.array(energies)

        max_H = np.max(energies)
        min_H = np.min(energies)
        mean_H = np.mean(energies)

        assert max_H < 5 * abs(initial_H), (
            f"Energy exploded: max={max_H}, initial={initial_H}"
        )
        assert min_H > -5 * abs(initial_H), (
            f"Energy collapsed: min={min_H}, initial={initial_H}"
        )

        H_range = (max_H - min_H) / abs(mean_H)
        assert H_range < 1.0, f"Energy fluctuation range too large: {H_range}"

    @pytest.mark.parametrize("device", DEVICES)
    def test_different_target_temperatures(self, device):
        """Test that NHC equilibrates to different target temperatures."""
        num_atoms = 100
        spring_k = 1.0
        tau = 0.1
        dt_val = 0.01
        chain_length = 3
        num_steps = 3000
        equilibration_steps = 1500

        np.random.seed(42)

        ndof = 3 * num_atoms - 3
        masses_np = np.ones(num_atoms, dtype=np.float32)

        def measure_temperature(target_temp):
            chain_masses = wp.empty(chain_length, dtype=wp.float64, device=device)
            nhc_compute_masses(
                ndof, target_temp, tau, chain_length, chain_masses, device=device
            )
            initial_pos = np.random.randn(num_atoms, 3).astype(np.float32) * 0.5
            initial_vel = np.random.randn(num_atoms, 3).astype(np.float32) * 0.1

            positions = wp.array(initial_pos.copy(), dtype=wp.vec3f, device=device)
            velocities = wp.array(initial_vel.copy(), dtype=wp.vec3f, device=device)
            forces = wp.zeros(num_atoms, dtype=wp.vec3f, device=device)
            masses = wp.array(masses_np, dtype=wp.float32, device=device)

            eta = wp.zeros(chain_length, dtype=wp.float64, device=device)
            eta_dot = wp.zeros(chain_length, dtype=wp.float64, device=device)
            eta_mass = chain_masses
            target_temp_arr = wp.array([target_temp], dtype=wp.float64, device=device)
            dt = wp.array([dt_val], dtype=wp.float32, device=device)
            ndof_arr = wp.array([float(ndof)], dtype=wp.float64, device=device)

            def compute_forces_fn():
                compute_harmonic_forces(positions, forces, spring_k)

            compute_forces_fn()

            temps = []
            for step in range(num_steps):
                run_nhc_nvt_step(
                    positions,
                    velocities,
                    forces,
                    masses,
                    eta,
                    eta_dot,
                    eta_mass,
                    target_temp_arr,
                    dt,
                    ndof_arr,
                    compute_forces_fn,
                    nloops=3,
                    device=device,
                )

                if step >= equilibration_steps and step % 50 == 0:
                    wp.synchronize_device(device)
                    vel_np = velocities.numpy()
                    temp = compute_temperature_np(vel_np, masses_np, ndof)
                    temps.append(temp)

            return np.mean(temps)

        temp_05 = measure_temperature(0.5)
        temp_10 = measure_temperature(1.0)
        temp_20 = measure_temperature(2.0)

        assert temp_05 < temp_10 < temp_20, (
            f"Temperatures should scale: {temp_05:.3f}, {temp_10:.3f}, {temp_20:.3f}"
        )

        np.testing.assert_allclose(temp_05, 0.5, rtol=0.3)
        np.testing.assert_allclose(temp_10, 1.0, rtol=0.3)
        np.testing.assert_allclose(temp_20, 2.0, rtol=0.3)

    @pytest.mark.parametrize("device", DEVICES)
    def test_chain_length_one(self, device):
        """Test NHC with chain_length=1 (no chain coupling)."""
        num_atoms = 50
        chain_length = 1
        target_temp = 1.0
        tau = 0.1
        ndof = 3 * num_atoms - 3
        dt_val = 0.01

        np.random.seed(42)
        masses_np = np.ones(num_atoms, dtype=np.float32)
        chain_masses = wp.empty(chain_length, dtype=wp.float64, device=device)
        nhc_compute_masses(
            ndof, target_temp, tau, chain_length, chain_masses, device=device
        )

        positions = wp.array(
            np.random.randn(num_atoms, 3).astype(np.float32) * 0.5,
            dtype=wp.vec3f,
            device=device,
        )
        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np.float32) * 0.1,
            dtype=wp.vec3f,
            device=device,
        )
        forces = wp.zeros(num_atoms, dtype=wp.vec3f, device=device)
        masses = wp.array(masses_np, dtype=wp.float32, device=device)

        eta = wp.zeros(chain_length, dtype=wp.float32, device=device)
        eta_dot = wp.zeros(chain_length, dtype=wp.float32, device=device)
        eta_mass = wp.array(chain_masses, dtype=wp.float32, device=device)
        target_temp_arr = wp.array([target_temp], dtype=wp.float32, device=device)
        dt = wp.array([dt_val], dtype=wp.float32, device=device)
        ndof_arr = wp.array([float(ndof)], dtype=wp.float32, device=device)

        for _ in range(10):
            nhc_thermostat_chain_update(
                velocities,
                masses,
                eta,
                eta_dot,
                eta_mass,
                target_temp_arr,
                dt,
                ndof_arr,
                ke2=wp.empty(1, dtype=wp.float32, device=device),
                total_scale=wp.ones(1, dtype=wp.float32, device=device),
                step_scale=wp.empty(1, dtype=wp.float32, device=device),
                dt_chain=wp.empty(1, dtype=wp.float32, device=device),
                nloops=1,
                device=device,
            )
            nhc_velocity_half_step(velocities, forces, masses, dt, device=device)
            nhc_position_update(positions, velocities, dt, device=device)

        wp.synchronize_device(device)
        assert positions.shape[0] == num_atoms

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("nloops", [1, 3, 5, 7])
    def test_different_nloops(self, device, nloops):
        """Test NHC with different nloops values (Yoshida-Suzuki factorization)."""
        num_atoms = 30
        chain_length = 3
        target_temp = 1.0
        tau = 0.1
        ndof = 3 * num_atoms - 3
        dt_val = 0.01

        np.random.seed(42)
        masses_np = np.ones(num_atoms, dtype=np.float32)
        chain_masses = wp.empty(chain_length, dtype=wp.float64, device=device)
        nhc_compute_masses(
            ndof, target_temp, tau, chain_length, chain_masses, device=device
        )

        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np.float32) * 0.1,
            dtype=wp.vec3f,
            device=device,
        )
        masses = wp.array(masses_np, dtype=wp.float32, device=device)

        eta = wp.zeros(chain_length, dtype=wp.float32, device=device)
        eta_dot = wp.zeros(chain_length, dtype=wp.float32, device=device)
        eta_mass = wp.array(chain_masses, dtype=wp.float32, device=device)
        target_temp_arr = wp.array([target_temp], dtype=wp.float32, device=device)
        dt = wp.array([dt_val], dtype=wp.float32, device=device)
        ndof_arr = wp.array([float(ndof)], dtype=wp.float32, device=device)

        nhc_thermostat_chain_update(
            velocities,
            masses,
            eta,
            eta_dot,
            eta_mass,
            target_temp_arr,
            dt,
            ndof_arr,
            ke2=wp.empty(1, dtype=wp.float32, device=device),
            total_scale=wp.ones(1, dtype=wp.float32, device=device),
            step_scale=wp.empty(1, dtype=wp.float32, device=device),
            dt_chain=wp.empty(1, dtype=wp.float32, device=device),
            nloops=nloops,
            device=device,
        )

        wp.synchronize_device(device)
        assert velocities.shape[0] == num_atoms


# ==============================================================================
# Atom Pointer (CSR) Batch Mode Tests
# ==============================================================================


class TestNHCAtomPtr:
    """Test atom_ptr batch mode functionality for Nosé-Hoover Chain integrator.

    Note: nhc_thermostat_chain_update does not use atom_ptr because it operates
    on per-system chain variables (eta, eta_dot). Only velocity_half_step and
    position_update support atom_ptr batch mode.
    """

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_atom_ptr_velocity_half_step_runs(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test that nhc_velocity_half_step executes with atom_ptr."""
        # 3 systems with different sizes: 10, 25, 15 atoms
        atom_counts = [10, 25, 15]
        total_atoms = sum(atom_counts)
        np.random.seed(42)

        # Create CSR-style atom_ptr
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

        nhc_velocity_half_step(velocities, forces, masses, dt, atom_ptr=atom_ptr)
        wp.synchronize_device(device)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_atom_ptr_position_update_runs(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test that nhc_position_update executes with atom_ptr."""
        atom_counts = [10, 25, 15]
        total_atoms = sum(atom_counts)
        np.random.seed(42)

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
        dt = wp.array([0.001, 0.002, 0.0015], dtype=dtype_scalar, device=device)

        nhc_position_update(positions, velocities, dt, atom_ptr=atom_ptr)
        wp.synchronize_device(device)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_atom_ptr_velocity_half_step_out(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test non-mutating velocity_half_step with atom_ptr."""
        atom_counts = [10, 20, 10]
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
        vel_out = nhc_velocity_half_step_out(
            velocities,
            forces,
            masses,
            dt,
            velocities_out,
            atom_ptr=atom_ptr,
            device=device,
        )

        wp.synchronize_device(device)

        # Check input preserved
        np.testing.assert_array_equal(velocities.numpy(), vel_orig)

        # Check output modified
        assert vel_out.shape[0] == total_atoms
        assert not np.allclose(vel_out.numpy(), vel_orig)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_atom_ptr_position_update_out(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test non-mutating position_update with atom_ptr."""
        atom_counts = [15, 15, 10]
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
        dt = wp.array([0.001, 0.001, 0.001], dtype=dtype_scalar, device=device)

        pos_orig = positions.numpy().copy()

        positions_out = wp.empty_like(positions)
        pos_out = nhc_position_update_out(
            positions, velocities, dt, positions_out, atom_ptr=atom_ptr, device=device
        )

        wp.synchronize_device(device)

        np.testing.assert_array_equal(positions.numpy(), pos_orig)
        assert pos_out.shape[0] == total_atoms
        assert not np.allclose(pos_out.numpy(), pos_orig)

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
            [[0.1, 0.0, 0.0], [0.1, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=np_dtype
        )

        positions = wp.array(pos, dtype=dtype_vec, device=device)
        velocities = wp.array(vel, dtype=dtype_vec, device=device)

        # Different timesteps: dt[1] = 2*dt[0], dt[2] = 3*dt[0]
        dt_vals = [0.01, 0.02, 0.03]
        dt = wp.array(dt_vals, dtype=dtype_scalar, device=device)

        nhc_position_update(positions, velocities, dt, atom_ptr=atom_ptr)
        wp.synchronize_device(device)

        result_pos = positions.numpy()

        # Displacement should be: v * dt
        displacement_0 = result_pos[0, 0] - 1.0
        displacement_1 = result_pos[1, 0] - 1.0
        displacement_2 = result_pos[2, 0] - 1.0

        # Check ratios: displacement ~ dt
        ratio_1_0 = displacement_1 / displacement_0
        ratio_2_0 = displacement_2 / displacement_0

        np.testing.assert_allclose(ratio_1_0, 2.0, rtol=0.01)  # 0.02/0.01 = 2
        np.testing.assert_allclose(ratio_2_0, 3.0, rtol=0.01)  # 0.03/0.01 = 3

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
        atom_ptr = wp.array([0, 20, 40, 60], dtype=wp.int32, device=device)

        # Execute with batch_idx
        nhc_velocity_half_step(
            velocities_batch, forces_batch, masses_batch, dt_batch, batch_idx=batch_idx
        )
        nhc_position_update(
            positions_batch, velocities_batch, dt_batch, batch_idx=batch_idx
        )

        # Execute with atom_ptr
        nhc_velocity_half_step(
            velocities_ptr, forces_ptr, masses_ptr, dt_ptr, atom_ptr=atom_ptr
        )
        nhc_position_update(positions_ptr, velocities_ptr, dt_ptr, atom_ptr=atom_ptr)

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
    def test_atom_ptr_mutual_exclusivity_velocity_half_step(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test that providing both batch_idx and atom_ptr raises ValueError for velocity_half_step."""
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
            nhc_velocity_half_step(
                velocities,
                forces,
                masses,
                dt,
                batch_idx=batch_idx,
                atom_ptr=atom_ptr,
            )

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
        dt = wp.array([0.001, 0.001], dtype=dtype_scalar, device=device)

        batch_idx = wp.array([0] * 10 + [1] * 10, dtype=wp.int32, device=device)
        atom_ptr = wp.array([0, 10, 20], dtype=wp.int32, device=device)

        # Should raise ValueError
        with pytest.raises(ValueError, match="Provide batch_idx OR atom_ptr, not both"):
            nhc_position_update(
                positions,
                velocities,
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

        pos_orig = positions.numpy().copy()
        vel_orig = velocities.numpy().copy()

        # Execute integration steps
        nhc_velocity_half_step(velocities, forces, masses, dt, atom_ptr=atom_ptr)
        nhc_position_update(positions, velocities, dt, atom_ptr=atom_ptr)
        wp.synchronize_device(device)

        result_pos = positions.numpy()
        result_vel = velocities.numpy()

        # Verify all systems were updated
        offset = 0
        for sys_id in range(num_systems):
            n = atom_counts[sys_id]
            sys_pos_orig = pos_orig[offset : offset + n]
            sys_pos_result = result_pos[offset : offset + n]
            sys_vel_orig = vel_orig[offset : offset + n]
            sys_vel_result = result_vel[offset : offset + n]

            # Each system should have moved
            assert not np.allclose(sys_pos_orig, sys_pos_result), (
                f"System {sys_id} (size={n}) positions not updated"
            )
            assert not np.allclose(sys_vel_orig, sys_vel_result), (
                f"System {sys_id} (size={n}) velocities not updated"
            )
            offset += n


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
