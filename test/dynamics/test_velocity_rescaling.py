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
Comprehensive tests for velocity rescaling thermostat.

Tests cover:
- _compute_rescale_factor utility
- velocity_rescale (in-place)
- velocity_rescale_out (non-mutating)
- Single and batched modes
- Float32 and float64 support
"""

import numpy as np
import pytest
import warp as wp

from nvalchemiops.dynamics.integrators.velocity_rescaling import (
    _compute_rescale_factor,
    velocity_rescale,
    velocity_rescale_out,
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
# Fixtures
# ==============================================================================


@pytest.fixture
def device():
    """Return CUDA device."""
    return "cuda:0"


# ==============================================================================
# Test compute_rescale_factor
# ==============================================================================


class TestComputeRescaleFactor:
    """Tests for compute_rescale_factor utility function."""

    def test_basic_rescaling(self):
        """Test basic temperature rescaling factor calculation."""
        current_temp = 200.0
        target_temp = 300.0

        factor = _compute_rescale_factor(current_temp, target_temp)

        expected = np.sqrt(target_temp / current_temp)
        np.testing.assert_allclose(factor, expected, rtol=1e-6)

    def test_cooling(self):
        """Test rescaling factor for cooling (target < current)."""
        current_temp = 400.0
        target_temp = 300.0

        factor = _compute_rescale_factor(current_temp, target_temp)

        # Factor should be < 1 for cooling
        assert factor < 1.0
        expected = np.sqrt(target_temp / current_temp)
        np.testing.assert_allclose(factor, expected, rtol=1e-6)

    def test_heating(self):
        """Test rescaling factor for heating (target > current)."""
        current_temp = 200.0
        target_temp = 400.0

        factor = _compute_rescale_factor(current_temp, target_temp)

        # Factor should be > 1 for heating
        assert factor > 1.0
        expected = np.sqrt(target_temp / current_temp)
        np.testing.assert_allclose(factor, expected, rtol=1e-6)

    def test_no_rescaling(self):
        """Test rescaling factor when temperatures are equal."""
        current_temp = 300.0
        target_temp = 300.0

        factor = _compute_rescale_factor(current_temp, target_temp)

        np.testing.assert_allclose(factor, 1.0, rtol=1e-6)

    def test_zero_current_temperature(self):
        """Test handling of zero current temperature (returns 1.0)."""
        factor = _compute_rescale_factor(0.0, 300.0)
        assert factor == 1.0

    def test_negative_current_temperature(self):
        """Test handling of negative current temperature (returns 1.0)."""
        factor = _compute_rescale_factor(-100.0, 300.0)
        assert factor == 1.0

    def test_very_small_temperature(self):
        """Test with very small temperature difference."""
        current_temp = 300.0
        target_temp = 300.0001

        factor = _compute_rescale_factor(current_temp, target_temp)

        # Should be very close to 1
        assert 0.99 < factor < 1.01


# ==============================================================================
# Test velocity_rescale (in-place)
# ==============================================================================


class TestVelocityRescale:
    """Tests for in-place velocity rescaling."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_basic_rescaling(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test basic velocity rescaling."""
        num_atoms = 10
        np.random.seed(42)

        velocities_np = np.random.randn(num_atoms, 3).astype(np_dtype)
        velocities = wp.array(velocities_np, dtype=dtype_vec, device=device)

        scale_factor = 1.5
        scale = wp.array([scale_factor], dtype=dtype_scalar, device=device)

        velocity_rescale(velocities, scale, device=device)

        wp.synchronize_device(device)

        result_np = velocities.numpy()
        expected = velocities_np * scale_factor

        np.testing.assert_allclose(result_np, expected, rtol=1e-5)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_cooling_rescale(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test velocity rescaling for cooling."""
        num_atoms = 5
        np.random.seed(43)

        velocities_np = np.random.randn(num_atoms, 3).astype(np_dtype)
        velocities = wp.array(velocities_np, dtype=dtype_vec, device=device)

        scale_factor = 0.8  # Cooling
        scale = wp.array([scale_factor], dtype=dtype_scalar, device=device)

        velocity_rescale(velocities, scale, device=device)

        wp.synchronize_device(device)

        result_np = velocities.numpy()
        # Kinetic energy should decrease
        initial_ke = np.sum(velocities_np**2)
        final_ke = np.sum(result_np**2)

        assert final_ke < initial_ke

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_batched_rescaling(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test batched velocity rescaling with different factors per system."""
        # Two systems: 3 atoms and 4 atoms
        atom_counts = [3, 4]
        total_atoms = sum(atom_counts)
        num_systems = len(atom_counts)

        np.random.seed(44)
        velocities_np = np.random.randn(total_atoms, 3).astype(np_dtype)
        velocities = wp.array(velocities_np, dtype=dtype_vec, device=device)

        # Different scale factors per system
        scale_factors = [1.2, 0.9]
        scale = wp.array(scale_factors, dtype=dtype_scalar, device=device)

        # Create batch_idx
        batch_idx_np = np.concatenate(
            [np.full(count, i, dtype=np.int32) for i, count in enumerate(atom_counts)]
        )
        batch_idx = wp.array(batch_idx_np, dtype=wp.int32, device=device)

        velocity_rescale(velocities, scale, batch_idx=batch_idx, device=device)

        wp.synchronize_device(device)

        result_np = velocities.numpy()

        # Check each system separately
        offset = 0
        for sys_id in range(num_systems):
            n = atom_counts[sys_id]
            expected = velocities_np[offset : offset + n] * scale_factors[sys_id]
            np.testing.assert_allclose(
                result_np[offset : offset + n], expected, rtol=1e-5
            )
            offset += n

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_device_inference(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test that device is inferred from velocities array."""
        num_atoms = 5
        np.random.seed(45)

        velocities_np = np.random.randn(num_atoms, 3).astype(np_dtype)
        velocities = wp.array(velocities_np, dtype=dtype_vec, device=device)

        scale = wp.array([1.5], dtype=dtype_scalar, device=device)

        # Don't pass device - should be inferred
        velocity_rescale(velocities, scale)

        wp.synchronize_device(device)

        result_np = velocities.numpy()
        expected = velocities_np * 1.5

        np.testing.assert_allclose(result_np, expected, rtol=1e-5)


# ==============================================================================
# Test velocity_rescale_out (non-mutating)
# ==============================================================================


class TestVelocityRescaleOut:
    """Tests for non-mutating velocity rescaling."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_basic_rescaling(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test basic non-mutating velocity rescaling."""
        num_atoms = 10
        np.random.seed(46)

        velocities_np = np.random.randn(num_atoms, 3).astype(np_dtype)
        velocities = wp.array(velocities_np, dtype=dtype_vec, device=device)

        scale_factor = 1.5
        scale = wp.array([scale_factor], dtype=dtype_scalar, device=device)

        velocities_out = wp.empty_like(velocities)
        velocity_rescale_out(velocities, scale, velocities_out, device=device)

        wp.synchronize_device(device)

        # Original unchanged
        np.testing.assert_allclose(velocities.numpy(), velocities_np, rtol=1e-6)

        # Output scaled
        result_np = velocities_out.numpy()
        expected = velocities_np * scale_factor

        np.testing.assert_allclose(result_np, expected, rtol=1e-5)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_preallocated_output(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test non-mutating rescaling with pre-allocated output."""
        num_atoms = 10
        np.random.seed(47)

        velocities_np = np.random.randn(num_atoms, 3).astype(np_dtype)
        velocities = wp.array(velocities_np, dtype=dtype_vec, device=device)

        scale_factor = 1.3
        scale = wp.array([scale_factor], dtype=dtype_scalar, device=device)

        # Pre-allocate output
        velocities_out = wp.zeros(num_atoms, dtype=dtype_vec, device=device)

        result = velocity_rescale_out(
            velocities, scale, velocities_out=velocities_out, device=device
        )

        wp.synchronize_device(device)

        # Should return the same array
        assert result is velocities_out

        # Check values
        result_np = result.numpy()
        expected = velocities_np * scale_factor

        np.testing.assert_allclose(result_np, expected, rtol=1e-5)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_batched_rescaling(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test batched non-mutating velocity rescaling."""
        # Two systems: 3 atoms and 4 atoms
        atom_counts = [3, 4]
        total_atoms = sum(atom_counts)
        num_systems = len(atom_counts)

        np.random.seed(48)
        velocities_np = np.random.randn(total_atoms, 3).astype(np_dtype)
        velocities = wp.array(velocities_np, dtype=dtype_vec, device=device)

        # Different scale factors per system
        scale_factors = [1.1, 0.85]
        scale = wp.array(scale_factors, dtype=dtype_scalar, device=device)

        # Create batch_idx
        batch_idx_np = np.concatenate(
            [np.full(count, i, dtype=np.int32) for i, count in enumerate(atom_counts)]
        )
        batch_idx = wp.array(batch_idx_np, dtype=wp.int32, device=device)

        velocities_out = wp.empty_like(velocities)
        velocity_rescale_out(
            velocities, scale, velocities_out, batch_idx=batch_idx, device=device
        )

        wp.synchronize_device(device)

        # Original unchanged
        np.testing.assert_allclose(velocities.numpy(), velocities_np, rtol=1e-6)

        # Check each system separately
        result_np = velocities_out.numpy()
        offset = 0
        for sys_id in range(num_systems):
            n = atom_counts[sys_id]
            expected = velocities_np[offset : offset + n] * scale_factors[sys_id]
            np.testing.assert_allclose(
                result_np[offset : offset + n], expected, rtol=1e-5
            )
            offset += n

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_device_inference(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test that device is inferred from velocities array."""
        num_atoms = 5
        np.random.seed(49)

        velocities_np = np.random.randn(num_atoms, 3).astype(np_dtype)
        velocities = wp.array(velocities_np, dtype=dtype_vec, device=device)

        scale = wp.array([1.5], dtype=dtype_scalar, device=device)

        # Don't pass device - should be inferred
        velocities_out = wp.empty_like(velocities)
        velocity_rescale_out(velocities, scale, velocities_out)

        wp.synchronize_device(device)

        result_np = velocities_out.numpy()
        expected = velocities_np * 1.5

        np.testing.assert_allclose(result_np, expected, rtol=1e-5)


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestVelocityRescalingIntegration:
    """Integration tests for velocity rescaling workflow."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_full_rescaling_workflow(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test full workflow: compute factor, then rescale."""
        num_atoms = 100
        np.random.seed(50)

        # Initial velocities sampled from Maxwell-Boltzmann
        masses_np = np.ones(num_atoms, dtype=np_dtype) * 12.0  # Carbon-like
        velocities_np = np.random.randn(num_atoms, 3).astype(np_dtype)

        # Scale to approximate initial temperature

        # Compute initial kinetic energy
        ke_initial = 0.5 * np.sum(masses_np[:, None] * velocities_np**2)

        # Compute target KE for target temperature
        target_temp = 400.0
        ndof = 3 * num_atoms - 3  # Remove COM motion DOF
        kB = 8.617333e-5  # eV/K

        # Current temperature from KE
        current_temp = 2 * ke_initial / (ndof * kB)

        # Compute rescale factor
        factor = _compute_rescale_factor(current_temp, target_temp)

        # Apply rescaling
        velocities = wp.array(velocities_np, dtype=dtype_vec, device=device)
        scale = wp.array([factor], dtype=dtype_scalar, device=device)

        velocity_rescale(velocities, scale, device=device)

        wp.synchronize_device(device)

        # Verify new temperature
        vel_scaled = velocities.numpy()
        ke_final = 0.5 * np.sum(masses_np[:, None] * vel_scaled**2)
        final_temp = 2 * ke_final / (ndof * kB)

        np.testing.assert_allclose(final_temp, target_temp, rtol=1e-5)


# ==============================================================================
# Atom Pointer (CSR) Batch Mode Tests
# ==============================================================================


class TestVelocityRescalingAtomPtr:
    """Test atom_ptr batch mode functionality for velocity rescaling."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_atom_ptr_rescaling_runs(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test that velocity_rescale executes with atom_ptr."""
        # 3 systems with different sizes: 10, 25, 15 atoms
        atom_counts = [10, 25, 15]
        total_atoms = sum(atom_counts)
        np.random.seed(42)

        # Create CSR-style atom_ptr
        atom_ptr_np = np.concatenate([[0], np.cumsum(atom_counts)]).astype(np.int32)
        atom_ptr = wp.array(atom_ptr_np, dtype=wp.int32, device=device)

        velocities_np = np.random.randn(total_atoms, 3).astype(np_dtype)
        velocities = wp.array(velocities_np, dtype=dtype_vec, device=device)

        # Different scale factors per system
        scale_factors = [1.2, 0.9, 1.5]
        scale = wp.array(scale_factors, dtype=dtype_scalar, device=device)

        velocity_rescale(velocities, scale, atom_ptr=atom_ptr, device=device)

        wp.synchronize_device(device)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_atom_ptr_rescaling_out(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test non-mutating velocity_rescale_out with atom_ptr."""
        atom_counts = [10, 20, 10]
        total_atoms = sum(atom_counts)

        atom_ptr_np = np.concatenate([[0], np.cumsum(atom_counts)]).astype(np.int32)
        atom_ptr = wp.array(atom_ptr_np, dtype=wp.int32, device=device)

        velocities_np = np.random.randn(total_atoms, 3).astype(np_dtype)
        velocities = wp.array(velocities_np, dtype=dtype_vec, device=device)

        scale_factors = [1.1, 0.85, 1.3]
        scale = wp.array(scale_factors, dtype=dtype_scalar, device=device)

        vel_orig = velocities.numpy().copy()

        vel_out = wp.empty_like(velocities)
        velocity_rescale_out(
            velocities, scale, vel_out, atom_ptr=atom_ptr, device=device
        )

        wp.synchronize_device(device)

        # Check input preserved
        np.testing.assert_array_equal(velocities.numpy(), vel_orig)

        # Check output modified
        assert vel_out.shape[0] == total_atoms
        assert not np.allclose(vel_out.numpy(), vel_orig)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_atom_ptr_uses_per_system_scale_factors(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test that atom_ptr mode uses per-system scale factors correctly."""
        # 3 systems with 1 atom each for easy verification
        num_systems = 3

        atom_ptr_np = np.array([0, 1, 2, 3], dtype=np.int32)
        atom_ptr = wp.array(atom_ptr_np, dtype=wp.int32, device=device)

        # Same initial velocities
        vel = np.array(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np_dtype
        )
        velocities = wp.array(vel, dtype=dtype_vec, device=device)

        # Different scale factors: 1.0, 2.0, 3.0
        scale_factors = [1.0, 2.0, 3.0]
        scale = wp.array(scale_factors, dtype=dtype_scalar, device=device)

        velocity_rescale(velocities, scale, atom_ptr=atom_ptr, device=device)
        wp.synchronize_device(device)

        result_vel = velocities.numpy()

        # Check each system got its correct scale factor
        for sys_id in range(num_systems):
            expected_vel = vel[sys_id] * scale_factors[sys_id]
            np.testing.assert_allclose(result_vel[sys_id], expected_vel, rtol=1e-5)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_atom_ptr_vs_batch_idx_equivalence(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test that atom_ptr and batch_idx produce identical results for same-sized systems."""
        num_systems = 3
        atoms_per_system = 20
        total_atoms = num_systems * atoms_per_system

        np.random.seed(42)
        initial_vel = np.random.randn(total_atoms, 3).astype(np_dtype)
        scale_factors = [1.2, 0.9, 1.5]

        # Setup for batch_idx mode
        velocities_batch = wp.array(initial_vel.copy(), dtype=dtype_vec, device=device)
        scale_batch = wp.array(scale_factors, dtype=dtype_scalar, device=device)
        batch_idx = wp.array(
            np.repeat(np.arange(num_systems), atoms_per_system).astype(np.int32),
            dtype=wp.int32,
            device=device,
        )

        # Setup for atom_ptr mode
        velocities_ptr = wp.array(initial_vel.copy(), dtype=dtype_vec, device=device)
        scale_ptr = wp.array(scale_factors, dtype=dtype_scalar, device=device)
        atom_ptr = wp.array([0, 20, 40, 60], dtype=wp.int32, device=device)

        # Execute with batch_idx
        velocity_rescale(
            velocities_batch, scale_batch, batch_idx=batch_idx, device=device
        )

        # Execute with atom_ptr
        velocity_rescale(velocities_ptr, scale_ptr, atom_ptr=atom_ptr, device=device)

        wp.synchronize_device(device)

        # Results should be identical
        np.testing.assert_allclose(
            velocities_batch.numpy(), velocities_ptr.numpy(), rtol=1e-6
        )

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_atom_ptr_mutual_exclusivity(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test that providing both batch_idx and atom_ptr raises ValueError."""
        total_atoms = 20

        velocities = wp.array(
            np.random.randn(total_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        scale = wp.array([1.2, 0.9], dtype=dtype_scalar, device=device)

        batch_idx = wp.array([0] * 10 + [1] * 10, dtype=wp.int32, device=device)
        atom_ptr = wp.array([0, 10, 20], dtype=wp.int32, device=device)

        # Should raise ValueError
        with pytest.raises(ValueError, match="Provide batch_idx OR atom_ptr, not both"):
            velocity_rescale(
                velocities,
                scale,
                batch_idx=batch_idx,
                atom_ptr=atom_ptr,
                device=device,
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

        np.random.seed(43)

        atom_ptr_np = np.concatenate([[0], np.cumsum(atom_counts)]).astype(np.int32)
        atom_ptr = wp.array(atom_ptr_np, dtype=wp.int32, device=device)

        velocities_np = np.random.randn(total_atoms, 3).astype(np_dtype)
        velocities = wp.array(velocities_np, dtype=dtype_vec, device=device)

        scale_factors = [1.1, 0.9, 1.3, 0.8]
        scale = wp.array(scale_factors, dtype=dtype_scalar, device=device)

        vel_orig = velocities.numpy().copy()

        # Execute rescaling
        velocity_rescale(velocities, scale, atom_ptr=atom_ptr, device=device)
        wp.synchronize_device(device)

        result_vel = velocities.numpy()

        # Verify all systems were updated correctly
        offset = 0
        for sys_id in range(num_systems):
            n = atom_counts[sys_id]
            sys_vel_orig = vel_orig[offset : offset + n]
            sys_vel_result = result_vel[offset : offset + n]
            expected = sys_vel_orig * scale_factors[sys_id]

            np.testing.assert_allclose(sys_vel_result, expected, rtol=1e-5)
            offset += n


# ==============================================================================
# Parametric Parity Tests
# ==============================================================================


MODE_CONFIGS = [
    pytest.param("single", id="single"),
    pytest.param("batch_idx", id="batch_idx"),
    pytest.param("atom_ptr", id="atom_ptr"),
]


class TestVelocityRescalingParity:
    """Parametric parity tests across all execution modes and dtypes.

    Verifies that single, batch_idx, and atom_ptr modes produce
    identical numerical results for the same physical setup.
    """

    @staticmethod
    def _make_batch_data(
        num_systems, atoms_per_system, np_dtype, dtype_vec, dtype_scalar, device
    ):
        """Create consistent test data for all modes."""
        total_atoms = num_systems * atoms_per_system
        np.random.seed(99)

        velocities_np = np.random.randn(total_atoms, 3).astype(np_dtype)
        scale_factors = np.random.uniform(0.5, 2.0, size=num_systems).astype(np_dtype)

        batch_idx_np = np.repeat(
            np.arange(num_systems, dtype=np.int32), atoms_per_system
        )
        atom_ptr_np = np.arange(0, total_atoms + 1, atoms_per_system, dtype=np.int32)

        return {
            "velocities_np": velocities_np,
            "scale_factors_np": scale_factors,
            "batch_idx": wp.array(batch_idx_np, dtype=wp.int32, device=device),
            "atom_ptr": wp.array(atom_ptr_np, dtype=wp.int32, device=device),
            "total_atoms": total_atoms,
            "dtype_vec": dtype_vec,
            "dtype_scalar": dtype_scalar,
        }

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    @pytest.mark.parametrize("mode", MODE_CONFIGS)
    def test_inplace_parity(self, device, dtype_vec, dtype_scalar, np_dtype, mode):
        """Verify in-place rescaling produces correct results in all modes."""
        num_systems = 3
        atoms_per_system = 20

        data = self._make_batch_data(
            num_systems, atoms_per_system, np_dtype, dtype_vec, dtype_scalar, device
        )
        velocities = wp.array(
            data["velocities_np"].copy(), dtype=dtype_vec, device=device
        )
        scale = wp.array(data["scale_factors_np"], dtype=dtype_scalar, device=device)

        if mode == "single":
            # For single mode, use only the first system's data
            vel_single = wp.array(
                data["velocities_np"][:atoms_per_system].copy(),
                dtype=dtype_vec,
                device=device,
            )
            scale_single = wp.array(
                [data["scale_factors_np"][0]], dtype=dtype_scalar, device=device
            )
            velocity_rescale(vel_single, scale_single, device=device)
            wp.synchronize_device(device)
            expected = (
                data["velocities_np"][:atoms_per_system] * data["scale_factors_np"][0]
            )
            np.testing.assert_allclose(vel_single.numpy(), expected, rtol=1e-5)
        elif mode == "batch_idx":
            velocity_rescale(
                velocities, scale, batch_idx=data["batch_idx"], device=device
            )
            wp.synchronize_device(device)
            result = velocities.numpy()
            for s in range(num_systems):
                start = s * atoms_per_system
                end = start + atoms_per_system
                expected = (
                    data["velocities_np"][start:end] * data["scale_factors_np"][s]
                )
                np.testing.assert_allclose(result[start:end], expected, rtol=1e-5)
        else:  # atom_ptr
            velocity_rescale(
                velocities, scale, atom_ptr=data["atom_ptr"], device=device
            )
            wp.synchronize_device(device)
            result = velocities.numpy()
            for s in range(num_systems):
                start = s * atoms_per_system
                end = start + atoms_per_system
                expected = (
                    data["velocities_np"][start:end] * data["scale_factors_np"][s]
                )
                np.testing.assert_allclose(result[start:end], expected, rtol=1e-5)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    @pytest.mark.parametrize("mode", MODE_CONFIGS)
    def test_out_parity(self, device, dtype_vec, dtype_scalar, np_dtype, mode):
        """Verify non-mutating rescaling produces correct results in all modes."""
        num_systems = 3
        atoms_per_system = 20

        data = self._make_batch_data(
            num_systems, atoms_per_system, np_dtype, dtype_vec, dtype_scalar, device
        )
        velocities = wp.array(
            data["velocities_np"].copy(), dtype=dtype_vec, device=device
        )
        scale = wp.array(data["scale_factors_np"], dtype=dtype_scalar, device=device)

        if mode == "single":
            vel_single = wp.array(
                data["velocities_np"][:atoms_per_system].copy(),
                dtype=dtype_vec,
                device=device,
            )
            scale_single = wp.array(
                [data["scale_factors_np"][0]], dtype=dtype_scalar, device=device
            )
            vel_out = wp.empty_like(vel_single)
            result = velocity_rescale_out(
                vel_single, scale_single, vel_out, device=device
            )
            wp.synchronize_device(device)
            # Input preserved
            np.testing.assert_allclose(
                vel_single.numpy(),
                data["velocities_np"][:atoms_per_system],
                rtol=1e-6,
            )
            # Output correct
            expected = (
                data["velocities_np"][:atoms_per_system] * data["scale_factors_np"][0]
            )
            np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)
            assert result is vel_out
        elif mode == "batch_idx":
            vel_out = wp.empty_like(velocities)
            result = velocity_rescale_out(
                velocities, scale, vel_out, batch_idx=data["batch_idx"], device=device
            )
            wp.synchronize_device(device)
            np.testing.assert_allclose(
                velocities.numpy(), data["velocities_np"], rtol=1e-6
            )
            result_np = result.numpy()
            for s in range(num_systems):
                start = s * atoms_per_system
                end = start + atoms_per_system
                expected = (
                    data["velocities_np"][start:end] * data["scale_factors_np"][s]
                )
                np.testing.assert_allclose(result_np[start:end], expected, rtol=1e-5)
        else:  # atom_ptr
            vel_out = wp.empty_like(velocities)
            result = velocity_rescale_out(
                velocities, scale, vel_out, atom_ptr=data["atom_ptr"], device=device
            )
            wp.synchronize_device(device)
            np.testing.assert_allclose(
                velocities.numpy(), data["velocities_np"], rtol=1e-6
            )
            result_np = result.numpy()
            for s in range(num_systems):
                start = s * atoms_per_system
                end = start + atoms_per_system
                expected = (
                    data["velocities_np"][start:end] * data["scale_factors_np"][s]
                )
                np.testing.assert_allclose(result_np[start:end], expected, rtol=1e-5)

    @pytest.mark.parametrize("device", DEVICES)
    def test_out_requires_velocities_out(self, device):
        """Verify velocity_rescale_out raises TypeError without velocities_out."""
        velocities = wp.zeros(10, dtype=wp.vec3f, device=device)
        scale = wp.array([1.5], dtype=wp.float32, device=device)

        with pytest.raises(TypeError):
            velocity_rescale_out(velocities, scale, device=device)
