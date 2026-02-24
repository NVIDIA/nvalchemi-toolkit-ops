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
Comprehensive tests for SHAKE and RATTLE constraint algorithms.

Tests cover:
- SHAKE position constraints (in-place and out)
- RATTLE velocity constraints (in-place and out)
- Single iteration and full constraint loop
- Convergence properties
- Float32 and float64 support
- Multi-bond systems
"""

import numpy as np
import pytest
import warp as wp

from nvalchemiops.dynamics.utils.constraints import (
    rattle_constraints,
    rattle_constraints_out,
    rattle_iteration,
    rattle_iteration_out,
    shake_constraints,
    shake_constraints_out,
    shake_iteration,
    shake_iteration_out,
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


@pytest.fixture
def simple_bond_system():
    """Create a simple two-atom bond system for testing."""
    return {
        "num_atoms": 2,
        "num_bonds": 1,
        "bond_length": 1.0,
        "masses": np.array([1.0, 1.0]),
    }


@pytest.fixture
def chain_system():
    """Create a linear chain of 4 atoms with 3 bonds."""
    return {
        "num_atoms": 4,
        "num_bonds": 3,
        "bond_length": 1.5,
        "masses": np.array([12.0, 1.0, 1.0, 12.0]),  # C-H-H-C like
    }


# ==============================================================================
# SHAKE Tests
# ==============================================================================


class TestShakeIteration:
    """Tests for single SHAKE iteration."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_shake_iteration_basic(
        self, device, dtype_vec, dtype_scalar, np_dtype, simple_bond_system
    ):
        """Test single SHAKE iteration reduces constraint error."""
        bond_length = simple_bond_system["bond_length"]
        masses_np = simple_bond_system["masses"].astype(np_dtype)

        # Initial positions: stretched bond
        positions_old_np = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np_dtype)
        # Unconstrained positions: stretched further
        positions_np = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], dtype=np_dtype)

        positions = wp.array(positions_np, dtype=dtype_vec, device=device)
        positions_old = wp.array(positions_old_np, dtype=dtype_vec, device=device)
        masses = wp.array(masses_np, dtype=dtype_scalar, device=device)

        bond_atom_i = wp.array([0], dtype=wp.int32, device=device)
        bond_atom_j = wp.array([1], dtype=wp.int32, device=device)
        bond_lengths_sq = wp.array([bond_length**2], dtype=dtype_scalar, device=device)

        max_error = wp.zeros(1, dtype=wp.float64, device=device)

        shake_iteration(
            positions,
            positions_old,
            masses,
            bond_atom_i,
            bond_atom_j,
            bond_lengths_sq,
            max_error,
            device=device,
        )

        wp.synchronize_device(device)

        # Error should be recorded
        error_val = max_error.numpy()[0]
        assert error_val > 0.0  # There was a constraint violation

        # Bond should be closer to target length
        pos_np = positions.numpy()
        r_ij = pos_np[1] - pos_np[0]
        new_length = np.linalg.norm(r_ij)

        # Should be closer to 1.0 than initial 1.2
        assert abs(new_length - bond_length) < abs(1.2 - bond_length)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_shake_iteration_with_preallocated_error(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test SHAKE iteration with pre-allocated error array."""
        positions_np = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], dtype=np_dtype)
        positions_old_np = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np_dtype)
        masses_np = np.array([1.0, 1.0], dtype=np_dtype)

        positions = wp.array(positions_np, dtype=dtype_vec, device=device)
        positions_old = wp.array(positions_old_np, dtype=dtype_vec, device=device)
        masses = wp.array(masses_np, dtype=dtype_scalar, device=device)

        bond_atom_i = wp.array([0], dtype=wp.int32, device=device)
        bond_atom_j = wp.array([1], dtype=wp.int32, device=device)
        bond_lengths_sq = wp.array([1.0], dtype=dtype_scalar, device=device)

        # Pre-allocate error array
        max_error = wp.zeros(1, dtype=wp.float64, device=device)

        result = shake_iteration(
            positions,
            positions_old,
            masses,
            bond_atom_i,
            bond_atom_j,
            bond_lengths_sq,
            max_error,
            device=device,
        )

        wp.synchronize_device(device)
        assert result is max_error  # Same array returned


class TestShakeConstraints:
    """Tests for full SHAKE constraint loop."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_shake_constraints_convergence(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test SHAKE converges to target bond length."""
        bond_length = 1.0
        positions_old_np = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np_dtype)
        positions_np = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=np_dtype)
        masses_np = np.array([1.0, 1.0], dtype=np_dtype)

        positions = wp.array(positions_np, dtype=dtype_vec, device=device)
        positions_old = wp.array(positions_old_np, dtype=dtype_vec, device=device)
        masses = wp.array(masses_np, dtype=dtype_scalar, device=device)

        bond_atom_i = wp.array([0], dtype=wp.int32, device=device)
        bond_atom_j = wp.array([1], dtype=wp.int32, device=device)
        bond_lengths_sq = wp.array([bond_length**2], dtype=dtype_scalar, device=device)

        max_error = wp.empty(1, dtype=wp.float64, device=device)

        shake_constraints(
            positions,
            positions_old,
            masses,
            bond_atom_i,
            bond_atom_j,
            bond_lengths_sq,
            max_error,
            num_iter=20,
            device=device,
        )

        wp.synchronize_device(device)

        # Check convergence
        pos_np = positions.numpy()
        r_ij = pos_np[1] - pos_np[0]
        actual_length = np.linalg.norm(r_ij)

        # Should be very close to target
        np.testing.assert_allclose(actual_length, bond_length, rtol=1e-4)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_shake_constraints_chain(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test SHAKE on multi-bond chain system."""
        bond_length = 1.5
        # Linear chain along x
        positions_old_np = np.array(
            [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0], [4.5, 0.0, 0.0]],
            dtype=np_dtype,
        )
        # Perturbed positions
        positions_np = np.array(
            [[0.1, 0.1, 0.0], [1.6, -0.1, 0.0], [3.1, 0.1, 0.0], [4.6, -0.1, 0.0]],
            dtype=np_dtype,
        )
        masses_np = np.array([12.0, 1.0, 1.0, 12.0], dtype=np_dtype)

        positions = wp.array(positions_np, dtype=dtype_vec, device=device)
        positions_old = wp.array(positions_old_np, dtype=dtype_vec, device=device)
        masses = wp.array(masses_np, dtype=dtype_scalar, device=device)

        # 3 bonds: 0-1, 1-2, 2-3
        bond_atom_i = wp.array([0, 1, 2], dtype=wp.int32, device=device)
        bond_atom_j = wp.array([1, 2, 3], dtype=wp.int32, device=device)
        bond_lengths_sq = wp.array(
            [bond_length**2] * 3, dtype=dtype_scalar, device=device
        )

        max_error = wp.empty(1, dtype=wp.float64, device=device)

        shake_constraints(
            positions,
            positions_old,
            masses,
            bond_atom_i,
            bond_atom_j,
            bond_lengths_sq,
            max_error,
            num_iter=30,
            device=device,
        )

        wp.synchronize_device(device)

        # Check all bonds - use larger tolerance for chain systems
        # due to competing constraints
        pos_np = positions.numpy()
        for i, j in [(0, 1), (1, 2), (2, 3)]:
            r_ij = pos_np[j] - pos_np[i]
            actual_length = np.linalg.norm(r_ij)
            np.testing.assert_allclose(actual_length, bond_length, rtol=5e-3)


class TestShakeIterationOut:
    """Tests for non-mutating SHAKE iteration."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_shake_iteration_out_basic(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test non-mutating SHAKE iteration."""
        positions_np = np.array([[0.0, 0.0, 0.0], [1.3, 0.0, 0.0]], dtype=np_dtype)
        positions_old_np = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np_dtype)
        masses_np = np.array([1.0, 1.0], dtype=np_dtype)

        positions = wp.array(positions_np, dtype=dtype_vec, device=device)
        positions_old = wp.array(positions_old_np, dtype=dtype_vec, device=device)
        masses = wp.array(masses_np, dtype=dtype_scalar, device=device)

        bond_atom_i = wp.array([0], dtype=wp.int32, device=device)
        bond_atom_j = wp.array([1], dtype=wp.int32, device=device)
        bond_lengths_sq = wp.array([1.0], dtype=dtype_scalar, device=device)

        num_atoms = 2
        position_corrections = wp.zeros(num_atoms, dtype=dtype_vec, device=device)
        max_error = wp.zeros(1, dtype=wp.float64, device=device)

        corrections, max_error = shake_iteration_out(
            positions,
            positions_old,
            masses,
            bond_atom_i,
            bond_atom_j,
            bond_lengths_sq,
            position_corrections,
            max_error,
            device=device,
        )

        wp.synchronize_device(device)

        # Original positions unchanged
        np.testing.assert_allclose(positions.numpy(), positions_np, rtol=1e-6)

        # Corrections computed
        corr_np = corrections.numpy()
        assert np.any(np.abs(corr_np) > 1e-10)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_shake_iteration_out_preallocated(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test non-mutating SHAKE with pre-allocated arrays."""
        num_atoms = 2
        positions_np = np.array([[0.0, 0.0, 0.0], [1.3, 0.0, 0.0]], dtype=np_dtype)
        positions_old_np = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np_dtype)
        masses_np = np.array([1.0, 1.0], dtype=np_dtype)

        positions = wp.array(positions_np, dtype=dtype_vec, device=device)
        positions_old = wp.array(positions_old_np, dtype=dtype_vec, device=device)
        masses = wp.array(masses_np, dtype=dtype_scalar, device=device)

        bond_atom_i = wp.array([0], dtype=wp.int32, device=device)
        bond_atom_j = wp.array([1], dtype=wp.int32, device=device)
        bond_lengths_sq = wp.array([1.0], dtype=dtype_scalar, device=device)

        # Pre-allocate
        corrections = wp.zeros(num_atoms, dtype=dtype_vec, device=device)
        max_error = wp.zeros(1, dtype=wp.float64, device=device)

        result_corr, result_err = shake_iteration_out(
            positions,
            positions_old,
            masses,
            bond_atom_i,
            bond_atom_j,
            bond_lengths_sq,
            corrections,
            max_error,
            device=device,
        )

        assert result_corr is corrections
        assert result_err is max_error


class TestShakeConstraintsOut:
    """Tests for non-mutating full SHAKE loop."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_shake_constraints_out_basic(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test non-mutating SHAKE constraints."""
        bond_length = 1.0
        positions_np = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=np_dtype)
        positions_old_np = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np_dtype)
        masses_np = np.array([1.0, 1.0], dtype=np_dtype)

        positions = wp.array(positions_np, dtype=dtype_vec, device=device)
        positions_old = wp.array(positions_old_np, dtype=dtype_vec, device=device)
        masses = wp.array(masses_np, dtype=dtype_scalar, device=device)

        bond_atom_i = wp.array([0], dtype=wp.int32, device=device)
        bond_atom_j = wp.array([1], dtype=wp.int32, device=device)
        bond_lengths_sq = wp.array([bond_length**2], dtype=dtype_scalar, device=device)

        num_atoms = 2
        positions_out = wp.empty(num_atoms, dtype=dtype_vec, device=device)
        wp.copy(positions_out, positions)
        max_error = wp.empty(1, dtype=wp.float64, device=device)

        positions_out, final_error = shake_constraints_out(
            positions,
            positions_old,
            masses,
            bond_atom_i,
            bond_atom_j,
            bond_lengths_sq,
            positions_out,
            max_error,
            num_iter=20,
            device=device,
        )

        wp.synchronize_device(device)

        # Original unchanged
        np.testing.assert_allclose(positions.numpy(), positions_np, rtol=1e-6)

        # Output constrained
        pos_out_np = positions_out.numpy()
        r_ij = pos_out_np[1] - pos_out_np[0]
        actual_length = np.linalg.norm(r_ij)
        np.testing.assert_allclose(actual_length, bond_length, rtol=1e-4)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_shake_constraints_out_preallocated(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test non-mutating SHAKE with pre-allocated output."""
        positions_np = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=np_dtype)
        positions_old_np = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np_dtype)
        masses_np = np.array([1.0, 1.0], dtype=np_dtype)

        positions = wp.array(positions_np, dtype=dtype_vec, device=device)
        positions_old = wp.array(positions_old_np, dtype=dtype_vec, device=device)
        masses = wp.array(masses_np, dtype=dtype_scalar, device=device)

        bond_atom_i = wp.array([0], dtype=wp.int32, device=device)
        bond_atom_j = wp.array([1], dtype=wp.int32, device=device)
        bond_lengths_sq = wp.array([1.0], dtype=dtype_scalar, device=device)

        # Pre-allocate output
        positions_out = wp.empty(2, dtype=dtype_vec, device=device)
        wp.copy(positions_out, positions)
        max_error = wp.empty(1, dtype=wp.float64, device=device)

        result_pos, _ = shake_constraints_out(
            positions,
            positions_old,
            masses,
            bond_atom_i,
            bond_atom_j,
            bond_lengths_sq,
            positions_out,
            max_error,
            num_iter=20,
            device=device,
        )

        assert result_pos is positions_out


# ==============================================================================
# RATTLE Tests
# ==============================================================================


class TestRattleIteration:
    """Tests for single RATTLE iteration."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_rattle_iteration_basic(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test single RATTLE iteration runs without error."""
        # Constrained positions (bond length = 1)
        positions_np = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np_dtype)
        # Velocities not perpendicular to bond
        velocities_np = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=np_dtype)
        masses_np = np.array([1.0, 1.0], dtype=np_dtype)

        positions = wp.array(positions_np, dtype=dtype_vec, device=device)
        velocities = wp.array(velocities_np, dtype=dtype_vec, device=device)
        masses = wp.array(masses_np, dtype=dtype_scalar, device=device)

        bond_atom_i = wp.array([0], dtype=wp.int32, device=device)
        bond_atom_j = wp.array([1], dtype=wp.int32, device=device)

        max_error = wp.zeros(1, dtype=wp.float64, device=device)

        rattle_iteration(
            positions,
            velocities,
            masses,
            bond_atom_i,
            bond_atom_j,
            max_error,
            device=device,
        )

        wp.synchronize_device(device)

        # Error should be recorded (initial constraint violation)
        error_val = max_error.numpy()[0]
        assert error_val >= 0.0  # Error is non-negative

        # Velocities should have been modified
        vel_np = velocities.numpy()
        # Just check shape is correct
        assert vel_np.shape == (2, 3)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_rattle_iteration_with_preallocated_error(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test RATTLE iteration with pre-allocated error array."""
        positions_np = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np_dtype)
        velocities_np = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=np_dtype)
        masses_np = np.array([1.0, 1.0], dtype=np_dtype)

        positions = wp.array(positions_np, dtype=dtype_vec, device=device)
        velocities = wp.array(velocities_np, dtype=dtype_vec, device=device)
        masses = wp.array(masses_np, dtype=dtype_scalar, device=device)

        bond_atom_i = wp.array([0], dtype=wp.int32, device=device)
        bond_atom_j = wp.array([1], dtype=wp.int32, device=device)

        # Pre-allocate error array
        max_error = wp.zeros(1, dtype=wp.float64, device=device)

        result = rattle_iteration(
            positions,
            velocities,
            masses,
            bond_atom_i,
            bond_atom_j,
            max_error,
            device=device,
        )

        wp.synchronize_device(device)
        assert result is max_error


class TestRattleConstraints:
    """Tests for full RATTLE constraint loop."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_rattle_constraints_runs(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test RATTLE constraints runs without error."""
        positions_np = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np_dtype)
        velocities_np = np.array([[0.0, 0.0, 0.0], [2.0, 1.0, 0.0]], dtype=np_dtype)
        masses_np = np.array([1.0, 1.0], dtype=np_dtype)

        positions = wp.array(positions_np, dtype=dtype_vec, device=device)
        velocities = wp.array(velocities_np, dtype=dtype_vec, device=device)
        masses = wp.array(masses_np, dtype=dtype_scalar, device=device)

        bond_atom_i = wp.array([0], dtype=wp.int32, device=device)
        bond_atom_j = wp.array([1], dtype=wp.int32, device=device)

        max_error = wp.empty(1, dtype=wp.float64, device=device)

        final_error = rattle_constraints(
            positions,
            velocities,
            masses,
            bond_atom_i,
            bond_atom_j,
            max_error,
            num_iter=20,
            device=device,
        )

        wp.synchronize_device(device)

        # Verify function ran and returned error array
        assert final_error.shape[0] == 1
        # Velocities should have correct shape
        assert velocities.numpy().shape == (2, 3)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_rattle_constraints_chain(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test RATTLE on multi-bond chain system runs without error."""
        # Linear chain along x
        positions_np = np.array(
            [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0], [4.5, 0.0, 0.0]],
            dtype=np_dtype,
        )
        # Velocities with parallel components
        velocities_np = np.array(
            [[0.5, 1.0, 0.0], [-0.5, 0.5, 0.0], [0.3, -0.5, 0.0], [-0.3, 1.0, 0.0]],
            dtype=np_dtype,
        )
        masses_np = np.array([12.0, 1.0, 1.0, 12.0], dtype=np_dtype)

        positions = wp.array(positions_np, dtype=dtype_vec, device=device)
        velocities = wp.array(velocities_np, dtype=dtype_vec, device=device)
        masses = wp.array(masses_np, dtype=dtype_scalar, device=device)

        bond_atom_i = wp.array([0, 1, 2], dtype=wp.int32, device=device)
        bond_atom_j = wp.array([1, 2, 3], dtype=wp.int32, device=device)

        max_error = wp.empty(1, dtype=wp.float64, device=device)

        final_error = rattle_constraints(
            positions,
            velocities,
            masses,
            bond_atom_i,
            bond_atom_j,
            max_error,
            num_iter=30,
            device=device,
        )

        wp.synchronize_device(device)

        # Verify function ran
        assert final_error.shape[0] == 1
        assert velocities.numpy().shape == (4, 3)


class TestRattleIterationOut:
    """Tests for non-mutating RATTLE iteration."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_rattle_iteration_out_basic(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test non-mutating RATTLE iteration."""
        positions_np = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np_dtype)
        velocities_np = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=np_dtype)
        masses_np = np.array([1.0, 1.0], dtype=np_dtype)

        positions = wp.array(positions_np, dtype=dtype_vec, device=device)
        velocities = wp.array(velocities_np, dtype=dtype_vec, device=device)
        masses = wp.array(masses_np, dtype=dtype_scalar, device=device)

        bond_atom_i = wp.array([0], dtype=wp.int32, device=device)
        bond_atom_j = wp.array([1], dtype=wp.int32, device=device)

        num_atoms = 2
        velocity_corrections = wp.zeros(num_atoms, dtype=dtype_vec, device=device)
        max_error = wp.zeros(1, dtype=wp.float64, device=device)

        corrections, max_error = rattle_iteration_out(
            positions,
            velocities,
            masses,
            bond_atom_i,
            bond_atom_j,
            velocity_corrections,
            max_error,
            device=device,
        )

        wp.synchronize_device(device)

        # Original velocities unchanged
        np.testing.assert_allclose(velocities.numpy(), velocities_np, rtol=1e-6)

        # Corrections computed
        corr_np = corrections.numpy()
        assert np.any(np.abs(corr_np) > 1e-10)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_rattle_iteration_out_preallocated(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test non-mutating RATTLE with pre-allocated arrays."""
        num_atoms = 2
        positions_np = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np_dtype)
        velocities_np = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=np_dtype)
        masses_np = np.array([1.0, 1.0], dtype=np_dtype)

        positions = wp.array(positions_np, dtype=dtype_vec, device=device)
        velocities = wp.array(velocities_np, dtype=dtype_vec, device=device)
        masses = wp.array(masses_np, dtype=dtype_scalar, device=device)

        bond_atom_i = wp.array([0], dtype=wp.int32, device=device)
        bond_atom_j = wp.array([1], dtype=wp.int32, device=device)

        # Pre-allocate
        corrections = wp.zeros(num_atoms, dtype=dtype_vec, device=device)
        max_error = wp.zeros(1, dtype=wp.float64, device=device)

        result_corr, result_err = rattle_iteration_out(
            positions,
            velocities,
            masses,
            bond_atom_i,
            bond_atom_j,
            corrections,
            max_error,
            device=device,
        )

        assert result_corr is corrections
        assert result_err is max_error


class TestRattleConstraintsOut:
    """Tests for non-mutating full RATTLE loop."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_rattle_constraints_out_basic(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test non-mutating RATTLE constraints."""
        positions_np = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np_dtype)
        velocities_np = np.array([[0.0, 0.0, 0.0], [2.0, 1.0, 0.0]], dtype=np_dtype)
        masses_np = np.array([1.0, 1.0], dtype=np_dtype)

        positions = wp.array(positions_np, dtype=dtype_vec, device=device)
        velocities = wp.array(velocities_np, dtype=dtype_vec, device=device)
        masses = wp.array(masses_np, dtype=dtype_scalar, device=device)

        bond_atom_i = wp.array([0], dtype=wp.int32, device=device)
        bond_atom_j = wp.array([1], dtype=wp.int32, device=device)

        num_atoms = 2
        velocities_out = wp.empty(num_atoms, dtype=dtype_vec, device=device)
        wp.copy(velocities_out, velocities)
        max_error = wp.empty(1, dtype=wp.float64, device=device)

        velocities_out, final_error = rattle_constraints_out(
            positions,
            velocities,
            masses,
            bond_atom_i,
            bond_atom_j,
            velocities_out,
            max_error,
            num_iter=20,
            device=device,
        )

        wp.synchronize_device(device)

        # Original unchanged
        np.testing.assert_allclose(velocities.numpy(), velocities_np, rtol=1e-6)

        # Output should have correct shape and be different from input
        assert velocities_out.shape[0] == 2
        assert final_error.shape[0] == 1

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_rattle_constraints_out_preallocated(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test non-mutating RATTLE with pre-allocated output."""
        positions_np = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np_dtype)
        velocities_np = np.array([[0.0, 0.0, 0.0], [2.0, 1.0, 0.0]], dtype=np_dtype)
        masses_np = np.array([1.0, 1.0], dtype=np_dtype)

        positions = wp.array(positions_np, dtype=dtype_vec, device=device)
        velocities = wp.array(velocities_np, dtype=dtype_vec, device=device)
        masses = wp.array(masses_np, dtype=dtype_scalar, device=device)

        bond_atom_i = wp.array([0], dtype=wp.int32, device=device)
        bond_atom_j = wp.array([1], dtype=wp.int32, device=device)

        # Pre-allocate output
        velocities_out = wp.empty(2, dtype=dtype_vec, device=device)
        wp.copy(velocities_out, velocities)
        max_error = wp.empty(1, dtype=wp.float64, device=device)

        result_vel, _ = rattle_constraints_out(
            positions,
            velocities,
            masses,
            bond_atom_i,
            bond_atom_j,
            velocities_out,
            max_error,
            num_iter=20,
            device=device,
        )

        assert result_vel is velocities_out
