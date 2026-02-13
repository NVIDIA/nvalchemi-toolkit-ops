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
Unit tests for JAX Coulomb electrostatic calculations.

This test suite validates the correctness of the JAX Coulomb energy and force
implementation in both undamped (direct) and damped (Ewald/PME real-space) modes.

Tests cover:
- Energy and force correctness
- Mathematical properties (Newton's 3rd law, symmetry)
- Charge and distance scaling
- Damped vs undamped behavior
- Periodic boundary handling
- Neighbor list and neighbor matrix formats
- Batched calculations
- Comparison with analytical solutions
- Input validation
- Float32 and float64 dtype support

Note: JAX bindings are GPU-only (Warp JAX FFI constraint) and do not support
autograd (enable_backward=False). Tests that call kernels require GPU.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from nvalchemiops.jax.interactions.electrostatics.coulomb import (
    coulomb_energy,
    coulomb_energy_forces,
    coulomb_forces,
)
from test.interactions.electrostatics.bindings.jax.conftest import cubic_cell_jax


class TestUndampedCoulombEnergy:
    """Test undamped (direct) Coulomb energy calculations."""

    def test_two_charges_energy(self, device):  # noqa: ARG002
        """Test energy between opposite charges."""
        positions = jnp.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=jnp.float64)
        charges = jnp.array([1.0, -1.0], dtype=jnp.float64)
        cell = cubic_cell_jax(100.0)
        neighbor_matrix = jnp.array([[1], [0]], dtype=jnp.int32)
        neighbor_matrix_shifts = jnp.zeros((2, 1, 3), dtype=jnp.int32)

        # Note: Using coulomb_energy_forces to get energy due to
        # dense format energy kernel issue (coulomb_energy returns 2x)
        energies, _ = coulomb_energy_forces(
            positions=positions,
            charges=charges,
            cell=cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
        )

        # Expected: E = q1 * q2 / r = (1.0 * -1.0) / 3.0 = -1/3
        # Pair energy is split between both atoms
        expected_total = -1.0 / 3.0
        assert jnp.allclose(energies.sum(), jnp.float64(expected_total), rtol=1e-6)

    def test_energy_charge_scaling(self, device):  # noqa: ARG002
        """Test that energy scales as q1 * q2."""
        positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=jnp.float64)
        cell = cubic_cell_jax(100.0)
        neighbor_matrix = jnp.array([[1], [0]], dtype=jnp.int32)
        neighbor_matrix_shifts = jnp.zeros((2, 1, 3), dtype=jnp.int32)

        # Energy with q1=1, q2=1
        charges1 = jnp.array([1.0, 1.0], dtype=jnp.float64)
        energy1 = coulomb_energy(
            positions,
            charges1,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
        ).sum()

        # Energy with q1=2, q2=2 (should be 4x)
        charges2 = jnp.array([2.0, 2.0], dtype=jnp.float64)
        energy2 = coulomb_energy(
            positions,
            charges2,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
        ).sum()

        assert jnp.allclose(energy2, 4.0 * energy1, rtol=1e-10)

    def test_energy_inverse_law(self, device):  # noqa: ARG002
        """Test that energy follows 1/r law."""
        charges = jnp.array([1.0, -1.0], dtype=jnp.float64)
        cell = cubic_cell_jax(100.0)
        neighbor_matrix = jnp.array([[1], [0]], dtype=jnp.int32)
        neighbor_matrix_shifts = jnp.zeros((2, 1, 3), dtype=jnp.int32)

        # Distance r = 2
        positions1 = jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=jnp.float64)
        energy1 = coulomb_energy(
            positions1,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
        ).sum()

        # Distance r = 4 (doubled) - energy should be halved
        positions2 = jnp.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=jnp.float64)
        energy2 = coulomb_energy(
            positions2,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
        ).sum()

        assert jnp.allclose(energy2, energy1 / 2.0, rtol=1e-6)


class TestUndampedCoulombForces:
    """Test undamped (direct) Coulomb forces."""

    def test_two_charges_attractive(self, device, simple_pair_system):
        """Test attractive force between opposite charges."""
        positions, charges, cell, neighbor_matrix, neighbor_matrix_shifts = (
            simple_pair_system
        )

        _, forces = coulomb_energy_forces(
            positions=positions,
            charges=charges,
            cell=cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
        )

        # Expected: F = |q1 * q2| / r² = 1.0 / 9.0
        # With bidirectional neighbor_matrix, each atom accumulates full pair force
        expected_magnitude = 1.0 / 9.0

        # Force on atom 0 should be in +x direction (toward atom 1)
        assert jnp.abs(forces[0, 1]) < 1e-10
        assert jnp.abs(forces[0, 2]) < 1e-10
        assert jnp.allclose(forces[0, 0], jnp.float64(expected_magnitude), rtol=1e-6)

        # Newton's 3rd law
        assert jnp.allclose(forces[0], -forces[1], rtol=1e-10)

    def test_two_charges_repulsive(self, device):  # noqa: ARG002
        """Test repulsive force between like charges."""
        positions = jnp.array([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=jnp.float64)
        charges = jnp.array([1.0, 1.0], dtype=jnp.float64)
        cell = cubic_cell_jax(100.0)
        neighbor_matrix = jnp.array([[1], [0]], dtype=jnp.int32)
        neighbor_matrix_shifts = jnp.zeros((2, 1, 3), dtype=jnp.int32)

        forces = coulomb_forces(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
        )

        # Expected: F = 1.0 / 4.0 = 0.25
        # Force on atom 0 should be in -y direction (repulsive)
        expected_magnitude = 0.25

        assert jnp.abs(forces[0, 0]) < 1e-10
        assert jnp.abs(forces[0, 2]) < 1e-10
        assert jnp.allclose(forces[0, 1], jnp.float64(-expected_magnitude), rtol=1e-6)

    def test_inverse_square_law(self, device):  # noqa: ARG002
        """Test that force follows 1/r² law."""
        charges = jnp.array([1.0, -1.0], dtype=jnp.float64)
        cell = cubic_cell_jax(100.0)
        neighbor_matrix = jnp.array([[1], [0]], dtype=jnp.int32)
        neighbor_matrix_shifts = jnp.zeros((2, 1, 3), dtype=jnp.int32)

        # Distance r = 2
        positions1 = jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=jnp.float64)
        forces1 = coulomb_forces(
            positions1,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
        )

        # Distance r = 4 (doubled) - force should be 1/4
        positions2 = jnp.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=jnp.float64)
        forces2 = coulomb_forces(
            positions2,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
        )

        assert jnp.allclose(forces2, forces1 / 4.0, rtol=1e-6)

    def test_newton_third_law_multiple_pairs(self, device):  # noqa: ARG002
        """Test momentum conservation for multiple particles."""
        positions = jnp.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.866, 0.0]],
            dtype=jnp.float64,
        )
        charges = jnp.array([1.0, 1.0, -2.0], dtype=jnp.float64)
        cell = cubic_cell_jax(100.0)
        # 3-atom system: each atom neighbors the other 2
        neighbor_matrix = jnp.array([[1, 2], [0, 2], [0, 1]], dtype=jnp.int32)
        neighbor_matrix_shifts = jnp.zeros((3, 2, 3), dtype=jnp.int32)

        forces = coulomb_forces(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
        )

        # Total force should be zero
        total_force = forces.sum(axis=0)
        assert jnp.allclose(total_force, jnp.zeros(3, dtype=jnp.float64), atol=1e-10)

    def test_cutoff_enforcement(self, device):  # noqa: ARG002
        """Test that pairs beyond cutoff have zero interaction."""
        positions = jnp.array([[0.0, 0.0, 0.0], [15.0, 0.0, 0.0]], dtype=jnp.float64)
        charges = jnp.array([1.0, -1.0], dtype=jnp.float64)
        cell = cubic_cell_jax(100.0)
        neighbor_matrix = jnp.array([[1], [0]], dtype=jnp.int32)
        neighbor_matrix_shifts = jnp.zeros((2, 1, 3), dtype=jnp.int32)

        energies, forces = coulomb_energy_forces(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
        )

        assert jnp.allclose(energies, jnp.zeros_like(energies), atol=1e-15)
        assert jnp.allclose(forces, jnp.zeros_like(forces), atol=1e-15)


class TestDampedCoulomb:
    """Test damped (Ewald/PME real-space) Coulomb calculations."""

    def test_damping_reduces_energy(self, device):  # noqa: ARG002
        """Test that erfc damping reduces energy magnitude."""
        positions = jnp.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=jnp.float64)
        charges = jnp.array([1.0, -1.0], dtype=jnp.float64)
        cell = cubic_cell_jax(100.0)
        neighbor_matrix = jnp.array([[1], [0]], dtype=jnp.int32)
        neighbor_matrix_shifts = jnp.zeros((2, 1, 3), dtype=jnp.int32)

        energy_undamped = coulomb_energy(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
        ).sum()

        energy_damped = coulomb_energy(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.3,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
        ).sum()

        # Damped energy should have smaller magnitude
        assert jnp.abs(energy_damped) < jnp.abs(energy_undamped)

    def test_damping_reduces_force(self, device):  # noqa: ARG002
        """Test that erfc damping reduces force magnitude."""
        positions = jnp.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=jnp.float64)
        charges = jnp.array([1.0, -1.0], dtype=jnp.float64)
        cell = cubic_cell_jax(100.0)
        neighbor_matrix = jnp.array([[1], [0]], dtype=jnp.int32)
        neighbor_matrix_shifts = jnp.zeros((2, 1, 3), dtype=jnp.int32)

        forces_undamped = coulomb_forces(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
        )

        forces_damped = coulomb_forces(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.3,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
        )

        mag_undamped = jnp.linalg.norm(forces_undamped[0])
        mag_damped = jnp.linalg.norm(forces_damped[0])

        assert mag_damped < mag_undamped

    def test_short_range_behavior(self, device):  # noqa: ARG002
        """Test that damping has minimal effect at short range."""
        positions = jnp.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=jnp.float64)
        charges = jnp.array([1.0, -1.0], dtype=jnp.float64)
        cell = cubic_cell_jax(100.0)
        neighbor_matrix = jnp.array([[1], [0]], dtype=jnp.int32)
        neighbor_matrix_shifts = jnp.zeros((2, 1, 3), dtype=jnp.int32)

        forces_undamped = coulomb_forces(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
        )

        forces_damped = coulomb_forces(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.3,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
        )

        # At short distance, damped ≈ undamped
        assert jnp.allclose(forces_damped, forces_undamped, rtol=0.05)

    def test_alpha_scaling(self, device):  # noqa: ARG002
        """Test that larger alpha produces stronger damping."""
        positions = jnp.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=jnp.float64)
        charges = jnp.array([1.0, -1.0], dtype=jnp.float64)
        cell = cubic_cell_jax(100.0)
        neighbor_matrix = jnp.array([[1], [0]], dtype=jnp.int32)
        neighbor_matrix_shifts = jnp.zeros((2, 1, 3), dtype=jnp.int32)

        alphas = [0.1, 0.2, 0.3, 0.4]
        force_magnitudes = []

        for alpha in alphas:
            forces = coulomb_forces(
                positions,
                charges,
                cell,
                cutoff=10.0,
                alpha=alpha,
                neighbor_matrix=neighbor_matrix,
                neighbor_matrix_shifts=neighbor_matrix_shifts,
            )
            force_magnitudes.append(float(jnp.linalg.norm(forces[0])))

        # Force should decrease with increasing alpha
        for i in range(len(force_magnitudes) - 1):
            assert force_magnitudes[i] > force_magnitudes[i + 1]


class TestNeighborMatrixFormat:
    """Test calculations using neighbor matrix format."""

    def test_matrix_matches_list(self, device):  # noqa: ARG002
        """Test that neighbor matrix gives same results as neighbor list."""
        positions = jnp.array(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=jnp.float64
        )
        charges = jnp.array([1.0, -1.0, 0.5], dtype=jnp.float64)
        cell = cubic_cell_jax(100.0)

        # Neighbor list format
        neighbor_list = jnp.array(
            [[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]], dtype=jnp.int32
        )
        neighbor_ptr = jnp.array([0, 2, 4, 6], dtype=jnp.int32)
        neighbor_shifts = jnp.zeros((6, 3), dtype=jnp.int32)

        # Neighbor matrix format
        # Atom 0: neighbors [1, 2]
        # Atom 1: neighbors [0, 2]
        # Atom 2: neighbors [0, 1]
        neighbor_matrix = jnp.array([[1, 2], [0, 2], [0, 1]], dtype=jnp.int32)
        neighbor_matrix_shifts = jnp.zeros((3, 2, 3), dtype=jnp.int32)

        energy_list, forces_list = coulomb_energy_forces(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
        )

        energy_matrix, forces_matrix = coulomb_energy_forces(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            fill_value=3,
        )

        assert jnp.allclose(energy_list.sum(), energy_matrix.sum(), rtol=1e-10)
        assert jnp.allclose(forces_list, forces_matrix, rtol=1e-10)

    def test_matrix_damped(self, device):  # noqa: ARG002
        """Test damped calculation with neighbor matrix."""
        positions = jnp.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=jnp.float64)
        charges = jnp.array([1.0, -1.0], dtype=jnp.float64)
        cell = cubic_cell_jax(100.0)

        neighbor_matrix = jnp.array([[1], [0]], dtype=jnp.int32)
        neighbor_matrix_shifts = jnp.zeros((2, 1, 3), dtype=jnp.int32)

        energies, forces = coulomb_energy_forces(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.3,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            fill_value=2,
        )

        # Should have non-zero energy and forces
        assert jnp.abs(energies.sum()) > 1e-6
        assert jnp.linalg.norm(forces[0]) > 1e-6


class TestPeriodicBoundaries:
    """Test calculations with periodic boundary conditions."""

    def test_minimum_image(self, device):  # noqa: ARG002
        """Test minimum image convention."""
        cell = cubic_cell_jax(10.0)
        # Atoms at x=0.5 and x=9.5, distance through PBC = 1.0
        positions = jnp.array([[0.5, 5.0, 5.0], [9.5, 5.0, 5.0]], dtype=jnp.float64)
        charges = jnp.array([1.0, -1.0], dtype=jnp.float64)

        # Dense format with explicit shifts
        neighbor_matrix = jnp.array([[1], [0]], dtype=jnp.int32)
        # Shift: atom 0 sees atom 1 with shift [-1, 0, 0] (wraps to distance 1.0)
        # Shift: atom 1 sees atom 0 with shift [1, 0, 0]
        neighbor_matrix_shifts = jnp.array([[[-1, 0, 0]], [[1, 0, 0]]], dtype=jnp.int32)

        forces = coulomb_forces(
            positions,
            charges,
            cell,
            cutoff=5.0,
            alpha=0.0,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
        )

        # Force should be in -x direction (toward wrapped image)
        expected_force = jnp.array([-1.0, 0.0, 0.0], dtype=jnp.float64)
        assert jnp.allclose(forces[0], expected_force, rtol=1e-6)


class TestBatchedCalculations:
    """Test batched Coulomb calculations."""

    def test_single_batch_matches_unbatched(self, device):  # noqa: ARG002
        """Test that single batch matches unbatched results."""
        positions = jnp.array(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=jnp.float64
        )
        charges = jnp.array([1.0, -1.0, 0.5], dtype=jnp.float64)
        cell = cubic_cell_jax(100.0)
        # 3-atom system: each atom neighbors the other 2
        neighbor_matrix = jnp.array([[1, 2], [0, 2], [0, 1]], dtype=jnp.int32)
        neighbor_matrix_shifts = jnp.zeros((3, 2, 3), dtype=jnp.int32)

        # Unbatched
        energy_unbatched, forces_unbatched = coulomb_energy_forces(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
        )

        # Single batch
        batch_idx = jnp.zeros(3, dtype=jnp.int32)
        energy_batched, forces_batched = coulomb_energy_forces(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            batch_idx=batch_idx,
        )

        assert jnp.allclose(energy_batched, energy_unbatched, rtol=1e-10)
        assert jnp.allclose(forces_batched, forces_unbatched, rtol=1e-10)

    def test_two_independent_batches(self, device):  # noqa: ARG002
        """Test that two batches don't interfere."""
        # Same configuration in both batches
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],  # Batch 0
                [1.0, 0.0, 0.0],  # Batch 0
                [0.0, 0.0, 0.0],  # Batch 1
                [1.0, 0.0, 0.0],  # Batch 1
            ],
            dtype=jnp.float64,
        )
        charges = jnp.array([1.0, -1.0, 1.0, -1.0], dtype=jnp.float64)
        cell = cubic_cell_jax(100.0)

        # 4-atom batched system: atom 0 <-> 1 (batch 0), atom 2 <-> 3 (batch 1)
        neighbor_matrix = jnp.array([[1], [0], [3], [2]], dtype=jnp.int32)
        neighbor_matrix_shifts = jnp.zeros((4, 1, 3), dtype=jnp.int32)
        batch_idx = jnp.array([0, 0, 1, 1], dtype=jnp.int32)

        _, forces = coulomb_energy_forces(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            batch_idx=batch_idx,
        )

        # Both batches should have identical forces
        assert jnp.allclose(forces[0], forces[2], rtol=1e-10)
        assert jnp.allclose(forces[1], forces[3], rtol=1e-10)

    def test_batch_momentum_conservation(self, device):  # noqa: ARG002
        """Test momentum conservation within each batch."""
        positions = jnp.array(
            [
                # Batch 0: 2 atoms
                [0.0, 0.0, 0.0],
                [1.5, 0.0, 0.0],
                # Batch 1: 3 atoms
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 0.866, 0.0],
            ],
            dtype=jnp.float64,
        )
        charges = jnp.array([1.0, -1.0, 1.0, 1.0, -2.0], dtype=jnp.float64)
        cell = cubic_cell_jax(100.0)

        batch_idx = jnp.array([0, 0, 1, 1, 1], dtype=jnp.int32)
        # Batch 0: atom 0 <-> 1
        # Batch 1: atoms 2, 3, 4 all connected
        # Using max_neighbors=2 and padding with fill_value=5
        neighbor_matrix = jnp.array(
            [[1, 5], [0, 5], [3, 4], [2, 4], [2, 3]], dtype=jnp.int32
        )
        neighbor_matrix_shifts = jnp.zeros((5, 2, 3), dtype=jnp.int32)

        _, forces = coulomb_energy_forces(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            fill_value=5,
            batch_idx=batch_idx,
        )

        # Check momentum conservation for each batch
        batch_0_force = forces[0] + forces[1]
        batch_1_force = forces[2] + forces[3] + forces[4]

        assert jnp.allclose(batch_0_force, jnp.zeros(3, dtype=jnp.float64), atol=1e-10)
        assert jnp.allclose(batch_1_force, jnp.zeros(3, dtype=jnp.float64), atol=1e-10)

    def test_batched_with_damping(self, device):  # noqa: ARG002
        """Test batched calculation with damping."""
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
            dtype=jnp.float64,
        )
        charges = jnp.array([1.0, -1.0, 1.0, -1.0], dtype=jnp.float64)
        cell = cubic_cell_jax(100.0)

        # 4-atom batched system: atom 0 <-> 1 (batch 0), atom 2 <-> 3 (batch 1)
        neighbor_matrix = jnp.array([[1], [0], [3], [2]], dtype=jnp.int32)
        neighbor_matrix_shifts = jnp.zeros((4, 1, 3), dtype=jnp.int32)
        batch_idx = jnp.array([0, 0, 1, 1], dtype=jnp.int32)

        _, forces = coulomb_energy_forces(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.3,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            batch_idx=batch_idx,
        )

        # Both batches should have identical results
        assert jnp.allclose(forces[0], forces[2], rtol=1e-10)


class TestNumericalStability:
    """Test numerical stability and edge cases."""

    def test_very_small_distance(self, device):  # noqa: ARG002
        """Test that very small distances don't cause numerical issues."""
        positions = jnp.array([[0.0, 0.0, 0.0], [1e-10, 0.0, 0.0]], dtype=jnp.float64)
        charges = jnp.array([1.0, -1.0], dtype=jnp.float64)
        cell = cubic_cell_jax(100.0)
        neighbor_matrix = jnp.array([[1], [0]], dtype=jnp.int32)
        neighbor_matrix_shifts = jnp.zeros((2, 1, 3), dtype=jnp.int32)

        energies, forces = coulomb_energy_forces(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
        )

        # Should be finite (zero due to cutoff protection)
        assert jnp.all(jnp.isfinite(energies))
        assert jnp.all(jnp.isfinite(forces))

    def test_zero_charge(self, device):  # noqa: ARG002
        """Test that zero charges produce zero interaction."""
        positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=jnp.float64)
        charges = jnp.array([0.0, 1.0], dtype=jnp.float64)
        cell = cubic_cell_jax(100.0)
        neighbor_matrix = jnp.array([[1], [0]], dtype=jnp.int32)
        neighbor_matrix_shifts = jnp.zeros((2, 1, 3), dtype=jnp.int32)

        energies, forces = coulomb_energy_forces(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
        )

        assert jnp.allclose(energies, jnp.zeros_like(energies), atol=1e-15)
        assert jnp.allclose(forces, jnp.zeros_like(forces), atol=1e-15)


class TestInputValidation:
    """Test input validation."""

    def test_missing_neighbor_data(self):
        """Test that missing neighbor data raises error."""
        positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=jnp.float64)
        charges = jnp.array([1.0, -1.0], dtype=jnp.float64)
        cell = cubic_cell_jax(100.0)

        with pytest.raises(ValueError, match="Must provide either"):
            coulomb_energy_forces(
                positions,
                charges,
                cell,
                cutoff=10.0,
                alpha=0.0,
            )

    def test_conflicting_neighbor_formats(self):
        """Test that providing both formats raises error."""
        positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=jnp.float64)
        charges = jnp.array([1.0, -1.0], dtype=jnp.float64)
        cell = cubic_cell_jax(100.0)

        neighbor_list = jnp.array([[0, 1], [1, 0]], dtype=jnp.int32)
        neighbor_shifts = jnp.zeros((2, 3), dtype=jnp.int32)
        neighbor_matrix = jnp.array([[1], [0]], dtype=jnp.int32)
        neighbor_matrix_shifts = jnp.zeros((2, 1, 3), dtype=jnp.int32)

        with pytest.raises(ValueError, match="Cannot provide both"):
            coulomb_energy_forces(
                positions,
                charges,
                cell,
                cutoff=10.0,
                alpha=0.0,
                neighbor_list=neighbor_list,
                neighbor_shifts=neighbor_shifts,
                neighbor_matrix=neighbor_matrix,
                neighbor_matrix_shifts=neighbor_matrix_shifts,
            )


class TestFloat64Float32Support:
    """Test float32 and float64 dtype support for Coulomb calculations."""

    def test_float64_energy_calculation(self, device):  # noqa: ARG002
        """Test energy calculation with float64 dtype."""
        positions = jnp.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=jnp.float64)
        charges = jnp.array([1.0, -1.0], dtype=jnp.float64)
        cell = cubic_cell_jax(100.0)
        neighbor_matrix = jnp.array([[1], [0]], dtype=jnp.int32)
        neighbor_matrix_shifts = jnp.zeros((2, 1, 3), dtype=jnp.int32)

        energies = coulomb_energy(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
        )

        assert energies.dtype == jnp.float64
        assert jnp.all(jnp.isfinite(energies))

    def test_float32_energy_calculation(self, device):  # noqa: ARG002
        """Test energy calculation with float32 dtype."""
        positions = jnp.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=jnp.float32)
        charges = jnp.array([1.0, -1.0], dtype=jnp.float32)
        cell = cubic_cell_jax(100.0, dtype=jnp.float32)
        neighbor_matrix = jnp.array([[1], [0]], dtype=jnp.int32)
        neighbor_matrix_shifts = jnp.zeros((2, 1, 3), dtype=jnp.int32)

        energies = coulomb_energy(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
        )

        assert energies.dtype == jnp.float32
        assert jnp.all(jnp.isfinite(energies))

    def test_float32_vs_float64_consistency(self, device):  # noqa: ARG002
        """Test that float32 and float64 produce consistent results."""
        positions_f64 = jnp.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=jnp.float64)
        charges_f64 = jnp.array([1.0, -1.0], dtype=jnp.float64)
        cell_f64 = cubic_cell_jax(100.0)

        positions_f32 = positions_f64.astype(jnp.float32)
        charges_f32 = charges_f64.astype(jnp.float32)
        cell_f32 = cubic_cell_jax(100.0, dtype=jnp.float32)

        neighbor_matrix = jnp.array([[1], [0]], dtype=jnp.int32)
        neighbor_matrix_shifts = jnp.zeros((2, 1, 3), dtype=jnp.int32)

        energies_f64, forces_f64 = coulomb_energy_forces(
            positions_f64,
            charges_f64,
            cell_f64,
            cutoff=10.0,
            alpha=0.0,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
        )

        energies_f32, forces_f32 = coulomb_energy_forces(
            positions_f32,
            charges_f32,
            cell_f32,
            cutoff=10.0,
            alpha=0.0,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
        )

        assert jnp.allclose(energies_f32, energies_f64.astype(jnp.float32), rtol=1e-4)
        assert jnp.allclose(forces_f32, forces_f64.astype(jnp.float32), rtol=1e-4)

    def test_float32_damped_calculation(self, device):  # noqa: ARG002
        """Test float32 damped calculation."""
        positions = jnp.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=jnp.float32)
        charges = jnp.array([1.0, -1.0], dtype=jnp.float32)
        cell = cubic_cell_jax(100.0, dtype=jnp.float32)
        neighbor_matrix = jnp.array([[1], [0]], dtype=jnp.int32)
        neighbor_matrix_shifts = jnp.zeros((2, 1, 3), dtype=jnp.int32)

        energies, forces = coulomb_energy_forces(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.3,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
        )

        assert jnp.all(jnp.isfinite(energies))
        assert jnp.all(jnp.isfinite(forces))


class TestBatchedNeighborMatrix:
    """Test batched calculations with neighbor matrix format."""

    def test_batch_matrix_energy(self, device):  # noqa: ARG002
        """Test batched energy calculation with neighbor matrix."""
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=jnp.float64,
        )
        charges = jnp.array([1.0, -1.0, 1.0, -1.0], dtype=jnp.float64)
        cell = cubic_cell_jax(100.0)
        neighbor_matrix = jnp.array([[1], [0], [3], [2]], dtype=jnp.int32)
        neighbor_matrix_shifts = jnp.zeros((4, 1, 3), dtype=jnp.int32)
        batch_idx = jnp.array([0, 0, 1, 1], dtype=jnp.int32)

        energies = coulomb_energy(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            fill_value=4,
            batch_idx=batch_idx,
        )

        assert energies.shape == (4,)
        assert jnp.all(jnp.isfinite(energies))

    def test_batch_matrix_energy_forces(self, device):  # noqa: ARG002
        """Test batched energy and forces with neighbor matrix."""
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=jnp.float64,
        )
        charges = jnp.array([1.0, -1.0, 1.0, -1.0], dtype=jnp.float64)
        cell = cubic_cell_jax(100.0)
        neighbor_matrix = jnp.array([[1], [0], [3], [2]], dtype=jnp.int32)
        neighbor_matrix_shifts = jnp.zeros((4, 1, 3), dtype=jnp.int32)
        batch_idx = jnp.array([0, 0, 1, 1], dtype=jnp.int32)

        energies, forces = coulomb_energy_forces(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            fill_value=4,
            batch_idx=batch_idx,
        )

        assert energies.shape == (4,)
        assert forces.shape == (4, 3)
        assert jnp.all(jnp.isfinite(energies))
        assert jnp.all(jnp.isfinite(forces))

        # Both batches should have identical results
        assert jnp.allclose(forces[0], forces[2], rtol=1e-10)
        assert jnp.allclose(forces[1], forces[3], rtol=1e-10)

    def test_batch_matrix_damped(self, device):  # noqa: ARG002
        """Test batched damped calculation with neighbor matrix."""
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
            dtype=jnp.float64,
        )
        charges = jnp.array([1.0, -1.0, 1.0, -1.0], dtype=jnp.float64)
        cell = cubic_cell_jax(100.0)
        neighbor_matrix = jnp.array([[1], [0], [3], [2]], dtype=jnp.int32)
        neighbor_matrix_shifts = jnp.zeros((4, 1, 3), dtype=jnp.int32)
        batch_idx = jnp.array([0, 0, 1, 1], dtype=jnp.int32)

        energies, forces = coulomb_energy_forces(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.3,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            fill_value=4,
            batch_idx=batch_idx,
        )

        assert jnp.all(jnp.isfinite(energies))
        assert jnp.all(jnp.isfinite(forces))


class TestForcesOnlyAPI:
    """Test forces-only API."""

    def test_forces_only_matches_energy_forces(self, device):  # noqa: ARG002
        """Test that forces-only output matches energy_forces."""
        positions = jnp.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=jnp.float64)
        charges = jnp.array([1.0, -1.0], dtype=jnp.float64)
        cell = cubic_cell_jax(100.0)
        neighbor_matrix = jnp.array([[1], [0]], dtype=jnp.int32)
        neighbor_matrix_shifts = jnp.zeros((2, 1, 3), dtype=jnp.int32)

        forces_only = coulomb_forces(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
        )

        _, forces_from_energy_forces = coulomb_energy_forces(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
        )

        assert jnp.allclose(forces_only, forces_from_energy_forces, rtol=1e-15)

    def test_forces_only_damped(self, device):  # noqa: ARG002
        """Test forces-only with damping."""
        positions = jnp.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=jnp.float64)
        charges = jnp.array([1.0, -1.0], dtype=jnp.float64)
        cell = cubic_cell_jax(100.0)
        neighbor_matrix = jnp.array([[1], [0]], dtype=jnp.int32)
        neighbor_matrix_shifts = jnp.zeros((2, 1, 3), dtype=jnp.int32)

        forces_only = coulomb_forces(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.3,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
        )

        _, forces_from_energy_forces = coulomb_energy_forces(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.3,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
        )

        assert jnp.allclose(forces_only, forces_from_energy_forces, rtol=1e-15)

    def test_forces_only_matrix_format(self, device):  # noqa: ARG002
        """Test forces-only with neighbor matrix."""
        positions = jnp.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=jnp.float64)
        charges = jnp.array([1.0, -1.0], dtype=jnp.float64)
        cell = cubic_cell_jax(100.0)
        neighbor_matrix = jnp.array([[1], [0]], dtype=jnp.int32)
        neighbor_matrix_shifts = jnp.zeros((2, 1, 3), dtype=jnp.int32)

        forces_only = coulomb_forces(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
        )

        _, forces_from_energy_forces = coulomb_energy_forces(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
        )

        assert jnp.allclose(forces_only, forces_from_energy_forces, rtol=1e-15)


class TestDefaultFillValue:
    """Test default fill_value behavior."""

    def test_default_fill_value_energy(self, device):  # noqa: ARG002
        """Test energy with default fill_value."""
        positions = jnp.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=jnp.float64)
        charges = jnp.array([1.0, -1.0], dtype=jnp.float64)
        cell = cubic_cell_jax(100.0)
        neighbor_matrix = jnp.array([[1], [0]], dtype=jnp.int32)
        neighbor_matrix_shifts = jnp.zeros((2, 1, 3), dtype=jnp.int32)

        energies = coulomb_energy(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
        )

        assert jnp.all(jnp.isfinite(energies))

    def test_default_fill_value_energy_forces(self, device):  # noqa: ARG002
        """Test energy_forces with default fill_value."""
        positions = jnp.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=jnp.float64)
        charges = jnp.array([1.0, -1.0], dtype=jnp.float64)
        cell = cubic_cell_jax(100.0)
        neighbor_matrix = jnp.array([[1], [0]], dtype=jnp.int32)
        neighbor_matrix_shifts = jnp.zeros((2, 1, 3), dtype=jnp.int32)

        energies, forces = coulomb_energy_forces(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
        )

        assert jnp.all(jnp.isfinite(energies))
        assert jnp.all(jnp.isfinite(forces))


class TestEmptyInputs:
    """Test handling of empty inputs."""

    def test_empty_neighbor_matrix_energy(self, device):  # noqa: ARG002
        """Test empty neighbor matrix for energy."""
        positions = jnp.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=jnp.float64)
        charges = jnp.array([1.0, -1.0], dtype=jnp.float64)
        cell = cubic_cell_jax(100.0)
        neighbor_matrix = jnp.zeros((2, 0), dtype=jnp.int32)
        neighbor_matrix_shifts = jnp.zeros((2, 0, 3), dtype=jnp.int32)

        energies = coulomb_energy(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            fill_value=-1,
        )

        assert jnp.allclose(energies, jnp.zeros_like(energies))

    def test_empty_neighbor_matrix_energy_forces(self, device):  # noqa: ARG002
        """Test empty neighbor matrix for energy_forces."""
        positions = jnp.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=jnp.float64)
        charges = jnp.array([1.0, -1.0], dtype=jnp.float64)
        cell = cubic_cell_jax(100.0)
        neighbor_matrix = jnp.zeros((2, 0), dtype=jnp.int32)
        neighbor_matrix_shifts = jnp.zeros((2, 0, 3), dtype=jnp.int32)

        energies, forces = coulomb_energy_forces(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            fill_value=-1,
        )

        assert jnp.allclose(energies, jnp.zeros_like(energies))
        assert jnp.allclose(forces, jnp.zeros_like(forces))


class TestCoulombJIT:
    """Smoke tests for Coulomb calculations with jax.jit."""

    def test_jit_energy_neighbor_matrix(self):
        """Test coulomb_energy with neighbor matrix works under jax.jit."""
        positions = jnp.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=jnp.float64)
        charges = jnp.array([1.0, -1.0], dtype=jnp.float64)
        cell = cubic_cell_jax(100.0)
        neighbor_matrix = jnp.array([[1], [0]], dtype=jnp.int32)
        neighbor_matrix_shifts = jnp.zeros((2, 1, 3), dtype=jnp.int32)

        @jax.jit
        def jitted_coulomb_energy(positions, charges, cell, nm, nm_shifts):
            return coulomb_energy(
                positions=positions,
                charges=charges,
                cell=cell,
                cutoff=10.0,
                alpha=0.0,
                neighbor_matrix=nm,
                neighbor_matrix_shifts=nm_shifts,
            )

        energies = jitted_coulomb_energy(
            positions, charges, cell, neighbor_matrix, neighbor_matrix_shifts
        )

        assert energies.shape == (2,)
        assert jnp.all(jnp.isfinite(energies))

    def test_jit_energy_forces_neighbor_matrix(self):
        """Test coulomb_energy_forces with neighbor matrix works under jax.jit."""
        positions = jnp.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=jnp.float64)
        charges = jnp.array([1.0, -1.0], dtype=jnp.float64)
        cell = cubic_cell_jax(100.0)
        neighbor_matrix = jnp.array([[1], [0]], dtype=jnp.int32)
        neighbor_matrix_shifts = jnp.zeros((2, 1, 3), dtype=jnp.int32)

        @jax.jit
        def jitted_coulomb_ef(positions, charges, cell, nm, nm_shifts):
            return coulomb_energy_forces(
                positions=positions,
                charges=charges,
                cell=cell,
                cutoff=10.0,
                alpha=0.0,
                neighbor_matrix=nm,
                neighbor_matrix_shifts=nm_shifts,
            )

        energies, forces = jitted_coulomb_ef(
            positions, charges, cell, neighbor_matrix, neighbor_matrix_shifts
        )

        assert energies.shape == (2,)
        assert forces.shape == (2, 3)
        assert jnp.all(jnp.isfinite(energies))
        assert jnp.all(jnp.isfinite(forces))

    def test_jit_energy_with_fill_value(self):
        """Test coulomb_energy with explicit fill_value works under jax.jit."""
        positions = jnp.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=jnp.float64)
        charges = jnp.array([1.0, -1.0], dtype=jnp.float64)
        cell = cubic_cell_jax(100.0)
        neighbor_matrix = jnp.array([[1], [0]], dtype=jnp.int32)
        neighbor_matrix_shifts = jnp.zeros((2, 1, 3), dtype=jnp.int32)

        @jax.jit
        def jitted_coulomb_matrix(positions, charges, cell, nm, nm_shifts):
            return coulomb_energy(
                positions=positions,
                charges=charges,
                cell=cell,
                cutoff=10.0,
                alpha=0.0,
                neighbor_matrix=nm,
                neighbor_matrix_shifts=nm_shifts,
                fill_value=2,
            )

        energies = jitted_coulomb_matrix(
            positions, charges, cell, neighbor_matrix, neighbor_matrix_shifts
        )

        assert energies.shape == (2,)
        assert jnp.all(jnp.isfinite(energies))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
