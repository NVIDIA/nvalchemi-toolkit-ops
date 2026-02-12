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
Unit tests for Coulomb warp kernel launchers.

This test suite validates the correctness of the Coulomb warp launchers
(coulomb_energy, coulomb_energy_forces, etc.) directly using warp arrays.

Tests cover:
- Undamped Coulomb energy (alpha=0)
- Damped Coulomb energy (alpha>0, Ewald real-space)
- Energy and force consistency
- Both neighbor list (CSR) and neighbor matrix formats
- Batched calculations
- Regression tests with hardcoded expected values

These tests use warp arrays directly and do not require PyTorch.
For PyTorch binding tests, see test/interactions/electrostatics/bindings/torch/test_coulomb.py
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from nvalchemiops.interactions.electrostatics.coulomb import (
    batch_coulomb_energy,
    batch_coulomb_energy_forces,
    batch_coulomb_energy_forces_matrix,
    batch_coulomb_energy_matrix,
    coulomb_energy,
    coulomb_energy_forces,
    coulomb_energy_forces_matrix,
    coulomb_energy_matrix,
)

# ==============================================================================
# Test Fixtures
# ==============================================================================


@pytest.fixture(scope="session")
def two_atom_system():
    """Simple two-atom system for basic Coulomb tests.

    Two atoms along x-axis:
    - Atom 0 at origin with charge +1
    - Atom 1 at (3, 0, 0) with charge -1

    Distance r = 3.0
    Expected undamped energy: E = q1*q2/r = 1*(-1)/3 = -1/3

    Returns
    -------
    dict
        System parameters (numpy arrays)
    """
    positions = np.array(
        [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    charges = np.array([1.0, -1.0], dtype=np.float64)
    # Large cell to avoid periodic interactions
    cell = np.array(
        [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
        dtype=np.float64,
    )
    # CSR format: atom 0 has neighbor 1, atom 1 has neighbor 0
    idx_j = np.array([1, 0], dtype=np.int32)
    neighbor_ptr = np.array([0, 1, 2], dtype=np.int32)
    unit_shifts = np.array([[0, 0, 0], [0, 0, 0]], dtype=np.int32)

    return {
        "positions": positions,
        "charges": charges,
        "cell": cell,
        "idx_j": idx_j,
        "neighbor_ptr": neighbor_ptr,
        "unit_shifts": unit_shifts,
        "num_atoms": 2,
        "distance": 3.0,
    }


@pytest.fixture(scope="session")
def two_atom_matrix_system():
    """Two-atom system in neighbor matrix format.

    Returns
    -------
    dict
        System parameters (numpy arrays)
    """
    positions = np.array(
        [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    charges = np.array([1.0, -1.0], dtype=np.float64)
    cell = np.array(
        [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
        dtype=np.float64,
    )
    # Neighbor matrix: shape (N, max_neighbors)
    # atom 0: neighbor 1; atom 1: neighbor 0
    # Use fill_value = 999 for padding
    neighbor_matrix = np.array([[1, 999], [0, 999]], dtype=np.int32)
    neighbor_shifts = np.array(
        [[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]], dtype=np.int32
    )

    return {
        "positions": positions,
        "charges": charges,
        "cell": cell,
        "neighbor_matrix": neighbor_matrix,
        "neighbor_shifts": neighbor_shifts,
        "fill_value": 999,
        "num_atoms": 2,
        "distance": 3.0,
    }


@pytest.fixture(scope="session")
def three_atom_system():
    """Three-atom linear system for testing multiple interactions.

    Atoms along x-axis:
    - Atom 0 at (0, 0, 0) with charge +1
    - Atom 1 at (2, 0, 0) with charge -1
    - Atom 2 at (5, 0, 0) with charge +0.5

    Distances: r01=2, r02=5, r12=3
    Expected undamped energies:
    - E_01 = 1*(-1)/2 = -0.5
    - E_02 = 1*0.5/5 = 0.1
    - E_12 = (-1)*0.5/3 = -1/6
    - Total = -0.5 + 0.1 - 1/6 = -0.56666...

    Returns
    -------
    dict
        System parameters (numpy arrays)
    """
    positions = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    charges = np.array([1.0, -1.0, 0.5], dtype=np.float64)
    cell = np.array(
        [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
        dtype=np.float64,
    )
    # CSR format (half neighbor list - each pair once)
    # atom 0: neighbors 1, 2
    # atom 1: neighbor 2
    # atom 2: no neighbors (all pairs already counted)
    idx_j = np.array([1, 2, 2], dtype=np.int32)
    neighbor_ptr = np.array([0, 2, 3, 3], dtype=np.int32)
    unit_shifts = np.zeros((3, 3), dtype=np.int32)

    return {
        "positions": positions,
        "charges": charges,
        "cell": cell,
        "idx_j": idx_j,
        "neighbor_ptr": neighbor_ptr,
        "unit_shifts": unit_shifts,
        "num_atoms": 3,
    }


@pytest.fixture(scope="session")
def batch_two_systems():
    """Two independent two-atom systems in a batch.

    System 0: Atoms at (0,0,0) and (2,0,0), charges +1 and -1
    System 1: Atoms at (0,0,0) and (4,0,0), charges +2 and -1

    Returns
    -------
    dict
        Batched system parameters (numpy arrays)
    """
    # Concatenated positions
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],  # System 0
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],  # System 1
        ],
        dtype=np.float64,
    )
    charges = np.array([1.0, -1.0, 2.0, -1.0], dtype=np.float64)
    # Per-system cell matrices
    cell = np.array(
        [
            [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]],
            [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]],
        ],
        dtype=np.float64,
    )
    # CSR format (half neighbor list)
    # Atom 0 (sys 0): neighbor 1
    # Atom 1 (sys 0): no neighbors
    # Atom 2 (sys 1): neighbor 3
    # Atom 3 (sys 1): no neighbors
    idx_j = np.array([1, 3], dtype=np.int32)
    neighbor_ptr = np.array([0, 1, 1, 2, 2], dtype=np.int32)
    unit_shifts = np.zeros((2, 3), dtype=np.int32)
    batch_idx = np.array([0, 0, 1, 1], dtype=np.int32)

    return {
        "positions": positions,
        "charges": charges,
        "cell": cell,
        "idx_j": idx_j,
        "neighbor_ptr": neighbor_ptr,
        "unit_shifts": unit_shifts,
        "batch_idx": batch_idx,
        "num_atoms": 4,
    }


@pytest.fixture(scope="session")
def batch_two_systems_matrix():
    """Two independent two-atom systems in a batch using neighbor matrix format.

    System 0: Atoms at (0,0,0) and (2,0,0) with charges +1 and -1
    System 1: Atoms at (0,0,0) and (4,0,0) with charges +2 and -1

    Returns
    -------
    dict
        Batched system parameters with neighbor matrix format (numpy arrays)
    """
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [
                0.0,
                0.0,
                0.0,
            ],
            [4.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    charges = np.array([1.0, -1, 2.0, -1], dtype=np.float64)
    cell = np.array(
        [
            [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]],
            [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]],
        ],
        dtype=np.float64,
    )
    neighbor_matrix = np.array([[1, 999], [0, 999], [3, 999], [2, 999]], dtype=np.int32)
    neighbor_shifts = np.zeros((4, 2, 3), dtype=np.int32)
    batch_idx = np.array([0, 0, 1, 1], dtype=np.int32)
    return {
        "positions": positions,
        "charges": charges,
        "cell": cell,
        "neighbor_matrix": neighbor_matrix,
        "neighbor_shifts": neighbor_shifts,
        "batch_idx": batch_idx,
        "fill_value": 999,
        "num_atoms": 4,
    }


# ==============================================================================
# Helper Functions
# ==============================================================================


def prepare_csr_inputs(system: dict, device: str) -> dict:
    """Prepare CSR-format inputs as warp arrays."""
    return {
        "positions": wp.from_numpy(system["positions"], dtype=wp.vec3d, device=device),
        "charges": wp.from_numpy(system["charges"], dtype=wp.float64, device=device),
        "cell": wp.from_numpy(system["cell"], dtype=wp.mat33d, device=device),
        "idx_j": wp.from_numpy(system["idx_j"], dtype=wp.int32, device=device),
        "neighbor_ptr": wp.from_numpy(
            system["neighbor_ptr"], dtype=wp.int32, device=device
        ),
        "unit_shifts": wp.from_numpy(
            system["unit_shifts"], dtype=wp.vec3i, device=device
        ),
    }


def prepare_matrix_inputs(system: dict, device: str) -> dict:
    """Prepare neighbor matrix format inputs as warp arrays."""
    return {
        "positions": wp.from_numpy(system["positions"], dtype=wp.vec3d, device=device),
        "charges": wp.from_numpy(system["charges"], dtype=wp.float64, device=device),
        "cell": wp.from_numpy(system["cell"], dtype=wp.mat33d, device=device),
        "neighbor_matrix": wp.from_numpy(
            system["neighbor_matrix"], dtype=wp.int32, device=device
        ),
        "neighbor_shifts": wp.from_numpy(
            system["neighbor_shifts"], dtype=wp.vec3i, device=device
        ),
        "fill_value": system["fill_value"],
    }


def allocate_energy_output(num_atoms: int, device: str) -> wp.array:
    """Allocate zero-initialized energy output array."""
    return wp.zeros(num_atoms, dtype=wp.float64, device=device)


def allocate_force_output(num_atoms: int, device: str) -> wp.array:
    """Allocate zero-initialized force output array."""
    return wp.zeros(num_atoms, dtype=wp.vec3d, device=device)


# ==============================================================================
# Test Class: Undamped Coulomb Energy (CSR Format)
# ==============================================================================


class TestWpCoulombEnergyCsr:
    """Tests for coulomb_energy with CSR neighbor list format."""

    def test_two_opposite_charges_undamped(self, device, two_atom_system):
        """Test undamped energy between opposite charges.

        E = q1*q2/r = (1)*(-1)/3 = -1/3
        With half neighbor list, each pair contributes 0.5 * E to each atom.
        Total = -1/3
        """
        inputs = prepare_csr_inputs(two_atom_system, device)
        energies = allocate_energy_output(two_atom_system["num_atoms"], device)

        coulomb_energy(
            positions=inputs["positions"],
            charges=inputs["charges"],
            cell=inputs["cell"],
            idx_j=inputs["idx_j"],
            neighbor_ptr=inputs["neighbor_ptr"],
            unit_shifts=inputs["unit_shifts"],
            cutoff=10.0,
            alpha=0.0,  # Undamped
            energies=energies,
            device=device,
        )

        result = energies.numpy()
        expected_total = -1.0 / 3.0
        assert sum(result) == pytest.approx(expected_total, rel=1e-10)

    def test_like_charges_positive_energy(self, device):
        """Test that like charges have positive energy."""
        positions = np.array(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        charges = np.array([1.0, 1.0], dtype=np.float64)  # Same sign
        cell = np.array(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=np.float64,
        )
        idx_j = np.array([1, 0], dtype=np.int32)
        neighbor_ptr = np.array([0, 1, 2], dtype=np.int32)
        unit_shifts = np.zeros((2, 3), dtype=np.int32)

        positions_wp = wp.from_numpy(positions, dtype=wp.vec3d, device=device)
        charges_wp = wp.from_numpy(charges, dtype=wp.float64, device=device)
        cell_wp = wp.from_numpy(cell, dtype=wp.mat33d, device=device)
        idx_j_wp = wp.from_numpy(idx_j, dtype=wp.int32, device=device)
        neighbor_ptr_wp = wp.from_numpy(neighbor_ptr, dtype=wp.int32, device=device)
        unit_shifts_wp = wp.from_numpy(unit_shifts, dtype=wp.vec3i, device=device)
        energies = allocate_energy_output(2, device)

        coulomb_energy(
            positions=positions_wp,
            charges=charges_wp,
            cell=cell_wp,
            idx_j=idx_j_wp,
            neighbor_ptr=neighbor_ptr_wp,
            unit_shifts=unit_shifts_wp,
            cutoff=10.0,
            alpha=0.0,
            energies=energies,
            device=device,
        )

        result = energies.numpy()
        # E = q1*q2/r = 1*1/2 = 0.5
        assert sum(result) == pytest.approx(0.5, rel=1e-10)
        assert sum(result) > 0  # Positive energy for like charges

    def test_energy_scales_with_distance(self, device):
        """Test that energy scales as 1/r."""
        # Test at two distances
        for distance, expected_energy in [(2.0, -0.5), (4.0, -0.25)]:
            positions = np.array(
                [[0.0, 0.0, 0.0], [distance, 0.0, 0.0]],
                dtype=np.float64,
            )
            charges = np.array([1.0, -1.0], dtype=np.float64)
            cell = np.array(
                [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
                dtype=np.float64,
            )
            idx_j = np.array([1, 0], dtype=np.int32)
            neighbor_ptr = np.array([0, 1, 2], dtype=np.int32)
            unit_shifts = np.zeros((2, 3), dtype=np.int32)

            positions_wp = wp.from_numpy(positions, dtype=wp.vec3d, device=device)
            charges_wp = wp.from_numpy(charges, dtype=wp.float64, device=device)
            cell_wp = wp.from_numpy(cell, dtype=wp.mat33d, device=device)
            idx_j_wp = wp.from_numpy(idx_j, dtype=wp.int32, device=device)
            neighbor_ptr_wp = wp.from_numpy(neighbor_ptr, dtype=wp.int32, device=device)
            unit_shifts_wp = wp.from_numpy(unit_shifts, dtype=wp.vec3i, device=device)
            energies = allocate_energy_output(2, device)

            coulomb_energy(
                positions=positions_wp,
                charges=charges_wp,
                cell=cell_wp,
                idx_j=idx_j_wp,
                neighbor_ptr=neighbor_ptr_wp,
                unit_shifts=unit_shifts_wp,
                cutoff=10.0,
                alpha=0.0,
                energies=energies,
                device=device,
            )

            result = energies.numpy()
            assert sum(result) == pytest.approx(expected_energy, rel=1e-10)

    def test_cutoff_excludes_far_pairs(self, device):
        """Test that pairs beyond cutoff are excluded."""
        positions = np.array(
            [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        charges = np.array([1.0, -1.0], dtype=np.float64)
        cell = np.array(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=np.float64,
        )
        idx_j = np.array([1, 0], dtype=np.int32)
        neighbor_ptr = np.array([0, 1, 2], dtype=np.int32)
        unit_shifts = np.zeros((2, 3), dtype=np.int32)

        positions_wp = wp.from_numpy(positions, dtype=wp.vec3d, device=device)
        charges_wp = wp.from_numpy(charges, dtype=wp.float64, device=device)
        cell_wp = wp.from_numpy(cell, dtype=wp.mat33d, device=device)
        idx_j_wp = wp.from_numpy(idx_j, dtype=wp.int32, device=device)
        neighbor_ptr_wp = wp.from_numpy(neighbor_ptr, dtype=wp.int32, device=device)
        unit_shifts_wp = wp.from_numpy(unit_shifts, dtype=wp.vec3i, device=device)
        energies = allocate_energy_output(2, device)

        # Use cutoff smaller than distance
        coulomb_energy(
            positions=positions_wp,
            charges=charges_wp,
            cell=cell_wp,
            idx_j=idx_j_wp,
            neighbor_ptr=neighbor_ptr_wp,
            unit_shifts=unit_shifts_wp,
            cutoff=3.0,  # Distance is 5.0, so pair should be excluded
            alpha=0.0,
            energies=energies,
            device=device,
        )

        result = energies.numpy()
        assert sum(result) == pytest.approx(0.0, abs=1e-15)


# ==============================================================================
# Test Class: Damped Coulomb Energy (Ewald Real-Space)
# ==============================================================================


class TestWpCoulombEnergyDamped:
    """Tests for damped Coulomb energy (alpha > 0)."""

    def test_damped_smaller_than_undamped(self, device, two_atom_system):
        """Damped energy should have smaller magnitude than undamped."""
        inputs = prepare_csr_inputs(two_atom_system, device)

        # Undamped
        energies_undamped = allocate_energy_output(two_atom_system["num_atoms"], device)
        coulomb_energy(
            positions=inputs["positions"],
            charges=inputs["charges"],
            cell=inputs["cell"],
            idx_j=inputs["idx_j"],
            neighbor_ptr=inputs["neighbor_ptr"],
            unit_shifts=inputs["unit_shifts"],
            cutoff=10.0,
            alpha=0.0,
            energies=energies_undamped,
            device=device,
        )

        # Damped with alpha = 0.5
        energies_damped = allocate_energy_output(two_atom_system["num_atoms"], device)
        coulomb_energy(
            positions=inputs["positions"],
            charges=inputs["charges"],
            cell=inputs["cell"],
            idx_j=inputs["idx_j"],
            neighbor_ptr=inputs["neighbor_ptr"],
            unit_shifts=inputs["unit_shifts"],
            cutoff=10.0,
            alpha=0.5,
            energies=energies_damped,
            device=device,
        )

        undamped_total = sum(energies_undamped.numpy())
        damped_total = sum(energies_damped.numpy())

        # Damped should have smaller magnitude (erfc(alpha*r) < 1)
        assert abs(damped_total) < abs(undamped_total)

    def test_damped_energy_erfc_formula(self, device, two_atom_system):
        """Test damped energy follows erfc(alpha*r)/r formula.

        Note: The kernel uses wp_erfc which may differ slightly from Python's
        math.erfc at high precision. We use a looser tolerance to account for
        this implementation difference.
        """
        inputs = prepare_csr_inputs(two_atom_system, device)
        r = two_atom_system["distance"]
        alpha = 0.3

        energies = allocate_energy_output(two_atom_system["num_atoms"], device)
        coulomb_energy(
            positions=inputs["positions"],
            charges=inputs["charges"],
            cell=inputs["cell"],
            idx_j=inputs["idx_j"],
            neighbor_ptr=inputs["neighbor_ptr"],
            unit_shifts=inputs["unit_shifts"],
            cutoff=10.0,
            alpha=alpha,
            energies=energies,
            device=device,
        )

        result = energies.numpy()
        # E = q1*q2 * erfc(alpha*r) / r
        # With full neighbor list and 0.5 prefactor, total = q1*q2 * erfc(alpha*r) / r
        erfc_term = math.erfc(alpha * r)
        expected = (1.0 * -1.0) * erfc_term / r
        # Use looser tolerance due to erfc implementation differences
        assert sum(result) == pytest.approx(expected, rel=1e-4)


# ==============================================================================
# Test Class: Coulomb Energy and Forces
# ==============================================================================


class TestWpCoulombEnergyForces:
    """Tests for coulomb_energy_forces."""

    def test_forces_opposite_direction_to_energy_gradient(
        self, device, two_atom_system
    ):
        """Test forces point in correct direction (attractive for opposite charges)."""
        inputs = prepare_csr_inputs(two_atom_system, device)
        num_atoms = two_atom_system["num_atoms"]

        energies = allocate_energy_output(num_atoms, device)
        forces = allocate_force_output(num_atoms, device)

        coulomb_energy_forces(
            positions=inputs["positions"],
            charges=inputs["charges"],
            cell=inputs["cell"],
            idx_j=inputs["idx_j"],
            neighbor_ptr=inputs["neighbor_ptr"],
            unit_shifts=inputs["unit_shifts"],
            cutoff=10.0,
            alpha=0.0,
            energies=energies,
            forces=forces,
            device=device,
        )

        result_forces = forces.numpy()
        # Atoms have opposite charges, so force should be attractive
        # Atom 0 at origin should be pulled toward positive x (toward atom 1)
        # Atom 1 should be pulled toward negative x (toward atom 0)
        assert result_forces[0, 0] > 0  # Force on atom 0 in +x direction
        assert result_forces[1, 0] < 0  # Force on atom 1 in -x direction

    def test_forces_satisfy_newtons_third_law(self, device, two_atom_system):
        """Test that F_ij = -F_ji (Newton's third law)."""
        inputs = prepare_csr_inputs(two_atom_system, device)
        num_atoms = two_atom_system["num_atoms"]

        energies = allocate_energy_output(num_atoms, device)
        forces = allocate_force_output(num_atoms, device)

        coulomb_energy_forces(
            positions=inputs["positions"],
            charges=inputs["charges"],
            cell=inputs["cell"],
            idx_j=inputs["idx_j"],
            neighbor_ptr=inputs["neighbor_ptr"],
            unit_shifts=inputs["unit_shifts"],
            cutoff=10.0,
            alpha=0.0,
            energies=energies,
            forces=forces,
            device=device,
        )

        result_forces = forces.numpy()
        # Sum of all forces should be zero
        total_force = np.sum(result_forces, axis=0)
        np.testing.assert_allclose(total_force, [0.0, 0.0, 0.0], atol=1e-14)

    def test_force_magnitude_undamped(self, device, two_atom_system):
        """Test undamped force magnitude: F = q1*q2/r^2."""
        inputs = prepare_csr_inputs(two_atom_system, device)
        num_atoms = two_atom_system["num_atoms"]
        r = two_atom_system["distance"]

        energies = allocate_energy_output(num_atoms, device)
        forces = allocate_force_output(num_atoms, device)

        coulomb_energy_forces(
            positions=inputs["positions"],
            charges=inputs["charges"],
            cell=inputs["cell"],
            idx_j=inputs["idx_j"],
            neighbor_ptr=inputs["neighbor_ptr"],
            unit_shifts=inputs["unit_shifts"],
            cutoff=10.0,
            alpha=0.0,
            energies=energies,
            forces=forces,
            device=device,
        )

        result_forces = forces.numpy()
        # Expected force magnitude: |F| = |q1*q2|/r^2 = 1/9
        expected_magnitude = 1.0 / (r * r)
        actual_magnitude = np.linalg.norm(result_forces[0])
        assert actual_magnitude == pytest.approx(expected_magnitude, rel=1e-10)

    def test_energy_consistent_with_energy_only(self, device, two_atom_system):
        """Energy from energy_forces should match energy_only kernel."""
        inputs = prepare_csr_inputs(two_atom_system, device)
        num_atoms = two_atom_system["num_atoms"]

        # Energy only
        energies_only = allocate_energy_output(num_atoms, device)
        coulomb_energy(
            positions=inputs["positions"],
            charges=inputs["charges"],
            cell=inputs["cell"],
            idx_j=inputs["idx_j"],
            neighbor_ptr=inputs["neighbor_ptr"],
            unit_shifts=inputs["unit_shifts"],
            cutoff=10.0,
            alpha=0.0,
            energies=energies_only,
            device=device,
        )

        # Energy + forces
        energies_combined = allocate_energy_output(num_atoms, device)
        forces = allocate_force_output(num_atoms, device)
        coulomb_energy_forces(
            positions=inputs["positions"],
            charges=inputs["charges"],
            cell=inputs["cell"],
            idx_j=inputs["idx_j"],
            neighbor_ptr=inputs["neighbor_ptr"],
            unit_shifts=inputs["unit_shifts"],
            cutoff=10.0,
            alpha=0.0,
            energies=energies_combined,
            forces=forces,
            device=device,
        )

        np.testing.assert_allclose(
            energies_only.numpy(), energies_combined.numpy(), rtol=1e-14
        )


# ==============================================================================
# Test Class: Neighbor Matrix Format
# ==============================================================================


class TestWpCoulombEnergyMatrix:
    """Tests for neighbor matrix format kernels."""

    def test_energy_matches_csr_format(
        self, device, two_atom_system, two_atom_matrix_system
    ):
        """Energy from matrix format should match CSR format."""
        # CSR format
        csr_inputs = prepare_csr_inputs(two_atom_system, device)
        energies_csr = allocate_energy_output(two_atom_system["num_atoms"], device)
        coulomb_energy(
            positions=csr_inputs["positions"],
            charges=csr_inputs["charges"],
            cell=csr_inputs["cell"],
            idx_j=csr_inputs["idx_j"],
            neighbor_ptr=csr_inputs["neighbor_ptr"],
            unit_shifts=csr_inputs["unit_shifts"],
            cutoff=10.0,
            alpha=0.0,
            energies=energies_csr,
            device=device,
        )

        # Matrix format
        mat_inputs = prepare_matrix_inputs(two_atom_matrix_system, device)
        energies_mat = allocate_energy_output(
            two_atom_matrix_system["num_atoms"], device
        )
        coulomb_energy_matrix(
            positions=mat_inputs["positions"],
            charges=mat_inputs["charges"],
            cell=mat_inputs["cell"],
            neighbor_matrix=mat_inputs["neighbor_matrix"],
            neighbor_matrix_shifts=mat_inputs["neighbor_shifts"],
            cutoff=10.0,
            alpha=0.0,
            fill_value=mat_inputs["fill_value"],
            energies=energies_mat,
            device=device,
        )

        # Total energies should be close (matrix format counts each pair twice)
        csr_total = sum(energies_csr.numpy())
        mat_total = sum(energies_mat.numpy())
        # Matrix format doesn't use 0.5 prefactor, so it counts full energy per pair
        # and each pair is counted twice (once from each atom's perspective)
        # So mat_total = 2 * csr_total if csr uses half neighbor list
        assert mat_total == pytest.approx(2 * csr_total, rel=1e-10)

    def test_energy_forces_matrix(self, device, two_atom_matrix_system):
        """Test energy and forces with matrix format."""
        mat_inputs = prepare_matrix_inputs(two_atom_matrix_system, device)
        num_atoms = two_atom_matrix_system["num_atoms"]

        energies = allocate_energy_output(num_atoms, device)
        forces = allocate_force_output(num_atoms, device)

        coulomb_energy_forces_matrix(
            positions=mat_inputs["positions"],
            charges=mat_inputs["charges"],
            cell=mat_inputs["cell"],
            neighbor_matrix=mat_inputs["neighbor_matrix"],
            neighbor_matrix_shifts=mat_inputs["neighbor_shifts"],
            cutoff=10.0,
            alpha=0.0,
            fill_value=mat_inputs["fill_value"],
            energies=energies,
            forces=forces,
            device=device,
        )

        result_forces = forces.numpy()
        # Newton's third law should still hold
        total_force = np.sum(result_forces, axis=0)
        np.testing.assert_allclose(total_force, [0.0, 0.0, 0.0], atol=1e-14)


# ==============================================================================
# Test Class: Batched Calculations
# ==============================================================================


class TestWpBatchCoulombEnergy:
    """Tests for batched Coulomb kernels."""

    def test_batch_energy_sum(self, device, batch_two_systems):
        """Test that batch energy is sum of individual system energies."""
        positions = wp.from_numpy(
            batch_two_systems["positions"], dtype=wp.vec3d, device=device
        )
        charges = wp.from_numpy(
            batch_two_systems["charges"], dtype=wp.float64, device=device
        )
        cell = wp.from_numpy(batch_two_systems["cell"], dtype=wp.mat33d, device=device)
        idx_j = wp.from_numpy(batch_two_systems["idx_j"], dtype=wp.int32, device=device)
        neighbor_ptr = wp.from_numpy(
            batch_two_systems["neighbor_ptr"], dtype=wp.int32, device=device
        )
        unit_shifts = wp.from_numpy(
            batch_two_systems["unit_shifts"], dtype=wp.vec3i, device=device
        )
        batch_idx = wp.from_numpy(
            batch_two_systems["batch_idx"], dtype=wp.int32, device=device
        )

        energies = allocate_energy_output(batch_two_systems["num_atoms"], device)

        batch_coulomb_energy(
            positions=positions,
            charges=charges,
            cell=cell,
            idx_j=idx_j,
            neighbor_ptr=neighbor_ptr,
            unit_shifts=unit_shifts,
            batch_idx=batch_idx,
            cutoff=10.0,
            alpha=0.0,
            energies=energies,
            device=device,
        )

        result = energies.numpy()
        # System 0: E = q1*q2/r = 1*(-1)/2 = -0.5
        # System 1: E = q1*q2/r = 2*(-1)/4 = -0.5
        # Per-atom contributions with half neighbor list
        # System 0 atoms: sum = -0.5 * 0.5 = -0.25 (split between atoms)
        # Actually with half NL: atom 0 gets full 0.5 * E
        system_0_energy = result[0] + result[1]
        system_1_energy = result[2] + result[3]

        assert system_0_energy == pytest.approx(
            -0.5 / 2, rel=1e-10
        )  # Half from prefactor
        assert system_1_energy == pytest.approx(-0.5 / 2, rel=1e-10)

    def test_batch_forces_newton_third_law_per_system(self, device, batch_two_systems):
        """Forces should sum to zero within each system."""
        positions = wp.from_numpy(
            batch_two_systems["positions"], dtype=wp.vec3d, device=device
        )
        charges = wp.from_numpy(
            batch_two_systems["charges"], dtype=wp.float64, device=device
        )
        cell = wp.from_numpy(batch_two_systems["cell"], dtype=wp.mat33d, device=device)
        idx_j = wp.from_numpy(batch_two_systems["idx_j"], dtype=wp.int32, device=device)
        neighbor_ptr = wp.from_numpy(
            batch_two_systems["neighbor_ptr"], dtype=wp.int32, device=device
        )
        unit_shifts = wp.from_numpy(
            batch_two_systems["unit_shifts"], dtype=wp.vec3i, device=device
        )
        batch_idx = wp.from_numpy(
            batch_two_systems["batch_idx"], dtype=wp.int32, device=device
        )

        energies = allocate_energy_output(batch_two_systems["num_atoms"], device)
        forces = allocate_force_output(batch_two_systems["num_atoms"], device)

        batch_coulomb_energy_forces(
            positions=positions,
            charges=charges,
            cell=cell,
            idx_j=idx_j,
            neighbor_ptr=neighbor_ptr,
            unit_shifts=unit_shifts,
            batch_idx=batch_idx,
            cutoff=10.0,
            alpha=0.0,
            energies=energies,
            forces=forces,
            device=device,
        )

        result_forces = forces.numpy()
        # System 0: atoms 0, 1
        system_0_force = result_forces[0] + result_forces[1]
        np.testing.assert_allclose(system_0_force, [0.0, 0.0, 0.0], atol=1e-14)

        # System 1: atoms 2, 3
        system_1_force = result_forces[2] + result_forces[3]
        np.testing.assert_allclose(system_1_force, [0.0, 0.0, 0.0], atol=1e-14)


# ==============================================================================
# Test Class: Batched Calculations with Neighbor Matrix Format
# ==============================================================================


class TestWpBatchCoulombEnergyMatrix:
    """Tests for batched Coulomb kernels with neighbor matrix format."""

    def test_batch_matrix_energy(self, device, batch_two_systems_matrix):
        """Test batch energy calculation with neighbor matrix format."""
        positions = wp.from_numpy(
            batch_two_systems_matrix["positions"], dtype=wp.vec3d, device=device
        )
        charges = wp.from_numpy(
            batch_two_systems_matrix["charges"], dtype=wp.float64, device=device
        )
        cell = wp.from_numpy(
            batch_two_systems_matrix["cell"], dtype=wp.mat33d, device=device
        )
        neighbor_matrix = wp.from_numpy(
            batch_two_systems_matrix["neighbor_matrix"], dtype=wp.int32, device=device
        )
        neighbor_shifts = wp.from_numpy(
            batch_two_systems_matrix["neighbor_shifts"], dtype=wp.vec3i, device=device
        )
        batch_idx = wp.from_numpy(
            batch_two_systems_matrix["batch_idx"], dtype=wp.int32, device=device
        )

        energies = allocate_energy_output(batch_two_systems_matrix["num_atoms"], device)

        batch_coulomb_energy_matrix(
            positions=positions,
            charges=charges,
            cell=cell,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_shifts,
            batch_idx=batch_idx,
            cutoff=10.0,
            alpha=0.0,
            fill_value=batch_two_systems_matrix["fill_value"],
            energies=energies,
            device=device,
        )

        result = energies.numpy()
        # System 0: E = q1*q2/r = 1*(-1)/2 = -0.5 (counted twice in matrix format)
        # System 1: E = q1*q2/r = 2*(-1)/4 = -0.5 (counted twice in matrix format)
        system_0_energy = result[0] + result[1]
        system_1_energy = result[2] + result[3]

        # Matrix format counts each pair twice (full neighbor list)
        assert system_0_energy == pytest.approx(-1.0, rel=1e-10)
        assert system_1_energy == pytest.approx(-1.0, rel=1e-10)

    def test_batch_matrix_energy_forces(self, device, batch_two_systems_matrix):
        """Test batch energy and forces with neighbor matrix format."""
        positions = wp.from_numpy(
            batch_two_systems_matrix["positions"], dtype=wp.vec3d, device=device
        )
        charges = wp.from_numpy(
            batch_two_systems_matrix["charges"], dtype=wp.float64, device=device
        )
        cell = wp.from_numpy(
            batch_two_systems_matrix["cell"], dtype=wp.mat33d, device=device
        )
        neighbor_matrix = wp.from_numpy(
            batch_two_systems_matrix["neighbor_matrix"], dtype=wp.int32, device=device
        )
        neighbor_shifts = wp.from_numpy(
            batch_two_systems_matrix["neighbor_shifts"], dtype=wp.vec3i, device=device
        )
        batch_idx = wp.from_numpy(
            batch_two_systems_matrix["batch_idx"], dtype=wp.int32, device=device
        )

        energies = allocate_energy_output(batch_two_systems_matrix["num_atoms"], device)
        forces = allocate_force_output(batch_two_systems_matrix["num_atoms"], device)

        batch_coulomb_energy_forces_matrix(
            positions=positions,
            charges=charges,
            cell=cell,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_shifts,
            batch_idx=batch_idx,
            cutoff=10.0,
            alpha=0.0,
            fill_value=batch_two_systems_matrix["fill_value"],
            energies=energies,
            forces=forces,
            device=device,
        )

        result_forces = forces.numpy()
        # Newton's third law should hold within each system
        # System 0: atoms 0, 1
        system_0_force = result_forces[0] + result_forces[1]
        np.testing.assert_allclose(system_0_force, [0.0, 0.0, 0.0], atol=1e-14)

        # System 1: atoms 2, 3
        system_1_force = result_forces[2] + result_forces[3]
        np.testing.assert_allclose(system_1_force, [0.0, 0.0, 0.0], atol=1e-14)

    def test_batch_matrix_damped(self, device, batch_two_systems_matrix):
        """Test batch damped energy with neighbor matrix format."""
        positions = wp.from_numpy(
            batch_two_systems_matrix["positions"], dtype=wp.vec3d, device=device
        )
        charges = wp.from_numpy(
            batch_two_systems_matrix["charges"], dtype=wp.float64, device=device
        )
        cell = wp.from_numpy(
            batch_two_systems_matrix["cell"], dtype=wp.mat33d, device=device
        )
        neighbor_matrix = wp.from_numpy(
            batch_two_systems_matrix["neighbor_matrix"], dtype=wp.int32, device=device
        )
        neighbor_shifts = wp.from_numpy(
            batch_two_systems_matrix["neighbor_shifts"], dtype=wp.vec3i, device=device
        )
        batch_idx = wp.from_numpy(
            batch_two_systems_matrix["batch_idx"], dtype=wp.int32, device=device
        )

        # Undamped
        energies_undamped = allocate_energy_output(
            batch_two_systems_matrix["num_atoms"], device
        )
        batch_coulomb_energy_matrix(
            positions=positions,
            charges=charges,
            cell=cell,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_shifts,
            batch_idx=batch_idx,
            cutoff=10.0,
            alpha=0.0,
            fill_value=batch_two_systems_matrix["fill_value"],
            energies=energies_undamped,
            device=device,
        )

        # Damped
        energies_damped = allocate_energy_output(
            batch_two_systems_matrix["num_atoms"], device
        )
        batch_coulomb_energy_matrix(
            positions=positions,
            charges=charges,
            cell=cell,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_shifts,
            batch_idx=batch_idx,
            cutoff=10.0,
            alpha=0.5,
            fill_value=batch_two_systems_matrix["fill_value"],
            energies=energies_damped,
            device=device,
        )

        undamped_total = sum(energies_undamped.numpy())
        damped_total = sum(energies_damped.numpy())

        # Damped should have smaller magnitude
        assert abs(damped_total) < abs(undamped_total)


# ==============================================================================
# Regression Tests with Hardcoded Values
# ==============================================================================


class TestCoulombRegressionValues:
    """Regression tests with hardcoded expected outputs.

    These values were generated by running the warp kernels with known inputs
    to establish baseline behavior. They serve as regression tests to catch
    unintended changes to kernel behavior.
    """

    def test_regression_undamped_energy_two_atoms(self, device):
        """Regression test: two atoms at distance 3.0 with charges +1, -1.

        Expected energy: -1/3 = -0.333333...
        """
        positions = np.array(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        charges = np.array([1.0, -1.0], dtype=np.float64)
        cell = np.array(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=np.float64,
        )
        idx_j = np.array([1, 0], dtype=np.int32)
        neighbor_ptr = np.array([0, 1, 2], dtype=np.int32)
        unit_shifts = np.zeros((2, 3), dtype=np.int32)

        positions_wp = wp.from_numpy(positions, dtype=wp.vec3d, device=device)
        charges_wp = wp.from_numpy(charges, dtype=wp.float64, device=device)
        cell_wp = wp.from_numpy(cell, dtype=wp.mat33d, device=device)
        idx_j_wp = wp.from_numpy(idx_j, dtype=wp.int32, device=device)
        neighbor_ptr_wp = wp.from_numpy(neighbor_ptr, dtype=wp.int32, device=device)
        unit_shifts_wp = wp.from_numpy(unit_shifts, dtype=wp.vec3i, device=device)
        energies = allocate_energy_output(2, device)

        coulomb_energy(
            positions=positions_wp,
            charges=charges_wp,
            cell=cell_wp,
            idx_j=idx_j_wp,
            neighbor_ptr=neighbor_ptr_wp,
            unit_shifts=unit_shifts_wp,
            cutoff=10.0,
            alpha=0.0,
            energies=energies,
            device=device,
        )

        result = energies.numpy()
        # Hardcoded regression value
        expected_total = -0.3333333333333333
        assert sum(result) == pytest.approx(expected_total, rel=1e-12)

    def test_regression_damped_energy_alpha_0p5(self, device):
        """Regression test: damped energy with alpha=0.5, distance=3.0.

        E = q1*q2 * erfc(alpha*r) / r
        alpha = 0.5, r = 3.0
        erfc(1.5) = 0.0339...
        E = (1)*(-1) * 0.0339... / 3.0 = -0.01129...

        Note: Hardcoded value from actual kernel output to catch regressions.
        """
        positions = np.array(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        charges = np.array([1.0, -1.0], dtype=np.float64)
        cell = np.array(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=np.float64,
        )
        idx_j = np.array([1, 0], dtype=np.int32)
        neighbor_ptr = np.array([0, 1, 2], dtype=np.int32)
        unit_shifts = np.zeros((2, 3), dtype=np.int32)

        positions_wp = wp.from_numpy(positions, dtype=wp.vec3d, device=device)
        charges_wp = wp.from_numpy(charges, dtype=wp.float64, device=device)
        cell_wp = wp.from_numpy(cell, dtype=wp.mat33d, device=device)
        idx_j_wp = wp.from_numpy(idx_j, dtype=wp.int32, device=device)
        neighbor_ptr_wp = wp.from_numpy(neighbor_ptr, dtype=wp.int32, device=device)
        unit_shifts_wp = wp.from_numpy(unit_shifts, dtype=wp.vec3i, device=device)
        energies = allocate_energy_output(2, device)

        coulomb_energy(
            positions=positions_wp,
            charges=charges_wp,
            cell=cell_wp,
            idx_j=idx_j_wp,
            neighbor_ptr=neighbor_ptr_wp,
            unit_shifts=unit_shifts_wp,
            cutoff=10.0,
            alpha=0.5,
            energies=energies,
            device=device,
        )

        result = energies.numpy()
        # Hardcoded regression value from actual kernel output
        # Note: This differs slightly from math.erfc due to wp_erfc implementation
        expected_total = -0.011298244912778213
        assert sum(result) == pytest.approx(expected_total, rel=1e-10)

    def test_regression_force_magnitude(self, device):
        """Regression test: force magnitude for two atoms at distance 3.0.

        Undamped force: |F| = |q1*q2|/r^2 = 1/9 = 0.111111...
        """
        positions = np.array(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        charges = np.array([1.0, -1.0], dtype=np.float64)
        cell = np.array(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=np.float64,
        )
        idx_j = np.array([1, 0], dtype=np.int32)
        neighbor_ptr = np.array([0, 1, 2], dtype=np.int32)
        unit_shifts = np.zeros((2, 3), dtype=np.int32)

        positions_wp = wp.from_numpy(positions, dtype=wp.vec3d, device=device)
        charges_wp = wp.from_numpy(charges, dtype=wp.float64, device=device)
        cell_wp = wp.from_numpy(cell, dtype=wp.mat33d, device=device)
        idx_j_wp = wp.from_numpy(idx_j, dtype=wp.int32, device=device)
        neighbor_ptr_wp = wp.from_numpy(neighbor_ptr, dtype=wp.int32, device=device)
        unit_shifts_wp = wp.from_numpy(unit_shifts, dtype=wp.vec3i, device=device)
        energies = allocate_energy_output(2, device)
        forces = allocate_force_output(2, device)

        coulomb_energy_forces(
            positions=positions_wp,
            charges=charges_wp,
            cell=cell_wp,
            idx_j=idx_j_wp,
            neighbor_ptr=neighbor_ptr_wp,
            unit_shifts=unit_shifts_wp,
            cutoff=10.0,
            alpha=0.0,
            energies=energies,
            forces=forces,
            device=device,
        )

        result_forces = forces.numpy()
        # Hardcoded regression values
        # Atom 0 force: [0.111111..., 0, 0] (attractive toward atom 1)
        # Atom 1 force: [-0.111111..., 0, 0] (attractive toward atom 0)
        expected_force_0 = np.array([0.1111111111111111, 0.0, 0.0])
        expected_force_1 = np.array([-0.1111111111111111, 0.0, 0.0])

        np.testing.assert_allclose(result_forces[0], expected_force_0, rtol=1e-12)
        np.testing.assert_allclose(result_forces[1], expected_force_1, rtol=1e-12)

    def test_regression_three_atom_energy(self, device, three_atom_system):
        """Regression test: three atoms with multiple interactions.

        Atoms: (0,0,0) q=+1, (2,0,0) q=-1, (5,0,0) q=+0.5
        Distances: r01=2, r02=5, r12=3
        Energies (half neighbor list):
        - E_01 = 0.5 * (1)*(-1)/2 = -0.25
        - E_02 = 0.5 * (1)*(0.5)/5 = 0.05
        - E_12 = 0.5 * (-1)*(0.5)/3 = -0.0833...
        - Total = -0.25 + 0.05 - 0.0833... = -0.2833...
        """
        inputs = prepare_csr_inputs(three_atom_system, device)
        energies = allocate_energy_output(three_atom_system["num_atoms"], device)

        coulomb_energy(
            positions=inputs["positions"],
            charges=inputs["charges"],
            cell=inputs["cell"],
            idx_j=inputs["idx_j"],
            neighbor_ptr=inputs["neighbor_ptr"],
            unit_shifts=inputs["unit_shifts"],
            cutoff=10.0,
            alpha=0.0,
            energies=energies,
            device=device,
        )

        result = energies.numpy()
        # Hardcoded regression value
        expected_total = -0.28333333333333333
        assert sum(result) == pytest.approx(expected_total, rel=1e-12)

    def test_regression_damped_force(self, device):
        """Regression test: damped force with alpha=0.3.

        Damped force:
        F = q1*q2 * [erfc(alpha*r)/r^2 + 2*alpha/sqrt(pi) * exp(-alpha^2*r^2)/r] / r * r_ij

        Note: Hardcoded value from actual kernel output to catch regressions.
        """
        positions = np.array(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        charges = np.array([1.0, -1.0], dtype=np.float64)
        cell = np.array(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=np.float64,
        )
        idx_j = np.array([1, 0], dtype=np.int32)
        neighbor_ptr = np.array([0, 1, 2], dtype=np.int32)
        unit_shifts = np.zeros((2, 3), dtype=np.int32)

        positions_wp = wp.from_numpy(positions, dtype=wp.vec3d, device=device)
        charges_wp = wp.from_numpy(charges, dtype=wp.float64, device=device)
        cell_wp = wp.from_numpy(cell, dtype=wp.mat33d, device=device)
        idx_j_wp = wp.from_numpy(idx_j, dtype=wp.int32, device=device)
        neighbor_ptr_wp = wp.from_numpy(neighbor_ptr, dtype=wp.int32, device=device)
        unit_shifts_wp = wp.from_numpy(unit_shifts, dtype=wp.vec3i, device=device)
        energies = allocate_energy_output(2, device)
        forces = allocate_force_output(2, device)

        coulomb_energy_forces(
            positions=positions_wp,
            charges=charges_wp,
            cell=cell_wp,
            idx_j=idx_j_wp,
            neighbor_ptr=neighbor_ptr_wp,
            unit_shifts=unit_shifts_wp,
            cutoff=10.0,
            alpha=0.3,
            energies=energies,
            forces=forces,
            device=device,
        )

        result_forces = forces.numpy()
        # Hardcoded regression values from actual kernel output
        # Force is in +x direction on atom 0 (attractive, opposite charges)
        expected_fx = 0.07276263
        assert result_forces[0, 0] == pytest.approx(expected_fx, rel=1e-6)
        assert result_forces[1, 0] == pytest.approx(-expected_fx, rel=1e-6)
        # y and z components should be zero
        assert result_forces[0, 1] == pytest.approx(0.0, abs=1e-14)
        assert result_forces[0, 2] == pytest.approx(0.0, abs=1e-14)
