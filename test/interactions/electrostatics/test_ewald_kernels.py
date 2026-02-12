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
Unit tests for Ewald warp kernel launchers.

This test suite validates the correctness of the Ewald warp launchers
(ewald_real_space_*, ewald_reciprocal_space_*, etc.) directly using warp arrays.

Tests cover:
- Real-space energy calculations (damped Coulomb with erfc)
- Real-space energy and forces
- Reciprocal-space structure factor computation
- Reciprocal-space energy calculation
- Self-energy and background corrections
- Both CSR and neighbor matrix formats
- Batched calculations
- Float32 and float64 dtypes

These tests use warp arrays directly and do not require PyTorch.
For PyTorch binding tests, see test/interactions/electrostatics/bindings/torch/test_ewald.py
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from nvalchemiops.interactions.electrostatics.ewald_kernels import (
    batch_ewald_real_space_energy,
    batch_ewald_real_space_energy_forces,
    batch_ewald_real_space_energy_forces_charge_grad,
    batch_ewald_real_space_energy_forces_charge_grad_matrix,
    batch_ewald_real_space_energy_forces_matrix,
    batch_ewald_real_space_energy_matrix,
    batch_ewald_reciprocal_space_compute_energy,
    batch_ewald_reciprocal_space_energy_forces,
    batch_ewald_reciprocal_space_energy_forces_charge_grad,
    batch_ewald_reciprocal_space_fill_structure_factors,
    batch_ewald_subtract_self_energy,
    ewald_real_space_energy,
    ewald_real_space_energy_forces,
    ewald_real_space_energy_forces_charge_grad,
    ewald_real_space_energy_forces_charge_grad_matrix,
    ewald_real_space_energy_forces_matrix,
    ewald_real_space_energy_matrix,
    ewald_reciprocal_space_compute_energy,
    ewald_reciprocal_space_energy_forces,
    ewald_reciprocal_space_energy_forces_charge_grad,
    ewald_reciprocal_space_fill_structure_factors,
    ewald_subtract_self_energy,
)

# ==============================================================================
# Test Fixtures
# ==============================================================================


@pytest.fixture(scope="session")
def two_atom_system():
    """Simple two-atom system for basic Ewald tests.

    Two atoms along x-axis:
    - Atom 0 at origin with charge +1
    - Atom 1 at (3, 0, 0) with charge -1

    Distance r = 3.0
    Expected damped real-space energy: E = q1*q2 * erfc(alpha*r) / r

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
    # Large cell to reduce periodic effects
    cell = np.array(
        [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
        dtype=np.float64,
    )
    # CSR format: atom 0 has neighbor 1, atom 1 has neighbor 0 (half neighbor list)
    idx_j = np.array([1], dtype=np.int32)
    neighbor_ptr = np.array([0, 1, 1], dtype=np.int32)
    unit_shifts = np.array([[0, 0, 0]], dtype=np.int32)

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
    # atom 0: neighbor 1; atom 1: neighbor 0 (full neighbor list)
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

    System 0: Atoms at (0,0,0) and (2,0,0), charges +1 and -1, distance=2
    System 1: Atoms at (0,0,0) and (4,0,0), charges +2 and -1, distance=4

    Returns
    -------
    dict
        Batched system parameters (numpy arrays)
    """
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

    Returns
    -------
    dict
        Batched system parameters with neighbor matrix format (numpy arrays)
    """
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
    cell = np.array(
        [
            [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]],
            [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]],
        ],
        dtype=np.float64,
    )
    # Full neighbor list: each atom knows about its neighbor in the same system
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


@pytest.fixture(scope="session")
def simple_kvector_system():
    """System with k-vectors for reciprocal-space tests.

    Small cubic cell with simple k-vectors for testing.

    Returns
    -------
    dict
        System parameters including k-vectors
    """
    # 2 atoms in a small cubic cell
    positions = np.array(
        [[0.0, 0.0, 0.0], [2.5, 2.5, 2.5]],
        dtype=np.float64,
    )
    charges = np.array([1.0, -1.0], dtype=np.float64)
    L = 10.0  # Cell length
    cell = np.array(
        [[[L, 0.0, 0.0], [0.0, L, 0.0], [0.0, 0.0, L]]],
        dtype=np.float64,
    )
    volume = L**3

    # Simple k-vectors: 2π/L * (n_x, n_y, n_z) for small integers
    # Using half-space convention (n_z >= 0 with special handling)
    k_factor = 2.0 * np.pi / L
    k_vectors = (
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
            ],
            dtype=np.float64,
        )
        * k_factor
    )

    return {
        "positions": positions,
        "charges": charges,
        "cell": cell,
        "k_vectors": k_vectors,
        "volume": volume,
        "num_atoms": 2,
        "num_k": 6,
    }


# ==============================================================================
# Helper Functions
# ==============================================================================


def prepare_csr_inputs(system: dict, device: str, dtype=wp.float64) -> dict:
    """Prepare CSR-format inputs as warp arrays."""
    vec_dtype = wp.vec3d if dtype == wp.float64 else wp.vec3f
    mat_dtype = wp.mat33d if dtype == wp.float64 else wp.mat33f

    return {
        "positions": wp.from_numpy(
            system["positions"].astype(
                np.float64 if dtype == wp.float64 else np.float32
            ),
            dtype=vec_dtype,
            device=device,
        ),
        "charges": wp.from_numpy(
            system["charges"].astype(np.float64 if dtype == wp.float64 else np.float32),
            dtype=dtype,
            device=device,
        ),
        "cell": wp.from_numpy(
            system["cell"].astype(np.float64 if dtype == wp.float64 else np.float32),
            dtype=mat_dtype,
            device=device,
        ),
        "idx_j": wp.from_numpy(system["idx_j"], dtype=wp.int32, device=device),
        "neighbor_ptr": wp.from_numpy(
            system["neighbor_ptr"], dtype=wp.int32, device=device
        ),
        "unit_shifts": wp.from_numpy(
            system["unit_shifts"], dtype=wp.vec3i, device=device
        ),
    }


def prepare_matrix_inputs(system: dict, device: str, dtype=wp.float64) -> dict:
    """Prepare neighbor matrix format inputs as warp arrays."""
    vec_dtype = wp.vec3d if dtype == wp.float64 else wp.vec3f
    mat_dtype = wp.mat33d if dtype == wp.float64 else wp.mat33f

    return {
        "positions": wp.from_numpy(
            system["positions"].astype(
                np.float64 if dtype == wp.float64 else np.float32
            ),
            dtype=vec_dtype,
            device=device,
        ),
        "charges": wp.from_numpy(
            system["charges"].astype(np.float64 if dtype == wp.float64 else np.float32),
            dtype=dtype,
            device=device,
        ),
        "cell": wp.from_numpy(
            system["cell"].astype(np.float64 if dtype == wp.float64 else np.float32),
            dtype=mat_dtype,
            device=device,
        ),
        "neighbor_matrix": wp.from_numpy(
            system["neighbor_matrix"], dtype=wp.int32, device=device
        ),
        "neighbor_shifts": wp.from_numpy(
            system["neighbor_shifts"], dtype=wp.vec3i, device=device
        ),
        "fill_value": system["fill_value"],
    }


def allocate_energy_output(num_atoms: int, device: str) -> wp.array:
    """Allocate zero-initialized energy output array (always float64 for accumulators)."""
    return wp.zeros(num_atoms, dtype=wp.float64, device=device)


def allocate_force_output(num_atoms: int, device: str, dtype=wp.float64) -> wp.array:
    """Allocate zero-initialized force output array."""
    vec_dtype = wp.vec3d if dtype == wp.float64 else wp.vec3f
    return wp.zeros(num_atoms, dtype=vec_dtype, device=device)


def make_alpha_array(alpha: float, device: str, dtype=wp.float64) -> wp.array:
    """Create a 1-element alpha array for Ewald kernels."""
    return wp.from_numpy(
        np.array([alpha], dtype=np.float64 if dtype == wp.float64 else np.float32),
        dtype=dtype,
        device=device,
    )


def allocate_charge_grad_output(num_atoms: int, device: str) -> wp.array:
    """Allocate zero-initialized charge gradient output array (always float64)."""
    return wp.zeros(num_atoms, dtype=wp.float64, device=device)


# ==============================================================================
# Test Class: Ewald Real-Space Energy (CSR Format)
# ==============================================================================


class TestWpEwaldRealSpaceEnergyCsr:
    """Tests for ewald_real_space_energy with CSR neighbor list format."""

    def test_two_opposite_charges_damped(self, device, two_atom_system):
        """Test damped energy between opposite charges.

        E = q1*q2 * erfc(alpha*r) / r
        For r=3, alpha=0.3: erfc(0.9) ≈ 0.2031
        Expected = 1*(-1) * erfc(0.9) / 3 ≈ -0.0677
        """
        inputs = prepare_csr_inputs(two_atom_system, device)
        energies = allocate_energy_output(two_atom_system["num_atoms"], device)
        alpha = 0.3
        alpha_arr = make_alpha_array(alpha, device)

        ewald_real_space_energy(
            positions=inputs["positions"],
            charges=inputs["charges"],
            cell=inputs["cell"],
            idx_j=inputs["idx_j"],
            neighbor_ptr=inputs["neighbor_ptr"],
            unit_shifts=inputs["unit_shifts"],
            alpha=alpha_arr,
            pair_energies=energies,
            wp_dtype=wp.float64,
            device=device,
        )

        result = energies.numpy()
        r = two_atom_system["distance"]
        erfc_term = math.erfc(alpha * r)
        # Half neighbor list: full energy assigned to first atom of pair (0.5 prefactor)
        expected_total = 0.5 * (1.0 * -1.0) * erfc_term / r
        assert sum(result) == pytest.approx(expected_total, rel=1e-4)

    def test_damping_reduces_energy_magnitude(self, device, two_atom_system):
        """Higher alpha should reduce energy magnitude (more damping)."""
        inputs = prepare_csr_inputs(two_atom_system, device)

        # Lower alpha (less damping)
        energies_low = allocate_energy_output(two_atom_system["num_atoms"], device)
        alpha_low = make_alpha_array(0.1, device)
        ewald_real_space_energy(
            positions=inputs["positions"],
            charges=inputs["charges"],
            cell=inputs["cell"],
            idx_j=inputs["idx_j"],
            neighbor_ptr=inputs["neighbor_ptr"],
            unit_shifts=inputs["unit_shifts"],
            alpha=alpha_low,
            pair_energies=energies_low,
            wp_dtype=wp.float64,
            device=device,
        )

        # Higher alpha (more damping)
        energies_high = allocate_energy_output(two_atom_system["num_atoms"], device)
        alpha_high = make_alpha_array(0.5, device)
        ewald_real_space_energy(
            positions=inputs["positions"],
            charges=inputs["charges"],
            cell=inputs["cell"],
            idx_j=inputs["idx_j"],
            neighbor_ptr=inputs["neighbor_ptr"],
            unit_shifts=inputs["unit_shifts"],
            alpha=alpha_high,
            pair_energies=energies_high,
            wp_dtype=wp.float64,
            device=device,
        )

        low_total = abs(sum(energies_low.numpy()))
        high_total = abs(sum(energies_high.numpy()))

        # Higher alpha -> smaller magnitude
        assert high_total < low_total

    def test_three_atom_system(self, device, three_atom_system):
        """Test real-space energy with three atoms."""
        inputs = prepare_csr_inputs(three_atom_system, device)
        energies = allocate_energy_output(three_atom_system["num_atoms"], device)
        alpha = 0.2
        alpha_arr = make_alpha_array(alpha, device)

        ewald_real_space_energy(
            positions=inputs["positions"],
            charges=inputs["charges"],
            cell=inputs["cell"],
            idx_j=inputs["idx_j"],
            neighbor_ptr=inputs["neighbor_ptr"],
            unit_shifts=inputs["unit_shifts"],
            alpha=alpha_arr,
            pair_energies=energies,
            wp_dtype=wp.float64,
            device=device,
        )

        result = energies.numpy()
        # Manual calculation for 3-atom system with half neighbor list
        # Pairs: (0,1) r=2, (0,2) r=5, (1,2) r=3
        q0, q1, q2 = 1.0, -1.0, 0.5
        e01 = 0.5 * q0 * q1 * math.erfc(alpha * 2.0) / 2.0
        e02 = 0.5 * q0 * q2 * math.erfc(alpha * 5.0) / 5.0
        e12 = 0.5 * q1 * q2 * math.erfc(alpha * 3.0) / 3.0
        expected_total = e01 + e02 + e12

        assert sum(result) == pytest.approx(expected_total, rel=1e-4)

    @pytest.mark.parametrize("dtype", [wp.float32, wp.float64])
    def test_dtype_flexibility(self, device, two_atom_system, dtype):
        """Test that both float32 and float64 work."""
        inputs = prepare_csr_inputs(two_atom_system, device, dtype=dtype)
        energies = allocate_energy_output(two_atom_system["num_atoms"], device)
        alpha_arr = make_alpha_array(0.3, device, dtype=dtype)

        ewald_real_space_energy(
            positions=inputs["positions"],
            charges=inputs["charges"],
            cell=inputs["cell"],
            idx_j=inputs["idx_j"],
            neighbor_ptr=inputs["neighbor_ptr"],
            unit_shifts=inputs["unit_shifts"],
            alpha=alpha_arr,
            pair_energies=energies,
            wp_dtype=dtype,
            device=device,
        )

        result = energies.numpy()
        # Just check it runs and gives reasonable result
        assert not np.isnan(result).any()
        assert sum(result) < 0  # Attractive interaction


# ==============================================================================
# Test Class: Ewald Real-Space Energy (Matrix Format)
# ==============================================================================


class TestWpEwaldRealSpaceEnergyMatrix:
    """Tests for wp_ewald_real_space_energy_matrix with neighbor matrix format."""

    def test_matrix_format_energy(self, device, two_atom_matrix_system):
        """Test energy calculation with neighbor matrix format."""
        inputs = prepare_matrix_inputs(two_atom_matrix_system, device)
        energies = allocate_energy_output(two_atom_matrix_system["num_atoms"], device)
        alpha = 0.3
        alpha_arr = make_alpha_array(alpha, device)

        ewald_real_space_energy_matrix(
            positions=inputs["positions"],
            charges=inputs["charges"],
            cell=inputs["cell"],
            neighbor_matrix=inputs["neighbor_matrix"],
            unit_shifts_matrix=inputs["neighbor_shifts"],
            mask_value=inputs["fill_value"],
            alpha=alpha_arr,
            pair_energies=energies,
            wp_dtype=wp.float64,
            device=device,
        )

        result = energies.numpy()
        r = two_atom_matrix_system["distance"]
        erfc_term = math.erfc(alpha * r)
        # Matrix format with full neighbor list: each atom computes 0.5 * E for its neighbors
        # So total = 2 * 0.5 * E = E (since each pair counted from both sides)
        expected_single_pair = (1.0 * -1.0) * erfc_term / r
        assert sum(result) == pytest.approx(expected_single_pair, rel=1e-4)


# ==============================================================================
# Test Class: Ewald Real-Space Energy and Forces
# ==============================================================================


class TestWpEwaldRealSpaceEnergyForces:
    """Tests for wp_ewald_real_space_energy_forces."""

    def test_forces_direction(self, device, two_atom_system):
        """Test forces point in correct direction (attractive for opposite charges)."""
        inputs = prepare_csr_inputs(two_atom_system, device)
        num_atoms = two_atom_system["num_atoms"]

        energies = allocate_energy_output(num_atoms, device)
        forces = allocate_force_output(num_atoms, device)
        alpha_arr = make_alpha_array(0.3, device)

        ewald_real_space_energy_forces(
            positions=inputs["positions"],
            charges=inputs["charges"],
            cell=inputs["cell"],
            idx_j=inputs["idx_j"],
            neighbor_ptr=inputs["neighbor_ptr"],
            unit_shifts=inputs["unit_shifts"],
            alpha=alpha_arr,
            pair_energies=energies,
            atomic_forces=forces,
            wp_dtype=wp.float64,
            device=device,
        )

        result_forces = forces.numpy()
        # Atom 0 at origin, atom 1 at (3,0,0)
        # Opposite charges -> attractive -> atom 0 should be pushed toward +x
        assert result_forces[0, 0] > 0  # Force on atom 0 in +x direction

    def test_newtons_third_law(self, device, two_atom_system):
        """Test that forces sum to zero (Newton's third law)."""
        inputs = prepare_csr_inputs(two_atom_system, device)
        num_atoms = two_atom_system["num_atoms"]

        energies = allocate_energy_output(num_atoms, device)
        forces = allocate_force_output(num_atoms, device)
        alpha_arr = make_alpha_array(0.3, device)

        ewald_real_space_energy_forces(
            positions=inputs["positions"],
            charges=inputs["charges"],
            cell=inputs["cell"],
            idx_j=inputs["idx_j"],
            neighbor_ptr=inputs["neighbor_ptr"],
            unit_shifts=inputs["unit_shifts"],
            alpha=alpha_arr,
            pair_energies=energies,
            atomic_forces=forces,
            wp_dtype=wp.float64,
            device=device,
        )

        result_forces = forces.numpy()
        total_force = np.sum(result_forces, axis=0)
        np.testing.assert_allclose(total_force, [0.0, 0.0, 0.0], atol=1e-12)

    def test_energy_matches_energy_only_kernel(self, device, two_atom_system):
        """Energy from energy+forces kernel should match energy-only kernel."""
        inputs = prepare_csr_inputs(two_atom_system, device)
        num_atoms = two_atom_system["num_atoms"]
        alpha_arr = make_alpha_array(0.3, device)

        # Energy only
        energies_only = allocate_energy_output(num_atoms, device)
        ewald_real_space_energy(
            positions=inputs["positions"],
            charges=inputs["charges"],
            cell=inputs["cell"],
            idx_j=inputs["idx_j"],
            neighbor_ptr=inputs["neighbor_ptr"],
            unit_shifts=inputs["unit_shifts"],
            alpha=alpha_arr,
            pair_energies=energies_only,
            wp_dtype=wp.float64,
            device=device,
        )

        # Energy + forces
        energies_combined = allocate_energy_output(num_atoms, device)
        forces = allocate_force_output(num_atoms, device)
        alpha_arr2 = make_alpha_array(0.3, device)
        ewald_real_space_energy_forces(
            positions=inputs["positions"],
            charges=inputs["charges"],
            cell=inputs["cell"],
            idx_j=inputs["idx_j"],
            neighbor_ptr=inputs["neighbor_ptr"],
            unit_shifts=inputs["unit_shifts"],
            alpha=alpha_arr2,
            pair_energies=energies_combined,
            atomic_forces=forces,
            wp_dtype=wp.float64,
            device=device,
        )

        np.testing.assert_allclose(
            energies_only.numpy(), energies_combined.numpy(), rtol=1e-12
        )


class TestWpEwaldRealSpaceEnergyForcesMatrix:
    """Tests for wp_ewald_real_space_energy_forces_matrix."""

    def test_matrix_forces_sum_to_zero(self, device, two_atom_matrix_system):
        """Test Newton's third law with matrix format."""
        inputs = prepare_matrix_inputs(two_atom_matrix_system, device)
        num_atoms = two_atom_matrix_system["num_atoms"]

        energies = allocate_energy_output(num_atoms, device)
        forces = allocate_force_output(num_atoms, device)
        alpha_arr = make_alpha_array(0.3, device)

        ewald_real_space_energy_forces_matrix(
            positions=inputs["positions"],
            charges=inputs["charges"],
            cell=inputs["cell"],
            neighbor_matrix=inputs["neighbor_matrix"],
            unit_shifts_matrix=inputs["neighbor_shifts"],
            mask_value=inputs["fill_value"],
            alpha=alpha_arr,
            pair_energies=energies,
            atomic_forces=forces,
            wp_dtype=wp.float64,
            device=device,
        )

        result_forces = forces.numpy()
        total_force = np.sum(result_forces, axis=0)
        np.testing.assert_allclose(total_force, [0.0, 0.0, 0.0], atol=1e-12)


# ==============================================================================
# Test Class: Batched Ewald Real-Space
# ==============================================================================


class TestWpBatchEwaldRealSpaceEnergy:
    """Tests for batched Ewald real-space kernels."""

    def test_batch_energy_independent_systems(self, device, batch_two_systems):
        """Test that batch energy correctly handles independent systems."""
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
        # Per-system alpha values
        alpha_arr = wp.from_numpy(
            np.array([0.3, 0.3], dtype=np.float64), dtype=wp.float64, device=device
        )

        energies = allocate_energy_output(batch_two_systems["num_atoms"], device)

        batch_ewald_real_space_energy(
            positions=positions,
            charges=charges,
            cell=cell,
            batch_id=batch_idx,
            idx_j=idx_j,
            neighbor_ptr=neighbor_ptr,
            unit_shifts=unit_shifts,
            alpha=alpha_arr,
            pair_energies=energies,
            wp_dtype=wp.float64,
            device=device,
        )

        result = energies.numpy()

        # System 0: atoms 0,1, distance=2, charges +1,-1
        # System 1: atoms 2,3, distance=4, charges +2,-1
        alpha = 0.3
        # Half neighbor list with 0.5 prefactor
        expected_sys0 = 0.5 * (1.0 * -1.0) * math.erfc(alpha * 2.0) / 2.0
        expected_sys1 = 0.5 * (2.0 * -1.0) * math.erfc(alpha * 4.0) / 4.0

        system_0_energy = result[0] + result[1]
        system_1_energy = result[2] + result[3]

        assert system_0_energy == pytest.approx(expected_sys0, rel=1e-4)
        assert system_1_energy == pytest.approx(expected_sys1, rel=1e-4)

    def test_batch_energy_forces(self, device, batch_two_systems):
        """Test batched energy and forces computation."""
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
        alpha_arr = wp.from_numpy(
            np.array([0.3, 0.3], dtype=np.float64), dtype=wp.float64, device=device
        )

        energies = allocate_energy_output(batch_two_systems["num_atoms"], device)
        forces = allocate_force_output(batch_two_systems["num_atoms"], device)

        batch_ewald_real_space_energy_forces(
            positions=positions,
            charges=charges,
            cell=cell,
            batch_id=batch_idx,
            idx_j=idx_j,
            neighbor_ptr=neighbor_ptr,
            unit_shifts=unit_shifts,
            alpha=alpha_arr,
            pair_energies=energies,
            atomic_forces=forces,
            wp_dtype=wp.float64,
            device=device,
        )

        result_forces = forces.numpy()

        # Total force for each system should be zero
        sys0_total = result_forces[0] + result_forces[1]
        sys1_total = result_forces[2] + result_forces[3]

        np.testing.assert_allclose(sys0_total, [0.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(sys1_total, [0.0, 0.0, 0.0], atol=1e-12)


class TestWpBatchEwaldRealSpaceMatrix:
    """Tests for batched Ewald real-space with neighbor matrix format."""

    def test_batch_matrix_energy(self, device, batch_two_systems_matrix):
        """Test batched energy with matrix format."""
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
        alpha_arr = wp.from_numpy(
            np.array([0.3, 0.3], dtype=np.float64), dtype=wp.float64, device=device
        )

        energies = allocate_energy_output(batch_two_systems_matrix["num_atoms"], device)

        batch_ewald_real_space_energy_matrix(
            positions=positions,
            charges=charges,
            cell=cell,
            batch_id=batch_idx,
            neighbor_matrix=neighbor_matrix,
            unit_shifts_matrix=neighbor_shifts,
            mask_value=batch_two_systems_matrix["fill_value"],
            alpha=alpha_arr,
            pair_energies=energies,
            wp_dtype=wp.float64,
            device=device,
        )

        result = energies.numpy()
        # Verify we get reasonable results (not NaN/inf)
        assert not np.isnan(result).any()
        assert not np.isinf(result).any()
        # Both systems have attractive interactions
        assert sum(result[:2]) < 0  # System 0
        assert sum(result[2:]) < 0  # System 1


# ==============================================================================
# Test Class: Reciprocal Space Structure Factors
# ==============================================================================


class TestWpEwaldReciprocalSpaceStructureFactors:
    """Tests for wp_ewald_reciprocal_space_fill_structure_factors."""

    def test_structure_factors_computation(self, device, simple_kvector_system):
        """Test structure factor computation."""
        sys = simple_kvector_system
        dtype = wp.float64
        vec_dtype = wp.vec3d
        mat_dtype = wp.mat33d

        positions = wp.from_numpy(sys["positions"], dtype=vec_dtype, device=device)
        charges = wp.from_numpy(sys["charges"], dtype=dtype, device=device)
        k_vectors = wp.from_numpy(sys["k_vectors"], dtype=vec_dtype, device=device)
        cell = wp.from_numpy(sys["cell"], dtype=mat_dtype, device=device)
        alpha_arr = make_alpha_array(0.5, device, dtype)

        num_k = sys["num_k"]
        num_atoms = sys["num_atoms"]

        # Allocate outputs
        total_charge = wp.zeros(1, dtype=wp.float64, device=device)
        cos_k_dot_r = wp.zeros((num_k, num_atoms), dtype=wp.float64, device=device)
        sin_k_dot_r = wp.zeros((num_k, num_atoms), dtype=wp.float64, device=device)
        real_sf = wp.zeros(num_k, dtype=wp.float64, device=device)
        imag_sf = wp.zeros(num_k, dtype=wp.float64, device=device)

        ewald_reciprocal_space_fill_structure_factors(
            positions=positions,
            charges=charges,
            k_vectors=k_vectors,
            cell=cell,
            alpha=alpha_arr,
            total_charge=total_charge,
            cos_k_dot_r=cos_k_dot_r,
            sin_k_dot_r=sin_k_dot_r,
            real_structure_factors=real_sf,
            imag_structure_factors=imag_sf,
            wp_dtype=dtype,
            device=device,
        )

        # Verify outputs are computed
        cos_arr = cos_k_dot_r.numpy()
        sin_arr = sin_k_dot_r.numpy()
        real_sf_arr = real_sf.numpy()
        imag_sf_arr = imag_sf.numpy()

        # cos and sin should be in [-1, 1]
        assert np.all(cos_arr >= -1.0 - 1e-10)
        assert np.all(cos_arr <= 1.0 + 1e-10)
        assert np.all(sin_arr >= -1.0 - 1e-10)
        assert np.all(sin_arr <= 1.0 + 1e-10)

        # Structure factors should be computed
        assert not np.isnan(real_sf_arr).any()
        assert not np.isnan(imag_sf_arr).any()

    def test_neutral_system_total_charge(self, device, simple_kvector_system):
        """Test that total_charge is computed correctly for neutral system."""
        sys = simple_kvector_system
        dtype = wp.float64
        vec_dtype = wp.vec3d
        mat_dtype = wp.mat33d

        positions = wp.from_numpy(sys["positions"], dtype=vec_dtype, device=device)
        charges = wp.from_numpy(sys["charges"], dtype=dtype, device=device)
        k_vectors = wp.from_numpy(sys["k_vectors"], dtype=vec_dtype, device=device)
        cell = wp.from_numpy(sys["cell"], dtype=mat_dtype, device=device)
        alpha_arr = make_alpha_array(0.5, device, dtype)

        num_k = sys["num_k"]
        num_atoms = sys["num_atoms"]

        total_charge = wp.zeros(1, dtype=wp.float64, device=device)
        cos_k_dot_r = wp.zeros((num_k, num_atoms), dtype=wp.float64, device=device)
        sin_k_dot_r = wp.zeros((num_k, num_atoms), dtype=wp.float64, device=device)
        real_sf = wp.zeros(num_k, dtype=wp.float64, device=device)
        imag_sf = wp.zeros(num_k, dtype=wp.float64, device=device)

        ewald_reciprocal_space_fill_structure_factors(
            positions=positions,
            charges=charges,
            k_vectors=k_vectors,
            cell=cell,
            alpha=alpha_arr,
            total_charge=total_charge,
            cos_k_dot_r=cos_k_dot_r,
            sin_k_dot_r=sin_k_dot_r,
            real_structure_factors=real_sf,
            imag_structure_factors=imag_sf,
            wp_dtype=dtype,
            device=device,
        )

        # For neutral system (sum of charges = 0), Q_total/V should be 0
        tc_val = total_charge.numpy()[0]
        # Q_total = sum(charges) = 1 + (-1) = 0
        # total_charge stores Q_total / V, so should be 0
        assert tc_val == pytest.approx(0.0, abs=1e-10)


# ==============================================================================
# Test Class: Reciprocal Space Energy Computation
# ==============================================================================


class TestWpEwaldReciprocalSpaceEnergy:
    """Tests for wp_ewald_reciprocal_space_compute_energy."""

    def test_reciprocal_energy_computation(self, device, simple_kvector_system):
        """Test reciprocal-space energy computation."""
        sys = simple_kvector_system
        dtype = wp.float64
        vec_dtype = wp.vec3d
        mat_dtype = wp.mat33d

        positions = wp.from_numpy(sys["positions"], dtype=vec_dtype, device=device)
        charges = wp.from_numpy(sys["charges"], dtype=dtype, device=device)
        k_vectors = wp.from_numpy(sys["k_vectors"], dtype=vec_dtype, device=device)
        cell = wp.from_numpy(sys["cell"], dtype=mat_dtype, device=device)
        alpha_arr = make_alpha_array(0.5, device, dtype)

        num_k = sys["num_k"]
        num_atoms = sys["num_atoms"]

        # First compute structure factors
        total_charge = wp.zeros(1, dtype=wp.float64, device=device)
        cos_k_dot_r = wp.zeros((num_k, num_atoms), dtype=wp.float64, device=device)
        sin_k_dot_r = wp.zeros((num_k, num_atoms), dtype=wp.float64, device=device)
        real_sf = wp.zeros(num_k, dtype=wp.float64, device=device)
        imag_sf = wp.zeros(num_k, dtype=wp.float64, device=device)

        ewald_reciprocal_space_fill_structure_factors(
            positions=positions,
            charges=charges,
            k_vectors=k_vectors,
            cell=cell,
            alpha=alpha_arr,
            total_charge=total_charge,
            cos_k_dot_r=cos_k_dot_r,
            sin_k_dot_r=sin_k_dot_r,
            real_structure_factors=real_sf,
            imag_structure_factors=imag_sf,
            wp_dtype=dtype,
            device=device,
        )

        # Then compute energy
        reciprocal_energies = wp.zeros(num_atoms, dtype=wp.float64, device=device)

        ewald_reciprocal_space_compute_energy(
            charges=charges,
            cos_k_dot_r=cos_k_dot_r,
            sin_k_dot_r=sin_k_dot_r,
            real_structure_factors=real_sf,
            imag_structure_factors=imag_sf,
            reciprocal_energies=reciprocal_energies,
            wp_dtype=dtype,
            device=device,
        )

        energies = reciprocal_energies.numpy()
        # Energy should be computed (not NaN/inf)
        assert not np.isnan(energies).any()
        assert not np.isinf(energies).any()


# ==============================================================================
# Test Class: Self-Energy Subtraction
# ==============================================================================


class TestWpEwaldSubtractSelfEnergy:
    """Tests for wp_ewald_subtract_self_energy."""

    def test_self_energy_correction(self, device, simple_kvector_system):
        """Test self-energy correction is applied."""
        sys = simple_kvector_system
        dtype = wp.float64
        num_atoms = sys["num_atoms"]

        charges = wp.from_numpy(sys["charges"], dtype=dtype, device=device)
        alpha = 0.5
        alpha_arr = make_alpha_array(alpha, device, dtype)

        # Neutral system: total_charge/V = 0
        total_charge = wp.zeros(1, dtype=wp.float64, device=device)

        # Fake input energies
        energy_in = wp.from_numpy(
            np.array([1.0, 2.0], dtype=np.float64), dtype=wp.float64, device=device
        )
        energy_out = wp.zeros(num_atoms, dtype=wp.float64, device=device)

        ewald_subtract_self_energy(
            charges=charges,
            alpha=alpha_arr,
            total_charge=total_charge,
            energy_in=energy_in,
            energy_out=energy_out,
            wp_dtype=dtype,
            device=device,
        )

        result = energy_out.numpy()
        # Self-energy = alpha/sqrt(pi) * q^2 for each atom
        # For charges [1, -1] and alpha=0.5:
        # self_0 = 0.5/sqrt(pi) * 1^2 ≈ 0.282
        # self_1 = 0.5/sqrt(pi) * (-1)^2 ≈ 0.282
        self_factor = alpha / math.sqrt(math.pi)
        expected_0 = 1.0 - self_factor * 1.0**2  # energy_in - self
        expected_1 = 2.0 - self_factor * (-1.0) ** 2

        assert result[0] == pytest.approx(expected_0, rel=1e-6)
        assert result[1] == pytest.approx(expected_1, rel=1e-6)


# ==============================================================================
# Test Class: Reciprocal Space Energy and Forces
# ==============================================================================


class TestWpEwaldReciprocalSpaceEnergyForces:
    """Tests for wp_ewald_reciprocal_space_energy_forces."""

    def test_reciprocal_forces_sum_to_zero(self, device, simple_kvector_system):
        """Test that reciprocal-space forces sum to zero."""
        sys = simple_kvector_system
        dtype = wp.float64
        vec_dtype = wp.vec3d
        mat_dtype = wp.mat33d

        positions = wp.from_numpy(sys["positions"], dtype=vec_dtype, device=device)
        charges = wp.from_numpy(sys["charges"], dtype=dtype, device=device)
        k_vectors = wp.from_numpy(sys["k_vectors"], dtype=vec_dtype, device=device)
        cell = wp.from_numpy(sys["cell"], dtype=mat_dtype, device=device)
        alpha_arr = make_alpha_array(0.5, device, dtype)

        num_k = sys["num_k"]
        num_atoms = sys["num_atoms"]

        # First compute structure factors
        total_charge = wp.zeros(1, dtype=wp.float64, device=device)
        cos_k_dot_r = wp.zeros((num_k, num_atoms), dtype=wp.float64, device=device)
        sin_k_dot_r = wp.zeros((num_k, num_atoms), dtype=wp.float64, device=device)
        real_sf = wp.zeros(num_k, dtype=wp.float64, device=device)
        imag_sf = wp.zeros(num_k, dtype=wp.float64, device=device)

        ewald_reciprocal_space_fill_structure_factors(
            positions=positions,
            charges=charges,
            k_vectors=k_vectors,
            cell=cell,
            alpha=alpha_arr,
            total_charge=total_charge,
            cos_k_dot_r=cos_k_dot_r,
            sin_k_dot_r=sin_k_dot_r,
            real_structure_factors=real_sf,
            imag_structure_factors=imag_sf,
            wp_dtype=dtype,
            device=device,
        )

        # Compute energy and forces
        reciprocal_energies = wp.zeros(num_atoms, dtype=wp.float64, device=device)
        forces = allocate_force_output(num_atoms, device)

        ewald_reciprocal_space_energy_forces(
            charges=charges,
            k_vectors=k_vectors,
            cos_k_dot_r=cos_k_dot_r,
            sin_k_dot_r=sin_k_dot_r,
            real_structure_factors=real_sf,
            imag_structure_factors=imag_sf,
            reciprocal_energies=reciprocal_energies,
            atomic_forces=forces,
            wp_dtype=dtype,
            device=device,
        )

        result_forces = forces.numpy()
        total_force = np.sum(result_forces, axis=0)
        # Total force should be zero (momentum conservation)
        np.testing.assert_allclose(total_force, [0.0, 0.0, 0.0], atol=1e-10)


# ==============================================================================
# Test Class: Regression Tests
# ==============================================================================


class TestEwaldKernelsRegression:
    """Regression tests with hardcoded expected values."""

    def test_real_space_energy_regression(self, device):
        """Regression test for real-space energy with known values."""
        # Two atoms, distance=3, charges +1 and -1, alpha=0.3
        positions = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float64)
        charges = np.array([1.0, -1.0], dtype=np.float64)
        cell = np.array(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=np.float64,
        )
        idx_j = np.array([1], dtype=np.int32)
        neighbor_ptr = np.array([0, 1, 1], dtype=np.int32)
        unit_shifts = np.array([[0, 0, 0]], dtype=np.int32)

        positions_wp = wp.from_numpy(positions, dtype=wp.vec3d, device=device)
        charges_wp = wp.from_numpy(charges, dtype=wp.float64, device=device)
        cell_wp = wp.from_numpy(cell, dtype=wp.mat33d, device=device)
        idx_j_wp = wp.from_numpy(idx_j, dtype=wp.int32, device=device)
        neighbor_ptr_wp = wp.from_numpy(neighbor_ptr, dtype=wp.int32, device=device)
        unit_shifts_wp = wp.from_numpy(unit_shifts, dtype=wp.vec3i, device=device)
        alpha_arr = make_alpha_array(0.3, device)

        energies = allocate_energy_output(2, device)

        ewald_real_space_energy(
            positions=positions_wp,
            charges=charges_wp,
            cell=cell_wp,
            idx_j=idx_j_wp,
            neighbor_ptr=neighbor_ptr_wp,
            unit_shifts=unit_shifts_wp,
            alpha=alpha_arr,
            pair_energies=energies,
            wp_dtype=wp.float64,
            device=device,
        )

        result = energies.numpy()
        # Expected: 0.5 * q1 * q2 * erfc(alpha*r) / r
        # = 0.5 * 1 * (-1) * erfc(0.9) / 3
        # erfc(0.9) ≈ 0.20309
        expected = 0.5 * (1.0) * (-1.0) * math.erfc(0.3 * 3.0) / 3.0
        assert sum(result) == pytest.approx(expected, rel=1e-4)


# ==============================================================================
# Test Class: Ewald Real-Space Energy Forces with Charge Gradients (CSR Format)
# ==============================================================================


class TestWpEwaldRealSpaceEnergyForcesChargeGrad:
    """Tests for wp_ewald_real_space_energy_forces_charge_grad with CSR format."""

    def test_charge_grad_shape(self, device, two_atom_system):
        """Test that charge gradients have correct shape."""
        inputs = prepare_csr_inputs(two_atom_system, device)
        num_atoms = two_atom_system["num_atoms"]

        energies = allocate_energy_output(num_atoms, device)
        forces = allocate_force_output(num_atoms, device)
        charge_grads = allocate_charge_grad_output(num_atoms, device)
        alpha_arr = make_alpha_array(0.3, device)

        ewald_real_space_energy_forces_charge_grad(
            positions=inputs["positions"],
            charges=inputs["charges"],
            cell=inputs["cell"],
            idx_j=inputs["idx_j"],
            neighbor_ptr=inputs["neighbor_ptr"],
            unit_shifts=inputs["unit_shifts"],
            alpha=alpha_arr,
            pair_energies=energies,
            atomic_forces=forces,
            charge_gradients=charge_grads,
            wp_dtype=wp.float64,
            device=device,
        )

        result_grads = charge_grads.numpy()
        assert result_grads.shape == (num_atoms,)
        assert not np.isnan(result_grads).any()

    def test_energy_matches_energy_forces_kernel(self, device, two_atom_system):
        """Energy from charge_grad kernel should match energy_forces kernel."""
        inputs = prepare_csr_inputs(two_atom_system, device)
        num_atoms = two_atom_system["num_atoms"]
        alpha_arr = make_alpha_array(0.3, device)

        # Energy + forces only
        energies_ef = allocate_energy_output(num_atoms, device)
        forces_ef = allocate_force_output(num_atoms, device)
        alpha_arr2 = make_alpha_array(0.3, device)
        ewald_real_space_energy_forces(
            positions=inputs["positions"],
            charges=inputs["charges"],
            cell=inputs["cell"],
            idx_j=inputs["idx_j"],
            neighbor_ptr=inputs["neighbor_ptr"],
            unit_shifts=inputs["unit_shifts"],
            alpha=alpha_arr2,
            pair_energies=energies_ef,
            atomic_forces=forces_ef,
            wp_dtype=wp.float64,
            device=device,
        )

        # Energy + forces + charge gradients
        energies_cg = allocate_energy_output(num_atoms, device)
        forces_cg = allocate_force_output(num_atoms, device)
        charge_grads = allocate_charge_grad_output(num_atoms, device)
        ewald_real_space_energy_forces_charge_grad(
            positions=inputs["positions"],
            charges=inputs["charges"],
            cell=inputs["cell"],
            idx_j=inputs["idx_j"],
            neighbor_ptr=inputs["neighbor_ptr"],
            unit_shifts=inputs["unit_shifts"],
            alpha=alpha_arr,
            pair_energies=energies_cg,
            atomic_forces=forces_cg,
            charge_gradients=charge_grads,
            wp_dtype=wp.float64,
            device=device,
        )

        np.testing.assert_allclose(energies_ef.numpy(), energies_cg.numpy(), rtol=1e-12)
        np.testing.assert_allclose(forces_ef.numpy(), forces_cg.numpy(), rtol=1e-12)

    def test_charge_gradient_values(self, device, two_atom_system):
        """Test charge gradients are physically reasonable."""
        inputs = prepare_csr_inputs(two_atom_system, device)
        num_atoms = two_atom_system["num_atoms"]

        energies = allocate_energy_output(num_atoms, device)
        forces = allocate_force_output(num_atoms, device)
        charge_grads = allocate_charge_grad_output(num_atoms, device)
        alpha_arr = make_alpha_array(0.3, device)

        ewald_real_space_energy_forces_charge_grad(
            positions=inputs["positions"],
            charges=inputs["charges"],
            cell=inputs["cell"],
            idx_j=inputs["idx_j"],
            neighbor_ptr=inputs["neighbor_ptr"],
            unit_shifts=inputs["unit_shifts"],
            alpha=alpha_arr,
            pair_energies=energies,
            atomic_forces=forces,
            charge_gradients=charge_grads,
            wp_dtype=wp.float64,
            device=device,
        )

        result_grads = charge_grads.numpy()
        # dE/dq_i = sum_j q_j * erfc(alpha*r_ij) / r_ij
        # For two opposite charges, gradients should have opposite signs
        # (increasing q0 makes the interaction more negative, and vice versa)
        assert not np.allclose(result_grads[0], result_grads[1])


# ==============================================================================
# Test Class: Ewald Real-Space Energy Forces Charge Grad (Matrix Format)
# ==============================================================================


class TestWpEwaldRealSpaceEnergyForcesChargeGradMatrix:
    """Tests for wp_ewald_real_space_energy_forces_charge_grad_matrix."""

    def test_matrix_charge_grad_shape(self, device, two_atom_matrix_system):
        """Test charge gradients have correct shape with matrix format."""
        inputs = prepare_matrix_inputs(two_atom_matrix_system, device)
        num_atoms = two_atom_matrix_system["num_atoms"]

        energies = allocate_energy_output(num_atoms, device)
        forces = allocate_force_output(num_atoms, device)
        charge_grads = allocate_charge_grad_output(num_atoms, device)
        alpha_arr = make_alpha_array(0.3, device)

        ewald_real_space_energy_forces_charge_grad_matrix(
            positions=inputs["positions"],
            charges=inputs["charges"],
            cell=inputs["cell"],
            neighbor_matrix=inputs["neighbor_matrix"],
            unit_shifts_matrix=inputs["neighbor_shifts"],
            mask_value=inputs["fill_value"],
            alpha=alpha_arr,
            pair_energies=energies,
            atomic_forces=forces,
            charge_gradients=charge_grads,
            wp_dtype=wp.float64,
            device=device,
        )

        result_grads = charge_grads.numpy()
        assert result_grads.shape == (num_atoms,)
        assert not np.isnan(result_grads).any()

    def test_matrix_forces_sum_to_zero(self, device, two_atom_matrix_system):
        """Test Newton's third law with matrix format + charge grads."""
        inputs = prepare_matrix_inputs(two_atom_matrix_system, device)
        num_atoms = two_atom_matrix_system["num_atoms"]

        energies = allocate_energy_output(num_atoms, device)
        forces = allocate_force_output(num_atoms, device)
        charge_grads = allocate_charge_grad_output(num_atoms, device)
        alpha_arr = make_alpha_array(0.3, device)

        ewald_real_space_energy_forces_charge_grad_matrix(
            positions=inputs["positions"],
            charges=inputs["charges"],
            cell=inputs["cell"],
            neighbor_matrix=inputs["neighbor_matrix"],
            unit_shifts_matrix=inputs["neighbor_shifts"],
            mask_value=inputs["fill_value"],
            alpha=alpha_arr,
            pair_energies=energies,
            atomic_forces=forces,
            charge_gradients=charge_grads,
            wp_dtype=wp.float64,
            device=device,
        )

        result_forces = forces.numpy()
        total_force = np.sum(result_forces, axis=0)
        np.testing.assert_allclose(total_force, [0.0, 0.0, 0.0], atol=1e-12)


# ==============================================================================
# Test Class: Batch Ewald Real-Space Energy Forces (Matrix Format)
# ==============================================================================


class TestWpBatchEwaldRealSpaceEnergyForcesMatrix:
    """Tests for wp_batch_ewald_real_space_energy_forces_matrix."""

    def test_batch_matrix_forces_sum_to_zero(self, device, batch_two_systems_matrix):
        """Test Newton's third law for batched matrix format."""
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
        alpha_arr = wp.from_numpy(
            np.array([0.3, 0.3], dtype=np.float64), dtype=wp.float64, device=device
        )

        num_atoms = batch_two_systems_matrix["num_atoms"]
        energies = allocate_energy_output(num_atoms, device)
        forces = allocate_force_output(num_atoms, device)

        batch_ewald_real_space_energy_forces_matrix(
            positions=positions,
            charges=charges,
            cell=cell,
            batch_id=batch_idx,
            neighbor_matrix=neighbor_matrix,
            unit_shifts_matrix=neighbor_shifts,
            mask_value=batch_two_systems_matrix["fill_value"],
            alpha=alpha_arr,
            pair_energies=energies,
            atomic_forces=forces,
            wp_dtype=wp.float64,
            device=device,
        )

        result_forces = forces.numpy()
        # Per-system momentum conservation
        sys0_total = result_forces[0] + result_forces[1]
        sys1_total = result_forces[2] + result_forces[3]

        np.testing.assert_allclose(sys0_total, [0.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(sys1_total, [0.0, 0.0, 0.0], atol=1e-12)


# ==============================================================================
# Test Class: Batch Ewald Real-Space Energy Forces Charge Grad (CSR Format)
# ==============================================================================


class TestWpBatchEwaldRealSpaceEnergyForcesChargeGrad:
    """Tests for wp_batch_ewald_real_space_energy_forces_charge_grad."""

    def test_batch_charge_grad_shape(self, device, batch_two_systems):
        """Test batch charge gradients have correct shape."""
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
        alpha_arr = wp.from_numpy(
            np.array([0.3, 0.3], dtype=np.float64), dtype=wp.float64, device=device
        )

        num_atoms = batch_two_systems["num_atoms"]
        energies = allocate_energy_output(num_atoms, device)
        forces = allocate_force_output(num_atoms, device)
        charge_grads = allocate_charge_grad_output(num_atoms, device)

        batch_ewald_real_space_energy_forces_charge_grad(
            positions=positions,
            charges=charges,
            cell=cell,
            batch_id=batch_idx,
            idx_j=idx_j,
            neighbor_ptr=neighbor_ptr,
            unit_shifts=unit_shifts,
            alpha=alpha_arr,
            pair_energies=energies,
            atomic_forces=forces,
            charge_gradients=charge_grads,
            wp_dtype=wp.float64,
            device=device,
        )

        result_grads = charge_grads.numpy()
        assert result_grads.shape == (num_atoms,)
        assert not np.isnan(result_grads).any()


# ==============================================================================
# Test Class: Batch Ewald Real-Space Energy Forces Charge Grad (Matrix Format)
# ==============================================================================


class TestWpBatchEwaldRealSpaceEnergyForcesChargeGradMatrix:
    """Tests for wp_batch_ewald_real_space_energy_forces_charge_grad_matrix."""

    def test_batch_matrix_charge_grad_shape(self, device, batch_two_systems_matrix):
        """Test batch charge gradients with matrix format."""
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
        alpha_arr = wp.from_numpy(
            np.array([0.3, 0.3], dtype=np.float64), dtype=wp.float64, device=device
        )

        num_atoms = batch_two_systems_matrix["num_atoms"]
        energies = allocate_energy_output(num_atoms, device)
        forces = allocate_force_output(num_atoms, device)
        charge_grads = allocate_charge_grad_output(num_atoms, device)

        batch_ewald_real_space_energy_forces_charge_grad_matrix(
            positions=positions,
            charges=charges,
            cell=cell,
            batch_id=batch_idx,
            neighbor_matrix=neighbor_matrix,
            unit_shifts_matrix=neighbor_shifts,
            mask_value=batch_two_systems_matrix["fill_value"],
            alpha=alpha_arr,
            pair_energies=energies,
            atomic_forces=forces,
            charge_gradients=charge_grads,
            wp_dtype=wp.float64,
            device=device,
        )

        result_grads = charge_grads.numpy()
        assert result_grads.shape == (num_atoms,)
        assert not np.isnan(result_grads).any()

        # Forces should still sum to zero per system
        result_forces = forces.numpy()
        sys0_total = result_forces[0] + result_forces[1]
        sys1_total = result_forces[2] + result_forces[3]
        np.testing.assert_allclose(sys0_total, [0.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(sys1_total, [0.0, 0.0, 0.0], atol=1e-12)


# ==============================================================================
# Test Class: Reciprocal Space Energy Forces with Charge Gradients
# ==============================================================================


class TestWpEwaldReciprocalSpaceEnergyForcesChargeGrad:
    """Tests for wp_ewald_reciprocal_space_energy_forces_charge_grad."""

    def test_reciprocal_charge_grad_shape(self, device, simple_kvector_system):
        """Test reciprocal-space charge gradients have correct shape."""
        sys = simple_kvector_system
        dtype = wp.float64
        vec_dtype = wp.vec3d
        mat_dtype = wp.mat33d

        positions = wp.from_numpy(sys["positions"], dtype=vec_dtype, device=device)
        charges = wp.from_numpy(sys["charges"], dtype=dtype, device=device)
        k_vectors = wp.from_numpy(sys["k_vectors"], dtype=vec_dtype, device=device)
        cell = wp.from_numpy(sys["cell"], dtype=mat_dtype, device=device)
        alpha_arr = make_alpha_array(0.5, device, dtype)

        num_k = sys["num_k"]
        num_atoms = sys["num_atoms"]

        # First compute structure factors
        total_charge = wp.zeros(1, dtype=wp.float64, device=device)
        cos_k_dot_r = wp.zeros((num_k, num_atoms), dtype=wp.float64, device=device)
        sin_k_dot_r = wp.zeros((num_k, num_atoms), dtype=wp.float64, device=device)
        real_sf = wp.zeros(num_k, dtype=wp.float64, device=device)
        imag_sf = wp.zeros(num_k, dtype=wp.float64, device=device)

        ewald_reciprocal_space_fill_structure_factors(
            positions=positions,
            charges=charges,
            k_vectors=k_vectors,
            cell=cell,
            alpha=alpha_arr,
            total_charge=total_charge,
            cos_k_dot_r=cos_k_dot_r,
            sin_k_dot_r=sin_k_dot_r,
            real_structure_factors=real_sf,
            imag_structure_factors=imag_sf,
            wp_dtype=dtype,
            device=device,
        )

        # Compute energy, forces, and charge gradients
        reciprocal_energies = wp.zeros(num_atoms, dtype=wp.float64, device=device)
        forces = allocate_force_output(num_atoms, device)
        charge_grads = allocate_charge_grad_output(num_atoms, device)

        ewald_reciprocal_space_energy_forces_charge_grad(
            charges=charges,
            k_vectors=k_vectors,
            cos_k_dot_r=cos_k_dot_r,
            sin_k_dot_r=sin_k_dot_r,
            real_structure_factors=real_sf,
            imag_structure_factors=imag_sf,
            reciprocal_energies=reciprocal_energies,
            atomic_forces=forces,
            charge_gradients=charge_grads,
            wp_dtype=dtype,
            device=device,
        )

        result_grads = charge_grads.numpy()
        assert result_grads.shape == (num_atoms,)
        assert not np.isnan(result_grads).any()

    def test_reciprocal_charge_grad_forces_sum_to_zero(
        self, device, simple_kvector_system
    ):
        """Test that forces still sum to zero with charge_grad kernel."""
        sys = simple_kvector_system
        dtype = wp.float64
        vec_dtype = wp.vec3d
        mat_dtype = wp.mat33d

        positions = wp.from_numpy(sys["positions"], dtype=vec_dtype, device=device)
        charges = wp.from_numpy(sys["charges"], dtype=dtype, device=device)
        k_vectors = wp.from_numpy(sys["k_vectors"], dtype=vec_dtype, device=device)
        cell = wp.from_numpy(sys["cell"], dtype=mat_dtype, device=device)
        alpha_arr = make_alpha_array(0.5, device, dtype)

        num_k = sys["num_k"]
        num_atoms = sys["num_atoms"]

        # First compute structure factors
        total_charge = wp.zeros(1, dtype=wp.float64, device=device)
        cos_k_dot_r = wp.zeros((num_k, num_atoms), dtype=wp.float64, device=device)
        sin_k_dot_r = wp.zeros((num_k, num_atoms), dtype=wp.float64, device=device)
        real_sf = wp.zeros(num_k, dtype=wp.float64, device=device)
        imag_sf = wp.zeros(num_k, dtype=wp.float64, device=device)

        ewald_reciprocal_space_fill_structure_factors(
            positions=positions,
            charges=charges,
            k_vectors=k_vectors,
            cell=cell,
            alpha=alpha_arr,
            total_charge=total_charge,
            cos_k_dot_r=cos_k_dot_r,
            sin_k_dot_r=sin_k_dot_r,
            real_structure_factors=real_sf,
            imag_structure_factors=imag_sf,
            wp_dtype=dtype,
            device=device,
        )

        # Compute energy, forces, and charge gradients
        reciprocal_energies = wp.zeros(num_atoms, dtype=wp.float64, device=device)
        forces = allocate_force_output(num_atoms, device)
        charge_grads = allocate_charge_grad_output(num_atoms, device)

        ewald_reciprocal_space_energy_forces_charge_grad(
            charges=charges,
            k_vectors=k_vectors,
            cos_k_dot_r=cos_k_dot_r,
            sin_k_dot_r=sin_k_dot_r,
            real_structure_factors=real_sf,
            imag_structure_factors=imag_sf,
            reciprocal_energies=reciprocal_energies,
            atomic_forces=forces,
            charge_gradients=charge_grads,
            wp_dtype=dtype,
            device=device,
        )

        result_forces = forces.numpy()
        total_force = np.sum(result_forces, axis=0)
        np.testing.assert_allclose(total_force, [0.0, 0.0, 0.0], atol=1e-10)


# ==============================================================================
# Test Class: Batched Reciprocal Space Structure Factors
# ==============================================================================


@pytest.fixture(scope="session")
def batch_kvector_system():
    """Two systems with k-vectors for batched reciprocal-space tests.

    Returns
    -------
    dict
        Batched system parameters including k-vectors
    """
    # 2 systems, 2 atoms each
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.5, 2.5, 2.5],  # System 0
            [0.0, 0.0, 0.0],
            [3.0, 3.0, 3.0],  # System 1
        ],
        dtype=np.float64,
    )
    charges = np.array([1.0, -1.0, 0.5, -0.5], dtype=np.float64)
    L = 10.0
    cell = np.array(
        [
            [[L, 0.0, 0.0], [0.0, L, 0.0], [0.0, 0.0, L]],
            [[L, 0.0, 0.0], [0.0, L, 0.0], [0.0, 0.0, L]],
        ],
        dtype=np.float64,
    )
    volume = L**3

    # k-vectors for both systems (same k-vectors)
    k_factor = 2.0 * np.pi / L
    k_vecs_single = (
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        * k_factor
    )
    # Shape: (B, K, 3)
    k_vectors = np.stack([k_vecs_single, k_vecs_single], axis=0)

    batch_idx = np.array([0, 0, 1, 1], dtype=np.int32)
    atom_start = np.array([0, 2], dtype=np.int32)
    atom_end = np.array([2, 4], dtype=np.int32)

    return {
        "positions": positions,
        "charges": charges,
        "cell": cell,
        "k_vectors": k_vectors,
        "volume": volume,
        "batch_idx": batch_idx,
        "atom_start": atom_start,
        "atom_end": atom_end,
        "num_atoms": 4,
        "num_k": 4,
        "num_systems": 2,
    }


class TestWpBatchEwaldReciprocalSpaceStructureFactors:
    """Tests for wp_batch_ewald_reciprocal_space_fill_structure_factors."""

    def test_batch_structure_factors_shape(self, device, batch_kvector_system):
        """Test batched structure factor outputs have correct shapes."""
        sys = batch_kvector_system
        dtype = wp.float64
        vec_dtype = wp.vec3d
        mat_dtype = wp.mat33d

        positions = wp.from_numpy(sys["positions"], dtype=vec_dtype, device=device)
        charges = wp.from_numpy(sys["charges"], dtype=dtype, device=device)
        k_vectors = wp.from_numpy(sys["k_vectors"], dtype=vec_dtype, device=device)
        cell = wp.from_numpy(sys["cell"], dtype=mat_dtype, device=device)
        alpha_arr = wp.from_numpy(
            np.array([0.5, 0.5], dtype=np.float64), dtype=dtype, device=device
        )
        atom_start = wp.from_numpy(sys["atom_start"], dtype=wp.int32, device=device)
        atom_end = wp.from_numpy(sys["atom_end"], dtype=wp.int32, device=device)

        num_k = sys["num_k"]
        num_atoms = sys["num_atoms"]
        num_systems = sys["num_systems"]

        # Allocate outputs
        total_charges = wp.zeros(num_systems, dtype=wp.float64, device=device)
        cos_k_dot_r = wp.zeros((num_k, num_atoms), dtype=wp.float64, device=device)
        sin_k_dot_r = wp.zeros((num_k, num_atoms), dtype=wp.float64, device=device)
        real_sf = wp.zeros((num_systems, num_k), dtype=wp.float64, device=device)
        imag_sf = wp.zeros((num_systems, num_k), dtype=wp.float64, device=device)

        # max_blocks_per_system: ceil(max_atoms_per_system / BATCH_BLOCK_SIZE)
        max_blocks_per_system = 1

        batch_ewald_reciprocal_space_fill_structure_factors(
            positions=positions,
            charges=charges,
            k_vectors=k_vectors,
            cell=cell,
            alpha=alpha_arr,
            atom_start=atom_start,
            atom_end=atom_end,
            total_charges=total_charges,
            cos_k_dot_r=cos_k_dot_r,
            sin_k_dot_r=sin_k_dot_r,
            real_structure_factors=real_sf,
            imag_structure_factors=imag_sf,
            num_k=num_k,
            num_systems=num_systems,
            max_blocks_per_system=max_blocks_per_system,
            wp_dtype=dtype,
            device=device,
        )

        # Verify shapes
        assert total_charges.numpy().shape == (num_systems,)
        assert cos_k_dot_r.numpy().shape == (num_k, num_atoms)
        assert sin_k_dot_r.numpy().shape == (num_k, num_atoms)
        assert real_sf.numpy().shape == (num_systems, num_k)
        assert imag_sf.numpy().shape == (num_systems, num_k)

        # Verify no NaN values
        assert not np.isnan(cos_k_dot_r.numpy()).any()
        assert not np.isnan(sin_k_dot_r.numpy()).any()
        assert not np.isnan(real_sf.numpy()).any()
        assert not np.isnan(imag_sf.numpy()).any()


# ==============================================================================
# Test Class: Batched Reciprocal Space Energy Computation
# ==============================================================================


class TestWpBatchEwaldReciprocalSpaceComputeEnergy:
    """Tests for wp_batch_ewald_reciprocal_space_compute_energy."""

    def test_batch_reciprocal_energy_shape(self, device, batch_kvector_system):
        """Test batched reciprocal energy has correct shape."""
        sys = batch_kvector_system
        dtype = wp.float64
        vec_dtype = wp.vec3d
        mat_dtype = wp.mat33d

        positions = wp.from_numpy(sys["positions"], dtype=vec_dtype, device=device)
        charges = wp.from_numpy(sys["charges"], dtype=dtype, device=device)
        k_vectors = wp.from_numpy(sys["k_vectors"], dtype=vec_dtype, device=device)
        cell = wp.from_numpy(sys["cell"], dtype=mat_dtype, device=device)
        alpha_arr = wp.from_numpy(
            np.array([0.5, 0.5], dtype=np.float64), dtype=dtype, device=device
        )
        atom_start = wp.from_numpy(sys["atom_start"], dtype=wp.int32, device=device)
        atom_end = wp.from_numpy(sys["atom_end"], dtype=wp.int32, device=device)
        batch_idx = wp.from_numpy(sys["batch_idx"], dtype=wp.int32, device=device)

        num_k = sys["num_k"]
        num_atoms = sys["num_atoms"]
        num_systems = sys["num_systems"]

        # First compute structure factors
        total_charges = wp.zeros(num_systems, dtype=wp.float64, device=device)
        cos_k_dot_r = wp.zeros((num_k, num_atoms), dtype=wp.float64, device=device)
        sin_k_dot_r = wp.zeros((num_k, num_atoms), dtype=wp.float64, device=device)
        real_sf = wp.zeros((num_systems, num_k), dtype=wp.float64, device=device)
        imag_sf = wp.zeros((num_systems, num_k), dtype=wp.float64, device=device)

        max_blocks_per_system = 1

        batch_ewald_reciprocal_space_fill_structure_factors(
            positions=positions,
            charges=charges,
            k_vectors=k_vectors,
            cell=cell,
            alpha=alpha_arr,
            atom_start=atom_start,
            atom_end=atom_end,
            total_charges=total_charges,
            cos_k_dot_r=cos_k_dot_r,
            sin_k_dot_r=sin_k_dot_r,
            real_structure_factors=real_sf,
            imag_structure_factors=imag_sf,
            num_k=num_k,
            num_systems=num_systems,
            max_blocks_per_system=max_blocks_per_system,
            wp_dtype=dtype,
            device=device,
        )

        # Then compute energy
        reciprocal_energies = wp.zeros(num_atoms, dtype=wp.float64, device=device)

        batch_ewald_reciprocal_space_compute_energy(
            charges=charges,
            batch_id=batch_idx,
            cos_k_dot_r=cos_k_dot_r,
            sin_k_dot_r=sin_k_dot_r,
            real_structure_factors=real_sf,
            imag_structure_factors=imag_sf,
            reciprocal_energies=reciprocal_energies,
            wp_dtype=dtype,
            device=device,
        )

        energies = reciprocal_energies.numpy()
        assert energies.shape == (num_atoms,)
        assert not np.isnan(energies).any()
        assert not np.isinf(energies).any()


# ==============================================================================
# Test Class: Batched Self-Energy Subtraction
# ==============================================================================


class TestWpBatchEwaldSubtractSelfEnergy:
    """Tests for wp_batch_ewald_subtract_self_energy."""

    def test_batch_self_energy_correction(self, device, batch_kvector_system):
        """Test batched self-energy correction is applied correctly."""
        sys = batch_kvector_system
        dtype = wp.float64
        num_atoms = sys["num_atoms"]
        num_systems = sys["num_systems"]

        charges = wp.from_numpy(sys["charges"], dtype=dtype, device=device)
        batch_idx = wp.from_numpy(sys["batch_idx"], dtype=wp.int32, device=device)
        alpha_arr = wp.from_numpy(
            np.array([0.5, 0.5], dtype=np.float64), dtype=dtype, device=device
        )

        # Neutral systems: total_charge/V = 0 for each
        total_charges = wp.zeros(num_systems, dtype=wp.float64, device=device)

        # Fake input energies
        energy_in = wp.from_numpy(
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
            dtype=wp.float64,
            device=device,
        )
        energy_out = wp.zeros(num_atoms, dtype=wp.float64, device=device)

        batch_ewald_subtract_self_energy(
            charges=charges,
            batch_idx=batch_idx,
            alpha=alpha_arr,
            total_charges=total_charges,
            energy_in=energy_in,
            energy_out=energy_out,
            wp_dtype=dtype,
            device=device,
        )

        result = energy_out.numpy()
        # Self-energy = alpha/sqrt(pi) * q^2 for each atom
        alpha = 0.5
        self_factor = alpha / math.sqrt(math.pi)
        charges_np = sys["charges"]
        energy_in_np = np.array([1.0, 2.0, 3.0, 4.0])

        for i in range(num_atoms):
            expected = energy_in_np[i] - self_factor * charges_np[i] ** 2
            assert result[i] == pytest.approx(expected, rel=1e-6)


# ==============================================================================
# Test Class: Batched Reciprocal Space Energy Forces
# ==============================================================================


class TestWpBatchEwaldReciprocalSpaceEnergyForces:
    """Tests for wp_batch_ewald_reciprocal_space_energy_forces."""

    def test_batch_reciprocal_forces_sum_to_zero(self, device, batch_kvector_system):
        """Test that batched reciprocal-space forces sum to zero per system."""
        sys = batch_kvector_system
        dtype = wp.float64
        vec_dtype = wp.vec3d
        mat_dtype = wp.mat33d

        positions = wp.from_numpy(sys["positions"], dtype=vec_dtype, device=device)
        charges = wp.from_numpy(sys["charges"], dtype=dtype, device=device)
        k_vectors = wp.from_numpy(sys["k_vectors"], dtype=vec_dtype, device=device)
        cell = wp.from_numpy(sys["cell"], dtype=mat_dtype, device=device)
        alpha_arr = wp.from_numpy(
            np.array([0.5, 0.5], dtype=np.float64), dtype=dtype, device=device
        )
        atom_start = wp.from_numpy(sys["atom_start"], dtype=wp.int32, device=device)
        atom_end = wp.from_numpy(sys["atom_end"], dtype=wp.int32, device=device)
        batch_idx = wp.from_numpy(sys["batch_idx"], dtype=wp.int32, device=device)

        num_k = sys["num_k"]
        num_atoms = sys["num_atoms"]
        num_systems = sys["num_systems"]

        # First compute structure factors
        total_charges = wp.zeros(num_systems, dtype=wp.float64, device=device)
        cos_k_dot_r = wp.zeros((num_k, num_atoms), dtype=wp.float64, device=device)
        sin_k_dot_r = wp.zeros((num_k, num_atoms), dtype=wp.float64, device=device)
        real_sf = wp.zeros((num_systems, num_k), dtype=wp.float64, device=device)
        imag_sf = wp.zeros((num_systems, num_k), dtype=wp.float64, device=device)

        max_blocks_per_system = 1

        batch_ewald_reciprocal_space_fill_structure_factors(
            positions=positions,
            charges=charges,
            k_vectors=k_vectors,
            cell=cell,
            alpha=alpha_arr,
            atom_start=atom_start,
            atom_end=atom_end,
            total_charges=total_charges,
            cos_k_dot_r=cos_k_dot_r,
            sin_k_dot_r=sin_k_dot_r,
            real_structure_factors=real_sf,
            imag_structure_factors=imag_sf,
            num_k=num_k,
            num_systems=num_systems,
            max_blocks_per_system=max_blocks_per_system,
            wp_dtype=dtype,
            device=device,
        )

        # Compute energy and forces
        reciprocal_energies = wp.zeros(num_atoms, dtype=wp.float64, device=device)
        forces = allocate_force_output(num_atoms, device)

        batch_ewald_reciprocal_space_energy_forces(
            charges=charges,
            batch_id=batch_idx,
            k_vectors=k_vectors,
            cos_k_dot_r=cos_k_dot_r,
            sin_k_dot_r=sin_k_dot_r,
            real_structure_factors=real_sf,
            imag_structure_factors=imag_sf,
            reciprocal_energies=reciprocal_energies,
            atomic_forces=forces,
            wp_dtype=dtype,
            device=device,
        )

        result_forces = forces.numpy()
        # Per-system momentum conservation
        sys0_total = result_forces[0] + result_forces[1]
        sys1_total = result_forces[2] + result_forces[3]

        np.testing.assert_allclose(sys0_total, [0.0, 0.0, 0.0], atol=1e-10)
        np.testing.assert_allclose(sys1_total, [0.0, 0.0, 0.0], atol=1e-10)


# ==============================================================================
# Test Class: Batched Reciprocal Space Energy Forces Charge Grad
# ==============================================================================


class TestWpBatchEwaldReciprocalSpaceEnergyForcesChargeGrad:
    """Tests for wp_batch_ewald_reciprocal_space_energy_forces_charge_grad."""

    def test_batch_reciprocal_charge_grad_shape(self, device, batch_kvector_system):
        """Test batched charge gradients have correct shape."""
        sys = batch_kvector_system
        dtype = wp.float64
        vec_dtype = wp.vec3d
        mat_dtype = wp.mat33d

        positions = wp.from_numpy(sys["positions"], dtype=vec_dtype, device=device)
        charges = wp.from_numpy(sys["charges"], dtype=dtype, device=device)
        k_vectors = wp.from_numpy(sys["k_vectors"], dtype=vec_dtype, device=device)
        cell = wp.from_numpy(sys["cell"], dtype=mat_dtype, device=device)
        alpha_arr = wp.from_numpy(
            np.array([0.5, 0.5], dtype=np.float64), dtype=dtype, device=device
        )
        atom_start = wp.from_numpy(sys["atom_start"], dtype=wp.int32, device=device)
        atom_end = wp.from_numpy(sys["atom_end"], dtype=wp.int32, device=device)
        batch_idx = wp.from_numpy(sys["batch_idx"], dtype=wp.int32, device=device)

        num_k = sys["num_k"]
        num_atoms = sys["num_atoms"]
        num_systems = sys["num_systems"]

        # First compute structure factors
        total_charges = wp.zeros(num_systems, dtype=wp.float64, device=device)
        cos_k_dot_r = wp.zeros((num_k, num_atoms), dtype=wp.float64, device=device)
        sin_k_dot_r = wp.zeros((num_k, num_atoms), dtype=wp.float64, device=device)
        real_sf = wp.zeros((num_systems, num_k), dtype=wp.float64, device=device)
        imag_sf = wp.zeros((num_systems, num_k), dtype=wp.float64, device=device)

        max_blocks_per_system = 1

        batch_ewald_reciprocal_space_fill_structure_factors(
            positions=positions,
            charges=charges,
            k_vectors=k_vectors,
            cell=cell,
            alpha=alpha_arr,
            atom_start=atom_start,
            atom_end=atom_end,
            total_charges=total_charges,
            cos_k_dot_r=cos_k_dot_r,
            sin_k_dot_r=sin_k_dot_r,
            real_structure_factors=real_sf,
            imag_structure_factors=imag_sf,
            num_k=num_k,
            num_systems=num_systems,
            max_blocks_per_system=max_blocks_per_system,
            wp_dtype=dtype,
            device=device,
        )

        # Compute energy, forces, and charge gradients
        reciprocal_energies = wp.zeros(num_atoms, dtype=wp.float64, device=device)
        forces = allocate_force_output(num_atoms, device)
        charge_grads = allocate_charge_grad_output(num_atoms, device)

        batch_ewald_reciprocal_space_energy_forces_charge_grad(
            charges=charges,
            batch_id=batch_idx,
            k_vectors=k_vectors,
            cos_k_dot_r=cos_k_dot_r,
            sin_k_dot_r=sin_k_dot_r,
            real_structure_factors=real_sf,
            imag_structure_factors=imag_sf,
            reciprocal_energies=reciprocal_energies,
            atomic_forces=forces,
            charge_gradients=charge_grads,
            wp_dtype=dtype,
            device=device,
        )

        result_grads = charge_grads.numpy()
        assert result_grads.shape == (num_atoms,)
        assert not np.isnan(result_grads).any()

        # Forces should still sum to zero per system
        result_forces = forces.numpy()
        sys0_total = result_forces[0] + result_forces[1]
        sys1_total = result_forces[2] + result_forces[3]
        np.testing.assert_allclose(sys0_total, [0.0, 0.0, 0.0], atol=1e-10)
        np.testing.assert_allclose(sys1_total, [0.0, 0.0, 0.0], atol=1e-10)
