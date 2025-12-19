"""
Unit tests for Coulomb electrostatic calculations.

This test suite validates the correctness of the Coulomb energy and force
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
"""

import pytest
import torch

from nvalchemiops.interactions.electrostatics.coulomb import (
    coulomb_energy,
    coulomb_energy_forces,
    coulomb_forces,
)
from nvalchemiops.neighborlist import neighbor_list as neighbor_list_fn


@pytest.fixture
def device():
    """Get available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def simple_pair_system(device):
    """Two-atom system for basic tests."""
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        dtype=torch.float64,
        device=device,
    )
    charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
    cell = torch.tensor(
        [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
        dtype=torch.float64,
        device=device,
    )
    neighbor_list = torch.tensor([[0], [1]], dtype=torch.int32, device=device)
    neighbor_ptr = torch.tensor([0, 1], dtype=torch.int32, device=device)
    neighbor_shifts = torch.zeros((1, 3), dtype=torch.int32, device=device)
    return positions, charges, cell, neighbor_list, neighbor_ptr, neighbor_shifts


class TestUndampedCoulombEnergy:
    """Test undamped (direct) Coulomb energy calculations."""

    def test_two_charges_energy(self, device):
        """Test energy between opposite charges."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        cell = torch.tensor(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=torch.float64,
            device=device,
        )
        neighbor_list = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        neighbor_ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        neighbor_shifts = torch.zeros((2, 3), dtype=torch.int32, device=device)

        energies = coulomb_energy(
            positions=positions,
            charges=charges,
            cell=cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
        )

        # Expected: E = q1 * q2 / r = (1.0 * -1.0) / 3.0 = -1/3
        # Pair energy is split between both atoms
        expected_total = -1.0 / 3.0
        assert torch.allclose(
            energies.sum(),
            torch.tensor(expected_total, dtype=torch.float64, device=device),
            rtol=1e-6,
        )

    def test_energy_charge_scaling(self, device):
        """Test that energy scales as q1 * q2."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=torch.float64,
            device=device,
        )
        cell = torch.tensor(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=torch.float64,
            device=device,
        )
        neighbor_list = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        neighbor_ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        neighbor_shifts = torch.zeros((2, 3), dtype=torch.int32, device=device)

        # Energy with q1=1, q2=1
        charges1 = torch.tensor([1.0, 1.0], dtype=torch.float64, device=device)
        energy1 = coulomb_energy(
            positions,
            charges1,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
        ).sum()

        # Energy with q1=2, q2=2 (should be 4x)
        charges2 = torch.tensor([2.0, 2.0], dtype=torch.float64, device=device)
        energy2 = coulomb_energy(
            positions,
            charges2,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
        ).sum()

        assert torch.allclose(energy2, 4.0 * energy1, rtol=1e-10)

    def test_energy_inverse_law(self, device):
        """Test that energy follows 1/r law."""
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        cell = torch.tensor(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=torch.float64,
            device=device,
        )
        neighbor_list = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        neighbor_ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        neighbor_shifts = torch.zeros((2, 3), dtype=torch.int32, device=device)

        # Distance r = 2
        positions1 = torch.tensor(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=torch.float64,
            device=device,
        )
        energy1 = coulomb_energy(
            positions1,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
        ).sum()

        # Distance r = 4 (doubled) - energy should be halved
        positions2 = torch.tensor(
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            dtype=torch.float64,
            device=device,
        )
        energy2 = coulomb_energy(
            positions2,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
        ).sum()

        assert torch.allclose(energy2, energy1 / 2.0, rtol=1e-6)


class TestUndampedCoulombForces:
    """Test undamped (direct) Coulomb forces."""

    def test_two_charges_attractive(self, device, simple_pair_system):
        """Test attractive force between opposite charges."""
        positions, charges, cell, neighbor_list, neighbor_ptr, neighbor_shifts = (
            simple_pair_system
        )

        _, forces = coulomb_energy_forces(
            positions=positions,
            charges=charges,
            cell=cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
        )

        # Expected: F = 0.5*|q1 * q2| / r² = 1.0 / 9.0
        expected_magnitude = 1.0 / 18.0

        # Force on atom 0 should be in +x direction (toward atom 1)
        assert forces[0, 1].abs() < 1e-10
        assert forces[0, 2].abs() < 1e-10
        assert torch.allclose(
            forces[0, 0],
            torch.tensor(expected_magnitude, dtype=torch.float64, device=device),
            rtol=1e-6,
        )

        # Newton's 3rd law
        assert torch.allclose(forces[0], -forces[1], rtol=1e-10)

    def test_two_charges_repulsive(self, device):
        """Test repulsive force between like charges."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, 1.0], dtype=torch.float64, device=device)
        cell = torch.tensor(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=torch.float64,
            device=device,
        )
        neighbor_list = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        neighbor_ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        neighbor_shifts = torch.zeros((2, 3), dtype=torch.int32, device=device)

        forces = coulomb_forces(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
        )

        # Expected: F = 1.0 / 4.0 = 0.25
        # Force on atom 0 should be in -y direction (repulsive)
        expected_magnitude = 0.25

        assert forces[0, 0].abs() < 1e-10
        assert forces[0, 2].abs() < 1e-10
        assert torch.allclose(
            forces[0, 1],
            torch.tensor(-expected_magnitude, dtype=torch.float64, device=device),
            rtol=1e-6,
        )

    def test_inverse_square_law(self, device):
        """Test that force follows 1/r² law."""
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        cell = torch.tensor(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=torch.float64,
            device=device,
        )
        neighbor_list = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        neighbor_ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        neighbor_shifts = torch.zeros((2, 3), dtype=torch.int32, device=device)

        # Distance r = 2
        positions1 = torch.tensor(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=torch.float64,
            device=device,
        )
        forces1 = coulomb_forces(
            positions1,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
        )

        # Distance r = 4 (doubled) - force should be 1/4
        positions2 = torch.tensor(
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            dtype=torch.float64,
            device=device,
        )
        forces2 = coulomb_forces(
            positions2,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
        )

        assert torch.allclose(forces2, forces1 / 4.0, rtol=1e-6)

    def test_newton_third_law_multiple_pairs(self, device):
        """Test momentum conservation for multiple particles."""
        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 0.866, 0.0],
            ],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, 1.0, -2.0], dtype=torch.float64, device=device)
        cell = torch.tensor(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=torch.float64,
            device=device,
        )
        neighbor_list = torch.tensor(
            [[0, 0, 1, 1, 2, 2], [1, 2, 1, 2, 0, 1]], dtype=torch.int32, device=device
        )
        neighbor_ptr = torch.tensor([0, 2, 4, 6], dtype=torch.int32, device=device)
        neighbor_shifts = torch.zeros((3, 3), dtype=torch.int32, device=device)

        forces = coulomb_forces(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
        )

        # Total force should be zero
        total_force = forces.sum(dim=0)
        assert torch.allclose(
            total_force,
            torch.zeros(3, dtype=torch.float64, device=device),
            atol=1e-10,
        )

    def test_cutoff_enforcement(self, device):
        """Test that pairs beyond cutoff have zero interaction."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [15.0, 0.0, 0.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        cell = torch.tensor(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=torch.float64,
            device=device,
        )
        neighbor_list = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        neighbor_ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        neighbor_shifts = torch.zeros((2, 3), dtype=torch.int32, device=device)

        energies, forces = coulomb_energy_forces(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
        )

        assert torch.allclose(energies, torch.zeros_like(energies), atol=1e-15)
        assert torch.allclose(forces, torch.zeros_like(forces), atol=1e-15)


class TestDampedCoulomb:
    """Test damped (Ewald/PME real-space) Coulomb calculations."""

    def test_damping_reduces_energy(self, device):
        """Test that erfc damping reduces energy magnitude."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        cell = torch.tensor(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=torch.float64,
            device=device,
        )
        neighbor_list = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        neighbor_ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        neighbor_shifts = torch.zeros((2, 3), dtype=torch.int32, device=device)

        energy_undamped = coulomb_energy(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
        ).sum()

        energy_damped = coulomb_energy(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.3,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
        ).sum()

        # Damped energy should have smaller magnitude
        assert energy_damped.abs() < energy_undamped.abs()

    def test_damping_reduces_force(self, device):
        """Test that erfc damping reduces force magnitude."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        cell = torch.tensor(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=torch.float64,
            device=device,
        )
        neighbor_list = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        neighbor_ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        neighbor_shifts = torch.zeros((2, 3), dtype=torch.int32, device=device)

        forces_undamped = coulomb_forces(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
        )

        forces_damped = coulomb_forces(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.3,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
        )

        mag_undamped = torch.norm(forces_undamped[0], dim=0)
        mag_damped = torch.norm(forces_damped[0], dim=0)

        assert mag_damped < mag_undamped

    def test_short_range_behavior(self, device):
        """Test that damping has minimal effect at short range."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        cell = torch.tensor(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=torch.float64,
            device=device,
        )
        neighbor_list = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        neighbor_ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        neighbor_shifts = torch.zeros((2, 3), dtype=torch.int32, device=device)

        forces_undamped = coulomb_forces(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
        )

        forces_damped = coulomb_forces(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.3,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
        )

        # At short distance, damped ≈ undamped
        assert torch.allclose(forces_damped, forces_undamped, rtol=0.05)

    def test_alpha_scaling(self, device):
        """Test that larger alpha produces stronger damping."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        cell = torch.tensor(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=torch.float64,
            device=device,
        )
        neighbor_list = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        neighbor_ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        neighbor_shifts = torch.zeros((2, 3), dtype=torch.int32, device=device)

        alphas = [0.1, 0.2, 0.3, 0.4]
        force_magnitudes = []

        for alpha in alphas:
            forces = coulomb_forces(
                positions,
                charges,
                cell,
                cutoff=10.0,
                alpha=alpha,
                neighbor_list=neighbor_list,
                neighbor_ptr=neighbor_ptr,
                neighbor_shifts=neighbor_shifts,
            )
            force_magnitudes.append(torch.norm(forces[0], dim=0).item())

        # Force should decrease with increasing alpha
        for i in range(len(force_magnitudes) - 1):
            assert force_magnitudes[i] > force_magnitudes[i + 1]


class TestNeighborMatrixFormat:
    """Test calculations using neighbor matrix format."""

    def test_matrix_matches_list(self, device):
        """Test that neighbor matrix gives same results as neighbor list."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0, 0.5], dtype=torch.float64, device=device)
        cell = torch.tensor(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=torch.float64,
            device=device,
        )

        # Neighbor list format
        neighbor_list = torch.tensor(
            [[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]], dtype=torch.int32, device=device
        )
        neighbor_ptr = torch.tensor([0, 2, 4, 6], dtype=torch.int32, device=device)
        neighbor_shifts = torch.zeros((6, 3), dtype=torch.int32, device=device)

        # Neighbor matrix format
        # Atom 0: neighbors [1, 2]
        # Atom 1: neighbors [0, 2]
        # Atom 2: neighbors [0, 1]
        neighbor_matrix = torch.tensor(
            [[1, 2], [0, 2], [0, 1]],
            dtype=torch.int32,
            device=device,
        )
        neighbor_matrix_shifts = torch.zeros(
            (3, 2, 3), dtype=torch.int32, device=device
        )

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

        assert torch.allclose(energy_list.sum(), energy_matrix.sum(), rtol=1e-10)
        assert torch.allclose(forces_list, forces_matrix, rtol=1e-10)

    def test_matrix_damped(self, device):
        """Test damped calculation with neighbor matrix."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        cell = torch.tensor(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=torch.float64,
            device=device,
        )

        neighbor_matrix = torch.tensor([[1], [0]], dtype=torch.int32, device=device)
        neighbor_matrix_shifts = torch.zeros(
            (2, 1, 3), dtype=torch.int32, device=device
        )

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
        assert energies.sum().abs() > 1e-6
        assert torch.norm(forces[0], dim=0) > 1e-6


class TestPeriodicBoundaries:
    """Test calculations with periodic boundary conditions."""

    def test_minimum_image(self, device):
        """Test minimum image convention."""
        cell = torch.tensor(
            [[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]],
            dtype=torch.float64,
            device=device,
        )
        # Atoms at x=0.5 and x=9.5, distance through PBC = 1.0
        positions = torch.tensor(
            [[0.5, 5.0, 5.0], [9.5, 5.0, 5.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)

        neighbor_list = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        neighbor_ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        # Shift: atom 1 in -1 cell (wraps to distance 1.0)
        neighbor_shifts = torch.tensor(
            [[-1, 0, 0], [1, 0, 0]], dtype=torch.int32, device=device
        )

        forces = coulomb_forces(
            positions,
            charges,
            cell,
            cutoff=5.0,
            alpha=0.0,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
        )

        # Force should be in -x direction (toward wrapped image)
        expected_force = torch.tensor(
            [-1.0, 0.0, 0.0], dtype=torch.float64, device=device
        )
        assert torch.allclose(forces[0], expected_force, rtol=1e-6)


class TestBatchedCalculations:
    """Test batched Coulomb calculations."""

    def test_single_batch_matches_unbatched(self, device):
        """Test that single batch matches unbatched results."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0, 0.5], dtype=torch.float64, device=device)
        cell = torch.tensor(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=torch.float64,
            device=device,
        )
        neighbor_list = torch.tensor(
            [[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]], dtype=torch.int32, device=device
        )
        neighbor_ptr = torch.tensor([0, 2, 4, 6], dtype=torch.int32, device=device)
        neighbor_shifts = torch.zeros((6, 3), dtype=torch.int32, device=device)

        # Unbatched
        energy_unbatched, forces_unbatched = coulomb_energy_forces(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
        )

        # Single batch
        batch_idx = torch.zeros(3, dtype=torch.int32, device=device)
        energy_batched, forces_batched = coulomb_energy_forces(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            batch_idx=batch_idx,
        )

        assert torch.allclose(energy_batched, energy_unbatched, rtol=1e-10)
        assert torch.allclose(forces_batched, forces_unbatched, rtol=1e-10)

    def test_two_independent_batches(self, device):
        """Test that two batches don't interfere."""
        # Same configuration in both batches
        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # Batch 0
                [1.0, 0.0, 0.0],  # Batch 0
                [0.0, 0.0, 0.0],  # Batch 1
                [1.0, 0.0, 0.0],  # Batch 1
            ],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor(
            [1.0, -1.0, 1.0, -1.0], dtype=torch.float64, device=device
        )
        cell = torch.tensor(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=torch.float64,
            device=device,
        )

        neighbor_list = torch.tensor(
            [[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.int32, device=device
        )
        neighbor_ptr = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32, device=device)
        neighbor_shifts = torch.zeros((4, 3), dtype=torch.int32, device=device)
        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)

        _, forces = coulomb_energy_forces(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            batch_idx=batch_idx,
        )

        # Both batches should have identical forces
        assert torch.allclose(forces[0], forces[2], rtol=1e-10)
        assert torch.allclose(forces[1], forces[3], rtol=1e-10)

    def test_batch_momentum_conservation(self, device):
        """Test momentum conservation within each batch."""
        positions = torch.tensor(
            [
                # Batch 0: 2 atoms
                [0.0, 0.0, 0.0],
                [1.5, 0.0, 0.0],
                # Batch 1: 3 atoms
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 0.866, 0.0],
            ],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor(
            [1.0, -1.0, 1.0, 1.0, -2.0],
            dtype=torch.float64,
            device=device,
        )
        cell = torch.tensor(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=torch.float64,
            device=device,
        )
        pbc = torch.tensor([True, True, True], dtype=torch.bool, device=device)

        batch_idx = torch.tensor([0, 0, 1, 1, 1], dtype=torch.int32, device=device)
        neighbor_list, neighbor_ptr, neighbor_shifts = neighbor_list_fn(
            positions,
            5.0,
            cell=cell,
            pbc=pbc,
            batch_idx=batch_idx,
            return_neighbor_list=True,
        )

        _, forces = coulomb_energy_forces(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            batch_idx=batch_idx,
        )

        # Check momentum conservation for each batch
        batch_0_force = forces[0] + forces[1]
        batch_1_force = forces[2] + forces[3] + forces[4]

        assert torch.allclose(
            batch_0_force,
            torch.zeros(3, dtype=torch.float64, device=device),
            atol=1e-10,
        )
        assert torch.allclose(
            batch_1_force,
            torch.zeros(3, dtype=torch.float64, device=device),
            atol=1e-10,
        )

    def test_batched_with_damping(self, device):
        """Test batched calculation with damping."""
        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor(
            [1.0, -1.0, 1.0, -1.0], dtype=torch.float64, device=device
        )
        cell = torch.tensor(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=torch.float64,
            device=device,
        )

        neighbor_list = torch.tensor(
            [[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.int32, device=device
        )
        neighbor_ptr = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32, device=device)
        neighbor_shifts = torch.zeros((4, 3), dtype=torch.int32, device=device)
        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)

        _, forces = coulomb_energy_forces(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.3,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            batch_idx=batch_idx,
        )

        # Both batches should have identical results
        assert torch.allclose(forces[0], forces[2], rtol=1e-10)


class TestNumericalStability:
    """Test numerical stability and edge cases."""

    def test_very_small_distance(self, device):
        """Test that very small distances don't cause numerical issues."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1e-10, 0.0, 0.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        cell = torch.tensor(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=torch.float64,
            device=device,
        )
        neighbor_list = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        neighbor_ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        neighbor_shifts = torch.zeros((2, 3), dtype=torch.int32, device=device)

        energies, forces = coulomb_energy_forces(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
        )

        # Should be finite (zero due to cutoff protection)
        assert torch.all(torch.isfinite(energies))
        assert torch.all(torch.isfinite(forces))

    def test_zero_charge(self, device):
        """Test that zero charges produce zero interaction."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([0.0, 1.0], dtype=torch.float64, device=device)
        cell = torch.tensor(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=torch.float64,
            device=device,
        )
        neighbor_list = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        neighbor_ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        neighbor_shifts = torch.zeros((2, 3), dtype=torch.int32, device=device)

        energies, forces = coulomb_energy_forces(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
        )

        assert torch.allclose(energies, torch.zeros_like(energies), atol=1e-15)
        assert torch.allclose(forces, torch.zeros_like(forces), atol=1e-15)

    def test_empty_neighbor_list(self, device):
        """Test handling of empty neighbor list."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        cell = torch.tensor(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=torch.float64,
            device=device,
        )

        # Empty neighbor list
        neighbor_list = torch.zeros((2, 0), dtype=torch.int32, device=device)
        neighbor_ptr = torch.zeros(1, dtype=torch.int32, device=device)
        neighbor_shifts = torch.zeros((0, 3), dtype=torch.int32, device=device)

        energies, forces = coulomb_energy_forces(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
        )

        assert torch.allclose(energies, torch.zeros_like(energies))
        assert torch.allclose(forces, torch.zeros_like(forces))


class TestInputValidation:
    """Test input validation."""

    def test_missing_neighbor_data(self, device):
        """Test that missing neighbor data raises error."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        cell = torch.tensor(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=torch.float64,
            device=device,
        )

        with pytest.raises(ValueError, match="Must provide either"):
            coulomb_energy_forces(
                positions,
                charges,
                cell,
                cutoff=10.0,
                alpha=0.0,
            )

    def test_conflicting_neighbor_formats(self, device):
        """Test that providing both formats raises error."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        cell = torch.tensor(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=torch.float64,
            device=device,
        )

        neighbor_list = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        neighbor_shifts = torch.zeros((2, 3), dtype=torch.int32, device=device)
        neighbor_matrix = torch.tensor([[1], [0]], dtype=torch.int32, device=device)
        neighbor_matrix_shifts = torch.zeros(
            (2, 1, 3), dtype=torch.int32, device=device
        )

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


class TestAutograd:
    """Test automatic differentiation support."""

    def test_energy_gradient_vs_explicit_forces(self, device):
        """Test that autograd of energy matches explicit forces."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        cell = torch.tensor(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=torch.float64,
            device=device,
        )
        neighbor_list = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        neighbor_ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        neighbor_shifts = torch.zeros((2, 3), dtype=torch.int32, device=device)

        # Get explicit forces
        _, explicit_forces = coulomb_energy_forces(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
        )

        # Get autograd forces
        energies = coulomb_energy(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
        )
        energies.sum().backward()
        autograd_forces = -positions.grad

        assert torch.allclose(
            autograd_forces, explicit_forces, rtol=1e-5, atol=1e-10
        ), (
            f"Autograd forces {autograd_forces} don't match explicit forces {explicit_forces}"
        )

    def test_damped_energy_gradient_vs_explicit_forces(self, device):
        """Test that autograd of damped energy matches explicit forces."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        cell = torch.tensor(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=torch.float64,
            device=device,
        )
        neighbor_list = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        neighbor_ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        neighbor_shifts = torch.zeros((2, 3), dtype=torch.int32, device=device)

        # Get explicit forces
        _, explicit_forces = coulomb_energy_forces(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.3,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
        )

        # Get autograd forces
        energies = coulomb_energy(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.3,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
        )
        energies.sum().backward()
        autograd_forces = -positions.grad

        assert torch.allclose(
            autograd_forces, explicit_forces, rtol=1e-5, atol=1e-10
        ), (
            f"Autograd forces {autograd_forces} don't match explicit forces {explicit_forces}"
        )

    def test_charge_gradient(self, device):
        """Test that gradients flow through charges."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor(
            [1.0, -1.0], dtype=torch.float64, device=device, requires_grad=True
        )
        cell = torch.tensor(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=torch.float64,
            device=device,
        )
        neighbor_list = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        neighbor_ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        neighbor_shifts = torch.zeros((2, 3), dtype=torch.int32, device=device)

        energies = coulomb_energy(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
        )
        energies.sum().backward()

        assert charges.grad is not None, "Gradients should flow through charges"
        assert torch.all(torch.isfinite(charges.grad)), (
            "Charge gradients should be finite"
        )

    def test_cell_gradient(self, device):
        """Test that gradients flow through cell."""
        positions = torch.tensor(
            [[0.5, 5.0, 5.0], [9.5, 5.0, 5.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        cell = torch.tensor(
            [[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )
        neighbor_list = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        neighbor_ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        # Periodic shift to make them close
        neighbor_shifts = torch.tensor(
            [[-1, 0, 0], [1, 0, 0]], dtype=torch.int32, device=device
        )

        energies = coulomb_energy(
            positions,
            charges,
            cell,
            cutoff=5.0,
            alpha=0.0,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
        )
        energies.sum().backward()

        assert cell.grad is not None, "Gradients should flow through cell"
        assert torch.all(torch.isfinite(cell.grad)), "Cell gradients should be finite"

    def test_matrix_format_autograd(self, device):
        """Test autograd with neighbor matrix format."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        cell = torch.tensor(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=torch.float64,
            device=device,
        )
        neighbor_matrix = torch.tensor([[1], [0]], dtype=torch.int32, device=device)
        neighbor_matrix_shifts = torch.zeros(
            (2, 1, 3), dtype=torch.int32, device=device
        )

        energies = coulomb_energy(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            fill_value=2,
        )
        energies.sum().backward()

        assert positions.grad is not None, "Gradients should flow through positions"
        assert torch.all(torch.isfinite(positions.grad)), (
            "Position gradients should be finite"
        )

    def test_batched_autograd(self, device):
        """Test autograd with batched calculations."""
        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )
        charges = torch.tensor(
            [1.0, -1.0, 1.0, -1.0], dtype=torch.float64, device=device
        )
        cell = torch.tensor(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=torch.float64,
            device=device,
        )
        neighbor_list = torch.tensor(
            [[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.int32, device=device
        )
        neighbor_ptr = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32, device=device)
        neighbor_shifts = torch.zeros((4, 3), dtype=torch.int32, device=device)
        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)

        energies = coulomb_energy(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            batch_idx=batch_idx,
        )
        energies.sum().backward()

        assert positions.grad is not None, (
            "Gradients should flow through positions in batched mode"
        )
        assert torch.all(torch.isfinite(positions.grad)), (
            "Position gradients should be finite"
        )

    def test_finite_difference_check(self, device):
        """Test gradients against finite differences."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0, 0.5], dtype=torch.float64, device=device)
        cell = torch.tensor(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=torch.float64,
            device=device,
        )
        neighbor_list = torch.tensor(
            [[0, 0, 1, 1, 2, 2], [1, 2, 1, 2, 0, 1]], dtype=torch.int32, device=device
        )
        neighbor_ptr = torch.tensor([0, 2, 4, 6], dtype=torch.int32, device=device)
        neighbor_shifts = torch.zeros((6, 3), dtype=torch.int32, device=device)

        eps = 1e-6
        num_forces = torch.zeros_like(positions)

        for i in range(positions.shape[0]):
            for j in range(3):
                # Positive perturbation
                pos_plus = positions.clone()
                pos_plus[i, j] += eps
                e_plus = coulomb_energy(
                    pos_plus,
                    charges,
                    cell,
                    cutoff=10.0,
                    alpha=0.0,
                    neighbor_list=neighbor_list,
                    neighbor_ptr=neighbor_ptr,
                    neighbor_shifts=neighbor_shifts,
                ).sum()

                # Negative perturbation
                pos_minus = positions.clone()
                pos_minus[i, j] -= eps
                e_minus = coulomb_energy(
                    pos_minus,
                    charges,
                    cell,
                    cutoff=10.0,
                    alpha=0.0,
                    neighbor_list=neighbor_list,
                    neighbor_ptr=neighbor_ptr,
                    neighbor_shifts=neighbor_shifts,
                ).sum()

                # Numerical gradient
                num_forces[i, j] = -(e_plus - e_minus) / (2 * eps)

        # Analytical forces
        _, analytical_forces = coulomb_energy_forces(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
        )

        assert torch.allclose(num_forces, analytical_forces, rtol=1e-4, atol=1e-8), (
            f"Numerical forces {num_forces} don't match analytical forces {analytical_forces}"
        )


class TestFloat32Support:
    """Test float32 dtype support for Coulomb calculations."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_float32_energy_calculation(self, device):
        """Test energy calculation with float32 dtype."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            dtype=torch.float32,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float32, device=device)
        cell = torch.tensor(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=torch.float32,
            device=device,
        )
        neighbor_list = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        neighbor_ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        neighbor_shifts = torch.zeros((2, 3), dtype=torch.int32, device=device)

        energies = coulomb_energy(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
        )

        assert energies.dtype == torch.float32
        assert torch.isfinite(energies).all()
        # Expected: E = q1 * q2 / r = -1/3
        expected_total = -1.0 / 3.0
        assert torch.allclose(
            energies.sum(),
            torch.tensor(expected_total, dtype=torch.float32, device=device),
            rtol=1e-4,
        )

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_float32_forces_calculation(self, device):
        """Test forces calculation with float32 dtype."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            dtype=torch.float32,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float32, device=device)
        cell = torch.tensor(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=torch.float32,
            device=device,
        )
        neighbor_list = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        neighbor_ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        neighbor_shifts = torch.zeros((2, 3), dtype=torch.int32, device=device)

        energies, forces = coulomb_energy_forces(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
        )

        assert energies.dtype == torch.float32
        assert forces.dtype == torch.float32
        assert torch.isfinite(forces).all()
        # Check Newton's 3rd law
        assert torch.allclose(forces[0], -forces[1], rtol=1e-4)

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_float32_vs_float64_consistency(self, device):
        """Test that float32 and float64 produce consistent results."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions_f32 = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            dtype=torch.float32,
            device=device,
        )
        charges_f32 = torch.tensor([1.0, -1.0], dtype=torch.float32, device=device)
        cell_f32 = torch.tensor(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=torch.float32,
            device=device,
        )
        neighbor_list = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        neighbor_ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        neighbor_shifts = torch.zeros((2, 3), dtype=torch.int32, device=device)

        e_f32, f_f32 = coulomb_energy_forces(
            positions_f32,
            charges_f32,
            cell_f32,
            cutoff=10.0,
            alpha=0.0,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
        )

        e_f64, f_f64 = coulomb_energy_forces(
            positions_f32.double(),
            charges_f32.double(),
            cell_f32.double(),
            cutoff=10.0,
            alpha=0.0,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
        )

        # Results should be close (within float32 precision)
        assert torch.allclose(e_f32.double(), e_f64, rtol=1e-4, atol=1e-5)
        assert torch.allclose(f_f32.double(), f_f64, rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_float32_damped_calculation(self, device):
        """Test damped Coulomb with float32 dtype."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
            dtype=torch.float32,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float32, device=device)
        cell = torch.tensor(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=torch.float32,
            device=device,
        )
        neighbor_list = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        neighbor_ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        neighbor_shifts = torch.zeros((2, 3), dtype=torch.int32, device=device)

        energies, forces = coulomb_energy_forces(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.3,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
        )

        assert energies.dtype == torch.float32
        assert forces.dtype == torch.float32
        assert torch.isfinite(energies).all()
        assert torch.isfinite(forces).all()


class TestBatchedNeighborMatrix:
    """Test batched calculations with neighbor matrix format."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_batch_matrix_energy(self, device):
        """Test batched energy calculation with neighbor matrix format."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        # Two systems with 2 atoms each
        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor(
            [1.0, -1.0, 1.0, -1.0], dtype=torch.float64, device=device
        )
        cell = torch.tensor(
            [
                [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]],
                [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]],
            ],
            dtype=torch.float64,
            device=device,
        )
        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)

        # Neighbor matrix: each atom sees its pair
        neighbor_matrix = torch.tensor(
            [[1], [0], [3], [2]], dtype=torch.int32, device=device
        )
        neighbor_matrix_shifts = torch.zeros(
            (4, 1, 3), dtype=torch.int32, device=device
        )

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
        assert torch.isfinite(energies).all()

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_batch_matrix_energy_forces(self, device):
        """Test batched energy and forces with neighbor matrix format."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        # Two systems with 2 atoms each
        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor(
            [1.0, -1.0, 1.0, -1.0], dtype=torch.float64, device=device
        )
        cell = torch.tensor(
            [
                [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]],
                [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]],
            ],
            dtype=torch.float64,
            device=device,
        )
        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)

        neighbor_matrix = torch.tensor(
            [[1], [0], [3], [2]], dtype=torch.int32, device=device
        )
        neighbor_matrix_shifts = torch.zeros(
            (4, 1, 3), dtype=torch.int32, device=device
        )

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
        assert torch.isfinite(energies).all()
        assert torch.isfinite(forces).all()

        # Both batches should have identical forces
        assert torch.allclose(forces[0], forces[2], rtol=1e-10)
        assert torch.allclose(forces[1], forces[3], rtol=1e-10)

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_batch_matrix_damped(self, device):
        """Test batched damped calculation with neighbor matrix format."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor(
            [1.0, -1.0, 1.0, -1.0], dtype=torch.float64, device=device
        )
        cell = torch.tensor(
            [
                [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]],
                [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]],
            ],
            dtype=torch.float64,
            device=device,
        )
        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)

        neighbor_matrix = torch.tensor(
            [[1], [0], [3], [2]], dtype=torch.int32, device=device
        )
        neighbor_matrix_shifts = torch.zeros(
            (4, 1, 3), dtype=torch.int32, device=device
        )

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

        assert torch.isfinite(energies).all()
        assert torch.isfinite(forces).all()


class TestForcesOnlyAPI:
    """Test the forces-only API coulomb_forces."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_forces_only_matches_energy_forces(self, device):
        """Test that coulomb_forces matches coulomb_energy_forces."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        cell = torch.tensor(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=torch.float64,
            device=device,
        )
        neighbor_list = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        neighbor_ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        neighbor_shifts = torch.zeros((2, 3), dtype=torch.int32, device=device)

        forces_only = coulomb_forces(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
        )

        _, forces_combined = coulomb_energy_forces(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
        )

        assert torch.allclose(forces_only, forces_combined, rtol=1e-10)

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_forces_only_damped(self, device):
        """Test forces-only API with damping."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        cell = torch.tensor(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=torch.float64,
            device=device,
        )
        neighbor_list = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        neighbor_ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        neighbor_shifts = torch.zeros((2, 3), dtype=torch.int32, device=device)

        forces_only = coulomb_forces(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.3,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
        )

        _, forces_combined = coulomb_energy_forces(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.3,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
        )

        assert torch.allclose(forces_only, forces_combined, rtol=1e-10)

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_forces_only_matrix_format(self, device):
        """Test forces-only API with neighbor matrix format."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        cell = torch.tensor(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=torch.float64,
            device=device,
        )
        neighbor_matrix = torch.tensor([[1], [0]], dtype=torch.int32, device=device)
        neighbor_matrix_shifts = torch.zeros(
            (2, 1, 3), dtype=torch.int32, device=device
        )

        forces_only = coulomb_forces(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            fill_value=2,
        )

        _, forces_combined = coulomb_energy_forces(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            fill_value=2,
        )

        assert torch.allclose(forces_only, forces_combined, rtol=1e-10)


class TestDefaultFillValue:
    """Test that default fill_value is handled correctly."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_default_fill_value_energy(self, device):
        """Test energy calculation without explicit fill_value."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        cell = torch.tensor(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=torch.float64,
            device=device,
        )
        neighbor_matrix = torch.tensor([[1], [0]], dtype=torch.int32, device=device)
        neighbor_matrix_shifts = torch.zeros(
            (2, 1, 3), dtype=torch.int32, device=device
        )

        # Without explicit fill_value
        energies = coulomb_energy(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
        )

        assert torch.isfinite(energies).all()

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_default_fill_value_energy_forces(self, device):
        """Test energy and forces calculation without explicit fill_value."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        cell = torch.tensor(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=torch.float64,
            device=device,
        )
        neighbor_matrix = torch.tensor([[1], [0]], dtype=torch.int32, device=device)
        neighbor_matrix_shifts = torch.zeros(
            (2, 1, 3), dtype=torch.int32, device=device
        )

        # Without explicit fill_value
        energies, forces = coulomb_energy_forces(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
        )

        assert torch.isfinite(energies).all()
        assert torch.isfinite(forces).all()


class TestEmptyInputs:
    """Test edge cases with empty/zero inputs."""

    def test_empty_neighbor_matrix_energy(self, device):
        """Test energy with empty neighbor matrix (max_neighbors=0)."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        cell = torch.tensor(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=torch.float64,
            device=device,
        )
        # Empty neighbor matrix (0 neighbors)
        neighbor_matrix = torch.zeros((2, 0), dtype=torch.int32, device=device)
        neighbor_matrix_shifts = torch.zeros(
            (2, 0, 3), dtype=torch.int32, device=device
        )

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

        # With no neighbors, energy should be zero
        assert energies.shape == (2,)
        assert torch.allclose(
            energies, torch.zeros(2, device=device, dtype=torch.float64)
        )

    def test_empty_neighbor_matrix_energy_forces(self, device):
        """Test energy and forces with empty neighbor matrix."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        cell = torch.tensor(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=torch.float64,
            device=device,
        )
        neighbor_matrix = torch.zeros((2, 0), dtype=torch.int32, device=device)
        neighbor_matrix_shifts = torch.zeros(
            (2, 0, 3), dtype=torch.int32, device=device
        )

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

        assert energies.shape == (2,)
        assert forces.shape == (2, 3)
        assert torch.allclose(
            energies, torch.zeros(2, device=device, dtype=torch.float64)
        )
        assert torch.allclose(
            forces, torch.zeros((2, 3), device=device, dtype=torch.float64)
        )

    def test_batch_empty_neighbor_list_energy(self, device):
        """Test batch energy with empty neighbor list."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        cell = torch.tensor(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=torch.float64,
            device=device,
        )
        batch_idx = torch.tensor([0, 0], dtype=torch.int32, device=device)
        neighbor_list = torch.zeros((2, 0), dtype=torch.int32, device=device)
        neighbor_ptr = torch.zeros(1, dtype=torch.int32, device=device)
        neighbor_shifts = torch.zeros((0, 3), dtype=torch.int32, device=device)

        energies = coulomb_energy(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            batch_idx=batch_idx,
        )

        assert energies.shape == (2,)
        assert torch.allclose(
            energies, torch.zeros(2, device=device, dtype=torch.float64)
        )

    def test_batch_empty_neighbor_list_energy_forces(self, device):
        """Test batch energy and forces with empty neighbor list."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        cell = torch.tensor(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=torch.float64,
            device=device,
        )
        batch_idx = torch.tensor([0, 0], dtype=torch.int32, device=device)
        neighbor_list = torch.zeros((2, 0), dtype=torch.int32, device=device)
        neighbor_ptr = torch.zeros(1, dtype=torch.int32, device=device)
        neighbor_shifts = torch.zeros((0, 3), dtype=torch.int32, device=device)

        energies, forces = coulomb_energy_forces(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            batch_idx=batch_idx,
        )

        assert energies.shape == (2,)
        assert forces.shape == (2, 3)
        assert torch.allclose(
            energies, torch.zeros(2, device=device, dtype=torch.float64)
        )
        assert torch.allclose(
            forces, torch.zeros((2, 3), device=device, dtype=torch.float64)
        )

    def test_batch_empty_neighbor_matrix_energy(self, device):
        """Test batch energy with empty neighbor matrix."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        cell = torch.tensor(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=torch.float64,
            device=device,
        )
        batch_idx = torch.tensor([0, 0], dtype=torch.int32, device=device)
        neighbor_matrix = torch.zeros((2, 0), dtype=torch.int32, device=device)
        neighbor_matrix_shifts = torch.zeros(
            (2, 0, 3), dtype=torch.int32, device=device
        )

        energies = coulomb_energy(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            fill_value=-1,
            batch_idx=batch_idx,
        )

        assert energies.shape == (2,)
        assert torch.allclose(
            energies, torch.zeros(2, device=device, dtype=torch.float64)
        )

    def test_batch_empty_neighbor_matrix_energy_forces(self, device):
        """Test batch energy and forces with empty neighbor matrix."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        cell = torch.tensor(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=torch.float64,
            device=device,
        )
        batch_idx = torch.tensor([0, 0], dtype=torch.int32, device=device)
        neighbor_matrix = torch.zeros((2, 0), dtype=torch.int32, device=device)
        neighbor_matrix_shifts = torch.zeros(
            (2, 0, 3), dtype=torch.int32, device=device
        )

        energies, forces = coulomb_energy_forces(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            fill_value=-1,
            batch_idx=batch_idx,
        )

        assert energies.shape == (2,)
        assert forces.shape == (2, 3)
        assert torch.allclose(
            energies, torch.zeros(2, device=device, dtype=torch.float64)
        )
        assert torch.allclose(
            forces, torch.zeros((2, 3), device=device, dtype=torch.float64)
        )


class TestMatrixFormatAutograd:
    """Test autograd with neighbor matrix format."""

    def test_matrix_energy_autograd(self, device):
        """Test energy autograd with matrix format."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )
        charges = torch.tensor(
            [1.0, -1.0], dtype=torch.float64, device=device, requires_grad=True
        )
        cell = torch.tensor(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )
        neighbor_matrix = torch.tensor([[1], [0]], dtype=torch.int32, device=device)
        neighbor_matrix_shifts = torch.zeros(
            (2, 1, 3), dtype=torch.int32, device=device
        )

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

        total_energy = energies.sum()
        total_energy.backward()

        assert positions.grad is not None
        assert charges.grad is not None
        assert cell.grad is not None
        assert positions.grad.shape == positions.shape
        assert charges.grad.shape == charges.shape
        assert cell.grad.shape == cell.shape

    def test_matrix_energy_forces_autograd(self, device):
        """Test energy_forces autograd with matrix format."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )
        charges = torch.tensor(
            [1.0, -1.0], dtype=torch.float64, device=device, requires_grad=True
        )
        cell = torch.tensor(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )
        neighbor_matrix = torch.tensor([[1], [0]], dtype=torch.int32, device=device)
        neighbor_matrix_shifts = torch.zeros(
            (2, 1, 3), dtype=torch.int32, device=device
        )

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

        # Backward on energy
        total_energy = energies.sum()
        total_energy.backward()

        assert positions.grad is not None
        assert charges.grad is not None
        assert cell.grad is not None


class TestBatchAutograd:
    """Test autograd with batch functions."""

    def test_batch_list_energy_autograd(self, device):
        """Test batch energy autograd with neighbor list."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [10.0, 0.0, 0.0], [13.0, 0.0, 0.0]],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )
        charges = torch.tensor(
            [1.0, -1.0, 1.0, -1.0],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )
        cell = torch.tensor(
            [
                [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]],
                [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]],
            ],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )
        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)
        neighbor_list = torch.tensor(
            [[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.int32, device=device
        )
        neighbor_ptr = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32, device=device)
        neighbor_shifts = torch.zeros((4, 3), dtype=torch.int32, device=device)

        energies = coulomb_energy(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            batch_idx=batch_idx,
        )

        total_energy = energies.sum()
        total_energy.backward()

        assert positions.grad is not None
        assert charges.grad is not None
        assert cell.grad is not None

    def test_batch_list_energy_forces_autograd(self, device):
        """Test batch energy_forces autograd with neighbor list."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [10.0, 0.0, 0.0], [13.0, 0.0, 0.0]],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )
        charges = torch.tensor(
            [1.0, -1.0, 1.0, -1.0],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )
        cell = torch.tensor(
            [
                [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]],
                [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]],
            ],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )
        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)
        neighbor_list = torch.tensor(
            [[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.int32, device=device
        )
        neighbor_ptr = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32, device=device)
        neighbor_shifts = torch.zeros((4, 3), dtype=torch.int32, device=device)

        energies, forces = coulomb_energy_forces(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            batch_idx=batch_idx,
        )

        total_energy = energies.sum()
        total_energy.backward()

        assert positions.grad is not None
        assert charges.grad is not None
        assert cell.grad is not None

    def test_batch_matrix_energy_autograd(self, device):
        """Test batch energy autograd with neighbor matrix."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [10.0, 0.0, 0.0], [13.0, 0.0, 0.0]],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )
        charges = torch.tensor(
            [1.0, -1.0, 1.0, -1.0],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )
        cell = torch.tensor(
            [
                [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]],
                [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]],
            ],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )
        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)
        neighbor_matrix = torch.tensor(
            [[1, -1], [0, -1], [3, -1], [2, -1]], dtype=torch.int32, device=device
        )
        neighbor_matrix_shifts = torch.zeros(
            (4, 2, 3), dtype=torch.int32, device=device
        )

        energies = coulomb_energy(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            fill_value=-1,
            batch_idx=batch_idx,
        )

        total_energy = energies.sum()
        total_energy.backward()

        assert positions.grad is not None
        assert charges.grad is not None
        assert cell.grad is not None

    def test_batch_matrix_energy_forces_autograd(self, device):
        """Test batch energy_forces autograd with neighbor matrix."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [10.0, 0.0, 0.0], [13.0, 0.0, 0.0]],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )
        charges = torch.tensor(
            [1.0, -1.0, 1.0, -1.0],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )
        cell = torch.tensor(
            [
                [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]],
                [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]],
            ],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )
        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)
        neighbor_matrix = torch.tensor(
            [[1, -1], [0, -1], [3, -1], [2, -1]], dtype=torch.int32, device=device
        )
        neighbor_matrix_shifts = torch.zeros(
            (4, 2, 3), dtype=torch.int32, device=device
        )

        energies, forces = coulomb_energy_forces(
            positions,
            charges,
            cell,
            cutoff=10.0,
            alpha=0.0,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            fill_value=-1,
            batch_idx=batch_idx,
        )

        total_energy = energies.sum()
        total_energy.backward()

        assert positions.grad is not None
        assert charges.grad is not None
        assert cell.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
