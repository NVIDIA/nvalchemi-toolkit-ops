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
Unified Test Suite for Particle Mesh Ewald (PME) Implementation
================================================================

This test suite validates the correctness of the unified PME API:

1. Unit Tests - Basic API functionality and shapes
2. Correctness Tests - Validation against torchpme reference
3. Batch Tests - Batch vs single-system consistency
4. Autograd Tests - Gradient computation validation
5. Conservation Laws - Momentum and energy properties
"""

from importlib import import_module

import pytest
import torch

from nvalchemiops.torch.interactions.electrostatics import (
    particle_mesh_ewald,
    pme_reciprocal_space,
)
from nvalchemiops.torch.interactions.electrostatics.ewald import ewald_real_space
from nvalchemiops.torch.interactions.electrostatics.k_vectors import (
    generate_k_vectors_pme,
)
from nvalchemiops.torch.neighbors import batch_cell_list, cell_list

# Check for optional dependencies
try:
    _ = import_module("torchpme")
    HAS_TORCHPME = True
    from torchpme import PMECalculator
    from torchpme.potentials import CoulombPotential
except ModuleNotFoundError:
    HAS_TORCHPME = False
    PMECalculator = None
    CoulombPotential = None

# Import test utilities for crystal structure generation
from .test_utils import (
    VIRIAL_DTYPE,
    create_cscl_supercell,
    create_wurtzite_system,
    create_zincblende_system,
    fd_virial_full,
    get_virial_neighbor_data,
    make_non_neutral_system,
    make_virial_batch_cscl_system,
    make_virial_cscl_system,
)

###########################################################################################
########################### Helper Functions ##############################################
###########################################################################################


def create_simple_system(
    device: torch.device,
    dtype: torch.dtype = torch.float64,
    num_atoms: int = 4,
    cell_size: float = 10.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a simple test system with random positions and neutral charges."""
    positions = (
        torch.rand((num_atoms, 3), dtype=dtype, device=device) * cell_size * 0.8
        + cell_size * 0.1
    )
    charges = torch.randn(num_atoms, dtype=dtype, device=device)
    charges[-1] = -charges[:-1].sum()  # Make neutral
    cell = torch.eye(3, dtype=dtype, device=device) * cell_size
    return positions, charges, cell


def create_dipole_system(
    device: torch.device,
    dtype: torch.dtype = torch.float64,
    separation: float = 2.0,
    cell_size: float = 10.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a simple dipole system."""
    center = cell_size / 2
    positions = torch.tensor(
        [
            [center - separation / 2, center, center],
            [center + separation / 2, center, center],
        ],
        dtype=dtype,
        device=device,
    )
    charges = torch.tensor([1.0, -1.0], dtype=dtype, device=device)
    cell = torch.eye(3, dtype=dtype, device=device) * cell_size
    return positions, charges, cell


def calculate_pme_reciprocal_energy_torchpme(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    mesh_spacing: float,
    alpha: float,
    spline_order: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Calculate PME reciprocal-space energy using torchpme as reference."""
    import math

    if not HAS_TORCHPME:
        pytest.skip("torchpme not available")

    # torchpme uses smearing σ where Gaussian is exp(-r²/(2σ²))
    # Standard Ewald uses exp(-α²r²), so σ = 1/(√2·α)
    smearing = 1.0 / (math.sqrt(2.0) * alpha)
    potential = CoulombPotential(smearing=smearing).to(device=device, dtype=dtype)
    charges_pme = charges.unsqueeze(1)

    calculator = PMECalculator(
        potential=potential,
        mesh_spacing=mesh_spacing,
        interpolation_nodes=spline_order,
        full_neighbor_list=True,
        prefactor=1.0,
    ).to(device=device, dtype=dtype)

    # Ensure cell is 2D for torchpme
    cell_2d = cell.squeeze(0) if cell.dim() == 3 else cell

    reciprocal_potential = calculator._compute_kspace(charges_pme, cell_2d, positions)

    return (reciprocal_potential * charges_pme).flatten()


###########################################################################################
########################### Dtype Tests ####################################################
###########################################################################################


class TestDtypeSupport:
    """Test that PME functions support both float32 and float64 dtypes."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_pme_reciprocal_dtype_returns_correct_type(self, device, dtype):
        """Test that pme_reciprocal_space returns tensors in input dtype."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions, charges, cell = create_dipole_system(device, dtype=dtype)
        alpha = 0.3

        # Test energy-only
        energies = pme_reciprocal_space(
            positions,
            charges,
            cell.unsqueeze(0),
            alpha=alpha,
            mesh_dimensions=(16, 16, 16),
            spline_order=4,
            compute_forces=False,
        )
        assert energies.dtype == dtype, f"Expected {dtype}, got {energies.dtype}"

        # Test with forces
        energies, forces = pme_reciprocal_space(
            positions,
            charges,
            cell.unsqueeze(0),
            alpha=alpha,
            mesh_dimensions=(16, 16, 16),
            spline_order=4,
            compute_forces=True,
        )
        assert energies.dtype == dtype, f"Expected {dtype}, got {energies.dtype}"
        assert forces.dtype == dtype, f"Expected {dtype}, got {forces.dtype}"

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_pme_batch_dtype_returns_correct_type(self, device, dtype):
        """Test that batch PME returns tensors in input dtype."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        # Create two systems
        pos1, chg1, cell1 = create_dipole_system(device, dtype=dtype)
        pos2, chg2, cell2 = create_dipole_system(device, dtype=dtype, separation=3.0)

        positions = torch.cat([pos1, pos2], dim=0)
        charges = torch.cat([chg1, chg2], dim=0)
        cells = torch.stack([cell1, cell2], dim=0)
        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)

        # Test energy-only
        energies = pme_reciprocal_space(
            positions,
            charges,
            cells,
            alpha=0.3,
            mesh_dimensions=(16, 16, 16),
            spline_order=4,
            batch_idx=batch_idx,
            compute_forces=False,
        )
        assert energies.dtype == dtype, f"Expected {dtype}, got {energies.dtype}"

        # Test with forces
        energies, forces = pme_reciprocal_space(
            positions,
            charges,
            cells,
            alpha=0.3,
            mesh_dimensions=(16, 16, 16),
            spline_order=4,
            batch_idx=batch_idx,
            compute_forces=True,
        )
        assert energies.dtype == dtype, f"Expected {dtype}, got {energies.dtype}"
        assert forces.dtype == dtype, f"Expected {dtype}, got {forces.dtype}"

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_float32_vs_float64_consistency(self, device):
        """Test that float32 and float64 produce consistent results."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        # Create systems in both dtypes
        pos_f32, chg_f32, cell_f32 = create_dipole_system(device, dtype=torch.float32)
        pos_f64, chg_f64, cell_f64 = create_dipole_system(device, dtype=torch.float64)

        # Use same values
        pos_f64 = pos_f32.double()
        chg_f64 = chg_f32.double()
        cell_f64 = cell_f32.double()

        e_f32, f_f32 = pme_reciprocal_space(
            pos_f32,
            chg_f32,
            cell_f32.unsqueeze(0),
            alpha=0.3,
            mesh_dimensions=(16, 16, 16),
            spline_order=4,
            compute_forces=True,
        )
        e_f64, f_f64 = pme_reciprocal_space(
            pos_f64,
            chg_f64,
            cell_f64.unsqueeze(0),
            alpha=0.3,
            mesh_dimensions=(16, 16, 16),
            spline_order=4,
            compute_forces=True,
        )

        # Results should be close (within float32 precision)
        assert torch.allclose(e_f32.double(), e_f64, rtol=1e-4, atol=1e-5), (
            f"Energy mismatch: f32={e_f32.sum()}, f64={e_f64.sum()}"
        )
        assert torch.allclose(f_f32.double(), f_f64, rtol=1e-4, atol=1e-5), (
            f"Forces mismatch: f32={f_f32}, f64={f_f64}"
        )


###########################################################################################
########################### Unit Tests: API Shapes and Basic Behavior #####################
###########################################################################################


class TestPMEReciprocalSpaceAPI:
    """Test basic API functionality for pme_reciprocal_space."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_output_shape_energy_only(self, device):
        """Test output shape when compute_forces=False."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions, charges, cell = create_simple_system(device, num_atoms=5)

        result = pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=0.3,
            mesh_dimensions=(16, 16, 16),
            compute_forces=False,
        )

        assert result.shape == (5,), f"Energy shape mismatch: {result.shape}"
        assert result.dtype == positions.dtype

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_output_shape_energy_forces(self, device):
        """Test output shape when compute_forces=True."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions, charges, cell = create_simple_system(device, num_atoms=5)

        energies, forces = pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=0.3,
            mesh_dimensions=(16, 16, 16),
            compute_forces=True,
        )

        assert energies.shape == (5,), f"Energy shape mismatch: {energies.shape}"
        assert forces.shape == (5, 3), f"Force shape mismatch: {forces.shape}"

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_batch_output_shape(self, device):
        """Test output shape for batched calculation."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        # Two systems with 3 and 4 atoms
        positions = torch.rand((7, 3), dtype=torch.float64, device=device) * 8.0
        charges = torch.randn(7, dtype=torch.float64, device=device)
        batch_idx = torch.tensor(
            [0, 0, 0, 1, 1, 1, 1], dtype=torch.int32, device=device
        )
        cells = (
            torch.eye(3, dtype=torch.float64, device=device)
            .unsqueeze(0)
            .expand(2, -1, -1)
            * 10.0
        )

        energies, forces = pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cells,
            alpha=0.3,
            mesh_dimensions=(16, 16, 16),
            batch_idx=batch_idx,
            compute_forces=True,
        )

        assert energies.shape == (7,), f"Batch energy shape mismatch: {energies.shape}"
        assert forces.shape == (7, 3), f"Batch force shape mismatch: {forces.shape}"

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_empty_system(self, device):
        """Test handling of empty system."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.zeros((0, 3), dtype=torch.float64, device=device)
        charges = torch.zeros(0, dtype=torch.float64, device=device)
        cell = torch.eye(3, dtype=torch.float64, device=device) * 10.0

        energies, forces = pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=0.3,
            mesh_dimensions=(16, 16, 16),
            compute_forces=True,
        )

        assert energies.shape == (0,)
        assert forces.shape == (0, 3)

    @pytest.mark.parametrize("spline_order", [2, 3, 4])
    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_different_spline_orders(self, spline_order, device):
        """Test that different spline orders produce valid results."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions, charges, cell = create_dipole_system(device)

        energies, forces = pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=0.3,
            mesh_dimensions=(16, 16, 16),
            spline_order=spline_order,
            compute_forces=True,
        )

        assert torch.all(torch.isfinite(energies)), (
            f"Non-finite energies for order {spline_order}"
        )
        assert torch.all(torch.isfinite(forces)), (
            f"Non-finite forces for order {spline_order}"
        )


###########################################################################################
########################### Conservation Law Tests ########################################
###########################################################################################


class TestPMEConservationLaws:
    """Test momentum conservation and symmetry properties."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_momentum_conservation(self, device):
        """Test that net force is zero for neutral system."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions, charges, cell = create_simple_system(device, num_atoms=6)

        _, forces = pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=0.3,
            mesh_dimensions=(20, 20, 20),
            compute_forces=True,
        )

        net_force = forces.sum(dim=0)
        assert torch.allclose(
            net_force, torch.zeros(3, dtype=torch.float64, device=device), atol=1e-4
        ), f"Momentum not conserved: net force = {net_force}"

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_translation_invariance(self, device):
        """Test that energy is invariant under translation."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions, charges, cell = create_dipole_system(device)

        energy1 = pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=0.3,
            mesh_dimensions=(16, 16, 16),
            compute_forces=False,
        )

        # Translate all atoms
        translation = torch.tensor([1.5, 0.5, -0.3], dtype=torch.float64, device=device)
        positions2 = positions + translation

        energy2 = pme_reciprocal_space(
            positions=positions2,
            charges=charges,
            cell=cell,
            alpha=0.3,
            mesh_dimensions=(16, 16, 16),
            compute_forces=False,
        )

        assert torch.allclose(energy1.sum(), energy2.sum(), rtol=1e-4), (
            f"Energy not translation invariant: {energy1.sum()} vs {energy2.sum()}"
        )

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_opposite_charges_opposite_forces(self, device):
        """Test that opposite charges in same field get opposite forces."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions, charges, cell = create_dipole_system(device)

        _, forces = pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=0.3,
            mesh_dimensions=(16, 16, 16),
            compute_forces=True,
        )

        # For a symmetric dipole, forces should be equal and opposite
        assert torch.allclose(forces[0], -forces[1], rtol=1e-6), (
            f"Forces not equal and opposite: {forces[0]} vs {-forces[1]}"
        )


###########################################################################################
########################### Mesh Size Convergence Tests ###################################
###########################################################################################


class TestPMEConvergence:
    """Test that results converge with finer mesh."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_mesh_size_convergence(self, device):
        """Test that energy converges as mesh size increases."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions, charges, cell = create_dipole_system(device)

        mesh_sizes = [4, 8, 16, 64]
        energies = []

        for mesh_size in mesh_sizes:
            energy = pme_reciprocal_space(
                positions=positions,
                charges=charges,
                cell=cell,
                alpha=0.3,
                mesh_dimensions=(mesh_size, mesh_size, mesh_size),
                compute_forces=False,
            )
            energies.append(energy.sum().item())

        # Check convergence: differences should decrease
        diff_1 = abs(energies[1] - energies[0])
        diff_2 = abs(energies[2] - energies[1])
        diff_3 = abs(energies[3] - energies[2])

        assert diff_2 < diff_1, f"Energy not converging: {diff_1} -> {diff_2}"
        assert diff_3 < diff_2, f"Energy not converging: {diff_2} -> {diff_3}"


###########################################################################################
########################### Correctness Tests: Against TorchPME ###########################
###########################################################################################


@pytest.mark.skipif(not HAS_TORCHPME, reason="torchpme is not installed")
class TestPMECorrectnessTorchPME:
    """Validate PME implementation against torchpme reference."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    @pytest.mark.parametrize("alpha", [0.3, 0.5, 1.0])
    @pytest.mark.parametrize("mesh_spacing", [0.3, 0.5])
    def test_reciprocal_energy_matches_torchpme(self, device, alpha, mesh_spacing):
        """Test that reciprocal energy matches torchpme."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        dtype = torch.float64

        positions, charges, cell = create_dipole_system(device, dtype=dtype)

        # Estimate mesh size from spacing
        cell_lengths = torch.norm(cell, dim=1)
        mesh_dims = tuple(
            int(torch.ceil(length / mesh_spacing).item()) for length in cell_lengths
        )

        # Our implementation
        our_energy = pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            mesh_dimensions=mesh_dims,
            spline_order=4,
            compute_forces=False,
        )

        # TorchPME reference
        torchpme_energy = calculate_pme_reciprocal_energy_torchpme(
            positions, charges, cell, mesh_spacing, alpha, 4, device, dtype
        )

        assert torch.allclose(
            our_energy.sum(), torchpme_energy.sum(), rtol=1e-2, atol=1e-3
        ), (
            f"Energy mismatch: ours={our_energy.sum().item():.6f}, "
            f"torchpme={torchpme_energy.sum().item():.6f}"
        )

    @pytest.mark.parametrize("size", [1, 2])
    @pytest.mark.parametrize("system_fn", ["cscl", "wurtzite", "zincblende"])
    @pytest.mark.parametrize("alpha", [0.3, 0.5])
    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_crystal_systems_match_torchpme(self, size, system_fn, alpha, device):
        """Test PME on crystal systems against torchpme."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        dtype = torch.float64

        # Get system function
        system_fns = {
            "cscl": create_cscl_supercell,
            "wurtzite": create_wurtzite_system,
            "zincblende": create_zincblende_system,
        }
        system = system_fns[system_fn](size)

        cell = torch.tensor(system.cell, dtype=dtype, device=device)
        positions = torch.tensor(system.positions, dtype=dtype, device=device)
        charges = torch.tensor(system.charges, dtype=dtype, device=device)

        mesh_spacing = 0.5
        cell_lengths = torch.norm(cell, dim=1)
        mesh_dims = tuple(
            int(torch.ceil(length / mesh_spacing).item()) for length in cell_lengths
        )

        # Our implementation
        our_energy = pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            mesh_dimensions=mesh_dims,
            spline_order=4,
            compute_forces=False,
        )

        # TorchPME reference
        torchpme_energy = calculate_pme_reciprocal_energy_torchpme(
            positions, charges, cell, mesh_spacing, alpha, 4, device, dtype
        )

        assert torch.allclose(
            our_energy.sum(), torchpme_energy.sum(), rtol=1e-2, atol=1e-3
        ), (
            f"{system_fn} size={size} alpha={alpha}: "
            f"ours={our_energy.sum().item():.6f}, torchpme={torchpme_energy.sum().item():.6f}"
        )

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    @pytest.mark.parametrize("alpha", [0.3, 0.5, 0.75])
    @pytest.mark.parametrize("mesh_spacing", [0.3, 0.5])
    @pytest.mark.parametrize(
        "system_fn",
        [create_cscl_supercell, create_wurtzite_system, create_zincblende_system],
    )
    def test_reciprocal_energy_positions_grad_matches_torchpme(
        self, device, alpha, mesh_spacing, system_fn
    ):
        """Test that reciprocal energy matches torchpme."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        dtype = torch.float64

        system = system_fn(3)
        positions = torch.tensor(system.positions, dtype=dtype, device=device)
        charges = torch.tensor(system.charges, dtype=dtype, device=device)
        cell = torch.tensor(system.cell, dtype=dtype, device=device)

        # Estimate mesh size from spacing
        cell_lengths = torch.norm(cell, dim=1)
        mesh_dims = tuple(
            int(torch.ceil(length / mesh_spacing).item()) for length in cell_lengths
        )

        # Our implementation
        our_positions = positions.clone().requires_grad_(True)
        our_energy = pme_reciprocal_space(
            positions=our_positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            mesh_dimensions=mesh_dims,
            spline_order=4,
            compute_forces=False,
        )
        our_energy.sum().backward()
        our_forces = -our_positions.grad.clone()

        # TorchPME reference
        positions_torchpme = positions.clone().requires_grad_(True)
        torchpme_energy = calculate_pme_reciprocal_energy_torchpme(
            positions_torchpme, charges, cell, mesh_spacing, alpha, 4, device, dtype
        )
        torchpme_energy.sum().backward()
        torchpme_forces = -positions_torchpme.grad.clone()

        print(our_energy.sum(), our_forces[:2])
        print(torchpme_energy.sum(), torchpme_forces[:2])

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    @pytest.mark.parametrize("alpha", [0.3, 0.5])
    @pytest.mark.parametrize("mesh_spacing", [0.1, 0.5, 0.75])
    @pytest.mark.parametrize(
        "system_fn",
        [create_cscl_supercell, create_wurtzite_system, create_zincblende_system],
    )
    def test_reciprocal_energy_charges_grad_matches_torchpme(
        self, device, alpha, mesh_spacing, system_fn
    ):
        """Test that reciprocal energy matches torchpme."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        dtype = torch.float64

        system = system_fn(3)
        positions = torch.tensor(system.positions, dtype=dtype, device=device)
        charges = torch.tensor(system.charges, dtype=dtype, device=device)
        cell = torch.tensor(system.cell, dtype=dtype, device=device)

        # Estimate mesh size from spacing
        cell_lengths = torch.norm(cell, dim=1)
        mesh_dims = tuple(
            int(torch.ceil(length / mesh_spacing).item()) for length in cell_lengths
        )

        # Our implementation
        our_charges = charges.clone().requires_grad_(True)
        our_energy = pme_reciprocal_space(
            positions=positions,
            charges=our_charges,
            cell=cell,
            alpha=alpha,
            mesh_dimensions=mesh_dims,
            spline_order=4,
            compute_forces=False,
        )
        our_energy.sum().backward()
        our_grad = -our_charges.grad.clone()

        # TorchPME reference
        charges_torchpme = charges.clone().requires_grad_(True)
        torchpme_energy = calculate_pme_reciprocal_energy_torchpme(
            positions, charges_torchpme, cell, mesh_spacing, alpha, 4, device, dtype
        )
        # Use charges_torchpme (not charges) to get full gradient: d(q*φ)/dq = φ + q*dφ/dq
        torchpme_energy.sum().backward()
        torchpme_grad = -charges_torchpme.grad.clone()
        assert torch.allclose(our_grad, torchpme_grad, rtol=1e-3, atol=1e-3), (
            f"Grad mismatch: ours={our_grad}, torchpme={torchpme_grad}"
        )

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    @pytest.mark.parametrize("alpha", [0.3, 0.5])
    @pytest.mark.parametrize("mesh_spacing", [0.3, 0.5, 0.75])
    @pytest.mark.parametrize(
        "system_fn",
        [create_cscl_supercell, create_wurtzite_system, create_zincblende_system],
    )
    def test_reciprocal_energy_cell_grad_matches_torchpme(
        self, device, alpha, mesh_spacing, system_fn
    ):
        """Test that reciprocal energy matches torchpme."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        dtype = torch.float64

        system = system_fn(3)
        positions = torch.tensor(system.positions, dtype=dtype, device=device)
        charges = torch.tensor(system.charges, dtype=dtype, device=device)
        cell = torch.tensor(system.cell, dtype=dtype, device=device)

        # Estimate mesh size from spacing
        cell_lengths = torch.norm(cell, dim=1)
        mesh_dims = tuple(
            int(torch.ceil(length / mesh_spacing).item()) for length in cell_lengths
        )

        # Our implementation
        our_cell = cell.clone().requires_grad_(True)
        our_energy = pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=our_cell,
            alpha=alpha,
            mesh_dimensions=mesh_dims,
            spline_order=4,
            compute_forces=False,
        )
        our_energy.sum().backward()
        our_grad = -our_cell.grad.clone()

        # TorchPME reference
        cell_torchpme = cell.clone().requires_grad_(True)
        torchpme_energy = calculate_pme_reciprocal_energy_torchpme(
            positions, charges, cell_torchpme, mesh_spacing, alpha, 4, device, dtype
        )
        torchpme_energy.sum().backward()
        torchpme_grad = -cell_torchpme.grad.clone()
        assert torch.allclose(our_grad, torchpme_grad, rtol=1e-2, atol=1e-2), (
            f"Grad mismatch: ours={our_grad}, torchpme={torchpme_grad}"
        )


###########################################################################################
########################### Batch vs Single-System Consistency ############################
###########################################################################################


class TestPMEBatchConsistency:
    """Test that batch processing matches single-system processing."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_batch_single_system_matches(self, device):
        """Test batch with size 1 matches single-system."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions, charges, cell = create_dipole_system(device)

        # Single-system
        energy_single, forces_single = pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=0.3,
            mesh_dimensions=(16, 16, 16),
            compute_forces=True,
        )

        # Batch with size 1
        batch_idx = torch.zeros(positions.shape[0], dtype=torch.int32, device=device)
        energy_batch, forces_batch = pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell.unsqueeze(0),
            alpha=0.3,
            mesh_dimensions=(16, 16, 16),
            batch_idx=batch_idx,
            compute_forces=True,
        )

        assert torch.allclose(energy_batch.sum(), energy_single.sum(), rtol=1e-6), (
            f"Energy mismatch: batch={energy_batch.sum()}, single={energy_single.sum()}"
        )
        assert torch.allclose(forces_batch, forces_single, rtol=1e-6), (
            "Forces mismatch between batch and single-system"
        )

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_batch_multiple_systems_vs_sequential(self, device):
        """Test batch with multiple systems matches sequential single-system calls."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        dtype = torch.float64

        num_systems = 3

        # Create independent systems
        systems = []
        for i in range(num_systems):
            pos, chg, cell = create_simple_system(
                device, dtype, num_atoms=4 + i, cell_size=8.0 + i
            )
            systems.append((pos, chg, cell))

        # Sequential single-system calls
        energies_single = []
        forces_single = []
        for pos, chg, cell in systems:
            e, f = pme_reciprocal_space(
                positions=pos,
                charges=chg,
                cell=cell,
                alpha=0.3,
                mesh_dimensions=(16, 16, 16),
                compute_forces=True,
            )
            energies_single.append(e)
            forces_single.append(f)

        # Batch processing
        positions_batch = torch.cat([s[0] for s in systems], dim=0)
        charges_batch = torch.cat([s[1] for s in systems], dim=0)
        cells_batch = torch.stack([s[2] for s in systems], dim=0)

        atoms_per_system = [s[0].shape[0] for s in systems]
        batch_idx = torch.repeat_interleave(
            torch.arange(num_systems, device=device),
            torch.tensor(atoms_per_system, device=device),
        ).to(torch.int32)

        energies_batch, forces_batch = pme_reciprocal_space(
            positions=positions_batch,
            charges=charges_batch,
            cell=cells_batch,
            alpha=0.3,
            mesh_dimensions=(16, 16, 16),
            batch_idx=batch_idx,
            compute_forces=True,
        )

        # Compare per-system
        start_idx = 0
        for sys_idx, n_atoms in enumerate(atoms_per_system):
            end_idx = start_idx + n_atoms

            e_batch = energies_batch[start_idx:end_idx].sum()
            e_single = energies_single[sys_idx].sum()

            assert torch.allclose(e_batch, e_single, rtol=1e-4, atol=1e-6), (
                f"System {sys_idx}: Energy mismatch batch={e_batch} single={e_single}"
            )

            f_batch = forces_batch[start_idx:end_idx]
            f_single = forces_single[sys_idx]

            assert torch.allclose(f_batch, f_single, rtol=1e-4, atol=1e-6), (
                f"System {sys_idx}: Forces mismatch"
            )

            start_idx = end_idx

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_batch_different_cells(self, device):
        """Test batch with different cell sizes per system."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        dtype = torch.float64

        # Two systems with different cell sizes
        pos1 = torch.tensor(
            [[2.5, 2.5, 2.5], [3.5, 3.5, 3.5]], dtype=dtype, device=device
        )
        chg1 = torch.tensor([1.0, -1.0], dtype=dtype, device=device)
        cell1 = torch.eye(3, dtype=dtype, device=device) * 6.0

        pos2 = torch.tensor(
            [[4.0, 4.0, 4.0], [6.0, 6.0, 6.0]], dtype=dtype, device=device
        )
        chg2 = torch.tensor([0.5, -0.5], dtype=dtype, device=device)
        cell2 = torch.eye(3, dtype=dtype, device=device) * 10.0

        # Single-system calculations
        e1_single, f1_single = pme_reciprocal_space(
            positions=pos1,
            charges=chg1,
            cell=cell1,
            alpha=0.3,
            mesh_dimensions=(16, 16, 16),
            compute_forces=True,
        )
        e2_single, f2_single = pme_reciprocal_space(
            positions=pos2,
            charges=chg2,
            cell=cell2,
            alpha=0.3,
            mesh_dimensions=(16, 16, 16),
            compute_forces=True,
        )

        # Batch calculation
        positions_batch = torch.cat([pos1, pos2], dim=0)
        charges_batch = torch.cat([chg1, chg2], dim=0)
        cells_batch = torch.stack([cell1, cell2], dim=0)
        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)

        e_batch, f_batch = pme_reciprocal_space(
            positions=positions_batch,
            charges=charges_batch,
            cell=cells_batch,
            alpha=0.3,
            mesh_dimensions=(16, 16, 16),
            batch_idx=batch_idx,
            compute_forces=True,
        )

        # Compare
        assert torch.allclose(e_batch[:2].sum(), e1_single.sum(), rtol=1e-4)
        assert torch.allclose(e_batch[2:].sum(), e2_single.sum(), rtol=1e-4)
        assert torch.allclose(f_batch[:2], f1_single, rtol=1e-4)
        assert torch.allclose(f_batch[2:], f2_single, rtol=1e-4)

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_batch_conservation_per_system(self, device):
        """Test momentum conservation for each system in batch."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        num_systems = 3
        atoms_per_system = [4, 5, 3]

        # Create neutral systems
        positions_list = []
        charges_list = []
        for n_atoms in atoms_per_system:
            pos = torch.rand((n_atoms, 3), dtype=torch.float64, device=device) * 8.0
            chg = torch.randn(n_atoms, dtype=torch.float64, device=device)
            chg[-1] = -chg[:-1].sum()  # Neutralize
            positions_list.append(pos)
            charges_list.append(chg)

        positions = torch.cat(positions_list, dim=0)
        charges = torch.cat(charges_list, dim=0)
        cells = (
            torch.eye(3, dtype=torch.float64, device=device)
            .unsqueeze(0)
            .expand(num_systems, -1, -1)
            * 10.0
        )
        batch_idx = torch.repeat_interleave(
            torch.arange(num_systems, device=device),
            torch.tensor(atoms_per_system, device=device),
        ).to(torch.int32)

        _, forces = pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cells,
            alpha=0.3,
            mesh_dimensions=(16, 16, 16),
            batch_idx=batch_idx,
            compute_forces=True,
        )

        # Check momentum conservation per system
        start_idx = 0
        for sys_idx, n_atoms in enumerate(atoms_per_system):
            end_idx = start_idx + n_atoms
            net_force = forces[start_idx:end_idx].sum(dim=0)
            assert torch.allclose(
                net_force, torch.zeros(3, dtype=torch.float64, device=device), atol=1e-3
            ), f"System {sys_idx}: Net force = {net_force}"
            start_idx = end_idx

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    @pytest.mark.parametrize("system_fn", ["cscl", "wurtzite", "zincblende"])
    def test_batch_autograd_positions_vs_single(self, device, system_fn):
        """Test batch position gradients match single-system gradients."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        dtype = torch.float64

        system_fns = {
            "cscl": create_cscl_supercell,
            "wurtzite": create_wurtzite_system,
            "zincblende": create_zincblende_system,
        }

        # Create two systems of the same type with different sizes
        system1 = system_fns[system_fn](1)
        system2 = system_fns[system_fn](2)

        pos1 = torch.tensor(system1.positions, dtype=dtype, device=device)
        chg1 = torch.tensor(system1.charges, dtype=dtype, device=device)
        cell1 = torch.tensor(system1.cell, dtype=dtype, device=device)

        pos2 = torch.tensor(system2.positions, dtype=dtype, device=device)
        chg2 = torch.tensor(system2.charges, dtype=dtype, device=device)
        cell2 = torch.tensor(system2.cell, dtype=dtype, device=device)

        mesh_dims = (16, 16, 16)
        alpha = 0.3

        # Single-system gradients
        pos1_single = pos1.clone().requires_grad_(True)
        e1 = pme_reciprocal_space(
            positions=pos1_single,
            charges=chg1,
            cell=cell1,
            alpha=alpha,
            mesh_dimensions=mesh_dims,
            compute_forces=False,
        )
        e1.sum().backward()
        grad1_single = pos1_single.grad.clone()

        pos2_single = pos2.clone().requires_grad_(True)
        e2 = pme_reciprocal_space(
            positions=pos2_single,
            charges=chg2,
            cell=cell2,
            alpha=alpha,
            mesh_dimensions=mesh_dims,
            compute_forces=False,
        )
        e2.sum().backward()
        grad2_single = pos2_single.grad.clone()

        # Batch gradients
        n1, n2 = pos1.shape[0], pos2.shape[0]
        positions_batch = torch.cat([pos1, pos2], dim=0).clone().requires_grad_(True)
        charges_batch = torch.cat([chg1, chg2], dim=0)
        cells_batch = torch.stack([cell1, cell2], dim=0)
        batch_idx = torch.tensor([0] * n1 + [1] * n2, dtype=torch.int32, device=device)

        e_batch = pme_reciprocal_space(
            positions=positions_batch,
            charges=charges_batch,
            cell=cells_batch,
            alpha=alpha,
            mesh_dimensions=mesh_dims,
            batch_idx=batch_idx,
            compute_forces=False,
        )
        e_batch.sum().backward()

        grad1_batch = positions_batch.grad[:n1]
        grad2_batch = positions_batch.grad[n1:]

        assert torch.allclose(grad1_batch, grad1_single, rtol=1e-4, atol=1e-6), (
            f"{system_fn}: System 1 position gradients mismatch"
        )
        assert torch.allclose(grad2_batch, grad2_single, rtol=1e-4, atol=1e-6), (
            f"{system_fn}: System 2 position gradients mismatch"
        )

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    @pytest.mark.parametrize("system_fn", ["cscl", "wurtzite", "zincblende"])
    def test_batch_autograd_charges_vs_single(self, device, system_fn):
        """Test batch charge gradients match single-system gradients."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        dtype = torch.float64

        system_fns = {
            "cscl": create_cscl_supercell,
            "wurtzite": create_wurtzite_system,
            "zincblende": create_zincblende_system,
        }

        # Create two systems
        system1 = system_fns[system_fn](1)
        system2 = system_fns[system_fn](2)

        pos1 = torch.tensor(system1.positions, dtype=dtype, device=device)
        chg1 = torch.tensor(system1.charges, dtype=dtype, device=device)
        cell1 = torch.tensor(system1.cell, dtype=dtype, device=device)

        pos2 = torch.tensor(system2.positions, dtype=dtype, device=device)
        chg2 = torch.tensor(system2.charges, dtype=dtype, device=device)
        cell2 = torch.tensor(system2.cell, dtype=dtype, device=device)

        mesh_dims = (16, 16, 16)
        alpha = 0.3

        # Single-system gradients
        chg1_single = chg1.clone().requires_grad_(True)
        e1 = pme_reciprocal_space(
            positions=pos1,
            charges=chg1_single,
            cell=cell1,
            alpha=alpha,
            mesh_dimensions=mesh_dims,
            compute_forces=False,
        )
        e1.sum().backward()
        grad1_single = chg1_single.grad.clone()

        chg2_single = chg2.clone().requires_grad_(True)
        e2 = pme_reciprocal_space(
            positions=pos2,
            charges=chg2_single,
            cell=cell2,
            alpha=alpha,
            mesh_dimensions=mesh_dims,
            compute_forces=False,
        )
        e2.sum().backward()
        grad2_single = chg2_single.grad.clone()

        # Batch gradients
        n1, n2 = pos1.shape[0], pos2.shape[0]
        positions_batch = torch.cat([pos1, pos2], dim=0)
        charges_batch = torch.cat([chg1, chg2], dim=0).clone().requires_grad_(True)
        cells_batch = torch.stack([cell1, cell2], dim=0)
        batch_idx = torch.tensor([0] * n1 + [1] * n2, dtype=torch.int32, device=device)

        e_batch = pme_reciprocal_space(
            positions=positions_batch,
            charges=charges_batch,
            cell=cells_batch,
            alpha=alpha,
            mesh_dimensions=mesh_dims,
            batch_idx=batch_idx,
            compute_forces=False,
        )
        e_batch.sum().backward()

        grad1_batch = charges_batch.grad[:n1]
        grad2_batch = charges_batch.grad[n1:]

        assert torch.allclose(grad1_batch, grad1_single, rtol=1e-4, atol=1e-6), (
            f"{system_fn}: System 1 charge gradients mismatch"
        )
        assert torch.allclose(grad2_batch, grad2_single, rtol=1e-4, atol=1e-6), (
            f"{system_fn}: System 2 charge gradients mismatch"
        )

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    @pytest.mark.parametrize("system_fn", ["cscl", "wurtzite", "zincblende"])
    def test_batch_autograd_cell_vs_single(self, device, system_fn):
        """Test batch cell gradients match single-system gradients."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        dtype = torch.float64

        system_fns = {
            "cscl": create_cscl_supercell,
            "wurtzite": create_wurtzite_system,
            "zincblende": create_zincblende_system,
        }

        # Create two systems
        system1 = system_fns[system_fn](1)
        system2 = system_fns[system_fn](2)

        pos1 = torch.tensor(system1.positions, dtype=dtype, device=device)
        chg1 = torch.tensor(system1.charges, dtype=dtype, device=device)
        cell1 = torch.tensor(system1.cell, dtype=dtype, device=device)

        pos2 = torch.tensor(system2.positions, dtype=dtype, device=device)
        chg2 = torch.tensor(system2.charges, dtype=dtype, device=device)
        cell2 = torch.tensor(system2.cell, dtype=dtype, device=device)

        mesh_dims = (16, 16, 16)
        alpha = 0.3

        # Single-system gradients
        cell1_single = cell1.clone().requires_grad_(True)
        e1 = pme_reciprocal_space(
            positions=pos1,
            charges=chg1,
            cell=cell1_single,
            alpha=alpha,
            mesh_dimensions=mesh_dims,
            compute_forces=False,
        )
        e1.sum().backward()
        grad1_single = cell1_single.grad.clone()

        cell2_single = cell2.clone().requires_grad_(True)
        e2 = pme_reciprocal_space(
            positions=pos2,
            charges=chg2,
            cell=cell2_single,
            alpha=alpha,
            mesh_dimensions=mesh_dims,
            compute_forces=False,
        )
        e2.sum().backward()
        grad2_single = cell2_single.grad.clone()

        # Batch gradients
        n1, n2 = pos1.shape[0], pos2.shape[0]
        positions_batch = torch.cat([pos1, pos2], dim=0)
        charges_batch = torch.cat([chg1, chg2], dim=0)
        cells_batch = torch.stack([cell1, cell2], dim=0).clone().requires_grad_(True)
        batch_idx = torch.tensor([0] * n1 + [1] * n2, dtype=torch.int32, device=device)

        e_batch = pme_reciprocal_space(
            positions=positions_batch,
            charges=charges_batch,
            cell=cells_batch,
            alpha=alpha,
            mesh_dimensions=mesh_dims,
            batch_idx=batch_idx,
            compute_forces=False,
        )
        e_batch.sum().backward()

        grad1_batch = cells_batch.grad[0]
        grad2_batch = cells_batch.grad[1]

        assert torch.allclose(grad1_batch, grad1_single, rtol=1e-4, atol=1e-6), (
            f"{system_fn}: System 1 cell gradients mismatch:\n"
            f"  Batch: {grad1_batch}\n  Single: {grad1_single}"
        )
        assert torch.allclose(grad2_batch, grad2_single, rtol=1e-4, atol=1e-6), (
            f"{system_fn}: System 2 cell gradients mismatch:\n"
            f"  Batch: {grad2_batch}\n  Single: {grad2_single}"
        )

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    @pytest.mark.parametrize("system_fn", ["cscl", "wurtzite", "zincblende"])
    def test_batch_explicit_forces_vs_single(self, device, system_fn):
        """Test batch explicit forces match single-system explicit forces."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        dtype = torch.float64

        system_fns = {
            "cscl": create_cscl_supercell,
            "wurtzite": create_wurtzite_system,
            "zincblende": create_zincblende_system,
        }

        # Create two systems
        system1 = system_fns[system_fn](1)
        system2 = system_fns[system_fn](2)

        pos1 = torch.tensor(system1.positions, dtype=dtype, device=device)
        chg1 = torch.tensor(system1.charges, dtype=dtype, device=device)
        cell1 = torch.tensor(system1.cell, dtype=dtype, device=device)

        pos2 = torch.tensor(system2.positions, dtype=dtype, device=device)
        chg2 = torch.tensor(system2.charges, dtype=dtype, device=device)
        cell2 = torch.tensor(system2.cell, dtype=dtype, device=device)

        mesh_dims = (16, 16, 16)
        alpha = 0.3

        # Single-system forces
        _, forces1_single = pme_reciprocal_space(
            positions=pos1,
            charges=chg1,
            cell=cell1,
            alpha=alpha,
            mesh_dimensions=mesh_dims,
            compute_forces=True,
        )

        _, forces2_single = pme_reciprocal_space(
            positions=pos2,
            charges=chg2,
            cell=cell2,
            alpha=alpha,
            mesh_dimensions=mesh_dims,
            compute_forces=True,
        )

        # Batch forces
        n1, n2 = pos1.shape[0], pos2.shape[0]
        positions_batch = torch.cat([pos1, pos2], dim=0)
        charges_batch = torch.cat([chg1, chg2], dim=0)
        cells_batch = torch.stack([cell1, cell2], dim=0)
        batch_idx = torch.tensor([0] * n1 + [1] * n2, dtype=torch.int32, device=device)

        _, forces_batch = pme_reciprocal_space(
            positions=positions_batch,
            charges=charges_batch,
            cell=cells_batch,
            alpha=alpha,
            mesh_dimensions=mesh_dims,
            batch_idx=batch_idx,
            compute_forces=True,
        )

        forces1_batch = forces_batch[:n1]
        forces2_batch = forces_batch[n1:]

        assert torch.allclose(forces1_batch, forces1_single, rtol=1e-4, atol=1e-6), (
            f"{system_fn}: System 1 forces mismatch"
        )
        assert torch.allclose(forces2_batch, forces2_single, rtol=1e-4, atol=1e-6), (
            f"{system_fn}: System 2 forces mismatch"
        )


###########################################################################################
########################### Autograd Tests ################################################
###########################################################################################


class TestPMEAutograd:
    """Test autograd functionality for PME operations."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_energy_autograd_positions(self, device, dtype):
        """Test gradients w.r.t. positions in energy calculation."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions, charges, cell = create_dipole_system(device, dtype=dtype)
        positions = positions.clone().requires_grad_(True)

        energies = pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=0.3,
            mesh_dimensions=(16, 16, 16),
            compute_forces=False,
        )

        loss = energies.sum()
        loss.backward()

        assert positions.grad is not None, "Position gradients not computed"
        assert positions.grad.shape == positions.shape
        assert torch.all(torch.isfinite(positions.grad))

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_energy_autograd_charges(self, device, dtype):
        """Test gradients w.r.t. charges in energy calculation."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions, charges, cell = create_dipole_system(device, dtype=dtype)
        charges = charges.clone().requires_grad_(True)

        energies = pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=0.3,
            mesh_dimensions=(16, 16, 16),
            compute_forces=False,
        )

        loss = energies.sum()
        loss.backward()

        assert charges.grad is not None, "Charge gradients not computed"
        assert charges.grad.shape == charges.shape
        assert torch.all(torch.isfinite(charges.grad))

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_forces_match_negative_energy_gradient(self, device):
        """Test that explicit forces match -dE/dr from autograd."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions, charges, cell = create_dipole_system(device)
        positions_ad = positions.clone().requires_grad_(True)

        mesh_spacing = 0.5
        # Estimate mesh size from spacing
        cell_lengths = torch.norm(cell, dim=1)
        mesh_dims = tuple(
            int(torch.ceil(length / mesh_spacing).item()) for length in cell_lengths
        )

        # Compute explicit forces
        _, explicit_forces = pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=0.3,
            mesh_dimensions=mesh_dims,
            compute_forces=True,
        )

        # Compute autograd forces
        energies_ad = pme_reciprocal_space(
            positions=positions_ad,
            charges=charges,
            cell=cell,
            alpha=0.3,
            mesh_dimensions=(20, 20, 20),
            compute_forces=False,
        )

        total_energy = energies_ad.sum()
        total_energy.backward()
        autograd_forces = -positions_ad.grad

        # Compare
        assert torch.allclose(explicit_forces, autograd_forces, rtol=1e-3, atol=1e-4), (
            f"Forces mismatch:\n"
            f"  Explicit: {explicit_forces}\n"
            f"  Autograd: {autograd_forces}\n"
            f"  Diff: {(explicit_forces - autograd_forces).abs().max()}"
        )

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_batch_autograd_positions(self, device):
        """Test gradients w.r.t. positions in batch calculation."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        # Create positions as leaf tensor (don't multiply before requires_grad)
        positions = (
            (torch.rand((6, 3), dtype=torch.float64, device=device) * 8.0)
            .clone()
            .requires_grad_(True)
        )
        charges = torch.randn(6, dtype=torch.float64, device=device)
        batch_idx = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int32, device=device)
        cells = (
            torch.eye(3, dtype=torch.float64, device=device)
            .unsqueeze(0)
            .expand(2, -1, -1)
            .contiguous()
            * 10.0
        )

        energies = pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cells,
            alpha=0.3,
            mesh_dimensions=(16, 16, 16),
            batch_idx=batch_idx,
            compute_forces=False,
        )

        loss = energies.sum()
        loss.backward()

        assert positions.grad is not None
        assert torch.all(torch.isfinite(positions.grad))


###########################################################################################
########################### Forces vs Numerical Gradient ##################################
###########################################################################################


class TestPMEForcesNumericalGradient:
    """Validate forces against numerical gradients."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_forces_vs_finite_differences(self, device):
        """Test that analytical forces match finite difference gradients."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions, charges, cell = create_dipole_system(device)

        # Slightly perturb positions to avoid symmetric configurations
        positions = positions + torch.randn_like(positions) * 0.1

        # Analytical forces
        _, analytical_forces = pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=0.5,
            mesh_dimensions=(24, 24, 24),
            compute_forces=True,
        )

        # Numerical forces via finite differences
        h = 1e-5
        numerical_forces = torch.zeros_like(positions)

        for atom_idx in range(positions.shape[0]):
            for coord_idx in range(3):
                # Forward
                pos_plus = positions.clone()
                pos_plus[atom_idx, coord_idx] += h
                e_plus = pme_reciprocal_space(
                    positions=pos_plus,
                    charges=charges,
                    cell=cell,
                    alpha=0.5,
                    mesh_dimensions=(24, 24, 24),
                    compute_forces=False,
                )

                # Backward
                pos_minus = positions.clone()
                pos_minus[atom_idx, coord_idx] -= h
                e_minus = pme_reciprocal_space(
                    positions=pos_minus,
                    charges=charges,
                    cell=cell,
                    alpha=0.5,
                    mesh_dimensions=(24, 24, 24),
                    compute_forces=False,
                )

                # Central difference: F = -dE/dr
                numerical_forces[atom_idx, coord_idx] = -(
                    e_plus.sum() - e_minus.sum()
                ) / (2 * h)

        assert torch.allclose(
            analytical_forces, numerical_forces, rtol=1e-2, atol=1e-4
        ), (
            f"Forces don't match numerical gradient:\n"
            f"  Max diff: {(analytical_forces - numerical_forces).abs().max()}\n"
            f"  Analytical: {analytical_forces}\n"
            f"  Numerical: {numerical_forces}"
        )


###########################################################################################
########################### Full PME (Real + Reciprocal) Tests ############################
###########################################################################################


class TestParticleMeshEwald:
    """Test the combined particle_mesh_ewald function."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_full_pme_output_shape(self, device):
        """Test output shape of full PME calculation."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions, charges, cell = create_simple_system(device, num_atoms=5)

        # Create simple neighbor matrix (all pairs)
        num_atoms = positions.shape[0]
        neighbor_matrix = torch.zeros(
            (num_atoms, num_atoms - 1), dtype=torch.int32, device=device
        )
        for i in range(num_atoms):
            neighbors = [j for j in range(num_atoms) if j != i]
            neighbor_matrix[i] = torch.tensor(
                neighbors, dtype=torch.int32, device=device
            )
        neighbor_matrix_shifts = torch.zeros(
            (num_atoms, num_atoms - 1, 3), dtype=torch.int32, device=device
        )

        energies, forces = particle_mesh_ewald(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=0.3,
            mesh_dimensions=(16, 16, 16),
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            compute_forces=True,
        )

        assert energies.shape == (5,)
        assert forces.shape == (5, 3)
        assert torch.all(torch.isfinite(energies))
        assert torch.all(torch.isfinite(forces))


class TestSingleAtomSystem:
    """Test handling of single atom systems."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_single_atom_pme(self, device):
        """Test PME with single atom."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[5.0, 5.0, 5.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0], dtype=torch.float64, device=device)
        cell = torch.eye(3, dtype=torch.float64, device=device) * 10.0

        energies, forces = pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=0.3,
            mesh_dimensions=(16, 16, 16),
            compute_forces=True,
        )

        assert energies.shape == (1,)
        assert forces.shape == (1, 3)
        assert torch.isfinite(energies).all()
        assert torch.isfinite(forces).all()


class TestNonCubicCells:
    """Test PME with non-cubic simulation cells."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_orthorhombic_cell(self, device):
        """Test PME with orthorhombic cell."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        # Orthorhombic cell
        cell = torch.tensor(
            [[8.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 12.0]],
            dtype=torch.float64,
            device=device,
        )
        positions = torch.tensor(
            [[2.0, 5.0, 6.0], [6.0, 5.0, 6.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)

        energies, forces = pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=0.3,
            mesh_dimensions=(16, 20, 24),
            compute_forces=True,
        )

        assert torch.isfinite(energies).all()
        assert torch.isfinite(forces).all()
        # Momentum conservation
        net_force = forces.sum(dim=0)
        assert torch.allclose(
            net_force, torch.zeros(3, dtype=torch.float64, device=device), atol=1e-4
        )

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_triclinic_cell(self, device):
        """Test PME with triclinic cell."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        # Triclinic cell
        cell = torch.tensor(
            [[10.0, 0.0, 0.0], [2.0, 10.0, 0.0], [1.0, 1.0, 10.0]],
            dtype=torch.float64,
            device=device,
        )
        positions = torch.tensor(
            [[2.0, 5.0, 5.0], [7.0, 5.0, 5.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)

        energies, forces = pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=0.3,
            mesh_dimensions=(16, 16, 16),
            compute_forces=True,
        )

        assert torch.isfinite(energies).all()
        assert torch.isfinite(forces).all()


class TestPrecomputedKVectors:
    """Test PME with precomputed k-vectors."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_precomputed_kvectors(self, device):
        """Test that precomputed k-vectors give same results."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        from nvalchemiops.torch.interactions.electrostatics.k_vectors import (
            generate_k_vectors_pme,
        )

        positions, charges, cell = create_dipole_system(device)
        mesh_dims = (16, 16, 16)
        alpha = 0.3

        # Without precomputed k-vectors
        energies1, forces1 = pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            mesh_dimensions=mesh_dims,
            compute_forces=True,
        )

        # With precomputed k-vectors
        k_vectors, k_squared = generate_k_vectors_pme(cell, mesh_dims)
        energies2, forces2 = pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            mesh_dimensions=mesh_dims,
            compute_forces=True,
            k_vectors=k_vectors,
            k_squared=k_squared,
        )

        assert torch.allclose(energies1, energies2, rtol=1e-6)
        assert torch.allclose(forces1, forces2, rtol=1e-6)


class TestSplineOrders:
    """Test different spline interpolation orders."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    @pytest.mark.parametrize("spline_order", [2, 3, 4, 5, 6])
    def test_spline_order_convergence(self, device, spline_order):
        """Test that higher spline orders give valid results."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions, charges, cell = create_dipole_system(device)

        energies, forces = pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=0.3,
            mesh_dimensions=(32, 32, 32),
            spline_order=spline_order,
            compute_forces=True,
        )

        assert torch.isfinite(energies).all()
        assert torch.isfinite(forces).all()
        # Momentum conservation
        net_force = forces.sum(dim=0)
        assert torch.allclose(
            net_force, torch.zeros(3, dtype=torch.float64, device=device), atol=1e-3
        )


class TestFullPMENeighborList:
    """Test full PME with neighbor list format."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_full_pme_neighbor_list(self, device):
        """Test full PME with neighbor list format."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions, charges, cell = create_simple_system(device, num_atoms=4)
        num_atoms = positions.shape[0]
        # Create neighbor list (all pairs)
        neighbor_list, neighbor_ptr, neighbor_shifts = cell_list(
            positions,
            cutoff=5.0,
            cell=cell,
            pbc=torch.tensor([True, True, True], dtype=torch.bool, device=device),
            return_neighbor_list=True,
        )

        energies, forces = particle_mesh_ewald(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=0.3,
            mesh_dimensions=(16, 16, 16),
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            compute_forces=True,
        )

        assert energies.shape == (num_atoms,)
        assert forces.shape == (num_atoms, 3)
        assert torch.isfinite(energies).all()
        assert torch.isfinite(forces).all()


class TestCellGradients:
    """Test gradients with respect to cell matrix."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_cell_gradient_finite(self, device):
        """Test that cell gradients are finite."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions, charges, cell = create_dipole_system(device)
        cell = cell.clone().requires_grad_(True)

        energies = pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=0.3,
            mesh_dimensions=(16, 16, 16),
            compute_forces=False,
        )
        energies.sum().backward()

        assert cell.grad is not None
        assert torch.isfinite(cell.grad).all()


class TestAlphaSensitivity:
    """Test sensitivity to alpha parameter."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_alpha_affects_energy(self, device):
        """Test that different alpha values affect energy."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions, charges, cell = create_dipole_system(device)

        energies_low_alpha = pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=0.2,
            mesh_dimensions=(16, 16, 16),
            compute_forces=False,
        )

        energies_high_alpha = pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=0.5,
            mesh_dimensions=(16, 16, 16),
            compute_forces=False,
        )

        # Different alpha should give different energies
        assert not torch.allclose(energies_low_alpha, energies_high_alpha)


class TestZeroCharges:
    """Test behavior with zero charges."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_zero_charges_zero_energy(self, device):
        """Test that zero charges give zero energy."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[2.0, 5.0, 5.0], [8.0, 5.0, 5.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([0.0, 0.0], dtype=torch.float64, device=device)
        cell = torch.eye(3, dtype=torch.float64, device=device) * 10.0

        energies, forces = pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=0.3,
            mesh_dimensions=(16, 16, 16),
            compute_forces=True,
        )

        assert torch.allclose(energies, torch.zeros_like(energies), atol=1e-10)
        assert torch.allclose(forces, torch.zeros_like(forces), atol=1e-10)


class TestBatchWithDifferentAlpha:
    """Test batch calculations with per-system alpha."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_batch_per_system_alpha(self, device):
        """Test batch with different alpha per system."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        # Create two systems
        pos1, chg1, cell1 = create_dipole_system(device)
        pos2, chg2, cell2 = create_dipole_system(device, separation=3.0)

        positions = torch.cat([pos1, pos2], dim=0)
        charges = torch.cat([chg1, chg2], dim=0)
        cells = torch.stack([cell1, cell2], dim=0)
        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)

        # Different alpha per system
        alphas = torch.tensor([0.2, 0.5], dtype=torch.float64, device=device)

        energies, forces = pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cells,
            alpha=alphas,
            mesh_dimensions=(16, 16, 16),
            batch_idx=batch_idx,
            compute_forces=True,
        )

        assert energies.shape == (4,)
        assert forces.shape == (4, 3)
        assert torch.isfinite(energies).all()
        assert torch.isfinite(forces).all()


class TestPrepareAlphaPME:
    """Test _prepare_alpha edge cases in PME for coverage."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_scalar_alpha_tensor_0d(self, device):
        """Test 0-dimensional alpha tensor expansion (line 189)."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions, charges, cell = create_simple_system(device, num_atoms=5)

        # 0-dimensional tensor (scalar tensor)
        alpha = torch.tensor(0.3, dtype=torch.float64, device=device)

        energies = pme_reciprocal_space(
            positions,
            charges,
            cell,
            alpha=alpha,  # 0-dim tensor
            mesh_dimensions=(16, 16, 16),
            compute_forces=False,
        )

        assert torch.isfinite(energies).all()

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_alpha_wrong_size_raises_error(self, device):
        """Test alpha tensor with wrong number of elements raises ValueError (line 191)."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions, charges, cell = create_simple_system(device, num_atoms=5)

        # Alpha tensor with wrong size (2 values for 1 system)
        alpha = torch.tensor([0.3, 0.5], dtype=torch.float64, device=device)

        with pytest.raises(ValueError):
            pme_reciprocal_space(
                positions,
                charges,
                cell,
                alpha=alpha,  # Wrong size
                mesh_dimensions=(16, 16, 16),
                compute_forces=False,
            )

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_alpha_invalid_type_raises_error(self, device):
        """Test non-float, non-tensor alpha raises TypeError (line 196)."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions, charges, cell = create_simple_system(device, num_atoms=5)

        with pytest.raises(TypeError):
            pme_reciprocal_space(
                positions,
                charges,
                cell,
                alpha="invalid",  # String is not valid
                mesh_dimensions=(16, 16, 16),
                compute_forces=False,
            )


class TestPMEMeshDimensionErrors:
    """Test mesh dimension error handling for coverage."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_no_mesh_dimensions_or_spacing_raises_error(self, device):
        """Test ValueError when neither mesh_dimensions nor mesh_spacing (lines 1277-1280)."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions, charges, cell = create_simple_system(device, num_atoms=5)

        with pytest.raises(
            ValueError, match="Either mesh_dimensions or mesh_spacing must be provided"
        ):
            pme_reciprocal_space(
                positions,
                charges,
                cell,
                alpha=0.3,
                mesh_dimensions=None,  # Not provided
                mesh_spacing=None,  # Not provided
                compute_forces=False,
            )

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_mesh_spacing_path(self, device):
        """Test mesh_spacing path for dimension computation (line 1281-1284)."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions, charges, cell = create_simple_system(device, num_atoms=5)

        # Use mesh_spacing instead of mesh_dimensions
        energies = pme_reciprocal_space(
            positions,
            charges,
            cell,
            alpha=0.3,
            mesh_spacing=0.5,  # Use spacing
            compute_forces=False,
        )

        assert torch.isfinite(energies).all()


class TestParticleMeshEwaldAutoEstimation:
    """Test particle_mesh_ewald auto-estimation paths for coverage."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_auto_estimate_alpha(self, device):
        """Test auto-estimation of alpha in particle_mesh_ewald (lines 1463-1466)."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions, charges, cell = create_simple_system(device, num_atoms=5)
        neighbor_list, neighbor_ptr, neighbor_shifts = cell_list(
            positions,
            cutoff=5.0,
            cell=cell,
            pbc=torch.tensor([True, True, True], dtype=torch.bool, device=device),
            return_neighbor_list=True,
        )
        # Call without alpha - should auto-estimate
        energies, forces = particle_mesh_ewald(
            positions,
            charges,
            cell,
            alpha=None,  # Auto-estimate
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            compute_forces=True,
        )

        assert energies.shape == (5,)
        assert forces.shape == (5, 3)
        assert torch.isfinite(energies).all()
        assert torch.isfinite(forces).all()

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_mesh_spacing_in_particle_mesh_ewald(self, device):
        """Test mesh_spacing path in particle_mesh_ewald (lines 1476-1477)."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions, charges, cell = create_simple_system(device, num_atoms=5)
        neighbor_list, neighbor_ptr, neighbor_shifts = cell_list(
            positions,
            cutoff=5.0,
            cell=cell,
            pbc=torch.tensor([True, True, True], dtype=torch.bool, device=device),
            return_neighbor_list=True,
        )

        energies, forces = particle_mesh_ewald(
            positions,
            charges,
            cell,
            alpha=0.3,
            mesh_spacing=0.5,  # Use spacing instead of dimensions
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            compute_forces=True,
        )

        assert energies.shape == (5,)
        assert forces.shape == (5, 3)
        assert torch.isfinite(energies).all()
        assert torch.isfinite(forces).all()

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_accuracy_based_mesh_estimation(self, device):
        """Test accuracy-based mesh dimension estimation (lines 1478-1480)."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions, charges, cell = create_simple_system(device, num_atoms=5)
        neighbor_list, neighbor_ptr, neighbor_shifts = cell_list(
            positions,
            cutoff=5.0,
            cell=cell,
            pbc=torch.tensor([True, True, True], dtype=torch.bool, device=device),
            return_neighbor_list=True,
        )

        # Provide alpha but no mesh_dimensions or mesh_spacing
        # Should use accuracy-based estimation
        energies, forces = particle_mesh_ewald(
            positions,
            charges,
            cell,
            alpha=0.3,
            mesh_dimensions=None,
            mesh_spacing=None,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            compute_forces=True,
            accuracy=1e-4,
        )

        assert energies.shape == (5,)
        assert forces.shape == (5, 3)
        assert torch.isfinite(energies).all()
        assert torch.isfinite(forces).all()

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_default_mask_value_pme(self, device):
        """Test particle_mesh_ewald with default mask_value (None -> num_atoms)."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions, charges, cell = create_simple_system(device, num_atoms=5)

        # Use neighbor matrix format without explicit mask_value
        neighbor_matrix = torch.tensor(
            [[1, -1], [0, 2], [1, 3], [2, 4], [3, -1]], dtype=torch.int32, device=device
        )
        neighbor_matrix_shifts = torch.zeros(
            (5, 2, 3), dtype=torch.int32, device=device
        )

        energies, forces = particle_mesh_ewald(
            positions,
            charges,
            cell,
            alpha=0.3,
            mesh_dimensions=(16, 16, 16),
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            # mask_value=None -> defaults to num_atoms
            compute_forces=True,
        )

        assert energies.shape == (5,)
        assert forces.shape == (5, 3)
        assert torch.isfinite(energies).all()
        assert torch.isfinite(forces).all()

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_auto_mesh_from_alpha_estimation(self, device):
        """Test mesh_dimensions auto-derived from alpha estimation (lines 1465-1467)."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions, charges, cell = create_simple_system(device, num_atoms=5)
        neighbor_list, neighbor_ptr, neighbor_shifts = cell_list(
            positions,
            cutoff=5.0,
            cell=cell,
            pbc=torch.tensor([True, True, True], dtype=torch.bool, device=device),
            return_neighbor_list=True,
        )

        # alpha=None triggers estimate_pme_parameters which sets alpha AND mesh_dimensions
        # Neither mesh_dimensions nor mesh_spacing provided
        energies, forces = particle_mesh_ewald(
            positions,
            charges,
            cell,
            alpha=None,  # Triggers auto-estimation
            mesh_dimensions=None,  # Will be set from params
            mesh_spacing=None,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            compute_forces=True,
        )

        assert energies.shape == (5,)
        assert forces.shape == (5, 3)
        assert torch.isfinite(energies).all()
        assert torch.isfinite(forces).all()


class TestBatchPMEShapePaths:
    """Test batch PME shape helper code paths."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_batch_reciprocal_space_single_system(self, device):
        """Test batch reciprocal space with single system (3D k_squared)."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions, charges, cell = create_simple_system(device, num_atoms=5)
        batch_idx = torch.zeros(5, dtype=torch.int32, device=device)  # All same batch

        energies, forces = pme_reciprocal_space(
            positions,
            charges,
            cell,
            alpha=0.3,
            mesh_dimensions=(16, 16, 16),
            batch_idx=batch_idx,
            compute_forces=True,
        )

        assert energies.shape == (5,)
        assert forces.shape == (5, 3)
        assert torch.isfinite(energies).all()
        assert torch.isfinite(forces).all()

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_batch_reciprocal_space_multi_system(self, device):
        """Test batch reciprocal space with multiple systems (exercises batch shape helpers)."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        # Create two batched systems
        positions = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor(
            [1.0, -1.0, 0.5, -0.5], dtype=torch.float64, device=device
        )
        cell = torch.stack(
            [
                torch.eye(3, dtype=torch.float64, device=device) * 10.0,
                torch.eye(3, dtype=torch.float64, device=device) * 10.0,
            ]
        )
        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)

        energies, forces = pme_reciprocal_space(
            positions,
            charges,
            cell,
            alpha=torch.tensor([0.3, 0.3], dtype=torch.float64, device=device),
            mesh_dimensions=(16, 16, 16),
            batch_idx=batch_idx,
            compute_forces=True,
        )

        assert energies.shape == (4,)
        assert forces.shape == (4, 3)
        assert torch.isfinite(energies).all()
        assert torch.isfinite(forces).all()


class TestPMEChargeGradients:
    """Test explicit charge gradients (compute_charge_gradients=True) for PME."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_reciprocal_charge_grad_matches_autograd(self, device):
        """Test that explicit charge gradients match autograd for pme_reciprocal_space."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions, charges, cell = create_simple_system(device, num_atoms=4)

        # Get explicit charge gradients
        energies, charge_grads = pme_reciprocal_space(
            positions,
            charges,
            cell,
            alpha=0.3,
            mesh_dimensions=(16, 16, 16),
            compute_forces=False,
            compute_charge_gradients=True,
        )

        # Get autograd charge gradients
        charges_ad = charges.clone().requires_grad_(True)
        energies_ad = pme_reciprocal_space(
            positions,
            charges_ad,
            cell,
            alpha=0.3,
            mesh_dimensions=(16, 16, 16),
            compute_forces=False,
            compute_charge_gradients=False,
        )
        energies_ad.sum().backward()
        autograd_charge_grad = charges_ad.grad.clone()

        assert torch.allclose(
            charge_grads, autograd_charge_grad, rtol=1e-4, atol=1e-7
        ), (
            f"Charge gradients mismatch: explicit={charge_grads}, "
            f"autograd={autograd_charge_grad}"
        )

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_reciprocal_charge_grad_with_forces(self, device):
        """Test charge gradients when compute_forces=True for pme_reciprocal_space."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions, charges, cell = create_simple_system(device, num_atoms=4)

        # Get explicit charge gradients with forces
        energies, forces, charge_grads = pme_reciprocal_space(
            positions,
            charges,
            cell,
            alpha=0.3,
            mesh_dimensions=(16, 16, 16),
            compute_forces=True,
            compute_charge_gradients=True,
        )

        assert energies.shape == (4,)
        assert forces.shape == (4, 3)
        assert charge_grads.shape == (4,)
        assert torch.isfinite(energies).all()
        assert torch.isfinite(forces).all()
        assert torch.isfinite(charge_grads).all()

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_full_pme_charge_grad_matches_autograd(self, device):
        """Test that explicit charge gradients match autograd for particle_mesh_ewald."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions, charges, cell = create_simple_system(device, num_atoms=4)
        neighbor_list, neighbor_ptr, neighbor_shifts = cell_list(
            positions,
            cutoff=5.0,
            cell=cell,
            pbc=torch.tensor([True, True, True], dtype=torch.bool, device=device),
            return_neighbor_list=True,
        )

        # Get explicit charge gradients
        energies, charge_grads = particle_mesh_ewald(
            positions,
            charges,
            cell,
            alpha=0.3,
            mesh_dimensions=(16, 16, 16),
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            compute_forces=False,
            compute_charge_gradients=True,
        )

        # Get autograd charge gradients
        charges_ad = charges.clone().requires_grad_(True)
        energies_ad = particle_mesh_ewald(
            positions,
            charges_ad,
            cell,
            alpha=0.3,
            mesh_dimensions=(16, 16, 16),
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            compute_forces=False,
            compute_charge_gradients=False,
        )
        energies_ad.sum().backward()
        autograd_charge_grad = charges_ad.grad.clone()

        assert torch.allclose(
            charge_grads, autograd_charge_grad, rtol=1e-4, atol=1e-7
        ), (
            f"Charge gradients mismatch: explicit={charge_grads}, "
            f"autograd={autograd_charge_grad}"
        )

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_full_pme_charge_grad_with_forces(self, device):
        """Test charge gradients when compute_forces=True for particle_mesh_ewald."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions, charges, cell = create_simple_system(device, num_atoms=4)
        neighbor_list, neighbor_ptr, neighbor_shifts = cell_list(
            positions,
            cutoff=5.0,
            cell=cell,
            pbc=torch.tensor([True, True, True], dtype=torch.bool, device=device),
            return_neighbor_list=True,
        )

        # Get explicit charge gradients with forces
        energies, forces, charge_grads = particle_mesh_ewald(
            positions,
            charges,
            cell,
            alpha=0.3,
            mesh_dimensions=(16, 16, 16),
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            compute_forces=True,
            compute_charge_gradients=True,
        )

        assert energies.shape == (4,)
        assert forces.shape == (4, 3)
        assert charge_grads.shape == (4,)
        assert torch.isfinite(energies).all()
        assert torch.isfinite(forces).all()
        assert torch.isfinite(charge_grads).all()

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_batch_reciprocal_charge_grad(self, device):
        """Test charge gradients for batch pme_reciprocal_space."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        # Create batched system
        positions = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor(
            [1.0, -1.0, 0.5, -0.5], dtype=torch.float64, device=device
        )
        cell = torch.stack(
            [
                torch.eye(3, dtype=torch.float64, device=device) * 10.0,
                torch.eye(3, dtype=torch.float64, device=device) * 10.0,
            ]
        )
        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)

        # Get explicit charge gradients
        energies, charge_grads = pme_reciprocal_space(
            positions,
            charges,
            cell,
            alpha=torch.tensor([0.3, 0.3], dtype=torch.float64, device=device),
            mesh_dimensions=(16, 16, 16),
            batch_idx=batch_idx,
            compute_forces=False,
            compute_charge_gradients=True,
        )

        # Get autograd charge gradients
        charges_ad = charges.clone().requires_grad_(True)
        energies_ad = pme_reciprocal_space(
            positions,
            charges_ad,
            cell,
            alpha=torch.tensor([0.3, 0.3], dtype=torch.float64, device=device),
            mesh_dimensions=(16, 16, 16),
            batch_idx=batch_idx,
            compute_forces=False,
            compute_charge_gradients=False,
        )
        energies_ad.sum().backward()
        autograd_charge_grad = charges_ad.grad.clone()

        assert torch.allclose(
            charge_grads, autograd_charge_grad, rtol=1e-4, atol=1e-7
        ), (
            f"Batch charge gradients mismatch: explicit={charge_grads}, "
            f"autograd={autograd_charge_grad}"
        )

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_batch_full_pme_charge_grad(self, device):
        """Test charge gradients for batch particle_mesh_ewald."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        # Create batched system
        positions = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor(
            [1.0, -1.0, 0.5, -0.5], dtype=torch.float64, device=device
        )
        cell = torch.stack(
            [
                torch.eye(3, dtype=torch.float64, device=device) * 10.0,
                torch.eye(3, dtype=torch.float64, device=device) * 10.0,
            ]
        )
        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)
        pbc = torch.tensor(
            [[True, True, True], [True, True, True]], dtype=torch.bool, device=device
        )

        neighbor_list, neighbor_ptr, neighbor_shifts = batch_cell_list(
            positions,
            cutoff=5.0,
            cell=cell,
            pbc=pbc,
            batch_idx=batch_idx,
            return_neighbor_list=True,
        )

        print("Neighbor list shape:", neighbor_list.shape)
        print("Neighbor ptr shape:", neighbor_ptr.shape)
        print("Neighbor shifts shape:", neighbor_shifts.shape)

        # Get explicit charge gradients with forces
        energies, forces, charge_grads = particle_mesh_ewald(
            positions,
            charges,
            cell,
            alpha=torch.tensor([0.3, 0.3], dtype=torch.float64, device=device),
            mesh_dimensions=(16, 16, 16),
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            batch_idx=batch_idx,
            compute_forces=True,
            compute_charge_gradients=True,
        )

        # Get autograd charge gradients
        charges_ad = charges.clone().requires_grad_(True)
        energies_ad = particle_mesh_ewald(
            positions,
            charges_ad,
            cell,
            alpha=torch.tensor([0.3, 0.3], dtype=torch.float64, device=device),
            mesh_dimensions=(16, 16, 16),
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            batch_idx=batch_idx,
            compute_forces=False,
            compute_charge_gradients=False,
        )
        energies_ad.sum().backward()
        autograd_charge_grad = charges_ad.grad.clone()

        assert torch.allclose(
            charge_grads, autograd_charge_grad, rtol=1e-4, atol=1e-7
        ), (
            f"Batch charge gradients mismatch: explicit={charge_grads}, "
            f"autograd={autograd_charge_grad}"
        )


###########################################################################################
########################### Virial Tests ##################################################
###########################################################################################


class TestPMEReciprocalVirial:
    """Test PME reciprocal-space virial against FD and basic properties."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_pme_reciprocal_virial_shape(self, device):
        """PME reciprocal virial has shape (1, 3, 3)."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        positions, charges, cell = make_virial_cscl_system(2, device=device)
        alpha = torch.tensor([0.3], dtype=VIRIAL_DTYPE, device=device)

        result = pme_reciprocal_space(
            positions,
            charges,
            cell,
            alpha,
            mesh_dimensions=(16, 16, 16),
            compute_forces=True,
            compute_virial=True,
        )
        virial = result[2]
        assert virial.shape == (1, 3, 3)
        assert virial.dtype == VIRIAL_DTYPE

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_pme_reciprocal_virial_fd(self, device):
        """PME reciprocal virial matches FD strain derivative."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        positions, charges, cell = make_virial_cscl_system(2, device=device)
        alpha = torch.tensor([0.3], dtype=VIRIAL_DTYPE, device=device)
        mesh_dims = (16, 16, 16)

        def energy_fn(pos, c):
            return pme_reciprocal_space(
                pos,
                charges,
                c,
                alpha,
                mesh_dimensions=mesh_dims,
                compute_forces=False,
            ).sum()

        result = pme_reciprocal_space(
            positions,
            charges,
            cell,
            alpha,
            mesh_dimensions=mesh_dims,
            compute_forces=True,
            compute_virial=True,
        )
        explicit_virial = result[2].squeeze(0)
        fd_virial = fd_virial_full(energy_fn, positions, cell, device)

        torch.testing.assert_close(
            explicit_virial,
            fd_virial,
            atol=1e-2,
            rtol=1e-2,
            msg="PME reciprocal virial does not match FD",
        )

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_pme_reciprocal_virial_symmetry(self, device):
        """PME reciprocal virial is symmetric for cubic system."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        positions, charges, cell = make_virial_cscl_system(2, device=device)
        alpha = torch.tensor([0.3], dtype=VIRIAL_DTYPE, device=device)

        result = pme_reciprocal_space(
            positions,
            charges,
            cell,
            alpha,
            mesh_dimensions=(16, 16, 16),
            compute_forces=True,
            compute_virial=True,
        )
        virial = result[2].squeeze(0)
        torch.testing.assert_close(
            virial,
            virial.T,
            atol=1e-6,
            rtol=1e-6,
            msg="PME reciprocal virial is not symmetric",
        )

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_pme_reciprocal_virial_dtype(self, device, dtype):
        """PME reciprocal virial dtype matches input."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        positions, charges, cell = make_virial_cscl_system(
            1, dtype=dtype, device=device
        )
        alpha = torch.tensor([0.3], dtype=dtype, device=device)

        result = pme_reciprocal_space(
            positions,
            charges,
            cell,
            alpha,
            mesh_dimensions=(8, 8, 8),
            compute_forces=True,
            compute_virial=True,
        )
        virial = result[2]
        assert virial.dtype == dtype


class TestPMEReciprocalVirialMeshConvergence:
    """PME virial converges with increasing mesh density."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_virial_mesh_convergence(self, device):
        """PME virial converges as mesh_dimensions increase."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        positions, charges, cell = make_virial_cscl_system(2, device=device)
        alpha = torch.tensor([0.3], dtype=VIRIAL_DTYPE, device=device)

        ref_result = pme_reciprocal_space(
            positions,
            charges,
            cell,
            alpha,
            mesh_dimensions=(32, 32, 32),
            compute_forces=True,
            compute_virial=True,
        )
        ref_virial = ref_result[2].squeeze(0)

        prev_err = float("inf")
        for mesh_size in [8, 16, 32]:
            result = pme_reciprocal_space(
                positions,
                charges,
                cell,
                alpha,
                mesh_dimensions=(mesh_size, mesh_size, mesh_size),
                compute_forces=True,
                compute_virial=True,
            )
            virial = result[2].squeeze(0)
            err = (virial - ref_virial).abs().max().item()
            assert err <= prev_err + 1e-10, (
                f"Virial did not converge: mesh={mesh_size}, err={err}, prev_err={prev_err}"
            )
            prev_err = err


class TestPMEReciprocalVirialSplineOrders:
    """PME virial with different spline orders."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    @pytest.mark.parametrize("spline_order", [3, 4, 5, 6])
    def test_virial_spline_orders(self, device, spline_order):
        """PME virial is finite and well-behaved for various spline orders."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        positions, charges, cell = make_virial_cscl_system(1, device=device)
        alpha = torch.tensor([0.3], dtype=VIRIAL_DTYPE, device=device)

        result = pme_reciprocal_space(
            positions,
            charges,
            cell,
            alpha,
            mesh_dimensions=(16, 16, 16),
            spline_order=spline_order,
            compute_forces=True,
            compute_virial=True,
        )
        virial = result[2]
        assert virial.shape == (1, 3, 3)
        assert torch.isfinite(virial).all(), (
            f"Non-finite virial for spline_order={spline_order}"
        )


class TestPMEReciprocalVirialBatch:
    """Batch PME virial matches single-system PME virial."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_batch_pme_reciprocal_virial_shape(self, device):
        """Batch PME reciprocal virial has shape (B, 3, 3)."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        positions, charges, cell, alpha, batch_idx, _, _, _, _, _ = (
            make_virial_batch_cscl_system(1, device=device)
        )

        result = pme_reciprocal_space(
            positions,
            charges,
            cell,
            alpha,
            mesh_dimensions=(8, 8, 8),
            batch_idx=batch_idx,
            compute_forces=True,
            compute_virial=True,
        )
        virial = result[2]
        assert virial.shape == (2, 3, 3)

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_batch_pme_reciprocal_virial_fd(self, device):
        """Batch PME reciprocal virial per-system matches single-system FD."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        positions, charges, cell, alpha, batch_idx, pos_s, q_s, cell_s, alpha_s, _ = (
            make_virial_batch_cscl_system(1, device=device)
        )
        mesh_dims = (8, 8, 8)

        def energy_fn(pos, c):
            return pme_reciprocal_space(
                pos,
                q_s,
                c,
                alpha_s,
                mesh_dimensions=mesh_dims,
                compute_forces=False,
            ).sum()

        fd_virial = fd_virial_full(energy_fn, pos_s, cell_s, device)

        batch_result = pme_reciprocal_space(
            positions,
            charges,
            cell,
            alpha,
            mesh_dimensions=mesh_dims,
            batch_idx=batch_idx,
            compute_forces=True,
            compute_virial=True,
        )
        batch_virial = batch_result[2]

        torch.testing.assert_close(
            batch_virial[0],
            fd_virial,
            atol=1e-2,
            rtol=1e-2,
            msg="Batch PME virial[0] does not match single-system FD",
        )
        torch.testing.assert_close(
            batch_virial[1],
            fd_virial,
            atol=1e-2,
            rtol=1e-2,
            msg="Batch PME virial[1] does not match single-system FD",
        )

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_batch_pme_reciprocal_virial_matches_single(self, device):
        """Batch PME reciprocal virial[i] matches single-system virial."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        positions, charges, cell, alpha, batch_idx, pos_s, q_s, cell_s, alpha_s, _ = (
            make_virial_batch_cscl_system(1, device=device)
        )

        single_result = pme_reciprocal_space(
            pos_s,
            q_s,
            cell_s,
            alpha_s,
            mesh_dimensions=(8, 8, 8),
            compute_forces=True,
            compute_virial=True,
        )
        single_virial = single_result[2]

        batch_result = pme_reciprocal_space(
            positions,
            charges,
            cell,
            alpha,
            mesh_dimensions=(8, 8, 8),
            batch_idx=batch_idx,
            compute_forces=True,
            compute_virial=True,
        )
        batch_virial = batch_result[2]

        torch.testing.assert_close(
            batch_virial[0:1],
            single_virial,
            atol=1e-5,
            rtol=1e-5,
            msg="Batch PME virial[0] != single virial",
        )
        torch.testing.assert_close(
            batch_virial[1:2],
            single_virial,
            atol=1e-5,
            rtol=1e-5,
            msg="Batch PME virial[1] != single virial",
        )


class TestFullPMEVirial:
    """Test full particle_mesh_ewald (real + reciprocal) virial."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_full_pme_virial_shape(self, device):
        """Full PME virial has shape (1, 3, 3)."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        positions, charges, cell = make_virial_cscl_system(2, device=device)
        alpha = torch.tensor([0.3], dtype=VIRIAL_DTYPE, device=device)
        cutoff = 6.0
        nl, nptr, us = get_virial_neighbor_data(positions, cell, cutoff)

        result = particle_mesh_ewald(
            positions,
            charges,
            cell,
            alpha=alpha,
            mesh_dimensions=(16, 16, 16),
            neighbor_list=nl,
            neighbor_ptr=nptr,
            neighbor_shifts=us,
            compute_forces=True,
            compute_virial=True,
        )
        virial = result[2]
        assert virial.shape == (1, 3, 3)

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_full_pme_virial_fd(self, device):
        """Full PME virial matches FD strain derivative."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        positions, charges, cell = make_virial_cscl_system(2, device=device)
        alpha = torch.tensor([0.3], dtype=VIRIAL_DTYPE, device=device)
        cutoff = 6.0
        mesh_dims = (16, 16, 16)
        pbc = torch.tensor([True, True, True], dtype=torch.bool, device=device)

        def energy_fn(pos, c):
            nl_new, np_new, us_new = cell_list(
                pos,
                cutoff,
                c.squeeze(0),
                pbc,
                return_neighbor_list=True,
            )
            return particle_mesh_ewald(
                pos,
                charges,
                c,
                alpha=alpha,
                mesh_dimensions=mesh_dims,
                neighbor_list=nl_new,
                neighbor_ptr=np_new,
                neighbor_shifts=us_new,
                compute_forces=False,
            ).sum()

        nl, nptr, us = get_virial_neighbor_data(positions, cell, cutoff)
        result = particle_mesh_ewald(
            positions,
            charges,
            cell,
            alpha=alpha,
            mesh_dimensions=mesh_dims,
            neighbor_list=nl,
            neighbor_ptr=nptr,
            neighbor_shifts=us,
            compute_forces=True,
            compute_virial=True,
        )
        explicit_virial = result[2].squeeze(0)
        fd_virial = fd_virial_full(energy_fn, positions, cell, device)

        torch.testing.assert_close(
            explicit_virial,
            fd_virial,
            atol=1e-2,
            rtol=1e-2,
            msg="Full PME virial does not match FD",
        )

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_full_pme_virial_sum_of_components(self, device):
        """Full PME virial = real-space virial + reciprocal virial."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        positions, charges, cell = make_virial_cscl_system(2, device=device)
        alpha = torch.tensor([0.3], dtype=VIRIAL_DTYPE, device=device)
        cutoff = 6.0
        mesh_dims = (16, 16, 16)
        nl, nptr, us = get_virial_neighbor_data(positions, cell, cutoff)

        rs_result = ewald_real_space(
            positions,
            charges,
            cell,
            alpha,
            neighbor_list=nl,
            neighbor_ptr=nptr,
            neighbor_shifts=us,
            compute_forces=True,
            compute_virial=True,
        )
        real_virial = rs_result[2]

        rec_result = pme_reciprocal_space(
            positions,
            charges,
            cell,
            alpha,
            mesh_dimensions=mesh_dims,
            compute_forces=True,
            compute_virial=True,
        )
        recip_virial = rec_result[2]

        total_result = particle_mesh_ewald(
            positions,
            charges,
            cell,
            alpha=alpha,
            mesh_dimensions=mesh_dims,
            neighbor_list=nl,
            neighbor_ptr=nptr,
            neighbor_shifts=us,
            compute_forces=True,
            compute_virial=True,
        )
        total_virial = total_result[2]

        torch.testing.assert_close(
            total_virial,
            real_virial + recip_virial,
            atol=1e-6,
            rtol=1e-6,
            msg="Full PME virial != real + reciprocal virial",
        )

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_full_pme_virial_without_forces(self, device):
        """particle_mesh_ewald with compute_forces=False + compute_virial=True."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        positions, charges, cell = make_virial_cscl_system(2, device=device)
        alpha = torch.tensor([0.3], dtype=VIRIAL_DTYPE, device=device)
        cutoff = 6.0
        nl, nptr, us = get_virial_neighbor_data(positions, cell, cutoff)

        result = particle_mesh_ewald(
            positions,
            charges,
            cell,
            alpha=alpha,
            mesh_dimensions=(16, 16, 16),
            neighbor_list=nl,
            neighbor_ptr=nptr,
            neighbor_shifts=us,
            compute_forces=False,
            compute_virial=True,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        energies, virial = result
        assert virial.shape == (1, 3, 3)

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_full_pme_virial_with_charge_gradients(self, device):
        """particle_mesh_ewald with forces + charge_grads + virial returns 4-tuple."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        positions, charges, cell = make_virial_cscl_system(2, device=device)
        alpha = torch.tensor([0.3], dtype=VIRIAL_DTYPE, device=device)
        cutoff = 6.0
        nl, nptr, us = get_virial_neighbor_data(positions, cell, cutoff)

        result = particle_mesh_ewald(
            positions,
            charges,
            cell,
            alpha=alpha,
            mesh_dimensions=(16, 16, 16),
            neighbor_list=nl,
            neighbor_ptr=nptr,
            neighbor_shifts=us,
            compute_forces=True,
            compute_charge_gradients=True,
            compute_virial=True,
        )
        assert isinstance(result, tuple)
        assert len(result) == 4
        energies, forces, charge_grads, virial = result
        assert virial.shape == (1, 3, 3)
        assert charge_grads.shape == (positions.shape[0],)


class TestPMEVirialNonCubicCells:
    """PME virial with non-cubic simulation cells, validated against FD."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_orthorhombic_cell_pme_virial_fd(self, device):
        """PME reciprocal virial matches FD on orthorhombic cell."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        cell = torch.tensor(
            [[[8.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 12.0]]],
            dtype=VIRIAL_DTYPE,
            device=device,
        )
        positions = torch.tensor(
            [[2.0, 5.0, 6.0], [6.0, 5.0, 6.0]],
            dtype=VIRIAL_DTYPE,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0], dtype=VIRIAL_DTYPE, device=device)
        alpha = torch.tensor([0.3], dtype=VIRIAL_DTYPE, device=device)
        mesh_dims = (8, 10, 12)

        def energy_fn(pos, c):
            return pme_reciprocal_space(
                pos,
                charges,
                c,
                alpha,
                mesh_dimensions=mesh_dims,
                compute_forces=False,
            ).sum()

        result = pme_reciprocal_space(
            positions,
            charges,
            cell,
            alpha,
            mesh_dimensions=mesh_dims,
            compute_forces=True,
            compute_virial=True,
        )
        virial = result[2]
        assert virial.shape == (1, 3, 3)
        assert torch.isfinite(virial).all()

        explicit_virial = virial.squeeze(0)
        fd_virial = fd_virial_full(energy_fn, positions, cell, device)
        torch.testing.assert_close(
            explicit_virial,
            fd_virial,
            atol=1e-2,
            rtol=1e-2,
            msg="Orthorhombic PME reciprocal virial does not match FD",
        )

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_triclinic_cell_pme_virial_fd(self, device):
        """PME reciprocal virial matches FD on triclinic cell."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        cell = torch.tensor(
            [[[10.0, 0.0, 0.0], [2.0, 10.0, 0.0], [1.0, 1.0, 10.0]]],
            dtype=VIRIAL_DTYPE,
            device=device,
        )
        positions = torch.tensor(
            [[2.0, 5.0, 5.0], [7.0, 5.0, 5.0]],
            dtype=VIRIAL_DTYPE,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0], dtype=VIRIAL_DTYPE, device=device)
        alpha = torch.tensor([0.3], dtype=VIRIAL_DTYPE, device=device)
        mesh_dims = (10, 10, 10)

        def energy_fn(pos, c):
            return pme_reciprocal_space(
                pos,
                charges,
                c,
                alpha,
                mesh_dimensions=mesh_dims,
                compute_forces=False,
            ).sum()

        result = pme_reciprocal_space(
            positions,
            charges,
            cell,
            alpha,
            mesh_dimensions=mesh_dims,
            compute_forces=True,
            compute_virial=True,
        )
        virial = result[2]
        assert virial.shape == (1, 3, 3)
        assert torch.isfinite(virial).all()

        explicit_virial = virial.squeeze(0)
        fd_virial = fd_virial_full(energy_fn, positions, cell, device)
        torch.testing.assert_close(
            explicit_virial,
            fd_virial,
            atol=1e-2,
            rtol=1e-2,
            msg="Triclinic PME reciprocal virial does not match FD",
        )


class TestPMEVirialPrecomputedKVectors:
    """Verify precomputed k_vectors/k_squared produce identical virial."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_precomputed_kvectors_virial_matches(self, device):
        """PME virial with precomputed k-vectors matches auto-generated."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        positions, charges, cell = make_virial_cscl_system(2, device=device)
        alpha = torch.tensor([0.3], dtype=VIRIAL_DTYPE, device=device)
        mesh_dims = (16, 16, 16)

        result_auto = pme_reciprocal_space(
            positions,
            charges,
            cell,
            alpha,
            mesh_dimensions=mesh_dims,
            compute_forces=True,
            compute_virial=True,
        )
        virial_auto = result_auto[2]

        k_vectors, k_squared = generate_k_vectors_pme(cell, mesh_dims)
        result_pre = pme_reciprocal_space(
            positions,
            charges,
            cell,
            alpha,
            mesh_dimensions=mesh_dims,
            k_vectors=k_vectors,
            k_squared=k_squared,
            compute_forces=True,
            compute_virial=True,
        )
        virial_pre = result_pre[2]

        torch.testing.assert_close(
            virial_auto,
            virial_pre,
            atol=1e-6,
            rtol=1e-6,
            msg="PME virial with precomputed k-vectors != auto-generated",
        )


class TestPMEVirialCrystalSystems:
    """PME virial FD tests over multiple crystal structures."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    @pytest.mark.parametrize(
        "crystal_factory",
        [
            pytest.param(lambda: create_cscl_supercell(1), id="cscl"),
            pytest.param(lambda: create_wurtzite_system(1), id="wurtzite"),
            pytest.param(lambda: create_zincblende_system(1), id="zincblende"),
        ],
    )
    def test_full_pme_virial_fd_crystals(self, device, crystal_factory):
        """Full PME virial matches FD for various crystal systems."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        crystal = crystal_factory()
        positions = torch.tensor(crystal.positions, dtype=VIRIAL_DTYPE, device=device)
        charges = torch.tensor(crystal.charges, dtype=VIRIAL_DTYPE, device=device)
        cell = torch.tensor(crystal.cell, dtype=VIRIAL_DTYPE, device=device).unsqueeze(
            0
        )
        alpha = torch.tensor([0.3], dtype=VIRIAL_DTYPE, device=device)
        cutoff = 6.0
        mesh_dims = (16, 16, 16)
        pbc = torch.tensor([True, True, True], dtype=torch.bool, device=device)

        def energy_fn(pos, c):
            nl, nptr, us = cell_list(
                pos,
                cutoff,
                c.squeeze(0),
                pbc,
                return_neighbor_list=True,
            )
            return particle_mesh_ewald(
                pos,
                charges,
                c,
                alpha=alpha,
                mesh_dimensions=mesh_dims,
                neighbor_list=nl,
                neighbor_ptr=nptr,
                neighbor_shifts=us,
                compute_forces=False,
            ).sum()

        nl, nptr, us = get_virial_neighbor_data(positions, cell, cutoff)
        result = particle_mesh_ewald(
            positions,
            charges,
            cell,
            alpha=alpha,
            mesh_dimensions=mesh_dims,
            neighbor_list=nl,
            neighbor_ptr=nptr,
            neighbor_shifts=us,
            compute_forces=True,
            compute_virial=True,
        )
        explicit_virial = result[2].squeeze(0)
        fd_virial = fd_virial_full(energy_fn, positions, cell, device)

        torch.testing.assert_close(
            explicit_virial,
            fd_virial,
            atol=1e-2,
            rtol=1e-2,
            msg="Full PME virial does not match FD",
        )


class TestPMENonNeutralVirial:
    """Virial FD tests for non-neutral (Q != 0) systems."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_pme_reciprocal_virial_fd_non_neutral(self, device):
        """PME reciprocal virial matches FD for a non-neutral system."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        positions, charges, cell = make_non_neutral_system(device)
        alpha = torch.tensor([0.3], dtype=VIRIAL_DTYPE, device=device)
        mesh_dims = (16, 16, 16)

        def energy_fn(pos, c):
            return pme_reciprocal_space(
                pos,
                charges,
                c,
                alpha,
                mesh_dimensions=mesh_dims,
                compute_forces=False,
            ).sum()

        result = pme_reciprocal_space(
            positions,
            charges,
            cell,
            alpha,
            mesh_dimensions=mesh_dims,
            compute_forces=True,
            compute_virial=True,
        )
        explicit_virial = result[2].squeeze(0)
        fd_virial = fd_virial_full(energy_fn, positions, cell, device)

        torch.testing.assert_close(
            explicit_virial,
            fd_virial,
            atol=2e-2,
            rtol=2e-2,
            msg="PME reciprocal virial does not match FD for non-neutral system",
        )


class TestPMEDifferentiableVirial:
    """Stress-loss gradients through PME virial path."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_pme_stress_loss_backprop_enabled(self, device, dtype):
        """PME stress loss contributes gradients when compute_virial=True."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        positions, charges, cell = make_virial_cscl_system(
            1, dtype=dtype, device=device
        )
        charges = charges.clone().requires_grad_(True)
        alpha = torch.tensor([0.3], dtype=dtype, device=device)
        nl, nptr, us = get_virial_neighbor_data(positions, cell, cutoff=6.0)

        _, _, virial = particle_mesh_ewald(
            positions,
            charges,
            cell,
            alpha=alpha,
            mesh_dimensions=(16, 16, 16),
            neighbor_list=nl,
            neighbor_ptr=nptr,
            neighbor_shifts=us,
            compute_forces=True,
            compute_virial=True,
        )

        stress_loss = virial.pow(2).sum()
        stress_loss.backward()

        assert charges.grad is not None
        assert torch.isfinite(charges.grad).all()
        assert charges.grad.abs().sum() > 0

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_pme_virial_fd_charges(self, device):
        """PME virial backward gives FD-correct charge gradients."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        positions, charges, cell = make_virial_cscl_system(1, device=device)
        alpha = torch.tensor([0.3], dtype=VIRIAL_DTYPE, device=device)
        nl, nptr, us = get_virial_neighbor_data(positions, cell, cutoff=6.0)

        def virial_sum(chg):
            _, _, v = particle_mesh_ewald(
                positions,
                chg,
                cell,
                alpha=alpha,
                mesh_dimensions=(16, 16, 16),
                neighbor_list=nl,
                neighbor_ptr=nptr,
                neighbor_shifts=us,
                compute_forces=True,
                compute_virial=True,
            )
            return v.sum()

        chg = charges.clone().requires_grad_(True)
        loss = virial_sum(chg)
        loss.backward()
        ad_grad = chg.grad.clone()

        h = 1e-5
        for i in range(min(4, len(charges))):
            cp = charges.clone()
            cp[i] += h
            cm = charges.clone()
            cm[i] -= h
            fd = (virial_sum(cp).item() - virial_sum(cm).item()) / (2 * h)
            rel = abs(ad_grad[i].item() - fd) / (abs(fd) + 1e-30)
            assert rel < 0.02, (
                f"atom {i}: AD={ad_grad[i].item():.8e}, FD={fd:.8e}, rel={rel:.2e}"
            )


def _torchpme_pme_energy(positions, charges, cell, alpha, mesh_spacing, device):
    """Compute PME reciprocal energy via torchpme PMECalculator."""
    import math

    smearing = 1.0 / (math.sqrt(2.0) * alpha)
    potential = CoulombPotential(smearing=smearing).to(
        device=device, dtype=VIRIAL_DTYPE
    )
    calculator = PMECalculator(
        potential=potential,
        mesh_spacing=mesh_spacing,
        interpolation_nodes=4,
        full_neighbor_list=True,
        prefactor=1.0,
    ).to(device=device, dtype=VIRIAL_DTYPE)
    charges_col = charges.unsqueeze(1)
    cell_2d = cell.squeeze(0) if cell.dim() == 3 else cell
    potentials = calculator._compute_kspace(charges_col, cell_2d, positions)
    return (charges_col * potentials).flatten().sum()


@pytest.mark.skipif(not HAS_TORCHPME, reason="torchpme not installed")
class TestPMEVirialTorchPMEParity:
    """Cross-validate PME virial against torchpme via FD on torchpme energies."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_pme_reciprocal_virial_vs_torchpme_fd(self, device):
        """PME reciprocal virial matches FD of torchpme PME reciprocal energy."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        positions, charges, cell = make_virial_cscl_system(2, device=device)
        alpha_val = 0.3
        alpha = torch.tensor([alpha_val], dtype=VIRIAL_DTYPE, device=device)
        mesh_spacing = 1.0

        def torchpme_energy_fn(pos, c):
            return _torchpme_pme_energy(
                pos,
                charges,
                c,
                alpha_val,
                mesh_spacing,
                device,
            )

        fd_virial = fd_virial_full(torchpme_energy_fn, positions, cell, device, h=1e-5)

        result = pme_reciprocal_space(
            positions,
            charges,
            cell,
            alpha,
            mesh_spacing=mesh_spacing,
            compute_forces=True,
            compute_virial=True,
        )
        our_virial = result[2].squeeze(0)

        torch.testing.assert_close(
            our_virial,
            fd_virial,
            atol=5e-2,
            rtol=5e-2,
            msg="PME reciprocal virial does not match torchpme FD virial",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
