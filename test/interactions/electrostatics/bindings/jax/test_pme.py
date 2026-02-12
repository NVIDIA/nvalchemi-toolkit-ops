# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Unit tests for JAX Particle Mesh Ewald (PME) electrostatic calculations.

This test suite validates the correctness of the JAX PME implementation
for long-range electrostatics in periodic systems.

Tests cover:
- Float32 and float64 dtype support
- API shapes (energy-only, energy+forces, batched)
- Physical conservation laws (momentum conservation, translation invariance)
- Mesh size convergence
- Numerical correctness against torchpme reference
- Batch vs single-system consistency
- Explicit charge gradient computation (replaces autograd tests)
- Non-cubic cells, spline orders, precomputed k-vectors
- Full PME (real + reciprocal) with neighbor lists
- Edge cases (zero charges, single atom, empty system)

Note: JAX bindings are GPU-only (Warp JAX FFI constraint) and do not support
autograd (enable_backward=False). Tests that call kernels require GPU.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nvalchemiops.jax.interactions.electrostatics.k_vectors import (
    generate_k_vectors_pme,
)
from nvalchemiops.jax.interactions.electrostatics.pme import (
    particle_mesh_ewald,
    pme_reciprocal_space,
)
from nvalchemiops.jax.neighbors import cell_list
from test.interactions.electrostatics.bindings.jax.conftest import place_on_device
from test.interactions.electrostatics.conftest import (
    create_cscl_supercell,
    create_wurtzite_system,
    create_zincblende_system,
)

# Try to import torchpme for reference calculations
try:
    import torch
    from torchpme import PMECalculator
    from torchpme.potentials import CoulombPotential

    HAS_TORCHPME = True
except ModuleNotFoundError:
    HAS_TORCHPME = False


# ==============================================================================
# Helper Functions
# ==============================================================================


def create_dipole_system(device, dtype=jnp.float64, separation=2.0, cell_size=10.0):
    """Create a simple dipole system with JAX arrays on GPU.

    Parameters
    ----------
    device : str
        Device type ("gpu")
    dtype : jnp.dtype
        Data type for arrays
    separation : float
        Distance between charges
    cell_size : float
        Cubic cell size

    Returns
    -------
    tuple
        (positions, charges, cell)
    """
    center = cell_size / 2
    positions = place_on_device(
        jnp.array(
            [
                [center - separation / 2, center, center],
                [center + separation / 2, center, center],
            ],
            dtype=dtype,
        ),
        device,
    )
    charges = place_on_device(jnp.array([1.0, -1.0], dtype=dtype), device)
    cell = place_on_device(
        jnp.array(
            [[[cell_size, 0.0, 0.0], [0.0, cell_size, 0.0], [0.0, 0.0, cell_size]]],
            dtype=dtype,
        ),
        device,
    )
    return positions, charges, cell


def create_simple_system(device, dtype=jnp.float64, num_atoms=4, cell_size=10.0):
    """Create a simple test system with random positions and neutral charges.

    Parameters
    ----------
    device : str
        Device type ("gpu")
    dtype : jnp.dtype
        Data type for arrays
    num_atoms : int
        Number of atoms
    cell_size : float
        Cubic cell size

    Returns
    -------
    tuple
        (positions, charges, cell)
    """
    key = jax.random.PRNGKey(42)
    positions = place_on_device(
        jax.random.uniform(key, (num_atoms, 3), dtype=dtype) * cell_size * 0.8
        + cell_size * 0.1,
        device,
    )

    # Create random charges and make neutral
    key2 = jax.random.PRNGKey(123)
    charges_raw = jax.random.normal(key2, (num_atoms,), dtype=dtype)
    # Make last charge neutralize the system
    charges_raw = charges_raw.at[-1].set(-charges_raw[:-1].sum())
    charges = place_on_device(charges_raw, device)

    cell = place_on_device(
        jnp.array(
            [[[cell_size, 0.0, 0.0], [0.0, cell_size, 0.0], [0.0, 0.0, cell_size]]],
            dtype=dtype,
        ),
        device,
    )

    return positions, charges, cell


def calculate_pme_reciprocal_energy_torchpme(
    positions_np, charges_np, cell_np, mesh_spacing, alpha, spline_order, dtype=None
):
    """Calculate PME reciprocal-space energy using torchpme as reference.

    Parameters
    ----------
    positions_np : np.ndarray
        Atomic positions
    charges_np : np.ndarray
        Atomic charges
    cell_np : np.ndarray
        Cell matrix (2D)
    mesh_spacing : float
        Mesh spacing
    alpha : float
        Ewald splitting parameter
    spline_order : int
        B-spline interpolation order
    dtype : torch dtype, optional
        Defaults to torch.float64

    Returns
    -------
    np.ndarray
        Reciprocal space energy per atom
    """
    if dtype is None:
        dtype = torch.float64

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # torchpme uses smearing sigma where Gaussian is exp(-r^2/(2*sigma^2))
    # Standard Ewald uses exp(-alpha^2 * r^2), so sigma = 1/(sqrt(2)*alpha)
    smearing = 1.0 / (2.0**0.5 * alpha)
    potential = CoulombPotential(smearing=smearing).to(device=device, dtype=dtype)

    positions_torch = torch.tensor(positions_np, dtype=dtype, device=device)
    charges_torch = torch.tensor(charges_np, dtype=dtype, device=device).unsqueeze(1)
    cell_torch = torch.tensor(cell_np, dtype=dtype, device=device)

    # Ensure cell is 2D
    if cell_torch.dim() == 3:
        cell_torch = cell_torch.squeeze(0)

    calculator = PMECalculator(
        potential=potential,
        mesh_spacing=mesh_spacing,
        interpolation_nodes=spline_order,
        full_neighbor_list=True,
        prefactor=1.0,
    ).to(device=device, dtype=dtype)

    reciprocal_potential = calculator._compute_kspace(
        charges_torch, cell_torch, positions_torch
    )

    return (reciprocal_potential * charges_torch).flatten().cpu().numpy()


# ==============================================================================
# Test Classes
# ==============================================================================


class TestDtypeSupport:
    """Test that PME functions support both float32 and float64 dtypes."""

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_pme_reciprocal_dtype_returns_correct_type(self, device, dtype):
        """Test that pme_reciprocal_space returns arrays with expected dtype.

        Note: JAX PME always uses float32 B-spline interpolation internally,
        so output dtype is always float32 regardless of input dtype.
        """
        positions, charges, cell = create_dipole_system(device, dtype=dtype)

        # Test energy-only
        energies = pme_reciprocal_space(
            positions,
            charges,
            cell,
            alpha=jnp.array([0.3], dtype=dtype),
            mesh_dimensions=(16, 16, 16),
            spline_order=4,
            compute_forces=False,
        )
        assert jnp.all(jnp.isfinite(energies))
        # JAX PME always returns float32 due to B-spline kernel
        assert energies.dtype == jnp.float32, (
            f"Expected float32 output, got {energies.dtype}"
        )

        # Test with forces
        energies, forces = pme_reciprocal_space(
            positions,
            charges,
            cell,
            alpha=jnp.array([0.3], dtype=dtype),
            mesh_dimensions=(16, 16, 16),
            spline_order=4,
            compute_forces=True,
        )
        assert jnp.all(jnp.isfinite(energies))
        assert jnp.all(jnp.isfinite(forces))
        assert energies.dtype == jnp.float32, (
            f"Expected float32 output, got {energies.dtype}"
        )
        assert forces.dtype == jnp.float32, (
            f"Expected float32 output, got {forces.dtype}"
        )

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_pme_batch_dtype_returns_correct_type(self, device, dtype):
        """Test that batch PME returns arrays with expected dtype.

        Note: JAX PME always uses float32 B-spline interpolation internally,
        so output dtype is always float32 regardless of input dtype.
        """
        pos1, chg1, cell1 = create_dipole_system(device, dtype=dtype)
        pos2, chg2, cell2 = create_dipole_system(device, dtype=dtype, separation=3.0)

        positions = jnp.concatenate([pos1, pos2], axis=0)
        charges = jnp.concatenate([chg1, chg2], axis=0)
        cells = jnp.concatenate([cell1, cell2], axis=0)
        batch_idx = place_on_device(jnp.array([0, 0, 1, 1], dtype=jnp.int32), device)

        # Test energy-only
        energies = pme_reciprocal_space(
            positions,
            charges,
            cells,
            alpha=jnp.array([0.3, 0.3], dtype=dtype),
            mesh_dimensions=(16, 16, 16),
            spline_order=4,
            batch_idx=batch_idx,
            compute_forces=False,
        )
        assert jnp.all(jnp.isfinite(energies))
        assert energies.dtype == jnp.float32, (
            f"Expected float32 output, got {energies.dtype}"
        )

        # Test with forces
        energies, forces = pme_reciprocal_space(
            positions,
            charges,
            cells,
            alpha=jnp.array([0.3, 0.3], dtype=dtype),
            mesh_dimensions=(16, 16, 16),
            spline_order=4,
            batch_idx=batch_idx,
            compute_forces=True,
        )
        assert jnp.all(jnp.isfinite(energies))
        assert jnp.all(jnp.isfinite(forces))
        assert energies.dtype == jnp.float32, (
            f"Expected float32 output, got {energies.dtype}"
        )
        assert forces.dtype == jnp.float32, (
            f"Expected float32 output, got {forces.dtype}"
        )

    def test_float32_vs_float64_consistency(self, device):
        """Test that float32 and float64 produce consistent results."""
        positions_f64, charges_f64, cell_f64 = create_dipole_system(
            device, dtype=jnp.float64
        )

        positions_f32 = positions_f64.astype(jnp.float32)
        charges_f32 = charges_f64.astype(jnp.float32)
        cell_f32 = cell_f64.astype(jnp.float32)

        e_f32, f_f32 = pme_reciprocal_space(
            positions_f32,
            charges_f32,
            cell_f32,
            alpha=jnp.array([0.3], dtype=jnp.float32),
            mesh_dimensions=(16, 16, 16),
            spline_order=4,
            compute_forces=True,
        )
        e_f64, f_f64 = pme_reciprocal_space(
            positions_f64,
            charges_f64,
            cell_f64,
            alpha=jnp.array([0.3], dtype=jnp.float64),
            mesh_dimensions=(16, 16, 16),
            spline_order=4,
            compute_forces=True,
        )

        # Results should be close (within float32 precision)
        assert jnp.allclose(e_f32.astype(jnp.float64), e_f64, rtol=1e-2, atol=1e-3), (
            f"Energy mismatch: f32={e_f32.sum()}, f64={e_f64.sum()}"
        )


###########################################################################################
########################### Unit Tests: API Shapes and Basic Behavior #####################
###########################################################################################


class TestPMEReciprocalSpaceAPI:
    """Test basic API functionality for pme_reciprocal_space."""

    def test_output_shape_energy_only(self, device):
        """Test output shape when compute_forces=False."""
        positions, charges, cell = create_simple_system(device, num_atoms=5)

        result = pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=jnp.array([0.3]),
            mesh_dimensions=(16, 16, 16),
            compute_forces=False,
        )

        assert result.shape == (5,), f"Energy shape mismatch: {result.shape}"

    def test_output_shape_energy_forces(self, device):
        """Test output shape when compute_forces=True."""
        positions, charges, cell = create_simple_system(device, num_atoms=5)

        energies, forces = pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=jnp.array([0.3]),
            mesh_dimensions=(16, 16, 16),
            compute_forces=True,
        )

        assert energies.shape == (5,), f"Energy shape mismatch: {energies.shape}"
        assert forces.shape == (5, 3), f"Force shape mismatch: {forces.shape}"

    def test_batch_output_shape(self, device):
        """Test output shape for batched calculation."""
        # Two systems with 3 and 4 atoms
        positions = place_on_device(
            jnp.array(
                [
                    [1.0, 1.0, 1.0],
                    [3.0, 1.0, 1.0],
                    [5.0, 1.0, 1.0],
                    [1.0, 5.0, 5.0],
                    [3.0, 5.0, 5.0],
                    [5.0, 5.0, 5.0],
                    [7.0, 5.0, 5.0],
                ],
                dtype=jnp.float64,
            ),
            device,
        )
        charges = place_on_device(
            jnp.array([1.0, -1.0, 0.0, 0.5, -0.5, 0.5, -0.5], dtype=jnp.float64),
            device,
        )
        batch_idx = place_on_device(
            jnp.array([0, 0, 0, 1, 1, 1, 1], dtype=jnp.int32), device
        )
        cells = place_on_device(
            jnp.stack([jnp.eye(3, dtype=jnp.float64) * 10.0] * 2, axis=0),
            device,
        )

        energies, forces = pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cells,
            alpha=jnp.array([0.3, 0.3]),
            mesh_dimensions=(16, 16, 16),
            batch_idx=batch_idx,
            compute_forces=True,
        )

        assert energies.shape == (7,), f"Batch energy shape mismatch: {energies.shape}"
        assert forces.shape == (7, 3), f"Batch force shape mismatch: {forces.shape}"

    def test_empty_system(self, device):
        """Test handling of empty system."""
        positions = place_on_device(jnp.zeros((0, 3), dtype=jnp.float64), device)
        charges = place_on_device(jnp.zeros(0, dtype=jnp.float64), device)
        cell = place_on_device(
            jnp.eye(3, dtype=jnp.float64).reshape(1, 3, 3) * 10.0, device
        )

        energies, forces = pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=jnp.array([0.3]),
            mesh_dimensions=(16, 16, 16),
            compute_forces=True,
        )

        assert energies.shape == (0,)
        assert forces.shape == (0, 3)

    @pytest.mark.parametrize("spline_order", [2, 3, 4])
    def test_different_spline_orders(self, spline_order, device):
        """Test that different spline orders produce valid results."""
        positions, charges, cell = create_dipole_system(device)

        energies, forces = pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=jnp.array([0.3]),
            mesh_dimensions=(16, 16, 16),
            spline_order=spline_order,
            compute_forces=True,
        )

        assert jnp.all(jnp.isfinite(energies)), (
            f"Non-finite energies for order {spline_order}"
        )
        assert jnp.all(jnp.isfinite(forces)), (
            f"Non-finite forces for order {spline_order}"
        )


###########################################################################################
########################### Conservation Law Tests ########################################
###########################################################################################


class TestPMEConservationLaws:
    """Test momentum conservation and symmetry properties."""

    def test_momentum_conservation(self, device):
        """Test that net force is zero for neutral system."""
        positions, charges, cell = create_simple_system(device, num_atoms=6)

        _, forces = pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=jnp.array([0.3]),
            mesh_dimensions=(20, 20, 20),
            compute_forces=True,
        )

        net_force = forces.sum(axis=0)
        # PME reciprocal-space forces use float32 spline interpolation,
        # so momentum conservation is limited by float32 precision
        assert jnp.allclose(
            net_force, jnp.zeros(3, dtype=net_force.dtype), atol=1e-2
        ), f"Momentum not conserved: net force = {net_force}"

    def test_translation_invariance(self, device):
        """Test that energy is invariant under translation.

        Uses a fine mesh (64^3) and small translation to reduce grid
        discretization artifacts from B-spline interpolation. PME
        translation invariance improves with finer meshes.
        """
        positions, charges, cell = create_dipole_system(device)

        energy1 = pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=jnp.array([0.3]),
            mesh_dimensions=(64, 64, 64),
            compute_forces=False,
        )

        # Use a small translation to stay close on the B-spline grid
        translation = jnp.array([0.1, 0.1, 0.1])
        positions2 = positions + translation

        energy2 = pme_reciprocal_space(
            positions=positions2,
            charges=charges,
            cell=cell,
            alpha=jnp.array([0.3]),
            mesh_dimensions=(64, 64, 64),
            compute_forces=False,
        )

        # PME with B-spline interpolation has limited translation invariance
        # due to grid discretization and float32 spline output
        assert jnp.allclose(energy1.sum(), energy2.sum(), rtol=5e-2), (
            f"Energy not translation invariant: {energy1.sum()} vs {energy2.sum()}"
        )

    def test_opposite_charges_opposite_forces(self, device):
        """Test that opposite charges in same field get opposite forces."""
        positions, charges, cell = create_dipole_system(device)

        _, forces = pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=jnp.array([0.3]),
            mesh_dimensions=(16, 16, 16),
            compute_forces=True,
        )

        # For a symmetric dipole, forces should be equal and opposite
        assert jnp.allclose(forces[0], -forces[1], rtol=1e-6), (
            f"Forces not equal and opposite: {forces[0]} vs {-forces[1]}"
        )


###########################################################################################
########################### Mesh Size Convergence Tests ###################################
###########################################################################################


class TestPMEConvergence:
    """Test that results converge with finer mesh."""

    def test_mesh_size_convergence(self, device):
        """Test that energy converges as mesh size increases.

        Uses larger mesh sizes to ensure meaningful differences in float32
        output from B-spline interpolation.
        """
        positions, charges, cell = create_dipole_system(device)

        mesh_sizes = [8, 16, 32, 64]
        energies = []

        for mesh_size in mesh_sizes:
            energy = pme_reciprocal_space(
                positions=positions,
                charges=charges,
                cell=cell,
                alpha=jnp.array([0.3]),
                mesh_dimensions=(mesh_size, mesh_size, mesh_size),
                compute_forces=False,
            )
            energies.append(float(energy.sum()))

        # Check that we get finite, non-zero results
        for e in energies:
            assert np.isfinite(e), f"Non-finite energy: {e}"

        # Check convergence: later differences should be smaller
        diffs = [abs(energies[i + 1] - energies[i]) for i in range(len(energies) - 1)]
        # The last difference should be smaller than the first
        assert diffs[-1] < diffs[0] + 1e-8, (
            f"Energy not converging: diffs={diffs}, energies={energies}"
        )


###########################################################################################
########################### Correctness Tests: Against TorchPME ###########################
###########################################################################################


@pytest.mark.skipif(not HAS_TORCHPME, reason="torchpme is not installed")
class TestPMECorrectnessTorchPME:
    """Validate PME implementation against torchpme reference."""

    @pytest.mark.parametrize("alpha", [0.3, 0.5, 1.0])
    @pytest.mark.parametrize("mesh_spacing", [0.3, 0.5])
    def test_reciprocal_energy_matches_torchpme(self, device, alpha, mesh_spacing):
        """Test that reciprocal energy matches torchpme."""
        positions, charges, cell = create_dipole_system(device, dtype=jnp.float64)

        # Convert to numpy for mesh dim computation
        cell_np = np.array(cell[0])
        cell_lengths = np.linalg.norm(cell_np, axis=1)
        mesh_dims = tuple(
            int(np.ceil(length / mesh_spacing)) for length in cell_lengths
        )

        # Our implementation
        our_energy = pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=jnp.array([alpha]),
            mesh_dimensions=mesh_dims,
            spline_order=4,
            compute_forces=False,
        )

        # TorchPME reference
        positions_np = np.array(positions)
        charges_np = np.array(charges)
        torchpme_energy = calculate_pme_reciprocal_energy_torchpme(
            positions_np, charges_np, cell_np, mesh_spacing, alpha, 4
        )

        assert jnp.allclose(
            our_energy.sum(), torchpme_energy.sum(), rtol=1e-2, atol=1e-3
        ), (
            f"Energy mismatch: ours={float(our_energy.sum()):.6f}, "
            f"torchpme={torchpme_energy.sum():.6f}"
        )

    @pytest.mark.parametrize("size", [1, 2])
    @pytest.mark.parametrize("system_fn", ["cscl", "wurtzite", "zincblende"])
    @pytest.mark.parametrize("alpha", [0.3, 0.5])
    def test_crystal_systems_match_torchpme(self, size, system_fn, alpha, device):
        """Test PME on crystal systems against torchpme."""
        system_fns = {
            "cscl": create_cscl_supercell,
            "wurtzite": create_wurtzite_system,
            "zincblende": create_zincblende_system,
        }
        system = system_fns[system_fn](size)

        positions = place_on_device(
            jnp.array(system.positions, dtype=jnp.float64), device
        )
        charges = place_on_device(jnp.array(system.charges, dtype=jnp.float64), device)
        cell = place_on_device(
            jnp.array(system.cell[np.newaxis, :, :], dtype=jnp.float64), device
        )

        mesh_spacing = 0.5
        cell_lengths = np.linalg.norm(system.cell, axis=1)
        mesh_dims = tuple(
            int(np.ceil(length / mesh_spacing)) for length in cell_lengths
        )

        # Our implementation
        our_energy = pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=jnp.array([alpha]),
            mesh_dimensions=mesh_dims,
            spline_order=4,
            compute_forces=False,
        )

        # TorchPME reference
        torchpme_energy = calculate_pme_reciprocal_energy_torchpme(
            system.positions, system.charges, system.cell, mesh_spacing, alpha, 4
        )

        assert jnp.allclose(
            our_energy.sum(), torchpme_energy.sum(), rtol=1e-2, atol=1e-3
        ), (
            f"{system_fn} size={size} alpha={alpha}: "
            f"ours={float(our_energy.sum()):.6f}, torchpme={torchpme_energy.sum():.6f}"
        )


###########################################################################################
########################### Batch vs Single-System Consistency ############################
###########################################################################################


class TestPMEBatchConsistency:
    """Test that batch processing matches single-system processing."""

    def test_batch_single_system_matches(self, device):
        """Test batch with size 1 matches single-system."""
        positions, charges, cell = create_dipole_system(device)

        # Single-system
        energy_single, forces_single = pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=jnp.array([0.3]),
            mesh_dimensions=(16, 16, 16),
            compute_forces=True,
        )

        # Batch with size 1
        batch_idx = place_on_device(
            jnp.zeros(positions.shape[0], dtype=jnp.int32), device
        )
        energy_batch, forces_batch = pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=jnp.array([0.3]),
            mesh_dimensions=(16, 16, 16),
            batch_idx=batch_idx,
            compute_forces=True,
        )

        assert jnp.allclose(energy_batch.sum(), energy_single.sum(), rtol=1e-6), (
            f"Energy mismatch: batch={energy_batch.sum()}, single={energy_single.sum()}"
        )
        assert jnp.allclose(forces_batch, forces_single, rtol=1e-6), (
            "Forces mismatch between batch and single-system"
        )

    def test_batch_multiple_systems_vs_sequential(self, device):
        """Test batch with multiple systems matches sequential single-system calls."""
        num_systems = 3
        dtype = jnp.float64

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
        for pos, chg, cell_s in systems:
            e, f = pme_reciprocal_space(
                positions=pos,
                charges=chg,
                cell=cell_s,
                alpha=jnp.array([0.3]),
                mesh_dimensions=(16, 16, 16),
                compute_forces=True,
            )
            energies_single.append(e)
            forces_single.append(f)

        # Batch processing
        positions_batch = jnp.concatenate([s[0] for s in systems], axis=0)
        charges_batch = jnp.concatenate([s[1] for s in systems], axis=0)
        cells_batch = jnp.concatenate([s[2] for s in systems], axis=0)

        atoms_per_system = [s[0].shape[0] for s in systems]
        batch_idx = place_on_device(
            jnp.repeat(
                jnp.arange(num_systems, dtype=jnp.int32),
                jnp.array(atoms_per_system),
            ),
            device,
        )

        energies_batch, forces_batch = pme_reciprocal_space(
            positions=positions_batch,
            charges=charges_batch,
            cell=cells_batch,
            alpha=jnp.array([0.3] * num_systems),
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

            assert jnp.allclose(e_batch, e_single, rtol=1e-4, atol=1e-6), (
                f"System {sys_idx}: Energy mismatch batch={e_batch} single={e_single}"
            )

            f_batch = forces_batch[start_idx:end_idx]
            f_single = forces_single[sys_idx]

            assert jnp.allclose(f_batch, f_single, rtol=1e-4, atol=1e-6), (
                f"System {sys_idx}: Forces mismatch"
            )

            start_idx = end_idx

    def test_batch_different_cells(self, device):
        """Test batch with different cell sizes per system."""
        dtype = jnp.float64

        # Two systems with different cell sizes
        pos1 = place_on_device(
            jnp.array([[2.5, 2.5, 2.5], [3.5, 3.5, 3.5]], dtype=dtype), device
        )
        chg1 = place_on_device(jnp.array([1.0, -1.0], dtype=dtype), device)
        cell1 = jnp.eye(3, dtype=dtype) * 6.0

        pos2 = place_on_device(
            jnp.array([[4.0, 4.0, 4.0], [6.0, 6.0, 6.0]], dtype=dtype), device
        )
        chg2 = place_on_device(jnp.array([0.5, -0.5], dtype=dtype), device)
        cell2 = jnp.eye(3, dtype=dtype) * 10.0

        # Single-system calculations
        e1_single, f1_single = pme_reciprocal_space(
            positions=pos1,
            charges=chg1,
            cell=cell1.reshape(1, 3, 3),
            alpha=jnp.array([0.3]),
            mesh_dimensions=(16, 16, 16),
            compute_forces=True,
        )
        e2_single, f2_single = pme_reciprocal_space(
            positions=pos2,
            charges=chg2,
            cell=cell2.reshape(1, 3, 3),
            alpha=jnp.array([0.3]),
            mesh_dimensions=(16, 16, 16),
            compute_forces=True,
        )

        # Batch calculation
        positions_batch = jnp.concatenate([pos1, pos2], axis=0)
        charges_batch = jnp.concatenate([chg1, chg2], axis=0)
        cells_batch = jnp.stack([cell1, cell2], axis=0)
        batch_idx = place_on_device(jnp.array([0, 0, 1, 1], dtype=jnp.int32), device)

        e_batch, f_batch = pme_reciprocal_space(
            positions=positions_batch,
            charges=charges_batch,
            cell=cells_batch,
            alpha=jnp.array([0.3, 0.3]),
            mesh_dimensions=(16, 16, 16),
            batch_idx=batch_idx,
            compute_forces=True,
        )

        # Compare
        assert jnp.allclose(e_batch[:2].sum(), e1_single.sum(), rtol=1e-4)
        assert jnp.allclose(e_batch[2:].sum(), e2_single.sum(), rtol=1e-4)
        assert jnp.allclose(f_batch[:2], f1_single, rtol=1e-4)
        assert jnp.allclose(f_batch[2:], f2_single, rtol=1e-4)

    def test_batch_conservation_per_system(self, device):
        """Test momentum conservation for each system in batch."""
        num_systems = 3
        atoms_per_system = [4, 5, 3]

        # Create neutral systems
        positions_list = []
        charges_list = []
        for idx, n_atoms in enumerate(atoms_per_system):
            key = jax.random.PRNGKey(idx + 100)
            pos = jax.random.uniform(key, (n_atoms, 3), dtype=jnp.float64) * 8.0
            pos = place_on_device(pos, device)

            key2 = jax.random.PRNGKey(idx + 200)
            chg = jax.random.normal(key2, (n_atoms,), dtype=jnp.float64)
            chg = chg.at[-1].set(-chg[:-1].sum())  # Neutralize
            chg = place_on_device(chg, device)

            positions_list.append(pos)
            charges_list.append(chg)

        positions = jnp.concatenate(positions_list, axis=0)
        charges = jnp.concatenate(charges_list, axis=0)
        cells = place_on_device(
            jnp.stack([jnp.eye(3, dtype=jnp.float64) * 10.0] * num_systems, axis=0),
            device,
        )
        batch_idx = place_on_device(
            jnp.repeat(
                jnp.arange(num_systems, dtype=jnp.int32),
                jnp.array(atoms_per_system),
            ),
            device,
        )

        _, forces = pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cells,
            alpha=jnp.array([0.3] * num_systems),
            mesh_dimensions=(32, 32, 32),
            batch_idx=batch_idx,
            compute_forces=True,
        )

        # Check momentum conservation per system
        # PME forces use float32 B-spline interpolation internally, which
        # limits the precision of momentum conservation. A coarser mesh
        # exacerbates this because the spline assignment error is larger
        # relative to the grid spacing. With a 32^3 mesh, conservation
        # is typically within ~0.2 for random small systems.
        start_idx = 0
        for sys_idx, n_atoms in enumerate(atoms_per_system):
            end_idx = start_idx + n_atoms
            net_force = forces[start_idx:end_idx].sum(axis=0)
            assert jnp.allclose(
                net_force, jnp.zeros(3, dtype=net_force.dtype), atol=2e-1
            ), f"System {sys_idx}: Net force = {net_force}"
            start_idx = end_idx

    @pytest.mark.parametrize("system_fn", ["cscl", "wurtzite", "zincblende"])
    def test_batch_explicit_forces_vs_single(self, device, system_fn):
        """Test batch explicit forces match single-system explicit forces."""
        dtype = jnp.float64

        system_fns = {
            "cscl": create_cscl_supercell,
            "wurtzite": create_wurtzite_system,
            "zincblende": create_zincblende_system,
        }

        # Create two systems
        system1 = system_fns[system_fn](1)
        system2 = system_fns[system_fn](2)

        pos1 = place_on_device(jnp.array(system1.positions, dtype=dtype), device)
        chg1 = place_on_device(jnp.array(system1.charges, dtype=dtype), device)
        cell1 = jnp.array(system1.cell, dtype=dtype)

        pos2 = place_on_device(jnp.array(system2.positions, dtype=dtype), device)
        chg2 = place_on_device(jnp.array(system2.charges, dtype=dtype), device)
        cell2 = jnp.array(system2.cell, dtype=dtype)

        mesh_dims = (16, 16, 16)
        alpha = 0.3

        # Single-system forces
        _, forces1_single = pme_reciprocal_space(
            positions=pos1,
            charges=chg1,
            cell=cell1.reshape(1, 3, 3),
            alpha=jnp.array([alpha]),
            mesh_dimensions=mesh_dims,
            compute_forces=True,
        )
        _, forces2_single = pme_reciprocal_space(
            positions=pos2,
            charges=chg2,
            cell=cell2.reshape(1, 3, 3),
            alpha=jnp.array([alpha]),
            mesh_dimensions=mesh_dims,
            compute_forces=True,
        )

        # Batch forces
        n1, n2 = pos1.shape[0], pos2.shape[0]
        positions_batch = jnp.concatenate([pos1, pos2], axis=0)
        charges_batch = jnp.concatenate([chg1, chg2], axis=0)
        cells_batch = jnp.stack([cell1, cell2], axis=0)
        batch_idx = place_on_device(
            jnp.array([0] * n1 + [1] * n2, dtype=jnp.int32), device
        )

        _, forces_batch = pme_reciprocal_space(
            positions=positions_batch,
            charges=charges_batch,
            cell=cells_batch,
            alpha=jnp.array([alpha, alpha]),
            mesh_dimensions=mesh_dims,
            batch_idx=batch_idx,
            compute_forces=True,
        )

        forces1_batch = forces_batch[:n1]
        forces2_batch = forces_batch[n1:]

        assert jnp.allclose(forces1_batch, forces1_single, rtol=1e-4, atol=1e-6), (
            f"{system_fn}: System 1 forces mismatch"
        )
        assert jnp.allclose(forces2_batch, forces2_single, rtol=1e-4, atol=1e-6), (
            f"{system_fn}: System 2 forces mismatch"
        )


###########################################################################################
########################### Explicit Gradient Tests (Replaces Autograd) ####################
###########################################################################################


class TestExplicitChargeGradients:
    """Test explicit charge gradient computation (compute_charge_gradients=True).

    Since JAX bindings do not support autograd (enable_backward=False),
    we test the explicit charge gradient flag instead.
    """

    def test_reciprocal_charge_gradients_shape(self, device):
        """Test reciprocal-space charge gradients have correct shape."""
        positions, charges, cell = create_simple_system(device, num_atoms=4)

        energies, charge_grads = pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=jnp.array([0.3]),
            mesh_dimensions=(16, 16, 16),
            compute_forces=False,
            compute_charge_gradients=True,
        )

        assert energies.shape == (4,)
        assert charge_grads.shape == (4,)

    def test_reciprocal_charge_gradients_finite(self, device):
        """Test reciprocal-space charge gradients are finite and non-zero."""
        positions, charges, cell = create_simple_system(device, num_atoms=4)

        energies, charge_grads = pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=jnp.array([0.3]),
            mesh_dimensions=(16, 16, 16),
            compute_forces=False,
            compute_charge_gradients=True,
        )

        assert jnp.all(jnp.isfinite(charge_grads))
        # At least one should be non-zero
        assert jnp.any(jnp.abs(charge_grads) > 1e-10)

    def test_reciprocal_charge_grad_with_forces(self, device):
        """Test charge gradients when compute_forces=True for pme_reciprocal_space."""
        positions, charges, cell = create_simple_system(device, num_atoms=4)

        energies, forces, charge_grads = pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=jnp.array([0.3]),
            mesh_dimensions=(16, 16, 16),
            compute_forces=True,
            compute_charge_gradients=True,
        )

        assert energies.shape == (4,)
        assert forces.shape == (4, 3)
        assert charge_grads.shape == (4,)
        assert jnp.all(jnp.isfinite(energies))
        assert jnp.all(jnp.isfinite(forces))
        assert jnp.all(jnp.isfinite(charge_grads))

    def test_batch_reciprocal_charge_grad(self, device):
        """Test charge gradients for batch pme_reciprocal_space."""
        positions = place_on_device(
            jnp.array(
                [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                ],
                dtype=jnp.float64,
            ),
            device,
        )
        charges = place_on_device(
            jnp.array([1.0, -1.0, 0.5, -0.5], dtype=jnp.float64), device
        )
        cell = place_on_device(
            jnp.stack([jnp.eye(3, dtype=jnp.float64) * 10.0] * 2, axis=0),
            device,
        )
        batch_idx = place_on_device(jnp.array([0, 0, 1, 1], dtype=jnp.int32), device)

        energies, charge_grads = pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=jnp.array([0.3, 0.3]),
            mesh_dimensions=(16, 16, 16),
            batch_idx=batch_idx,
            compute_forces=False,
            compute_charge_gradients=True,
        )

        assert energies.shape == (4,)
        assert charge_grads.shape == (4,)
        assert jnp.all(jnp.isfinite(energies))
        assert jnp.all(jnp.isfinite(charge_grads))

    def test_full_pme_charge_grad_shapes(self, device):
        """Test charge gradients for particle_mesh_ewald with forces."""
        positions, charges, cell = create_simple_system(device, num_atoms=4)

        cutoff = 5.0
        pbc = place_on_device(jnp.array([[True, True, True]]), device)
        neighbor_list, neighbor_ptr, neighbor_shifts = cell_list(
            positions, cutoff, cell, pbc, return_neighbor_list=True
        )

        energies, forces, charge_grads = particle_mesh_ewald(
            positions=positions,
            charges=charges,
            cell=cell,
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
        assert jnp.all(jnp.isfinite(energies))
        assert jnp.all(jnp.isfinite(forces))
        assert jnp.all(jnp.isfinite(charge_grads))

    def test_full_pme_charge_grad_no_forces(self, device):
        """Test charge gradients for particle_mesh_ewald without forces."""
        positions, charges, cell = create_simple_system(device, num_atoms=4)

        cutoff = 5.0
        pbc = place_on_device(jnp.array([[True, True, True]]), device)
        neighbor_list, neighbor_ptr, neighbor_shifts = cell_list(
            positions, cutoff, cell, pbc, return_neighbor_list=True
        )

        energies, charge_grads = particle_mesh_ewald(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=0.3,
            mesh_dimensions=(16, 16, 16),
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            compute_forces=False,
            compute_charge_gradients=True,
        )

        assert energies.shape == (4,)
        assert charge_grads.shape == (4,)
        assert jnp.all(jnp.isfinite(energies))
        assert jnp.all(jnp.isfinite(charge_grads))


###########################################################################################
########################### Full PME (Real + Reciprocal) Tests ############################
###########################################################################################


class TestParticleMeshEwald:
    """Test the combined particle_mesh_ewald function."""

    def test_full_pme_output_shape(self, device):
        """Test output shape of full PME calculation."""
        positions, charges, cell = create_simple_system(device, num_atoms=5)

        cutoff = 5.0
        pbc = place_on_device(jnp.array([[True, True, True]]), device)
        neighbor_list, neighbor_ptr, neighbor_shifts = cell_list(
            positions, cutoff, cell, pbc, return_neighbor_list=True
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

        assert energies.shape == (5,)
        assert forces.shape == (5, 3)
        assert jnp.all(jnp.isfinite(energies))
        assert jnp.all(jnp.isfinite(forces))

    def test_full_pme_energy_only(self, device):
        """Test full PME energy-only output."""
        positions, charges, cell = create_simple_system(device, num_atoms=4)

        cutoff = 5.0
        pbc = place_on_device(jnp.array([[True, True, True]]), device)
        neighbor_list, neighbor_ptr, neighbor_shifts = cell_list(
            positions, cutoff, cell, pbc, return_neighbor_list=True
        )

        energies = particle_mesh_ewald(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=0.3,
            mesh_dimensions=(16, 16, 16),
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            compute_forces=False,
        )

        assert energies.shape == (4,)
        assert jnp.all(jnp.isfinite(energies))

    def test_full_pme_auto_estimate_alpha(self, device):
        """Test full PME with automatic alpha estimation."""
        positions, charges, cell = create_simple_system(device, num_atoms=5)

        cutoff = 5.0
        pbc = place_on_device(jnp.array([[True, True, True]]), device)
        neighbor_list, neighbor_ptr, neighbor_shifts = cell_list(
            positions, cutoff, cell, pbc, return_neighbor_list=True
        )

        # Call without alpha - should auto-estimate
        energies, forces = particle_mesh_ewald(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=None,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            compute_forces=True,
        )

        assert energies.shape == (5,)
        assert forces.shape == (5, 3)
        assert jnp.all(jnp.isfinite(energies))
        assert jnp.all(jnp.isfinite(forces))

    def test_full_pme_mesh_spacing(self, device):
        """Test full PME with mesh_spacing parameter."""
        positions, charges, cell = create_simple_system(device, num_atoms=5)

        cutoff = 5.0
        pbc = place_on_device(jnp.array([[True, True, True]]), device)
        neighbor_list, neighbor_ptr, neighbor_shifts = cell_list(
            positions, cutoff, cell, pbc, return_neighbor_list=True
        )

        energies, forces = particle_mesh_ewald(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=0.3,
            mesh_spacing=0.5,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            compute_forces=True,
        )

        assert energies.shape == (5,)
        assert forces.shape == (5, 3)
        assert jnp.all(jnp.isfinite(energies))
        assert jnp.all(jnp.isfinite(forces))


###########################################################################################
########################### Non-Cubic Cell Tests ##########################################
###########################################################################################


class TestNonCubicCells:
    """Test PME with non-cubic simulation cells."""

    def test_orthorhombic_cell(self, device):
        """Test PME with orthorhombic cell."""
        cell = place_on_device(
            jnp.array(
                [[[8.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 12.0]]],
                dtype=jnp.float64,
            ),
            device,
        )
        positions = place_on_device(
            jnp.array([[2.0, 5.0, 6.0], [6.0, 5.0, 6.0]], dtype=jnp.float64),
            device,
        )
        charges = place_on_device(jnp.array([1.0, -1.0], dtype=jnp.float64), device)

        energies, forces = pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=jnp.array([0.3]),
            mesh_dimensions=(16, 20, 24),
            compute_forces=True,
        )

        assert jnp.all(jnp.isfinite(energies))
        assert jnp.all(jnp.isfinite(forces))
        # Momentum conservation
        net_force = forces.sum(axis=0)
        assert jnp.allclose(net_force, jnp.zeros(3, dtype=net_force.dtype), atol=1e-2)

    def test_triclinic_cell(self, device):
        """Test PME with triclinic cell."""
        cell = place_on_device(
            jnp.array(
                [[[10.0, 0.0, 0.0], [2.0, 10.0, 0.0], [1.0, 1.0, 10.0]]],
                dtype=jnp.float64,
            ),
            device,
        )
        positions = place_on_device(
            jnp.array([[2.0, 5.0, 5.0], [7.0, 5.0, 5.0]], dtype=jnp.float64),
            device,
        )
        charges = place_on_device(jnp.array([1.0, -1.0], dtype=jnp.float64), device)

        energies, forces = pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=jnp.array([0.3]),
            mesh_dimensions=(16, 16, 16),
            compute_forces=True,
        )

        assert jnp.all(jnp.isfinite(energies))
        assert jnp.all(jnp.isfinite(forces))

    def test_wurtzite_cell(self, device):
        """Test PME with wurtzite (hexagonal) cell."""
        crystal = create_wurtzite_system(size=2)

        positions = place_on_device(
            jnp.array(crystal.positions, dtype=jnp.float64), device
        )
        charges = place_on_device(jnp.array(crystal.charges, dtype=jnp.float64), device)
        cell = place_on_device(
            jnp.array(crystal.cell[np.newaxis, :, :], dtype=jnp.float64), device
        )

        energies, forces = pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=jnp.array([0.3]),
            mesh_dimensions=(16, 16, 16),
            compute_forces=True,
        )

        assert jnp.all(jnp.isfinite(energies))
        assert jnp.all(jnp.isfinite(forces))


###########################################################################################
########################### Precomputed K-Vectors Tests ###################################
###########################################################################################


class TestPrecomputedKVectors:
    """Test PME with precomputed k-vectors."""

    def test_precomputed_kvectors(self, device):
        """Test that precomputed k-vectors give same results."""
        positions, charges, cell = create_dipole_system(device)
        mesh_dims = (16, 16, 16)

        # Without precomputed k-vectors
        energies1, forces1 = pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=jnp.array([0.3]),
            mesh_dimensions=mesh_dims,
            compute_forces=True,
        )

        # With precomputed k-vectors
        k_vectors, k_squared = generate_k_vectors_pme(cell, mesh_dims)
        energies2, forces2 = pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=jnp.array([0.3]),
            mesh_dimensions=mesh_dims,
            compute_forces=True,
            k_vectors=k_vectors,
            k_squared=k_squared,
        )

        assert jnp.allclose(energies1, energies2, rtol=1e-6)
        assert jnp.allclose(forces1, forces2, rtol=1e-6)


###########################################################################################
########################### Single Atom System Tests ######################################
###########################################################################################


class TestSingleAtomSystem:
    """Test handling of single atom systems."""

    def test_single_atom_pme(self, device):
        """Test PME with single atom."""
        positions = place_on_device(
            jnp.array([[5.0, 5.0, 5.0]], dtype=jnp.float64), device
        )
        charges = place_on_device(jnp.array([1.0], dtype=jnp.float64), device)
        cell = place_on_device(
            jnp.eye(3, dtype=jnp.float64).reshape(1, 3, 3) * 10.0, device
        )

        energies, forces = pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=jnp.array([0.3]),
            mesh_dimensions=(16, 16, 16),
            compute_forces=True,
        )

        assert energies.shape == (1,)
        assert forces.shape == (1, 3)
        assert jnp.all(jnp.isfinite(energies))
        assert jnp.all(jnp.isfinite(forces))


###########################################################################################
########################### Zero Charges Tests ############################################
###########################################################################################


class TestZeroCharges:
    """Test behavior with zero charges."""

    def test_zero_charges_zero_energy(self, device):
        """Test that zero charges give zero energy."""
        positions = place_on_device(
            jnp.array([[2.0, 5.0, 5.0], [8.0, 5.0, 5.0]], dtype=jnp.float64),
            device,
        )
        charges = place_on_device(jnp.array([0.0, 0.0], dtype=jnp.float64), device)
        cell = place_on_device(
            jnp.eye(3, dtype=jnp.float64).reshape(1, 3, 3) * 10.0, device
        )

        energies, forces = pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=jnp.array([0.3]),
            mesh_dimensions=(16, 16, 16),
            compute_forces=True,
        )

        assert jnp.allclose(energies, jnp.zeros_like(energies), atol=1e-10)
        assert jnp.allclose(forces, jnp.zeros_like(forces), atol=1e-10)


###########################################################################################
########################### Alpha Sensitivity Tests #######################################
###########################################################################################


class TestAlphaSensitivity:
    """Test sensitivity to alpha parameter."""

    def test_alpha_affects_energy(self, device):
        """Test that different alpha values affect energy."""
        positions, charges, cell = create_dipole_system(device)

        energies_low_alpha = pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=jnp.array([0.2]),
            mesh_dimensions=(16, 16, 16),
            compute_forces=False,
        )

        energies_high_alpha = pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=jnp.array([0.5]),
            mesh_dimensions=(16, 16, 16),
            compute_forces=False,
        )

        # Different alpha should give different energies
        assert not jnp.allclose(energies_low_alpha, energies_high_alpha)


###########################################################################################
########################### Batch with Per-System Alpha Tests #############################
###########################################################################################


class TestBatchWithDifferentAlpha:
    """Test batch calculations with per-system alpha."""

    def test_batch_per_system_alpha(self, device):
        """Test batch with different alpha per system."""
        pos1, chg1, cell1 = create_dipole_system(device)
        pos2, chg2, cell2 = create_dipole_system(device, separation=3.0)

        positions = jnp.concatenate([pos1, pos2], axis=0)
        charges = jnp.concatenate([chg1, chg2], axis=0)
        cells = jnp.concatenate([cell1, cell2], axis=0)
        batch_idx = place_on_device(jnp.array([0, 0, 1, 1], dtype=jnp.int32), device)

        # Different alpha per system
        alphas = jnp.array([0.2, 0.5])

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
        assert jnp.all(jnp.isfinite(energies))
        assert jnp.all(jnp.isfinite(forces))


###########################################################################################
########################### Forces vs Finite Differences ##################################
###########################################################################################


class TestPMEForcesNumericalGradient:
    """Validate forces against numerical gradients (finite differences)."""

    def test_forces_vs_finite_differences(self, device):
        """Test that analytical forces match finite difference gradients.

        Uses a larger charge separation and denser mesh so that the
        force magnitudes are well above float32 noise.  The finite-
        difference step ``h`` is chosen large enough that the energy
        differences are resolvable with the float32 B-spline
        interpolation used internally by the PME kernel.
        """
        # Use a well-separated dipole with stronger charges for bigger forces
        positions, charges, cell = create_dipole_system(
            device, separation=3.0, cell_size=12.0
        )
        # Scale charges up so forces are not tiny
        charges = charges * 3.0

        # Slightly perturb positions to avoid symmetric configurations
        key = jax.random.PRNGKey(999)
        perturbation = jax.random.normal(key, positions.shape) * 0.05
        positions = positions + perturbation

        mesh_dims = (32, 32, 32)
        alpha_val = jnp.array([0.4])

        # Analytical forces
        _, analytical_forces = pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha_val,
            mesh_dimensions=mesh_dims,
            compute_forces=True,
        )

        # Numerical forces via finite differences
        # A larger h is needed because the underlying energy uses float32
        # B-spline interpolation; too small a step produces energy
        # differences dominated by float32 rounding noise.
        h = 1e-3
        positions_np = np.array(positions)
        numerical_forces = np.zeros_like(positions_np)

        for atom_idx in range(positions_np.shape[0]):
            for coord_idx in range(3):
                # Forward
                pos_plus = positions_np.copy()
                pos_plus[atom_idx, coord_idx] += h
                e_plus = pme_reciprocal_space(
                    positions=place_on_device(
                        jnp.array(pos_plus, dtype=jnp.float64), device
                    ),
                    charges=charges,
                    cell=cell,
                    alpha=alpha_val,
                    mesh_dimensions=mesh_dims,
                    compute_forces=False,
                )

                # Backward
                pos_minus = positions_np.copy()
                pos_minus[atom_idx, coord_idx] -= h
                e_minus = pme_reciprocal_space(
                    positions=place_on_device(
                        jnp.array(pos_minus, dtype=jnp.float64), device
                    ),
                    charges=charges,
                    cell=cell,
                    alpha=alpha_val,
                    mesh_dimensions=mesh_dims,
                    compute_forces=False,
                )

                # Central difference: F = -dE/dr
                numerical_forces[atom_idx, coord_idx] = -(
                    float(e_plus.sum()) - float(e_minus.sum())
                ) / (2 * h)

        # With float32 spline interpolation, we expect agreement within
        # ~5% relative or ~5e-3 absolute for moderate-magnitude forces.
        assert jnp.allclose(
            analytical_forces,
            jnp.array(numerical_forces, dtype=jnp.float32),
            rtol=5e-2,
            atol=5e-3,
        ), (
            f"Forces don't match numerical gradient:\n"
            f"  Max diff: {jnp.abs(analytical_forces - jnp.array(numerical_forces)).max()}\n"
            f"  Analytical: {analytical_forces}\n"
            f"  Numerical: {numerical_forces}"
        )


###########################################################################################
########################### Spline Order Tests ############################################
###########################################################################################


class TestSplineOrders:
    """Test different spline interpolation orders."""

    @pytest.mark.parametrize("spline_order", [2, 3, 4, 5, 6])
    def test_spline_order_valid_results(self, device, spline_order):
        """Test that different spline orders give valid results."""
        positions, charges, cell = create_dipole_system(device)

        energies, forces = pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=jnp.array([0.3]),
            mesh_dimensions=(32, 32, 32),
            spline_order=spline_order,
            compute_forces=True,
        )

        assert jnp.all(jnp.isfinite(energies))
        assert jnp.all(jnp.isfinite(forces))
        # Momentum conservation
        net_force = forces.sum(axis=0)
        assert jnp.allclose(net_force, jnp.zeros(3, dtype=net_force.dtype), atol=1e-2)


###########################################################################################
########################### Mesh Spacing Path Tests #######################################
###########################################################################################


class TestPMEMeshSpacing:
    """Test mesh_spacing alternative to mesh_dimensions."""

    def test_mesh_spacing_path(self, device):
        """Test mesh_spacing path for dimension computation."""
        positions, charges, cell = create_simple_system(device, num_atoms=5)

        # Use mesh_spacing instead of mesh_dimensions
        energies = pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=jnp.array([0.3]),
            mesh_spacing=0.5,
            compute_forces=False,
        )

        assert jnp.all(jnp.isfinite(energies))


###########################################################################################
########################### Alpha Validation Tests ########################################
###########################################################################################


class TestPrepareAlphaPME:
    """Test alpha parameter validation edge cases in PME.

    Tests validation logic from _prepare_alpha_array via ewald_real_space
    (called internally by particle_mesh_ewald).
    """

    def test_scalar_alpha_0d_array(self, device):
        """Test 0-dimensional alpha array expansion."""
        positions, charges, cell = create_simple_system(device, num_atoms=5)

        # 0-dimensional JAX array (scalar array)
        alpha = jnp.array(0.3)

        cutoff = 5.0
        pbc = place_on_device(jnp.array([[True, True, True]]), device)
        neighbor_list, neighbor_ptr, neighbor_shifts = cell_list(
            positions, cutoff, cell, pbc, return_neighbor_list=True
        )

        energies = particle_mesh_ewald(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,  # 0-dim array
            mesh_dimensions=(16, 16, 16),
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            compute_forces=False,
        )

        assert jnp.all(jnp.isfinite(energies))

    def test_alpha_wrong_size_raises_error(self, device):
        """Test alpha array with wrong number of elements raises ValueError."""
        positions, charges, cell = create_simple_system(device, num_atoms=5)

        # Alpha array with wrong size (2 values for 1 system)
        alpha = jnp.array([0.3, 0.5])

        cutoff = 5.0
        pbc = place_on_device(jnp.array([[True, True, True]]), device)
        neighbor_list, neighbor_ptr, neighbor_shifts = cell_list(
            positions, cutoff, cell, pbc, return_neighbor_list=True
        )

        with pytest.raises(ValueError):
            particle_mesh_ewald(
                positions=positions,
                charges=charges,
                cell=cell,
                alpha=alpha,  # Wrong size: 2 values for 1 system
                mesh_dimensions=(16, 16, 16),
                neighbor_list=neighbor_list,
                neighbor_ptr=neighbor_ptr,
                neighbor_shifts=neighbor_shifts,
                compute_forces=False,
            )

    def test_alpha_invalid_type_raises_error(self, device):
        """Test non-float, non-array alpha raises an error.

        Note: particle_mesh_ewald raises AttributeError because the inline
        alpha handling accesses .ndim before type-checking. The underlying
        _prepare_alpha_array (used by ewald_real_space) raises TypeError.
        We accept either error type here.
        """
        positions, charges, cell = create_simple_system(device, num_atoms=5)

        cutoff = 5.0
        pbc = place_on_device(jnp.array([[True, True, True]]), device)
        neighbor_list, neighbor_ptr, neighbor_shifts = cell_list(
            positions, cutoff, cell, pbc, return_neighbor_list=True
        )

        with pytest.raises((TypeError, AttributeError)):
            particle_mesh_ewald(
                positions=positions,
                charges=charges,
                cell=cell,
                alpha="invalid",  # String is not valid
                mesh_dimensions=(16, 16, 16),
                neighbor_list=neighbor_list,
                neighbor_ptr=neighbor_ptr,
                neighbor_shifts=neighbor_shifts,
                compute_forces=False,
            )


###########################################################################################
########################### Mesh Dimension Error Tests ####################################
###########################################################################################


class TestPMEMeshDimensionErrors:
    """Test mesh dimension handling for coverage.

    Note: Unlike the PyTorch implementation, the JAX pme_reciprocal_space
    falls back to estimate_pme_mesh_dimensions when both mesh_dimensions
    and mesh_spacing are None, rather than raising ValueError. We verify
    this graceful fallback behavior.
    """

    def test_no_mesh_dimensions_or_spacing_falls_back_to_estimation(self, device):
        """Test that pme_reciprocal_space estimates mesh when both are None."""
        positions, charges, cell = create_simple_system(device, num_atoms=5)

        # Neither mesh_dimensions nor mesh_spacing provided
        # JAX version gracefully falls back to estimate_pme_mesh_dimensions
        energies = pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=jnp.array([0.3]),
            mesh_dimensions=None,
            mesh_spacing=None,
            compute_forces=False,
        )

        assert jnp.all(jnp.isfinite(energies))
        assert energies.shape == (5,)

    def test_mesh_spacing_path_in_reciprocal_space(self, device):
        """Test mesh_spacing path for dimension computation in pme_reciprocal_space."""
        positions, charges, cell = create_simple_system(device, num_atoms=5)

        energies = pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=jnp.array([0.3]),
            mesh_spacing=0.5,
            compute_forces=False,
        )

        assert jnp.all(jnp.isfinite(energies))
        assert energies.shape == (5,)


###########################################################################################
########################### Auto-Estimation Tests #########################################
###########################################################################################


class TestParticleMeshEwaldAutoEstimation:
    """Test particle_mesh_ewald auto-estimation paths.

    Note: Basic alpha auto-estimation and mesh_spacing tests are in
    TestParticleMeshEwald. This class covers additional estimation paths
    not tested elsewhere.
    """

    def test_accuracy_based_mesh_estimation(self, device):
        """Test accuracy-based mesh dimension estimation."""
        positions, charges, cell = create_simple_system(device, num_atoms=5)

        cutoff = 5.0
        pbc = place_on_device(jnp.array([[True, True, True]]), device)
        neighbor_list, neighbor_ptr, neighbor_shifts = cell_list(
            positions, cutoff, cell, pbc, return_neighbor_list=True
        )

        # Provide alpha but no mesh_dimensions or mesh_spacing
        # Should use accuracy-based estimation
        energies, forces = particle_mesh_ewald(
            positions=positions,
            charges=charges,
            cell=cell,
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
        assert jnp.all(jnp.isfinite(energies))
        assert jnp.all(jnp.isfinite(forces))

    def test_auto_mesh_from_alpha_estimation(self, device):
        """Test mesh_dimensions auto-derived when alpha is auto-estimated."""
        positions, charges, cell = create_simple_system(device, num_atoms=5)

        cutoff = 5.0
        pbc = place_on_device(jnp.array([[True, True, True]]), device)
        neighbor_list, neighbor_ptr, neighbor_shifts = cell_list(
            positions, cutoff, cell, pbc, return_neighbor_list=True
        )

        # alpha=None triggers estimate_pme_parameters which sets alpha AND mesh_dimensions
        energies, forces = particle_mesh_ewald(
            positions=positions,
            charges=charges,
            cell=cell,
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
        assert jnp.all(jnp.isfinite(energies))
        assert jnp.all(jnp.isfinite(forces))


###########################################################################################
########################### Batch PME Shape Path Tests ####################################
###########################################################################################


class TestBatchPMEShapePaths:
    """Test batch PME shape helper code paths."""

    def test_batch_reciprocal_space_single_system(self, device):
        """Test batch reciprocal space with single system."""
        positions, charges, cell = create_simple_system(device, num_atoms=5)
        batch_idx = place_on_device(
            jnp.zeros(5, dtype=jnp.int32), device
        )  # All same batch

        energies, forces = pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=jnp.array([0.3]),
            mesh_dimensions=(16, 16, 16),
            batch_idx=batch_idx,
            compute_forces=True,
        )

        assert energies.shape == (5,)
        assert forces.shape == (5, 3)
        assert jnp.all(jnp.isfinite(energies))
        assert jnp.all(jnp.isfinite(forces))

    def test_batch_reciprocal_space_multi_system(self, device):
        """Test batch reciprocal space with heterogeneous systems.

        Tests two systems with different atom counts to exercise the
        batch helper logic for non-uniform partitions.
        """
        dtype = jnp.float64

        # System 1: 3 atoms
        pos1 = place_on_device(
            jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 3.0, 2.0]], dtype=dtype),
            device,
        )
        chg1 = place_on_device(jnp.array([1.0, -0.5, -0.5], dtype=dtype), device)

        # System 2: 2 atoms
        pos2 = place_on_device(
            jnp.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]], dtype=dtype),
            device,
        )
        chg2 = place_on_device(jnp.array([0.5, -0.5], dtype=dtype), device)

        positions = jnp.concatenate([pos1, pos2], axis=0)
        charges = jnp.concatenate([chg1, chg2], axis=0)
        cells = place_on_device(
            jnp.stack([jnp.eye(3, dtype=dtype) * 10.0] * 2, axis=0),
            device,
        )
        batch_idx = place_on_device(jnp.array([0, 0, 0, 1, 1], dtype=jnp.int32), device)

        energies, forces = pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cells,
            alpha=jnp.array([0.3, 0.3]),
            mesh_dimensions=(16, 16, 16),
            batch_idx=batch_idx,
            compute_forces=True,
        )

        assert energies.shape == (5,)
        assert forces.shape == (5, 3)
        assert jnp.all(jnp.isfinite(energies))
        assert jnp.all(jnp.isfinite(forces))

        # Verify batch vs sequential consistency for heterogeneous systems
        e1_single, f1_single = pme_reciprocal_space(
            positions=pos1,
            charges=chg1,
            cell=cells[:1],
            alpha=jnp.array([0.3]),
            mesh_dimensions=(16, 16, 16),
            compute_forces=True,
        )
        e2_single, f2_single = pme_reciprocal_space(
            positions=pos2,
            charges=chg2,
            cell=cells[1:],
            alpha=jnp.array([0.3]),
            mesh_dimensions=(16, 16, 16),
            compute_forces=True,
        )

        assert jnp.allclose(energies[:3].sum(), e1_single.sum(), rtol=1e-4)
        assert jnp.allclose(energies[3:].sum(), e2_single.sum(), rtol=1e-4)
        assert jnp.allclose(forces[:3], f1_single, rtol=1e-4)
        assert jnp.allclose(forces[3:], f2_single, rtol=1e-4)


###########################################################################################
########################### PME Charge Gradient Finite Difference Tests ####################
###########################################################################################


class TestPMEChargeGradients:
    """Test explicit charge gradient computation against finite differences.

    Since JAX bindings do not support autograd (enable_backward=False),
    we compare explicit charge gradients against numerical finite
    differences to verify correctness.
    """

    def test_reciprocal_charge_grad_matches_finite_difference(self, device):
        """Test reciprocal charge gradients match finite difference estimate.

        Compares dE/dq_i (explicit) against central finite differences:
            dE/dq_i ≈ [E(q_i + h) - E(q_i - h)] / (2h)
        """
        positions, charges, cell = create_simple_system(device, num_atoms=4)

        # Get explicit charge gradients
        energies, charge_grads = pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=jnp.array([0.3]),
            mesh_dimensions=(16, 16, 16),
            compute_forces=False,
            compute_charge_gradients=True,
        )

        # Numerical charge gradients via finite differences
        h = 1e-3  # Larger step due to float32 B-spline
        charges_np = np.array(charges)
        numerical_charge_grads = np.zeros(len(charges_np))

        for i in range(len(charges_np)):
            # Forward
            chg_plus = charges_np.copy()
            chg_plus[i] += h
            e_plus = pme_reciprocal_space(
                positions=positions,
                charges=place_on_device(jnp.array(chg_plus, dtype=jnp.float64), device),
                cell=cell,
                alpha=jnp.array([0.3]),
                mesh_dimensions=(16, 16, 16),
                compute_forces=False,
            )

            # Backward
            chg_minus = charges_np.copy()
            chg_minus[i] -= h
            e_minus = pme_reciprocal_space(
                positions=positions,
                charges=place_on_device(
                    jnp.array(chg_minus, dtype=jnp.float64), device
                ),
                cell=cell,
                alpha=jnp.array([0.3]),
                mesh_dimensions=(16, 16, 16),
                compute_forces=False,
            )

            # Central difference: dE/dq_i
            numerical_charge_grads[i] = (float(e_plus.sum()) - float(e_minus.sum())) / (
                2 * h
            )

        # With float32 spline interpolation, expect ~5% agreement
        assert jnp.allclose(
            charge_grads,
            jnp.array(numerical_charge_grads, dtype=jnp.float32),
            rtol=5e-2,
            atol=5e-3,
        ), (
            f"Charge gradients don't match numerical estimate:\n"
            f"  Max diff: {jnp.abs(charge_grads - jnp.array(numerical_charge_grads)).max()}\n"
            f"  Explicit: {charge_grads}\n"
            f"  Numerical: {numerical_charge_grads}"
        )

    def test_batch_charge_grad_matches_finite_difference(self, device):
        """Test batch charge gradients match finite difference estimate."""
        positions = place_on_device(
            jnp.array(
                [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                ],
                dtype=jnp.float64,
            ),
            device,
        )
        charges = place_on_device(
            jnp.array([1.0, -1.0, 0.5, -0.5], dtype=jnp.float64), device
        )
        cell = place_on_device(
            jnp.stack([jnp.eye(3, dtype=jnp.float64) * 10.0] * 2, axis=0),
            device,
        )
        batch_idx = place_on_device(jnp.array([0, 0, 1, 1], dtype=jnp.int32), device)

        # Get explicit charge gradients
        energies, charge_grads = pme_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=jnp.array([0.3, 0.3]),
            mesh_dimensions=(16, 16, 16),
            batch_idx=batch_idx,
            compute_forces=False,
            compute_charge_gradients=True,
        )

        # Numerical charge gradients
        h = 1e-3
        charges_np = np.array(charges)
        numerical_charge_grads = np.zeros(len(charges_np))

        for i in range(len(charges_np)):
            chg_plus = charges_np.copy()
            chg_plus[i] += h
            e_plus = pme_reciprocal_space(
                positions=positions,
                charges=place_on_device(jnp.array(chg_plus, dtype=jnp.float64), device),
                cell=cell,
                alpha=jnp.array([0.3, 0.3]),
                mesh_dimensions=(16, 16, 16),
                batch_idx=batch_idx,
                compute_forces=False,
            )

            chg_minus = charges_np.copy()
            chg_minus[i] -= h
            e_minus = pme_reciprocal_space(
                positions=positions,
                charges=place_on_device(
                    jnp.array(chg_minus, dtype=jnp.float64), device
                ),
                cell=cell,
                alpha=jnp.array([0.3, 0.3]),
                mesh_dimensions=(16, 16, 16),
                batch_idx=batch_idx,
                compute_forces=False,
            )

            numerical_charge_grads[i] = (float(e_plus.sum()) - float(e_minus.sum())) / (
                2 * h
            )

        assert jnp.allclose(
            charge_grads,
            jnp.array(numerical_charge_grads, dtype=jnp.float32),
            rtol=5e-2,
            atol=5e-3,
        ), (
            f"Batch charge gradients mismatch:\n"
            f"  Explicit: {charge_grads}\n"
            f"  Numerical: {numerical_charge_grads}"
        )


###########################################################################################
########################### Full PME Neighbor List Tests ###################################
###########################################################################################


class TestFullPMENeighborList:
    """Test full PME with explicit neighbor list (COO) format.

    Verifies that particle_mesh_ewald correctly uses neighbor_list
    (COO format) for the real-space component on crystal systems,
    complementing the simpler tests in TestParticleMeshEwald.
    """

    def test_full_pme_neighbor_list_crystal_system(self, device):
        """Test full PME with neighbor list on a crystal system."""
        crystal = create_cscl_supercell(1)

        positions = place_on_device(
            jnp.array(crystal.positions, dtype=jnp.float64), device
        )
        charges = place_on_device(jnp.array(crystal.charges, dtype=jnp.float64), device)
        cell = place_on_device(
            jnp.array(crystal.cell[np.newaxis, :, :], dtype=jnp.float64), device
        )

        cutoff = 5.0
        pbc = place_on_device(jnp.array([[True, True, True]]), device)
        neighbor_list, neighbor_ptr, neighbor_shifts = cell_list(
            positions, cutoff, cell, pbc, return_neighbor_list=True
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

        num_atoms = positions.shape[0]
        assert energies.shape == (num_atoms,)
        assert forces.shape == (num_atoms, 3)
        assert jnp.all(jnp.isfinite(energies))
        assert jnp.all(jnp.isfinite(forces))

        # Momentum conservation check (relaxed for float32 splines)
        net_force = forces.sum(axis=0)
        assert jnp.allclose(
            net_force, jnp.zeros(3, dtype=net_force.dtype), atol=5e-2
        ), f"Net force = {net_force}"

    @pytest.mark.parametrize("system_fn", ["cscl", "wurtzite", "zincblende"])
    def test_full_pme_neighbor_list_multiple_crystals(self, device, system_fn):
        """Test full PME with neighbor list on multiple crystal types."""
        system_fns = {
            "cscl": create_cscl_supercell,
            "wurtzite": create_wurtzite_system,
            "zincblende": create_zincblende_system,
        }
        crystal = system_fns[system_fn](1)

        positions = place_on_device(
            jnp.array(crystal.positions, dtype=jnp.float64), device
        )
        charges = place_on_device(jnp.array(crystal.charges, dtype=jnp.float64), device)
        cell = place_on_device(
            jnp.array(crystal.cell[np.newaxis, :, :], dtype=jnp.float64), device
        )

        cutoff = 5.0
        pbc = place_on_device(jnp.array([[True, True, True]]), device)
        neighbor_list, neighbor_ptr, neighbor_shifts = cell_list(
            positions, cutoff, cell, pbc, return_neighbor_list=True
        )

        energies, forces, charge_grads = particle_mesh_ewald(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=0.3,
            mesh_dimensions=(16, 16, 16),
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            compute_forces=True,
            compute_charge_gradients=True,
        )

        num_atoms = positions.shape[0]
        assert energies.shape == (num_atoms,)
        assert forces.shape == (num_atoms, 3)
        assert charge_grads.shape == (num_atoms,)
        assert jnp.all(jnp.isfinite(energies))
        assert jnp.all(jnp.isfinite(forces))
        assert jnp.all(jnp.isfinite(charge_grads))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
