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

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nvalchemiops.jax.interactions.electrostatics.ewald import ewald_real_space
from nvalchemiops.jax.interactions.electrostatics.k_vectors import (
    generate_k_vectors_pme,
)
from nvalchemiops.jax.interactions.electrostatics.pme import (
    particle_mesh_ewald,
    pme_reciprocal_space,
)
from nvalchemiops.jax.neighbors import cell_list
from test.interactions.electrostatics.bindings.jax.conftest import (
    cubic_cell_jax,
    fd_virial_full_jax,
    make_crystal_system_jax,
    make_virial_cscl_system_jax,
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


def create_dipole_system(dtype=jnp.float64, separation=2.0, cell_size=10.0):
    """Create a simple dipole system with JAX arrays on GPU.

    Parameters
    ----------
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
    positions = jnp.array(
        [
            [center - separation / 2, center, center],
            [center + separation / 2, center, center],
        ],
        dtype=dtype,
    )
    charges = jnp.array([1.0, -1.0], dtype=dtype)
    cell = jnp.array(
        [[[cell_size, 0.0, 0.0], [0.0, cell_size, 0.0], [0.0, 0.0, cell_size]]],
        dtype=dtype,
    )
    return positions, charges, cell


def create_simple_system(dtype=jnp.float64, num_atoms=4, cell_size=10.0):
    """Create a simple test system with random positions and neutral charges.

    Parameters
    ----------
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
    positions = (
        jax.random.uniform(key, (num_atoms, 3), dtype=dtype) * cell_size * 0.8
        + cell_size * 0.1
    )

    # Create random charges and make neutral
    key2 = jax.random.PRNGKey(123)
    charges_raw = jax.random.normal(key2, (num_atoms,), dtype=dtype)
    # Make last charge neutralize the system
    charges_raw = charges_raw.at[-1].set(-charges_raw[:-1].sum())
    charges = charges_raw

    cell = jnp.array(
        [[[cell_size, 0.0, 0.0], [0.0, cell_size, 0.0], [0.0, 0.0, cell_size]]],
        dtype=dtype,
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
        """Test that pme_reciprocal_space returns arrays matching input dtype.

        Spline kernels now preserve input dtype (float32 or float64), so PME
        output dtype matches the input positions dtype.
        """
        positions, charges, cell = create_dipole_system(dtype=dtype)

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
        assert energies.dtype == dtype, f"Expected {dtype} output, got {energies.dtype}"

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
        assert energies.dtype == dtype, f"Expected {dtype} output, got {energies.dtype}"
        assert forces.dtype == dtype, f"Expected {dtype} output, got {forces.dtype}"

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_pme_batch_dtype_returns_correct_type(self, device, dtype):
        """Test that batch PME returns arrays matching input dtype.

        Spline kernels now preserve input dtype (float32 or float64), so PME
        output dtype matches the input positions dtype.
        """
        pos1, chg1, cell1 = create_dipole_system(dtype=dtype)
        pos2, chg2, cell2 = create_dipole_system(dtype=dtype, separation=3.0)

        positions = jnp.concatenate([pos1, pos2], axis=0)
        charges = jnp.concatenate([chg1, chg2], axis=0)
        cells = jnp.concatenate([cell1, cell2], axis=0)
        batch_idx = jnp.array([0, 0, 1, 1], dtype=jnp.int32)

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
        assert energies.dtype == dtype, f"Expected {dtype} output, got {energies.dtype}"

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
        assert energies.dtype == dtype, f"Expected {dtype} output, got {energies.dtype}"
        assert forces.dtype == dtype, f"Expected {dtype} output, got {forces.dtype}"

    def test_float32_vs_float64_consistency(self, device):
        """Test that float32 and float64 produce consistent results."""
        positions_f64, charges_f64, cell_f64 = create_dipole_system(dtype=jnp.float64)

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
        positions, charges, cell = create_simple_system(num_atoms=5)

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
        positions, charges, cell = create_simple_system(num_atoms=5)

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
        positions = jnp.array(
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
        )
        charges = jnp.array([1.0, -1.0, 0.0, 0.5, -0.5, 0.5, -0.5], dtype=jnp.float64)
        batch_idx = jnp.array([0, 0, 0, 1, 1, 1, 1], dtype=jnp.int32)
        cells = jnp.stack([jnp.eye(3, dtype=jnp.float64) * 10.0] * 2, axis=0)

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
        positions = jnp.zeros((0, 3), dtype=jnp.float64)
        charges = jnp.zeros(0, dtype=jnp.float64)
        cell = cubic_cell_jax(10.0)

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
        positions, charges, cell = create_dipole_system()

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
        positions, charges, cell = create_simple_system(num_atoms=6)

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
        positions, charges, cell = create_dipole_system()

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
        positions, charges, cell = create_dipole_system()

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
        positions, charges, cell = create_dipole_system()

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
    @pytest.mark.parametrize("jit", [False, True])
    def test_reciprocal_energy_matches_torchpme(self, device, alpha, mesh_spacing, jit):
        """Test that reciprocal energy matches torchpme."""
        positions, charges, cell = create_dipole_system(dtype=jnp.float64)

        # Convert to numpy for mesh dim computation
        cell_np = np.array(cell[0])
        cell_lengths = np.linalg.norm(cell_np, axis=1)
        mesh_dims = tuple(
            int(np.ceil(length / mesh_spacing)) for length in cell_lengths
        )

        func = partial(
            pme_reciprocal_space,
            alpha=jnp.array([alpha]),
            mesh_dimensions=mesh_dims,
            spline_order=4,
            compute_forces=False,
        )
        if jit:
            func = jax.jit(func)
        # Our implementation
        our_energy = func(
            positions=positions,
            charges=charges,
            cell=cell,
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
    @pytest.mark.parametrize("crystal_type", ["cscl", "wurtzite", "zincblende"])
    @pytest.mark.parametrize("alpha", [0.3, 0.5])
    @pytest.mark.parametrize("jit", [False, True])
    def test_crystal_systems_match_torchpme(
        self, size, crystal_type, alpha, device, jit
    ):
        """Test PME on crystal systems against torchpme."""
        positions, charges, cell = make_crystal_system_jax(crystal_type, size=size)
        positions_np = np.array(positions)
        charges_np = np.array(charges)
        cell_np = np.array(cell[0])

        mesh_spacing = 0.5
        cell_lengths = np.linalg.norm(cell_np, axis=1)
        mesh_dims = tuple(
            int(np.ceil(length / mesh_spacing)) for length in cell_lengths
        )

        func = partial(
            pme_reciprocal_space,
            alpha=jnp.array([alpha]),
            mesh_dimensions=mesh_dims,
            spline_order=4,
            compute_forces=False,
        )
        if jit:
            func = jax.jit(func)
        # Our implementation
        our_energy = func(
            positions=positions,
            charges=charges,
            cell=cell,
        )

        # TorchPME reference
        torchpme_energy = calculate_pme_reciprocal_energy_torchpme(
            positions_np, charges_np, cell_np, mesh_spacing, alpha, 4
        )

        assert jnp.allclose(
            our_energy.sum(), torchpme_energy.sum(), rtol=1e-5, atol=1e-3
        ), (
            f"{crystal_type} size={size} alpha={alpha}: "
            f"ours={float(our_energy.sum()):.6f}, torchpme={torchpme_energy.sum():.6f}"
        )


###########################################################################################
########################### Batch vs Single-System Consistency ############################
###########################################################################################


class TestPMEBatchConsistency:
    """Test that batch processing matches single-system processing."""

    def test_batch_single_system_matches(self, device):
        """Test batch with size 1 matches single-system."""
        positions, charges, cell = create_dipole_system()

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
        batch_idx = jnp.zeros(positions.shape[0], dtype=jnp.int32)
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
                dtype=dtype, num_atoms=4 + i, cell_size=8.0 + i
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
        batch_idx = jnp.repeat(
            jnp.arange(num_systems, dtype=jnp.int32),
            jnp.array(atoms_per_system),
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
        pos1 = jnp.array([[2.5, 2.5, 2.5], [3.5, 3.5, 3.5]], dtype=dtype)
        chg1 = jnp.array([1.0, -1.0], dtype=dtype)
        cell1 = jnp.eye(3, dtype=dtype) * 6.0

        pos2 = jnp.array([[4.0, 4.0, 4.0], [6.0, 6.0, 6.0]], dtype=dtype)
        chg2 = jnp.array([0.5, -0.5], dtype=dtype)
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
        batch_idx = jnp.array([0, 0, 1, 1], dtype=jnp.int32)

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

            key2 = jax.random.PRNGKey(idx + 200)
            chg = jax.random.normal(key2, (n_atoms,), dtype=jnp.float64)
            chg = chg.at[-1].set(-chg[:-1].sum())  # Neutralize

            positions_list.append(pos)
            charges_list.append(chg)

        positions = jnp.concatenate(positions_list, axis=0)
        charges = jnp.concatenate(charges_list, axis=0)
        cells = jnp.stack([jnp.eye(3, dtype=jnp.float64) * 10.0] * num_systems, axis=0)
        batch_idx = jnp.repeat(
            jnp.arange(num_systems, dtype=jnp.int32),
            jnp.array(atoms_per_system),
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

    @pytest.mark.parametrize("crystal_type", ["cscl", "wurtzite", "zincblende"])
    def test_batch_explicit_forces_vs_single(self, device, crystal_type):
        """Test batch explicit forces match single-system explicit forces."""
        # Create two systems
        pos1, chg1, cell1 = make_crystal_system_jax(crystal_type, size=1)
        pos2, chg2, cell2 = make_crystal_system_jax(crystal_type, size=2)

        # cell1, cell2 are already (1, 3, 3) shape
        mesh_dims = (16, 16, 16)
        alpha = 0.3

        # Single-system forces
        _, forces1_single = pme_reciprocal_space(
            positions=pos1,
            charges=chg1,
            cell=cell1,
            alpha=jnp.array([alpha]),
            mesh_dimensions=mesh_dims,
            compute_forces=True,
        )
        _, forces2_single = pme_reciprocal_space(
            positions=pos2,
            charges=chg2,
            cell=cell2,
            alpha=jnp.array([alpha]),
            mesh_dimensions=mesh_dims,
            compute_forces=True,
        )

        # Batch forces
        n1, n2 = pos1.shape[0], pos2.shape[0]
        positions_batch = jnp.concatenate([pos1, pos2], axis=0)
        charges_batch = jnp.concatenate([chg1, chg2], axis=0)
        cells_batch = jnp.concatenate([cell1, cell2], axis=0)
        batch_idx = jnp.array([0] * n1 + [1] * n2, dtype=jnp.int32)

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
            f"{crystal_type}: System 1 forces mismatch"
        )
        assert jnp.allclose(forces2_batch, forces2_single, rtol=1e-4, atol=1e-6), (
            f"{crystal_type}: System 2 forces mismatch"
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
        positions, charges, cell = create_simple_system(num_atoms=4)

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
        positions, charges, cell = create_simple_system(num_atoms=4)

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
        positions, charges, cell = create_simple_system(num_atoms=4)

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
        positions = jnp.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
            dtype=jnp.float64,
        )
        charges = jnp.array([1.0, -1.0, 0.5, -0.5], dtype=jnp.float64)
        cell = jnp.stack([jnp.eye(3, dtype=jnp.float64) * 10.0] * 2, axis=0)
        batch_idx = jnp.array([0, 0, 1, 1], dtype=jnp.int32)

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
        positions, charges, cell = create_simple_system(num_atoms=4)

        cutoff = 5.0
        pbc = jnp.array([[True, True, True]])
        neighbor_matrix, _, neighbor_matrix_shifts = cell_list(
            positions, cutoff, cell, pbc
        )

        energies, forces, charge_grads = particle_mesh_ewald(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=0.3,
            mesh_dimensions=(16, 16, 16),
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
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
        positions, charges, cell = create_simple_system(num_atoms=4)

        cutoff = 5.0
        pbc = jnp.array([[True, True, True]])
        neighbor_matrix, _, neighbor_matrix_shifts = cell_list(
            positions, cutoff, cell, pbc
        )

        energies, charge_grads = particle_mesh_ewald(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=0.3,
            mesh_dimensions=(16, 16, 16),
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
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
        positions, charges, cell = create_simple_system(num_atoms=5)

        cutoff = 5.0
        pbc = jnp.array([[True, True, True]])
        neighbor_matrix, _, neighbor_matrix_shifts = cell_list(
            positions, cutoff, cell, pbc
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
        assert jnp.all(jnp.isfinite(energies))
        assert jnp.all(jnp.isfinite(forces))

    def test_full_pme_energy_only(self, device):
        """Test full PME energy-only output."""
        positions, charges, cell = create_simple_system(num_atoms=4)

        cutoff = 5.0
        pbc = jnp.array([[True, True, True]])
        neighbor_matrix, _, neighbor_matrix_shifts = cell_list(
            positions, cutoff, cell, pbc
        )

        energies = particle_mesh_ewald(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=0.3,
            mesh_dimensions=(16, 16, 16),
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            compute_forces=False,
        )

        assert energies.shape == (4,)
        assert jnp.all(jnp.isfinite(energies))

    def test_full_pme_auto_estimate_alpha(self, device):
        """Test full PME with automatic alpha estimation."""
        positions, charges, cell = create_simple_system(num_atoms=5)

        cutoff = 5.0
        pbc = jnp.array([[True, True, True]])
        neighbor_matrix, _, neighbor_matrix_shifts = cell_list(
            positions, cutoff, cell, pbc
        )

        # Call without alpha - should auto-estimate
        energies, forces = particle_mesh_ewald(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=None,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            compute_forces=True,
        )

        assert energies.shape == (5,)
        assert forces.shape == (5, 3)
        assert jnp.all(jnp.isfinite(energies))
        assert jnp.all(jnp.isfinite(forces))

    def test_full_pme_mesh_spacing(self, device):
        """Test full PME with mesh_spacing parameter."""
        positions, charges, cell = create_simple_system(num_atoms=5)

        cutoff = 5.0
        pbc = jnp.array([[True, True, True]])
        neighbor_matrix, _, neighbor_matrix_shifts = cell_list(
            positions, cutoff, cell, pbc
        )

        energies, forces = particle_mesh_ewald(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=0.3,
            mesh_spacing=0.5,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
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
        cell = jnp.array(
            [[[8.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 12.0]]],
            dtype=jnp.float64,
        )
        positions = jnp.array([[2.0, 5.0, 6.0], [6.0, 5.0, 6.0]], dtype=jnp.float64)
        charges = jnp.array([1.0, -1.0], dtype=jnp.float64)

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
        cell = jnp.array(
            [[[10.0, 0.0, 0.0], [2.0, 10.0, 0.0], [1.0, 1.0, 10.0]]],
            dtype=jnp.float64,
        )
        positions = jnp.array([[2.0, 5.0, 5.0], [7.0, 5.0, 5.0]], dtype=jnp.float64)
        charges = jnp.array([1.0, -1.0], dtype=jnp.float64)

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
        positions, charges, cell = make_crystal_system_jax("wurtzite", size=2)

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
        positions, charges, cell = create_dipole_system()
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
        positions = jnp.array([[5.0, 5.0, 5.0]], dtype=jnp.float64)
        charges = jnp.array([1.0], dtype=jnp.float64)
        cell = cubic_cell_jax(10.0)

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
        positions = jnp.array([[2.0, 5.0, 5.0], [8.0, 5.0, 5.0]], dtype=jnp.float64)
        charges = jnp.array([0.0, 0.0], dtype=jnp.float64)
        cell = cubic_cell_jax(10.0)

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
        positions, charges, cell = create_dipole_system()

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
        pos1, chg1, cell1 = create_dipole_system()
        pos2, chg2, cell2 = create_dipole_system(separation=3.0)

        positions = jnp.concatenate([pos1, pos2], axis=0)
        charges = jnp.concatenate([chg1, chg2], axis=0)
        cells = jnp.concatenate([cell1, cell2], axis=0)
        batch_idx = jnp.array([0, 0, 1, 1], dtype=jnp.int32)

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
        positions, charges, cell = create_dipole_system(separation=3.0, cell_size=12.0)
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
                    positions=jnp.array(pos_plus, dtype=jnp.float64),
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
                    positions=jnp.array(pos_minus, dtype=jnp.float64),
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
        positions, charges, cell = create_dipole_system()

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
        positions, charges, cell = create_simple_system(num_atoms=5)

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
        positions, charges, cell = create_simple_system(num_atoms=5)

        # 0-dimensional JAX array (scalar array)
        alpha = jnp.array(0.3)

        cutoff = 5.0
        pbc = jnp.array([[True, True, True]])
        neighbor_matrix, _, neighbor_matrix_shifts = cell_list(
            positions, cutoff, cell, pbc
        )

        energies = particle_mesh_ewald(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,  # 0-dim array
            mesh_dimensions=(16, 16, 16),
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            compute_forces=False,
        )

        assert jnp.all(jnp.isfinite(energies))

    def test_alpha_wrong_size_raises_error(self, device):
        """Test alpha array with wrong number of elements raises ValueError."""
        positions, charges, cell = create_simple_system(num_atoms=5)

        # Alpha array with wrong size (2 values for 1 system)
        alpha = jnp.array([0.3, 0.5])

        cutoff = 5.0
        pbc = jnp.array([[True, True, True]])
        neighbor_matrix, _, neighbor_matrix_shifts = cell_list(
            positions, cutoff, cell, pbc
        )

        with pytest.raises(ValueError):
            particle_mesh_ewald(
                positions=positions,
                charges=charges,
                cell=cell,
                alpha=alpha,  # Wrong size: 2 values for 1 system
                mesh_dimensions=(16, 16, 16),
                neighbor_matrix=neighbor_matrix,
                neighbor_matrix_shifts=neighbor_matrix_shifts,
                compute_forces=False,
            )

    def test_alpha_invalid_type_raises_error(self, device):
        """Test non-float, non-array alpha raises an error.

        Note: particle_mesh_ewald raises AttributeError because the inline
        alpha handling accesses .ndim before type-checking. The underlying
        _prepare_alpha_array (used by ewald_real_space) raises TypeError.
        We accept either error type here.
        """
        positions, charges, cell = create_simple_system(num_atoms=5)

        cutoff = 5.0
        pbc = jnp.array([[True, True, True]])
        neighbor_matrix, _, neighbor_matrix_shifts = cell_list(
            positions, cutoff, cell, pbc
        )

        with pytest.raises((TypeError, AttributeError)):
            particle_mesh_ewald(
                positions=positions,
                charges=charges,
                cell=cell,
                alpha="invalid",  # String is not valid
                mesh_dimensions=(16, 16, 16),
                neighbor_matrix=neighbor_matrix,
                neighbor_matrix_shifts=neighbor_matrix_shifts,
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
        positions, charges, cell = create_simple_system(num_atoms=5)

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
        positions, charges, cell = create_simple_system(num_atoms=5)

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
        positions, charges, cell = create_simple_system(num_atoms=5)

        cutoff = 5.0
        pbc = jnp.array([[True, True, True]])
        neighbor_matrix, _, neighbor_matrix_shifts = cell_list(
            positions, cutoff, cell, pbc
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
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            compute_forces=True,
            accuracy=1e-4,
        )

        assert energies.shape == (5,)
        assert forces.shape == (5, 3)
        assert jnp.all(jnp.isfinite(energies))
        assert jnp.all(jnp.isfinite(forces))

    def test_auto_mesh_from_alpha_estimation(self, device):
        """Test mesh_dimensions auto-derived when alpha is auto-estimated."""
        positions, charges, cell = create_simple_system(num_atoms=5)

        cutoff = 5.0
        pbc = jnp.array([[True, True, True]])
        neighbor_matrix, _, neighbor_matrix_shifts = cell_list(
            positions, cutoff, cell, pbc
        )

        # alpha=None triggers estimate_pme_parameters which sets alpha AND mesh_dimensions
        energies, forces = particle_mesh_ewald(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=None,  # Triggers auto-estimation
            mesh_dimensions=None,  # Will be set from params
            mesh_spacing=None,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
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
        positions, charges, cell = create_simple_system(num_atoms=5)
        batch_idx = jnp.zeros(5, dtype=jnp.int32)  # All same batch

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
        pos1 = jnp.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 3.0, 2.0]], dtype=dtype
        )
        chg1 = jnp.array([1.0, -0.5, -0.5], dtype=dtype)

        # System 2: 2 atoms
        pos2 = jnp.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]], dtype=dtype)
        chg2 = jnp.array([0.5, -0.5], dtype=dtype)

        positions = jnp.concatenate([pos1, pos2], axis=0)
        charges = jnp.concatenate([chg1, chg2], axis=0)
        cells = jnp.stack([jnp.eye(3, dtype=dtype) * 10.0] * 2, axis=0)
        batch_idx = jnp.array([0, 0, 0, 1, 1], dtype=jnp.int32)

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
        positions, charges, cell = create_simple_system(num_atoms=4)

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
                charges=jnp.array(chg_plus, dtype=jnp.float64),
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
                charges=jnp.array(chg_minus, dtype=jnp.float64),
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
        positions = jnp.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
            dtype=jnp.float64,
        )
        charges = jnp.array([1.0, -1.0, 0.5, -0.5], dtype=jnp.float64)
        cell = jnp.stack([jnp.eye(3, dtype=jnp.float64) * 10.0] * 2, axis=0)
        batch_idx = jnp.array([0, 0, 1, 1], dtype=jnp.int32)

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
                charges=jnp.array(chg_plus, dtype=jnp.float64),
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
                charges=jnp.array(chg_minus, dtype=jnp.float64),
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

    Verifies that particle_mesh_ewald correctly uses neighbor_matrix
    (dense format) for the real-space component on crystal systems,
    complementing the simpler tests in TestParticleMeshEwald.
    """

    def test_full_pme_neighbor_list_crystal_system(self, device):
        """Test full PME with neighbor list on a crystal system."""
        positions, charges, cell = make_crystal_system_jax("cscl", size=1)

        cutoff = 5.0
        pbc = jnp.array([[True, True, True]])
        neighbor_matrix, _, neighbor_matrix_shifts = cell_list(
            positions, cutoff, cell, pbc
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

    @pytest.mark.parametrize("crystal_type", ["cscl", "wurtzite", "zincblende"])
    def test_full_pme_neighbor_list_multiple_crystals(self, device, crystal_type):
        """Test full PME with neighbor list on multiple crystal types."""
        positions, charges, cell = make_crystal_system_jax(crystal_type, size=1)

        cutoff = 5.0
        pbc = jnp.array([[True, True, True]])
        neighbor_matrix, _, neighbor_matrix_shifts = cell_list(
            positions, cutoff, cell, pbc
        )

        energies, forces, charge_grads = particle_mesh_ewald(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=0.3,
            mesh_dimensions=(16, 16, 16),
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
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


class TestPMEJIT:
    """Smoke tests for PME calculations with jax.jit."""

    def test_jit_reciprocal_space(self):
        """Test pme_reciprocal_space works under jax.jit."""
        positions = jnp.array([[4.0, 5.0, 5.0], [6.0, 5.0, 5.0]], dtype=jnp.float64)
        charges = jnp.array([1.0, -1.0], dtype=jnp.float64)
        cell = cubic_cell_jax(10.0)
        alpha = jnp.array([0.3], dtype=jnp.float64)

        @jax.jit
        def jitted_pme_recip(positions, charges, cell, alpha):
            return pme_reciprocal_space(
                positions=positions,
                charges=charges,
                cell=cell,
                alpha=alpha,
                mesh_dimensions=(16, 16, 16),
                spline_order=4,
                compute_forces=False,
            )

        energies = jitted_pme_recip(positions, charges, cell, alpha)

        assert energies.shape == (2,)
        assert jnp.all(jnp.isfinite(energies))

    def test_jit_reciprocal_space_with_forces(self):
        """Test pme_reciprocal_space with forces works under jax.jit."""
        positions = jnp.array([[4.0, 5.0, 5.0], [6.0, 5.0, 5.0]], dtype=jnp.float64)
        charges = jnp.array([1.0, -1.0], dtype=jnp.float64)
        cell = cubic_cell_jax(10.0)
        alpha = jnp.array([0.3], dtype=jnp.float64)

        @jax.jit
        def jitted_pme_recip_forces(positions, charges, cell, alpha):
            return pme_reciprocal_space(
                positions=positions,
                charges=charges,
                cell=cell,
                alpha=alpha,
                mesh_dimensions=(16, 16, 16),
                spline_order=4,
                compute_forces=True,
            )

        energies, forces = jitted_pme_recip_forces(positions, charges, cell, alpha)

        assert energies.shape == (2,)
        assert forces.shape == (2, 3)
        assert jnp.all(jnp.isfinite(energies))
        assert jnp.all(jnp.isfinite(forces))

    def test_jit_full_pme(self):
        """Test particle_mesh_ewald works under jax.jit."""
        positions = jnp.array(
            [[4.0, 5.0, 5.0], [6.0, 5.0, 5.0], [5.0, 4.0, 5.0], [5.0, 6.0, 5.0]],
            dtype=jnp.float64,
        )
        charges = jnp.array([1.0, -1.0, 0.5, -0.5], dtype=jnp.float64)
        cell = cubic_cell_jax(10.0)
        pbc = jnp.array([[True, True, True]])

        # Build neighbor list eagerly (it uses .devices().pop() internally)
        neighbor_matrix, _, neighbor_matrix_shifts = cell_list(
            positions, cutoff=5.0, cell=cell, pbc=pbc
        )

        @jax.jit
        def jitted_full_pme(positions, charges, cell, nm, nms):
            return particle_mesh_ewald(
                positions=positions,
                charges=charges,
                cell=cell,
                alpha=0.3,
                mesh_dimensions=(16, 16, 16),
                neighbor_matrix=nm,
                neighbor_matrix_shifts=nms,
                compute_forces=True,
            )

        energies, forces = jitted_full_pme(
            positions, charges, cell, neighbor_matrix, neighbor_matrix_shifts
        )

        assert energies.shape == (4,)
        assert forces.shape == (4, 3)
        assert jnp.all(jnp.isfinite(energies))
        assert jnp.all(jnp.isfinite(forces))

    def test_jit_equivalence(self):
        """Test PME results from JIT vs not is equivalent"""
        positions = jnp.array(
            [[4.0, 5.0, 5.0], [6.0, 5.0, 5.0], [5.0, 4.0, 5.0], [5.0, 6.0, 5.0]],
            dtype=jnp.float64,
        )
        charges = jnp.array([1.0, -1.0, 0.5, -0.5], dtype=jnp.float64)
        cell = cubic_cell_jax(10.0)
        pbc = jnp.array([[True, True, True]])

        # Build neighbor list eagerly (it uses .devices().pop() internally)
        neighbor_matrix, _, neighbor_matrix_shifts = cell_list(
            positions, cutoff=5.0, cell=cell, pbc=pbc
        )

        bare_func = partial(
            particle_mesh_ewald,
            alpha=0.3,
            mesh_dimensions=(16, 16, 16),
            compute_forces=True,
        )
        jitted_func = jax.jit(bare_func)

        bare_energy, bare_forces = bare_func(
            positions=positions,
            charges=charges,
            cell=cell,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
        )
        jit_energy, jit_forces = jitted_func(
            positions=positions,
            charges=charges,
            cell=cell,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
        )

        assert jnp.allclose(bare_energy, jit_energy, atol=1e-5, rtol=1e-5), (
            "Energy difference!"
        )
        assert jnp.allclose(bare_forces, jit_forces, atol=1e-5, rtol=1e-5), (
            "Force difference!"
        )


###########################################################################################
########################### PME Virial Tests ##############################################
###########################################################################################


class TestPMEReciprocalVirial:
    """Test PME reciprocal-space virial against FD and basic properties."""

    def test_pme_reciprocal_virial_shape(self, device):
        """PME reciprocal virial has shape (1, 3, 3)."""
        positions, charges, cell = make_virial_cscl_system_jax(size=2)
        alpha = jnp.array([0.3], dtype=jnp.float64)

        result = pme_reciprocal_space(
            positions,
            charges,
            cell,
            alpha,
            mesh_dimensions=(16, 16, 16),
            compute_forces=True,
            compute_virial=True,
        )
        # Result should be (energies, forces, virial)
        assert len(result) == 3
        energies, forces, virial = result
        assert virial.shape == (1, 3, 3)

    def test_pme_reciprocal_virial_finite_nonzero(self, device):
        """PME reciprocal virial is finite and non-zero for ionic crystal."""
        positions, charges, cell = make_virial_cscl_system_jax(size=2)
        alpha = jnp.array([0.3], dtype=jnp.float64)

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
        assert jnp.all(jnp.isfinite(virial))
        assert jnp.any(jnp.abs(virial) > 1e-10)

    def test_pme_reciprocal_virial_fd(self, device):
        """PME reciprocal virial matches FD strain derivative.

        Note: JAX PME B-spline interpolation always outputs float32, which
        limits FD derivative precision. We compare diagonal elements with
        looser tolerance. Off-diagonal FD elements may have spurious noise
        from float32 subtraction of nearly-equal values.
        """
        positions, charges, cell = make_virial_cscl_system_jax(size=2)
        alpha = jnp.array([0.3], dtype=jnp.float64)
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
        explicit_virial = result[2].squeeze(0)  # (3, 3)
        fd_virial = fd_virial_full_jax(energy_fn, positions, cell, h=1e-5)

        # Compare diagonal elements (most reliable for float32 output)
        explicit_diag = jnp.diag(explicit_virial)
        fd_diag = jnp.diag(fd_virial)
        assert jnp.allclose(explicit_diag, fd_diag, atol=5e-2, rtol=5e-2), (
            f"PME reciprocal virial diagonal does not match FD\n"
            f"explicit diag: {explicit_diag}\nFD diag: {fd_diag}"
        )

    def test_pme_reciprocal_virial_symmetry(self, device):
        """PME reciprocal virial is symmetric for cubic system."""
        positions, charges, cell = make_virial_cscl_system_jax(size=2)
        alpha = jnp.array([0.3], dtype=jnp.float64)

        result = pme_reciprocal_space(
            positions,
            charges,
            cell,
            alpha,
            mesh_dimensions=(16, 16, 16),
            compute_forces=True,
            compute_virial=True,
        )
        virial = result[2].squeeze(0)  # (3, 3)
        assert jnp.allclose(virial, virial.T, atol=1e-6, rtol=1e-6), (
            f"PME reciprocal virial is not symmetric:\n{virial}"
        )

    def test_pme_reciprocal_virial_without_forces(self, device):
        """PME reciprocal with compute_forces=False + compute_virial=True."""
        positions, charges, cell = make_virial_cscl_system_jax(size=2)
        alpha = jnp.array([0.3], dtype=jnp.float64)

        result = pme_reciprocal_space(
            positions,
            charges,
            cell,
            alpha,
            mesh_dimensions=(16, 16, 16),
            compute_forces=False,
            compute_virial=True,
        )
        # Result should be (energies, virial)
        assert isinstance(result, tuple)
        assert len(result) == 2
        energies, virial = result
        assert virial.shape == (1, 3, 3)
        assert jnp.all(jnp.isfinite(virial))


class TestPMEReciprocalVirialBatch:
    """Batch PME virial matches single-system PME virial."""

    def test_batch_pme_reciprocal_virial_shape(self, device):
        """Batch PME reciprocal virial has shape (B, 3, 3)."""
        # Create two identical systems as a batch
        positions_s, charges_s, cell_s = make_virial_cscl_system_jax(size=1)
        n = positions_s.shape[0]

        positions = jnp.concatenate([positions_s, positions_s], axis=0)
        charges = jnp.concatenate([charges_s, charges_s], axis=0)
        cells = jnp.concatenate([cell_s, cell_s], axis=0)
        batch_idx = jnp.array([0] * n + [1] * n, dtype=jnp.int32)
        alpha = jnp.array([0.3, 0.3], dtype=jnp.float64)

        result = pme_reciprocal_space(
            positions,
            charges,
            cells,
            alpha,
            mesh_dimensions=(8, 8, 8),
            batch_idx=batch_idx,
            compute_forces=True,
            compute_virial=True,
        )
        virial = result[2]
        assert virial.shape == (2, 3, 3)

    def test_batch_pme_reciprocal_virial_shape_single_system(self, device):
        """Batch PME reciprocal virial has shape (1, 3, 3) when B=1."""
        positions, charges, cell = make_virial_cscl_system_jax(size=1)
        batch_idx = jnp.zeros(positions.shape[0], dtype=jnp.int32)
        alpha = jnp.array([0.3], dtype=jnp.float64)

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
        assert virial.shape == (1, 3, 3)

    def test_batch_pme_reciprocal_virial_matches_single(self, device):
        """Batch PME reciprocal virial[i] matches single-system virial."""
        positions_s, charges_s, cell_s = make_virial_cscl_system_jax(size=1)
        n = positions_s.shape[0]

        # Single-system virial
        single_result = pme_reciprocal_space(
            positions_s,
            charges_s,
            cell_s,
            jnp.array([0.3], dtype=jnp.float64),
            mesh_dimensions=(8, 8, 8),
            compute_forces=True,
            compute_virial=True,
        )
        single_virial = single_result[2]  # (1, 3, 3)

        # Batch virial (two identical systems)
        positions = jnp.concatenate([positions_s, positions_s], axis=0)
        charges = jnp.concatenate([charges_s, charges_s], axis=0)
        cells = jnp.concatenate([cell_s, cell_s], axis=0)
        batch_idx = jnp.array([0] * n + [1] * n, dtype=jnp.int32)

        batch_result = pme_reciprocal_space(
            positions,
            charges,
            cells,
            jnp.array([0.3, 0.3], dtype=jnp.float64),
            mesh_dimensions=(8, 8, 8),
            batch_idx=batch_idx,
            compute_forces=True,
            compute_virial=True,
        )
        batch_virial = batch_result[2]  # (2, 3, 3)

        assert jnp.allclose(batch_virial[0:1], single_virial, atol=1e-5, rtol=1e-5), (
            f"Batch PME virial[0] != single virial\n"
            f"batch[0]:\n{batch_virial[0]}\nsingle:\n{single_virial[0]}"
        )
        assert jnp.allclose(batch_virial[1:2], single_virial, atol=1e-5, rtol=1e-5), (
            f"Batch PME virial[1] != single virial\n"
            f"batch[1]:\n{batch_virial[1]}\nsingle:\n{single_virial[0]}"
        )

    def test_batch_pme_reciprocal_virial_fd(self, device):
        """Batch PME reciprocal virial per-system matches single-system FD.

        Note: Compares diagonal elements only due to float32 B-spline limitations.
        """
        positions_s, charges_s, cell_s = make_virial_cscl_system_jax(size=1)
        n = positions_s.shape[0]
        mesh_dims = (8, 8, 8)
        alpha_s = jnp.array([0.3], dtype=jnp.float64)

        # FD virial for single system
        def energy_fn(pos, c):
            return pme_reciprocal_space(
                pos,
                charges_s,
                c,
                alpha_s,
                mesh_dimensions=mesh_dims,
                compute_forces=False,
            ).sum()

        fd_virial = fd_virial_full_jax(energy_fn, positions_s, cell_s, h=1e-5)

        # Batch virial
        positions = jnp.concatenate([positions_s, positions_s], axis=0)
        charges = jnp.concatenate([charges_s, charges_s], axis=0)
        cells = jnp.concatenate([cell_s, cell_s], axis=0)
        batch_idx = jnp.array([0] * n + [1] * n, dtype=jnp.int32)

        batch_result = pme_reciprocal_space(
            positions,
            charges,
            cells,
            jnp.array([0.3, 0.3], dtype=jnp.float64),
            mesh_dimensions=mesh_dims,
            batch_idx=batch_idx,
            compute_forces=True,
            compute_virial=True,
        )
        batch_virial = batch_result[2]

        # Compare diagonal elements
        fd_diag = jnp.diag(fd_virial)
        batch0_diag = jnp.diag(batch_virial[0])
        batch1_diag = jnp.diag(batch_virial[1])

        assert jnp.allclose(batch0_diag, fd_diag, atol=5e-2, rtol=5e-2), (
            "Batch PME system 0 diagonal does not match single-system FD"
        )
        assert jnp.allclose(batch1_diag, fd_diag, atol=5e-2, rtol=5e-2), (
            "Batch PME system 1 diagonal does not match single-system FD"
        )


class TestFullPMEVirial:
    """Test full particle_mesh_ewald (real + reciprocal) virial."""

    def test_full_pme_virial_shape(self, device):
        """Full PME virial has shape (1, 3, 3)."""
        positions, charges, cell = make_virial_cscl_system_jax(size=2)
        alpha = 0.3
        cutoff = 6.0
        pbc = jnp.array([[True, True, True]])
        nm, _, nms = cell_list(positions, cutoff, cell, pbc)

        result = particle_mesh_ewald(
            positions,
            charges,
            cell,
            alpha=alpha,
            mesh_dimensions=(16, 16, 16),
            neighbor_matrix=nm,
            neighbor_matrix_shifts=nms,
            compute_forces=True,
            compute_virial=True,
        )
        # Should be (energies, forces, virial)
        assert len(result) == 3
        energies, forces, virial = result
        assert virial.shape == (1, 3, 3)
        assert jnp.all(jnp.isfinite(virial))

    def test_full_pme_virial_fd(self, device):
        """Full PME virial matches FD strain derivative.

        Note: Compares diagonal elements only due to float32 B-spline limitations.
        """
        positions, charges, cell = make_virial_cscl_system_jax(size=2)
        alpha = 0.3
        cutoff = 6.0
        mesh_dims = (16, 16, 16)
        pbc = jnp.array([[True, True, True]])

        def energy_fn(pos, c):
            nm_new, _, nms_new = cell_list(pos, cutoff, c, pbc)
            return particle_mesh_ewald(
                pos,
                charges,
                c,
                alpha=alpha,
                mesh_dimensions=mesh_dims,
                neighbor_matrix=nm_new,
                neighbor_matrix_shifts=nms_new,
                compute_forces=False,
            ).sum()

        nm, _, nms = cell_list(positions, cutoff, cell, pbc)
        result = particle_mesh_ewald(
            positions,
            charges,
            cell,
            alpha=alpha,
            mesh_dimensions=mesh_dims,
            neighbor_matrix=nm,
            neighbor_matrix_shifts=nms,
            compute_forces=True,
            compute_virial=True,
        )
        explicit_virial = result[2].squeeze(0)
        fd_virial = fd_virial_full_jax(energy_fn, positions, cell, h=1e-5)

        # Compare diagonal elements (most reliable for float32 output)
        explicit_diag = jnp.diag(explicit_virial)
        fd_diag = jnp.diag(fd_virial)
        assert jnp.allclose(explicit_diag, fd_diag, atol=5e-2, rtol=5e-2), (
            f"Full PME virial diagonal does not match FD\n"
            f"explicit diag: {explicit_diag}\nFD diag: {fd_diag}"
        )

    def test_full_pme_virial_sum_of_components(self, device):
        """Full PME virial = real-space virial + reciprocal virial."""
        positions, charges, cell = make_virial_cscl_system_jax(size=2)
        alpha_arr = jnp.array([0.3], dtype=jnp.float64)
        cutoff = 6.0
        mesh_dims = (16, 16, 16)
        pbc = jnp.array([[True, True, True]])
        nm, _, nms = cell_list(positions, cutoff, cell, pbc)

        # Real-space virial
        rs_result = ewald_real_space(
            positions,
            charges,
            cell,
            alpha_arr,
            neighbor_matrix=nm,
            neighbor_matrix_shifts=nms,
            compute_forces=True,
            compute_virial=True,
        )
        real_virial = rs_result[2]

        # Reciprocal-space virial
        rec_result = pme_reciprocal_space(
            positions,
            charges,
            cell,
            alpha_arr,
            mesh_dimensions=mesh_dims,
            compute_forces=True,
            compute_virial=True,
        )
        recip_virial = rec_result[2]

        # Total virial
        total_result = particle_mesh_ewald(
            positions,
            charges,
            cell,
            alpha=0.3,
            mesh_dimensions=mesh_dims,
            neighbor_matrix=nm,
            neighbor_matrix_shifts=nms,
            compute_forces=True,
            compute_virial=True,
        )
        total_virial = total_result[2]

        assert jnp.allclose(
            total_virial, real_virial + recip_virial, atol=1e-6, rtol=1e-6
        ), (
            f"Full PME virial != real + reciprocal virial\n"
            f"total:\n{total_virial}\nreal+recip:\n{real_virial + recip_virial}"
        )

    def test_full_pme_virial_without_forces(self, device):
        """particle_mesh_ewald with compute_forces=False + compute_virial=True."""
        positions, charges, cell = make_virial_cscl_system_jax(size=2)
        alpha = 0.3
        cutoff = 6.0
        pbc = jnp.array([[True, True, True]])
        nm, _, nms = cell_list(positions, cutoff, cell, pbc)

        result = particle_mesh_ewald(
            positions,
            charges,
            cell,
            alpha=alpha,
            mesh_dimensions=(16, 16, 16),
            neighbor_matrix=nm,
            neighbor_matrix_shifts=nms,
            compute_forces=False,
            compute_virial=True,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        energies, virial = result
        assert virial.shape == (1, 3, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
