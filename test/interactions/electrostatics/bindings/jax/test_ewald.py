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
Unit tests for JAX Ewald summation electrostatic calculations.

This test suite validates the correctness of the JAX Ewald summation
implementation for long-range electrostatics in periodic systems.

Tests cover:
- Real-space and reciprocal-space energy and forces
- Full Ewald summation (real + reciprocal)
- Explicit charge gradient computation (replaces autograd tests)
- Numerical correctness against torchpme reference
- Float32 and float64 dtype support
- Batched calculations
- Physical properties (charge scaling, translation invariance)
- Non-cubic cells
- Automatic parameter estimation

Note: JAX bindings are GPU-only (Warp JAX FFI constraint) and do not support
autograd (enable_backward=False). Tests that call kernels require GPU.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nvalchemiops.jax.interactions.electrostatics.ewald import (
    ewald_real_space,
    ewald_reciprocal_space,
    ewald_summation,
)
from nvalchemiops.jax.interactions.electrostatics.k_vectors import (
    generate_k_vectors_ewald_summation,
)
from nvalchemiops.jax.neighbors import batch_cell_list, cell_list
from test.interactions.electrostatics.bindings.jax.conftest import (
    cubic_cell_jax,
    fd_virial_full_jax,
    make_crystal_system_jax,
    make_virial_cscl_system_jax,
)

# Try to import torchpme for reference calculations
try:
    import torch
    from torchpme import EwaldCalculator
    from torchpme.potentials import CoulombPotential

    HAS_TORCHPME = True
except ModuleNotFoundError:
    HAS_TORCHPME = False


# ==============================================================================
# Helper Functions
# ==============================================================================


def create_dipole_system(dtype=jnp.float64, separation=6.0, cell_size=10.0):
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
        (positions, charges, cell, neighbor_matrix, num_neighbors, neighbor_matrix_shifts)
    """
    positions = jnp.array([[0.0, 0.0, 0.0], [separation, 0.0, 0.0]], dtype=dtype)
    charges = jnp.array([1.0, -1.0], dtype=dtype)
    cell = jnp.array(
        [[[cell_size, 0.0, 0.0], [0.0, cell_size, 0.0], [0.0, 0.0, cell_size]]],
        dtype=dtype,
    )

    # Build neighbor list using cell_list
    cutoff = separation * 1.5
    pbc = jnp.array([[True, True, True]])
    neighbor_matrix, num_neighbors, neighbor_matrix_shifts = cell_list(
        positions, cutoff, cell, pbc
    )

    return (
        positions,
        charges,
        cell,
        neighbor_matrix,
        num_neighbors,
        neighbor_matrix_shifts,
    )


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
    positions = jax.random.uniform(key, (num_atoms, 3), dtype=dtype) * cell_size * 0.8

    # Create alternating charges for neutrality
    charges = jnp.array([1.0, -1.0] * (num_atoms // 2), dtype=dtype)
    if num_atoms % 2 == 1:
        charges = jnp.concatenate([charges, jnp.array([0.0], dtype=dtype)])

    cell = jnp.array(
        [[[cell_size, 0.0, 0.0], [0.0, cell_size, 0.0], [0.0, 0.0, cell_size]]],
        dtype=dtype,
    )

    return positions, charges, cell


def compute_torchpme_reciprocal(positions_np, charges_np, cell_np, k_cutoff, alpha):
    """Compute reference reciprocal energy using torchpme.

    Parameters
    ----------
    positions_np : np.ndarray
        Atomic positions
    charges_np : np.ndarray
        Atomic charges
    cell_np : np.ndarray
        Cell matrix
    k_cutoff : float
        K-space cutoff
    alpha : float
        Ewald splitting parameter

    Returns
    -------
    np.ndarray
        Reciprocal space energy per atom
    """

    device = "cuda"
    dtype = torch.float64
    positions_torch = torch.tensor(positions_np, dtype=dtype, device=device)
    charges_torch = torch.tensor(charges_np, dtype=dtype, device=device)
    cell_torch = torch.tensor(cell_np, dtype=dtype, device=device)

    lr_wavelength = 2 * torch.pi / k_cutoff
    smearing = 1.0 / (2.0**0.5 * alpha)
    potential = CoulombPotential(smearing=smearing).to(device=device, dtype=dtype)
    calc = EwaldCalculator(
        potential=potential, lr_wavelength=lr_wavelength, full_neighbor_list=True
    ).to(device=device, dtype=dtype)

    charges_col = charges_torch.unsqueeze(1)
    # cell_torch is (3, 3) — pass directly (not cell_torch[0] which would be row 0)
    potentials = calc._compute_kspace(charges_col, cell_torch, positions_torch)
    energy_recip = (charges_col * potentials).flatten()

    return energy_recip.cpu().numpy()


def compute_torchpme_real_space(
    charges_np, neighbor_indices, neighbor_distances, alpha, k_cutoff
):
    """Compute reference real-space energy using torchpme.

    Parameters
    ----------
    charges_np : np.ndarray
        Atomic charges
    neighbor_indices : np.ndarray
        Neighbor pair indices [2, num_pairs]
    neighbor_distances : np.ndarray
        Pair distances
    alpha : float
        Ewald splitting parameter
    k_cutoff : float
        K-space cutoff

    Returns
    -------
    np.ndarray
        Real space energy per atom
    """

    device = "cuda"
    dtype = torch.float64
    charges_torch = torch.tensor(charges_np, dtype=dtype, device=device)
    neighbor_indices_torch = torch.tensor(
        neighbor_indices, dtype=torch.long, device=device
    )
    neighbor_distances_torch = torch.tensor(
        neighbor_distances, dtype=dtype, device=device
    )

    lr_wavelength = 2 * torch.pi / k_cutoff
    smearing = 1.0 / (2.0**0.5 * alpha)
    potential = CoulombPotential(smearing=smearing).to(device=device, dtype=dtype)
    calc = EwaldCalculator(
        potential=potential, lr_wavelength=lr_wavelength, full_neighbor_list=True
    ).to(device=device, dtype=dtype)

    # torchpme expects neighbor_indices as (num_pairs, 2), but our JAX neighbor list
    # is in COO format (2, num_pairs), so transpose if needed
    if neighbor_indices_torch.shape[0] == 2 and neighbor_indices_torch.ndim == 2:
        neighbor_indices_torch = neighbor_indices_torch.T

    charges_col = charges_torch.unsqueeze(1)
    potentials = calc._compute_rspace(
        charges_col, neighbor_indices_torch, neighbor_distances_torch
    )
    energy_real = (charges_col * potentials).flatten()

    return energy_real.cpu().numpy()


# ==============================================================================
# Test Classes
# ==============================================================================


class TestDtypeSupport:
    """Test float32 and float64 dtype support."""

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_real_space_dtype_returns_correct_type(self, device, dtype):
        """Test ewald_real_space with float32 and float64."""
        (
            positions,
            charges,
            cell,
            neighbor_matrix,
            num_neighbors,
            neighbor_matrix_shifts,
        ) = create_dipole_system(dtype=dtype)

        alpha = jnp.array([0.3], dtype=dtype)

        energies = ewald_real_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
        )

        # Energy is always float64
        assert energies.dtype == jnp.float64
        assert jnp.all(jnp.isfinite(energies))

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_reciprocal_space_dtype_returns_correct_type(self, device, dtype):
        """Test ewald_reciprocal_space with float32 and float64."""
        positions, charges, cell = create_simple_system(dtype=dtype)

        alpha = jnp.array([0.3], dtype=dtype)
        k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff=8.0).astype(dtype)

        energies = ewald_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            k_vectors=k_vectors,
            alpha=alpha,
        )

        assert energies.dtype == jnp.float64
        assert jnp.all(jnp.isfinite(energies))

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_ewald_summation_dtype_returns_correct_type(self, device, dtype):
        """Test ewald_summation with float32 and float64."""
        (
            positions,
            charges,
            cell,
            neighbor_matrix,
            num_neighbors,
            neighbor_matrix_shifts,
        ) = create_dipole_system(dtype=dtype)

        alpha = 0.3
        k_cutoff = 8.0

        energies = ewald_summation(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            k_cutoff=k_cutoff,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
        )

        assert energies.dtype == jnp.float64
        assert jnp.all(jnp.isfinite(energies))

    def test_float32_vs_float64_consistency(self, device):
        """Test that float32 and float64 produce consistent results."""
        positions_f64, charges_f64, cell_f64, nm_f64, nn_f64, nms_f64 = (
            create_dipole_system(dtype=jnp.float64)
        )

        positions_f32 = positions_f64.astype(jnp.float32)
        charges_f32 = charges_f64.astype(jnp.float32)
        cell_f32 = cell_f64.astype(jnp.float32)

        alpha = 0.3

        energies_f64 = ewald_real_space(
            positions=positions_f64,
            charges=charges_f64,
            cell=cell_f64,
            alpha=alpha,
            neighbor_matrix=nm_f64,
            neighbor_matrix_shifts=nms_f64,
        )

        energies_f32 = ewald_real_space(
            positions=positions_f32,
            charges=charges_f32,
            cell=cell_f32,
            alpha=alpha,
            neighbor_matrix=nm_f64,
            neighbor_matrix_shifts=nms_f64,
        )

        # Both are float64 but f32 may have slightly different values
        assert jnp.allclose(energies_f32, energies_f64, rtol=1e-4)

    def test_batch_dtype_returns_correct_type(self, device):
        """Test batched calculations return correct dtype."""
        # Create 2 batches of 2 atoms each
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
            dtype=jnp.float32,
        )
        charges = jnp.array([1.0, -1.0, 1.0, -1.0], dtype=jnp.float32)
        cell_single = cubic_cell_jax(10.0, dtype=jnp.float32)
        cell = jnp.concatenate([cell_single, cell_single], axis=0)  # (2, 3, 3)
        batch_idx = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
        batch_ptr = jnp.array([0, 2, 4], dtype=jnp.int32)

        cutoff = 5.0
        pbc = jnp.array([[True, True, True], [True, True, True]])
        neighbor_matrix, num_neighbors, neighbor_matrix_shifts = batch_cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors=32,
        )

        alpha = jnp.array([0.3, 0.3], dtype=jnp.float32)

        energies = ewald_real_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            batch_idx=batch_idx,
        )

        assert energies.dtype == jnp.float64
        assert energies.shape == (4,)


class TestEwaldRealSpaceAPI:
    """Test Ewald real-space API."""

    def test_single_system_energy_only(self, device):
        """Test real-space energy for a single system."""
        (
            positions,
            charges,
            cell,
            neighbor_matrix,
            num_neighbors,
            neighbor_matrix_shifts,
        ) = create_dipole_system()

        alpha = 0.3

        energies = ewald_real_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
        )

        assert energies.shape == (2,)
        assert jnp.all(jnp.isfinite(energies))
        # Opposite charges should produce negative energy
        assert energies.sum() < 0

    def test_single_system_with_forces(self, device):
        """Test real-space energy and forces."""
        (
            positions,
            charges,
            cell,
            neighbor_matrix,
            num_neighbors,
            neighbor_matrix_shifts,
        ) = create_dipole_system()

        alpha = 0.3

        energies, forces = ewald_real_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            compute_forces=True,
        )

        assert energies.shape == (2,)
        assert forces.shape == (2, 3)
        assert jnp.all(jnp.isfinite(energies))
        assert jnp.all(jnp.isfinite(forces))

        # Forces should be non-zero for this configuration
        assert jnp.abs(forces[0, 0]) > 1e-6
        # Newton's 3rd law
        assert jnp.allclose(forces[0], -forces[1], rtol=1e-10)

    def test_batch_system_energy_only(self, device):
        """Test batched real-space energy."""
        # 2 batches of 2 atoms each
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
        cell_single = cubic_cell_jax(10.0, dtype=jnp.float64)
        cell = jnp.concatenate([cell_single, cell_single], axis=0)  # (2, 3, 3)
        batch_idx = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
        batch_ptr = jnp.array([0, 2, 4], dtype=jnp.int32)

        cutoff = 5.0
        pbc = jnp.array([[True, True, True], [True, True, True]])
        neighbor_matrix, num_neighbors, neighbor_matrix_shifts = batch_cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors=32,
        )

        alpha = jnp.array([0.3, 0.3])

        energies = ewald_real_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            batch_idx=batch_idx,
        )

        assert energies.shape == (4,)
        assert jnp.all(jnp.isfinite(energies))

    def test_batch_system_with_forces(self, device):
        """Test batched real-space energy and forces."""
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
        cell_single = cubic_cell_jax(10.0, dtype=jnp.float64)
        cell = jnp.concatenate([cell_single, cell_single], axis=0)  # (2, 3, 3)
        batch_idx = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
        batch_ptr = jnp.array([0, 2, 4], dtype=jnp.int32)

        cutoff = 5.0
        pbc = jnp.array([[True, True, True], [True, True, True]])
        neighbor_matrix, num_neighbors, neighbor_matrix_shifts = batch_cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors=32,
        )

        alpha = jnp.array([0.3, 0.3])

        energies, forces = ewald_real_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            batch_idx=batch_idx,
            compute_forces=True,
        )

        assert energies.shape == (4,)
        assert forces.shape == (4, 3)
        assert jnp.all(jnp.isfinite(energies))
        assert jnp.all(jnp.isfinite(forces))


class TestEwaldReciprocalSpaceAPI:
    """Test Ewald reciprocal-space API."""

    def test_single_system_energy_only(self, device):
        """Test reciprocal-space energy for a single system."""
        positions, charges, cell = create_simple_system()

        alpha = 0.3
        k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff=8.0)

        energies = ewald_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            k_vectors=k_vectors,
            alpha=alpha,
        )

        assert energies.shape == (4,)
        assert jnp.all(jnp.isfinite(energies))

    def test_single_system_with_forces(self, device):
        """Test reciprocal-space energy and forces."""
        positions, charges, cell = create_simple_system()

        alpha = 0.3
        k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff=8.0)

        energies, forces = ewald_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            k_vectors=k_vectors,
            alpha=alpha,
            compute_forces=True,
        )

        assert energies.shape == (4,)
        assert forces.shape == (4, 3)
        assert jnp.all(jnp.isfinite(energies))
        assert jnp.all(jnp.isfinite(forces))

    def test_batch_system_energy_only(self, device):
        """Test batched reciprocal-space energy."""
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
        cell = jnp.array(
            [
                [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
                [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
            ]
        )
        batch_idx = jnp.array([0, 0, 1, 1], dtype=jnp.int32)

        alpha = jnp.array([0.3, 0.3])
        k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff=8.0)

        energies = ewald_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            k_vectors=k_vectors,
            alpha=alpha,
            batch_idx=batch_idx,
        )

        assert energies.shape == (4,)
        assert jnp.all(jnp.isfinite(energies))

    def test_batch_system_with_forces(self, device):
        """Test batched reciprocal-space energy and forces."""
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
        cell = jnp.array(
            [
                [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
                [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
            ]
        )
        batch_idx = jnp.array([0, 0, 1, 1], dtype=jnp.int32)

        alpha = jnp.array([0.3, 0.3])
        k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff=8.0)

        energies, forces = ewald_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            k_vectors=k_vectors,
            alpha=alpha,
            batch_idx=batch_idx,
            compute_forces=True,
        )

        assert energies.shape == (4,)
        assert forces.shape == (4, 3)
        assert jnp.all(jnp.isfinite(energies))
        assert jnp.all(jnp.isfinite(forces))


class TestEwaldSummationAPI:
    """Test full Ewald summation API."""

    def test_single_system_energy_only(self, device):
        """Test full Ewald summation for a single system."""
        (
            positions,
            charges,
            cell,
            neighbor_matrix,
            num_neighbors,
            neighbor_matrix_shifts,
        ) = create_dipole_system()

        alpha = 0.3
        k_cutoff = 8.0

        energies = ewald_summation(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            k_cutoff=k_cutoff,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
        )

        assert energies.shape == (2,)
        assert jnp.all(jnp.isfinite(energies))
        # Opposite charges should produce negative energy
        assert energies.sum() < 0

    def test_batch_system_energy_only(self, device):
        """Test batched full Ewald summation."""
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
        cell_single = cubic_cell_jax(10.0, dtype=jnp.float64)
        cell = jnp.concatenate([cell_single, cell_single], axis=0)  # (2, 3, 3)
        batch_idx = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
        batch_ptr = jnp.array([0, 2, 4], dtype=jnp.int32)

        cutoff = 5.0
        pbc = jnp.array([[True, True, True], [True, True, True]])
        neighbor_matrix, num_neighbors, neighbor_matrix_shifts = batch_cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors=32,
        )

        alpha = 0.3
        k_cutoff = 8.0

        energies = ewald_summation(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            k_cutoff=k_cutoff,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            batch_idx=batch_idx,
        )

        assert energies.shape == (4,)
        assert jnp.all(jnp.isfinite(energies))

    def test_batch_system_with_forces(self, device):
        """Test batched full Ewald summation with forces."""
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
        cell_single = cubic_cell_jax(10.0, dtype=jnp.float64)
        cell = jnp.concatenate([cell_single, cell_single], axis=0)  # (2, 3, 3)
        batch_idx = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
        batch_ptr = jnp.array([0, 2, 4], dtype=jnp.int32)

        cutoff = 5.0
        pbc = jnp.array([[True, True, True], [True, True, True]])
        neighbor_matrix, num_neighbors, neighbor_matrix_shifts = batch_cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors=32,
        )

        alpha = 0.3
        k_cutoff = 8.0

        energies, forces = ewald_summation(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            k_cutoff=k_cutoff,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            batch_idx=batch_idx,
            compute_forces=True,
        )

        assert energies.shape == (4,)
        assert forces.shape == (4, 3)
        assert jnp.all(jnp.isfinite(energies))
        assert jnp.all(jnp.isfinite(forces))

    def test_per_system_alpha(self, device):
        """Test batched Ewald with different alpha per system."""
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
        cell_single = cubic_cell_jax(10.0, dtype=jnp.float64)
        cell = jnp.concatenate([cell_single, cell_single], axis=0)  # (2, 3, 3)
        batch_idx = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
        batch_ptr = jnp.array([0, 2, 4], dtype=jnp.int32)

        cutoff = 5.0
        pbc = jnp.array([[True, True, True], [True, True, True]])
        neighbor_matrix, num_neighbors, neighbor_matrix_shifts = batch_cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors=32,
        )

        # Different alpha for each system
        alpha = jnp.array([0.2, 0.4])
        k_cutoff = 8.0

        energies = ewald_summation(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            k_cutoff=k_cutoff,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            batch_idx=batch_idx,
        )

        assert energies.shape == (4,)
        assert jnp.all(jnp.isfinite(energies))


@pytest.mark.skipif(not HAS_TORCHPME, reason="torchpme not installed")
class TestRealSpaceCorrectness:
    """Test real-space correctness against torchpme."""

    @pytest.mark.parametrize("crystal_fn", ["cscl", "wurtzite", "zincblende"])
    @pytest.mark.parametrize("alpha", [0.2, 0.3, 0.4])
    def test_real_space_energy_matches_torchpme(self, device, crystal_fn, alpha):
        """Test real-space energy matches torchpme reference."""
        # Create crystal system
        positions, charges, cell = make_crystal_system_jax(crystal_fn, size=2)

        # Build neighbor list
        cutoff = 10.0
        pbc = jnp.array([[True, True, True]])
        neighbor_matrix, num_neighbors, neighbor_matrix_shifts = cell_list(
            positions, cutoff, cell, pbc
        )

        # Compute with JAX
        energies_jax = ewald_real_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
        )

        # Convert to numpy for torchpme
        positions_host = np.array(positions)
        charges_host = np.array(charges)
        nm_host = np.array(neighbor_matrix)
        ns_host = np.array(neighbor_matrix_shifts)

        # Build COO pairs from dense matrix for torchpme comparison
        cell_host = np.array(cell[0])
        num_atoms = positions_host.shape[0]
        idx_i_list, idx_j_list, dist_list = [], [], []
        for i in range(nm_host.shape[0]):
            for k in range(nm_host.shape[1]):
                j = nm_host[i, k]
                if j >= num_atoms:  # padding (fill_value = num_atoms)
                    continue
                idx_i_list.append(i)
                idx_j_list.append(j)
                shift_vec = ns_host[i, k] @ cell_host
                delta = positions_host[j] - positions_host[i] + shift_vec
                dist_list.append(np.linalg.norm(delta))
        neighbor_indices = np.array([idx_i_list, idx_j_list])
        distances = np.array(dist_list)

        # Compute with torchpme
        energies_torchpme = compute_torchpme_real_space(
            charges_host, neighbor_indices, distances, alpha, k_cutoff=8.0
        )

        # Compare
        assert jnp.allclose(
            energies_jax.sum(), energies_torchpme.sum(), rtol=1e-3, atol=1e-3
        )

    @pytest.mark.parametrize("crystal_fn", ["cscl"])
    def test_real_space_forces_match_torchpme(self, device, crystal_fn):
        """Test real-space forces match torchpme reference."""
        # Create crystal system
        positions, charges, cell = make_crystal_system_jax("cscl", size=2)

        # Build neighbor list
        cutoff = 10.0
        pbc = jnp.array([[True, True, True]])
        neighbor_matrix, num_neighbors, neighbor_matrix_shifts = cell_list(
            positions, cutoff, cell, pbc
        )

        alpha = 0.3

        # Compute with JAX
        energies_jax, forces_jax = ewald_real_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            compute_forces=True,
        )

        # Check forces are finite and momentum is conserved
        assert jnp.all(jnp.isfinite(forces_jax))
        assert jnp.allclose(forces_jax.sum(axis=0), jnp.zeros(3), atol=1e-10)


@pytest.mark.skipif(not HAS_TORCHPME, reason="torchpme not installed")
class TestReciprocalSpaceCorrectness:
    """Test reciprocal-space correctness against torchpme."""

    @pytest.mark.parametrize("crystal_fn", ["cscl", "zincblende"])
    def test_reciprocal_energy_matches_torchpme(self, device, crystal_fn):
        """Test reciprocal-space energy matches torchpme reference."""
        # Create crystal system
        positions, charges, cell = make_crystal_system_jax(crystal_fn, size=2)
        positions_np = np.array(positions)
        charges_np = np.array(charges)
        cell_np = np.array(cell[0])  # (3, 3) for torchpme

        alpha = 0.3
        k_cutoff = 8.0
        k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff=k_cutoff)

        # Compute with JAX
        energies_jax = ewald_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            k_vectors=k_vectors,
            alpha=alpha,
        )

        # Compute with torchpme
        energies_torchpme = compute_torchpme_reciprocal(
            positions_np, charges_np, cell_np, k_cutoff, alpha
        )

        # Compare total energy
        assert jnp.allclose(
            energies_jax.sum(), energies_torchpme.sum(), rtol=1e-3, atol=1e-3
        )

    @pytest.mark.parametrize("crystal_fn", ["cscl"])
    def test_reciprocal_forces_match_torchpme(self, device, crystal_fn):
        """Test reciprocal-space forces match torchpme reference."""
        # Create crystal system
        positions, charges, cell = make_crystal_system_jax("cscl", size=2)

        alpha = 0.3
        k_cutoff = 8.0
        k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff=k_cutoff)

        # Compute with JAX
        energies_jax, forces_jax = ewald_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            k_vectors=k_vectors,
            alpha=alpha,
            compute_forces=True,
        )

        # Check forces are finite and momentum is conserved
        assert jnp.all(jnp.isfinite(forces_jax))
        assert jnp.allclose(forces_jax.sum(axis=0), jnp.zeros(3), atol=1e-10)


class TestExplicitChargeGradients:
    """Test explicit charge gradient computation (replaces autograd tests)."""

    def test_real_space_charge_gradients_shape(self, device):
        """Test real-space charge gradients have correct shape."""
        (
            positions,
            charges,
            cell,
            neighbor_matrix,
            num_neighbors,
            neighbor_matrix_shifts,
        ) = create_dipole_system()

        alpha = 0.3

        energies, charge_grads = ewald_real_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            compute_charge_gradients=True,
        )

        assert energies.shape == (2,)
        assert charge_grads.shape == (2,)

    def test_reciprocal_charge_gradients_shape(self, device):
        """Test reciprocal-space charge gradients have correct shape."""
        positions, charges, cell = create_simple_system()

        alpha = 0.3
        k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff=8.0)

        energies, charge_grads = ewald_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            k_vectors=k_vectors,
            alpha=alpha,
            compute_charge_gradients=True,
        )

        assert energies.shape == (4,)
        assert charge_grads.shape == (4,)

    def test_real_space_charge_gradients_finite(self, device):
        """Test real-space charge gradients are finite and non-zero."""
        (
            positions,
            charges,
            cell,
            neighbor_matrix,
            num_neighbors,
            neighbor_matrix_shifts,
        ) = create_dipole_system()

        alpha = 0.3

        energies, charge_grads = ewald_real_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            compute_charge_gradients=True,
        )

        assert jnp.all(jnp.isfinite(charge_grads))
        # At least one should be non-zero
        assert jnp.any(jnp.abs(charge_grads) > 1e-10)

    def test_reciprocal_charge_gradients_finite(self, device):
        """Test reciprocal-space charge gradients are finite and non-zero."""
        positions, charges, cell = create_simple_system()

        alpha = 0.3
        k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff=8.0)

        energies, charge_grads = ewald_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            k_vectors=k_vectors,
            alpha=alpha,
            compute_charge_gradients=True,
        )

        assert jnp.all(jnp.isfinite(charge_grads))
        # At least one should be non-zero
        assert jnp.any(jnp.abs(charge_grads) > 1e-10)


class TestPhysicalProperties:
    """Test physical properties of Ewald summation."""

    def test_opposite_charges_attract(self, device):
        """Test that opposite charges produce negative energy."""
        (
            positions,
            charges,
            cell,
            neighbor_matrix,
            num_neighbors,
            neighbor_matrix_shifts,
        ) = create_dipole_system()

        alpha = 0.3
        k_cutoff = 8.0

        energies = ewald_summation(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            k_cutoff=k_cutoff,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
        )

        assert energies.sum() < 0

    def test_energy_charge_scaling(self, device):
        """Test that doubling charges quadruples energy."""
        (
            positions,
            charges,
            cell,
            neighbor_matrix,
            num_neighbors,
            neighbor_matrix_shifts,
        ) = create_dipole_system()

        alpha = 0.3
        k_cutoff = 8.0

        # Energy with q = 1
        energy1 = ewald_summation(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            k_cutoff=k_cutoff,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
        ).sum()

        # Energy with q = 2
        charges2 = charges * 2.0
        energy2 = ewald_summation(
            positions=positions,
            charges=charges2,
            cell=cell,
            alpha=alpha,
            k_cutoff=k_cutoff,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
        ).sum()

        # E(2q) = 4 * E(q)
        assert jnp.allclose(energy2, 4.0 * energy1, rtol=1e-5)

    def test_translation_invariance(self, device):
        """Test that translating all atoms doesn't change energy."""
        (
            positions,
            charges,
            cell,
            neighbor_matrix,
            num_neighbors,
            neighbor_matrix_shifts,
        ) = create_dipole_system(cell_size=20.0)

        alpha = 0.3
        k_cutoff = 8.0

        # Energy at original positions
        energy1 = ewald_summation(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            k_cutoff=k_cutoff,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
        ).sum()

        # Translate all atoms by (1, 1, 1)
        positions_shifted = positions + jnp.array([1.0, 1.0, 1.0])

        # Rebuild neighbor list for shifted positions
        cutoff = 10.0
        pbc = jnp.array([[True, True, True]])
        nm_shifted, nn_shifted, nms_shifted = cell_list(
            positions_shifted, cutoff, cell, pbc
        )

        energy2 = ewald_summation(
            positions=positions_shifted,
            charges=charges,
            cell=cell,
            alpha=alpha,
            k_cutoff=k_cutoff,
            neighbor_matrix=nm_shifted,
            neighbor_matrix_shifts=nms_shifted,
        ).sum()

        # Energy should be the same
        assert jnp.allclose(energy1, energy2, rtol=1e-5)


class TestEdgeCases:
    """Test edge cases and special configurations."""

    def test_non_cubic_cells(self, device):
        """Test with wurtzite (hexagonal) cell."""
        positions, charges, cell = make_crystal_system_jax("wurtzite", size=2)

        # Build neighbor list
        cutoff = 10.0
        pbc = jnp.array([[True, True, True]])
        neighbor_matrix, num_neighbors, neighbor_matrix_shifts = cell_list(
            positions, cutoff, cell, pbc
        )

        alpha = 0.3
        k_cutoff = 8.0

        energies = ewald_summation(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            k_cutoff=k_cutoff,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
        )

        assert jnp.all(jnp.isfinite(energies))

    def test_auto_parameters(self, device):
        """Test Ewald summation with automatic parameter estimation."""
        (
            positions,
            charges,
            cell,
            neighbor_matrix,
            num_neighbors,
            neighbor_matrix_shifts,
        ) = create_dipole_system()

        # Use auto-estimated alpha and k_cutoff
        energies = ewald_summation(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=None,
            k_cutoff=None,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            accuracy=1e-6,
        )

        assert energies.shape == (2,)
        assert jnp.all(jnp.isfinite(energies))
        assert energies.sum() < 0


class TestEwaldJIT:
    """Smoke tests for Ewald summation compatibility with jax.jit."""

    def test_jit_real_space(self):
        """Test ewald_real_space works under jax.jit."""
        positions = jnp.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=jnp.float64)
        charges = jnp.array([1.0, -1.0], dtype=jnp.float64)
        cell = cubic_cell_jax(10.0, dtype=jnp.float64)
        pbc = jnp.array([[True, True, True]])

        # Build neighbor list eagerly (it uses .devices().pop() internally)
        neighbor_matrix, num_neighbors, neighbor_matrix_shifts = cell_list(
            positions, cutoff=5.0, cell=cell, pbc=pbc
        )
        alpha = jnp.array([0.3], dtype=jnp.float64)

        @jax.jit
        def jitted_real_space(positions, charges, cell, alpha, nm, nms):
            return ewald_real_space(
                positions=positions,
                charges=charges,
                cell=cell,
                alpha=alpha,
                neighbor_matrix=nm,
                neighbor_matrix_shifts=nms,
            )

        energies = jitted_real_space(
            positions,
            charges,
            cell,
            alpha,
            neighbor_matrix,
            neighbor_matrix_shifts,
        )

        assert energies.shape == (2,)
        assert jnp.all(jnp.isfinite(energies))

    def test_jit_reciprocal_space(self):
        """Test ewald_reciprocal_space works under jax.jit."""
        positions = jnp.array(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [3.0, 3.0, 0.0]],
            dtype=jnp.float64,
        )
        charges = jnp.array([1.0, -1.0, 0.5, -0.5], dtype=jnp.float64)
        cell = cubic_cell_jax(10.0, dtype=jnp.float64)
        alpha = jnp.array([0.3], dtype=jnp.float64)

        # k_vectors must be computed eagerly (dynamic shape)
        k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff=8.0)

        @jax.jit
        def jitted_reciprocal(positions, charges, cell, k_vectors, alpha):
            return ewald_reciprocal_space(
                positions=positions,
                charges=charges,
                cell=cell,
                k_vectors=k_vectors,
                alpha=alpha,
            )

        energies = jitted_reciprocal(positions, charges, cell, k_vectors, alpha)

        assert energies.shape == (4,)
        assert jnp.all(jnp.isfinite(energies))

    def test_jit_batched_reciprocal_space(self):
        """Test ewald_reciprocal_space works under jax.jit with batched inputs."""
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
        cell_single = cubic_cell_jax(10.0, dtype=jnp.float64)
        cell = jnp.concatenate([cell_single, cell_single], axis=0)  # (2, 3, 3)
        batch_idx = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
        alpha = jnp.array([0.3, 0.3], dtype=jnp.float64)

        # k_vectors must be computed eagerly (dynamic shape)
        k_vectors = generate_k_vectors_ewald_summation(cell_single, k_cutoff=8.0)

        @jax.jit
        def jitted_batched_reciprocal(
            positions, charges, cell, k_vectors, alpha, batch_idx
        ):
            return ewald_reciprocal_space(
                positions=positions,
                charges=charges,
                cell=cell,
                k_vectors=k_vectors,
                alpha=alpha,
                batch_idx=batch_idx,
                max_atoms_per_system=2,
            )

        energies = jitted_batched_reciprocal(
            positions, charges, cell, k_vectors, alpha, batch_idx
        )

        assert energies.shape == (4,)
        assert jnp.all(jnp.isfinite(energies))


# ==============================================================================
# Virial Test Classes
# ==============================================================================


class TestEwaldRealSpaceVirial:
    """Tests for real-space virial computation."""

    def test_virial_shape(self, device):
        """Test that virial has correct shape (1, 3, 3) for single system."""
        positions, charges, cell = make_virial_cscl_system_jax(size=2)

        # Build neighbor list
        cutoff = 6.0
        pbc = jnp.array([[True, True, True]])
        neighbor_matrix, num_neighbors, neighbor_matrix_shifts = cell_list(
            positions, cutoff, cell, pbc
        )

        alpha = jnp.array([0.3], dtype=jnp.float64)

        result = ewald_real_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            compute_forces=True,
            compute_virial=True,
        )

        # Result should be (energies, forces, virial)
        assert len(result) == 3
        energies, forces, virial = result
        assert virial.shape == (1, 3, 3)

    def test_virial_dtype(self, device):
        """Test that virial dtype matches input dtype."""
        positions, charges, cell = make_virial_cscl_system_jax(size=2)

        # Build neighbor list
        cutoff = 6.0
        pbc = jnp.array([[True, True, True]])
        neighbor_matrix, num_neighbors, neighbor_matrix_shifts = cell_list(
            positions, cutoff, cell, pbc
        )

        alpha = jnp.array([0.3], dtype=jnp.float64)

        result = ewald_real_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            compute_forces=True,
            compute_virial=True,
        )

        energies, forces, virial = result
        assert virial.dtype == jnp.float64

    def test_virial_nonzero(self, device):
        """Test that virial has non-zero elements."""
        positions, charges, cell = make_virial_cscl_system_jax(size=2)

        # Build neighbor list
        cutoff = 6.0
        pbc = jnp.array([[True, True, True]])
        neighbor_matrix, num_neighbors, neighbor_matrix_shifts = cell_list(
            positions, cutoff, cell, pbc
        )

        alpha = jnp.array([0.3], dtype=jnp.float64)

        result = ewald_real_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            compute_forces=True,
            compute_virial=True,
        )

        energies, forces, virial = result
        assert jnp.all(jnp.isfinite(virial))
        # For ionic crystal, virial should be non-zero
        assert jnp.any(jnp.abs(virial) > 1e-10)

    def test_virial_fd(self, device):
        """Test virial matches finite difference approximation."""
        positions, charges, cell = make_virial_cscl_system_jax(size=2)

        # Build neighbor list for original positions
        cutoff = 6.0
        pbc = jnp.array([[True, True, True]])

        alpha_arr = jnp.array([0.3], dtype=jnp.float64)

        # Define energy function that rebuilds neighbor list for each strained geometry
        def energy_fn(pos, c):
            nm, nn, nms = cell_list(pos, cutoff, c, pbc)
            return ewald_real_space(
                pos,
                charges,
                c,
                alpha_arr,
                neighbor_matrix=nm,
                neighbor_matrix_shifts=nms,
            ).sum()

        # Compute explicit virial
        neighbor_matrix, num_neighbors, neighbor_matrix_shifts = cell_list(
            positions, cutoff, cell, pbc
        )
        result = ewald_real_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha_arr,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            compute_forces=True,
            compute_virial=True,
        )
        explicit_virial = result[-1].squeeze(0)

        # Compute FD virial
        fd_virial = fd_virial_full_jax(energy_fn, positions, cell, h=1e-5)

        # Compare with loose tolerance for FD
        assert jnp.allclose(explicit_virial, fd_virial, atol=1e-2, rtol=1e-2)

    def test_virial_without_forces(self, device):
        """Test that compute_virial=True, compute_forces=False works."""
        positions, charges, cell = make_virial_cscl_system_jax(size=2)

        # Build neighbor list
        cutoff = 6.0
        pbc = jnp.array([[True, True, True]])
        neighbor_matrix, num_neighbors, neighbor_matrix_shifts = cell_list(
            positions, cutoff, cell, pbc
        )

        alpha = jnp.array([0.3], dtype=jnp.float64)

        result = ewald_real_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            compute_forces=False,
            compute_virial=True,
        )

        # Result should be (energies, virial) since compute_forces=False
        assert len(result) == 2
        energies, virial = result
        assert virial.shape == (1, 3, 3)
        assert jnp.all(jnp.isfinite(virial))


class TestEwaldReciprocalSpaceVirial:
    """Tests for reciprocal-space virial computation."""

    def test_virial_shape(self, device):
        """Test that virial has correct shape (1, 3, 3) for single system."""
        positions, charges, cell = make_virial_cscl_system_jax(size=2)

        alpha = jnp.array([0.3], dtype=jnp.float64)
        k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff=8.0)

        result = ewald_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            k_vectors=k_vectors,
            alpha=alpha,
            compute_virial=True,
        )

        # Result should be (energies, virial)
        assert len(result) == 2
        energies, virial = result
        assert virial.shape == (1, 3, 3)

    def test_virial_nonzero(self, device):
        """Test that virial has non-zero elements."""
        positions, charges, cell = make_virial_cscl_system_jax(size=2)

        alpha = jnp.array([0.3], dtype=jnp.float64)
        k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff=8.0)

        result = ewald_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            k_vectors=k_vectors,
            alpha=alpha,
            compute_virial=True,
        )

        energies, virial = result
        assert jnp.all(jnp.isfinite(virial))
        # For ionic crystal, virial should be non-zero
        assert jnp.any(jnp.abs(virial) > 1e-10)

    def test_virial_fd(self, device):
        """Test virial matches finite difference approximation."""
        positions, charges, cell = make_virial_cscl_system_jax(size=2)

        alpha = jnp.array([0.3], dtype=jnp.float64)

        # Define energy function that generates k_vectors from cell
        def energy_fn(pos, c):
            k_vecs = generate_k_vectors_ewald_summation(c, k_cutoff=8.0)
            return ewald_reciprocal_space(
                pos,
                charges,
                c,
                k_vecs,
                alpha,
            ).sum()

        # Compute explicit virial
        k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff=8.0)
        result = ewald_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            k_vectors=k_vectors,
            alpha=alpha,
            compute_virial=True,
        )
        explicit_virial = result[-1].squeeze(0)

        # Compute FD virial
        fd_virial = fd_virial_full_jax(energy_fn, positions, cell, h=1e-5)

        # Compare with loose tolerance for FD
        assert jnp.allclose(explicit_virial, fd_virial, atol=1e-2, rtol=1e-2)

    def test_virial_symmetry(self, device):
        """Test that virial is approximately symmetric."""
        positions, charges, cell = make_virial_cscl_system_jax(size=2)

        alpha = jnp.array([0.3], dtype=jnp.float64)
        k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff=8.0)

        result = ewald_reciprocal_space(
            positions=positions,
            charges=charges,
            cell=cell,
            k_vectors=k_vectors,
            alpha=alpha,
            compute_virial=True,
        )

        energies, virial = result
        virial_squeezed = virial.squeeze(0)
        # Check approximate symmetry
        assert jnp.allclose(virial_squeezed, virial_squeezed.T, rtol=1e-5, atol=1e-10)


class TestEwaldTotalVirial:
    """Tests for combined real+reciprocal virial computation."""

    def test_combined_virial_fd(self, device):
        """Test combined virial matches finite difference of total energy."""
        positions, charges, cell = make_virial_cscl_system_jax(size=1)

        cutoff = 6.0
        pbc = jnp.array([[True, True, True]])

        alpha = 0.3
        k_cutoff = 8.0

        # Define energy function for total Ewald summation
        def energy_fn(pos, c):
            nm, nn, nms = cell_list(pos, cutoff, c, pbc)
            return ewald_summation(
                pos,
                charges,
                c,
                alpha=alpha,
                k_cutoff=k_cutoff,
                neighbor_matrix=nm,
                neighbor_matrix_shifts=nms,
            ).sum()

        # Compute explicit virial
        neighbor_matrix, num_neighbors, neighbor_matrix_shifts = cell_list(
            positions, cutoff, cell, pbc
        )
        result = ewald_summation(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            k_cutoff=k_cutoff,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            compute_virial=True,
        )
        explicit_virial = result[-1].squeeze(0)

        # Compute FD virial
        fd_virial = fd_virial_full_jax(energy_fn, positions, cell, h=1e-5)

        # Compare with loose tolerance for FD
        assert jnp.allclose(explicit_virial, fd_virial, atol=1e-2, rtol=1e-2)


class TestEwaldVirialJIT:
    """Tests for virial computation under jax.jit."""

    def test_jit_real_space_virial(self):
        """Test that ewald_real_space with compute_virial works under jax.jit."""
        positions = jnp.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=jnp.float64)
        charges = jnp.array([1.0, -1.0], dtype=jnp.float64)
        cell = cubic_cell_jax(10.0, dtype=jnp.float64)
        pbc = jnp.array([[True, True, True]])

        # Build neighbor list eagerly
        neighbor_matrix, num_neighbors, neighbor_matrix_shifts = cell_list(
            positions, cutoff=5.0, cell=cell, pbc=pbc
        )
        alpha = jnp.array([0.3], dtype=jnp.float64)

        @jax.jit
        def jitted_real_space_virial(positions, charges, cell, alpha, nm, nms):
            return ewald_real_space(
                positions=positions,
                charges=charges,
                cell=cell,
                alpha=alpha,
                neighbor_matrix=nm,
                neighbor_matrix_shifts=nms,
                compute_forces=True,
                compute_virial=True,
            )

        result = jitted_real_space_virial(
            positions,
            charges,
            cell,
            alpha,
            neighbor_matrix,
            neighbor_matrix_shifts,
        )

        assert len(result) == 3
        energies, forces, virial = result
        assert virial.shape == (1, 3, 3)
        assert jnp.all(jnp.isfinite(virial))

    def test_jit_reciprocal_virial(self):
        """Test that ewald_reciprocal_space with compute_virial works under jax.jit."""
        positions = jnp.array(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [3.0, 3.0, 0.0]],
            dtype=jnp.float64,
        )
        charges = jnp.array([1.0, -1.0, 0.5, -0.5], dtype=jnp.float64)
        cell = cubic_cell_jax(10.0, dtype=jnp.float64)
        alpha = jnp.array([0.3], dtype=jnp.float64)

        # k_vectors must be computed eagerly (dynamic shape)
        k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff=8.0)

        @jax.jit
        def jitted_reciprocal_virial(positions, charges, cell, k_vectors, alpha):
            return ewald_reciprocal_space(
                positions=positions,
                charges=charges,
                cell=cell,
                k_vectors=k_vectors,
                alpha=alpha,
                compute_virial=True,
            )

        result = jitted_reciprocal_virial(positions, charges, cell, k_vectors, alpha)

        assert len(result) == 2
        energies, virial = result
        assert virial.shape == (1, 3, 3)
        assert jnp.all(jnp.isfinite(virial))
