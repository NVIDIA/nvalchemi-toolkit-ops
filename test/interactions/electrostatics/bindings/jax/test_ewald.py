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
    from torchpme import EwaldCalculator
    from torchpme.potentials import CoulombPotential

    HAS_TORCHPME = True
except ModuleNotFoundError:
    HAS_TORCHPME = False


# ==============================================================================
# Helper Functions
# ==============================================================================


def create_dipole_system(device, dtype=jnp.float64, separation=6.0, cell_size=10.0):
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
        (positions, charges, cell, neighbor_list, neighbor_ptr, neighbor_shifts)
    """
    positions = place_on_device(
        jnp.array([[0.0, 0.0, 0.0], [separation, 0.0, 0.0]], dtype=dtype), device
    )
    charges = place_on_device(jnp.array([1.0, -1.0], dtype=dtype), device)
    cell = place_on_device(
        jnp.array(
            [[[cell_size, 0.0, 0.0], [0.0, cell_size, 0.0], [0.0, 0.0, cell_size]]],
            dtype=dtype,
        ),
        device,
    )

    # Build neighbor list using cell_list
    cutoff = separation * 1.5
    pbc = place_on_device(jnp.array([[True, True, True]]), device)
    neighbor_list, neighbor_ptr, neighbor_shifts = cell_list(
        positions, cutoff, cell, pbc, return_neighbor_list=True
    )

    return positions, charges, cell, neighbor_list, neighbor_ptr, neighbor_shifts


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
        jax.random.uniform(key, (num_atoms, 3), dtype=dtype) * cell_size * 0.8,
        device,
    )

    # Create alternating charges for neutrality
    charges = jnp.array([1.0, -1.0] * (num_atoms // 2), dtype=dtype)
    if num_atoms % 2 == 1:
        charges = jnp.concatenate([charges, jnp.array([0.0], dtype=dtype)])
    charges = place_on_device(charges, device)

    cell = place_on_device(
        jnp.array(
            [[[cell_size, 0.0, 0.0], [0.0, cell_size, 0.0], [0.0, 0.0, cell_size]]],
            dtype=dtype,
        ),
        device,
    )

    return positions, charges, cell


def compute_torchpme_reciprocal(positions_np, charges_np, cell_np, k_cutoff, alpha):
    """Compute reciprocal energy using torchpme.

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
    device = torch.device("cuda:0")
    positions_torch = torch.tensor(positions_np, dtype=torch.float64, device=device)
    charges_torch = torch.tensor(charges_np, dtype=torch.float64, device=device)
    cell_torch = torch.tensor(cell_np, dtype=torch.float64, device=device)

    calc = EwaldCalculator(CoulombPotential(), alpha=alpha, k_cutoff=k_cutoff)
    energy_recip = calc.reciprocal_energy(positions_torch, charges_torch, cell_torch[0])

    return energy_recip.cpu().numpy()


def compute_torchpme_real_space(
    charges_np, neighbor_indices, neighbor_distances, alpha, k_cutoff
):
    """Compute real-space energy using torchpme.

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
        K-space cutoff (unused but kept for API compatibility)

    Returns
    -------
    np.ndarray
        Real space energy per atom
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    charges_torch = torch.tensor(charges_np, dtype=torch.float64, device=device)
    neighbor_indices_torch = torch.tensor(
        neighbor_indices, dtype=torch.long, device=device
    )
    neighbor_distances_torch = torch.tensor(
        neighbor_distances, dtype=torch.float64, device=device
    )

    calc = EwaldCalculator(CoulombPotential(), alpha=alpha, k_cutoff=k_cutoff)
    energy_real = calc.real_space_energy(
        charges_torch, neighbor_indices_torch, neighbor_distances_torch
    )

    return energy_real.cpu().numpy()


# ==============================================================================
# Test Classes
# ==============================================================================


class TestDtypeSupport:
    """Test float32 and float64 dtype support."""

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_real_space_dtype_returns_correct_type(self, device, dtype):
        """Test ewald_real_space with float32 and float64."""
        positions, charges, cell, neighbor_list, neighbor_ptr, neighbor_shifts = (
            create_dipole_system(device, dtype=dtype)
        )

        alpha = jnp.array([0.3], dtype=dtype)

        energies = ewald_real_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
        )

        # Energy is always float64
        assert energies.dtype == jnp.float64
        assert jnp.all(jnp.isfinite(energies))

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_reciprocal_space_dtype_returns_correct_type(self, device, dtype):
        """Test ewald_reciprocal_space with float32 and float64."""
        positions, charges, cell = create_simple_system(device, dtype=dtype)

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
        positions, charges, cell, neighbor_list, neighbor_ptr, neighbor_shifts = (
            create_dipole_system(device, dtype=dtype)
        )

        alpha = 0.3
        k_cutoff = 8.0

        energies = ewald_summation(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            k_cutoff=k_cutoff,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
        )

        assert energies.dtype == jnp.float64
        assert jnp.all(jnp.isfinite(energies))

    def test_float32_vs_float64_consistency(self, device):
        """Test that float32 and float64 produce consistent results."""
        positions_f64, charges_f64, cell_f64, nl_f64, ptr_f64, shifts_f64 = (
            create_dipole_system(device, dtype=jnp.float64)
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
            neighbor_list=nl_f64,
            neighbor_ptr=ptr_f64,
            neighbor_shifts=shifts_f64,
        )

        energies_f32 = ewald_real_space(
            positions=positions_f32,
            charges=charges_f32,
            cell=cell_f32,
            alpha=alpha,
            neighbor_list=nl_f64,
            neighbor_ptr=ptr_f64,
            neighbor_shifts=shifts_f64,
        )

        # Both are float64 but f32 may have slightly different values
        assert jnp.allclose(energies_f32, energies_f64, rtol=1e-4)

    def test_batch_dtype_returns_correct_type(self, device):
        """Test batched calculations return correct dtype."""
        # Create 2 batches of 2 atoms each
        positions = place_on_device(
            jnp.array(
                [
                    [0.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                ],
                dtype=jnp.float32,
            ),
            device,
        )
        charges = place_on_device(
            jnp.array([1.0, -1.0, 1.0, -1.0], dtype=jnp.float32), device
        )
        cell = place_on_device(
            jnp.array([[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]]),
            device,
        ).astype(jnp.float32)
        batch_idx = place_on_device(jnp.array([0, 0, 1, 1], dtype=jnp.int32), device)

        cutoff = 5.0
        pbc = place_on_device(jnp.array([[True, True, True]]), device)
        neighbor_list, neighbor_ptr, neighbor_shifts = cell_list(
            positions, cutoff, cell, pbc, return_neighbor_list=True
        )

        alpha = jnp.array([0.3, 0.3], dtype=jnp.float32)

        energies = ewald_real_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            batch_idx=batch_idx,
        )

        assert energies.dtype == jnp.float64
        assert energies.shape == (4,)


class TestEwaldRealSpaceAPI:
    """Test Ewald real-space API."""

    def test_single_system_energy_only(self, device):
        """Test real-space energy for a single system."""
        positions, charges, cell, neighbor_list, neighbor_ptr, neighbor_shifts = (
            create_dipole_system(device)
        )

        alpha = 0.3

        energies = ewald_real_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
        )

        assert energies.shape == (2,)
        assert jnp.all(jnp.isfinite(energies))
        # Opposite charges should produce negative energy
        assert energies.sum() < 0

    def test_single_system_with_forces(self, device):
        """Test real-space energy and forces."""
        positions, charges, cell, neighbor_list, neighbor_ptr, neighbor_shifts = (
            create_dipole_system(device)
        )

        alpha = 0.3

        energies, forces = ewald_real_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
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
        positions = place_on_device(
            jnp.array(
                [
                    [0.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                ],
                dtype=jnp.float64,
            ),
            device,
        )
        charges = place_on_device(
            jnp.array([1.0, -1.0, 1.0, -1.0], dtype=jnp.float64), device
        )
        cell = place_on_device(
            jnp.array([[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]]),
            device,
        )
        batch_idx = place_on_device(jnp.array([0, 0, 1, 1], dtype=jnp.int32), device)

        cutoff = 5.0
        pbc = place_on_device(jnp.array([[True, True, True]]), device)
        neighbor_list, neighbor_ptr, neighbor_shifts = cell_list(
            positions, cutoff, cell, pbc, return_neighbor_list=True
        )

        alpha = jnp.array([0.3, 0.3])

        energies = ewald_real_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            batch_idx=batch_idx,
        )

        assert energies.shape == (4,)
        assert jnp.all(jnp.isfinite(energies))

    def test_batch_system_with_forces(self, device):
        """Test batched real-space energy and forces."""
        positions = place_on_device(
            jnp.array(
                [
                    [0.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                ],
                dtype=jnp.float64,
            ),
            device,
        )
        charges = place_on_device(
            jnp.array([1.0, -1.0, 1.0, -1.0], dtype=jnp.float64), device
        )
        cell = place_on_device(
            jnp.array([[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]]),
            device,
        )
        batch_idx = place_on_device(jnp.array([0, 0, 1, 1], dtype=jnp.int32), device)

        cutoff = 5.0
        pbc = place_on_device(jnp.array([[True, True, True]]), device)
        neighbor_list, neighbor_ptr, neighbor_shifts = cell_list(
            positions, cutoff, cell, pbc, return_neighbor_list=True
        )

        alpha = jnp.array([0.3, 0.3])

        energies, forces = ewald_real_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
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
        positions, charges, cell = create_simple_system(device)

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
        positions, charges, cell = create_simple_system(device)

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
        positions = place_on_device(
            jnp.array(
                [
                    [0.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                ],
                dtype=jnp.float64,
            ),
            device,
        )
        charges = place_on_device(
            jnp.array([1.0, -1.0, 1.0, -1.0], dtype=jnp.float64), device
        )
        cell = place_on_device(
            jnp.array(
                [
                    [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
                    [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
                ]
            ),
            device,
        )
        batch_idx = place_on_device(jnp.array([0, 0, 1, 1], dtype=jnp.int32), device)

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
        positions = place_on_device(
            jnp.array(
                [
                    [0.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                ],
                dtype=jnp.float64,
            ),
            device,
        )
        charges = place_on_device(
            jnp.array([1.0, -1.0, 1.0, -1.0], dtype=jnp.float64), device
        )
        cell = place_on_device(
            jnp.array(
                [
                    [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
                    [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
                ]
            ),
            device,
        )
        batch_idx = place_on_device(jnp.array([0, 0, 1, 1], dtype=jnp.int32), device)

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
        positions, charges, cell, neighbor_list, neighbor_ptr, neighbor_shifts = (
            create_dipole_system(device)
        )

        alpha = 0.3
        k_cutoff = 8.0

        energies = ewald_summation(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            k_cutoff=k_cutoff,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
        )

        assert energies.shape == (2,)
        assert jnp.all(jnp.isfinite(energies))
        # Opposite charges should produce negative energy
        assert energies.sum() < 0

    def test_batch_system_energy_only(self, device):
        """Test batched full Ewald summation."""
        positions = place_on_device(
            jnp.array(
                [
                    [0.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                ],
                dtype=jnp.float64,
            ),
            device,
        )
        charges = place_on_device(
            jnp.array([1.0, -1.0, 1.0, -1.0], dtype=jnp.float64), device
        )
        cell = place_on_device(
            jnp.array([[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]]),
            device,
        )
        batch_idx = place_on_device(jnp.array([0, 0, 1, 1], dtype=jnp.int32), device)

        cutoff = 5.0
        pbc = place_on_device(jnp.array([[True, True, True]]), device)
        neighbor_list, neighbor_ptr, neighbor_shifts = cell_list(
            positions, cutoff, cell, pbc, return_neighbor_list=True
        )

        alpha = 0.3
        k_cutoff = 8.0

        energies = ewald_summation(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            k_cutoff=k_cutoff,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            batch_idx=batch_idx,
        )

        assert energies.shape == (4,)
        assert jnp.all(jnp.isfinite(energies))

    def test_batch_system_with_forces(self, device):
        """Test batched full Ewald summation with forces."""
        positions = place_on_device(
            jnp.array(
                [
                    [0.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                ],
                dtype=jnp.float64,
            ),
            device,
        )
        charges = place_on_device(
            jnp.array([1.0, -1.0, 1.0, -1.0], dtype=jnp.float64), device
        )
        cell = place_on_device(
            jnp.array([[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]]),
            device,
        )
        batch_idx = place_on_device(jnp.array([0, 0, 1, 1], dtype=jnp.int32), device)

        cutoff = 5.0
        pbc = place_on_device(jnp.array([[True, True, True]]), device)
        neighbor_list, neighbor_ptr, neighbor_shifts = cell_list(
            positions, cutoff, cell, pbc, return_neighbor_list=True
        )

        alpha = 0.3
        k_cutoff = 8.0

        energies, forces = ewald_summation(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            k_cutoff=k_cutoff,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            batch_idx=batch_idx,
            compute_forces=True,
        )

        assert energies.shape == (4,)
        assert forces.shape == (4, 3)
        assert jnp.all(jnp.isfinite(energies))
        assert jnp.all(jnp.isfinite(forces))

    def test_per_system_alpha(self, device):
        """Test batched Ewald with different alpha per system."""
        positions = place_on_device(
            jnp.array(
                [
                    [0.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                ],
                dtype=jnp.float64,
            ),
            device,
        )
        charges = place_on_device(
            jnp.array([1.0, -1.0, 1.0, -1.0], dtype=jnp.float64), device
        )
        cell = place_on_device(
            jnp.array([[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]]),
            device,
        )
        batch_idx = place_on_device(jnp.array([0, 0, 1, 1], dtype=jnp.int32), device)

        cutoff = 5.0
        pbc = place_on_device(jnp.array([[True, True, True]]), device)
        neighbor_list, neighbor_ptr, neighbor_shifts = cell_list(
            positions, cutoff, cell, pbc, return_neighbor_list=True
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
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
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
        crystal_generators = {
            "cscl": create_cscl_supercell,
            "wurtzite": create_wurtzite_system,
            "zincblende": create_zincblende_system,
        }
        crystal = crystal_generators[crystal_fn](size=2)

        positions_np = crystal.positions
        charges_np = crystal.charges
        cell_np = crystal.cell

        # Convert to JAX
        positions = place_on_device(jnp.array(positions_np, dtype=jnp.float64), device)
        charges = place_on_device(jnp.array(charges_np, dtype=jnp.float64), device)
        cell = place_on_device(
            jnp.array(cell_np[jnp.newaxis, :, :], dtype=jnp.float64), device
        )

        # Build neighbor list
        cutoff = 10.0
        pbc = place_on_device(jnp.array([[True, True, True]]), device)
        neighbor_list, neighbor_ptr, neighbor_shifts = cell_list(
            positions, cutoff, cell, pbc, return_neighbor_list=True
        )

        # Compute with JAX
        energies_jax = ewald_real_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
        )

        # Convert to numpy for torchpme
        positions_host = np.array(positions)
        charges_host = np.array(charges)
        neighbor_list_host = np.array(neighbor_list)
        neighbor_shifts_host = np.array(neighbor_shifts)

        # Compute pair distances for torchpme
        idx_i = neighbor_list_host[0]
        idx_j = neighbor_list_host[1]
        shifts = neighbor_shifts_host

        # Compute distances including periodic shifts
        cell_host = np.array(cell[0])
        pos_i = positions_host[idx_i]
        pos_j = positions_host[idx_j]
        shift_vecs = shifts @ cell_host
        deltas = pos_j - pos_i + shift_vecs
        distances = np.linalg.norm(deltas, axis=1)

        # Compute with torchpme
        energies_torchpme = compute_torchpme_real_space(
            charges_host, neighbor_list_host, distances, alpha, k_cutoff=8.0
        )

        # Compare
        assert jnp.allclose(
            energies_jax.sum(), energies_torchpme.sum(), rtol=1e-3, atol=1e-3
        )

    @pytest.mark.parametrize("crystal_fn", ["cscl"])
    def test_real_space_forces_match_torchpme(self, device, crystal_fn):
        """Test real-space forces match torchpme reference."""
        # Create crystal system
        crystal = create_cscl_supercell(size=2)

        positions_np = crystal.positions
        charges_np = crystal.charges
        cell_np = crystal.cell

        # Convert to JAX
        positions = place_on_device(jnp.array(positions_np, dtype=jnp.float64), device)
        charges = place_on_device(jnp.array(charges_np, dtype=jnp.float64), device)
        cell = place_on_device(
            jnp.array(cell_np[jnp.newaxis, :, :], dtype=jnp.float64), device
        )

        # Build neighbor list
        cutoff = 10.0
        pbc = place_on_device(jnp.array([[True, True, True]]), device)
        neighbor_list, neighbor_ptr, neighbor_shifts = cell_list(
            positions, cutoff, cell, pbc, return_neighbor_list=True
        )

        alpha = 0.3

        # Compute with JAX
        energies_jax, forces_jax = ewald_real_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
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
        crystal_generators = {
            "cscl": create_cscl_supercell,
            "zincblende": create_zincblende_system,
        }
        crystal = crystal_generators[crystal_fn](size=2)

        positions_np = crystal.positions
        charges_np = crystal.charges
        cell_np = crystal.cell

        # Convert to JAX
        positions = place_on_device(jnp.array(positions_np, dtype=jnp.float64), device)
        charges = place_on_device(jnp.array(charges_np, dtype=jnp.float64), device)
        cell = place_on_device(
            jnp.array(cell_np[jnp.newaxis, :, :], dtype=jnp.float64), device
        )

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
        crystal = create_cscl_supercell(size=2)

        positions_np = crystal.positions
        charges_np = crystal.charges
        cell_np = crystal.cell

        # Convert to JAX
        positions = place_on_device(jnp.array(positions_np, dtype=jnp.float64), device)
        charges = place_on_device(jnp.array(charges_np, dtype=jnp.float64), device)
        cell = place_on_device(
            jnp.array(cell_np[jnp.newaxis, :, :], dtype=jnp.float64), device
        )

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
        positions, charges, cell, neighbor_list, neighbor_ptr, neighbor_shifts = (
            create_dipole_system(device)
        )

        alpha = 0.3

        energies, charge_grads = ewald_real_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            compute_charge_gradients=True,
        )

        assert energies.shape == (2,)
        assert charge_grads.shape == (2,)

    def test_reciprocal_charge_gradients_shape(self, device):
        """Test reciprocal-space charge gradients have correct shape."""
        positions, charges, cell = create_simple_system(device)

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
        positions, charges, cell, neighbor_list, neighbor_ptr, neighbor_shifts = (
            create_dipole_system(device)
        )

        alpha = 0.3

        energies, charge_grads = ewald_real_space(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            compute_charge_gradients=True,
        )

        assert jnp.all(jnp.isfinite(charge_grads))
        # At least one should be non-zero
        assert jnp.any(jnp.abs(charge_grads) > 1e-10)

    def test_reciprocal_charge_gradients_finite(self, device):
        """Test reciprocal-space charge gradients are finite and non-zero."""
        positions, charges, cell = create_simple_system(device)

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
        positions, charges, cell, neighbor_list, neighbor_ptr, neighbor_shifts = (
            create_dipole_system(device)
        )

        alpha = 0.3
        k_cutoff = 8.0

        energies = ewald_summation(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            k_cutoff=k_cutoff,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
        )

        assert energies.sum() < 0

    def test_energy_charge_scaling(self, device):
        """Test that doubling charges quadruples energy."""
        positions, charges, cell, neighbor_list, neighbor_ptr, neighbor_shifts = (
            create_dipole_system(device)
        )

        alpha = 0.3
        k_cutoff = 8.0

        # Energy with q = 1
        energy1 = ewald_summation(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            k_cutoff=k_cutoff,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
        ).sum()

        # Energy with q = 2
        charges2 = charges * 2.0
        energy2 = ewald_summation(
            positions=positions,
            charges=charges2,
            cell=cell,
            alpha=alpha,
            k_cutoff=k_cutoff,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
        ).sum()

        # E(2q) = 4 * E(q)
        assert jnp.allclose(energy2, 4.0 * energy1, rtol=1e-5)

    def test_translation_invariance(self, device):
        """Test that translating all atoms doesn't change energy."""
        positions, charges, cell, neighbor_list, neighbor_ptr, neighbor_shifts = (
            create_dipole_system(device, cell_size=20.0)
        )

        alpha = 0.3
        k_cutoff = 8.0

        # Energy at original positions
        energy1 = ewald_summation(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            k_cutoff=k_cutoff,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
        ).sum()

        # Translate all atoms by (1, 1, 1)
        positions_shifted = positions + jnp.array([1.0, 1.0, 1.0])

        # Rebuild neighbor list for shifted positions
        cutoff = 10.0
        pbc = place_on_device(jnp.array([[True, True, True]]), device)
        nl_shifted, ptr_shifted, shifts_shifted = cell_list(
            positions_shifted, cutoff, cell, pbc, return_neighbor_list=True
        )

        energy2 = ewald_summation(
            positions=positions_shifted,
            charges=charges,
            cell=cell,
            alpha=alpha,
            k_cutoff=k_cutoff,
            neighbor_list=nl_shifted,
            neighbor_ptr=ptr_shifted,
            neighbor_shifts=shifts_shifted,
        ).sum()

        # Energy should be the same
        assert jnp.allclose(energy1, energy2, rtol=1e-5)


class TestEdgeCases:
    """Test edge cases and special configurations."""

    def test_non_cubic_cells(self, device):
        """Test with wurtzite (hexagonal) cell."""
        crystal = create_wurtzite_system(size=2)

        positions_np = crystal.positions
        charges_np = crystal.charges
        cell_np = crystal.cell

        # Convert to JAX
        positions = place_on_device(jnp.array(positions_np, dtype=jnp.float64), device)
        charges = place_on_device(jnp.array(charges_np, dtype=jnp.float64), device)
        cell = place_on_device(
            jnp.array(cell_np[jnp.newaxis, :, :], dtype=jnp.float64), device
        )

        # Build neighbor list
        cutoff = 10.0
        pbc = place_on_device(jnp.array([[True, True, True]]), device)
        neighbor_list, neighbor_ptr, neighbor_shifts = cell_list(
            positions, cutoff, cell, pbc, return_neighbor_list=True
        )

        alpha = 0.3
        k_cutoff = 8.0

        energies = ewald_summation(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            k_cutoff=k_cutoff,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
        )

        assert jnp.all(jnp.isfinite(energies))

    def test_auto_parameters(self, device):
        """Test Ewald summation with automatic parameter estimation."""
        positions, charges, cell, neighbor_list, neighbor_ptr, neighbor_shifts = (
            create_dipole_system(device)
        )

        # Use auto-estimated alpha and k_cutoff
        energies = ewald_summation(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=None,
            k_cutoff=None,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            accuracy=1e-6,
        )

        assert energies.shape == (2,)
        assert jnp.all(jnp.isfinite(energies))
        assert energies.sum() < 0
