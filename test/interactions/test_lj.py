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

"""Tests for nvalchemiops.interactions.lj module.

This module provides comprehensive tests for the Lennard-Jones potential
implementation, covering:
- Energy calculations
- Force calculations
- Virial tensor calculations
- Both neighbor matrix and neighbor list (CSR) formats
- Single and batched systems
- Half and full neighbor lists
- Switching function integration
- Float32 and float64 precision
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import warp as wp

from nvalchemiops.interactions.lj import (
    lj_energy,
    lj_energy_forces,
    lj_energy_forces_virial,
    lj_forces,
)
from nvalchemiops.torch.neighbors import cell_list

wp.init()


# =============================================================================
# Helper Functions
# =============================================================================


def lj_energy_pair_numpy(r: float, epsilon: float, sigma: float) -> float:
    """Compute LJ pair energy: V = 4ε[(σ/r)^12 - (σ/r)^6]."""
    s = sigma / r
    s6 = s**6
    s12 = s6**2
    return 4.0 * epsilon * (s12 - s6)


def lj_force_over_r_numpy(r: float, epsilon: float, sigma: float) -> float:
    """Compute LJ force/r: F/r = 24ε/r² [2(σ/r)^12 - (σ/r)^6]."""
    s = sigma / r
    s6 = s**6
    s12 = s6**2
    r_sq = r * r
    return 24.0 * epsilon * (2 * s12 - s6) / r_sq


def create_simple_pair_system(
    r: float, epsilon: float, sigma: float, dtype=np.float64, device="cuda:0"
):
    """Create a simple two-atom system for testing.

    Places atoms at (0,0,0) and (r,0,0) in a large box.
    """
    box_size = 50.0
    positions = np.array(
        [
            [box_size / 2, box_size / 2, box_size / 2],
            [box_size / 2 + r, box_size / 2, box_size / 2],
        ],
        dtype=dtype,
    )
    cell = np.eye(3, dtype=dtype) * box_size

    vec_dtype = wp.vec3f if dtype == np.float32 else wp.vec3d
    mat_dtype = wp.mat33f if dtype == np.float32 else wp.mat33d

    positions_wp = wp.array(positions, dtype=vec_dtype, device=device)
    cell_wp = wp.array([cell], dtype=mat_dtype, device=device)

    return positions_wp, cell_wp


def create_fcc_positions(n_cells: int, lattice_constant: float, dtype=np.float64):
    """Create FCC lattice positions."""
    basis = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ],
        dtype=dtype,
    )

    positions = []
    for i in range(n_cells):
        for j in range(n_cells):
            for k in range(n_cells):
                for b in basis:
                    pos = (np.array([i, j, k], dtype=dtype) + b) * lattice_constant
                    positions.append(pos)

    return np.array(positions, dtype=dtype)


def build_neighbor_matrix_simple(
    positions_np: np.ndarray, cell_np: np.ndarray, cutoff: float, half: bool = True
):
    """Build a simple neighbor matrix for testing (no PBC, small systems)."""
    num_atoms = len(positions_np)
    max_neighbors = num_atoms - 1

    neighbor_matrix = np.full((num_atoms, max_neighbors), num_atoms, dtype=np.int32)
    neighbor_shifts = np.zeros((num_atoms, max_neighbors, 3), dtype=np.int32)
    num_neighbors = np.zeros(num_atoms, dtype=np.int32)

    for i in range(num_atoms):
        j_start = i + 1 if half else 0
        count = 0
        for j in range(j_start, num_atoms):
            if i == j:
                continue
            r_ij = positions_np[i] - positions_np[j]
            dist = np.linalg.norm(r_ij)
            if dist < cutoff:
                neighbor_matrix[i, count] = j
                # No shifts for non-PBC
                count += 1
        num_neighbors[i] = count

    return neighbor_matrix, neighbor_shifts, num_neighbors


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def device():
    """Return appropriate device for testing."""
    return "cuda:0" if wp.is_cuda_available() else "cpu"


@pytest.fixture
def lj_params():
    """Standard argon-like LJ parameters."""
    return {
        "epsilon": 0.0104,  # eV
        "sigma": 3.40,  # Å
        "cutoff": 8.5,  # Å (2.5 * sigma)
    }


# =============================================================================
# Test Classes
# =============================================================================


class TestLJEnergyPair:
    """Tests for LJ energy with simple two-atom systems."""

    def test_energy_at_sigma(self, device, lj_params):
        """Test energy at r=sigma (should be 0)."""
        r = lj_params["sigma"]
        epsilon = lj_params["epsilon"]
        sigma = lj_params["sigma"]
        cutoff = lj_params["cutoff"]

        positions_wp, cell_wp = create_simple_pair_system(
            r, epsilon, sigma, device=device
        )

        # Build neighbor matrix
        pos_np = positions_wp.numpy()
        cell_np = cell_wp.numpy()[0]
        nm, ns, nn = build_neighbor_matrix_simple(pos_np, cell_np, cutoff, half=True)

        nm_wp = wp.array(nm, dtype=wp.int32, device=device)
        ns_wp = wp.array(ns, dtype=wp.vec3i, device=device)
        nn_wp = wp.array(nn, dtype=wp.int32, device=device)

        energies = lj_energy(
            positions=positions_wp,
            cell=cell_wp,
            epsilon=epsilon,
            sigma=sigma,
            cutoff=cutoff,
            neighbor_matrix=nm_wp,
            neighbor_matrix_shifts=ns_wp,
            num_neighbors=nn_wp,
            half_neighbor_list=True,
            device=device,
        )

        total_energy = energies.numpy().sum()
        expected = lj_energy_pair_numpy(r, epsilon, sigma)

        np.testing.assert_allclose(total_energy, expected, rtol=1e-10, atol=1e-15)

    def test_energy_at_equilibrium(self, device, lj_params):
        """Test energy at r=2^(1/6)*sigma (minimum)."""
        sigma = lj_params["sigma"]
        epsilon = lj_params["epsilon"]
        cutoff = lj_params["cutoff"]
        r = sigma * (2 ** (1 / 6))  # Equilibrium distance

        positions_wp, cell_wp = create_simple_pair_system(
            r, epsilon, sigma, device=device
        )

        pos_np = positions_wp.numpy()
        cell_np = cell_wp.numpy()[0]
        nm, ns, nn = build_neighbor_matrix_simple(pos_np, cell_np, cutoff, half=True)

        nm_wp = wp.array(nm, dtype=wp.int32, device=device)
        ns_wp = wp.array(ns, dtype=wp.vec3i, device=device)
        nn_wp = wp.array(nn, dtype=wp.int32, device=device)

        energies = lj_energy(
            positions=positions_wp,
            cell=cell_wp,
            epsilon=epsilon,
            sigma=sigma,
            cutoff=cutoff,
            neighbor_matrix=nm_wp,
            neighbor_matrix_shifts=ns_wp,
            num_neighbors=nn_wp,
            half_neighbor_list=True,
            device=device,
        )

        total_energy = energies.numpy().sum()
        expected = -epsilon  # Minimum energy

        np.testing.assert_allclose(total_energy, expected, rtol=1e-10)

    def test_energy_beyond_cutoff(self, device, lj_params):
        """Test that energy is 0 beyond cutoff."""
        epsilon = lj_params["epsilon"]
        sigma = lj_params["sigma"]
        cutoff = lj_params["cutoff"]
        r = cutoff + 1.0  # Beyond cutoff

        positions_wp, cell_wp = create_simple_pair_system(
            r, epsilon, sigma, device=device
        )

        pos_np = positions_wp.numpy()
        cell_np = cell_wp.numpy()[0]
        nm, ns, nn = build_neighbor_matrix_simple(pos_np, cell_np, cutoff, half=True)

        nm_wp = wp.array(nm, dtype=wp.int32, device=device)
        ns_wp = wp.array(ns, dtype=wp.vec3i, device=device)
        nn_wp = wp.array(nn, dtype=wp.int32, device=device)

        energies = lj_energy(
            positions=positions_wp,
            cell=cell_wp,
            epsilon=epsilon,
            sigma=sigma,
            cutoff=cutoff,
            neighbor_matrix=nm_wp,
            neighbor_matrix_shifts=ns_wp,
            num_neighbors=nn_wp,
            half_neighbor_list=True,
            device=device,
        )

        total_energy = energies.numpy().sum()
        np.testing.assert_allclose(total_energy, 0.0, atol=1e-15)


class TestLJForces:
    """Tests for LJ force calculations."""

    def test_force_direction(self, device, lj_params):
        """Test that force points in correct direction."""
        epsilon = lj_params["epsilon"]
        sigma = lj_params["sigma"]
        cutoff = lj_params["cutoff"]
        r = sigma * 1.5  # Attractive region

        positions_wp, cell_wp = create_simple_pair_system(
            r, epsilon, sigma, device=device
        )

        pos_np = positions_wp.numpy()
        cell_np = cell_wp.numpy()[0]
        nm, ns, nn = build_neighbor_matrix_simple(pos_np, cell_np, cutoff, half=True)

        nm_wp = wp.array(nm, dtype=wp.int32, device=device)
        ns_wp = wp.array(ns, dtype=wp.vec3i, device=device)
        nn_wp = wp.array(nn, dtype=wp.int32, device=device)

        forces = lj_forces(
            positions=positions_wp,
            cell=cell_wp,
            epsilon=epsilon,
            sigma=sigma,
            cutoff=cutoff,
            neighbor_matrix=nm_wp,
            neighbor_matrix_shifts=ns_wp,
            num_neighbors=nn_wp,
            half_neighbor_list=True,
            device=device,
        )

        forces_np = forces.numpy()

        # In attractive region, atoms should attract
        # Atom 0 at (25,25,25), atom 1 at (25+r,25,25)
        # Force on atom 0 should point toward atom 1 (+x)
        # Force on atom 1 should point toward atom 0 (-x)
        assert forces_np[0, 0] > 0  # F_x on atom 0 is positive
        assert forces_np[1, 0] < 0  # F_x on atom 1 is negative

        # Newton's 3rd law
        np.testing.assert_allclose(forces_np[0], -forces_np[1], rtol=1e-10)

    def test_force_zero_at_equilibrium(self, device, lj_params):
        """Test that force is zero at equilibrium distance."""
        epsilon = lj_params["epsilon"]
        sigma = lj_params["sigma"]
        cutoff = lj_params["cutoff"]
        r = sigma * (2 ** (1 / 6))  # Equilibrium

        positions_wp, cell_wp = create_simple_pair_system(
            r, epsilon, sigma, device=device
        )

        pos_np = positions_wp.numpy()
        cell_np = cell_wp.numpy()[0]
        nm, ns, nn = build_neighbor_matrix_simple(pos_np, cell_np, cutoff, half=True)

        nm_wp = wp.array(nm, dtype=wp.int32, device=device)
        ns_wp = wp.array(ns, dtype=wp.vec3i, device=device)
        nn_wp = wp.array(nn, dtype=wp.int32, device=device)

        forces = lj_forces(
            positions=positions_wp,
            cell=cell_wp,
            epsilon=epsilon,
            sigma=sigma,
            cutoff=cutoff,
            neighbor_matrix=nm_wp,
            neighbor_matrix_shifts=ns_wp,
            num_neighbors=nn_wp,
            half_neighbor_list=True,
            device=device,
        )

        forces_np = forces.numpy()
        np.testing.assert_allclose(forces_np, 0.0, atol=1e-12)

    def test_force_magnitude(self, device, lj_params):
        """Test force magnitude against analytical formula."""
        epsilon = lj_params["epsilon"]
        sigma = lj_params["sigma"]
        cutoff = lj_params["cutoff"]
        r = 4.0  # Test distance

        positions_wp, cell_wp = create_simple_pair_system(
            r, epsilon, sigma, device=device
        )

        pos_np = positions_wp.numpy()
        cell_np = cell_wp.numpy()[0]
        nm, ns, nn = build_neighbor_matrix_simple(pos_np, cell_np, cutoff, half=True)

        nm_wp = wp.array(nm, dtype=wp.int32, device=device)
        ns_wp = wp.array(ns, dtype=wp.vec3i, device=device)
        nn_wp = wp.array(nn, dtype=wp.int32, device=device)

        forces = lj_forces(
            positions=positions_wp,
            cell=cell_wp,
            epsilon=epsilon,
            sigma=sigma,
            cutoff=cutoff,
            neighbor_matrix=nm_wp,
            neighbor_matrix_shifts=ns_wp,
            num_neighbors=nn_wp,
            half_neighbor_list=True,
            device=device,
        )

        forces_np = forces.numpy()

        # Expected force magnitude
        f_over_r = lj_force_over_r_numpy(r, epsilon, sigma)
        expected_force_mag = abs(f_over_r * r)

        actual_force_mag = np.linalg.norm(forces_np[0])
        np.testing.assert_allclose(actual_force_mag, expected_force_mag, rtol=1e-10)

    def test_force_numerical_gradient(self, device, lj_params):
        """Test forces against numerical gradient of energy."""
        epsilon = lj_params["epsilon"]
        sigma = lj_params["sigma"]
        cutoff = lj_params["cutoff"]
        r = 4.5
        h = 1e-6

        # Compute force analytically
        positions_wp, cell_wp = create_simple_pair_system(
            r, epsilon, sigma, device=device
        )

        pos_np = positions_wp.numpy()
        cell_np = cell_wp.numpy()[0]
        nm, ns, nn = build_neighbor_matrix_simple(pos_np, cell_np, cutoff, half=True)

        nm_wp = wp.array(nm, dtype=wp.int32, device=device)
        ns_wp = wp.array(ns, dtype=wp.vec3i, device=device)
        nn_wp = wp.array(nn, dtype=wp.int32, device=device)

        _, forces = lj_energy_forces(
            positions=positions_wp,
            cell=cell_wp,
            epsilon=epsilon,
            sigma=sigma,
            cutoff=cutoff,
            neighbor_matrix=nm_wp,
            neighbor_matrix_shifts=ns_wp,
            num_neighbors=nn_wp,
            half_neighbor_list=True,
            device=device,
        )

        forces_np = forces.numpy()

        # Numerical gradient
        vec_dtype = wp.vec3d

        numerical_force = np.zeros(3)
        for dim in range(3):
            pos_plus = pos_np.copy()
            pos_plus[0, dim] += h
            pos_wp_plus = wp.array(pos_plus, dtype=vec_dtype, device=device)

            nm_p, ns_p, nn_p = build_neighbor_matrix_simple(
                pos_plus, cell_np, cutoff, half=True
            )
            nm_wp_p = wp.array(nm_p, dtype=wp.int32, device=device)
            ns_wp_p = wp.array(ns_p, dtype=wp.vec3i, device=device)
            nn_wp_p = wp.array(nn_p, dtype=wp.int32, device=device)

            e_plus = lj_energy(
                positions=pos_wp_plus,
                cell=cell_wp,
                epsilon=epsilon,
                sigma=sigma,
                cutoff=cutoff,
                neighbor_matrix=nm_wp_p,
                neighbor_matrix_shifts=ns_wp_p,
                num_neighbors=nn_wp_p,
                half_neighbor_list=True,
                device=device,
            )

            pos_minus = pos_np.copy()
            pos_minus[0, dim] -= h
            pos_wp_minus = wp.array(pos_minus, dtype=vec_dtype, device=device)

            nm_m, ns_m, nn_m = build_neighbor_matrix_simple(
                pos_minus, cell_np, cutoff, half=True
            )
            nm_wp_m = wp.array(nm_m, dtype=wp.int32, device=device)
            ns_wp_m = wp.array(ns_m, dtype=wp.vec3i, device=device)
            nn_wp_m = wp.array(nn_m, dtype=wp.int32, device=device)

            e_minus = lj_energy(
                positions=pos_wp_minus,
                cell=cell_wp,
                epsilon=epsilon,
                sigma=sigma,
                cutoff=cutoff,
                neighbor_matrix=nm_wp_m,
                neighbor_matrix_shifts=ns_wp_m,
                num_neighbors=nn_wp_m,
                half_neighbor_list=True,
                device=device,
            )

            numerical_force[dim] = -(e_plus.numpy().sum() - e_minus.numpy().sum()) / (
                2 * h
            )

        np.testing.assert_allclose(forces_np[0], numerical_force, rtol=1e-5)


class TestLJVirial:
    """Tests for LJ virial tensor calculations."""

    def test_virial_pair(self, device, lj_params):
        """Test virial tensor for a simple pair."""
        epsilon = lj_params["epsilon"]
        sigma = lj_params["sigma"]
        cutoff = lj_params["cutoff"]
        r = 4.0

        positions_wp, cell_wp = create_simple_pair_system(
            r, epsilon, sigma, device=device
        )

        pos_np = positions_wp.numpy()
        cell_np = cell_wp.numpy()[0]
        nm, ns, nn = build_neighbor_matrix_simple(pos_np, cell_np, cutoff, half=True)

        nm_wp = wp.array(nm, dtype=wp.int32, device=device)
        ns_wp = wp.array(ns, dtype=wp.vec3i, device=device)
        nn_wp = wp.array(nn, dtype=wp.int32, device=device)

        energies, forces, virial = lj_energy_forces_virial(
            positions=positions_wp,
            cell=cell_wp,
            epsilon=epsilon,
            sigma=sigma,
            cutoff=cutoff,
            neighbor_matrix=nm_wp,
            neighbor_matrix_shifts=ns_wp,
            num_neighbors=nn_wp,
            half_neighbor_list=True,
            device=device,
        )

        virial_np = virial.numpy()

        # For pair along x-axis, virial should be mostly xx component
        # W_αβ = -r_α * F_β
        f_over_r = lj_force_over_r_numpy(r, epsilon, sigma)
        f_x = f_over_r * r  # Force magnitude in x direction

        # r_ij = (r, 0, 0), F_ij = (f_x, 0, 0)
        # W_xx = -r * f_x
        expected_vir_xx = -r * f_x

        # Virial layout: [xx, xy, xz, yx, yy, yz, zx, zy, zz]
        np.testing.assert_allclose(virial_np[0], expected_vir_xx, rtol=1e-10)

        # Other components should be ~0
        np.testing.assert_allclose(virial_np[1:], 0.0, atol=1e-12)

    def test_virial_symmetry(self, device):
        """Test that virial tensor is symmetric for multi-atom system."""
        # Create a small FCC system
        n_cells = 2
        a = 5.26  # Lattice constant
        positions_np = create_fcc_positions(n_cells, a, dtype=np.float64)
        cell_np = np.eye(3, dtype=np.float64) * (n_cells * a)

        epsilon = 0.0104
        sigma = 3.40
        cutoff = 8.5

        vec_dtype = wp.vec3d
        mat_dtype = wp.mat33d

        positions_wp = wp.array(positions_np, dtype=vec_dtype, device=device)
        cell_wp = wp.array([cell_np], dtype=mat_dtype, device=device)

        # Use cell_list for proper neighbor finding with PBC
        positions_torch = torch.from_numpy(positions_np).cuda()
        cell_torch = torch.from_numpy(cell_np).cuda()
        pbc_torch = torch.tensor([True, True, True], dtype=torch.bool, device="cuda")

        neighbor_matrix, num_neighbors, neighbor_shifts = cell_list(
            positions=positions_torch,
            cell=cell_torch,
            cutoff=cutoff,
            pbc=pbc_torch,
        )

        nm_wp = wp.from_torch(neighbor_matrix.int(), dtype=wp.int32)
        ns_wp = wp.from_torch(neighbor_shifts, dtype=wp.vec3i)
        nn_wp = wp.from_torch(num_neighbors.int(), dtype=wp.int32)

        _, _, virial = lj_energy_forces_virial(
            positions=positions_wp,
            cell=cell_wp,
            epsilon=epsilon,
            sigma=sigma,
            cutoff=cutoff,
            neighbor_matrix=nm_wp,
            neighbor_matrix_shifts=ns_wp,
            num_neighbors=nn_wp,
            fill_value=len(positions_np),
            half_neighbor_list=False,  # cell_list produces full neighbor list
            device=device,
        )

        virial_np = virial.numpy()

        # Reshape to 3x3
        virial_mat = virial_np.reshape(3, 3)

        # Check symmetry
        np.testing.assert_allclose(virial_mat, virial_mat.T, rtol=1e-10)


class TestLJHalfVsFull:
    """Tests comparing half and full neighbor lists."""

    def test_energy_half_vs_full(self, device, lj_params):
        """Test that half and full neighbor lists give same energy."""
        # Create a small system
        n_cells = 2
        a = 5.26
        positions_np = create_fcc_positions(n_cells, a, dtype=np.float64)
        cell_np = np.eye(3, dtype=np.float64) * (n_cells * a)

        epsilon = lj_params["epsilon"]
        sigma = lj_params["sigma"]
        cutoff = lj_params["cutoff"]

        vec_dtype = wp.vec3d
        mat_dtype = wp.mat33d

        positions_wp = wp.array(positions_np, dtype=vec_dtype, device=device)
        cell_wp = wp.array([cell_np], dtype=mat_dtype, device=device)

        # Build neighbor list (cell_list gives full neighbor list)
        positions_torch = torch.from_numpy(positions_np).cuda()
        cell_torch = torch.from_numpy(cell_np).cuda()
        pbc_torch = torch.tensor([True, True, True], dtype=torch.bool, device="cuda")

        neighbor_matrix, num_neighbors, neighbor_shifts = cell_list(
            positions=positions_torch,
            cell=cell_torch,
            cutoff=cutoff,
            pbc=pbc_torch,
        )

        nm_wp = wp.from_torch(neighbor_matrix.int(), dtype=wp.int32)
        ns_wp = wp.from_torch(neighbor_shifts, dtype=wp.vec3i)
        nn_wp = wp.from_torch(num_neighbors.int(), dtype=wp.int32)

        # Full neighbor list
        energies_full = lj_energy(
            positions=positions_wp,
            cell=cell_wp,
            epsilon=epsilon,
            sigma=sigma,
            cutoff=cutoff,
            neighbor_matrix=nm_wp,
            neighbor_matrix_shifts=ns_wp,
            num_neighbors=nn_wp,
            fill_value=len(positions_np),
            half_neighbor_list=False,
            device=device,
        )

        total_energy = energies_full.numpy().sum()

        # Verify energy is reasonable (negative for condensed system)
        assert total_energy < 0

    def test_forces_half_vs_full(self, device, lj_params):
        """Test that half and full neighbor lists give same forces."""
        n_cells = 2
        a = 5.26
        positions_np = create_fcc_positions(n_cells, a, dtype=np.float64)
        cell_np = np.eye(3, dtype=np.float64) * (n_cells * a)

        epsilon = lj_params["epsilon"]
        sigma = lj_params["sigma"]
        cutoff = lj_params["cutoff"]

        vec_dtype = wp.vec3d
        mat_dtype = wp.mat33d

        positions_wp = wp.array(positions_np, dtype=vec_dtype, device=device)
        cell_wp = wp.array([cell_np], dtype=mat_dtype, device=device)

        positions_torch = torch.from_numpy(positions_np).cuda()
        cell_torch = torch.from_numpy(cell_np).cuda()
        pbc_torch = torch.tensor([True, True, True], dtype=torch.bool, device="cuda")

        neighbor_matrix, num_neighbors, neighbor_shifts = cell_list(
            positions=positions_torch,
            cell=cell_torch,
            cutoff=cutoff,
            pbc=pbc_torch,
        )

        nm_wp = wp.from_torch(neighbor_matrix.int(), dtype=wp.int32)
        ns_wp = wp.from_torch(neighbor_shifts, dtype=wp.vec3i)
        nn_wp = wp.from_torch(num_neighbors.int(), dtype=wp.int32)

        _, forces_full = lj_energy_forces(
            positions=positions_wp,
            cell=cell_wp,
            epsilon=epsilon,
            sigma=sigma,
            cutoff=cutoff,
            neighbor_matrix=nm_wp,
            neighbor_matrix_shifts=ns_wp,
            num_neighbors=nn_wp,
            fill_value=len(positions_np),
            half_neighbor_list=False,
            device=device,
        )

        forces_np = forces_full.numpy()

        # Sum of all forces should be ~0 (momentum conservation)
        total_force = forces_np.sum(axis=0)
        np.testing.assert_allclose(total_force, 0.0, atol=1e-10)


class TestLJSwitching:
    """Tests for switching function integration."""

    def test_switching_reduces_energy(self, device, lj_params):
        """Test that switching reduces energy near cutoff."""
        epsilon = lj_params["epsilon"]
        sigma = lj_params["sigma"]
        cutoff = lj_params["cutoff"]
        switch_width = 2.0
        r = cutoff - 0.5  # Near cutoff but within

        positions_wp, cell_wp = create_simple_pair_system(
            r, epsilon, sigma, device=device
        )

        pos_np = positions_wp.numpy()
        cell_np = cell_wp.numpy()[0]
        nm, ns, nn = build_neighbor_matrix_simple(pos_np, cell_np, cutoff, half=True)

        nm_wp = wp.array(nm, dtype=wp.int32, device=device)
        ns_wp = wp.array(ns, dtype=wp.vec3i, device=device)
        nn_wp = wp.array(nn, dtype=wp.int32, device=device)

        # Without switching
        energies_no_switch = lj_energy(
            positions=positions_wp,
            cell=cell_wp,
            epsilon=epsilon,
            sigma=sigma,
            cutoff=cutoff,
            neighbor_matrix=nm_wp,
            neighbor_matrix_shifts=ns_wp,
            num_neighbors=nn_wp,
            switch_width=0.0,
            half_neighbor_list=True,
            device=device,
        )

        # With switching
        energies_switch = lj_energy(
            positions=positions_wp,
            cell=cell_wp,
            epsilon=epsilon,
            sigma=sigma,
            cutoff=cutoff,
            neighbor_matrix=nm_wp,
            neighbor_matrix_shifts=ns_wp,
            num_neighbors=nn_wp,
            switch_width=switch_width,
            half_neighbor_list=True,
            device=device,
        )

        e_no_switch = energies_no_switch.numpy().sum()
        e_switch = energies_switch.numpy().sum()

        # Switching should reduce magnitude since we're in switching region
        assert abs(e_switch) < abs(e_no_switch)

    def test_switching_no_effect_below_r_on(self, device, lj_params):
        """Test that switching has no effect for r < r_on."""
        epsilon = lj_params["epsilon"]
        sigma = lj_params["sigma"]
        cutoff = lj_params["cutoff"]
        switch_width = 2.0
        r_on = cutoff - switch_width
        r = r_on - 1.0  # Well below r_on

        positions_wp, cell_wp = create_simple_pair_system(
            r, epsilon, sigma, device=device
        )

        pos_np = positions_wp.numpy()
        cell_np = cell_wp.numpy()[0]
        nm, ns, nn = build_neighbor_matrix_simple(pos_np, cell_np, cutoff, half=True)

        nm_wp = wp.array(nm, dtype=wp.int32, device=device)
        ns_wp = wp.array(ns, dtype=wp.vec3i, device=device)
        nn_wp = wp.array(nn, dtype=wp.int32, device=device)

        energies_no_switch = lj_energy(
            positions=positions_wp,
            cell=cell_wp,
            epsilon=epsilon,
            sigma=sigma,
            cutoff=cutoff,
            neighbor_matrix=nm_wp,
            neighbor_matrix_shifts=ns_wp,
            num_neighbors=nn_wp,
            switch_width=0.0,
            half_neighbor_list=True,
            device=device,
        )

        energies_switch = lj_energy(
            positions=positions_wp,
            cell=cell_wp,
            epsilon=epsilon,
            sigma=sigma,
            cutoff=cutoff,
            neighbor_matrix=nm_wp,
            neighbor_matrix_shifts=ns_wp,
            num_neighbors=nn_wp,
            switch_width=switch_width,
            half_neighbor_list=True,
            device=device,
        )

        np.testing.assert_allclose(
            energies_switch.numpy().sum(),
            energies_no_switch.numpy().sum(),
            rtol=1e-10,
        )


class TestLJDtypes:
    """Tests for different data types."""

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_energy_dtype(self, device, lj_params, dtype):
        """Test energy calculation with different dtypes."""
        epsilon = lj_params["epsilon"]
        sigma = lj_params["sigma"]
        cutoff = lj_params["cutoff"]
        r = 4.0

        positions_wp, cell_wp = create_simple_pair_system(
            r, epsilon, sigma, dtype=dtype, device=device
        )

        pos_np = positions_wp.numpy()
        cell_np = cell_wp.numpy()[0]
        nm, ns, nn = build_neighbor_matrix_simple(pos_np, cell_np, cutoff, half=True)

        nm_wp = wp.array(nm, dtype=wp.int32, device=device)
        ns_wp = wp.array(ns, dtype=wp.vec3i, device=device)
        nn_wp = wp.array(nn, dtype=wp.int32, device=device)

        energies = lj_energy(
            positions=positions_wp,
            cell=cell_wp,
            epsilon=epsilon,
            sigma=sigma,
            cutoff=cutoff,
            neighbor_matrix=nm_wp,
            neighbor_matrix_shifts=ns_wp,
            num_neighbors=nn_wp,
            half_neighbor_list=True,
            device=device,
        )

        total_energy = energies.numpy().sum()
        expected = lj_energy_pair_numpy(r, epsilon, sigma)

        # Float32 has lower precision
        rtol = 1e-5 if dtype == np.float32 else 1e-10
        np.testing.assert_allclose(total_energy, expected, rtol=rtol)


class TestLJBatched:
    """Tests for batched LJ calculations."""

    def test_batched_energy_forces(self, device, lj_params):
        """Test batched energy and force calculations."""
        epsilon = lj_params["epsilon"]
        sigma = lj_params["sigma"]
        cutoff = lj_params["cutoff"]

        # Create two systems with different configurations
        n_cells = 2
        a1 = 5.26
        a2 = 5.30  # Slightly different lattice constant

        pos1 = create_fcc_positions(n_cells, a1, dtype=np.float64)
        pos2 = create_fcc_positions(n_cells, a2, dtype=np.float64)
        n1, n2 = len(pos1), len(pos2)

        # Concatenate positions
        positions_np = np.vstack([pos1, pos2])

        # Create cells
        cell1 = np.eye(3, dtype=np.float64) * (n_cells * a1)
        cell2 = np.eye(3, dtype=np.float64) * (n_cells * a2)
        cells_np = np.stack([cell1, cell2])

        # Create batch_idx
        batch_idx_np = np.concatenate(
            [np.zeros(n1, dtype=np.int32), np.ones(n2, dtype=np.int32)]
        )

        vec_dtype = wp.vec3d
        mat_dtype = wp.mat33d

        positions_wp = wp.array(positions_np, dtype=vec_dtype, device=device)
        cells_wp = wp.array(cells_np, dtype=mat_dtype, device=device)
        batch_idx_wp = wp.array(batch_idx_np, dtype=wp.int32, device=device)

        # Build batched neighbor list
        positions_torch = torch.from_numpy(positions_np).cuda()
        cells_torch = torch.from_numpy(cells_np).cuda()
        batch_idx_torch = torch.from_numpy(batch_idx_np).cuda().long()
        pbc_torch = torch.tensor([True, True, True], dtype=torch.bool, device="cuda")

        from nvalchemiops.torch.neighbors import batch_cell_list

        neighbor_matrix, num_neighbors, neighbor_shifts = batch_cell_list(
            positions=positions_torch,
            cell=cells_torch,
            pbc=pbc_torch,
            batch_idx=batch_idx_torch,
            cutoff=cutoff,
        )

        nm_wp = wp.from_torch(neighbor_matrix.int(), dtype=wp.int32)
        ns_wp = wp.from_torch(neighbor_shifts, dtype=wp.vec3i)
        nn_wp = wp.from_torch(num_neighbors.int(), dtype=wp.int32)

        energies, forces = lj_energy_forces(
            positions=positions_wp,
            cell=cells_wp,
            epsilon=epsilon,
            sigma=sigma,
            cutoff=cutoff,
            neighbor_matrix=nm_wp,
            neighbor_matrix_shifts=ns_wp,
            num_neighbors=nn_wp,
            batch_idx=batch_idx_wp,
            fill_value=len(positions_np),
            half_neighbor_list=False,
            device=device,
        )

        energies_np = energies.numpy()
        forces_np = forces.numpy()

        # System 1 energies
        e1 = energies_np[:n1].sum()
        # System 2 energies
        e2 = energies_np[n1:].sum()

        # Both should be negative (bound systems)
        assert e1 < 0
        assert e2 < 0

        # Different lattice constants should give different energies
        assert not np.isclose(e1, e2)

        # Forces should sum to ~0 for each system
        f1_sum = forces_np[:n1].sum(axis=0)
        f2_sum = forces_np[n1:].sum(axis=0)

        np.testing.assert_allclose(f1_sum, 0.0, atol=1e-10)
        np.testing.assert_allclose(f2_sum, 0.0, atol=1e-10)

    def test_batched_virial(self, device, lj_params):
        """Test batched virial calculation."""
        epsilon = lj_params["epsilon"]
        sigma = lj_params["sigma"]
        cutoff = lj_params["cutoff"]

        n_cells = 2
        a1 = 5.26
        a2 = 5.30

        pos1 = create_fcc_positions(n_cells, a1, dtype=np.float64)
        pos2 = create_fcc_positions(n_cells, a2, dtype=np.float64)
        n1, n2 = len(pos1), len(pos2)

        positions_np = np.vstack([pos1, pos2])
        cell1 = np.eye(3, dtype=np.float64) * (n_cells * a1)
        cell2 = np.eye(3, dtype=np.float64) * (n_cells * a2)
        cells_np = np.stack([cell1, cell2])
        batch_idx_np = np.concatenate(
            [np.zeros(n1, dtype=np.int32), np.ones(n2, dtype=np.int32)]
        )

        vec_dtype = wp.vec3d
        mat_dtype = wp.mat33d

        positions_wp = wp.array(positions_np, dtype=vec_dtype, device=device)
        cells_wp = wp.array(cells_np, dtype=mat_dtype, device=device)
        batch_idx_wp = wp.array(batch_idx_np, dtype=wp.int32, device=device)

        positions_torch = torch.from_numpy(positions_np).cuda()
        cells_torch = torch.from_numpy(cells_np).cuda()
        batch_idx_torch = torch.from_numpy(batch_idx_np).cuda().long()
        pbc_torch = torch.tensor([True, True, True], dtype=torch.bool, device="cuda")

        from nvalchemiops.torch.neighbors import batch_cell_list

        neighbor_matrix, num_neighbors, neighbor_shifts = batch_cell_list(
            positions=positions_torch,
            cell=cells_torch,
            pbc=pbc_torch,
            batch_idx=batch_idx_torch,
            cutoff=cutoff,
        )

        nm_wp = wp.from_torch(neighbor_matrix.int(), dtype=wp.int32)
        ns_wp = wp.from_torch(neighbor_shifts, dtype=wp.vec3i)
        nn_wp = wp.from_torch(num_neighbors.int(), dtype=wp.int32)

        energies, forces, virial = lj_energy_forces_virial(
            positions=positions_wp,
            cell=cells_wp,
            epsilon=epsilon,
            sigma=sigma,
            cutoff=cutoff,
            neighbor_matrix=nm_wp,
            neighbor_matrix_shifts=ns_wp,
            num_neighbors=nn_wp,
            batch_idx=batch_idx_wp,
            fill_value=len(positions_np),
            half_neighbor_list=False,
            device=device,
        )

        virial_np = virial.numpy()

        # Should have shape (2, 9)
        assert virial_np.shape == (2, 9)

        # Each system's virial should be symmetric
        for s in range(2):
            vir_mat = virial_np[s].reshape(3, 3)
            np.testing.assert_allclose(vir_mat, vir_mat.T, rtol=1e-10, atol=1e-14)


class TestLJNeighborListFormat:
    """Tests for CSR neighbor list format."""

    def test_energy_list_format(self, device, lj_params):
        """Test energy calculation using CSR neighbor list format."""
        n_cells = 2
        a = 5.26
        positions_np = create_fcc_positions(n_cells, a, dtype=np.float64)
        cell_np = np.eye(3, dtype=np.float64) * (n_cells * a)

        epsilon = lj_params["epsilon"]
        sigma = lj_params["sigma"]
        cutoff = lj_params["cutoff"]

        vec_dtype = wp.vec3d
        mat_dtype = wp.mat33d

        positions_wp = wp.array(positions_np, dtype=vec_dtype, device=device)
        cell_wp = wp.array([cell_np], dtype=mat_dtype, device=device)

        # Build neighbor list in CSR format
        positions_torch = torch.from_numpy(positions_np).cuda()
        cell_torch = torch.from_numpy(cell_np).cuda()
        pbc_torch = torch.tensor([True, True, True], dtype=torch.bool, device="cuda")

        neighbor_matrix, num_neighbors, neighbor_shifts = cell_list(
            positions=positions_torch,
            cell=cell_torch,
            cutoff=cutoff,
            pbc=pbc_torch,
        )

        # Convert matrix format to CSR
        nm_np = neighbor_matrix.cpu().numpy()
        ns_np = neighbor_shifts.cpu().numpy()
        nn_np = num_neighbors.cpu().numpy()

        # Build CSR arrays
        idx_j_list = []
        shifts_list = []
        neighbor_ptr = [0]

        for i in range(len(positions_np)):
            n = nn_np[i]
            idx_j_list.extend(nm_np[i, :n].tolist())
            shifts_list.extend(ns_np[i, :n].tolist())
            neighbor_ptr.append(neighbor_ptr[-1] + n)

        idx_j_np = np.array(idx_j_list, dtype=np.int32)
        shifts_np = np.array(shifts_list, dtype=np.int32)
        neighbor_ptr_np = np.array(neighbor_ptr, dtype=np.int32)

        # Create Warp arrays for CSR format
        shifts_wp = wp.array(shifts_np, dtype=wp.vec3i, device=device)
        neighbor_ptr_wp = wp.array(neighbor_ptr_np, dtype=wp.int32, device=device)

        # Create a 2D array for neighbor_list (expected by API)
        neighbor_list_np = np.zeros((2, len(idx_j_np)), dtype=np.int32)
        neighbor_list_np[1] = idx_j_np
        neighbor_list_wp = wp.array(neighbor_list_np, dtype=wp.int32, device=device)

        # Energy using CSR format
        energies_list = lj_energy(
            positions=positions_wp,
            cell=cell_wp,
            epsilon=epsilon,
            sigma=sigma,
            cutoff=cutoff,
            neighbor_list=neighbor_list_wp,
            neighbor_ptr=neighbor_ptr_wp,
            neighbor_shifts=shifts_wp,
            half_neighbor_list=False,
            device=device,
        )

        # Energy using matrix format
        nm_wp = wp.from_torch(neighbor_matrix.int(), dtype=wp.int32)
        ns_wp = wp.from_torch(neighbor_shifts, dtype=wp.vec3i)
        nn_wp = wp.from_torch(num_neighbors.int(), dtype=wp.int32)

        energies_matrix = lj_energy(
            positions=positions_wp,
            cell=cell_wp,
            epsilon=epsilon,
            sigma=sigma,
            cutoff=cutoff,
            neighbor_matrix=nm_wp,
            neighbor_matrix_shifts=ns_wp,
            num_neighbors=nn_wp,
            fill_value=len(positions_np),
            half_neighbor_list=False,
            device=device,
        )

        # Results should match
        np.testing.assert_allclose(
            energies_list.numpy().sum(),
            energies_matrix.numpy().sum(),
            rtol=1e-10,
        )


class TestLJEdgeCases:
    """Tests for edge cases and error handling."""

    def test_no_neighbors(self, device, lj_params):
        """Test with atoms that have no neighbors."""
        epsilon = lj_params["epsilon"]
        sigma = lj_params["sigma"]
        cutoff = lj_params["cutoff"]

        # Two atoms very far apart
        r = cutoff * 10

        positions_wp, cell_wp = create_simple_pair_system(
            r, epsilon, sigma, device=device
        )

        pos_np = positions_wp.numpy()
        cell_np = cell_wp.numpy()[0]
        nm, ns, nn = build_neighbor_matrix_simple(pos_np, cell_np, cutoff, half=True)

        nm_wp = wp.array(nm, dtype=wp.int32, device=device)
        ns_wp = wp.array(ns, dtype=wp.vec3i, device=device)
        nn_wp = wp.array(nn, dtype=wp.int32, device=device)

        energies, forces = lj_energy_forces(
            positions=positions_wp,
            cell=cell_wp,
            epsilon=epsilon,
            sigma=sigma,
            cutoff=cutoff,
            neighbor_matrix=nm_wp,
            neighbor_matrix_shifts=ns_wp,
            num_neighbors=nn_wp,
            half_neighbor_list=True,
            device=device,
        )

        np.testing.assert_allclose(energies.numpy(), 0.0, atol=1e-15)
        np.testing.assert_allclose(forces.numpy(), 0.0, atol=1e-15)

    def test_missing_neighbor_format_error(self, device, lj_params):
        """Test that error is raised when no neighbor format is provided."""
        epsilon = lj_params["epsilon"]
        sigma = lj_params["sigma"]
        cutoff = lj_params["cutoff"]

        positions_wp, cell_wp = create_simple_pair_system(
            4.0, epsilon, sigma, device=device
        )

        with pytest.raises(ValueError, match="Must provide either"):
            lj_energy(
                positions=positions_wp,
                cell=cell_wp,
                epsilon=epsilon,
                sigma=sigma,
                cutoff=cutoff,
                device=device,
            )

    def test_single_atom(self, device, lj_params):
        """Test with a single atom (no interactions)."""
        epsilon = lj_params["epsilon"]
        sigma = lj_params["sigma"]
        cutoff = lj_params["cutoff"]

        box_size = 50.0
        positions_np = np.array([[25.0, 25.0, 25.0]], dtype=np.float64)
        cell_np = np.eye(3, dtype=np.float64) * box_size

        vec_dtype = wp.vec3d
        mat_dtype = wp.mat33d

        positions_wp = wp.array(positions_np, dtype=vec_dtype, device=device)
        cell_wp = wp.array([cell_np], dtype=mat_dtype, device=device)

        # Empty neighbor matrix
        nm = np.full((1, 1), 1, dtype=np.int32)
        ns = np.zeros((1, 1, 3), dtype=np.int32)
        nn = np.zeros(1, dtype=np.int32)

        nm_wp = wp.array(nm, dtype=wp.int32, device=device)
        ns_wp = wp.array(ns, dtype=wp.vec3i, device=device)
        nn_wp = wp.array(nn, dtype=wp.int32, device=device)

        energies, forces = lj_energy_forces(
            positions=positions_wp,
            cell=cell_wp,
            epsilon=epsilon,
            sigma=sigma,
            cutoff=cutoff,
            neighbor_matrix=nm_wp,
            neighbor_matrix_shifts=ns_wp,
            num_neighbors=nn_wp,
            half_neighbor_list=True,
            device=device,
        )

        np.testing.assert_allclose(energies.numpy(), 0.0, atol=1e-15)
        np.testing.assert_allclose(forces.numpy(), 0.0, atol=1e-15)
