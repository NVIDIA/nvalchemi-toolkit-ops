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
Tests for NPT and NPH integrators.

This module tests both NPT (isothermal-isobaric) and NPH (isenthalpic-isobaric)
integrators, including API tests, correctness tests, and physics tests.
"""

import numpy as np
import pytest
import warp as wp

from nvalchemiops.dynamics.integrators.npt import (
    compute_barostat_mass,
    compute_barostat_potential_energy,
    compute_cell_kinetic_energy,
    compute_pressure_tensor,
    compute_scalar_pressure,
    nph_barostat_half_step,
    nph_position_update,
    nph_position_update_out,
    nph_velocity_half_step,
    nph_velocity_half_step_out,
    npt_barostat_half_step,
    npt_cell_update,
    npt_cell_update_out,
    npt_position_update,
    npt_position_update_out,
    npt_velocity_half_step,
    npt_velocity_half_step_out,
    run_nph_step,
    run_npt_step,
    vec3d,
    vec3f,
    vec9d,
    vec9f,
)
from nvalchemiops.dynamics.utils.cell_utils import (
    compute_cell_inverse,
    compute_cell_volume,
)

# ==============================================================================
# Test Configuration
# ==============================================================================

DEVICES = ["cuda:0"]


# ==============================================================================
# Helper Functions
# ==============================================================================


def make_cell(cell_np, dtype, device):
    """Create a (1,) shaped cell array from a (3,3) numpy array."""
    mat_dtype = wp.mat33f if dtype == "float32" else wp.mat33d
    np_dtype = np.float32 if dtype == "float32" else np.float64
    cell_np = cell_np.astype(np_dtype)
    mat = mat_dtype(
        cell_np[0, 0],
        cell_np[0, 1],
        cell_np[0, 2],
        cell_np[1, 0],
        cell_np[1, 1],
        cell_np[1, 2],
        cell_np[2, 0],
        cell_np[2, 1],
        cell_np[2, 2],
    )
    return wp.array([mat], dtype=mat_dtype, device=device)


def make_cells_batch(cells_np, dtype, device):
    """Create a (B,) shaped cell array from a (B, 3, 3) numpy array."""
    mat_dtype = wp.mat33f if dtype == "float32" else wp.mat33d
    np_dtype = np.float32 if dtype == "float32" else np.float64
    num_systems = cells_np.shape[0]
    mats = []
    for i in range(num_systems):
        c = cells_np[i].astype(np_dtype)
        mats.append(
            mat_dtype(
                c[0, 0],
                c[0, 1],
                c[0, 2],
                c[1, 0],
                c[1, 1],
                c[1, 2],
                c[2, 0],
                c[2, 1],
                c[2, 2],
            )
        )
    return wp.array(mats, dtype=mat_dtype, device=device)


def setup_npt_system(num_atoms, dtype, device, seed=42, chain_length=3):
    """Set up a test system for NPT simulation."""
    np.random.seed(seed)

    vec_dtype = wp.vec3f if dtype == "float32" else wp.vec3d
    mat_dtype = wp.mat33f if dtype == "float32" else wp.mat33d
    scalar_dtype = wp.float32 if dtype == "float32" else wp.float64
    tensor_dtype = vec9f if dtype == "float32" else vec9d
    np_dtype = np.float32 if dtype == "float32" else np.float64

    positions = wp.array(
        np.random.rand(num_atoms, 3).astype(np_dtype) * 8.0 + 1.0,
        dtype=vec_dtype,
        device=device,
    )
    velocities = wp.array(
        np.random.randn(num_atoms, 3).astype(np_dtype) * 0.3,
        dtype=vec_dtype,
        device=device,
    )
    forces = wp.zeros(num_atoms, dtype=vec_dtype, device=device)
    masses = wp.array(
        np.ones(num_atoms, dtype=np_dtype), dtype=scalar_dtype, device=device
    )

    cell_np = np.diag([10.0, 10.0, 10.0]).astype(np_dtype)
    h_dot_np = np.zeros((3, 3), dtype=np_dtype)

    cell_mat = mat_dtype(
        cell_np[0, 0],
        cell_np[0, 1],
        cell_np[0, 2],
        cell_np[1, 0],
        cell_np[1, 1],
        cell_np[1, 2],
        cell_np[2, 0],
        cell_np[2, 1],
        cell_np[2, 2],
    )
    h_dot_mat = mat_dtype(
        h_dot_np[0, 0],
        h_dot_np[0, 1],
        h_dot_np[0, 2],
        h_dot_np[1, 0],
        h_dot_np[1, 1],
        h_dot_np[1, 2],
        h_dot_np[2, 0],
        h_dot_np[2, 1],
        h_dot_np[2, 2],
    )

    cells = wp.array([cell_mat], dtype=mat_dtype, device=device)
    cell_velocities = wp.array([h_dot_mat], dtype=mat_dtype, device=device)
    virial_tensors = wp.zeros(1, dtype=tensor_dtype, device=device)

    eta = wp.zeros((1, chain_length), dtype=scalar_dtype, device=device)
    eta_dot = wp.zeros((1, chain_length), dtype=scalar_dtype, device=device)
    thermostat_masses = wp.array(
        np.ones((1, chain_length), dtype=np_dtype) * 20.0,
        dtype=scalar_dtype,
        device=device,
    )

    cell_masses = wp.array([200.0], dtype=scalar_dtype, device=device)
    target_temperature = wp.array([1.0], dtype=scalar_dtype, device=device)
    target_pressure = wp.array([0.1], dtype=scalar_dtype, device=device)

    # Scratch arrays for refactored APIs
    pressure_tensors = wp.empty(1, dtype=tensor_dtype, device=device)
    volumes = wp.empty(1, dtype=scalar_dtype, device=device)
    compute_cell_volume(cells, volumes, device=device)
    kinetic_energy = wp.zeros(1, dtype=scalar_dtype, device=device)
    cells_inv = wp.empty(1, dtype=mat_dtype, device=device)
    compute_cell_inverse(cells, cells_inv, device=device)
    kinetic_tensors = wp.zeros((1, 9), dtype=scalar_dtype, device=device)
    num_atoms_per_system = wp.array([num_atoms], dtype=wp.int32, device=device)
    dt = wp.array([0.001], dtype=scalar_dtype, device=device)

    return {
        "positions": positions,
        "velocities": velocities,
        "forces": forces,
        "masses": masses,
        "cells": cells,
        "cell_velocities": cell_velocities,
        "virial_tensors": virial_tensors,
        "eta": eta,
        "eta_dot": eta_dot,
        "thermostat_masses": thermostat_masses,
        "cell_masses": cell_masses,
        "target_temperature": target_temperature,
        "target_pressure": target_pressure,
        "chain_length": chain_length,
        "num_atoms": num_atoms,
        "pressure_tensors": pressure_tensors,
        "volumes": volumes,
        "kinetic_energy": kinetic_energy,
        "cells_inv": cells_inv,
        "kinetic_tensors": kinetic_tensors,
        "num_atoms_per_system": num_atoms_per_system,
        "dt": dt,
    }


def setup_nph_system(num_atoms, dtype, device, seed=42):
    """Set up a test system for NPH simulation (no thermostat state)."""
    np.random.seed(seed)

    vec_dtype = wp.vec3f if dtype == "float32" else wp.vec3d
    mat_dtype = wp.mat33f if dtype == "float32" else wp.mat33d
    scalar_dtype = wp.float32 if dtype == "float32" else wp.float64
    tensor_dtype = vec9f if dtype == "float32" else vec9d
    np_dtype = np.float32 if dtype == "float32" else np.float64

    positions = wp.array(
        np.random.rand(num_atoms, 3).astype(np_dtype) * 8.0 + 1.0,
        dtype=vec_dtype,
        device=device,
    )
    velocities = wp.array(
        np.random.randn(num_atoms, 3).astype(np_dtype) * 0.3,
        dtype=vec_dtype,
        device=device,
    )
    forces = wp.zeros(num_atoms, dtype=vec_dtype, device=device)
    masses = wp.array(
        np.ones(num_atoms, dtype=np_dtype), dtype=scalar_dtype, device=device
    )

    cell_np = np.diag([10.0, 10.0, 10.0]).astype(np_dtype)
    h_dot_np = np.zeros((3, 3), dtype=np_dtype)

    cell_mat = mat_dtype(
        cell_np[0, 0],
        cell_np[0, 1],
        cell_np[0, 2],
        cell_np[1, 0],
        cell_np[1, 1],
        cell_np[1, 2],
        cell_np[2, 0],
        cell_np[2, 1],
        cell_np[2, 2],
    )
    h_dot_mat = mat_dtype(
        h_dot_np[0, 0],
        h_dot_np[0, 1],
        h_dot_np[0, 2],
        h_dot_np[1, 0],
        h_dot_np[1, 1],
        h_dot_np[1, 2],
        h_dot_np[2, 0],
        h_dot_np[2, 1],
        h_dot_np[2, 2],
    )

    cells = wp.array([cell_mat], dtype=mat_dtype, device=device)
    cell_velocities = wp.array([h_dot_mat], dtype=mat_dtype, device=device)
    virial_tensors = wp.zeros(1, dtype=tensor_dtype, device=device)

    cell_masses = wp.array([200.0], dtype=scalar_dtype, device=device)
    target_pressure = wp.array([0.1], dtype=scalar_dtype, device=device)

    # Scratch arrays for refactored APIs
    pressure_tensors = wp.empty(1, dtype=tensor_dtype, device=device)
    volumes = wp.empty(1, dtype=scalar_dtype, device=device)
    compute_cell_volume(cells, volumes, device=device)
    kinetic_energy = wp.zeros(1, dtype=scalar_dtype, device=device)
    cells_inv = wp.empty(1, dtype=mat_dtype, device=device)
    compute_cell_inverse(cells, cells_inv, device=device)
    kinetic_tensors = wp.zeros((1, 9), dtype=scalar_dtype, device=device)
    num_atoms_per_system = wp.array([num_atoms], dtype=wp.int32, device=device)
    dt = wp.array([0.001], dtype=scalar_dtype, device=device)

    return {
        "positions": positions,
        "velocities": velocities,
        "forces": forces,
        "masses": masses,
        "cells": cells,
        "cell_velocities": cell_velocities,
        "virial_tensors": virial_tensors,
        "cell_masses": cell_masses,
        "target_pressure": target_pressure,
        "num_atoms": num_atoms,
        "pressure_tensors": pressure_tensors,
        "volumes": volumes,
        "kinetic_energy": kinetic_energy,
        "cells_inv": cells_inv,
        "kinetic_tensors": kinetic_tensors,
        "num_atoms_per_system": num_atoms_per_system,
        "dt": dt,
    }


# ==============================================================================
# 1. Pressure Tensor API Tests
# ==============================================================================


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
class TestPressureTensorAPI:
    """API tests for pressure tensor computation."""

    def test_compute_pressure_tensor_runs(self, dtype, device):
        """Test that compute_pressure_tensor runs without errors."""
        vec_dtype = wp.vec3f if dtype == "float32" else wp.vec3d
        scalar_dtype = wp.float32 if dtype == "float32" else wp.float64
        tensor_dtype = vec9f if dtype == "float32" else vec9d
        np_dtype = np.float32 if dtype == "float32" else np.float64

        num_atoms = 10
        np.random.seed(42)

        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=vec_dtype,
            device=device,
        )
        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype), dtype=scalar_dtype, device=device
        )
        virial_tensors = wp.zeros(1, dtype=tensor_dtype, device=device)

        cell_np = np.diag([10.0, 10.0, 10.0])
        cells = make_cell(cell_np, dtype, device)

        kinetic_tensors = wp.zeros((1, 9), dtype=scalar_dtype, device=device)
        pressure_tensors = wp.empty(1, dtype=tensor_dtype, device=device)
        volumes = wp.empty(1, dtype=scalar_dtype, device=device)
        compute_cell_volume(cells, volumes, device=device)

        pressure_tensors = compute_pressure_tensor(
            velocities,
            masses,
            virial_tensors,
            cells,
            kinetic_tensors,
            pressure_tensors,
            volumes,
            device=device,
        )
        wp.synchronize_device(device)

        assert pressure_tensors.shape[0] == 1
        assert pressure_tensors.dtype == tensor_dtype

    def test_compute_scalar_pressure_runs(self, dtype, device):
        """Test that compute_scalar_pressure runs without errors."""
        tensor_dtype = vec9f if dtype == "float32" else vec9d
        scalar_dtype = wp.float32 if dtype == "float32" else wp.float64
        np_dtype = np.float32 if dtype == "float32" else np.float64

        tensor_np = np.zeros(9, dtype=np_dtype)
        tensor_np[0] = 1.0  # xx
        tensor_np[4] = 2.0  # yy
        tensor_np[8] = 3.0  # zz

        tensor_vec = tensor_dtype(*tensor_np)
        pressure_tensors = wp.array([tensor_vec], dtype=tensor_dtype, device=device)

        scalar_pressures_out = wp.empty(1, dtype=scalar_dtype, device=device)
        scalar_pressures = compute_scalar_pressure(
            pressure_tensors,
            scalar_pressures_out,
            device=device,
        )
        wp.synchronize_device(device)

        assert scalar_pressures.shape[0] == 1
        expected = (1.0 + 2.0 + 3.0) / 3.0
        np.testing.assert_allclose(scalar_pressures.numpy()[0], expected, rtol=1e-5)


# ==============================================================================
# 2. Barostat Utilities API Tests
# ==============================================================================


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
class TestBarostatUtilitiesAPI:
    """API tests for barostat utilities."""

    def test_compute_barostat_mass(self, dtype, device):
        """Test barostat mass computation."""
        scalar_dtype = wp.float32 if dtype == "float32" else wp.float64

        target_temp = wp.array([1.0], dtype=scalar_dtype, device=device)
        tau_p_arr = wp.array([1.0], dtype=scalar_dtype, device=device)
        num_atoms_arr = wp.array([100], dtype=wp.int32, device=device)
        masses_out = wp.empty(1, dtype=scalar_dtype, device=device)

        W = compute_barostat_mass(
            target_temp,
            tau_p_arr,
            num_atoms_arr,
            masses_out,
            device=device,
        )
        # W = (N_f + d) * kT * τ²  = (300 + 3) * 1.0 * 1.0 = 303
        W_np = W.numpy()
        assert W_np[0] > 0
        np.testing.assert_allclose(W_np[0], 303.0, rtol=1e-5)

    def test_compute_cell_kinetic_energy_runs(self, dtype, device):
        """Test cell kinetic energy computation."""
        mat_dtype = wp.mat33f if dtype == "float32" else wp.mat33d
        scalar_dtype = wp.float32 if dtype == "float32" else wp.float64
        np_dtype = np.float32 if dtype == "float32" else np.float64

        h_dot_np = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]], dtype=np_dtype)
        h_dot_mat = mat_dtype(
            h_dot_np[0, 0],
            h_dot_np[0, 1],
            h_dot_np[0, 2],
            h_dot_np[1, 0],
            h_dot_np[1, 1],
            h_dot_np[1, 2],
            h_dot_np[2, 0],
            h_dot_np[2, 1],
            h_dot_np[2, 2],
        )
        cell_velocities = wp.array([h_dot_mat], dtype=mat_dtype, device=device)
        cells_inv = wp.array(
            [mat_dtype(1, 0, 0, 0, 1, 0, 0, 0, 1)], dtype=mat_dtype, device=device
        )
        cell_masses = wp.array([100.0], dtype=scalar_dtype, device=device)
        ke_out = wp.empty(1, dtype=scalar_dtype, device=device)

        ke = compute_cell_kinetic_energy(
            cell_velocities, cell_masses, ke_out, cells_inv=cells_inv, device=device
        )
        wp.synchronize_device(device)

        assert ke.shape[0] == 1
        # KE = 0.5 * W * ||ε̇||²_F where ε̇ = ḣ h⁻¹
        # With h⁻¹ = I: ε̇ = ḣ, so ||ε̇||²_F = 3 * 0.1² = 0.03
        expected = 0.5 * 100.0 * 0.03
        np.testing.assert_allclose(ke.numpy()[0], expected, rtol=1e-5)

    def test_compute_cell_kinetic_energy_non_identity_cell(self, dtype, device):
        """Test cell KE with non-identity h⁻¹ to exercise strain-rate path."""
        mat_dtype = wp.mat33f if dtype == "float32" else wp.mat33d
        scalar_dtype = wp.float32 if dtype == "float32" else wp.float64
        np_dtype = np.float32 if dtype == "float32" else np.float64

        # Non-cubic cell: h = diag(2, 3, 4), h⁻¹ = diag(0.5, 1/3, 0.25)
        h_inv_np = np.diag([0.5, 1.0 / 3.0, 0.25]).astype(np_dtype)
        # ḣ = diag(0.2, 0.3, 0.4)
        h_dot_np = np.diag([0.2, 0.3, 0.4]).astype(np_dtype)
        # ε̇ = ḣ h⁻¹ = diag(0.1, 0.1, 0.1) — same strain rate, different ḣ
        eps_dot_np = h_dot_np @ h_inv_np

        h_dot_mat = mat_dtype(*h_dot_np.flatten())
        h_inv_mat = mat_dtype(*h_inv_np.flatten())
        cell_velocities = wp.array([h_dot_mat], dtype=mat_dtype, device=device)
        cells_inv = wp.array([h_inv_mat], dtype=mat_dtype, device=device)
        W = 100.0
        cell_masses = wp.array([W], dtype=scalar_dtype, device=device)
        ke_out = wp.empty(1, dtype=scalar_dtype, device=device)

        ke = compute_cell_kinetic_energy(
            cell_velocities, cell_masses, ke_out, cells_inv=cells_inv, device=device
        )
        wp.synchronize_device(device)

        # KE = 0.5 * W * ||ε̇||²_F = 0.5 * 100 * (0.1² + 0.1² + 0.1²) = 1.5
        frob_sq = float(np.sum(eps_dot_np**2))
        expected = 0.5 * W * frob_sq
        np.testing.assert_allclose(ke.numpy()[0], expected, rtol=1e-5)

        # Verify this differs from naive ||ḣ||²_F
        naive_frob_sq = float(np.sum(h_dot_np**2))
        assert not np.isclose(frob_sq, naive_frob_sq), (
            "Test is degenerate: ||ε̇||² == ||ḣ||² — use a non-identity h⁻¹"
        )

    def test_compute_cell_kinetic_energy_volumes_fallback(self, dtype, device):
        """Test cell KE using volumes-only fallback (no cells_inv) for non-unit cubic cell."""
        mat_dtype = wp.mat33f if dtype == "float32" else wp.mat33d
        scalar_dtype = wp.float32 if dtype == "float32" else wp.float64

        L = 10.0
        V = L**3
        h_dot_np = np.diag([0.1, 0.1, 0.1])
        h_dot_mat = mat_dtype(*h_dot_np.flatten())
        cell_velocities = wp.array([h_dot_mat], dtype=mat_dtype, device=device)

        volumes = wp.array([V], dtype=scalar_dtype, device=device)
        W = 100.0
        cell_masses = wp.array([W], dtype=scalar_dtype, device=device)
        ke_out = wp.empty(1, dtype=scalar_dtype, device=device)

        ke = compute_cell_kinetic_energy(
            cell_velocities, cell_masses, ke_out, volumes=volumes, device=device
        )
        wp.synchronize_device(device)

        # ε̇ = ḣ · V^{-1/3}·I = diag(0.01, 0.01, 0.01)
        eps_dot = 0.1 / L
        expected = 0.5 * W * 3.0 * eps_dot**2
        np.testing.assert_allclose(ke.numpy()[0], expected, rtol=1e-4)

    def test_compute_barostat_potential_energy_runs(self, dtype, device):
        """Test barostat potential energy computation."""
        scalar_dtype = wp.float32 if dtype == "float32" else wp.float64

        target_pressures = wp.array([0.1], dtype=scalar_dtype, device=device)
        volumes = wp.array([1000.0], dtype=scalar_dtype, device=device)
        pe_out = wp.empty(1, dtype=scalar_dtype, device=device)

        pe = compute_barostat_potential_energy(
            target_pressures,
            volumes,
            pe_out,
            device=device,
        )
        wp.synchronize_device(device)

        assert pe.shape[0] == 1
        expected = 0.1 * 1000.0
        np.testing.assert_allclose(pe.numpy()[0], expected, rtol=1e-5)


# ==============================================================================
# 3. NPT Integration API Tests
# ==============================================================================


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
class TestNPTIntegrationAPI:
    """API tests for NPT integration functions."""

    def test_npt_velocity_half_step_runs(self, dtype, device):
        """Test that npt_velocity_half_step runs without errors."""
        system = setup_npt_system(10, dtype, device)

        npt_velocity_half_step(
            system["velocities"],
            system["masses"],
            system["forces"],
            system["cell_velocities"],
            system["volumes"],
            system["eta_dot"],
            system["num_atoms_per_system"],
            dt=system["dt"],
            cells_inv=system["cells_inv"],
            device=device,
        )
        wp.synchronize_device(device)

    def test_npt_position_update_runs(self, dtype, device):
        """Test that npt_position_update runs without errors."""
        system = setup_npt_system(10, dtype, device)

        npt_position_update(
            system["positions"],
            system["velocities"],
            system["cells"],
            system["cell_velocities"],
            dt=system["dt"],
            cells_inv=system["cells_inv"],
            device=device,
        )
        wp.synchronize_device(device)

    def test_npt_cell_update_runs(self, dtype, device):
        """Test that npt_cell_update runs without errors."""
        system = setup_npt_system(10, dtype, device)

        npt_cell_update(
            system["cells"], system["cell_velocities"], dt=system["dt"], device=device
        )
        wp.synchronize_device(device)

    def test_run_npt_step_runs(self, dtype, device):
        """Test that run_npt_step runs without errors."""
        system = setup_npt_system(10, dtype, device)

        run_npt_step(
            system["positions"],
            system["velocities"],
            system["forces"],
            system["masses"],
            system["cells"],
            system["cell_velocities"],
            system["virial_tensors"],
            system["eta"],
            system["eta_dot"],
            system["thermostat_masses"],
            system["cell_masses"],
            system["target_temperature"],
            system["target_pressure"],
            system["num_atoms_per_system"],
            system["chain_length"],
            dt=system["dt"],
            pressure_tensors=system["pressure_tensors"],
            volumes=system["volumes"],
            kinetic_energy=system["kinetic_energy"],
            cells_inv=system["cells_inv"],
            kinetic_tensors=system["kinetic_tensors"],
            num_atoms_per_system=system["num_atoms_per_system"],
            device=device,
        )
        wp.synchronize_device(device)


# ==============================================================================
# 4. NPH Integration API Tests
# ==============================================================================


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
class TestNPHIntegrationAPI:
    """API tests for NPH integration functions."""

    def test_nph_velocity_half_step_runs(self, dtype, device):
        """Test that nph_velocity_half_step runs without errors."""
        system = setup_nph_system(10, dtype, device)

        nph_velocity_half_step(
            system["velocities"],
            system["masses"],
            system["forces"],
            system["cell_velocities"],
            system["volumes"],
            system["num_atoms_per_system"],
            dt=system["dt"],
            cells_inv=system["cells_inv"],
            device=device,
        )
        wp.synchronize_device(device)

    def test_nph_position_update_runs(self, dtype, device):
        """Test that nph_position_update runs without errors."""
        system = setup_nph_system(10, dtype, device)

        nph_position_update(
            system["positions"],
            system["velocities"],
            system["cells"],
            system["cell_velocities"],
            dt=system["dt"],
            cells_inv=system["cells_inv"],
            device=device,
        )
        wp.synchronize_device(device)

    def test_run_nph_step_runs(self, dtype, device):
        """Test that run_nph_step runs without errors."""
        system = setup_nph_system(10, dtype, device)

        run_nph_step(
            system["positions"],
            system["velocities"],
            system["forces"],
            system["masses"],
            system["cells"],
            system["cell_velocities"],
            system["virial_tensors"],
            system["cell_masses"],
            system["target_pressure"],
            system["num_atoms_per_system"],
            dt=system["dt"],
            pressure_tensors=system["pressure_tensors"],
            volumes=system["volumes"],
            kinetic_energy=system["kinetic_energy"],
            cells_inv=system["cells_inv"],
            kinetic_tensors=system["kinetic_tensors"],
            num_atoms_per_system=system["num_atoms_per_system"],
            device=device,
        )
        wp.synchronize_device(device)


# ==============================================================================
# 5. Non-Mutating API Tests
# ==============================================================================


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
class TestNonMutatingAPIs:
    """Tests for non-mutating (out) API variants."""

    def test_npt_velocity_half_step_out_runs(self, dtype, device):
        """Test that npt_velocity_half_step_out runs without errors."""
        system = setup_npt_system(10, dtype, device)

        velocities_out = wp.empty_like(system["velocities"])
        velocities_out = npt_velocity_half_step_out(
            system["velocities"],
            system["masses"],
            system["forces"],
            system["cell_velocities"],
            system["volumes"],
            system["eta_dot"],
            system["num_atoms_per_system"],
            system["dt"],
            velocities_out,
            cells_inv=system["cells_inv"],
            device=device,
        )
        wp.synchronize_device(device)

        assert velocities_out.shape[0] == system["num_atoms"]

    def test_npt_position_update_out_runs(self, dtype, device):
        """Test that npt_position_update_out runs without errors."""
        system = setup_npt_system(10, dtype, device)

        positions_out = wp.empty_like(system["positions"])
        positions_out = npt_position_update_out(
            system["positions"],
            system["velocities"],
            system["cells"],
            system["cell_velocities"],
            system["dt"],
            positions_out,
            cells_inv=system["cells_inv"],
            device=device,
        )
        wp.synchronize_device(device)

        assert positions_out.shape[0] == system["num_atoms"]

    def test_npt_cell_update_out_runs(self, dtype, device):
        """Test that npt_cell_update_out runs without errors."""
        system = setup_npt_system(10, dtype, device)

        cells_out = wp.empty_like(system["cells"])
        cells_out = npt_cell_update_out(
            system["cells"],
            system["cell_velocities"],
            system["dt"],
            cells_out,
            device=device,
        )
        wp.synchronize_device(device)

        assert cells_out.shape[0] == 1

    def test_nph_velocity_half_step_out_runs(self, dtype, device):
        """Test that nph_velocity_half_step_out runs without errors."""
        system = setup_nph_system(10, dtype, device)

        velocities_out = wp.empty_like(system["velocities"])
        velocities_out = nph_velocity_half_step_out(
            system["velocities"],
            system["masses"],
            system["forces"],
            system["cell_velocities"],
            system["volumes"],
            system["num_atoms_per_system"],
            system["dt"],
            velocities_out,
            cells_inv=system["cells_inv"],
            device=device,
        )
        wp.synchronize_device(device)

        assert velocities_out.shape[0] == system["num_atoms"]

    def test_nph_position_update_out_runs(self, dtype, device):
        """Test that nph_position_update_out runs without errors."""
        system = setup_nph_system(10, dtype, device)

        positions_out = wp.empty_like(system["positions"])
        positions_out = nph_position_update_out(
            system["positions"],
            system["velocities"],
            system["cells"],
            system["cell_velocities"],
            system["dt"],
            positions_out,
            cells_inv=system["cells_inv"],
            device=device,
        )
        wp.synchronize_device(device)

        assert positions_out.shape[0] == system["num_atoms"]


# ==============================================================================
# 6. Mutating vs Non-Mutating Consistency Tests
# ==============================================================================


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
class TestMutatingNonMutatingConsistency:
    """Test that mutating and non-mutating APIs give same results."""

    def test_npt_position_update_consistency(self, dtype, device):
        """Test that mutating and non-mutating position update give same results."""
        system1 = setup_npt_system(10, dtype, device, seed=42)
        system2 = setup_npt_system(10, dtype, device, seed=42)

        # Mutating
        npt_position_update(
            system1["positions"],
            system1["velocities"],
            system1["cells"],
            system1["cell_velocities"],
            system1["dt"],
            cells_inv=system1["cells_inv"],
            device=device,
        )

        # Non-mutating
        positions_out = wp.empty_like(system2["positions"])
        positions_out = npt_position_update_out(
            system2["positions"],
            system2["velocities"],
            system2["cells"],
            system2["cell_velocities"],
            system2["dt"],
            positions_out,
            cells_inv=system2["cells_inv"],
            device=device,
        )

        wp.synchronize_device(device)

        np.testing.assert_allclose(
            system1["positions"].numpy(), positions_out.numpy(), rtol=1e-5
        )

    def test_npt_velocity_update_consistency(self, dtype, device):
        """Test that mutating and non-mutating velocity update give same results."""
        system1 = setup_npt_system(10, dtype, device, seed=42)
        system2 = setup_npt_system(10, dtype, device, seed=42)

        volumes = system1["volumes"]
        cells_inv = system1["cells_inv"]

        # Mutating
        npt_velocity_half_step(
            system1["velocities"],
            system1["masses"],
            system1["forces"],
            system1["cell_velocities"],
            volumes,
            system1["eta_dot"],
            system1["num_atoms_per_system"],
            system1["dt"],
            cells_inv=cells_inv,
            device=device,
        )

        # Non-mutating
        velocities_out = wp.empty_like(system2["velocities"])
        velocities_out = npt_velocity_half_step_out(
            system2["velocities"],
            system2["masses"],
            system2["forces"],
            system2["cell_velocities"],
            volumes,
            system2["eta_dot"],
            system2["num_atoms_per_system"],
            system2["dt"],
            velocities_out,
            cells_inv=cells_inv,
            device=device,
        )

        wp.synchronize_device(device)

        np.testing.assert_allclose(
            system1["velocities"].numpy(), velocities_out.numpy(), rtol=1e-5
        )


# ==============================================================================
# 7. Physics Tests - NPT
# ==============================================================================


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
class TestNPTPhysics:
    """Physics tests for NPT integrator."""

    def test_cell_expands_under_high_internal_pressure(self, dtype, device):
        """Test that cell expands when internal pressure exceeds external."""
        system = setup_npt_system(20, dtype, device)

        # Set high velocities to create high internal pressure
        vec_dtype = wp.vec3f if dtype == "float32" else wp.vec3d
        np_dtype = np.float32 if dtype == "float32" else np.float64
        np.random.seed(42)

        system["velocities"] = wp.array(
            np.random.randn(20, 3).astype(np_dtype) * 2.0,  # High velocities
            dtype=vec_dtype,
            device=device,
        )

        compute_cell_volume(system["cells"], system["volumes"], device=device)
        initial_volume = system["volumes"].numpy()[0]

        # Run many steps
        for _ in range(100):
            run_npt_step(
                system["positions"],
                system["velocities"],
                system["forces"],
                system["masses"],
                system["cells"],
                system["cell_velocities"],
                system["virial_tensors"],
                system["eta"],
                system["eta_dot"],
                system["thermostat_masses"],
                system["cell_masses"],
                system["target_temperature"],
                system["target_pressure"],
                system["num_atoms_per_system"],
                system["chain_length"],
                dt=system["dt"],
                pressure_tensors=system["pressure_tensors"],
                volumes=system["volumes"],
                kinetic_energy=system["kinetic_energy"],
                cells_inv=system["cells_inv"],
                kinetic_tensors=system["kinetic_tensors"],
                num_atoms_per_system=system["num_atoms_per_system"],
                device=device,
            )

        wp.synchronize_device(device)

        compute_cell_volume(system["cells"], system["volumes"], device=device)
        final_volume = system["volumes"].numpy()[0]

        # Volume should change (expand due to high kinetic pressure)
        # We don't check direction because it depends on the balance of forces
        assert initial_volume != final_volume

    def test_thermostat_modifies_kinetic_energy(self, dtype, device):
        """Test that NPT thermostat affects kinetic energy."""
        system = setup_npt_system(20, dtype, device)

        from nvalchemiops.dynamics.utils.thermostat_utils import compute_kinetic_energy

        scalar_dtype = wp.float32 if dtype == "float32" else wp.float64
        ke_tmp = wp.zeros(1, dtype=scalar_dtype, device=device)
        compute_kinetic_energy(
            system["velocities"],
            system["masses"],
            ke_tmp,
            device=device,
        )
        initial_ke = ke_tmp.numpy().sum()

        for _ in range(50):
            run_npt_step(
                system["positions"],
                system["velocities"],
                system["forces"],
                system["masses"],
                system["cells"],
                system["cell_velocities"],
                system["virial_tensors"],
                system["eta"],
                system["eta_dot"],
                system["thermostat_masses"],
                system["cell_masses"],
                system["target_temperature"],
                system["target_pressure"],
                system["num_atoms_per_system"],
                system["chain_length"],
                dt=system["dt"],
                pressure_tensors=system["pressure_tensors"],
                volumes=system["volumes"],
                kinetic_energy=system["kinetic_energy"],
                cells_inv=system["cells_inv"],
                kinetic_tensors=system["kinetic_tensors"],
                num_atoms_per_system=system["num_atoms_per_system"],
                device=device,
            )

        wp.synchronize_device(device)

        ke_tmp.zero_()
        compute_kinetic_energy(
            system["velocities"],
            system["masses"],
            ke_tmp,
            device=device,
        )
        final_ke = ke_tmp.numpy().sum()

        # Thermostat should change KE towards target
        assert initial_ke != final_ke


# ==============================================================================
# 8. Physics Tests - NPH
# ==============================================================================


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
class TestNPHPhysics:
    """Physics tests for NPH integrator."""

    def test_nph_cell_responds_to_pressure(self, dtype, device):
        """Test that NPH cell volume changes under pressure imbalance."""
        system = setup_nph_system(20, dtype, device)

        # Set high velocities to create high internal pressure
        vec_dtype = wp.vec3f if dtype == "float32" else wp.vec3d
        np_dtype = np.float32 if dtype == "float32" else np.float64
        np.random.seed(42)

        system["velocities"] = wp.array(
            np.random.randn(20, 3).astype(np_dtype) * 2.0,
            dtype=vec_dtype,
            device=device,
        )

        compute_cell_volume(system["cells"], system["volumes"], device=device)
        initial_volume = system["volumes"].numpy()[0]

        for _ in range(100):
            run_nph_step(
                system["positions"],
                system["velocities"],
                system["forces"],
                system["masses"],
                system["cells"],
                system["cell_velocities"],
                system["virial_tensors"],
                system["cell_masses"],
                system["target_pressure"],
                system["num_atoms_per_system"],
                dt=system["dt"],
                pressure_tensors=system["pressure_tensors"],
                volumes=system["volumes"],
                kinetic_energy=system["kinetic_energy"],
                cells_inv=system["cells_inv"],
                kinetic_tensors=system["kinetic_tensors"],
                num_atoms_per_system=system["num_atoms_per_system"],
                device=device,
            )

        wp.synchronize_device(device)

        compute_cell_volume(system["cells"], system["volumes"], device=device)
        final_volume = system["volumes"].numpy()[0]

        # Volume should change
        assert initial_volume != final_volume

    def test_nph_no_thermostat_coupling(self, dtype, device):
        """Test that NPH differs from NPT due to lack of thermostat.

        NPT has thermostat coupling in velocity update: drag = (coupling * eps_dot + eta_dot_1) * v
        NPH has no thermostat coupling: drag = coupling * eps_dot * v

        We verify this by checking that the eta_dot (thermostat velocity) changes in NPT
        but NPH has no thermostat state at all.
        """
        # Set up NPT system with non-equilibrium initial conditions
        npt_system = setup_npt_system(10, dtype, device, seed=123)

        # Record initial thermostat state
        initial_eta_dot = npt_system["eta_dot"].numpy().copy()

        # Run NPT - the thermostat chain should evolve
        for _ in range(100):
            run_npt_step(
                npt_system["positions"],
                npt_system["velocities"],
                npt_system["forces"],
                npt_system["masses"],
                npt_system["cells"],
                npt_system["cell_velocities"],
                npt_system["virial_tensors"],
                npt_system["eta"],
                npt_system["eta_dot"],
                npt_system["thermostat_masses"],
                npt_system["cell_masses"],
                npt_system["target_temperature"],
                npt_system["target_pressure"],
                npt_system["num_atoms_per_system"],
                npt_system["chain_length"],
                dt=npt_system["dt"],
                pressure_tensors=npt_system["pressure_tensors"],
                volumes=npt_system["volumes"],
                kinetic_energy=npt_system["kinetic_energy"],
                cells_inv=npt_system["cells_inv"],
                kinetic_tensors=npt_system["kinetic_tensors"],
                num_atoms_per_system=npt_system["num_atoms_per_system"],
                device=device,
            )

        wp.synchronize_device(device)

        # NPT thermostat state should have changed
        final_eta_dot = npt_system["eta_dot"].numpy()

        # The thermostat velocities should have evolved from their initial values
        # This confirms the thermostat is active in NPT
        assert not np.allclose(initial_eta_dot, final_eta_dot, atol=1e-10), (
            "NPT thermostat chain should evolve (eta_dot should change)"
        )

        # NPH doesn't have thermostat state - that's the key difference
        # We verify NPH runs without thermostat by checking it doesn't require eta/eta_dot
        nph_system = setup_nph_system(10, dtype, device, seed=123)

        for _ in range(100):
            run_nph_step(
                nph_system["positions"],
                nph_system["velocities"],
                nph_system["forces"],
                nph_system["masses"],
                nph_system["cells"],
                nph_system["cell_velocities"],
                nph_system["virial_tensors"],
                nph_system["cell_masses"],
                nph_system["target_pressure"],
                nph_system["num_atoms_per_system"],
                dt=nph_system["dt"],
                pressure_tensors=nph_system["pressure_tensors"],
                volumes=nph_system["volumes"],
                kinetic_energy=nph_system["kinetic_energy"],
                cells_inv=nph_system["cells_inv"],
                kinetic_tensors=nph_system["kinetic_tensors"],
                num_atoms_per_system=nph_system["num_atoms_per_system"],
                device=device,
            )

        wp.synchronize_device(device)

        # NPH should complete without any thermostat - this is the fundamental difference
        # The test passing means NPH doesn't need/use thermostat state


# ==============================================================================
# 9. Batched Tests
# ==============================================================================


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
class TestBatchedIntegration:
    """Tests for batched integration."""

    def test_batched_pressure_tensor(self, dtype, device):
        """Test batched pressure tensor computation."""
        vec_dtype = wp.vec3f if dtype == "float32" else wp.vec3d
        mat_dtype = wp.mat33f if dtype == "float32" else wp.mat33d
        scalar_dtype = wp.float32 if dtype == "float32" else wp.float64
        tensor_dtype = vec9f if dtype == "float32" else vec9d
        np_dtype = np.float32 if dtype == "float32" else np.float64

        num_systems = 2
        atoms_per_system = 10
        num_atoms = num_systems * atoms_per_system

        np.random.seed(42)

        batch_idx_np = np.repeat(np.arange(num_systems), atoms_per_system).astype(
            np.int32
        )
        batch_idx = wp.array(batch_idx_np, dtype=wp.int32, device=device)

        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=vec_dtype,
            device=device,
        )
        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype), dtype=scalar_dtype, device=device
        )

        # Create cells
        cell1_np = np.diag([10.0, 10.0, 10.0]).astype(np_dtype)
        cell2_np = np.diag([12.0, 12.0, 12.0]).astype(np_dtype)

        cell1_mat = mat_dtype(
            cell1_np[0, 0],
            cell1_np[0, 1],
            cell1_np[0, 2],
            cell1_np[1, 0],
            cell1_np[1, 1],
            cell1_np[1, 2],
            cell1_np[2, 0],
            cell1_np[2, 1],
            cell1_np[2, 2],
        )
        cell2_mat = mat_dtype(
            cell2_np[0, 0],
            cell2_np[0, 1],
            cell2_np[0, 2],
            cell2_np[1, 0],
            cell2_np[1, 1],
            cell2_np[1, 2],
            cell2_np[2, 0],
            cell2_np[2, 1],
            cell2_np[2, 2],
        )
        cells = wp.array([cell1_mat, cell2_mat], dtype=mat_dtype, device=device)

        virial_tensors = wp.zeros(num_systems, dtype=tensor_dtype, device=device)

        kinetic_tensors = wp.zeros((num_systems, 9), dtype=scalar_dtype, device=device)
        pressure_tensors = wp.empty(num_systems, dtype=tensor_dtype, device=device)
        volumes = wp.empty(num_systems, dtype=scalar_dtype, device=device)
        compute_cell_volume(cells, volumes, device=device)

        pressure_tensors = compute_pressure_tensor(
            velocities,
            masses,
            virial_tensors,
            cells,
            kinetic_tensors,
            pressure_tensors,
            volumes,
            batch_idx=batch_idx,
            device=device,
        )
        wp.synchronize_device(device)

        assert pressure_tensors.shape[0] == num_systems

    def test_batched_nph_velocity_update(self, dtype, device):
        """Test batched NPH velocity update."""
        vec_dtype = wp.vec3f if dtype == "float32" else wp.vec3d
        mat_dtype = wp.mat33f if dtype == "float32" else wp.mat33d
        scalar_dtype = wp.float32 if dtype == "float32" else wp.float64
        np_dtype = np.float32 if dtype == "float32" else np.float64

        num_systems = 2
        atoms_per_system = 10
        num_atoms = num_systems * atoms_per_system

        np.random.seed(42)

        batch_idx_np = np.repeat(np.arange(num_systems), atoms_per_system).astype(
            np.int32
        )
        batch_idx = wp.array(batch_idx_np, dtype=wp.int32, device=device)
        num_atoms_per_system = wp.array(
            [atoms_per_system, atoms_per_system], dtype=wp.int32, device=device
        )

        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype) * 0.3,
            dtype=vec_dtype,
            device=device,
        )
        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype), dtype=scalar_dtype, device=device
        )
        forces = wp.zeros(num_atoms, dtype=vec_dtype, device=device)

        cell_np = np.diag([10.0, 10.0, 10.0]).astype(np_dtype)
        h_dot_np = np.zeros((3, 3), dtype=np_dtype)

        cell_mat = mat_dtype(
            cell_np[0, 0],
            cell_np[0, 1],
            cell_np[0, 2],
            cell_np[1, 0],
            cell_np[1, 1],
            cell_np[1, 2],
            cell_np[2, 0],
            cell_np[2, 1],
            cell_np[2, 2],
        )
        h_dot_mat = mat_dtype(
            h_dot_np[0, 0],
            h_dot_np[0, 1],
            h_dot_np[0, 2],
            h_dot_np[1, 0],
            h_dot_np[1, 1],
            h_dot_np[1, 2],
            h_dot_np[2, 0],
            h_dot_np[2, 1],
            h_dot_np[2, 2],
        )
        cells = wp.array([cell_mat, cell_mat], dtype=mat_dtype, device=device)
        cell_velocities = wp.array(
            [h_dot_mat, h_dot_mat], dtype=mat_dtype, device=device
        )

        volumes = wp.empty(num_systems, dtype=scalar_dtype, device=device)
        compute_cell_volume(cells, volumes, device=device)

        cells_inv = wp.empty(num_systems, dtype=mat_dtype, device=device)
        compute_cell_inverse(cells, cells_inv, device=device)

        dt = wp.array([0.001, 0.001], dtype=scalar_dtype, device=device)
        nph_velocity_half_step(
            velocities,
            masses,
            forces,
            cell_velocities,
            volumes,
            num_atoms_per_system,
            dt=dt,
            batch_idx=batch_idx,
            num_atoms_per_system=num_atoms_per_system,
            cells_inv=cells_inv,
            device=device,
        )
        wp.synchronize_device(device)

        assert velocities.shape[0] == num_atoms


# ==============================================================================
# Anisotropic Pressure Control Tests
# ==============================================================================


def setup_aniso_system(num_atoms, dtype, device, seed=42):
    """Set up a system for anisotropic pressure tests."""
    np.random.seed(seed)

    np_dtype = np.float32 if dtype == "float32" else np.float64
    vec_dtype = wp.vec3f if dtype == "float32" else wp.vec3d
    scalar_dtype = wp.float32 if dtype == "float32" else wp.float64
    mat_dtype = wp.mat33f if dtype == "float32" else wp.mat33d
    vec9_dtype = vec9f if dtype == "float32" else vec9d
    vec3_dtype = vec3f if dtype == "float32" else vec3d

    # Positions and velocities
    positions = wp.array(
        np.random.uniform(0, 10, (num_atoms, 3)).astype(np_dtype),
        dtype=vec_dtype,
        device=device,
    )
    velocities = wp.array(
        np.random.randn(num_atoms, 3).astype(np_dtype) * 0.5,
        dtype=vec_dtype,
        device=device,
    )
    masses = wp.array(
        np.ones(num_atoms, dtype=np_dtype), dtype=scalar_dtype, device=device
    )
    forces = wp.zeros(num_atoms, dtype=vec_dtype, device=device)

    # Cell (orthorhombic for anisotropic tests)
    cell_np = np.diag([10.0, 12.0, 8.0]).astype(np_dtype)  # Different box lengths
    cell_mat = mat_dtype(
        cell_np[0, 0],
        cell_np[0, 1],
        cell_np[0, 2],
        cell_np[1, 0],
        cell_np[1, 1],
        cell_np[1, 2],
        cell_np[2, 0],
        cell_np[2, 1],
        cell_np[2, 2],
    )
    cells = wp.array([cell_mat], dtype=mat_dtype, device=device)

    # Cell velocity
    h_dot_np = np.zeros((3, 3), dtype=np_dtype)
    h_dot_mat = mat_dtype(
        h_dot_np[0, 0],
        h_dot_np[0, 1],
        h_dot_np[0, 2],
        h_dot_np[1, 0],
        h_dot_np[1, 1],
        h_dot_np[1, 2],
        h_dot_np[2, 0],
        h_dot_np[2, 1],
        h_dot_np[2, 2],
    )
    cell_velocities = wp.array([h_dot_mat], dtype=mat_dtype, device=device)

    # Volume
    volumes = wp.empty(1, dtype=scalar_dtype, device=device)
    compute_cell_volume(cells, volumes, device=device)

    # Cell inverse
    cells_inv = wp.empty(1, dtype=mat_dtype, device=device)
    compute_cell_inverse(cells, cells_inv, device=device)

    # Thermostat state
    chain_length = 3
    eta = wp.zeros((1, chain_length), dtype=scalar_dtype, device=device)
    eta_dot = wp.zeros((1, chain_length), dtype=scalar_dtype, device=device)
    thermostat_masses = wp.array(
        np.ones((1, chain_length), dtype=np_dtype), dtype=scalar_dtype, device=device
    )

    # Cell mass
    cell_masses = wp.array(
        np.array([100.0], dtype=np_dtype), dtype=scalar_dtype, device=device
    )

    # Virial tensor (isotropic pressure)
    virial_np = np.array(
        [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]], dtype=np_dtype
    )
    virial_tensors = wp.array(virial_np, dtype=vec9_dtype, device=device)

    # Anisotropic target pressure (vec3)
    target_pressure_aniso = wp.array(
        np.array([[1.0, 2.0, 0.5]], dtype=np_dtype),  # Different pressures for x, y, z
        dtype=vec3_dtype,
        device=device,
    )

    # Triclinic target pressure (vec9)
    target_pressure_tri = wp.array(
        np.array([[1.0, 0.1, 0.2, 0.1, 2.0, 0.15, 0.2, 0.15, 0.5]], dtype=np_dtype),
        dtype=vec9_dtype,
        device=device,
    )

    # Scalar target pressure
    target_pressure_scalar = wp.array(
        np.array([1.0], dtype=np_dtype), dtype=scalar_dtype, device=device
    )

    # Kinetic energy
    kinetic_energy = wp.array(
        np.array([5.0], dtype=np_dtype), dtype=scalar_dtype, device=device
    )

    # Number of atoms
    num_atoms_per_system = wp.array([num_atoms], dtype=wp.int32, device=device)

    dt = wp.array([0.001], dtype=scalar_dtype, device=device)

    return {
        "positions": positions,
        "velocities": velocities,
        "masses": masses,
        "forces": forces,
        "cells": cells,
        "cell_velocities": cell_velocities,
        "volumes": volumes,
        "cells_inv": cells_inv,
        "eta": eta,
        "eta_dot": eta_dot,
        "thermostat_masses": thermostat_masses,
        "cell_masses": cell_masses,
        "virial_tensors": virial_tensors,
        "target_pressure_aniso": target_pressure_aniso,
        "target_pressure_tri": target_pressure_tri,
        "target_pressure_scalar": target_pressure_scalar,
        "kinetic_energy": kinetic_energy,
        "num_atoms_per_system": num_atoms_per_system,
        "num_atoms": num_atoms,
        "chain_length": chain_length,
        "dt": dt,
    }


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
class TestAnisotropicBarostat:
    """Tests for anisotropic barostat functionality.

    These tests verify the unified API dispatches correctly based on
    target_pressures dtype:
    - scalar (float32/float64) -> isotropic
    - vec3 -> anisotropic (orthorhombic)
    - vec9 -> triclinic (full tensor)
    """

    def test_npt_barostat_aniso_runs(self, dtype, device):
        """Test that anisotropic NPT barostat runs via unified API (vec3 target)."""
        system = setup_aniso_system(10, dtype, device)

        # Unified API auto-dispatches to aniso kernel based on vec3 dtype
        npt_barostat_half_step(
            system["cell_velocities"],
            system["virial_tensors"],
            system["target_pressure_aniso"],  # vec3 -> anisotropic dispatch
            system["volumes"],
            system["cell_masses"],
            system["kinetic_energy"],
            system["num_atoms_per_system"],
            system["eta_dot"],
            dt=system["dt"],
            device=device,
        )
        wp.synchronize_device(device)

        # Cell velocities should have been updated
        h_dot = system["cell_velocities"].numpy()[0]
        assert h_dot.shape == (3, 3)

    def test_nph_barostat_aniso_runs(self, dtype, device):
        """Test that anisotropic NPH barostat runs via unified API (vec3 target)."""
        system = setup_aniso_system(10, dtype, device)

        # Unified API auto-dispatches to aniso kernel based on vec3 dtype
        nph_barostat_half_step(
            system["cell_velocities"],
            system["virial_tensors"],
            system["target_pressure_aniso"],  # vec3 -> anisotropic dispatch
            system["volumes"],
            system["cell_masses"],
            system["kinetic_energy"],
            system["num_atoms_per_system"],
            dt=system["dt"],
            device=device,
        )
        wp.synchronize_device(device)

        h_dot = system["cell_velocities"].numpy()[0]
        assert h_dot.shape == (3, 3)

    def test_npt_barostat_triclinic_runs(self, dtype, device):
        """Test that triclinic NPT barostat runs via unified API (vec9 target)."""
        system = setup_aniso_system(10, dtype, device)

        # Unified API auto-dispatches to triclinic kernel based on vec9 dtype
        npt_barostat_half_step(
            system["cell_velocities"],
            system["virial_tensors"],
            system["target_pressure_tri"],  # vec9 -> triclinic dispatch
            system["volumes"],
            system["cell_masses"],
            system["kinetic_energy"],
            system["num_atoms_per_system"],
            system["eta_dot"],
            dt=system["dt"],
            device=device,
        )
        wp.synchronize_device(device)

        h_dot = system["cell_velocities"].numpy()[0]
        assert h_dot.shape == (3, 3)

    def test_nph_barostat_triclinic_runs(self, dtype, device):
        """Test that triclinic NPH barostat runs via unified API (vec9 target)."""
        system = setup_aniso_system(10, dtype, device)

        # Unified API auto-dispatches to triclinic kernel based on vec9 dtype
        nph_barostat_half_step(
            system["cell_velocities"],
            system["virial_tensors"],
            system["target_pressure_tri"],  # vec9 -> triclinic dispatch
            system["volumes"],
            system["cell_masses"],
            system["kinetic_energy"],
            system["num_atoms_per_system"],
            dt=system["dt"],
            device=device,
        )
        wp.synchronize_device(device)

        h_dot = system["cell_velocities"].numpy()[0]
        assert h_dot.shape == (3, 3)

    def test_aniso_updates_diagonal_independently(self, dtype, device):
        """Test that anisotropic mode updates diagonal cell velocities independently."""
        system = setup_aniso_system(10, dtype, device)

        # Different target pressures for each axis (vec3 -> anisotropic)
        scalar_dtype = wp.float32 if dtype == "float32" else wp.float64
        dt_large = wp.array([0.01], dtype=scalar_dtype, device=device)
        nph_barostat_half_step(
            system["cell_velocities"],
            system["virial_tensors"],
            system["target_pressure_aniso"],
            system["volumes"],
            system["cell_masses"],
            system["kinetic_energy"],
            system["num_atoms_per_system"],
            dt=dt_large,
            device=device,
        )
        wp.synchronize_device(device)

        h_dot = system["cell_velocities"].numpy()[0]

        # Diagonal elements should be updated (potentially different values)
        # Off-diagonal should remain zero for orthorhombic
        assert abs(h_dot[0, 1]) < 1e-10
        assert abs(h_dot[0, 2]) < 1e-10
        assert abs(h_dot[1, 0]) < 1e-10
        assert abs(h_dot[1, 2]) < 1e-10
        assert abs(h_dot[2, 0]) < 1e-10
        assert abs(h_dot[2, 1]) < 1e-10

    def test_triclinic_updates_all_components(self, dtype, device):
        """Test that triclinic mode can update all cell velocity components."""
        system = setup_aniso_system(10, dtype, device)

        # Full stress tensor (vec9 -> triclinic)
        scalar_dtype = wp.float32 if dtype == "float32" else wp.float64
        dt_large = wp.array([0.01], dtype=scalar_dtype, device=device)
        nph_barostat_half_step(
            system["cell_velocities"],
            system["virial_tensors"],
            system["target_pressure_tri"],
            system["volumes"],
            system["cell_masses"],
            system["kinetic_energy"],
            system["num_atoms_per_system"],
            dt=dt_large,
            device=device,
        )
        wp.synchronize_device(device)

        h_dot = system["cell_velocities"].numpy()[0]

        # All components should be updated (not necessarily non-zero, but should have run)
        assert h_dot.shape == (3, 3)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
class TestAnisotropicVelocityUpdate:
    """Tests for anisotropic velocity update functionality.

    These tests use the unified API with mode="anisotropic" parameter.
    """

    def test_npt_velocity_aniso_runs(self, dtype, device):
        """Test that anisotropic NPT velocity update runs via mode parameter."""
        system = setup_aniso_system(10, dtype, device)

        npt_velocity_half_step(
            system["velocities"],
            system["masses"],
            system["forces"],
            system["cell_velocities"],
            system["volumes"],
            system["eta_dot"],
            system["num_atoms_per_system"],
            dt=system["dt"],
            cells_inv=system["cells_inv"],
            mode="anisotropic",  # Use unified API with mode
            device=device,
        )
        wp.synchronize_device(device)

        assert system["velocities"].shape[0] == 10

    def test_nph_velocity_aniso_runs(self, dtype, device):
        """Test that anisotropic NPH velocity update runs via mode parameter."""
        system = setup_aniso_system(10, dtype, device)

        nph_velocity_half_step(
            system["velocities"],
            system["masses"],
            system["forces"],
            system["cell_velocities"],
            system["volumes"],
            system["num_atoms_per_system"],
            dt=system["dt"],
            cells_inv=system["cells_inv"],
            mode="anisotropic",  # Use unified API with mode
            device=device,
        )
        wp.synchronize_device(device)

        assert system["velocities"].shape[0] == 10

    def test_npt_velocity_aniso_out_runs(self, dtype, device):
        """Test that non-mutating anisotropic NPT velocity update runs."""
        system = setup_aniso_system(10, dtype, device)

        vel_out = wp.empty_like(system["velocities"])
        vel_out = npt_velocity_half_step_out(
            system["velocities"],
            system["masses"],
            system["forces"],
            system["cell_velocities"],
            system["volumes"],
            system["eta_dot"],
            system["num_atoms_per_system"],
            system["dt"],
            vel_out,
            cells_inv=system["cells_inv"],
            mode="anisotropic",  # Use mode parameter
            device=device,
        )
        wp.synchronize_device(device)

        assert vel_out.shape[0] == 10
        assert vel_out.dtype == system["velocities"].dtype

    def test_nph_velocity_aniso_out_runs(self, dtype, device):
        """Test that non-mutating anisotropic NPH velocity update runs."""
        system = setup_aniso_system(10, dtype, device)

        vel_out = wp.empty_like(system["velocities"])
        vel_out = nph_velocity_half_step_out(
            system["velocities"],
            system["masses"],
            system["forces"],
            system["cell_velocities"],
            system["volumes"],
            system["num_atoms_per_system"],
            system["dt"],
            vel_out,
            cells_inv=system["cells_inv"],
            mode="anisotropic",  # Use mode parameter
            device=device,
        )
        wp.synchronize_device(device)

        assert vel_out.shape[0] == 10
        assert vel_out.dtype == system["velocities"].dtype


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
class TestAnisotropicPhysics:
    """Physics tests for anisotropic pressure control using unified API."""

    def test_aniso_different_pressures_cause_different_cell_evolution(
        self, dtype, device
    ):
        """Test that different target pressures cause different cell evolutions."""
        np_dtype = np.float32 if dtype == "float32" else np.float64
        vec3_dtype = vec3f if dtype == "float32" else vec3d

        # Create two systems with same initial conditions
        system1 = setup_aniso_system(10, dtype, device, seed=123)
        system2 = setup_aniso_system(10, dtype, device, seed=123)

        # Different target pressures (vec3 -> anisotropic dispatch)
        # System 1: High x pressure, low y, z
        target1 = wp.array(
            np.array([[5.0, 0.1, 0.1]], dtype=np_dtype), dtype=vec3_dtype, device=device
        )
        # System 2: Low x pressure, high y, z
        target2 = wp.array(
            np.array([[0.1, 5.0, 5.0]], dtype=np_dtype), dtype=vec3_dtype, device=device
        )

        scalar_dtype = wp.float32 if dtype == "float32" else wp.float64
        dt_large = wp.array([0.01], dtype=scalar_dtype, device=device)
        for _ in range(10):
            # Unified API dispatches to aniso kernel based on vec3 dtype
            nph_barostat_half_step(
                system1["cell_velocities"],
                system1["virial_tensors"],
                target1,
                system1["volumes"],
                system1["cell_masses"],
                system1["kinetic_energy"],
                system1["num_atoms_per_system"],
                dt=dt_large,
                device=device,
            )
            nph_barostat_half_step(
                system2["cell_velocities"],
                system2["virial_tensors"],
                target2,
                system2["volumes"],
                system2["cell_masses"],
                system2["kinetic_energy"],
                system2["num_atoms_per_system"],
                dt=dt_large,
                device=device,
            )

        wp.synchronize_device(device)

        h_dot1 = system1["cell_velocities"].numpy()[0]
        h_dot2 = system2["cell_velocities"].numpy()[0]

        # Cell velocities should evolve differently based on target pressures
        assert not np.allclose(h_dot1, h_dot2, atol=1e-10), (
            "Different target pressures should cause different cell velocity evolution"
        )

    def test_isotropic_vs_anisotropic_with_uniform_pressure(self, dtype, device):
        """Test that isotropic and anisotropic give similar results with uniform pressure."""
        np_dtype = np.float32 if dtype == "float32" else np.float64
        scalar_dtype = wp.float32 if dtype == "float32" else wp.float64
        vec3_dtype = vec3f if dtype == "float32" else vec3d

        # Create two identical systems
        system_iso = setup_aniso_system(10, dtype, device, seed=456)
        system_aniso = setup_aniso_system(10, dtype, device, seed=456)

        # Same pressure for all axes (uniform)
        P = 1.0
        # scalar dtype -> isotropic dispatch
        target_iso = wp.array(
            np.array([P], dtype=np_dtype), dtype=scalar_dtype, device=device
        )
        # vec3 dtype -> anisotropic dispatch
        target_aniso = wp.array(
            np.array([[P, P, P]], dtype=np_dtype), dtype=vec3_dtype, device=device
        )

        scalar_dtype = wp.float32 if dtype == "float32" else wp.float64
        dt_arr = wp.array([0.001], dtype=scalar_dtype, device=device)
        # Run isotropic (auto-dispatched based on scalar dtype)
        nph_barostat_half_step(
            system_iso["cell_velocities"],
            system_iso["virial_tensors"],
            target_iso,
            system_iso["volumes"],
            system_iso["cell_masses"],
            system_iso["kinetic_energy"],
            system_iso["num_atoms_per_system"],
            dt=dt_arr,
            device=device,
        )

        # Run anisotropic (auto-dispatched based on vec3 dtype)
        nph_barostat_half_step(
            system_aniso["cell_velocities"],
            system_aniso["virial_tensors"],
            target_aniso,
            system_aniso["volumes"],
            system_aniso["cell_masses"],
            system_aniso["kinetic_energy"],
            system_aniso["num_atoms_per_system"],
            dt=dt_arr,
            device=device,
        )

        wp.synchronize_device(device)

        h_dot_iso = system_iso["cell_velocities"].numpy()[0]
        h_dot_aniso = system_aniso["cell_velocities"].numpy()[0]

        # With uniform target pressure, the diagonal elements should be the same
        atol = 1e-5 if dtype == "float32" else 1e-10
        assert np.allclose(h_dot_iso[0, 0], h_dot_aniso[0, 0], atol=atol)
        assert np.allclose(h_dot_iso[1, 1], h_dot_aniso[1, 1], atol=atol)
        assert np.allclose(h_dot_iso[2, 2], h_dot_aniso[2, 2], atol=atol)


# ==============================================================================
# Coverage Tests - Triclinic Velocity Coupling
# ==============================================================================


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
class TestTriclinicVelocityCoupling:
    """Tests for triclinic velocity coupling mode."""

    def setup_triclinic_system(self, num_atoms, dtype, device):
        """Helper to set up a triclinic system."""
        vec_dtype = wp.vec3f if dtype == "float32" else wp.vec3d
        mat_dtype = wp.mat33f if dtype == "float32" else wp.mat33d
        scalar_dtype = wp.float32 if dtype == "float32" else wp.float64
        np_dtype = np.float32 if dtype == "float32" else np.float64

        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype) * 0.1,
            dtype=vec_dtype,
            device=device,
        )
        forces = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=vec_dtype,
            device=device,
        )
        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype),
            dtype=scalar_dtype,
            device=device,
        )

        # Triclinic cell
        cell_np = np.array(
            [[10.0, 0.5, 0.2], [0.0, 10.0, 0.3], [0.0, 0.0, 10.0]], dtype=np_dtype
        )

        cell_inv_np = np.linalg.inv(cell_np).astype(np_dtype)
        cell_inv_mat = mat_dtype(
            cell_inv_np[0, 0],
            cell_inv_np[0, 1],
            cell_inv_np[0, 2],
            cell_inv_np[1, 0],
            cell_inv_np[1, 1],
            cell_inv_np[1, 2],
            cell_inv_np[2, 0],
            cell_inv_np[2, 1],
            cell_inv_np[2, 2],
        )
        cells_inv = wp.array([cell_inv_mat], dtype=mat_dtype, device=device)

        h_dot_np = np.array(
            [[0.01, 0.001, 0.0], [0.0, 0.01, 0.001], [0.0, 0.0, 0.01]], dtype=np_dtype
        )
        h_dot_mat = mat_dtype(
            h_dot_np[0, 0],
            h_dot_np[0, 1],
            h_dot_np[0, 2],
            h_dot_np[1, 0],
            h_dot_np[1, 1],
            h_dot_np[1, 2],
            h_dot_np[2, 0],
            h_dot_np[2, 1],
            h_dot_np[2, 2],
        )
        cell_velocities = wp.array([h_dot_mat], dtype=mat_dtype, device=device)

        volumes = wp.array([np.linalg.det(cell_np)], dtype=scalar_dtype, device=device)
        eta_dots = wp.zeros((1, 3), dtype=scalar_dtype, device=device)
        num_atoms_per_system = wp.array([num_atoms], dtype=wp.int32, device=device)
        dt = wp.array([0.001], dtype=scalar_dtype, device=device)

        return {
            "velocities": velocities,
            "forces": forces,
            "masses": masses,
            "cells_inv": cells_inv,
            "cell_velocities": cell_velocities,
            "volumes": volumes,
            "eta_dots": eta_dots,
            "num_atoms": num_atoms,
            "num_atoms_per_system": num_atoms_per_system,
            "dt": dt,
        }

    def test_npt_velocity_half_step_triclinic_single(self, dtype, device):
        """Test NPT triclinic velocity half-step for single system."""
        system = self.setup_triclinic_system(20, dtype, device)

        npt_velocity_half_step(
            system["velocities"],
            system["masses"],
            system["forces"],
            system["cell_velocities"],
            system["volumes"],
            system["eta_dots"],
            num_atoms=system["num_atoms_per_system"],
            dt=system["dt"],
            cells_inv=system["cells_inv"],
            mode="triclinic",
            device=device,
        )

        assert system["velocities"].shape[0] == system["num_atoms"]

    def test_npt_velocity_half_step_out_triclinic_single(self, dtype, device):
        """Test NPT triclinic velocity half-step (non-mutating) for single system."""
        system = self.setup_triclinic_system(20, dtype, device)
        vel_orig = system["velocities"].numpy().copy()

        vel_out = wp.empty_like(system["velocities"])
        vel_out = npt_velocity_half_step_out(
            system["velocities"],
            system["masses"],
            system["forces"],
            system["cell_velocities"],
            system["volumes"],
            system["eta_dots"],
            system["num_atoms_per_system"],
            system["dt"],
            vel_out,
            cells_inv=system["cells_inv"],
            mode="triclinic",
            device=device,
        )

        np.testing.assert_array_equal(system["velocities"].numpy(), vel_orig)
        assert vel_out.shape[0] == system["num_atoms"]

    def test_nph_velocity_half_step_triclinic_single(self, dtype, device):
        """Test NPH triclinic velocity half-step for single system."""
        system = self.setup_triclinic_system(20, dtype, device)

        nph_velocity_half_step(
            system["velocities"],
            system["masses"],
            system["forces"],
            system["cell_velocities"],
            system["volumes"],
            num_atoms=system["num_atoms_per_system"],
            dt=system["dt"],
            cells_inv=system["cells_inv"],
            mode="triclinic",
            device=device,
        )

        assert system["velocities"].shape[0] == system["num_atoms"]

    def test_nph_velocity_half_step_out_triclinic_single(self, dtype, device):
        """Test NPH triclinic velocity half-step (non-mutating) for single system."""
        system = self.setup_triclinic_system(20, dtype, device)
        vel_orig = system["velocities"].numpy().copy()

        vel_out = wp.empty_like(system["velocities"])
        vel_out = nph_velocity_half_step_out(
            system["velocities"],
            system["masses"],
            system["forces"],
            system["cell_velocities"],
            system["volumes"],
            system["num_atoms_per_system"],
            system["dt"],
            vel_out,
            cells_inv=system["cells_inv"],
            mode="triclinic",
            device=device,
        )

        np.testing.assert_array_equal(system["velocities"].numpy(), vel_orig)
        assert vel_out.shape[0] == system["num_atoms"]

    def test_npt_velocity_triclinic_batched(self, dtype, device):
        """Test NPT triclinic velocity half-step for batched systems."""
        num_atoms = 40
        num_systems = 2
        vec_dtype = wp.vec3f if dtype == "float32" else wp.vec3d
        mat_dtype = wp.mat33f if dtype == "float32" else wp.mat33d
        scalar_dtype = wp.float32 if dtype == "float32" else wp.float64
        np_dtype = np.float32 if dtype == "float32" else np.float64

        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype) * 0.1,
            dtype=vec_dtype,
            device=device,
        )
        forces = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=vec_dtype,
            device=device,
        )
        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype),
            dtype=scalar_dtype,
            device=device,
        )
        batch_idx = wp.array(
            np.array([0] * 20 + [1] * 20, dtype=np.int32),
            dtype=wp.int32,
            device=device,
        )
        num_atoms_per_system = wp.array([20, 20], dtype=wp.int32, device=device)

        # Create cells for batch
        cell_np = np.array(
            [[10.0, 0.5, 0.2], [0.0, 10.0, 0.3], [0.0, 0.0, 10.0]], dtype=np_dtype
        )
        cell_inv_np = np.linalg.inv(cell_np).astype(np_dtype)
        h_dot_np = np.array(
            [[0.01, 0.001, 0.0], [0.0, 0.01, 0.001], [0.0, 0.0, 0.01]], dtype=np_dtype
        )

        cell_inv_mat = mat_dtype(
            cell_inv_np[0, 0],
            cell_inv_np[0, 1],
            cell_inv_np[0, 2],
            cell_inv_np[1, 0],
            cell_inv_np[1, 1],
            cell_inv_np[1, 2],
            cell_inv_np[2, 0],
            cell_inv_np[2, 1],
            cell_inv_np[2, 2],
        )
        h_dot_mat = mat_dtype(
            h_dot_np[0, 0],
            h_dot_np[0, 1],
            h_dot_np[0, 2],
            h_dot_np[1, 0],
            h_dot_np[1, 1],
            h_dot_np[1, 2],
            h_dot_np[2, 0],
            h_dot_np[2, 1],
            h_dot_np[2, 2],
        )

        cell_velocities = wp.array(
            [h_dot_mat, h_dot_mat], dtype=mat_dtype, device=device
        )
        cells_inv = wp.array(
            [cell_inv_mat, cell_inv_mat], dtype=mat_dtype, device=device
        )
        volumes = wp.array(
            [np.linalg.det(cell_np), np.linalg.det(cell_np)],
            dtype=scalar_dtype,
            device=device,
        )
        eta_dots = wp.zeros((num_systems, 3), dtype=scalar_dtype, device=device)

        dt = wp.array([0.001, 0.001], dtype=scalar_dtype, device=device)
        npt_velocity_half_step(
            velocities,
            masses,
            forces,
            cell_velocities,
            volumes,
            eta_dots,
            num_atoms=num_atoms_per_system,
            dt=dt,
            batch_idx=batch_idx,
            num_atoms_per_system=num_atoms_per_system,
            cells_inv=cells_inv,
            mode="triclinic",
            device=device,
        )

        assert velocities.shape[0] == num_atoms


# ==============================================================================
# Coverage Tests - Barostat Mass Computation
# ==============================================================================


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
class TestBarostatMassCoverage:
    """Coverage tests for barostat mass computation."""

    def test_compute_barostat_mass_batched_arrays(self, dtype, device):
        """Test barostat mass computation with batched wp.array inputs."""
        scalar_dtype = wp.float32 if dtype == "float32" else wp.float64

        temps = wp.array([1.0, 2.0, 1.5], dtype=scalar_dtype, device=device)
        tau_p = wp.array([1.0, 0.5, 2.0], dtype=scalar_dtype, device=device)
        num_atoms = wp.array([100, 200, 150], dtype=wp.int32, device=device)
        masses_out = wp.empty(3, dtype=scalar_dtype, device=device)

        W = compute_barostat_mass(
            temps,
            tau_p,
            num_atoms,
            masses_out,
            device=device,
        )

        assert W.shape[0] == 3
        # Verify formula: W = (3N + 3) * T * τ²
        W_np = W.numpy()
        np.testing.assert_allclose(W_np[0], (300 + 3) * 1.0 * 1.0, rtol=1e-5)
        np.testing.assert_allclose(W_np[1], (600 + 3) * 2.0 * 0.25, rtol=1e-5)
        np.testing.assert_allclose(W_np[2], (450 + 3) * 1.5 * 4.0, rtol=1e-5)

    def test_compute_barostat_mass_mixed_scalar_array(self, dtype, device):
        """Test barostat mass computation with pre-broadcast array inputs."""
        scalar_dtype = wp.float32 if dtype == "float32" else wp.float64

        # All inputs must be wp.arrays; caller pre-broadcasts
        target_temp = wp.array([1.0, 1.0], dtype=scalar_dtype, device=device)
        tau_p_arr = wp.array([1.0, 1.0], dtype=scalar_dtype, device=device)
        num_atoms = wp.array([100, 200], dtype=wp.int32, device=device)
        masses_out = wp.empty(2, dtype=scalar_dtype, device=device)

        W = compute_barostat_mass(
            target_temp,
            tau_p_arr,
            num_atoms,
            masses_out,
            device=device,
        )

        assert W.shape[0] == 2
        W_np = W.numpy()
        np.testing.assert_allclose(W_np[0], 303.0, rtol=1e-5)
        np.testing.assert_allclose(W_np[1], 603.0, rtol=1e-5)

    def test_compute_barostat_mass_list_inputs(self, dtype, device):
        """Test barostat mass computation with wp.array inputs."""
        scalar_dtype = wp.float32 if dtype == "float32" else wp.float64

        target_temp = wp.array([1.0, 2.0], dtype=scalar_dtype, device=device)
        tau_p_arr = wp.array([1.0, 0.5], dtype=scalar_dtype, device=device)
        num_atoms_arr = wp.array([100, 200], dtype=wp.int32, device=device)
        masses_out = wp.empty(2, dtype=scalar_dtype, device=device)

        W = compute_barostat_mass(
            target_temp,
            tau_p_arr,
            num_atoms_arr,
            masses_out,
            device=device,
        )

        assert W.shape[0] == 2
        W_np = W.numpy()
        np.testing.assert_allclose(W_np[0], 303.0, rtol=1e-5)
        # W = (3N + 3) * T * τ² = 603 * 2.0 * 0.25 = 301.5
        np.testing.assert_allclose(W_np[1], 603.0 * 2.0 * 0.25, rtol=1e-5)


# ==============================================================================
# Coverage Tests - Explicit Aniso/Triclinic Functions
# ==============================================================================


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
class TestExplicitBarostatFunctions:
    """Test explicit aniso/triclinic barostat functions for coverage."""

    def test_npt_barostat_half_step_aniso(self, dtype, device):
        """Test explicit NPT anisotropic barostat."""
        from nvalchemiops.dynamics.integrators.npt import npt_barostat_half_step_aniso

        num_systems = 2
        num_atoms = 20
        mat_dtype = wp.mat33f if dtype == "float32" else wp.mat33d
        tensor_dtype = vec9f if dtype == "float32" else vec9d
        scalar_dtype = wp.float32 if dtype == "float32" else wp.float64
        vec3_dtype = wp.vec3f if dtype == "float32" else wp.vec3d

        cell_velocities = wp.zeros(num_systems, dtype=mat_dtype, device=device)
        pressure_tensors = wp.array(
            [tensor_dtype(1e-4, 0, 0, 0, 1e-4, 0, 0, 0, 1e-4)] * num_systems,
            dtype=tensor_dtype,
            device=device,
        )
        target_pressures = wp.array(
            [vec3_dtype(1e-4, 1e-4, 1e-4)] * num_systems,
            dtype=vec3_dtype,
            device=device,
        )
        volumes = wp.array([1000.0, 1000.0], dtype=scalar_dtype, device=device)
        cell_masses = wp.array([100.0, 100.0], dtype=scalar_dtype, device=device)
        kinetic_energy = wp.array([10.0, 10.0], dtype=scalar_dtype, device=device)
        num_atoms_per_system = wp.array(
            [num_atoms, num_atoms], dtype=wp.int32, device=device
        )
        eta_dots = wp.zeros((num_systems, 3), dtype=scalar_dtype, device=device)

        dt = wp.array([0.001, 0.001], dtype=scalar_dtype, device=device)
        npt_barostat_half_step_aniso(
            cell_velocities,
            pressure_tensors,
            target_pressures,
            volumes,
            cell_masses,
            kinetic_energy,
            num_atoms_per_system,
            eta_dots,
            dt=dt,
            device=device,
        )

        wp.synchronize_device(device)
        assert cell_velocities.shape[0] == num_systems

    def test_npt_barostat_half_step_triclinic(self, dtype, device):
        """Test explicit NPT triclinic barostat."""
        from nvalchemiops.dynamics.integrators.npt import (
            npt_barostat_half_step_triclinic,
        )

        num_systems = 2
        num_atoms = 20
        mat_dtype = wp.mat33f if dtype == "float32" else wp.mat33d
        tensor_dtype = vec9f if dtype == "float32" else vec9d
        scalar_dtype = wp.float32 if dtype == "float32" else wp.float64

        cell_velocities = wp.zeros(num_systems, dtype=mat_dtype, device=device)
        pressure_tensors = wp.array(
            [tensor_dtype(1e-4, 0, 0, 0, 1e-4, 0, 0, 0, 1e-4)] * num_systems,
            dtype=tensor_dtype,
            device=device,
        )
        target_pressures = wp.array(
            [tensor_dtype(1e-4, 0, 0, 0, 1e-4, 0, 0, 0, 1e-4)] * num_systems,
            dtype=tensor_dtype,
            device=device,
        )
        volumes = wp.array([1000.0, 1000.0], dtype=scalar_dtype, device=device)
        cell_masses = wp.array([100.0, 100.0], dtype=scalar_dtype, device=device)
        kinetic_energy = wp.array([10.0, 10.0], dtype=scalar_dtype, device=device)
        num_atoms_per_system = wp.array(
            [num_atoms, num_atoms], dtype=wp.int32, device=device
        )
        eta_dots = wp.zeros((num_systems, 3), dtype=scalar_dtype, device=device)

        dt = wp.array([0.001, 0.001], dtype=scalar_dtype, device=device)
        npt_barostat_half_step_triclinic(
            cell_velocities,
            pressure_tensors,
            target_pressures,
            volumes,
            cell_masses,
            kinetic_energy,
            num_atoms_per_system,
            eta_dots,
            dt=dt,
            device=device,
        )

        wp.synchronize_device(device)
        assert cell_velocities.shape[0] == num_systems

    def test_nph_barostat_half_step_aniso(self, dtype, device):
        """Test explicit NPH anisotropic barostat."""
        from nvalchemiops.dynamics.integrators.npt import nph_barostat_half_step_aniso

        num_systems = 2
        num_atoms = 20
        mat_dtype = wp.mat33f if dtype == "float32" else wp.mat33d
        tensor_dtype = vec9f if dtype == "float32" else vec9d
        scalar_dtype = wp.float32 if dtype == "float32" else wp.float64
        vec3_dtype = wp.vec3f if dtype == "float32" else wp.vec3d

        cell_velocities = wp.zeros(num_systems, dtype=mat_dtype, device=device)
        pressure_tensors = wp.array(
            [tensor_dtype(1e-4, 0, 0, 0, 1e-4, 0, 0, 0, 1e-4)] * num_systems,
            dtype=tensor_dtype,
            device=device,
        )
        target_pressures = wp.array(
            [vec3_dtype(1e-4, 1e-4, 1e-4)] * num_systems,
            dtype=vec3_dtype,
            device=device,
        )
        volumes = wp.array([1000.0, 1000.0], dtype=scalar_dtype, device=device)
        cell_masses = wp.array([100.0, 100.0], dtype=scalar_dtype, device=device)
        kinetic_energy = wp.array([10.0, 10.0], dtype=scalar_dtype, device=device)
        num_atoms_per_system = wp.array(
            [num_atoms, num_atoms], dtype=wp.int32, device=device
        )

        dt = wp.array([0.001, 0.001], dtype=scalar_dtype, device=device)
        nph_barostat_half_step_aniso(
            cell_velocities,
            pressure_tensors,
            target_pressures,
            volumes,
            cell_masses,
            kinetic_energy,
            num_atoms_per_system,
            dt=dt,
            device=device,
        )

        wp.synchronize_device(device)
        assert cell_velocities.shape[0] == num_systems

    def test_nph_barostat_half_step_triclinic(self, dtype, device):
        """Test explicit NPH triclinic barostat."""
        from nvalchemiops.dynamics.integrators.npt import (
            nph_barostat_half_step_triclinic,
        )

        num_systems = 2
        num_atoms = 20
        mat_dtype = wp.mat33f if dtype == "float32" else wp.mat33d
        tensor_dtype = vec9f if dtype == "float32" else vec9d
        scalar_dtype = wp.float32 if dtype == "float32" else wp.float64

        cell_velocities = wp.zeros(num_systems, dtype=mat_dtype, device=device)
        pressure_tensors = wp.array(
            [tensor_dtype(1e-4, 0, 0, 0, 1e-4, 0, 0, 0, 1e-4)] * num_systems,
            dtype=tensor_dtype,
            device=device,
        )
        target_pressures = wp.array(
            [tensor_dtype(1e-4, 0, 0, 0, 1e-4, 0, 0, 0, 1e-4)] * num_systems,
            dtype=tensor_dtype,
            device=device,
        )
        volumes = wp.array([1000.0, 1000.0], dtype=scalar_dtype, device=device)
        cell_masses = wp.array([100.0, 100.0], dtype=scalar_dtype, device=device)
        kinetic_energy = wp.array([10.0, 10.0], dtype=scalar_dtype, device=device)
        num_atoms_per_system = wp.array(
            [num_atoms, num_atoms], dtype=wp.int32, device=device
        )

        dt = wp.array([0.001, 0.001], dtype=scalar_dtype, device=device)
        nph_barostat_half_step_triclinic(
            cell_velocities,
            pressure_tensors,
            target_pressures,
            volumes,
            cell_masses,
            kinetic_energy,
            num_atoms_per_system,
            dt=dt,
            device=device,
        )

        wp.synchronize_device(device)
        assert cell_velocities.shape[0] == num_systems


# ==============================================================================
# Coverage Tests - Device Inference and Pre-allocation
# ==============================================================================


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
class TestNPTDeviceInference:
    """Test device inference for NPT/NPH functions."""

    def test_compute_pressure_tensor_device_inference(self, dtype, device):
        """Test device inference for compute_pressure_tensor."""
        num_atoms = 20
        np.random.seed(42)
        np_dtype = np.float32 if dtype == "float32" else np.float64
        vec_dtype = wp.vec3f if dtype == "float32" else wp.vec3d
        scalar_dtype = wp.float32 if dtype == "float32" else wp.float64
        tensor_dtype = vec9f if dtype == "float32" else vec9d

        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype) * 0.1,
            dtype=vec_dtype,
            device=device,
        )
        masses = wp.ones(num_atoms, dtype=scalar_dtype, device=device)
        cell_np = np.diag([10.0, 10.0, 10.0]).astype(np_dtype)
        cells = make_cell(cell_np, dtype, device)
        virial = wp.zeros(1, dtype=tensor_dtype, device=device)

        kinetic_tensors = wp.zeros((1, 9), dtype=scalar_dtype, device=device)
        pressure_tensors = wp.empty(1, dtype=tensor_dtype, device=device)
        volumes = wp.empty(1, dtype=scalar_dtype, device=device)
        compute_cell_volume(cells, volumes, device=device)

        # Call without explicit device (note: virial_tensors comes before cells)
        result = compute_pressure_tensor(
            velocities,
            masses,
            virial,
            cells,
            kinetic_tensors,
            pressure_tensors,
            volumes,
        )

        wp.synchronize_device(device)
        assert result.shape[0] == 1

    def test_compute_scalar_pressure_device_inference(self, dtype, device):
        """Test device inference for compute_scalar_pressure."""
        tensor_dtype = vec9f if dtype == "float32" else vec9d
        scalar_dtype = wp.float32 if dtype == "float32" else wp.float64
        pressure_tensor = wp.array(
            [tensor_dtype(0.1, 0, 0, 0, 0.1, 0, 0, 0, 0.1)],
            dtype=tensor_dtype,
            device=device,
        )

        scalar_out = wp.empty(1, dtype=scalar_dtype, device=device)
        # Call without explicit device
        result = compute_scalar_pressure(pressure_tensor, scalar_out)

        wp.synchronize_device(device)
        np.testing.assert_allclose(result.numpy()[0], 0.1, rtol=1e-5)

    def test_compute_cell_kinetic_energy_device_inference(self, dtype, device):
        """Test device inference for compute_cell_kinetic_energy."""
        mat_dtype = wp.mat33f if dtype == "float32" else wp.mat33d
        scalar_dtype = wp.float32 if dtype == "float32" else wp.float64
        cell_velocities = wp.zeros(1, dtype=mat_dtype, device=device)
        cells_inv = wp.array(
            [mat_dtype(1, 0, 0, 0, 1, 0, 0, 0, 1)], dtype=mat_dtype, device=device
        )
        cell_masses = wp.array([100.0], dtype=scalar_dtype, device=device)
        ke_out = wp.empty(1, dtype=scalar_dtype, device=device)

        # Call without explicit device
        result = compute_cell_kinetic_energy(
            cell_velocities, cell_masses, ke_out, cells_inv=cells_inv
        )

        wp.synchronize_device(device)
        assert result.shape[0] == 1

    def test_npt_position_update_device_inference(self, dtype, device):
        """Test device inference for npt_position_update."""
        num_atoms = 10
        np.random.seed(42)
        np_dtype = np.float32 if dtype == "float32" else np.float64
        vec_dtype = wp.vec3f if dtype == "float32" else wp.vec3d
        mat_dtype = wp.mat33f if dtype == "float32" else wp.mat33d

        positions = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=vec_dtype,
            device=device,
        )
        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype) * 0.1,
            dtype=vec_dtype,
            device=device,
        )
        cell_np = np.diag([10.0, 10.0, 10.0]).astype(np_dtype)
        cells = make_cell(cell_np, dtype, device)
        cell_velocities = wp.zeros(1, dtype=mat_dtype, device=device)
        cells_inv = wp.empty(1, dtype=mat_dtype, device=device)
        compute_cell_inverse(cells, cells_inv, device=device)

        scalar_dtype = wp.float32 if dtype == "float32" else wp.float64
        dt = wp.array([0.001], dtype=scalar_dtype, device=device)
        # Call without explicit device (should infer from positions)
        npt_position_update(
            positions, velocities, cells, cell_velocities, dt=dt, cells_inv=cells_inv
        )

        wp.synchronize_device(device)
        assert positions.shape[0] == num_atoms

    def test_npt_position_update_batched(self, dtype, device):
        """Test batched NPT position update."""
        num_atoms = 20
        num_systems = 2
        np.random.seed(42)
        np_dtype = np.float32 if dtype == "float32" else np.float64
        vec_dtype = wp.vec3f if dtype == "float32" else wp.vec3d

        positions = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=vec_dtype,
            device=device,
        )
        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype) * 0.1,
            dtype=vec_dtype,
            device=device,
        )
        batch_idx = wp.array([0] * 10 + [1] * 10, dtype=wp.int32, device=device)
        cell_np = np.diag([10.0, 10.0, 10.0]).astype(np_dtype)
        cells_np = np.stack([cell_np, cell_np])
        cells = make_cells_batch(cells_np, dtype, device)
        mat_dtype = wp.mat33f if dtype == "float32" else wp.mat33d
        cell_velocities = wp.zeros(num_systems, dtype=mat_dtype, device=device)
        cells_inv = wp.empty(num_systems, dtype=mat_dtype, device=device)
        compute_cell_inverse(cells, cells_inv, device=device)

        scalar_dtype = wp.float32 if dtype == "float32" else wp.float64
        dt = wp.array([0.001, 0.001], dtype=scalar_dtype, device=device)
        npt_position_update(
            positions,
            velocities,
            cells,
            cell_velocities,
            dt=dt,
            cells_inv=cells_inv,
            batch_idx=batch_idx,
            device=device,
        )

        wp.synchronize_device(device)
        assert positions.shape[0] == num_atoms

    def test_npt_velocity_out_batched_triclinic(self, dtype, device):
        """Test batched NPT velocity update out with triclinic mode."""
        num_atoms = 20
        num_systems = 2
        np.random.seed(42)
        np_dtype = np.float32 if dtype == "float32" else np.float64
        vec_dtype = wp.vec3f if dtype == "float32" else wp.vec3d
        mat_dtype = wp.mat33f if dtype == "float32" else wp.mat33d
        scalar_dtype = wp.float32 if dtype == "float32" else wp.float64

        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype) * 0.1,
            dtype=vec_dtype,
            device=device,
        )
        masses = wp.ones(num_atoms, dtype=scalar_dtype, device=device)
        forces = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype) * 0.01,
            dtype=vec_dtype,
            device=device,
        )
        batch_idx = wp.array([0] * 10 + [1] * 10, dtype=wp.int32, device=device)

        cell_np = np.diag([10.0, 10.0, 10.0]).astype(np_dtype)
        cell_inv_np = np.linalg.inv(cell_np).astype(np_dtype)
        cells_inv_np_batch = np.stack([cell_inv_np, cell_inv_np])
        cells_inv = make_cells_batch(cells_inv_np_batch, dtype, device)
        cell_velocities = wp.zeros(num_systems, dtype=mat_dtype, device=device)
        volumes = wp.array([1000.0] * num_systems, dtype=scalar_dtype, device=device)
        eta_dots = wp.zeros((num_systems, 3), dtype=scalar_dtype, device=device)
        num_atoms_per_system = wp.array([10, 10], dtype=wp.int32, device=device)

        result = wp.empty_like(velocities)
        dt = wp.array([0.001, 0.001], dtype=scalar_dtype, device=device)
        result = npt_velocity_half_step_out(
            velocities,
            masses,
            forces,
            cell_velocities,
            volumes,
            eta_dots,
            num_atoms_per_system,
            dt,
            result,
            batch_idx=batch_idx,
            num_atoms_per_system=num_atoms_per_system,
            cells_inv=cells_inv,
            mode="triclinic",
            device=device,
        )

        wp.synchronize_device(device)
        assert result.shape[0] == num_atoms

    def test_nph_velocity_out_batched_triclinic(self, dtype, device):
        """Test batched NPH velocity update out with triclinic mode."""
        num_atoms = 20
        num_systems = 2
        np.random.seed(42)
        np_dtype = np.float32 if dtype == "float32" else np.float64
        vec_dtype = wp.vec3f if dtype == "float32" else wp.vec3d
        mat_dtype = wp.mat33f if dtype == "float32" else wp.mat33d
        scalar_dtype = wp.float32 if dtype == "float32" else wp.float64

        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype) * 0.1,
            dtype=vec_dtype,
            device=device,
        )
        masses = wp.ones(num_atoms, dtype=scalar_dtype, device=device)
        forces = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype) * 0.01,
            dtype=vec_dtype,
            device=device,
        )
        batch_idx = wp.array([0] * 10 + [1] * 10, dtype=wp.int32, device=device)

        cell_np = np.diag([10.0, 10.0, 10.0]).astype(np_dtype)
        cell_inv_np = np.linalg.inv(cell_np).astype(np_dtype)
        cells_inv_np_batch = np.stack([cell_inv_np, cell_inv_np])
        cells_inv = make_cells_batch(cells_inv_np_batch, dtype, device)
        cell_velocities = wp.zeros(num_systems, dtype=mat_dtype, device=device)
        volumes = wp.array([1000.0] * num_systems, dtype=scalar_dtype, device=device)
        num_atoms_per_system = wp.array([10, 10], dtype=wp.int32, device=device)

        result = wp.empty_like(velocities)
        dt = wp.array([0.001, 0.001], dtype=scalar_dtype, device=device)
        result = nph_velocity_half_step_out(
            velocities,
            masses,
            forces,
            cell_velocities,
            volumes,
            num_atoms_per_system,
            dt,
            result,
            batch_idx=batch_idx,
            num_atoms_per_system=num_atoms_per_system,
            cells_inv=cells_inv,
            mode="triclinic",
            device=device,
        )

        wp.synchronize_device(device)
        assert result.shape[0] == num_atoms


@pytest.mark.parametrize("device", DEVICES)
class TestNPTCoverageExtras:
    """Additional tests for NPT/NPH coverage."""

    def test_compute_scalar_pressure_preallocated(self, device):
        """Test compute_scalar_pressure with pre-allocated output."""
        pressure_tensor = wp.array(
            [vec9f(0.1, 0, 0, 0, 0.2, 0, 0, 0, 0.3)],
            dtype=vec9f,
            device=device,
        )
        scalar_out = wp.empty(1, dtype=wp.float32, device=device)

        result = compute_scalar_pressure(pressure_tensor, scalar_out)

        wp.synchronize_device(device)
        assert result is scalar_out
        # P_scalar = (0.1 + 0.2 + 0.3) / 3 = 0.2
        np.testing.assert_allclose(result.numpy()[0], 0.2, rtol=1e-5)

    def test_compute_barostat_mass_device_inference(self, device):
        """Test compute_barostat_mass with device inference."""
        target_temp = wp.array([1.0], dtype=wp.float32, device=device)
        tau_p_arr = wp.array([1.0], dtype=wp.float32, device=device)
        num_atoms_arr = wp.array([100], dtype=wp.int32, device=device)
        masses_out = wp.empty(1, dtype=wp.float32, device=device)

        # Don't pass device, let it default
        W = compute_barostat_mass(
            target_temp,
            tau_p_arr,
            num_atoms_arr,
            masses_out,
        )

        assert W.shape[0] == 1
        # W = (3N + 3) * T * τ² = 303 * 1.0 * 1.0 = 303.0
        np.testing.assert_allclose(W.numpy()[0], 303.0, rtol=1e-5)

    def test_compute_barostat_mass_broadcast_scalars(self, device):
        """Test compute_barostat_mass with pre-broadcast arrays."""
        target_temp = wp.array([1.0, 1.0], dtype=wp.float32, device=device)
        tau_p_arr = wp.array([1.0, 1.0], dtype=wp.float32, device=device)
        num_atoms_arr = wp.array([100, 200], dtype=wp.int32, device=device)
        masses_out = wp.empty(2, dtype=wp.float32, device=device)

        W = compute_barostat_mass(
            target_temp,
            tau_p_arr,
            num_atoms_arr,
            masses_out,
            device=device,
        )

        assert W.shape[0] == 2
        np.testing.assert_allclose(W.numpy()[0], 303.0, rtol=1e-5)
        np.testing.assert_allclose(W.numpy()[1], 603.0, rtol=1e-5)

    def test_npt_barostat_half_step_device_inference(self, device):
        """Test npt_barostat_half_step with device inference."""
        num_systems = 1
        chain_length = 3

        cell_velocities = wp.zeros(num_systems, dtype=wp.mat33f, device=device)
        pressure_tensors = wp.empty(num_systems, dtype=vec9f, device=device)
        target_pressures = wp.array([0.1], dtype=wp.float32, device=device)
        volumes = wp.array([1000.0], dtype=wp.float32, device=device)
        cell_masses = wp.array([100.0], dtype=wp.float32, device=device)
        kinetic_energy = wp.array([10.0], dtype=wp.float32, device=device)
        num_atoms_per_system = wp.array([100], dtype=wp.int32, device=device)
        # eta_dots must be 2D: (B, chain_length)
        eta_dots = wp.zeros(
            (num_systems, chain_length), dtype=wp.float32, device=device
        )

        dt = wp.array([0.001], dtype=wp.float32, device=device)
        # Don't pass device
        npt_barostat_half_step(
            cell_velocities,
            pressure_tensors,
            target_pressures,
            volumes,
            cell_masses,
            kinetic_energy,
            num_atoms_per_system,
            eta_dots,
            dt=dt,
        )

        wp.synchronize_device(device)
        assert cell_velocities.shape[0] == num_systems

    def test_nph_velocity_half_step_batched_triclinic(self, device):
        """Test nph_velocity_half_step with batched triclinic mode."""
        num_atoms = 20
        num_systems = 2
        np.random.seed(42)

        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np.float32) * 0.1,
            dtype=wp.vec3f,
            device=device,
        )
        masses = wp.ones(num_atoms, dtype=wp.float32, device=device)
        forces = wp.zeros(num_atoms, dtype=wp.vec3f, device=device)
        batch_idx = wp.array([0] * 10 + [1] * 10, dtype=wp.int32, device=device)

        cell_np = np.diag([10.0, 10.0, 10.0]).astype(np.float32)
        cell_inv_np = np.linalg.inv(cell_np).astype(np.float32)
        cells_inv_np_batch = np.stack([cell_inv_np, cell_inv_np])
        cells_inv = make_cells_batch(cells_inv_np_batch, "float32", device)
        cell_velocities = wp.zeros(num_systems, dtype=wp.mat33f, device=device)
        volumes = wp.array([1000.0] * num_systems, dtype=wp.float32, device=device)
        num_atoms_per_system = wp.array([10, 10], dtype=wp.int32, device=device)

        dt = wp.array([0.001, 0.001], dtype=wp.float32, device=device)
        # Mutating batched triclinic
        nph_velocity_half_step(
            velocities,
            masses,
            forces,
            cell_velocities,
            volumes,
            num_atoms=num_atoms_per_system,
            dt=dt,
            batch_idx=batch_idx,
            num_atoms_per_system=num_atoms_per_system,
            cells_inv=cells_inv,
            mode="triclinic",
            device=device,
        )

        wp.synchronize_device(device)
        assert velocities.shape[0] == num_atoms

    def test_npt_velocity_half_step_batched_triclinic(self, device):
        """Test npt_velocity_half_step with batched triclinic mode."""
        num_atoms = 20
        num_systems = 2
        chain_length = 3
        np.random.seed(42)

        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np.float32) * 0.1,
            dtype=wp.vec3f,
            device=device,
        )
        masses = wp.ones(num_atoms, dtype=wp.float32, device=device)
        forces = wp.zeros(num_atoms, dtype=wp.vec3f, device=device)
        batch_idx = wp.array([0] * 10 + [1] * 10, dtype=wp.int32, device=device)

        cell_np = np.diag([10.0, 10.0, 10.0]).astype(np.float32)
        cell_inv_np = np.linalg.inv(cell_np).astype(np.float32)
        cells_inv_np_batch = np.stack([cell_inv_np, cell_inv_np])
        cells_inv = make_cells_batch(cells_inv_np_batch, "float32", device)
        cell_velocities = wp.zeros(num_systems, dtype=wp.mat33f, device=device)
        volumes = wp.array([1000.0] * num_systems, dtype=wp.float32, device=device)
        # eta_dots must be 2D: (B, chain_length)
        eta_dots = wp.zeros(
            (num_systems, chain_length), dtype=wp.float32, device=device
        )
        num_atoms_per_system = wp.array([10, 10], dtype=wp.int32, device=device)

        dt = wp.array([0.001, 0.001], dtype=wp.float32, device=device)
        # Mutating batched triclinic
        npt_velocity_half_step(
            velocities,
            masses,
            forces,
            cell_velocities,
            volumes,
            eta_dots,
            num_atoms=num_atoms_per_system,
            dt=dt,
            batch_idx=batch_idx,
            num_atoms_per_system=num_atoms_per_system,
            cells_inv=cells_inv,
            mode="triclinic",
            device=device,
        )

        wp.synchronize_device(device)
        assert velocities.shape[0] == num_atoms

    def test_explicit_aniso_barostat_functions(self, device):
        """Test explicit aniso barostat half step functions."""
        from nvalchemiops.dynamics.integrators.npt import (
            nph_barostat_half_step_aniso,
            nph_barostat_half_step_triclinic,
            npt_barostat_half_step_aniso,
            npt_barostat_half_step_triclinic,
        )

        num_systems = 1
        chain_length = 3

        cell_velocities = wp.zeros(num_systems, dtype=wp.mat33f, device=device)
        pressure_tensors = wp.empty(num_systems, dtype=vec9f, device=device)
        volumes = wp.array([1000.0], dtype=wp.float32, device=device)
        cell_masses = wp.array([100.0], dtype=wp.float32, device=device)
        kinetic_energy = wp.array([10.0], dtype=wp.float32, device=device)
        num_atoms_per_system = wp.array([100], dtype=wp.int32, device=device)
        # eta_dots must be 2D: (B, chain_length)
        eta_dots = wp.zeros(
            (num_systems, chain_length), dtype=wp.float32, device=device
        )

        dt = wp.array([0.001], dtype=wp.float32, device=device)
        # Test NPT aniso
        target_pressures_aniso = wp.array(
            [wp.vec3f(0.1, 0.1, 0.1)], dtype=wp.vec3f, device=device
        )
        npt_barostat_half_step_aniso(
            cell_velocities,
            pressure_tensors,
            target_pressures_aniso,
            volumes,
            cell_masses,
            kinetic_energy,
            num_atoms_per_system,
            eta_dots,
            dt=dt,
            device=device,
        )

        # Test NPT triclinic
        target_pressures_tri = wp.zeros(num_systems, dtype=vec9f, device=device)
        npt_barostat_half_step_triclinic(
            cell_velocities,
            pressure_tensors,
            target_pressures_tri,
            volumes,
            cell_masses,
            kinetic_energy,
            num_atoms_per_system,
            eta_dots,
            dt=dt,
            device=device,
        )

        # Test NPH aniso
        nph_barostat_half_step_aniso(
            cell_velocities,
            pressure_tensors,
            target_pressures_aniso,
            volumes,
            cell_masses,
            kinetic_energy,
            num_atoms_per_system,
            dt=dt,
            device=device,
        )

        # Test NPH triclinic
        nph_barostat_half_step_triclinic(
            cell_velocities,
            pressure_tensors,
            target_pressures_tri,
            volumes,
            cell_masses,
            kinetic_energy,
            num_atoms_per_system,
            dt=dt,
            device=device,
        )

        wp.synchronize_device(device)
        assert cell_velocities.shape[0] == num_systems

    def test_npt_position_update_out_batched(self, device):
        """Test npt_position_update_out with batched mode."""
        num_atoms = 20
        num_systems = 2
        np.random.seed(42)

        positions = wp.array(
            np.random.randn(num_atoms, 3).astype(np.float32),
            dtype=wp.vec3f,
            device=device,
        )
        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np.float32) * 0.1,
            dtype=wp.vec3f,
            device=device,
        )
        batch_idx = wp.array([0] * 10 + [1] * 10, dtype=wp.int32, device=device)

        cell_np = np.diag([10.0, 10.0, 10.0]).astype(np.float32)
        cells_np = np.stack([cell_np, cell_np])
        cells = make_cells_batch(cells_np, "float32", device)
        cell_velocities = wp.zeros(num_systems, dtype=wp.mat33f, device=device)
        cells_inv = wp.empty(num_systems, dtype=wp.mat33f, device=device)
        compute_cell_inverse(cells, cells_inv, device=device)

        dt = wp.array([0.001, 0.001], dtype=wp.float32, device=device)
        # Pre-allocate output
        result = wp.empty_like(positions)
        result = npt_position_update_out(
            positions,
            velocities,
            cells,
            cell_velocities,
            dt,
            result,
            cells_inv=cells_inv,
            batch_idx=batch_idx,
            device=device,
        )

        wp.synchronize_device(device)
        assert result.shape[0] == num_atoms

    def test_nph_position_update_out_batched(self, device):
        """Test nph_position_update_out with batched mode."""
        num_atoms = 20
        num_systems = 2
        np.random.seed(42)

        positions = wp.array(
            np.random.randn(num_atoms, 3).astype(np.float32),
            dtype=wp.vec3f,
            device=device,
        )
        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np.float32) * 0.1,
            dtype=wp.vec3f,
            device=device,
        )
        batch_idx = wp.array([0] * 10 + [1] * 10, dtype=wp.int32, device=device)

        cell_np = np.diag([10.0, 10.0, 10.0]).astype(np.float32)
        cells_np = np.stack([cell_np, cell_np])
        cells = make_cells_batch(cells_np, "float32", device)
        cell_velocities = wp.zeros(num_systems, dtype=wp.mat33f, device=device)
        cells_inv = wp.empty(num_systems, dtype=wp.mat33f, device=device)
        compute_cell_inverse(cells, cells_inv, device=device)

        dt = wp.array([0.001, 0.001], dtype=wp.float32, device=device)
        # Pre-allocate output
        result = wp.empty_like(positions)
        result = nph_position_update_out(
            positions,
            velocities,
            cells,
            cell_velocities,
            dt,
            result,
            cells_inv=cells_inv,
            batch_idx=batch_idx,
            device=device,
        )

        wp.synchronize_device(device)
        assert result.shape[0] == num_atoms


# ==============================================================================
# Additional Coverage Tests
# ==============================================================================


class TestAdditionalCoverage:
    """Additional tests to improve coverage for edge cases and device inference."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_compute_barostat_mass_N_broadcast(self, device):
        """Test compute_barostat_mass with pre-broadcast N_arr for multiple systems."""
        # All inputs must be wp.arrays; caller pre-broadcasts
        T_arr = wp.array([1.0, 1.5], dtype=wp.float32, device=device)
        tau_arr = wp.array([1.0, 1.0], dtype=wp.float32, device=device)
        num_atoms_arr = wp.array([100, 100], dtype=wp.int32, device=device)
        masses_out = wp.empty(2, dtype=wp.float32, device=device)

        W = compute_barostat_mass(
            T_arr,
            tau_arr,
            num_atoms_arr,
            masses_out,
            device=device,
        )

        wp.synchronize_device(device)
        assert W.shape[0] == 2
        # W = (3N + 3) * T * τ² = 303 * T
        np.testing.assert_allclose(W.numpy()[0], 303.0, rtol=1e-5)
        np.testing.assert_allclose(W.numpy()[1], 454.5, rtol=1e-5)

    @pytest.mark.parametrize("device", DEVICES)
    def test_compute_barostat_potential_device_inference(self, device):
        """Test compute_barostat_potential_energy with device inference."""
        target_pressures = wp.array([0.1], dtype=wp.float32, device=device)
        volumes = wp.array([1000.0], dtype=wp.float32, device=device)
        pe_out = wp.empty(1, dtype=wp.float32, device=device)

        # Don't pass device
        result = compute_barostat_potential_energy(target_pressures, volumes, pe_out)

        wp.synchronize_device(device)
        assert result.shape[0] == 1
        # PE = P * V = 0.1 * 1000 = 100.0
        np.testing.assert_allclose(result.numpy()[0], 100.0, rtol=1e-5)

    @pytest.mark.parametrize("device", DEVICES)
    def test_compute_barostat_potential_output_allocation(self, device):
        """Test compute_barostat_potential_energy with pre-allocated output."""
        target_pressures = wp.array([0.1, 0.2], dtype=wp.float32, device=device)
        volumes = wp.array([1000.0, 500.0], dtype=wp.float32, device=device)
        pe_out = wp.empty(2, dtype=wp.float32, device=device)

        result = compute_barostat_potential_energy(
            target_pressures,
            volumes,
            pe_out,
            device=device,
        )

        wp.synchronize_device(device)
        assert result.shape[0] == 2
        np.testing.assert_allclose(result.numpy()[0], 100.0, rtol=1e-5)
        np.testing.assert_allclose(result.numpy()[1], 100.0, rtol=1e-5)

    @pytest.mark.parametrize("device", DEVICES)
    def test_compute_cell_kinetic_energy_preallocated(self, device):
        """Test compute_cell_kinetic_energy with pre-allocated output."""
        num_systems = 2
        cell_vel_np = np.diag([0.1, 0.1, 0.1]).astype(np.float32)
        cell_velocities = wp.array(
            [wp.mat33f(*cell_vel_np.flatten())] * num_systems,
            dtype=wp.mat33f,
            device=device,
        )
        cell_masses = wp.array([100.0, 100.0], dtype=wp.float32, device=device)
        cells_inv = wp.array(
            [wp.mat33f(1, 0, 0, 0, 1, 0, 0, 0, 1)] * num_systems,
            dtype=wp.mat33f,
            device=device,
        )
        kinetic_energy = wp.empty(num_systems, dtype=wp.float32, device=device)

        result = compute_cell_kinetic_energy(
            cell_velocities,
            cell_masses,
            kinetic_energy,
            cells_inv=cells_inv,
            device=device,
        )

        wp.synchronize_device(device)
        assert result is kinetic_energy
        # KE = 0.5 * W * ||ε̇||²_F where ε̇ = ḣ h⁻¹
        # With h⁻¹ = I: ε̇ = ḣ, so ||ε̇||²_F = 3 * 0.1² = 0.03
        np.testing.assert_allclose(result.numpy()[0], 1.5, rtol=1e-4)

    @pytest.mark.parametrize("device", DEVICES)
    def test_npt_thermostat_half_step_device_inference(self, device):
        """Test npt_thermostat_half_step with device inference."""
        from nvalchemiops.dynamics.integrators.npt import npt_thermostat_half_step

        num_systems = 1
        chain_length = 3

        eta = wp.zeros((num_systems, chain_length), dtype=wp.float64, device=device)
        eta_dot = wp.zeros((num_systems, chain_length), dtype=wp.float64, device=device)
        kinetic_energy = wp.array([10.0], dtype=wp.float64, device=device)
        target_temperature = wp.array([1.0], dtype=wp.float64, device=device)
        thermostat_masses = wp.ones(
            (num_systems, chain_length), dtype=wp.float64, device=device
        )
        num_atoms_per_system = wp.array([100], dtype=wp.int32, device=device)

        dt = wp.array([0.0005], dtype=wp.float64, device=device)
        # Don't pass device
        npt_thermostat_half_step(
            eta,
            eta_dot,
            kinetic_energy,
            target_temperature,
            thermostat_masses,
            num_atoms_per_system,
            chain_length=chain_length,
            dt=dt,
        )

        wp.synchronize_device(device)
        # Just check it ran without error

    @pytest.mark.parametrize("device", DEVICES)
    def test_nph_barostat_half_step_device_inference(self, device):
        """Test nph_barostat_half_step with device inference."""
        num_systems = 1

        cell_velocities = wp.zeros(num_systems, dtype=wp.mat33f, device=device)
        pressure_tensors = wp.empty(num_systems, dtype=vec9f, device=device)
        target_pressures = wp.array([0.1], dtype=wp.float32, device=device)
        volumes = wp.array([1000.0], dtype=wp.float32, device=device)
        cell_masses = wp.array([100.0], dtype=wp.float32, device=device)
        kinetic_energy = wp.array([10.0], dtype=wp.float32, device=device)
        num_atoms_per_system = wp.array([100], dtype=wp.int32, device=device)

        dt = wp.array([0.001], dtype=wp.float32, device=device)
        # Don't pass device
        nph_barostat_half_step(
            cell_velocities,
            pressure_tensors,
            target_pressures,
            volumes,
            cell_masses,
            kinetic_energy,
            num_atoms_per_system,
            dt=dt,
        )

        wp.synchronize_device(device)
        # Just check it ran without error

    @pytest.mark.parametrize("device", DEVICES)
    def test_npt_cell_update_device_inference(self, device):
        """Test npt_cell_update with device inference."""
        num_systems = 1

        cell_np = np.diag([10.0, 10.0, 10.0]).astype(np.float32)
        cells_np = np.stack([cell_np])
        cells = make_cells_batch(cells_np, "float32", device)
        cell_velocities = wp.zeros(num_systems, dtype=wp.mat33f, device=device)

        dt = wp.array([0.001], dtype=wp.float32, device=device)
        # Don't pass device
        npt_cell_update(cells, cell_velocities, dt=dt)

        wp.synchronize_device(device)
        # Just check it ran without error

    @pytest.mark.parametrize("device", DEVICES)
    def test_npt_cell_update_out_device_inference(self, device):
        """Test npt_cell_update_out with device inference."""
        num_systems = 1

        cell_np = np.diag([10.0, 10.0, 10.0]).astype(np.float32)
        cells_np = np.stack([cell_np])
        cells = make_cells_batch(cells_np, "float32", device)
        cell_velocities = wp.zeros(num_systems, dtype=wp.mat33f, device=device)

        dt = wp.array([0.001], dtype=wp.float32, device=device)
        # Don't pass device
        result = wp.empty_like(cells)
        result = npt_cell_update_out(cells, cell_velocities, dt, result)

        wp.synchronize_device(device)
        assert result.shape[0] == num_systems

    @pytest.mark.parametrize("device", DEVICES)
    def test_npt_velocity_half_step_device_inference(self, device):
        """Test npt_velocity_half_step with device inference."""
        num_atoms = 10
        num_systems = 1
        np.random.seed(42)

        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np.float32) * 0.1,
            dtype=wp.vec3f,
            device=device,
        )
        masses = wp.ones(num_atoms, dtype=wp.float32, device=device)
        forces = wp.zeros(num_atoms, dtype=wp.vec3f, device=device)

        cell_np = np.diag([10.0, 10.0, 10.0]).astype(np.float32)
        cell_inv_np = np.linalg.inv(cell_np).astype(np.float32)
        cells_inv = wp.array(
            [wp.mat33f(*cell_inv_np.flatten())], dtype=wp.mat33f, device=device
        )
        cell_velocities = wp.zeros(num_systems, dtype=wp.mat33f, device=device)
        volumes = wp.array([1000.0], dtype=wp.float32, device=device)
        eta_dot_0 = wp.zeros((num_systems, 1), dtype=wp.float32, device=device)

        num_atoms_arr = wp.array([num_atoms], dtype=wp.int32, device=device)
        dt = wp.array([0.001], dtype=wp.float32, device=device)
        # Don't pass device
        npt_velocity_half_step(
            velocities,
            masses,
            forces,
            cell_velocities,
            volumes,
            eta_dot_0,
            num_atoms_arr,
            dt=dt,
            cells_inv=cells_inv,
        )

        wp.synchronize_device(device)
        # Just check it ran without error


# ==============================================================================
# Single vs Batch Equivalence
# ==============================================================================


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
class TestSingleBatchEquivalence:
    """Verify single-system and batch_idx dispatch paths produce identical results."""

    def _setup(self, dtype, device, seed=42):
        """Create a single-system test fixture usable by both dispatch paths."""
        np.random.seed(seed)
        vec_dtype = wp.vec3f if dtype == "float32" else wp.vec3d
        mat_dtype = wp.mat33f if dtype == "float32" else wp.mat33d
        scalar_dtype = wp.float32 if dtype == "float32" else wp.float64
        tensor_dtype = vec9f if dtype == "float32" else vec9d
        np_dtype = np.float32 if dtype == "float32" else np.float64

        num_atoms = 20

        positions = wp.array(
            np.random.rand(num_atoms, 3).astype(np_dtype) * 8.0 + 1.0,
            dtype=vec_dtype,
            device=device,
        )
        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype) * 0.3,
            dtype=vec_dtype,
            device=device,
        )
        forces = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype) * 0.01,
            dtype=vec_dtype,
            device=device,
        )
        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype) * 12.0,
            dtype=scalar_dtype,
            device=device,
        )

        cell_np = np.diag([10.0, 10.0, 10.0]).astype(np_dtype)
        cells = make_cell(cell_np, dtype, device)

        cell_inv_np = np.linalg.inv(cell_np).astype(np_dtype)
        cells_inv = make_cell(cell_inv_np, dtype, device)

        cell_velocities = wp.zeros(1, dtype=mat_dtype, device=device)
        virial_tensors = wp.zeros(1, dtype=tensor_dtype, device=device)
        volumes = wp.empty(1, dtype=scalar_dtype, device=device)
        compute_cell_volume(cells, volumes, device=device)
        kinetic_tensors = wp.zeros((1, 9), dtype=scalar_dtype, device=device)
        num_atoms_per_system = wp.array([num_atoms], dtype=wp.int32, device=device)
        batch_idx = wp.zeros(num_atoms, dtype=wp.int32, device=device)

        eta_dots = wp.zeros((1, 1), dtype=scalar_dtype, device=device)
        dt = wp.array([0.001], dtype=scalar_dtype, device=device)

        return dict(
            num_atoms=num_atoms,
            positions=positions,
            velocities=velocities,
            forces=forces,
            masses=masses,
            cells=cells,
            cells_inv=cells_inv,
            cell_velocities=cell_velocities,
            virial_tensors=virial_tensors,
            volumes=volumes,
            kinetic_tensors=kinetic_tensors,
            num_atoms_per_system=num_atoms_per_system,
            batch_idx=batch_idx,
            eta_dots=eta_dots,
            dt=dt,
            vec_dtype=vec_dtype,
            scalar_dtype=scalar_dtype,
            tensor_dtype=tensor_dtype,
        )

    # -- compute_pressure_tensor ------------------------------------------

    def test_pressure_tensor_equivalence(self, dtype, device):
        s = self._setup(dtype, device)

        kt_single = wp.zeros((1, 9), dtype=s["scalar_dtype"], device=device)
        pt_single = wp.empty(1, dtype=s["tensor_dtype"], device=device)
        compute_pressure_tensor(
            s["velocities"],
            s["masses"],
            s["virial_tensors"],
            s["cells"],
            kt_single,
            pt_single,
            s["volumes"],
            batch_idx=None,
            device=device,
        )

        kt_batch = wp.zeros((1, 9), dtype=s["scalar_dtype"], device=device)
        pt_batch = wp.empty(1, dtype=s["tensor_dtype"], device=device)
        compute_pressure_tensor(
            s["velocities"],
            s["masses"],
            s["virial_tensors"],
            s["cells"],
            kt_batch,
            pt_batch,
            s["volumes"],
            batch_idx=s["batch_idx"],
            device=device,
        )
        wp.synchronize_device(device)

        np.testing.assert_allclose(
            pt_single.numpy(),
            pt_batch.numpy(),
            rtol=1e-5,
            atol=1e-7,
        )

    # -- npt_position_update_out ------------------------------------------

    def test_npt_position_update_equivalence(self, dtype, device):
        s = self._setup(dtype, device)

        pos_out_single = wp.empty_like(s["positions"])
        npt_position_update_out(
            s["positions"],
            s["velocities"],
            s["cells"],
            s["cell_velocities"],
            dt=s["dt"],
            positions_out=pos_out_single,
            cells_inv=s["cells_inv"],
            batch_idx=None,
            device=device,
        )

        pos_out_batch = wp.empty_like(s["positions"])
        npt_position_update_out(
            s["positions"],
            s["velocities"],
            s["cells"],
            s["cell_velocities"],
            dt=s["dt"],
            positions_out=pos_out_batch,
            cells_inv=s["cells_inv"],
            batch_idx=s["batch_idx"],
            device=device,
        )
        wp.synchronize_device(device)

        np.testing.assert_allclose(
            pos_out_single.numpy(),
            pos_out_batch.numpy(),
            rtol=1e-5,
            atol=1e-7,
        )

    # -- npt_velocity_half_step_out (all modes) ---------------------------

    @pytest.mark.parametrize("mode", ["isotropic", "anisotropic", "triclinic"])
    def test_npt_velocity_equivalence(self, dtype, device, mode):
        s = self._setup(dtype, device)

        vel_out_single = wp.empty_like(s["velocities"])
        npt_velocity_half_step_out(
            s["velocities"],
            s["masses"],
            s["forces"],
            s["cell_velocities"],
            s["volumes"],
            s["eta_dots"],
            s["num_atoms_per_system"],
            dt=s["dt"],
            velocities_out=vel_out_single,
            batch_idx=None,
            num_atoms_per_system=s["num_atoms_per_system"],
            cells_inv=s["cells_inv"],
            mode=mode,
            device=device,
        )

        vel_out_batch = wp.empty_like(s["velocities"])
        npt_velocity_half_step_out(
            s["velocities"],
            s["masses"],
            s["forces"],
            s["cell_velocities"],
            s["volumes"],
            s["eta_dots"],
            s["num_atoms_per_system"],
            dt=s["dt"],
            velocities_out=vel_out_batch,
            batch_idx=s["batch_idx"],
            num_atoms_per_system=s["num_atoms_per_system"],
            cells_inv=s["cells_inv"],
            mode=mode,
            device=device,
        )
        wp.synchronize_device(device)

        np.testing.assert_allclose(
            vel_out_single.numpy(),
            vel_out_batch.numpy(),
            rtol=1e-5,
            atol=1e-7,
        )

    # -- nph_velocity_half_step_out (all modes) ---------------------------

    @pytest.mark.parametrize("mode", ["isotropic", "anisotropic", "triclinic"])
    def test_nph_velocity_equivalence(self, dtype, device, mode):
        s = self._setup(dtype, device)

        vel_out_single = wp.empty_like(s["velocities"])
        nph_velocity_half_step_out(
            s["velocities"],
            s["masses"],
            s["forces"],
            s["cell_velocities"],
            s["volumes"],
            s["num_atoms_per_system"],
            dt=s["dt"],
            velocities_out=vel_out_single,
            batch_idx=None,
            num_atoms_per_system=s["num_atoms_per_system"],
            cells_inv=s["cells_inv"],
            mode=mode,
            device=device,
        )

        vel_out_batch = wp.empty_like(s["velocities"])
        nph_velocity_half_step_out(
            s["velocities"],
            s["masses"],
            s["forces"],
            s["cell_velocities"],
            s["volumes"],
            s["num_atoms_per_system"],
            dt=s["dt"],
            velocities_out=vel_out_batch,
            batch_idx=s["batch_idx"],
            num_atoms_per_system=s["num_atoms_per_system"],
            cells_inv=s["cells_inv"],
            mode=mode,
            device=device,
        )
        wp.synchronize_device(device)

        np.testing.assert_allclose(
            vel_out_single.numpy(),
            vel_out_batch.numpy(),
            rtol=1e-5,
            atol=1e-7,
        )


# ==============================================================================
# Virial sign convention tests (library API only)
# ==============================================================================


class TestVirialSignConvention:
    """Verify compute_pressure_tensor passes virial through without negation.

    The library convention is W = -dE/dε (see conventions.md).  The pressure
    tensor kernel computes P = (KE_tensor + W) / V.  These tests confirm W
    is added directly — any internal negation would be caught.
    """

    @pytest.mark.parametrize("device", DEVICES)
    def test_pressure_tensor_virial_sign(self, device):
        """P = W/V when kinetic contribution is zero."""
        scalar_dtype = wp.float64
        vec_dtype = wp.vec3d
        vec9_dtype = vec9d
        mat_dtype = wp.mat33d

        # Cubic cell V = 1000
        cell_np = np.diag([10.0, 10.0, 10.0])
        cells = wp.array(
            [mat_dtype(*cell_np.flatten())], dtype=mat_dtype, device=device
        )
        volumes = wp.empty(1, dtype=scalar_dtype, device=device)
        compute_cell_volume(cells, volumes=volumes, device=device)

        # Known virial W = (10, 1, 2, 3, 20, 4, 5, 6, 30) row-major
        W = np.array([10.0, 1.0, 2.0, 3.0, 20.0, 4.0, 5.0, 6.0, 30.0])
        virial_tensors = wp.array(W.reshape(1, 9), dtype=vec9_dtype, device=device)

        # Zero velocities → kinetic tensor = 0
        num_atoms = 1
        velocities = wp.zeros(num_atoms, dtype=vec_dtype, device=device)
        masses = wp.array([1.0], dtype=scalar_dtype, device=device)

        kinetic_tensors = wp.zeros((1, 9), dtype=scalar_dtype, device=device)
        pressure_tensors = wp.empty(1, dtype=vec9_dtype, device=device)

        compute_pressure_tensor(
            velocities,
            masses,
            virial_tensors,
            cells,
            kinetic_tensors,
            pressure_tensors,
            volumes,
            device=device,
        )
        wp.synchronize_device(device)

        P = pressure_tensors.numpy()[0]
        V = 1000.0
        expected = W / V
        np.testing.assert_allclose(P, expected, rtol=1e-12)

    @pytest.mark.parametrize("device", DEVICES)
    def test_pressure_tensor_with_kinetic(self, device):
        """P = (KE_tensor + W) / V with both contributions non-zero."""
        scalar_dtype = wp.float64
        vec_dtype = wp.vec3d
        vec9_dtype = vec9d
        mat_dtype = wp.mat33d

        cell_np = np.diag([10.0, 10.0, 10.0])
        cells = wp.array(
            [mat_dtype(*cell_np.flatten())], dtype=mat_dtype, device=device
        )
        volumes = wp.empty(1, dtype=scalar_dtype, device=device)
        compute_cell_volume(cells, volumes=volumes, device=device)

        W = np.array([10.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 30.0])
        virial_tensors = wp.array(W.reshape(1, 9), dtype=vec9_dtype, device=device)

        # Single atom with velocity (1, 2, 3), mass 2
        # KE_tensor_ij = m * v_i * v_j
        # KE = 2 * [[1,2,3],[2,4,6],[3,6,9]] row-major
        velocities = wp.array([[1.0, 2.0, 3.0]], dtype=vec_dtype, device=device)
        masses = wp.array([2.0], dtype=scalar_dtype, device=device)

        kinetic_tensors = wp.zeros((1, 9), dtype=scalar_dtype, device=device)
        pressure_tensors = wp.empty(1, dtype=vec9_dtype, device=device)

        compute_pressure_tensor(
            velocities,
            masses,
            virial_tensors,
            cells,
            kinetic_tensors,
            pressure_tensors,
            volumes,
            device=device,
        )
        wp.synchronize_device(device)

        P = pressure_tensors.numpy()[0]
        V = 1000.0
        m = 2.0
        v = np.array([1.0, 2.0, 3.0])
        KE = m * np.outer(v, v).flatten()
        expected = (KE + W) / V
        np.testing.assert_allclose(P, expected, rtol=1e-10)


# ==============================================================================
# Non-cubic cells_inv integration tests for drag
# ==============================================================================


class TestNonCubicDrag:
    """Verify drag uses ε̇ = ḣ h⁻¹ (not ḣ/V) with orthorhombic cells."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_isotropic_drag_non_cubic(self, device):
        """Isotropic velocity half-step with orthorhombic cell h = diag(10,12,8).

        With ḣ = diag(0.1, 0.12, 0.08), we get ε̇ = ḣ h⁻¹ = diag(0.01, 0.01, 0.01)
        so Tr(ε̇)/3 = 0.01.  The naive formula Tr(ḣ)/(3V) = 0.3/(3*960) ≈ 1.04e-4
        gives a very different drag.  Large dt amplifies the difference.
        """
        np_dtype = np.float64
        scalar_dtype = wp.float64
        mat_dtype = wp.mat33d
        vec_dtype = wp.vec3d

        num_atoms = 1
        dt_val = 1.0

        # Orthorhombic cell
        h_np = np.diag([10.0, 12.0, 8.0]).astype(np_dtype)
        h_inv_np = np.linalg.inv(h_np).astype(np_dtype)

        cells_inv = wp.array(
            [mat_dtype(*h_inv_np.flatten())], dtype=mat_dtype, device=device
        )

        # ḣ chosen so ε̇ = ḣ h⁻¹ = 0.01 * I (uniform strain rate)
        h_dot_np = np.diag([0.1, 0.12, 0.08]).astype(np_dtype)
        cell_velocities = wp.array(
            [mat_dtype(*h_dot_np.flatten())], dtype=mat_dtype, device=device
        )

        # Single atom: v = (1,1,1), m = 1, F = 0, η̇₁ = 0
        v_np = np.array([[1.0, 1.0, 1.0]], dtype=np_dtype)
        velocities = wp.array(v_np, dtype=vec_dtype, device=device)
        masses = wp.array([1.0], dtype=scalar_dtype, device=device)
        forces = wp.zeros(num_atoms, dtype=vec_dtype, device=device)

        V = float(np.linalg.det(h_np))
        volumes = wp.array([V], dtype=scalar_dtype, device=device)
        eta_dot = wp.zeros((1, 1), dtype=scalar_dtype, device=device)
        num_atoms_arr = wp.array([num_atoms], dtype=wp.int32, device=device)
        dt = wp.array([dt_val], dtype=scalar_dtype, device=device)

        vel_out = wp.empty(num_atoms, dtype=vec_dtype, device=device)
        npt_velocity_half_step_out(
            velocities,
            masses,
            forces,
            cell_velocities,
            volumes,
            eta_dot,
            num_atoms_arr,
            dt,
            vel_out,
            cells_inv=cells_inv,
            mode="isotropic",
            device=device,
        )
        wp.synchronize_device(device)

        # Expected: v_new = v + dt/2 * (F/m - drag)
        # drag = coupling * Tr(ε̇)/3 * v  (η̇₁ = 0)
        # coupling = 1 + 1/N_f = 1 + 1/3 = 4/3
        # Tr(ε̇)/3 = 0.01
        # drag = (4/3) * 0.01 * v = 0.01333... * v
        eps_dot = h_dot_np @ h_inv_np
        trace_eps_3 = np.trace(eps_dot) / 3.0
        N_f = 3.0 * num_atoms
        coupling = 1.0 + 1.0 / N_f
        drag = coupling * trace_eps_3 * v_np
        expected = v_np + (dt_val / 2.0) * (0.0 - drag)

        actual = vel_out.numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-10)

        # Also verify this differs from the naive Tr(ḣ)/(3V) formula
        V = np.linalg.det(h_np)
        naive_eps = (h_dot_np[0, 0] + h_dot_np[1, 1] + h_dot_np[2, 2]) / (3.0 * V)
        naive_drag = coupling * naive_eps * v_np
        naive_expected = v_np + (dt_val / 2.0) * (0.0 - naive_drag)
        assert not np.allclose(actual, naive_expected, rtol=1e-5), (
            "Test is degenerate: correct and naive formulas give the same result"
        )

    @pytest.mark.parametrize("device", DEVICES)
    def test_anisotropic_drag_non_cubic(self, device):
        """Anisotropic velocity half-step with non-uniform strain rates.

        h = diag(10, 12, 8), ḣ = diag(0.2, 0.12, 0.16)
        → ε̇ = diag(0.02, 0.01, 0.02) — different per-axis strain rates.
        Large dt amplifies the difference vs the naive ḣ/V formula.
        """
        np_dtype = np.float64
        scalar_dtype = wp.float64
        mat_dtype = wp.mat33d
        vec_dtype = wp.vec3d

        num_atoms = 1
        dt_val = 1.0

        h_np = np.diag([10.0, 12.0, 8.0]).astype(np_dtype)
        h_inv_np = np.linalg.inv(h_np).astype(np_dtype)
        cells_inv = wp.array(
            [mat_dtype(*h_inv_np.flatten())], dtype=mat_dtype, device=device
        )

        # Non-uniform ḣ → different ε̇ per axis
        h_dot_np = np.diag([0.2, 0.12, 0.16]).astype(np_dtype)
        cell_velocities = wp.array(
            [mat_dtype(*h_dot_np.flatten())], dtype=mat_dtype, device=device
        )

        v_np = np.array([[1.0, 2.0, 3.0]], dtype=np_dtype)
        velocities = wp.array(v_np, dtype=vec_dtype, device=device)
        masses = wp.array([1.0], dtype=scalar_dtype, device=device)
        forces = wp.zeros(num_atoms, dtype=vec_dtype, device=device)

        V = float(np.linalg.det(h_np))
        volumes = wp.array([V], dtype=scalar_dtype, device=device)
        eta_dot = wp.zeros((1, 1), dtype=scalar_dtype, device=device)
        num_atoms_arr = wp.array([num_atoms], dtype=wp.int32, device=device)
        dt = wp.array([dt_val], dtype=scalar_dtype, device=device)

        vel_out = wp.empty(num_atoms, dtype=vec_dtype, device=device)
        npt_velocity_half_step_out(
            velocities,
            masses,
            forces,
            cell_velocities,
            volumes,
            eta_dot,
            num_atoms_arr,
            dt,
            vel_out,
            cells_inv=cells_inv,
            mode="anisotropic",
            device=device,
        )
        wp.synchronize_device(device)

        # drag_i = (coupling * ε̇_ii + η̇₁) * v_i
        eps_dot = h_dot_np @ h_inv_np
        N_f = 3.0 * num_atoms
        coupling = 1.0 + 1.0 / N_f
        drag = np.array(
            [
                [
                    (coupling * eps_dot[0, 0]) * v_np[0, 0],
                    (coupling * eps_dot[1, 1]) * v_np[0, 1],
                    (coupling * eps_dot[2, 2]) * v_np[0, 2],
                ]
            ]
        )
        expected = v_np + (dt_val / 2.0) * (0.0 - drag)

        actual = vel_out.numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-10)

        # Naive formula: ε̇_ii = ḣ_ii / V gives uniform rate, losing axis info
        V = np.linalg.det(h_np)
        naive_drag = np.array(
            [
                [
                    (coupling * h_dot_np[0, 0] / V) * v_np[0, 0],
                    (coupling * h_dot_np[1, 1] / V) * v_np[0, 1],
                    (coupling * h_dot_np[2, 2] / V) * v_np[0, 2],
                ]
            ]
        )
        naive_expected = v_np + (dt_val / 2.0) * (0.0 - naive_drag)
        assert not np.allclose(actual, naive_expected, rtol=1e-5), (
            "Test is degenerate: correct and naive formulas give the same result"
        )
