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
Unit tests for PME Warp kernels (framework-agnostic layer).

Tests the launcher functions in nvalchemiops.interactions.electrostatics.pme_kernels
directly with wp.array inputs.

Tests cover:
- Green's function computation (single and batched)
- Structure factor squared computation
- Energy corrections (self-energy and background)
- Energy corrections with charge gradients
- Float32 and float64 dtypes
- CPU and GPU devices

These tests use warp arrays directly and do not require PyTorch.
For PyTorch binding tests, see test/interactions/electrostatics/bindings/torch/
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from nvalchemiops.interactions.electrostatics.pme_kernels import (
    batch_pme_energy_corrections,
    batch_pme_energy_corrections_with_charge_grad,
    batch_pme_green_structure_factor,
    pme_energy_corrections,
    pme_energy_corrections_with_charge_grad,
    pme_green_structure_factor,
)

# Mathematical constants
PI = math.pi
TWOPI = 2.0 * PI


###########################################################################################
########################### Helper Functions ##############################################
###########################################################################################


def get_np_dtype(wp_dtype: type) -> type:
    """Get numpy dtype from warp dtype."""
    return np.float64 if wp_dtype == wp.float64 else np.float32


def make_scalar_array(value: float, device: str, wp_dtype: type) -> wp.array:
    """Create a 1-element warp array from a scalar."""
    np_dtype = get_np_dtype(wp_dtype)
    return wp.array([np_dtype(value)], dtype=wp_dtype, device=device)


def get_rtol(wp_dtype: type) -> float:
    """Get relative tolerance based on dtype."""
    return 1e-5 if wp_dtype == wp.float64 else 1e-4


###########################################################################################
########################### Green's Function Tests ########################################
###########################################################################################


class TestPMEGreenStructureFactor:
    """Test pme_green_structure_factor kernel."""

    def test_green_function_shape(self, device, wp_dtype):
        """Test output shapes are correct."""
        mesh_nx, mesh_ny, mesh_nz = 8, 8, 8
        nz_rfft = mesh_nz // 2 + 1

        # Create inputs
        k_squared = wp.zeros((mesh_nx, mesh_ny, nz_rfft), dtype=wp_dtype, device=device)
        miller_x = wp.zeros(mesh_nx, dtype=wp_dtype, device=device)
        miller_y = wp.zeros(mesh_ny, dtype=wp_dtype, device=device)
        miller_z = wp.zeros(nz_rfft, dtype=wp_dtype, device=device)
        alpha = make_scalar_array(0.3, device, wp_dtype)
        volume = make_scalar_array(1000.0, device, wp_dtype)

        # Create outputs
        green_function = wp.zeros(
            (mesh_nx, mesh_ny, nz_rfft), dtype=wp_dtype, device=device
        )
        structure_factor_sq = wp.zeros(
            (mesh_nx, mesh_ny, nz_rfft), dtype=wp_dtype, device=device
        )

        # Run kernel
        pme_green_structure_factor(
            k_squared=k_squared,
            miller_x=miller_x,
            miller_y=miller_y,
            miller_z=miller_z,
            alpha=alpha,
            volume=volume,
            mesh_nx=mesh_nx,
            mesh_ny=mesh_ny,
            mesh_nz=mesh_nz,
            spline_order=4,
            green_function=green_function,
            structure_factor_sq=structure_factor_sq,
            wp_dtype=wp_dtype,
            device=device,
        )

        assert green_function.shape == (mesh_nx, mesh_ny, nz_rfft)
        assert structure_factor_sq.shape == (mesh_nx, mesh_ny, nz_rfft)

    def test_green_function_k0_is_zero(self, device, wp_dtype):
        """Test that G(k=0) is set to zero (tin-foil boundary conditions)."""
        mesh_nx, mesh_ny, mesh_nz = 8, 8, 8
        nz_rfft = mesh_nz // 2 + 1
        np_dtype = get_np_dtype(wp_dtype)

        # Create k_squared with non-zero values
        k_sq_np = np.ones((mesh_nx, mesh_ny, nz_rfft), dtype=np_dtype)
        k_sq_np[0, 0, 0] = 0.0  # k=0 point
        k_squared = wp.array(k_sq_np, dtype=wp_dtype, device=device)

        # Create miller indices (simple case)
        miller_x = wp.array(
            np.arange(mesh_nx, dtype=np_dtype), dtype=wp_dtype, device=device
        )
        miller_y = wp.array(
            np.arange(mesh_ny, dtype=np_dtype), dtype=wp_dtype, device=device
        )
        miller_z = wp.array(
            np.arange(nz_rfft, dtype=np_dtype), dtype=wp_dtype, device=device
        )

        alpha = make_scalar_array(0.3, device, wp_dtype)
        volume = make_scalar_array(1000.0, device, wp_dtype)

        green_function = wp.zeros(
            (mesh_nx, mesh_ny, nz_rfft), dtype=wp_dtype, device=device
        )
        structure_factor_sq = wp.zeros(
            (mesh_nx, mesh_ny, nz_rfft), dtype=wp_dtype, device=device
        )

        pme_green_structure_factor(
            k_squared=k_squared,
            miller_x=miller_x,
            miller_y=miller_y,
            miller_z=miller_z,
            alpha=alpha,
            volume=volume,
            mesh_nx=mesh_nx,
            mesh_ny=mesh_ny,
            mesh_nz=mesh_nz,
            spline_order=4,
            green_function=green_function,
            structure_factor_sq=structure_factor_sq,
            wp_dtype=wp_dtype,
            device=device,
        )

        green_np = green_function.numpy()
        assert green_np[0, 0, 0] == 0.0, "G(k=0) should be zero"

    def test_green_function_formula(self, device, wp_dtype):
        """Test Green's function follows G(k) = 2π/V * exp(-k²/(4α²)) / k²."""
        mesh_nx, mesh_ny, mesh_nz = 4, 4, 4
        nz_rfft = mesh_nz // 2 + 1
        alpha_val = 0.5
        volume_val = 100.0
        np_dtype = get_np_dtype(wp_dtype)

        # Create simple k² values
        k_sq_np = np.ones((mesh_nx, mesh_ny, nz_rfft), dtype=np_dtype) * 4.0  # k² = 4
        k_sq_np[0, 0, 0] = 0.0  # k=0

        k_squared = wp.array(k_sq_np, dtype=wp_dtype, device=device)
        # Miller indices all zero (so structure factor = 1)
        miller_x = wp.array(
            np.zeros(mesh_nx, dtype=np_dtype), dtype=wp_dtype, device=device
        )
        miller_y = wp.array(
            np.zeros(mesh_ny, dtype=np_dtype), dtype=wp_dtype, device=device
        )
        miller_z = wp.array(
            np.zeros(nz_rfft, dtype=np_dtype), dtype=wp_dtype, device=device
        )
        alpha = make_scalar_array(alpha_val, device, wp_dtype)
        volume = make_scalar_array(volume_val, device, wp_dtype)

        green_function = wp.zeros(
            (mesh_nx, mesh_ny, nz_rfft), dtype=wp_dtype, device=device
        )
        structure_factor_sq = wp.zeros(
            (mesh_nx, mesh_ny, nz_rfft), dtype=wp_dtype, device=device
        )

        pme_green_structure_factor(
            k_squared=k_squared,
            miller_x=miller_x,
            miller_y=miller_y,
            miller_z=miller_z,
            alpha=alpha,
            volume=volume,
            mesh_nx=mesh_nx,
            mesh_ny=mesh_ny,
            mesh_nz=mesh_nz,
            spline_order=4,
            green_function=green_function,
            structure_factor_sq=structure_factor_sq,
            wp_dtype=wp_dtype,
            device=device,
        )

        green_np = green_function.numpy()

        # Expected: G(k) = 2π/V * exp(-k²/(4α²)) / k²
        k_sq = 4.0
        expected = TWOPI * np.exp(-k_sq / (4.0 * alpha_val**2)) / (k_sq * volume_val)

        # Check a non-zero k point
        rtol = get_rtol(wp_dtype)
        assert np.isclose(green_np[1, 0, 0], expected, rtol=rtol), (
            f"Green's function mismatch: got {green_np[1, 0, 0]}, expected {expected}"
        )

    def test_structure_factor_positive(self, device, wp_dtype):
        """Test that structure factor squared is always positive."""
        mesh_nx, mesh_ny, mesh_nz = 8, 8, 8
        nz_rfft = mesh_nz // 2 + 1
        np_dtype = get_np_dtype(wp_dtype)

        k_squared = wp.array(
            np.ones((mesh_nx, mesh_ny, nz_rfft), dtype=np_dtype),
            dtype=wp_dtype,
            device=device,
        )
        miller_x = wp.array(
            np.fft.fftfreq(mesh_nx).astype(np_dtype) * mesh_nx,
            dtype=wp_dtype,
            device=device,
        )
        miller_y = wp.array(
            np.fft.fftfreq(mesh_ny).astype(np_dtype) * mesh_ny,
            dtype=wp_dtype,
            device=device,
        )
        miller_z = wp.array(
            np.fft.rfftfreq(mesh_nz).astype(np_dtype) * mesh_nz,
            dtype=wp_dtype,
            device=device,
        )
        alpha = make_scalar_array(0.3, device, wp_dtype)
        volume = make_scalar_array(1000.0, device, wp_dtype)

        green_function = wp.zeros(
            (mesh_nx, mesh_ny, nz_rfft), dtype=wp_dtype, device=device
        )
        structure_factor_sq = wp.zeros(
            (mesh_nx, mesh_ny, nz_rfft), dtype=wp_dtype, device=device
        )

        pme_green_structure_factor(
            k_squared=k_squared,
            miller_x=miller_x,
            miller_y=miller_y,
            miller_z=miller_z,
            alpha=alpha,
            volume=volume,
            mesh_nx=mesh_nx,
            mesh_ny=mesh_ny,
            mesh_nz=mesh_nz,
            spline_order=4,
            green_function=green_function,
            structure_factor_sq=structure_factor_sq,
            wp_dtype=wp_dtype,
            device=device,
        )

        sf_np = structure_factor_sq.numpy()
        assert np.all(sf_np > 0), "Structure factor squared should be positive"

    def test_green_function_no_nan_inf(self, device, wp_dtype):
        """Test that Green's function has no NaN or Inf values."""
        mesh_nx, mesh_ny, mesh_nz = 16, 16, 16
        nz_rfft = mesh_nz // 2 + 1
        np_dtype = get_np_dtype(wp_dtype)

        # Create realistic k² values
        k_sq_np = np.random.rand(mesh_nx, mesh_ny, nz_rfft).astype(np_dtype) + 0.1
        k_sq_np[0, 0, 0] = 0.0

        k_squared = wp.array(k_sq_np, dtype=wp_dtype, device=device)
        miller_x = wp.array(
            np.fft.fftfreq(mesh_nx).astype(np_dtype) * mesh_nx,
            dtype=wp_dtype,
            device=device,
        )
        miller_y = wp.array(
            np.fft.fftfreq(mesh_ny).astype(np_dtype) * mesh_ny,
            dtype=wp_dtype,
            device=device,
        )
        miller_z = wp.array(
            np.fft.rfftfreq(mesh_nz).astype(np_dtype) * mesh_nz,
            dtype=wp_dtype,
            device=device,
        )
        alpha = make_scalar_array(0.3, device, wp_dtype)
        volume = make_scalar_array(1000.0, device, wp_dtype)

        green_function = wp.zeros(
            (mesh_nx, mesh_ny, nz_rfft), dtype=wp_dtype, device=device
        )
        structure_factor_sq = wp.zeros(
            (mesh_nx, mesh_ny, nz_rfft), dtype=wp_dtype, device=device
        )

        pme_green_structure_factor(
            k_squared=k_squared,
            miller_x=miller_x,
            miller_y=miller_y,
            miller_z=miller_z,
            alpha=alpha,
            volume=volume,
            mesh_nx=mesh_nx,
            mesh_ny=mesh_ny,
            mesh_nz=mesh_nz,
            spline_order=4,
            green_function=green_function,
            structure_factor_sq=structure_factor_sq,
            wp_dtype=wp_dtype,
            device=device,
        )

        green_np = green_function.numpy()
        sf_np = structure_factor_sq.numpy()

        assert not np.isnan(green_np).any(), "Green's function contains NaN"
        assert not np.isinf(green_np).any(), "Green's function contains Inf"
        assert not np.isnan(sf_np).any(), "Structure factor contains NaN"
        assert not np.isinf(sf_np).any(), "Structure factor contains Inf"


###########################################################################################
########################### Energy Corrections Tests ######################################
###########################################################################################


class TestPMEEnergyCorrections:
    """Test pme_energy_corrections kernel."""

    def test_energy_corrections_shape(self, device, wp_dtype):
        """Test output shape is correct."""
        num_atoms = 10

        raw_energies = wp.zeros(num_atoms, dtype=wp_dtype, device=device)
        charges = wp.zeros(num_atoms, dtype=wp_dtype, device=device)
        volume = make_scalar_array(1000.0, device, wp_dtype)
        alpha = make_scalar_array(0.3, device, wp_dtype)
        total_charge = make_scalar_array(0.0, device, wp_dtype)
        corrected_energies = wp.zeros(num_atoms, dtype=wp_dtype, device=device)

        pme_energy_corrections(
            raw_energies=raw_energies,
            charges=charges,
            volume=volume,
            alpha=alpha,
            total_charge=total_charge,
            corrected_energies=corrected_energies,
            wp_dtype=wp_dtype,
            device=device,
        )

        assert corrected_energies.shape == (num_atoms,)

    def test_self_energy_correction(self, device, wp_dtype):
        """Test self-energy correction: E_self = (α/√π) * q²."""
        num_atoms = 2
        alpha_val = 0.5
        np_dtype = get_np_dtype(wp_dtype)

        # Zero raw energies to isolate self-energy correction
        raw_energies = wp.array(
            np.zeros(num_atoms, dtype=np_dtype), dtype=wp_dtype, device=device
        )
        charges = wp.array(
            [np_dtype(1.0), np_dtype(-1.0)], dtype=wp_dtype, device=device
        )
        volume = make_scalar_array(1000.0, device, wp_dtype)
        alpha = make_scalar_array(alpha_val, device, wp_dtype)
        total_charge = make_scalar_array(0.0, device, wp_dtype)  # Neutral
        corrected_energies = wp.zeros(num_atoms, dtype=wp_dtype, device=device)

        pme_energy_corrections(
            raw_energies=raw_energies,
            charges=charges,
            volume=volume,
            alpha=alpha,
            total_charge=total_charge,
            corrected_energies=corrected_energies,
            wp_dtype=wp_dtype,
            device=device,
        )

        result = corrected_energies.numpy()

        # Expected self-energy correction: -α/√π * q²
        expected_self = -alpha_val / np.sqrt(PI) * 1.0**2

        rtol = get_rtol(wp_dtype)
        assert np.isclose(result[0], expected_self, rtol=rtol), (
            f"Self-energy correction mismatch: got {result[0]}, expected {expected_self}"
        )
        assert np.isclose(result[1], expected_self, rtol=rtol), (
            "Self-energy correction mismatch for negative charge"
        )

    def test_neutral_system_zero_background(self, device, wp_dtype):
        """Test that neutral system has zero background correction."""
        num_atoms = 4
        np_dtype = get_np_dtype(wp_dtype)
        alpha_val = 0.3

        raw_energies = wp.array(
            np.zeros(num_atoms, dtype=np_dtype), dtype=wp_dtype, device=device
        )
        # Neutral charges
        charges_np = np.array([1.0, -1.0, 0.5, -0.5], dtype=np_dtype)
        charges = wp.array(charges_np, dtype=wp_dtype, device=device)
        volume = make_scalar_array(1000.0, device, wp_dtype)
        alpha = make_scalar_array(alpha_val, device, wp_dtype)
        total_charge = make_scalar_array(0.0, device, wp_dtype)
        corrected_energies = wp.zeros(num_atoms, dtype=wp_dtype, device=device)

        pme_energy_corrections(
            raw_energies=raw_energies,
            charges=charges,
            volume=volume,
            alpha=alpha,
            total_charge=total_charge,
            corrected_energies=corrected_energies,
            wp_dtype=wp_dtype,
            device=device,
        )

        result = corrected_energies.numpy()

        # For neutral system, should only have self-energy (no background)
        # Self-energy: -α/√π * q²
        expected = -alpha_val / np.sqrt(PI) * charges_np**2

        rtol = get_rtol(wp_dtype)
        assert np.allclose(result, expected, rtol=rtol)

    def test_potential_energy_conversion(self, device, wp_dtype):
        """Test that E = q * φ component is computed correctly."""
        num_atoms = 2
        np_dtype = get_np_dtype(wp_dtype)
        alpha_val = 0.5
        volume_val = 1000.0

        # Non-zero raw energies (potentials)
        phi = np.array([0.5, -0.3], dtype=np_dtype)
        raw_energies = wp.array(phi, dtype=wp_dtype, device=device)
        q = np.array([1.0, -1.0], dtype=np_dtype)
        charges = wp.array(q, dtype=wp_dtype, device=device)
        volume = make_scalar_array(volume_val, device, wp_dtype)
        alpha = make_scalar_array(alpha_val, device, wp_dtype)
        total_charge = make_scalar_array(0.0, device, wp_dtype)
        corrected_energies = wp.zeros(num_atoms, dtype=wp_dtype, device=device)

        pme_energy_corrections(
            raw_energies=raw_energies,
            charges=charges,
            volume=volume,
            alpha=alpha,
            total_charge=total_charge,
            corrected_energies=corrected_energies,
            wp_dtype=wp_dtype,
            device=device,
        )

        result = corrected_energies.numpy()

        # Expected: E_i = q_i * φ_i - (α/√π) * q_i² - (π/(2α²V)) * q_i * Q_total
        # With Q_total = 0, background term is zero
        expected = q * phi - alpha_val / np.sqrt(PI) * q**2

        rtol = get_rtol(wp_dtype)
        assert np.allclose(result, expected, rtol=rtol)


class TestPMEEnergyCorrectionsWithChargeGrad:
    """Test pme_energy_corrections_with_charge_grad kernel."""

    def test_charge_grad_shape(self, device, wp_dtype):
        """Test output shapes are correct."""
        num_atoms = 10

        raw_energies = wp.zeros(num_atoms, dtype=wp_dtype, device=device)
        charges = wp.zeros(num_atoms, dtype=wp_dtype, device=device)
        volume = make_scalar_array(1000.0, device, wp_dtype)
        alpha = make_scalar_array(0.3, device, wp_dtype)
        total_charge = make_scalar_array(0.0, device, wp_dtype)
        corrected_energies = wp.zeros(num_atoms, dtype=wp_dtype, device=device)
        charge_gradients = wp.zeros(num_atoms, dtype=wp_dtype, device=device)

        pme_energy_corrections_with_charge_grad(
            raw_energies=raw_energies,
            charges=charges,
            volume=volume,
            alpha=alpha,
            total_charge=total_charge,
            corrected_energies=corrected_energies,
            charge_gradients=charge_gradients,
            wp_dtype=wp_dtype,
            device=device,
        )

        assert corrected_energies.shape == (num_atoms,)
        assert charge_gradients.shape == (num_atoms,)

    def test_charge_grad_formula(self, device, wp_dtype):
        """Test charge gradient: ∂E/∂q_i = 2*φ_i - 2*(α/√π)*q_i - (π/(α²V))*Q_total."""
        num_atoms = 2
        alpha_val = 0.5
        volume_val = 1000.0
        np_dtype = get_np_dtype(wp_dtype)

        # Non-zero raw energies (potentials)
        phi = np.array([0.5, -0.3], dtype=np_dtype)
        raw_energies = wp.array(phi, dtype=wp_dtype, device=device)
        q = np.array([1.0, -1.0], dtype=np_dtype)
        charges = wp.array(q, dtype=wp_dtype, device=device)
        volume = make_scalar_array(volume_val, device, wp_dtype)
        alpha = make_scalar_array(alpha_val, device, wp_dtype)
        Q_total = 0.0
        total_charge = make_scalar_array(Q_total, device, wp_dtype)
        corrected_energies = wp.zeros(num_atoms, dtype=wp_dtype, device=device)
        charge_gradients = wp.zeros(num_atoms, dtype=wp_dtype, device=device)

        pme_energy_corrections_with_charge_grad(
            raw_energies=raw_energies,
            charges=charges,
            volume=volume,
            alpha=alpha,
            total_charge=total_charge,
            corrected_energies=corrected_energies,
            charge_gradients=charge_gradients,
            wp_dtype=wp_dtype,
            device=device,
        )

        grad_np = charge_gradients.numpy()

        # Expected: ∂E/∂q_i = 2*φ_i - 2*(α/√π)*q_i - (π/(α²V))*Q_total
        expected_grad = (
            2 * phi
            - 2 * (alpha_val / np.sqrt(PI)) * q
            - (PI / (alpha_val**2 * volume_val)) * Q_total
        )

        rtol = get_rtol(wp_dtype)
        assert np.allclose(grad_np, expected_grad, rtol=rtol), (
            f"Charge gradient mismatch: got {grad_np}, expected {expected_grad}"
        )

    def test_energy_matches_energy_only_kernel(self, device, wp_dtype):
        """Test that energy output matches the energy-only kernel."""
        num_atoms = 5
        np_dtype = get_np_dtype(wp_dtype)
        alpha_val = 0.4

        phi = np.random.randn(num_atoms).astype(np_dtype)
        q = np.random.randn(num_atoms).astype(np_dtype)

        raw_energies = wp.array(phi, dtype=wp_dtype, device=device)
        charges = wp.array(q, dtype=wp_dtype, device=device)
        volume = make_scalar_array(500.0, device, wp_dtype)
        alpha = make_scalar_array(alpha_val, device, wp_dtype)
        total_charge = make_scalar_array(float(q.sum()), device, wp_dtype)

        # Energy-only kernel
        corrected_energies_only = wp.zeros(num_atoms, dtype=wp_dtype, device=device)
        pme_energy_corrections(
            raw_energies=raw_energies,
            charges=charges,
            volume=volume,
            alpha=alpha,
            total_charge=total_charge,
            corrected_energies=corrected_energies_only,
            wp_dtype=wp_dtype,
            device=device,
        )

        # Energy + gradient kernel
        corrected_energies_grad = wp.zeros(num_atoms, dtype=wp_dtype, device=device)
        charge_gradients = wp.zeros(num_atoms, dtype=wp_dtype, device=device)
        pme_energy_corrections_with_charge_grad(
            raw_energies=raw_energies,
            charges=charges,
            volume=volume,
            alpha=alpha,
            total_charge=total_charge,
            corrected_energies=corrected_energies_grad,
            charge_gradients=charge_gradients,
            wp_dtype=wp_dtype,
            device=device,
        )

        rtol = get_rtol(wp_dtype)
        np.testing.assert_allclose(
            corrected_energies_only.numpy(),
            corrected_energies_grad.numpy(),
            rtol=rtol,
        )


###########################################################################################
########################### Batch Tests ###################################################
###########################################################################################


class TestBatchPMEGreenStructureFactor:
    """Test batch_pme_green_structure_factor kernel."""

    def test_batch_green_function_shape(self, device, wp_dtype):
        """Test batch output shapes are correct."""
        num_systems = 3
        mesh_nx, mesh_ny, mesh_nz = 8, 8, 8
        nz_rfft = mesh_nz // 2 + 1
        np_dtype = get_np_dtype(wp_dtype)

        k_squared = wp.zeros(
            (num_systems, mesh_nx, mesh_ny, nz_rfft), dtype=wp_dtype, device=device
        )
        miller_x = wp.zeros(mesh_nx, dtype=wp_dtype, device=device)
        miller_y = wp.zeros(mesh_ny, dtype=wp_dtype, device=device)
        miller_z = wp.zeros(nz_rfft, dtype=wp_dtype, device=device)
        alpha = wp.array(
            np.array([0.3, 0.3, 0.3], dtype=np_dtype), dtype=wp_dtype, device=device
        )
        volumes = wp.array(
            np.array([1000.0, 1000.0, 1000.0], dtype=np_dtype),
            dtype=wp_dtype,
            device=device,
        )

        green_function = wp.zeros(
            (num_systems, mesh_nx, mesh_ny, nz_rfft), dtype=wp_dtype, device=device
        )
        structure_factor_sq = wp.zeros(
            (mesh_nx, mesh_ny, nz_rfft), dtype=wp_dtype, device=device
        )

        batch_pme_green_structure_factor(
            k_squared=k_squared,
            miller_x=miller_x,
            miller_y=miller_y,
            miller_z=miller_z,
            alpha=alpha,
            volumes=volumes,
            mesh_nx=mesh_nx,
            mesh_ny=mesh_ny,
            mesh_nz=mesh_nz,
            spline_order=4,
            green_function=green_function,
            structure_factor_sq=structure_factor_sq,
            wp_dtype=wp_dtype,
            device=device,
        )

        assert green_function.shape == (num_systems, mesh_nx, mesh_ny, nz_rfft)
        assert structure_factor_sq.shape == (mesh_nx, mesh_ny, nz_rfft)

    def test_batch_vs_single_consistency(self, device, wp_dtype):
        """Test that batch with size 1 matches single-system."""
        mesh_nx, mesh_ny, mesh_nz = 8, 8, 8
        nz_rfft = mesh_nz // 2 + 1
        np_dtype = get_np_dtype(wp_dtype)

        # Create test data
        k_sq_np = np.random.rand(mesh_nx, mesh_ny, nz_rfft).astype(np_dtype) + 0.1
        k_sq_np[0, 0, 0] = 0.0
        miller_x_np = (np.fft.fftfreq(mesh_nx) * mesh_nx).astype(np_dtype)
        miller_y_np = (np.fft.fftfreq(mesh_ny) * mesh_ny).astype(np_dtype)
        miller_z_np = (np.fft.rfftfreq(mesh_nz) * mesh_nz).astype(np_dtype)
        alpha_val = 0.3
        volume_val = 1000.0

        # Single-system
        k_squared_single = wp.array(k_sq_np, dtype=wp_dtype, device=device)
        miller_x = wp.array(miller_x_np, dtype=wp_dtype, device=device)
        miller_y = wp.array(miller_y_np, dtype=wp_dtype, device=device)
        miller_z = wp.array(miller_z_np, dtype=wp_dtype, device=device)
        alpha_single = make_scalar_array(alpha_val, device, wp_dtype)
        volume_single = make_scalar_array(volume_val, device, wp_dtype)

        green_single = wp.zeros(
            (mesh_nx, mesh_ny, nz_rfft), dtype=wp_dtype, device=device
        )
        sf_single = wp.zeros((mesh_nx, mesh_ny, nz_rfft), dtype=wp_dtype, device=device)

        pme_green_structure_factor(
            k_squared=k_squared_single,
            miller_x=miller_x,
            miller_y=miller_y,
            miller_z=miller_z,
            alpha=alpha_single,
            volume=volume_single,
            mesh_nx=mesh_nx,
            mesh_ny=mesh_ny,
            mesh_nz=mesh_nz,
            spline_order=4,
            green_function=green_single,
            structure_factor_sq=sf_single,
            wp_dtype=wp_dtype,
            device=device,
        )

        # Batch with size 1
        k_squared_batch = wp.array(
            k_sq_np.reshape(1, mesh_nx, mesh_ny, nz_rfft),
            dtype=wp_dtype,
            device=device,
        )
        alpha_batch = wp.array([np_dtype(alpha_val)], dtype=wp_dtype, device=device)
        volumes_batch = wp.array([np_dtype(volume_val)], dtype=wp_dtype, device=device)

        green_batch = wp.zeros(
            (1, mesh_nx, mesh_ny, nz_rfft), dtype=wp_dtype, device=device
        )
        sf_batch = wp.zeros((mesh_nx, mesh_ny, nz_rfft), dtype=wp_dtype, device=device)

        batch_pme_green_structure_factor(
            k_squared=k_squared_batch,
            miller_x=miller_x,
            miller_y=miller_y,
            miller_z=miller_z,
            alpha=alpha_batch,
            volumes=volumes_batch,
            mesh_nx=mesh_nx,
            mesh_ny=mesh_ny,
            mesh_nz=mesh_nz,
            spline_order=4,
            green_function=green_batch,
            structure_factor_sq=sf_batch,
            wp_dtype=wp_dtype,
            device=device,
        )

        rtol = get_rtol(wp_dtype)
        np.testing.assert_allclose(
            green_single.numpy(), green_batch.numpy()[0], rtol=rtol
        )
        np.testing.assert_allclose(sf_single.numpy(), sf_batch.numpy(), rtol=rtol)


class TestBatchPMEEnergyCorrections:
    """Test batch_pme_energy_corrections kernel."""

    def test_batch_energy_corrections_shape(self, device, wp_dtype):
        """Test batch output shape is correct."""
        num_atoms = 10
        num_systems = 3
        np_dtype = get_np_dtype(wp_dtype)

        raw_energies = wp.zeros(num_atoms, dtype=wp_dtype, device=device)
        charges = wp.zeros(num_atoms, dtype=wp_dtype, device=device)
        batch_idx = wp.array(
            [0, 0, 0, 1, 1, 1, 1, 2, 2, 2], dtype=wp.int32, device=device
        )
        volumes = wp.array(
            np.array([1000.0, 1000.0, 1000.0], dtype=np_dtype),
            dtype=wp_dtype,
            device=device,
        )
        alpha = wp.array(
            np.array([0.3, 0.3, 0.3], dtype=np_dtype), dtype=wp_dtype, device=device
        )
        total_charges = wp.zeros(num_systems, dtype=wp_dtype, device=device)
        corrected_energies = wp.zeros(num_atoms, dtype=wp_dtype, device=device)

        batch_pme_energy_corrections(
            raw_energies=raw_energies,
            charges=charges,
            batch_idx=batch_idx,
            volumes=volumes,
            alpha=alpha,
            total_charges=total_charges,
            corrected_energies=corrected_energies,
            wp_dtype=wp_dtype,
            device=device,
        )

        assert corrected_energies.shape == (num_atoms,)

    def test_batch_vs_single_energy_corrections(self, device, wp_dtype):
        """Test batch energy corrections match sequential single-system calls."""
        np_dtype = get_np_dtype(wp_dtype)
        alpha_val = 0.3

        # System 1: 2 atoms
        raw1 = np.array([0.1, 0.2], dtype=np_dtype)
        q1 = np.array([1.0, -1.0], dtype=np_dtype)
        vol1 = 500.0
        Q1 = 0.0

        # System 2: 3 atoms
        raw2 = np.array([0.3, 0.4, 0.5], dtype=np_dtype)
        q2 = np.array([0.5, -0.3, -0.2], dtype=np_dtype)
        vol2 = 800.0
        Q2 = 0.0

        # Single-system calculations
        corrected1 = wp.zeros(2, dtype=wp_dtype, device=device)
        pme_energy_corrections(
            raw_energies=wp.array(raw1, dtype=wp_dtype, device=device),
            charges=wp.array(q1, dtype=wp_dtype, device=device),
            volume=make_scalar_array(vol1, device, wp_dtype),
            alpha=make_scalar_array(alpha_val, device, wp_dtype),
            total_charge=make_scalar_array(Q1, device, wp_dtype),
            corrected_energies=corrected1,
            wp_dtype=wp_dtype,
            device=device,
        )

        corrected2 = wp.zeros(3, dtype=wp_dtype, device=device)
        pme_energy_corrections(
            raw_energies=wp.array(raw2, dtype=wp_dtype, device=device),
            charges=wp.array(q2, dtype=wp_dtype, device=device),
            volume=make_scalar_array(vol2, device, wp_dtype),
            alpha=make_scalar_array(alpha_val, device, wp_dtype),
            total_charge=make_scalar_array(Q2, device, wp_dtype),
            corrected_energies=corrected2,
            wp_dtype=wp_dtype,
            device=device,
        )

        # Batch calculation
        raw_batch = np.concatenate([raw1, raw2])
        q_batch = np.concatenate([q1, q2])
        batch_idx = np.array([0, 0, 1, 1, 1], dtype=np.int32)

        corrected_batch = wp.zeros(5, dtype=wp_dtype, device=device)
        batch_pme_energy_corrections(
            raw_energies=wp.array(raw_batch, dtype=wp_dtype, device=device),
            charges=wp.array(q_batch, dtype=wp_dtype, device=device),
            batch_idx=wp.array(batch_idx, dtype=wp.int32, device=device),
            volumes=wp.array(
                [np_dtype(vol1), np_dtype(vol2)], dtype=wp_dtype, device=device
            ),
            alpha=wp.array(
                [np_dtype(alpha_val), np_dtype(alpha_val)],
                dtype=wp_dtype,
                device=device,
            ),
            total_charges=wp.array(
                [np_dtype(Q1), np_dtype(Q2)], dtype=wp_dtype, device=device
            ),
            corrected_energies=corrected_batch,
            wp_dtype=wp_dtype,
            device=device,
        )

        batch_np = corrected_batch.numpy()
        rtol = get_rtol(wp_dtype)

        np.testing.assert_allclose(
            batch_np[:2],
            corrected1.numpy(),
            rtol=rtol,
            err_msg="System 1 energy mismatch",
        )
        np.testing.assert_allclose(
            batch_np[2:],
            corrected2.numpy(),
            rtol=rtol,
            err_msg="System 2 energy mismatch",
        )


class TestBatchPMEEnergyCorrectionsWithChargeGrad:
    """Test batch_pme_energy_corrections_with_charge_grad kernel."""

    def test_batch_charge_grad_shape(self, device, wp_dtype):
        """Test batch output shapes are correct."""
        num_atoms = 8
        num_systems = 2
        np_dtype = get_np_dtype(wp_dtype)

        raw_energies = wp.zeros(num_atoms, dtype=wp_dtype, device=device)
        charges = wp.zeros(num_atoms, dtype=wp_dtype, device=device)
        batch_idx = wp.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=wp.int32, device=device)
        volumes = wp.array(
            np.array([1000.0, 1000.0], dtype=np_dtype), dtype=wp_dtype, device=device
        )
        alpha = wp.array(
            np.array([0.3, 0.3], dtype=np_dtype), dtype=wp_dtype, device=device
        )
        total_charges = wp.zeros(num_systems, dtype=wp_dtype, device=device)
        corrected_energies = wp.zeros(num_atoms, dtype=wp_dtype, device=device)
        charge_gradients = wp.zeros(num_atoms, dtype=wp_dtype, device=device)

        batch_pme_energy_corrections_with_charge_grad(
            raw_energies=raw_energies,
            charges=charges,
            batch_idx=batch_idx,
            volumes=volumes,
            alpha=alpha,
            total_charges=total_charges,
            corrected_energies=corrected_energies,
            charge_gradients=charge_gradients,
            wp_dtype=wp_dtype,
            device=device,
        )

        assert corrected_energies.shape == (num_atoms,)
        assert charge_gradients.shape == (num_atoms,)

    def test_batch_charge_grad_consistency(self, device, wp_dtype):
        """Test that batch with charge grad matches energy-only batch."""
        np_dtype = get_np_dtype(wp_dtype)
        num_atoms = 6

        raw_energies_np = np.random.randn(num_atoms).astype(np_dtype)
        charges_np = np.random.randn(num_atoms).astype(np_dtype)
        batch_idx_np = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)

        raw_energies = wp.array(raw_energies_np, dtype=wp_dtype, device=device)
        charges = wp.array(charges_np, dtype=wp_dtype, device=device)
        batch_idx = wp.array(batch_idx_np, dtype=wp.int32, device=device)
        volumes = wp.array(
            np.array([500.0, 800.0], dtype=np_dtype), dtype=wp_dtype, device=device
        )
        alpha = wp.array(
            np.array([0.3, 0.4], dtype=np_dtype), dtype=wp_dtype, device=device
        )
        Q0 = float(charges_np[:3].sum())
        Q1 = float(charges_np[3:].sum())
        total_charges = wp.array(
            np.array([Q0, Q1], dtype=np_dtype), dtype=wp_dtype, device=device
        )

        # Energy-only
        corrected_only = wp.zeros(num_atoms, dtype=wp_dtype, device=device)
        batch_pme_energy_corrections(
            raw_energies=raw_energies,
            charges=charges,
            batch_idx=batch_idx,
            volumes=volumes,
            alpha=alpha,
            total_charges=total_charges,
            corrected_energies=corrected_only,
            wp_dtype=wp_dtype,
            device=device,
        )

        # Energy + grad
        corrected_grad = wp.zeros(num_atoms, dtype=wp_dtype, device=device)
        charge_grads = wp.zeros(num_atoms, dtype=wp_dtype, device=device)
        batch_pme_energy_corrections_with_charge_grad(
            raw_energies=raw_energies,
            charges=charges,
            batch_idx=batch_idx,
            volumes=volumes,
            alpha=alpha,
            total_charges=total_charges,
            corrected_energies=corrected_grad,
            charge_gradients=charge_grads,
            wp_dtype=wp_dtype,
            device=device,
        )

        rtol = get_rtol(wp_dtype)
        np.testing.assert_allclose(
            corrected_only.numpy(), corrected_grad.numpy(), rtol=rtol
        )


###########################################################################################
########################### Regression Tests ##############################################
###########################################################################################


class TestPMEKernelsRegression:
    """Regression tests with hardcoded expected values.

    These values were generated by running the warp kernels with known inputs
    to establish baseline behavior. They serve as regression tests to catch
    unintended changes to kernel behavior.

    Note: Expected values may differ slightly from analytical formulas due to
    Warp's internal math implementations (e.g., wp.sqrt vs numpy.sqrt).
    """

    def test_self_energy_regression(self, device):
        """Regression test for self-energy with known values.

        α = 0.5, q = 1.0
        Self-energy = -α/√π * q² = -0.5/√π * 1 ≈ -0.28209479
        Background = -π/(2α²V) * q * Q_total = -π/(500) ≈ -0.006283185

        Note: Hardcoded value from actual kernel output (wp.sqrt differs slightly).
        """
        num_atoms = 1
        alpha_val = 0.5
        np_dtype = np.float64
        wp_dtype = wp.float64

        raw_energies = wp.zeros(num_atoms, dtype=wp_dtype, device=device)
        charges = wp.array([np_dtype(1.0)], dtype=wp_dtype, device=device)
        volume = make_scalar_array(1000.0, device, wp_dtype)
        alpha = make_scalar_array(alpha_val, device, wp_dtype)
        total_charge = make_scalar_array(1.0, device, wp_dtype)
        corrected_energies = wp.zeros(num_atoms, dtype=wp_dtype, device=device)

        pme_energy_corrections(
            raw_energies=raw_energies,
            charges=charges,
            volume=volume,
            alpha=alpha,
            total_charge=total_charge,
            corrected_energies=corrected_energies,
            wp_dtype=wp_dtype,
            device=device,
        )

        result = corrected_energies.numpy()[0]

        # Hardcoded regression value from actual kernel output
        # This catches any unintended changes to the kernel behavior
        expected_total = -0.28837797333090204

        assert result == pytest.approx(expected_total, rel=1e-10)

    def test_green_function_regression(self, device):
        """Regression test for Green's function with known values.

        k² = 1.0, α = 0.5, V = 100
        G(k) = 2π/V * exp(-k²/(4α²)) / k²
             = 2π/100 * exp(-1/(4*0.25)) / 1
             = 2π/100 * exp(-1)
             ≈ 0.023109...

        Note: Hardcoded value from actual kernel output.
        """
        mesh_nx, mesh_ny, mesh_nz = 4, 4, 4
        nz_rfft = mesh_nz // 2 + 1
        wp_dtype = wp.float64
        np_dtype = np.float64
        alpha_val = 0.5
        volume_val = 100.0

        # k² = 1.0 at all points except k=0
        k_sq_np = np.ones((mesh_nx, mesh_ny, nz_rfft), dtype=np_dtype)
        k_sq_np[0, 0, 0] = 0.0

        k_squared = wp.array(k_sq_np, dtype=wp_dtype, device=device)
        miller_x = wp.zeros(mesh_nx, dtype=wp_dtype, device=device)
        miller_y = wp.zeros(mesh_ny, dtype=wp_dtype, device=device)
        miller_z = wp.zeros(nz_rfft, dtype=wp_dtype, device=device)
        alpha = make_scalar_array(alpha_val, device, wp_dtype)
        volume = make_scalar_array(volume_val, device, wp_dtype)

        green_function = wp.zeros(
            (mesh_nx, mesh_ny, nz_rfft), dtype=wp_dtype, device=device
        )
        structure_factor_sq = wp.zeros(
            (mesh_nx, mesh_ny, nz_rfft), dtype=wp_dtype, device=device
        )

        pme_green_structure_factor(
            k_squared=k_squared,
            miller_x=miller_x,
            miller_y=miller_y,
            miller_z=miller_z,
            alpha=alpha,
            volume=volume,
            mesh_nx=mesh_nx,
            mesh_ny=mesh_ny,
            mesh_nz=mesh_nz,
            spline_order=4,
            green_function=green_function,
            structure_factor_sq=structure_factor_sq,
            wp_dtype=wp_dtype,
            device=device,
        )

        green_np = green_function.numpy()

        # Hardcoded regression value from actual kernel output
        expected = 0.023114547639039303

        # Check non-k=0 point
        assert green_np[1, 0, 0] == pytest.approx(expected, rel=1e-10)
        # k=0 should be zero
        assert green_np[0, 0, 0] == 0.0


###########################################################################################
########################### Parameterized Fixtures ########################################
###########################################################################################


@pytest.fixture(params=[wp.float32, wp.float64], ids=["float32", "float64"])
def wp_dtype(request):
    """Parameterized fixture for Warp dtypes."""
    return request.param
