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
Tests for cell_utils.py - Cell utilities for NPT/NPH simulations.

Covers:
- Cell volume computation
- Cell inverse computation
- Position scaling/remapping with cell changes
- Position wrapping
- Coordinate transformations (Cartesian <-> fractional)
- Batched operations
- Both float32 and float64 precision

All tests use the (B, 3, 3) cell formalism where cells are arrays of mat33.
"""

import numpy as np
import pytest
import warp as wp

from nvalchemiops.dynamics.utils import (
    apply_strain_to_cell,
    cartesian_to_fractional,
    compute_cell_inverse,
    compute_cell_volume,
    compute_strain_tensor,
    fractional_to_cartesian,
    scale_positions_with_cell,
    scale_positions_with_cell_out,
    wrap_positions_to_cell,
    wrap_positions_to_cell_out,
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
    # Warp mat33 is constructed row-major
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
    num_systems = cells_np.shape[0]
    mats = []
    for i in range(num_systems):
        c = cells_np[i]
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


def cell_to_numpy(cells_wp, sys_idx=0):
    """Extract a cell from warp array to numpy (3, 3)."""
    wp.synchronize_device(cells_wp.device)
    mat = cells_wp.numpy()[sys_idx]
    # mat is a flat array in row-major order
    return mat.reshape(3, 3)


# ==============================================================================
# Cell Volume Tests
# ==============================================================================


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("cell_dtype", ["float32", "float64"])
class TestCellVolume:
    """Tests for compute_cell_volume."""

    def test_cubic_cell_volume(self, cell_dtype, device):
        """Test volume of cubic cell."""
        tol = 1e-4 if cell_dtype == "float32" else 1e-10

        cell_np = np.array(
            [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]], dtype=np.float64
        )
        cells = make_cell(cell_np, cell_dtype, device)

        scalar_wp = wp.float32 if cell_dtype == "float32" else wp.float64
        volumes = wp.empty(1, dtype=scalar_wp, device=device)
        compute_cell_volume(cells, volumes, device=device)
        wp.synchronize_device(device)

        assert abs(volumes.numpy()[0] - 1000.0) < tol

    def test_orthorhombic_cell_volume(self, cell_dtype, device):
        """Test volume of orthorhombic cell."""
        tol = 1e-4 if cell_dtype == "float32" else 1e-10

        cell_np = np.array(
            [[10.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 30.0]], dtype=np.float64
        )
        cells = make_cell(cell_np, cell_dtype, device)

        scalar_wp = wp.float32 if cell_dtype == "float32" else wp.float64
        volumes = wp.empty(1, dtype=scalar_wp, device=device)
        compute_cell_volume(cells, volumes, device=device)
        wp.synchronize_device(device)

        assert abs(volumes.numpy()[0] - 6000.0) < tol

    def test_triclinic_cell_volume(self, cell_dtype, device):
        """Test volume of triclinic cell."""
        tol = 1e-3 if cell_dtype == "float32" else 1e-10

        cell_np = np.array(
            [[10.0, 0.0, 0.0], [2.0, 9.0, 0.0], [1.0, 2.0, 8.0]], dtype=np.float64
        )
        cells = make_cell(cell_np, cell_dtype, device)

        scalar_wp = wp.float32 if cell_dtype == "float32" else wp.float64
        volumes = wp.empty(1, dtype=scalar_wp, device=device)
        compute_cell_volume(cells, volumes, device=device)
        wp.synchronize_device(device)

        expected = abs(np.linalg.det(cell_np))
        assert abs(volumes.numpy()[0] - expected) < tol

    def test_batched_cell_volume(self, cell_dtype, device):
        """Test volume computation for batched cells."""
        tol = 1e-3 if cell_dtype == "float32" else 1e-8

        num_systems = 3
        cells_np = np.zeros((num_systems, 3, 3), dtype=np.float64)

        # Different cubic cells
        for i in range(num_systems):
            L = 10.0 + i * 5.0  # 10, 15, 20 Å
            cells_np[i] = np.diag([L, L, L])

        cells = make_cells_batch(cells_np, cell_dtype, device)
        scalar_wp = wp.float32 if cell_dtype == "float32" else wp.float64
        volumes = wp.empty(num_systems, dtype=scalar_wp, device=device)
        compute_cell_volume(cells, volumes, device=device)
        wp.synchronize_device(device)

        volumes_np = volumes.numpy()
        expected = [1000.0, 3375.0, 8000.0]  # L^3

        for i in range(num_systems):
            assert abs(volumes_np[i] - expected[i]) < tol


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("cell_dtype", ["float32", "float64"])
class TestCellInverse:
    """Tests for compute_cell_inverse."""

    def test_cubic_inverse(self, cell_dtype, device):
        """Test inverse of cubic cell."""
        rtol = 1e-4 if cell_dtype == "float32" else 1e-10

        cell_np = np.diag([10.0, 10.0, 10.0])
        cells = make_cell(cell_np, cell_dtype, device)

        mat_dtype = wp.mat33f if cell_dtype == "float32" else wp.mat33d
        cells_inv = wp.empty(1, dtype=mat_dtype, device=device)
        compute_cell_inverse(cells, cells_inv, device=device)
        wp.synchronize_device(device)

        result = cell_to_numpy(cells_inv, 0)
        expected = np.diag([0.1, 0.1, 0.1])
        np.testing.assert_allclose(result, expected, rtol=rtol)

    def test_triclinic_inverse(self, cell_dtype, device):
        """Test inverse of triclinic cell."""
        rtol = 1e-4 if cell_dtype == "float32" else 1e-10

        cell_np = np.array(
            [[10.0, 0.0, 0.0], [2.0, 9.0, 0.0], [1.0, 2.0, 8.0]], dtype=np.float64
        )
        cells = make_cell(cell_np, cell_dtype, device)

        mat_dtype = wp.mat33f if cell_dtype == "float32" else wp.mat33d
        cells_inv = wp.empty(1, dtype=mat_dtype, device=device)
        compute_cell_inverse(cells, cells_inv, device=device)
        wp.synchronize_device(device)

        result = cell_to_numpy(cells_inv, 0)
        expected = np.linalg.inv(cell_np)
        np.testing.assert_allclose(result, expected, rtol=rtol)

    def test_inverse_identity(self, cell_dtype, device):
        """Test that cell @ cell_inv = I."""
        atol = 1e-4 if cell_dtype == "float32" else 1e-10

        cell_np = np.array(
            [[10.0, 0.0, 0.0], [2.0, 9.0, 0.0], [1.0, 2.0, 8.0]], dtype=np.float64
        )
        cells = make_cell(cell_np, cell_dtype, device)

        mat_dtype = wp.mat33f if cell_dtype == "float32" else wp.mat33d
        cells_inv = wp.empty(1, dtype=mat_dtype, device=device)
        compute_cell_inverse(cells, cells_inv, device=device)
        wp.synchronize_device(device)

        inv_np = cell_to_numpy(cells_inv, 0)
        product = cell_np @ inv_np
        np.testing.assert_allclose(product, np.eye(3), atol=atol)


# ==============================================================================
# Strain Tensor Tests
# ==============================================================================


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
class TestStrainTensor:
    """Tests for strain tensor computations."""

    def test_zero_strain_identity(self, dtype, device):
        """Test that identical cells give zero strain."""
        atol = 1e-6 if dtype == "float32" else 1e-12
        mat_dtype = wp.mat33f if dtype == "float32" else wp.mat33d
        cell_np = np.diag([10.0, 10.0, 10.0])
        cells = make_cell(cell_np, dtype, device)

        cells_ref_inv = wp.empty(1, dtype=mat_dtype, device=device)
        compute_cell_inverse(cells, cells_ref_inv, device=device)
        strains = wp.empty(1, dtype=mat_dtype, device=device)
        compute_strain_tensor(cells, cells_ref_inv, strains, device=device)
        wp.synchronize_device(device)

        result = cell_to_numpy(strains, 0)
        np.testing.assert_allclose(result, np.zeros((3, 3)), atol=atol)

    def test_isotropic_expansion(self, dtype, device):
        """Test strain for isotropic expansion."""
        rtol = 1e-4 if dtype == "float32" else 1e-10
        mat_dtype = wp.mat33f if dtype == "float32" else wp.mat33d
        cell_ref_np = np.diag([10.0, 10.0, 10.0])
        cell_new_np = np.diag([11.0, 11.0, 11.0])  # 10% expansion

        cells_ref = make_cell(cell_ref_np, dtype, device)
        cells_new = make_cell(cell_new_np, dtype, device)

        cells_ref_inv = wp.empty(1, dtype=mat_dtype, device=device)
        compute_cell_inverse(cells_ref, cells_ref_inv, device=device)
        strains = wp.empty(1, dtype=mat_dtype, device=device)
        compute_strain_tensor(cells_new, cells_ref_inv, strains, device=device)
        wp.synchronize_device(device)

        result = cell_to_numpy(strains, 0)
        expected = np.diag([0.1, 0.1, 0.1])  # 10% strain
        np.testing.assert_allclose(result, expected, rtol=rtol)

    def test_apply_strain_roundtrip(self, dtype, device):
        """Test that applying computed strain recovers original cell."""
        rtol = 1e-4 if dtype == "float32" else 1e-10
        np_dtype = np.float32 if dtype == "float32" else np.float64
        mat_dtype = wp.mat33f if dtype == "float32" else wp.mat33d
        cell_ref_np = np.diag([10.0, 10.0, 10.0])
        cell_new_np = np.array(
            [[11.0, 0.5, 0.0], [0.0, 10.5, 0.2], [0.0, 0.0, 10.0]], dtype=np_dtype
        )

        cells_ref = make_cell(cell_ref_np, dtype, device)
        cells_new = make_cell(cell_new_np, dtype, device)

        # Compute strain
        cells_ref_inv = wp.empty(1, dtype=mat_dtype, device=device)
        compute_cell_inverse(cells_ref, cells_ref_inv, device=device)
        strains = wp.empty(1, dtype=mat_dtype, device=device)
        compute_strain_tensor(cells_new, cells_ref_inv, strains, device=device)
        wp.synchronize_device(device)

        # Apply strain to reference
        cells_recovered = wp.empty(1, dtype=mat_dtype, device=device)
        apply_strain_to_cell(cells_ref, strains, cells_recovered, device=device)
        wp.synchronize_device(device)

        result = cell_to_numpy(cells_recovered, 0)
        np.testing.assert_allclose(result, cell_new_np, rtol=rtol)

    def test_strain_with_precomputed_inverse(self, dtype, device):
        """Test compute_strain_tensor with pre-computed cells_ref_inv."""
        rtol = 1e-4 if dtype == "float32" else 1e-10
        mat_dtype = wp.mat33f if dtype == "float32" else wp.mat33d
        cell_ref_np = np.diag([10.0, 10.0, 10.0])
        cell_new_np = np.diag([11.0, 11.0, 11.0])

        cells_ref = make_cell(cell_ref_np, dtype, device)
        cells_new = make_cell(cell_new_np, dtype, device)

        # Pre-compute inverse
        cells_ref_inv = wp.empty(1, dtype=mat_dtype, device=device)
        compute_cell_inverse(cells_ref, cells_ref_inv, device=device)
        wp.synchronize_device(device)

        # Use pre-computed inverse
        strains = wp.empty(1, dtype=mat_dtype, device=device)
        compute_strain_tensor(cells_new, cells_ref_inv, strains, device=device)
        wp.synchronize_device(device)

        result = cell_to_numpy(strains, 0)
        expected = np.diag([0.1, 0.1, 0.1])
        np.testing.assert_allclose(result, expected, rtol=rtol)


# ==============================================================================
# Position Scaling Tests
# ==============================================================================


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
class TestPositionScaling:
    """Tests for position scaling with cell changes."""

    def test_scale_positions_maintains_fractional(self, dtype, device):
        """Test that scaling maintains fractional coordinates."""
        vec_dtype = wp.vec3f if dtype == "float32" else wp.vec3d
        scalar_dtype = np.float32 if dtype == "float32" else np.float64

        num_atoms = 10
        np.random.seed(42)

        # Original cubic cell
        cell_old_np = np.diag([10.0, 10.0, 10.0])
        # New cell (expanded)
        cell_new_np = np.diag([12.0, 12.0, 12.0])

        # Random positions in original cell
        positions_np = np.random.rand(num_atoms, 3).astype(scalar_dtype) * 10.0

        # Compute original fractional coordinates
        fractional_old = positions_np @ np.linalg.inv(cell_old_np)

        cells_old = make_cell(cell_old_np, dtype, device)
        cells_new = make_cell(cell_new_np, dtype, device)
        positions = wp.array(positions_np, dtype=vec_dtype, device=device)

        # Pre-compute inverse of old cell
        mat_dtype = wp.mat33f if dtype == "float32" else wp.mat33d
        cells_old_inv = wp.empty(1, dtype=mat_dtype, device=device)
        compute_cell_inverse(cells_old, cells_old_inv, device=device)

        # Test single-system version (no batch_idx)
        scale_positions_with_cell(positions, cells_new, cells_old_inv, device=device)
        wp.synchronize_device(device)

        positions_scaled = positions.numpy()

        # Compute new fractional coordinates
        fractional_new = positions_scaled @ np.linalg.inv(cell_new_np)

        # Fractional coords should be preserved
        np.testing.assert_allclose(fractional_new, fractional_old, rtol=1e-4)

    def test_scale_positions_out_preserves_input(self, dtype, device):
        """Test that non-mutating version preserves input."""
        vec_dtype = wp.vec3f if dtype == "float32" else wp.vec3d
        scalar_dtype = np.float32 if dtype == "float32" else np.float64

        positions_np = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [2.0, 2.0, 2.0],
                [5.0, 5.0, 5.0],
            ],
            dtype=scalar_dtype,
        )

        cell_old_np = np.diag([10.0, 10.0, 10.0])
        cell_new_np = np.diag([20.0, 20.0, 20.0])

        cells_old = make_cell(cell_old_np, dtype, device)
        cells_new = make_cell(cell_new_np, dtype, device)
        positions = wp.array(positions_np.copy(), dtype=vec_dtype, device=device)

        # Pre-compute inverse of old cell
        mat_dtype = wp.mat33f if dtype == "float32" else wp.mat33d
        cells_old_inv = wp.empty(1, dtype=mat_dtype, device=device)
        compute_cell_inverse(cells_old, cells_old_inv, device=device)

        # Test single-system version (no batch_idx)
        positions_out = wp.empty(positions.shape[0], dtype=vec_dtype, device=device)
        scale_positions_with_cell_out(
            positions, cells_new, cells_old_inv, positions_out, device=device
        )
        wp.synchronize_device(device)

        # Input should be unchanged
        np.testing.assert_allclose(positions.numpy(), positions_np, rtol=1e-5)

        # Output should be scaled (2x for isotropic doubling)
        expected_scaled = positions_np * 2.0
        np.testing.assert_allclose(positions_out.numpy(), expected_scaled, rtol=1e-4)

    def test_scale_with_precomputed_inverse(self, dtype, device):
        """Test scale_positions with pre-computed cells_old_inv."""
        vec_dtype = wp.vec3f if dtype == "float32" else wp.vec3d
        scalar_dtype = np.float32 if dtype == "float32" else np.float64

        positions_np = np.array([[5.0, 5.0, 5.0]], dtype=scalar_dtype)
        cell_old_np = np.diag([10.0, 10.0, 10.0])
        cell_new_np = np.diag([20.0, 20.0, 20.0])

        cells_old = make_cell(cell_old_np, dtype, device)
        cells_new = make_cell(cell_new_np, dtype, device)

        # Pre-compute inverse
        mat_dtype = wp.mat33f if dtype == "float32" else wp.mat33d
        cells_old_inv = wp.empty(1, dtype=mat_dtype, device=device)
        compute_cell_inverse(cells_old, cells_old_inv, device=device)
        wp.synchronize_device(device)

        positions = wp.array(positions_np.copy(), dtype=vec_dtype, device=device)

        # Use pre-computed inverse (single-system, no batch_idx)
        scale_positions_with_cell(positions, cells_new, cells_old_inv, device=device)
        wp.synchronize_device(device)

        # Should be scaled 2x
        np.testing.assert_allclose(positions.numpy()[0], [10.0, 10.0, 10.0], rtol=1e-4)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
class TestPositionScalingBatched:
    """Tests for batched position scaling."""

    def test_batched_scale_positions(self, dtype, device):
        """Test batched position scaling."""
        vec_dtype = wp.vec3f if dtype == "float32" else wp.vec3d
        scalar_dtype = np.float32 if dtype == "float32" else np.float64

        num_systems = 2
        atoms_per_system = 5
        num_atoms = num_systems * atoms_per_system

        np.random.seed(42)

        # Create batch index
        batch_idx_np = np.repeat(np.arange(num_systems), atoms_per_system).astype(
            np.int32
        )

        # Different cells for each system
        cells_old_np = np.zeros((num_systems, 3, 3), dtype=np.float64)
        cells_new_np = np.zeros((num_systems, 3, 3), dtype=np.float64)

        cells_old_np[0] = np.diag([10.0, 10.0, 10.0])
        cells_old_np[1] = np.diag([20.0, 20.0, 20.0])
        cells_new_np[0] = np.diag([12.0, 12.0, 12.0])  # 20% expansion
        cells_new_np[1] = np.diag([18.0, 18.0, 18.0])  # 10% compression

        # Random positions
        positions_np = np.random.rand(num_atoms, 3).astype(scalar_dtype)
        # Scale to respective cell sizes
        for i in range(num_atoms):
            sys_id = batch_idx_np[i]
            positions_np[i] *= cells_old_np[sys_id, 0, 0]

        batch_idx = wp.array(batch_idx_np, dtype=wp.int32, device=device)
        cells_old = make_cells_batch(cells_old_np, dtype, device)
        cells_new = make_cells_batch(cells_new_np, dtype, device)
        positions = wp.array(positions_np.copy(), dtype=vec_dtype, device=device)

        # Pre-compute inverse of old cells
        mat_dtype = wp.mat33f if dtype == "float32" else wp.mat33d
        cells_old_inv = wp.empty(num_systems, dtype=mat_dtype, device=device)
        compute_cell_inverse(cells_old, cells_old_inv, device=device)

        # Test batched version with batch_idx keyword
        scale_positions_with_cell(
            positions,
            cells_new,
            cells_old_inv,
            batch_idx=batch_idx,
            device=device,
        )
        wp.synchronize_device(device)

        positions_scaled = positions.numpy()

        # Check each system independently
        for sys_id in range(num_systems):
            mask = batch_idx_np == sys_id
            pos_old = positions_np[mask]
            pos_new = positions_scaled[mask]

            # Compute fractional coordinates
            frac_old = pos_old @ np.linalg.inv(cells_old_np[sys_id])
            frac_new = pos_new @ np.linalg.inv(cells_new_np[sys_id])

            np.testing.assert_allclose(frac_new, frac_old, rtol=1e-4)


# ==============================================================================
# Position Wrapping Tests
# ==============================================================================


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
class TestPositionWrapping:
    """Tests for position wrapping into primary cell."""

    def test_wrap_positions_cubic(self, dtype, device):
        """Test wrapping in cubic cell."""
        vec_dtype = wp.vec3f if dtype == "float32" else wp.vec3d
        scalar_dtype = np.float32 if dtype == "float32" else np.float64

        # Positions outside cell
        positions_np = np.array(
            [
                [15.0, 5.0, 5.0],  # x > L
                [-3.0, 5.0, 5.0],  # x < 0
                [25.0, -8.0, 12.0],  # Multiple wraps needed
            ],
            dtype=scalar_dtype,
        )

        cell_np = np.diag([10.0, 10.0, 10.0])

        cells = make_cell(cell_np, dtype, device)
        positions = wp.array(positions_np.copy(), dtype=vec_dtype, device=device)

        # Pre-compute inverse
        mat_dtype = wp.mat33f if dtype == "float32" else wp.mat33d
        cells_inv = wp.empty(1, dtype=mat_dtype, device=device)
        compute_cell_inverse(cells, cells_inv, device=device)

        # Test single-system version (no batch_idx)
        wrap_positions_to_cell(positions, cells, cells_inv, device=device)
        wp.synchronize_device(device)

        wrapped = positions.numpy()

        # All positions should be in [0, L)
        assert np.all(wrapped >= 0.0 - 1e-6)
        assert np.all(wrapped < 10.0 + 1e-6)

        # Check specific values
        np.testing.assert_allclose(wrapped[0], [5.0, 5.0, 5.0], atol=1e-4)
        np.testing.assert_allclose(wrapped[1], [7.0, 5.0, 5.0], atol=1e-4)
        np.testing.assert_allclose(wrapped[2], [5.0, 2.0, 2.0], atol=1e-4)

    def test_wrap_positions_triclinic(self, dtype, device):
        """Test wrapping in triclinic cell."""
        vec_dtype = wp.vec3f if dtype == "float32" else wp.vec3d
        scalar_dtype = np.float32 if dtype == "float32" else np.float64

        cell_np = np.array(
            [[10.0, 0.0, 0.0], [2.0, 9.0, 0.0], [1.0, 2.0, 8.0]], dtype=np.float64
        )

        # Position outside cell
        positions_np = np.array(
            [
                [15.0, 15.0, 15.0],
            ],
            dtype=scalar_dtype,
        )

        cells = make_cell(cell_np, dtype, device)
        positions = wp.array(positions_np.copy(), dtype=vec_dtype, device=device)

        # Pre-compute inverse
        mat_dtype = wp.mat33f if dtype == "float32" else wp.mat33d
        cells_inv = wp.empty(1, dtype=mat_dtype, device=device)
        compute_cell_inverse(cells, cells_inv, device=device)

        # Test single-system version (no batch_idx)
        wrap_positions_to_cell(positions, cells, cells_inv, device=device)
        wp.synchronize_device(device)

        wrapped = positions.numpy()

        # Verify wrapped position is in fractional [0, 1)
        fractional = wrapped @ np.linalg.inv(cell_np)
        assert np.all(fractional >= 0.0 - 1e-5)
        assert np.all(fractional < 1.0 + 1e-5)

    def test_wrap_out_preserves_input(self, dtype, device):
        """Test that non-mutating wrap preserves input."""
        vec_dtype = wp.vec3f if dtype == "float32" else wp.vec3d
        scalar_dtype = np.float32 if dtype == "float32" else np.float64
        mat_dtype = wp.mat33f if dtype == "float32" else wp.mat33d

        positions_np = np.array([[15.0, 5.0, 5.0]], dtype=scalar_dtype)
        cell_np = np.diag([10.0, 10.0, 10.0])

        cells = make_cell(cell_np, dtype, device)
        positions = wp.array(positions_np.copy(), dtype=vec_dtype, device=device)

        # Pre-compute inverse
        cells_inv = wp.empty(1, dtype=mat_dtype, device=device)
        compute_cell_inverse(cells, cells_inv, device=device)

        # Test single-system version (no batch_idx)
        wrapped = wp.empty(positions.shape[0], dtype=vec_dtype, device=device)
        wrap_positions_to_cell_out(positions, cells, cells_inv, wrapped, device=device)
        wp.synchronize_device(device)

        # Input unchanged
        np.testing.assert_allclose(positions.numpy(), positions_np, rtol=1e-5)

        # Output is wrapped
        np.testing.assert_allclose(wrapped.numpy()[0], [5.0, 5.0, 5.0], atol=1e-4)

    def test_wrap_with_precomputed_inverse(self, dtype, device):
        """Test wrap_positions with pre-computed cells_inv."""
        vec_dtype = wp.vec3f if dtype == "float32" else wp.vec3d
        scalar_dtype = np.float32 if dtype == "float32" else np.float64

        positions_np = np.array([[15.0, 5.0, 5.0]], dtype=scalar_dtype)
        cell_np = np.diag([10.0, 10.0, 10.0])

        cells = make_cell(cell_np, dtype, device)
        # Pre-compute inverse
        mat_dtype = wp.mat33f if dtype == "float32" else wp.mat33d
        cells_inv = wp.empty(1, dtype=mat_dtype, device=device)
        compute_cell_inverse(cells, cells_inv, device=device)
        wp.synchronize_device(device)

        positions = wp.array(positions_np.copy(), dtype=vec_dtype, device=device)

        # Use pre-computed inverse (single-system, no batch_idx)
        wrap_positions_to_cell(positions, cells, cells_inv, device=device)
        wp.synchronize_device(device)

        np.testing.assert_allclose(positions.numpy()[0], [5.0, 5.0, 5.0], atol=1e-4)


# ==============================================================================
# Coordinate Transformation Tests
# ==============================================================================


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
class TestCoordinateTransformations:
    """Tests for Cartesian <-> fractional coordinate transformations."""

    def test_cartesian_to_fractional_cubic(self, dtype, device):
        """Test Cartesian to fractional conversion in cubic cell."""
        vec_dtype = wp.vec3f if dtype == "float32" else wp.vec3d
        scalar_dtype = np.float32 if dtype == "float32" else np.float64
        mat_dtype = wp.mat33f if dtype == "float32" else wp.mat33d

        cell_np = np.diag([10.0, 10.0, 10.0])
        positions_np = np.array(
            [
                [5.0, 5.0, 5.0],
                [0.0, 0.0, 0.0],
                [10.0, 10.0, 10.0],
            ],
            dtype=scalar_dtype,
        )

        cells = make_cell(cell_np, dtype, device)
        positions = wp.array(positions_np, dtype=vec_dtype, device=device)

        # Pre-compute inverse
        cells_inv = wp.empty(1, dtype=mat_dtype, device=device)
        compute_cell_inverse(cells, cells_inv, device=device)

        # Test single-system version (no batch_idx)
        fractional = wp.empty(positions.shape[0], dtype=vec_dtype, device=device)
        cartesian_to_fractional(positions, cells_inv, fractional, device=device)
        wp.synchronize_device(device)

        expected = np.array(
            [
                [0.5, 0.5, 0.5],
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ]
        )

        np.testing.assert_allclose(fractional.numpy(), expected, rtol=1e-4)

    def test_fractional_to_cartesian_cubic(self, dtype, device):
        """Test fractional to Cartesian conversion in cubic cell."""
        vec_dtype = wp.vec3f if dtype == "float32" else wp.vec3d
        scalar_dtype = np.float32 if dtype == "float32" else np.float64

        cell_np = np.diag([10.0, 10.0, 10.0])
        fractional_np = np.array(
            [
                [0.5, 0.5, 0.5],
                [0.0, 0.0, 0.0],
                [0.25, 0.75, 0.1],
            ],
            dtype=scalar_dtype,
        )

        cells = make_cell(cell_np, dtype, device)
        fractional = wp.array(fractional_np, dtype=vec_dtype, device=device)

        # Test single-system version (no batch_idx)
        positions = wp.empty(fractional.shape[0], dtype=vec_dtype, device=device)
        fractional_to_cartesian(fractional, cells, positions, device=device)
        wp.synchronize_device(device)

        expected = np.array(
            [
                [5.0, 5.0, 5.0],
                [0.0, 0.0, 0.0],
                [2.5, 7.5, 1.0],
            ]
        )

        np.testing.assert_allclose(positions.numpy(), expected, rtol=1e-4)

    def test_roundtrip_cartesian_fractional_cartesian(self, dtype, device):
        """Test roundtrip conversion preserves coordinates."""
        vec_dtype = wp.vec3f if dtype == "float32" else wp.vec3d
        scalar_dtype = np.float32 if dtype == "float32" else np.float64
        mat_dtype = wp.mat33f if dtype == "float32" else wp.mat33d

        cell_np = np.array(
            [[10.0, 0.0, 0.0], [2.0, 9.0, 0.0], [1.0, 2.0, 8.0]], dtype=np.float64
        )

        positions_np = np.array(
            [
                [5.0, 3.0, 2.0],
                [8.0, 7.0, 6.0],
            ],
            dtype=scalar_dtype,
        )

        cells = make_cell(cell_np, dtype, device)
        positions = wp.array(positions_np, dtype=vec_dtype, device=device)

        # Pre-compute inverse
        cells_inv = wp.empty(1, dtype=mat_dtype, device=device)
        compute_cell_inverse(cells, cells_inv, device=device)

        num_atoms = positions_np.shape[0]

        # Test single-system version (no batch_idx)
        # Cartesian -> Fractional -> Cartesian
        fractional = wp.empty(num_atoms, dtype=vec_dtype, device=device)
        cartesian_to_fractional(positions, cells_inv, fractional, device=device)
        positions_back = wp.empty(num_atoms, dtype=vec_dtype, device=device)
        fractional_to_cartesian(fractional, cells, positions_back, device=device)
        wp.synchronize_device(device)

        np.testing.assert_allclose(positions_back.numpy(), positions_np, rtol=1e-4)


# ==============================================================================
# Coverage Tests - Single System Paths Without batch_idx
# ==============================================================================


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
class TestCellUtilsSingleSystemCoverage:
    """Coverage tests for single-system cell utility paths (no batch_idx)."""

    def test_scale_positions_single_system(self, dtype, device):
        """Test position scaling for single system without batch_idx."""
        num_atoms = 20
        vec_dtype = wp.vec3f if dtype == "float32" else wp.vec3d
        np_dtype = np.float32 if dtype == "float32" else np.float64

        positions = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype) * 5.0,
            dtype=vec_dtype,
            device=device,
        )
        cell_np = np.array(
            [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]], dtype=np_dtype
        )
        cell_new_np = np.array(
            [[12.0, 0.0, 0.0], [0.0, 12.0, 0.0], [0.0, 0.0, 12.0]], dtype=np_dtype
        )

        cells = make_cell(cell_np, dtype, device)
        cells_new = make_cell(cell_new_np, dtype, device)

        mat_dtype = wp.mat33f if dtype == "float32" else wp.mat33d
        cells_inv = wp.empty(1, dtype=mat_dtype, device=device)
        compute_cell_inverse(cells, cells_inv, device=device)

        scale_positions_with_cell(positions, cells_new, cells_inv, device=device)

        assert positions.shape[0] == num_atoms

    def test_scale_positions_out_single_system(self, dtype, device):
        """Test non-mutating position scaling for single system preserves input."""
        num_atoms = 20
        vec_dtype = wp.vec3f if dtype == "float32" else wp.vec3d
        np_dtype = np.float32 if dtype == "float32" else np.float64
        mat_dtype = wp.mat33f if dtype == "float32" else wp.mat33d

        positions = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype) * 5.0,
            dtype=vec_dtype,
            device=device,
        )
        cell_np = np.array(
            [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]], dtype=np_dtype
        )
        cell_new_np = np.array(
            [[12.0, 0.0, 0.0], [0.0, 12.0, 0.0], [0.0, 0.0, 12.0]], dtype=np_dtype
        )

        cells = make_cell(cell_np, dtype, device)
        cells_new = make_cell(cell_new_np, dtype, device)

        cells_inv = wp.empty(1, dtype=mat_dtype, device=device)
        compute_cell_inverse(cells, cells_inv, device=device)

        pos_orig = positions.numpy().copy()

        pos_out = wp.empty(num_atoms, dtype=vec_dtype, device=device)
        scale_positions_with_cell_out(
            positions, cells_new, cells_inv, pos_out, device=device
        )

        np.testing.assert_array_equal(positions.numpy(), pos_orig)
        assert pos_out.shape[0] == num_atoms

    def test_wrap_positions_single_system(self, dtype, device):
        """Test position wrapping for single system without batch_idx."""
        num_atoms = 20
        vec_dtype = wp.vec3f if dtype == "float32" else wp.vec3d
        np_dtype = np.float32 if dtype == "float32" else np.float64

        positions = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype) * 20.0,
            dtype=vec_dtype,
            device=device,
        )
        cell_np = np.array(
            [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]], dtype=np_dtype
        )

        cells = make_cell(cell_np, dtype, device)

        mat_dtype = wp.mat33f if dtype == "float32" else wp.mat33d
        cells_inv = wp.empty(1, dtype=mat_dtype, device=device)
        compute_cell_inverse(cells, cells_inv, device=device)

        wrap_positions_to_cell(positions, cells, cells_inv, device=device)

        assert positions.shape[0] == num_atoms

    def test_wrap_positions_out_single_system(self, dtype, device):
        """Test non-mutating position wrapping for single system preserves input."""
        num_atoms = 20
        vec_dtype = wp.vec3f if dtype == "float32" else wp.vec3d
        np_dtype = np.float32 if dtype == "float32" else np.float64
        mat_dtype = wp.mat33f if dtype == "float32" else wp.mat33d

        positions = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype) * 20.0,
            dtype=vec_dtype,
            device=device,
        )
        cell_np = np.array(
            [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]], dtype=np_dtype
        )

        cells = make_cell(cell_np, dtype, device)

        cells_inv = wp.empty(1, dtype=mat_dtype, device=device)
        compute_cell_inverse(cells, cells_inv, device=device)

        pos_orig = positions.numpy().copy()

        pos_out = wp.empty(num_atoms, dtype=vec_dtype, device=device)
        wrap_positions_to_cell_out(positions, cells, cells_inv, pos_out, device=device)

        np.testing.assert_array_equal(positions.numpy(), pos_orig)
        assert pos_out.shape[0] == num_atoms

    def test_cartesian_to_fractional_single_system(self, dtype, device):
        """Test Cartesian to fractional conversion for single system."""
        num_atoms = 20
        vec_dtype = wp.vec3f if dtype == "float32" else wp.vec3d
        np_dtype = np.float32 if dtype == "float32" else np.float64
        mat_dtype = wp.mat33f if dtype == "float32" else wp.mat33d

        positions = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype) * 5.0,
            dtype=vec_dtype,
            device=device,
        )
        cell_np = np.array(
            [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]], dtype=np_dtype
        )

        cells = make_cell(cell_np, dtype, device)

        cells_inv = wp.empty(1, dtype=mat_dtype, device=device)
        compute_cell_inverse(cells, cells_inv, device=device)

        fractional = wp.empty(num_atoms, dtype=vec_dtype, device=device)
        cartesian_to_fractional(positions, cells_inv, fractional, device=device)

        assert fractional.shape[0] == num_atoms

    def test_fractional_to_cartesian_single_system(self, dtype, device):
        """Test fractional to Cartesian conversion for single system."""
        num_atoms = 20
        vec_dtype = wp.vec3f if dtype == "float32" else wp.vec3d
        np_dtype = np.float32 if dtype == "float32" else np.float64

        # Fractional coordinates in [0, 1)
        fractional = wp.array(
            np.random.rand(num_atoms, 3).astype(np_dtype),
            dtype=vec_dtype,
            device=device,
        )
        cell_np = np.array(
            [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]], dtype=np_dtype
        )

        cells = make_cell(cell_np, dtype, device)

        cartesian = wp.empty(num_atoms, dtype=vec_dtype, device=device)
        fractional_to_cartesian(fractional, cells, cartesian, device=device)

        assert cartesian.shape[0] == num_atoms


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
class TestCellUtilsCoverageExtras:
    """Additional coverage tests for edge cases."""

    def test_compute_cell_volume_device_inference(self, dtype, device):
        """Test device inference for compute_cell_volume."""
        np_dtype = np.float32 if dtype == "float32" else np.float64
        scalar_wp = wp.float32 if dtype == "float32" else wp.float64
        cell_np = np.diag([10.0, 10.0, 10.0]).astype(np_dtype)
        cells = make_cell(cell_np, dtype, device)

        # Call without explicit device
        volumes = wp.empty(1, dtype=scalar_wp, device=device)
        compute_cell_volume(cells, volumes)

        wp.synchronize_device(device)
        np.testing.assert_allclose(volumes.numpy()[0], 1000.0, rtol=1e-5)

    def test_compute_cell_inverse_device_inference(self, dtype, device):
        """Test device inference for compute_cell_inverse."""
        np_dtype = np.float32 if dtype == "float32" else np.float64
        mat_dtype = wp.mat33f if dtype == "float32" else wp.mat33d
        cell_np = np.diag([10.0, 10.0, 10.0]).astype(np_dtype)
        cells = make_cell(cell_np, dtype, device)

        # Call without explicit device
        cells_inv = wp.empty(1, dtype=mat_dtype, device=device)
        compute_cell_inverse(cells, cells_inv)

        wp.synchronize_device(device)
        assert cells_inv.shape[0] == 1

    def test_scale_positions_with_cells_inv_precomputed(self, dtype, device):
        """Test scale_positions_with_cell with pre-computed cells_old_inv."""
        num_atoms = 20
        np.random.seed(42)
        np_dtype = np.float32 if dtype == "float32" else np.float64
        vec_dtype = wp.vec3f if dtype == "float32" else wp.vec3d
        mat_dtype = wp.mat33f if dtype == "float32" else wp.mat33d

        positions = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype) * 5.0,
            dtype=vec_dtype,
            device=device,
        )
        cell_np = np.diag([10.0, 10.0, 10.0]).astype(np_dtype)
        cell_new_np = np.diag([12.0, 12.0, 12.0]).astype(np_dtype)

        cells = make_cell(cell_np, dtype, device)
        cells_new = make_cell(cell_new_np, dtype, device)

        pos_orig = positions.numpy().copy()

        # Pre-compute cells_old_inv
        cells_old_inv = wp.empty(1, dtype=mat_dtype, device=device)
        compute_cell_inverse(cells, cells_old_inv, device=device)

        scale_positions_with_cell(positions, cells_new, cells_old_inv, device=device)

        wp.synchronize_device(device)
        # Positions should have changed
        assert not np.allclose(positions.numpy(), pos_orig)

    def test_wrap_positions_with_cells_inv_precomputed(self, dtype, device):
        """Test wrap_positions_to_cell with pre-computed cells_inv."""
        num_atoms = 20
        np.random.seed(42)
        np_dtype = np.float32 if dtype == "float32" else np.float64
        vec_dtype = wp.vec3f if dtype == "float32" else wp.vec3d
        mat_dtype = wp.mat33f if dtype == "float32" else wp.mat33d

        positions = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype) * 20.0,
            dtype=vec_dtype,
            device=device,
        )
        cell_np = np.diag([10.0, 10.0, 10.0]).astype(np_dtype)
        cells = make_cell(cell_np, dtype, device)

        # Pre-compute cells_inv
        cells_inv = wp.empty(1, dtype=mat_dtype, device=device)
        compute_cell_inverse(cells, cells_inv, device=device)

        wrap_positions_to_cell(positions, cells, cells_inv, device=device)

        wp.synchronize_device(device)
        assert positions.shape[0] == num_atoms

    def test_cartesian_to_fractional_with_cells_inv_provided(self, dtype, device):
        """Test cartesian_to_fractional when cells_inv is provided."""
        num_atoms = 20
        np.random.seed(42)
        np_dtype = np.float32 if dtype == "float32" else np.float64
        vec_dtype = wp.vec3f if dtype == "float32" else wp.vec3d

        positions = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype) * 5.0,
            dtype=vec_dtype,
            device=device,
        )
        cell_np = np.diag([10.0, 10.0, 10.0]).astype(np_dtype)
        cell_inv_np = np.linalg.inv(cell_np).astype(np_dtype)

        cells_inv = make_cell(cell_inv_np, dtype, device)

        # Call with pre-computed cells_inv
        fractional = wp.empty(num_atoms, dtype=vec_dtype, device=device)
        cartesian_to_fractional(positions, cells_inv, fractional, device=device)

        wp.synchronize_device(device)
        assert fractional.shape[0] == num_atoms

    def test_compute_cell_volume_preallocated(self, dtype, device):
        """Test compute_cell_volume with pre-allocated output."""
        np_dtype = np.float32 if dtype == "float32" else np.float64
        scalar_dtype = wp.float32 if dtype == "float32" else wp.float64
        cell_np = np.diag([10.0, 10.0, 10.0]).astype(np_dtype)
        cells = make_cell(cell_np, dtype, device)

        # Pre-allocate volumes
        volumes = wp.empty(1, dtype=scalar_dtype, device=device)

        result = compute_cell_volume(cells, volumes, device=device)

        wp.synchronize_device(device)
        assert result is volumes
        np.testing.assert_allclose(volumes.numpy()[0], 1000.0, rtol=1e-5)


# ==============================================================================
# Coverage Tests - Batched Operations
# ==============================================================================


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
class TestCellUtilsBatchedCoverage:
    """Coverage tests for batched cell utility operations."""

    def test_wrap_positions_batched(self, dtype, device):
        """Test wrap_positions_to_cell with batched systems."""
        num_atoms = 20
        num_systems = 2
        np.random.seed(42)
        np_dtype = np.float32 if dtype == "float32" else np.float64
        vec_dtype = wp.vec3f if dtype == "float32" else wp.vec3d
        mat_dtype = wp.mat33f if dtype == "float32" else wp.mat33d

        positions = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype) * 20.0,
            dtype=vec_dtype,
            device=device,
        )
        batch_idx = wp.array([0] * 10 + [1] * 10, dtype=wp.int32, device=device)

        cell_np = np.diag([10.0, 10.0, 10.0]).astype(np_dtype)
        cells = make_cells_batch(np.stack([cell_np] * num_systems), dtype, device)

        cells_inv = wp.empty(num_systems, dtype=mat_dtype, device=device)
        compute_cell_inverse(cells, cells_inv, device=device)

        wrap_positions_to_cell(
            positions, cells, cells_inv, batch_idx=batch_idx, device=device
        )

        wp.synchronize_device(device)
        # Positions should be wrapped into cell
        pos_np = positions.numpy()
        for i in range(num_atoms):
            for dim in range(3):
                assert -0.01 <= pos_np[i, dim] <= 10.01

    def test_wrap_positions_out_batched(self, dtype, device):
        """Test wrap_positions_to_cell_out with batched systems."""
        num_atoms = 20
        num_systems = 2
        np.random.seed(42)
        np_dtype = np.float32 if dtype == "float32" else np.float64
        vec_dtype = wp.vec3f if dtype == "float32" else wp.vec3d
        mat_dtype = wp.mat33f if dtype == "float32" else wp.mat33d

        positions = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype) * 20.0,
            dtype=vec_dtype,
            device=device,
        )
        batch_idx = wp.array([0] * 10 + [1] * 10, dtype=wp.int32, device=device)

        cell_np = np.diag([10.0, 10.0, 10.0]).astype(np_dtype)
        cells = make_cells_batch(np.stack([cell_np] * num_systems), dtype, device)

        cells_inv = wp.empty(num_systems, dtype=mat_dtype, device=device)
        compute_cell_inverse(cells, cells_inv, device=device)

        pos_orig = positions.numpy().copy()
        positions_out = wp.empty(num_atoms, dtype=vec_dtype, device=device)
        result = wrap_positions_to_cell_out(
            positions,
            cells,
            cells_inv,
            positions_out,
            batch_idx=batch_idx,
            device=device,
        )

        wp.synchronize_device(device)
        # Original positions preserved
        np.testing.assert_allclose(positions.numpy(), pos_orig)
        # Output wrapped
        assert result.shape[0] == num_atoms

    def test_cartesian_to_fractional_batched(self, dtype, device):
        """Test cartesian_to_fractional with batched systems."""
        num_atoms = 20
        num_systems = 2
        np.random.seed(42)
        np_dtype = np.float32 if dtype == "float32" else np.float64
        vec_dtype = wp.vec3f if dtype == "float32" else wp.vec3d
        mat_dtype = wp.mat33f if dtype == "float32" else wp.mat33d

        positions = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype) * 5.0,
            dtype=vec_dtype,
            device=device,
        )
        batch_idx = wp.array([0] * 10 + [1] * 10, dtype=wp.int32, device=device)

        cell_np = np.diag([10.0, 10.0, 10.0]).astype(np_dtype)
        cells = make_cells_batch(np.stack([cell_np] * num_systems), dtype, device)

        cells_inv = wp.empty(num_systems, dtype=mat_dtype, device=device)
        compute_cell_inverse(cells, cells_inv, device=device)

        fractional_out = wp.empty(num_atoms, dtype=vec_dtype, device=device)
        result = cartesian_to_fractional(
            positions, cells_inv, fractional_out, batch_idx=batch_idx, device=device
        )

        wp.synchronize_device(device)
        # For cubic cell, fractional = cartesian / L
        pos_np = positions.numpy()
        frac_np = result.numpy()
        np.testing.assert_allclose(frac_np, pos_np / 10.0, rtol=1e-5)

    def test_fractional_to_cartesian_batched(self, dtype, device):
        """Test fractional_to_cartesian with batched systems."""
        num_atoms = 20
        num_systems = 2
        np.random.seed(42)
        np_dtype = np.float32 if dtype == "float32" else np.float64
        vec_dtype = wp.vec3f if dtype == "float32" else wp.vec3d

        fractional = wp.array(
            np.random.rand(num_atoms, 3).astype(np_dtype),
            dtype=vec_dtype,
            device=device,
        )
        batch_idx = wp.array([0] * 10 + [1] * 10, dtype=wp.int32, device=device)

        cell_np = np.diag([10.0, 10.0, 10.0]).astype(np_dtype)
        cells = make_cells_batch(np.stack([cell_np] * num_systems), dtype, device)

        positions_out = wp.empty(num_atoms, dtype=vec_dtype, device=device)
        result = fractional_to_cartesian(
            fractional, cells, positions_out, batch_idx=batch_idx, device=device
        )

        wp.synchronize_device(device)
        # For cubic cell, cartesian = fractional * L
        frac_np = fractional.numpy()
        cart_np = result.numpy()
        np.testing.assert_allclose(cart_np, frac_np * 10.0, rtol=1e-5)

    def test_scale_positions_batched(self, dtype, device):
        """Test scale_positions_with_cell with batched systems."""
        num_atoms = 20
        num_systems = 2
        np.random.seed(42)
        np_dtype = np.float32 if dtype == "float32" else np.float64
        vec_dtype = wp.vec3f if dtype == "float32" else wp.vec3d
        mat_dtype = wp.mat33f if dtype == "float32" else wp.mat33d

        positions = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype) * 5.0,
            dtype=vec_dtype,
            device=device,
        )
        batch_idx = wp.array([0] * 10 + [1] * 10, dtype=wp.int32, device=device)

        cell_old_np = np.diag([10.0, 10.0, 10.0]).astype(np_dtype)
        cell_new_np = np.diag([12.0, 12.0, 12.0]).astype(np_dtype)

        cells_old = make_cells_batch(
            np.stack([cell_old_np] * num_systems), dtype, device
        )
        cells_new = make_cells_batch(
            np.stack([cell_new_np] * num_systems), dtype, device
        )

        cells_old_inv = wp.empty(num_systems, dtype=mat_dtype, device=device)
        compute_cell_inverse(cells_old, cells_old_inv, device=device)

        pos_orig = positions.numpy().copy()
        scale_positions_with_cell(
            positions,
            cells_new,
            cells_old_inv,
            batch_idx=batch_idx,
            device=device,
        )

        wp.synchronize_device(device)
        # Positions should scale by 12/10 = 1.2
        np.testing.assert_allclose(positions.numpy(), pos_orig * 1.2, rtol=1e-5)

    def test_scale_positions_out_batched(self, dtype, device):
        """Test scale_positions_with_cell_out with batched systems."""
        num_atoms = 20
        num_systems = 2
        np.random.seed(42)
        np_dtype = np.float32 if dtype == "float32" else np.float64
        vec_dtype = wp.vec3f if dtype == "float32" else wp.vec3d
        mat_dtype = wp.mat33f if dtype == "float32" else wp.mat33d

        positions = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype) * 5.0,
            dtype=vec_dtype,
            device=device,
        )
        batch_idx = wp.array([0] * 10 + [1] * 10, dtype=wp.int32, device=device)

        cell_old_np = np.diag([10.0, 10.0, 10.0]).astype(np_dtype)
        cell_new_np = np.diag([12.0, 12.0, 12.0]).astype(np_dtype)

        cells_old = make_cells_batch(
            np.stack([cell_old_np] * num_systems), dtype, device
        )
        cells_new = make_cells_batch(
            np.stack([cell_new_np] * num_systems), dtype, device
        )

        cells_old_inv = wp.empty(num_systems, dtype=mat_dtype, device=device)
        compute_cell_inverse(cells_old, cells_old_inv, device=device)

        pos_orig = positions.numpy().copy()
        positions_out = wp.empty(num_atoms, dtype=vec_dtype, device=device)
        result = scale_positions_with_cell_out(
            positions,
            cells_new,
            cells_old_inv,
            positions_out,
            batch_idx=batch_idx,
            device=device,
        )

        wp.synchronize_device(device)
        # Original preserved
        np.testing.assert_allclose(positions.numpy(), pos_orig)
        # Output scaled
        np.testing.assert_allclose(result.numpy(), pos_orig * 1.2, rtol=1e-5)


# ==============================================================================
# Coverage Tests - Pre-allocated Outputs
# ==============================================================================


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
class TestCellUtilsPreallocatedOutputs:
    """Coverage tests for pre-allocated output arrays."""

    def test_compute_cell_inverse_preallocated(self, dtype, device):
        """Test compute_cell_inverse with pre-allocated output."""
        np_dtype = np.float32 if dtype == "float32" else np.float64
        mat_dtype = wp.mat33f if dtype == "float32" else wp.mat33d
        cell_np = np.diag([10.0, 10.0, 10.0]).astype(np_dtype)
        cells = make_cell(cell_np, dtype, device)

        # Pre-allocate output
        cells_inv = wp.empty(1, dtype=mat_dtype, device=device)

        result = compute_cell_inverse(cells, cells_inv, device=device)

        wp.synchronize_device(device)
        assert result is cells_inv
        inv_np = np.array(cells_inv.numpy()[0]).reshape(3, 3)
        expected = np.linalg.inv(cell_np)
        np.testing.assert_allclose(inv_np, expected, rtol=1e-5)

    def test_cartesian_to_fractional_preallocated(self, dtype, device):
        """Test cartesian_to_fractional with pre-allocated output."""
        num_atoms = 10
        np.random.seed(42)
        np_dtype = np.float32 if dtype == "float32" else np.float64
        vec_dtype = wp.vec3f if dtype == "float32" else wp.vec3d
        mat_dtype = wp.mat33f if dtype == "float32" else wp.mat33d

        positions = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype) * 5.0,
            dtype=vec_dtype,
            device=device,
        )
        cell_np = np.diag([10.0, 10.0, 10.0]).astype(np_dtype)
        cells = make_cell(cell_np, dtype, device)

        # Pre-compute inverse
        cells_inv = wp.empty(1, dtype=mat_dtype, device=device)
        compute_cell_inverse(cells, cells_inv, device=device)

        # Pre-allocate output
        fractional = wp.empty(num_atoms, dtype=vec_dtype, device=device)

        result = cartesian_to_fractional(
            positions, cells_inv, fractional, device=device
        )

        wp.synchronize_device(device)
        assert result is fractional
        np.testing.assert_allclose(
            fractional.numpy(), positions.numpy() / 10.0, rtol=1e-5
        )

    def test_fractional_to_cartesian_preallocated(self, dtype, device):
        """Test fractional_to_cartesian with pre-allocated output."""
        num_atoms = 10
        np.random.seed(42)
        np_dtype = np.float32 if dtype == "float32" else np.float64
        vec_dtype = wp.vec3f if dtype == "float32" else wp.vec3d

        fractional = wp.array(
            np.random.rand(num_atoms, 3).astype(np_dtype),
            dtype=vec_dtype,
            device=device,
        )
        cell_np = np.diag([10.0, 10.0, 10.0]).astype(np_dtype)
        cells = make_cell(cell_np, dtype, device)

        # Pre-allocate output
        positions = wp.empty(num_atoms, dtype=vec_dtype, device=device)

        result = fractional_to_cartesian(fractional, cells, positions, device=device)

        wp.synchronize_device(device)
        assert result is positions
        np.testing.assert_allclose(
            positions.numpy(), fractional.numpy() * 10.0, rtol=1e-5
        )

    def test_wrap_positions_out_preallocated(self, dtype, device):
        """Test wrap_positions_to_cell_out with pre-allocated output."""
        num_atoms = 10
        np.random.seed(42)
        np_dtype = np.float32 if dtype == "float32" else np.float64
        vec_dtype = wp.vec3f if dtype == "float32" else wp.vec3d
        mat_dtype = wp.mat33f if dtype == "float32" else wp.mat33d

        positions = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype) * 20.0,
            dtype=vec_dtype,
            device=device,
        )
        cell_np = np.diag([10.0, 10.0, 10.0]).astype(np_dtype)
        cells = make_cell(cell_np, dtype, device)

        # Pre-compute inverse
        cells_inv = wp.empty(1, dtype=mat_dtype, device=device)
        compute_cell_inverse(cells, cells_inv, device=device)

        # Pre-allocate output
        positions_out = wp.empty(num_atoms, dtype=vec_dtype, device=device)

        result = wrap_positions_to_cell_out(
            positions, cells, cells_inv, positions_out, device=device
        )

        wp.synchronize_device(device)
        assert result is positions_out

    def test_scale_positions_out_preallocated(self, dtype, device):
        """Test scale_positions_with_cell_out with pre-allocated output."""
        num_atoms = 10
        np.random.seed(42)
        np_dtype = np.float32 if dtype == "float32" else np.float64
        vec_dtype = wp.vec3f if dtype == "float32" else wp.vec3d
        mat_dtype = wp.mat33f if dtype == "float32" else wp.mat33d

        positions = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype) * 5.0,
            dtype=vec_dtype,
            device=device,
        )
        cell_old_np = np.diag([10.0, 10.0, 10.0]).astype(np_dtype)
        cell_new_np = np.diag([12.0, 12.0, 12.0]).astype(np_dtype)
        cells_old = make_cell(cell_old_np, dtype, device)
        cells_new = make_cell(cell_new_np, dtype, device)

        # Pre-compute inverse of old cell
        cells_old_inv = wp.empty(1, dtype=mat_dtype, device=device)
        compute_cell_inverse(cells_old, cells_old_inv, device=device)

        # Pre-allocate output
        positions_out = wp.empty(num_atoms, dtype=vec_dtype, device=device)

        result = scale_positions_with_cell_out(
            positions,
            cells_new,
            cells_old_inv,
            positions_out,
            device=device,
        )

        wp.synchronize_device(device)
        assert result is positions_out


# ==============================================================================
# Device Inference Tests
# ==============================================================================


class TestDeviceInference:
    """Test device inference when device=None is passed."""

    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    @pytest.mark.parametrize("device", DEVICES)
    def test_compute_cell_volume_device_inference(self, dtype, device):
        """Test compute_cell_volume infers device from cells."""
        np_dtype = np.float32 if dtype == "float32" else np.float64
        scalar_wp = wp.float32 if dtype == "float32" else wp.float64
        cell_np = np.diag([10.0, 10.0, 10.0]).astype(np_dtype)
        cells = make_cell(cell_np, dtype, device)

        # Don't pass device
        volumes = wp.empty(1, dtype=scalar_wp, device=device)
        compute_cell_volume(cells, volumes)

        wp.synchronize_device(device)
        assert volumes.device == device

    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    @pytest.mark.parametrize("device", DEVICES)
    def test_compute_cell_inverse_device_inference(self, dtype, device):
        """Test compute_cell_inverse infers device from cells."""
        np_dtype = np.float32 if dtype == "float32" else np.float64
        mat_dtype = wp.mat33f if dtype == "float32" else wp.mat33d
        cell_np = np.diag([10.0, 10.0, 10.0]).astype(np_dtype)
        cells = make_cell(cell_np, dtype, device)

        # Don't pass device
        cells_inv = wp.empty(1, dtype=mat_dtype, device=device)
        compute_cell_inverse(cells, cells_inv)

        wp.synchronize_device(device)
        assert cells_inv.device == device

    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    @pytest.mark.parametrize("device", DEVICES)
    def test_compute_strain_tensor_device_inference(self, dtype, device):
        """Test compute_strain_tensor infers device from cells."""
        np_dtype = np.float32 if dtype == "float32" else np.float64
        mat_dtype = wp.mat33f if dtype == "float32" else wp.mat33d
        cell_np = np.diag([10.0, 10.0, 10.0]).astype(np_dtype)
        cell_ref_np = np.diag([9.0, 9.0, 9.0]).astype(np_dtype)
        cells = make_cell(cell_np, dtype, device)
        cells_ref = make_cell(cell_ref_np, dtype, device)

        # Pre-compute inverse of reference cell
        cells_ref_inv = wp.empty(1, dtype=mat_dtype, device=device)
        compute_cell_inverse(cells_ref, cells_ref_inv, device=device)

        # Don't pass device
        strains = wp.empty(1, dtype=mat_dtype, device=device)
        compute_strain_tensor(cells, cells_ref_inv, strains)

        wp.synchronize_device(device)
        assert strains.device == device

    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    @pytest.mark.parametrize("device", DEVICES)
    def test_apply_strain_to_cell_device_inference(self, dtype, device):
        """Test apply_strain_to_cell infers device from cells."""
        np_dtype = np.float32 if dtype == "float32" else np.float64
        mat_dtype = wp.mat33f if dtype == "float32" else wp.mat33d

        cell_np = np.diag([10.0, 10.0, 10.0]).astype(np_dtype)
        strain_np = np.eye(3, dtype=np_dtype) * 0.01
        cells = make_cell(cell_np, dtype, device)
        strains = wp.array(
            [mat_dtype(*strain_np.flatten())], dtype=mat_dtype, device=device
        )

        # Don't pass device
        cells_out = wp.empty(1, dtype=mat_dtype, device=device)
        apply_strain_to_cell(cells, strains, cells_out)

        wp.synchronize_device(device)
        assert cells_out.device == device

    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    @pytest.mark.parametrize("device", DEVICES)
    def test_scale_positions_device_inference(self, dtype, device):
        """Test scale_positions_with_cell infers device from positions."""
        num_atoms = 5
        np_dtype = np.float32 if dtype == "float32" else np.float64
        vec_dtype = wp.vec3f if dtype == "float32" else wp.vec3d
        mat_dtype = wp.mat33f if dtype == "float32" else wp.mat33d

        positions = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=vec_dtype,
            device=device,
        )
        cell_old_np = np.diag([10.0, 10.0, 10.0]).astype(np_dtype)
        cell_new_np = np.diag([11.0, 11.0, 11.0]).astype(np_dtype)
        cells_old = make_cell(cell_old_np, dtype, device)
        cells_new = make_cell(cell_new_np, dtype, device)

        cells_old_inv = wp.empty(1, dtype=mat_dtype, device=device)
        compute_cell_inverse(cells_old, cells_old_inv, device=device)

        # Don't pass device
        scale_positions_with_cell(positions, cells_new, cells_old_inv)

        wp.synchronize_device(device)
        # Just check it ran without error

    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    @pytest.mark.parametrize("device", DEVICES)
    def test_wrap_positions_device_inference(self, dtype, device):
        """Test wrap_positions_to_cell infers device from positions."""
        num_atoms = 5
        np_dtype = np.float32 if dtype == "float32" else np.float64
        vec_dtype = wp.vec3f if dtype == "float32" else wp.vec3d
        mat_dtype = wp.mat33f if dtype == "float32" else wp.mat33d

        positions = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype) * 20.0,
            dtype=vec_dtype,
            device=device,
        )
        cell_np = np.diag([10.0, 10.0, 10.0]).astype(np_dtype)
        cells = make_cell(cell_np, dtype, device)

        cells_inv = wp.empty(1, dtype=mat_dtype, device=device)
        compute_cell_inverse(cells, cells_inv, device=device)

        # Don't pass device
        wrap_positions_to_cell(positions, cells, cells_inv)

        wp.synchronize_device(device)
        # Just check it ran without error


# ==============================================================================
# Error Case Tests
# ==============================================================================


class TestErrorCases:
    """Test error handling in cell utilities."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_compute_strain_missing_ref_error(self, device):
        """Test compute_strain_tensor raises TypeError without required args."""
        np_dtype = np.float64
        cell_np = np.diag([10.0, 10.0, 10.0]).astype(np_dtype)
        cells = make_cell(cell_np, "float64", device)

        with pytest.raises(TypeError):
            compute_strain_tensor(cells, device=device)

    @pytest.mark.parametrize("device", DEVICES)
    def test_scale_positions_missing_old_cell_error(self, device):
        """Test scale_positions_with_cell raises TypeError without required args."""
        num_atoms = 5
        np_dtype = np.float64
        vec_dtype = wp.vec3d

        positions = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=vec_dtype,
            device=device,
        )
        cell_new_np = np.diag([11.0, 11.0, 11.0]).astype(np_dtype)
        cells_new = make_cell(cell_new_np, "float64", device)

        with pytest.raises(TypeError):
            scale_positions_with_cell(positions, cells_new, device=device)

    @pytest.mark.parametrize("device", DEVICES)
    def test_wrap_positions_missing_cells_error(self, device):
        """Test wrap_positions_to_cell raises TypeError without required args."""
        num_atoms = 5
        np_dtype = np.float64
        vec_dtype = wp.vec3d

        positions = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=vec_dtype,
            device=device,
        )

        with pytest.raises(TypeError):
            wrap_positions_to_cell(positions, device=device)

    @pytest.mark.parametrize("device", DEVICES)
    def test_wrap_positions_out_missing_cells_error(self, device):
        """Test wrap_positions_to_cell_out raises TypeError without required args."""
        num_atoms = 5
        np_dtype = np.float64
        vec_dtype = wp.vec3d

        positions = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=vec_dtype,
            device=device,
        )

        with pytest.raises(TypeError):
            wrap_positions_to_cell_out(positions, device=device)

    @pytest.mark.parametrize("device", DEVICES)
    def test_scale_positions_out_missing_old_cell_error(self, device):
        """Test scale_positions_with_cell_out raises TypeError without required args."""
        num_atoms = 5
        np_dtype = np.float64
        vec_dtype = wp.vec3d

        positions = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=vec_dtype,
            device=device,
        )
        cell_new_np = np.diag([11.0, 11.0, 11.0]).astype(np_dtype)
        cells_new = make_cell(cell_new_np, "float64", device)

        with pytest.raises(TypeError):
            scale_positions_with_cell_out(positions, cells_new=cells_new, device=device)
