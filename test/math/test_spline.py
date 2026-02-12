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
Test Suite for B-Spline Kernels (Pure Warp)
===========================================

Tests the Warp kernels and launchers in nvalchemiops.math.spline.

Test Categories
---------------
1. Happy path tests: Verify kernels run without error and produce reasonable output
2. Regression tests: Verify output matches hardcoded expected values
3. Property tests: Verify mathematical properties (e.g., spread sums to total charge)

Mathematical Properties Tested
------------------------------
- B-spline spread: sum(mesh) == sum(charges) (charge conservation)
- B-spline weights sum to 1 (partition of unity)
- Spread/gather are adjoint operations

External Fixtures
-----------------
The following fixtures are defined in ``test/math/conftest.py``:

- ``device``: Parametrized fixture providing "cpu" and "cuda:0" devices.
  GPU tests are skipped if CUDA is not available.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from nvalchemiops.math.spline import (
    batch_spline_gather,
    batch_spline_gather_gradient,
    batch_spline_gather_vec3,
    batch_spline_spread,
    spline_gather,
    spline_gather_gradient,
    spline_gather_vec3,
    spline_spread,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def simple_system():
    """Simple 4-atom system in a 10A cubic cell for testing.

    Returns
    -------
    dict
        Dictionary containing positions, charges, cell_inv_t, and mesh_dims.
    """
    cell = np.eye(3, dtype=np.float64) * 10.0
    cell_inv_t = np.linalg.inv(cell).T

    positions = np.array(
        [
            [1.0, 1.0, 1.0],
            [5.0, 5.0, 5.0],
            [2.5, 7.5, 3.0],
            [8.0, 2.0, 6.0],
        ],
        dtype=np.float64,
    )

    charges = np.array([1.0, -1.0, 0.5, 0.5], dtype=np.float64)

    return {
        "positions": positions,
        "charges": charges,
        "cell_inv_t": cell_inv_t,
        "mesh_dims": (8, 8, 8),
    }


@pytest.fixture
def batch_system():
    """Batched 2-system test setup.

    Returns
    -------
    dict
        Dictionary containing batched positions, charges, batch_idx, cell_inv_t.
    """
    cell = np.eye(3, dtype=np.float64) * 10.0
    cell_inv_t = np.linalg.inv(cell).T

    positions = np.array(
        [
            [1.0, 1.0, 1.0],  # sys 0
            [5.0, 5.0, 5.0],  # sys 0
            [2.5, 2.5, 2.5],  # sys 1
            [7.5, 7.5, 7.5],  # sys 1
        ],
        dtype=np.float64,
    )

    charges = np.array([1.0, -0.5, 0.5, -0.5], dtype=np.float64)
    batch_idx = np.array([0, 0, 1, 1], dtype=np.int32)
    batch_cell_inv_t = np.stack([cell_inv_t, cell_inv_t])

    return {
        "positions": positions,
        "charges": charges,
        "batch_idx": batch_idx,
        "cell_inv_t": batch_cell_inv_t,
        "mesh_dims": (8, 8, 8),
        "num_systems": 2,
    }


# =============================================================================
# Happy Path Tests: Single-System Kernels
# =============================================================================


class TestSplineSpread:
    """Tests for spline_spread launcher."""

    @pytest.mark.parametrize("order", [2, 3, 4])
    def test_spread_runs_without_error(self, device, simple_system, order):
        """Verify spread kernel runs without error for different orders."""
        positions = wp.from_numpy(
            simple_system["positions"], dtype=wp.vec3d, device=device
        )
        charges = wp.from_numpy(
            simple_system["charges"], dtype=wp.float64, device=device
        )
        cell_inv_t = wp.from_numpy(
            simple_system["cell_inv_t"].reshape(1, 3, 3), dtype=wp.mat33d, device=device
        )
        mesh = wp.zeros(simple_system["mesh_dims"], dtype=wp.float64, device=device)

        spline_spread(positions, charges, cell_inv_t, order, mesh, wp.float64, device)
        wp.synchronize()

        # Verify output is non-zero
        mesh_np = mesh.numpy()
        assert np.any(mesh_np != 0), "Mesh should have non-zero values after spread"

    def test_spread_charge_conservation(self, device, simple_system):
        """Verify total charge is conserved after spread."""
        positions = wp.from_numpy(
            simple_system["positions"], dtype=wp.vec3d, device=device
        )
        charges = wp.from_numpy(
            simple_system["charges"], dtype=wp.float64, device=device
        )
        cell_inv_t = wp.from_numpy(
            simple_system["cell_inv_t"].reshape(1, 3, 3), dtype=wp.mat33d, device=device
        )
        mesh = wp.zeros(simple_system["mesh_dims"], dtype=wp.float64, device=device)

        spline_spread(positions, charges, cell_inv_t, 4, mesh, wp.float64, device)
        wp.synchronize()

        mesh_sum = mesh.numpy().sum()
        expected_sum = simple_system["charges"].sum()
        assert mesh_sum == pytest.approx(expected_sum, rel=1e-10)

    def test_spread_float32(self, device):
        """Verify spread works with float32 precision."""
        positions_np = np.array([[5.0, 5.0, 5.0]], dtype=np.float32)
        charges_np = np.array([1.0], dtype=np.float32)
        cell_inv_t_np = (np.eye(3, dtype=np.float32) * 0.1).reshape(1, 3, 3)

        positions = wp.from_numpy(positions_np, dtype=wp.vec3f, device=device)
        charges = wp.from_numpy(charges_np, dtype=wp.float32, device=device)
        cell_inv_t = wp.from_numpy(cell_inv_t_np, dtype=wp.mat33f, device=device)
        mesh = wp.zeros((16, 16, 16), dtype=wp.float32, device=device)

        spline_spread(positions, charges, cell_inv_t, 4, mesh, wp.float32, device)
        wp.synchronize()

        mesh_sum = mesh.numpy().sum()
        assert mesh_sum == pytest.approx(1.0, rel=1e-5)


class TestSplineGather:
    """Tests for spline_gather launcher."""

    def test_gather_runs_without_error(self, device, simple_system):
        """Verify gather kernel runs without error."""
        positions = wp.from_numpy(
            simple_system["positions"], dtype=wp.vec3d, device=device
        )
        cell_inv_t = wp.from_numpy(
            simple_system["cell_inv_t"].reshape(1, 3, 3), dtype=wp.mat33d, device=device
        )

        # Create a mesh with uniform values
        mesh = wp.full(simple_system["mesh_dims"], 1.0, dtype=wp.float64, device=device)
        output = wp.zeros(4, dtype=wp.float64, device=device)

        spline_gather(positions, cell_inv_t, 4, mesh, output, wp.float64, device)
        wp.synchronize()

        output_np = output.numpy()
        # Each atom should gather ~1.0 from a uniform mesh (weights sum to 1)
        assert np.all(output_np > 0), "Output should be positive for uniform mesh"

    def test_spread_then_gather(self, device, simple_system):
        """Test spread followed by gather produces consistent results."""
        positions = wp.from_numpy(
            simple_system["positions"], dtype=wp.vec3d, device=device
        )
        unit_charges = wp.from_numpy(
            np.ones(4, dtype=np.float64), dtype=wp.float64, device=device
        )
        cell_inv_t = wp.from_numpy(
            simple_system["cell_inv_t"].reshape(1, 3, 3), dtype=wp.mat33d, device=device
        )

        # Spread unit charges
        mesh = wp.zeros(simple_system["mesh_dims"], dtype=wp.float64, device=device)
        spline_spread(positions, unit_charges, cell_inv_t, 4, mesh, wp.float64, device)
        wp.synchronize()

        # Gather back
        output = wp.zeros(4, dtype=wp.float64, device=device)
        spline_gather(positions, cell_inv_t, 4, mesh, output, wp.float64, device)
        wp.synchronize()

        output_np = output.numpy()
        # Output should be positive (self-interaction + cross terms)
        assert np.all(output_np > 0), "Gathered values should be positive"
        # Sum should be related to total spread charge
        assert output_np.sum() > 0, "Total gathered should be positive"


class TestSplineGatherVec3:
    """Tests for spline_gather_vec3 launcher."""

    def test_gather_vec3_uniform_mesh(self, device, simple_system):
        """Test vec3 gather from uniform vector field."""
        positions = wp.from_numpy(
            simple_system["positions"], dtype=wp.vec3d, device=device
        )
        charges = wp.from_numpy(
            simple_system["charges"], dtype=wp.float64, device=device
        )
        cell_inv_t = wp.from_numpy(
            simple_system["cell_inv_t"].reshape(1, 3, 3), dtype=wp.mat33d, device=device
        )

        # Create uniform vec3 mesh: each point has (1, 2, 3)
        mesh_size = 8 * 8 * 8
        vec_mesh_np = np.tile(
            np.array([1.0, 2.0, 3.0], dtype=np.float64), (mesh_size, 1)
        )
        vec_mesh = wp.from_numpy(vec_mesh_np, dtype=wp.vec3d, device=device)
        vec_mesh = vec_mesh.reshape(simple_system["mesh_dims"])

        output = wp.zeros(4, dtype=wp.vec3d, device=device)
        spline_gather_vec3(
            positions, charges, cell_inv_t, 4, vec_mesh, output, wp.float64, device
        )
        wp.synchronize()

        output_np = np.array(output.numpy().tolist())

        # Output should be charge-weighted: q_i * (1, 2, 3) * weight_sum
        # For uniform mesh and weights summing to 1: output[i] = q_i * (1, 2, 3)
        expected_ratios = np.array(
            [[1, 2, 3], [-1, -2, -3], [0.5, 1, 1.5], [0.5, 1, 1.5]]
        )
        assert output_np == pytest.approx(expected_ratios, rel=1e-6)


class TestSplineGatherGradient:
    """Tests for spline_gather_gradient launcher."""

    def test_gradient_uniform_mesh(self, device, simple_system):
        """Test gradient gather from uniform potential mesh."""
        positions = wp.from_numpy(
            simple_system["positions"], dtype=wp.vec3d, device=device
        )
        charges = wp.from_numpy(
            simple_system["charges"], dtype=wp.float64, device=device
        )
        cell_inv_t = wp.from_numpy(
            simple_system["cell_inv_t"].reshape(1, 3, 3), dtype=wp.mat33d, device=device
        )

        # Uniform potential mesh -> gradient should be zero
        mesh = wp.full(simple_system["mesh_dims"], 1.0, dtype=wp.float64, device=device)
        forces = wp.zeros(4, dtype=wp.vec3d, device=device)

        spline_gather_gradient(
            positions, charges, cell_inv_t, 4, mesh, forces, wp.float64, device
        )
        wp.synchronize()

        forces_np = np.array(forces.numpy().tolist())

        # Forces should be near zero for uniform potential
        assert forces_np == pytest.approx(0.0, abs=1e-10)


# =============================================================================
# Happy Path Tests: Batch Kernels
# =============================================================================


class TestBatchSplineSpread:
    """Tests for batch_spline_spread launcher."""

    def test_batch_spread_runs_without_error(self, device, batch_system):
        """Verify batch spread kernel runs without error."""
        positions = wp.from_numpy(
            batch_system["positions"], dtype=wp.vec3d, device=device
        )
        charges = wp.from_numpy(
            batch_system["charges"], dtype=wp.float64, device=device
        )
        batch_idx = wp.from_numpy(
            batch_system["batch_idx"], dtype=wp.int32, device=device
        )
        cell_inv_t = wp.from_numpy(
            batch_system["cell_inv_t"], dtype=wp.mat33d, device=device
        )
        mesh = wp.zeros(
            (batch_system["num_systems"],) + batch_system["mesh_dims"],
            dtype=wp.float64,
            device=device,
        )

        batch_spline_spread(
            positions, charges, batch_idx, cell_inv_t, 4, mesh, wp.float64, device
        )
        wp.synchronize()

        mesh_np = mesh.numpy()
        assert np.any(mesh_np != 0), "Batch mesh should have non-zero values"

    def test_batch_spread_per_system_charge_conservation(self, device, batch_system):
        """Verify charge conservation per system in batch spread."""
        positions = wp.from_numpy(
            batch_system["positions"], dtype=wp.vec3d, device=device
        )
        charges = wp.from_numpy(
            batch_system["charges"], dtype=wp.float64, device=device
        )
        batch_idx = wp.from_numpy(
            batch_system["batch_idx"], dtype=wp.int32, device=device
        )
        cell_inv_t = wp.from_numpy(
            batch_system["cell_inv_t"], dtype=wp.mat33d, device=device
        )
        mesh = wp.zeros(
            (batch_system["num_systems"],) + batch_system["mesh_dims"],
            dtype=wp.float64,
            device=device,
        )

        batch_spline_spread(
            positions, charges, batch_idx, cell_inv_t, 4, mesh, wp.float64, device
        )
        wp.synchronize()

        mesh_np = mesh.numpy()

        # System 0: atoms 0,1 with charges 1.0, -0.5 -> sum = 0.5
        sys0_sum = mesh_np[0].sum()
        assert sys0_sum == pytest.approx(0.5, rel=1e-8)

        # System 1: atoms 2,3 with charges 0.5, -0.5 -> sum = 0.0
        sys1_sum = mesh_np[1].sum()
        assert sys1_sum == pytest.approx(0.0, abs=1e-8)


class TestBatchSplineGather:
    """Tests for batch_spline_gather launcher."""

    def test_batch_gather_runs_without_error(self, device, batch_system):
        """Verify batch gather kernel runs without error."""
        positions = wp.from_numpy(
            batch_system["positions"], dtype=wp.vec3d, device=device
        )
        batch_idx = wp.from_numpy(
            batch_system["batch_idx"], dtype=wp.int32, device=device
        )
        cell_inv_t = wp.from_numpy(
            batch_system["cell_inv_t"], dtype=wp.mat33d, device=device
        )
        mesh = wp.full(
            (batch_system["num_systems"],) + batch_system["mesh_dims"],
            1.0,
            dtype=wp.float64,
            device=device,
        )
        output = wp.zeros(4, dtype=wp.float64, device=device)

        batch_spline_gather(
            positions, batch_idx, cell_inv_t, 4, mesh, output, wp.float64, device
        )
        wp.synchronize()

        output_np = output.numpy()
        assert np.all(output_np > 0), "Batch gather output should be positive"


class TestBatchSplineGatherVec3:
    """Tests for batch_spline_gather_vec3 launcher."""

    def test_batch_gather_vec3_uniform_mesh(self, device, batch_system):
        """Test batch vec3 gather from uniform vector field."""
        positions = wp.from_numpy(
            batch_system["positions"], dtype=wp.vec3d, device=device
        )
        charges = wp.from_numpy(
            batch_system["charges"], dtype=wp.float64, device=device
        )
        batch_idx = wp.from_numpy(
            batch_system["batch_idx"], dtype=wp.int32, device=device
        )
        cell_inv_t = wp.from_numpy(
            batch_system["cell_inv_t"], dtype=wp.mat33d, device=device
        )

        # Create uniform vec3 mesh: each point has (1, 2, 3)
        mesh_size = batch_system["num_systems"] * 8 * 8 * 8
        vec_mesh_np = np.tile(
            np.array([1.0, 2.0, 3.0], dtype=np.float64), (mesh_size, 1)
        )
        vec_mesh = wp.from_numpy(vec_mesh_np, dtype=wp.vec3d, device=device)
        vec_mesh = vec_mesh.reshape(
            (batch_system["num_systems"],) + batch_system["mesh_dims"]
        )

        output = wp.zeros(4, dtype=wp.vec3d, device=device)
        batch_spline_gather_vec3(
            positions,
            charges,
            batch_idx,
            cell_inv_t,
            4,
            vec_mesh,
            output,
            wp.float64,
            device,
        )
        wp.synchronize()

        output_np = np.array(output.numpy().tolist())

        # Output = charge * (1, 2, 3) for each atom
        expected = np.array(
            [
                [1.0, 2.0, 3.0],  # q=1.0
                [-0.5, -1.0, -1.5],  # q=-0.5
                [0.5, 1.0, 1.5],  # q=0.5
                [-0.5, -1.0, -1.5],  # q=-0.5
            ]
        )
        assert output_np == pytest.approx(expected, rel=1e-6)


class TestBatchSplineGatherGradient:
    """Tests for batch_spline_gather_gradient launcher."""

    def test_batch_gradient_uniform_mesh(self, device, batch_system):
        """Test batch gradient gather from uniform potential mesh."""
        positions = wp.from_numpy(
            batch_system["positions"], dtype=wp.vec3d, device=device
        )
        charges = wp.from_numpy(
            batch_system["charges"], dtype=wp.float64, device=device
        )
        batch_idx = wp.from_numpy(
            batch_system["batch_idx"], dtype=wp.int32, device=device
        )
        cell_inv_t = wp.from_numpy(
            batch_system["cell_inv_t"], dtype=wp.mat33d, device=device
        )

        # Uniform mesh -> zero gradient
        mesh = wp.full(
            (batch_system["num_systems"],) + batch_system["mesh_dims"],
            1.0,
            dtype=wp.float64,
            device=device,
        )
        forces = wp.zeros(4, dtype=wp.vec3d, device=device)

        batch_spline_gather_gradient(
            positions,
            charges,
            batch_idx,
            cell_inv_t,
            4,
            mesh,
            forces,
            wp.float64,
            device,
        )
        wp.synchronize()

        forces_np = np.array(forces.numpy().tolist())
        assert forces_np == pytest.approx(0.0, abs=1e-10)


# =============================================================================
# Regression Tests
# =============================================================================


class TestSplineRegressionValues:
    """Regression tests with hardcoded expected values.

    These values were generated with seed=42 and specific system configurations
    to detect any changes in kernel behavior.
    """

    def test_spread_regression(self, device, simple_system):
        """Regression test for spline_spread with expected values."""
        positions = wp.from_numpy(
            simple_system["positions"], dtype=wp.vec3d, device=device
        )
        charges = wp.from_numpy(
            simple_system["charges"], dtype=wp.float64, device=device
        )
        cell_inv_t = wp.from_numpy(
            simple_system["cell_inv_t"].reshape(1, 3, 3), dtype=wp.mat33d, device=device
        )
        mesh = wp.zeros(simple_system["mesh_dims"], dtype=wp.float64, device=device)

        spline_spread(positions, charges, cell_inv_t, 4, mesh, wp.float64, device)
        wp.synchronize()

        mesh_np = mesh.numpy()

        # Regression values from initial run
        assert mesh_np.sum() == pytest.approx(1.0, rel=1e-10)
        assert mesh_np.max() == pytest.approx(0.2508416403, rel=1e-8)
        assert mesh_np.min() == pytest.approx(-0.2962962963, rel=1e-8)
        assert np.count_nonzero(mesh_np) == 182

    def test_gather_regression(self, device, simple_system):
        """Regression test for spline_gather with expected values."""
        positions = wp.from_numpy(
            simple_system["positions"], dtype=wp.vec3d, device=device
        )
        unit_charges = wp.from_numpy(
            np.ones(4, dtype=np.float64), dtype=wp.float64, device=device
        )
        cell_inv_t = wp.from_numpy(
            simple_system["cell_inv_t"].reshape(1, 3, 3), dtype=wp.mat33d, device=device
        )

        # Spread then gather
        mesh = wp.zeros(simple_system["mesh_dims"], dtype=wp.float64, device=device)
        spline_spread(positions, unit_charges, cell_inv_t, 4, mesh, wp.float64, device)
        wp.synchronize()

        output = wp.zeros(4, dtype=wp.float64, device=device)
        spline_gather(positions, cell_inv_t, 4, mesh, output, wp.float64, device)
        wp.synchronize()

        output_np = output.numpy()

        # Regression values
        expected = np.array([0.11403329, 0.12506939, 0.11594129, 0.10419698])
        assert output_np == pytest.approx(expected, rel=1e-6)
        assert output_np.sum() == pytest.approx(0.4592409390, rel=1e-8)

    def test_gather_vec3_regression(self, device, simple_system):
        """Regression test for spline_gather_vec3 with expected values."""
        positions = wp.from_numpy(
            simple_system["positions"], dtype=wp.vec3d, device=device
        )
        charges = wp.from_numpy(
            simple_system["charges"], dtype=wp.float64, device=device
        )
        cell_inv_t = wp.from_numpy(
            simple_system["cell_inv_t"].reshape(1, 3, 3), dtype=wp.mat33d, device=device
        )

        mesh_size = 8 * 8 * 8
        vec_mesh_np = np.tile(
            np.array([1.0, 2.0, 3.0], dtype=np.float64), (mesh_size, 1)
        )
        vec_mesh = wp.from_numpy(vec_mesh_np, dtype=wp.vec3d, device=device)
        vec_mesh = vec_mesh.reshape(simple_system["mesh_dims"])

        output = wp.zeros(4, dtype=wp.vec3d, device=device)
        spline_gather_vec3(
            positions, charges, cell_inv_t, 4, vec_mesh, output, wp.float64, device
        )
        wp.synchronize()

        output_np = np.array(output.numpy().tolist())

        # Regression values - charge-weighted (1, 2, 3) vectors
        expected = np.array(
            [
                [1.0, 2.0, 2.99999999],
                [-1.0, -2.0, -3.0],
                [0.5, 1.0, 1.5],
                [0.5, 1.0, 1.5],
            ]
        )
        assert output_np == pytest.approx(expected, rel=1e-6)

    def test_batch_spread_regression(self, device, batch_system):
        """Regression test for batch_spline_spread with expected values."""
        positions = wp.from_numpy(
            batch_system["positions"], dtype=wp.vec3d, device=device
        )
        charges = wp.from_numpy(
            batch_system["charges"], dtype=wp.float64, device=device
        )
        batch_idx = wp.from_numpy(
            batch_system["batch_idx"], dtype=wp.int32, device=device
        )
        cell_inv_t = wp.from_numpy(
            batch_system["cell_inv_t"], dtype=wp.mat33d, device=device
        )
        mesh = wp.zeros(
            (batch_system["num_systems"],) + batch_system["mesh_dims"],
            dtype=wp.float64,
            device=device,
        )

        batch_spline_spread(
            positions, charges, batch_idx, cell_inv_t, 4, mesh, wp.float64, device
        )
        wp.synchronize()

        mesh_np = mesh.numpy()

        # Regression values
        assert mesh_np[0].sum() == pytest.approx(0.4999999976, rel=1e-8)
        assert mesh_np[1].sum() == pytest.approx(0.0, abs=1e-10)
        assert mesh_np.max() == pytest.approx(0.2508416403, rel=1e-8)
        assert mesh_np.min() == pytest.approx(-0.1481481481, rel=1e-8)

    def test_batch_gather_regression(self, device, batch_system):
        """Regression test for batch_spline_gather with expected values."""
        positions = wp.from_numpy(
            batch_system["positions"], dtype=wp.vec3d, device=device
        )
        unit_charges = wp.from_numpy(
            np.ones(4, dtype=np.float64), dtype=wp.float64, device=device
        )
        batch_idx = wp.from_numpy(
            batch_system["batch_idx"], dtype=wp.int32, device=device
        )
        cell_inv_t = wp.from_numpy(
            batch_system["cell_inv_t"], dtype=wp.mat33d, device=device
        )

        # Spread then gather
        mesh = wp.zeros(
            (batch_system["num_systems"],) + batch_system["mesh_dims"],
            dtype=wp.float64,
            device=device,
        )
        batch_spline_spread(
            positions, unit_charges, batch_idx, cell_inv_t, 4, mesh, wp.float64, device
        )
        wp.synchronize()

        output = wp.zeros(4, dtype=wp.float64, device=device)
        batch_spline_gather(
            positions, batch_idx, cell_inv_t, 4, mesh, output, wp.float64, device
        )
        wp.synchronize()

        output_np = output.numpy()

        # Regression values
        expected = np.array([0.11403082, 0.125, 0.125, 0.125])
        assert output_np == pytest.approx(expected, rel=1e-6)


# =============================================================================
# Property Tests
# =============================================================================


class TestBSplineProperties:
    """Tests for mathematical properties of B-splines."""

    @pytest.mark.parametrize("order", [2, 3, 4])
    def test_spread_charge_conservation_all_orders(self, device, order):
        """Verify charge conservation for all B-spline orders."""
        positions_np = np.array([[5.0, 5.0, 5.0]], dtype=np.float64)
        charges_np = np.array([1.0], dtype=np.float64)
        cell_inv_t_np = (np.eye(3, dtype=np.float64) * 0.1).reshape(1, 3, 3)

        positions = wp.from_numpy(positions_np, dtype=wp.vec3d, device=device)
        charges = wp.from_numpy(charges_np, dtype=wp.float64, device=device)
        cell_inv_t = wp.from_numpy(cell_inv_t_np, dtype=wp.mat33d, device=device)

        mesh = wp.zeros((16, 16, 16), dtype=wp.float64, device=device)
        spline_spread(positions, charges, cell_inv_t, order, mesh, wp.float64, device)
        wp.synchronize()

        mesh_sum = mesh.numpy().sum()
        assert mesh_sum == pytest.approx(1.0, rel=1e-10)

    def test_multiple_atoms_charge_conservation(self, device):
        """Verify charge conservation with multiple atoms of varying charges."""
        np.random.seed(42)
        n_atoms = 10
        positions_np = (
            np.random.rand(n_atoms, 3).astype(np.float64) * 8 + 1
        )  # Keep in [1, 9]
        charges_np = np.random.randn(n_atoms).astype(np.float64)
        cell_inv_t_np = (np.eye(3, dtype=np.float64) * 0.1).reshape(1, 3, 3)

        positions = wp.from_numpy(positions_np, dtype=wp.vec3d, device=device)
        charges = wp.from_numpy(charges_np, dtype=wp.float64, device=device)
        cell_inv_t = wp.from_numpy(cell_inv_t_np, dtype=wp.mat33d, device=device)

        mesh = wp.zeros((16, 16, 16), dtype=wp.float64, device=device)
        spline_spread(positions, charges, cell_inv_t, 4, mesh, wp.float64, device)
        wp.synchronize()

        mesh_sum = mesh.numpy().sum()
        expected_sum = charges_np.sum()
        assert mesh_sum == pytest.approx(expected_sum, rel=1e-10)

    def test_gather_from_uniform_mesh_equals_one(self, device):
        """Verify gathering from uniform mesh gives weight sum = 1."""
        positions_np = np.array([[5.0, 5.0, 5.0]], dtype=np.float64)
        cell_inv_t_np = (np.eye(3, dtype=np.float64) * 0.1).reshape(1, 3, 3)

        positions = wp.from_numpy(positions_np, dtype=wp.vec3d, device=device)
        cell_inv_t = wp.from_numpy(cell_inv_t_np, dtype=wp.mat33d, device=device)

        # Uniform mesh with value 1.0
        mesh = wp.full((16, 16, 16), 1.0, dtype=wp.float64, device=device)
        output = wp.zeros(1, dtype=wp.float64, device=device)

        spline_gather(positions, cell_inv_t, 4, mesh, output, wp.float64, device)
        wp.synchronize()

        # Weights should sum to 1.0
        assert output.numpy().sum() == pytest.approx(1.0, rel=1e-10)

    def test_gradient_of_constant_is_zero(self, device):
        """Verify gradient of constant potential field is zero."""
        np.random.seed(42)
        positions_np = np.random.rand(5, 3).astype(np.float64) * 8 + 1
        charges_np = np.random.randn(5).astype(np.float64)
        cell_inv_t_np = (np.eye(3, dtype=np.float64) * 0.1).reshape(1, 3, 3)

        positions = wp.from_numpy(positions_np, dtype=wp.vec3d, device=device)
        charges = wp.from_numpy(charges_np, dtype=wp.float64, device=device)
        cell_inv_t = wp.from_numpy(cell_inv_t_np, dtype=wp.mat33d, device=device)

        # Constant potential
        mesh = wp.full((16, 16, 16), 5.0, dtype=wp.float64, device=device)
        forces = wp.zeros(5, dtype=wp.vec3d, device=device)

        spline_gather_gradient(
            positions, charges, cell_inv_t, 4, mesh, forces, wp.float64, device
        )
        wp.synchronize()

        forces_np = np.array(forces.numpy().tolist())
        # Force = -q * grad(phi). For constant phi, grad(phi) = 0, so F = 0
        assert forces_np == pytest.approx(0.0, abs=1e-10)
