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

from typing import Any

import numpy as np
import pytest
import warp as wp

from nvalchemiops.math.spline import (
    batch_spline_gather,
    batch_spline_gather_gradient,
    batch_spline_gather_vec3,
    batch_spline_spread,
    bspline_derivative,
    bspline_second_derivative,
    bspline_weight,
    bspline_weight_hessian_3d,
    spline_gather,
    spline_gather_gradient,
    spline_gather_vec3,
    spline_spread,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture(scope="module")
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


@pytest.fixture(scope="class")
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

    @pytest.mark.parametrize("order", [2, 3, 4, 5, 6])
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
        assert mesh_np.sum() == pytest.approx(1.0, rel=1e-8)
        assert mesh_np.max() == pytest.approx(0.2508416403, rel=1e-8)
        assert mesh_np.min() == pytest.approx(-0.2962962963, rel=1e-8)
        assert np.count_nonzero(np.abs(mesh_np) > 1e-8) == 181

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

    @pytest.mark.parametrize("order", [2, 3, 4, 5, 6])
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


# =============================================================================
# B-spline second-derivative + 3D Hessian (Phase 3 chunk MA)
# =============================================================================
#
# Trampoline kernels exposing the new ``@wp.func`` primitives so the tests can
# compare against numpy-side analytical / finite-difference oracles. These
# kernels live next to the test class so they don't pollute the public spline
# surface; the ``@wp.func``s under test are exported from
# nvalchemiops.math.spline.


@wp.kernel
def _test_bspline_weight_kernel(
    u_in: wp.array(dtype=Any),
    order: wp.int32,
    out: wp.array(dtype=Any),
):
    """Trampoline: write ``M_n(u_in[i])`` to ``out[i]`` for parity tests."""
    i = wp.tid()
    out[i] = bspline_weight(u_in[i], order)


@wp.kernel
def _test_bspline_derivative_kernel(
    u_in: wp.array(dtype=Any),
    order: wp.int32,
    out: wp.array(dtype=Any),
):
    """Trampoline: write ``M_n'(u_in[i])`` to ``out[i]`` for parity tests."""
    i = wp.tid()
    out[i] = bspline_derivative(u_in[i], order)


@wp.kernel
def _test_bspline_second_derivative_kernel(
    u_in: wp.array(dtype=Any),
    order: wp.int32,
    out: wp.array(dtype=Any),
):
    """Trampoline: write ``B''(u_in[i])`` to ``out[i]`` for parity tests."""
    i = wp.tid()
    out[i] = bspline_second_derivative(u_in[i], order)


@wp.kernel
def _test_bspline_weight_hessian_3d_kernel(
    theta_in: wp.array(dtype=Any),
    offset_in: wp.array(dtype=wp.vec3i),
    order: wp.int32,
    mesh_dims: wp.vec3i,
    diag_out: wp.array(dtype=Any),
    off_out: wp.array(dtype=Any),
):
    """Trampoline: ``(diag, off)`` of the 3D B-spline Hessian per input."""
    i = wp.tid()
    diag, off = bspline_weight_hessian_3d(theta_in[i], offset_in[i], order, mesh_dims)
    diag_out[i] = diag
    off_out[i] = off


def _truncated_power_bspline(u: np.ndarray, order: int, deriv: int) -> np.ndarray:
    r"""Closed-form cardinal B-spline (and its derivatives) via truncated powers.

    The cardinal B-spline of order ``n`` is the ``n``-fold convolution of the
    unit boxcar ``χ_[0,1)`` and admits the explicit form

    .. math::

        M_n(u) = \frac{1}{(n-1)!} \sum_{j=0}^{n} (-1)^j \binom{n}{j}
                 (u - j)_+^{n-1}

    where ``(x)_+ = max(0, x)``. Differentiating ``deriv`` times reduces the
    exponent and the leading factorial accordingly. Valid for any
    ``deriv ≤ n - 1``; returns zeros otherwise (matches the convention of the
    Warp primitives, which return zero when the spline order is too low to
    support the requested derivative).

    Independent of the piecewise polynomials used inside
    :func:`bspline_weight` / :func:`bspline_derivative` /
    :func:`bspline_second_derivative` — making this the right oracle for
    parity tests of those functions.
    """
    import math

    n = order
    k = n - 1 - deriv  # exponent of the truncated-power term
    if n < 1 or k < 0:
        return np.zeros_like(u, dtype=np.float64)
    out = np.zeros_like(u, dtype=np.float64)
    norm = math.factorial(k)
    for j in range(n + 1):
        sign = -1.0 if (j & 1) else 1.0
        coef = math.comb(n, j)
        diff = u - j
        # Strict ``> 0`` so the boundary diff == 0 maps to the zero branch
        # (right-continuous truncated power).
        out += sign * coef * np.where(diff > 0, np.power(diff, k), 0.0)
    return out / norm


def _bspline_weight_np(u: np.ndarray, order: int) -> np.ndarray:
    """Pure-numpy ``M_n(u)`` mirroring ``bspline_weight``."""
    return _truncated_power_bspline(u, order, deriv=0)


def _bspline_derivative_np(u: np.ndarray, order: int) -> np.ndarray:
    """Pure-numpy ``M_n'(u)`` mirroring ``bspline_derivative``."""
    return _truncated_power_bspline(u, order, deriv=1)


def _bspline_second_derivative_np(u: np.ndarray, order: int) -> np.ndarray:
    """Pure-numpy ``M_n''(u)`` mirroring ``bspline_second_derivative``."""
    return _truncated_power_bspline(u, order, deriv=2)


def _bspline_via_convolution(u: np.ndarray, order: int, h: float = 1e-4) -> np.ndarray:
    r"""Cardinal B-spline via numerical n-fold convolution of the unit boxcar.

    Independent oracle for the analytical pieces — provides a check that
    the polynomial coefficients in :func:`bspline_weight` actually
    represent the n-fold convolution of ``χ_[0,1)`` rather than just
    matching the closed-form truncated-power expression. With
    ``h = 1e-4`` the trapezoidal-rule convolution converges to ~1e-8
    accuracy, which is well below the breakpoint behavior we want to
    detect (the cardinal pieces match to machine precision; if any
    polynomial were wrong even by one term the residual would be O(1)).
    """
    n = order
    grid = np.arange(0, n + 1, h)
    box = ((grid >= 0) & (grid < 1)).astype(np.float64)
    spline = box.copy()
    for _ in range(n - 1):
        spline = np.convolve(spline, box) * h
    out_grid = np.arange(0, len(spline)) * h
    return np.interp(u, out_grid, spline)


class TestBSplineWeightHigherOrder:
    r"""``bspline_weight`` and ``bspline_derivative`` match independent oracles
    across all supported orders.

    The closed-form truncated-power oracle and the explicit n-fold-convolution
    oracle agree at machine precision for the well-conditioned interior of the
    spline support, providing two independent paths to the same ground truth.
    """

    @pytest.mark.parametrize("order", [1, 2, 3, 4, 5, 6])
    def test_weight_truncated_power_parity(self, device, order):
        """``M_n(u)`` matches the truncated-power closed form on a sweep
        of ``u`` ∈ (0, n).

        ``atol`` widens with ``order`` because the truncated-power
        oracle accumulates O(C(n, n/2) · u^(n-1)) cancellation —
        order 4 boundary residual ~1e-15, order 6 boundary residual
        ~1e-12. The kernel itself uses direct piecewise polynomials
        and is 1-2 ULP precise."""
        u_in = np.linspace(0.001, float(order) - 0.001, 64, dtype=np.float64)

        u_wp = wp.from_numpy(u_in, dtype=wp.float64, device=device)
        out_wp = wp.zeros(u_in.shape[0], dtype=wp.float64, device=device)
        wp.launch(
            _test_bspline_weight_kernel,
            dim=u_in.shape[0],
            inputs=[u_wp, order, out_wp],
            device=device,
        )
        wp.synchronize()

        expected = _bspline_weight_np(u_in, order)
        atol = 1e-13 if order <= 4 else 1e-11
        np.testing.assert_allclose(out_wp.numpy(), expected, atol=atol, rtol=1e-10)

    @pytest.mark.parametrize("order", [2, 3, 4, 5, 6])
    def test_derivative_truncated_power_parity(self, device, order):
        """``M_n'(u)`` matches the truncated-power closed form on a sweep
        of ``u`` ∈ (0, n)."""
        u_in = np.linspace(0.001, float(order) - 0.001, 64, dtype=np.float64)

        u_wp = wp.from_numpy(u_in, dtype=wp.float64, device=device)
        out_wp = wp.zeros(u_in.shape[0], dtype=wp.float64, device=device)
        wp.launch(
            _test_bspline_derivative_kernel,
            dim=u_in.shape[0],
            inputs=[u_wp, order, out_wp],
            device=device,
        )
        wp.synchronize()

        expected = _bspline_derivative_np(u_in, order)
        atol = 1e-13 if order <= 4 else 1e-11
        np.testing.assert_allclose(out_wp.numpy(), expected, atol=atol, rtol=1e-10)

    @pytest.mark.parametrize("order", [5, 6])
    def test_weight_n_fold_convolution_parity(self, device, order):
        """``M_n(u)`` matches a numerical n-fold convolution of the unit
        boxcar — independent of the closed-form polynomial pieces.

        Sanity check that the new orders 5/6 polynomial coefficients
        actually express the cardinal B-spline (n-fold convolution of
        ``χ_[0,1)``). Loose tolerance because the convolution oracle is
        a trapezoidal-rule approximation; if any piece were qualitatively
        wrong the residual would be O(1)."""
        u_in = np.linspace(0.05, float(order) - 0.05, 32, dtype=np.float64)

        u_wp = wp.from_numpy(u_in, dtype=wp.float64, device=device)
        out_wp = wp.zeros(u_in.shape[0], dtype=wp.float64, device=device)
        wp.launch(
            _test_bspline_weight_kernel,
            dim=u_in.shape[0],
            inputs=[u_wp, order, out_wp],
            device=device,
        )
        wp.synchronize()

        expected = _bspline_via_convolution(u_in, order, h=1e-4)
        # Discrete np.convolve is rectangular-rule, so accuracy is O(h),
        # not O(h²). Loose atol — the point is qualitative parity, not
        # bit-precision: any wrong polynomial coefficient would give
        # O(1) residual, easily caught at 1e-3.
        np.testing.assert_allclose(out_wp.numpy(), expected, atol=1e-3)

    @pytest.mark.parametrize("order", [1, 2, 3, 4, 5, 6])
    def test_partition_of_unity(self, device, order):
        """``Σ_k M_n(θ + k) = 1`` for any ``θ ∈ [0, 1)`` — the defining
        property of cardinal B-splines (sum across a unit-shifted family
        is identically one). Verifies normalization across orders."""
        rng = np.random.default_rng(0xB5E)
        thetas = rng.uniform(0.0, 1.0, size=16).astype(np.float64)
        # u_k = θ + (k - n/2) for k = 0..n-1 covers the support [0, n).
        # bspline_weight tests one (u, order) at a time; assemble the sum
        # for each θ.
        for theta in thetas:
            u_vec = np.array([theta + i for i in range(order)], dtype=np.float64)
            # Shift so all u_vec entries land in [0, order).
            # For θ ∈ [0, 1) and k = 0..n-1, u = θ + k ∈ [0, n) trivially.
            u_wp = wp.from_numpy(u_vec, dtype=wp.float64, device=device)
            out_wp = wp.zeros(u_vec.shape[0], dtype=wp.float64, device=device)
            wp.launch(
                _test_bspline_weight_kernel,
                dim=u_vec.shape[0],
                inputs=[u_wp, order, out_wp],
                device=device,
            )
            wp.synchronize()
            # Tolerance scales with order because higher-order pieces
            # involve larger polynomial coefficients (O(10⁴) at order 6)
            # that introduce ULP-scale rounding when summed.
            assert out_wp.numpy().sum() == pytest.approx(1.0, abs=1e-12)


class TestBSplineSecondDerivative:
    r"""``bspline_second_derivative`` matches analytical and finite-difference oracles."""

    @pytest.mark.parametrize("order", [3, 4])
    def test_known_values(self, device, order):
        """Hand-computed values at the midpoint of each piece."""
        if order == 4:
            u_in = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float64)
            # B''(u): u → u in [0,1); 4-3u in [1,2); 3u-8 in [2,3); 4-u in [3,4)
            expected = np.array([0.5, -0.5, -0.5, 0.5], dtype=np.float64)
        else:  # order == 3
            u_in = np.array([0.5, 1.5, 2.5], dtype=np.float64)
            # B''(u): 1, -2, 1
            expected = np.array([1.0, -2.0, 1.0], dtype=np.float64)

        u_wp = wp.from_numpy(u_in, dtype=wp.float64, device=device)
        out_wp = wp.zeros(u_in.shape[0], dtype=wp.float64, device=device)
        wp.launch(
            _test_bspline_second_derivative_kernel,
            dim=u_in.shape[0],
            inputs=[u_wp, order, out_wp],
            device=device,
        )
        wp.synchronize()
        np.testing.assert_allclose(out_wp.numpy(), expected, atol=1e-15)

    @pytest.mark.parametrize("order", [3, 4, 5, 6])
    def test_analytical_parity(self, device, order):
        """Sweep of ``u`` ∈ (0, order); each ``B''(u)`` matches the
        truncated-power oracle evaluated independently in numpy.

        Uses an analytical oracle (rather than finite differences)
        because central FD second-derivative roundoff at h=1e-5 is
        ~ε/h² per evaluation, which the mesh_dims² scale (~500) in
        the downstream Hessian test amplifies past 1e-5 relative.
        ``atol`` is widened to ``1e-12`` for orders ≥ 5 because the
        truncated-power oracle accumulates O(10⁴) cancellations at
        order 6 — 1-2 ULP per term × 7 terms ≈ 1e-12 absolute. The
        kernel itself uses direct piecewise polynomials (no
        cancellation), so the residual is dominated by the oracle.
        """
        u_in = np.linspace(0.001, float(order) - 0.001, 64, dtype=np.float64)

        u_wp = wp.from_numpy(u_in, dtype=wp.float64, device=device)
        out_wp = wp.zeros(u_in.shape[0], dtype=wp.float64, device=device)
        wp.launch(
            _test_bspline_second_derivative_kernel,
            dim=u_in.shape[0],
            inputs=[u_wp, order, out_wp],
            device=device,
        )
        wp.synchronize()

        expected = _bspline_second_derivative_np(u_in, order)
        atol = 1e-13 if order <= 4 else 1e-11
        np.testing.assert_allclose(out_wp.numpy(), expected, atol=atol, rtol=1e-10)


class TestBSplineWeightHessian3D:
    r"""``bspline_weight_hessian_3d`` matches the analytical product rule.

    The 3D weight is :math:`w(\theta) = M_n(u_x) M_n(u_y) M_n(u_z)`. Its
    Hessian factorizes via the product rule into a 6-component symmetric
    tensor; we verify each component against the explicit factorization
    using ``bspline_weight`` / ``bspline_derivative`` /
    ``bspline_second_derivative`` evaluated independently.
    """

    @pytest.mark.parametrize("order", [3, 4])
    def test_factorization_parity(self, device, order):
        # Sample a handful of (theta, offset) pairs in the interior of the
        # spline support so each axis lands cleanly inside the [0, order)
        # parameter range.
        rng = np.random.default_rng(0xC0DE)
        n_pts = 32
        theta_np = rng.uniform(0.0, 1.0, size=(n_pts, 3)).astype(np.float64)
        # Pick offsets such that u_α = order/2 + θ - offset stays well
        # inside [0.1, order - 0.1) — avoid the FD-unfriendly breakpoints
        # entirely by taking offset_α = 1 (so u_α ∈ [order/2, order/2 + 1)
        # for order = 4, that's [2, 3)).
        offset_np = np.full((n_pts, 3), 1, dtype=np.int32)
        mesh_dims_np = np.array([16, 24, 20], dtype=np.int32)

        theta_wp = wp.from_numpy(theta_np, dtype=wp.vec3d, device=device)
        offset_wp = wp.from_numpy(offset_np, dtype=wp.vec3i, device=device)
        diag_wp = wp.zeros(n_pts, dtype=wp.vec3d, device=device)
        off_wp = wp.zeros(n_pts, dtype=wp.vec3d, device=device)

        wp.launch(
            _test_bspline_weight_hessian_3d_kernel,
            dim=n_pts,
            inputs=[
                theta_wp,
                offset_wp,
                order,
                wp.vec3i(
                    int(mesh_dims_np[0]), int(mesh_dims_np[1]), int(mesh_dims_np[2])
                ),
                diag_wp,
                off_wp,
            ],
            device=device,
        )
        wp.synchronize()

        # Numpy oracle — evaluate the same factorization independently.
        half_order = order * 0.5
        u_x = half_order + theta_np[:, 0] - offset_np[:, 0]
        u_y = half_order + theta_np[:, 1] - offset_np[:, 1]
        u_z = half_order + theta_np[:, 2] - offset_np[:, 2]

        # Cross-check: u must be in [0, order) by construction.
        assert ((u_x >= 0) & (u_x < order)).all()
        assert ((u_y >= 0) & (u_y < order)).all()
        assert ((u_z >= 0) & (u_z < order)).all()

        w_x = _bspline_weight_np(u_x, order)
        w_y = _bspline_weight_np(u_y, order)
        w_z = _bspline_weight_np(u_z, order)
        dw_x = _bspline_derivative_np(u_x, order)
        dw_y = _bspline_derivative_np(u_y, order)
        dw_z = _bspline_derivative_np(u_z, order)
        ddw_x = _bspline_second_derivative_np(u_x, order)
        ddw_y = _bspline_second_derivative_np(u_y, order)
        ddw_z = _bspline_second_derivative_np(u_z, order)

        md_x, md_y, md_z = (
            float(mesh_dims_np[0]),
            float(mesh_dims_np[1]),
            float(mesh_dims_np[2]),
        )
        diag_expected = np.stack(
            [
                ddw_x * w_y * w_z * md_x * md_x,
                w_x * ddw_y * w_z * md_y * md_y,
                w_x * w_y * ddw_z * md_z * md_z,
            ],
            axis=1,
        )
        off_expected = np.stack(
            [
                dw_x * dw_y * w_z * md_x * md_y,
                dw_x * w_y * dw_z * md_x * md_z,
                w_x * dw_y * dw_z * md_y * md_z,
            ],
            axis=1,
        )

        diag_got = np.array(diag_wp.numpy().tolist())
        off_got = np.array(off_wp.numpy().tolist())
        np.testing.assert_allclose(diag_got, diag_expected, atol=1e-13)
        np.testing.assert_allclose(off_got, off_expected, atol=1e-13)

    def test_zero_outside_support(self, device):
        """Hessian must be zero when ``u`` falls outside ``[0, order)``."""
        # offset[0] = -1 forces u_x = order/2 + theta - (-1) = order/2 + 1 + theta;
        # for order = 4 that's u_x ∈ [3, 4) which is in support, NOT what
        # we want. Use offset[0] = order so u_x = order/2 + theta - order ∈
        # [-order/2, -order/2+1) — out of [0, order).
        order = 4
        theta_np = np.array([[0.5, 0.5, 0.5]], dtype=np.float64)
        offset_np = np.array([[order, 0, 0]], dtype=np.int32)  # x out-of-support
        mesh_dims_np = np.array([16, 16, 16], dtype=np.int32)

        theta_wp = wp.from_numpy(theta_np, dtype=wp.vec3d, device=device)
        offset_wp = wp.from_numpy(offset_np, dtype=wp.vec3i, device=device)
        diag_wp = wp.zeros(1, dtype=wp.vec3d, device=device)
        off_wp = wp.zeros(1, dtype=wp.vec3d, device=device)

        wp.launch(
            _test_bspline_weight_hessian_3d_kernel,
            dim=1,
            inputs=[
                theta_wp,
                offset_wp,
                order,
                wp.vec3i(*[int(x) for x in mesh_dims_np]),
                diag_wp,
                off_wp,
            ],
            device=device,
        )
        wp.synchronize()

        np.testing.assert_array_equal(
            np.array(diag_wp.numpy().tolist())[0], np.zeros(3)
        )
        np.testing.assert_array_equal(np.array(off_wp.numpy().tolist())[0], np.zeros(3))
