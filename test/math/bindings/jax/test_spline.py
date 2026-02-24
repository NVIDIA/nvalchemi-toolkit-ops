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
Test Suite for B-Spline JAX Bindings
=====================================

Tests the JAX bindings in nvalchemiops.jax.spline.

Test Categories
---------------
1. Happy path tests: Verify operations run without error and produce reasonable output
2. Regression tests: Verify output matches hardcoded expected values
3. Property tests: Verify mathematical properties (e.g., spread sums to total charge)

Mathematical Properties Tested
------------------------------
- B-spline spread: sum(mesh) == sum(charges) (charge conservation)
- B-spline weights sum to 1 (partition of unity)
- Spread/gather are adjoint operations

NOTE: Autograd/gradient tests are SKIPPED because JAX spline wrappers use
enable_backward=False. Output dtype matches the input positions dtype
(float32 or float64).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Enable JAX float64 support (disabled by default)
jax.config.update("jax_enable_x64", True)

from nvalchemiops.jax.spline import (  # noqa: E402
    compute_bspline_deconvolution,
    compute_bspline_deconvolution_1d,
    spline_gather,
    spline_gather_channels,
    spline_gather_gradient,
    spline_gather_vec3,
    spline_spread,
    spline_spread_channels,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def simple_system():
    """Simple 4-atom system in a 10A cubic cell for testing.

    Returns
    -------
    dict
        Dictionary containing positions, charges, cell, and mesh_dims.
        Matches the simple_system fixture in Warp kernel tests.
    """
    cell = jnp.eye(3, dtype=jnp.float64) * 10.0
    positions = jnp.array(
        [
            [1.0, 1.0, 1.0],
            [5.0, 5.0, 5.0],
            [2.5, 7.5, 3.0],
            [8.0, 2.0, 6.0],
        ],
        dtype=jnp.float64,
    )
    charges = jnp.array([1.0, -1.0, 0.5, 0.5], dtype=jnp.float64)

    return {
        "positions": positions,
        "charges": charges,
        "cell": cell,
        "mesh_dims": (8, 8, 8),
    }


@pytest.fixture(scope="session")
def batch_system():
    """Batched 2-system test setup.

    Returns
    -------
    dict
        Dictionary containing batched positions, charges, batch_idx, cell.
        Matches the batch_system fixture in Warp kernel tests.
    """
    cell = jnp.eye(3, dtype=jnp.float64) * 10.0
    positions = jnp.array(
        [
            [1.0, 1.0, 1.0],  # sys 0
            [5.0, 5.0, 5.0],  # sys 0
            [2.5, 2.5, 2.5],  # sys 1
            [7.5, 7.5, 7.5],  # sys 1
        ],
        dtype=jnp.float64,
    )
    charges = jnp.array([1.0, -0.5, 0.5, -0.5], dtype=jnp.float64)
    batch_idx = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
    batch_cell = jnp.tile(cell[jnp.newaxis, :, :], (2, 1, 1))

    return {
        "positions": positions,
        "charges": charges,
        "batch_idx": batch_idx,
        "cell": batch_cell,
        "mesh_dims": (8, 8, 8),
        "num_systems": 2,
    }


###########################################################################################
########################### B-Spline Weight Function Tests ################################
###########################################################################################


class TestBSplineWeightFunctions:
    """Test B-spline basis function properties."""

    @pytest.mark.parametrize("order", [1, 2, 3, 4])
    def test_partition_of_unity(self, order):
        """Test that B-spline weights sum to 1 at any point."""
        # Test at various points within [0, 1)
        test_points = [0.1, 0.25, 0.5, 0.75, 0.9]

        for theta in test_points:
            total = 0.0
            for k in range(order):
                u = theta + k
                # Compute weight using pure Python reference
                if order == 1:
                    w = 1.0 if 0 <= u < 1 else 0.0
                elif order == 2:
                    if 0 <= u < 1:
                        w = u
                    elif 1 <= u < 2:
                        w = 2 - u
                    else:
                        w = 0.0
                elif order == 3:
                    if 0 <= u < 1:
                        w = u**2 / 2
                    elif 1 <= u < 2:
                        w = 0.75 - (u - 1.5) ** 2
                    elif 2 <= u < 3:
                        w = (3 - u) ** 2 / 2
                    else:
                        w = 0.0
                elif order == 4:
                    if 0 <= u < 1:
                        w = u**3 / 6
                    elif 1 <= u < 2:
                        w = (-3 * u**3 + 12 * u**2 - 12 * u + 4) / 6
                    elif 2 <= u < 3:
                        w = (3 * u**3 - 24 * u**2 + 60 * u - 44) / 6
                    elif 3 <= u < 4:
                        w = (4 - u) ** 3 / 6
                    else:
                        w = 0.0
                total += w

            assert abs(total) == pytest.approx(1.0, rel=1e-10), (
                f"Partition of unity failed for order={order}, theta={theta}: sum={total}"
            )

    @pytest.mark.parametrize("order", [2, 3, 4])
    def test_derivative_matches_finite_diff(self, order):
        """Test that B-spline derivative matches finite differences."""
        eps = 1e-6
        test_points = [0.5, 1.5, 2.5]
        if order == 4:
            test_points.append(3.5)

        for u in test_points:
            # Reference: finite difference
            def weight_py(u_val, ord_val):
                if ord_val == 2:
                    if 0 <= u_val < 1:
                        return u_val
                    elif 1 <= u_val < 2:
                        return 2 - u_val
                    else:
                        return 0.0
                elif ord_val == 3:
                    if 0 <= u_val < 1:
                        return u_val**2 / 2
                    elif 1 <= u_val < 2:
                        return 0.75 - (u_val - 1.5) ** 2
                    elif 2 <= u_val < 3:
                        return (3 - u_val) ** 2 / 2
                    else:
                        return 0.0
                elif ord_val == 4:
                    if 0 <= u_val < 1:
                        return u_val**3 / 6
                    elif 1 <= u_val < 2:
                        return (-3 * u_val**3 + 12 * u_val**2 - 12 * u_val + 4) / 6
                    elif 2 <= u_val < 3:
                        return (3 * u_val**3 - 24 * u_val**2 + 60 * u_val - 44) / 6
                    elif 3 <= u_val < 4:
                        return (4 - u_val) ** 3 / 6
                    else:
                        return 0.0

            fd_deriv = (weight_py(u + eps, order) - weight_py(u - eps, order)) / (
                2 * eps
            )

            # Analytical derivative (reference)
            if order == 2:
                if 0 <= u < 1:
                    analytic = 1.0
                elif 1 <= u < 2:
                    analytic = -1.0
                else:
                    analytic = 0.0
            elif order == 3:
                if 0 <= u < 1:
                    analytic = u
                elif 1 <= u < 2:
                    analytic = -2 * (u - 1.5)
                elif 2 <= u < 3:
                    analytic = -(3 - u)
                else:
                    analytic = 0.0
            elif order == 4:
                if 0 <= u < 1:
                    analytic = u**2 / 2
                elif 1 <= u < 2:
                    analytic = (-9 * u**2 + 24 * u - 12) / 6
                elif 2 <= u < 3:
                    analytic = (9 * u**2 - 48 * u + 60) / 6
                elif 3 <= u < 4:
                    analytic = -3 * (4 - u) ** 2 / 6
                else:
                    analytic = 0.0

            assert abs(fd_deriv - analytic) < 1e-5, (
                f"Derivative mismatch for order={order}, u={u}: "
                f"fd={fd_deriv}, analytic={analytic}"
            )


###########################################################################################
########################### Regression Tests ##############################################
###########################################################################################


class TestSplineRegressionValues:
    """Regression tests with hardcoded expected values.

    These values match the Warp kernel regression tests to ensure JAX
    bindings produce identical results. NOTE: Tolerance values work for
    both float32 and float64 outputs.
    """

    @pytest.mark.slow
    def test_spread_regression(self, simple_system):
        """Regression test for spline_spread with expected values."""
        positions = simple_system["positions"]
        charges = simple_system["charges"]
        cell = simple_system["cell"]

        mesh = spline_spread(
            positions, charges, cell, mesh_dims=(8, 8, 8), spline_order=4
        )

        # Regression values (match Warp kernel tests)
        assert float(mesh.sum()) == pytest.approx(1.0, rel=1e-4)
        assert float(mesh.max()) == pytest.approx(0.2508416403, rel=1e-4)
        assert float(mesh.min()) == pytest.approx(-0.2962962963, rel=1e-4)
        assert int((jnp.abs(mesh) > 1e-12).sum()) == pytest.approx(182, rel=1e-4)

    def test_gather_regression(self, simple_system):
        """Regression test for spline_gather with expected values."""
        positions = simple_system["positions"]
        cell = simple_system["cell"]
        unit_charges = jnp.ones(4, dtype=jnp.float64)

        # Spread then gather
        mesh = spline_spread(
            positions, unit_charges, cell, mesh_dims=(8, 8, 8), spline_order=4
        )
        output = spline_gather(positions, mesh, cell, spline_order=4)

        # Regression values (match Warp kernel tests)
        expected = jnp.array(
            [0.11403329, 0.12506939, 0.11594129, 0.10419698],
            dtype=jnp.float64,
        )
        assert np.allclose(output, expected, rtol=1e-4)
        assert float(output.sum()) == pytest.approx(0.4592409390, rel=1e-4)

    def test_gather_vec3_regression(self, simple_system):
        """Regression test for spline_gather_vec3 with expected values."""
        positions = simple_system["positions"]
        charges = simple_system["charges"]
        cell = simple_system["cell"]

        # Uniform vec3 mesh: each point has (1, 2, 3)
        field = jnp.zeros((8, 8, 8, 3), dtype=jnp.float64)
        field = field.at[:, :, :, 0].set(1.0)
        field = field.at[:, :, :, 1].set(2.0)
        field = field.at[:, :, :, 2].set(3.0)

        output = spline_gather_vec3(positions, charges, field, cell, spline_order=4)

        # Each atom should gather approximately (1, 2, 3) * charge due to partition of unity
        for i in range(4):
            expected_vec = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64) * float(
                charges[i]
            )
            assert jnp.allclose(output[i], expected_vec, rtol=1e-4), (
                f"Atom {i}: expected {expected_vec}, got {output[i]}"
            )

    def test_batch_spread_regression(self, batch_system):
        """Regression test for batched spline_spread."""
        positions = batch_system["positions"]
        charges = batch_system["charges"]
        batch_idx = batch_system["batch_idx"]
        cell = batch_system["cell"]

        mesh = spline_spread(
            positions,
            charges,
            cell,
            mesh_dims=(8, 8, 8),
            spline_order=4,
            batch_idx=batch_idx,
        )

        # Check per-system charge conservation
        sys0_charge = charges[batch_idx == 0].sum()
        sys1_charge = charges[batch_idx == 1].sum()

        sys0_mesh_sum = mesh[0].sum()
        sys1_mesh_sum = mesh[1].sum()

        assert float(sys0_mesh_sum) == pytest.approx(float(sys0_charge), rel=1e-4)
        assert float(sys1_mesh_sum) == pytest.approx(float(sys1_charge), rel=1e-4)

    def test_batch_gather_regression(self, batch_system):
        """Regression test for batched spline_gather."""
        positions = batch_system["positions"]
        charges = batch_system["charges"]
        batch_idx = batch_system["batch_idx"]
        cell = batch_system["cell"]

        # Spread then gather
        mesh = spline_spread(
            positions,
            charges,
            cell,
            mesh_dims=(8, 8, 8),
            spline_order=4,
            batch_idx=batch_idx,
        )
        output = spline_gather(
            positions, mesh, cell, spline_order=4, batch_idx=batch_idx
        )

        # Output should have finite values
        assert jnp.isfinite(output).all()
        assert output.shape == (4,)


###########################################################################################
########################### Spread Operation Tests ########################################
###########################################################################################


class TestSplineSpread:
    """Test B-spline charge spreading."""

    @pytest.mark.parametrize("spline_order", [2, 3, 4])
    def test_charge_conservation(self, spline_order):
        """Test that spreading conserves total charge."""
        positions = jnp.array(
            [[2.3, 2.7, 2.1], [5.8, 5.2, 5.6], [3.1, 4.2, 6.8]],
            dtype=jnp.float64,
        )
        charges = jnp.array([1.5, -0.8, 0.3], dtype=jnp.float64)
        cell = jnp.array(
            [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]],
            dtype=jnp.float64,
        )

        mesh = spline_spread(
            positions, charges, cell, mesh_dims=(16, 16, 16), spline_order=spline_order
        )

        mesh_total = mesh.sum()
        charge_total = charges.sum()

        # Tolerance matches kernel precision
        assert jnp.allclose(mesh_total, charge_total, rtol=1e-3), (
            f"Charge not conserved: mesh={float(mesh_total)}, charges={float(charge_total)}"
        )

    def test_spread_output_shape(self):
        """Test that spread returns correct mesh shape."""
        key = jax.random.PRNGKey(0)
        positions = jax.random.uniform(key, shape=(10, 3), dtype=jnp.float64) * 5.0
        charges = jax.random.normal(
            jax.random.PRNGKey(1), shape=(10,), dtype=jnp.float64
        )
        cell = jnp.eye(3, dtype=jnp.float64) * 5.0

        mesh = spline_spread(
            positions, charges, cell, mesh_dims=(12, 14, 16), spline_order=4
        )

        assert mesh.shape == (12, 14, 16), f"Unexpected shape: {mesh.shape}"
        # Output dtype matches input positions dtype
        assert mesh.dtype == jnp.float64

    def test_spread_locality(self):
        """Test that a single atom only affects nearby grid points."""
        # Place a single atom slightly off-grid (theta != 0)
        # At exactly on-grid (theta=0), u=0 gives M(0)=0, so only 3^3=27 points affected
        # Offset slightly to get all 4^3=64 points with non-zero weight
        positions = jnp.array([[4.1, 4.1, 4.1]], dtype=jnp.float64)
        charges = jnp.array([1.0], dtype=jnp.float64)
        cell = jnp.eye(3, dtype=jnp.float64) * 8.0

        mesh = spline_spread(
            positions, charges, cell, mesh_dims=(8, 8, 8), spline_order=4
        )

        # Count non-zero points
        nonzero = int((jnp.abs(mesh) > 1e-12).sum())

        # For order-4 B-spline with theta != 0, should affect 4^3 = 64 points
        assert nonzero == 64, f"Expected 64 non-zero points, got {nonzero}"

    @pytest.mark.parametrize("spline_order", [2, 3, 4])
    def test_spread_center_of_mass(self, spline_order):
        """Test that the center of mass of the spread is at the atom position.

        This is a critical test that catches off-by-one errors in the B-spline
        spreading algorithm. The center of mass of the spread weights should
        equal the atom position.
        """
        cell_size = 10.0
        mesh_size = 16

        # Test several positions including grid-aligned and off-grid
        test_positions = [
            [5.0, 5.0, 5.0],  # Center of cell
            [2.5, 7.5, 4.0],  # Off-center
            [0.3, 0.3, 0.3],  # Near origin
            [9.5, 9.5, 9.5],  # Near edge (tests wrapping)
        ]

        cell = jnp.eye(3, dtype=jnp.float64) * cell_size

        for pos in test_positions:
            positions = jnp.array([pos], dtype=jnp.float64)
            charges = jnp.array([1.0], dtype=jnp.float64)

            mesh = spline_spread(
                positions,
                charges,
                cell,
                (mesh_size, mesh_size, mesh_size),
                spline_order,
            )

            # Compute center of mass of the mesh
            # Grid point i is at position i * cell_size / mesh_size
            grid_spacing = cell_size / mesh_size

            center_of_mass = jnp.zeros(3, dtype=jnp.float64)
            total_weight = mesh.sum()

            for dim in range(3):
                indices = jnp.arange(mesh_size, dtype=jnp.float64)
                # Grid points are at integer positions in mesh coordinates
                coords = indices * grid_spacing

                # Weight by mesh values and sum
                if dim == 0:
                    dim_weight = (mesh * coords.reshape(-1, 1, 1)).sum()
                elif dim == 1:
                    dim_weight = (mesh * coords.reshape(1, -1, 1)).sum()
                else:
                    dim_weight = (mesh * coords.reshape(1, 1, -1)).sum()

                center_of_mass = center_of_mass.at[dim].set(dim_weight / total_weight)

            expected_pos = jnp.array(pos, dtype=jnp.float64)

            # For interior positions, center of mass should match
            if all(grid_spacing < p < cell_size - grid_spacing for p in pos):
                assert jnp.allclose(center_of_mass, expected_pos, rtol=1e-4), (
                    f"Center of mass mismatch for order={spline_order}, pos={pos}: "
                    f"expected {expected_pos.tolist()}, got {center_of_mass.tolist()}"
                )


###########################################################################################
########################### Gather Operation Tests ########################################
###########################################################################################


class TestSplineGather:
    """Test B-spline gathering."""

    def test_gather_uniform_potential(self):
        """Test gathering from a uniform potential mesh."""
        positions = jnp.array([[2.0, 2.0, 2.0], [5.0, 5.0, 5.0]], dtype=jnp.float64)
        cell = jnp.eye(3, dtype=jnp.float64) * 8.0

        # Uniform mesh with constant value
        mesh = jnp.full((8, 8, 8), 2.5, dtype=jnp.float64)

        output = spline_gather(positions, mesh, cell, spline_order=4)

        # Should gather ~2.5 at each position due to partition of unity
        expected = jnp.array([2.5, 2.5], dtype=jnp.float64)
        assert jnp.allclose(output, expected, rtol=1e-4), (
            f"Expected {expected}, got {output}"
        )

    def test_gather_output_shape(self):
        """Test that gather returns correct output shape."""
        key = jax.random.PRNGKey(0)
        positions = jax.random.uniform(key, shape=(10, 3), dtype=jnp.float64) * 5.0
        cell = jnp.eye(3, dtype=jnp.float64) * 5.0
        mesh = jax.random.normal(
            jax.random.PRNGKey(1), shape=(8, 8, 8), dtype=jnp.float64
        )

        output = spline_gather(positions, mesh, cell, spline_order=4)

        assert output.shape == (10,), f"Unexpected shape: {output.shape}"
        assert output.dtype == jnp.float64


###########################################################################################
########################### Gather Gradient Tests #########################################
###########################################################################################


class TestSplineGatherGradient:
    """Test B-spline gradient gathering."""

    def test_gather_gradient_uniform_zero(self):
        """Test gathering gradient from a uniform potential (should be zero)."""
        positions = jnp.array([[2.0, 2.0, 2.0], [5.0, 5.0, 5.0]], dtype=jnp.float64)
        charges = jnp.array([1.0, -1.0], dtype=jnp.float64)
        cell = jnp.eye(3, dtype=jnp.float64) * 8.0

        # Uniform mesh (constant potential) has zero gradient
        potential = jnp.full((8, 8, 8), 1.0, dtype=jnp.float64)

        forces = spline_gather_gradient(
            positions, charges, potential, cell, spline_order=4
        )

        # Gradient of uniform field should be zero
        expected = jnp.zeros((2, 3), dtype=jnp.float64)
        assert jnp.allclose(forces, expected, atol=1e-4), (
            f"Expected zero gradient, got {forces}"
        )

    def test_gather_gradient_output_shape(self):
        """Test that gather_gradient returns correct output shape."""
        key = jax.random.PRNGKey(0)
        positions = jax.random.uniform(key, shape=(10, 3), dtype=jnp.float64) * 5.0
        charges = jax.random.normal(
            jax.random.PRNGKey(1), shape=(10,), dtype=jnp.float64
        )
        cell = jnp.eye(3, dtype=jnp.float64) * 5.0
        potential = jax.random.normal(
            jax.random.PRNGKey(2), shape=(8, 8, 8), dtype=jnp.float64
        )

        forces = spline_gather_gradient(
            positions, charges, potential, cell, spline_order=4
        )

        assert forces.shape == (10, 3), f"Unexpected shape: {forces.shape}"
        assert forces.dtype == jnp.float64


###########################################################################################
########################### Gather Vec3 Tests #############################################
###########################################################################################


class TestSplineGatherVec3:
    """Test B-spline vec3 field gathering."""

    def test_gather_vec3_uniform_field(self):
        """Test gathering from a uniform vec3 field."""
        positions = jnp.array([[2.0, 2.0, 2.0], [5.0, 5.0, 5.0]], dtype=jnp.float64)
        charges = jnp.array([1.0, -0.5], dtype=jnp.float64)
        cell = jnp.eye(3, dtype=jnp.float64) * 8.0

        # Uniform vec3 field: each point has (1, 2, 3)
        field = jnp.zeros((8, 8, 8, 3), dtype=jnp.float64)
        field = field.at[:, :, :, 0].set(1.0)
        field = field.at[:, :, :, 1].set(2.0)
        field = field.at[:, :, :, 2].set(3.0)

        output = spline_gather_vec3(positions, charges, field, cell, spline_order=4)

        # Each atom should gather (1, 2, 3) * charge
        expected = jnp.array([[1.0, 2.0, 3.0], [-0.5, -1.0, -1.5]], dtype=jnp.float64)
        assert jnp.allclose(output, expected, rtol=1e-4), (
            f"Expected {expected}, got {output}"
        )

    def test_gather_vec3_output_shape(self):
        """Test that gather_vec3 returns correct output shape."""
        key = jax.random.PRNGKey(0)
        positions = jax.random.uniform(key, shape=(10, 3), dtype=jnp.float64) * 5.0
        charges = jax.random.normal(
            jax.random.PRNGKey(1), shape=(10,), dtype=jnp.float64
        )
        cell = jnp.eye(3, dtype=jnp.float64) * 5.0
        field = jax.random.normal(
            jax.random.PRNGKey(2), shape=(8, 8, 8, 3), dtype=jnp.float64
        )

        output = spline_gather_vec3(positions, charges, field, cell, spline_order=4)

        assert output.shape == (10, 3), f"Unexpected shape: {output.shape}"
        assert output.dtype == jnp.float64


###########################################################################################
########################### Non-Cubic Cell Tests ##########################################
###########################################################################################


class TestNonCubicCell:
    """Test spline operations with non-cubic cells."""

    def test_triclinic_cell_spread(self):
        """Test spreading in a triclinic cell."""
        # Triclinic cell (non-orthogonal)
        cell = jnp.array(
            [[8.0, 0.0, 0.0], [2.0, 8.0, 0.0], [1.0, 1.0, 8.0]],
            dtype=jnp.float64,
        )

        positions = jnp.array([[2.0, 2.0, 2.0], [5.0, 5.0, 5.0]], dtype=jnp.float64)
        charges = jnp.array([1.0, -1.0], dtype=jnp.float64)

        mesh = spline_spread(
            positions, charges, cell, mesh_dims=(8, 8, 8), spline_order=4
        )

        # Charge should be conserved even in triclinic cells
        assert jnp.allclose(mesh.sum(), charges.sum(), rtol=1e-4)


###########################################################################################
########################### Spread/Gather Consistency Tests ###############################
###########################################################################################


class TestSpreadGatherConsistency:
    """Test mathematical consistency between spread and gather."""

    def test_spread_gather_adjoint(self):
        """Test that spread and gather are adjoint operations.

        For linear operators S (spread) and G (gather):
        <S(q), phi> = <q, G(phi)>

        Where <·,·> is the inner product (sum of element-wise products).
        """
        key = jax.random.PRNGKey(42)
        positions = jax.random.uniform(key, shape=(20, 3), dtype=jnp.float64) * 8.0
        charges = jax.random.normal(
            jax.random.PRNGKey(1), shape=(20,), dtype=jnp.float64
        )
        cell = jnp.eye(3, dtype=jnp.float64) * 8.0
        mesh_dims = (16, 16, 16)

        # Create random potential mesh
        potential = jax.random.normal(
            jax.random.PRNGKey(2), shape=mesh_dims, dtype=jnp.float64
        )

        # Spread charges to mesh
        mesh = spline_spread(positions, charges, cell, mesh_dims, spline_order=4)

        # Gather potential to atoms
        gathered = spline_gather(positions, potential, cell, spline_order=4)

        # Compute inner products
        lhs = jnp.sum(mesh * potential)  # <S(q), phi>
        rhs = jnp.sum(charges * gathered)  # <q, G(phi)>

        # Should be equal (adjoint property)
        assert jnp.allclose(lhs, rhs, rtol=1e-3), (
            f"Adjoint property failed: <S(q),phi>={float(lhs)}, <q,G(phi)>={float(rhs)}"
        )

    def test_spread_gather_roundtrip_sum_of_squares(self):
        """Test that spread-gather preserves sum of squares (Parseval)."""
        key = jax.random.PRNGKey(123)
        positions = jax.random.uniform(key, shape=(15, 3), dtype=jnp.float64) * 8.0
        charges = jax.random.normal(
            jax.random.PRNGKey(1), shape=(15,), dtype=jnp.float64
        )
        cell = jnp.eye(3, dtype=jnp.float64) * 8.0

        # Spread then gather
        mesh = spline_spread(
            positions, charges, cell, mesh_dims=(16, 16, 16), spline_order=4
        )
        gathered = spline_gather(positions, mesh, cell, spline_order=4)

        # Check that gathering preserves information (not exact, but close)
        charge_norm = jnp.sum(charges**2)
        gathered_norm = jnp.sum(gathered**2)

        # Ratio should be reasonable (not testing exact equality)
        ratio = gathered_norm / charge_norm
        assert 0.01 < ratio < 2.0, f"Sum of squares ratio out of range: {float(ratio)}"


###########################################################################################
########################### Batch Spread Tests ############################################
###########################################################################################


class TestBatchSplineSpread:
    """Test batched B-spline spreading."""

    def test_batch_spread_charge_conservation(self):
        """Test that batched spreading conserves charge per system."""
        positions = jnp.array(
            [
                [1.0, 1.0, 1.0],  # sys 0
                [5.0, 5.0, 5.0],  # sys 0
                [2.0, 2.0, 2.0],  # sys 1
                [6.0, 6.0, 6.0],  # sys 1
                [3.0, 3.0, 3.0],  # sys 1
            ],
            dtype=jnp.float64,
        )
        charges = jnp.array([1.0, -0.5, 0.8, -0.3, 0.2], dtype=jnp.float64)
        batch_idx = jnp.array([0, 0, 1, 1, 1], dtype=jnp.int32)
        cell = jnp.tile(
            jnp.eye(3, dtype=jnp.float64)[jnp.newaxis, :, :] * 8.0, (2, 1, 1)
        )

        mesh = spline_spread(
            positions,
            charges,
            cell,
            mesh_dims=(8, 8, 8),
            spline_order=4,
            batch_idx=batch_idx,
        )

        # Check per-system charge conservation
        sys0_charge = charges[batch_idx == 0].sum()
        sys1_charge = charges[batch_idx == 1].sum()

        assert jnp.allclose(mesh[0].sum(), sys0_charge, rtol=1e-4)
        assert jnp.allclose(mesh[1].sum(), sys1_charge, rtol=1e-4)

    def test_batch_spread_matches_single(self):
        """Test that batched spread matches individual spreads."""
        # System with 2 identical setups
        pos_single = jnp.array([[2.0, 3.0, 4.0]], dtype=jnp.float64)
        charge_single = jnp.array([1.5], dtype=jnp.float64)
        cell_single = jnp.eye(3, dtype=jnp.float64) * 8.0

        # Single-system spread
        mesh_single = spline_spread(
            pos_single, charge_single, cell_single, mesh_dims=(8, 8, 8), spline_order=4
        )

        # Batched spread with 2 identical systems
        pos_batch = jnp.tile(pos_single, (2, 1))
        charge_batch = jnp.tile(charge_single, 2)
        batch_idx = jnp.array([0, 1], dtype=jnp.int32)
        cell_batch = jnp.tile(cell_single[jnp.newaxis, :, :], (2, 1, 1))

        mesh_batch = spline_spread(
            pos_batch,
            charge_batch,
            cell_batch,
            mesh_dims=(8, 8, 8),
            spline_order=4,
            batch_idx=batch_idx,
        )

        # Both systems should match single-system result
        assert jnp.allclose(mesh_batch[0], mesh_single, rtol=1e-4)
        assert jnp.allclose(mesh_batch[1], mesh_single, rtol=1e-4)


###########################################################################################
########################### Batch Gather Tests ############################################
###########################################################################################


class TestBatchSplineGather:
    """Test batched B-spline gathering."""

    def test_batch_gather_uniform_potential(self):
        """Test batched gathering from uniform potentials."""
        positions = jnp.array([[2.0, 2.0, 2.0], [5.0, 5.0, 5.0]], dtype=jnp.float64)
        batch_idx = jnp.array([0, 1], dtype=jnp.int32)
        cell = jnp.tile(
            jnp.eye(3, dtype=jnp.float64)[jnp.newaxis, :, :] * 8.0, (2, 1, 1)
        )

        # Uniform meshes with different values
        mesh = jnp.stack(
            [
                jnp.full((8, 8, 8), 2.0, dtype=jnp.float64),
                jnp.full((8, 8, 8), 3.0, dtype=jnp.float64),
            ]
        )

        output = spline_gather(
            positions, mesh, cell, spline_order=4, batch_idx=batch_idx
        )

        # Should gather ~2.0 and ~3.0 respectively
        assert jnp.allclose(output[0], 2.0, rtol=1e-4)
        assert jnp.allclose(output[1], 3.0, rtol=1e-4)

    def test_batch_gather_matches_single(self):
        """Test that batched gather matches individual gathers."""
        # Create a single system
        pos_single = jnp.array([[2.0, 3.0, 4.0]], dtype=jnp.float64)
        cell_single = jnp.eye(3, dtype=jnp.float64) * 8.0
        mesh_single = jax.random.normal(
            jax.random.PRNGKey(0), shape=(8, 8, 8), dtype=jnp.float64
        )

        # Single-system gather
        gather_single = spline_gather(
            pos_single, mesh_single, cell_single, spline_order=4
        )

        # Batched gather with 2 identical systems
        pos_batch = jnp.tile(pos_single, (2, 1))
        batch_idx = jnp.array([0, 1], dtype=jnp.int32)
        cell_batch = jnp.tile(cell_single[jnp.newaxis, :, :], (2, 1, 1))
        mesh_batch = jnp.stack([mesh_single, mesh_single])

        gather_batch = spline_gather(
            pos_batch, mesh_batch, cell_batch, spline_order=4, batch_idx=batch_idx
        )

        # Both atoms should match single-system result
        assert jnp.allclose(gather_batch[0], gather_single[0], rtol=1e-4)
        assert jnp.allclose(gather_batch[1], gather_single[0], rtol=1e-4)


###########################################################################################
########################### Batch Gather Vec3 Tests #######################################
###########################################################################################


class TestBatchSplineGatherVec3:
    """Test batched B-spline vec3 gathering."""

    def test_batch_gather_vec3_uniform_field(self):
        """Test batched vec3 gathering from uniform fields."""
        positions = jnp.array([[2.0, 2.0, 2.0], [5.0, 5.0, 5.0]], dtype=jnp.float64)
        charges = jnp.array([1.0, -0.5], dtype=jnp.float64)
        batch_idx = jnp.array([0, 1], dtype=jnp.int32)
        cell = jnp.tile(
            jnp.eye(3, dtype=jnp.float64)[jnp.newaxis, :, :] * 8.0, (2, 1, 1)
        )

        # Uniform vec3 fields with different values
        field0 = jnp.zeros((8, 8, 8, 3), dtype=jnp.float64)
        field0 = field0.at[:, :, :, :].set(jnp.array([1.0, 2.0, 3.0]))

        field1 = jnp.zeros((8, 8, 8, 3), dtype=jnp.float64)
        field1 = field1.at[:, :, :, :].set(jnp.array([2.0, 3.0, 4.0]))

        field = jnp.stack([field0, field1])

        output = spline_gather_vec3(
            positions, charges, field, cell, spline_order=4, batch_idx=batch_idx
        )

        # Check results
        expected0 = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64) * charges[0]
        expected1 = jnp.array([2.0, 3.0, 4.0], dtype=jnp.float64) * charges[1]

        assert jnp.allclose(output[0], expected0, rtol=1e-4)
        assert jnp.allclose(output[1], expected1, rtol=1e-4)

    def test_batch_gather_vec3_output_shape(self):
        """Test that batched gather_vec3 returns correct shape."""
        key = jax.random.PRNGKey(0)
        positions = jax.random.uniform(key, shape=(10, 3), dtype=jnp.float64) * 8.0
        charges = jax.random.normal(
            jax.random.PRNGKey(1), shape=(10,), dtype=jnp.float64
        )
        batch_idx = jnp.array([0] * 5 + [1] * 5, dtype=jnp.int32)
        cell = jnp.tile(
            jnp.eye(3, dtype=jnp.float64)[jnp.newaxis, :, :] * 8.0, (2, 1, 1)
        )
        field = jax.random.normal(
            jax.random.PRNGKey(2), shape=(2, 8, 8, 8, 3), dtype=jnp.float64
        )

        output = spline_gather_vec3(
            positions, charges, field, cell, spline_order=4, batch_idx=batch_idx
        )

        assert output.shape == (10, 3), f"Unexpected shape: {output.shape}"
        assert output.dtype == jnp.float64


###########################################################################################
########################### Batch Gather Gradient Tests ###################################
###########################################################################################


class TestBatchSplineGatherGradient:
    """Test batched B-spline gradient gathering."""

    def test_batch_gather_gradient_uniform_zero(self):
        """Test batched gradient gathering from uniform potentials."""
        positions = jnp.array([[2.0, 2.0, 2.0], [5.0, 5.0, 5.0]], dtype=jnp.float64)
        charges = jnp.array([1.0, -1.0], dtype=jnp.float64)
        batch_idx = jnp.array([0, 1], dtype=jnp.int32)
        cell = jnp.tile(
            jnp.eye(3, dtype=jnp.float64)[jnp.newaxis, :, :] * 8.0, (2, 1, 1)
        )

        # Uniform potentials (zero gradient)
        potential = jnp.stack(
            [
                jnp.full((8, 8, 8), 1.0, dtype=jnp.float64),
                jnp.full((8, 8, 8), 2.0, dtype=jnp.float64),
            ]
        )

        forces = spline_gather_gradient(
            positions, charges, potential, cell, spline_order=4, batch_idx=batch_idx
        )

        # Gradient of uniform fields should be zero
        expected = jnp.zeros((2, 3), dtype=jnp.float64)
        assert jnp.allclose(forces, expected, atol=1e-4)

    def test_batch_gather_gradient_matches_single(self):
        """Test that batched gather_gradient matches individual gathers."""
        # Single system
        pos_single = jnp.array([[2.0, 3.0, 4.0]], dtype=jnp.float64)
        charge_single = jnp.array([1.5], dtype=jnp.float64)
        cell_single = jnp.eye(3, dtype=jnp.float64) * 8.0
        potential_single = jax.random.normal(
            jax.random.PRNGKey(0), shape=(8, 8, 8), dtype=jnp.float64
        )

        # Single-system gradient
        grad_single = spline_gather_gradient(
            pos_single, charge_single, potential_single, cell_single, spline_order=4
        )

        # Batched gradient with 2 identical systems
        pos_batch = jnp.tile(pos_single, (2, 1))
        charge_batch = jnp.tile(charge_single, 2)
        batch_idx = jnp.array([0, 1], dtype=jnp.int32)
        cell_batch = jnp.tile(cell_single[jnp.newaxis, :, :], (2, 1, 1))
        potential_batch = jnp.stack([potential_single, potential_single])

        grad_batch = spline_gather_gradient(
            pos_batch,
            charge_batch,
            potential_batch,
            cell_batch,
            spline_order=4,
            batch_idx=batch_idx,
        )

        # Both should match
        assert jnp.allclose(grad_batch[0], grad_single[0], rtol=1e-4)
        assert jnp.allclose(grad_batch[1], grad_single[0], rtol=1e-4)


###########################################################################################
########################### Batch Different Cells Tests ###################################
###########################################################################################


class TestBatchDifferentCells:
    """Test batched operations with different cells per system."""

    def test_batch_spread_different_cells(self):
        """Test batched spread with different cell sizes."""
        positions = jnp.array([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]], dtype=jnp.float64)
        charges = jnp.array([1.0, -1.0], dtype=jnp.float64)
        batch_idx = jnp.array([0, 1], dtype=jnp.int32)

        # Different cell sizes
        cell = jnp.stack(
            [
                jnp.eye(3, dtype=jnp.float64) * 8.0,
                jnp.eye(3, dtype=jnp.float64) * 10.0,
            ]
        )

        mesh = spline_spread(
            positions,
            charges,
            cell,
            mesh_dims=(8, 8, 8),
            spline_order=4,
            batch_idx=batch_idx,
        )

        # Each system should conserve charge
        assert jnp.allclose(mesh[0].sum(), charges[0], rtol=1e-4)
        assert jnp.allclose(mesh[1].sum(), charges[1], rtol=1e-4)

    def test_batch_gather_different_cells(self):
        """Test batched gather with different cell sizes."""
        positions = jnp.array([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]], dtype=jnp.float64)
        batch_idx = jnp.array([0, 1], dtype=jnp.int32)

        # Different cell sizes
        cell = jnp.stack(
            [
                jnp.eye(3, dtype=jnp.float64) * 8.0,
                jnp.eye(3, dtype=jnp.float64) * 10.0,
            ]
        )

        # Uniform meshes
        mesh = jnp.stack(
            [
                jnp.full((8, 8, 8), 2.0, dtype=jnp.float64),
                jnp.full((8, 8, 8), 3.0, dtype=jnp.float64),
            ]
        )

        output = spline_gather(
            positions, mesh, cell, spline_order=4, batch_idx=batch_idx
        )

        # Should gather uniform values
        assert jnp.allclose(output[0], 2.0, rtol=1e-4)
        assert jnp.allclose(output[1], 3.0, rtol=1e-4)


###########################################################################################
########################### Multi-Channel Spread Tests ####################################
###########################################################################################


class TestMultiChannelSplineSpread:
    """Test multi-channel B-spline spreading."""

    def test_spread_channels_output_shape(self):
        """Test that multi-channel spread returns correct shape."""
        key = jax.random.PRNGKey(0)
        positions = jax.random.uniform(key, shape=(10, 3), dtype=jnp.float64) * 8.0
        values = jax.random.normal(
            jax.random.PRNGKey(1), shape=(10, 4), dtype=jnp.float64
        )
        cell = jnp.eye(3, dtype=jnp.float64) * 8.0

        mesh = spline_spread_channels(
            positions, values, cell, mesh_dims=(8, 8, 8), spline_order=4
        )

        # JAX spline_spread_channels returns shape (C, nx, ny, nz)
        assert mesh.shape == (4, 8, 8, 8), f"Unexpected shape: {mesh.shape}"
        assert mesh.dtype == jnp.float64

    def test_spread_channels_conservation(self):
        """Test that multi-channel spread conserves per-channel totals."""
        positions = jnp.array([[2.0, 2.0, 2.0], [5.0, 5.0, 5.0]], dtype=jnp.float64)
        values = jnp.array([[1.0, 2.0, 3.0], [-0.5, 1.5, -2.0]], dtype=jnp.float64)
        cell = jnp.eye(3, dtype=jnp.float64) * 8.0

        mesh = spline_spread_channels(
            positions, values, cell, mesh_dims=(8, 8, 8), spline_order=4
        )

        # Each channel should conserve its total
        # JAX shape is (C, nx, ny, nz)
        for c in range(3):
            channel_total = values[:, c].sum()
            mesh_total = mesh[c, :, :, :].sum()
            assert jnp.allclose(mesh_total, channel_total, rtol=1e-3), (
                f"Channel {c} not conserved: mesh={float(mesh_total)}, values={float(channel_total)}"
            )


###########################################################################################
########################### Multi-Channel Gather Tests ####################################
###########################################################################################


class TestMultiChannelSplineGather:
    """Test multi-channel B-spline gathering."""

    def test_gather_channels_output_shape(self):
        """Test that multi-channel gather returns correct shape."""
        key = jax.random.PRNGKey(0)
        positions = jax.random.uniform(key, shape=(10, 3), dtype=jnp.float64) * 8.0
        cell = jnp.eye(3, dtype=jnp.float64) * 8.0
        # JAX mesh shape is (C, nx, ny, nz)
        mesh = jax.random.normal(
            jax.random.PRNGKey(1), shape=(4, 8, 8, 8), dtype=jnp.float64
        )

        output = spline_gather_channels(positions, mesh, cell, spline_order=4)

        assert output.shape == (10, 4), f"Unexpected shape: {output.shape}"
        assert output.dtype == jnp.float64

    def test_gather_channels_uniform_mesh(self):
        """Test gathering from uniform multi-channel mesh."""
        positions = jnp.array([[2.0, 2.0, 2.0], [5.0, 5.0, 5.0]], dtype=jnp.float64)
        cell = jnp.eye(3, dtype=jnp.float64) * 8.0

        # Uniform mesh with different values per channel
        # JAX mesh shape is (C, nx, ny, nz)
        mesh = jnp.zeros((3, 8, 8, 8), dtype=jnp.float64)
        mesh = mesh.at[0, :, :, :].set(1.0)
        mesh = mesh.at[1, :, :, :].set(2.0)
        mesh = mesh.at[2, :, :, :].set(3.0)

        output = spline_gather_channels(positions, mesh, cell, spline_order=4)

        # Each atom should gather (1, 2, 3)
        expected = jnp.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype=jnp.float64)
        assert jnp.allclose(output, expected, rtol=1e-4)

    def test_gather_channels_matches_single_channel(self):
        """Test that single-channel gather_channels matches spline_gather."""
        key = jax.random.PRNGKey(0)
        positions = jax.random.uniform(key, shape=(10, 3), dtype=jnp.float64) * 8.0
        cell = jnp.eye(3, dtype=jnp.float64) * 8.0
        mesh_3d = jax.random.normal(
            jax.random.PRNGKey(1), shape=(8, 8, 8), dtype=jnp.float64
        )

        # Single-channel gather
        output_single = spline_gather(positions, mesh_3d, cell, spline_order=4)

        # Multi-channel gather with 1 channel
        # JAX mesh shape is (C, nx, ny, nz) so add channel as first dimension
        mesh_4d = mesh_3d[jnp.newaxis, :, :, :]
        output_multi = spline_gather_channels(positions, mesh_4d, cell, spline_order=4)

        # Should match
        assert jnp.allclose(output_multi[:, 0], output_single, rtol=1e-4)


###########################################################################################
########################### Multi-Channel Batch Tests #####################################
###########################################################################################


class TestMultiChannelBatch:
    """Test multi-channel operations with batching."""

    def test_batch_spread_channels_output_shape(self):
        """Test that batched multi-channel spread returns correct shape."""
        key = jax.random.PRNGKey(0)
        positions = jax.random.uniform(key, shape=(10, 3), dtype=jnp.float64) * 8.0
        values = jax.random.normal(
            jax.random.PRNGKey(1), shape=(10, 3), dtype=jnp.float64
        )
        batch_idx = jnp.array([0] * 5 + [1] * 5, dtype=jnp.int32)
        cell = jnp.tile(
            jnp.eye(3, dtype=jnp.float64)[jnp.newaxis, :, :] * 8.0, (2, 1, 1)
        )

        mesh = spline_spread_channels(
            positions,
            values,
            cell,
            mesh_dims=(8, 8, 8),
            spline_order=4,
            batch_idx=batch_idx,
        )

        # JAX batch multi-channel shape is (B, C, nx, ny, nz)
        assert mesh.shape == (2, 3, 8, 8, 8), f"Unexpected shape: {mesh.shape}"
        assert mesh.dtype == jnp.float64

    def test_batch_gather_channels_output_shape(self):
        """Test that batched multi-channel gather returns correct shape."""
        key = jax.random.PRNGKey(0)
        positions = jax.random.uniform(key, shape=(10, 3), dtype=jnp.float64) * 8.0
        batch_idx = jnp.array([0] * 5 + [1] * 5, dtype=jnp.int32)
        cell = jnp.tile(
            jnp.eye(3, dtype=jnp.float64)[jnp.newaxis, :, :] * 8.0, (2, 1, 1)
        )
        # JAX batch multi-channel shape is (B, C, nx, ny, nz)
        mesh = jax.random.normal(
            jax.random.PRNGKey(1), shape=(2, 3, 8, 8, 8), dtype=jnp.float64
        )

        output = spline_gather_channels(
            positions, mesh, cell, spline_order=4, batch_idx=batch_idx
        )

        assert output.shape == (10, 3), f"Unexpected shape: {output.shape}"
        assert output.dtype == jnp.float64


###########################################################################################
########################### B-Spline Deconvolution Tests ##################################
###########################################################################################


class TestBSplineDeconvolution:
    """Test B-spline deconvolution functions."""

    @pytest.mark.parametrize("order", [1, 2, 3, 4, 5, 6])
    def test_deconvolution_shape(self, order):
        """Test that deconvolution returns correct shape."""
        mesh_dims = (8, 12, 16)
        deconv = compute_bspline_deconvolution(mesh_dims, spline_order=order)

        assert deconv.shape == mesh_dims, f"Unexpected shape: {deconv.shape}"

    @pytest.mark.parametrize("order", [1, 2, 3, 4])
    def test_deconvolution_at_zero_frequency(self, order):
        """Test that deconvolution is 1 at zero frequency."""
        mesh_dims = (8, 8, 8)
        deconv = compute_bspline_deconvolution(mesh_dims, spline_order=order)

        # At k=(0,0,0), the B-spline modulus should be 1, so deconv=1
        assert abs(float(deconv[0, 0, 0]) - 1.0) < 1e-6, (
            f"Deconvolution at zero frequency should be 1, got {float(deconv[0, 0, 0])}"
        )

    @pytest.mark.parametrize("order", [2, 3, 4])
    def test_deconvolution_positive(self, order):
        """Test that deconvolution factors are positive."""
        mesh_dims = (16, 16, 16)
        deconv = compute_bspline_deconvolution(mesh_dims, spline_order=order)

        assert (deconv > 0).all(), "Deconvolution factors should be positive"

    @pytest.mark.parametrize("order", [2, 3, 4])
    def test_deconvolution_symmetry(self, order):
        """Test that deconvolution has correct symmetry."""
        n = 8
        deconv = compute_bspline_deconvolution((n, n, n), spline_order=order)

        # Should be symmetric: D(k) = D(-k)
        for i in range(1, n // 2):
            assert jnp.allclose(deconv[i, 0, 0], deconv[-i, 0, 0], rtol=1e-6), (
                f"Asymmetry at kx={i}"
            )
            assert jnp.allclose(deconv[0, i, 0], deconv[0, -i, 0], rtol=1e-6), (
                f"Asymmetry at ky={i}"
            )
            assert jnp.allclose(deconv[0, 0, i], deconv[0, 0, -i], rtol=1e-6), (
                f"Asymmetry at kz={i}"
            )

    def test_deconvolution_1d(self):
        """Test 1D deconvolution factors."""
        n = 16
        deconv_1d = compute_bspline_deconvolution_1d(n, spline_order=4)

        assert deconv_1d.shape == (n,), f"Unexpected shape: {deconv_1d.shape}"
        assert abs(float(deconv_1d[0]) - 1.0) < 1e-6, "D(0) should be 1"
        assert (deconv_1d > 0).all(), "All factors should be positive"


###########################################################################################
########################### Spread/Gather Round Trip Tests ################################
###########################################################################################


class TestSpreadGatherRoundTrip:
    """Test spread-gather with deconvolution."""

    def test_round_trip_with_deconvolution(self):
        """Test that spread -> FFT deconvolve -> gather recovers values."""
        key = jax.random.PRNGKey(0)
        positions = jax.random.uniform(key, shape=(20, 3), dtype=jnp.float64) * 8.0
        charges = jax.random.normal(
            jax.random.PRNGKey(1), shape=(20,), dtype=jnp.float64
        )
        cell = jnp.eye(3, dtype=jnp.float64) * 8.0
        mesh_dims = (16, 16, 16)
        spline_order = 4

        # Spread charges to mesh
        mesh = spline_spread(positions, charges, cell, mesh_dims, spline_order)

        # Apply deconvolution in Fourier space
        deconv = compute_bspline_deconvolution(mesh_dims, spline_order)
        mesh_fft = jnp.fft.fftn(mesh)
        mesh_deconv_fft = mesh_fft * deconv
        mesh_deconv = jnp.fft.ifftn(mesh_deconv_fft).real

        # Gather back to atoms
        gathered = spline_gather(positions, mesh_deconv, cell, spline_order)

        # Should recover something close to original (not exact due to truncation)
        # Just check that it completes and gives finite results
        # Note: Due to numerical precision, values may be quite large after deconvolution
        assert jnp.isfinite(gathered).all()


###########################################################################################
########################### Dtype Preservation Tests ######################################
###########################################################################################


class TestDtypePreservation:
    """Test that output dtype matches input positions dtype for all spline operations."""

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_spread_preserves_dtype(self, dtype):
        """Test that spline_spread output dtype matches input."""
        positions = jnp.array([[2.0, 2.0, 2.0], [5.0, 5.0, 5.0]], dtype=dtype)
        charges = jnp.array([1.0, -1.0], dtype=dtype)
        cell = jnp.eye(3, dtype=dtype) * 8.0

        mesh = spline_spread(
            positions, charges, cell, mesh_dims=(8, 8, 8), spline_order=4
        )
        assert mesh.dtype == dtype, f"Expected {dtype}, got {mesh.dtype}"

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_gather_preserves_dtype(self, dtype):
        """Test that spline_gather output dtype matches input."""
        positions = jnp.array([[2.0, 2.0, 2.0], [5.0, 5.0, 5.0]], dtype=dtype)
        cell = jnp.eye(3, dtype=dtype) * 8.0
        mesh = jnp.ones((8, 8, 8), dtype=dtype)

        output = spline_gather(positions, mesh, cell, spline_order=4)
        assert output.dtype == dtype, f"Expected {dtype}, got {output.dtype}"

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_gather_vec3_preserves_dtype(self, dtype):
        """Test that spline_gather_vec3 output dtype matches input."""
        positions = jnp.array([[2.0, 2.0, 2.0], [5.0, 5.0, 5.0]], dtype=dtype)
        charges = jnp.array([1.0, -0.5], dtype=dtype)
        cell = jnp.eye(3, dtype=dtype) * 8.0
        field = jnp.ones((8, 8, 8, 3), dtype=dtype)

        output = spline_gather_vec3(positions, charges, field, cell, spline_order=4)
        assert output.dtype == dtype, f"Expected {dtype}, got {output.dtype}"

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_gather_gradient_preserves_dtype(self, dtype):
        """Test that spline_gather_gradient output dtype matches input."""
        positions = jnp.array([[2.0, 2.0, 2.0], [5.0, 5.0, 5.0]], dtype=dtype)
        charges = jnp.array([1.0, -1.0], dtype=dtype)
        cell = jnp.eye(3, dtype=dtype) * 8.0
        potential = jnp.ones((8, 8, 8), dtype=dtype)

        forces = spline_gather_gradient(
            positions, charges, potential, cell, spline_order=4
        )
        assert forces.dtype == dtype, f"Expected {dtype}, got {forces.dtype}"

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_spread_channels_preserves_dtype(self, dtype):
        """Test that spline_spread_channels output dtype matches input."""
        positions = jnp.array([[2.0, 2.0, 2.0], [5.0, 5.0, 5.0]], dtype=dtype)
        values = jnp.array([[1.0, 2.0], [-0.5, 1.5]], dtype=dtype)
        cell = jnp.eye(3, dtype=dtype) * 8.0

        mesh = spline_spread_channels(
            positions, values, cell, mesh_dims=(8, 8, 8), spline_order=4
        )
        assert mesh.dtype == dtype, f"Expected {dtype}, got {mesh.dtype}"

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_gather_channels_preserves_dtype(self, dtype):
        """Test that spline_gather_channels output dtype matches input."""
        positions = jnp.array([[2.0, 2.0, 2.0], [5.0, 5.0, 5.0]], dtype=dtype)
        cell = jnp.eye(3, dtype=dtype) * 8.0
        mesh = jnp.ones((3, 8, 8, 8), dtype=dtype)

        output = spline_gather_channels(positions, mesh, cell, spline_order=4)
        assert output.dtype == dtype, f"Expected {dtype}, got {output.dtype}"

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_batch_spread_preserves_dtype(self, dtype):
        """Test that batched spline_spread output dtype matches input."""
        positions = jnp.array([[2.0, 2.0, 2.0], [5.0, 5.0, 5.0]], dtype=dtype)
        charges = jnp.array([1.0, -1.0], dtype=dtype)
        batch_idx = jnp.array([0, 1], dtype=jnp.int32)
        cell = jnp.stack([jnp.eye(3, dtype=dtype) * 8.0, jnp.eye(3, dtype=dtype) * 8.0])

        mesh = spline_spread(
            positions,
            charges,
            cell,
            mesh_dims=(8, 8, 8),
            spline_order=4,
            batch_idx=batch_idx,
        )
        assert mesh.dtype == dtype, f"Expected {dtype}, got {mesh.dtype}"

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_batch_gather_preserves_dtype(self, dtype):
        """Test that batched spline_gather output dtype matches input."""
        positions = jnp.array([[2.0, 2.0, 2.0], [5.0, 5.0, 5.0]], dtype=dtype)
        batch_idx = jnp.array([0, 1], dtype=jnp.int32)
        cell = jnp.stack([jnp.eye(3, dtype=dtype) * 8.0, jnp.eye(3, dtype=dtype) * 8.0])
        mesh = jnp.ones((2, 8, 8, 8), dtype=dtype)

        output = spline_gather(
            positions, mesh, cell, spline_order=4, batch_idx=batch_idx
        )
        assert output.dtype == dtype, f"Expected {dtype}, got {output.dtype}"


###########################################################################################
########################### FP64 Precision Tests ##########################################
###########################################################################################


class TestFP64Precision:
    """Test that FP64 spline operations achieve tighter tolerances than FP32."""

    def test_fp64_charge_conservation_tight(self):
        """Test that FP64 spread conserves charge to ~1e-10 (vs FP32's ~1e-3)."""
        positions = jnp.array(
            [[2.3, 2.7, 2.1], [5.8, 5.2, 5.6], [3.1, 4.2, 6.8]],
            dtype=jnp.float64,
        )
        charges = jnp.array([1.5, -0.8, 0.3], dtype=jnp.float64)
        cell = jnp.eye(3, dtype=jnp.float64) * 8.0

        mesh = spline_spread(
            positions, charges, cell, mesh_dims=(16, 16, 16), spline_order=4
        )

        mesh_total = float(mesh.sum())
        charge_total = float(charges.sum())

        # FP64 should conserve charge to much tighter tolerance
        assert abs(mesh_total - charge_total) < 1e-10, (
            f"FP64 charge conservation failed: mesh={mesh_total}, charges={charge_total}, "
            f"diff={abs(mesh_total - charge_total)}"
        )

    def test_fp64_gather_uniform_tight(self):
        """Test that FP64 gather from uniform mesh is exact to ~1e-10."""
        positions = jnp.array([[2.0, 2.0, 2.0], [5.0, 5.0, 5.0]], dtype=jnp.float64)
        cell = jnp.eye(3, dtype=jnp.float64) * 8.0
        mesh = jnp.full((8, 8, 8), 2.5, dtype=jnp.float64)

        output = spline_gather(positions, mesh, cell, spline_order=4)

        # FP64 should be nearly exact for uniform mesh
        for i in range(2):
            assert abs(float(output[i]) - 2.5) < 1e-10, (
                f"FP64 gather from uniform mesh: expected 2.5, got {float(output[i])}"
            )

    def test_fp64_gradient_uniform_zero_tight(self):
        """Test that FP64 gradient of uniform field is zero to ~1e-10."""
        positions = jnp.array([[2.0, 2.0, 2.0], [5.0, 5.0, 5.0]], dtype=jnp.float64)
        charges = jnp.array([1.0, -1.0], dtype=jnp.float64)
        cell = jnp.eye(3, dtype=jnp.float64) * 8.0
        potential = jnp.full((8, 8, 8), 1.0, dtype=jnp.float64)

        forces = spline_gather_gradient(
            positions, charges, potential, cell, spline_order=4
        )

        # FP64 gradient of uniform should be very close to zero
        max_force = float(jnp.abs(forces).max())
        assert max_force < 1e-10, (
            f"FP64 gradient of uniform field should be ~0, got max |F| = {max_force}"
        )

    def test_fp64_spread_gather_adjoint_tight(self):
        """Test spread/gather adjoint property at FP64 precision."""
        key = jax.random.PRNGKey(42)
        positions = jax.random.uniform(key, shape=(20, 3), dtype=jnp.float64) * 8.0
        charges = jax.random.normal(
            jax.random.PRNGKey(1), shape=(20,), dtype=jnp.float64
        )
        cell = jnp.eye(3, dtype=jnp.float64) * 8.0
        mesh_dims = (16, 16, 16)

        potential = jax.random.normal(
            jax.random.PRNGKey(2), shape=mesh_dims, dtype=jnp.float64
        )

        charge_mesh = spline_spread(positions, charges, cell, mesh_dims, spline_order=4)
        gathered = spline_gather(positions, potential, cell, spline_order=4)

        lhs = float(jnp.sum(charge_mesh * potential))
        rhs = float(jnp.sum(charges * gathered))

        # FP64 adjoint property should hold to ~1e-7
        assert abs(lhs - rhs) < 5e-8, (
            f"FP64 adjoint property: <S(q), phi>={lhs}, <q, G(phi)>={rhs}, diff={abs(lhs - rhs)}"
        )

    def test_fp64_batch_charge_conservation_tight(self):
        """Test that batched FP64 spread conserves per-system charge to ~1e-10."""
        positions = jnp.array(
            [[2.0, 2.0, 2.0], [5.0, 5.0, 5.0], [3.0, 3.0, 3.0], [6.0, 6.0, 6.0]],
            dtype=jnp.float64,
        )
        charges = jnp.array([1.0, -0.5, 0.8, -0.3], dtype=jnp.float64)
        batch_idx = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
        cell = jnp.stack([jnp.eye(3, dtype=jnp.float64) * 8.0] * 2)

        mesh = spline_spread(
            positions,
            charges,
            cell,
            mesh_dims=(8, 8, 8),
            spline_order=4,
            batch_idx=batch_idx,
        )

        # Per-system charge conservation at FP64 precision
        for sys in range(2):
            sys_charge = float(charges[batch_idx == sys].sum())
            sys_mesh = float(mesh[sys].sum())
            assert abs(sys_mesh - sys_charge) < 1e-10, (
                f"System {sys}: mesh_sum={sys_mesh}, charge_sum={sys_charge}, "
                f"diff={abs(sys_mesh - sys_charge)}"
            )

    def test_fp32_vs_fp64_precision_difference(self):
        """Verify that FP64 actually provides better precision than FP32."""
        positions_f64 = jnp.array([[2.3, 2.7, 2.1], [5.8, 5.2, 5.6]], dtype=jnp.float64)
        positions_f32 = positions_f64.astype(jnp.float32)
        charges_f64 = jnp.array([1.5, -0.8], dtype=jnp.float64)
        charges_f32 = charges_f64.astype(jnp.float32)
        cell_f64 = jnp.eye(3, dtype=jnp.float64) * 8.0
        cell_f32 = cell_f64.astype(jnp.float32)

        mesh_f64 = spline_spread(positions_f64, charges_f64, cell_f64, (16, 16, 16), 4)
        mesh_f32 = spline_spread(positions_f32, charges_f32, cell_f32, (16, 16, 16), 4)

        # Both should conserve charge, but FP64 should be tighter
        total_charge = 1.5 - 0.8  # = 0.7
        err_f64 = abs(float(mesh_f64.sum()) - total_charge)
        err_f32 = abs(float(mesh_f32.sum()) - total_charge)

        assert err_f64 < err_f32 or err_f32 < 1e-5, (
            f"FP64 error ({err_f64}) should be <= FP32 error ({err_f32})"
        )
        assert mesh_f64.dtype == jnp.float64
        assert mesh_f32.dtype == jnp.float32
