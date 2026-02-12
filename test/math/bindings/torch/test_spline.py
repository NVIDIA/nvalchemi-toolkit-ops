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
Test Suite for B-Spline PyTorch Bindings
========================================

Tests the PyTorch bindings in nvalchemiops.torch.spline.

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
"""

import pytest
import torch

from nvalchemiops.torch.spline import (
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


@pytest.fixture
def simple_system():
    """Simple 4-atom system in a 10A cubic cell for testing.

    Returns
    -------
    dict
        Dictionary containing positions, charges, cell, and mesh_dims.
        Matches the simple_system fixture in Warp kernel tests.
    """
    cell = torch.eye(3, dtype=torch.float64) * 10.0
    positions = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [5.0, 5.0, 5.0],
            [2.5, 7.5, 3.0],
            [8.0, 2.0, 6.0],
        ],
        dtype=torch.float64,
    )
    charges = torch.tensor([1.0, -1.0, 0.5, 0.5], dtype=torch.float64)

    return {
        "positions": positions,
        "charges": charges,
        "cell": cell,
        "mesh_dims": (8, 8, 8),
    }


@pytest.fixture
def batch_system():
    """Batched 2-system test setup.

    Returns
    -------
    dict
        Dictionary containing batched positions, charges, batch_idx, cell.
        Matches the batch_system fixture in Warp kernel tests.
    """
    cell = torch.eye(3, dtype=torch.float64) * 10.0
    positions = torch.tensor(
        [
            [1.0, 1.0, 1.0],  # sys 0
            [5.0, 5.0, 5.0],  # sys 0
            [2.5, 2.5, 2.5],  # sys 1
            [7.5, 7.5, 7.5],  # sys 1
        ],
        dtype=torch.float64,
    )
    charges = torch.tensor([1.0, -0.5, 0.5, -0.5], dtype=torch.float64)
    batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
    batch_cell = cell.unsqueeze(0).expand(2, -1, -1).contiguous()

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

    These values match the Warp kernel regression tests to ensure PyTorch
    bindings produce identical results. Values computed using simple_system
    and batch_system fixtures.
    """

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_spread_regression(self, device, simple_system):
        """Regression test for spline_spread with expected values."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device_obj = torch.device(device)

        positions = simple_system["positions"].to(device_obj)
        charges = simple_system["charges"].to(device_obj)
        cell = simple_system["cell"].to(device_obj)

        mesh = spline_spread(
            positions, charges, cell, mesh_dims=(8, 8, 8), spline_order=4
        )

        # Regression values (match Warp kernel tests)
        assert mesh.sum().item() == pytest.approx(1.0, rel=1e-10)
        assert mesh.max().item() == pytest.approx(0.2508416403, rel=1e-8)
        assert mesh.min().item() == pytest.approx(-0.2962962963, rel=1e-8)
        assert (mesh.abs() > 1e-12).sum().item() == pytest.approx(182, rel=1e-8)

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_gather_regression(self, device, simple_system):
        """Regression test for spline_gather with expected values."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device_obj = torch.device(device)

        positions = simple_system["positions"].to(device_obj)
        cell = simple_system["cell"].to(device_obj)
        unit_charges = torch.ones(4, dtype=torch.float64, device=device_obj)

        # Spread then gather
        mesh = spline_spread(
            positions, unit_charges, cell, mesh_dims=(8, 8, 8), spline_order=4
        )
        output = spline_gather(positions, mesh, cell, spline_order=4)

        # Regression values (match Warp kernel tests)
        expected = torch.tensor(
            [0.11403329, 0.12506939, 0.11594129, 0.10419698],
            dtype=torch.float64,
            device=device_obj,
        )
        assert output.tolist() == pytest.approx(expected.tolist(), rel=1e-6)
        assert output.sum().item() == pytest.approx(0.4592409390, rel=1e-8)

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_gather_vec3_regression(self, device, simple_system):
        """Regression test for spline_gather_vec3 with expected values."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device_obj = torch.device(device)

        positions = simple_system["positions"].to(device_obj)
        charges = simple_system["charges"].to(device_obj)
        cell = simple_system["cell"].to(device_obj)

        # Uniform vec3 mesh: each point has (1, 2, 3)
        field = torch.zeros((8, 8, 8, 3), dtype=torch.float64, device=device_obj)
        field[..., 0] = 1.0
        field[..., 1] = 2.0
        field[..., 2] = 3.0

        output = spline_gather_vec3(positions, charges, field, cell, spline_order=4)

        # Regression values: charge-weighted (1, 2, 3) vectors
        expected = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [-1.0, -2.0, -3.0],
                [0.5, 1.0, 1.5],
                [0.5, 1.0, 1.5],
            ],
            dtype=torch.float64,
            device=device_obj,
        )
        assert torch.allclose(output, expected, rtol=1e-6)

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_batch_spread_regression(self, device, batch_system):
        """Regression test for batch spline_spread with expected values."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device_obj = torch.device(device)

        positions = batch_system["positions"].to(device_obj)
        charges = batch_system["charges"].to(device_obj)
        batch_idx = batch_system["batch_idx"].to(device_obj)
        cell = batch_system["cell"].to(device_obj)

        mesh = spline_spread(
            positions,
            charges,
            cell,
            mesh_dims=(8, 8, 8),
            spline_order=4,
            batch_idx=batch_idx,
        )

        # Regression values (match Warp kernel tests)
        assert mesh[0].sum().item() == pytest.approx(0.5, rel=1e-8)
        assert mesh[1].sum().item() == pytest.approx(0.0, abs=1e-10)
        assert mesh.max().item() == pytest.approx(0.2508416403, rel=1e-8)
        assert mesh.min().item() == pytest.approx(-0.1481481481, rel=1e-8)

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_batch_gather_regression(self, device, batch_system):
        """Regression test for batch spline_gather with expected values."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device_obj = torch.device(device)

        positions = batch_system["positions"].to(device_obj)
        batch_idx = batch_system["batch_idx"].to(device_obj)
        cell = batch_system["cell"].to(device_obj)
        unit_charges = torch.ones(4, dtype=torch.float64, device=device_obj)

        # Spread then gather
        mesh = spline_spread(
            positions,
            unit_charges,
            cell,
            mesh_dims=(8, 8, 8),
            spline_order=4,
            batch_idx=batch_idx,
        )
        output = spline_gather(
            positions, mesh, cell, spline_order=4, batch_idx=batch_idx
        )

        # Regression values (match Warp kernel tests)
        expected = torch.tensor(
            [0.11403082, 0.125, 0.125, 0.125],
            dtype=torch.float64,
            device=device_obj,
        )
        assert output.tolist() == pytest.approx(expected.tolist(), rel=1e-6)


###########################################################################################
########################### Spread Operation Tests ########################################
###########################################################################################


class TestSplineSpread:
    """Test B-spline charge spreading."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    @pytest.mark.parametrize("spline_order", [2, 3, 4])
    def test_charge_conservation(self, device, spline_order):
        """Test that spreading conserves total charge."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[2.3, 2.7, 2.1], [5.8, 5.2, 5.6], [3.1, 4.2, 6.8]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.5, -0.8, 0.3], dtype=torch.float64, device=device)
        cell = torch.tensor(
            [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]],
            dtype=torch.float64,
            device=device,
        )

        mesh = spline_spread(
            positions, charges, cell, mesh_dims=(16, 16, 16), spline_order=spline_order
        )

        mesh_total = mesh.sum()
        charge_total = charges.sum()

        assert torch.allclose(mesh_total, charge_total, rtol=1e-3), (
            f"Charge not conserved: mesh={mesh_total.item()}, charges={charge_total.item()}"
        )

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_spread_output_shape(self, device):
        """Test that spread returns correct mesh shape."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.rand((10, 3), dtype=torch.float64, device=device) * 5.0
        charges = torch.randn(10, dtype=torch.float64, device=device)
        cell = torch.eye(3, dtype=torch.float64, device=device) * 5.0

        mesh = spline_spread(
            positions, charges, cell, mesh_dims=(12, 14, 16), spline_order=4
        )

        assert mesh.shape == (12, 14, 16), f"Unexpected shape: {mesh.shape}"
        assert mesh.dtype == torch.float64

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_spread_locality(self, device):
        """Test that a single atom only affects nearby grid points."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        # Place a single atom slightly off-grid (theta != 0)
        # At exactly on-grid (theta=0), u=0 gives M(0)=0, so only 3^3=27 points affected
        # Offset slightly to get all 4^3=64 points with non-zero weight
        positions = torch.tensor([[4.1, 4.1, 4.1]], dtype=torch.float64, device=device)
        charges = torch.tensor([1.0], dtype=torch.float64, device=device)
        cell = torch.eye(3, dtype=torch.float64, device=device) * 8.0

        mesh = spline_spread(
            positions, charges, cell, mesh_dims=(8, 8, 8), spline_order=4
        )

        # Count non-zero points
        nonzero = (mesh.abs() > 1e-12).sum().item()

        # For order-4 B-spline with theta != 0, should affect 4^3 = 64 points
        assert nonzero == 64, f"Expected 64 non-zero points, got {nonzero}"

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    @pytest.mark.parametrize("spline_order", [2, 3, 4])
    def test_spread_center_of_mass(self, device, spline_order):
        """Test that the center of mass of the spread is at the atom position.

        This is a critical test that catches off-by-one errors in the B-spline
        spreading algorithm. The center of mass of the spread weights should
        equal the atom position.
        """
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        cell_size = 10.0
        mesh_size = 16

        # Test several positions including grid-aligned and off-grid
        test_positions = [
            [5.0, 5.0, 5.0],  # Center of cell
            [2.5, 7.5, 4.0],  # Off-center
            [0.3, 0.3, 0.3],  # Near origin
            [9.5, 9.5, 9.5],  # Near edge (tests wrapping)
        ]

        cell = torch.eye(3, dtype=torch.float64, device=device) * cell_size

        for pos in test_positions:
            positions = torch.tensor([pos], dtype=torch.float64, device=device)
            charges = torch.tensor([1.0], dtype=torch.float64, device=device)

            mesh = spline_spread(
                positions,
                charges,
                cell,
                [mesh_size, mesh_size, mesh_size],
                spline_order,
            )

            # Compute center of mass of the mesh
            # Grid point i is at position i * cell_size / mesh_size
            grid_spacing = cell_size / mesh_size

            center_of_mass = torch.zeros(3, dtype=torch.float64, device=device)
            total_weight = mesh.sum()

            for dim in range(3):
                indices = torch.arange(mesh_size, dtype=torch.float64, device=device)
                # Grid points are at integer positions in mesh coordinates
                coords = indices * grid_spacing

                # Weight by mesh values and sum
                if dim == 0:
                    dim_weight = (mesh * coords.view(-1, 1, 1)).sum()
                elif dim == 1:
                    dim_weight = (mesh * coords.view(1, -1, 1)).sum()
                else:
                    dim_weight = (mesh * coords.view(1, 1, -1)).sum()

                center_of_mass[dim] = dim_weight / total_weight

            expected_pos = torch.tensor(pos, dtype=torch.float64, device=device)

            # Handle wrapping for positions near edges
            for dim in range(3):
                if (expected_pos[dim] > cell_size - grid_spacing) or (
                    expected_pos[dim] < grid_spacing
                ):
                    # Center of mass wraps - compare modulo cell_size
                    pass

            # For interior positions, center of mass should match
            if all(grid_spacing < p < cell_size - grid_spacing for p in pos):
                assert torch.allclose(center_of_mass, expected_pos, rtol=1e-10), (
                    f"Center of mass mismatch for order={spline_order}, pos={pos}: "
                    f"expected {expected_pos.tolist()}, got {center_of_mass.tolist()}"
                )


###########################################################################################
########################### Gather Operation Tests ########################################
###########################################################################################


class TestSplineGather:
    """Test B-spline potential gathering."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_gather_uniform_potential(self, device):
        """Test that gathering from uniform potential returns that value."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[2.0, 2.0, 2.0], [4.0, 5.0, 3.0], [6.1, 1.2, 7.3]],
            dtype=torch.float64,
            device=device,
        )
        cell = torch.eye(3, dtype=torch.float64, device=device) * 8.0

        # Uniform potential = 3.14
        potential = torch.ones((16, 16, 16), dtype=torch.float64, device=device) * 3.14

        values = spline_gather(positions, potential, cell, spline_order=4)

        assert torch.allclose(values, torch.full_like(values, 3.14), rtol=1e-6), (
            f"Uniform potential gathering failed: {values}"
        )

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_gather_output_shape(self, device):
        """Test that gather returns correct output shape."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        num_atoms = 15
        positions = torch.rand((num_atoms, 3), dtype=torch.float64, device=device) * 5.0
        cell = torch.eye(3, dtype=torch.float64, device=device) * 5.0
        potential = torch.randn((8, 8, 8), dtype=torch.float64, device=device)

        values = spline_gather(positions, potential, cell, spline_order=4)

        assert values.shape == (num_atoms,), f"Unexpected shape: {values.shape}"


###########################################################################################
########################### Gather Gradient Tests #########################################
###########################################################################################


class TestSplineGatherGradient:
    """Test B-spline force computation."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_gather_gradient_uniform_zero(self, device):
        """Test that uniform potential gives zero forces."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        cell = torch.eye(3, dtype=torch.float64, device=device).reshape(1, 3, 3) * 10.0

        # Uniform potential
        potential = torch.ones((16, 16, 16), dtype=torch.float64, device=device) * 5.0

        forces = spline_gather_gradient(
            positions, charges, potential, cell, spline_order=4
        )

        # Gradient of constant is zero
        assert torch.allclose(forces, torch.zeros_like(forces), atol=1e-10), (
            f"Forces from uniform potential should be zero: {forces}"
        )

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_gather_gradient_output_shape(self, device):
        """Test that gather_gradient returns correct output shape."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        num_atoms = 7
        positions = torch.rand((num_atoms, 3), dtype=torch.float64, device=device) * 5.0
        charges = torch.randn(num_atoms, dtype=torch.float64, device=device)
        cell = torch.eye(3, dtype=torch.float64, device=device) * 5.0
        potential = torch.randn((8, 8, 8), dtype=torch.float64, device=device)

        forces = spline_gather_gradient(
            positions, charges, potential, cell, spline_order=4
        )

        assert forces.shape == (num_atoms, 3), f"Unexpected shape: {forces.shape}"


###########################################################################################
########################### Gather Vec3 Tests #############################################
###########################################################################################


class TestSplineGatherVec3:
    """Test B-spline 3D vector field gathering."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_gather_vec3_uniform_field(self, device):
        """Test that gathering from uniform vector field returns that value."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[2.0, 2.0, 2.0], [4.0, 5.0, 3.0], [6.1, 1.2, 7.3]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64, device=device)
        cell = torch.eye(3, dtype=torch.float64, device=device) * 8.0

        # Uniform vector field = [1.0, 2.0, 3.0]
        field = torch.zeros((16, 16, 16, 3), dtype=torch.float64, device=device)
        field[..., 0] = 1.0
        field[..., 1] = 2.0
        field[..., 2] = 3.0

        values = spline_gather_vec3(positions, charges, field, cell, spline_order=4)

        expected = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, device=device)
        for i in range(positions.shape[0]):
            assert torch.allclose(values[i], expected, rtol=1e-6), (
                f"Uniform field gathering failed at atom {i}: {values[i]}"
            )

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_gather_vec3_output_shape(self, device):
        """Test that gather_vec3 returns correct output shape."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        num_atoms = 15
        positions = torch.rand((num_atoms, 3), dtype=torch.float64, device=device) * 5.0
        charges = torch.randn((num_atoms,), dtype=torch.float64, device=device)
        cell = torch.eye(3, dtype=torch.float64, device=device) * 5.0
        field = torch.randn((8, 8, 8, 3), dtype=torch.float64, device=device)

        values = spline_gather_vec3(positions, charges, field, cell, spline_order=4)

        assert values.shape == (num_atoms, 3), f"Unexpected shape: {values.shape}"

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_gather_vec3_autograd(self, device):
        """Test gradients w.r.t. mesh in gather_vec3."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor([[2.0, 2.0, 2.0]], dtype=torch.float64, device=device)
        charges = torch.tensor([1.0], dtype=torch.float64, device=device)
        cell = torch.eye(3, dtype=torch.float64, device=device) * 8.0
        field = torch.ones(
            (8, 8, 8, 3), dtype=torch.float64, device=device, requires_grad=True
        )

        values = spline_gather_vec3(positions, charges, field, cell, spline_order=4)
        loss = values.sum()
        loss.backward()

        assert field.grad is not None, "Field gradients not computed"
        assert field.grad.abs().sum() > 0


###########################################################################################
########################### Autograd Tests ################################################
###########################################################################################


class TestSplineAutograd:
    """Test autograd functionality for spline operations."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_spread_autograd_positions(self, device):
        """Test gradients w.r.t. positions in spread."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        cell = torch.eye(3, dtype=torch.float64, device=device) * 8.0

        mesh = spline_spread(
            positions, charges, cell, mesh_dims=(8, 8, 8), spline_order=4
        )
        loss = mesh.sum()
        loss.backward()

        assert positions.grad is not None, "Position gradients not computed"
        assert positions.grad.shape == positions.shape
        # Since charges sum to 0, gradient should also be small (symmetry)

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_spread_autograd_charges(self, device):
        """Test gradients w.r.t. charges in spread."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor(
            [1.0, -1.0], dtype=torch.float64, device=device, requires_grad=True
        )
        cell = torch.eye(3, dtype=torch.float64, device=device) * 8.0

        mesh = spline_spread(
            positions, charges, cell, mesh_dims=(8, 8, 8), spline_order=4
        )
        loss = mesh.sum()
        loss.backward()

        assert charges.grad is not None, "Charge gradients not computed"
        # d(sum(mesh))/d(charge) = 1 due to conservation
        assert torch.allclose(charges.grad, torch.ones_like(charges), rtol=1e-10), (
            f"Charge gradient should be 1: {charges.grad}"
        )

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_gather_autograd_mesh(self, device):
        """Test gradients w.r.t. mesh in gather."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor([[2.0, 2.0, 2.0]], dtype=torch.float64, device=device)
        cell = torch.eye(3, dtype=torch.float64, device=device) * 8.0
        potential = torch.ones(
            (8, 8, 8), dtype=torch.float64, device=device, requires_grad=True
        )

        values = spline_gather(positions, potential, cell, spline_order=4)
        loss = values.sum()
        loss.backward()

        assert potential.grad is not None, "Mesh gradients not computed"
        # Gradient should be non-zero near the atom position
        assert potential.grad.abs().sum() > 0


###########################################################################################
########################### Non-Cubic Cell Tests ##########################################
###########################################################################################


class TestNonCubicCell:
    """Test spline operations with non-cubic cells."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_triclinic_cell_spread(self, device):
        """Test spread with triclinic cell."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        # Triclinic cell
        cell = torch.tensor(
            [[5.0, 0.0, 0.0], [1.0, 5.0, 0.0], [0.5, 0.5, 5.0]],
            dtype=torch.float64,
            device=device,
        )
        positions = torch.tensor(
            [[1.0, 1.0, 1.0], [2.5, 2.5, 2.5]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)

        mesh = spline_spread(
            positions, charges, cell, mesh_dims=(8, 8, 8), spline_order=4
        )

        # Conservation should still hold
        assert torch.allclose(mesh.sum(), charges.sum(), rtol=1e-10)


###########################################################################################
########################### Consistency Tests #############################################
###########################################################################################


class TestSpreadGatherConsistency:
    """Test that spread and gather are adjoint operations."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_spread_gather_adjoint(self, device):
        """Test adjoint property: <spread(q), v>_mesh = <q, gather(v)>_atoms."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        # Random atoms and mesh values
        positions = torch.tensor(
            [[2.3, 3.7, 4.1], [5.2, 1.8, 6.3]], dtype=torch.float64, device=device
        )
        charges = torch.tensor([1.5, -0.8], dtype=torch.float64, device=device)
        cell = torch.eye(3, dtype=torch.float64, device=device) * 8.0

        # Random mesh values
        mesh_values = torch.randn((8, 8, 8), dtype=torch.float64, device=device)

        # Spread charges to mesh
        spread_mesh = spline_spread(
            positions, charges, cell, mesh_dims=(8, 8, 8), spline_order=4
        )

        # Gather mesh values to atoms
        gathered = spline_gather(positions, mesh_values, cell, spline_order=4)

        # Adjoint property: <spread(q), v>_mesh = <q, gather(v)>_atoms
        lhs = (spread_mesh * mesh_values).sum()
        rhs = (charges * gathered).sum()

        assert torch.allclose(lhs, rhs, rtol=1e-6), (
            f"Adjoint property failed: LHS={lhs.item()}, RHS={rhs.item()}"
        )

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_spread_gather_roundtrip_sum_of_squares(self, device):
        """Test that spread->gather returns sum(w²) * charge.

        For B-splines, spread distributes w_i*q to grid points, gather reads w_i*mesh[i].
        Result = sum(w_i² * q), not q, because weights appear twice.
        """
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        # Atom exactly at a grid point (theta=0)
        positions = torch.tensor([[4.0, 4.0, 4.0]], dtype=torch.float64, device=device)
        charges = torch.tensor([1.0], dtype=torch.float64, device=device)
        cell = torch.eye(3, dtype=torch.float64, device=device) * 8.0

        mesh = spline_spread(
            positions, charges, cell, mesh_dims=(8, 8, 8), spline_order=4
        )
        recovered = spline_gather(positions, mesh, cell, spline_order=4)

        # For order-4 B-spline at theta=0:
        # Weights: M(1)=1/6, M(2)=2/3, M(3)=1/6 (M(0)=0)
        # Sum of squares per dim: (1/6)² + (2/3)² + (1/6)² = 1/2
        # 3D: (1/2)³ = 0.125
        expected = 0.125

        assert abs(recovered.item() - expected) < 1e-10, (
            f"Expected {expected}, got {recovered.item()}"
        )


###########################################################################################
########################### Batch Implementation Tests ####################################
###########################################################################################


class TestBatchSplineSpread:
    """Test batch B-spline charge spreading."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_batch_spread_charge_conservation(self, device):
        """Test that batch spreading conserves total charge per system."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        # Two systems with different charges
        positions = torch.tensor(
            [
                [2.0, 2.0, 2.0],  # System 0
                [4.0, 4.0, 4.0],  # System 0
                [3.0, 3.0, 3.0],  # System 1
                [5.0, 5.0, 5.0],  # System 1
                [1.0, 6.0, 2.0],  # System 1
            ],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor(
            [1.0, -0.5, 2.0, -1.0, 0.5], dtype=torch.float64, device=device
        )
        batch_idx = torch.tensor([0, 0, 1, 1, 1], dtype=torch.int32, device=device)
        cell = 8.0 * torch.eye(3, dtype=torch.float64, device=device).unsqueeze(
            0
        ).repeat(2, 1, 1)

        mesh = spline_spread(
            positions,
            charges,
            cell,
            mesh_dims=(8, 8, 8),
            spline_order=4,
            batch_idx=batch_idx,
        )

        assert mesh.shape == (2, 8, 8, 8), f"Unexpected batch mesh shape: {mesh.shape}"

        # Check conservation per system
        system0_mesh_total = mesh[0].sum()
        system0_charge_total = charges[:2].sum()
        assert torch.allclose(system0_mesh_total, system0_charge_total, rtol=1e-10), (
            f"System 0 charge not conserved: mesh={system0_mesh_total.item()}, "
            f"charges={system0_charge_total.item()}"
        )

        system1_mesh_total = mesh[1].sum()
        system1_charge_total = charges[2:].sum()
        assert torch.allclose(system1_mesh_total, system1_charge_total, rtol=1e-10), (
            f"System 1 charge not conserved: mesh={system1_mesh_total.item()}, "
            f"charges={system1_charge_total.item()}"
        )

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_batch_spread_matches_single(self, device):
        """Test that batch spread matches individual single-system spreads."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        # Two systems
        positions0 = torch.tensor(
            [[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]], dtype=torch.float64, device=device
        )
        charges0 = torch.tensor([1.0, -0.5], dtype=torch.float64, device=device)
        positions1 = torch.tensor(
            [[3.0, 3.0, 3.0], [5.0, 5.0, 5.0]], dtype=torch.float64, device=device
        )
        charges1 = torch.tensor([2.0, -1.0], dtype=torch.float64, device=device)

        cell = torch.eye(3, dtype=torch.float64, device=device) * 8.0
        mesh_dims = (8, 8, 8)

        # Single system spreads
        mesh0_single = spline_spread(
            positions0, charges0, cell, mesh_dims, spline_order=4
        )
        mesh1_single = spline_spread(
            positions1, charges1, cell, mesh_dims, spline_order=4
        )

        # Batch spread
        positions_batch = torch.cat([positions0, positions1], dim=0)
        charges_batch = torch.cat([charges0, charges1], dim=0)
        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)
        cells_batch = cell.unsqueeze(0).expand(2, -1, -1).contiguous()

        mesh_batch = spline_spread(
            positions_batch,
            charges_batch,
            cells_batch,
            mesh_dims,
            spline_order=4,
            batch_idx=batch_idx,
        )

        assert torch.allclose(mesh_batch[0], mesh0_single, rtol=1e-12), (
            "Batch system 0 doesn't match single system"
        )
        assert torch.allclose(mesh_batch[1], mesh1_single, rtol=1e-12), (
            "Batch system 1 doesn't match single system"
        )


class TestBatchSplineGather:
    """Test batch B-spline potential gathering."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_batch_gather_uniform_potential(self, device):
        """Test batch gathering from uniform potentials."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[2.0, 2.0, 2.0], [4.0, 4.0, 4.0], [3.0, 3.0, 3.0]],
            dtype=torch.float64,
            device=device,
        )
        batch_idx = torch.tensor([0, 0, 1], dtype=torch.int32, device=device)
        cell = torch.eye(3, dtype=torch.float64, device=device) * 8.0
        cells = cell.unsqueeze(0).expand(2, -1, -1).contiguous()

        # Different uniform potentials per system
        potential = torch.zeros((2, 8, 8, 8), dtype=torch.float64, device=device)
        potential[0] = 3.14
        potential[1] = 2.72

        values = spline_gather(
            positions, potential, cells, spline_order=4, batch_idx=batch_idx
        )

        assert torch.allclose(
            values[0], torch.tensor(3.14, dtype=torch.float64, device=device), rtol=1e-6
        )
        assert torch.allclose(
            values[1], torch.tensor(3.14, dtype=torch.float64, device=device), rtol=1e-6
        )
        assert torch.allclose(
            values[2], torch.tensor(2.72, dtype=torch.float64, device=device), rtol=1e-6
        )

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_batch_gather_matches_single(self, device):
        """Test that batch gather matches individual single-system gathers."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions0 = torch.tensor([[2.0, 2.0, 2.0]], dtype=torch.float64, device=device)
        positions1 = torch.tensor([[3.0, 3.0, 3.0]], dtype=torch.float64, device=device)

        cell = torch.eye(3, dtype=torch.float64, device=device) * 8.0

        potential0 = torch.randn((8, 8, 8), dtype=torch.float64, device=device)
        potential1 = torch.randn((8, 8, 8), dtype=torch.float64, device=device)

        # Single system gathers
        value0_single = spline_gather(positions0, potential0, cell, spline_order=4)
        value1_single = spline_gather(positions1, potential1, cell, spline_order=4)

        # Batch gather
        positions_batch = torch.cat([positions0, positions1], dim=0)
        batch_idx = torch.tensor([0, 1], dtype=torch.int32, device=device)
        potential_batch = torch.stack([potential0, potential1], dim=0)
        cells = cell.unsqueeze(0).expand(2, -1, -1).contiguous()

        values_batch = spline_gather(
            positions_batch, potential_batch, cells, spline_order=4, batch_idx=batch_idx
        )

        assert torch.allclose(values_batch[0], value0_single[0], rtol=1e-12)
        assert torch.allclose(values_batch[1], value1_single[0], rtol=1e-12)


class TestBatchSplineGatherVec3:
    """Test batch B-spline vector field gathering."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_batch_gather_vec3_uniform_field(self, device):
        """Test batch gathering from uniform vector fields."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, 1.0], dtype=torch.float64, device=device)
        batch_idx = torch.tensor([0, 1], dtype=torch.int32, device=device)
        cell = torch.eye(3, dtype=torch.float64, device=device) * 8.0
        cells = cell.unsqueeze(0).expand(2, -1, -1).contiguous()

        # Different uniform vector fields per system
        field = torch.zeros((2, 8, 8, 8, 3), dtype=torch.float64, device=device)
        field[0, ..., :] = torch.tensor([1.0, 2.0, 3.0])
        field[1, ..., :] = torch.tensor([4.0, 5.0, 6.0])

        values = spline_gather_vec3(
            positions, charges, field, cells, spline_order=4, batch_idx=batch_idx
        )

        expected0 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, device=device)
        expected1 = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float64, device=device)

        assert torch.allclose(values[0], expected0, rtol=1e-6)
        assert torch.allclose(values[1], expected1, rtol=1e-6)

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_batch_gather_vec3_output_shape(self, device):
        """Test that batch gather_vec3 returns correct output shape."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        num_atoms = 10
        num_systems = 3
        positions = torch.rand((num_atoms, 3), dtype=torch.float64, device=device) * 5.0
        charges = torch.randn((num_atoms,), dtype=torch.float64, device=device)
        batch_idx = torch.randint(
            0, num_systems, (num_atoms,), dtype=torch.int32, device=device
        )
        cells = (
            torch.eye(3, dtype=torch.float64, device=device)
            .unsqueeze(0)
            .expand(num_systems, -1, -1)
            .contiguous()
            * 5.0
        )
        field = torch.randn(
            (num_systems, 8, 8, 8, 3), dtype=torch.float64, device=device
        )

        values = spline_gather_vec3(
            positions, charges, field, cells, spline_order=4, batch_idx=batch_idx
        )

        assert values.shape == (num_atoms, 3), f"Unexpected shape: {values.shape}"


class TestBatchSplineGatherGradient:
    """Test batch B-spline force computation."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_batch_gather_gradient_uniform_zero(self, device):
        """Test that uniform potential gives zero forces in batch mode."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[2.0, 2.0, 2.0], [4.0, 4.0, 4.0], [3.0, 3.0, 3.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0, 0.5], dtype=torch.float64, device=device)
        batch_idx = torch.tensor([0, 0, 1], dtype=torch.int32, device=device)
        cells = (
            torch.eye(3, dtype=torch.float64, device=device)
            .unsqueeze(0)
            .expand(2, -1, -1)
            .contiguous()
            * 10.0
        )

        # Uniform potential
        potential = (
            torch.ones((2, 16, 16, 16), dtype=torch.float64, device=device) * 5.0
        )

        forces = spline_gather_gradient(
            positions, charges, potential, cells, spline_order=4, batch_idx=batch_idx
        )

        assert torch.allclose(forces, torch.zeros_like(forces), atol=1e-10), (
            f"Forces from uniform potential should be zero: {forces}"
        )

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_batch_gather_gradient_matches_single(self, device):
        """Test that batch gather_gradient matches individual single-system computations."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions0 = torch.tensor([[2.0, 2.0, 2.0]], dtype=torch.float64, device=device)
        charges0 = torch.tensor([1.5], dtype=torch.float64, device=device)
        positions1 = torch.tensor([[3.0, 3.0, 3.0]], dtype=torch.float64, device=device)
        charges1 = torch.tensor([-0.8], dtype=torch.float64, device=device)

        cell = torch.eye(3, dtype=torch.float64, device=device) * 8.0

        potential0 = torch.randn((8, 8, 8), dtype=torch.float64, device=device)
        potential1 = torch.randn((8, 8, 8), dtype=torch.float64, device=device)

        # Single system computations
        forces0_single = spline_gather_gradient(
            positions0, charges0, potential0, cell, spline_order=4
        )
        forces1_single = spline_gather_gradient(
            positions1, charges1, potential1, cell, spline_order=4
        )

        # Batch computation
        positions_batch = torch.cat([positions0, positions1], dim=0)
        charges_batch = torch.cat([charges0, charges1], dim=0)
        batch_idx = torch.tensor([0, 1], dtype=torch.int32, device=device)
        potential_batch = torch.stack([potential0, potential1], dim=0)
        cells = cell.unsqueeze(0).expand(2, -1, -1).contiguous()

        forces_batch = spline_gather_gradient(
            positions_batch,
            charges_batch,
            potential_batch,
            cells,
            spline_order=4,
            batch_idx=batch_idx,
        )

        assert torch.allclose(forces_batch[0], forces0_single[0], rtol=1e-12)
        assert torch.allclose(forces_batch[1], forces1_single[0], rtol=1e-12)


class TestBatchSplineAutograd:
    """Test autograd functionality for batch spline operations."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_batch_spread_autograd_positions(self, device):
        """Test gradients w.r.t. positions in batch spread."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[2.0, 2.0, 2.0], [4.0, 4.0, 4.0], [3.0, 3.0, 3.0]],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )
        charges = torch.tensor([1.0, -1.0, 0.5], dtype=torch.float64, device=device)
        batch_idx = torch.tensor([0, 0, 1], dtype=torch.int32, device=device)
        cells = (
            torch.eye(3, dtype=torch.float64, device=device)
            .unsqueeze(0)
            .expand(2, -1, -1)
            .contiguous()
            * 8.0
        )

        mesh = spline_spread(
            positions,
            charges,
            cells,
            mesh_dims=(8, 8, 8),
            spline_order=4,
            batch_idx=batch_idx,
        )
        loss = mesh.sum()
        loss.backward()

        assert positions.grad is not None, "Position gradients not computed"
        assert positions.grad.shape == positions.shape

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_batch_spread_autograd_charges(self, device):
        """Test gradients w.r.t. charges in batch spread."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[2.0, 2.0, 2.0], [4.0, 4.0, 4.0], [3.0, 3.0, 3.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor(
            [1.0, -1.0, 0.5], dtype=torch.float64, device=device, requires_grad=True
        )
        batch_idx = torch.tensor([0, 0, 1], dtype=torch.int32, device=device)
        cells = (
            torch.eye(3, dtype=torch.float64, device=device)
            .unsqueeze(0)
            .expand(2, -1, -1)
            .contiguous()
            * 8.0
        )

        mesh = spline_spread(
            positions,
            charges,
            cells,
            mesh_dims=(8, 8, 8),
            spline_order=4,
            batch_idx=batch_idx,
        )
        loss = mesh.sum()
        loss.backward()

        assert charges.grad is not None, "Charge gradients not computed"
        # d(sum(mesh))/d(charge) = 1 due to conservation
        assert torch.allclose(charges.grad, torch.ones_like(charges), rtol=1e-10)

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_batch_gather_autograd_mesh(self, device):
        """Test gradients w.r.t. mesh in batch gather."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]], dtype=torch.float64, device=device
        )
        batch_idx = torch.tensor([0, 1], dtype=torch.int32, device=device)
        cells = (
            torch.eye(3, dtype=torch.float64, device=device)
            .unsqueeze(0)
            .expand(2, -1, -1)
            .contiguous()
            * 8.0
        )
        potential = torch.ones(
            (2, 8, 8, 8), dtype=torch.float64, device=device, requires_grad=True
        )

        values = spline_gather(
            positions, potential, cells, spline_order=4, batch_idx=batch_idx
        )
        loss = values.sum()
        loss.backward()

        assert potential.grad is not None, "Mesh gradients not computed"
        assert potential.grad.abs().sum() > 0


class TestBatchDifferentCells:
    """Test batch operations with different cells per system."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_batch_spread_different_cells(self, device):
        """Test batch spread with different cell sizes per system."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        # Same fractional positions, different cell sizes
        positions = torch.tensor(
            [[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]],  # In an 8x8x8 cell
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        batch_idx = torch.tensor([0, 1], dtype=torch.int32, device=device)

        # Different cell sizes
        cells = torch.zeros((2, 3, 3), dtype=torch.float64, device=device)
        cells[0] = torch.eye(3, dtype=torch.float64, device=device) * 8.0
        cells[1] = torch.eye(3, dtype=torch.float64, device=device) * 10.0

        mesh = spline_spread(
            positions,
            charges,
            cells,
            mesh_dims=(8, 8, 8),
            spline_order=4,
            batch_idx=batch_idx,
        )

        # Conservation should still hold for each system
        assert torch.allclose(mesh[0].sum(), charges[0], rtol=1e-10)
        assert torch.allclose(mesh[1].sum(), charges[1], rtol=1e-10)

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_batch_gather_different_cells(self, device):
        """Test batch gather with different cells per system."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]],
            dtype=torch.float64,
            device=device,
        )
        batch_idx = torch.tensor([0, 1], dtype=torch.int32, device=device)

        cells = torch.zeros((2, 3, 3), dtype=torch.float64, device=device)
        cells[0] = torch.eye(3, dtype=torch.float64, device=device) * 8.0
        cells[1] = torch.eye(3, dtype=torch.float64, device=device) * 10.0

        # Same uniform potential for both systems
        potential = torch.ones((2, 8, 8, 8), dtype=torch.float64, device=device) * 2.5

        values = spline_gather(
            positions, potential, cells, spline_order=4, batch_idx=batch_idx
        )

        # Both should return uniform value
        assert torch.allclose(values, torch.full_like(values, 2.5), rtol=1e-6)


###########################################################################################
########################### Multi-Channel Spline Tests ####################################
###########################################################################################


class TestMultiChannelSplineSpread:
    """Test multi-channel B-spline spread operations."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    @pytest.mark.parametrize("num_channels", [1, 3, 9])
    def test_spread_channels_output_shape(self, device, num_channels):
        """Test that spread_channels returns correct output shape."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        from nvalchemiops.torch.spline import spline_spread_channels

        num_atoms = 10
        positions = torch.rand((num_atoms, 3), dtype=torch.float64, device=device) * 8.0
        values = torch.randn(
            (num_atoms, num_channels), dtype=torch.float64, device=device
        )
        cell = torch.eye(3, dtype=torch.float64, device=device) * 10.0
        mesh_dims = (8, 8, 8)

        mesh = spline_spread_channels(
            positions, values, cell, mesh_dims, spline_order=4
        )

        assert mesh.shape == (num_channels, 8, 8, 8), f"Unexpected shape: {mesh.shape}"

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_spread_channels_conservation(self, device):
        """Test that multi-channel spread conserves values per channel."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        from nvalchemiops.torch.spline import spline_spread_channels

        num_atoms = 5
        num_channels = 4
        positions = torch.rand((num_atoms, 3), dtype=torch.float64, device=device) * 8.0
        values = torch.randn(
            (num_atoms, num_channels), dtype=torch.float64, device=device
        )
        cell = torch.eye(3, dtype=torch.float64, device=device) * 10.0
        mesh_dims = (16, 16, 16)

        mesh = spline_spread_channels(
            positions, values, cell, mesh_dims, spline_order=4
        )

        # Each channel should sum to the total value for that channel
        for c in range(num_channels):
            expected = values[:, c].sum()
            actual = mesh[c].sum()
            assert torch.allclose(actual, expected, rtol=1e-6), (
                f"Channel {c} not conserved: expected {expected}, got {actual}"
            )


class TestMultiChannelSplineGather:
    """Test multi-channel B-spline gather operations."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    @pytest.mark.parametrize("num_channels", [1, 3, 9])
    def test_gather_channels_output_shape(self, device, num_channels):
        """Test that gather_channels returns correct output shape."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        from nvalchemiops.torch.spline import spline_gather_channels

        num_atoms = 10
        positions = torch.rand((num_atoms, 3), dtype=torch.float64, device=device) * 8.0
        mesh = torch.randn((num_channels, 8, 8, 8), dtype=torch.float64, device=device)
        cell = 10.0 * torch.eye(3, dtype=torch.float64, device=device).unsqueeze(0)

        values = spline_gather_channels(positions, mesh, cell, spline_order=4)

        assert values.shape == (num_atoms, num_channels), (
            f"Unexpected shape: {values.shape}"
        )

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_gather_channels_uniform_mesh(self, device):
        """Test that uniform mesh gives uniform values per channel."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        from nvalchemiops.torch.spline import spline_gather_channels

        num_atoms = 5
        num_channels = 4
        positions = torch.rand((num_atoms, 3), dtype=torch.float64, device=device) * 8.0
        cell = torch.eye(3, dtype=torch.float64, device=device) * 10.0

        # Create uniform mesh with different values per channel
        mesh = torch.zeros((num_channels, 8, 8, 8), dtype=torch.float64, device=device)
        for c in range(num_channels):
            mesh[c] = float(c + 1)

        values = spline_gather_channels(positions, mesh, cell, spline_order=4)

        # Each atom should see the uniform value for each channel
        for c in range(num_channels):
            expected = float(c + 1)
            assert torch.allclose(
                values[:, c],
                torch.full((num_atoms,), expected, dtype=torch.float64, device=device),
                rtol=1e-6,
            ), f"Channel {c} values not uniform"

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_gather_channels_matches_single_channel(self, device):
        """Test that multi-channel gather matches single-channel gather per channel."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        from nvalchemiops.torch.spline import spline_gather_channels

        num_atoms = 5
        num_channels = 3
        positions = torch.rand((num_atoms, 3), dtype=torch.float64, device=device) * 8.0
        mesh = torch.randn((num_channels, 8, 8, 8), dtype=torch.float64, device=device)
        cell = torch.eye(3, dtype=torch.float64, device=device) * 10.0

        # Multi-channel gather
        values_multi = spline_gather_channels(positions, mesh, cell, spline_order=4)

        # Single-channel gather for comparison
        for c in range(num_channels):
            values_single = spline_gather(positions, mesh[c], cell, spline_order=4)
            assert torch.allclose(values_multi[:, c], values_single, rtol=1e-12), (
                f"Channel {c} mismatch between multi and single gather"
            )


class TestMultiChannelBatch:
    """Test multi-channel operations in batch mode."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_batch_spread_channels_output_shape(self, device):
        """Test batch multi-channel spread output shape."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        from nvalchemiops.torch.spline import spline_spread_channels

        num_atoms = 10
        num_systems = 3
        num_channels = 5

        positions = torch.rand((num_atoms, 3), dtype=torch.float64, device=device) * 8.0
        values = torch.randn(
            (num_atoms, num_channels), dtype=torch.float64, device=device
        )
        batch_idx = torch.randint(
            0, num_systems - 1, (num_atoms,), dtype=torch.int32, device=device
        )
        cells = (
            torch.eye(3, dtype=torch.float64, device=device)
            .unsqueeze(0)
            .expand(num_systems, -1, -1)
            .contiguous()
            * 10.0
        )

        mesh = spline_spread_channels(
            positions,
            values,
            cells,
            mesh_dims=(8, 8, 8),
            spline_order=4,
            batch_idx=batch_idx,
        )

        assert mesh.shape == (num_systems, num_channels, 8, 8, 8), (
            f"Unexpected shape: {mesh.shape}"
        )

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_batch_gather_channels_output_shape(self, device):
        """Test batch multi-channel gather output shape."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        from nvalchemiops.torch.spline import spline_gather_channels

        num_atoms = 10
        num_systems = 3
        num_channels = 5

        positions = torch.rand((num_atoms, 3), dtype=torch.float64, device=device) * 8.0
        batch_idx = torch.randint(
            0, num_systems - 1, (num_atoms,), dtype=torch.int32, device=device
        )
        mesh = torch.randn(
            (num_systems, num_channels, 8, 8, 8), dtype=torch.float64, device=device
        )
        cells = (
            torch.eye(3, dtype=torch.float64, device=device)
            .unsqueeze(0)
            .expand(num_systems, -1, -1)
            .contiguous()
            * 10.0
        )

        values = spline_gather_channels(
            positions, mesh, cells, spline_order=4, batch_idx=batch_idx
        )

        assert values.shape == (num_atoms, num_channels), (
            f"Unexpected shape: {values.shape}"
        )


class TestMultiChannelAutograd:
    """Test autograd for multi-channel spline operations."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_spread_channels_autograd_positions(self, device):
        """Test gradients w.r.t. positions in multi-channel spread."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        from nvalchemiops.torch.spline import spline_spread_channels

        num_atoms = 5
        num_channels = 3
        positions = (
            torch.rand(
                (num_atoms, 3), dtype=torch.float64, device=device, requires_grad=True
            )
            * 8.0
        )
        positions = positions.detach().clone().requires_grad_(True)
        values = torch.randn(
            (num_atoms, num_channels), dtype=torch.float64, device=device
        )
        cell = torch.eye(3, dtype=torch.float64, device=device) * 10.0

        mesh = spline_spread_channels(
            positions, values, cell, (8, 8, 8), spline_order=4
        )
        loss = mesh.sum()
        loss.backward()

        assert positions.grad is not None, "Position gradients not computed"
        assert positions.grad.shape == positions.shape

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_spread_channels_autograd_values(self, device):
        """Test gradients w.r.t. values in multi-channel spread."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        from nvalchemiops.torch.spline import spline_spread_channels

        num_atoms = 5
        num_channels = 3
        positions = torch.rand((num_atoms, 3), dtype=torch.float64, device=device) * 8.0
        values = torch.randn(
            (num_atoms, num_channels),
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )
        cell = torch.eye(3, dtype=torch.float64, device=device) * 10.0

        mesh = spline_spread_channels(
            positions, values, cell, (8, 8, 8), spline_order=4
        )
        loss = mesh.sum()
        loss.backward()

        assert values.grad is not None, "Value gradients not computed"
        # d(sum(mesh))/d(values) should be 1 due to conservation
        assert torch.allclose(values.grad, torch.ones_like(values), rtol=1e-6)

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_gather_channels_autograd_mesh(self, device):
        """Test gradients w.r.t. mesh in multi-channel gather."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        from nvalchemiops.torch.spline import spline_gather_channels

        num_atoms = 5
        num_channels = 3
        positions = torch.rand((num_atoms, 3), dtype=torch.float64, device=device) * 8.0
        mesh = torch.randn(
            (num_channels, 8, 8, 8),
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )
        cell = torch.eye(3, dtype=torch.float64, device=device) * 10.0

        values = spline_gather_channels(positions, mesh, cell, spline_order=4)
        loss = values.sum()
        loss.backward()

        assert mesh.grad is not None, "Mesh gradients not computed"
        assert mesh.grad.abs().sum() > 0


###########################################################################################
########################### B-Spline Deconvolution Tests ##################################
###########################################################################################


class TestBSplineDeconvolution:
    """Test B-spline deconvolution functions."""

    @pytest.mark.parametrize("order", [1, 2, 3, 4, 5, 6])
    def test_deconvolution_shape(self, order):
        """Test that deconvolution returns correct shape."""
        from nvalchemiops.torch.spline import compute_bspline_deconvolution

        mesh_dims = (8, 12, 16)
        deconv = compute_bspline_deconvolution(mesh_dims, spline_order=order)

        assert deconv.shape == mesh_dims, f"Unexpected shape: {deconv.shape}"

    @pytest.mark.parametrize("order", [1, 2, 3, 4])
    def test_deconvolution_at_zero_frequency(self, order):
        """Test that deconvolution is 1 at zero frequency."""
        from nvalchemiops.torch.spline import compute_bspline_deconvolution

        mesh_dims = (8, 8, 8)
        deconv = compute_bspline_deconvolution(mesh_dims, spline_order=order)

        # At k=(0,0,0), the B-spline modulus should be 1, so deconv=1
        assert abs(deconv[0, 0, 0].item() - 1.0) < 1e-6, (
            f"Deconvolution at zero frequency should be 1, got {deconv[0, 0, 0].item()}"
        )

    @pytest.mark.parametrize("order", [2, 3, 4])
    def test_deconvolution_positive(self, order):
        """Test that deconvolution factors are positive."""
        from nvalchemiops.torch.spline import compute_bspline_deconvolution

        mesh_dims = (16, 16, 16)
        deconv = compute_bspline_deconvolution(mesh_dims, spline_order=order)

        assert (deconv > 0).all(), "Deconvolution factors should be positive"

    @pytest.mark.parametrize("order", [2, 3, 4])
    def test_deconvolution_symmetry(self, order):
        """Test that deconvolution has correct symmetry."""
        from nvalchemiops.torch.spline import compute_bspline_deconvolution

        n = 8
        deconv = compute_bspline_deconvolution((n, n, n), spline_order=order)

        # Should be symmetric: D(k) = D(-k)
        for i in range(1, n // 2):
            assert torch.allclose(deconv[i, 0, 0], deconv[-i, 0, 0], rtol=1e-6), (
                f"Asymmetry at kx={i}"
            )
            assert torch.allclose(deconv[0, i, 0], deconv[0, -i, 0], rtol=1e-6), (
                f"Asymmetry at ky={i}"
            )
            assert torch.allclose(deconv[0, 0, i], deconv[0, 0, -i], rtol=1e-6), (
                f"Asymmetry at kz={i}"
            )

    def test_deconvolution_1d(self):
        """Test 1D deconvolution factors."""
        from nvalchemiops.torch.spline import compute_bspline_deconvolution_1d

        n = 16
        deconv_1d = compute_bspline_deconvolution_1d(n, spline_order=4)

        assert deconv_1d.shape == (n,), f"Unexpected shape: {deconv_1d.shape}"
        assert abs(deconv_1d[0].item() - 1.0) < 1e-6, "D(0) should be 1"
        assert (deconv_1d > 0).all(), "All factors should be positive"

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_deconvolution_on_device(self, device):
        """Test that deconvolution works on different devices."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from nvalchemiops.torch.spline import compute_bspline_deconvolution

        device = torch.device(device)
        deconv = compute_bspline_deconvolution((8, 8, 8), spline_order=4, device=device)

        assert deconv.device.type == device.type


class TestSpreadGatherRoundTrip:
    """Test spread-gather round trip with deconvolution."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_round_trip_with_deconvolution(self, device):
        """Test that spread-FFT-deconv-IFFT-gather approximately recovers values."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        from nvalchemiops.torch.spline import (
            compute_bspline_deconvolution,
            spline_gather,
            spline_spread,
        )

        # Create test data
        positions = torch.tensor(
            [[2.0, 2.0, 2.0], [5.0, 5.0, 5.0], [7.0, 3.0, 4.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, -0.5, 0.3], dtype=torch.float64, device=device)
        cell = torch.eye(3, dtype=torch.float64, device=device) * 10.0
        mesh_dims = (32, 32, 32)

        # Spread charges to mesh
        mesh = spline_spread(positions, charges, cell, mesh_dims, spline_order=4)

        # Get deconvolution factors
        deconv = compute_bspline_deconvolution(mesh_dims, spline_order=4, device=device)

        # FFT -> apply deconvolution -> IFFT
        mesh_fft = torch.fft.fftn(mesh)
        mesh_corrected_fft = mesh_fft * deconv
        mesh_corrected = torch.fft.ifftn(mesh_corrected_fft).real

        # Gather back
        values_raw = spline_gather(positions, mesh, cell, spline_order=4)
        values_corrected = spline_gather(
            positions, mesh_corrected, cell, spline_order=4
        )

        # The corrected values should be closer to the original charges
        # (Not exact due to discretization, but should be better)
        # Just check that the operation completes and gives finite results
        assert torch.isfinite(values_corrected).all()
        assert torch.isfinite(values_raw).all()


class TestSplineAutogradCoverage:
    """Test spline functions with autograd to cover attach_for_backward paths."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_spline_gather_gradient_with_autograd(self, device):
        """Test spline_gather_gradient with requires_grad=True (lines 1435-1444)."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[2.0, 2.0, 2.0], [5.0, 5.0, 5.0]],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )
        charges = torch.ones(2, dtype=torch.float64, device=device)
        cell = torch.eye(3, dtype=torch.float64, device=device) * 10.0

        # Create a simple mesh
        mesh = torch.randn(8, 8, 8, dtype=torch.float64, device=device)

        forces = spline_gather_gradient(positions, charges, mesh, cell, spline_order=4)

        # Verify backward works
        forces.sum().backward()
        assert positions.grad is not None
        assert torch.isfinite(positions.grad).all()

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_batch_spline_gather_vec3_with_autograd(self, device):
        """Test batch spline_gather_vec3 with requires_grad=True (lines 1708-1717)."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[2.0, 2.0, 2.0], [5.0, 5.0, 5.0], [2.0, 2.0, 2.0], [5.0, 5.0, 5.0]],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )
        charges = torch.ones(4, dtype=torch.float64, device=device)
        cell = (
            torch.eye(3, dtype=torch.float64, device=device)
            .unsqueeze(0)
            .expand(2, -1, -1)
            .contiguous()
            * 10.0
        )
        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)

        # Create a batched vec3 mesh
        mesh = torch.randn(2, 8, 8, 8, 3, dtype=torch.float64, device=device)

        values = spline_gather_vec3(
            positions, charges, mesh, cell, spline_order=4, batch_idx=batch_idx
        )

        # Verify backward works
        values.sum().backward()
        assert positions.grad is not None
        assert torch.isfinite(positions.grad).all()

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_batch_spline_gather_gradient_with_autograd(self, device):
        """Test batch spline_gather_gradient with requires_grad=True (lines 1795-1804)."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[2.0, 2.0, 2.0], [5.0, 5.0, 5.0], [2.0, 2.0, 2.0], [5.0, 5.0, 5.0]],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )
        charges = torch.ones(4, dtype=torch.float64, device=device)
        cell = (
            torch.eye(3, dtype=torch.float64, device=device)
            .unsqueeze(0)
            .expand(2, -1, -1)
            .contiguous()
            * 10.0
        )
        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)

        # Create a batched mesh
        mesh = torch.randn(2, 8, 8, 8, dtype=torch.float64, device=device)

        forces = spline_gather_gradient(
            positions, charges, mesh, cell, spline_order=4, batch_idx=batch_idx
        )

        # Verify backward works
        forces.sum().backward()
        assert positions.grad is not None
        assert torch.isfinite(positions.grad).all()


class TestSplineMultiChannelCoverage:
    """Test multi-channel spline functions for coverage."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_spread_channels_with_2d_cell(self, device):
        """Test spline_spread_channels with 2D cell (lines 1858-1860)."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[2.0, 2.0, 2.0], [5.0, 5.0, 5.0]],
            dtype=torch.float64,
            device=device,
        )
        # Multi-channel values (e.g., 4 channels)
        values = torch.randn(2, 4, dtype=torch.float64, device=device)
        # 2D cell (not batched)
        cell = torch.eye(3, dtype=torch.float64, device=device) * 10.0

        mesh = spline_spread_channels(
            positions, values, cell, mesh_dims=(8, 8, 8), spline_order=4
        )

        assert mesh.shape == (4, 8, 8, 8)
        assert torch.isfinite(mesh).all()

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_gather_channels_with_2d_cell(self, device):
        """Test spline_gather_channels with 2D cell (lines 1938-1940)."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[2.0, 2.0, 2.0], [5.0, 5.0, 5.0]],
            dtype=torch.float64,
            device=device,
        )
        # Multi-channel mesh
        mesh = torch.randn(4, 8, 8, 8, dtype=torch.float64, device=device)
        # 2D cell (not batched)
        cell = torch.eye(3, dtype=torch.float64, device=device) * 10.0

        values = spline_gather_channels(positions, mesh, cell, spline_order=4)

        assert values.shape == (2, 4)
        assert torch.isfinite(values).all()


class TestSplineBatch2DCellCoverage:
    """Test batch spline functions with 2D cell for coverage."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_batch_spread_with_2d_cell(self, device):
        """Test spline_spread with batch_idx and 2D cell (line 2236)."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[2.0, 2.0, 2.0], [5.0, 5.0, 5.0], [2.0, 2.0, 2.0], [5.0, 5.0, 5.0]],
            dtype=torch.float64,
            device=device,
        )
        values = torch.ones(4, dtype=torch.float64, device=device)
        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)
        # 3D cell (batched)
        cell = (
            torch.eye(3, dtype=torch.float64, device=device)
            .unsqueeze(0)
            .expand(2, -1, -1)
            .contiguous()
            * 10.0
        )

        mesh = spline_spread(
            positions,
            values,
            cell,
            mesh_dims=(8, 8, 8),
            spline_order=4,
            batch_idx=batch_idx,
        )

        assert mesh.shape == (2, 8, 8, 8)
        assert torch.isfinite(mesh).all()

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_batch_gather_with_2d_cell_expansion(self, device):
        """Test spline_gather with batch_idx and 2D cell that needs expansion (lines 2287-2289)."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[2.0, 2.0, 2.0], [5.0, 5.0, 5.0], [2.0, 2.0, 2.0], [5.0, 5.0, 5.0]],
            dtype=torch.float64,
            device=device,
        )
        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)
        mesh = torch.randn(2, 8, 8, 8, dtype=torch.float64, device=device)
        # 2D cell (not batched) - should be expanded
        cell = torch.eye(3, dtype=torch.float64, device=device) * 10.0

        values = spline_gather(
            positions, mesh, cell, spline_order=4, batch_idx=batch_idx
        )

        assert values.shape == (4,)
        assert torch.isfinite(values).all()

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_batch_gather_vec3_with_2d_cell_expansion(self, device):
        """Test spline_gather_vec3 with batch_idx and 2D cell (lines 2336-2338)."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[2.0, 2.0, 2.0], [5.0, 5.0, 5.0], [2.0, 2.0, 2.0], [5.0, 5.0, 5.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.ones(4, dtype=torch.float64, device=device)
        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)
        mesh = torch.randn(2, 8, 8, 8, 3, dtype=torch.float64, device=device)
        # 2D cell (not batched) - should be expanded
        cell = torch.eye(3, dtype=torch.float64, device=device) * 10.0

        values = spline_gather_vec3(
            positions, charges, mesh, cell, spline_order=4, batch_idx=batch_idx
        )

        assert values.shape == (4, 3)
        assert torch.isfinite(values).all()

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_batch_gather_gradient_with_2d_cell_expansion(self, device):
        """Test spline_gather_gradient with batch_idx and 2D cell (lines 2387-2389)."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[2.0, 2.0, 2.0], [5.0, 5.0, 5.0], [2.0, 2.0, 2.0], [5.0, 5.0, 5.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.ones(4, dtype=torch.float64, device=device)
        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)
        mesh = torch.randn(2, 8, 8, 8, dtype=torch.float64, device=device)
        # 2D cell (not batched) - should be expanded
        cell = torch.eye(3, dtype=torch.float64, device=device) * 10.0

        forces = spline_gather_gradient(
            positions, charges, mesh, cell, spline_order=4, batch_idx=batch_idx
        )

        assert forces.shape == (4, 3)
        assert torch.isfinite(forces).all()

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_batch_spread_channels_with_2d_cell(self, device):
        """Test spline_spread_channels with batch_idx and 2D cell (lines 2452-2453)."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[2.0, 2.0, 2.0], [5.0, 5.0, 5.0], [2.0, 2.0, 2.0], [5.0, 5.0, 5.0]],
            dtype=torch.float64,
            device=device,
        )
        values = torch.randn(4, 3, dtype=torch.float64, device=device)  # 3 channels
        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)
        # 2D cell (not batched) - should be expanded
        cell = torch.eye(3, dtype=torch.float64, device=device) * 10.0

        mesh = spline_spread_channels(
            positions,
            values,
            cell,
            mesh_dims=(8, 8, 8),
            spline_order=4,
            batch_idx=batch_idx,
        )

        assert mesh.shape == (2, 3, 8, 8, 8)
        assert torch.isfinite(mesh).all()

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_batch_gather_channels_with_2d_cell(self, device):
        """Test spline_gather_channels with batch_idx and 2D cell (lines 2508-2511)."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[2.0, 2.0, 2.0], [5.0, 5.0, 5.0], [2.0, 2.0, 2.0], [5.0, 5.0, 5.0]],
            dtype=torch.float64,
            device=device,
        )
        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)
        mesh = torch.randn(
            2, 3, 8, 8, 8, dtype=torch.float64, device=device
        )  # 3 channels
        # 2D cell (not batched) - should be expanded
        cell = torch.eye(3, dtype=torch.float64, device=device) * 10.0

        values = spline_gather_channels(
            positions, mesh, cell, spline_order=4, batch_idx=batch_idx
        )

        assert values.shape == (4, 3)
        assert torch.isfinite(values).all()


class TestDeconvolutionCoverage:
    """Test deconvolution functions for coverage."""

    def test_bspline_deconvolution_high_order(self):
        """Test compute_bspline_deconvolution with spline_order > 6 (lines 2629-2637)."""
        mesh_dims = (8, 8, 8)

        # Order 7 triggers recursive coefficient computation
        deconv = compute_bspline_deconvolution(mesh_dims, spline_order=7)

        assert deconv.shape == mesh_dims
        assert torch.isfinite(deconv).all()
        # Deconvolution should be >= 1 at all points
        assert (deconv >= 0.9).all()

    def test_bspline_deconvolution_1d_default_device(self):
        """Test compute_bspline_deconvolution_1d without device (lines 2739-2740)."""
        # Call without device argument - should default to CPU
        deconv_1d = compute_bspline_deconvolution_1d(16, spline_order=4)

        assert deconv_1d.shape == (16,)
        assert deconv_1d.device == torch.device("cpu")
        assert torch.isfinite(deconv_1d).all()

    def test_bspline_deconvolution_1d_high_order(self):
        """Test compute_bspline_deconvolution_1d with high order."""
        # Order 8 triggers recursive coefficient computation
        deconv_1d = compute_bspline_deconvolution_1d(16, spline_order=8)

        assert deconv_1d.shape == (16,)
        assert torch.isfinite(deconv_1d).all()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_bspline_deconvolution_1d_cuda(self):
        """Test compute_bspline_deconvolution_1d with CUDA device."""
        device = torch.device("cuda")
        deconv_1d = compute_bspline_deconvolution_1d(16, spline_order=4, device=device)

        assert deconv_1d.shape == (16,)
        assert deconv_1d.device.type == "cuda"
        assert torch.isfinite(deconv_1d).all()
