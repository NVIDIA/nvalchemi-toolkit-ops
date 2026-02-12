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
Test suite for parameter estimation functions.

Tests the automatic parameter estimation for Ewald summation and PME methods.
"""

import math

import pytest
import torch

from nvalchemiops.torch.interactions.electrostatics.parameters import (
    EwaldParameters,
    PMEParameters,
    _count_atoms_per_system,
    estimate_ewald_parameters,
    estimate_pme_mesh_dimensions,
    estimate_pme_parameters,
    mesh_spacing_to_dimensions,
)
from nvalchemiops.torch.neighbors import cell_list


class TestCountAtomsPerSystem:
    """Tests for the _count_atoms_per_system helper function."""

    @pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda:0")])
    def test_single_system(self, device):
        """Test atom counting for single system."""
        positions = torch.randn(100, 3, device=device)
        counts = _count_atoms_per_system(positions, num_systems=1, batch_idx=None)

        assert counts.shape == (1,)
        assert counts[0].item() == 100
        assert counts.device == device
        assert counts.dtype == torch.int32

    @pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda:0")])
    def test_batch_uniform(self, device):
        """Test atom counting for batch with uniform distribution."""
        positions = torch.randn(30, 3, device=device)
        batch_idx = torch.tensor(
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
            ],
            dtype=torch.int32,
            device=device,
        )
        counts = _count_atoms_per_system(positions, num_systems=3, batch_idx=batch_idx)

        assert counts.shape == (3,)
        assert counts[0].item() == 10
        assert counts[1].item() == 10
        assert counts[2].item() == 10
        assert counts.dtype == torch.int32

    @pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda:0")])
    def test_batch_nonuniform(self, device):
        """Test atom counting for batch with non-uniform distribution."""
        positions = torch.randn(15, 3, device=device)
        batch_idx = torch.tensor(
            [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2],
            dtype=torch.int32,
            device=device,
        )
        counts = _count_atoms_per_system(positions, num_systems=3, batch_idx=batch_idx)

        assert counts.shape == (3,)
        assert counts[0].item() == 3
        assert counts[1].item() == 5
        assert counts[2].item() == 7


class TestEstimateEwaldParameters:
    """Tests for estimate_ewald_parameters function."""

    @pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda:0")])
    def test_single_system_returns_tensors(self, device):
        """Test that single-system mode returns tensor values."""
        positions = torch.randn(100, 3, device=device)
        cell = torch.eye(3, device=device).unsqueeze(0) * 20.0

        params = estimate_ewald_parameters(positions, cell, accuracy=1e-6)

        assert isinstance(params, EwaldParameters)
        assert isinstance(params.alpha, torch.Tensor)
        assert params.alpha.shape == (1,)
        assert isinstance(params.real_space_cutoff, torch.Tensor)
        assert params.real_space_cutoff.shape == (1,)
        assert isinstance(params.reciprocal_space_cutoff, torch.Tensor)
        assert params.reciprocal_space_cutoff.shape == (1,)

    @pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda:0")])
    def test_batch_returns_tensors(self, device):
        """Test that batch mode returns tensor values."""
        positions = torch.randn(100, 3, device=device)
        cells = torch.stack(
            [torch.eye(3, device=device) * 20.0, torch.eye(3, device=device) * 25.0]
        )
        batch_idx = torch.tensor([0] * 50 + [1] * 50, dtype=torch.int32, device=device)

        params = estimate_ewald_parameters(
            positions, cells, batch_idx=batch_idx, accuracy=1e-6
        )

        assert isinstance(params, EwaldParameters)
        assert isinstance(params.alpha, torch.Tensor)
        assert params.alpha.shape == (2,)
        assert isinstance(params.real_space_cutoff, torch.Tensor)
        assert params.real_space_cutoff.shape == (2,)

    @pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda:0")])
    def test_reasonable_alpha_values(self, device):
        """Test that alpha values are in reasonable range."""
        positions = torch.randn(100, 3, device=device)
        cell = torch.eye(3, device=device).unsqueeze(0) * 20.0

        params = estimate_ewald_parameters(positions, cell, accuracy=1e-6)

        # Alpha should typically be in range 0.1-1.0 for typical systems
        alpha_val = params.alpha.item()
        assert 0.05 < alpha_val < 2.0, f"alpha={alpha_val} out of expected range"

    @pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda:0")])
    def test_larger_cell_smaller_alpha(self, device):
        """Test that larger cells lead to smaller alpha."""
        positions = torch.randn(100, 3, device=device)

        cell_small = torch.eye(3, device=device).unsqueeze(0) * 15.0
        cell_large = torch.eye(3, device=device).unsqueeze(0) * 30.0

        params_small = estimate_ewald_parameters(positions, cell_small, accuracy=1e-6)
        params_large = estimate_ewald_parameters(positions, cell_large, accuracy=1e-6)

        assert params_large.alpha.item() < params_small.alpha.item()

    @pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda:0")])
    def test_more_atoms_larger_alpha(self, device):
        """Test that more atoms lead to larger alpha (for same volume)."""
        cell = torch.eye(3, device=device).unsqueeze(0) * 20.0

        positions_few = torch.randn(50, 3, device=device)
        positions_many = torch.randn(200, 3, device=device)

        params_few = estimate_ewald_parameters(positions_few, cell, accuracy=1e-6)
        params_many = estimate_ewald_parameters(positions_many, cell, accuracy=1e-6)

        assert params_many.alpha.item() > params_few.alpha.item()

    @pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda:0")])
    def test_higher_accuracy_larger_cutoffs(self, device):
        """Test that higher accuracy leads to larger cutoffs."""
        positions = torch.randn(100, 3, device=device)
        cell = torch.eye(3, device=device).unsqueeze(0) * 20.0

        params_low = estimate_ewald_parameters(positions, cell, accuracy=1e-4)
        params_high = estimate_ewald_parameters(positions, cell, accuracy=1e-8)

        assert (
            params_high.real_space_cutoff.item() > params_low.real_space_cutoff.item()
        )
        assert (
            params_high.reciprocal_space_cutoff.item()
            > params_low.reciprocal_space_cutoff.item()
        )

    @pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda:0")])
    def test_cutoff_product_independent_of_cell_size(self, device):
        """Test that r_cutoff * k_cutoff is roughly constant (depends only on accuracy)."""
        positions = torch.randn(100, 3, device=device)

        cell_small = torch.eye(3, device=device).unsqueeze(0) * 1.0
        cell_large = torch.eye(3, device=device).unsqueeze(0) * 30.0

        params_small = estimate_ewald_parameters(positions, cell_small, accuracy=1e-6)
        params_large = estimate_ewald_parameters(positions, cell_large, accuracy=1e-6)

        product_small = (
            params_small.real_space_cutoff.item()
            * params_small.reciprocal_space_cutoff.item()
        )
        product_large = (
            params_large.real_space_cutoff.item()
            * params_large.reciprocal_space_cutoff.item()
        )

        # Product should be the same (it's -2*log(accuracy))
        expected = -2.0 * math.log(1e-6)
        assert abs(product_small - expected) < 1e-5
        assert abs(product_large - expected) < 1e-5

    @pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda:0")])
    def test_batch_different_cells(self, device):
        """Test batch mode with systems of different sizes."""
        # Create two systems with different volumes
        n_atoms_per_system = 50
        positions = torch.randn(n_atoms_per_system * 2, 3, device=device)
        cells = torch.stack(
            [
                torch.eye(3, device=device) * 15.0,  # Smaller cell
                torch.eye(3, device=device) * 25.0,  # Larger cell
            ]
        )
        batch_idx = torch.tensor(
            [0] * n_atoms_per_system + [1] * n_atoms_per_system,
            dtype=torch.int32,
            device=device,
        )

        params = estimate_ewald_parameters(
            positions, cells, batch_idx=batch_idx, accuracy=1e-6
        )

        # Larger cell should have smaller alpha
        assert params.alpha[1].item() < params.alpha[0].item()

    @pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda:0")])
    def test_batch_different_atom_counts(self, device):
        """Test batch mode with systems having different atom counts."""
        # Create two systems: 30 atoms and 70 atoms
        positions = torch.randn(100, 3, device=device)
        cells = torch.stack(
            [
                torch.eye(3, device=device) * 20.0,
                torch.eye(3, device=device) * 20.0,
            ]
        )
        batch_idx = torch.tensor([0] * 30 + [1] * 70, dtype=torch.int32, device=device)

        params = estimate_ewald_parameters(
            positions, cells, batch_idx=batch_idx, accuracy=1e-6
        )

        # System with more atoms should have larger alpha
        assert params.alpha[1].item() > params.alpha[0].item()

    @pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda:0")])
    def test_2d_cell_unsqueezed(self, device):
        """Test that 2D cell input is handled correctly."""
        positions = torch.randn(100, 3, device=device)
        cell_2d = torch.eye(3, device=device) * 20.0

        params = estimate_ewald_parameters(positions, cell_2d, accuracy=1e-6)

        assert isinstance(params.alpha, torch.Tensor)
        assert params.alpha.shape == (1,)


class TestEstimatePMEMeshDimensions:
    """Tests for estimate_pme_mesh_dimensions function."""

    @pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda:0")])
    def test_returns_tuple(self, device):
        """Test that function returns a tuple of 3 integers."""
        cell = torch.eye(3, device=device).unsqueeze(0) * 20.0
        alpha = torch.tensor([0.3], device=device)

        dims = estimate_pme_mesh_dimensions(cell, alpha, accuracy=1e-6)

        assert isinstance(dims, tuple)
        assert len(dims) == 3
        assert all(isinstance(d, int) for d in dims)

    @pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda:0")])
    def test_power_of_two_dimensions(self, device):
        """Test that all dimensions are powers of 2."""
        cell = torch.eye(3, device=device).unsqueeze(0) * 20.0
        alpha = torch.tensor([0.3], device=device)

        dims = estimate_pme_mesh_dimensions(cell, alpha, accuracy=1e-6)

        for d in dims:
            # Check if power of 2: d & (d - 1) == 0
            assert d > 0 and (d & (d - 1)) == 0, f"{d} is not a power of 2"

    @pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda:0")])
    def test_larger_alpha_more_points(self, device):
        """Test that larger alpha leads to more mesh points."""
        cell = torch.eye(3, device=device).unsqueeze(0) * 20.0

        dims_small_alpha = estimate_pme_mesh_dimensions(
            cell, torch.tensor([0.2], device=device), accuracy=1e-6
        )
        dims_large_alpha = estimate_pme_mesh_dimensions(
            cell, torch.tensor([0.5], device=device), accuracy=1e-6
        )

        assert all(
            d_large >= d_small
            for d_large, d_small in zip(dims_large_alpha, dims_small_alpha)
        )

    @pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda:0")])
    def test_higher_accuracy_more_points(self, device):
        """Test that higher accuracy leads to more mesh points."""
        cell = torch.eye(3, device=device).unsqueeze(0) * 20.0
        alpha = torch.tensor([0.3], device=device)

        dims_low = estimate_pme_mesh_dimensions(cell, alpha, accuracy=1e-4)
        dims_high = estimate_pme_mesh_dimensions(cell, alpha, accuracy=1e-8)

        assert all(d_high >= d_low for d_high, d_low in zip(dims_high, dims_low))

    @pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda:0")])
    def test_rectangular_cell(self, device):
        """Test with a rectangular (non-cubic) cell."""
        cell = torch.diag(torch.tensor([10.0, 20.0, 30.0], device=device)).unsqueeze(0)
        alpha = torch.tensor([0.3], device=device)

        dims = estimate_pme_mesh_dimensions(cell, alpha, accuracy=1e-6)

        # Longer dimension should have more points (or equal if rounded to same power of 2)
        assert dims[0] <= dims[1] <= dims[2]

    @pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda:0")])
    def test_batch_uses_max(self, device):
        """Test that batch mode uses max dimensions across systems."""
        cells = torch.stack(
            [torch.eye(3, device=device) * 20.0, torch.eye(3, device=device) * 30.0]
        )
        alpha = torch.tensor([0.3, 0.3], device=device)

        dims = estimate_pme_mesh_dimensions(cells, alpha, accuracy=1e-6)

        # Should return tuple of 3 integers (max across batch)
        assert isinstance(dims, tuple)
        assert len(dims) == 3

    @pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda:0")])
    def test_batch_max_dimensions_correct(self, device):
        """Test that batch mode correctly computes max dimensions."""
        # Create two systems with different sizes
        cells = torch.stack(
            [torch.eye(3, device=device) * 15.0, torch.eye(3, device=device) * 30.0]
        )
        alpha = torch.tensor([0.3, 0.3], device=device)

        dims_batch = estimate_pme_mesh_dimensions(cells, alpha, accuracy=1e-6)

        # Compare with single-system for the larger cell
        dims_large = estimate_pme_mesh_dimensions(
            cells[1:2], torch.tensor([0.3], device=device), accuracy=1e-6
        )

        # Batch dims should be >= single system dims for larger cell
        assert all(
            d_batch >= d_large for d_batch, d_large in zip(dims_batch, dims_large)
        )

    @pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda:0")])
    def test_2d_cell_input(self, device):
        """Test that 2D cell input is handled correctly."""
        cell = torch.eye(3, device=device) * 20.0  # 2D cell
        alpha = torch.tensor([0.3], device=device)

        dims = estimate_pme_mesh_dimensions(cell, alpha, accuracy=1e-6)

        assert isinstance(dims, tuple)
        assert len(dims) == 3


class TestEstimatePMEParameters:
    """Tests for estimate_pme_parameters function."""

    @pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda:0")])
    def test_single_system_returns_correct_types(self, device):
        """Test that single-system mode returns correct types."""
        positions = torch.randn(100, 3, device=device)
        cell = torch.eye(3, device=device).unsqueeze(0) * 20.0

        params = estimate_pme_parameters(positions, cell, accuracy=1e-6)

        assert isinstance(params, PMEParameters)
        assert isinstance(params.alpha, torch.Tensor)
        assert params.alpha.shape == (1,)
        assert isinstance(params.mesh_dimensions, tuple)
        assert len(params.mesh_dimensions) == 3
        assert all(isinstance(d, int) for d in params.mesh_dimensions)
        assert isinstance(params.mesh_spacing, torch.Tensor)
        assert params.mesh_spacing.shape == (1, 3)
        assert isinstance(params.real_space_cutoff, torch.Tensor)
        assert params.real_space_cutoff.shape == (1,)

    @pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda:0")])
    def test_batch_returns_correct_shapes(self, device):
        """Test that batch mode returns correct tensor shapes."""
        positions = torch.randn(100, 3, device=device)
        cells = torch.stack(
            [torch.eye(3, device=device) * 20.0, torch.eye(3, device=device) * 25.0]
        )
        batch_idx = torch.tensor([0] * 50 + [1] * 50, dtype=torch.int32, device=device)

        params = estimate_pme_parameters(
            positions, cells, batch_idx=batch_idx, accuracy=1e-6
        )

        # Alpha and real_space_cutoff should have shape (B,)
        assert isinstance(params.alpha, torch.Tensor)
        assert params.alpha.shape == (2,)
        assert isinstance(params.real_space_cutoff, torch.Tensor)
        assert params.real_space_cutoff.shape == (2,)

        # mesh_dimensions should be a tuple of 3 integers (max across batch)
        assert isinstance(params.mesh_dimensions, tuple)
        assert len(params.mesh_dimensions) == 3
        assert all(isinstance(d, int) for d in params.mesh_dimensions)

        # mesh_spacing should have shape (B, 3)
        assert isinstance(params.mesh_spacing, torch.Tensor)
        assert params.mesh_spacing.shape == (2, 3)

    @pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda:0")])
    def test_mesh_dimensions_are_power_of_two(self, device):
        """Test that mesh dimensions are powers of 2."""
        positions = torch.randn(100, 3, device=device)
        cell = torch.eye(3, device=device).unsqueeze(0) * 20.0

        params = estimate_pme_parameters(positions, cell, accuracy=1e-6)

        for d in params.mesh_dimensions:
            assert d > 0 and (d & (d - 1)) == 0, f"{d} is not a power of 2"

    @pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda:0")])
    def test_consistency_with_ewald_parameters(self, device):
        """Test that alpha matches estimate_ewald_parameters."""
        positions = torch.randn(100, 3, device=device)
        cell = torch.eye(3, device=device).unsqueeze(0) * 20.0

        pme_params = estimate_pme_parameters(positions, cell, accuracy=1e-6)
        ewald_params = estimate_ewald_parameters(positions, cell, accuracy=1e-6)

        assert torch.allclose(pme_params.alpha, ewald_params.alpha)
        assert torch.allclose(
            pme_params.real_space_cutoff, ewald_params.real_space_cutoff
        )

    @pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda:0")])
    def test_mesh_spacing_varies_per_system(self, device):
        """Test that mesh_spacing varies per system in batch mode."""
        positions = torch.randn(100, 3, device=device)
        # Create two cells of different sizes
        cells = torch.stack(
            [torch.eye(3, device=device) * 20.0, torch.eye(3, device=device) * 30.0]
        )
        batch_idx = torch.tensor([0] * 50 + [1] * 50, dtype=torch.int32, device=device)

        params = estimate_pme_parameters(
            positions, cells, batch_idx=batch_idx, accuracy=1e-6
        )

        # mesh_spacing should be different for systems with different cell sizes
        # (same mesh_dimensions but different cell lengths)
        assert not torch.allclose(params.mesh_spacing[0], params.mesh_spacing[1])

        # Larger cell should have larger mesh spacing
        assert torch.all(params.mesh_spacing[1] > params.mesh_spacing[0])

    @pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda:0")])
    def test_mesh_spacing_consistent_with_dimensions(self, device):
        """Test that mesh_spacing = cell_lengths / mesh_dimensions."""
        positions = torch.randn(100, 3, device=device)
        cell = torch.eye(3, device=device).unsqueeze(0) * 20.0

        params = estimate_pme_parameters(positions, cell, accuracy=1e-6)

        # Compute expected spacing
        cell_lengths = torch.norm(cell, dim=2)  # (1, 3)
        mesh_dims_tensor = torch.tensor(
            params.mesh_dimensions, dtype=cell_lengths.dtype, device=device
        )
        expected_spacing = cell_lengths / mesh_dims_tensor

        assert torch.allclose(params.mesh_spacing, expected_spacing)


class TestMeshSpacingToDimensions:
    """Tests for mesh_spacing_to_dimensions function."""

    @pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda:0")])
    def test_returns_tensor(self, device):
        """Test that function returns a tensor."""
        cell = torch.eye(3, device=device).unsqueeze(0) * 20.0

        dims = mesh_spacing_to_dimensions(cell, mesh_spacing=0.5)

        assert isinstance(dims, tuple)
        assert len(dims) == 3
        assert all(isinstance(d, int) for d in dims)

    @pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda:0")])
    def test_power_of_two_dimensions(self, device):
        """Test that all dimensions are powers of 2."""
        cell = torch.eye(3, device=device).unsqueeze(0) * 20.0

        dims = mesh_spacing_to_dimensions(cell, mesh_spacing=0.5)

        for d_val in dims:
            assert d_val > 0 and (d_val & (d_val - 1)) == 0, (
                f"{d_val} is not a power of 2"
            )

    @pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda:0")])
    def test_smaller_spacing_more_points(self, device):
        """Test that smaller spacing leads to more mesh points."""
        cell = torch.eye(3, device=device).unsqueeze(0) * 20.0

        dims_large_spacing = mesh_spacing_to_dimensions(cell, mesh_spacing=1.0)
        dims_small_spacing = mesh_spacing_to_dimensions(cell, mesh_spacing=0.25)

        assert all(
            d_small >= d_large
            for d_small, d_large in zip(dims_small_spacing, dims_large_spacing)
        )

    @pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda:0")])
    def test_rectangular_cell(self, device):
        """Test with rectangular cell."""
        cell = torch.diag(torch.tensor([10.0, 20.0, 30.0], device=device)).unsqueeze(0)

        dims = mesh_spacing_to_dimensions(cell, mesh_spacing=0.5)

        # Longer dimension should have more points
        assert dims[0] <= dims[1] <= dims[2]

    @pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda:0")])
    def test_tensor_spacing_1d(self, device):
        """Test with 1D tensor mesh spacing (per-system, uniform direction)."""
        cells = torch.stack(
            [torch.eye(3, device=device) * 20.0, torch.eye(3, device=device) * 30.0]
        )
        spacing = torch.tensor([0.5, 0.5], device=device)

        dims = mesh_spacing_to_dimensions(cells, mesh_spacing=spacing)

        assert len(dims) == 3
        assert isinstance(dims, tuple)

    @pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda:0")])
    def test_tensor_spacing_2d(self, device):
        """Test with 2D tensor mesh spacing (per-system, per-direction)."""
        cells = torch.stack(
            [torch.eye(3, device=device) * 20.0, torch.eye(3, device=device) * 30.0]
        )
        # Different spacing for each direction
        spacing = torch.tensor([[0.5, 0.4, 0.3], [0.6, 0.5, 0.4]], device=device)

        dims = mesh_spacing_to_dimensions(cells, mesh_spacing=spacing)

        assert len(dims) == 3
        assert isinstance(dims, tuple)

    @pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda:0")])
    def test_invalid_spacing_shape_raises(self, device):
        """Test that invalid spacing shape raises an error."""
        cells = torch.stack(
            [torch.eye(3, device=device) * 20.0, torch.eye(3, device=device) * 30.0]
        )
        # Wrong batch size
        spacing = torch.tensor([0.5, 0.5, 0.5], device=device)

        with pytest.raises(ValueError):
            mesh_spacing_to_dimensions(cells, mesh_spacing=spacing)

    @pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda:0")])
    def test_2d_cell_input(self, device):
        """Test that 2D cell input is handled correctly."""
        cell = torch.eye(3, device=device) * 20.0  # 2D cell

        dims = mesh_spacing_to_dimensions(cell, mesh_spacing=0.5)

        assert len(dims) == 3
        assert isinstance(dims, tuple)


class TestIntegration:
    """Integration tests for parameter estimation."""

    @pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda:0")])
    def test_ewald_parameters_work_with_ewald_summation(self, device):
        """Test that estimated parameters can be used with ewald_summation."""
        pytest.importorskip("nvalchemiops.torch.interactions.electrostatics.ewald")

        from nvalchemiops.torch.interactions.electrostatics import ewald_summation

        # Create a simple system
        positions = torch.randn(20, 3, device=device, dtype=torch.float64) * 5.0 + 5.0
        charges = torch.randn(20, device=device, dtype=torch.float64)
        charges = charges - charges.mean()  # Neutralize
        cell = torch.eye(3, device=device, dtype=torch.float64).unsqueeze(0) * 15.0

        # Estimate parameters
        params = estimate_ewald_parameters(positions, cell, accuracy=1e-4)

        neighbor_list, neighbor_ptr, neighbor_shifts = cell_list(
            positions,
            5.0,
            cell=cell,
            pbc=torch.tensor([True, True, True], dtype=torch.bool, device=device),
            return_neighbor_list=True,
        )
        # This should run without error
        energies = ewald_summation(
            positions,
            charges,
            cell,
            alpha=params.alpha,
            k_cutoff=params.reciprocal_space_cutoff,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            compute_forces=False,
        )
        assert energies.shape == (20,)

    @pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda:0")])
    def test_pme_parameters_work_with_particle_mesh_ewald(self, device):
        """Test that estimated parameters can be used with particle_mesh_ewald."""
        pytest.importorskip("nvalchemiops.torch.interactions.electrostatics.pme")

        from nvalchemiops.torch.interactions.electrostatics import particle_mesh_ewald

        # Create a simple system
        positions = (
            torch.randn(20, 3, device=device, dtype=torch.float64) * 5.0 + 5.0
        ).requires_grad_(False)
        charges = torch.randn(20, device=device, dtype=torch.float64)
        charges = charges - charges.mean()  # Neutralize
        cell = torch.eye(3, device=device, dtype=torch.float64).unsqueeze(0) * 15.0

        # Estimate parameters
        params = estimate_pme_parameters(positions, cell, accuracy=1e-4)

        # Create a simple neighbor list
        neighbor_list, neighbor_ptr, neighbor_shifts = cell_list(
            positions,
            5.0,
            cell=cell,
            pbc=torch.tensor([True, True, True], dtype=torch.bool, device=device),
            return_neighbor_list=True,
        )

        # This should run without error
        energies = particle_mesh_ewald(
            positions,
            charges,
            cell,
            alpha=params.alpha,
            mesh_dimensions=tuple(params.mesh_dimensions),
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            compute_forces=False,
        )

        assert energies.shape == (20,)
