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
Test suite for JAX parameter estimation functions.

Tests the automatic parameter estimation for Ewald summation and PME methods.
"""

import math

import jax
import jax.numpy as jnp
import pytest

from nvalchemiops.jax.interactions.electrostatics.parameters import (
    EwaldParameters,
    PMEParameters,
    _count_atoms_per_system,
    estimate_ewald_parameters,
    estimate_pme_mesh_dimensions,
    estimate_pme_parameters,
    mesh_spacing_to_dimensions,
)


class TestCountAtomsPerSystem:
    """Tests for the _count_atoms_per_system helper function."""

    def test_single_system(self):
        """Test atom counting for single system."""
        positions = jax.random.normal(jax.random.PRNGKey(0), (100, 3))
        counts = _count_atoms_per_system(positions, num_systems=1, batch_idx=None)

        assert counts.shape == (1,)
        assert counts[0].item() == 100
        assert counts.dtype == jnp.int32

    def test_batch_uniform(self):
        """Test atom counting for batch with uniform distribution."""
        positions = jax.random.normal(jax.random.PRNGKey(0), (30, 3))
        batch_idx = jnp.array(
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
            dtype=jnp.int32,
        )
        counts = _count_atoms_per_system(positions, num_systems=3, batch_idx=batch_idx)

        assert counts.shape == (3,)
        assert counts[0].item() == 10
        assert counts[1].item() == 10
        assert counts[2].item() == 10
        assert counts.dtype == jnp.int32

    def test_batch_nonuniform(self):
        """Test atom counting for batch with non-uniform distribution."""
        positions = jax.random.normal(jax.random.PRNGKey(0), (15, 3))
        batch_idx = jnp.array(
            [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2],
            dtype=jnp.int32,
        )
        counts = _count_atoms_per_system(positions, num_systems=3, batch_idx=batch_idx)

        assert counts.shape == (3,)
        assert counts[0].item() == 3
        assert counts[1].item() == 5
        assert counts[2].item() == 7


class TestEstimateEwaldParameters:
    """Tests for estimate_ewald_parameters function."""

    def test_single_system_returns_tensors(self):
        """Test that single-system mode returns tensor values."""
        positions = jax.random.normal(jax.random.PRNGKey(0), (100, 3))
        cell = jnp.eye(3)[None, ...] * 20.0

        params = estimate_ewald_parameters(positions, cell, accuracy=1e-6)

        assert isinstance(params, EwaldParameters)
        assert isinstance(params.alpha, jax.Array)
        assert params.alpha.shape == (1,)
        assert isinstance(params.real_space_cutoff, jax.Array)
        assert params.real_space_cutoff.shape == (1,)
        assert isinstance(params.reciprocal_space_cutoff, jax.Array)
        assert params.reciprocal_space_cutoff.shape == (1,)

    def test_batch_returns_tensors(self):
        """Test that batch mode returns tensor values."""
        positions = jax.random.normal(jax.random.PRNGKey(0), (100, 3))
        cells = jnp.stack([jnp.eye(3) * 20.0, jnp.eye(3) * 25.0])
        batch_idx = jnp.array([0] * 50 + [1] * 50, dtype=jnp.int32)

        params = estimate_ewald_parameters(
            positions, cells, batch_idx=batch_idx, accuracy=1e-6
        )

        assert isinstance(params, EwaldParameters)
        assert isinstance(params.alpha, jax.Array)
        assert params.alpha.shape == (2,)
        assert isinstance(params.real_space_cutoff, jax.Array)
        assert params.real_space_cutoff.shape == (2,)

    def test_reasonable_alpha_values(self):
        """Test that alpha values are in reasonable range."""
        positions = jax.random.normal(jax.random.PRNGKey(0), (100, 3))
        cell = jnp.eye(3)[None, ...] * 20.0

        params = estimate_ewald_parameters(positions, cell, accuracy=1e-6)

        # Alpha should typically be in range 0.1-1.0 for typical systems
        alpha_val = float(params.alpha[0])
        assert 0.05 < alpha_val < 2.0, f"alpha={alpha_val} out of expected range"

    def test_larger_cell_smaller_alpha(self):
        """Test that larger cells lead to smaller alpha."""
        positions = jax.random.normal(jax.random.PRNGKey(0), (100, 3))

        cell_small = jnp.eye(3)[None, ...] * 15.0
        cell_large = jnp.eye(3)[None, ...] * 30.0

        params_small = estimate_ewald_parameters(positions, cell_small, accuracy=1e-6)
        params_large = estimate_ewald_parameters(positions, cell_large, accuracy=1e-6)

        assert float(params_large.alpha[0]) < float(params_small.alpha[0])

    def test_more_atoms_larger_alpha(self):
        """Test that more atoms lead to larger alpha (for same volume)."""
        cell = jnp.eye(3)[None, ...] * 20.0

        positions_few = jax.random.normal(jax.random.PRNGKey(0), (50, 3))
        positions_many = jax.random.normal(jax.random.PRNGKey(1), (200, 3))

        params_few = estimate_ewald_parameters(positions_few, cell, accuracy=1e-6)
        params_many = estimate_ewald_parameters(positions_many, cell, accuracy=1e-6)

        assert float(params_many.alpha[0]) > float(params_few.alpha[0])

    def test_higher_accuracy_larger_cutoffs(self):
        """Test that higher accuracy leads to larger cutoffs."""
        positions = jax.random.normal(jax.random.PRNGKey(0), (100, 3))
        cell = jnp.eye(3)[None, ...] * 20.0

        params_low = estimate_ewald_parameters(positions, cell, accuracy=1e-4)
        params_high = estimate_ewald_parameters(positions, cell, accuracy=1e-8)

        assert float(params_high.real_space_cutoff[0]) > float(
            params_low.real_space_cutoff[0]
        )
        assert float(params_high.reciprocal_space_cutoff[0]) > float(
            params_low.reciprocal_space_cutoff[0]
        )

    def test_cutoff_product_independent_of_cell_size(self):
        """Test that r_cutoff * k_cutoff is roughly constant (depends only on accuracy)."""
        positions = jax.random.normal(jax.random.PRNGKey(0), (100, 3))

        cell_small = jnp.eye(3)[None, ...] * 1.0
        cell_large = jnp.eye(3)[None, ...] * 30.0

        params_small = estimate_ewald_parameters(positions, cell_small, accuracy=1e-6)
        params_large = estimate_ewald_parameters(positions, cell_large, accuracy=1e-6)

        product_small = float(params_small.real_space_cutoff[0]) * float(
            params_small.reciprocal_space_cutoff[0]
        )
        product_large = float(params_large.real_space_cutoff[0]) * float(
            params_large.reciprocal_space_cutoff[0]
        )

        # Product should be the same (it's -2*log(accuracy))
        expected = -2.0 * math.log(1e-6)
        assert abs(product_small - expected) < 1e-5
        assert abs(product_large - expected) < 1e-5

    def test_batch_different_cells(self):
        """Test batch mode with systems of different sizes."""
        # Create two systems with different volumes
        n_atoms_per_system = 50
        positions = jax.random.normal(
            jax.random.PRNGKey(0), (n_atoms_per_system * 2, 3)
        )
        cells = jnp.stack(
            [
                jnp.eye(3) * 15.0,  # Smaller cell
                jnp.eye(3) * 25.0,  # Larger cell
            ]
        )
        batch_idx = jnp.array(
            [0] * n_atoms_per_system + [1] * n_atoms_per_system,
            dtype=jnp.int32,
        )

        params = estimate_ewald_parameters(
            positions, cells, batch_idx=batch_idx, accuracy=1e-6
        )

        # Larger cell should have smaller alpha
        assert float(params.alpha[1]) < float(params.alpha[0])

    def test_batch_different_atom_counts(self):
        """Test batch mode with systems having different atom counts."""
        # Create two systems: 30 atoms and 70 atoms
        positions = jax.random.normal(jax.random.PRNGKey(0), (100, 3))
        cells = jnp.stack(
            [
                jnp.eye(3) * 20.0,
                jnp.eye(3) * 20.0,
            ]
        )
        batch_idx = jnp.array([0] * 30 + [1] * 70, dtype=jnp.int32)

        params = estimate_ewald_parameters(
            positions, cells, batch_idx=batch_idx, accuracy=1e-6
        )

        # System with more atoms should have larger alpha
        assert float(params.alpha[1]) > float(params.alpha[0])

    def test_2d_cell_unsqueezed(self):
        """Test that 2D cell input is handled correctly."""
        positions = jax.random.normal(jax.random.PRNGKey(0), (100, 3))
        cell_2d = jnp.eye(3) * 20.0

        params = estimate_ewald_parameters(positions, cell_2d, accuracy=1e-6)

        assert isinstance(params.alpha, jax.Array)
        assert params.alpha.shape == (1,)


class TestEstimatePMEMeshDimensions:
    """Tests for estimate_pme_mesh_dimensions function."""

    def test_returns_tuple(self):
        """Test that function returns a tuple of 3 integers."""
        cell = jnp.eye(3)[None, ...] * 20.0
        alpha = jnp.array([0.3])

        dims = estimate_pme_mesh_dimensions(cell, alpha, accuracy=1e-6)

        assert isinstance(dims, tuple)
        assert len(dims) == 3
        assert all(isinstance(d, int) for d in dims)

    def test_power_of_two_dimensions(self):
        """Test that all dimensions are powers of 2."""
        cell = jnp.eye(3)[None, ...] * 20.0
        alpha = jnp.array([0.3])

        dims = estimate_pme_mesh_dimensions(cell, alpha, accuracy=1e-6)

        for d in dims:
            # Check if power of 2: d & (d - 1) == 0
            assert d > 0 and (d & (d - 1)) == 0, f"{d} is not a power of 2"

    def test_larger_alpha_more_points(self):
        """Test that larger alpha leads to more mesh points."""
        cell = jnp.eye(3)[None, ...] * 20.0

        dims_small_alpha = estimate_pme_mesh_dimensions(
            cell, jnp.array([0.2]), accuracy=1e-6
        )
        dims_large_alpha = estimate_pme_mesh_dimensions(
            cell, jnp.array([0.5]), accuracy=1e-6
        )

        assert all(
            d_large >= d_small
            for d_large, d_small in zip(dims_large_alpha, dims_small_alpha)
        )

    def test_higher_accuracy_more_points(self):
        """Test that higher accuracy leads to more mesh points."""
        cell = jnp.eye(3)[None, ...] * 20.0
        alpha = jnp.array([0.3])

        dims_low = estimate_pme_mesh_dimensions(cell, alpha, accuracy=1e-4)
        dims_high = estimate_pme_mesh_dimensions(cell, alpha, accuracy=1e-8)

        assert all(d_high >= d_low for d_high, d_low in zip(dims_high, dims_low))

    def test_rectangular_cell(self):
        """Test with a rectangular (non-cubic) cell."""
        cell = jnp.diag(jnp.array([10.0, 20.0, 30.0]))[None, ...]
        alpha = jnp.array([0.3])

        dims = estimate_pme_mesh_dimensions(cell, alpha, accuracy=1e-6)

        # Longer dimension should have more points (or equal if rounded to same power of 2)
        assert dims[0] <= dims[1] <= dims[2]

    def test_batch_uses_max(self):
        """Test that batch mode uses max dimensions across systems."""
        cells = jnp.stack([jnp.eye(3) * 20.0, jnp.eye(3) * 30.0])
        alpha = jnp.array([0.3, 0.3])

        dims = estimate_pme_mesh_dimensions(cells, alpha, accuracy=1e-6)

        # Should return tuple of 3 integers (max across batch)
        assert isinstance(dims, tuple)
        assert len(dims) == 3

    def test_batch_max_dimensions_correct(self):
        """Test that batch mode correctly computes max dimensions."""
        # Create two systems with different sizes
        cells = jnp.stack([jnp.eye(3) * 15.0, jnp.eye(3) * 30.0])
        alpha = jnp.array([0.3, 0.3])

        dims_batch = estimate_pme_mesh_dimensions(cells, alpha, accuracy=1e-6)

        # Compare with single-system for the larger cell
        dims_large = estimate_pme_mesh_dimensions(
            cells[1:2], jnp.array([0.3]), accuracy=1e-6
        )

        # Batch dims should be >= single system dims for larger cell
        assert all(
            d_batch >= d_large for d_batch, d_large in zip(dims_batch, dims_large)
        )

    def test_2d_cell_input(self):
        """Test that 2D cell input is handled correctly."""
        cell = jnp.eye(3) * 20.0  # 2D cell
        alpha = jnp.array([0.3])

        dims = estimate_pme_mesh_dimensions(cell, alpha, accuracy=1e-6)

        assert isinstance(dims, tuple)
        assert len(dims) == 3


class TestEstimatePMEParameters:
    """Tests for estimate_pme_parameters function."""

    def test_single_system_returns_correct_types(self):
        """Test that single-system mode returns correct types."""
        positions = jax.random.normal(jax.random.PRNGKey(0), (100, 3))
        cell = jnp.eye(3)[None, ...] * 20.0

        params = estimate_pme_parameters(positions, cell, accuracy=1e-6)

        assert isinstance(params, PMEParameters)
        assert isinstance(params.alpha, jax.Array)
        assert params.alpha.shape == (1,)
        assert isinstance(params.mesh_dimensions, tuple)
        assert len(params.mesh_dimensions) == 3
        assert all(isinstance(d, int) for d in params.mesh_dimensions)
        assert isinstance(params.mesh_spacing, jax.Array)
        assert params.mesh_spacing.shape == (1, 3)
        assert isinstance(params.real_space_cutoff, jax.Array)
        assert params.real_space_cutoff.shape == (1,)

    def test_batch_returns_correct_shapes(self):
        """Test that batch mode returns correct tensor shapes."""
        positions = jax.random.normal(jax.random.PRNGKey(0), (100, 3))
        cells = jnp.stack([jnp.eye(3) * 20.0, jnp.eye(3) * 25.0])
        batch_idx = jnp.array([0] * 50 + [1] * 50, dtype=jnp.int32)

        params = estimate_pme_parameters(
            positions, cells, batch_idx=batch_idx, accuracy=1e-6
        )

        # Alpha and real_space_cutoff should have shape (B,)
        assert isinstance(params.alpha, jax.Array)
        assert params.alpha.shape == (2,)
        assert isinstance(params.real_space_cutoff, jax.Array)
        assert params.real_space_cutoff.shape == (2,)

        # mesh_dimensions should be a tuple of 3 integers (max across batch)
        assert isinstance(params.mesh_dimensions, tuple)
        assert len(params.mesh_dimensions) == 3
        assert all(isinstance(d, int) for d in params.mesh_dimensions)

        # mesh_spacing should have shape (B, 3)
        assert isinstance(params.mesh_spacing, jax.Array)
        assert params.mesh_spacing.shape == (2, 3)

    def test_mesh_dimensions_are_power_of_two(self):
        """Test that mesh dimensions are powers of 2."""
        positions = jax.random.normal(jax.random.PRNGKey(0), (100, 3))
        cell = jnp.eye(3)[None, ...] * 20.0

        params = estimate_pme_parameters(positions, cell, accuracy=1e-6)

        for d in params.mesh_dimensions:
            assert d > 0 and (d & (d - 1)) == 0, f"{d} is not a power of 2"

    def test_consistency_with_ewald_parameters(self):
        """Test that alpha matches estimate_ewald_parameters."""
        positions = jax.random.normal(jax.random.PRNGKey(0), (100, 3))
        cell = jnp.eye(3)[None, ...] * 20.0

        pme_params = estimate_pme_parameters(positions, cell, accuracy=1e-6)
        ewald_params = estimate_ewald_parameters(positions, cell, accuracy=1e-6)

        assert jnp.allclose(pme_params.alpha, ewald_params.alpha)
        assert jnp.allclose(
            pme_params.real_space_cutoff, ewald_params.real_space_cutoff
        )

    def test_mesh_spacing_varies_per_system(self):
        """Test that mesh_spacing varies per system in batch mode."""
        positions = jax.random.normal(jax.random.PRNGKey(0), (100, 3))
        # Create two cells of different sizes
        cells = jnp.stack([jnp.eye(3) * 20.0, jnp.eye(3) * 30.0])
        batch_idx = jnp.array([0] * 50 + [1] * 50, dtype=jnp.int32)

        params = estimate_pme_parameters(
            positions, cells, batch_idx=batch_idx, accuracy=1e-6
        )

        # mesh_spacing should be different for systems with different cell sizes
        # (same mesh_dimensions but different cell lengths)
        assert not jnp.allclose(params.mesh_spacing[0], params.mesh_spacing[1])

        # Larger cell should have larger mesh spacing
        assert jnp.all(params.mesh_spacing[1] > params.mesh_spacing[0])

    def test_mesh_spacing_consistent_with_dimensions(self):
        """Test that mesh_spacing = cell_lengths / mesh_dimensions."""
        positions = jax.random.normal(jax.random.PRNGKey(0), (100, 3))
        cell = jnp.eye(3)[None, ...] * 20.0

        params = estimate_pme_parameters(positions, cell, accuracy=1e-6)

        # Compute expected spacing
        cell_lengths = jnp.linalg.norm(cell, axis=2)  # (1, 3)
        mesh_dims_tensor = jnp.array(params.mesh_dimensions, dtype=cell_lengths.dtype)
        expected_spacing = cell_lengths / mesh_dims_tensor

        assert jnp.allclose(params.mesh_spacing, expected_spacing)


class TestMeshSpacingToDimensions:
    """Tests for mesh_spacing_to_dimensions function."""

    def test_returns_tensor(self):
        """Test that function returns a tensor."""
        cell = jnp.eye(3)[None, ...] * 20.0

        dims = mesh_spacing_to_dimensions(cell, mesh_spacing=0.5)

        assert isinstance(dims, tuple)
        assert len(dims) == 3
        assert all(isinstance(d, int) for d in dims)

    def test_power_of_two_dimensions(self):
        """Test that all dimensions are powers of 2."""
        cell = jnp.eye(3)[None, ...] * 20.0

        dims = mesh_spacing_to_dimensions(cell, mesh_spacing=0.5)

        for d_val in dims:
            assert d_val > 0 and (d_val & (d_val - 1)) == 0, (
                f"{d_val} is not a power of 2"
            )

    def test_smaller_spacing_more_points(self):
        """Test that smaller spacing leads to more mesh points."""
        cell = jnp.eye(3)[None, ...] * 20.0

        dims_large_spacing = mesh_spacing_to_dimensions(cell, mesh_spacing=1.0)
        dims_small_spacing = mesh_spacing_to_dimensions(cell, mesh_spacing=0.25)

        assert all(
            d_small >= d_large
            for d_small, d_large in zip(dims_small_spacing, dims_large_spacing)
        )

    def test_rectangular_cell(self):
        """Test with rectangular cell."""
        cell = jnp.diag(jnp.array([10.0, 20.0, 30.0]))[None, ...]

        dims = mesh_spacing_to_dimensions(cell, mesh_spacing=0.5)

        # Longer dimension should have more points
        assert dims[0] <= dims[1] <= dims[2]

    def test_tensor_spacing_1d(self):
        """Test with 1D tensor mesh spacing (per-system, uniform direction)."""
        cells = jnp.stack([jnp.eye(3) * 20.0, jnp.eye(3) * 30.0])
        spacing = jnp.array([0.5, 0.5])

        dims = mesh_spacing_to_dimensions(cells, mesh_spacing=spacing)

        assert len(dims) == 3
        assert isinstance(dims, tuple)

    def test_tensor_spacing_2d(self):
        """Test with 2D tensor mesh spacing (per-system, per-direction)."""
        cells = jnp.stack([jnp.eye(3) * 20.0, jnp.eye(3) * 30.0])
        # Different spacing for each direction
        spacing = jnp.array([[0.5, 0.4, 0.3], [0.6, 0.5, 0.4]])

        dims = mesh_spacing_to_dimensions(cells, mesh_spacing=spacing)

        assert len(dims) == 3
        assert isinstance(dims, tuple)

    def test_invalid_spacing_shape_raises(self):
        """Test that invalid spacing shape raises an error."""
        cells = jnp.stack([jnp.eye(3) * 20.0, jnp.eye(3) * 30.0])
        # Wrong batch size
        spacing = jnp.array([0.5, 0.5, 0.5])

        with pytest.raises(ValueError):
            mesh_spacing_to_dimensions(cells, mesh_spacing=spacing)

    def test_2d_cell_input(self):
        """Test that 2D cell input is handled correctly."""
        cell = jnp.eye(3) * 20.0  # 2D cell

        dims = mesh_spacing_to_dimensions(cell, mesh_spacing=0.5)

        assert len(dims) == 3
        assert isinstance(dims, tuple)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
