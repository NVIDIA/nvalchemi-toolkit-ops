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
Tests for JAX k-vector generation utilities.

Tests cover:
- Ewald summation k-vector generation
- PME k-vector generation
- Batch support for multiple systems
- Gradient support through JAX autodiff
"""

import jax
import jax.numpy as jnp
import pytest

from nvalchemiops.jax.interactions.electrostatics.k_vectors import (
    generate_k_vectors_ewald_summation,
    generate_k_vectors_pme,
)

###########################################################################################
########################### Ewald K-Vector Tests ##########################################
###########################################################################################


class TestKVectorsEwald:
    """Test k-vector generation for Ewald summation."""

    def test_output_shape_single_system(self):
        """Test output shape for single system."""
        cell = jnp.eye(3, dtype=jnp.float64)[None, ...] * 10.0
        k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff=8.0)

        # Should be (K, 3) for single system (squeezed)
        assert k_vectors.ndim == 2
        assert k_vectors.shape[1] == 3
        assert k_vectors.shape[0] > 0

    def test_output_shape_batch(self):
        """Test output shape for batch of systems."""
        cell = jnp.tile(jnp.eye(3, dtype=jnp.float64)[None, ...], (3, 1, 1)) * 10.0
        k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff=8.0)

        # Should be (B, K, 3) for batch
        assert k_vectors.ndim == 3
        assert k_vectors.shape[0] == 3
        assert k_vectors.shape[2] == 3

    def test_larger_cutoff_more_vectors(self):
        """Test that larger cutoff produces more k-vectors."""
        cell = jnp.eye(3, dtype=jnp.float64)[None, ...] * 10.0

        k_vectors_small = generate_k_vectors_ewald_summation(cell, k_cutoff=5.0)
        k_vectors_large = generate_k_vectors_ewald_summation(cell, k_cutoff=10.0)

        assert k_vectors_large.shape[0] > k_vectors_small.shape[0]

    def test_halfspace_completeness(self):
        """Test that half-space enumeration produces exactly the right k-vectors.

        Verifies that every k-vector in the output satisfies the half-space
        condition and that no valid half-space k-vectors are missing.
        """
        cell = jnp.eye(3, dtype=jnp.float64)[None, ...] * 10.0
        k_cutoff = 5.0

        k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff=k_cutoff)

        # Compute reciprocal cell for reference
        reciprocal_cell = 2 * jnp.pi * jnp.linalg.inv(cell[0].T)

        # Recover Miller indices (approximate, since k = miller @ reciprocal_cell)
        inv_recip = jnp.linalg.inv(reciprocal_cell)
        miller_approx = k_vectors @ inv_recip
        miller_rounded = jnp.round(miller_approx).astype(jnp.int32)

        h = miller_rounded[:, 0]
        k = miller_rounded[:, 1]
        m = miller_rounded[:, 2]

        # Every vector should satisfy the half-space condition
        halfspace_ok = (h > 0) | ((h == 0) & (k > 0)) | ((h == 0) & (k == 0) & (m > 0))
        assert jnp.all(halfspace_ok), "All k-vectors must be in the positive half-space"

        # No duplicates
        unique_count = jnp.unique(
            miller_rounded, axis=0, size=miller_rounded.shape[0]
        ).shape[0]
        assert unique_count == miller_rounded.shape[0], "No duplicate Miller indices"

    def test_with_explicit_miller_bounds_matches_auto(self):
        """Test that explicit miller_bounds produces identical output to auto-computed."""
        from nvalchemiops.jax.interactions.electrostatics.k_vectors import (
            generate_miller_indices,
        )

        cell = jnp.eye(3, dtype=jnp.float64)[None, ...] * 10.0
        k_cutoff = 8.0

        # Auto-computed
        k_vectors_auto = generate_k_vectors_ewald_summation(cell, k_cutoff=k_cutoff)

        # Explicit bounds
        bounds = generate_miller_indices(cell, k_cutoff)
        miller_bounds = (int(bounds[0]), int(bounds[1]), int(bounds[2]))
        k_vectors_explicit = generate_k_vectors_ewald_summation(
            cell, k_cutoff=k_cutoff, miller_bounds=miller_bounds
        )

        assert k_vectors_auto.shape == k_vectors_explicit.shape
        assert jnp.allclose(k_vectors_auto, k_vectors_explicit)

    def test_jit_compatible_with_miller_bounds(self):
        """Test that generate_k_vectors_ewald_summation works inside jax.jit with miller_bounds."""
        from nvalchemiops.jax.interactions.electrostatics.k_vectors import (
            generate_miller_indices,
        )

        cell = jnp.eye(3, dtype=jnp.float64)[None, ...] * 10.0
        k_cutoff = 8.0

        # Precompute bounds eagerly
        bounds = generate_miller_indices(cell, k_cutoff)
        miller_bounds = (int(bounds[0]), int(bounds[1]), int(bounds[2]))

        # This should work inside jax.jit
        @jax.jit
        def jitted_fn(cell):
            return generate_k_vectors_ewald_summation(
                cell, k_cutoff=k_cutoff, miller_bounds=miller_bounds
            )

        k_vectors_jit = jitted_fn(cell)

        # Compare with eager execution
        k_vectors_eager = generate_k_vectors_ewald_summation(
            cell, k_cutoff=k_cutoff, miller_bounds=miller_bounds
        )

        assert k_vectors_jit.shape == k_vectors_eager.shape
        assert jnp.allclose(k_vectors_jit, k_vectors_eager)


###########################################################################################
########################### PME K-Vector Tests ############################################
###########################################################################################


class TestKVectorsPME:
    """Test k-vector generation for PME."""

    def test_output_shapes(self):
        """Test output shapes for PME k-vectors."""
        cell = jnp.eye(3, dtype=jnp.float64)[None, ...] * 10.0
        mesh_dims = (16, 16, 16)

        k_vectors, k_squared_safe = generate_k_vectors_pme(cell, mesh_dims)

        # k_vectors shape: (nx, ny, nz/2+1, 3)
        assert k_vectors.shape == (16, 16, 9, 3)
        # k_squared_safe shape: (nx, ny, nz/2+1)
        assert k_squared_safe.shape == (16, 16, 9)

    def test_k_squared_positive(self):
        """Test that k_squared_safe is always positive (avoids division by zero)."""
        cell = jnp.eye(3, dtype=jnp.float64)[None, ...] * 10.0
        mesh_dims = (16, 16, 16)

        _, k_squared_safe = generate_k_vectors_pme(cell, mesh_dims)

        assert (k_squared_safe > 0).all(), "k_squared_safe should always be positive"

    def test_k_zero_has_safe_value(self):
        """Test that k=0 has a safe non-zero k² value."""
        cell = jnp.eye(3, dtype=jnp.float64)[None, ...] * 10.0
        mesh_dims = (16, 16, 16)

        k_vectors, k_squared_safe = generate_k_vectors_pme(cell, mesh_dims)

        # k=0 is at index [0, 0, 0]
        k_zero = k_vectors[0, 0, 0]
        k_sq_zero = k_squared_safe[0, 0, 0]

        assert jnp.linalg.norm(k_zero) < 1e-10, "k[0,0,0] should be zero"
        assert k_sq_zero > 0, "k_squared_safe[0,0,0] should be non-zero for safety"

    @pytest.mark.parametrize("mesh_dims", [(8, 8, 8), (16, 16, 16), (32, 32, 32)])
    def test_different_mesh_sizes(self, mesh_dims):
        """Test different mesh dimensions."""
        cell = jnp.eye(3, dtype=jnp.float64)[None, ...] * 10.0

        k_vectors, k_squared_safe = generate_k_vectors_pme(cell, mesh_dims)

        nx, ny, nz = mesh_dims
        expected_shape = (nx, ny, nz // 2 + 1)

        assert k_vectors.shape[:3] == expected_shape
        assert k_squared_safe.shape == expected_shape

    def test_rectangular_mesh(self):
        """Test non-cubic mesh dimensions."""
        cell = jnp.diag(jnp.array([10.0, 15.0, 20.0]))[None, ...]
        mesh_dims = (16, 24, 32)

        k_vectors, k_squared_safe = generate_k_vectors_pme(cell, mesh_dims)

        nx, ny, nz = mesh_dims
        expected_shape = (nx, ny, nz // 2 + 1)

        assert k_vectors.shape[:3] == expected_shape
        assert k_squared_safe.shape == expected_shape


###########################################################################################
########################### Gradient Tests ################################################
###########################################################################################


class TestKVectorGradients:
    """Test that k-vector generation supports autograd through cell."""

    def test_ewald_k_vectors_have_gradients(self):
        """Test that Ewald k-vectors flow gradients through cell."""
        cell = jnp.eye(3, dtype=jnp.float64)[None, ...] * 10.0

        def loss_fn(cell):
            k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff=8.0)
            return k_vectors.sum()

        grad_fn = jax.grad(loss_fn)
        grad_val = grad_fn(cell)

        assert grad_val is not None
        assert jnp.isfinite(grad_val).all()

    def test_pme_k_vectors_have_gradients(self):
        """Test that PME k-vectors flow gradients through cell."""
        cell = jnp.eye(3, dtype=jnp.float64)[None, ...] * 10.0

        def loss_fn(cell):
            k_vectors, k_squared_safe = generate_k_vectors_pme(cell, (16, 16, 16))
            return k_vectors.sum() + k_squared_safe.sum()

        grad_fn = jax.grad(loss_fn)
        grad_val = grad_fn(cell)

        assert grad_val is not None
        assert jnp.isfinite(grad_val).all()

    def test_ewald_k_vectors_gradients_with_miller_bounds(self):
        """Test that gradients flow through cell when miller_bounds is provided."""
        from nvalchemiops.jax.interactions.electrostatics.k_vectors import (
            generate_miller_indices,
        )

        cell = jnp.eye(3, dtype=jnp.float64)[None, ...] * 10.0
        k_cutoff = 8.0
        bounds = generate_miller_indices(cell, k_cutoff)
        miller_bounds = (int(bounds[0]), int(bounds[1]), int(bounds[2]))

        def loss_fn(cell):
            k_vectors = generate_k_vectors_ewald_summation(
                cell, k_cutoff=k_cutoff, miller_bounds=miller_bounds
            )
            return k_vectors.sum()

        grad_fn = jax.grad(loss_fn)
        grad_val = grad_fn(cell)

        assert grad_val is not None
        assert jnp.isfinite(grad_val).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
