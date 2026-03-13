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

"""Tests for JAX bindings of naive dual cutoff neighbor list methods."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from nvalchemiops.jax.neighbors import naive_neighbor_list_dual_cutoff

from .conftest import create_simple_cubic_system_jax, requires_gpu

pytestmark = requires_gpu


class TestNaiveDualCutoffCorrectness:
    """Test correctness of naive dual cutoff neighbor list."""

    def test_matrix_format_no_pbc(self):
        """Test dual cutoff neighbor list in matrix format without PBC."""
        positions, _, _ = create_simple_cubic_system_jax(
            num_atoms=8, cell_size=2.0, dtype=jnp.float32
        )

        cutoff1 = 1.0
        cutoff2 = 1.5
        max_neighbors1 = 15
        max_neighbors2 = 25

        neighbor_matrix1, num_neighbors1, neighbor_matrix2, num_neighbors2 = (
            naive_neighbor_list_dual_cutoff(
                positions,
                cutoff1,
                cutoff2,
                max_neighbors1=max_neighbors1,
                max_neighbors2=max_neighbors2,
            )
        )

        # Verify output shapes and types
        assert neighbor_matrix1.shape == (8, max_neighbors1)
        assert neighbor_matrix2.shape == (8, max_neighbors2)
        assert num_neighbors1.shape == (8,)
        assert num_neighbors2.shape == (8,)
        assert neighbor_matrix1.dtype == jnp.int32
        assert neighbor_matrix2.dtype == jnp.int32
        assert num_neighbors1.dtype == jnp.int32
        assert num_neighbors2.dtype == jnp.int32

        # Verify neighbor counts are reasonable
        assert jnp.all(num_neighbors1 >= 0)
        assert jnp.all(num_neighbors2 >= 0)
        assert jnp.all(num_neighbors1 <= max_neighbors1)
        assert jnp.all(num_neighbors2 <= max_neighbors2)
        # Larger cutoff should find at least as many neighbors
        assert jnp.all(num_neighbors2 >= num_neighbors1)

    def test_matrix_format_with_pbc(self):
        """Test dual cutoff neighbor list in matrix format with PBC."""
        positions, cell, pbc = create_simple_cubic_system_jax(
            num_atoms=8, cell_size=2.0, dtype=jnp.float32
        )

        cutoff1 = 1.0
        cutoff2 = 1.5
        max_neighbors1 = 15
        max_neighbors2 = 25

        (
            neighbor_matrix1,
            num_neighbors1,
            neighbor_matrix_shifts1,
            neighbor_matrix2,
            num_neighbors2,
            neighbor_matrix_shifts2,
        ) = naive_neighbor_list_dual_cutoff(
            positions,
            cutoff1,
            cutoff2,
            pbc=pbc,
            cell=cell,
            max_neighbors1=max_neighbors1,
            max_neighbors2=max_neighbors2,
        )

        # Verify output shapes and types
        assert neighbor_matrix1.shape == (8, max_neighbors1)
        assert neighbor_matrix2.shape == (8, max_neighbors2)
        assert neighbor_matrix_shifts1.shape == (8, max_neighbors1, 3)
        assert neighbor_matrix_shifts2.shape == (8, max_neighbors2, 3)
        assert num_neighbors1.shape == (8,)
        assert num_neighbors2.shape == (8,)

        # Verify dtypes
        assert neighbor_matrix1.dtype == jnp.int32
        assert neighbor_matrix2.dtype == jnp.int32
        assert num_neighbors1.dtype == jnp.int32
        assert num_neighbors2.dtype == jnp.int32
        assert neighbor_matrix_shifts1.dtype == jnp.int32
        assert neighbor_matrix_shifts2.dtype == jnp.int32

        # Verify neighbor counts
        assert jnp.all(num_neighbors1 >= 0)
        assert jnp.all(num_neighbors2 >= 0)
        assert jnp.all(num_neighbors2 >= num_neighbors1)


class TestNaiveDualCutoffEdgeCases:
    """Test edge cases for naive dual cutoff neighbor list."""

    def test_single_atom(self):
        """Test with single atom (should have no neighbors)."""
        positions = jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32)

        cutoff1 = 1.0
        cutoff2 = 1.5

        neighbor_matrix1, num_neighbors1, neighbor_matrix2, num_neighbors2 = (
            naive_neighbor_list_dual_cutoff(
                positions,
                cutoff1,
                cutoff2,
                max_neighbors1=10,
                max_neighbors2=10,
            )
        )

        assert int(num_neighbors1[0]) == 0
        assert int(num_neighbors2[0]) == 0

    def test_identical_cutoffs(self):
        """Test with identical cutoffs (both lists should match)."""
        positions, _, _ = create_simple_cubic_system_jax(
            num_atoms=8, cell_size=2.0, dtype=jnp.float32
        )

        cutoff = 1.2
        max_neighbors = 20

        neighbor_matrix1, num_neighbors1, neighbor_matrix2, num_neighbors2 = (
            naive_neighbor_list_dual_cutoff(
                positions,
                cutoff,
                cutoff,
                max_neighbors1=max_neighbors,
                max_neighbors2=max_neighbors,
            )
        )

        # Neighbor counts should be identical
        assert jnp.all(num_neighbors1 == num_neighbors2)


# ==============================================================================
# Tests: return_neighbor_list=True (COO format)
# ==============================================================================


class TestDualCutoffListFormat:
    """Test dual cutoff neighbor list in COO list format."""

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_unbatched_list_format_no_pbc(self, dtype):
        """Test unbatched dual cutoff in list format without PBC."""
        positions, _, _ = create_simple_cubic_system_jax(
            num_atoms=8, cell_size=2.0, dtype=dtype
        )
        cutoff1 = 1.0
        cutoff2 = 1.5

        neighbor_list1, neighbor_ptr1, neighbor_list2, neighbor_ptr2 = (
            naive_neighbor_list_dual_cutoff(
                positions,
                cutoff1,
                cutoff2,
                max_neighbors1=15,
                max_neighbors2=25,
                return_neighbor_list=True,
            )
        )

        # Verify COO format shapes
        assert neighbor_list1.shape[0] == 2
        assert neighbor_list2.shape[0] == 2
        assert neighbor_ptr1.shape == (9,)
        assert neighbor_ptr2.shape == (9,)
        # Larger cutoff should find at least as many pairs
        assert neighbor_list2.shape[1] >= neighbor_list1.shape[1]

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_unbatched_list_format_with_pbc(self, dtype):
        """Test unbatched dual cutoff in list format with PBC."""
        positions, cell, pbc = create_simple_cubic_system_jax(
            num_atoms=8, cell_size=2.0, dtype=dtype
        )
        cutoff1 = 1.0
        cutoff2 = 1.5

        (
            neighbor_list1,
            neighbor_ptr1,
            neighbor_shifts1,
            neighbor_list2,
            neighbor_ptr2,
            neighbor_shifts2,
        ) = naive_neighbor_list_dual_cutoff(
            positions,
            cutoff1,
            cutoff2,
            cell=cell,
            pbc=pbc,
            max_neighbors1=15,
            max_neighbors2=25,
            return_neighbor_list=True,
        )

        # Verify COO format shapes
        assert neighbor_list1.shape[0] == 2
        assert neighbor_list2.shape[0] == 2
        assert neighbor_ptr1.shape == (9,)
        assert neighbor_ptr2.shape == (9,)
        assert neighbor_shifts1.shape[0] == neighbor_list1.shape[1]
        assert neighbor_shifts2.shape[0] == neighbor_list2.shape[1]
        assert neighbor_shifts1.shape[1] == 3
        assert neighbor_shifts2.shape[1] == 3


class TestNaiveDualCutoffJIT:
    """Smoke tests for naive_neighbor_list_dual_cutoff with jax.jit."""

    def test_jit_no_pbc(self):
        """Test dual cutoff without PBC works with jax.jit."""
        positions, _, _ = create_simple_cubic_system_jax(
            num_atoms=8, cell_size=2.0, dtype=jnp.float32
        )

        @jax.jit
        def jitted_dual(positions):
            return naive_neighbor_list_dual_cutoff(
                positions,
                cutoff1=1.0,
                cutoff2=1.5,
                max_neighbors1=15,
                max_neighbors2=25,
            )

        nm1, nn1, nm2, nn2 = jitted_dual(positions)

        assert nm1.shape == (8, 15)
        assert nm2.shape == (8, 25)
        assert nn1.shape == (8,)
        assert nn2.shape == (8,)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
class TestNaiveDualCutoffSelectiveRebuildFlags:
    """Test selective rebuild (rebuild_flags) for naive_neighbor_list_dual_cutoff JAX."""

    def test_no_rebuild_preserves_data(self, dtype):
        """Flag=False: neighbor data should remain unchanged."""
        positions, _, _ = create_simple_cubic_system_jax(
            num_atoms=8, cell_size=2.0, dtype=dtype
        )
        cutoff1 = 1.0
        cutoff2 = 1.5
        max_neighbors1 = 15
        max_neighbors2 = 25

        # Initial full build
        nm1, nn1, nm2, nn2 = naive_neighbor_list_dual_cutoff(
            positions,
            cutoff1,
            cutoff2,
            max_neighbors1=max_neighbors1,
            max_neighbors2=max_neighbors2,
        )

        saved_nn1 = jnp.array(nn1)
        saved_nn2 = jnp.array(nn2)

        # Selective rebuild with flag=False
        rebuild_flags = jnp.zeros(1, dtype=jnp.bool_)
        nm1b, nn1b, nm2b, nn2b = naive_neighbor_list_dual_cutoff(
            positions,
            cutoff1,
            cutoff2,
            max_neighbors1=max_neighbors1,
            max_neighbors2=max_neighbors2,
            neighbor_matrix1=nm1,
            neighbor_matrix2=nm2,
            num_neighbors1=nn1,
            num_neighbors2=nn2,
            rebuild_flags=rebuild_flags,
        )

        assert jnp.all(nn1b == saved_nn1), "nn1 must be unchanged when flag=False"
        assert jnp.all(nn2b == saved_nn2), "nn2 must be unchanged when flag=False"

    def test_rebuild_updates_data(self, dtype):
        """Flag=True: result should match a fresh full rebuild."""
        positions, _, _ = create_simple_cubic_system_jax(
            num_atoms=8, cell_size=2.0, dtype=dtype
        )
        cutoff1 = 1.0
        cutoff2 = 1.5
        max_neighbors1 = 15
        max_neighbors2 = 25

        # Reference: full build
        _, nn1_ref, _, nn2_ref = naive_neighbor_list_dual_cutoff(
            positions,
            cutoff1,
            cutoff2,
            max_neighbors1=max_neighbors1,
            max_neighbors2=max_neighbors2,
        )

        # Selective rebuild with flag=True
        nm1_stale = jnp.full((8, max_neighbors1), 99, dtype=jnp.int32)
        nm2_stale = jnp.full((8, max_neighbors2), 99, dtype=jnp.int32)
        nn1_stale = jnp.full((8,), 99, dtype=jnp.int32)
        nn2_stale = jnp.full((8,), 99, dtype=jnp.int32)

        rebuild_flags = jnp.ones(1, dtype=jnp.bool_)
        _, nn1b, _, nn2b = naive_neighbor_list_dual_cutoff(
            positions,
            cutoff1,
            cutoff2,
            max_neighbors1=max_neighbors1,
            max_neighbors2=max_neighbors2,
            neighbor_matrix1=nm1_stale,
            neighbor_matrix2=nm2_stale,
            num_neighbors1=nn1_stale,
            num_neighbors2=nn2_stale,
            rebuild_flags=rebuild_flags,
        )

        assert jnp.all(nn1b == nn1_ref), "nn1 should match full rebuild when flag=True"
        assert jnp.all(nn2b == nn2_ref), "nn2 should match full rebuild when flag=True"
