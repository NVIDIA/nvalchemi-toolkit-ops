# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for JAX bindings of cell list neighbor construction methods."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from nvalchemiops.jax.neighbors.unbatched import cell_list

from .conftest import requires_gpu

pytestmark = requires_gpu


class TestCellList:
    """Test cell_list function."""

    def test_single_atom_no_neighbors(self):
        """Test cell_list with single atom."""
        positions = jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32)
        cell = jnp.array([[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]])
        pbc = jnp.array([[True, True, True]])
        cutoff = 1.0

        neighbor_matrix, num_neighbors, shifts = cell_list(positions, cutoff, cell, pbc)

        assert num_neighbors.shape == (1,)
        assert int(num_neighbors[0]) == 0

    def test_cubic_system_with_pbc(self):
        """Test cell_list with cubic system."""
        # Simple cubic: 8 atoms at corners
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [2.0, 2.0, 0.0],
                [0.0, 0.0, 2.0],
                [2.0, 0.0, 2.0],
                [0.0, 2.0, 2.0],
                [2.0, 2.0, 2.0],
            ],
            dtype=jnp.float32,
        )
        cell = jnp.array([[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]])
        pbc = jnp.array([[True, True, True]])
        cutoff = 2.5  # Include nearest neighbors at distance 2.0

        neighbor_matrix, num_neighbors, shifts = cell_list(positions, cutoff, cell, pbc)

        assert neighbor_matrix.shape[0] == 8
        assert num_neighbors.shape == (8,)
        assert shifts.shape[0] == 8
        assert shifts.shape[2] == 3
        # Each atom should have at least some neighbors
        assert jnp.sum(num_neighbors) > 0

    def test_return_neighbor_list_format(self):
        """Test cell_list with return_neighbor_list=True."""
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
            ],
            dtype=jnp.float32,
        )
        cell = jnp.array([[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]])
        pbc = jnp.array([[True, True, True]])
        cutoff = 1.0

        neighbor_list, neighbor_ptr, shifts = cell_list(
            positions, cutoff, cell, pbc, return_neighbor_list=True
        )

        assert neighbor_list.shape[0] == 2  # COO format
        assert neighbor_ptr.shape == (3,)  # 2 atoms + 1
        assert shifts.shape[1] == 3

    def test_no_pbc(self):
        """Test cell_list without PBC."""
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
            ],
            dtype=jnp.float32,
        )
        cell = jnp.array([[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]])
        pbc = jnp.array([[False, False, False]])
        cutoff = 1.0

        neighbor_matrix, num_neighbors, shifts = cell_list(positions, cutoff, cell, pbc)

        # With no PBC, all shifts should be zero
        if int(jnp.sum(num_neighbors)) > 0:
            assert jnp.all(shifts == 0)

    def test_different_dtypes(self):
        """Test cell_list with different dtypes."""
        for dtype in [jnp.float32, jnp.float64]:
            positions = jnp.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=dtype)
            cell = jnp.array(
                [[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]], dtype=dtype
            )
            pbc = jnp.array([[True, True, True]])
            cutoff = 1.0

            neighbor_matrix, num_neighbors, shifts = cell_list(
                positions, cutoff, cell, pbc
            )

            assert neighbor_matrix.dtype == jnp.int32
            assert num_neighbors.dtype == jnp.int32
            assert shifts.dtype == jnp.int32


class TestCellListEdgeCases:
    """Edge case tests for cell_list."""

    def test_large_cutoff(self):
        """Large cutoff should still work correctly."""
        key = jax.random.PRNGKey(789)
        positions = jax.random.uniform(key, shape=(8, 3), dtype=jnp.float32) * 2.0
        cell = jnp.eye(3, dtype=jnp.float32).reshape(1, 3, 3) * 4.0
        pbc = jnp.array([[True, True, True]])

        nm, nn, shifts = cell_list(positions, cutoff=5.0, cell=cell, pbc=pbc)
        # Should find some neighbors
        assert int(jnp.sum(nn)) > 0

    def test_no_pbc_all_shifts_zero(self):
        """With no PBC, all shifts must be zero."""
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.5, 0.5, 0.0],
            ],
            dtype=jnp.float32,
        )
        cell = jnp.eye(3, dtype=jnp.float32).reshape(1, 3, 3) * 10.0
        pbc = jnp.array([[False, False, False]])

        nm, nn, shifts = cell_list(positions, cutoff=1.0, cell=cell, pbc=pbc)
        if int(jnp.sum(nn)) > 0:
            assert jnp.all(shifts == 0), "All shifts should be zero with no PBC"

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_dtype_output_consistency(self, dtype):
        """Output dtypes should always be int32 regardless of input dtype."""
        positions = jnp.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=dtype)
        cell = jnp.eye(3, dtype=dtype).reshape(1, 3, 3) * 10.0
        pbc = jnp.array([[True, True, True]])

        nm, nn, shifts = cell_list(positions, cutoff=1.0, cell=cell, pbc=pbc)
        assert nm.dtype == jnp.int32
        assert nn.dtype == jnp.int32
        assert shifts.dtype == jnp.int32

    def test_return_neighbor_list_format(self):
        """Cell list with return_neighbor_list=True should return COO format."""
        positions = jnp.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=jnp.float32)
        cell = jnp.eye(3, dtype=jnp.float32).reshape(1, 3, 3) * 10.0
        pbc = jnp.array([[True, True, True]])

        nl, ptr, shifts = cell_list(
            positions, cutoff=1.0, cell=cell, pbc=pbc, return_neighbor_list=True
        )
        assert nl.shape[0] == 2
        assert ptr.shape == (3,)
        assert shifts.shape[1] == 3
