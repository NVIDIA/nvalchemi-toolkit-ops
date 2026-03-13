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

"""Tests for JAX neighbor utility functions."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from nvalchemiops.jax.neighbors.neighbor_utils import (
    compute_naive_num_shifts,
    get_neighbor_list_from_neighbor_matrix,
    prepare_batch_idx_ptr,
)
from nvalchemiops.neighbors.neighbor_utils import NeighborOverflowError

from .conftest import requires_gpu

pytestmark = requires_gpu

# ==============================================================================
# Tests: compute_naive_num_shifts
# ==============================================================================


class TestComputeNaiveNumShifts:
    """Test compute_naive_num_shifts function."""

    def test_single_system_no_pbc(self):
        """Test with single system and no periodic boundary conditions."""
        cell = jnp.array([[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]])
        pbc = jnp.array([[False, False, False]])
        cutoff = 2.0

        shift_range, num_shifts, max_shifts = compute_naive_num_shifts(
            cell, cutoff, pbc
        )

        # No PBC should result in no shifts
        assert max_shifts == 1  # Only the zero shift
        assert num_shifts.shape == (1,)
        assert int(num_shifts[0]) == 1

    def test_single_system_full_pbc(self):
        """Test with single system and full periodic boundary conditions."""
        cell = jnp.array([[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]])
        pbc = jnp.array([[True, True, True]])
        cutoff = 2.0

        shift_range, num_shifts, max_shifts = compute_naive_num_shifts(
            cell, cutoff, pbc
        )

        # With PBC and cutoff=2.0, cell=5.0, should have shifts in each direction
        assert max_shifts > 1
        assert num_shifts.shape == (1,)
        assert max_shifts == int(num_shifts[0])

    def test_single_system_mixed_pbc(self):
        """Test with single system and mixed periodic boundary conditions."""
        cell = jnp.array([[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]])
        pbc = jnp.array([[True, True, False]])
        cutoff = 2.0

        shift_range, num_shifts, max_shifts = compute_naive_num_shifts(
            cell, cutoff, pbc
        )

        # Mixed PBC should have shifts in x and y but not z
        assert shift_range.shape == (1, 3)
        assert max_shifts >= 1

    def test_multiple_systems(self):
        """Test with multiple systems."""
        cells = jnp.array(
            [
                [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
                [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
            ]
        )
        pbcs = jnp.array([[True, True, True], [True, True, True]])
        cutoff = 2.0

        shift_range, num_shifts, max_shifts = compute_naive_num_shifts(
            cells, cutoff, pbcs
        )

        # Should have shifts for both systems
        assert num_shifts.shape == (2,)
        assert max_shifts > 0

    def test_large_cutoff(self):
        """Test with large cutoff relative to cell size."""
        cell = jnp.array([[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]])
        pbc = jnp.array([[True, True, True]])
        cutoff = 5.0

        shift_range, num_shifts, max_shifts = compute_naive_num_shifts(
            cell, cutoff, pbc
        )

        # Large cutoff should result in many shifts
        assert max_shifts > 1
        assert shift_range.shape == (1, 3)


# ==============================================================================
# Tests: get_neighbor_list_from_neighbor_matrix
# ==============================================================================


class TestGetNeighborListFromNeighborMatrix:
    """Test get_neighbor_list_from_neighbor_matrix function."""

    def test_empty_matrix(self):
        """Test with empty neighbor matrix."""
        neighbor_matrix = jnp.zeros((0, 10), dtype=jnp.int32)
        num_neighbors = jnp.zeros((0,), dtype=jnp.int32)

        neighbor_list, neighbor_ptr = get_neighbor_list_from_neighbor_matrix(
            neighbor_matrix, num_neighbors
        )

        assert neighbor_list.shape == (2, 0)
        assert neighbor_ptr.shape == (1,)

    def test_single_neighbor(self):
        """Test with single neighbor pair."""
        neighbor_matrix = jnp.array([[1, -1, -1]], dtype=jnp.int32)
        num_neighbors = jnp.array([1], dtype=jnp.int32)

        neighbor_list, neighbor_ptr = get_neighbor_list_from_neighbor_matrix(
            neighbor_matrix, num_neighbors, fill_value=-1
        )

        assert neighbor_list.shape[1] == 1
        assert neighbor_ptr.shape == (2,)
        assert int(neighbor_ptr[0]) == 0
        assert int(neighbor_ptr[1]) == 1

    def test_multiple_atoms_varying_neighbors(self):
        """Test with varying number of neighbors per atom."""
        neighbor_matrix = jnp.array(
            [
                [1, 2, -1],
                [0, -1, -1],
                [1, 0, 2],
            ],
            dtype=jnp.int32,
        )
        num_neighbors = jnp.array([2, 1, 3], dtype=jnp.int32)

        neighbor_list, neighbor_ptr = get_neighbor_list_from_neighbor_matrix(
            neighbor_matrix, num_neighbors, fill_value=-1
        )

        assert neighbor_list.shape[0] == 2
        assert neighbor_list.shape[1] == 6  # Total of 2+1+3 neighbors
        assert neighbor_ptr.shape == (4,)
        assert int(neighbor_ptr[0]) == 0
        assert int(neighbor_ptr[1]) == 2
        assert int(neighbor_ptr[2]) == 3
        assert int(neighbor_ptr[3]) == 6

    def test_with_shifts(self):
        """Test conversion with shift information."""
        neighbor_matrix = jnp.array([[1, 2]], dtype=jnp.int32)
        num_neighbors = jnp.array([2], dtype=jnp.int32)
        shifts = jnp.array([[[0, 0, 0], [1, 0, 0]]], dtype=jnp.int32)

        neighbor_list, neighbor_ptr, shift_list = (
            get_neighbor_list_from_neighbor_matrix(
                neighbor_matrix, num_neighbors, shifts, fill_value=-1
            )
        )

        assert neighbor_list.shape[1] == 2
        assert shift_list.shape == (2, 3)
        assert jnp.allclose(shift_list[0], jnp.array([0, 0, 0]))
        assert jnp.allclose(shift_list[1], jnp.array([1, 0, 0]))

    def test_overflow_error(self):
        """Test that overflow error is raised appropriately."""
        neighbor_matrix = jnp.array([[1, 2]], dtype=jnp.int32)
        num_neighbors = jnp.array([5], dtype=jnp.int32)  # More than matrix width

        with pytest.raises(NeighborOverflowError):
            get_neighbor_list_from_neighbor_matrix(
                neighbor_matrix, num_neighbors, fill_value=-1
            )


# ==============================================================================
# Tests: prepare_batch_idx_ptr
# ==============================================================================


class TestPrepareBatchIdxPtr:
    """Test prepare_batch_idx_ptr function."""

    def test_from_batch_idx(self):
        """Test conversion from batch_idx to batch_ptr."""
        batch_idx = jnp.array([0, 0, 0, 1, 1, 2], dtype=jnp.int32)

        result_idx, result_ptr = prepare_batch_idx_ptr(
            batch_idx=batch_idx, batch_ptr=None, num_atoms=6
        )

        assert result_idx.shape == (6,)
        assert result_ptr.shape == (4,)  # 3 systems + 1
        assert int(result_ptr[0]) == 0
        assert int(result_ptr[1]) == 3
        assert int(result_ptr[2]) == 5
        assert int(result_ptr[3]) == 6

    def test_from_batch_ptr(self):
        """Test conversion from batch_ptr to batch_idx."""
        batch_ptr = jnp.array([0, 3, 5, 6], dtype=jnp.int32)

        result_idx, result_ptr = prepare_batch_idx_ptr(
            batch_idx=None, batch_ptr=batch_ptr, num_atoms=6
        )

        assert result_idx.shape == (6,)
        assert result_ptr.shape == (4,)
        # Check that indices are correct
        expected_idx = jnp.array([0, 0, 0, 1, 1, 2], dtype=jnp.int32)
        assert jnp.allclose(result_idx, expected_idx)

    def test_both_provided(self):
        """Test when both batch_idx and batch_ptr are provided."""
        batch_idx = jnp.array([0, 0, 0, 1, 1, 2], dtype=jnp.int32)
        batch_ptr = jnp.array([0, 3, 5, 6], dtype=jnp.int32)

        result_idx, result_ptr = prepare_batch_idx_ptr(
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            num_atoms=6,
        )

        assert result_idx.shape == (6,)
        assert result_ptr.shape == (4,)

    def test_error_both_none(self):
        """Test that error is raised when both are None."""
        with pytest.raises(ValueError):
            prepare_batch_idx_ptr(batch_idx=None, batch_ptr=None, num_atoms=6)

    def test_single_system(self):
        """Test with single system."""
        batch_idx = jnp.array([0, 0, 0], dtype=jnp.int32)

        result_idx, result_ptr = prepare_batch_idx_ptr(
            batch_idx=batch_idx, batch_ptr=None, num_atoms=3
        )

        assert result_idx.shape == (3,)
        assert result_ptr.shape == (2,)  # 1 system + 1
        assert int(result_ptr[0]) == 0
        assert int(result_ptr[1]) == 3
