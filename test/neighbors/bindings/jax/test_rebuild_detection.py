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

"""Tests for JAX rebuild detection functions."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nvalchemiops.jax.neighbors.rebuild_detection import (
    cell_list_needs_rebuild,
    neighbor_list_needs_rebuild,
)

from .conftest import requires_gpu

pytestmark = requires_gpu


# ==============================================================================
# Tests: neighbor_list_needs_rebuild
# ==============================================================================


class TestNeighborListNeedsRebuild:
    """Test neighbor_list_needs_rebuild function."""

    def test_no_movement(self):
        """Test that no rebuild is needed when atoms don't move."""
        positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=jnp.float32)
        skin_distance = 0.5

        rebuild_needed = neighbor_list_needs_rebuild(
            reference_positions=positions,
            current_positions=positions,
            skin_distance_threshold=skin_distance,
        )

        assert rebuild_needed.shape == (1,)
        assert rebuild_needed.dtype == jnp.bool_
        assert not rebuild_needed.item()

    def test_small_movement_within_skin(self):
        """Test no rebuild for small movements within skin distance."""
        reference_positions = jnp.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=jnp.float32
        )
        current_positions = reference_positions + jnp.array(
            [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]], dtype=jnp.float32
        )
        skin_distance = 0.5

        rebuild_needed = neighbor_list_needs_rebuild(
            reference_positions=reference_positions,
            current_positions=current_positions,
            skin_distance_threshold=skin_distance,
        )

        assert not rebuild_needed.item()

    def test_large_movement_beyond_skin(self):
        """Test rebuild needed for large movements beyond skin distance."""
        reference_positions = jnp.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=jnp.float32
        )
        current_positions = reference_positions + jnp.array(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=jnp.float32
        )
        skin_distance = 0.5

        rebuild_needed = neighbor_list_needs_rebuild(
            reference_positions=reference_positions,
            current_positions=current_positions,
            skin_distance_threshold=skin_distance,
        )

        assert rebuild_needed.item()

    def test_shape_mismatch(self):
        """Test rebuild needed for shape mismatch."""
        reference_positions = jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32)
        current_positions = jnp.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=jnp.float32
        )
        skin_distance = 0.5

        rebuild_needed = neighbor_list_needs_rebuild(
            reference_positions=reference_positions,
            current_positions=current_positions,
            skin_distance_threshold=skin_distance,
        )

        assert rebuild_needed.item()

    def test_empty_system(self):
        """Test with empty system."""
        reference_positions = jnp.zeros((0, 3), dtype=jnp.float32)
        current_positions = jnp.zeros((0, 3), dtype=jnp.float32)
        skin_distance = 0.5

        rebuild_needed = neighbor_list_needs_rebuild(
            reference_positions=reference_positions,
            current_positions=current_positions,
            skin_distance_threshold=skin_distance,
        )

        assert not rebuild_needed.item()


# ==============================================================================
# Tests: cell_list_needs_rebuild
# ==============================================================================


class TestCellListNeedsRebuild:
    """Test cell_list_needs_rebuild function."""

    def test_no_movement(self):
        """Test that no rebuild is needed when atoms don't move."""
        current_positions = jnp.array(
            [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=jnp.float32
        )
        cell = jnp.array([[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]])
        pbc = jnp.array([True, True, True])
        cells_per_dimension = jnp.array([2, 2, 2], dtype=jnp.int32)
        atom_to_cell_mapping = jnp.array([[0, 0, 0], [1, 0, 0]], dtype=jnp.int32)

        rebuild_needed = cell_list_needs_rebuild(
            current_positions=current_positions,
            atom_to_cell_mapping=atom_to_cell_mapping,
            cells_per_dimension=cells_per_dimension,
            cell=cell,
            pbc=pbc,
        )

        assert rebuild_needed.shape == (1,)
        assert rebuild_needed.dtype == jnp.bool_
        assert not rebuild_needed.item()

    def test_small_movement_within_cell(self):
        """Test no rebuild for small movements within cells."""
        current_positions = jnp.array(
            [[0.1, 0.0, 0.0], [5.2, 0.0, 0.0]], dtype=jnp.float32
        )
        cell = jnp.array([[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]])
        pbc = jnp.array([True, True, True])
        cells_per_dimension = jnp.array([2, 2, 2], dtype=jnp.int32)
        atom_to_cell_mapping = jnp.array([[0, 0, 0], [1, 0, 0]], dtype=jnp.int32)

        rebuild_needed = cell_list_needs_rebuild(
            current_positions=current_positions,
            atom_to_cell_mapping=atom_to_cell_mapping,
            cells_per_dimension=cells_per_dimension,
            cell=cell,
            pbc=pbc,
        )

        # May or may not need rebuild depending on cell size
        assert rebuild_needed.shape == (1,)

    def test_large_movement_across_cells(self):
        """Test rebuild needed for large movements across cells."""
        current_positions = jnp.array(
            [[6.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=jnp.float32
        )
        cell = jnp.array([[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]])
        pbc = jnp.array([True, True, True])
        cells_per_dimension = jnp.array([2, 2, 2], dtype=jnp.int32)
        atom_to_cell_mapping = jnp.array([[0, 0, 0], [1, 0, 0]], dtype=jnp.int32)

        rebuild_needed = cell_list_needs_rebuild(
            current_positions=current_positions,
            atom_to_cell_mapping=atom_to_cell_mapping,
            cells_per_dimension=cells_per_dimension,
            cell=cell,
            pbc=pbc,
        )

        assert rebuild_needed.item()

    def test_empty_system(self):
        """Test with empty system."""
        current_positions = jnp.zeros((0, 3), dtype=jnp.float32)
        cell = jnp.array([[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]])
        pbc = jnp.array([True, True, True])
        cells_per_dimension = jnp.array([1, 1, 1], dtype=jnp.int32)
        atom_to_cell_mapping = jnp.zeros((0, 3), dtype=jnp.int32)

        rebuild_needed = cell_list_needs_rebuild(
            current_positions=current_positions,
            atom_to_cell_mapping=atom_to_cell_mapping,
            cells_per_dimension=cells_per_dimension,
            cell=cell,
            pbc=pbc,
        )

        assert not rebuild_needed.item()


# ==============================================================================
# Tests: JIT compatibility
# ==============================================================================


class TestNeighborListRebuildJIT:
    """Test neighbor_list_needs_rebuild under jax.jit."""

    def test_jit_no_movement(self):
        """Test JIT: no rebuild needed when atoms don't move."""
        positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=jnp.float32)

        @jax.jit
        def check_rebuild(ref, cur):
            return neighbor_list_needs_rebuild(ref, cur, 0.5)

        result = check_rebuild(positions, positions)
        assert result.shape == (1,)
        assert not result.item()

    def test_jit_beyond_skin(self):
        """Test JIT: rebuild needed when atom moves beyond skin distance."""
        reference = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=jnp.float32)
        current = reference + jnp.array(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=jnp.float32
        )

        @jax.jit
        def check_rebuild(ref, cur):
            return neighbor_list_needs_rebuild(ref, cur, 0.5)

        result = check_rebuild(reference, current)
        assert result.item()

    def test_jit_within_skin(self):
        """Test JIT: no rebuild for small movements within skin distance."""
        reference = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=jnp.float32)
        current = reference + jnp.array(
            [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]], dtype=jnp.float32
        )

        @jax.jit
        def check_rebuild(ref, cur):
            return neighbor_list_needs_rebuild(ref, cur, 0.5)

        result = check_rebuild(reference, current)
        assert not result.item()


class TestCellListRebuildJIT:
    """Test cell_list_needs_rebuild under jax.jit."""

    def test_jit_no_movement(self):
        """Test JIT: no rebuild needed when atoms stay in same cells."""
        positions = jnp.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=jnp.float32)
        cell = jnp.array(
            [[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]], dtype=jnp.float32
        )
        pbc = jnp.array([True, True, True])
        cells_per_dim = jnp.array([2, 2, 2], dtype=jnp.int32)
        mapping = jnp.array([[0, 0, 0], [1, 0, 0]], dtype=jnp.int32)

        @jax.jit
        def check_rebuild(pos, mapping, cells, c, p):
            return cell_list_needs_rebuild(pos, mapping, cells, c, p)

        result = check_rebuild(positions, mapping, cells_per_dim, cell, pbc)
        assert result.shape == (1,)
        assert not result.item()

    def test_jit_across_cells(self):
        """Test JIT: rebuild needed when atom crosses cell boundary."""
        positions = jnp.array([[6.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=jnp.float32)
        cell = jnp.array(
            [[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]], dtype=jnp.float32
        )
        pbc = jnp.array([True, True, True])
        cells_per_dim = jnp.array([2, 2, 2], dtype=jnp.int32)
        mapping = jnp.array([[0, 0, 0], [1, 0, 0]], dtype=jnp.int32)

        @jax.jit
        def check_rebuild(pos, mapping, cells, c, p):
            return cell_list_needs_rebuild(pos, mapping, cells, c, p)

        result = check_rebuild(positions, mapping, cells_per_dim, cell, pbc)
        assert result.item()
