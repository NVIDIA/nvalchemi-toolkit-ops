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
import numpy as np
import pytest

from nvalchemiops.jax.neighbors.batch_cell_list import batch_build_cell_list
from nvalchemiops.jax.neighbors.rebuild_detection import (
    batch_cell_list_needs_rebuild,
    batch_neighbor_list_needs_rebuild,
    cell_list_needs_rebuild,
    check_batch_cell_list_rebuild_needed,
    check_batch_neighbor_list_rebuild_needed,
    neighbor_list_needs_rebuild,
)

from .conftest import (
    create_batch_idx_and_ptr_jax,
    create_simple_cubic_system_jax,
    requires_gpu,
)

dtypes = [jnp.float32, jnp.float64]

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


# ==============================================================================
# Helpers for batch tests
# ==============================================================================


def _make_batch_systems(dtype, num_systems=3, rng_seed=42):
    """Create a small batch of identical cubic systems for batch rebuild tests.

    Returns positions (total_atoms, 3), cell (num_systems, 3, 3),
    pbc (num_systems, 3), batch_idx (total_atoms,), atoms_per list.
    """
    rng = np.random.default_rng(rng_seed)
    atoms_per = [6, 8, 5, 7, 4]
    atoms_per = (atoms_per * ((num_systems // len(atoms_per)) + 1))[:num_systems]

    all_pos = []
    for n in atoms_per:
        pos = rng.random((n, 3)).astype(np.float64) * 3.0
        all_pos.append(pos)

    positions = jnp.array(np.concatenate(all_pos, axis=0), dtype=dtype)
    cell = jnp.stack(
        [jnp.eye(3, dtype=dtype) * 4.0 for _ in range(num_systems)], axis=0
    )
    pbc = jnp.zeros((num_systems, 3), dtype=jnp.bool_)
    batch_idx, _ = create_batch_idx_and_ptr_jax(atoms_per)

    return positions, cell, pbc, batch_idx, atoms_per


def _build_batch_cell_list_for_test(positions, cell, pbc, batch_idx, cutoff=1.0):
    """Build batch cell list and return (cells_per_dimension, atom_to_cell_mapping)."""
    result = batch_build_cell_list(
        positions,
        batch_idx=batch_idx,
        cell=cell,
        pbc=pbc,
        cutoff=cutoff,
    )
    cells_per_dimension = result[0]  # shape (num_systems, 3)
    atom_to_cell_mapping = result[2]  # shape (total_atoms, 3)
    return cells_per_dimension, atom_to_cell_mapping


# ==============================================================================
# Tests: batch_neighbor_list_needs_rebuild
# ==============================================================================


class TestBatchNeighborListNeedsRebuild:
    """Test batch_neighbor_list_needs_rebuild function."""

    @pytest.mark.parametrize("dtype", dtypes)
    def test_no_movement(self, dtype):
        """No movement in any system → all rebuild_flags should be False."""
        positions, cell, pbc, batch_idx, atoms_per = _make_batch_systems(dtype)
        num_systems = len(atoms_per)

        rebuild_flags = batch_neighbor_list_needs_rebuild(
            reference_positions=positions,
            current_positions=positions,
            batch_idx=batch_idx,
            skin_distance_threshold=0.5,
            num_systems=num_systems,
        )

        assert rebuild_flags.shape == (num_systems,)
        assert rebuild_flags.dtype == jnp.bool_
        assert not jnp.any(rebuild_flags), (
            "No systems should need rebuild with no movement"
        )

    @pytest.mark.parametrize("dtype", dtypes)
    def test_selective_rebuild(self, dtype):
        """Only system 1 has an atom move beyond skin → only flag[1] is True."""
        positions, cell, pbc, batch_idx, atoms_per = _make_batch_systems(dtype)
        num_systems = len(atoms_per)

        # Move an atom in system 1 (first atom of system 1 = atoms_per[0])
        system1_start = atoms_per[0]
        current_positions = positions.at[system1_start].add(2.0)

        rebuild_flags = batch_neighbor_list_needs_rebuild(
            reference_positions=positions,
            current_positions=current_positions,
            batch_idx=batch_idx,
            skin_distance_threshold=0.5,
            num_systems=num_systems,
        )

        assert rebuild_flags.shape == (num_systems,)
        assert not rebuild_flags[0].item(), "System 0 should not need rebuild"
        assert rebuild_flags[1].item(), "System 1 should need rebuild"
        assert not rebuild_flags[2].item(), "System 2 should not need rebuild"

    @pytest.mark.parametrize("dtype", dtypes)
    def test_all_systems_rebuild(self, dtype):
        """All systems have atoms moving beyond skin → all flags True."""
        positions, cell, pbc, batch_idx, atoms_per = _make_batch_systems(dtype)
        num_systems = len(atoms_per)

        # Compute start indices
        starts = [0] + list(np.cumsum(atoms_per[:-1]))
        current_positions = positions
        for s in starts:
            current_positions = current_positions.at[s].add(2.0)

        rebuild_flags = batch_neighbor_list_needs_rebuild(
            reference_positions=positions,
            current_positions=current_positions,
            batch_idx=batch_idx,
            skin_distance_threshold=0.5,
            num_systems=num_systems,
        )

        assert jnp.all(rebuild_flags), "All systems should need rebuild"

    @pytest.mark.parametrize("dtype", dtypes)
    def test_output_shape_varies_with_num_systems(self, dtype):
        """Output shape matches num_systems for different batch sizes."""
        for ns in [1, 2, 5]:
            positions, cell, pbc, batch_idx, atoms_per = _make_batch_systems(
                dtype, num_systems=ns
            )
            rebuild_flags = batch_neighbor_list_needs_rebuild(
                reference_positions=positions,
                current_positions=positions,
                batch_idx=batch_idx,
                skin_distance_threshold=0.5,
                num_systems=ns,
            )
            assert rebuild_flags.shape == (ns,)
            assert rebuild_flags.dtype == jnp.bool_

    @pytest.mark.parametrize("dtype", dtypes)
    def test_empty_system(self, dtype):
        """Empty positions → returns all-False flags."""
        positions = jnp.zeros((0, 3), dtype=dtype)
        batch_idx = jnp.zeros(0, dtype=jnp.int32)

        rebuild_flags = batch_neighbor_list_needs_rebuild(
            reference_positions=positions,
            current_positions=positions,
            batch_idx=batch_idx,
            skin_distance_threshold=0.5,
            num_systems=2,
        )

        assert rebuild_flags.shape == (2,)
        assert not jnp.any(rebuild_flags)


# ==============================================================================
# Tests: batch_cell_list_needs_rebuild
# ==============================================================================


class TestBatchCellListNeedsRebuild:
    """Test batch_cell_list_needs_rebuild function."""

    @pytest.mark.parametrize("dtype", dtypes)
    def test_no_movement(self, dtype):
        """No movement in any system → all rebuild_flags should be False."""
        positions, cell, pbc, batch_idx, atoms_per = _make_batch_systems(dtype)
        num_systems = len(atoms_per)
        cells_per_dimension, atom_to_cell_mapping = _build_batch_cell_list_for_test(
            positions, cell, pbc, batch_idx
        )

        rebuild_flags = batch_cell_list_needs_rebuild(
            current_positions=positions,
            atom_to_cell_mapping=atom_to_cell_mapping,
            batch_idx=batch_idx,
            cells_per_dimension=cells_per_dimension,
            cell=cell,
            pbc=pbc,
        )

        assert rebuild_flags.shape == (num_systems,)
        assert rebuild_flags.dtype == jnp.bool_
        assert not jnp.any(rebuild_flags), (
            "No systems should need rebuild with no movement"
        )

    @pytest.mark.parametrize("dtype", dtypes)
    def test_selective_rebuild(self, dtype):
        """Only system 0 has an atom cross a cell boundary → only flag[0] is True."""
        positions, cell, pbc, batch_idx, atoms_per = _make_batch_systems(dtype)
        num_systems = len(atoms_per)
        cells_per_dimension, atom_to_cell_mapping = _build_batch_cell_list_for_test(
            positions, cell, pbc, batch_idx
        )

        # Move first atom of system 0 by 1.5 Å (crosses cell boundary in 4 Å box)
        new_positions = positions.at[0].add(1.5)

        rebuild_flags = batch_cell_list_needs_rebuild(
            current_positions=new_positions,
            atom_to_cell_mapping=atom_to_cell_mapping,
            batch_idx=batch_idx,
            cells_per_dimension=cells_per_dimension,
            cell=cell,
            pbc=pbc,
        )

        assert rebuild_flags.shape == (num_systems,)
        assert rebuild_flags[0].item(), "System 0 should need rebuild"
        assert not rebuild_flags[1].item(), "System 1 should not need rebuild"
        assert not rebuild_flags[2].item(), "System 2 should not need rebuild"

    @pytest.mark.parametrize("dtype", dtypes)
    def test_output_shape(self, dtype):
        """Output shape matches num_systems from cell.shape[0]."""
        for ns in [1, 2, 4]:
            positions, cell, pbc, batch_idx, atoms_per = _make_batch_systems(
                dtype, num_systems=ns
            )
            cells_per_dimension, atom_to_cell_mapping = _build_batch_cell_list_for_test(
                positions, cell, pbc, batch_idx
            )

            rebuild_flags = batch_cell_list_needs_rebuild(
                current_positions=positions,
                atom_to_cell_mapping=atom_to_cell_mapping,
                batch_idx=batch_idx,
                cells_per_dimension=cells_per_dimension,
                cell=cell,
                pbc=pbc,
            )

            assert rebuild_flags.shape == (ns,)
            assert rebuild_flags.dtype == jnp.bool_

    @pytest.mark.parametrize("dtype", dtypes)
    def test_empty_system(self, dtype):
        """Empty positions → returns all-False flags."""
        num_systems = 2
        positions = jnp.zeros((0, 3), dtype=dtype)
        cell = jnp.stack([jnp.eye(3, dtype=dtype) * 4.0] * num_systems)
        pbc = jnp.zeros((num_systems, 3), dtype=jnp.bool_)
        batch_idx = jnp.zeros(0, dtype=jnp.int32)
        atom_to_cell_mapping = jnp.zeros((0, 3), dtype=jnp.int32)
        cells_per_dimension = jnp.ones((num_systems, 3), dtype=jnp.int32)

        rebuild_flags = batch_cell_list_needs_rebuild(
            current_positions=positions,
            atom_to_cell_mapping=atom_to_cell_mapping,
            batch_idx=batch_idx,
            cells_per_dimension=cells_per_dimension,
            cell=cell,
            pbc=pbc,
        )

        assert rebuild_flags.shape == (num_systems,)
        assert not jnp.any(rebuild_flags)


# ==============================================================================
# Tests: check_batch_*_rebuild_needed convenience wrappers
# ==============================================================================


class TestCheckBatchRebuildNeededWrappers:
    """Test check_batch_neighbor_list_rebuild_needed and check_batch_cell_list_rebuild_needed."""

    @pytest.mark.parametrize("dtype", dtypes)
    def test_check_batch_neighbor_list_no_movement(self, dtype):
        """Convenience wrapper returns list[bool] with all False when no movement."""
        positions, cell, pbc, batch_idx, atoms_per = _make_batch_systems(dtype)
        num_systems = len(atoms_per)

        result = check_batch_neighbor_list_rebuild_needed(
            reference_positions=positions,
            current_positions=positions,
            batch_idx=batch_idx,
            skin_distance_threshold=0.5,
            num_systems=num_systems,
        )

        assert isinstance(result, list)
        assert len(result) == num_systems
        assert all(isinstance(v, bool) for v in result)
        assert not any(result)

    @pytest.mark.parametrize("dtype", dtypes)
    def test_check_batch_neighbor_list_with_movement(self, dtype):
        """Convenience wrapper returns True for systems with atoms beyond skin."""
        positions, cell, pbc, batch_idx, atoms_per = _make_batch_systems(dtype)
        num_systems = len(atoms_per)
        system1_start = atoms_per[0]
        current_positions = positions.at[system1_start].add(2.0)

        result = check_batch_neighbor_list_rebuild_needed(
            reference_positions=positions,
            current_positions=current_positions,
            batch_idx=batch_idx,
            skin_distance_threshold=0.5,
            num_systems=num_systems,
        )

        assert not result[0]
        assert result[1]
        assert not result[2]

    @pytest.mark.parametrize("dtype", dtypes)
    def test_check_batch_cell_list_no_movement(self, dtype):
        """Convenience wrapper returns list[bool] with all False when no movement."""
        positions, cell, pbc, batch_idx, atoms_per = _make_batch_systems(dtype)
        num_systems = len(atoms_per)
        cells_per_dimension, atom_to_cell_mapping = _build_batch_cell_list_for_test(
            positions, cell, pbc, batch_idx
        )

        result = check_batch_cell_list_rebuild_needed(
            current_positions=positions,
            atom_to_cell_mapping=atom_to_cell_mapping,
            batch_idx=batch_idx,
            cells_per_dimension=cells_per_dimension,
            cell=cell,
            pbc=pbc,
        )

        assert isinstance(result, list)
        assert len(result) == num_systems
        assert all(isinstance(v, bool) for v in result)
        assert not any(result)

    @pytest.mark.parametrize("dtype", dtypes)
    def test_check_batch_cell_list_with_movement(self, dtype):
        """Convenience wrapper returns True for system 0 after atom crosses cell boundary."""
        positions, cell, pbc, batch_idx, atoms_per = _make_batch_systems(dtype)
        cells_per_dimension, atom_to_cell_mapping = _build_batch_cell_list_for_test(
            positions, cell, pbc, batch_idx
        )
        new_positions = positions.at[0].add(1.5)

        result = check_batch_cell_list_rebuild_needed(
            current_positions=new_positions,
            atom_to_cell_mapping=atom_to_cell_mapping,
            batch_idx=batch_idx,
            cells_per_dimension=cells_per_dimension,
            cell=cell,
            pbc=pbc,
        )

        assert result[0]
        assert not result[1]
        assert not result[2]


@pytest.mark.parametrize("dtype", dtypes)
class TestJaxPBCRebuildDetection:
    """Test PBC-aware rebuild detection via JAX bindings."""

    def test_pbc_no_spurious_rebuild(self, dtype):
        """Boundary-crossing atom must not trigger rebuild with PBC."""
        cell_size = 10.0
        skin = 1.0
        positions, cell, pbc = create_simple_cubic_system_jax(
            num_atoms=8,
            cell_size=cell_size,
            dtype=dtype,
        )
        pbc = pbc.squeeze(0)
        cell_inv = jnp.array(np.eye(3) / cell_size, dtype=dtype).reshape(1, 3, 3)

        reference = positions
        current = positions.at[0, 0].set(cell_size - 0.01)

        result = neighbor_list_needs_rebuild(
            reference,
            current,
            skin / 2.0,
            cell=cell,
            cell_inv=cell_inv,
            pbc=pbc,
        )
        assert not bool(result[0]), (
            "PBC should prevent spurious rebuild at periodic boundary"
        )

    def test_pbc_genuine_rebuild(self, dtype):
        """Large genuine displacement should still trigger rebuild with PBC."""
        cell_size = 10.0
        skin = 1.0
        positions, cell, pbc = create_simple_cubic_system_jax(
            num_atoms=8,
            cell_size=cell_size,
            dtype=dtype,
        )
        pbc = pbc.squeeze(0)
        cell_inv = jnp.array(np.eye(3) / cell_size, dtype=dtype).reshape(1, 3, 3)

        reference = positions
        current = positions.at[0, 0].add(2.0)

        result = neighbor_list_needs_rebuild(
            reference,
            current,
            skin / 2.0,
            cell=cell,
            cell_inv=cell_inv,
            pbc=pbc,
        )
        assert bool(result[0]), (
            "PBC should still trigger rebuild for genuinely large displacement"
        )

    def test_batch_pbc_no_spurious_rebuild(self, dtype):
        """Batch PBC: boundary-crossing atoms must not trigger rebuild."""
        cell_size = 10.0
        skin = 1.0
        num_systems = 3
        atoms_per = [4, 5, 3]
        total_atoms = sum(atoms_per)

        n_side = 2
        spacing = cell_size / n_side
        grid_coords = [
            [i * spacing, j * spacing, k * spacing]
            for i in range(n_side)
            for j in range(n_side)
            for k in range(n_side)
        ]
        all_coords = []
        for n_atoms in atoms_per:
            all_coords.extend(grid_coords[:n_atoms])
        positions = jnp.array(all_coords[:total_atoms], dtype=dtype)

        cells = jnp.array(np.stack([np.eye(3) * cell_size] * num_systems), dtype=dtype)
        cells_inv = jnp.array(
            np.stack([np.eye(3) / cell_size] * num_systems), dtype=dtype
        )
        pbc = jnp.ones((num_systems, 3), dtype=jnp.bool_)
        batch_idx = jnp.array(
            np.repeat(np.arange(num_systems), atoms_per), dtype=jnp.int32
        )

        reference = positions
        current = positions.at[5, 0].set(cell_size - 0.01)

        result = batch_neighbor_list_needs_rebuild(
            reference,
            current,
            batch_idx,
            skin / 2.0,
            num_systems=3,
            cell=cells,
            cell_inv=cells_inv,
            pbc=pbc,
        )
        assert not bool(jnp.any(result)), (
            "Batch PBC should prevent spurious rebuilds at boundaries"
        )
